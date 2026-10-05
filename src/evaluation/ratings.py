from __future__ import annotations

from collections import defaultdict
from importlib.metadata import version
from itertools import permutations
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time
from typing import Optional, Sequence

from openskill.models import PlackettLuce

from src.checkpoints.locking import exclusive_writer
from src.checkpoints.io import atomic_write_json
from src.game.environment import GameConfig
from src.game.multiplayer import MatchConfig


OPEN_SKILL_VERSION = "6.2.0"
CAP_POLICIES = ("undecided", "points", "weighted-points", "tie", "exclude")
APPLICATION_ID = 0x50415052
SCHEMA_VERSION = 1
RATING_PARAMETERS = {
    "mu": 25.0, "sigma": 25.0 / 3, "beta": 25.0 / 6, "kappa": 0.0001,
    "tau": 0.0, "margin": 0.0, "limit_sigma": False, "balance": False,
}


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(value: str) -> None:
    if value == "random-v1":
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("competitor identity must be a checkpoint SHA-256 or random-v1")


def placements(points: Sequence[int], winner: Optional[int], ending: str, cap_policy: str) -> Optional[tuple[int, ...]]:
    if cap_policy not in CAP_POLICIES:
        raise ValueError("unknown cap policy")
    if len(points) < 2 or any(type(point) is not int or point < 0 for point in points):
        raise ValueError("penalties must contain nonnegative integer points for every player")
    if ending == "win":
        if type(winner) is not int or not 0 <= winner < len(points) or points[winner] != 0:
            raise ValueError("a declaration needs a winning seat with zero penalty")
        return tuple(
            1 if seat == winner else 2 + sum(
                other != winner and points[other] < mine
                for other in range(len(points))
            )
            for seat, mine in enumerate(points)
        )
    if ending != "turns_exhausted" or winner is not None:
        raise ValueError("only completed declarations and draw-limit endings have placements")
    if cap_policy in ("points", "weighted-points"):
        return tuple(1 + sum(other < mine for other in points) for mine in points)
    if cap_policy == "tie":
        return (1,) * len(points)
    return None


class RatingStore:
    def __init__(self, path: Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._connection = None
        self._lock = None
        self._projections: dict[tuple, _ProjectionState] = {}

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("use RatingStore as a context manager")
        return self._connection

    def __enter__(self) -> "RatingStore":
        if self._connection is not None:
            raise RuntimeError("rating store is already open")
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = exclusive_writer(self.path.with_suffix(self.path.suffix + ".lock"))
            self._lock.__enter__()
        opened = False
        try:
            self._connection = (
                sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
                if self.read_only else sqlite3.connect(self.path)
            )
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA busy_timeout=5000")
            app_id = self.connection.execute("PRAGMA application_id").fetchone()[0]
            schema = self.connection.execute("PRAGMA user_version").fetchone()[0]
            tables = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if not self.read_only and app_id == 0 and schema == 0 and not tables:
                self._initialize()
            elif app_id != APPLICATION_ID or schema != SCHEMA_VERSION:
                raise ValueError("not a supported Papplu rating database")
            if not self.read_only:
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                with self.connection:
                    self.connection.execute(
                        "UPDATE attempts SET finished=?, status='interrupted', error='coordinator restarted' WHERE finished IS NULL",
                        (time.time(),),
                    )
            opened = True
            return self
        finally:
            if not opened:
                self.close()

    def __exit__(self, *exception) -> None:
        self.close()

    def close(self) -> None:
        self._projections.clear()
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None

    def _write(self) -> None:
        if self.read_only:
            raise RuntimeError("rating store is read-only")

    def _initialize(self) -> None:
        self.connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
            INSERT INTO state VALUES (1,0);
            CREATE TABLE competitors (id TEXT PRIMARY KEY, label TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE protocols (id TEXT PRIMARY KEY, config TEXT NOT NULL);
            CREATE TABLE blocks (
                protocol TEXT NOT NULL REFERENCES protocols(id), ordinal INTEGER NOT NULL,
                seed INTEGER NOT NULL, lineup TEXT NOT NULL, PRIMARY KEY(protocol,ordinal)
            );
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, protocol TEXT NOT NULL, block INTEGER NOT NULL,
                position INTEGER NOT NULL, participants TEXT NOT NULL, seat_order TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','raw','complete','failed')),
                FOREIGN KEY(protocol,block) REFERENCES blocks(protocol,ordinal),
                UNIQUE(protocol,block,position)
            );
            CREATE TABLE results (
                job TEXT PRIMARY KEY REFERENCES jobs(id), winner INTEGER, ending TEXT NOT NULL,
                raw TEXT NOT NULL, points TEXT, proofs TEXT
            );
            CREATE TABLE attempts (
                id INTEGER PRIMARY KEY, job TEXT NOT NULL REFERENCES jobs(id), phase TEXT NOT NULL,
                started REAL NOT NULL, finished REAL, status TEXT NOT NULL, error TEXT
            );
            CREATE UNIQUE INDEX one_open_attempt ON attempts(job) WHERE finished IS NULL;
            CREATE INDEX pending_jobs ON jobs(protocol,status,block,position);
            PRAGMA application_id=%d;
            PRAGMA user_version=%d;
            COMMIT;
        """ % (APPLICATION_ID, SCHEMA_VERSION))

    def _changed(self) -> None:
        self.connection.execute("UPDATE state SET revision=revision+1 WHERE id=1")

    def register_competitor(self, identity: str, label: str, metadata: dict) -> bool:
        self._write()
        _identity(identity)
        if not isinstance(label, str) or not label:
            raise ValueError("competitor label is required")
        if not isinstance(metadata, dict):
            raise ValueError("competitor metadata must be an object")
        body = _json(metadata)
        existing = self.connection.execute("SELECT metadata FROM competitors WHERE id=?", (identity,)).fetchone()
        if existing is not None:
            if existing["metadata"] != body:
                raise ValueError("immutable competitor metadata changed")
            return False
        with self.connection:
            self.connection.execute("INSERT INTO competitors VALUES (?,?,?)", (identity, label, body))
            self._changed()
        return True

    def competitors(self) -> dict:
        return {
            row["id"]: {"label": row["label"], "metadata": json.loads(row["metadata"])}
            for row in self.connection.execute("SELECT id,label,metadata FROM competitors ORDER BY id")
        }

    def ensure_protocol(self, config: dict) -> str:
        self._write()
        required = {"players", "game", "encoding_version", "scoring_version", "policy", "seating_design", "seed"}
        if set(config) != required:
            raise ValueError("rating protocol fields do not match the schema")
        game = GameConfig.from_dict(config["game"])
        MatchConfig(game=game, players=config["players"])
        for name in ("encoding_version", "scoring_version"):
            if type(config[name]) is not int or config[name] < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 53:
            raise ValueError("protocol seed must be an exactly representable nonnegative integer")
        if config["policy"] != "greedy-v1" or config["seating_design"] != "all-permutations-v1":
            raise ValueError("unsupported evaluation policy or seating design")
        body = _json(config)
        identity = hashlib.sha256(body.encode("utf-8")).hexdigest()
        with self.connection:
            inserted = self.connection.execute("INSERT OR IGNORE INTO protocols VALUES (?,?)", (identity, body))
            if inserted.rowcount:
                self._changed()
        return identity

    def protocols(self) -> dict:
        return {
            row["id"]: json.loads(row["config"])
            for row in self.connection.execute("SELECT id,config FROM protocols ORDER BY id")
        }

    def _protocol(self, identity: str) -> dict:
        row = self.connection.execute("SELECT config FROM protocols WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("unknown rating protocol")
        return json.loads(row["config"])

    def blocks(self, protocol: str, after: int = -1) -> list[dict]:
        return [
            {"ordinal": row["ordinal"], "seed": row["seed"], "lineup": json.loads(row["lineup"])}
            for row in self.connection.execute(
                "SELECT ordinal,seed,lineup FROM blocks WHERE protocol=? AND ordinal>? ORDER BY ordinal", (protocol, after),
            )
        ]

    def schedule_block(self, protocol: str, ordinal: int, seed: int, lineup: Sequence[str]) -> list[dict]:
        self._write()
        config = self._protocol(protocol)
        if type(ordinal) is not int or ordinal < 0 or type(seed) is not int or not 0 <= seed < 2 ** 53:
            raise ValueError("invalid block ordinal or seed")
        if len(lineup) != config["players"] or len(set(lineup)) != len(lineup):
            raise ValueError("a lineup needs one distinct competitor per seat")
        known = self.competitors()
        if any(identity not in known for identity in lineup):
            raise ValueError("lineup contains an unregistered competitor")
        existing = self.connection.execute(
            "SELECT seed,lineup FROM blocks WHERE protocol=? AND ordinal=?", (protocol, ordinal),
        ).fetchone()
        if existing is not None:
            if existing["seed"] != seed or existing["lineup"] != _json(list(lineup)):
                raise ValueError("scheduled block conflicts with its immutable identity")
            return self.jobs(protocol, ordinal)
        next_ordinal = self.connection.execute(
            "SELECT COALESCE(MAX(ordinal)+1,0) FROM blocks WHERE protocol=?", (protocol,),
        ).fetchone()[0]
        if ordinal != next_ordinal:
            raise ValueError("blocks must be scheduled in contiguous order")
        orders = list(permutations(range(config["players"])))
        random.Random(seed ^ 0x6A09E667).shuffle(orders)
        with self.connection:
            self.connection.execute("INSERT INTO blocks VALUES (?,?,?,?)", (protocol, ordinal, seed, _json(list(lineup))))
            for position, order in enumerate(orders):
                participants = [lineup[index] for index in order]
                identity = hashlib.sha256(_json([protocol, ordinal, seed, participants, order]).encode("utf-8")).hexdigest()
                self.connection.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,'pending')",
                    (identity, protocol, ordinal, position, _json(participants), _json(order)),
                )
            self._changed()
        return self.jobs(protocol, ordinal)

    def jobs(self, protocol: Optional[str] = None, block: Optional[int] = None, pending_only: bool = False) -> list[dict]:
        clauses, parameters = [], []
        if protocol is not None:
            clauses.append("j.protocol=?")
            parameters.append(protocol)
        if block is not None:
            clauses.append("j.block=?")
            parameters.append(block)
        if pending_only:
            clauses.append("j.status IN ('pending','raw')")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            """SELECT j.*,b.seed,p.config,r.raw,r.winner,r.ending
               FROM jobs j JOIN blocks b ON b.protocol=j.protocol AND b.ordinal=j.block
               JOIN protocols p ON p.id=j.protocol LEFT JOIN results r ON r.job=j.id"""
            + where + " ORDER BY j.protocol,j.block,j.position",
            parameters,
        )
        result = []
        for row in rows:
            raw = None if row["raw"] is None else {
                **json.loads(row["raw"]), "winner": row["winner"], "terminal_reason": row["ending"],
            }
            result.append({
                "id": row["id"], "protocol": row["protocol"], "block": row["block"],
                "position": row["position"], "seed": row["seed"], "status": row["status"],
                "participants": json.loads(row["participants"]), "seat_order": json.loads(row["seat_order"]),
                "config": json.loads(row["config"]), "raw": raw,
            })
        return result

    def _job(self, identity: str) -> dict:
        row = self.connection.execute("SELECT protocol,block FROM jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("unknown tournament job")
        return next(job for job in self.jobs(row["protocol"], row["block"]) if job["id"] == identity)

    def start_attempt(self, identity: str) -> str:
        self._write()
        job = self._job(identity)
        if job["status"] not in ("pending", "raw"):
            raise ValueError("job is already settled")
        phase = "play" if job["raw"] is None else "score"
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(job,phase,started,status) VALUES (?,?,?,'running')",
                (identity, phase, time.time()),
            )
        return phase

    def _finish_attempt(self, identity: str, phase: str, status: str, error: Optional[str] = None) -> None:
        changed = self.connection.execute(
            "UPDATE attempts SET finished=?,status=?,error=? WHERE job=? AND phase=? AND finished IS NULL",
            (time.time(), status, error, identity, phase),
        ).rowcount
        if changed != 1:
            raise ValueError("job completion has no matching open attempt")

    def record_raw(self, identity: str, raw: dict) -> bool:
        self._write()
        job = self._job(identity)
        if job["raw"] is not None:
            if _json(job["raw"]) != _json(raw):
                raise ValueError("conflicting result for an existing match")
            return False
        game = GameConfig.from_dict(job["config"]["game"])
        players = len(job["participants"])
        snapshot = raw.get("terminal_snapshot")
        if not isinstance(snapshot, dict) or set(snapshot) != {"hands", "joker"}:
            raise ValueError("completed matches need a terminal hand snapshot")
        joker = snapshot["joker"]
        if type(joker) is not int or not 0 <= joker < 52:
            raise ValueError("invalid terminal joker")
        hands = snapshot["hands"]
        if len(hands) != players or any(
            len(hand) != 52 or any(type(count) is not int or count < 0 for count in hand)
            or sum(hand) != game.cards_in_hand for hand in hands
        ):
            raise ValueError("invalid terminal hand counts")
        if any(sum(hand[face] for hand in hands) + int(face == joker) > game.num_decks for face in range(52)):
            raise ValueError("terminal hands exceed the physical card supply")
        turns = raw["seat_turns"]
        if len(turns) != players or any(type(turn) is not int or not 0 <= turn <= game.max_turns for turn in turns):
            raise ValueError("invalid terminal turn counts")
        if type(raw["seed"]) is not int or raw["seed"] != job["seed"]:
            raise ValueError("match seed does not match the scheduled job")
        if type(raw["action_count"]) is not int or raw["action_count"] != 2 * sum(turns):
            raise ValueError("match action count does not describe completed turns")
        if type(raw["stock_remaining"]) is not int or not 0 <= raw["stock_remaining"] <= game.num_decks * 52:
            raise ValueError("invalid remaining stock count")
        telemetry = raw["telemetry"]
        for values in (telemetry["stock_draws"], telemetry["discard_draws"]):
            if len(values) != players or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("invalid draw telemetry")
        if any(telemetry["stock_draws"][seat] + telemetry["discard_draws"][seat] != turns[seat] for seat in range(players)):
            raise ValueError("draw telemetry does not match completed turns")
        refills = telemetry["refill_turns"]
        if any(type(turn) is not int or not 0 < turn < sum(turns) for turn in refills) or list(refills) != sorted(set(refills)):
            raise ValueError("invalid refill telemetry")
        winner, ending = raw["winner"], raw["terminal_reason"]
        if ending == "win":
            if type(winner) is not int or not 0 <= winner < players or turns[winner] < 1:
                raise ValueError("invalid declared winner")
        elif ending == "turns_exhausted":
            if winner is not None or any(turn != game.max_turns for turn in turns):
                raise ValueError("invalid draw-limit ending")
        else:
            raise ValueError("incomplete or invalid executions are not match results")
        context = {key: value for key, value in raw.items() if key not in ("winner", "terminal_reason")}
        with self.connection:
            self.connection.execute("INSERT INTO results(job,winner,ending,raw) VALUES (?,?,?,?)", (identity, winner, ending, _json(context)))
            self.connection.execute("UPDATE jobs SET status='raw' WHERE id=?", (identity,))
            self._finish_attempt(identity, "play", "completed")
            self._changed()
        return True

    def record_scores(self, identity: str, penalties: Sequence[dict], scoring_version: int) -> bool:
        self._write()
        job = self._job(identity)
        if job["raw"] is None:
            raise ValueError("cannot score a match that has not completed")
        if type(scoring_version) is not int or scoring_version != job["config"]["scoring_version"]:
            raise ValueError("scoring version does not match the protocol")
        if len(penalties) != len(job["participants"]):
            raise ValueError("one penalty is required for every participant")
        points = [penalty["points"] for penalty in penalties]
        if any(type(point) is not int or not 0 <= point <= 10 * job["config"]["game"]["cards_in_hand"] for point in points):
            raise ValueError("invalid penalty points")
        placements(points, job["raw"]["winner"], job["raw"]["terminal_reason"], "undecided")
        proofs = [{key: value for key, value in penalty.items() if key != "points"} for penalty in penalties]
        old = self.connection.execute("SELECT points,proofs FROM results WHERE job=?", (identity,)).fetchone()
        if old["points"] is not None:
            if old["points"] != _json(points) or old["proofs"] != _json(proofs):
                raise ValueError("conflicting scores for an existing match")
            return False
        with self.connection:
            self.connection.execute("UPDATE results SET points=?,proofs=? WHERE job=?", (_json(points), _json(proofs), identity))
            self.connection.execute("UPDATE jobs SET status='complete' WHERE id=?", (identity,))
            self._finish_attempt(identity, "score", "completed")
            self._changed()
        return True

    def fail_job(self, identity: str, reason: str, terminal: bool = False) -> None:
        self._write()
        job = self._job(identity)
        phase = "play" if job["raw"] is None else "score"
        with self.connection:
            self._finish_attempt(identity, phase, "failed", reason)
            if terminal:
                self.connection.execute("UPDATE jobs SET status='failed' WHERE id=?", (identity,))
            self._changed()

    def failure_count(self, identity: str) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE job=? AND status='failed'", (identity,),
        ).fetchone()[0]

    def progress(self, protocols: Optional[Sequence[str]] = None) -> dict:
        parameters = [] if protocols is None else list(protocols)
        where = "" if protocols is None else " WHERE protocol IN (%s)" % ",".join("?" for _ in parameters)
        counts = {row["status"]: row["count"] for row in self.connection.execute(
            "SELECT status,COUNT(*) AS count FROM jobs" + where + " GROUP BY status", parameters,
        )}
        return {
            "competitors": self.connection.execute("SELECT COUNT(*) FROM competitors").fetchone()[0],
            "protocols": self.connection.execute("SELECT COUNT(*) FROM protocols").fetchone()[0] if protocols is None else len(parameters),
            "blocks": self.connection.execute("SELECT COUNT(*) FROM blocks" + where, parameters).fetchone()[0],
            "pending_games": counts.get("pending", 0), "awaiting_scores": counts.get("raw", 0),
            "scored_games": counts.get("complete", 0), "failed_jobs": counts.get("failed", 0),
        }

    def report(
        self, protocol: str, reference: str, cap_policy: str,
        samples: int = 0, seed: int = 81, cap_weight: float = 0.25,
    ) -> dict:
        if cap_policy not in CAP_POLICIES or type(samples) is not int or samples < 0:
            raise ValueError("invalid report settings")
        _validate_weight(cap_weight)
        if version("openskill") != OPEN_SKILL_VERSION:
            raise RuntimeError("rating replay requires openskill==" + OPEN_SKILL_VERSION)
        if self.connection.in_transaction:
            raise RuntimeError("report requires a settled database transaction")
        cached = not self.read_only and samples == 0
        cache_key = (protocol, cap_policy, cap_weight, _json(RATING_PARAMETERS))
        if cached and cache_key in self._projections:
            projection = self._projections[cache_key]
        else:
            projection = _ProjectionState(cap_policy, cap_weight)
            if cached:
                self._projections[cache_key] = projection
        projected = False
        self.connection.execute("BEGIN")
        try:
            config = self._protocol(protocol)
            competitors = self.competitors()
            if reference not in competitors:
                raise ValueError("reference competitor is not registered")
            revision = self.connection.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]
            blocks = []
            incomplete = deferred = newly_settled_failed = 0
            failed = projection.failed_blocks
            cursor = projection.cursor
            prefix_open = True
            for block in self.blocks(protocol, after=projection.cursor):
                rows = self.connection.execute(
                    """SELECT j.status,j.participants,r.winner,r.ending,r.points
                       FROM jobs j LEFT JOIN results r ON r.job=j.id
                       WHERE j.protocol=? AND j.block=? ORDER BY j.position""",
                    (protocol, block["ordinal"]),
                ).fetchall()
                unsettled = any(row["status"] in ("pending", "raw") for row in rows)
                if any(row["status"] == "failed" for row in rows):
                    failed += 1
                if unsettled:
                    prefix_open = False
                    incomplete += 1
                if not prefix_open:
                    deferred += int(not unsettled)
                    continue
                cursor = block["ordinal"]
                if any(row["status"] == "failed" for row in rows):
                    newly_settled_failed += 1
                    continue
                blocks.append({
                    "id": block["ordinal"],
                    "games": [
                        {"participants": json.loads(row["participants"]), "winner": row["winner"],
                         "ending": row["ending"], "points": json.loads(row["points"])}
                        for row in rows
                    ],
                })
            projection.add_competitors(competitors)
            for block in blocks:
                projection.append(block)
            projection.cursor = cursor
            projection.failed_blocks += newly_settled_failed
            rows = projection.rows(reference)
            projected = True
        finally:
            self.connection.rollback()
            if cached and not projected:
                self._projections.pop(cache_key, None)
        if samples:
            if samples < 100:
                raise ValueError("bootstrap reports require at least 100 replicates")
            intervals = defaultdict(list)
            rng = random.Random(seed)
            if len(blocks) >= 30:
                for _ in range(samples):
                    replicated = _fold(rng.choices(blocks, k=len(blocks)), competitors, reference, cap_policy, cap_weight)
                    for identity, row in replicated.items():
                        if row["rating"] is not None:
                            intervals[identity].append(row["rating"])
            for identity, row in rows.items():
                values = sorted(intervals[identity])
                row["bootstrap_covered"] = len(values)
                if (
                    row["rating_blocks"] >= 30 and len(values) >= math.ceil(samples * 0.95)
                    and row["has_beaten"] and row["has_lost"]
                    and identity != reference
                ):
                    row["interval95"] = [_quantile(values, 0.025), _quantile(values, 0.975)]
                    if cap_policy not in ("undecided", "weighted-points") and not failed:
                        row["status"] = "estimated"
        return {
            "version": 1, "protocol": protocol, "config": config, "revision": revision,
            "reference": reference, "cap_policy": cap_policy,
            "scope": (
                "declaration-conditional" if cap_policy in ("undecided", "exclude")
                else "experimental-discounted-cap-evidence" if cap_policy == "weighted-points"
                else "all-completed-games"
            ),
            "cap_evidence_weight": cap_weight if cap_policy == "weighted-points" else 0 if cap_policy in ("undecided", "exclude") else 1,
            "cap_weight_validated": False if cap_policy == "weighted-points" else None,
            "discount_method": "gaussian-information-interpolation-v1" if cap_policy == "weighted-points" else None,
            "openskill_version": OPEN_SKILL_VERSION, "tau": 0,
            "rating_parameters": dict(RATING_PARAMETERS),
            "display": {"center": 1500, "scale": 40},
            "completed_blocks": projection.completed_blocks, "incomplete_blocks": incomplete, "failed_blocks": failed,
            "deferred_blocks": deferred,
            "mixed_cap_blocks": projection.mixed_cap_blocks,
            "raw_seatings_balanced": True,
            "effective_seatings_balanced": not (
                projection.mixed_cap_blocks
                and (cap_policy in ("undecided", "exclude") or cap_policy == "weighted-points" and cap_weight < 1)
            ),
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "uncertainty_unit": "complete independent deal block",
            "rows": sorted(rows.values(), key=lambda row: (row["rating"] is None, -(row["rating"] or 0), row["id"])),
        }


def _fold(blocks: list[dict], competitors: dict, reference: str, cap_policy: str, cap_weight: float = 0.25) -> dict:
    projection = _ProjectionState(cap_policy, cap_weight)
    projection.add_competitors(competitors)
    for block in blocks:
        projection.append(block)
    return projection.rows(reference)


class _ProjectionState:
    def __init__(self, cap_policy: str, cap_weight: float) -> None:
        self.model = PlackettLuce(**RATING_PARAMETERS)
        self.cap_policy = cap_policy
        self.cap_weight = cap_weight
        self.ratings = {}
        self.counters = {}
        self.cursor = -1
        self.completed_blocks = 0
        self.failed_blocks = 0
        self.mixed_cap_blocks = 0

    def add_competitors(self, competitors: dict) -> None:
        for identity, record in competitors.items():
            if identity not in self.counters:
                self.counters[identity] = {
                    "id": identity, "label": record["label"], "games": 0, "rated_games": 0,
                    "weighted_games": 0.0, "declaration_wins": 0, "cap_endings": 0, "penalty_sum": 0,
                    "blocks": set(), "rating_blocks": set(), "opponents": set(),
                    "has_beaten": False, "has_lost": False,
                }

    def append(self, block: dict) -> None:
        self.completed_blocks += 1
        endings = {game["ending"] for game in block["games"]}
        self.mixed_cap_blocks += int(len(endings) > 1)
        for game in block["games"]:
            identities = game["participants"]
            ranks = placements(game["points"], game["winner"], game["ending"], self.cap_policy)
            for seat, identity in enumerate(identities):
                row = self.counters[identity]
                row["games"] += 1
                row["blocks"].add(block["id"])
                row["declaration_wins"] += int(game["winner"] == seat)
                row["cap_endings"] += int(game["ending"] == "turns_exhausted")
                row["penalty_sum"] += game["points"][seat]
            weight = self.cap_weight if game["ending"] == "turns_exhausted" and self.cap_policy == "weighted-points" else 1.0
            if ranks is None or weight == 0:
                continue
            for identity in identities:
                if identity not in self.ratings:
                    self.ratings[identity] = self.model.rating(name=identity)
            teams = [[self.ratings[identity]] for identity in identities]
            prior = [(team[0].mu, team[0].sigma) for team in teams]
            updated = self.model.rate(teams, ranks=list(ranks))
            for seat, identity in enumerate(identities):
                if weight == 1:
                    self.ratings[identity] = updated[seat][0]
                else:
                    mu, sigma = discounted_posterior(
                        *prior[seat], updated[seat][0].mu, updated[seat][0].sigma, weight,
                    )
                    self.ratings[identity] = self.model.rating(mu=mu, sigma=sigma, name=identity)
                row = self.counters[identity]
                row["rated_games"] += 1
                row["weighted_games"] += weight
                row["rating_blocks"].add(block["id"])
                row["opponents"].update(other for other in identities if other != identity)
                row["has_beaten"] |= any(rank > ranks[seat] for rank in ranks)
                row["has_lost"] |= any(rank < ranks[seat] for rank in ranks)
    def rows(self, reference: str) -> dict:
        connected = {reference} if reference in self.ratings else set()
        frontier = list(connected)
        while frontier:
            for other in self.counters[frontier.pop()]["opponents"] - connected:
                connected.add(other)
                frontier.append(other)
        rows = {}
        for identity, counter in self.counters.items():
            row = dict(counter)
            rating = self.ratings.get(identity)
            row["mu"] = None if rating is None else rating.mu
            row["sigma"] = None if rating is None else rating.sigma
            row["rating"] = 1500 + 40 * (rating.mu - self.ratings[reference].mu) if identity in connected else None
            row["mean_penalty"] = row["penalty_sum"] / row["games"] if row["games"] else None
            row["declaration_rate"] = row["declaration_wins"] / row["games"] if row["games"] else None
            row["blocks"] = len(row["blocks"])
            row["rating_blocks"] = len(row["rating_blocks"])
            row["opponents"] = len(row["opponents"])
            row["interval95"] = None
            row["is_reference"] = identity == reference
            row["status"] = (
                "unrated" if rating is None else "unconnected" if identity not in connected
                else "anchor" if identity == reference else "provisional"
            )
            rows[identity] = row
        return rows


def _quantile(values: Sequence[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _validate_weight(weight: float) -> None:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("evidence weight must be a finite number in [0, 1]")


def discounted_posterior(
    prior_mu: float, prior_sigma: float, posterior_mu: float, posterior_sigma: float, weight: float,
) -> tuple[float, float]:
    _validate_weight(weight)
    for name, value in (
        ("prior_mu", prior_mu), ("prior_sigma", prior_sigma),
        ("posterior_mu", posterior_mu), ("posterior_sigma", posterior_sigma),
    ):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or name.endswith("sigma") and value <= 0
        ):
            raise ValueError("invalid Gaussian parameter: " + name)
    if weight == 0:
        return prior_mu, prior_sigma
    if weight == 1:
        return posterior_mu, posterior_sigma
    prior_precision = 1 / (prior_sigma * prior_sigma)
    posterior_precision = 1 / (posterior_sigma * posterior_sigma)
    precision = (1 - weight) * prior_precision + weight * posterior_precision
    information = (1 - weight) * prior_mu * prior_precision + weight * posterior_mu * posterior_precision
    if not math.isfinite(precision) or precision <= 0 or not math.isfinite(information):
        raise ValueError("invalid Gaussian evidence update")
    return information / precision, math.sqrt(1 / precision)


def write_rating_report(
    database: Path, output: Path, reference: str, cap_policy: str,
    protocol: Optional[str] = None, samples: int = 200, seed: int = 81,
    cap_weight: float = 0.25,
) -> dict:
    database, output = Path(database).resolve(), Path(output).resolve()
    reserved = {
        database, Path(str(database) + "-wal"), Path(str(database) + "-shm"),
        Path(str(database) + ".lock"), database.parent / "status.json",
        database.parent / "standings.json", database.parent / "standings-completed-only.json",
        database.parent / "models" / "registry.json",
    }
    if output.suffix.lower() != ".json" or output in reserved or output.exists() and output.samefile(database):
        raise ValueError("report output must be a separate JSON file, not coordinator state")
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_writer(output.with_suffix(output.suffix + ".lock")):
        with RatingStore(database, read_only=True) as store:
            profiles = store.protocols()
            if protocol is not None:
                if protocol not in profiles:
                    raise ValueError("unknown rating protocol")
                profiles = {protocol: profiles[protocol]}
            result = {
                "version": 1, "generated_unix_time": time.time(),
                "reports": [
                    store.report(identity, reference, cap_policy, samples, seed, cap_weight)
                    for identity in profiles
                ],
            }
        atomic_write_json(result, output)
    return result
