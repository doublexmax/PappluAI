from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.checkpoints.io import require_checkpoint_fields, require_checkpoint_int, file_sha256
from src.checkpoints.tensor import atomic_torch_save
from src.evaluation.arena import (
    CheckpointPolicy, Deal, EpsilonPolicy, MatchInterrupted, RandomPolicy, run_matches,
)
from src.game.environment import ENCODING_VERSION, GameConfig
from src.game.outcome import DealOutcome
from src.model.network import build_checkpoint, load_checkpoint
from src.model.search import TeacherConfig
from src.training.core import Trajectory
from src.training.optimization import Learner, LearnerConfig, LossWeights


LEAGUE_STATE_VERSION = 4
ALGORITHM = "champion_league_nstep_ddqn"
RANDOM_OPPONENT = "random"
TRAINING_SEED_BASE = 1_000_000_000_000


@dataclass(frozen=True)
class PfspConfig:
    champion_share: float = 0.35
    anchor_share: float = 0.15
    exponent: float = 2.0
    prior_games: float = 4.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if self.champion_share + self.anchor_share > 1:
            raise ValueError("PFSP fixed shares cannot exceed one")
        if self.exponent <= 0 or self.prior_games <= 0:
            raise ValueError("PFSP exponent and prior_games must be positive")


class OpponentScheduler:
    def __init__(self, config: PfspConfig = PfspConfig()) -> None:
        self.config = config
        self._scores: dict[tuple[int, str], tuple[float, int]] = {}

    def record(self, players: int, opponent_sha256: str, score: float) -> None:
        if type(players) is not int or not 2 <= players <= 6 or not isinstance(opponent_sha256, str) or not opponent_sha256:
            raise ValueError("PFSP needs a player count and opponent identity")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("PFSP placement score must be in [0, 1]")
        key = (players, opponent_sha256)
        total, games = self._scores.get(key, (0.0, 0))
        self._scores[key] = (total + score, games + 1)

    def update(self, outcome: DealOutcome, challenger_seat: int, opponent_hashes: Sequence[str]) -> None:
        players = len(outcome.penalties)
        if len(opponent_hashes) != players:
            raise ValueError("PFSP needs one identity per seat")
        for seat, identity in enumerate(opponent_hashes):
            if seat != challenger_seat:
                self.record(players, identity, (outcome.margin(challenger_seat, seat) + 1.0) * 0.5)

    def probabilities(
        self, players: int, opponents: Sequence[str], champion: str,
        anchors: Sequence[str] = (RANDOM_OPPONENT,),
    ) -> tuple[float, ...]:
        if not opponents or len(set(opponents)) != len(opponents):
            raise ValueError("PFSP opponents must be nonempty and distinct")
        champion_rows = [row for row, identity in enumerate(opponents) if identity == champion]
        anchor_rows = [row for row, identity in enumerate(opponents) if identity in anchors and identity != champion]
        pfsp_rows = [row for row in range(len(opponents)) if row not in champion_rows + anchor_rows]
        weights = [0.0] * len(opponents)
        remaining = 1.0
        for rows, share in (
            (champion_rows, self.config.champion_share), (anchor_rows, self.config.anchor_share),
        ):
            if rows:
                for row in rows:
                    weights[row] += share / len(rows)
                remaining -= share
        if not pfsp_rows:
            total = sum(weights)
            return tuple(weight / total if total else 1.0 / len(opponents) for weight in weights)
        hardness = []
        for row in pfsp_rows:
            total, games = self._scores.get((players, opponents[row]), (0.0, 0))
            rate = (total + self.config.prior_games * 0.5) / (games + self.config.prior_games)
            hardness.append((1.0 - rate) ** self.config.exponent)
        normalizer = sum(hardness)
        for row, value in zip(pfsp_rows, hardness):
            weights[row] += remaining * value / normalizer
        return tuple(weights)

    def choose(
        self, players: int, opponents: Sequence[str], champion: str,
        rng: random.Random, anchors: Sequence[str] = (RANDOM_OPPONENT,),
    ) -> int:
        return rng.choices(
            range(len(opponents)), weights=self.probabilities(players, opponents, champion, anchors), k=1,
        )[0]

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "scores": tuple((players, identity, total, games)
                            for (players, identity), (total, games) in sorted(self._scores.items())),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != {"config", "scores"} or state["config"] != asdict(self.config):
            raise ValueError("PFSP state config changed")
        if not isinstance(state["scores"], tuple):
            raise ValueError("PFSP scores must be a tuple")
        scores = {}
        for row in state["scores"]:
            if not isinstance(row, tuple) or len(row) != 4:
                raise ValueError("PFSP score row is invalid")
            players, identity, total, games = row
            if (
                type(players) is not int or not 2 <= players <= 6
                or not isinstance(identity, str) or not identity
                or type(games) is not int or games < 1
                or type(total) not in (int, float) or not math.isfinite(total) or not 0 <= total <= games
                or (players, identity) in scores
            ):
                raise ValueError("PFSP score row is invalid")
            scores[players, identity] = (float(total), games)
        self._scores = scores


@dataclass(frozen=True)
class LeagueConfig:
    game: GameConfig = field(default_factory=lambda: GameConfig(max_turns=60))
    batch_size: int = 128
    replay_capacity: int = 50_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    n_step: int = 5
    target_tau: float = 0.005
    updates_per_match: int = 4
    matches_per_round: int = 16
    grad_clip: float = 1.0
    losses: LossWeights = LossWeights()
    teacher: TeacherConfig = TeacherConfig()
    pfsp: PfspConfig = PfspConfig()
    epsilon_start: float = 0.2
    epsilon_end: float = 0.05
    epsilon_decay_matches: int = 10_000
    player_counts: Tuple[int, ...] = (2, 3, 4)

    def __post_init__(self) -> None:
        if (self.game.num_decks, self.game.cards_in_hand, self.game.required_sequences) != (3, 21, 5):
            raise ValueError("league training requires 3 decks, 21 cards, and 5 pure sequences")
        self.learner_config
        for name in ("updates_per_match", "matches_per_round", "epsilon_decay_matches"):
            require_checkpoint_int(getattr(self, name), name, minimum=1)
        for name in ("epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("%s must be in [0, 1]" % name)
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        if (
            not isinstance(self.player_counts, tuple) or not self.player_counts
            or any(type(count) is not int or count not in (2, 3, 4) for count in self.player_counts)
            or tuple(sorted(set(self.player_counts))) != self.player_counts
        ):
            raise ValueError("player_counts must be distinct and sorted from 2/3/4")
        if not isinstance(self.pfsp, PfspConfig):
            raise TypeError("pfsp must be a PfspConfig")

    @property
    def learner_config(self) -> LearnerConfig:
        return LearnerConfig(**{
            name: getattr(self, name) for name in LearnerConfig.__dataclass_fields__
        })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LeagueConfig":
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("league_config fields do not match this trainer")
        values = dict(raw)
        values["game"] = GameConfig.from_dict(values["game"])
        for name, kind in (("losses", LossWeights), ("teacher", TeacherConfig), ("pfsp", PfspConfig)):
            if not isinstance(values[name], dict) or set(values[name]) != set(kind.__dataclass_fields__):
                raise ValueError("league %s fields do not match" % name)
            values[name] = kind(**values[name])
        return cls(**values)


@dataclass(frozen=True)
class _Opponent:
    label: str
    sha256: str
    policy: Any


class LeagueSession:
    def __init__(
        self, initial_model: str, opponent_paths: Sequence[str] = (),
        config: Optional[LeagueConfig] = None, seed: int = 41,
    ) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an int")
        self.config = LeagueConfig() if config is None else config
        self.initial_seed = seed
        self.parent_sha256 = file_sha256(Path(initial_model))
        self.anchor_sha256 = self.parent_sha256
        self.last_exported_sha256: Optional[str] = None
        with torch.random.fork_rng(devices=[]):
            network, saved_game, _ = load_checkpoint(initial_model, device=torch.device("cpu"))
        _require_compatible_game(saved_game, self.config.game, "initial model")
        seed_source = random.Random(seed)
        self.deal_rng = random.Random(seed_source.getrandbits(64))
        self.opponent_rng = random.Random(seed_source.getrandbits(64))
        self.learner = Learner(network, self.config.learner_config, seed_source.getrandbits(63))
        self.scheduler = OpponentScheduler(self.config.pfsp)
        self._parent_path = str(initial_model)
        self._opponents = _load_opponents(self._parent_path, opponent_paths, self.config.game)
        self.opponent_hashes = tuple(opponent.sha256 for opponent in self._opponents)
        self.opponent_pool_generation = 0
        self.completed_matches = 0
        self.update_matches = 0
        self.wins = self.losses = self.draws = self.no_action_matches = self.epsilon_progress = 0
        self.at_match_boundary = True

    @property
    def network(self):
        return self.learner.online

    @property
    def optimizer(self):
        return self.learner.optimizer

    @property
    def replay(self):
        return self.learner.replay

    @property
    def total_updates(self) -> int:
        return self.learner.update_count

    @property
    def count_matches(self) -> Dict[int, int]:
        cycles, remainder = divmod(self.completed_matches, len(self.config.player_counts))
        return {count: cycles + int(index < remainder) for index, count in enumerate(self.config.player_counts)}

    @property
    def current_epsilon(self) -> float:
        fraction = min(1.0, self.epsilon_progress / float(self.config.epsilon_decay_matches))
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * fraction

    def train_match(self, deadline: Optional[float] = None) -> Dict[str, Any]:
        return self.train_round(1, deadline=deadline)[0]

    def train_round(self, matches: int, deadline: Optional[float] = None) -> list[Dict[str, Any]]:
        require_checkpoint_int(matches, "matches", minimum=1)
        rng_snapshot = (self.deal_rng.getstate(), self.opponent_rng.getstate())
        self.at_match_boundary = False
        collected = False
        try:
            epsilon = self.current_epsilon
            policies = [opponent.policy for opponent in self._opponents] + [EpsilonPolicy(self.network, epsilon)]
            deals, challenger_seats = [], []
            for offset in range(matches):
                occurrence, index = divmod(self.completed_matches + offset, len(self.config.player_counts))
                players = self.config.player_counts[index]
                challenger = occurrence % players
                seats = tuple(
                    len(self._opponents) if seat == challenger else self.scheduler.choose(
                        players, self.opponent_hashes, self.parent_sha256, self.opponent_rng,
                        anchors=(self.anchor_sha256, RANDOM_OPPONENT),
                    )
                    for seat in range(players)
                )
                deals.append(Deal(
                    TRAINING_SEED_BASE + self.deal_rng.getrandbits(39), seats, players, frozenset((challenger,)),
                ))
                challenger_seats.append(challenger)
            results = run_matches(deals, policies, self.config.game, deadline, record_terminal=False)
            if len(results) != matches:
                raise ValueError("arena returned the wrong number of matches")
            collected = True
        except BaseException:
            if not collected:
                self.deal_rng.setstate(rng_snapshot[0])
                self.opponent_rng.setstate(rng_snapshot[1])
                self.at_match_boundary = True
            raise

        trajectories, summaries = [], []
        for deal, challenger, result in zip(deals, challenger_seats, results):
            outcome = result.outcome
            if outcome is None:
                raise ValueError("league match omitted its scored outcome")
            identities = [
                self.parent_sha256 if seat == challenger else self.opponent_hashes[index]
                for seat, index in enumerate(deal.seats)
            ]
            self.scheduler.update(outcome, challenger, identities)
            decisions = result.trajectories[challenger]
            if decisions:
                trajectories.append(Trajectory(deal.players, decisions, outcome.seat(challenger)))
                self.update_matches += 1
            else:
                self.no_action_matches += 1
            self.completed_matches += 1
            self.epsilon_progress += 1
            if result.winner is None:
                self.draws += 1
            elif result.winner == challenger:
                self.wins += 1
            else:
                self.losses += 1
            summaries.append({
                "match": self.completed_matches, "seed": result.seed, "players": deal.players,
                "challenger_seat": challenger, "winner": result.winner, "won": result.winner == challenger,
                "terminal_reason": result.terminal_reason, "actions": len(decisions),
                "updated": bool(decisions), "epsilon": epsilon,
            })
        stats = self.learner.learn(trajectories, updates=self.config.updates_per_match * len(trajectories))
        if stats.updates:
            self.last_exported_sha256 = None
        self.at_match_boundary = True
        return summaries

    def state_dict(self) -> Dict[str, Any]:
        if not self.at_match_boundary:
            raise RuntimeError("cannot checkpoint a partial match")
        return {
            "league_state_version": LEAGUE_STATE_VERSION, "encoding_version": ENCODING_VERSION,
            "algorithm": ALGORITHM, "league_config": self.config.to_dict(),
            "model_spec": self.network.spec.to_dict(), "initial_seed": self.initial_seed,
            "parent_sha256": self.parent_sha256, "anchor_sha256": self.anchor_sha256,
            "last_exported_sha256": self.last_exported_sha256,
            "opponent_hashes": self.opponent_hashes, "opponent_pool_generation": self.opponent_pool_generation,
            "replay_preserved_across_pool_refresh": True, "learner": self.learner.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_states": {"deal": self.deal_rng.getstate(), "opponent": self.opponent_rng.getstate()},
            "counters": {
                name: getattr(self, name) for name in (
                    "completed_matches", "update_matches", "total_updates", "wins", "losses",
                    "draws", "no_action_matches", "epsilon_progress",
                )
            },
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        required = {
            "league_state_version", "encoding_version", "algorithm", "league_config", "model_spec",
            "initial_seed", "parent_sha256", "anchor_sha256", "last_exported_sha256",
            "opponent_hashes", "opponent_pool_generation", "replay_preserved_across_pool_refresh",
            "learner", "scheduler", "rng_states", "counters",
        }
        require_checkpoint_fields(payload, required, "league checkpoint")
        for name, expected in (
            ("league_state_version", LEAGUE_STATE_VERSION), ("encoding_version", ENCODING_VERSION),
            ("algorithm", ALGORITHM), ("model_spec", self.network.spec.to_dict()),
            ("initial_seed", self.initial_seed), ("parent_sha256", self.parent_sha256),
        ):
            if type(payload[name]) is not type(expected) or payload[name] != expected:
                raise ValueError("league checkpoint %s changed" % name)
        if LeagueConfig.from_dict(payload["league_config"]) != self.config:
            raise ValueError("league checkpoint config changed")
        if payload["opponent_hashes"] != self.opponent_hashes:
            raise ValueError("league checkpoint opponent pool changed")
        for name in ("anchor_sha256", "last_exported_sha256"):
            digest = payload[name]
            if digest is None and name == "last_exported_sha256":
                continue
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("league checkpoint %s is invalid" % name)
        generation = require_checkpoint_int(payload["opponent_pool_generation"], "opponent_pool_generation", minimum=0)
        if payload["replay_preserved_across_pool_refresh"] is not True:
            raise ValueError("league checkpoint replay refresh policy changed")
        counters = require_checkpoint_fields(payload["counters"], {
            "completed_matches", "update_matches", "total_updates", "wins", "losses",
            "draws", "no_action_matches", "epsilon_progress",
        }, "counters")
        counters = {name: require_checkpoint_int(value, name, minimum=0) for name, value in counters.items()}
        completed = counters["completed_matches"]
        if counters["wins"] + counters["losses"] + counters["draws"] != completed:
            raise ValueError("league outcomes do not sum to completed matches")
        if counters["update_matches"] + counters["no_action_matches"] != completed:
            raise ValueError("league update counters do not sum to matches")
        if counters["epsilon_progress"] != completed or counters["total_updates"] != counters["update_matches"] * self.config.updates_per_match:
            raise ValueError("league update count or epsilon progress is inconsistent")
        states = require_checkpoint_fields(payload["rng_states"], {"deal", "opponent"}, "rng_states")
        rngs = {}
        for name, state in states.items():
            rng = random.Random()
            try:
                rng.setstate(state)
            except (TypeError, ValueError, IndexError) as error:
                raise ValueError("%s RNG state is invalid" % name) from error
            rngs[name] = rng
        learner = Learner(deepcopy(self.network), self.config.learner_config)
        learner.load_state_dict(payload["learner"])
        if learner.update_count != counters["total_updates"]:
            raise ValueError("league and learner update counts differ")
        scheduler = OpponentScheduler(self.config.pfsp)
        scheduler.load_state_dict(payload["scheduler"])
        self.learner, self.scheduler = learner, scheduler
        self.deal_rng, self.opponent_rng = rngs["deal"], rngs["opponent"]
        self.anchor_sha256 = payload["anchor_sha256"]
        self.last_exported_sha256 = payload["last_exported_sha256"]
        self.opponent_pool_generation = generation
        for name, value in counters.items():
            if name != "total_updates":
                setattr(self, name, value)
        self.at_match_boundary = True

    @classmethod
    def load(
        cls, path: str, initial_model: str, opponent_paths: Sequence[str] = (),
        expected_config: Optional[LeagueConfig] = None,
    ) -> "LeagueSession":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("league checkpoint must be a dict")
        config = LeagueConfig.from_dict(payload.get("league_config", {}))
        if expected_config is not None and config != expected_config:
            raise ValueError("resume league_config does not match requested config")
        session = cls(initial_model, opponent_paths, config, payload.get("initial_seed"))
        session.load_state_dict(payload)
        return session

    def export_model(self, path: Path, role: str = "latest") -> None:
        atomic_torch_save(build_checkpoint(self.network, self.config.game, meta={
            "algorithm": ALGORITHM, "role": role, "matches": self.completed_matches,
            "updates": self.total_updates, "parent_sha256": self.parent_sha256,
            "opponent_hashes": self.opponent_hashes, "opponent_pool_generation": self.opponent_pool_generation,
            "replay_preserved_across_pool_refresh": True,
        }), path)
        self.last_exported_sha256 = file_sha256(Path(path))

    def follow_champion(self, champion_path: str, *, force_rebase: bool = False) -> bool:
        if not self.at_match_boundary:
            raise RuntimeError("cannot follow a champion during a match")
        digest = file_sha256(Path(champion_path))
        if not force_rebase and digest in (self.parent_sha256, self.last_exported_sha256):
            self.parent_sha256, self._parent_path = digest, str(champion_path)
            return False
        with torch.random.fork_rng(devices=[]):
            network, game, _ = load_checkpoint(champion_path, device=torch.device("cpu"))
        _require_compatible_game(game, self.config.game, "champion")
        self.learner.adopt(network.state_dict())
        self.last_exported_sha256 = None
        self.parent_sha256, self._parent_path = digest, str(champion_path)
        return True

    def refresh_opponent_pool(self, opponent_paths: Sequence[str]) -> None:
        if not self.at_match_boundary:
            raise RuntimeError("cannot refresh opponents during a match")
        opponents = _load_opponents(self._parent_path, opponent_paths, self.config.game)
        hashes = tuple(opponent.sha256 for opponent in opponents)
        if hashes != self.opponent_hashes:
            self._opponents, self.opponent_hashes = opponents, hashes
            self.opponent_pool_generation += 1


def save_session_checkpoint(session: LeagueSession, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session.export_model(output_dir / "latest-model.pt")
    atomic_torch_save(session.state_dict(), output_dir / "latest-state.pt")


def _require_compatible_game(actual: GameConfig, expected: GameConfig, label: str) -> None:
    if (actual.num_decks, actual.cards_in_hand, actual.required_sequences) != (
        expected.num_decks, expected.cards_in_hand, expected.required_sequences,
    ):
        raise ValueError("%s must use 3 decks, 21 cards, and 5 sequences" % label)


def _load_opponents(initial_model: str, opponent_paths: Sequence[str], game: GameConfig) -> Tuple[_Opponent, ...]:
    opponents, seen = [], set()
    for path in (initial_model, *opponent_paths, RANDOM_OPPONENT):
        if path == RANDOM_OPPONENT:
            if path not in seen:
                opponents.append(_Opponent(path, path, RandomPolicy()))
                seen.add(path)
            continue
        digest = file_sha256(Path(path))
        if digest in seen:
            continue
        with torch.random.fork_rng(devices=[]):
            policy = CheckpointPolicy(str(path))
            _require_compatible_game(policy.game_config, game, "opponent %s" % path)
        opponents.append(_Opponent(str(path), digest, policy))
        seen.add(digest)
    return tuple(opponents)
