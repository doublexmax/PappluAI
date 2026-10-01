from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class ModelRecord:
    id: str
    sha256: str
    filename: str
    origin: str
    metadata: Dict[str, Any]
    evidence: Dict[str, Any]
    parents: tuple


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ModelRegistry:
    """One writer owns registration and compare-and-swap promotion."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.champion_path = self.root / "champion.json"
        if not self.path.exists():
            atomic_json(self.path, {"version": 1, "models": {}})
        self._read()

    def _read(self) -> dict:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("models"), dict):
            raise ValueError("Unsupported model registry")
        return data

    def get(self, record_id: str) -> ModelRecord:
        try:
            raw = self._read()["models"][record_id]
        except KeyError as error:
            raise ValueError("Unknown model id: %s" % record_id) from error
        record = ModelRecord(**{**raw, "parents": tuple(raw["parents"])})
        if (
            record.id != record_id
            or record.filename != "model-" + record.sha256 + ".pt"
            or len(record.sha256) != 64
            or any(character not in "0123456789abcdef" for character in record.sha256)
        ):
            raise ValueError("Invalid model record: %s" % record_id)
        return record

    def model_path(self, record_id: str) -> Path:
        record = self.get(record_id)
        path = self.root / record.filename
        if file_sha256(path) != record.sha256:
            raise ValueError("Model checksum mismatch: %s" % record_id)
        return path

    def register(
        self,
        path: str,
        origin: str,
        evidence: Optional[dict] = None,
        parents: Sequence[str] = (),
    ) -> ModelRecord:
        from src.model import load_checkpoint

        source = Path(path)
        digest = file_sha256(source)
        identifier = digest[:20]
        database = self._read()
        if identifier in database["models"]:
            existing = self.get(identifier)
            if existing.sha256 != digest:
                raise ValueError("Model identifier collision")
            self.model_path(identifier)
            return existing
        network, config, _ = load_checkpoint(str(source))
        if (config.num_decks, config.cards_in_hand, config.required_sequences) != (3, 21, 5):
            raise ValueError("League models must use three decks, 21 cards, and five pure sequences")
        for parent in parents:
            self.get(parent)
        if not isinstance(origin, str) or not origin:
            raise ValueError("Model origin must be a nonempty string")
        filename = "model-" + digest + ".pt"
        record = ModelRecord(
            identifier, digest, filename, origin,
            {
                "architecture": network.architecture,
                "state_dim": network.state_dim,
                "num_actions": network.num_actions,
                "game_config": config.to_dict(),
            },
            dict(evidence or {}), tuple(parents),
        )
        destination = self.root / filename
        if destination.exists():
            if file_sha256(destination) != digest:
                raise ValueError("An immutable model file was modified")
        else:
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            if file_sha256(temporary) != digest:
                raise ValueError("Model changed while its snapshot was being copied")
            os.replace(temporary, destination)
        database["models"][identifier] = asdict(record)
        atomic_json(self.path, database)
        return record

    def initialize_champion(self, record_id: str) -> None:
        self.model_path(record_id)
        if self.champion_path.exists():
            if self.champion().id != record_id:
                raise ValueError("The initial champion is already pinned")
            return
        atomic_json(
            self.champion_path,
            {"version": 1, "generation": 0, "id": record_id, "history": []},
        )

    def champion(self) -> ModelRecord:
        data = json.loads(self.champion_path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("history"), list):
            raise ValueError("Unsupported champion pointer")
        record = self.get(data["id"])
        self.model_path(record.id)
        return record

    def promote(self, candidate_id: str, expected_id: str, evidence: dict) -> ModelRecord:
        current = self.champion()
        candidate = self.get(candidate_id)
        self.model_path(candidate.id)
        if current.id != expected_id:
            raise ValueError("Champion changed before promotion")
        if candidate.id == current.id:
            raise ValueError("A model cannot replace itself")
        if (
            evidence.get("accepted") is not True
            or evidence.get("selection_passed") is not True
            or evidence.get("phase") != "confirmation"
            or evidence.get("candidate_sha256") != candidate.sha256
            or evidence.get("champion_sha256") != current.sha256
        ):
            raise ValueError("Promotion requires accepted selection and confirmation for these exact models")
        if tuple(evidence.get("player_counts", ())) != (2, 3):
            raise ValueError("This champion gate requires two- and three-player evidence")
        rules = evidence.get("rules")
        if not isinstance(rules, dict) or rules.get("recycle_discard") is not True:
            raise ValueError("Promotion evidence must use discard-recycling rules")
        multiplayer = evidence.get("multiplayer")
        solo = evidence.get("solo")
        if (
            not isinstance(multiplayer, dict) or set(multiplayer) != {"2", "3"}
            or not isinstance(solo, dict)
        ):
            raise ValueError("Both player-count results and solo results are required")
        for result in multiplayer.values():
            value = result.get("candidate_minus_champion") if isinstance(result, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Each player count must retain or improve league performance")
        blocks = evidence.get("seed_blocks")
        games = solo.get("games")
        seed = evidence.get("seed")
        selection_seed = evidence.get("selection_seed")
        if (
            isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 32
            or isinstance(games, bool) or not isinstance(games, int) or games < 256
            or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(selection_seed, bool) or not isinstance(selection_seed, int)
            or abs(seed - selection_seed) < max(blocks, games) + 100_000
        ):
            raise ValueError("Promotion requires sufficiently sized, disjoint selection and confirmation deals")
        for name in ("multiplayer_gain", "solo_gain"):
            value = evidence.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Missing finite promotion metric: %s" % name)
        for name in ("multiplayer_ci95", "solo_ci95"):
            values = evidence.get(name)
            if (
                not isinstance(values, (list, tuple)) or len(values) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
                or values[0] > values[1]
            ):
                raise ValueError("Invalid promotion interval: %s" % name)
        from src.promotion import evidence_metrics

        samples = evidence.get("bootstrap_samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
            raise ValueError("Missing bootstrap sample count")
        recomputed = evidence_metrics(multiplayer, solo, blocks, seed, samples)
        for name, computed in recomputed.items():
            declared = evidence[name]
            pairs = zip(declared, computed) if isinstance(computed, list) else ((declared, computed),)
            if any(not math.isclose(left, right, abs_tol=1e-12) for left, right in pairs):
                raise ValueError("Promotion summary disagrees with raw evidence: %s" % name)
        if (
            evidence["multiplayer_gain"] < 0.05
            or evidence["multiplayer_ci95"][0] <= 0
            or evidence["solo_gain"] < 0
            or evidence["solo_ci95"][0] < -0.03
        ):
            raise ValueError("Candidate failed multiplayer improvement or solo retention")
        pointer = json.loads(self.champion_path.read_text(encoding="utf-8"))
        if pointer["id"] != expected_id:
            raise ValueError("Champion changed before the registry transaction")
        pointer["history"].append({
            "previous_id": current.id,
            "candidate_id": candidate.id,
            "evidence": evidence,
        })
        pointer["generation"] += 1
        pointer["id"] = candidate.id
        atomic_json(self.champion_path, pointer)
        return candidate
