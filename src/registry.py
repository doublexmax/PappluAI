from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Sequence

from src.checkpoints import atomic_write_json


@dataclass(frozen=True)
class ModelRecord:
    id: str
    sha256: str
    filename: str
    origin: str
    metadata: Dict[str, Any]
    evidence: Dict[str, Any]
    parents: tuple


def file_metadata(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def file_sha256(path: Path) -> str:
    return str(file_metadata(path)["sha256"])


class ModelRegistry:
    """One writer owns registration and compare-and-swap promotion."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.champion_path = self.root / "champion.json"
        if not self.path.exists():
            atomic_write_json({"version": 1, "models": {}}, self.path)
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
        atomic_write_json(database, self.path)
        return record

    def initialize_champion(self, record_id: str) -> None:
        self.model_path(record_id)
        if self.champion_path.exists():
            if self.champion().id != record_id:
                raise ValueError("The initial champion is already pinned")
            return
        atomic_write_json(
            {"version": 1, "generation": 0, "id": record_id, "history": []},
            self.champion_path,
        )

    def champion(self) -> ModelRecord:
        data = json.loads(self.champion_path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("history"), list):
            raise ValueError("Unsupported champion pointer")
        record = self.get(data["id"])
        self.model_path(record.id)
        return record

    def promote(self, candidate_id: str, expected_id: str, evidence: dict) -> ModelRecord:
        from src.promotion import (
            MIN_CONFIRMATION_SEED_BLOCKS,
            MIN_SELECTION_SEED_BLOCKS,
            validate_gate_evidence,
        )

        current = self.champion()
        candidate = self.get(candidate_id)
        self.model_path(candidate.id)
        if current.id != expected_id:
            raise ValueError("Champion changed before promotion")
        if candidate.id == current.id:
            raise ValueError("A model cannot replace itself")
        if (
            not isinstance(evidence, dict)
            or evidence.get("selection_passed") is not True
        ):
            raise ValueError("Promotion requires a passed selection gate")
        confirmation = validate_gate_evidence(
            evidence,
            phase="confirmation",
            candidate_sha256=candidate.sha256,
            champion_sha256=current.sha256,
            minimum_seed_blocks=MIN_CONFIRMATION_SEED_BLOCKS,
        )
        selection = validate_gate_evidence(
            evidence.get("selection"),
            phase="selection",
            candidate_sha256=candidate.sha256,
            champion_sha256=current.sha256,
            minimum_seed_blocks=MIN_SELECTION_SEED_BLOCKS,
        )
        if (
            evidence.get("selection_seed") != selection.seed
            or selection.rules != confirmation.rules
            or abs(confirmation.seed - selection.seed)
            < 100_000
            + max(
                confirmation.seed_blocks,
                confirmation.solo_games,
                selection.seed_blocks,
                selection.solo_games,
            )
        ):
            raise ValueError(
                "Promotion requires disjoint selection and confirmation deals"
            )
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
        atomic_write_json(pointer, self.champion_path)
        return candidate
