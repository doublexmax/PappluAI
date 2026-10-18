"""Finite champion-league improvement cycles with guarded promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.checkpoints import evidence
from src.checkpoints.io import (
    atomic_copy,
    atomic_write_json,
    file_metadata,
    file_sha256,
    require_checkpoint_fields,
    require_checkpoint_int,
)
from src.checkpoints.registry import ModelRegistry
from src.checkpoints.tensor import atomic_torch_save
from src.evaluation.arena import MatchInterrupted
from src.evaluation.promotion import GateInterrupted, evaluate_gate
from src.game.environment import ENCODING_VERSION, GameConfig
from src.model.network import CHECKPOINT_VERSION, build_checkpoint
from src.model.spec import ARCHITECTURE
from src.training.core import TerminalShaping
from src.training.league import LeagueConfig, LeagueSession, RANDOM_OPPONENT
from src.training.curriculum import (
    STATUS_SCHEMA_VERSION,
    TrainingConfig,
    TrainingSession,
)


IMPROVEMENT_STATE_VERSION = 2
ALGORITHM = "champion_league_mc"
PHASES = ("training", "selection", "confirmation")
LEAGUE_FINETUNE_LR = 2e-5
LEAGUE_FINETUNE_EPSILON_START = 0.04
LEAGUE_FINETUNE_EPSILON_END = 0.01
RESEARCH_FINETUNE_LR = 2e-5
RESEARCH_FINETUNE_EPSILON_START = 0.04
RESEARCH_FINETUNE_EPSILON_END = 0.01
DEFAULT_IMPROVE_SHAPING = TerminalShaping(speed=0.35, margin=0.25)


@dataclass(frozen=True)
class ImproveConfig:
    max_cycles: int = 64
    league_matches_per_cycle: int = 128
    research_episodes_per_cycle: int = 128
    seed: int = 41
    selection_blocks: int = evidence.MIN_SELECTION_SEED_BLOCKS
    confirmation_blocks: int = evidence.MIN_CONFIRMATION_SEED_BLOCKS
    solo_games: int = evidence.MIN_SOLO_GAMES
    checkpoint_every: int = 16
    max_turns: int = 60
    player_counts: Tuple[int, ...] = (2, 3, 4)
    gate_samples: int = 2_000
    recycle_discard: bool = True

    def __post_init__(self) -> None:
        if type(self.recycle_discard) is not bool:
            raise TypeError("recycle_discard must be a boolean")
        for name in (
            "max_cycles",
            "league_matches_per_cycle",
            "research_episodes_per_cycle",
            "selection_blocks",
            "confirmation_blocks",
            "solo_games",
            "checkpoint_every",
            "max_turns",
            "gate_samples",
        ):
            require_checkpoint_int(getattr(self, name), name, minimum=1)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(self.player_counts, tuple):
            raise TypeError("player_counts must be a tuple")
        evidence.require_player_counts(self.player_counts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ImproveConfig":
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("improve_config fields do not match this controller")
        values = dict(raw)
        if not isinstance(values["player_counts"], tuple):
            raise TypeError("player_counts must be a tuple")
        return cls(**values)


class ImprovementController:
    def __init__(
        self,
        initial_model: str,
        output_dir: Path,
        opponent_paths: Sequence[str] = (),
        config: Optional[ImproveConfig] = None,
        snapshots_dir: Optional[Path] = None,
    ) -> None:
        self.config = ImproveConfig() if config is None else config
        self.snapshots_dir = None if snapshots_dir is None else Path(snapshots_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any((self.output_dir / name).exists() for name in ("latest-state.pt", "status.json")):
            raise ValueError("Training outputs already exist; use --resume")
        self.metric_ids = _read_metric_ids(self.output_dir / ".active-metrics.jsonl")
        self.registry = ModelRegistry(self.output_dir)

        initial_evidence = _initial_evidence(
            Path(initial_model),
            self.config.recycle_discard,
        )
        baseline = self.registry.register(
            initial_model,
            origin="baseline",
            evidence=initial_evidence,
        )
        self.registry.initialize_champion(baseline.id)
        opponent_ids = [baseline.id]
        for path in opponent_paths:
            if path == RANDOM_OPPONENT:
                continue
            record = self.registry.register(
                path,
                origin="frozen_opponent",
            )
            if record.id not in opponent_ids:
                opponent_ids.append(record.id)
        self.baseline_id = baseline.id
        self.opponent_ids = _bounded_pool(tuple(opponent_ids), baseline.id)

        baseline_path = str(self.registry.model_path(self.baseline_id))
        league_paths = self._opponent_paths()
        batch_size = max(1, int(os.environ.get("PAPPLU_BATCH_SIZE", "64")))
        updates_per_step = max(1, int(os.environ.get("PAPPLU_UPDATES_PER_STEP", "4")))
        replay_capacity = max(100, int(os.environ.get("PAPPLU_REPLAY_CAPACITY", "10000")))
        scale_lr = os.environ.get("PAPPLU_SCALE_LR_BY_WIDTH") == "1"
        lr_scale = 1.0
        if scale_lr:
            from src.model.network import load_checkpoint
            w = load_checkpoint(baseline_path)[0].spec.width
            lr_scale = 64.0 / max(64.0, float(w))
        self.league = LeagueSession(
            baseline_path,
            league_paths,
            config=LeagueConfig(
                game=GameConfig(max_turns=self.config.max_turns, recycle_discard=self.config.recycle_discard),
                batch_size=batch_size,
                replay_capacity=replay_capacity,
                updates_per_match=updates_per_step,
                player_counts=self.config.player_counts,
                learning_rate=LEAGUE_FINETUNE_LR * lr_scale,
                epsilon_start=LEAGUE_FINETUNE_EPSILON_START,
                epsilon_end=LEAGUE_FINETUNE_EPSILON_END,
                shaping=DEFAULT_IMPROVE_SHAPING,
            ),
            seed=self.config.seed,
        )
        self._ingest_snapshots()
        self.research = TrainingSession(
            config=_research_config(self.config, lr_scale=lr_scale),
            seed=self.config.seed + 1,
            initial_model=baseline_path,
            initial_stage_index=2,
        )

        self.cycle_index = 0
        self.phase = "training"
        self.league_in_cycle = 0
        self.research_in_cycle = 0
        self.next_learner = "league"
        self.candidate_ids: Dict[str, str] = {}
        self.selection_results: Dict[str, Dict[str, Any]] = {}
        self.nominee_origin: Optional[str] = None
        self.nominee_id: Optional[str] = None
        self.confirmation_result: Optional[Dict[str, Any]] = None
        self.champion_at_cycle_id: Optional[str] = None
        self.promotions = 0
        self.promoted_cycles: Tuple[int, ...] = ()
        self.last_gate: Optional[Dict[str, Any]] = None
        self.best_score = initial_evidence.get(
            "current_solo_reference",
            {},
        ).get("win_rate")

    @classmethod
    def load(
        cls,
        path: str,
        output_dir: Path,
        snapshots_dir: Optional[Path] = None,
    ) -> "ImprovementController":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "improvement_state_version",
            "encoding_version",
            "algorithm",
            "config",
            "baseline_id",
            "opponent_ids",
            "cycle",
            "phase",
            "cursors",
            "evaluation",
            "promotions",
            "promoted_cycles",
            "last_gate",
            "best_score",
            "league_state",
            "research_state",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("improvement checkpoint fields do not match controller")
        if payload["improvement_state_version"] != IMPROVEMENT_STATE_VERSION:
            raise ValueError("improvement checkpoint version is incompatible")
        if payload["encoding_version"] != ENCODING_VERSION:
            raise ValueError("improvement checkpoint encoding is incompatible")
        if payload["algorithm"] != ALGORITHM:
            raise ValueError("improvement checkpoint algorithm is incompatible")

        controller = cls.__new__(cls)
        controller.config = ImproveConfig.from_dict(payload["config"])
        controller.snapshots_dir = None if snapshots_dir is None else Path(snapshots_dir)
        controller.output_dir = Path(output_dir)
        if (controller.output_dir / "metrics.jsonl").exists():
            atomic_copy(
                controller.output_dir / "metrics.jsonl",
                controller.output_dir / ".active-metrics.jsonl",
            )
        controller.metric_ids = _read_metric_ids(controller.output_dir / ".active-metrics.jsonl")
        controller.registry = ModelRegistry(controller.output_dir)
        controller.baseline_id = _record_id(payload["baseline_id"], "baseline_id")
        raw_opponents = payload["opponent_ids"]
        if not isinstance(raw_opponents, tuple) or not raw_opponents:
            raise ValueError("opponent_ids must be a nonempty tuple")
        controller.opponent_ids = tuple(
            _record_id(value, "opponent_ids") for value in raw_opponents
        )
        for record_id in controller.opponent_ids:
            controller.registry.model_path(record_id)
        controller.registry.model_path(controller.baseline_id)

        controller.cycle_index = require_checkpoint_int(
            payload["cycle"],
            "cycle",
            minimum=0,
        )
        controller.phase = payload["phase"]
        if controller.phase not in PHASES:
            raise ValueError("improvement phase is invalid")
        cursors = require_checkpoint_fields(
            payload["cursors"],
            {"league", "research", "next_learner"},
            "cursors",
        )
        controller.league_in_cycle = require_checkpoint_int(
            cursors["league"],
            "league cursor",
            minimum=0,
        )
        controller.research_in_cycle = require_checkpoint_int(
            cursors["research"],
            "research cursor",
            minimum=0,
        )
        controller.next_learner = cursors["next_learner"]
        if controller.next_learner not in ("league", "research"):
            raise ValueError("next_learner is invalid")

        evaluation = require_checkpoint_fields(
            payload["evaluation"],
            {
                "candidate_ids",
                "selection_results",
                "nominee_origin",
                "nominee_id",
                "confirmation_result",
                "champion_at_cycle_id",
                "pending_gate_seed",
            },
            "evaluation",
        )
        controller.candidate_ids = _string_dict(
            evaluation["candidate_ids"], "candidate_ids"
        )
        controller.selection_results = _mapping_dict(
            evaluation["selection_results"], "selection_results"
        )
        controller.nominee_origin = _optional_string(
            evaluation["nominee_origin"], "nominee_origin"
        )
        controller.nominee_id = _optional_string(
            evaluation["nominee_id"], "nominee_id"
        )
        controller.confirmation_result = _optional_dict(
            evaluation["confirmation_result"], "confirmation_result"
        )
        controller.champion_at_cycle_id = _optional_string(
            evaluation["champion_at_cycle_id"], "champion_at_cycle_id"
        )
        expected_seed = controller.pending_gate_seed
        if evaluation["pending_gate_seed"] != expected_seed:
            raise ValueError("pending gate seed is inconsistent")

        controller.promotions = require_checkpoint_int(
            payload["promotions"],
            "promotions",
            minimum=0,
        )
        raw_promoted = payload["promoted_cycles"]
        if not isinstance(raw_promoted, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_promoted
        ):
            raise ValueError("promoted_cycles is invalid")
        controller.promoted_cycles = raw_promoted
        controller.last_gate = _optional_dict(payload["last_gate"], "last_gate")
        score = payload["best_score"]
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= score <= 1
        ):
            raise ValueError("best_score is invalid")
        controller.best_score = None if score is None else float(score)

        baseline_path = str(controller.registry.model_path(controller.baseline_id))
        league_config = LeagueConfig.from_dict(
            payload["league_state"]["league_config"]
        )
        controller.league = LeagueSession(
            baseline_path,
            controller._opponent_paths(),
            config=league_config,
            seed=payload["league_state"]["initial_seed"],
        )
        controller.league.load_state_dict(payload["league_state"])
        controller.research = TrainingSession.from_state_dict(
            payload["research_state"],
            expected_config=TrainingConfig.from_dict(payload["research_state"]["training_config"]),
        )
        controller._validate_cursors()
        controller._reconcile_promotion()
        return controller

    @property
    def pending_gate_seed(self) -> Optional[int]:
        if self.phase == "selection":
            return 60_000_000 + self.cycle_index * 1_000_000
        if self.phase == "confirmation":
            return 60_500_000 + self.cycle_index * 1_000_000
        return None

    def state_dict(self) -> Dict[str, Any]:
        self._validate_cursors()
        return {
            "improvement_state_version": IMPROVEMENT_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "algorithm": ALGORITHM,
            "config": self.config.to_dict(),
            "baseline_id": self.baseline_id,
            "opponent_ids": self.opponent_ids,
            "cycle": self.cycle_index,
            "phase": self.phase,
            "cursors": {
                "league": self.league_in_cycle,
                "research": self.research_in_cycle,
                "next_learner": self.next_learner,
            },
            "evaluation": {
                "candidate_ids": dict(self.candidate_ids),
                "selection_results": dict(self.selection_results),
                "nominee_origin": self.nominee_origin,
                "nominee_id": self.nominee_id,
                "confirmation_result": self.confirmation_result,
                "champion_at_cycle_id": self.champion_at_cycle_id,
                "pending_gate_seed": self.pending_gate_seed,
            },
            "promotions": self.promotions,
            "promoted_cycles": self.promoted_cycles,
            "last_gate": self.last_gate,
            "best_score": self.best_score,
            "league_state": self.league.state_dict(),
            "research_state": self.research.state_dict(),
        }

    def run(
        self,
        max_seconds: float,
        stop_requested: Any = lambda: False,
    ) -> str:
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, (int, float))
            or not math.isfinite(float(max_seconds))
            or max_seconds < 0
        ):
            raise ValueError("max_seconds must be finite and nonnegative")
        started = time.monotonic()
        deadline = started + float(max_seconds)
        self.save_checkpoint("running", None, started)
        reason = "max_cycles"
        while self.cycle_index < self.config.max_cycles:
            if stop_requested():
                reason = "stopped"
                break
            if time.monotonic() >= deadline:
                reason = "budget"
                break
            if not self.step(deadline, started):
                reason = "budget"
                break
        self.save_checkpoint(reason, reason, started)
        return reason

    def step(self, deadline: float, started: Optional[float] = None) -> bool:
        if self.cycle_index >= self.config.max_cycles:
            return True
        if started is None:
            started = time.monotonic()
        if self.phase == "training":
            if (
                self.league_in_cycle == 0
                and self.research_in_cycle == 0
                and self._ingest_snapshots()
            ):
                self.save_checkpoint("running", None, started)
            progressed = self._training_step(deadline)
            if not progressed:
                return False
            if self._training_complete():
                self._prepare_candidates()
                self.save_checkpoint("running", None, started)
            elif (
                self.league_in_cycle + self.research_in_cycle
            ) % self.config.checkpoint_every == 0:
                self.save_checkpoint("running", None, started)
            return True
        if self.phase == "selection":
            self.save_checkpoint("running", None, started)
            try:
                self._selection_step(deadline)
            except (GateInterrupted, MatchInterrupted, TimeoutError):
                return False
            self.save_checkpoint("running", None, started)
            return True

        self.save_checkpoint("running", None, started)
        try:
            self._confirmation_step(deadline)
        except (GateInterrupted, MatchInterrupted, TimeoutError):
            return False
        self.save_checkpoint("running", None, started)
        return True

    def save_checkpoint(
        self,
        status: str = "running",
        stop_reason: Optional[str] = None,
        started: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not self.league.at_match_boundary or not self.research.at_episode_boundary:
            raise RuntimeError("cannot checkpoint partial learner work")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.jsonl").touch(exist_ok=True)
        self.league.export_model(self.output_dir / "latest-model.pt")
        atomic_copy(
            self.registry.model_path(self.registry.champion().id),
            self.output_dir / "best-model.pt",
        )
        atomic_copy(
            self.registry.model_path(self.baseline_id),
            self.output_dir / "baseline-model.pt",
        )
        atomic_torch_save(
            self.state_dict(), self.output_dir / "latest-state.pt"
        )
        active_metrics = self.output_dir / ".active-metrics.jsonl"
        if active_metrics.exists():
            atomic_copy(active_metrics, self.output_dir / "metrics.jsonl")
        files = _snapshot_manifest(self.output_dir)
        atomic_write_json(
            self._status_payload(
                files,
                status,
                stop_reason,
                time.monotonic() if started is None else started,
            ),
            self.output_dir / "status.json",
        )
        return files

    def _training_step(self, deadline: float) -> bool:
        can_league = (
            self.league_in_cycle < self.config.league_matches_per_cycle
        )
        can_research = (
            self.research_in_cycle < self.config.research_episodes_per_cycle
        )
        if not can_league and not can_research:
            return True
        learner = self.next_learner
        if learner == "league" and not can_league:
            learner = "research"
        elif learner == "research" and not can_research:
            learner = "league"
        if learner == "league":
            try:
                result = self.league.train_match(deadline=deadline)
            except MatchInterrupted:
                return False
            self.league_in_cycle += 1
            self.next_learner = "research"
            _append_metric_once(
                self.output_dir,
                "cycle-%d-league-%d"
                % (self.cycle_index, self.league_in_cycle),
                {
                    "event": "league_match",
                    "cycle": self.cycle_index,
                    **result,
                },
                self.metric_ids,
            )
            return True
        result = self.research.run_episode(deadline=deadline)
        if result is None:
            return False
        self.research_in_cycle += 1
        self.next_learner = "league"
        _append_metric_once(
            self.output_dir,
            "cycle-%d-research-%d"
            % (self.cycle_index, self.research_in_cycle),
            {
                "event": "research_episode",
                "cycle": self.cycle_index,
                **result,
            },
            self.metric_ids,
        )
        return True

    def _prepare_candidates(self) -> None:
        champion = self.registry.champion()
        self.champion_at_cycle_id = champion.id
        candidates = {}
        temporary = self.output_dir / ".league-candidate.pt"
        try:
            self.league.export_model(temporary, role="candidate")
            record = self.registry.register(
                str(temporary),
                origin="league",
                evidence={
                    "cycle": self.cycle_index,
                    "matches": self.league.completed_matches,
                    "updates": self.league.total_updates,
                },
                parents=(champion.id,),
            )
            candidates["league"] = record.id
        finally:
            temporary.unlink(missing_ok=True)

        temporary = self.output_dir / ".research-candidate.pt"
        try:
            atomic_torch_save(
                _research_model_payload(self.research, self.cycle_index),
                temporary,
            )
            record = self.registry.register(
                str(temporary),
                origin="research",
                evidence={
                    "cycle": self.cycle_index,
                    "episodes": self.research.total_episodes,
                    "updates": self.research.total_updates,
                },
                parents=(champion.id,),
            )
            candidates["research"] = record.id
        finally:
            temporary.unlink(missing_ok=True)

        self.candidate_ids = candidates
        self.selection_results = {}
        self.nominee_origin = None
        self.nominee_id = None
        self.confirmation_result = None
        self.phase = "selection"

    def _selection_step(self, deadline: Optional[float] = None) -> None:
        champion = self.registry.get(self.champion_at_cycle_id)
        for origin in ("league", "research"):
            if origin in self.selection_results:
                continue
            candidate_id = self.candidate_ids[origin]
            result = evaluate_gate(
                candidate_path=str(self.registry.model_path(candidate_id)),
                champion_path=str(self.registry.model_path(champion.id)),
                opponent_paths=self._gate_opponent_paths(champion.id),
                seed=self.pending_gate_seed,
                seed_blocks=self.config.selection_blocks,
                solo_games=self.config.solo_games,
                max_turns=self.config.max_turns,
                player_counts=self.config.player_counts,
                samples=self.config.gate_samples,
                deadline=deadline,
                recycle_discard=self.config.recycle_discard,
            )
            checked = _gate_result(result, "selection")
            self.selection_results[origin] = checked
            self.last_gate = checked
            _append_metric_once(
                self.output_dir,
                "cycle-%d-selection-%s" % (self.cycle_index, origin),
                {
                    "event": "gate",
                    "cycle": self.cycle_index,
                    "candidate_origin": origin,
                    **checked,
                },
                self.metric_ids,
            )
            return

        eligible = [
            origin
            for origin, result in self.selection_results.items()
            if result["accepted"]
        ]
        if not eligible:
            self._finish_cycle(promoted=False)
            return
        self.nominee_origin = max(
            eligible,
            key=lambda origin: (
                _gate_score(self.selection_results[origin]),
                origin == "league",
            ),
        )
        self.nominee_id = self.candidate_ids[self.nominee_origin]
        self.phase = "confirmation"

    def _confirmation_step(self, deadline: Optional[float] = None) -> None:
        champion = self.registry.get(self.champion_at_cycle_id)
        if self.confirmation_result is None:
            result = evaluate_gate(
                candidate_path=str(self.registry.model_path(self.nominee_id)),
                champion_path=str(self.registry.model_path(champion.id)),
                opponent_paths=self._gate_opponent_paths(champion.id),
                seed=self.pending_gate_seed,
                seed_blocks=self.config.confirmation_blocks,
                solo_games=self.config.solo_games,
                max_turns=self.config.max_turns,
                player_counts=self.config.player_counts,
                samples=self.config.gate_samples,
                deadline=deadline,
                recycle_discard=self.config.recycle_discard,
            )
            checked = _gate_result(result, "confirmation")
            checked["selection_passed"] = True
            self.confirmation_result = checked
            self.last_gate = checked
            _append_metric_once(
                self.output_dir,
                "cycle-%d-confirmation" % self.cycle_index,
                {
                    "event": "gate",
                    "cycle": self.cycle_index,
                    "candidate_origin": self.nominee_origin,
                    **checked,
                },
                self.metric_ids,
            )
            return

        promoted = bool(self.confirmation_result["accepted"])
        if promoted:
            current = self.registry.champion()
            if current.id != champion.id:
                if current.id == self.nominee_id:
                    self._record_promotion()
                    self._finish_cycle(promoted=True)
                    return
                raise RuntimeError("champion changed outside this controller")
            evidence = {
                **self.confirmation_result,
                "phase": "confirmation",
                "selection_passed": True,
                "accepted": True,
                "selection": self.selection_results[self.nominee_origin],
                "cycle": self.cycle_index,
                "candidate_origin": self.nominee_origin,
                "selection_seed": self.selection_results[
                    self.nominee_origin
                ]["seed"],
                "expected_champion_id": champion.id,
                "expected_champion_sha256": champion.sha256,
            }
            self.registry.promote(
                self.nominee_id,
                expected_id=champion.id,
                evidence=evidence,
                player_counts=self.config.player_counts,
            )
            self._record_promotion()
        self._finish_cycle(promoted=promoted)

    def _record_promotion(self) -> None:
        if self.cycle_index not in self.promoted_cycles:
            self.promoted_cycles = (*self.promoted_cycles, self.cycle_index)
            self.promotions += 1
            solo = self.confirmation_result.get("solo", {})
            games = solo.get("games")
            wins = solo.get("candidate_wins")
            if (
                isinstance(games, int)
                and not isinstance(games, bool)
                and games > 0
                and isinstance(wins, (int, float))
                and not isinstance(wins, bool)
            ):
                self.best_score = float(wins) / games

    def _finish_cycle(self, promoted: bool) -> None:
        if promoted and self.nominee_id is not None:
            self.opponent_ids = _bounded_pool(
                (*self.opponent_ids, self.nominee_id),
                self.baseline_id,
            )
            self.league.refresh_opponent_pool(self._opponent_paths())
        self.cycle_index += 1
        self.phase = "training"
        self.league_in_cycle = 0
        self.research_in_cycle = 0
        self.next_learner = "league"
        self.candidate_ids = {}
        self.selection_results = {}
        self.nominee_origin = None
        self.nominee_id = None
        self.confirmation_result = None
        self.champion_at_cycle_id = None
        self._ingest_snapshots()

    def _reconcile_promotion(self) -> None:
        current = self.registry.champion()
        if self.phase in ("selection", "confirmation"):
            if current.id == self.champion_at_cycle_id:
                return
            if (
                self.phase == "confirmation"
                and self.nominee_id is not None
                and current.id == self.nominee_id
            ):
                self._record_promotion()
                self._finish_cycle(promoted=True)
                return
            raise RuntimeError("champion pointer disagrees with controller state")

    def _training_complete(self) -> bool:
        return (
            self.league_in_cycle >= self.config.league_matches_per_cycle
            and self.research_in_cycle
            >= self.config.research_episodes_per_cycle
        )

    def _ingest_snapshots(self) -> bool:
        if self.snapshots_dir is None or not self.snapshots_dir.is_dir():
            return False
        known_ids = set(self.registry._read()["models"])
        new_ids = []
        for path in sorted(self.snapshots_dir.glob("model-*.pt")):
            match = re.fullmatch(r"model-([0-9a-f]{64})\.pt", path.name)
            if match is None or match[1][:20] in known_ids:
                continue
            try:
                record = self.registry.register(
                    str(path), origin="frozen_opponent", expected_sha256=match[1],
                )
            except (OSError, RuntimeError, TypeError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
                continue
            if record.id not in self.opponent_ids and record.origin == "frozen_opponent":
                new_ids.append(record.id)
        if not new_ids:
            return False
        self.opponent_ids = _bounded_pool((*self.opponent_ids, *new_ids), self.baseline_id)
        self.league.refresh_opponent_pool(self._opponent_paths())
        return True

    def _opponent_paths(self) -> Tuple[str, ...]:
        return tuple(
            str(self.registry.model_path(record_id))
            for record_id in self.opponent_ids
            if record_id != self.baseline_id
        )

    def _gate_opponent_paths(self, champion_id: str) -> list[str]:
        return [
            str(self.registry.model_path(record_id))
            for record_id in self.opponent_ids
            if record_id != champion_id
        ]

    def _validate_cursors(self) -> None:
        if self.league_in_cycle > self.config.league_matches_per_cycle:
            raise ValueError("league cycle cursor exceeds configured work")
        if self.research_in_cycle > self.config.research_episodes_per_cycle:
            raise ValueError("research cycle cursor exceeds configured work")
        if self.phase != "training" and not self._training_complete():
            raise ValueError("evaluation phase requires complete training cursors")
        if self.phase == "training" and (
            self.candidate_ids
            or self.selection_results
            or self.nominee_id is not None
            or self.champion_at_cycle_id is not None
        ):
            raise ValueError("training phase cannot retain evaluation state")
        if self.phase in ("selection", "confirmation"):
            if set(self.candidate_ids) != {"league", "research"}:
                raise ValueError("evaluation phase requires both candidates")
            if self.champion_at_cycle_id is None:
                raise ValueError("evaluation phase requires a frozen champion")
        if self.phase == "confirmation" and (
            self.nominee_origin not in ("league", "research")
            or self.nominee_id != self.candidate_ids[self.nominee_origin]
            or not self.selection_results.get(
                self.nominee_origin, {}
            ).get("accepted", False)
        ):
            raise ValueError("confirmation phase requires a selected nominee")

    def _status_payload(
        self,
        files: Mapping[str, Mapping[str, Any]],
        status: str,
        stop_reason: Optional[str],
        started: float,
    ) -> Dict[str, Any]:
        champion = self.registry.champion()
        return {
            "status_schema_version": STATUS_SCHEMA_VERSION,
            "status": status,
            "stop_reason": stop_reason,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "completed_cycles": self.cycle_index,
            "completed_episodes": (
                self.league.completed_matches + self.research.total_episodes
            ),
            "cumulative_episodes": (
                self.league.completed_matches + self.research.total_episodes
            ),
            "updates": self.league.total_updates + self.research.total_updates,
            "best_score": self.best_score,
            "stage": "%d/%s" % (self.cycle_index, self.phase),
            "distance": None,
            "training_state_version": IMPROVEMENT_STATE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "model_checkpoint_version": CHECKPOINT_VERSION,
            "algorithm": ALGORITHM,
            "rules": {
                "num_decks": 3,
                "cards_in_hand": 21,
                "required_sequences": 5,
                "max_turns": self.config.max_turns,
                "player_counts": self.config.player_counts,
                "recycle_discard": self.config.recycle_discard,
            },
            "champion_id": champion.id,
            "league_matches": self.league.completed_matches,
            "research_episodes": self.research.total_episodes,
            "promotions": self.promotions,
            "last_gate": self.last_gate,
            "files": {name: dict(metadata) for name, metadata in files.items()},
        }


def _research_config(config: ImproveConfig, lr_scale: float = 1.0) -> TrainingConfig:
    batch_size = max(1, int(os.environ.get("PAPPLU_BATCH_SIZE", "64")))
    updates_per_step = max(1, int(os.environ.get("PAPPLU_UPDATES_PER_STEP", "4")))
    replay_capacity = max(100, int(os.environ.get("PAPPLU_REPLAY_CAPACITY", "10000")))
    return TrainingConfig(
        learning_rate=RESEARCH_FINETUNE_LR * lr_scale,
        epsilon_start=RESEARCH_FINETUNE_EPSILON_START,
        epsilon_end=RESEARCH_FINETUNE_EPSILON_END,
        curriculum_fraction=0.25,
        max_turns=config.max_turns,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        updates_per_episode=updates_per_step,
        validation_games=config.solo_games,
        recycle_discard=config.recycle_discard,
        shaping=DEFAULT_IMPROVE_SHAPING,
    )


def _research_model_payload(
    session: TrainingSession, cycle: int
) -> Dict[str, Any]:
    return build_checkpoint(
        session.network,
        session.config.game_config,
        meta={
            "algorithm": ALGORITHM,
            "origin": "research",
            "cycle": cycle,
            "episodes": session.total_episodes,
            "updates": session.total_updates,
            "training_config": session.config.to_dict(),
        },
    )


def _gate_result(raw: Any, phase: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("promotion gate must return a dict")
    if not isinstance(raw.get("accepted"), bool):
        raise ValueError("promotion gate accepted result must be bool")
    result = dict(raw)
    result["phase"] = phase
    return result


def _gate_score(result: Mapping[str, Any]) -> Tuple[float, float]:
    return (
        _finite_number(result.get("multiplayer_gain"), "multiplayer_gain"),
        _finite_number(result.get("solo_gain"), "solo_gain"),
    )


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("%s must be finite" % name)
    return float(value)


def _initial_evidence(
    model: Path,
    current_recycle_discard: bool,
) -> dict:
    path = model.parent / "final-best.json"
    if not path.is_file():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("encoding_version") != ENCODING_VERSION
        or result.get("checkpoint_version") != CHECKPOINT_VERSION
        or result.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("Initial model evaluation requires the current dynamic schema")
    checkpoint_sha256 = file_sha256(model)
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Initial model evaluation belongs to a different checkpoint")
    outcomes = result.get("greedy", {}).get("outcomes")
    games = result.get("greedy", {}).get("games")
    wins = result.get("greedy", {}).get("wins")
    evaluated_rules = GameConfig.from_dict(result.get("game_config"))
    if (
        not isinstance(outcomes, list) or not outcomes
        or any(type(value) is not int or value not in (0, 1) for value in outcomes)
        or games != len(outcomes) or wins != sum(outcomes)
        or (
            evaluated_rules.num_decks, evaluated_rules.cards_in_hand,
            evaluated_rules.required_sequences, evaluated_rules.max_turns,
        ) != (3, 21, 5, 60)
    ):
        raise ValueError("Initial model evaluation must contain complete ordinary full21 outcomes")
    if evaluated_rules.recycle_discard != current_recycle_discard:
        raise ValueError("Initial model evaluation rules must match the current training rules")
    return {
        "current_solo_reference": {
            "checkpoint_sha256": checkpoint_sha256,
            "games": games,
            "wins": wins,
            "win_rate": wins / games,
            "outcomes": list(outcomes),
            "seed": result.get("game_seed"),
            "source": "frozen ordinary-deal evaluation",
            "recycle_discard": evaluated_rules.recycle_discard,
            "rules": evaluated_rules.to_dict(),
            "rule_regime": (
                "discard_recycling"
                if evaluated_rules.recycle_discard
                else "stock_exhaustion"
            ),
        },
    }


def _bounded_pool(record_ids: Sequence[str], baseline_id: str) -> Tuple[str, ...]:
    unique = []
    for record_id in record_ids:
        if record_id not in unique:
            unique.append(record_id)
    others = [record_id for record_id in unique if record_id != baseline_id]
    return (baseline_id, *others[-6:])


_IMMUTABLE_MODEL_META: Dict[Tuple[str, int, int], Dict[str, Any]] = {}


def _snapshot_manifest(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    files = {}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.name in ("status.json", "error.json")
            or (
                path.suffix not in (".pt", ".json")
                and path.name != "metrics.jsonl"
            )
        ):
            continue
        if path.name.startswith("model-") and path.name.endswith(".pt"):
            st = path.stat()
            cache_key = (str(path), st.st_size, st.st_mtime_ns)
            meta = _IMMUTABLE_MODEL_META.get(cache_key)
            if meta is None:
                meta = file_metadata(path)
                _IMMUTABLE_MODEL_META[cache_key] = meta
            files[path.name] = dict(meta)
        else:
            files[path.name] = file_metadata(path)
    required = {
        "latest-state.pt",
        "latest-model.pt",
        "best-model.pt",
        "baseline-model.pt",
        "metrics.jsonl",
        "registry.json",
        "champion.json",
    }
    missing = required - set(files)
    if missing:
        raise FileNotFoundError(
            "missing snapshot files: %s" % ", ".join(sorted(missing))
        )
    return files


def _read_metric_ids(path: Path) -> set:
    identifiers = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("Malformed metrics history at line %d" % line_number) from error
                if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
                    raise ValueError("Metrics history lacks an event identifier")
                identifiers.add(event["event_id"])
    return identifiers


def _append_metric_once(
    output_dir: Path, event_id: str, event: Mapping[str, Any],
    known_ids: Optional[set] = None,
) -> None:
    path = output_dir / ".active-metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    identifiers = _read_metric_ids(path) if known_ids is None else known_ids
    if event_id in identifiers:
        return
    payload = {
        "algorithm": ALGORITHM,
        "improvement_state_version": IMPROVEMENT_STATE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "event_id": event_id,
        **event,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    identifiers.add(event_id)


def _record_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a nonempty string" % name)
    return value


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _record_id(value, name)


def _string_dict(value: Any, name: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict" % name)
    return {str(key): _record_id(item, name) for key, item in value.items()}


def _mapping_dict(value: Any, name: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict) or any(
        not isinstance(item, dict) for item in value.values()
    ):
        raise ValueError("%s must contain dict values" % name)
    return {str(key): dict(item) for key, item in value.items()}


def _optional_dict(value: Any, name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict or None" % name)
    return dict(value)
