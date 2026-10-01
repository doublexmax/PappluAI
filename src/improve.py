"""Finite champion-league improvement cycles with guarded promotion."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.arena import MatchInterrupted
from src.environment import ENCODING_VERSION, GameConfig
from src.league_train import LeagueConfig, LeagueSession, RANDOM_OPPONENT
from src.long_train import (
    STATUS_SCHEMA_VERSION,
    TrainingConfig,
    TrainingSession,
    _atomic_json,
    _atomic_torch_save,
    _dict_with_fields,
    _nonnegative_int,
    _positive_int,
)
from src.promotion import GateInterrupted


IMPROVEMENT_STATE_VERSION = 1
ALGORITHM = "champion_league_mc"
PHASES = ("training", "selection", "confirmation")


@dataclass(frozen=True)
class ImproveConfig:
    max_cycles: int = 64
    league_matches_per_cycle: int = 128
    research_episodes_per_cycle: int = 128
    seed: int = 41
    selection_blocks: int = 32
    confirmation_blocks: int = 128
    solo_games: int = 256
    checkpoint_every: int = 16
    max_turns: int = 60
    player_counts: Tuple[int, ...] = (2, 3)
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
            _positive_int(getattr(self, name), name)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.max_turns != 60:
            raise ValueError("improvement cycles require 60-turn matches")
        if (
            not isinstance(self.player_counts, tuple)
            or not self.player_counts
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count not in (2, 3)
                for count in self.player_counts
            )
        ):
            raise ValueError("the first promotion scope supports players 2 and 3")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ImproveConfig":
        if isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__) - {"recycle_discard"}:
            raw = {**raw, "recycle_discard": False}
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
    ) -> None:
        from src.registry import ModelRegistry

        self.config = ImproveConfig() if config is None else config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any((self.output_dir / name).exists() for name in ("latest-state.pt", "status.json")):
            raise ValueError("Training outputs already exist; use --resume")
        self.metric_ids = _read_metric_ids(self.output_dir / ".active-metrics.jsonl")
        self.registry = ModelRegistry(self.output_dir)

        initial_evidence = _initial_evidence(Path(initial_model))
        baseline = self.registry.register(
            initial_model,
            origin="baseline",
            evidence={"general_access": True, "protected": True, **initial_evidence},
        )
        self.registry.initialize_champion(baseline.id)
        opponent_ids = [baseline.id]
        for path in opponent_paths:
            if path == RANDOM_OPPONENT:
                continue
            record = self.registry.register(
                path,
                origin="frozen_opponent",
                evidence={"general_access": False},
            )
            if record.id not in opponent_ids:
                opponent_ids.append(record.id)
        self.baseline_id = baseline.id
        self.opponent_ids = _bounded_pool(tuple(opponent_ids), baseline.id)

        baseline_path = str(self.registry.model_path(self.baseline_id))
        league_paths = self._opponent_paths()
        self.league = LeagueSession(
            baseline_path,
            league_paths,
            config=LeagueConfig(
                game=GameConfig(max_turns=60, recycle_discard=self.config.recycle_discard),
                player_counts=self.config.player_counts,
            ),
            seed=self.config.seed,
        )
        self.research = TrainingSession(
            config=_research_config(self.config),
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
        self.best_score = initial_evidence.get("verified_solo", {}).get("win_rate")

    @classmethod
    def load(cls, path: str, output_dir: Path) -> "ImprovementController":
        import torch
        from src.registry import ModelRegistry

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
        controller.output_dir = Path(output_dir)
        if (controller.output_dir / "metrics.jsonl").exists():
            _atomic_copy(
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

        controller.cycle_index = _nonnegative_int(payload["cycle"], "cycle")
        controller.phase = payload["phase"]
        if controller.phase not in PHASES:
            raise ValueError("improvement phase is invalid")
        cursors = _dict_with_fields(
            payload["cursors"],
            {"league", "research", "next_learner"},
            "cursors",
        )
        controller.league_in_cycle = _nonnegative_int(
            cursors["league"], "league cursor"
        )
        controller.research_in_cycle = _nonnegative_int(
            cursors["research"], "research cursor"
        )
        controller.next_learner = cursors["next_learner"]
        if controller.next_learner not in ("league", "research"):
            raise ValueError("next_learner is invalid")

        evaluation = _dict_with_fields(
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

        controller.promotions = _nonnegative_int(
            payload["promotions"], "promotions"
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
            expected_config=_research_config(controller.config),
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
        _atomic_copy(
            self.registry.model_path(self.registry.champion().id),
            self.output_dir / "best-model.pt",
        )
        _atomic_copy(
            self.registry.model_path(self.baseline_id),
            self.output_dir / "baseline-model.pt",
        )
        _atomic_torch_save(
            self.state_dict(), self.output_dir / "latest-state.pt"
        )
        active_metrics = self.output_dir / ".active-metrics.jsonl"
        if active_metrics.exists():
            _atomic_copy(active_metrics, self.output_dir / "metrics.jsonl")
        files = _snapshot_manifest(self.output_dir)
        _atomic_json(
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
            from src.arena import MatchInterrupted

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
            _atomic_torch_save(
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
        from src.promotion import evaluate_gate

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
        from src.promotion import evaluate_gate

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
        from src.model import CHECKPOINT_VERSION

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-model")
    source.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--max-cycles", type=int, default=64)
    parser.add_argument(
        "--league-matches",
        "--league-matches-per-cycle",
        dest="league_matches_per_cycle",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--research-episodes",
        "--research-episodes-per-cycle",
        dest="research_episodes_per_cycle",
        type=int,
        default=128,
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--selection-blocks", type=int, default=32)
    parser.add_argument("--confirmation-blocks", type=int, default=128)
    parser.add_argument("--solo-games", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    supplied_arguments = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = parser.parse_args(supplied_arguments)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    output_dir = Path(args.output_dir)
    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, request_stop)
        if args.resume:
            controller = ImprovementController.load(args.resume, output_dir)
            _validate_resume_options(controller, args, supplied_arguments)
        else:
            controller = ImprovementController(
                args.initial_model,
                output_dir,
                opponent_paths=args.opponent,
                config=ImproveConfig(
                    max_cycles=args.max_cycles,
                    league_matches_per_cycle=args.league_matches_per_cycle,
                    research_episodes_per_cycle=args.research_episodes_per_cycle,
                    seed=args.seed,
                    selection_blocks=args.selection_blocks,
                    confirmation_blocks=args.confirmation_blocks,
                    solo_games=args.solo_games,
                    checkpoint_every=args.checkpoint_every,
                ),
            )
        reason = controller.run(
            args.max_seconds,
            stop_requested=lambda: stop[0],
        )
        print(
            json.dumps(
                {
                    "status": reason,
                    "cycle": controller.cycle_index,
                    "champion_id": controller.registry.champion().id,
                    "league_matches": controller.league.completed_matches,
                    "research_episodes": controller.research.total_episodes,
                    "promotions": controller.promotions,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _validate_resume_options(controller, args, supplied_arguments):
    explicit = {value.split("=", 1)[0] for value in supplied_arguments if value.startswith("--")}
    learning_options = {
        "--league-matches": "league_matches_per_cycle",
        "--league-matches-per-cycle": "league_matches_per_cycle",
        "--research-episodes": "research_episodes_per_cycle",
        "--research-episodes-per-cycle": "research_episodes_per_cycle",
        "--selection-blocks": "selection_blocks",
        "--confirmation-blocks": "confirmation_blocks",
        "--solo-games": "solo_games",
        "--seed": "seed",
    }
    for option, field in learning_options.items():
        if option in explicit and getattr(args, field) != getattr(controller.config, field):
            raise ValueError("Resume cannot change %s" % option)
    for path in args.opponent:
        if path == RANDOM_OPPONENT:
            continue
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        record = controller.registry.get(digest[:20])
        if record.sha256 != digest or (
            record.origin != "frozen_opponent" and record.id != controller.baseline_id
        ):
            raise ValueError("Resume cannot add or replace the initial opponent pool")
    changes = {}
    if "--max-cycles" in explicit:
        _positive_int(args.max_cycles, "max_cycles")
        if args.max_cycles < controller.cycle_index:
            raise ValueError("max_cycles cannot be below completed cycles")
        changes["max_cycles"] = args.max_cycles
    if "--checkpoint-every" in explicit:
        _positive_int(args.checkpoint_every, "checkpoint_every")
        changes["checkpoint_every"] = args.checkpoint_every
    if changes:
        controller.config = replace(controller.config, **changes)


def _research_config(config: ImproveConfig) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=1e-4,
        curriculum_fraction=0.25,
        max_turns=60,
        replay_capacity=10_000,
        batch_size=64,
        updates_per_episode=4,
        validation_games=config.solo_games,
        recycle_discard=config.recycle_discard,
    )


def _research_model_payload(
    session: TrainingSession, cycle: int
) -> Dict[str, Any]:
    from src.model import build_checkpoint

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


def _initial_evidence(model: Path) -> dict:
    path = model.parent / "final-best.json"
    if not path.is_file():
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("checkpoint_sha256") != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError("Initial model evaluation belongs to a different checkpoint")
    outcomes = result.get("greedy", {}).get("outcomes")
    games = result.get("greedy", {}).get("games")
    wins = result.get("greedy", {}).get("wins")
    if (
        not isinstance(outcomes, list) or not outcomes
        or any(type(value) is not int or value not in (0, 1) for value in outcomes)
        or games != len(outcomes) or wins != sum(outcomes)
        or result.get("game_config") != {
            "num_decks": 3, "cards_in_hand": 21,
            "required_sequences": 5, "max_turns": 60,
        }
    ):
        raise ValueError("Initial model evaluation must contain complete ordinary full21 outcomes")
    return {
        "verified_solo": {
            "games": games, "wins": wins, "win_rate": wins / games,
            "seed": result.get("game_seed"),
            "source": "frozen ordinary-deal evaluation",
        },
    }


def _bounded_pool(record_ids: Sequence[str], baseline_id: str) -> Tuple[str, ...]:
    unique = []
    for record_id in record_ids:
        if record_id not in unique:
            unique.append(record_id)
    others = [record_id for record_id in unique if record_id != baseline_id]
    return (baseline_id, *others[-6:])


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
        files[path.name] = _file_metadata(path)
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


def _file_metadata(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return {"sha256": digest.hexdigest(), "bytes": size}
            size += len(chunk)
            digest.update(chunk)


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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(str(temporary), str(destination))


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


if __name__ == "__main__":
    sys.exit(main())
