from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.improve import (
    ALGORITHM,
    ImproveConfig,
    ImprovementController,
    build_arg_parser,
)
from src.registry import file_sha256

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


class TestMetricHistory(unittest.TestCase):
    def test_initial_evidence_accepts_legacy_and_recycling_rules_with_explicit_labels(self):
        from src.improve import _initial_evidence

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            model.write_bytes(b"same frozen weights")
            for recycling in (None, True, False):
                rules = {"num_decks": 3, "cards_in_hand": 21, "required_sequences": 5, "max_turns": 60}
                if recycling is not None:
                    rules["recycle_discard"] = recycling
                (root / "final-best.json").write_text(json.dumps({
                    "checkpoint_sha256": sha256(model),
                    "game_config": rules,
                    "greedy": {"games": 2, "wins": 1, "outcomes": [1, 0]},
                }))
                evidence = _initial_evidence(
                    model,
                    current_recycle_discard=True,
                )
                label = (
                    "current_solo_reference"
                    if recycling is True
                    else "legacy_solo_reference"
                )
                verified = evidence[label]
                self.assertEqual(set(evidence), {label})
                self.assertEqual(verified["win_rate"], 0.5)
                self.assertIs(verified["recycle_discard"], recycling is True)
                self.assertEqual(
                    verified["rule_regime"],
                    (
                        "discard_recycling"
                        if recycling is True
                        else "legacy_nonrecycling"
                    ),
                )

    def test_cached_event_index_avoids_rescanning_the_entire_log(self):
        from src.improve import _append_metric_once, _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identifiers = set()
            with mock.patch("src.improve.json.loads", side_effect=AssertionError("unexpected rescan")):
                _append_metric_once(output, "first", {"value": 1}, identifiers)
                _append_metric_once(output, "second", {"value": 2}, identifiers)
                _append_metric_once(output, "first", {"value": 99}, identifiers)
            self.assertEqual(_read_metric_ids(output / ".active-metrics.jsonl"), {"first", "second"})
            self.assertEqual(len((output / ".active-metrics.jsonl").read_text().splitlines()), 2)

    def test_malformed_history_is_reported_instead_of_silently_skipped(self):
        from src.improve import _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text("{incomplete", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                _read_metric_ids(path)


def sha256(path):
    return file_sha256(Path(path))


def gate_result(accepted):
    def evaluate(**kwargs):
        from src.promotion import balanced_seat_orders, evidence_metrics

        blocks = kwargs["seed_blocks"]
        multiplayer = {
            str(players): {
                "candidate_minus_champion": 0.5 if accepted else 0.0,
                "games": blocks * len(balanced_seat_orders(players)),
                "seed_block_margins": [0.5 if accepted else 0.0] * blocks,
            }
            for players in (2, 3)
        }
        solo = {
            "seed": kwargs["seed"] + 100_000,
            "games": kwargs["solo_games"],
            "candidate_outcomes": [0] * kwargs["solo_games"],
            "champion_outcomes": [0] * kwargs["solo_games"],
            "candidate_wins": 0,
            "champion_wins": 0,
        }
        samples = max(100, kwargs["samples"])
        return {
            "candidate_sha256": sha256(kwargs["candidate_path"]),
            "champion_sha256": sha256(kwargs["champion_path"]),
            "accepted": accepted,
            **evidence_metrics(multiplayer, solo, blocks, kwargs["seed"], samples),
            "bootstrap_samples": samples,
            "seed": kwargs["seed"],
            "seed_blocks": kwargs["seed_blocks"],
            "player_counts": list(kwargs["player_counts"]),
            "rules": {
                "num_decks": 3,
                "cards_in_hand": 21,
                "required_sequences": 5,
                "max_turns": kwargs["max_turns"],
                "recycle_discard": kwargs["recycle_discard"],
            },
            "pool_hashes": [
                sha256(path) for path in kwargs["opponent_paths"]
            ],
            "raw_outcomes": (),
            "multiplayer": multiplayer,
            "solo": solo,
            "scope": {"player_counts": kwargs["player_counts"]},
        }

    return evaluate


@unittest.skipIf(torch is None, TORCH_REASON)
class TestImprovementController(unittest.TestCase):
    def make_model(self, path):
        from src.environment import GameConfig
        from src.model import QNetwork, save_checkpoint

        save_checkpoint(
            str(path),
            QNetwork(architecture="suit_conv"),
            GameConfig(max_turns=60),
            meta={"best_score": 164 / 256},
        )

    def make_controller(self, root):
        model = root / "champion-source.pt"
        self.make_model(model)
        config = ImproveConfig(
            max_cycles=1,
            league_matches_per_cycle=1,
            research_episodes_per_cycle=1,
            selection_blocks=32,
            confirmation_blocks=128,
            solo_games=256,
            checkpoint_every=1,
            gate_samples=20,
        )
        return ImprovementController(model, root / "output", config=config)

    def prepare_distinct_candidates(self, controller):
        with torch.no_grad():
            next(iter(controller.league.network.parameters())).add_(0.01)
            next(iter(controller.research.network.parameters())).add_(0.02)
        controller.league_in_cycle = 1
        controller.research_in_cycle = 1
        controller._prepare_candidates()

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_research_is_independent_lower_lr_and_starts_at_distance_four(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            self.assertEqual(
                controller.research.config.learning_rate,
                1e-4,
            )
            self.assertLess(
                controller.research.config.learning_rate,
                1e-3,
            )
            self.assertEqual(controller.research.config.curriculum_fraction, 0.25)
            self.assertEqual(controller.research.stage_index, 2)
            self.assertEqual(controller.research.current_distance, 4)
            self.assertIsNot(
                controller.research.optimizer,
                controller.league.optimizer,
            )
            self.assertIsNot(
                controller.research.replay,
                controller.league.replay,
            )
            self.assertIsNone(controller.best_score)

    def test_legacy_solo_reference_does_not_seed_current_best_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "champion-source.pt"
            self.make_model(model)
            (root / "final-best.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": sha256(model),
                        "game_config": {
                            "num_decks": 3,
                            "cards_in_hand": 21,
                            "required_sequences": 5,
                            "max_turns": 60,
                            "recycle_discard": False,
                        },
                        "greedy": {
                            "games": 2,
                            "wins": 1,
                            "outcomes": [1, 0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            controller = ImprovementController(
                model,
                root / "output",
                config=ImproveConfig(
                    max_cycles=1,
                    league_matches_per_cycle=1,
                    research_episodes_per_cycle=1,
                    selection_blocks=32,
                    confirmation_blocks=128,
                    solo_games=256,
                ),
            )
            baseline = controller.registry.get(controller.baseline_id)
            self.assertIsNone(controller.best_score)
            self.assertIn("legacy_solo_reference", baseline.evidence)
            self.assertNotIn("current_solo_reference", baseline.evidence)

    def test_weak_selection_keeps_current_champion_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            original = controller.registry.champion()
            original_bytes = controller.registry.model_path(original.id).read_bytes()
            self.prepare_distinct_candidates(controller)
            with mock.patch(
                "src.promotion.evaluate_gate",
                side_effect=gate_result(False),
            ):
                controller._selection_step()
                controller._selection_step()
                controller._selection_step()
            current = controller.registry.champion()
            self.assertEqual(current.id, original.id)
            self.assertEqual(controller.phase, "training")
            self.assertEqual(controller.promotions, 0)
            self.assertEqual(
                controller.registry.model_path(original.id).read_bytes(),
                original_bytes,
            )
            self.assertEqual(original.evidence, {})

    def test_only_fresh_confirmation_can_change_champion(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            original = controller.registry.champion()
            original_bytes = controller.registry.model_path(
                original.id
            ).read_bytes()
            self.prepare_distinct_candidates(controller)
            decisions = iter(
                [
                    gate_result(True),
                    gate_result(False),
                    gate_result(True),
                ]
            )

            def evaluate(**kwargs):
                return next(decisions)(**kwargs)

            with mock.patch(
                "src.promotion.evaluate_gate",
                side_effect=evaluate,
            ):
                controller._selection_step()
                controller._selection_step()
                controller._selection_step()
                nominee = controller.nominee_id
                self.assertEqual(controller.registry.champion().id, original.id)
                controller._confirmation_step()
                self.assertEqual(controller.registry.champion().id, original.id)
                controller._confirmation_step()
            self.assertEqual(controller.registry.champion().id, nominee)
            self.assertEqual(
                controller.registry.model_path(original.id).read_bytes(),
                original_bytes,
            )
            self.assertEqual(controller.promotions, 1)
            self.assertEqual(controller.promoted_cycles, (0,))
            self.assertEqual(controller.league.opponent_pool_generation, 1)

    def test_selection_uses_one_bank_and_confirms_only_nominee(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            self.prepare_distinct_candidates(controller)
            calls = []
            decisions = iter((True, False, False))

            def evaluate(**kwargs):
                calls.append(dict(kwargs))
                return gate_result(next(decisions))(**kwargs)

            with mock.patch(
                "src.promotion.evaluate_gate",
                side_effect=evaluate,
            ):
                controller._selection_step()
                controller._selection_step()
                controller._selection_step()
                nominee_path = str(
                    controller.registry.model_path(controller.nominee_id)
                )
                controller._confirmation_step()
                self.assertEqual(controller.registry.champion().id, controller.baseline_id)
                controller._confirmation_step()
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0]["seed"], 60_000_000)
            self.assertEqual(calls[1]["seed"], 60_000_000)
            self.assertEqual(calls[2]["seed"], 60_500_000)
            self.assertEqual(calls[0]["seed_blocks"], 32)
            self.assertEqual(calls[1]["seed_blocks"], 32)
            self.assertEqual(calls[2]["seed_blocks"], 128)
            self.assertEqual(calls[2]["candidate_path"], nominee_path)
            self.assertEqual(controller.promotions, 0)

    def test_full_state_roundtrip_preserves_both_learners(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.make_controller(root)
            controller.save_checkpoint()
            before = controller.state_dict()
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"),
                controller.output_dir,
            )
            after = resumed.state_dict()
            self.assert_nested_equal(
                before["league_state"]["optimizer_state_dict"],
                after["league_state"]["optimizer_state_dict"],
            )
            self.assert_nested_equal(
                before["research_state"]["optimizer_state_dict"],
                after["research_state"]["optimizer_state_dict"],
            )
            self.assert_nested_equal(
                before["league_state"]["rng_states"],
                after["league_state"]["rng_states"],
            )
            self.assert_nested_equal(
                before["research_state"]["rng_states"],
                after["research_state"]["rng_states"],
            )
            self.assertEqual(after["config"], before["config"])
            self.assertEqual(after["opponent_ids"], before["opponent_ids"])

    def test_resume_can_extend_runtime_cap_but_not_learning_settings(self):
        from src.improve import _validate_resume_options

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            parser = build_arg_parser()
            arguments = [
                "--resume", "latest-state.pt", "--output-dir", "output",
                "--max-seconds", "60", "--max-cycles", "3",
            ]
            _validate_resume_options(controller, parser.parse_args(arguments), arguments)
            self.assertEqual(controller.config.max_cycles, 3)
            self.assertEqual(controller.config.league_matches_per_cycle, 1)
            invalid = arguments + ["--league-matches", "999"]
            with self.assertRaisesRegex(ValueError, "cannot change"):
                _validate_resume_options(controller, parser.parse_args(invalid), invalid)

    def test_solo_resume_rejects_nonfinite_optimizer_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            payload = controller.research.state_dict()
            payload["optimizer_state_dict"]["state"] = {
                0: {"exp_avg": torch.tensor([float("nan")])}
            }
            with self.assertRaisesRegex(ValueError, "non-finite"):
                from src.long_train import TrainingSession

                TrainingSession.from_state_dict(payload)

    def test_zero_budget_exports_truthful_flat_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            reason = controller.run(0)
            self.assertEqual(reason, "budget")
            status = json.loads(
                (controller.output_dir / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["algorithm"], ALGORITHM)
            self.assertEqual(status["completed_cycles"], 0)
            self.assertEqual(status["completed_episodes"], 0)
            self.assertEqual(status["league_matches"], 0)
            self.assertEqual(status["research_episodes"], 0)
            self.assertEqual(status["promotions"], 0)
            self.assertEqual(status["distance"], None)
            self.assertNotIn("genes", json.dumps(status).lower())
            required = {
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
                "metrics.jsonl",
                "registry.json",
                "champion.json",
            }
            self.assertTrue(required.issubset(status["files"]))
            for name, metadata in status["files"].items():
                path = controller.output_dir / name
                self.assertEqual(metadata["bytes"], path.stat().st_size)
                self.assertEqual(metadata["sha256"], sha256(path))

    def test_active_metrics_cannot_invalidate_published_snapshot(self):
        from src.improve import _append_metric_once

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.save_checkpoint()
            status = json.loads((controller.output_dir / "status.json").read_text())
            metadata = status["files"]["metrics.jsonl"]
            _append_metric_once(controller.output_dir, "new-event", {"event": "test"}, controller.metric_ids)
            self.assertEqual(sha256(controller.output_dir / "metrics.jsonl"), metadata["sha256"])
            self.assertTrue((controller.output_dir / ".active-metrics.jsonl").read_text())
            controller.save_checkpoint()
            self.assertIn("new-event", (controller.output_dir / "metrics.jsonl").read_text())

    def test_gate_budget_exceptions_checkpoint_without_partial_promotion(self):
        from src.arena import MatchInterrupted
        from src.promotion import GateInterrupted

        interruptions = (
            GateInterrupted("deadline"),
            MatchInterrupted("deadline"),
            TimeoutError("deadline"),
        )
        for interruption in interruptions:
            with self.subTest(exception=type(interruption).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    controller = self.make_controller(Path(temporary))
                    self.prepare_distinct_candidates(controller)
                    champion_id = controller.registry.champion().id
                    before = controller.state_dict()
                    with mock.patch(
                        "src.promotion.evaluate_gate",
                        side_effect=interruption,
                    ):
                        reason = controller.run(10)
                    self.assertEqual(reason, "budget")
                    self.assertEqual(controller.registry.champion().id, champion_id)
                    self.assertEqual(controller.selection_results, {})
                    self.assertIsNone(controller.last_gate)
                    saved = torch.load(
                        controller.output_dir / "latest-state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assert_nested_equal(
                        saved["league_state"], before["league_state"]
                    )
                    self.assert_nested_equal(
                        saved["research_state"], before["research_state"]
                    )
                    status = json.loads(
                        (controller.output_dir / "status.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(status["stop_reason"], "budget")
                    self.assertEqual(status["completed_cycles"], 0)
                    self.assertEqual(status["completed_episodes"], 0)

    def test_controller_state_uses_relative_registry_ids_not_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.save_checkpoint()
            payload = torch.load(
                controller.output_dir / "latest-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            encoded = repr(payload)
            self.assertNotIn("champion-source.pt", encoded)
            self.assertEqual(payload["baseline_id"], controller.baseline_id)
            self.assertIsInstance(payload["opponent_ids"], tuple)

    def test_remote_one_cycle_smoke_keeps_champion_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "champion-source.pt"
            self.make_model(model)
            controller = ImprovementController(
                model,
                root / "output",
                config=ImproveConfig(
                    max_cycles=1,
                    league_matches_per_cycle=2,
                    research_episodes_per_cycle=2,
                    selection_blocks=2,
                    confirmation_blocks=2,
                    solo_games=2,
                    checkpoint_every=2,
                    gate_samples=100,
                ),
            )
            original = controller.registry.champion().id
            reason = controller.run(300)
            self.assertEqual(reason, "max_cycles")
            self.assertEqual(controller.cycle_index, 1)
            self.assertEqual(controller.league.completed_matches, 2)
            self.assertEqual(controller.research.total_episodes, 2)
            self.assertEqual(controller.registry.champion().id, original)
            self.assertEqual(controller.promotions, 0)


class TestImproveCli(unittest.TestCase):
    def test_cli_requires_budget_and_exposes_bounded_cycle_overrides(self):
        parser = build_arg_parser()
        max_seconds = next(
            action
            for action in parser._actions
            if action.dest == "max_seconds"
        )
        self.assertTrue(max_seconds.required)
        args = parser.parse_args(
            [
                "--initial-model",
                "champion.pt",
                "--output-dir",
                "out",
                "--max-seconds",
                "60",
                "--max-cycles",
                "1",
                "--league-matches-per-cycle",
                "2",
                "--research-episodes-per-cycle",
                "2",
                "--selection-blocks",
                "2",
                "--confirmation-blocks",
                "2",
                "--solo-games",
                "2",
            ]
        )
        self.assertEqual(args.max_cycles, 1)
        self.assertEqual(args.league_matches_per_cycle, 2)
        self.assertEqual(args.research_episodes_per_cycle, 2)
        self.assertEqual(args.selection_blocks, 2)
        self.assertEqual(args.confirmation_blocks, 2)
        self.assertEqual(args.solo_games, 2)


if __name__ == "__main__":
    unittest.main()
