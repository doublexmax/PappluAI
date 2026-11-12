from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import src.training.improve as improve_runtime
from src.cli.improve import build_arg_parser
from src.training.improve import ALGORITHM, FinetuneSchedule, ImproveConfig, ImprovementController
from tests.improve_support import ControllerTestCase, sha256


class TestMetricHistory(unittest.TestCase):
    def test_initial_evidence_requires_current_schema_and_matching_rules(self):
        from src.training.improve import _initial_evidence

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            model.write_bytes(b"same frozen weights")
            rules = {
                "num_decks": 3, "cards_in_hand": 21, "required_sequences": 5,
                "max_turns": 60, "recycle_discard": True,
            }
            result = {
                "encoding_version": 2, "checkpoint_version": 4, "architecture": "axial_table",
                "checkpoint_sha256": sha256(model), "game_config": rules,
                "greedy": {"games": 2, "wins": 1, "outcomes": [1, 0]},
            }
            path = root / "final-best.json"
            path.write_text(json.dumps(result))
            evidence = _initial_evidence(model, current_recycle_discard=True)
            verified = evidence["current_solo_reference"]
            self.assertEqual(verified["win_rate"], 0.5)
            self.assertTrue(verified["recycle_discard"])
            for bad in (
                {**result, "encoding_version": 1},
                {**result, "game_config": {**rules, "recycle_discard": False}},
            ):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    _initial_evidence(model, current_recycle_discard=True)

    def test_cached_event_index_avoids_rescanning_the_entire_log(self):
        from src.training.improve import _append_metric_once, _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identifiers = set()
            with mock.patch("src.training.improve.json.loads", side_effect=AssertionError("unexpected rescan")):
                _append_metric_once(output, "first", {"value": 1}, identifiers)
                _append_metric_once(output, "second", {"value": 2}, identifiers)
                _append_metric_once(output, "first", {"value": 99}, identifiers)
            self.assertEqual(_read_metric_ids(output / ".active-metrics.jsonl"), {"first", "second"})
            events = [json.loads(line) for line in (output / ".active-metrics.jsonl").read_text().splitlines()]
            self.assertEqual([event["value"] for event in events], [1, 2])

    def test_malformed_history_is_reported_instead_of_silently_skipped(self):
        from src.training.improve import _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text("{incomplete", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                _read_metric_ids(path)


class TestImproveConfig(unittest.TestCase):
    def test_finetune_schedule_selects_by_verified_mastery_and_rejects_invalid_values(self):
        for rate, expected in (
            (None, (1e-4, 0.08, 0.02)), (0.9499, (1e-4, 0.08, 0.02)),
            (0.95, (2e-5, 0.04, 0.01)), (1.0, (2e-5, 0.04, 0.01)),
        ):
            schedule = FinetuneSchedule.for_solo_win_rate(rate)
            self.assertEqual((schedule.learning_rate, schedule.epsilon_start, schedule.epsilon_end), expected)
        for settings in (
            {"learning_rate": 0}, {"learning_rate": float("nan")}, {"learning_rate": True},
            {"epsilon_start": 1.01}, {"epsilon_end": -0.01}, {"epsilon_end": float("nan")},
            {"epsilon_start": 0.01, "epsilon_end": 0.02},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                FinetuneSchedule(**settings)

    def test_gate_worker_count_requires_a_positive_integer(self):
        for value in (0, -1, True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ImproveConfig(gate_workers=value)
        config = ImproveConfig(gate_workers=2)
        self.assertEqual(ImproveConfig.from_dict(config.to_dict()).gate_workers, 2)

    def test_rebase_threshold_requires_a_positive_integer(self):
        for value in (0, -1, True, 1.5, "8", None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ImproveConfig(rebase_after_rejections=value)
        config = ImproveConfig(rebase_after_rejections=2)
        self.assertEqual(ImproveConfig.from_dict(config.to_dict()).rebase_after_rejections, 2)

    def test_specialist_scopes_roundtrip_without_expanding_to_master(self):
        for counts in ((2,), (3,), (4,), (2, 4), (2, 3, 4)):
            with self.subTest(counts=counts):
                config = ImproveConfig(player_counts=counts)
                restored = ImproveConfig.from_dict(config.to_dict())
                self.assertEqual(restored.player_counts, counts)
                self.assertEqual(restored, config)

    def test_invalid_player_count_tuples_are_rejected(self):
        for counts in ((), (1,), (5,), (True,), (2.0,), ("2",), (2, 2), (3, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                ImproveConfig(player_counts=counts)
        for counts in (None, [2], "234", {2, 3, 4}):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(TypeError, "player_counts must be a tuple"):
                    ImproveConfig(player_counts=counts)
                raw = ImproveConfig().to_dict()
                raw["player_counts"] = counts
                with self.assertRaisesRegex(TypeError, "player_counts must be a tuple"):
                    ImproveConfig.from_dict(raw)


class TestImproveResume(ControllerTestCase):
    def test_real_cli_partial_override_preserves_mastery_defaults_and_rejects_resume_changes(self):
        from src.cli.improve import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, output = root / "initial.pt", root / "output"
            self.make_model(model)
            self.write_solo_reference(model, 95)
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main([
                    "--initial-model", str(model), "--output-dir", str(output), "--max-seconds", "0",
                    "--learning-rate=0.00015", "--epsilon-decay-matches", "77",
                ]), 0)
            state_path = output / "latest-state.pt"
            config = improve_runtime.torch.load(state_path, weights_only=True)["league_state"]["league_config"]
            fields = ("learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay_matches")
            self.assertEqual(tuple(config[field] for field in fields), (0.00015, 0.04, 0.01, 77))
            base = ["--resume", str(state_path), "--output-dir", str(output), "--max-seconds", "0"]
            matching = base + [
                "--learning-rate", "0.00015", "--epsilon-start=0.04",
                "--epsilon-end", "0.01", "--epsilon-decay-matches=77",
            ]
            for arguments in (base, matching):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main(arguments), 0)
            before = state_path.read_bytes()
            for option, value in (
                ("--learning-rate", "0.0002"), ("--epsilon-start", "0.05"),
                ("--epsilon-end", "0.02"), ("--epsilon-decay-matches", "78"),
            ):
                for extra in ([option, value], [option + "=" + value]):
                    with self.subTest(extra=extra), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                        self.assertEqual(main(base + extra), 2)
                        self.assertIn("Resume cannot change " + option, error.getvalue())
                    self.assertEqual(state_path.read_bytes(), before)

    def test_resume_can_extend_runtime_cap_but_not_learning_settings(self):
        from src.cli.improve import _validate_resume_options

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            parser = build_arg_parser()
            arguments = [
                "--resume", "latest-state.pt", "--output-dir", "output",
                "--max-seconds", "60", "--max-cycles", "3",
                "--workers", "2", "--checkpoint-every", "3",
            ]
            _validate_resume_options(improve_runtime, controller, parser.parse_args(arguments), arguments)
            self.assertEqual(controller.config.max_cycles, 3)
            self.assertEqual(controller.config.league_matches_per_cycle, 1)
            self.assertEqual(controller.config.gate_workers, 2)
            self.assertEqual(controller.config.checkpoint_every, 3)
            unchanged = ["--resume", "latest-state.pt", "--output-dir", "output", "--max-seconds", "60"]
            _validate_resume_options(improve_runtime, controller, parser.parse_args(unchanged), unchanged)
            self.assertEqual(controller.config.gate_workers, 2)
            controller.save_checkpoint()
            restored = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"), controller.output_dir,
            )
            self.assertEqual(restored.config.gate_workers, 2)
            for changed in (["--league-matches", "999"], ["--rebase-after-rejections", "2"], ["--matches-per-round", "4"]):
                invalid = arguments + changed
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    _validate_resume_options(improve_runtime, controller, parser.parse_args(invalid), invalid)

    def test_resume_preserves_specialist_scope_unless_explicitly_changed(self):
        from src.cli.improve import _validate_resume_options

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), player_counts=(3,))
            parser = build_arg_parser()
            base = ["--resume", "latest-state.pt", "--output-dir", "output", "--max-seconds", "60"]
            for extra in ([], ["--players", "3"]):
                arguments = base + extra
                _validate_resume_options(improve_runtime, controller, parser.parse_args(arguments), arguments)
                self.assertEqual(controller.config.player_counts, (3,))
            for counts in (["2"], ["2", "3", "4"], ["3", "3"]):
                arguments = base + ["--players", *counts]
                with self.assertRaisesRegex(ValueError, "Resume cannot change --players"):
                    _validate_resume_options(improve_runtime, controller, parser.parse_args(arguments), arguments)
                self.assertEqual(controller.config.player_counts, (3,))

    def test_zero_budget_exports_truthful_flat_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.make_controller(root)
            reason = controller.run(0)
            self.assertEqual(reason, "budget")
            status = json.loads((controller.output_dir / "status.json").read_text())
            self.assertEqual(status["stop_reason"], "budget")
            self.assertEqual(status["completed_cycles"], 0)
            self.assertEqual(status["completed_episodes"], 0)
            self.assertEqual(status["algorithm"], ALGORITHM)
            self.assertEqual(status["league_matches"], 0)
            self.assertEqual(status["updates"], 0)
            self.assertEqual(status["consecutive_rejections"], 0)
            self.assertNotIn("research_episodes", status)
            self.assertEqual(status["promotions"], 0)
            self.assertIsNone(status["distance"])
            self.assertNotIn("genes", json.dumps(status).lower())
            required = {
                "latest-state.pt", "latest-model.pt", "best-model.pt", "baseline-model.pt",
                "metrics.jsonl", "registry.json", "champion.json",
            }
            self.assertTrue(required.issubset(status["files"]))
            for name, metadata in status["files"].items():
                self.assertEqual(metadata["bytes"], (controller.output_dir / name).stat().st_size)
                self.assertEqual(sha256(controller.output_dir / name), metadata["sha256"])
            self.assertNotIn("research_state", controller.state_dict())
            self.assertEqual(ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"), controller.output_dir,
            ).cycle_index, 0)

    def test_active_metrics_cannot_invalidate_published_snapshot(self):
        from src.training.improve import _append_metric_once

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.run(0)
            status = json.loads((controller.output_dir / "status.json").read_text())
            expected = status["files"]["metrics.jsonl"]["sha256"]
            _append_metric_once(controller.output_dir, "new-event", {"event": "new"}, controller.metric_ids)
            self.assertEqual(sha256(controller.output_dir / "metrics.jsonl"), expected)
            self.assertTrue((controller.output_dir / ".active-metrics.jsonl").read_text())
            controller.save_checkpoint()
            updated = json.loads((controller.output_dir / "status.json").read_text())
            self.assertEqual(updated["files"]["metrics.jsonl"]["sha256"], sha256(controller.output_dir / "metrics.jsonl"))
            self.assertNotEqual(updated["files"]["metrics.jsonl"]["sha256"], expected)
            self.assertIn("new-event", (controller.output_dir / "metrics.jsonl").read_text())

    def test_snapshot_manifest_skips_gate_directories_and_hidden_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            gates = controller.output_dir / "gates"
            gates.mkdir()
            (gates / "completed.jsonl").write_text('{"complete":true}\n')
            (controller.output_dir / "not-a-file.json").mkdir()
            (controller.output_dir / ".temporary-model.pt").write_bytes(b"unfinished")
            files = controller.save_checkpoint()
            self.assertEqual(files["latest-state.pt"]["sha256"], sha256(controller.output_dir / "latest-state.pt"))
            self.assertEqual(set(files) & {"gates", "not-a-file.json", ".temporary-model.pt"}, set())
            self.assertEqual((gates / "completed.jsonl").read_text(), '{"complete":true}\n')


class TestImproveCli(unittest.TestCase):
    def test_removed_research_options_fail_before_loading_the_training_runtime(self):
        from src.cli.improve import main

        base = ["--initial-model", "champion.pt", "--output-dir", "out", "--max-seconds", "60"]
        for option in ("--research-episodes", "--research-episodes-per-cycle"):
            with self.subTest(option=option), mock.patch(
                "src.cli.improve.load_runtime", side_effect=AssertionError("runtime loaded"),
            ), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(main(base + [option, "2"]), 2)
                self.assertIn("unrecognized arguments", error.getvalue())
                self.assertIn(option, error.getvalue())

    def test_cli_exposes_specialist_scopes_and_optional_snapshot_directory(self):
        parser = build_arg_parser()
        base = ["--initial-model", "champion.pt", "--output-dir", "out", "--max-seconds", "60"]
        defaults = parser.parse_args(base)
        self.assertIsNone(defaults.players)
        self.assertIsNone(defaults.snapshots)
        self.assertEqual(defaults.workers, 1)
        for counts in (["2"], ["3"], ["4"], ["2", "3", "4"]):
            with self.subTest(counts=counts):
                args = parser.parse_args(base + ["--players", *counts, "--snapshots", "snapshots"])
                self.assertEqual(args.players, [int(count) for count in counts])
                self.assertEqual(args.snapshots, Path("snapshots"))

    def test_cli_requires_budget_and_exposes_bounded_cycle_overrides(self):
        parser = build_arg_parser()
        max_seconds = next(action for action in parser._actions if action.dest == "max_seconds")
        self.assertTrue(max_seconds.required)
        args = parser.parse_args([
            "--initial-model", "champion.pt", "--output-dir", "out", "--max-seconds", "60",
            "--max-cycles", "1", "--league-matches-per-cycle", "2", "--matches-per-round", "4",
            "--workers", "2", "--rebase-after-rejections", "3", "--selection-blocks", "2",
            "--confirmation-blocks", "2", "--solo-games", "2",
        ])
        self.assertEqual(args.max_cycles, 1)
        self.assertEqual(args.league_matches_per_cycle, 2)
        self.assertEqual(args.matches_per_round, 4)
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.rebase_after_rejections, 3)
        self.assertEqual(args.selection_blocks, 2)
        self.assertEqual(args.confirmation_blocks, 2)
        self.assertEqual(args.solo_games, 2)


if __name__ == "__main__":
    unittest.main()
