from __future__ import annotations

import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.checkpoints.locking import exclusive_writer
from src.cli.calibration import main
from src.evaluation.calibration import (
    CalibrationConfig,
    calibration_cases,
    summarize_calibration,
    upper_event_probability,
    validate_case_result,
)


def completed_record(case, winner=0):
    turns = (
        [case.cap] * case.players if winner is None
        else [int(seat <= winner) for seat in range(case.players)]
    )
    return {
        "case": case.payload(),
        "status": "completed",
        "result": {
            "seed": case.seed, "winner": winner,
            "terminal_reason": "turns_exhausted" if winner is None else "win",
            "seat_turns": turns, "action_count": 2 * sum(turns),
            "stock_remaining": 156 - 21 * case.players - 2,
            "telemetry": {
                "stock_draws": [0] * case.players,
                "discard_draws": turns.copy(),
                "refill_turns": [],
            },
        },
    }


class TestCalibrationSchedule(unittest.TestCase):
    def test_caps_share_deals_and_seats_are_randomized_independently_of_the_deck(self):
        config = CalibrationConfig(caps=(20, 60), blocks=32)
        cases = calibration_cases(config, ("a", "b", "c", "d", "e"))
        self.assertEqual(len(cases), 192)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        for players in config.player_counts:
            for block in range(config.blocks):
                grouped = [case for case in cases if (case.players, case.block) == (players, block)]
                self.assertEqual(len(grouped), 2)
                self.assertEqual({case.cap for case in grouped}, {20, 60})
                self.assertEqual(grouped[0].seed, grouped[1].seed)
                self.assertEqual(grouped[0].policies, grouped[1].policies)
                self.assertEqual(grouped[0].seat_order, grouped[1].seat_order)
            self.assertGreater(
                len({case.seat_order for case in cases if case.players == players}), 1,
            )
        self.assertEqual(cases, calibration_cases(config, ("a", "b", "c", "d", "e")))

    def test_invalid_configuration_and_duplicate_competitors_fail(self):
        for arguments in (
            {"blocks": True}, {"caps": (1, 1)}, {"player_counts": (1,)},
            {"workers": 0}, {"seed": -1}, {"match_seconds": float("nan")},
            {"max_extra_completion_rate": 0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CalibrationConfig(**arguments)
        with self.assertRaises(ValueError):
            calibration_cases(CalibrationConfig(), ("a", "a", "b", "c"))


class TestCalibrationEvidence(unittest.TestCase):
    def test_default_budget_can_support_each_table_size_with_zero_extra_events(self):
        config = CalibrationConfig()
        cases = calibration_cases(config, ("a", "b", "c", "d"))
        records = [completed_record(case) for case in cases]
        for case, record in zip(cases, records):
            validate_case_result(record, case)
        report = summarize_calibration(records, config)
        self.assertEqual(
            [(row["players"], row["cap"], row["status"]) for row in report["recommendations"]],
            [(2, 30, "supported_for_recorded_pool"), (3, 30, "supported_for_recorded_pool"),
             (4, 30, "supported_for_recorded_pool")],
        )
        for row in report["groups"]:
            if row["cap"] != 120:
                self.assertLess(
                    row["extra_completion_probability_upper"], config.max_extra_completion_rate,
                )

    def test_six_comparison_zero_event_boundary_is_94_blocks(self):
        for blocks, status, cap in (
            (93, "insufficient_evidence", None),
            (94, "supported_for_recorded_pool", 30),
        ):
            with self.subTest(blocks=blocks):
                config = CalibrationConfig(blocks=blocks)
                upper = upper_event_probability(0, blocks, 6)
                if cap is None:
                    self.assertGreater(upper, config.max_extra_completion_rate)
                else:
                    self.assertLessEqual(upper, config.max_extra_completion_rate)
                records = [
                    completed_record(case)
                    for case in calibration_cases(config, ("a", "b", "c", "d"))
                ]
                report = summarize_calibration(records, config)
                self.assertEqual(
                    [(row["players"], row["cap"], row["status"]) for row in report["recommendations"]],
                    [(2, cap, status), (3, cap, status), (4, cap, status)],
                )

    def test_winner_reversals_are_accepted_as_completion_only_evidence(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=128)
        records = [
            completed_record(case, winner=0 if case.cap == 1 else 1)
            for case in calibration_cases(config, ("a", "b"))
        ]
        report = summarize_calibration(records, config)
        self.assertEqual(report["recommendations"][0]["cap"], 1)
        self.assertEqual(report["groups"][0]["extra_completion_blocks"], 0)
        self.assertEqual(report["selection_rule"]["scope"], "completion_only")
        self.assertEqual(
            report["selection_rule"]["not_validated"],
            ["winner", "placement", "ranking", "capped_game_weight"],
        )
        self.assertTrue(report["selection_rule"]["cap_conditions_policy"])
        self.assertFalse(report["selection_rule"]["identical_action_prefix_guaranteed"])

    def test_both_all_capped_runs_are_accepted_without_any_declarations(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=128)
        records = [
            completed_record(case, winner=None)
            for case in calibration_cases(config, ("a", "b"))
        ]
        report = summarize_calibration(records, config)
        self.assertEqual(
            [(row["declarations"], row["capped_games"]) for row in report["groups"]],
            [(0, 128), (0, 128)],
        )
        self.assertEqual(report["recommendations"][0]["cap"], 1)
        self.assertEqual(report["recommendations"][0]["status"], "supported_for_recorded_pool")

    def test_one_successful_block_does_not_establish_a_safe_cap(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=1)
        cases = calibration_cases(config, ("a", "b"))
        report = summarize_calibration([completed_record(case) for case in cases], config)
        decision = report["recommendations"][0]
        self.assertIsNone(decision["cap"])
        self.assertEqual(decision["status"], "insufficient_evidence")
        self.assertAlmostEqual(upper_event_probability(0, 1, 1), 0.95)

    def test_complete_independent_blocks_can_support_a_smaller_cap(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=128)
        cases = calibration_cases(config, ("a", "b"))
        records = [completed_record(case) for case in cases]
        report = summarize_calibration(records, config)
        self.assertEqual(report["recommendations"][0]["cap"], 1)
        row = report["groups"][0]
        self.assertEqual(row["complete_paired_blocks"], 128)
        self.assertEqual(row["extra_completion_blocks"], 0)
        self.assertLess(row["extra_completion_probability_upper"], 0.05)
        incomplete = summarize_calibration(records[:-1], config)
        self.assertIsNone(incomplete["recommendations"][0]["cap"])
        self.assertEqual(incomplete["recommendations"][0]["status"], "incomplete_experiment")

    def test_multiple_caps_do_not_inflate_the_independent_deal_count(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(4,), blocks=128)
        cases = calibration_cases(config, ("a", "b", "c", "d"))
        records = [
            completed_record(case, winner=None if case.cap == 1 and case.block == 0 else 0)
            for case in cases
        ]
        report = summarize_calibration(records, config)
        self.assertEqual(report["groups"][0]["capped_games"], 1)
        self.assertEqual(report["groups"][0]["extra_completion_blocks"], 1)
        self.assertEqual(report["groups"][0]["complete_paired_blocks"], 128)

    def test_duplicate_deal_observations_are_rejected(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=1)
        record = completed_record(calibration_cases(config, ("a", "b"))[0])
        with self.assertRaisesRegex(ValueError, "one independently randomized seating"):
            summarize_calibration([record, record], config)

    def test_guard_is_not_used_as_a_success_shaped_fallback(self):
        config = CalibrationConfig(caps=(1, 2), player_counts=(2,), blocks=64)
        cases = calibration_cases(config, ("a", "b"))
        records = [completed_record(case, winner=None if case.cap == 1 else 0) for case in cases]
        report = summarize_calibration(records, config)
        self.assertIsNone(report["recommendations"][0]["cap"])
        self.assertEqual(report["recommendations"][0]["guard_cap"], 2)

    def test_saved_record_validation_checks_actual_work_and_endings(self):
        case = calibration_cases(
            CalibrationConfig(caps=(2,), player_counts=(2,), blocks=1), ("a", "b"),
        )[0]
        valid = completed_record(case)
        validate_case_result(valid, case)
        for change in (
            lambda value: value["result"].update(action_count=0),
            lambda value: value["result"].update(winner=True),
            lambda value: value["result"]["telemetry"].update(refill_turns=[1]),
            lambda value: value["result"]["telemetry"].update(stock_draws=[1, 0]),
            lambda value: value.update(status="interrupted"),
        ):
            broken = copy.deepcopy(valid)
            change(broken)
            with self.assertRaises(ValueError):
                validate_case_result(broken, case)


class TestCalibrationCLI(unittest.TestCase):
    def test_cli_default_matches_a_viable_config_default(self):
        configs = []

        def run_synthetic(models_dir, output_dir, config):
            self.assertEqual((models_dir, output_dir), (Path("models"), Path("results")))
            configs.append(config)
            records = [
                completed_record(case)
                for case in calibration_cases(config, ("a", "b", "c", "d"))
            ]
            return {**summarize_calibration(records, config), "execution_complete": True}

        runtime = SimpleNamespace(CalibrationConfig=CalibrationConfig, run_calibration=run_synthetic)
        with mock.patch("src.cli.calibration.load_runtime", return_value=runtime), mock.patch(
            "sys.stdout", new_callable=io.StringIO,
        ) as output:
            self.assertEqual(main(["--models-dir", "models", "--output-dir", "results"]), 0)
        self.assertEqual(configs, [CalibrationConfig()])
        for players in (2, 3, 4):
            self.assertIn(
                "%d players: supported_for_recorded_pool, cap=30" % players, output.getvalue(),
            )

    def test_help_does_not_import_torch(self):
        script = (
            "import builtins,runpy,sys;"
            "real=builtins.__import__;"
            "builtins.__import__=lambda name,*a,**k: "
            "(_ for _ in ()).throw(AssertionError('help imported torch')) "
            "if name=='torch' or name.startswith('torch.') else real(name,*a,**k);"
            "sys.argv=['src.cli.calibration','--help'];"
            "runpy.run_module('src.cli.calibration',run_name='__main__')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--blocks", result.stdout)


class TestWriterOwnership(unittest.TestCase):
    def test_only_one_writer_owns_a_directory_and_release_allows_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".writer.lock"
            with exclusive_writer(path):
                with self.assertRaisesRegex(RuntimeError, "another writer"):
                    with exclusive_writer(path):
                        self.fail("a second writer acquired the same lock")
            with exclusive_writer(path):
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
