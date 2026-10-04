from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from src.checkpoints.locking import exclusive_writer
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
