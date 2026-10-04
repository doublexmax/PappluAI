from __future__ import annotations

import os
import multiprocessing
import time
import unittest

from src.evaluation.execution import BoundedWorkers, WorkerFailure, WorkerResult


def square(value):
    return value * value


def sleep_or_return(value):
    if value == "hang":
        time.sleep(60)
    return value


def exit_worker(value):
    os._exit(7)


class TestBoundedWorkers(unittest.TestCase):
    def test_completed_jobs_and_repeated_batches_preserve_inputs(self):
        with BoundedWorkers(square, workers=2, task_seconds=5) as pool:
            first = list(pool.run((3, 1, 2)))
            second = list(pool.run((4,)))
        self.assertTrue(all(isinstance(result, WorkerResult) for result in first + second))
        self.assertEqual({result.job: result.value for result in first}, {3: 9, 1: 1, 2: 4})
        self.assertEqual(second[0].value, 16)

    def test_hung_job_is_terminated_and_later_jobs_can_finish(self):
        with BoundedWorkers(sleep_or_return, workers=1, task_seconds=0.2) as pool:
            results = list(pool.run(("hang", "next")))
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], WorkerFailure)
        self.assertEqual(results[0].job, "hang")
        self.assertEqual(results[0].kind, "timeout")
        self.assertLess(results[0].elapsed_seconds, 5)
        self.assertIsInstance(results[1], WorkerResult)
        self.assertEqual(results[1].value, "next")

    def test_worker_exit_is_not_a_game_result(self):
        with BoundedWorkers(exit_worker, workers=1, task_seconds=5) as pool:
            results = list(pool.run(("case",)))
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], WorkerFailure)
        self.assertEqual(results[0].kind, "worker_exit")
        self.assertIn("7", results[0].detail)

    def test_leaving_context_cancels_owned_workers(self):
        before = {process.pid for process in multiprocessing.active_children()}
        pool = BoundedWorkers(sleep_or_return, workers=2, task_seconds=10)
        with pool:
            outcomes = pool.run(("done", "hang"))
            self.assertIsInstance(next(outcomes), WorkerResult)
        self.assertEqual({process.pid for process in multiprocessing.active_children()}, before)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            next(outcomes)

    def test_invalid_limits_are_rejected(self):
        for arguments in ({"workers": True}, {"workers": 0}, {"task_seconds": 0}, {"task_seconds": float("nan")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BoundedWorkers(square, **arguments)


if __name__ == "__main__":
    unittest.main()
