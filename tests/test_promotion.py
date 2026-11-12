import random
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from src.checkpoints.evidence import (
    balanced_seat_orders,
    blocks_for_count,
    clustered_interval,
    evidence_metrics,
    gate_accepted,
    require_player_counts,
    stratified_interval,
)
from src.evaluation.promotion import (
    GateInterrupted,
    evaluate_gate,
)

try:
    import torch
except ImportError:
    torch = None


class TestPlayerCountEvidence(unittest.TestCase):
    def test_scaled_blocks_produce_the_selection_and_confirmation_game_budgets(self):
        for blocks, expected in (
            (32, [(128, 256), (64, 384), (32, 768)]),
            (128, [(512, 1024), (256, 1536), (128, 3072)]),
        ):
            with self.subTest(blocks=blocks):
                self.assertEqual([
                    (blocks_for_count(blocks, count),
                     blocks_for_count(blocks, count) * len(balanced_seat_orders(count)))
                    for count in (2, 3, 4)
                ], expected)

    def test_scope_requires_sorted_distinct_integer_counts(self):
        for counts in ((2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)):
            with self.subTest(counts=counts):
                self.assertEqual(require_player_counts(counts), counts)
        for counts in ((), (1,), (5,), (True,), (2.0,), ("2",), (2, 2), (3, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                require_player_counts(counts)
        for counts in (None, [2], "234", {2, 3, 4}):
            with self.subTest(counts=counts), self.assertRaises(TypeError):
                require_player_counts(counts)

    def test_metrics_average_only_the_configured_counts(self):
        multiplayer = {
            "2": {
                "blocks": 128,
                "candidate_minus_champion": 0.25,
                "placement_margins": [0.25] * 128,
                "declaration_margins": [0.0] * 128,
                "declaration_delta": 0.0,
                "games": 256,
            },
            "4": {
                "blocks": 32,
                "candidate_minus_champion": 0.5,
                "placement_margins": [0.5] * 32,
                "declaration_margins": [0.25] * 32,
                "declaration_delta": 0.25,
                "games": 768,
            },
        }
        solo = {
            "games": 256,
            "candidate_outcomes": [0] * 256,
            "champion_outcomes": [0] * 256,
            "candidate_wins": 0,
            "champion_wins": 0,
        }
        metrics = evidence_metrics(multiplayer, solo, 32, 41, 100, player_counts=[2, 4])
        self.assertEqual(metrics, {
            "multiplayer_gain": 0.375,
            "declaration_gain": 0.125,
            "multiplayer_ci95": [0.375, 0.375],
            "multiplayer_ci95_by_count": {"2": [0.25, 0.25], "4": [0.5, 0.5]},
            "solo_gain": 0.0,
            "solo_ci95": [0.0, 0.0],
        })
        self.assertTrue(gate_accepted(multiplayer, metrics, 32, 256, player_counts=(2, 4)))
        self.assertFalse(gate_accepted(multiplayer, metrics, 32, 256))
        specialist = evidence_metrics(
            {"2": multiplayer["2"]}, solo, 32, 41, 100, player_counts=(2,),
        )
        self.assertEqual(specialist["multiplayer_gain"], 0.25)
        self.assertEqual(specialist["multiplayer_ci95_by_count"], {"2": [0.25, 0.25]})
        self.assertTrue(gate_accepted({"2": multiplayer["2"]}, specialist, 32, 256, (2,)))
        for counts in ((2,), (2, 3, 4)):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "player_counts"):
                evidence_metrics(multiplayer, solo, 32, 41, 100, player_counts=counts)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_real_specialist_gate_returns_only_its_configured_results(self):
        from src.game.environment import GameConfig
        from src.model.network import QNetwork, save_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.pt"
            save_checkpoint(str(model), QNetwork(), GameConfig())
            for counts in ((2,), (3,), (4,), [2]):
                with self.subTest(counts=counts):
                    result = evaluate_gate(
                        str(model), str(model), (), seed=41, seed_blocks=1,
                        solo_games=1, max_turns=1, player_counts=counts, samples=100,
                    )
                    count = counts[0]
                    self.assertEqual(result["player_counts"], [count])
                    self.assertEqual(set(result["multiplayer"]), {str(count)})
                    self.assertEqual(
                        result["multiplayer"][str(count)]["games"],
                        {2: 8, 3: 12, 4: 24}[count],
                    )
                    self.assertEqual(result["multiplayer_gain"], 0.0)
                    self.assertEqual(result["multiplayer_ci95_by_count"], {str(count): [0.0, 0.0]})
                    self.assertFalse(result["accepted"])

    def test_gate_rejects_invalid_scopes_before_loading_models(self):
        for counts in ((), (True,), (2.0,), (2, 2), (4, 2), (5,)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                evaluate_gate("missing.pt", "missing.pt", (), 41, 1, 1, player_counts=counts)
        for counts in (None, "234", {2, 3, 4}):
            with self.subTest(counts=counts), self.assertRaises(TypeError):
                evaluate_gate("missing.pt", "missing.pt", (), 41, 1, 1, player_counts=counts)


class TestSeatClusterBootstrap(unittest.TestCase):
    def test_stratified_bootstrap_weights_counts_equally_despite_different_block_counts(self):
        self.assertEqual(stratified_interval({2: [1.0] * 128, 4: [0.0] * 32}, 100), [0.5, 0.5])
        values = {4: [-1.0, 0.0, 1.0], 2: [-1.0, 1.0], 3: [0.5]}
        before = random.getstate()
        self.assertEqual(
            stratified_interval(values, 100, 81),
            [-0.38888888888888884, 0.7222222222222222],
        )
        self.assertEqual(
            stratified_interval({count: values[count] for count in (2, 3, 4)}, 100, 81),
            [-0.38888888888888884, 0.7222222222222222],
        )
        self.assertEqual(random.getstate(), before)

    def test_stratified_bootstrap_rejects_empty_or_invalid_strata(self):
        for values in ({}, {2: []}, {2: [True]}, {3: [float("nan")]}, {4: [-1.01]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stratified_interval(values, 100)

    def test_four_players_cover_adjacent_and_opposite_contenders_equally(self):
        orders = balanced_seat_orders(4)
        self.assertEqual(len(set(orders)), 24)
        self.assertEqual(Counter(order.index(0) for order in orders), {0: 6, 1: 6, 2: 6, 3: 6})
        self.assertEqual(Counter((order.index(1) - order.index(0)) % 4 for order in orders),
                         {1: 8, 2: 8, 3: 8})

    def test_identical_contenders_cannot_gain_from_following_a_weak_opponent(self):
        orders = balanced_seat_orders(3)
        scores = []
        for order in orders:
            winner_seat = (order.index(2) + 1) % 3
            winner = order[winner_seat]
            scores.append(1 if winner == 0 else -1 if winner == 1 else 0)
        self.assertEqual(len(orders), 6)
        self.assertEqual(sum(scores), 0)
        self.assertEqual(set(orders), {
            (0, 1, 2), (1, 2, 0), (2, 0, 1),
            (1, 0, 2), (0, 2, 1), (2, 1, 0),
        })

    def test_constant_margin_has_exact_interval(self):
        for value, blocks in ((0.0, 128), (0.5, 64)):
            with self.subTest(value=value):
                self.assertEqual(
                    clustered_interval(
                        [value] * blocks,
                        samples=100,
                    ),
                    [value, value],
                )

    def test_bootstrap_preserves_global_random_state(self):
        before = random.getstate()
        values = [0, 0.5, -0.5, 1] * 32
        self.assertEqual(clustered_interval(values, samples=100), clustered_interval(values, samples=100))
        self.assertEqual(before, random.getstate())

    def test_invalid_or_nonfinite_scores_are_rejected(self):
        for values in (
            [],
            [float("nan")],
            [float("inf")],
            [2.0],
            [True],
            ["0.5"],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                clustered_interval(values, samples=100)

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_gate_restores_torch_threads_when_checkpoint_loading_fails(self):
        from src.game.environment import GameConfig
        from src.model.network import QNetwork, save_checkpoint
        from src.model.spec import ModelSpec

        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.pt"
                save_checkpoint(str(path), QNetwork(ModelSpec(8, 1, 1)), GameConfig())
                with mock.patch(
                    "src.evaluation.promotion.CheckpointPolicy",
                    side_effect=RuntimeError("bad checkpoint"),
                ), self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                    evaluate_gate(str(path), str(path), (), seed=1, seed_blocks=32, solo_games=256)
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(previous)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestGateExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        from src.game.environment import GameConfig
        from src.model.network import QNetwork, save_checkpoint
        from src.model.spec import ModelSpec

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate, self.champion = self.root / "candidate.pt", self.root / "champion.pt"
        with torch.random.fork_rng(devices=[]):
            for path, seed in ((self.candidate, 17), (self.champion, 31)):
                torch.manual_seed(seed)
                save_checkpoint(str(path), QNetwork(ModelSpec(8, 1, 1)), GameConfig())

    def gate(self, **overrides):
        arguments = {
            "candidate_path": str(self.candidate), "champion_path": str(self.champion),
            "opponent_paths": (), "seed": 41, "seed_blocks": 1,
            "solo_games": 35, "max_turns": 1, "samples": 100,
        }
        return evaluate_gate(**{**arguments, **overrides})

    def test_loser_placement_improves_without_either_contender_declaring(self):
        from src.evaluate import HandPenalty
        from src.game.outcome import DealOutcome

        def third_seat_wins(deals, *_args, **_kwargs):
            matches = []
            for deal in deals:
                candidate, champion, winner = (deal.seats.index(index) for index in range(3))
                points = [0] * 3
                points[candidate], points[champion] = 40, 80
                outcome = DealOutcome(winner, "win", tuple(
                    HandPenalty(value, (), (0,) * 52) for value in points
                ))
                matches.append(SimpleNamespace(winner=winner, outcome=outcome))
            return matches

        with mock.patch("src.evaluation.promotion.run_matches", side_effect=third_seat_wins):
            result = self.gate(player_counts=(3,), solo_games=1)
        self.assertEqual(result["multiplayer"]["3"], {
            "blocks": 2, "games": 12,
            "placement_margins": [1.0, 1.0], "declaration_margins": [0.0, 0.0],
            "candidate_minus_champion": 1.0, "declaration_delta": 0.0,
        })
        self.assertEqual(result["multiplayer_gain"], 1.0)
        self.assertEqual(result["declaration_gain"], 0.0)
        self.assertEqual(result["multiplayer_ci95"], [1.0, 1.0])

    def test_parallel_workers_produce_byte_identical_evidence_to_in_process_execution(self):
        serial = self.gate(workers=1)
        self.assertEqual(
            [(raw["blocks"], raw["games"]) for raw in serial["multiplayer"].values()],
            [(4, 8), (2, 12), (1, 24)],
        )
        self.assertEqual(serial["solo"]["candidate_outcomes"], [0] * 35)
        for workers in (2, 4):
            with self.subTest(workers=workers):
                parallel = self.gate(workers=workers)
                self.assertEqual(serial["multiplayer"], parallel["multiplayer"])
                self.assertEqual(serial["solo"], parallel["solo"])
                self.assertEqual(serial["multiplayer_ci95"], parallel["multiplayer_ci95"])
                self.assertEqual(json.dumps(serial), json.dumps(parallel))

    def test_interrupted_journal_preserves_blocks_and_resume_skips_them(self):
        from src.evaluation.arena import run_matches

        journal_dir = self.root / "gates"
        played = []

        def interrupt(deals, *args, **kwargs):
            if len(played) == 2:
                raise GateInterrupted("injected interruption")
            played.append((deals[0].players, deals[0].seed))
            return run_matches(deals, *args, **kwargs)

        with mock.patch("src.evaluation.promotion.run_matches", side_effect=interrupt):
            with self.assertRaisesRegex(GateInterrupted, "injected interruption"):
                self.gate(journal_dir=journal_dir)
        journal, = journal_dir.glob("*.jsonl")
        prefix = journal.read_bytes()
        records = [json.loads(line) for line in prefix.splitlines()]
        self.assertEqual([record["job"] for record in records], [
            ["multiplayer", 2, 0], ["multiplayer", 2, 1],
        ])
        self.assertEqual(played, [(2, 41), (2, 42)])
        with journal.open("a", encoding="utf-8") as stream:
            stream.write('{"interrupted":')
        played.clear()

        def record(deals, *args, **kwargs):
            played.append((deals[0].players, deals[0].seed))
            return run_matches(deals, *args, **kwargs)

        with mock.patch("src.evaluation.promotion.run_matches", side_effect=record):
            resumed = self.gate(journal_dir=journal_dir)
        self.assertEqual(played, [(2, 43), (2, 44), (3, 41), (3, 42), (4, 41)])
        self.assertTrue(journal.read_bytes().startswith(prefix))
        resumed_records = [json.loads(line) for line in journal.read_bytes().splitlines()]
        self.assertEqual(len(resumed_records), 9)
        self.assertTrue(journal.read_bytes().endswith(b"\n"))
        uninterrupted = self.gate()
        self.assertEqual(json.dumps(resumed), json.dumps(uninterrupted))
        with mock.patch("src.evaluation.promotion._GateRunner", side_effect=AssertionError("replayed cached work")):
            cached = self.gate(journal_dir=journal_dir, workers=2)
        self.assertEqual(json.dumps(cached), json.dumps(uninterrupted))

    def test_resume_skips_completed_paired_solo_slices(self):
        from src.evaluation.benchmark import evaluate_games

        journal_dir = self.root / "solo-journal"

        def interrupt(network, rules, games, seed, **kwargs):
            if seed == 100_073:
                raise GateInterrupted("second solo slice")
            return evaluate_games(network, rules, games, seed, **kwargs)

        with mock.patch("src.evaluation.promotion.evaluate_games", side_effect=interrupt):
            with self.assertRaisesRegex(GateInterrupted, "second solo slice"):
                self.gate(journal_dir=journal_dir)
        journal, = journal_dir.glob("*.jsonl")
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(len(records), 8)
        self.assertEqual(records[-1]["job"], ["solo", 0, 32])
        self.assertEqual(records[-1]["result"]["candidate_outcomes"], [0] * 32)
        with mock.patch(
            "src.evaluation.promotion.run_matches", side_effect=AssertionError("replayed completed block"),
        ), mock.patch(
            "src.evaluation.promotion.evaluate_games", wraps=evaluate_games,
        ) as solo:
            resumed = self.gate(journal_dir=journal_dir)
        self.assertEqual([call.args[3] for call in solo.call_args_list], [100_073, 100_073])
        self.assertEqual(json.dumps(resumed), json.dumps(self.gate()))

    def test_parallel_task_timeout_is_a_failure_and_restores_parent_threads(self):
        with self.assertRaisesRegex(RuntimeError, "Gate job timed out") as failure:
            self.gate(workers=2, task_seconds=0.000001)
        self.assertNotIsInstance(failure.exception, GateInterrupted)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_serial_task_timeouts_are_distinct_from_the_global_deadline(self):
        from src.evaluation.arena import MatchInterrupted

        for error in (TimeoutError("task deadline"), MatchInterrupted("match deadline")):
            with mock.patch("src.evaluation.promotion._GateRunner.run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "Gate job timed out") as failure:
                    self.gate(deadline=time.monotonic() + 120)
            self.assertNotIsInstance(failure.exception, GateInterrupted)
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaises(GateInterrupted):
                self.gate(workers=workers, deadline=0)

    def test_a_completed_serial_result_is_journaled_before_the_next_deadline_check(self):
        clock = [0.0]
        deadlines = []

        def finish(_runner, _job, *, deadline):
            deadlines.append(deadline)
            clock[0] = 11.0
            return {"placement_margin": 0.0, "declaration_margin": 0.0}

        directory = self.root / "finished-job"
        with mock.patch("src.evaluation.promotion.time.monotonic", side_effect=lambda: clock[0]), mock.patch(
            "src.evaluation.promotion._GateRunner.run", autospec=True, side_effect=finish,
        ), self.assertRaises(GateInterrupted):
            self.gate(journal_dir=directory, deadline=10.0, task_seconds=5.0)
        journal, = directory.glob("*.jsonl")
        records = [json.loads(line) for line in journal.read_bytes().splitlines()]
        self.assertEqual(deadlines, [5.0])
        self.assertEqual([record["job"] for record in records], [["multiplayer", 2, 0]])
        self.assertEqual(records[0]["result"], {"placement_margin": 0.0, "declaration_margin": 0.0})

    def test_settled_journal_corruption_is_never_silently_replayed(self):
        directory = self.root / "corrupt"
        self.gate(journal_dir=directory, player_counts=(2,), solo_games=1)
        journal, = directory.glob("*.jsonl")
        original = journal.read_bytes()
        record = json.loads(original.splitlines()[0])
        corruptions = [
            b"{broken}\n",
            (json.dumps({**record, "job": ["multiplayer", 2, 99]}) + "\n").encode(),
            (json.dumps({**record, "result": {"placement_margin": 2, "declaration_margin": 0}}) + "\n").encode(),
            (json.dumps({**record, "gate_id": "wrong"}) + "\n").encode(),
        ]
        for corruption in corruptions:
            with self.subTest(corruption=corruption):
                journal.write_bytes(corruption + original)
                with self.assertRaisesRegex(ValueError, "Malformed gate journal"):
                    self.gate(journal_dir=directory, player_counts=(2,), solo_games=1)

    def test_source_changes_use_a_different_journal_and_evidence_identity(self):
        directory = self.root / "source-bound"
        with mock.patch("src.evaluation.promotion.source_fingerprint", return_value="a" * 64):
            first = self.gate(journal_dir=directory, player_counts=(2,), solo_games=1)
        with mock.patch("src.evaluation.promotion.source_fingerprint", return_value="b" * 64):
            second = self.gate(journal_dir=directory, player_counts=(2,), solo_games=1)
        self.assertEqual(first["source_sha256"], "a" * 64)
        self.assertEqual(second["source_sha256"], "b" * 64)
        self.assertEqual(first["multiplayer"], second["multiplayer"])
        self.assertEqual(len(list(directory.glob("*.jsonl"))), 2)

    def test_gate_runner_rejects_checkpoint_swaps_before_loading_the_network(self):
        from src.checkpoints.io import file_sha256
        from src.evaluation.promotion import _GateRunner
        from src.game.environment import GameConfig

        hashes = [file_sha256(path) for path in (self.candidate, self.champion)]
        self.candidate.write_bytes(b"replaced checkpoint")
        with mock.patch("src.evaluation.arena.load_checkpoint", side_effect=AssertionError("loaded swapped model")):
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                _GateRunner(str(self.candidate), str(self.champion), (), GameConfig(), 41, hashes)

    def test_worker_source_mismatch_fails_before_loading_checkpoints(self):
        from src.evaluation.promotion import _init_gate_worker
        from src.game.environment import GameConfig

        with self.assertRaisesRegex(RuntimeError, "source differs"):
            _init_gate_worker(
                expected_source="wrong", candidate_path="missing.pt", champion_path="missing.pt",
                opponent_paths=(), rules=GameConfig(), seed=41, expected_hashes=(),
            )

    def test_invalid_execution_settings_fail_before_loading_checkpoints(self):
        for arguments in (
            {"workers": 0}, {"workers": True}, {"workers": 1.5},
            {"task_seconds": 0}, {"task_seconds": float("nan")}, {"samples": 99},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                evaluate_gate("missing.pt", "missing.pt", (), 41, 1, 1, **arguments)


if __name__ == "__main__":
    unittest.main()
