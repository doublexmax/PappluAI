import random
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.checkpoints.evidence import (
    balanced_seat_orders,
    clustered_interval,
    evidence_metrics,
    gate_accepted,
    require_player_counts,
)
from src.evaluation.promotion import (
    evaluate_gate,
)

try:
    import torch
except ImportError:
    torch = None


class TestPlayerCountEvidence(unittest.TestCase):
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
                "candidate_minus_champion": 0.25,
                "seed_block_margins": [0.25] * 32,
                "games": 64,
            },
            "4": {
                "candidate_minus_champion": 0.5,
                "seed_block_margins": [0.5] * 32,
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
                        {2: 2, 3: 6, 4: 24}[count],
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
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with mock.patch(
                "src.evaluation.promotion.CheckpointPolicy",
                side_effect=RuntimeError("bad checkpoint"),
            ), self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                evaluate_gate(
                    "candidate.pt",
                    "champion.pt",
                    (),
                    seed=1,
                    seed_blocks=32,
                    solo_games=256,
                )
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
