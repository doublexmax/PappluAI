import math
import unittest

from src.game.environment import GameConfig
from src.model.encoding import HEADER_TOKENS, SCHEMA
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import Deal, RandomPolicy, TransformerPolicy, head_to_head, resolve_policy, run_matches
    from src.model.network import GameTransformer


class FirstLegal:
    game = None

    def __init__(self):
        self.batches = []
        self.finished = set()

    def open(self):
        return self

    def act(self, turns):
        self.batches.append(tuple(turn.stream for turn in turns))
        return [(turn.legal & -turn.legal).bit_length() - 1 for turn in turns]

    def finish(self, stream):
        if stream in self.finished:
            raise AssertionError("stream was finished twice")
        self.finished.add(stream)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestArena(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = GameConfig(max_turns=1)

    def test_refills_a_fast_solo_slot_before_a_six_seat_table_ends(self):
        actor = FirstLegal()
        solo, large, next_solo = Deal(5, (0,)), Deal(17, (0,) * 6), Deal(21, (0,))
        results = list(run_matches((solo, large, next_solo), (actor,), self.game, tables=2))
        self.assertEqual([result.deal for result in results], [solo, next_solo, large])
        self.assertEqual([result.action_count for result in results], [2, 2, 12])
        self.assertIn(12, actor.batches[2])
        self.assertEqual(actor.finished, {0, 6, 7, 8, 9, 10, 11, 12})
        self.assertAlmostEqual(sum(results[-1].outcome.utilities()), 0)

    def test_recorded_uniform_behavior_uses_only_present_legal_actions(self):
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        policy = TransformerPolicy(model, self.game, sample=True, uniform_mix=1, record=True, precision="bf16")
        deals = (Deal(19, (0,), curriculum=1), Deal(37, (0, 0)))
        results = list(run_matches(deals, (policy,), self.game, tables=2))
        self.assertEqual(sum(results[0].telemetry.stock_draws + results[0].telemetry.discard_draws), 1)
        for result in results:
            traces = [trace for trace in result.traces if trace is not None]
            self.assertEqual(sum(len(trace.action) for trace in traces), result.action_count)
            for trace in traces:
                self.assertEqual(len(trace.prefix) // 8, trace.cut[-1])
                self.assertGreater(trace.cut[0], HEADER_TOKENS)
                self.assertEqual(len(trace.suffix), 240 * len(trace.action))
                for action, legal, logp in zip(trace.action, trace.legal, trace.behavior_logp):
                    self.assertTrue(legal & (1 << action))
                    self.assertAlmostEqual(logp, -math.log(legal.bit_count()))

    def test_deal_distinguishes_cold_solo_puzzle_and_multiplayer_curriculum(self):
        cold = Deal(19, (0,))
        puzzle = Deal(19, (0,), curriculum=0)
        multi = Deal(19, (0, 0), curriculum=2)
        results = {result.deal: result for result in run_matches((cold, puzzle, multi), (FirstLegal(),), self.game)}
        self.assertEqual((results[cold].action_count, results[puzzle].action_count), (2, 1))
        self.assertEqual(results[puzzle].telemetry.stock_draws, (0,))
        self.assertIn(results[multi].action_count, (2, 4))
        with self.assertRaisesRegex(ValueError, "solo discard puzzle"):
            list(run_matches((Deal(19, (0, 0), curriculum=0),), (FirstLegal(),), self.game))

    def test_paired_self_comparison_is_neutral(self):
        for players in (1, 3):
            with self.subTest(players=players):
                result = head_to_head(RandomPolicy(), RandomPolicy(), self.game, players, 3, 10, tables=2)
                self.assertAlmostEqual(result.advantage, 0)
                self.assertEqual(result.deals, 3 * max(2, players))
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            resolve_policy("heuristic")
