import math
from random import Random
from struct import unpack
import unittest
from unittest.mock import patch

from src.game.environment import ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, PappluEnv, Phase, discard_action
from src.model.encoding import HEADER_TOKENS, SCHEMA, legal_mask
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import Deal, RandomPolicy, TransformerPolicy, Turn, head_to_head, resolve_policy, run_matches
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

    def test_search_takes_a_forced_win_that_the_neural_prior_misses(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=1)
        env = PappluEnv(game)
        env._hand = [int(face in (1, 2, 25)) for face in range(52)]
        env._joker, env._discard, env._stock = 51, [3], [27]
        env._turns_remaining, env._phase = 1, Phase.DRAW
        env._start_history()
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy.bias[[0, discard_action(1)]] = 10
        blind = TransformerPolicy(model, game, record=True).open()
        search = TransformerPolicy(model, game, search_ply=1, record=True, sample=True, uniform_mix=1).open()
        obs = env.observe()
        draw = Turn(0, obs, Random(0), legal_mask(obs))
        with patch("src.evaluation.arena.is_valid_hand", side_effect=AssertionError("0-ply checked the rules")):
            self.assertEqual(blind.act([draw]), [0])
        self.assertEqual(search.act([draw]), [ACTION_TAKE_DISCARD])
        obs = env.step(ACTION_TAKE_DISCARD)
        discard = Turn(0, obs, Random(0), legal_mask(obs))
        self.assertEqual(blind.act([discard]), [discard_action(1)])
        self.assertEqual(search.act([discard]), [discard_action(25)])
        self.assertTrue(env.step(discard_action(25)).won)
        trace = search.finish(0)
        mass = unpack("=108H", trace.target_probs)
        self.assertEqual(mass[:2], (0, 65535))
        self.assertEqual(sum(mass[54:]), 65535)
        self.assertEqual(mass[54 + discard_action(25)], 65535)
        self.assertEqual(trace.behavior_logp, (0.0, 0.0))
        self.assertEqual(trace.forced, bytes((1, 1)))
        self.assertEqual(blind.finish(0).forced, bytes(2))

    def test_search_records_uniform_winning_discards_even_with_exploration(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=1)
        env = PappluEnv(game)
        env._hand = [int(face in (0, 1, 2)) for face in range(52)]
        env._joker, env._discard, env._stock = 51, [27], [3]
        env._turns_remaining, env._phase = 1, Phase.DRAW
        env._start_history()
        obs = env.step(0)
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        actor = TransformerPolicy(model, game, search_ply=1, record=True, sample=True, uniform_mix=1).open()
        action, = actor.act([Turn(0, obs, Random(7), legal_mask(obs))])
        trace = actor.finish(0)
        self.assertTrue(env.step(action).won)
        self.assertEqual(unpack("=54H", trace.target_probs)[2:6], (32768, 0, 0, 32768))
        self.assertAlmostEqual(trace.behavior_logp[0], -math.log(2))
        self.assertEqual(trace.forced, bytes((1,)))

    def test_search_preserves_prior_for_nonwinning_actions(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=2)
        env = PappluEnv(game)
        env._hand = [int(face in (1, 6, 14)) for face in range(52)]
        env._joker, env._discard, env._stock = 51, [20], [28]
        env._turns_remaining, env._phase = 2, Phase.DRAW
        env._start_history()
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy.bias[[0, discard_action(1)]] = 0.5

        blind = TransformerPolicy(model, game, record=True).open()
        search = TransformerPolicy(model, game, search_ply=1, record=True).open()
        obs = env.observe()
        with patch.object(model, "evaluate_candidates", side_effect=AssertionError("search evaluated afterstates")):
            for expected in (ACTION_DRAW_STOCK, discard_action(1)):
                turn = Turn(0, obs, Random(0), legal_mask(obs))
                self.assertEqual(blind.act([turn]), [expected])
                self.assertEqual(search.act([turn]), [expected])
                obs = env.step(expected)
        self.assertFalse(obs.won)
        self.assertEqual(search.finish(0), blind.finish(0))

    def test_refills_a_fast_solo_slot_before_a_six_seat_table_ends(self):
        actor = FirstLegal()
        solo, large, next_solo = Deal(5, (0,)), Deal(17, (0,) * 6), Deal(21, (0,))
        results = list(run_matches((solo, large, next_solo), (actor,), self.game, tables=2))
        self.assertEqual([result.deal for result in results], [solo, next_solo, large])
        self.assertEqual([result.action_count for result in results], [2, 2, 12])
        self.assertIn(12, actor.batches[2])
        self.assertEqual(actor.finished, {0, 6, 7, 8, 9, 10, 11, 12})
        self.assertEqual(results[-1].outcome.ending, "turns_exhausted")
        self.assertAlmostEqual(sum(results[-1].outcome.utilities()), -3)
        self.assertLessEqual(max(results[-1].outcome.utilities()), -0.2)

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
                self.assertEqual(trace.forced, bytes(len(trace.action)))
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

    def test_paired_solo_comparison_and_multiplayer_timeout_payoff(self):
        solo = head_to_head(RandomPolicy(), RandomPolicy(), self.game, 1, 3, 10, tables=2)
        multi = head_to_head(RandomPolicy(), RandomPolicy(), self.game, 3, 3, 10, tables=2)
        self.assertEqual((solo.advantage, solo.deals), (0.0, 6))
        self.assertEqual(multi.deals, 9)
        self.assertAlmostEqual(multi.advantage, -0.5)
        self.assertEqual((multi.win_rate, multi.opponent_win_rate), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            resolve_policy("heuristic")
