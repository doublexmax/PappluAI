import math
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluate import HandPenalty
from src.game.environment import ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, PappluEnv, Phase, discard_action
from src.game.outcome import DealOutcome
from src.model.encoding import HEADER_TOKENS, SCHEMA, legal_mask
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import (
        Deal, RandomPolicy, TransformerPolicy, Turn, head_to_head, rate_checkpoints, resolve_policy, run_matches, solve_elo,
    )
    from src.model.network import GameTransformer, save_checkpoint


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

    def test_actor_follows_legal_neural_logits_even_when_a_win_is_available(self):
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
            model.policy.bias[[ACTION_DRAW_STOCK, discard_action(1)]] = 10
        actor = TransformerPolicy(model, game, record=True).open()
        obs = env.observe()
        draw = Turn(0, obs, Random(0), legal_mask(obs))
        self.assertEqual(actor.act([draw]), [ACTION_DRAW_STOCK])
        obs = env.step(ACTION_TAKE_DISCARD)
        discard = Turn(0, obs, Random(0), legal_mask(obs))
        self.assertEqual(actor.act([discard]), [discard_action(1)])
        self.assertTrue(env.step(discard_action(25)).won)
        trace = actor.finish(0)
        self.assertEqual(trace.action, bytes((ACTION_DRAW_STOCK, discard_action(1))))
        self.assertEqual(trace.behavior_logp, (0.0, 0.0))

    def test_refills_a_fast_solo_slot_before_a_six_seat_table_ends(self):
        actor = FirstLegal()
        solo, large, next_solo = Deal(5, (0,)), Deal(17, (0,) * 6), Deal(21, (0,))
        results = list(run_matches((solo, large, next_solo), (actor,), self.game, tables=2))
        self.assertEqual([result.deal for result in results], [solo, next_solo, large])
        self.assertEqual([result.action_count for result in results], [2, 2, 12])
        self.assertIn(12, actor.batches[2])
        self.assertEqual(actor.finished, {0, 6, 7, 8, 9, 10, 11, 12})
        self.assertEqual(results[-1].outcome.ending, "turns_exhausted")
        self.assertEqual(sum(results[-1].outcome.utilities()), 0)
        self.assertLessEqual(max(results[-1].outcome.utilities()), 0.475)

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

    def test_solo_compares_declarations_then_deadwood_on_each_seed(self):
        examples = (
            ((True, 0), (False, 0)), ((False, 8), (False, 8)), ((False, 7), (False, 5)),
            ((True, 0), (True, 0)), ((False, 2), (False, 5)),
        )

        def finish(deals, *args, **kwargs):
            for deal in reversed(tuple(deals)):
                won, points = examples[deal.seed - 17][deal.seats[0]]
                outcome = DealOutcome(0 if won else None, "win" if won else "turns_exhausted",
                                      (HandPenalty(points, (), ()),), 210)
                yield SimpleNamespace(deal=deal, winner=outcome.winner, outcome=outcome, seat_turns=(3,))

        with patch("src.evaluation.arena.run_matches", finish):
            row = head_to_head(FirstLegal(), RandomPolicy(), self.game, 1, 5, 17)
        self.assertEqual((row.deals, row.matches, row.score, row.net_points), (5, 10, 0.6, 0.2))
        self.assertEqual((row.win_rate, row.opponent_win_rate, row.mean_win_turn), (0.4, 0.2, 3.0))
        self.assertAlmostEqual(row.losing_deadwood, 17 / 3)

    def test_three_seat_mirrors_count_every_cross_policy_pair_including_ties(self):
        def finish(deals, *args, **kwargs):
            queued = tuple(deals)
            self.assertEqual([(deal.seed, deal.seats) for deal in queued], [
                (17, (0, 1, 1)), (17, (1, 0, 0)), (17, (1, 0, 1)), (17, (0, 1, 0)),
                (17, (1, 1, 0)), (17, (0, 0, 1)),
            ])
            for deal in queued:
                winner = deal.seats.index(0)
                outcome = DealOutcome(winner, "win", tuple(
                    HandPenalty(0 if seat == winner else 9, (), ()) for seat in range(3)
                ), 210)
                yield SimpleNamespace(deal=deal, winner=winner, outcome=outcome, seat_turns=(1, 2, 3))

        with patch("src.evaluation.arena.run_matches", finish):
            row = head_to_head(FirstLegal(), RandomPolicy(), self.game, 3, 1, 17)
        self.assertEqual((row.matches, row.comparisons, row.seats_per_policy), (6, 12, 9))
        self.assertEqual((row.score, row.net_points), (0.875, 9.0))
        self.assertAlmostEqual(row.advantage, 0.3325)
        self.assertEqual((row.win_rate, row.opponent_win_rate, row.losing_deadwood), (2 / 3, 0.0, 9.0))
        self.assertEqual(row.mean_win_turn, 10 / 6)

    def test_swapping_policies_reverses_advantages_and_win_rates_from_solo_to_four_seats(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=3)
        for players, matches, comparisons in ((1, 2, 1), (2, 2, 2), (3, 6, 12), (4, 8, 24)):
            with self.subTest(players=players):
                row = head_to_head(FirstLegal(), RandomPolicy(), game, players, 12, 7, tables=5)
                swapped = head_to_head(RandomPolicy(), FirstLegal(), game, players, 12, 7, tables=5)
                self.assertEqual((row.deals, row.matches, row.comparisons), (12, 12 * matches, 12 * comparisons))
                self.assertAlmostEqual(swapped.score, 1 - row.score)
                self.assertEqual((swapped.net_points, swapped.advantage), (-row.net_points, -row.advantage))
                self.assertEqual((swapped.win_rate, swapped.opponent_win_rate), (row.opponent_win_rate, row.win_rate))

    def test_interruption_refuses_incomplete_pair_results(self):
        with self.assertRaisesRegex(InterruptedError, "paired deals"):
            head_to_head(FirstLegal(), RandomPolicy(), self.game, 4, 2, 7, stop=lambda: True)

    def test_league_rates_real_matching_checkpoints_and_rejects_non_checkpoints(self):
        game = GameConfig(max_turns=2)
        model = GameTransformer(ModelSpec(16, 1, 2, 32), SCHEMA)
        with TemporaryDirectory() as directory:
            paths = tuple(Path(directory) / f"{name}.pt" for name in ("anchor", "copy", "mismatch"))
            for path, config in zip(paths, (game, game, self.game)):
                save_checkpoint(path, model, config.to_dict(), {"version": 1})
            league = rate_checkpoints(paths[:2], deals=1, max_turns=1, anchor_elo=1200)
            with self.assertRaisesRegex(ValueError, "matching game configs"):
                rate_checkpoints(paths[1:], deals=1, max_turns=1)
            for token in ("random", "heuristic"):
                with self.assertRaisesRegex(ValueError, "only accepts GameTransformer"):
                    rate_checkpoints((paths[0], token), deals=1)
        self.assertEqual(league["matches"], 16)
        self.assertEqual(league["config"]["game"]["max_turns"], 1)
        self.assertEqual([row["seed"] for row in league["pairs"]], [972000, 973000, 974000])
        for i, row in enumerate(league["standings"]):
            self.assertEqual(row["label"], ("anchor", "copy")[i])
            self.assertEqual(row["elo_by_players"], {2: 1200, 3: 1200, 4: 1200})
            self.assertEqual((row["elo"], row["score"], row["seats"], row["net_points_per_seat"]), (1200, 0.5, 27, 0.0))

    def test_bradley_terry_recovers_known_strength_ratios_without_a_prior(self):
        ratings = solve_elo(((0, 10, 10), (20, 0, 10), (40, 20, 0)), anchor_index=1, anchor_elo=1400)
        self.assertEqual(ratings[1], 1400)
        for rating, strength in zip(ratings, (0.5, 1.0, 2.0)):
            self.assertAlmostEqual(rating, 1400 + 400 * math.log10(strength), places=8)
        expected = 1000 - 400 * math.log10(800)
        self.assertAlmostEqual(solve_elo(((0, 400), (0.5, 0)))[1], expected, places=8)

    def test_bradley_terry_fits_finite_ordered_ratings_without_a_loss_or_win(self):
        ratings = solve_elo(((0, 1024, 1024), (0, 0, 1024), (0, 0, 0)), anchor_index=1)
        self.assertTrue(all(map(math.isfinite, ratings)))
        self.assertGreater(ratings[0], ratings[1])
        self.assertGreater(ratings[1], ratings[2])
        self.assertEqual(ratings[1], 1000)
        self.assertAlmostEqual(ratings[0] + ratings[2], 2000, places=7)

    def test_single_matches_reject_a_removed_heuristic_policy(self):
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            resolve_policy("heuristic")
