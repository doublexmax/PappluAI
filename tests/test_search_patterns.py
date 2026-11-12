from __future__ import annotations

from collections import Counter
from dataclasses import replace
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from src.evaluate import evaluate_hand, pure_sequence_upper_bound
from src.game.belief import BeliefState
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation, Phase, discard_action,
)
from src.game.information import EventKind, InformationEvent, PlayerView
from src.game.lookahead import (
    Continues, Declares, Terminates, after_discard, after_draw, can_declare_ub, expand, outs_1ply,
    winning_discard,
)
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.game.outcome import solo_outcome
from src.model.encoding import CONTEXT_CHANNELS, LAYOUT, encode, feature_values

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.search import TeacherConfig, state_values, teacher_targets
    from src.training.core import Decision, Trajectory, TrajectoryReplay


def draw_observation():
    hand = tuple(int(face in (0, 1, 30)) for face in range(52))
    config = GameConfig(cards_in_hand=3, required_sequences=1)
    players = (PlayerView(3, 40, 1, 1, True), PlayerView(3, 40, 0, 0, True))
    event = InformationEvent(EventKind.INITIAL, 0, 3, 40, 148, 1, offered_top=2, initial_hand=hand)
    return Observation(hand, (2,), 51, Phase.DRAW, 40, 148, config, players, (event,))


class TestLongSequenceDeclarations(unittest.TestCase):
    def test_exact_joker_is_not_also_counted_as_a_natural_sequence(self):
        counts = tuple(int(face in (0, 1, 2, 17, 30, 43)) for face in range(52))
        self.assertEqual(pure_sequence_upper_bound(counts, 2), 1)
        obs = replace(
            draw_observation(), hand=counts, joker=2,
            config=GameConfig(cards_in_hand=6, required_sequences=2),
        )
        values = feature_values(encode(obs))
        context_offset = next(field.offset for field in LAYOUT if field.name == "context")
        self.assertEqual(values[context_offset + CONTEXT_CHANNELS.index("pure_quota_deficit")], 0.5)

    def test_two_exact_wildcards_can_fill_one_sequence_piece(self):
        counts = [0] * 52
        for face in (16, 7, 7, 19, 20, 21):
            counts[face] += 1
        result = evaluate_hand(counts, joker=7, required_sequences=2, cards_in_hand=6)
        self.assertTrue(result.is_valid)
        self.assertTrue(can_declare_ub(counts, 7, 2))
        self.assertEqual(sum(meld.is_pure for meld in result.melds), 2)

    def test_long_runs_remain_complete_declarations_with_their_full_sequence_quota(self):
        for length in range(6, 14):
            patterns = [tuple(range(start, start + length)) for start in range(14 - length)]
            patterns.append(tuple(range(14 - length, 13)) + (0,))
            for suit in range(4):
                joker = ((suit + 1) % 4) * 13 + 9
                for pattern in patterns:
                    for substitute in (False, True):
                        with self.subTest(length=length, suit=suit, pattern=pattern, substitute=substitute):
                            cards = [suit * 13 + rank for rank in pattern]
                            if substitute:
                                cards[0] = joker
                                cards[-1] = joker
                            counts = [cards.count(face) for face in range(52)]
                            result = evaluate_hand(counts, joker, length // 3, length)
                            self.assertTrue(result.is_valid)
                            self.assertTrue(can_declare_ub(counts, joker, length // 3))
                            self.assertGreaterEqual(sum(meld.is_pure for meld in result.melds), length // 3)
                            self.assertEqual(Counter(face for meld in result.melds for face in meld.cards), Counter(cards))

    def test_upper_bound_excludes_king_ace_two_but_allows_queen_king_ace(self):
        bad = [int(face in (0, 1, 12)) for face in range(52)]
        good = [int(face in (11, 12, 0)) for face in range(52)]
        self.assertFalse(can_declare_ub(bad, joker=7, required_sequences=1))
        self.assertTrue(can_declare_ub(good, joker=7, required_sequences=1))
        self.assertIsNone(winning_discard(bad, joker=7, required_sequences=1, cards_in_hand=3))


class TestBeliefLookahead(unittest.TestCase):
    def test_unseen_supply_matches_stock_and_unknown_opponent_cards(self):
        for players in (2, 4, 6):
            env = MultiplayerEnv(MatchConfig(players=players))
            env.reset(41)
            with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
                for turn in range(players * 2):
                    obs = env.observe()
                    belief = BeliefState.from_observation(obs)
                    hidden = sum(
                        player.hand_size - sum(belief.ledger.seats[seat].held)
                        for seat, player in enumerate(obs.players) if seat != 0
                    )
                    self.assertEqual(sum(belief.unseen), obs.stock_remaining + hidden)
                    self.assertEqual(belief.stock_weights, belief.unseen)
                    self.assertAlmostEqual(sum(p for _, p in belief.stock_probabilities()), 1.0)
                    env.step(ACTION_TAKE_DISCARD if turn % 2 else ACTION_DRAW_STOCK)
                    obs = env.observe()
                    env.step(discard_action(next(face for face, copies in enumerate(obs.hand) if copies)))

    def test_known_cards_preferences_and_passes_follow_relative_seats(self):
        for known in (0, 11, 12):
            obs = draw_observation()
            initial = replace(obs.history[0], offered_top=known)
            events = (
                InformationEvent(EventKind.TAKE_DISCARD, 1, 4, 40, 148, 0, known, known),
                InformationEvent(EventKind.DISCARD, 1, 3, 39, 148, 0, 16),
                InformationEvent(EventKind.DRAW_STOCK, 1, 4, 39, 147, 0, offered_top=16),
                InformationEvent(EventKind.DISCARD, 1, 3, 38, 147, 0, 28),
                InformationEvent(EventKind.TAKE_DISCARD, 1, 4, 38, 147, 0, 28, 28),
                InformationEvent(EventKind.DISCARD, 1, 3, 37, 147, 0, 28),
            )
            obs = replace(obs, joker=8, discard_pile=(16, 28), history=(initial, *events))
            belief = BeliefState.from_observation(obs)
            seat = belief.ledger.seats[1]
            self.assertEqual(sum(seat.held), 1)
            self.assertEqual(seat.held[known], 1)
            self.assertEqual(belief.take_rate(1), 3 / 8)
            self.assertEqual((seat.discarded[16], seat.passed[16]), (1, 1))
            self.assertEqual((seat.last_take[known], seat.last_discard[28], seat.last_pass[16]), (2, 7, 4))
            self.assertEqual(belief.feed_risk(known), 3 / 8)
            self.assertEqual(belief.feed_risk(known + 13), 3 / 16)
            self.assertEqual(belief.feed_risk(8), 3 / 8)
            self.assertEqual(belief.feed_risk(16), 0.0)
            self.assertEqual(belief.feed_risk(known, seat=2), 0.0)
            if known == 0:
                self.assertAlmostEqual(belief.feed_risk(12), 3 / 8 * 0.6)
                self.assertAlmostEqual(belief.feed_risk(11), 3 / 8 * 0.3)
                self.assertAlmostEqual(belief.feed_risk(1), 3 / 8 * 0.6)
            if known == 12:
                self.assertEqual(belief.feed_risk(1), 0.0)

    def test_own_public_takes_are_recorded_without_double_subtracting_supply(self):
        obs = draw_observation()
        drawn = after_draw(obs, 2, EventKind.TAKE_DISCARD)
        belief = BeliefState.from_observation(drawn)
        self.assertEqual(belief.ledger.seats[0].took[2], 1)
        self.assertEqual(belief.ledger.seats[0].held[2], 1)
        self.assertEqual(belief.unseen[2], 2)
        self.assertEqual(belief.take_rate(0), 2 / 6)
        self.assertEqual(belief.ledger.pile, ())

    def test_refill_limits_stock_to_the_recycled_discard_pool(self):
        env = MultiplayerEnv(MatchConfig(GameConfig(num_decks=1, max_turns=12), 2))
        env.reset(17)
        pool = None
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(9):
                obs = env.observe()
                face = env._stock[-1]
                env.step(ACTION_DRAW_STOCK)
                if pool is not None:
                    pool[face] -= 1
                    belief = BeliefState.from_observation(env.observe())
                    self.assertEqual(belief.stock_weights, tuple(min(n, belief.unseen[f]) for f, n in enumerate(pool)))
                    self.assertEqual(belief.stock_weights[face], 0)
                    break
                obs = env.observe()
                discarded = next(face for face, copies in enumerate(obs.hand) if copies)
                before = Counter((*obs.discard_pile, discarded))
                env.step(discard_action(discarded))
                if env.observe().history[-1].kind is EventKind.REFILL:
                    before[discarded] -= 1
                    pool = [before[face] for face in range(52)]
                    self.assertEqual(BeliefState.from_observation(env.observe()).stock_weights, tuple(pool))
        self.assertEqual(sum(pool), 7)

    def test_after_draw_matches_real_environment_transitions(self):
        for players in (2, 4, 6):
            for action, kind in ((0, EventKind.DRAW_STOCK), (1, EventKind.TAKE_DISCARD)):
                env = MultiplayerEnv(MatchConfig(players=players))
                env.reset(23)
                obs = env.observe()
                face = env._stock[-1] if action == 0 else obs.discard_top
                expected = after_draw(obs, face, kind)
                env.step(action)
                self.assertEqual(expected, env.observe())
                self.assertEqual(Observation.from_dict(expected.to_dict()), expected)

    def test_expand_keeps_complete_chance_nodes_and_exact_declarations(self):
        obs = draw_observation()
        branches = expand(obs)
        self.assertEqual(set(branches), {0, 1})
        self.assertAlmostEqual(sum(branch.probability for branch in branches[0]), 1.0)
        self.assertEqual(len(branches[0]), 52)
        discard = branches[1][0].observation
        self.assertEqual(winning_discard(discard.hand, 51, 1, 3), 30)
        self.assertEqual(expand(discard)[discard_action(30)], (Declares(1.0),))
        self.assertIsInstance(expand(discard)[discard_action(0)][0], Continues)
        self.assertAlmostEqual(outs_1ply(obs.hand, BeliefState.from_observation(obs), obs.config), 4 / 151)
        after = after_discard(discard, 0)
        self.assertEqual(after.players[0].hand_size, sum(after.hand))
        self.assertEqual(expand(after), {})
        self.assertEqual(expand(replace(obs, phase=Phase.TERMINAL)), {})

    def test_opponent_discard_views_do_not_expand_as_observer_decisions(self):
        discard = after_draw(draw_observation(), 2, EventKind.TAKE_DISCARD)
        other = replace(discard, history=(*discard.history[:-1], replace(discard.history[-1], actor=1)))
        self.assertEqual(expand(other), {})
        self.assertEqual(expand(discard)[discard_action(30)], (Declares(1.0),))

    def test_terminal_solo_discards_score_deadwood_and_multiplayer_exhaustion_is_skipped(self):
        discard = after_draw(draw_observation(), 2, EventKind.TAKE_DISCARD)
        solo = replace(discard, players=(replace(discard.players[0], next_player=0, previous_player=0),))
        for obs in (
            replace(solo, warm_start=True),
            replace(solo, turns_remaining=1, players=(replace(solo.players[0], turns_remaining=1),)),
        ):
            branches = expand(obs)
            self.assertEqual(branches, {
                discard_action(0): (Terminates(1.0, -0.6),),
                discard_action(1): (Terminates(1.0, -0.6551724137931034),),
                discard_action(2): (Terminates(1.0, -0.6491228070175439),),
                discard_action(30): (Declares(1.0),),
            })
        exhausted = replace(
            discard, turns_remaining=1,
            players=(replace(discard.players[0], turns_remaining=1),
                     replace(discard.players[1], turns_remaining=0, active=False)),
        )
        self.assertEqual(expand(exhausted), {})


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestBudgetedTeacher(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        class Values(torch.nn.Module):
            def forward(self, batch):
                return torch.full((len(batch.card), 54), 2.0, device=batch.card.device)

            def outputs(self, batch):
                return SimpleNamespace(after=torch.full((len(batch.card),), 0.4, device=batch.card.device))

        self.target = Values()

    def test_exact_wins_and_expected_stock_values_receive_the_draw_discount(self):
        result = teacher_targets(self.target, [draw_observation()], TeacherConfig(fraction=1),
                                 random.Random(1), 0.5, self.device)
        self.assertEqual(result.rows.tolist(), [0])
        self.assertEqual(result.mask[0].nonzero().flatten().tolist(), [0, 1])
        self.assertAlmostEqual(result.values[0, 1].item(), 0.5)
        self.assertAlmostEqual(result.values[0, 0].item(), 1 - 2 / 151, places=6)

    def test_child_budget_never_truncates_or_renormalizes_a_chance_node(self):
        result = teacher_targets(self.target, [draw_observation()], TeacherConfig(fraction=1, max_children=1),
                                 random.Random(1), 0.5, self.device)
        self.assertEqual(result.rows.tolist(), [])
        self.assertEqual(result.values.shape, (0, 54))
        complete = teacher_targets(
            self.target, [draw_observation()], TeacherConfig(fraction=1, max_children=53),
            random.Random(1), 0.5, self.device,
        )
        self.assertEqual(complete.mask[0].nonzero().flatten().tolist(), [0, 1])
        self.assertAlmostEqual(complete.values[0, 0].item(), 1 - 2 / 151, places=6)
        disabled = teacher_targets(self.target, [draw_observation()], TeacherConfig(fraction=0),
                                   random.Random(1), 1.0, self.device)
        self.assertEqual(disabled.values.shape, (0, 54))

    def test_exhausted_budget_stops_expansion_and_estimates_skip_oversized_rows(self):
        discard = after_draw(draw_observation(), 2, EventKind.TAKE_DISCARD)
        rng = mock.Mock()
        rng.shuffle.side_effect = lambda rows: None
        for budget, expected_rows in ((3, [0]), (8, [0, 2])):
            with mock.patch("src.model.search.expand", wraps=expand) as expanded:
                result = teacher_targets(
                    self.target, [discard, draw_observation(), discard],
                    TeacherConfig(fraction=1, max_children=budget), rng, 0.5, self.device,
                )
            self.assertEqual(result.rows.tolist(), expected_rows)
            self.assertEqual([call.args[0] for call in expanded.call_args_list], [discard] * len(expected_rows))
            for position in range(len(expected_rows)):
                self.assertEqual(result.mask[position].nonzero().flatten().tolist(), [2, 3, 4, 32])
                self.assertAlmostEqual(result.values[position, 2].item(), 0.4)
                self.assertEqual(result.values[position, 32].item(), 1.0)

    def test_terminal_discards_need_no_child_forward_even_when_only_one_child_is_budgeted(self):
        discard = after_draw(draw_observation(), 2, EventKind.TAKE_DISCARD)
        solo = replace(discard, warm_start=True, players=(replace(
            discard.players[0], next_player=0, previous_player=0,
        ),))
        with mock.patch.object(self.target, "forward", side_effect=AssertionError("terminal Q forward")), mock.patch.object(
            self.target, "outputs", side_effect=AssertionError("terminal afterstate forward"),
        ):
            result = teacher_targets(
                self.target, [solo], TeacherConfig(fraction=1, max_children=1),
                random.Random(1), 0.5, self.device,
            )
        self.assertEqual(result.mask[0].nonzero().flatten().tolist(), [2, 3, 4, 32])
        for action, expected, penalty in (
            (2, -0.6, 10), (3, -0.6551724137931034, 18),
            (4, -0.6491228070175439, 17), (32, 1.0, 0),
        ):
            remaining = list(solo.hand)
            remaining[action - 2] -= 1
            outcome = solo_outcome(remaining, solo.joker, action == 32, solo.config.required_sequences)
            replay = TrajectoryReplay(8)
            replay.append(Trajectory(1, (Decision(
                solo, encode(solo), action, (solo.hand,) + ((0,) * 52,) * 5,
            ),), outcome))
            sample = replay.sample(random.Random(1), 1)[0]
            self.assertEqual((sample.outcome.utility, sample.outcome.penalty), (expected, penalty))
            self.assertAlmostEqual(result.values[0, action].item(), expected, places=6)
            self.assertAlmostEqual(result.values[0, action].item(), sample.outcome.utility, places=6)

    def test_discard_declarations_are_exact_and_afterstates_are_not_discounted(self):
        discard = after_draw(draw_observation(), 2, EventKind.TAKE_DISCARD)
        result = teacher_targets(self.target, [discard], TeacherConfig(fraction=1),
                                 random.Random(1), 0.5, self.device)
        self.assertEqual(result.values[0, discard_action(30)].item(), 1.0)
        self.assertAlmostEqual(result.values[0, discard_action(0)].item(), 0.4)
        after = after_discard(discard, 0)
        values = state_values(self.target, [discard, after], self.device)
        self.assertAlmostEqual(values[0].item(), 1.0)
        self.assertAlmostEqual(values[1].item(), 0.4)


if __name__ == "__main__":
    unittest.main()
