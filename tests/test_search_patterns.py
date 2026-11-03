from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from src.evaluate import evaluate_hand
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation, Phase, discard_action,
)
from src.game.information import EventKind, InformationEvent, PlayerView
from src.game.multiplayer import MatchConfig, MultiplayerEnv

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.model.encoding import collate_observations
    from src.model.network import QNetwork, save_checkpoint
    from src.model.search import (
        BeliefState, SearchConfig, SearchPolicy, _DECOMP_CACHE, _PEN_CACHE,
        _after_draw, _cached_meld_val, _score_22_discards, can_declare_ub,
        outs_1ply, search, select_search_action, winning_discard,
    )


class TestLongSequenceDeclarations(unittest.TestCase):
    def test_two_exact_wildcards_can_fill_one_sequence_piece(self):
        counts = [0] * 52
        for face in (16, 7, 7, 19, 20, 21):
            counts[face] += 1
        result = evaluate_hand(counts, joker=7, required_sequences=2, cards_in_hand=6)
        self.assertTrue(result.is_valid)
        if torch is not None:
            self.assertTrue(can_declare_ub(counts, 7, 2))
        self.assertEqual(sum(meld.is_pure for meld in result.melds), 2)

    def test_long_runs_remain_complete_declarations_with_their_full_sequence_quota(self):
        for length in range(6, 14):
            patterns = [
                tuple(range(start, start + length))
                for start in range(14 - length)
            ]
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
                            counts = [0] * 52
                            for face in cards:
                                counts[face] += 1
                            result = evaluate_hand(
                                counts, joker,
                                required_sequences=length // 3,
                                cards_in_hand=length,
                            )
                            self.assertTrue(result.is_valid)
                            if torch is not None:
                                self.assertTrue(can_declare_ub(counts, joker, length // 3))
                            self.assertGreaterEqual(
                                sum(meld.is_pure for meld in result.melds),
                                length // 3,
                            )
                            self.assertEqual(
                                Counter(face for meld in result.melds for face in meld.cards),
                                Counter(cards),
                            )


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestBeliefSearch(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.network = QNetwork()
        _PEN_CACHE.clear()
        _DECOMP_CACHE.clear()

    @staticmethod
    def draw_observation():
        hand = tuple(int(face in (0, 1, 30)) for face in range(52))
        config = GameConfig(cards_in_hand=3, required_sequences=1)
        stock = 148
        players = (
            PlayerView(3, 40, 1, 1, True),
            PlayerView(3, 40, 0, 0, True),
        )
        event = InformationEvent(
            EventKind.INITIAL, 0, 3, 40, stock, 1, offered_top=2, initial_hand=hand,
        )
        return Observation(hand, (2,), 51, Phase.DRAW, 40, stock, config, players, (event,))

    @staticmethod
    def meld_discard_hand():
        faces = (
            0, 1, 2, 13, 14, 15, 26, 27, 28, 3, 4, 5,
            9, 22, 35, 10, 23, 36, 19, 20, 42, 50,
        )
        return tuple(faces.count(face) for face in range(52))

    def test_cached_meld_value_counts_complete_melds_and_residual_protos(self):
        hand = list(self.meld_discard_hand())
        hand[42] -= 1
        hand = tuple(hand)
        for required, expected in ((4, 5.22), (5, 5.32)):
            with self.subTest(required=required):
                self.assertAlmostEqual(_cached_meld_val(hand, 51, required, 21), expected)
        with mock.patch(
            "src.model.search.minimum_penalty", side_effect=AssertionError("already cached"),
        ):
            self.assertAlmostEqual(_cached_meld_val(hand, 51, 4, 21), 5.22)
            self.assertAlmostEqual(_cached_meld_val(hand, 51, 5, 21), 5.32)

    def test_score_22_discards_reuses_residuals_and_rechecks_top_meld_faces(self):
        hand = self.meld_discard_hand()
        faces = (13, 1, 0, 19, 20, 42, 50)
        q_values = [0.0] * 52
        expected = {0: 4.32, 1: 4.55, 13: 4.55, 19: 5.1, 20: 5.1, 42: 5.32, 50: 5.32}
        for preferred in (0, 13):
            with self.subTest(preferred=preferred):
                q_values[preferred] = 1.0
                scores = _score_22_discards(hand, faces, q_values, 51, 5, 21, top_non_u=1)
                self.assertEqual(set(scores), set(faces))
                for face, value in expected.items():
                    if preferred == 13:
                        value = 4.55 if face == 0 else 4.54 if face == 13 else value
                    self.assertAlmostEqual(scores[face], value)
                q_values[preferred] = 0.0
        self.assertEqual(sum(hand), 22)
        self.assertEqual(hand[0], 1)
        self.assertEqual(hand[13], 1)

    def test_meld_values_skip_hands_with_more_than_two_rank_wildcards(self):
        hand = list(self.meld_discard_hand())
        for before, after in ((42, 12), (50, 25), (19, 38)):
            hand[before] -= 1
            hand[after] += 1
        hand = tuple(hand)
        self.assertEqual(_cached_meld_val(hand, 51, 5, 21), 0.0)
        self.assertEqual(
            _score_22_discards(hand, (0, 12, 25, 38), [0.0] * 52, 51, 5, 21),
            {0: 0.0, 12: 0.0, 25: 0.0, 38: 0.0},
        )

    def test_unseen_supply_matches_stock_and_unknown_opponent_cards(self):
        for players in (2, 4, 6):
            env = MultiplayerEnv(MatchConfig(players=players))
            env.reset(41)
            with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
                for turn in range(players * 2):
                    obs = env.observe()
                    belief = BeliefState.from_observation(obs)
                    hidden = sum(
                        player.hand_size - sum(belief.known_held[seat])
                        for seat, player in enumerate(obs.players) if seat != 0
                    )
                    self.assertEqual(sum(belief.unseen_counts), obs.stock_remaining + hidden)
                    self.assertEqual(belief.stock_weights, belief.unseen_counts)
                    self.assertAlmostEqual(sum(p for _, p in belief.stock_probabilities()), 1.0)
                    env.step(ACTION_TAKE_DISCARD if turn % 2 else ACTION_DRAW_STOCK)
                    hand = env.observe().hand
                    env.step(discard_action(next(face for face, copies in enumerate(hand) if copies)))

    def test_known_cards_preferences_and_passes_follow_relative_seats(self):
        for known in (0, 11, 12):
            with self.subTest(known=known):
                obs = self.draw_observation()
                initial = replace(obs.history[0], offered_top=known)
                events = (
                    InformationEvent(EventKind.TAKE_DISCARD, 1, 4, 40, 148, 0, known, known),
                    InformationEvent(EventKind.DISCARD, 1, 3, 39, 148, 0, 16),
                    InformationEvent(EventKind.DRAW_STOCK, 1, 4, 39, 147, 0, offered_top=16),
                    InformationEvent(EventKind.DISCARD, 1, 3, 38, 147, 0, 28),
                    InformationEvent(EventKind.TAKE_DISCARD, 1, 4, 38, 147, 0, 28, 28),
                    InformationEvent(EventKind.DISCARD, 1, 3, 37, 147, 0, 28),
                )
                obs = replace(
                    obs, joker=8, discard_pile=(16, 28), history=(initial, *events),
                )
                belief = BeliefState.from_observation(obs)
                self.assertEqual(sum(belief.known_held[1]), 1)
                self.assertEqual(belief.known_held[1][known], 1)
                self.assertEqual(belief.take_rate[1], 3 / 8)
                self.assertEqual(belief.opp_safe[1][16], 2)
                self.assertEqual(belief.opp_safe[1][28], 2)
                self.assertEqual(belief.opp_wanted[1][known], 1.0)
                self.assertEqual(belief.opp_wanted[1][known + 13], 0.5)
                self.assertEqual(belief.opp_wanted[1][8], 1.0)
                self.assertEqual(belief.feed_risk(known), 3 / 8)
                self.assertEqual(belief.feed_risk(16), 0.0)
                self.assertEqual(belief.feed_risk(known, seat=2), 0.0)
                if known == 0:
                    self.assertEqual(belief.opp_wanted[1][12], 0.6)
                    self.assertEqual(belief.opp_wanted[1][11], 0.3)
                    self.assertEqual(belief.opp_wanted[1][1], 0.6)
                if known == 11:
                    self.assertEqual(belief.opp_wanted[1][0], 0.3)
                if known == 12:
                    self.assertEqual(belief.opp_wanted[1][0], 0.6)
                    self.assertEqual(belief.opp_wanted[1][1], 0.0)

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
                    expected = tuple(min(n, belief.unseen_counts[f]) for f, n in enumerate(pool))
                    self.assertEqual(belief.stock_weights, expected)
                    self.assertEqual(belief.stock_weights[face], 0)
                    break
                obs = env.observe()
                discarded = next(face for face, copies in enumerate(obs.hand) if copies)
                before = Counter((*obs.discard_pile, discarded))
                env.step(discard_action(discarded))
                if env.observe().history[-1].kind is EventKind.REFILL:
                    before[discarded] -= 1
                    pool = [before[face] for face in range(52)]
                    belief = BeliefState.from_observation(env.observe())
                    self.assertEqual(belief.stock_weights, tuple(pool))
        self.assertEqual(sum(pool), 7)

    def test_after_draw_matches_real_environment_transitions(self):
        for players in (2, 4, 6):
            for action, kind in (
                (ACTION_DRAW_STOCK, EventKind.DRAW_STOCK),
                (ACTION_TAKE_DISCARD, EventKind.TAKE_DISCARD),
            ):
                env = MultiplayerEnv(MatchConfig(players=players))
                env.reset(23)
                obs = env.observe()
                face = env._stock[-1] if action == ACTION_DRAW_STOCK else obs.discard_top
                expected = _after_draw(obs, face, kind)
                env.step(action)
                self.assertEqual(expected, env.observe())
                self.assertEqual(Observation.from_dict(expected.to_dict()), expected)

    def test_exact_wins_need_no_network_rows_in_either_phase(self):
        draw = self.draw_observation()
        discard = _after_draw(draw, 2, EventKind.TAKE_DISCARD)
        policy = SearchPolicy(self.network, draw.config)
        with mock.patch.object(self.network, "forward", side_effect=AssertionError("network not needed")):
            self.assertEqual(policy.act(draw, random.Random(9)), ACTION_TAKE_DISCARD)
            self.assertEqual(
                select_search_action(self.network, discard), discard_action(30),
            )
            self.assertEqual(search(self.network, discard).reason, "exact_win")
            self.assertEqual(search(self.network, draw).rows, 0)
        self.assertEqual(winning_discard(discard.hand, 51, 1, 3), 30)
        belief = BeliefState.from_observation(draw)
        self.assertAlmostEqual(outs_1ply(draw.hand, belief, draw.config), 4 / 151)

    def test_upper_bound_keeps_wild_sequences_and_rejects_king_ace_two(self):
        counts = [0] * 52
        for face in (16, 7, 7, 19, 20, 21):
            counts[face] += 1
        self.assertTrue(can_declare_ub(counts, joker=7, required_sequences=2))
        bad = [int(face in (0, 1, 12)) for face in range(52)]
        self.assertFalse(can_declare_ub(bad, joker=51, required_sequences=1))
        self.assertIsNone(winning_discard(bad, joker=51, required_sequences=1, cards_in_hand=3))

    def test_draw_lookahead_uses_expected_post_discard_values(self):
        class PreferTwo(QNetwork):
            def forward(self, batch):
                return (batch.cards[:, 2, :1] > 0).float().expand(-1, 54)

        result = search(
            PreferTwo(), self.draw_observation(),
            SearchConfig(exact_wins=False, feed_weight=0.0),
        )
        self.assertEqual(result.action, ACTION_TAKE_DISCARD)
        self.assertEqual(result.reason, "draw_lookahead")
        self.assertEqual(result.rows, 53)
        self.assertEqual(result.take_value, 1.0)
        self.assertAlmostEqual(result.stock_value, 2 / 151)

    def test_draw_lookahead_takes_discard_that_creates_outs(self):
        class ZeroQ(QNetwork):
            def forward(self, batch):
                return batch.cards.new_zeros((len(batch.cards), 54))

        obs = self.draw_observation()
        hand = tuple(int(face in (0, 30, 40)) for face in range(52))
        obs = replace(
            obs, hand=hand, discard_pile=(1,),
            history=(replace(obs.history[0], initial_hand=hand, offered_top=1),),
        )
        belief = BeliefState.from_observation(obs)
        self.assertEqual(outs_1ply(obs.hand, belief, obs.config), 0.0)
        child_hand = list(obs.hand)
        child_hand[obs.discard_top] += 1
        child_hand[40] -= 1
        self.assertGreater(outs_1ply(child_hand, belief, obs.config), 0.0)

        result = search(
            ZeroQ(), obs, SearchConfig(exact_wins=True, feed_weight=0.0),
        )
        self.assertEqual(result.action, ACTION_TAKE_DISCARD)
        self.assertEqual(result.reason, "draw_lookahead")
        self.assertGreater(result.take_value, result.stock_value)
        self.assertEqual(result.stock_value, 0.0)

    def test_draw_lookahead_matches_pile_and_stock_context(self):
        class PreferShortPile(QNetwork):
            def forward(self, batch):
                return -batch.cards[:, :, 1].sum(dim=1, keepdim=True).expand(-1, 54)

        class PreferMoreStock(QNetwork):
            def forward(self, batch):
                return (batch.context[:, 4:5] - 1.0).expand(-1, 54)

        for network, expected in (
            (PreferShortPile(), -1 / 3),
            (PreferMoreStock(), 147 / 156 - 1.0),
        ):
            with self.subTest(network=type(network).__name__):
                result = search(
                    network, self.draw_observation(),
                    SearchConfig(exact_wins=False, feed_weight=0.0),
                )
                self.assertEqual(result.reason, "draw_lookahead")
                self.assertEqual(result.rows, 53)
                self.assertAlmostEqual(result.take_value, expected)
                self.assertAlmostEqual(result.stock_value, expected)
                self.assertAlmostEqual(result.take_value, result.stock_value)

    def test_full_draw_blends_deduplicated_afterstates_without_diluting_meld_bonus(self):
        class DrawAfterstateQ(QNetwork):
            def forward(self, batch):
                values = batch.cards.new_full((len(batch.cards), 54), -100.0)
                for row, cards in enumerate(batch.cards):
                    hand = (cards[:, 0] * 3).round().to(torch.int64).tolist()
                    if sum(hand) == 22:
                        if hand[42]:
                            for face, value in ((0, 2.0), (2, 1.99), (4, 1.98)):
                                values[row, discard_action(face)] = value
                        else:
                            face = 48 if hand[48] else 50
                            values[row, discard_action(face)] = 0.5
                    else:
                        values[row, 0] = 0.5 * hand[0] + 0.5 * (1 - hand[42])
                        values[row, 1] = 100.0
                return values

        draw = self.draw_observation()
        hand = tuple(int(face in range(0, 42, 2)) for face in range(52))
        draw = replace(
            draw, hand=hand, discard_pile=(42,), config=GameConfig(), stock_remaining=112,
            players=tuple(replace(player, hand_size=21) for player in draw.players),
            history=(replace(
                draw.history[0], hand_size=21, stock_remaining=112,
                initial_hand=hand, offered_top=42,
            ),),
        )
        meld_values = dict.fromkeys(range(52), 1.0)
        meld_values[2] = 3.0
        cases = (
            (True, 2, 0.15, 1.9015, 0.88625, 7),
            (True, 2, 0.0, 1.7665, 0.85625, 7),
            (True, 1, 0.15, 2.0, 0.875, 4),
            (False, 2, 0.15, 2.0, 0.875, 4),
        )
        for exact, depth, outs, take, stock, rows in cases:
            seen = []

            def capture(observations, device):
                seen.append(tuple(observations))
                return collate_observations(observations, device)

            with (
                self.subTest(exact=exact, depth=depth, outs=outs),
                mock.patch.object(
                    BeliefState, "stock_probabilities",
                    return_value=((42, 0.25), (48, 0.25), (50, 0.5)),
                ),
                mock.patch("src.model.search._cached_meld_val", return_value=1.0),
                mock.patch("src.model.search._score_22_discards", return_value=meld_values),
                mock.patch(
                    "src.model.search.outs_1ply",
                    side_effect=lambda h, *_: 0.1 if h[42] else 0.2,
                ),
                mock.patch("src.model.search.collate_observations", side_effect=capture),
            ):
                result = search(
                    DrawAfterstateQ(), draw,
                    SearchConfig(
                        exact_wins=exact, rollout_depth=depth, rollout_beam=2,
                        feed_weight=0.0, outs_weight=outs,
                    ),
                )
                self.assertEqual(result.action, ACTION_TAKE_DISCARD)
                self.assertEqual(result.reason, "draw_lookahead")
                self.assertAlmostEqual(result.take_value, take, places=6)
                self.assertAlmostEqual(result.stock_value, stock, places=6)
                self.assertEqual(result.rows, rows)
                self.assertEqual(len(seen[0]), 4)
                if rows == 7:
                    expected_hands = {draw.hand}
                    for face in (0, 2):
                        post_hand = list(draw.hand)
                        post_hand[42] += 1
                        post_hand[face] -= 1
                        expected_hands.add(tuple(post_hand))
                    self.assertEqual(len(seen[1]), 3)
                    self.assertEqual({after.hand for after in seen[1]}, expected_hands)
                    for after in seen[1]:
                        self.assertEqual(after.discard_pile, (42,))
                        self.assertEqual(after.phase, Phase.DRAW)
                        self.assertEqual(after.players[0].hand_size, 21)
                        self.assertEqual(after.turns_remaining, 39)
                        self.assertEqual(after.stock_remaining, 111)
                        self.assertEqual(after.history[-2].kind, EventKind.DRAW_STOCK)
                        self.assertEqual(after.history[-1].kind, EventKind.DISCARD)

    def test_forced_greedy_and_discard_paths_return_legal_actions(self):
        obs = self.draw_observation()
        forced = replace(obs, stock_remaining=0)
        self.assertEqual(search(self.network, forced).action, ACTION_TAKE_DISCARD)
        self.assertEqual(search(self.network, forced).reason, "forced")
        greedy = search(self.network, obs, SearchConfig(exact_wins=False, draw_lookahead=False))
        self.assertIn(greedy.action, (ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD))
        self.assertEqual(greedy.reason, "greedy")
        self.assertEqual(greedy.rows, 1)
        discard = _after_draw(obs, 30, EventKind.DRAW_STOCK)
        result = search(self.network, discard, SearchConfig(exact_wins=False))
        self.assertIn(result.action, (discard_action(0), discard_action(1), discard_action(30)))
        self.assertEqual(result.reason, "discard_search")
        self.assertEqual(result.rows, 1)

    def test_discard_search_blends_q_values_with_post_discard_draw_values(self):
        class AfterstateQ(QNetwork):
            def __init__(self, q_margin):
                super().__init__()
                self.q_margin = q_margin

            def forward(self, batch):
                values = batch.cards.new_ones((len(batch.cards), 54))
                values[:, discard_action(0)] += self.q_margin
                values[:, 0] = (
                    4.0 - 6.0 * batch.cards[:, 42, 0] - 12.0 * batch.cards[:, 42, 1]
                )
                values[:, 1] = -values[:, 0]
                return values

        draw = self.draw_observation()
        hand = tuple(int(face in range(0, 42, 2)) for face in range(52))
        draw = replace(
            draw, hand=hand, config=GameConfig(), stock_remaining=112,
            players=tuple(replace(player, hand_size=21) for player in draw.players),
            history=(replace(
                draw.history[0], hand_size=21, stock_remaining=112, initial_hand=hand,
            ),),
        )
        discard = _after_draw(draw, 42, EventKind.DRAW_STOCK)
        for q_margin, expected_face in ((0.0, 42), (0.49, 42), (0.51, 0)):
            with self.subTest(q_margin=q_margin):
                result = search(
                    AfterstateQ(q_margin), discard,
                    SearchConfig(
                        exact_wins=True, rollout_depth=2, feed_weight=0.0, outs_weight=0.0,
                    ),
                )
                self.assertEqual(result.action, discard_action(expected_face))
                self.assertEqual(result.reason, "discard_search")
                self.assertEqual(result.rows, 23)

    def test_discard_search_adds_residual_meld_values_only_when_enabled(self):
        class ProtoBreakingQ(QNetwork):
            def forward(self, batch):
                values = batch.cards.new_zeros((len(batch.cards), 54))
                values[:, discard_action(19)] = 0.01
                return values

        draw = self.draw_observation()
        hand = list(self.meld_discard_hand())
        hand[50] -= 1
        hand = tuple(hand)
        draw = replace(
            draw, hand=hand, config=GameConfig(), stock_remaining=112,
            players=tuple(replace(player, hand_size=21) for player in draw.players),
            history=(replace(
                draw.history[0], hand_size=21, stock_remaining=112, initial_hand=hand,
            ),),
        )
        discard = _after_draw(draw, 50, EventKind.DRAW_STOCK)
        for exact, outs, expected, rows in (
            (True, 0.15, 42, 23), (True, 0.0, 19, 23), (False, 0.15, 19, 1),
        ):
            with (
                self.subTest(exact=exact, outs=outs),
                mock.patch("src.model.search.outs_1ply", return_value=0.0),
            ):
                result = search(
                    ProtoBreakingQ(), discard,
                    SearchConfig(exact_wins=exact, feed_weight=0.0, outs_weight=outs),
                )
                self.assertEqual(result.action, discard_action(expected))
                self.assertEqual(result.reason, "discard_search")
                self.assertEqual(result.rows, rows)

    def test_depth3_rollout_updates_post_discard_hand_size_and_history(self):
        obs = self.draw_observation()
        discard = _after_draw(obs, 30, EventKind.DRAW_STOCK)
        seen_batches = []
        real_collate = collate_observations

        def spy_collate(children, device=None):
            if len(children) > 1:
                for child in children:
                    self.assertEqual(child.players[0].hand_size, sum(child.hand))
                    self.assertEqual(child.players[0].turns_remaining, child.turns_remaining)
                    self.assertEqual(child.history[-2].kind, EventKind.DISCARD)
                    self.assertEqual(child.history[-1].kind, EventKind.DRAW_STOCK)
                seen_batches.append(len(children))
            return real_collate(children, device)

        with mock.patch("src.model.search.collate_observations", side_effect=spy_collate):
            result = search(
                self.network, discard,
                SearchConfig(exact_wins=False, rollout_depth=3, rollout_beam=4),
            )
        self.assertEqual(result.reason, "discard_search")
        self.assertEqual(seen_batches, [36])
        self.assertEqual(result.rows, 37)

    def test_search_policy_loads_real_checkpoints(self):
        obs = self.draw_observation()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.pt"
            save_checkpoint(str(path), self.network, obs.config)
            policy = SearchPolicy.from_checkpoint(str(path))
            self.assertEqual(policy.game_config, obs.config)
            self.assertEqual(policy.act(obs, random.Random(8)), ACTION_TAKE_DISCARD)


if __name__ == "__main__":
    unittest.main()
