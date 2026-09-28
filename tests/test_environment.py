from __future__ import annotations

import copy
from collections import Counter
import operator
import random
import unittest
from unittest import mock

from src.evaluate import hand_reward, is_valid_hand
from src.environment import (
    ACTION_DRAW_STOCK,
    ACTION_TAKE_DISCARD,
    ENCODING_VERSION,
    NUM_ACTIONS,
    STATE_DIM,
    GameConfig,
    Observation,
    PappluEnv,
    Phase,
    build_warm_start_deal,
    discard_action,
    encode_observation,
    legal_action_mask,
)


class TestGameConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GameConfig()
        self.assertEqual(cfg.num_decks, 3)
        self.assertEqual(cfg.cards_in_hand, 21)
        self.assertEqual(cfg.required_sequences, 5)

    def test_rejects_impossible_supply(self):
        with self.assertRaises(ValueError):
            GameConfig(num_decks=1, cards_in_hand=60)

    def test_rejects_sequences_vs_hand(self):
        with self.assertRaises(ValueError):
            GameConfig(cards_in_hand=3, required_sequences=2)

    def test_checkpoint_config_preserves_validation(self):
        for value in (True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                GameConfig.from_dict({**GameConfig().to_dict(), "num_decks": value})


class TestEnvironmentCore(unittest.TestCase):
    def test_seeded_reset_reproducible(self):
        env = PappluEnv(GameConfig(max_turns=5))
        a = env.reset(seed=7)
        b_env = PappluEnv(GameConfig(max_turns=5))
        b = b_env.reset(seed=7)
        self.assertEqual(a.hand, b.hand)
        self.assertEqual(a.joker, b.joker)
        self.assertEqual(a.discard_pile, b.discard_pile)
        self.assertEqual(a.stock_remaining, b.stock_remaining)
        self.assertEqual(a.phase, Phase.DRAW)

    def test_card_conservation_with_indicator_removed(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=6, required_sequences=1, max_turns=10)
        env = PappluEnv(cfg)
        obs = env.reset(seed=1)
        total = cfg.num_decks * 52
        self.assertEqual(env.total_cards_in_play(), total - 1)
        self.assertEqual(sum(obs.hand), cfg.cards_in_hand)
        self.assertEqual(len(obs.discard_pile), 1)
        self.assertEqual(
            sum(obs.hand) + obs.stock_remaining + len(obs.discard_pile),
            total - 1,
        )

    def test_immutable_observation(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5))
        obs = env.reset(seed=2)
        with self.assertRaises(TypeError):
            operator.setitem(obs.hand, 0, 99)
        before = obs.hand
        env.step(ACTION_DRAW_STOCK)
        self.assertEqual(before, obs.hand)

    def test_encoding_shape_and_joker_one_hot(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5))
        obs = env.reset(seed=3)
        vec = encode_observation(obs)
        self.assertEqual(len(vec), STATE_DIM)
        self.assertEqual(ENCODING_VERSION, 1)
        joker_slice = vec[104:156]
        self.assertEqual(sum(joker_slice), 1.0)
        self.assertEqual(joker_slice[obs.joker], 1.0)
        self.assertEqual(vec[156], 1.0)
        self.assertEqual(vec[157], 0.0)

    def test_draw_and_discard_legality(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=8))
        obs = env.reset(seed=4)
        mask = legal_action_mask(obs)
        self.assertTrue(mask[ACTION_DRAW_STOCK])
        self.assertTrue(mask[ACTION_TAKE_DISCARD])
        self.assertFalse(any(mask[2:]))

        snap = copy.deepcopy(obs)
        with self.assertRaises(ValueError):
            env.step(discard_action(0))
        self.assertEqual(env.observe().hand, snap.hand)
        self.assertEqual(env.observe().phase, Phase.DRAW)

        obs = env.step(ACTION_DRAW_STOCK)
        self.assertEqual(obs.phase, Phase.DISCARD)
        self.assertEqual(sum(obs.hand), 7)
        mask = legal_action_mask(obs)
        self.assertFalse(mask[ACTION_DRAW_STOCK])
        self.assertFalse(mask[ACTION_TAKE_DISCARD])
        legal_discards = [i for i, ok in enumerate(mask) if ok]
        self.assertTrue(legal_discards)
        for face, count in enumerate(obs.hand):
            self.assertEqual(mask[discard_action(face)], count > 0)

    def test_illegal_action_no_mutation(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5))
        env.reset(seed=5)
        before = env.observe()
        with self.assertRaises(ValueError):
            env.step(99)
        after = env.observe()
        self.assertEqual(before, after)

    def test_allow_discard_just_drawn(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5))
        obs = env.reset(seed=6)
        stock_before = obs.stock_remaining
        before_hand = obs.hand
        obs = env.step(ACTION_DRAW_STOCK)
        self.assertEqual(obs.stock_remaining, stock_before - 1)
        drawn = None
        for face in range(52):
            if obs.hand[face] == before_hand[face] + 1:
                drawn = face
                break
        self.assertIsNotNone(drawn)
        obs2 = env.step(discard_action(drawn))
        self.assertEqual(sum(obs2.hand), 6)
        self.assertEqual(obs2.discard_top, drawn)

    def test_take_discard_path(self):
        env = PappluEnv(GameConfig(cards_in_hand=6, required_sequences=1, max_turns=5))
        obs = env.reset(seed=8)
        top = obs.discard_top
        obs = env.step(ACTION_TAKE_DISCARD)
        self.assertEqual(obs.phase, Phase.DISCARD)
        self.assertGreaterEqual(obs.hand[top], 1)
        self.assertEqual(len(obs.discard_pile), 0)

    def test_reward_after_discard_uses_evaluator(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=5)
        env = PappluEnv(cfg)
        env.reset(seed=0)
        env._phase = Phase.DISCARD
        env._hand = [0] * 52
        env._hand[0] = 1
        env._hand[1] = 1
        env._hand[2] = 1
        env._hand[13 + 4] = 1
        env._joker = 20
        env._discard = []
        env._stock = [30, 31]
        env._turns_remaining = 5
        obs = env.step(discard_action(13 + 4))
        self.assertEqual(obs.last_reward, 1.0)
        self.assertTrue(obs.won)
        self.assertEqual(obs.phase, Phase.TERMINAL)
        self.assertTrue(
            is_valid_hand(obs.hand, obs.joker, required_sequences=1, cards_in_hand=3)
        )

    def test_invalid_discard_reward_zero(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=5)
        env = PappluEnv(cfg)
        env.reset(seed=0)
        env._phase = Phase.DISCARD
        env._hand = [0] * 52
        env._hand[0] = 1
        env._hand[1] = 1
        env._hand[10] = 1
        env._hand[20] = 1
        env._joker = 40
        env._discard = []
        env._stock = [30]
        env._turns_remaining = 5
        obs = env.step(discard_action(20))
        self.assertEqual(obs.last_reward, 0.0)
        self.assertFalse(obs.won)
        self.assertEqual(obs.phase, Phase.DRAW)

    def test_max_turns_terminates(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=1)
        env = PappluEnv(cfg)
        env.reset(seed=9)
        env.step(ACTION_DRAW_STOCK)
        obs = env.observe()
        face = next(i for i, c in enumerate(obs.hand) if c > 0)
        obs = env.step(discard_action(face))
        if not obs.won:
            self.assertEqual(obs.phase, Phase.TERMINAL)

    def test_stock_exhaustion_discard_still_playable(self):
        cfg = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=5)
        env = PappluEnv(cfg)
        env.reset(seed=10)
        env._stock = []
        env._discard = [7]
        env._phase = Phase.DRAW
        env._turns_remaining = 5
        env._hand = [0] * 52
        env._hand[0] = 1
        env._hand[1] = 1
        env._hand[2] = 1
        mask = legal_action_mask(env.observe())
        self.assertFalse(mask[ACTION_DRAW_STOCK])
        self.assertTrue(mask[ACTION_TAKE_DISCARD])
        obs = env.step(ACTION_TAKE_DISCARD)
        self.assertEqual(obs.phase, Phase.DISCARD)
        self.assertEqual(sum(obs.hand), 4)

    def test_no_stock_reshuffle(self):
        cfg = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=20)
        env = PappluEnv(cfg)
        env.reset(seed=11)
        env._stock = []
        env._discard = [5]
        env._phase = Phase.DRAW
        env._hand = [0] * 52
        env._hand[0] = env._hand[1] = env._hand[2] = 1
        env.step(ACTION_TAKE_DISCARD)
        env.step(discard_action(5))
        self.assertEqual(env.observe().stock_remaining, 0)
        self.assertGreaterEqual(len(env.observe().discard_pile), 1)

    def test_each_face_conserved_across_complete_episode(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=6, required_sequences=1, max_turns=10)
        env = PappluEnv(cfg)
        obs = env.reset(seed=0)
        rng = random.Random(0)
        while not obs.done:
            counts = Counter(env._stock) + Counter(obs.discard_pile)
            counts.update({face: count for face, count in enumerate(obs.hand)})
            counts[obs.joker] += 1
            self.assertEqual(counts, Counter({face: cfg.num_decks for face in range(52)}))
            actions = [i for i, legal in enumerate(legal_action_mask(obs)) if legal]
            obs = env.step(rng.choice(actions))

    def test_evaluator_error_preserves_pending_discard(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        env.reset(seed=0)
        obs = env.step(ACTION_DRAW_STOCK)
        face = next(face for face, count in enumerate(obs.hand) if count)
        with mock.patch("src.environment.hand_reward", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                env.step(discard_action(face))
        self.assertEqual(obs, env.observe())


class TestWarmStart(unittest.TestCase):
    def test_warm_start_3_1(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=5)
        env = PappluEnv(cfg)
        obs = env.reset_warm_start(seed=42)
        self.assertTrue(obs.warm_start)
        self.assertEqual(obs.phase, Phase.DISCARD)
        self.assertEqual(sum(obs.hand), 4)
        found = False
        for face, count in enumerate(obs.hand):
            if count <= 0:
                continue
            trial = list(obs.hand)
            trial[face] -= 1
            if hand_reward(trial, obs.joker, required_sequences=1, cards_in_hand=3) == 1.0:
                found = True
                break
        self.assertTrue(found)
        vec = encode_observation(obs)
        self.assertEqual(len(vec), STATE_DIM)

    def test_warm_start_6_1(self):
        cfg = GameConfig(num_decks=2, cards_in_hand=6, required_sequences=1, max_turns=5)
        rng = random.Random(0)
        hand, joker, stock = build_warm_start_deal(cfg, rng)
        self.assertEqual(sum(hand), 7)
        wins = 0
        for face, count in enumerate(hand):
            if count <= 0:
                continue
            trial = list(hand)
            trial[face] -= 1
            if is_valid_hand(trial, joker, required_sequences=1, cards_in_hand=6):
                wins += 1
        self.assertGreaterEqual(wins, 1)

    def test_warm_start_default_size(self):
        cfg = GameConfig(num_decks=3, cards_in_hand=21, required_sequences=5, max_turns=10)
        rng = random.Random(1)
        hand, joker, stock = build_warm_start_deal(cfg, rng)
        self.assertEqual(sum(hand), 22)
        self.assertEqual(
            sum(hand) + len(stock) + 1,
            cfg.num_decks * 52,
        )

    def test_warm_starts_cover_nonmultiple_hand_sizes(self):
        for size in (3, 4, 5, 6, 7, 8):
            for quota in (0, size // 3):
                with self.subTest(size=size, quota=quota):
                    cfg = GameConfig(num_decks=1, cards_in_hand=size, required_sequences=quota)
                    hand, joker, stock = build_warm_start_deal(cfg, random.Random(42))
                    counts = Counter(stock)
                    counts.update({face: count for face, count in enumerate(hand)})
                    counts[joker] += 1
                    self.assertEqual(counts, Counter({face: 1 for face in range(52)}))
                    self.assertTrue(any(
                        hand_reward(
                            [count - (index == face) for index, count in enumerate(hand)],
                            joker, cards_in_hand=size, required_sequences=quota,
                        )
                        for face, count in enumerate(hand) if count
                    ))

    def test_warm_puzzle_ends_after_losing_discard(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        obs = env.reset_warm_start(seed=42)
        for face, count in enumerate(obs.hand):
            if count:
                trial = list(obs.hand)
                trial[face] -= 1
                if hand_reward(trial, obs.joker, cards_in_hand=3, required_sequences=1) == 0:
                    result = env.step(discard_action(face))
                    self.assertTrue(result.done)
                    self.assertFalse(result.won)
                    self.assertEqual(result.last_reward, 0.0)
                    return
        self.fail("fixture must have a losing discard")


class TestCurriculumStart(unittest.TestCase):
    def test_full_hand_draw_start_is_invalid_and_conserves_every_face(self):
        cfg = GameConfig(max_turns=6)
        env = PappluEnv(cfg)
        obs = env.reset_curriculum(distance=4, seed=9001)

        self.assertEqual(sum(obs.hand), 21)
        self.assertEqual(obs.phase, Phase.DRAW)
        self.assertFalse(obs.warm_start)
        self.assertFalse(obs.won)
        self.assertFalse(
            is_valid_hand(
                obs.hand,
                obs.joker,
                required_sequences=cfg.required_sequences,
                cards_in_hand=cfg.cards_in_hand,
            )
        )
        self.assertFalse(hasattr(obs, "target"))

        counts = Counter(env._stock) + Counter(obs.discard_pile)
        counts.update({face: count for face, count in enumerate(obs.hand)})
        counts[obs.joker] += 1
        self.assertEqual(
            counts,
            Counter({face: cfg.num_decks for face in range(52)}),
        )
        self.assertEqual(
            sum(obs.hand) + obs.stock_remaining + len(obs.discard_pile),
            cfg.num_decks * 52 - 1,
        )

    def test_distance_counts_actual_target_card_changes(self):
        import src.environment as environment

        cfg = GameConfig(max_turns=5)
        captured = []
        real_builder = environment._build_natural_winning_deal

        def capture(config, rng):
            deal = real_builder(config, rng)
            captured.append(tuple(deal[0]))
            return deal

        env = PappluEnv(cfg)
        with mock.patch(
            "src.environment._build_natural_winning_deal",
            side_effect=capture,
        ):
            obs = env.reset_curriculum(distance=8, seed=72)

        target = captured[-1]
        removed = sum(max(0, before - after) for before, after in zip(target, obs.hand))
        added = sum(max(0, after - before) for before, after in zip(target, obs.hand))
        self.assertEqual(removed, 8)
        self.assertEqual(added, 8)

    def test_seeded_curriculum_reset_is_repeatable(self):
        cfg = GameConfig(max_turns=5)
        first = PappluEnv(cfg)
        second = PappluEnv(cfg)
        a = first.reset_curriculum(distance=2, seed=123456)
        b = second.reset_curriculum(distance=2, seed=123456)
        self.assertEqual(a, b)
        self.assertEqual(first._stock, second._stock)

    def test_curriculum_uses_normal_multiturn_termination(self):
        cfg = GameConfig(max_turns=3)
        env = PappluEnv(cfg)
        obs = env.reset_curriculum(distance=1, seed=81)
        self.assertFalse(obs.warm_start)

        turns = 0
        while not obs.done:
            hand_before = obs.hand
            obs = env.step(ACTION_DRAW_STOCK)
            drawn = next(
                face
                for face, (before, after) in enumerate(zip(hand_before, obs.hand))
                if after == before + 1
            )
            obs = env.step(discard_action(drawn))
            turns += 1
        self.assertEqual(turns, cfg.max_turns)
        self.assertFalse(obs.won)
        self.assertEqual(obs.phase, Phase.TERMINAL)

    def test_invalid_or_failed_curriculum_reset_does_not_mutate(self):
        env = PappluEnv(GameConfig(max_turns=5))
        env.reset(seed=99)
        before = env.observe()
        rng_before = env._rng.getstate()

        for distance in (0, 22, True, 1.5):
            with self.subTest(distance=distance), self.assertRaises(
                (TypeError, ValueError)
            ):
                env.reset_curriculum(distance)
            self.assertEqual(env.observe(), before)
            self.assertEqual(env._rng.getstate(), rng_before)

        with mock.patch("src.environment.is_valid_hand", return_value=True):
            with self.assertRaisesRegex(ValueError, "after 200 attempts"):
                env.reset_curriculum(1, seed=123)
        self.assertEqual(env.observe(), before)
        self.assertEqual(env._rng.getstate(), rng_before)


if __name__ == "__main__":
    unittest.main()
