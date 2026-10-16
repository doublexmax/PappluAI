from collections import Counter
import copy
from dataclasses import replace
import io
import random
import unittest
from unittest import mock

from src.game.environment import (
    GameConfig, Observation, PappluEnv, discard_action,
)
from src.game.information import EventKind
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.training.core import EpisodeReplay

try:
    import torch
except ImportError:
    torch = None


class TestInformationHistory(unittest.TestCase):
    def make_match(self, players=4, turns=12):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(cards_in_hand=3, required_sequences=1, max_turns=turns), players,
        ))
        env.reset(73)
        return env

    def test_stock_face_is_private_without_private_chronology_gaps(self):
        env = self.make_match()
        initial = env.observe()
        drawn_face = env._stock[-1]
        env.step(0)
        own = env.observe()
        env.step(discard_action(drawn_face))
        peer = env.observe()
        self.assertEqual(own.history[0].initial_hand, initial.hand)
        self.assertEqual(own.history[-1].face, drawn_face)
        self.assertEqual(own.history[-1].offered_top, initial.discard_top)
        self.assertEqual(peer.history[1].kind, EventKind.DRAW_STOCK)
        self.assertIsNone(peer.history[1].face)
        self.assertEqual(peer.history[1].actor, 3)
        self.assertEqual(peer.history[1].hand_size, 4)
        self.assertEqual(peer.history[1].stock_remaining, initial.stock_remaining - 1)
        self.assertEqual(peer.history[2].face, drawn_face)
        self.assertEqual(len(peer.history), 3)
        self.assertEqual(len(own.history), 2)
        self.assertEqual(peer.history[0].initial_hand, peer.hand)

    def test_hidden_stock_and_peer_cards_do_not_change_visible_observation(self):
        env = self.make_match()
        other = copy.deepcopy(env)
        other._stock.reverse()
        other._hands[1], other._hands[2] = other._hands[2], other._hands[1]
        self.assertEqual(env.observe(), other.observe())
        env.step(0)
        other.step(0)
        env._current_seat = other._current_seat = 1
        self.assertEqual(env._journal.observe(1)[1], other._journal.observe(1)[1])

    def test_discard_pickup_and_passed_top_remain_public(self):
        env = self.make_match(players=3)
        top = env.observe().discard_top
        env.step(1)
        env.step(discard_action(top))
        peer = env.observe()
        self.assertEqual(peer.history[1].kind, EventKind.TAKE_DISCARD)
        self.assertEqual(peer.history[1].face, top)
        self.assertEqual(peer.history[1].offered_top, top)
        self.assertEqual(peer.history[2].face, top)
        self.assertEqual(peer.history[1].actor, 2)

    def test_history_retains_own_old_draw_and_full_prefix_past_sixteen_events(self):
        env = self.make_match(turns=20)
        initial = env.observe()
        private_face = env._stock[-1]
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(20):
                env.step(0)
                face = next(i for i, n in enumerate(env.observe().hand) if n)
                env.step(discard_action(face))
        observation = env.observe()
        self.assertEqual(len(observation.history), 41)
        self.assertEqual(observation.history[0].initial_hand, initial.hand)
        self.assertEqual(observation.history[1].face, private_face)
        self.assertEqual(len(initial.history), 1)
        self.assertEqual(initial.history[0].initial_hand, initial.hand)

    def test_relative_projection_survives_cyclic_absolute_seat_relabeling(self):
        original = self.make_match()
        original._current_seat = 2
        rotated = copy.deepcopy(original)
        rotated._hands = [hand.copy() for hand in original._hands[2:] + original._hands[:2]]
        rotated._turns_remaining = original._turns_remaining[2:] + original._turns_remaining[:2]
        rotated._current_seat = 0
        rotated._journal.start(
            rotated._hands, rotated._turns_remaining, len(rotated._stock),
            rotated._discard[-1],
        )
        self.assertEqual(original.observe(), rotated.observe())
        original.step(0)
        rotated.step(0)
        self.assertEqual(original.observe(), rotated.observe())

    def test_player_views_encode_skipped_next_and_previous_players(self):
        env = self.make_match()
        env._turns_remaining[1] = 0
        players = env.observe().players
        self.assertEqual(players[0].next_player, 2)
        self.assertEqual(players[2].previous_player, 0)
        self.assertFalse(players[1].active)
        self.assertEqual(players[3].next_player, 0)
        self.assertEqual(tuple(p.hand_size for p in players), (3, 3, 3, 3))

    def test_constructed_starts_record_only_the_actual_initial_hand(self):
        env = PappluEnv(GameConfig(max_turns=5))
        for reset in (env.reset, env.reset_warm_start, lambda seed: env.reset_curriculum(4, seed)):
            obs = reset(seed=17)
            self.assertEqual(obs.history[0].initial_hand, obs.hand)
            self.assertEqual(len(obs.history), 1)
            self.assertEqual(obs.players[0].hand_size, sum(obs.hand))
            self.assertEqual(Observation.from_dict(obs.to_dict()), obs)

    def test_initial_history_rejects_consistently_corrupted_reset_state(self):
        original = self.make_match(players=2).observe().to_dict()
        wrong_stock = copy.deepcopy(original)
        wrong_stock["stock_remaining"] -= 1
        wrong_stock["history"][0]["stock_remaining"] -= 1
        reduced_turns = copy.deepcopy(original)
        reduced_turns["turns_remaining"] -= 1
        reduced_turns["players"][0]["turns_remaining"] -= 1
        reduced_turns["history"][0]["turns_remaining"] -= 1
        extra_card = copy.deepcopy(original)
        cards = list(extra_card["hand"])
        cards[next(face for face, count in enumerate(cards) if count == 0)] += 1
        extra_card["hand"] = extra_card["history"][0]["initial_hand"] = tuple(cards)
        extra_card["players"][0]["hand_size"] += 1
        extra_card["history"][0]["hand_size"] += 1
        extra_card["discard_pile"] = ()
        extra_card["history"][0]["offered_top"] = None
        extra_card["phase"] = 1
        for name, corrupted in (
            ("stock", wrong_stock), ("turns", reduced_turns), ("multiplayer warm start", extra_card),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "initial"):
                Observation.from_dict(corrupted)

    def test_initial_discard_shape_matches_the_start_type(self):
        env = PappluEnv(GameConfig(max_turns=5))
        ordinary = env.reset(17).to_dict()
        ordinary["discard_pile"] = ()
        ordinary["history"][0]["offered_top"] = None
        warm = env.reset_warm_start(17).to_dict()
        warm["discard_pile"] = (0,)
        warm["history"][0]["offered_top"] = 0
        for corrupted in (ordinary, warm):
            with self.subTest(warm=corrupted["warm_start"]), self.assertRaisesRegex(ValueError, "initial"):
                Observation.from_dict(corrupted)

    def test_ordinary_start_with_zero_stock_roundtrips(self):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(num_decks=1, cards_in_hand=25, required_sequences=1), 2,
        ))
        env.reset(73)
        observation = env.observe()
        self.assertEqual(observation.stock_remaining, 0)
        self.assertIsNotNone(observation.discard_top)
        self.assertEqual(Observation.from_dict(observation.to_dict()), observation)

    def test_replay_eviction_does_not_cut_history_or_expose_future_events(self):
        env = self.make_match(players=2)
        replay = EpisodeReplay(2)
        decisions = []
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(12):
                obs = env.observe()
                if env.view().current_seat == 0:
                    decisions.append((obs, 0, 0.0))
                env.step(0)
                env.step(discard_action(next(i for i, n in enumerate(env.observe().hand) if n)))
        replay.append(tuple(decisions))
        restored = EpisodeReplay(2)
        restored.load_state_dict(replay.state_dict())
        retained = replay.state_dict()["episodes"][0]
        self.assertEqual(len(retained["transitions"]), 2)
        self.assertEqual(len(retained["history"]), 21)
        self.assertEqual(tuple(t[1] for t in retained["transitions"]), (17, 21))
        first, second = tuple(t[0] for t in restored._episodes[0])
        self.assertEqual(len(first.history), 17)
        self.assertEqual(len(second.history), 21)
        self.assertEqual(len(env.observe().history), 25)
        self.assertEqual(restored.state_dict(), replay.state_dict())

    def test_primitive_load_rejects_privileged_and_missing_history(self):
        env = self.make_match()
        env.step(0)
        env._current_seat = 1
        obs = env.observe()
        raw = obs.to_dict()
        bad = copy.deepcopy(raw)
        bad["history"][1]["face"] = 17
        with self.assertRaisesRegex(ValueError, "opponent stock"):
            Observation.from_dict(bad)
        with self.assertRaisesRegex(ValueError, "complete initial history"):
            Observation.from_dict({**raw, "history": ()})
        with self.assertRaisesRegex(ValueError, "current information schema"):
            Observation.from_dict({"hand": obs.hand})

    def test_primitive_load_rejects_inconsistent_own_card_history(self):
        env = PappluEnv(GameConfig(max_turns=5))
        original = env.reset(17)
        observation = env.step(0)
        self.assertEqual(Observation.from_dict(observation.to_dict()), observation)
        wrong_face = observation.to_dict()
        wrong_face["history"][-1]["face"] = (
            wrong_face["history"][-1]["face"] + 1
        ) % 52
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(wrong_face)
        wrong_count = original.to_dict()
        wrong_count["history"][0]["hand_size"] = 0
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(wrong_count)
        missing_draw = observation.to_dict()
        missing_draw["history"] = missing_draw["history"][:-1]
        with self.assertRaisesRegex(ValueError, "own card history"):
            Observation.from_dict(missing_draw)
        wrong_turns = observation.to_dict()
        wrong_turns["players"][0]["turns_remaining"] -= 1
        with self.assertRaisesRegex(ValueError, "own player turns"):
            Observation.from_dict(wrong_turns)


class TestReplayJournals(unittest.TestCase):
    @staticmethod
    def episodes(players=3, rounds=2, num_decks=3, seed=73):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(
                num_decks=num_decks, cards_in_hand=3,
                required_sequences=1, max_turns=rounds,
            ), players,
        ))
        env.reset(seed)
        episodes = [[] for _ in range(players)]
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(rounds * players):
                seat = env.view().current_seat
                episodes[seat].append((env.observe(), 0, 0.125))
                env.step(0)
                obs = env.observe()
                action = discard_action(next(i for i, count in enumerate(obs.hand) if count))
                episodes[seat].append((obs, action, -0.25))
                env.step(action)
        return tuple(tuple(episode) for episode in episodes)

    def replay(self, **kwargs):
        replay = EpisodeReplay(500)
        for episode in self.episodes(**kwargs):
            replay.append(episode)
        return replay

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_weights_only_roundtrip_keeps_one_journal_and_shared_events(self):
        from src.model.encoding import collate_observations

        replay = self.replay()
        state = replay.state_dict()
        self.assertEqual(
            tuple(tuple(t[1] for t in e["transitions"]) for e in state["episodes"]),
            ((1, 2, 7, 8), (3, 4, 9, 10), (5, 6, 11, 12)),
        )
        self.assertEqual(tuple(len(e["history"]) for e in state["episodes"]), (8, 10, 12))
        for journal in state["episodes"]:
            for snapshot, _prefix, _action, _target in journal["transitions"]:
                self.assertNotIn("history", snapshot)
                self.assertNotIn("config", snapshot)
                self.assertNotIn("joker", snapshot)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = EpisodeReplay(500)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(tuple(restored._episodes), tuple(replay._episodes))
        for episode in restored._episodes:
            final_history = episode[-1][0].history
            for obs, _action, _target in episode:
                for index, event in enumerate(obs.history):
                    self.assertIs(event, final_history[index])
        original = [obs for e in replay._episodes for obs, _, _ in e]
        reloaded = [obs for e in restored._episodes for obs, _, _ in e]
        left = collate_observations(original, torch.device("cpu"))
        right = collate_observations(reloaded, torch.device("cpu"))
        for name in left.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)), name)
        self.assertEqual(
            [target for e in restored._episodes for _, _, target in e],
            [0.125, -0.25] * 6,
        )

    def test_malformed_prefixes_and_future_snapshots_are_rejected_atomically(self):
        replay = self.replay(players=2)
        before = replay.state_dict()
        for prefix in (0, -1, True, 1.0, 7):
            bad = copy.deepcopy(before)
            journal = bad["episodes"][0]
            transitions = list(journal["transitions"])
            snapshot, _prefix, action, target = transitions[0]
            transitions[0] = (snapshot, prefix, action, target)
            journal["transitions"] = tuple(transitions)
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "prefix"):
                replay.load_state_dict(bad)
            self.assertEqual(replay.state_dict(), before)
        bad = copy.deepcopy(before)
        journal = bad["episodes"][0]
        snapshot, _prefix, action, target = journal["transitions"][0]
        journal["transitions"] = ((snapshot, 2, action, target),)
        with self.assertRaisesRegex(ValueError, "own card history"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        journal = bad["episodes"][0]
        journal["transitions"] = tuple(reversed(journal["transitions"]))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            replay.load_state_dict(bad)
        self.assertEqual(replay.state_dict(), before)

    def test_corrupt_own_history_and_chronology_are_rejected(self):
        replay = self.replay(players=2)
        before = replay.state_dict()
        bad = copy.deepcopy(before)
        bad["episodes"][0]["history"][1]["face"] = (
            bad["episodes"][0]["history"][1]["face"] + 1
        ) % 52
        with self.assertRaisesRegex(ValueError, "own card history"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        history = list(bad["episodes"][0]["history"])
        history[1], history[2] = history[2], history[1]
        bad["episodes"][0]["history"] = tuple(history)
        with self.assertRaisesRegex(ValueError, "chronology"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        bad["episodes"][0]["history"][1]["stock_remaining"] += 1
        with self.assertRaisesRegex(ValueError, "chronology"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        snapshot = bad["episodes"][0]["transitions"][0][0]
        snapshot["turns_remaining"] -= 1
        snapshot["players"][0]["turns_remaining"] -= 1
        with self.assertRaisesRegex(ValueError, "own player turns"):
            replay.load_state_dict(bad)
        self.assertEqual(replay.state_dict(), before)

    def test_legal_action_must_match_its_recorded_history_event(self):
        replay = self.replay(players=2)
        original = replay.state_dict()
        corrupted = copy.deepcopy(original)
        journal = corrupted["episodes"][0]
        transitions = list(journal["transitions"])
        snapshot, prefix, _action, target = transitions[0]
        transitions[0] = (snapshot, prefix, 1, target)
        journal["transitions"] = tuple(transitions)
        with self.assertRaisesRegex(ValueError, "action does not match"):
            replay.load_state_dict(corrupted)
        self.assertEqual(replay.state_dict(), original)

    def test_snapshot_outcome_and_start_flags_match_history(self):
        env = PappluEnv(GameConfig(max_turns=1))
        initial = env.reset(17)
        for change in (
            {"won": True}, {"last_reward": 1.0}, {"warm_start": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Observation.from_dict({**initial.to_dict(), **change})
        env.step(0)
        terminal = env.step(discard_action(next(i for i, n in enumerate(env.observe().hand) if n)))
        self.assertEqual(Observation.from_dict(terminal.to_dict()), terminal)
        warm = env.reset_warm_start(17)
        self.assertEqual(Observation.from_dict(warm.to_dict()), warm)
        with self.assertRaisesRegex(ValueError, "warm_start"):
            Observation.from_dict({**warm.to_dict(), "warm_start": False})

    def test_mixed_seats_configurations_and_episode_histories_are_rejected(self):
        episodes = self.episodes(players=2)
        mixed = EpisodeReplay(500)
        mixed.append((episodes[0][0], episodes[1][0]))
        with self.assertRaisesRegex(ValueError, "mixes seat or episode"):
            mixed.state_dict()
        obs, action, target = episodes[0][-1]
        changed = replace(obs, config=replace(obs.config, max_turns=3))
        mixed = EpisodeReplay(500)
        mixed.append(episodes[0][:-1] + ((changed, action, target),))
        with self.assertRaisesRegex(ValueError, "configuration"):
            mixed.state_dict()
        replay = self.replay(players=2)
        before = replay.state_dict()
        bad = copy.deepcopy(before)
        journal = bad["episodes"][0]
        peer = bad["episodes"][1]["transitions"][0][0]
        _snapshot, prefix, action, target = journal["transitions"][0]
        journal["transitions"] = ((peer, prefix, action, target),)
        with self.assertRaisesRegex(ValueError, "own card history"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        bad["episodes"][0]["history"] += bad["episodes"][1]["history"]
        with self.assertRaisesRegex(ValueError, "later initial event"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        bad["episodes"][0]["transitions"][0][0]["config"] = bad["episodes"][0]["config"]
        with self.assertRaisesRegex(ValueError, "current information schema"):
            replay.load_state_dict(bad)
        bad = copy.deepcopy(before)
        bad["episodes"][0]["transitions"][0][0]["players"] += (
            bad["episodes"][0]["transitions"][0][0]["players"][0],
        )
        with self.assertRaisesRegex(ValueError, "player count"):
            replay.load_state_dict(bad)
        self.assertEqual(replay.state_dict(), before)

    def test_refill_journals_reconstruct_public_table_at_each_prefix(self):
        replay = self.replay(players=2, rounds=25, num_decks=1)
        state = replay.state_dict()
        self.assertIn(
            int(EventKind.REFILL),
            [event["kind"] for event in state["episodes"][0]["history"]],
        )
        restored = EpisodeReplay(500)
        restored.load_state_dict(state)
        self.assertEqual(tuple(restored._episodes), tuple(replay._episodes))
        self.assertEqual(len(restored), 100)


class TestCountBalancedReplay(unittest.TestCase):
    def test_counts_are_balanced_before_episodes_and_decisions(self):
        replay = EpisodeReplay(100)
        for players, decisions in ((2, 30), (3, 12), (4, 1)):
            env = MultiplayerEnv(MatchConfig(
                GameConfig(cards_in_hand=3, required_sequences=1), players,
            ))
            env.reset(players)
            replay.append(tuple((env.observe(), 0, 0.0) for _ in range(decisions)))
        batch = replay.sample(random.Random(81), 30)
        self.assertEqual(Counter(len(item[0].players) for item in batch), {2: 10, 3: 10, 4: 10})
        replay.sample(random.Random(17), 1)
        restored = EpisodeReplay(100)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(
            replay.sample(random.Random(18), 11),
            restored.sample(random.Random(18), 11),
        )


if __name__ == "__main__":
    unittest.main()
