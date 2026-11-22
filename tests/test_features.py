from dataclasses import replace
import unittest

from src.evaluate import is_valid_hand
from src.game.environment import ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, PappluEnv, discard_action
from src.game.information import EventKind
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import FIELDS, HEADER_TOKENS, Kind, PAD, encode, legal_mask, validate_game


def tokens(data):
    return [data[i:i + FIELDS] for i in range(0, len(data), FIELDS)]


class TestByteEncoding(unittest.TestCase):
    def test_header_and_full_hand_snapshots_are_lossless(self):
        env = PappluEnv(GameConfig(num_decks=4, max_turns=235))
        obs = env.reset_warm_start(410)
        row = encode(obs)
        tape, suffix = tokens(row.prefix), tokens(row.suffix)
        self.assertEqual(len(tape), 30)
        self.assertEqual(tape[:7], [
            bytes((Kind.RULE_DECKS, 24, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.RULE_HAND, 41, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.RULE_SEQUENCES, 25, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.RULE_TURNS, 255, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.RULE_RECYCLE, 21, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.RULE_PLAYERS, 21, 0, 0, 0, 0, 0, 0)),
            bytes((Kind.JOKER, 0, obs.joker + 1, obs.joker // 13 + 1, obs.joker % 13 + 1, 0, 0, 0)),
        ])
        for data, hand, kind, has_pointer in (
            (tape[7:29], obs.history[0].initial_hand, Kind.HAND_INIT, False),
            (suffix[8:], obs.hand, Kind.HAND, True),
        ):
            expected = [
                bytes((kind, 10 + n, face + 1, face // 13 + 1, face % 13 + 1, 0, 0, face + 3 if has_pointer else 0))
                for face, n in enumerate(hand) if n
            ]
            self.assertEqual(data, expected + [PAD] * (len(data) - len(expected)))
        self.assertEqual(len(suffix), 30)
        self.assertEqual(tape[-1], bytes((Kind.EVENT_INITIAL, 1, 53, 5, 14, 53, 121, 0)))
        self.assertEqual(suffix[0], bytes((Kind.ACT, 1, 53, 5, 14, 100, 121, 0)))
        self.assertEqual(suffix[1], bytes((Kind.TABLE_STATE, 8, 0, 0, 0, 84, 222, 0)))
        self.assertEqual(suffix[2], bytes((Kind.SEAT, 1, 0, 0, 0, 82, 121, 0)))
        self.assertTrue(all(token[7] == 0 for token in tape + suffix[:8]))

    def test_schema_rejects_games_that_overflow_the_caps_or_discard_slots(self):
        validate_game(GameConfig(num_decks=4, cards_in_hand=21, max_turns=235))
        with self.assertRaisesRegex(ValueError, "at most 4 decks"):
            validate_game(GameConfig(num_decks=5))
        with self.assertRaisesRegex(ValueError, "at most 235 turns"):
            validate_game(GameConfig(max_turns=236))
        with self.assertRaisesRegex(ValueError, "at most 21 cards before drawing"):
            validate_game(GameConfig(cards_in_hand=22))

    def test_all_twenty_two_distinct_discard_faces_have_a_pointer(self):
        obs = PappluEnv().reset_warm_start(410)
        faces = tuple(range(21)) + (51,)
        hand = tuple(int(face in faces) for face in range(52))
        row = encode(replace(obs, hand=hand))
        suffix = tokens(row.suffix)
        self.assertEqual(len(suffix), 30)
        self.assertEqual(suffix[8], bytes((Kind.HAND, 11, 1, 1, 1, 0, 0, 3)))
        self.assertEqual(suffix[-1], bytes((Kind.HAND, 11, 52, 4, 13, 0, 0, 54)))
        self.assertEqual([token[7] for token in suffix[8:]], [face + 3 for face in faces])
        self.assertEqual(row.legal, sum(1 << (face + 2) for face in faces))

    def test_face_zero_uses_one_byte_in_every_face_slot(self):
        obs = PappluEnv().reset(410)
        hand = (1,) + (0,) * 51
        initial = replace(obs.history[0], initial_hand=hand)
        discard = replace(initial, kind=EventKind.DISCARD, face=0, offered_top=51)
        row = encode(replace(obs, hand=hand, joker=0, discard_pile=(0,), history=(initial, discard)))
        tape, suffix = tokens(row.prefix), tokens(row.suffix)
        self.assertEqual([token[2:5] for token in (tape[6], tape[7], tape[-1], suffix[0], suffix[8])], [b"\1\1\1"] * 5)
        self.assertEqual(tape[-1][5], 52)
        self.assertEqual([token[2:5] for token in (tape[0], tape[-2], suffix[1], suffix[-1])],
                         [b"\0\0\0", bytes((53, 5, 14)), b"\0\0\0", b"\0\0\0"])

    def test_multiplayer_deltas_keep_private_draws_redacted_and_seats_relative(self):
        env = MultiplayerEnv(MatchConfig(GameConfig(), 3))
        env.reset(803)
        first = encode(env.observe())
        env.step(ACTION_DRAW_STOCK)
        drawn = env.observe()
        second = encode(drawn, since=first.events)
        mask = legal_mask(drawn)
        env.step((mask & -mask).bit_length() - 1)
        other = env.observe()
        other_row = encode(other)
        self.assertEqual(first.prefix + second.prefix, encode(drawn).prefix)
        self.assertEqual(second.cut, HEADER_TOKENS + 2)
        self.assertEqual(encode(drawn, since=second.events).prefix, b"")
        observed = tokens(other_row.prefix)[HEADER_TOKENS + 1]
        self.assertEqual(observed, bytes((
            Kind.EVENT_INITIAL + EventKind.DRAW_STOCK, 3, 53, 5, 14, drawn.discard_top + 1, 41, 0,
        )))
        own = other.players[0]
        self.assertEqual(tokens(other_row.suffix)[1], bytes((
            Kind.TABLE_STATE, 7, 0, 0, 0, 83 + 2 * own.next_player + int(own.active), 175, 0,
        )))
        for seat, view in enumerate(other.players):
            self.assertEqual(tokens(other_row.suffix)[2 + seat], bytes((
                Kind.SEAT, seat + 1, 0, 0, 0, 60 + view.hand_size, 1 + view.turns_remaining, 0,
            )))
        self.assertEqual(tokens(other_row.suffix)[5:8],
                         [bytes((Kind.SEAT, seat, 0, 0, 0, 0, 0, 0)) for seat in (4, 5, 6)])

    def test_restricts_take_back_but_preserves_a_forced_action(self):
        env = PappluEnv()
        initial = env.reset(410)
        discard = env.step(ACTION_TAKE_DISCARD)
        taken = 1 << discard_action(initial.discard_top)
        self.assertFalse(legal_mask(discard) & taken)
        same_face = tuple(22 if face == initial.discard_top else 0 for face in range(52))
        self.assertEqual(legal_mask(replace(discard, hand=same_face)), taken)
        keep = (legal_mask(discard) & -legal_mask(discard)).bit_length() - 1
        after = env.step(keep)
        self.assertEqual(legal_mask(after), 1 << ACTION_DRAW_STOCK)
        self.assertEqual(legal_mask(replace(after, stock_remaining=0)), 1 << ACTION_TAKE_DISCARD)
        with self.assertRaisesRegex(ValueError, "event cursor"):
            encode(initial, since=2)

    def test_solo_cannot_take_back_its_top_discard_after_refill(self):
        config = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=50)
        env = PappluEnv(config)
        obs = env.reset(0)
        self.assertFalse(is_valid_hand(obs.hand, obs.joker, 1, 3))
        for _ in range(obs.stock_remaining):
            drawn = env.step(ACTION_DRAW_STOCK)
            obs = env.step(discard_action(drawn.history[-1].face))
        self.assertIs(obs.history[-1].kind, EventKind.REFILL)
        self.assertGreater(obs.stock_remaining, 0)
        self.assertEqual(legal_mask(obs), 1 << ACTION_DRAW_STOCK)
