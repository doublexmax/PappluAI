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
        env = PappluEnv(GameConfig(num_decks=5, max_turns=254))
        obs = env.reset_warm_start(410)
        row = encode(obs)
        tape, suffix = tokens(row.prefix), tokens(row.suffix)
        self.assertEqual(len(tape), 30)
        self.assertEqual(tape[:7], [
            bytes((Kind.RULE_DECKS, 0, 0, 0, 6, 0, 0, 0)),
            bytes((Kind.RULE_HAND, 0, 0, 0, 22, 0, 0, 0)),
            bytes((Kind.RULE_SEQUENCES, 0, 0, 0, 6, 0, 0, 0)),
            bytes((Kind.RULE_TURNS, 0, 0, 0, 0, 255, 0, 0)),
            bytes((Kind.RULE_RECYCLE, 0, 0, 0, 2, 0, 0, 0)),
            bytes((Kind.RULE_PLAYERS, 0, 0, 0, 2, 0, 0, 0)),
            bytes((Kind.JOKER, 0, obs.joker + 1, 0, 0, 0, 0, 0)),
        ])
        for data, hand, kind in (
            (tape[7:29], obs.history[0].initial_hand, Kind.HAND_INIT),
            (suffix[7:], obs.hand, Kind.HAND),
        ):
            expected = [bytes((kind, 0, face + 1, 0, n + 1, 0, 0, 0)) for face, n in enumerate(hand) if n]
            self.assertEqual(data, expected + [PAD] * (len(data) - len(expected)))
        self.assertEqual(len(suffix), 30)
        self.assertEqual(tape[-1], bytes((Kind.EVENT_INITIAL, 1, 53, 53, 23, 255, obs.stock_remaining + 1, 1)))
        self.assertEqual(suffix[0], bytes((Kind.ACT_DISCARD, 1, 53, 0, 1, 255, obs.stock_remaining + 1, 1)))
        self.assertEqual(suffix[1], bytes((Kind.SEAT, 1, 0, 0, 23, 255, 0, 25)))

    def test_schema_rejects_games_where_a_used_counter_could_lose_information(self):
        validate_game(GameConfig(num_decks=5, cards_in_hand=5, required_sequences=0, max_turns=254))
        with self.assertRaisesRegex(ValueError, "at most 254 cards"):
            validate_game(GameConfig(num_decks=5, cards_in_hand=4, required_sequences=0))
        with self.assertRaisesRegex(ValueError, "at most 254 turns"):
            validate_game(GameConfig(max_turns=255))

    def test_face_zero_uses_one_byte_in_every_face_slot(self):
        obs = PappluEnv().reset(410)
        hand = (1,) + (0,) * 51
        initial = replace(obs.history[0], initial_hand=hand)
        discard = replace(initial, kind=EventKind.DISCARD, face=0, offered_top=51)
        row = encode(replace(obs, hand=hand, joker=0, discard_pile=(0,), history=(initial, discard)))
        tape, suffix = tokens(row.prefix), tokens(row.suffix)
        self.assertEqual([token[2] for token in (tape[6], tape[7], tape[-1], suffix[0], suffix[7])], [1] * 5)
        self.assertEqual(tape[-1][3], 52)
        self.assertEqual([token[2] for token in (tape[0], tape[-2], suffix[1], suffix[-1])], [0, 53, 0, 0])

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
        self.assertEqual(observed[:4], bytes((Kind.EVENT_INITIAL + EventKind.DRAW_STOCK, 3, 53, drawn.discard_top + 1)))
        for seat, view in enumerate(other.players):
            self.assertEqual(tokens(other_row.suffix)[1 + seat], bytes((
                Kind.SEAT, seat + 1, 0, 0, view.hand_size + 1, view.turns_remaining + 1,
                0, 17 + 8 * view.active + view.next_player,
            )))
        self.assertEqual(tokens(other_row.suffix)[4:7],
                         [bytes((Kind.SEAT, seat, 0, 0, 0, 0, 0, 1)) for seat in (4, 5, 6)])

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
