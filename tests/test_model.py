import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from src.game.environment import GameConfig, PappluEnv, discard_action, legal_action_mask
from src.game.information import EventKind
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import _meld_membership, encode
from src.model.spec import ModelSpec, ModelSpecError

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import CheckpointPolicy
    from src.model.network import (
        QNetwork, checkpoint_payload, collate, collate_observations, load_checkpoint,
        masked_q_values, save_checkpoint, select_action, select_greedy_action,
    )


class TestModelSpec(unittest.TestCase):
    def test_roundtrip_requires_all_architecture_dimensions(self):
        spec = ModelSpec(128, 3)
        self.assertEqual(ModelSpec.from_dict({"width": 128, "blocks": 3}), spec)
        self.assertEqual(spec.to_dict(), {"width": 128, "blocks": 3})
        for raw in (
            None, {"width": 64}, {"blocks": 2}, {"width": 64, "heads": 4, "blocks": 2},
            {"width": 64, "blocks": 2, "dropout": 0}, {"width": True, "blocks": 2},
            {"width": 64.0, "blocks": 2}, {"width": 64, "blocks": True},
        ):
            with self.subTest(raw=raw), self.assertRaises(ModelSpecError):
                ModelSpec.from_dict(raw)
        for arguments in ((4, 1), (64, 0), (True, 1), (64, True), (64.0, 2), (64, 2.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ModelSpecError):
                ModelSpec(*arguments)
        self.assertEqual(ModelSpec(9, 1).to_dict(), {"width": 9, "blocks": 1})


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestCardNetwork(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.network = QNetwork()
        self.device = torch.device("cpu")

    def observation(self, players=4, turns=5, takes=False):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(cards_in_hand=3, required_sequences=1, max_turns=30), players,
        ))
        env.reset(19)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(turns):
                env.step(int(takes))
                env.step(discard_action(next(i for i, n in enumerate(env.observe().hand) if n)))
        return env.observe()

    def values(self, *observations):
        return self.network(collate_observations(observations, self.device))

    def test_one_network_scores_variable_tables_and_sparse_histories(self):
        observations = [PappluEnv().reset(17)] + [
            self.observation(players, turns=10) for players in (2, 3, 4, 5, 6)
        ]
        values = self.values(*observations)
        self.assertEqual(values.shape, (6, 54))
        self.assertTrue(torch.isfinite(values).all())
        torch.testing.assert_close(
            self.network.policy(collate_observations(observations, self.device)), values, rtol=0, atol=0,
        )

    def test_cached_features_collate_without_a_non_cpu_default_device_leak(self):
        observations = [self.observation(2, 1), self.observation(3, 2)]
        expected = self.values(*observations)
        features = [encode(obs) for obs in observations]
        previous = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            batch = collate(features, self.device)
            self.assertEqual(batch.card.device.type, "cpu")
            self.assertEqual(batch.card.shape, (2, 52, 22))
            self.assertEqual(batch.seat.shape, (2, 6, 9))
            self.assertEqual(batch.context.shape, (2, 22))
            torch.testing.assert_close(self.network(batch), expected, rtol=0, atol=0)
        finally:
            torch.set_default_device(previous)

    def test_batch_composition_does_not_change_action_values(self):
        short, long = self.observation(2, 1), self.observation(6, 20)
        torch.testing.assert_close(self.values(short)[0], self.values(short, long)[0], rtol=1e-5, atol=1e-6)

    def test_opposite_seat_public_takes_and_discards_change_action_values(self):
        obs = self.observation(4, 8, takes=True)
        baseline = self.values(obs)
        for kind in (EventKind.TAKE_DISCARD, EventKind.DISCARD):
            position = next(i for i, event in enumerate(obs.history) if event.actor == 2 and event.kind is kind)
            event = obs.history[position]
            changed = replace(obs, history=(
                *obs.history[:position], replace(event, face=(event.face + 1) % 52), *obs.history[position + 1:],
            ))
            self.assertGreater(float((baseline - self.values(changed)).abs().max()), 1e-7)
        initial = list(obs.history[0].initial_hand)
        present = next(i for i, n in enumerate(initial) if n)
        initial[present] -= 1
        initial[(present + 1) % 52] += 1
        changed = replace(obs, history=(replace(obs.history[0], initial_hand=tuple(initial)), *obs.history[1:]))
        torch.testing.assert_close(baseline, self.values(changed), rtol=0, atol=0)

    def test_actual_next_player_relationship_changes_scores(self):
        obs = self.observation(4, 4)
        changed = replace(obs, players=(replace(obs.players[0], next_player=2, previous_player=1), *obs.players[1:]))
        self.assertGreater(float((self.values(obs) - self.values(changed)).abs().max()), 1e-7)

    def test_shared_checkpoint_policy_has_no_cross_game_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_checkpoint(str(path), self.network, GameConfig())
            policy = CheckpointPolicy(str(path))
            first = self.observation(2, 3)
            before = policy.act(first, random.Random(1))
            with torch.no_grad():
                values_before = policy.network(collate_observations([first], self.device)).clone()
            for count in (4, 3, 2, 6):
                obs = self.observation(count, count)
                self.assertTrue(legal_action_mask(obs)[policy.act(obs, random.Random(count))])
            self.assertEqual(before, policy.act(first, random.Random(999)))
            with torch.no_grad():
                values_after = policy.network(collate_observations([first], self.device))
            torch.testing.assert_close(values_before, values_after, rtol=0, atol=0)

    def test_mask_and_exploration_use_only_legal_actions(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        env.reset(12)
        obs = env.step(0)
        mask = legal_action_mask(obs)
        with torch.no_grad():
            self.network.policy.face_out.bias.fill_(-1e10)
        values = masked_q_values(self.network, obs, mask)
        self.assertTrue(torch.isneginf(values[0]))
        self.assertTrue(torch.isfinite(values[2:][torch.tensor(mask[2:])]).all())
        self.assertTrue(mask[select_greedy_action(self.network, obs)])
        seen = {select_action(self.network, obs, 1.0, random.Random(seed)) for seed in range(100)}
        self.assertEqual(seen, {i for i, allowed in enumerate(mask) if allowed})
        with self.assertRaisesRegex(ValueError, "no legal actions"):
            select_greedy_action(self.network, replace(obs, phase=type(obs.phase).TERMINAL))

    def test_card_relationships_change_action_specific_values(self):
        obs = self.observation(3, 2)
        hand = list(obs.hand)
        present = next(i for i, n in enumerate(hand) if n)
        hand[present] -= 1
        hand[(present + 2) % 52] += 1
        delta = self.values(obs)[0] - self.values(replace(obs, hand=tuple(hand)))[0]
        self.assertGreater(float(delta.abs().max()), 1e-7)
        self.assertGreater(float(delta[2:].std()), 1e-7)

    def test_suit_permutation_relabels_action_values(self):
        obs, suits = self.observation(4, 12), (2, 0, 3, 1)

        def face(index):
            return None if index is None else suits[index // 13] * 13 + index % 13

        def counts(hand):
            result = [0] * 52
            for index, count in enumerate(hand):
                result[face(index)] = count
            return tuple(result)

        changed = replace(
            obs, hand=counts(obs.hand), joker=face(obs.joker),
            discard_pile=tuple(face(index) for index in obs.discard_pile),
            history=tuple(replace(
                event, face=face(event.face), offered_top=face(event.offered_top),
                initial_hand=counts(event.initial_hand) if event.initial_hand else (),
            ) for event in obs.history),
        )
        actions = [0, 1] + [discard_action(face(index)) for index in range(52)]
        torch.testing.assert_close(self.values(obs), self.values(changed)[:, actions], rtol=1e-5, atol=1e-6)

    def test_draw_head_can_select_both_stock_and_discard(self):
        observations = [self.observation(2, turns) for turns in range(1, 9)]
        selected = set()
        for seed in range(8):
            torch.manual_seed(100 + seed)
            net = QNetwork()
            net.eval()
            selected.update(select_greedy_action(net, obs, device=self.device) for obs in observations)
        self.assertEqual(selected, {0, 1})

    def test_q_only_policy_returns_dueling_values_for_legal_and_terminal_rows(self):
        batch = collate_observations([self.observation(), PappluEnv().reset_warm_start(3)], self.device)
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.zero_()
            self.network.policy.value_out.bias.fill_(0.25)
            self.network.policy.draw_out.bias.copy_(torch.tensor([1.0, 3.0]))
            self.network.policy.face_out.bias.fill_(7.0)
        expected = torch.tensor([[-0.75, 1.25] + [5.25] * 52, [-5.75, -3.75] + [0.25] * 52])
        torch.testing.assert_close(self.network(batch), expected, rtol=0, atol=0)
        terminal = replace(batch, legal=torch.zeros_like(batch.legal))
        expected_terminal = torch.tensor([[1.25, 3.25] + [7.25] * 52] * 2)
        torch.testing.assert_close(self.network(terminal), expected_terminal, rtol=0, atol=0)
        self.assertEqual({name.split(".")[0] for name in self.network.state_dict()}, {"policy"})

    def test_legal_centering_ignores_illegal_advantages_and_terminal_mask_is_finite(self):
        batch = collate_observations([PappluEnv().reset(41)], self.device)
        with torch.no_grad():
            baseline = self.network(batch)
            self.network.policy.face_out.bias.add_(100)
            changed = self.network(batch)
        torch.testing.assert_close(baseline[:, :2], changed[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, 2:] - baseline[:, 2:], torch.full((1, 52), 100.0), rtol=1e-5, atol=1e-5)
        terminal = replace(batch, legal=torch.zeros_like(batch.legal))
        self.assertTrue(torch.isfinite(self.network(terminal)).all())

    def test_model_has_no_stochastic_or_running_inference_state(self):
        obs = self.observation(3, 5)
        self.network.train()
        first = self.values(obs)
        self.network.eval()
        torch.testing.assert_close(first, self.values(obs), rtol=0, atol=0)

    def test_checkpoint_roundtrip_preserves_nondefault_spec_and_wire_format(self):
        self.network = QNetwork(ModelSpec(32, 3))
        obs = self.observation()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "model.pt"
            save_checkpoint(str(path), self.network, obs.config, {"note": "fresh"})
            loaded, config, meta = load_checkpoint(str(path))
            self.assertEqual((config, meta, loaded.spec), (obs.config, {"note": "fresh"}, self.network.spec))
            torch.testing.assert_close(self.values(obs), loaded(collate_observations([obs], self.device)), rtol=0, atol=0)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["architecture"], "card_mlp")
            self.assertEqual(payload["checkpoint_version"], 5)
            self.assertEqual(payload["encoding_version"], 5)
            self.assertEqual(payload["model_spec"], {"width": 32, "blocks": 3})
            self.assertNotIn("state_dim", payload)

    def test_bad_or_old_checkpoints_do_not_mutate_supplied_network(self):
        baseline = copy.deepcopy(self.network.state_dict())
        changes = [
            {"checkpoint_version": 1}, {"checkpoint_version": 2}, {"checkpoint_version": 3},
            {"checkpoint_version": 4}, {"checkpoint_version": True}, {"checkpoint_version": 5.0},
            {"encoding_version": 1}, {"encoding_version": 2}, {"encoding_version": 3},
            {"encoding_version": 4}, {"encoding_version": 5.0},
            {"architecture": "mlp"}, {"architecture": "axial_table"}, {"state_dim": 164},
            {"model_spec": {"width": 32, "blocks": 2}}, {"num_actions": 12},
            {"model_spec": {"width": 64, "heads": 4, "blocks": 2}},
            {"model_spec": {"width": 64, "blocks": True}}, {"meta": []},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pt"
            for change in changes:
                payload = checkpoint_payload(QNetwork(), GameConfig())
                payload.update(change)
                torch.save(payload, path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_checkpoint(str(path), network=self.network)
                for name, value in self.network.state_dict().items():
                    self.assertTrue(torch.equal(baseline[name], value), name)
            for defect in ("shape", "nan", "dtype"):
                payload = checkpoint_payload(QNetwork(), GameConfig())
                last = next(reversed(payload["model_state_dict"]))
                tensor = payload["model_state_dict"][last]
                if defect == "shape":
                    payload["model_state_dict"][last] = torch.zeros(2)
                elif defect == "nan":
                    tensor.reshape(-1)[0] = float("nan")
                else:
                    payload["model_state_dict"][last] = tensor.to(torch.float64)
                torch.save(payload, path)
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    load_checkpoint(str(path), network=self.network)
                for name, value in self.network.state_dict().items():
                    self.assertTrue(torch.equal(baseline[name], value), name)
            save_checkpoint(str(path), QNetwork(), GameConfig())
            with self.assertRaisesRegex(ValueError, "game_config"):
                load_checkpoint(str(path), expected_config=GameConfig(max_turns=1))

    def assert_meld_parity(self, observations):
        batch = collate_observations(observations, self.device)
        cards = batch.card[:, :, 13:18].reshape(len(observations), 4, 13, 5)
        context = batch.context[:, 17:18]
        self.assertEqual(cards.shape, (len(observations), 4, 13, 5))
        self.assertEqual(context.shape, (len(observations), 1))
        for idx, obs in enumerate(observations):
            natural, pure, any_meld = _meld_membership(obs.hand, obs.joker)
            expected, unavoidable = [], 0
            for face, copies in enumerate(obs.hand):
                suit, rank = divmod(face, 13)
                neighbors = (1, 12) if rank == 0 else (11, 0) if rank == 12 else (rank - 1, rank + 1)
                expected.append((
                    sum(obs.hand[suit * 13 + neighbor] > 0 for neighbor in neighbors) / 2.0,
                    sum(obs.hand[other * 13 + rank] > 0 for other in range(4) if other != suit) / 3.0,
                    natural[face], pure[face], any_meld[face],
                ))
                pts = 0 if rank == obs.joker % 13 else (10 if rank == 0 or rank >= 10 else rank + 1)
                unavoidable += copies * pts * (1 - any_meld[face])
            torch.testing.assert_close(
                cards[idx].reshape(52, 5), torch.tensor(expected, dtype=torch.float32),
                rtol=0, atol=0,
            )
            self.assertAlmostEqual(
                context[idx, 0].item(), unavoidable / (10.0 * obs.config.cards_in_hand), places=6,
            )
        return cards

    def test_encoded_meld_channels_match_membership_across_random_deals(self):
        observations = []
        for decks, hand_size, sequences in ((1, 3, 1), (2, 7, 2), (3, 21, 5), (6, 30, 10)):
            env = PappluEnv(GameConfig(
                num_decks=decks, cards_in_hand=hand_size, required_sequences=sequences,
            ))
            for seed in range(64):
                observations.append(env.reset(seed))
                observations.append(env.step(seed % 2))
        self.assert_meld_parity(observations)

    def test_encoded_meld_channels_handle_aces_papplus_wilds_and_physical_sets(self):
        cases = (
            ((0, 1, 2), 7, (0, 1, 2), (0, 1, 2), (0, 1, 2)),
            ((11, 12, 0), 7, (0, 11, 12), (0, 11, 12), (0, 11, 12)),
            ((12, 0, 1), 7, (), (), ()),
            ((16, 17, 7), 7, (), (7, 16, 17), (7, 16, 17)),
            ((16, 17, 33), 7, (), (), (16, 17, 33)),
            ((32, 33, 34), 7, (32, 33, 34), (32, 33, 34), (32, 33, 34)),
            ((5, 5, 5), 7, (), (5,), (5,)),
            ((5, 18, 31, 44), 7, (), (), (5, 18, 31, 44)),
            ((5, 5, 7), 7, (), (), ()),
            ((7, 20, 33), 7, (), (), (7, 20, 33)),
            ((7, 7, 7), 7, (), (7,), (7,)),
            ((), 7, (), (), ()),
        )
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        observations = [replace(
            obs, hand=tuple(cards.count(face) for face in range(52)), joker=joker,
        ) for cards, joker, *_ in cases]
        extras = self.assert_meld_parity(observations).reshape(len(cases), 52, 5)
        for idx, (_, _, natural, pure, any_meld) in enumerate(cases):
            for channel, expected in zip((2, 3, 4), (natural, pure, any_meld)):
                self.assertEqual(
                    tuple(face for face in range(52) if extras[idx, face, channel].item()), expected,
                )

    def test_model_sizes_include_disjoint_card_and_context_inputs(self):
        for spec, count in ((ModelSpec(64, 2), 127172), (ModelSpec(128, 3), 673412)):
            network = QNetwork(spec)
            self.assertEqual(sum(p.numel() for p in network.parameters()), count)
            self.assertEqual(network.policy.card_embed.in_features, 22)
            self.assertEqual(network.policy.table_embed.in_features, 76)
            self.assertEqual(network(collate_observations([PappluEnv().reset(17)], self.device)).shape, (1, 54))


if __name__ == "__main__":
    unittest.main()
