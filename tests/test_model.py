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
from src.model.encoding import encode
from src.model.spec import ModelSpec, ModelSpecError

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import CheckpointPolicy
    from src.model.network import (
        CardBlock, QNetwork, checkpoint_payload, collate, collate_observations, greedy_actions,
        load_checkpoint, save_checkpoint,
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
            self.assertEqual(batch.card.shape, (2, 52, 14))
            self.assertEqual(batch.seat.shape, (2, 6, 9))
            self.assertEqual(batch.context.shape, (2, 17))
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

    def test_greedy_masks_illegal_values_and_rejects_terminal_rows(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        env.reset(12)
        obs = replace(env.step(0), hand=(1, 1, 1, 1) + (0,) * 48)
        batch = collate_observations([obs], self.device)
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.zero_()
            self.network.policy.value_out.bias.fill_(-1e10)
            self.network.policy.face_out.bias.fill_(-1e10)
        values = self.network(batch)
        self.assertEqual(values[0, 0].item(), 0.0)
        torch.testing.assert_close(values[batch.legal.bool()], torch.full((4,), -1e10), rtol=0, atol=0)
        self.assertEqual(greedy_actions(self.network, batch).tolist(), [2])
        terminal = replace(obs, phase=type(obs.phase).TERMINAL)
        with self.assertRaisesRegex(ValueError, "no legal actions"):
            greedy_actions(self.network, collate_observations([obs, terminal], self.device))

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
        batch = collate_observations(observations, self.device)
        selected = set()
        for seed in range(8):
            torch.manual_seed(100 + seed)
            net = QNetwork()
            net.eval()
            selected.update(greedy_actions(net, batch).tolist())
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
            self.assertEqual(payload["checkpoint_version"], 6)
            self.assertEqual(payload["encoding_version"], 6)
            self.assertEqual(payload["model_spec"], {"width": 32, "blocks": 3})
            self.assertNotIn("state_dim", payload)

    def test_bad_or_old_checkpoints_do_not_mutate_supplied_network(self):
        baseline = copy.deepcopy(self.network.state_dict())
        changes = [
            {"checkpoint_version": 1}, {"checkpoint_version": 2}, {"checkpoint_version": 3},
            {"checkpoint_version": 4}, {"checkpoint_version": 5},
            {"checkpoint_version": True}, {"checkpoint_version": 6.0},
            {"encoding_version": 1}, {"encoding_version": 2}, {"encoding_version": 3},
            {"encoding_version": 4}, {"encoding_version": 5}, {"encoding_version": 6.0},
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

    def test_card_block_multiplies_the_silu_gate_by_the_second_projection(self):
        block = CardBlock(8)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            block.card_ffn.bias[:8].fill_(1)
            block.card_ffn.bias[8:].fill_(2)
            block.card_out.weight[0, 0] = 1
        cards, table = block(torch.zeros(1, 4, 13, 8), torch.zeros(1, 8))
        torch.testing.assert_close(cards[..., 0], torch.full((1, 4, 13), 1.4621172), rtol=0, atol=1e-7)
        torch.testing.assert_close(cards[..., 1:], torch.zeros(1, 4, 13, 7), rtol=0, atol=0)
        torch.testing.assert_close(table, torch.zeros(1, 8), rtol=0, atol=0)

    def test_model_sizes_include_only_observable_card_and_context_inputs(self):
        for spec, count in ((ModelSpec(), 342804), (ModelSpec(128, 3), 867972)):
            network = QNetwork(spec)
            self.assertEqual(sum(p.numel() for p in network.parameters()), count)
            self.assertEqual(network.policy.card_embed.in_features, 14)
            self.assertEqual(network.policy.table_embed.in_features, 71)
            self.assertEqual(network(collate_observations([PappluEnv().reset(17)], self.device)).shape, (1, 54))


if __name__ == "__main__":
    unittest.main()
