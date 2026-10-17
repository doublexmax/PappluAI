import copy
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from src.game.environment import GameConfig, PappluEnv, discard_action, legal_action_mask
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.training.core import EpisodeReplay

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.arena import CheckpointPolicy
    from src.model.encoding import collate_observations
    from src.model.network import (
        QNetwork, build_checkpoint, expand_network_width, load_checkpoint, masked_q_values,
        save_checkpoint, select_action, select_greedy_action,
    )
    from src.model.spec import ModelSpec
    from src.training.optimization import train_batch


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDynamicNetwork(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.network = QNetwork()
        self.device = torch.device("cpu")

    def observation(self, players=4, turns=5):
        env = MultiplayerEnv(MatchConfig(
            GameConfig(cards_in_hand=3, required_sequences=1, max_turns=30), players,
        ))
        env.reset(19)
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            for _ in range(turns):
                env.step(0)
                env.step(discard_action(next(i for i, n in enumerate(env.observe().hand) if n)))
        return env.observe()

    def values(self, *observations):
        return self.network(collate_observations(observations, self.device))

    def test_one_network_scores_variable_tables_and_full_histories(self):
        observations = [PappluEnv().reset(17)] + [
            self.observation(players, turns=10) for players in (2, 3, 4, 5, 6)
        ]
        values = self.values(*observations)
        self.assertEqual(values.shape, (6, 54))
        self.assertTrue(torch.isfinite(values).all())

    def test_history_lengths_stay_on_cpu_with_a_non_cpu_default_device(self):
        observations = [self.observation(2, 1), self.observation(3, 2)]
        previous = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            batch = collate_observations(observations, self.device)
            self.assertEqual(batch.history_lengths.device.type, "cpu")
            self.assertEqual(batch.history_lengths.tolist(), [3, 5])
            values = self.network(batch)
            self.assertEqual(values.shape, (2, 54))
            self.assertTrue(torch.isfinite(values).all())
        finally:
            torch.set_default_device(previous)

    def test_training_reduces_selected_action_error_on_a_controlled_batch(self):
        observations = [self.observation(players, turns=10) for players in (2, 3, 4)]
        targets = torch.tensor([1.0, 0.5, 0.0])
        batch = [(obs, 0, float(target)) for obs, target in zip(observations, targets)]
        before = torch.nn.functional.mse_loss(self.values(*observations)[:, 0], targets).item()
        loss = train_batch(
            self.network, torch.optim.Adam(self.network.parameters(), lr=1e-4),
            torch.nn.MSELoss(), batch, self.device, 1.0,
        )
        self.assertAlmostEqual(loss, before)
        after = torch.nn.functional.mse_loss(self.values(*observations)[:, 0], targets).item()
        self.assertLess(after, before)

    def test_warm_start_training_generalizes_to_unseen_deals(self):
        config = GameConfig(
            num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3,
        )
        env = PappluEnv(config)
        transitions = []
        training_deals = set()
        for seed in range(64):
            obs = env.reset_warm_start(seed)
            training_deals.add((obs.hand, obs.joker))
            for face, count in enumerate(obs.hand):
                if count:
                    env.reset_warm_start(seed)
                    action = discard_action(face)
                    transitions.append((obs, action, env.step(action).last_reward))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        rng = random.Random(47)
        for _ in range(64):
            train_batch(
                self.network, optimizer, torch.nn.MSELoss(),
                rng.sample(transitions, 64), self.device, 1.0,
            )
        greedy_wins = 0
        random_wins = 0
        evaluation_rng = random.Random(600000)
        for seed in range(600000, 600100):
            obs = env.reset_warm_start(seed)
            self.assertNotIn((obs.hand, obs.joker), training_deals)
            greedy_wins += env.step(select_greedy_action(self.network, obs)).won
            obs = env.reset_warm_start(seed)
            action = select_action(self.network, obs, 1.0, evaluation_rng)
            random_wins += env.step(action).won
        self.assertGreaterEqual(greedy_wins, 80)
        self.assertGreater(greedy_wins, random_wins)

    def test_history_and_player_padding_do_not_change_values(self):
        short = self.observation(2, 1)
        long = self.observation(6, 20)
        alone = self.values(short)[0]
        together = self.values(short, long)[0]
        torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-6)
        batch = collate_observations([short, long], self.device)
        poisoned = replace(
            batch,
            players=batch.players.masked_fill(~batch.player_mask[:, :, None], 777.0),
            event_features=batch.event_features.masked_fill(~batch.history_mask[:, :, None], -777.0),
        )
        torch.testing.assert_close(self.network(batch), self.network(poisoned), rtol=1e-5, atol=1e-6)

    def test_early_history_and_chronology_affect_the_same_current_hand(self):
        obs = self.observation(4, 12)
        early = replace(obs.history[1], face=(obs.history[1].face + 1) % 52)
        changed = replace(obs, history=(obs.history[0], early, *obs.history[2:]))
        reversed_pair = replace(obs, history=(
            *obs.history[:3], obs.history[4], obs.history[3], *obs.history[5:],
        ))
        baseline = self.values(obs)
        self.assertGreater(float((baseline - self.values(changed)).abs().max()), 1e-7)
        self.assertGreater(float((baseline - self.values(reversed_pair)).abs().max()), 1e-7)
        initial = list(obs.history[0].initial_hand)
        present = next(i for i, n in enumerate(initial) if n)
        initial[present] -= 1
        initial[(present + 1) % 52] += 1
        changed_initial = replace(obs, history=(
            replace(obs.history[0], initial_hand=tuple(initial)), *obs.history[1:],
        ))
        self.assertGreater(float((baseline - self.values(changed_initial)).abs().max()), 1e-7)

    def test_actual_next_player_relationship_changes_scores(self):
        obs = self.observation(4, 4)
        changed = replace(obs, players=(
            replace(obs.players[0], next_player=2, previous_player=1), *obs.players[1:],
        ))
        self.assertGreater(float((self.values(obs) - self.values(changed)).abs().max()), 1e-7)

    def test_replay_and_inference_use_identical_information(self):
        obs = self.observation(4, 12)
        replay = EpisodeReplay(3)
        replay.append(((obs, 0, 0.5),))
        restored = EpisodeReplay(3)
        restored.load_state_dict(replay.state_dict())
        replay_obs = restored.sample(random.Random(4), 1)[0][0]
        self.assertEqual(replay_obs, obs)
        torch.testing.assert_close(self.values(obs), self.values(replay_obs), rtol=0, atol=0)

    def test_shared_checkpoint_policy_does_not_remember_other_seats_or_games(self):
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
            after = policy.act(first, random.Random(999))
            self.assertEqual(before, after)
            with torch.no_grad():
                values_after = policy.network(collate_observations([first], self.device))
            torch.testing.assert_close(values_before, values_after, rtol=0, atol=0)

    def test_mask_and_exploration_use_only_legal_actions(self):
        env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1))
        env.reset(12)
        obs = env.step(0)
        mask = legal_action_mask(obs)
        with torch.no_grad():
            self.network.score[-1].bias.fill_(-1e10)
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
        changed = replace(obs, hand=tuple(hand))
        delta = self.values(obs)[0] - self.values(changed)[0]
        self.assertGreater(float(delta.abs().max()), 1e-7)
        self.assertGreater(float(delta[2:].std()), 1e-7)

    def test_suit_permutation_relabels_action_values(self):
        obs = self.observation(4, 12)
        suits = (2, 0, 3, 1)

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
        torch.testing.assert_close(
            self.values(obs), self.values(changed)[:, actions], rtol=1e-5, atol=1e-6,
        )

    def test_model_has_no_stochastic_or_running_inference_state(self):
        obs = self.observation(3, 5)
        self.network.train()
        first = self.values(obs)
        self.network.eval()
        torch.testing.assert_close(first, self.values(obs), rtol=0, atol=0)

    def test_width_expansion_preserves_values_and_parameter_counts(self):
        observations = [
            PappluEnv().reset(41), PappluEnv().reset_warm_start(43),
            self.observation(2, 4), self.observation(4, 12), self.observation(6, 8),
        ]
        batch = collate_observations(observations, self.device)
        self.network.eval()
        with torch.no_grad():
            expected = self.network(batch)
        source_state = copy.deepcopy(self.network.state_dict())
        for width, count in ((128, 272_651), (192, 605_579)):
            with self.subTest(width=width):
                expanded = expand_network_width(self.network, ModelSpec(width, 8))
                self.assertEqual(sum(p.numel() for p in expanded.parameters()), count)
                self.assertFalse(expanded.training)
                with torch.no_grad():
                    actual = expanded(batch)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
                self.assertLess(float((actual - expected).abs().max()), 1e-5)
        for name, value in self.network.state_dict().items():
            torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)

    def test_expanded_copies_diverge_on_the_first_adam_update(self):
        batch = collate_observations(
            [PappluEnv().reset(41), self.observation(4, 12)], self.device,
        )
        for width in (128, 192):
            with self.subTest(width=width):
                expanded = expand_network_width(self.network, ModelSpec(width, 8))
                torch.testing.assert_close(
                    expanded.suit_conv1.weight[:64], expanded.suit_conv1.weight[64:128],
                    rtol=0, atol=0,
                )
                optimizer = torch.optim.Adam(expanded.parameters(), lr=1e-3)
                target = torch.linspace(-0.7, 0.7, 54)
                loss = (expanded(batch) - target).square().mean()
                loss.backward()
                head = width // 8
                padded = [
                    seat * head + dimension
                    for seat in range(8) for dimension in range(head)
                    if seat >= 4 or dimension >= 16
                ]
                gradient = expanded.card_attention.out_proj.weight.grad[:, padded]
                self.assertGreater(float(gradient.abs().max()), 1e-8)
                if width == 192:
                    gradient = expanded.card_attention.in_proj_weight.grad[
                        width + torch.tensor(padded)
                    ]
                    self.assertGreater(float(gradient.abs().max()), 1e-10)
                optimizer.step()
                divergence = (
                    expanded.suit_conv1.weight[:64] - expanded.suit_conv1.weight[64:128]
                ).detach().abs().max()
                self.assertGreater(float(divergence), 1e-7)

    def test_width_expansion_is_seeded_and_does_not_advance_global_rng(self):
        state = torch.get_rng_state().clone()
        first = expand_network_width(self.network, ModelSpec(128, 8), seed=17)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        second = expand_network_width(self.network, ModelSpec(128, 8), seed=17)
        third = expand_network_width(self.network, ModelSpec(128, 8), seed=19)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
        self.assertGreater(
            float((first.suit_conv2.weight - third.suit_conv2.weight).detach().abs().max()), 1e-5,
        )

    def test_identity_expansion_is_an_independent_copy_and_invalid_specs_fail(self):
        expanded = expand_network_width(self.network, self.network.spec)
        obs = self.observation()
        batch = collate_observations([obs], self.device)
        with torch.no_grad():
            expected = self.network(batch).clone()
            torch.testing.assert_close(expanded(batch), expected, rtol=0, atol=0)
            expanded.score[-1].bias.add_(1.0)
            torch.testing.assert_close(self.network(batch), expected, rtol=0, atol=0)
        for spec in (ModelSpec(32, 4), ModelSpec(128, 2), ModelSpec(128, 16), ModelSpec(96, 4)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                expand_network_width(self.network, spec)

    def test_checkpoint_roundtrip_preserves_nondefault_spec_and_wire_format(self):
        self.network = QNetwork(ModelSpec(width=32, heads=2))
        obs = self.observation()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "model.pt"
            save_checkpoint(str(path), self.network, obs.config, {"note": "fresh"})
            loaded, config, meta = load_checkpoint(str(path))
            self.assertEqual(config, obs.config)
            self.assertEqual(meta, {"note": "fresh"})
            self.assertEqual(loaded.spec, self.network.spec)
            torch.testing.assert_close(
                self.values(obs), loaded(collate_observations([obs], self.device)), rtol=0, atol=0,
            )
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["architecture"], "dynamic_table")
            self.assertEqual(payload["checkpoint_version"], 3)
            self.assertEqual(payload["encoding_version"], 2)
            self.assertEqual(payload["model_spec"], {"width": 32, "heads": 2})
            self.assertNotIn("state_dim", payload)

    def test_bad_or_old_checkpoints_do_not_mutate_supplied_network(self):
        baseline = copy.deepcopy(self.network.state_dict())
        changes = [
            {"checkpoint_version": 1}, {"checkpoint_version": 2},
            {"encoding_version": 1}, {"architecture": "mlp"}, {"state_dim": 164},
            {"model_spec": {"width": 32, "heads": 4}}, {"num_actions": 12},
            {"model_spec": {"width": 64, "heads": 3}}, {"meta": []},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pt"
            for change in changes:
                payload = build_checkpoint(QNetwork(), GameConfig())
                payload.update(change)
                torch.save(payload, path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_checkpoint(str(path), network=self.network)
                for name, value in self.network.state_dict().items():
                    self.assertTrue(torch.equal(baseline[name], value), name)
            for defect in ("shape", "nan", "dtype"):
                payload = build_checkpoint(QNetwork(), GameConfig())
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


if __name__ == "__main__":
    unittest.main()
