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
        QNetwork, build_checkpoint, load_checkpoint, masked_q_values,
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
        self.assertEqual([len(obs.history) for obs in observations], [1, 21, 21, 21, 21, 21])

    def test_gradients_reach_history_cards_relations_and_action_queries(self):
        observations = [self.observation(players, turns=10) for players in (2, 3, 4)]
        batch = [(obs, 0, target) for obs, target in zip(observations, (1.0, 0.5, 0.0))]
        before = copy.deepcopy(self.network.state_dict())
        loss = train_batch(
            self.network, torch.optim.Adam(self.network.parameters(), lr=1e-4),
            torch.nn.MSELoss(), batch, self.device, 1.0,
        )
        self.assertGreater(loss, 0.0)
        for name in (
            "rank_embedding.weight", "suit_embedding.weight", "card_features.weight",
            "card_conv.weight", "card_attention.in_proj_weight", "initial_cards.weight",
            "event_features.weight", "event_cards.weight", "history_encoder.weight_ih_l0",
            "player_history_attention.in_proj_weight", "turn_relations.weight",
            "player_attention.in_proj_weight", "action_query.weight",
            "action_player_attention.in_proj_weight", "action_card_attention.in_proj_weight",
        ):
            with self.subTest(name=name):
                gradient = dict(self.network.named_parameters())[name].grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.0)
                self.assertFalse(torch.equal(before[name], self.network.state_dict()[name]))

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

    def test_model_has_no_stochastic_or_running_inference_state(self):
        obs = self.observation(3, 5)
        self.network.train()
        first = self.values(obs)
        self.network.eval()
        torch.testing.assert_close(first, self.values(obs), rtol=0, atol=0)
        self.assertFalse(any(isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d))
                             for module in self.network.modules()))

    def test_checkpoint_roundtrip_and_current_metadata(self):
        obs = self.observation()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "model.pt"
            save_checkpoint(str(path), self.network, obs.config, {"note": "fresh"})
            loaded, config, meta = load_checkpoint(str(path))
            self.assertEqual(config, obs.config)
            self.assertEqual(meta, {"note": "fresh"})
            torch.testing.assert_close(
                self.values(obs), loaded(collate_observations([obs], self.device)), rtol=0, atol=0,
            )
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload["architecture"], "dynamic_table")
            self.assertEqual(payload["checkpoint_version"], 3)
            self.assertEqual(payload["encoding_version"], 2)
            self.assertEqual(payload["model_spec"], {"width": 64, "heads": 4})
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
