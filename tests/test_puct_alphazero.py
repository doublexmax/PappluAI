from __future__ import annotations

from dataclasses import replace
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import tokenize
import ast
import unittest

import torch

from src.checkpoints.io import file_sha256
from src.evaluation.arena import play_match
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation, Phase,
    discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationEvent, PlayerView
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.model.encoding import collate_observations
from src.model.network import QNetwork, load_checkpoint, save_checkpoint
from src.model.policy_value import META_KEY, PolicyValueNet
from src.model.puct import PUCTConfig, PUCTPolicy, puct_search
from src.model.search import _after_draw
from src.model.spec import ModelSpec
from src.training.alphazero import (
    AlphaZeroConfig, AlphaZeroReplay, AlphaZeroSample, AlphaZeroSession,
    train_alphazero_batch,
)
from src.training.core import TerminalShaping


ROOT = Path(__file__).resolve().parents[1]


def draw_position(faces=(0, 15, 30), top=44, turns=3):
    game = GameConfig(cards_in_hand=len(faces), required_sequences=1, max_turns=turns)
    hand = tuple(faces.count(face) for face in range(52))
    stock = game.num_decks * 52 - len(faces) * 2 - 2
    players = (
        PlayerView(len(faces), turns, 1, 1, True),
        PlayerView(len(faces), turns, 0, 0, True),
    )
    initial = InformationEvent(
        EventKind.INITIAL, 0, len(faces), turns, stock, 1,
        offered_top=top, initial_hand=hand,
    )
    return Observation(hand, (top,), 51, Phase.DRAW, turns, stock, game, players, (initial,))


def constant_model(game, draw=0.0, discard=0.0):
    network = QNetwork(ModelSpec(8, 2))
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.draw_head[2].bias[0] = draw
        network.score[2].bias[0] = discard
    return PolicyValueNet(network, game)


def small_search(**settings):
    return PUCTConfig(**{
        "simulations": 4, "max_stock_outcomes": 2, "opponent_outcomes": 2,
        "meld_weight": 0.0, "outs_weight": 0.0, "feed_weight": 0.0,
        **settings,
    })


class AfterstateNetwork(QNetwork):
    def __init__(self):
        super().__init__(ModelSpec(8, 2))

    def forward(self, batch):
        q = torch.zeros((batch.cards.shape[0], 54), device=batch.cards.device)
        q[:, 0] = batch.cards[:, 0, 0] * 3
        q[:, discard_action(0)] = 0.8
        return q


class DepthNetwork(QNetwork):
    def __init__(self):
        super().__init__(ModelSpec(8, 2))

    def forward(self, batch):
        depth = (
            (batch.event_kind == int(EventKind.DISCARD))
            & (batch.event_actor == 0) & batch.history_mask
        ).sum(dim=1)
        return (depth[:, None].float() * 0.25).expand(-1, 54)


class NonnegativeStockNetwork(QNetwork):
    def forward(self, batch):
        if bool((batch.context[:, 4] < 0).any()):
            raise AssertionError("search evaluated negative stock context")
        if bool(((batch.event_features[:, :, 5] < 0) & batch.history_mask).any()):
            raise AssertionError("search evaluated negative stock history")
        return super().forward(batch)


class DecisionOnlyPolicyValueNet(PolicyValueNet):
    def forward(self, batch, mask):
        nondecision = (
            (batch.context[:, 0] > 0.5) & (batch.context[:, 4] == 0) & (batch.top == 52)
        )
        if bool(nondecision.any()):
            raise AssertionError("a value-only afterstate entered the policy evaluator")
        return super().forward(batch, mask)


class TorchTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)


class TestPolicyValue(TorchTest):
    def test_draw_and_discard_values_and_mask_support(self):
        draw = draw_position()
        discard = _after_draw(draw, 44, EventKind.DRAW_STOCK)
        model = constant_model(draw.config, draw=0.2, discard=0.8)
        states = collate_observations((draw, discard), torch.device("cpu"))
        mask = torch.tensor([legal_action_mask(draw), legal_action_mask(discard)])
        out = model(states, mask)
        self.assertEqual(tuple(out.q.shape), (2, 54))
        self.assertEqual(tuple(out.position_value.shape), (2,))
        self.assertTrue(torch.isneginf(out.log_policy[~mask]).all())
        torch.testing.assert_close(out.log_policy.exp().sum(dim=1), torch.ones(2))
        torch.testing.assert_close(out.temperature, torch.tensor([0.05, 0.05]))
        torch.testing.assert_close(out.position_value, torch.tensor([0.2, 0.5]))
        evaluated = model.evaluate_positions((draw, discard))
        self.assertEqual(evaluated[0].actions, (0, 1))
        self.assertEqual(evaluated[1].actions, (2, 17, 32, 46))
        self.assertAlmostEqual(evaluated[0].position_value, 0.2)
        self.assertAlmostEqual(evaluated[1].position_value, 0.5)
        self.assertEqual(evaluated[0].prior, (0.5, 0.5))
        self.assertEqual(evaluated[1].prior, (0.25, 0.25, 0.25, 0.25))
        self.assertEqual(model.evaluate_position(draw), evaluated[0])
        self.assertEqual(model.evaluate_positions(()), ())

    def test_mask_validation(self):
        obs = draw_position()
        model = constant_model(obs.config)
        states = collate_observations((obs,), torch.device("cpu"))
        for mask in (
            torch.ones((1, 54), dtype=torch.long),
            torch.ones((2, 54), dtype=torch.bool),
            torch.ones((1, 53), dtype=torch.bool),
            torch.zeros((1, 54), dtype=torch.bool),
        ):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                model(states, mask)
        with self.assertRaises(ValueError):
            model.evaluate_position(replace(obs, phase=Phase.TERMINAL))

    def test_standard_checkpoint_warm_start_and_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "q.pt")
            output = str(Path(root) / "pv.pt")
            game = GameConfig(max_turns=2)
            network = QNetwork(ModelSpec(8, 2))
            save_checkpoint(path, network, game, {"origin": "warm_start"})
            model = PolicyValueNet.from_checkpoint(path)
            env = MultiplayerEnv(MatchConfig(game, 2))
            env.reset(41)
            draw = env.observe()
            env.step(ACTION_DRAW_STOCK)
            discard = env.observe()
            states = collate_observations((draw, discard), torch.device("cpu"))
            raw = network(states).detach()
            evaluated = model.evaluate_positions((draw, discard))
            self.assertAlmostEqual(evaluated[0].position_value, float(raw[0, 0]), places=6)
            self.assertEqual(sum(draw.hand), 21)
            self.assertEqual(sum(discard.hand), 22)
            for result, row in zip(evaluated, raw):
                self.assertAlmostEqual(sum(result.prior), 1.0, places=7)
                for action, score in zip(result.actions, result.q_values):
                    self.assertAlmostEqual(score, float(row[action]), places=6)
            model = PolicyValueNet(
                model.qnet, game, draw_temperature=0.08, discard_temperature=0.12,
                gamma=0.9, shaping=TerminalShaping(speed=0.1, margin=0.2),
            )
            digest = model.save(output, {"origin": "test"})
            self.assertEqual(digest, file_sha256(Path(output)))
            restored = PolicyValueNet.from_checkpoint(output)
            raw_model, raw_game, meta = load_checkpoint(output)
            self.assertEqual(raw_model.architecture, "dynamic_table")
            self.assertEqual(raw_game, game)
            self.assertEqual(meta["origin"], "test")
            self.assertAlmostEqual(meta[META_KEY]["draw_temperature"], 0.08)
            self.assertAlmostEqual(meta[META_KEY]["discard_temperature"], 0.12)
            self.assertEqual(restored.gamma, 0.9)
            self.assertEqual(restored.shaping, TerminalShaping(speed=0.1, margin=0.2))
            torch.testing.assert_close(model.log_temperature, restored.log_temperature)
            for before, after in zip(
                model.evaluate_positions((draw, discard)),
                restored.evaluate_positions((draw, discard)),
            ):
                self.assertAlmostEqual(before.position_value, after.position_value, places=7)
                torch.testing.assert_close(torch.tensor(before.prior), torch.tensor(after.prior))

    @unittest.skipUnless(
        (ROOT / "checkpoints" / "champion-2p-dynamic-table.pt").exists(),
        "champion checkpoint is not present",
    )
    def test_champion_warm_start(self):
        model = PolicyValueNet.from_checkpoint(
            str(ROOT / "checkpoints" / "champion-2p-dynamic-table.pt"),
        )
        env = MultiplayerEnv(MatchConfig(model.game_config, 2))
        env.reset(17)
        obs = env.observe()
        evaluated = model.evaluate_position(obs)
        with torch.no_grad():
            expected = model.qnet(collate_observations((obs,), torch.device("cpu")))[0, 0]
        self.assertEqual(sum(obs.hand), 21)
        self.assertAlmostEqual(evaluated.position_value, float(expected), places=6)
        self.assertEqual(evaluated.actions, (0, 1))
        self.assertAlmostEqual(sum(evaluated.prior), 1.0, places=7)


class TestPUCT(TorchTest):
    def test_forced_and_terminal_positions(self):
        obs = draw_position()
        forced = replace(obs, discard_pile=())
        result = puct_search(QNetwork(ModelSpec(8, 2)), forced)
        self.assertEqual(result.action, 0)
        self.assertEqual(result.policy_target, ((0, 1.0),))
        self.assertEqual(result.visits, ((0, 0),))
        self.assertEqual(result.rows, 0)
        self.assertEqual(result.reason, "forced")
        with self.assertRaises(ValueError):
            puct_search(QNetwork(), replace(obs, phase=Phase.TERMINAL))

    def test_exact_draw_and_discard_wins(self):
        obs = draw_position((0, 1, 30), top=2)
        network = QNetwork(ModelSpec(8, 2))
        draw = puct_search(network, obs)
        discard = puct_search(network, _after_draw(obs, 2, EventKind.TAKE_DISCARD))
        for result, action in ((draw, 1), (discard, 32)):
            self.assertEqual(result.reason, "exact_win")
            self.assertEqual(result.action, action)
            self.assertEqual(result.root_value, 1.0)
            self.assertEqual(result.rows, 0)
            self.assertEqual(sum(probability for _, probability in result.policy_target), 1.0)
            self.assertEqual(dict(result.policy_target)[action], 1.0)
            self.assertTrue(all(visits == 0 for _, visits in result.visits))

    def test_discard_search_uses_21_card_afterstate_value(self):
        obs = _after_draw(draw_position(), 44, EventKind.DRAW_STOCK)
        model = PolicyValueNet(AfterstateNetwork(), obs.config)
        direct = puct_search(model, obs, small_search(simulations=0, afterstate_weight=0))
        position = puct_search(model, obs, small_search(simulations=0, afterstate_weight=1))
        self.assertEqual(direct.action, 2)
        self.assertEqual(position.action, 17)
        self.assertEqual(position.rows, 5)
        self.assertEqual(tuple(action for action, _ in position.policy_target), (2, 17, 32, 46))
        self.assertTrue(all(probability > 0 for _, probability in position.policy_target))
        self.assertAlmostEqual(sum(probability for _, probability in position.policy_target), 1.0)

    def test_raw_q_network_draw_search_uses_afterstate_expectations(self):
        obs = draw_position()
        model = constant_model(obs.config, draw=0.2, discard=0.8)
        result = puct_search(model.qnet, obs, small_search(simulations=0))
        self.assertAlmostEqual(result.root_value, 0.68, places=6)
        self.assertEqual(result.visits, ((0, 0), (1, 0)))
        self.assertEqual(result.policy_target, ((0, 0.5), (1, 0.5)))

    def test_completed_policy_uses_discounted_backup_not_visit_frequencies(self):
        obs = draw_position()
        base = constant_model(obs.config, draw=0.2, discard=0.8)
        model = PolicyValueNet(base.qnet, obs.config, gamma=0.9)
        result = puct_search(model, obs, small_search(simulations=1))
        self.assertEqual(result.visits, ((0, 1), (1, 0)))
        self.assertAlmostEqual(dict(result.policy_target)[0], 0.091122961, places=6)
        self.assertAlmostEqual(dict(result.policy_target)[1], 0.908877039, places=6)
        self.assertAlmostEqual(result.root_value, 0.669520859, places=6)

    def test_completed_q_backup_and_visits_are_deterministic(self):
        obs = _after_draw(draw_position(), 44, EventKind.DRAW_STOCK)
        model = PolicyValueNet(DepthNetwork(), obs.config)
        initial = puct_search(model, obs, small_search(simulations=0))
        config = small_search(simulations=16)
        result = puct_search(model, obs, config, random.Random(1))
        repeated = puct_search(model, obs, config, random.Random(999))
        self.assertEqual(result, repeated)
        self.assertAlmostEqual(initial.root_value, 0.05, places=7)
        self.assertGreater(result.root_value, 0.1)
        self.assertEqual(sum(visits for _, visits in result.visits), 16)
        self.assertGreater(result.rows, initial.rows)
        self.assertEqual(result.reason, "puct_search")
        self.assertAlmostEqual(sum(probability for _, probability in result.policy_target), 1.0)
        self.assertTrue(all(probability > 0 for _, probability in result.policy_target))

    def test_taken_card_is_not_immediately_rediscarded(self):
        obs = _after_draw(draw_position(), 44, EventKind.TAKE_DISCARD)
        result = puct_search(constant_model(obs.config), obs, small_search())
        self.assertNotEqual(result.action, 46)
        self.assertEqual(dict(result.visits)[46], 0)
        self.assertGreater(dict(result.policy_target)[46], 0)
        self.assertEqual(len(result.policy_target), 4)

    def test_depth_search_does_not_draw_from_exhausted_stock(self):
        game = GameConfig(num_decks=1, max_turns=8, recycle_discard=False)
        env = MultiplayerEnv(MatchConfig(game, 2))
        env.reset(41)
        while env.view().stock_remaining:
            env.step(ACTION_DRAW_STOCK)
            if not env.view().stock_remaining:
                break
            obs = env.observe()
            env.step(discard_action(next(face for face, count in enumerate(obs.hand) if count)))
        obs = env.observe()
        self.assertEqual(obs.stock_remaining, 0)
        self.assertEqual(obs.phase, Phase.DISCARD)
        model = PolicyValueNet(NonnegativeStockNetwork(ModelSpec(8, 2)), game)
        result = puct_search(model, obs, small_search(
            simulations=8, max_stock_outcomes=1, opponent_outcomes=1,
        ))
        self.assertEqual(sum(visits for _, visits in result.visits), 8)
        self.assertTrue(legal_action_mask(obs)[result.action])
        self.assertTrue(math.isfinite(result.root_value))
        self.assertAlmostEqual(sum(probability for _, probability in result.policy_target), 1.0)

    def test_value_only_afterstates_do_not_invent_legal_draws(self):
        game = GameConfig(num_decks=1, cards_in_hand=25, max_turns=2, recycle_discard=False)
        env = MultiplayerEnv(MatchConfig(game, 2))
        env.reset(41)
        env.step(ACTION_TAKE_DISCARD)
        obs = env.observe()
        self.assertEqual(obs.stock_remaining, 0)
        self.assertEqual(obs.discard_pile, ())
        model = DecisionOnlyPolicyValueNet(QNetwork(ModelSpec(8, 2)), game)
        result = puct_search(model, obs, small_search(
            simulations=2, max_stock_outcomes=1, opponent_outcomes=1,
        ))
        self.assertEqual(result.reason, "puct_search")
        self.assertEqual(sum(visits for _, visits in result.visits), 2)
        self.assertTrue(legal_action_mask(obs)[result.action])
        self.assertGreater(result.rows, 1)

    def test_exploration_samples_and_is_reproducible(self):
        obs = draw_position()
        model = constant_model(obs.config)
        config = small_search(simulations=0, temperature=1.0)
        self.assertEqual(
            puct_search(model, obs, config, random.Random(7)),
            puct_search(model, obs, config, random.Random(7)),
        )
        actions = {
            puct_search(model, obs, config, random.Random(seed)).action
            for seed in range(12)
        }
        self.assertEqual(actions, {0, 1})

    def test_policy_completes_two_player_match(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=2)
        model = PolicyValueNet(QNetwork(ModelSpec(8, 2)), game)
        policy = PUCTPolicy(model, config=small_search(simulations=2))
        result = play_match(
            (policy, policy), MatchConfig(game, 2), 41,
            record_trajectories=True, record_terminal=True,
        )
        self.assertEqual(result.terminal_reason, "turns_exhausted")
        self.assertEqual(result.action_count, 8)
        self.assertEqual(result.seat_turns, (2, 2))
        self.assertEqual(sum(len(steps) for steps in result.trajectories), 8)
        self.assertIsNotNone(result.terminal_snapshot)
        for steps in result.trajectories:
            for obs, action, _ in steps:
                self.assertTrue(legal_action_mask(obs)[action])

    def test_config_rejects_invalid_values(self):
        for settings in (
            {"simulations": True}, {"simulations": -1}, {"max_stock_outcomes": 0},
            {"opponent_outcomes": -1}, {"c_puct": float("nan")},
            {"dirichlet_alpha": 0}, {"dirichlet_eps": 1.1},
            {"afterstate_weight": 1.1}, {"temperature": -1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PUCTConfig(**settings)


class TestAlphaZeroTraining(TorchTest):
    def test_loss_uses_blended_position_target(self):
        obs = draw_position()
        model = constant_model(obs.config)
        sample = AlphaZeroSample(obs, 0, ((0, 0.5), (1, 0.5)), 0.0, 1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loss = train_alphazero_batch(model, optimizer, (sample,), bootstrap_blend=0.25)
        self.assertAlmostEqual(loss.value_loss, 0.28125, places=7)
        self.assertAlmostEqual(loss.q_loss, 0.28125, places=7)
        self.assertAlmostEqual(loss.policy_loss, 0.05 * math.log(2), places=7)
        self.assertAlmostEqual(loss.total_loss, 0.456532359, places=7)

    def test_training_updates_21_card_value_and_policy_and_reduces_losses(self):
        game = GameConfig(max_turns=2)
        env = MultiplayerEnv(MatchConfig(game, 2))
        env.reset(41)
        obs = env.observe()
        model = PolicyValueNet(QNetwork(ModelSpec(16, 2)), game)
        before = model.evaluate_position(obs)
        draw_head_before = model.qnet.draw_head[2].bias.detach().clone()
        batch = (AlphaZeroSample(obs, 1, ((0, 0.1), (1, 0.9)), 0.8, 0.8),) * 4
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        losses = [train_alphazero_batch(model, optimizer, batch) for _ in range(80)]
        after = model.evaluate_position(obs)
        self.assertEqual(sum(obs.hand), 21)
        self.assertGreater(after.position_value, 0.7)
        self.assertGreater(after.position_value, before.position_value + 0.7)
        self.assertGreater(after.prior[1], 0.85)
        self.assertGreater(after.prior[1], before.prior[1])
        self.assertFalse(torch.equal(draw_head_before, model.qnet.draw_head[2].bias))
        for name in ("total_loss", "value_loss", "policy_loss"):
            self.assertLess(getattr(losses[-1], name), getattr(losses[0], name))
        self.assertLess(losses[-1].value_loss, 0.001)
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_replay_is_bounded_and_round_trips_observations(self):
        obs = draw_position()
        samples = tuple(
            AlphaZeroSample(obs, 0, ((0, 1.0),), 0.25, float(index))
            for index in range(5)
        )
        replay = AlphaZeroReplay(3)
        replay.append_samples(samples)
        self.assertEqual(len(replay), 3)
        self.assertEqual(
            tuple(item["outcome_value"] for item in replay.state_dict()["samples"]),
            (2.0, 3.0, 4.0),
        )
        restored = AlphaZeroReplay(3)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(
            replay.sample(random.Random(7), 8), restored.sample(random.Random(7), 8),
        )
        state = replay.state_dict()
        broken = {**state, "samples": ({**state["samples"][0], "policy_target": ((0, 0.2),)},)}
        with self.assertRaises(ValueError):
            restored.load_state_dict(broken)
        self.assertEqual(restored.state_dict(), state)
        with self.assertRaises(ValueError):
            AlphaZeroReplay(3).sample(random.Random(0), 1)

    def test_sample_rejects_illegal_and_unnormalized_targets(self):
        obs = draw_position()
        for action, target, value in (
            (2, ((0, 1.0),), 0.0), (0, ((2, 1.0),), 0.0),
            (0, ((0, 0.5),), 0.0), (0, ((0, 0.5), (0, 0.5)), 0.0),
            (0, ((0, 1.0),), float("nan")),
        ):
            with self.subTest(action=action, target=target), self.assertRaises(ValueError):
                AlphaZeroSample(obs, action, target, value, 0.0)

    def test_session_training_export_and_resume(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=2)
        config = AlphaZeroConfig(
            game=game, search=small_search(simulations=2, temperature=1),
            batch_size=4, replay_capacity=64, updates_per_match=2,
            gamma=0.9, shaping=TerminalShaping(margin=0.4),
        )
        model = PolicyValueNet(QNetwork(ModelSpec(8, 2)), game)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        session = AlphaZeroSession(model, config)
        summary = session.train_match(seed=41)
        self.assertEqual(summary["matches"], 1)
        self.assertEqual(summary["train_updates"], 2)
        self.assertEqual(summary["samples"], 8)
        self.assertAlmostEqual(summary["bootstrap_blend"], 0.65)
        self.assertEqual(len(session.replay), 8)
        self.assertTrue(any(
            not torch.equal(before[name], tensor) for name, tensor in model.state_dict().items()
        ))
        for trajectory in (session.last_samples[:4], session.last_samples[4:]):
            self.assertGreater(trajectory[-1].outcome_value, 0)
            for first, following in zip(trajectory, trajectory[1:]):
                self.assertAlmostEqual(first.outcome_value, 0.9 * following.outcome_value)
        with tempfile.TemporaryDirectory() as root:
            exported = str(Path(root) / "model.pt")
            checkpoint = str(Path(root) / "session.pt")
            self.assertEqual(session.export_model(exported), file_sha256(Path(exported)))
            network, loaded_game, meta = load_checkpoint(exported)
            self.assertEqual(network.architecture, "dynamic_table")
            self.assertEqual(loaded_game, game)
            self.assertEqual(meta["algorithm"], "alphazero_puct")
            self.assertEqual(meta["completed_matches"], 1)
            policy = PUCTPolicy.from_checkpoint(exported, config=small_search(simulations=0))
            obs = session.last_samples[0].observation
            self.assertTrue(legal_action_mask(obs)[policy.act(obs, random.Random(0))])
            self.assertEqual(session.save_checkpoint(checkpoint), file_sha256(Path(checkpoint)))
            restored = AlphaZeroSession.from_checkpoint(checkpoint)
            self.assertEqual(restored.config, config)
            self.assertEqual(restored.replay.state_dict(), session.replay.state_dict())
            expected = session.train_match()
            resumed = restored.train_match()
            self.assertEqual(resumed, expected)
            for name, tensor in session.model.state_dict().items():
                torch.testing.assert_close(tensor, restored.model.state_dict()[name], rtol=0, atol=0)

    def test_frozen_opponent_is_not_trained(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=1)
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "opponent.pt")
            constant_model(game).save(path)
            config = AlphaZeroConfig(
                game=game, search=small_search(simulations=1, temperature=1),
                batch_size=2, updates_per_match=1, opponent_paths=(path,),
            )
            session = AlphaZeroSession(
                PolicyValueNet(QNetwork(ModelSpec(8, 2)), game), config,
            )
            opponent = session.opponents[0].network
            before = {name: tensor.clone() for name, tensor in opponent.state_dict().items()}
            summary = session.train_match(seed=41)
            self.assertEqual(summary["samples"], 2)
            self.assertFalse(any(parameter.requires_grad for parameter in opponent.parameters()))
            for name, tensor in opponent.state_dict().items():
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_dynamic_bootstrap_blend_schedule(self):
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=1)
        config = AlphaZeroConfig(
            game=game, search=small_search(simulations=0),
            updates_per_match=0, bootstrap_blend=0.65,
            bootstrap_blend_end=0.15, bootstrap_decay_matches=2,
        )
        session = AlphaZeroSession(PolicyValueNet(QNetwork(ModelSpec(8, 2)), game), config)
        self.assertAlmostEqual(session.current_bootstrap_blend, 0.65)
        first = session.train_match(seed=41)
        self.assertAlmostEqual(first["bootstrap_blend"], 0.65)
        self.assertAlmostEqual(session.current_bootstrap_blend, 0.40)
        second = session.train_match(seed=42)
        self.assertAlmostEqual(second["bootstrap_blend"], 0.40)
        self.assertAlmostEqual(session.current_bootstrap_blend, 0.15)
        third = session.train_match(seed=43)
        self.assertAlmostEqual(third["bootstrap_blend"], 0.15)
        fixed = AlphaZeroConfig(game=game, bootstrap_blend=0.65, bootstrap_blend_end=None)
        self.assertAlmostEqual(fixed.bootstrap_blend_at(50), 0.65)


class TestAlphaZeroCLI(unittest.TestCase):
    def test_help_parses_without_torch(self):
        script = (
            "import builtins,runpy,sys\n"
            "original=builtins.__import__\n"
            "def blocked(name,*args,**kwargs):\n"
            " if name=='torch' or name.startswith('torch.'):\n"
            "  raise ModuleNotFoundError('blocked torch',name='torch')\n"
            " return original(name,*args,**kwargs)\n"
            "builtins.__import__=blocked\n"
            "sys.argv=['src.cli.alphazero','--help']\n"
            "runpy.run_module('src.cli.alphazero',run_name='__main__')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--matches", result.stdout)
        self.assertIn("--resume", result.stdout)
        self.assertIn("policy/value", result.stdout)

    def test_cli_trains_exports_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = str(Path(root) / "session.pt")
            exported = str(Path(root) / "model.pt")
            command = [
                sys.executable, "-m", "src.cli.alphazero", "--matches", "1",
                "--cards-in-hand", "3", "--required-sequences", "1", "--max-turns", "1",
                "--simulations", "1", "--stock-outcomes", "1", "--opponent-outcomes", "1",
                "--batch-size", "2", "--updates-per-match", "1",
                "--checkpoint", checkpoint, "--export", exported,
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(summary["matches"], 1)
            self.assertEqual(summary["train_updates"], 1)
            self.assertEqual(summary["export_sha256"], file_sha256(Path(exported)))
            network, config, meta = load_checkpoint(exported)
            self.assertEqual(network.architecture, "dynamic_table")
            self.assertEqual(config.cards_in_hand, 3)
            self.assertEqual(meta["algorithm"], "alphazero_puct")
            resumed = subprocess.run(
                [sys.executable, "-m", "src.cli.alphazero", "--resume", checkpoint,
                 "--matches", "1", "--export", exported],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            summary = json.loads(resumed.stdout.splitlines()[-1])
            self.assertEqual(summary["matches"], 2)
            self.assertEqual(summary["train_updates"], 2)
            self.assertEqual(summary["export_sha256"], file_sha256(Path(exported)))

    def test_added_modules_have_no_comments_or_docstrings(self):
        for relative in (
            "src/model/policy_value.py", "src/model/puct.py",
            "src/training/alphazero.py", "src/cli/alphazero.py",
            "tests/test_puct_alphazero.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            comments = [
                token for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            tree = ast.parse(source)
            docstrings = [
                ast.get_docstring(node)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and ast.get_docstring(node) is not None
            ]
            self.assertEqual(comments, [], relative)
            self.assertEqual(docstrings, [], relative)


if __name__ == "__main__":
    unittest.main()
