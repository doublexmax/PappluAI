import copy
from dataclasses import replace
import io
import random
import unittest
from unittest import mock

import torch

from src.evaluation.arena import (
    Deal, EpsilonPolicy, GreedyPolicy, RandomPolicy, play_match, run_matches,
)
from src.game.environment import GameConfig, PappluEnv, discard_action
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.game.outcome import SeatOutcome, solo_outcome
from src.model.encoding import encode, legal_actions
from src.model.network import NetworkOutputs, QNetwork
from src.model.search import TeacherConfig
from src.model.spec import ModelSpec
from src.training.core import Decision, Sample, Trajectory, TrajectoryReplay
from src.training.league import OpponentScheduler, PfspConfig
from src.training.optimization import Learner, LearnerConfig, LossWeights


def decision(seed=19, players=1):
    game = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=20)
    if players == 1:
        obs = PappluEnv(game).reset(seed)
        hands = (obs.hand,) + ((0,) * 52,) * 5
    else:
        env = MultiplayerEnv(MatchConfig(game, players))
        env.reset(seed)
        obs, hands = env.observe(), env.private_hands(0)
    return Decision(obs, encode(obs), 0, hands)


class TableNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(54))

    def forward(self, batch):
        return self.values[None].expand(batch.card.shape[0], -1)

    def outputs(self, batch):
        q = self(batch)
        zero = q * 0
        return NetworkOutputs(q, zero, zero, zero[:, :1, None].expand(-1, 6, 52), zero[:, 0])


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class TestTrajectoryReplay(unittest.TestCase):
    def trajectory(self, length, seed=19, players=1):
        return Trajectory(players, (decision(seed, players),) * length, SeatOutcome(-0.2, False, 20))

    def test_capacity_trims_only_oldest_decisions_and_keeps_their_targets(self):
        replay = TrajectoryReplay(5)
        replay.append(self.trajectory(3, seed=1))
        replay.append(self.trajectory(4, seed=2))
        self.assertEqual(len(replay), 5)
        self.assertEqual([len(t.decisions) for t in replay._trajectories], [1, 4])
        indices = mock.Mock()
        indices.randrange.side_effect = range(2, 7)
        samples = replay.sample(indices, 5, n_step=1, gamma=0.5)
        self.assertEqual([s.decision.observation for s in samples], [
            decision(1).observation, *[decision(2).observation] * 4,
        ])
        self.assertEqual([s.scale for s in samples], [1.0, 0.5, 0.5, 0.5, 1.0])
        self.assertEqual([s.bootstrap is None for s in samples], [True, False, False, False, True])
        replay.append(self.trajectory(20, seed=3))
        self.assertEqual(len(replay), 5)
        self.assertEqual([len(t.decisions) for t in replay._trajectories], [5])
        self.assertEqual({s.decision.observation for s in replay.sample(random.Random(2), 20)}, {decision(3).observation})

    def test_n_step_bootstrap_and_terminal_discount_have_literal_scales(self):
        records = tuple(decision(seed) for seed in range(7))
        replay = TrajectoryReplay(10)
        replay.append(Trajectory(1, records, SeatOutcome(-0.75, False, 40)))
        rng = mock.Mock()
        rng.randrange.side_effect = range(7)
        samples = replay.sample(rng, 7, n_step=5, gamma=0.5)
        self.assertEqual([s.scale for s in samples], [1 / 32, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1])
        self.assertEqual([s.bootstrap for s in samples], [records[5], records[6], None, None, None, None, None])
        self.assertEqual([s.outcome.utility for s in samples], [-0.75] * 7)
        self.assertEqual(
            [s.scale * s.outcome.utility for s in samples[2:]],
            [-0.046875, -0.09375, -0.1875, -0.375, -0.75],
        )

    def test_decisions_are_uniform_within_each_round_robin_player_count(self):
        replay = TrajectoryReplay(100)
        short = self.trajectory(2, seed=1, players=2)
        long = self.trajectory(20, seed=2, players=2)
        replay.append(short)
        replay.append(long)
        replay.append(self.trajectory(1, seed=3, players=3))
        replay.append(self.trajectory(40, seed=4, players=4))
        samples = replay.sample(random.Random(73), 33_000)
        self.assertEqual([sum(len(s.decision.observation.players) == count for s in samples) for count in (2, 3, 4)], [11_000] * 3)
        short_draws = sum(s.decision is short.decisions[0] for s in samples)
        long_draws = sum(s.decision is long.decisions[0] for s in samples)
        self.assertTrue(9 < long_draws / short_draws < 11)
        self.assertTrue(900 < short_draws < 1100)

    def test_roundtrip_preserves_cursor_features_and_weights_only_journals(self):
        replay = TrajectoryReplay(20)
        for count in (2, 3, 4):
            replay.append(self.trajectory(3, players=count))
        replay.sample(random.Random(1), 1)
        stream = io.BytesIO()
        torch.save(replay.state_dict(), stream)
        stream.seek(0)
        restored = TrajectoryReplay(20)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        left, right = replay.sample(random.Random(7), 11), restored.sample(random.Random(7), 11)
        self.assertEqual(left, right)
        self.assertEqual([len(s.decision.observation.players) for s in right], [3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4])
        self.assertEqual(replay.state_dict(), restored.state_dict())
        journal = restored.state_dict()["trajectories"][0]
        self.assertEqual(len(journal["decisions"][0][3]), 312)
        self.assertNotIn("history", journal["decisions"][0][0])

    def test_invalid_load_is_atomic_and_reencodes_once_per_valid_decision(self):
        replay = TrajectoryReplay(10)
        replay.append(self.trajectory(2))
        before = replay.state_dict()
        bad_states = []
        for field, value in (("version", 2), ("capacity", 11), ("sample_cursor", -1)):
            bad_states.append({**before, field: value})
        for bad_action, bad_hidden in ((54, None), (True, None), (0, (0,) * 312)):
            state = copy.deepcopy(before)
            journal = state["trajectories"][0]
            snapshot, prefix, _, hidden = journal["decisions"][0]
            journal["decisions"] = ((snapshot, prefix, bad_action, hidden if bad_hidden is None else bad_hidden),)
            bad_states.append(state)
        for state in bad_states:
            with self.assertRaises((ValueError, TypeError)):
                replay.load_state_dict(state)
            self.assertEqual(replay.state_dict(), before)
        with mock.patch("src.training.core.encode", wraps=encode) as encoded:
            replay.load_state_dict(before)
        self.assertEqual(encoded.call_count, 2)
        self.assertEqual(replay.state_dict(), before)


class TestLearner(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def config(self, **overrides):
        return LearnerConfig(**{
            "batch_size": 4, "replay_capacity": 50, "teacher": TeacherConfig(fraction=0),
            **overrides,
        })

    def test_double_dqn_online_selects_legal_action_and_target_evaluates_it(self):
        learner = Learner(TableNetwork(), self.config(
            gamma=0.5, n_step=1, losses=LossWeights(0, 0, 0), weight_decay=0,
        ))
        with torch.no_grad():
            learner.online.values[:3] = torch.tensor([2.0, 3.0, 1e6])
            learner.target.values[:3] = torch.tensor([99.0, 7.0, 1e9])
        record = decision()
        learner.append([Trajectory(1, (record, record), SeatOutcome(-1.0, False, 100))])
        boot = Sample(record, SeatOutcome(-1, False, 100), record, 0.5)
        terminal = Sample(record, SeatOutcome(-1, False, 100), None, 1.0)
        with mock.patch.object(learner.replay, "sample", return_value=[boot, terminal]):
            stats = learner.update()
        self.assertAlmostEqual(stats.q_loss, 1.75, places=6)
        self.assertEqual((stats.samples, stats.updates, learner.update_count), (2, 1, 1))

    def test_cached_update_never_encodes_current_or_bootstrap_observations(self):
        learner = Learner(QNetwork(ModelSpec(8, 1, 1)), self.config(n_step=1))
        records = (decision(1), decision(2), decision(3))
        learner.append([Trajectory(1, records, SeatOutcome(-1, False, 50))])
        with mock.patch("src.model.encoding.encode", side_effect=AssertionError("cached state was encoded")), mock.patch(
            "src.model.network.encode", side_effect=AssertionError("network encoded cached state"),
        ), mock.patch("src.training.core.encode", side_effect=AssertionError("replay encoded cached state")):
            stats = learner.update(2)
        self.assertEqual((stats.samples, stats.updates), (6, 2))
        self.assertEqual(stats.belief_loss, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))

    def test_shaped_terminal_target_does_not_change_binary_win_or_exact_penalty_heads(self):
        learner = Learner(TableNetwork(), self.config(weight_decay=0))
        hand = tuple(int(face in (0, 12, 26, 38)) for face in range(52))
        outcome = solo_outcome(hand, 7, False, 1)
        learner.append([Trajectory(1, (decision(),), outcome)])
        stats = learner.update()
        self.assertAlmostEqual(stats.q_loss, 0.28125, places=6)
        self.assertAlmostEqual(stats.win_loss, 0.69314718, places=6)
        self.assertAlmostEqual(stats.penalty_loss, 0.08, places=6)
        self.assertEqual(learner.replay.state_dict()["trajectories"][0]["outcome"], (-0.75, False, 40))

    def test_auxiliary_losses_mask_absent_seats_and_scale_penalties_and_decks(self):
        class BeliefNetwork(TableNetwork):
            def outputs(self, batch):
                out = super().outputs(batch)
                belief = torch.full_like(out.belief, 100.0)
                belief[:, 1] = 1.0
                return replace(out, belief=belief + out.belief)

        learner = Learner(BeliefNetwork(), self.config())
        record = decision(players=2)
        learner.append([Trajectory(2, (record,), SeatOutcome(-1, False, 50))])
        stats = learner.update()
        self.assertAlmostEqual(stats.win_loss, 0.69314718, places=6)
        self.assertAlmostEqual(stats.penalty_loss, 0.125, places=6)
        self.assertAlmostEqual(stats.belief_loss, 1.28441553, places=6)
        self.assertAlmostEqual(stats.q_loss, 0.5, places=6)

    def test_afterstate_and_teacher_distillation_run_on_teacher_schedule(self):
        learner = Learner(QNetwork(ModelSpec(8, 1, 1)), self.config(
            teacher=TeacherConfig(fraction=1, every_updates=2, max_children=8),
        ))
        env = PappluEnv(GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3))
        env.reset(99)
        obs = env.step(0)
        action = discard_action(next(face for face, count in enumerate(obs.hand) if count))
        record = Decision(obs, encode(obs), action, (obs.hand,) + ((0,) * 52,) * 5)
        learner.append([Trajectory(1, (record,), SeatOutcome(-0.2, False, 20))])
        after_before = learner.online.aux.after_out.weight.detach().clone()
        first = learner.update()
        self.assertEqual(first.after_loss, 0)
        self.assertEqual(first.teacher_loss, 0)
        self.assertTrue(torch.equal(after_before, learner.online.aux.after_out.weight))
        second = learner.update()
        self.assertGreater(second.after_loss, 0)
        self.assertGreater(second.teacher_loss, 0)
        self.assertFalse(torch.equal(after_before, learner.online.aux.after_out.weight))

    def test_terminal_discards_do_not_train_the_afterstate_head(self):
        for warm_start, won, turns in ((True, False, 3), (False, True, 3), (False, False, 1)):
            with self.subTest(warm_start=warm_start, won=won, turns=turns):
                learner = Learner(QNetwork(ModelSpec(8, 1, 1)), self.config(
                    teacher=TeacherConfig(fraction=1, every_updates=1, max_children=8),
                ))
                env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1, max_turns=turns))
                if warm_start:
                    obs = env.reset_warm_start(99)
                else:
                    env.reset(99)
                    obs = env.step(0)
                action = discard_action(next(face for face, count in enumerate(obs.hand) if count))
                record = Decision(obs, encode(obs), action, (obs.hand,) + ((0,) * 52,) * 5)
                learner.append([Trajectory(1, (record,), SeatOutcome(1.0 if won else -1.0, won, 20))])
                stats = learner.update()
                self.assertEqual(stats.after_loss, 0)
                self.assertGreater(stats.q_loss, 0)

    def test_polyak_and_adopt_update_both_models_without_resetting_replay_or_counters(self):
        learner = Learner(QNetwork(ModelSpec(8, 1, 1)), self.config(target_tau=0.25))
        learner.append([Trajectory(1, (decision(),), SeatOutcome(1, True, 0))])
        old_target = copy.deepcopy(learner.target.state_dict())
        learner.update()
        for name, value in learner.target.state_dict().items():
            torch.testing.assert_close(value, old_target[name].lerp(learner.online.state_dict()[name], 0.25), rtol=0, atol=0)
        replay, rng = learner.replay, learner.rng
        counts = learner.decision_count, learner.update_count
        replacement = QNetwork(ModelSpec(8, 1, 1)).state_dict()
        learner.adopt(replacement)
        assert_tree_equal(self, learner.online.state_dict(), replacement)
        assert_tree_equal(self, learner.target.state_dict(), replacement)
        self.assertEqual(learner.optimizer.state_dict()["state"], {})
        self.assertIs(learner.replay, replay)
        self.assertIs(learner.rng, rng)
        self.assertEqual((learner.decision_count, learner.update_count), counts)
        self.assertFalse(learner.target.training)
        self.assertTrue(all(not p.requires_grad for p in learner.target.parameters()))

    def test_exact_resume_includes_target_optimizer_and_rng_without_global_rng_changes(self):
        config = self.config(teacher=TeacherConfig(fraction=0.25, every_updates=2, max_children=8))
        learner = Learner(QNetwork(ModelSpec(8, 1, 1)), config, seed=71)
        learner.learn([Trajectory(1, (decision(),), SeatOutcome(1, True, 0))])
        saved = learner.state_dict()
        resumed = Learner(QNetwork(ModelSpec(8, 1, 1)), config)
        external = torch.get_rng_state().clone()
        resumed.load_state_dict(saved)
        left, right = learner.update(3), resumed.update(3)
        self.assertEqual(left, right)
        assert_tree_equal(self, learner.state_dict(), resumed.state_dict())
        self.assertTrue(torch.equal(external, torch.get_rng_state()))
        self.assertFalse(torch.equal(saved["online"]["policy.card_embed.weight"], learner.online.state_dict()["policy.card_embed.weight"]))

    def test_corrupt_state_never_mutates_a_live_learner(self):
        learner = Learner(QNetwork(ModelSpec(8, 1, 1)), self.config())
        learner.learn([Trajectory(1, (decision(),), SeatOutcome(1, True, 0))])
        before = learner.state_dict()
        for field in (
            "online", "target", "optimizer", "optimizer_step", "optimizer_variance",
            "replay", "rng", "torch_rng_state", "update_count",
        ):
            state = copy.deepcopy(before)
            if field in ("online", "target"):
                next(iter(state[field].values())).fill_(float("nan"))
            elif field == "optimizer":
                next(iter(state[field]["state"].values()))["exp_avg"] = torch.zeros(1)
            elif field == "optimizer_step":
                next(iter(state["optimizer"]["state"].values()))["step"] = torch.tensor(True)
            elif field == "optimizer_variance":
                next(iter(state["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
            elif field == "replay":
                state[field]["version"] = 2
            elif field == "rng":
                state[field] = ()
            elif field == "torch_rng_state":
                state[field] = torch.zeros(1, dtype=torch.uint8)
            else:
                state[field] = -1
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                learner.load_state_dict(state)
            assert_tree_equal(self, learner.state_dict(), before)


class TestBatchedMatches(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_eight_concurrent_matches_equal_scalar_actions_features_and_outcomes(self):
        network = QNetwork(ModelSpec(8, 1, 1))
        policies = [EpsilonPolicy(network, 0.45), RandomPolicy(), GreedyPolicy(network)]
        game = GameConfig(num_decks=2, cards_in_hand=3, required_sequences=1, max_turns=3)
        deals = [
            Deal(700 + index, tuple(seat % 3 for seat in range(players)), players, frozenset(range(players)))
            for index, players in enumerate((2, 3, 4, 2, 3, 4, 2, 3))
        ]
        batched = run_matches(deals, policies, game, record_terminal=True)
        scalar = [
            play_match([policies[row] for row in deal.seats], MatchConfig(game, deal.players), deal.seed,
                       record_trajectories=True, record_terminal=True)
            for deal in deals
        ]
        self.assertEqual(batched, scalar)
        self.assertEqual([result.seed for result in batched], list(range(700, 708)))
        self.assertGreater(sum(result.action_count for result in batched), 100)
        for result in batched:
            for trajectory in result.trajectories:
                for record in trajectory:
                    self.assertEqual(record.features, encode(record.observation))
                    self.assertEqual(record.hidden_hands[0], record.observation.hand)

    def test_epsilon_batch_uses_each_rows_rng_and_one_greedy_forward(self):
        network = QNetwork(ModelSpec(8, 1, 1))
        features = [decision(seed).features for seed in range(8)]
        expected_rngs = [random.Random(seed) for seed in range(8)]
        rngs = [random.Random(seed) for seed in range(8)]
        policy = EpsilonPolicy(network, 0.5)
        expected = [policy.act_features([row], [rng])[0] for row, rng in zip(features, expected_rngs)]
        with mock.patch.object(network, "forward", wraps=network.forward) as forward:
            actions = policy.act_features(features, rngs)
        self.assertEqual(actions, expected)
        self.assertEqual([rng.getstate() for rng in rngs], [rng.getstate() for rng in expected_rngs])
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(forward.call_args.args[0].card.shape[0], 4)
        self.assertTrue(all(action in legal_actions(row) for action, row in zip(actions, features)))

    def test_unscored_matches_preserve_play_and_only_retain_requested_terminal_snapshots(self):
        policies = [RandomPolicy(), RandomPolicy()]
        game = GameConfig(cards_in_hand=3, required_sequences=1, max_turns=2)
        deals = [Deal(17, (0, 1), 2), Deal(31, (1, 0), 2)]
        scored = run_matches(deals, policies, game)
        for record_terminal in (False, True):
            with mock.patch("src.evaluation.arena.DealOutcome.score", side_effect=AssertionError("scored unrequested match")):
                results = run_matches(
                    deals, policies, game, score_outcome=False, record_terminal=record_terminal,
                )
            self.assertEqual([r.outcome for r in results], [None, None])
            self.assertEqual([r.action_count for r in results], [8, 8])
            self.assertEqual([r.telemetry for r in results], [r.telemetry for r in scored])
            self.assertEqual([r.terminal_snapshot is not None for r in results], [record_terminal] * 2)
        self.assertEqual(
            play_match(policies, MatchConfig(game, 2), 17, score_outcome=False),
            run_matches(deals[:1], policies, game, score_outcome=False)[0],
        )


class TestOpponentScheduler(unittest.TestCase):
    def test_fixed_only_pools_normalize_without_applying_pfsp_again(self):
        scheduler = OpponentScheduler()
        for _ in range(20):
            scheduler.record(2, "champion", 0.0)
            scheduler.record(2, "random", 1.0)
        self.assertEqual(scheduler.probabilities(2, ("champion", "random"), "champion"), (0.7, 0.3))
        self.assertEqual(scheduler.probabilities(2, ("champion",), "champion"), (1.0,))
        self.assertEqual(scheduler.probabilities(2, ("random",), "champion"), (1.0,))
        zero_shares = OpponentScheduler(PfspConfig(champion_share=0, anchor_share=0))
        self.assertEqual(zero_shares.probabilities(2, ("champion", "random"), "champion"), (0.5, 0.5))

    def test_pfsp_prefers_tough_opponents_and_keeps_champion_and_anchor_shares(self):
        scheduler = OpponentScheduler()
        identities = ("champion", "random", "tough", "beaten")
        for _ in range(20):
            scheduler.record(2, "tough", 0.0)
            scheduler.record(2, "beaten", 1.0)
        probabilities = scheduler.probabilities(2, identities, "champion")
        self.assertAlmostEqual(probabilities[0], 0.35)
        self.assertAlmostEqual(probabilities[1], 0.15)
        self.assertGreater(probabilities[2], 100 * probabilities[3])
        draws = [scheduler.choose(2, identities, "champion", random.Random(seed)) for seed in range(1000)]
        self.assertGreater(draws.count(2), 450)
        self.assertLess(draws.count(3), 10)
        self.assertEqual(scheduler.probabilities(3, identities, "champion"), (0.35, 0.15, 0.25, 0.25))
        restored = OpponentScheduler(PfspConfig())
        restored.load_state_dict(scheduler.state_dict())
        self.assertEqual(restored.probabilities(2, identities, "champion"), probabilities)
        restored.probabilities(2, ("champion", "random", "new"), "champion")
        self.assertEqual(restored.probabilities(2, identities, "champion"), probabilities)


if __name__ == "__main__":
    unittest.main()
