from collections import Counter
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.cli._runtime import MissingTrainingDependency
from src.cli.train import build_arg_parser, main
from src.evaluate import evaluate_hand
from src.game.environment import GameConfig, PappluEnv
from src.game.multiplayer import MultiplayerEnv
from src.model.encoding import unpack
from src.model.spec import ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from concurrent.futures import ProcessPoolExecutor
    from src.model.network import QNetwork, load_checkpoint, save_checkpoint
    from src.training.train import STATE_VERSION, TrainConfig, Trainer, _Rollout, collect_round


ROOT = Path(__file__).resolve().parents[1]


class TestTrainCli(unittest.TestCase):
    def test_help_and_invalid_arguments_do_not_load_training(self):
        with mock.patch("src.cli.train.load_runtime", side_effect=AssertionError("runtime loaded")):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["--help"]), 0)
                self.assertIn("winning-trajectory margins", output.getvalue())
            with mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(main([]), 2)
                self.assertIn("--output-dir", error.getvalue())
            for flag in ("--warm-start-fraction", "--curriculum-distances", "--pfsp-exponent",
                         "--promote", "--rebase-after-rejections", "--heads", "--target-tau",
                         "--epsilon-decay-episodes"):
                with self.subTest(flag=flag), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                    self.assertEqual(main(["--output-dir", "unused", flag, "1"]), 2)
                    self.assertIn("unrecognized arguments", error.getvalue())
        parsed = build_arg_parser().parse_args(["--output-dir", "run", "--players", "1", "3", "--actors", "2"])
        self.assertEqual((parsed.player_counts, parsed.actors), ([1, 3], 2))

    def test_stationary_epsilon_and_learning_rate_flags_parse(self):
        parsed = build_arg_parser().parse_args([
            "--output-dir", "run", "--updates-per-episode", "3",
            "--lr-decay-updates", "12", "--torch-threads", "2", "--epsilons", "0.01", "0.05",
        ])
        self.assertEqual((parsed.updates_per_episode, parsed.lr_decay_updates, parsed.torch_threads), (3, 12, 2))
        self.assertEqual(parsed.epsilons, [0.01, 0.05])
        self.assertEqual(build_arg_parser().parse_args(["--output-dir", "run", "--margin-weight", "0.25"]).margin_weight, 0.25)

    def test_missing_dependency_is_reported_without_creating_a_run(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "src.cli.train.load_runtime", side_effect=MissingTrainingDependency("PyTorch is missing"),
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
            path = Path(temporary) / "run"
            self.assertEqual(main(["--output-dir", str(path)]), 2)
            self.assertIn("PyTorch is missing", error.getvalue())
            self.assertFalse(path.exists())

    def test_command_help_runs_with_torch_and_numpy_blocked(self):
        script = """
import builtins, runpy, sys
real = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('help imported ' + name)
    return real(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = ['src.cli.train', '--help']
runpy.run_module('src.cli.train', run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--self-play-share", result.stdout)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestSelfPlayTrainer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.overrides = {
            "num_decks": 1, "cards_in_hand": 3, "required_sequences": 1, "max_turns": 2,
            "player_counts": (1, 2), "round_episodes": 4, "updates_per_episode": 1,
            "replay_capacity": 64, "batch_size": 4, "validate_every": 4, "validation_games": 1,
            "checkpoint_every": 4, "self_play_share": 1.0, "max_pool": 2,
        }

    def test_self_play_and_sparring_record_every_seats_negative_outcome(self):
        torch.manual_seed(41)
        network = QNetwork(ModelSpec(9, 1))
        config = TrainConfig(**{
            **self.overrides, "player_counts": (1, 2, 3, 4), "max_turns": 1, "epsilons": (1.0,),
        })
        with tempfile.TemporaryDirectory() as temporary, mock.patch("src.game.environment.hand_reward", return_value=0.0), mock.patch(
            "src.game.multiplayer.hand_reward", return_value=0.0,
        ):
            best = str(Path(temporary) / "best.pt")
            save_checkpoint(best, network, config.game)
            trajectories = collect_round(config, network, best, [], 0, 4)
            opponents = collect_round(TrainConfig(**{**config.__dict__, "self_play_share": 0.0}), network, best, [], 0, 4)
        observed = Counter(round(unpack(t.steps[0].features)["context"][12] * 6) for t in trajectories)
        self.assertEqual(observed, {1: 1, 2: 2, 3: 3, 4: 4})
        self.assertEqual(Counter(round(unpack(t.steps[0].features)["context"][12] * 6) for t in opponents),
                         {1: 1, 2: 2, 3: 3, 4: 4})
        self.assertEqual([len(t.steps) for t in trajectories], [2] * 10)
        self.assertTrue(all(-1.0 <= t.utility <= -0.6 for t in [*trajectories, *opponents]))
        self.assertEqual([unpack(t.steps[0].features)["context"][0] for t in trajectories], [1.0] * 10)
        self.assertEqual({step.greedy for trajectory in trajectories for step in trajectory.steps}, {False})
        self.assertEqual({step.greedy for trajectory in opponents for step in trajectory.steps}, {False, True})

    def test_each_learner_seat_keeps_its_stationary_exploration_rate(self):
        config = TrainConfig(**{
            **self.overrides, "player_counts": (1, 2, 3, 4), "max_turns": 3, "epsilons": (0.0, 1.0),
        })
        with mock.patch("src.game.environment.hand_reward", return_value=0.0), mock.patch(
            "src.game.multiplayer.hand_reward", return_value=0.0,
        ):
            trajectories = collect_round(config, QNetwork(ModelSpec(9, 1)), "unused", [], 0, 4)
        self.assertEqual(len(trajectories), 10)
        self.assertEqual({trajectory.steps[0].greedy for trajectory in trajectories}, {False, True})
        for trajectory in trajectories:
            self.assertEqual(len(trajectory.steps), 6)
            self.assertEqual({step.greedy for step in trajectory.steps}, {trajectory.steps[0].greedy})

    def test_three_and_four_player_sparring_retain_real_greedy_opponent_wins(self):
        config = TrainConfig(
            num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=8,
            player_counts=(3, 4), self_play_share=0.0, epsilons=(1.0,), replay_capacity=64,
        )
        network = QNetwork(ModelSpec(9, 1))
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
        with tempfile.TemporaryDirectory() as temporary:
            best = str(Path(temporary) / "best.pt")
            save_checkpoint(best, network, config.game)
            for start, players, steps in ((1, 4, 10), (16, 3, 4)):
                with self.subTest(players=players):
                    trajectories = collect_round(config, network, best, [], start, 1)
                    self.assertEqual(len(trajectories), players)
                    winner = trajectories[0]
                    self.assertEqual((winner.utility, len(winner.steps)), (1.0, steps))
                    self.assertEqual({step.greedy for step in winner.steps}, {True})
                    self.assertTrue(all(trajectory.utility < 0 for trajectory in trajectories[1:]))
                    final = winner.steps[-1]
                    cards = unpack(final.features)["card"]
                    hand = [round(cards[face * 22]) for face in range(52)]
                    hand[final.action - 2] -= 1
                    joker = next(face for face in range(52) if cards[face * 22 + 6])
                    self.assertTrue(evaluate_hand(hand, joker, required_sequences=1, cards_in_hand=3).is_valid)

    def test_sparring_boundaries_select_champion_heuristic_random_and_uniform_pool(self):
        config = TrainConfig(**{
            **self.overrides, "player_counts": (6,), "max_turns": 1, "self_play_share": 0.0,
            "epsilons": (1.0,),
        })
        network = QNetwork(ModelSpec(9, 1))
        real_rng = random.Random
        with tempfile.TemporaryDirectory() as temporary:
            best = str(Path(temporary) / "best.pt")
            save_checkpoint(best, network, config.game)
            for pool, expected in (([best] * 3, (0, 1, 2, 3, 4, 5)), ([], (0, 1, 2, 3, 1, 1))):
                with self.subTest(pool=pool):
                    deal_rng = mock.Mock()
                    deal_rng.random.side_effect = (1.0, 0.449, 0.45, 0.65, 0.70, 0.999)
                    deal_rng.randrange.side_effect = (0, 4, 5)
                    first = [True]
                    deals = []

                    def seeded(seed=None):
                        if first[0]:
                            first[0] = False
                            return deal_rng
                        return real_rng(seed)

                    def rollout(*args):
                        deals.append(args[1])
                        return _Rollout(*args)

                    with mock.patch("src.training.train.random.Random", side_effect=seeded), mock.patch(
                        "src.training.train._Rollout", side_effect=rollout,
                    ), mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
                        trajectories = collect_round(config, network, best, pool, 0, 1)
                    self.assertEqual(deals[0].seats, expected)
                    self.assertEqual(deals[0].record_seats, frozenset(range(6)))
                    self.assertEqual(tuple(len(trajectory.steps) for trajectory in trajectories), (2,) * 6)
                    self.assertEqual(tuple(trajectory.steps[0].greedy for trajectory in trajectories),
                                     (False, True, True, False, True, True))
                    self.assertTrue(all(trajectory.utility <= -0.6 for trajectory in trajectories))

    def test_resume_matches_uninterrupted_weights_replay_optimizer_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            whole = Trainer.open(root / "whole", self.overrides, initial=ModelSpec(9, 1))
            self.assertEqual(whole.run(max_episodes=8), "max_episodes")
            partial = Trainer.open(root / "split", self.overrides, initial=ModelSpec(9, 1))
            partial.run(max_episodes=4)
            resumed = Trainer.open(root / "split")
            resumed.run(max_episodes=8)
            self.assertEqual(whole.summary(), resumed.summary())
            self.assertEqual(whole.learner.replay.state_dict(), resumed.learner.replay.state_dict())
            self.assertEqual(whole.learner.rng.getstate(), resumed.learner.rng.getstate())
            for name, value in whole.learner.online.state_dict().items():
                torch.testing.assert_close(value, resumed.learner.online.state_dict()[name], rtol=0, atol=0)
            for left, right in zip(whole.learner.optimizer.state.values(), resumed.learner.optimizer.state.values()):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual((root / "whole/metrics.jsonl").read_text(), (root / "split/metrics.jsonl").read_text())
            status = json.loads((root / "split/status.json").read_text())
            self.assertEqual((status["status"], status["episodes"], status["updates"]), ("max_episodes", 8, 8))
            for name in ("latest-state.pt", "latest-model.pt", "best-model.pt", "initial-model.pt"):
                self.assertTrue((root / "split" / name).is_file())
            latest, game, meta = load_checkpoint(str(root / "split/latest-model.pt"))
            self.assertEqual((latest.spec, game, meta["episodes"]), (ModelSpec(9, 1), resumed.config.game, 8))
            self.assertEqual(meta["algorithm"], "dmc")

    def test_validation_is_fixed_and_periodic_snapshots_roll_without_repromoting_ties(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), {
                **self.overrides, "max_turns": 1, "round_episodes": 1,
                "updates_per_episode": 0, "validate_every": 1,
            }, initial=ModelSpec(9, 1))
            trainer.run(max_episodes=1)
            first = copy.deepcopy(trainer.validation)
            self.assertEqual(set(first), {"score", "solo", "2p_heuristic", "2p_best", "vs_best", "promoted"})
            self.assertGreaterEqual(first["score"], 0)
            self.assertLessEqual(first["score"], 1)
            self.assertEqual((first["2p_best"], first["vs_best"], first["promoted"]), (0.5, 0.5, True))
            selected = (Path(temporary) / "best-model.pt").read_bytes()
            trainer.run(max_episodes=3)
            self.assertEqual(trainer.validation, {**first, "promoted": False})
            self.assertEqual(trainer.best_score, first["score"])
            self.assertEqual(trainer.snapshot_pool, [
                "opponents/snap-000000000002.pt", "opponents/snap-000000000003.pt",
            ])
            self.assertEqual((Path(temporary) / "best-model.pt").read_bytes(), selected)
            self.assertTrue((Path(temporary) / "initial-model.pt").is_file())
            self.assertTrue((Path(temporary) / "opponents/best-000000000001.pt").is_file())
            best, _, meta = load_checkpoint(str(Path(temporary) / "best-model.pt"))
            self.assertEqual(meta["episodes"], 1)
            for name, value in trainer.learner.online.state_dict().items():
                torch.testing.assert_close(value, best.state_dict()[name], rtol=0, atol=0)

    def test_validation_metrics_match_with_one_or_two_actors(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = []
            for actors in (1, 2):
                with Trainer.open(Path(temporary) / str(actors), {
                    **self.overrides, "actors": actors, "player_counts": (1, 2, 3),
                    "validation_games": 9, "max_turns": 1,
                }, initial=ModelSpec(9, 1)) as trainer:
                    trainer.learner.online.train()
                    results.append(trainer.validate())
                    self.assertFalse(trainer.learner.online.training)
                    self.assertEqual(trainer._executor is None, actors == 1)
            self.assertEqual(results[0], results[1])
            self.assertEqual((results[0]["2p_best"], results[0]["3p_best"], results[0]["vs_best"]), (0.5, 0.5, 0.5))
            self.assertIs(results[0]["promoted"], True)
            self.assertAlmostEqual(results[0]["score"], sum(results[0][key] for key in (
                "solo", "2p_heuristic", "3p_heuristic",
            )) / 3)

    def test_stronger_model_replaces_reigning_best_after_real_head_to_head_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weak = QNetwork(ModelSpec(9, 1))
            with torch.no_grad():
                for parameter in weak.parameters():
                    parameter.zero_()
                weak.policy.card_embed.weight[0, 8] = 1
                weak.policy.card_embed.bias[1] = 1
                weak.policy.card_final.weight.fill_(1)
                weak.policy.face_local.weight[0, 0] = 1
                weak.policy.face_out.weight[0, 0] = -1
                weak.policy.draw_out.bias[0] = 1
            save_checkpoint(str(root / "weak.pt"), weak, TrainConfig(**self.overrides).game)
            with Trainer.open(root / "run", {
                **self.overrides, "player_counts": (2, 3), "validation_games": 9,
            }, initial=root / "weak.pt") as trainer:
                baseline = trainer.validate()
                self.assertEqual((baseline["vs_best"], baseline["promoted"]), (0.5, True))
                selected = (trainer.root / "best-model.pt").read_bytes()
                with torch.no_grad():
                    trainer.learner.online.policy.face_out.weight[0, 0] = 1
                trainer.episodes = 7
                result = trainer.validate()
                self.assertEqual(result["2p_best"], 1.0)
                self.assertAlmostEqual(result["3p_best"], 101 / 108)
                self.assertAlmostEqual(result["vs_best"], 209 / 216)
                self.assertIs(result["promoted"], True)
                self.assertEqual(trainer.best_score, max(baseline["score"], result["score"]))
                self.assertNotEqual((trainer.root / "best-model.pt").read_bytes(), selected)
                best, _, meta = load_checkpoint(str(trainer.root / "best-model.pt"))
                self.assertEqual(meta["episodes"], 7)
                for name, value in trainer.learner.online.state_dict().items():
                    torch.testing.assert_close(value, best.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(trainer.snapshot_pool[-1], "opponents/best-000000000007.pt")

    def test_promotion_thresholds_anchor_floor_and_fresh_snapshot_contents(self):
        cases = (
            (0.80, 0.50, False, "snap", 0.80),
            (0.77, 0.51, False, "snap", 0.80),
            (0.78, 0.51, True, "best", 0.80),
            (0.82, 0.50, False, "snap", 0.80),
            (0.83, 0.44, False, None, 0.80),
            (0.83, 0.48, False, "snap", 0.80),
            (0.83, 0.49, True, "best", 0.83),
            (0.79, 0.44, False, None, 0.80),
            (0.85, 0.52, True, "best", 0.85),
        )
        with tempfile.TemporaryDirectory() as temporary, Trainer.open(Path(temporary), {
            **self.overrides, "player_counts": (2,),
        }, initial=ModelSpec(9, 1)) as trainer:
            trainer.best_score = 0.80
            for episode, (score, vs_best, promoted, snapshot, best_score) in enumerate(cases, 1):
                with self.subTest(score=score, vs_best=vs_best):
                    trainer.best_score = 0.80
                    with torch.no_grad():
                        trainer.learner.online.policy.face_out.weight.add_(0.1)
                    trainer.episodes = episode
                    selected = (trainer.root / "best-model.pt").read_bytes()
                    pool = list(trainer.snapshot_pool)
                    credits = {"2p_heuristic": 2 * score, "2p_best": 2 * vs_best}
                    with mock.patch("src.training.train._validation_job", side_effect=lambda job: (job[3], credits[job[3]])):
                        self.assertEqual(trainer.validate(), {
                            "score": score, "2p_heuristic": score,
                            "2p_best": vs_best, "vs_best": vs_best, "promoted": promoted,
                        })
                    self.assertEqual(trainer.best_score, best_score)
                    if not promoted:
                        self.assertEqual((trainer.root / "best-model.pt").read_bytes(), selected)
                    if snapshot is None:
                        self.assertEqual(trainer.snapshot_pool, pool)
                    else:
                        self.assertEqual(trainer.snapshot_pool[-1], "opponents/%s-%012d.pt" % (snapshot, episode))
                        saved, _, meta = load_checkpoint(str(trainer.root / trainer.snapshot_pool[-1]))
                        self.assertEqual(meta["episodes"], episode)
                        for name, value in trainer.learner.online.state_dict().items():
                            torch.testing.assert_close(value, saved.state_dict()[name], rtol=0, atol=0)
                    self.assertTrue((trainer.root / "initial-model.pt").is_file())

    def test_solo_validation_does_not_repromote_equal_scores(self):
        with tempfile.TemporaryDirectory() as temporary, Trainer.open(Path(temporary), {
            **self.overrides, "player_counts": (1,), "max_turns": 1,
        }, initial=ModelSpec(9, 1)) as trainer:
            for episode in (1, 2):
                trainer.episodes = episode
                self.assertEqual(trainer.validate(), {"score": 0.0, "solo": 0.0, "promoted": episode == 1})
            self.assertEqual(trainer.best_score, 0.0)
            self.assertEqual(trainer.snapshot_pool, [
                "initial-model.pt", "opponents/best-000000000001.pt",
            ])

    def test_spawned_actors_reuse_one_pool_and_close_it_after_multiple_rounds(self):
        executors = []

        def create_pool(**kwargs):
            pool = ProcessPoolExecutor(**kwargs)
            executors.append(pool)
            return pool

        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), {
                **self.overrides, "actors": 2, "round_episodes": 2,
            }, initial=ModelSpec(9, 1))
            before = copy.deepcopy(trainer.learner.online.state_dict())
            with mock.patch("src.training.train.ProcessPoolExecutor", side_effect=create_pool):
                self.assertEqual(trainer.run(max_episodes=4), "max_episodes")
            self.assertEqual((trainer.episodes, trainer.learner.update_count), (4, 4))
            self.assertEqual(len(executors), 1)
            self.assertIsNone(trainer._executor)
            self.assertIsNone(executors[0]._processes)
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.learner.online.state_dict().items()))
            resumed = Trainer.open(Path(temporary), {"actors": 1})
            self.assertEqual(resumed.run(max_episodes=5), "max_episodes")
            self.assertEqual(resumed.episodes, 5)

    def test_solo_and_multiplayer_validation_use_only_fixed_odd_deal_seeds(self):
        solo_seeds, multiplayer_seeds = [], []
        solo_reset, multiplayer_reset = PappluEnv.reset, MultiplayerEnv.reset

        def reset_solo(env, seed=None):
            solo_seeds.append(seed)
            return solo_reset(env, seed)

        def reset_multiplayer(env, seed):
            multiplayer_seeds.append(seed)
            return multiplayer_reset(env, seed)

        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), {
                **self.overrides, "validation_games": 3, "max_turns": 1,
            }, initial=ModelSpec(9, 1))
            with mock.patch.object(PappluEnv, "reset", reset_solo), mock.patch.object(
                MultiplayerEnv, "reset", reset_multiplayer,
            ):
                result = trainer.validate()
            self.assertEqual(solo_seeds, [30000001, 30000003, 30000005])
            self.assertEqual(multiplayer_seeds, [30000013, 30000013, 30000015, 30000015, 30000017, 30000017] * 2)
            self.assertGreaterEqual(result["score"], 0)
            self.assertLessEqual(result["score"], 1)

    def test_stop_and_time_limits_save_clean_resumable_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), self.overrides, initial=ModelSpec(9, 1))
            self.assertEqual(trainer.run(stop=lambda: True), "stopped")
            self.assertEqual(trainer.episodes, 0)
            self.assertEqual(trainer.run(max_seconds=1e-12), "budget")
            self.assertEqual(trainer.episodes, 0)
            stopped = iter((False, True))
            self.assertEqual(trainer.run(stop=lambda: next(stopped)), "stopped")
            self.assertEqual(trainer.episodes, 4)
            resumed = Trainer.open(Path(temporary))
            self.assertEqual(resumed.run(max_episodes=4), "max_episodes")
            self.assertEqual(resumed.learner.replay.state_dict(), trainer.learner.replay.state_dict())

    def test_initial_checkpoint_and_optional_opponents_are_copied_into_the_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.pt"
            game = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=2)
            save_checkpoint(str(initial), QNetwork(ModelSpec(9, 1)), game)
            trainer = Trainer.open(root / "run", {
                "player_counts": (2,), "replay_capacity": 64, "validation_games": 0,
            }, initial=initial, opponents=[initial])
            self.assertEqual((trainer.learner.online.spec, trainer.config.game), (ModelSpec(9, 1), game))
            self.assertEqual(trainer.snapshot_pool, ["initial-model.pt", "opponents/seed-000.pt"])
            initial.unlink()
            self.assertEqual(trainer.run(max_episodes=2), "max_episodes")
            resumed = Trainer.open(root / "run")
            self.assertEqual(resumed.run(max_episodes=3), "max_episodes")

    def test_resume_rejects_recipe_changes_and_missing_opponents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Trainer.open(root, self.overrides, initial=ModelSpec(9, 1))
            with self.assertRaisesRegex(ValueError, "cannot change"):
                Trainer.open(root, {"gamma": 1.0})
            with self.assertRaisesRegex(ValueError, "fresh runs"):
                Trainer.open(root, initial=ModelSpec(9, 1))
            (root / "initial-model.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing opponent"):
                Trainer.open(root)

    def test_invalid_recipe_and_limits_fail_at_the_boundary(self):
        for change in ({"gamma": 0}, {"gamma": float("nan")}, {"actors": 0}, {"player_counts": (1, 1)},
                       {"self_play_share": 1.1}, {"batch_size": True}, {"learning_rate": 0},
                       {"margin_weight": -0.01}, {"margin_weight": float("inf")}, {"margin_weight": True},
                       {"lr_decay_updates": -1}, {"lr_decay_updates": True}, {"epsilons": ()},
                       {"epsilons": (True,)}, {"epsilons": (float("nan"),)}, {"epsilons": (1.1,)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TrainConfig(**change)
        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer.open(Path(temporary), self.overrides, initial=ModelSpec(9, 1))
            for limits in ({"max_episodes": -1}, {"max_seconds": 0}, {"max_seconds": float("inf")}):
                with self.subTest(limits=limits), self.assertRaises(ValueError):
                    trainer.run(**limits)

    def test_real_cli_runs_and_resumes_the_saved_nondefault_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, "-m", "src.cli.train", "--output-dir", temporary]
            flags = [
                "--players", "1", "2", "--max-turns", "1", "--round-episodes", "2",
                "--batch-size", "2", "--replay-capacity", "32", "--validation-games", "1",
                "--validate-every", "2", "--width", "9", "--blocks", "1", "--max-episodes", "2",
                "--updates-per-episode", "3", "--lr-decay-updates", "12", "--torch-threads", "2",
                "--epsilons", "0.01", "0.05",
                "--margin-weight", "0.25",
            ]
            fresh = subprocess.run(command + flags, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)
            resumed = subprocess.run(command + ["--max-episodes", "4"], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            report = json.loads(resumed.stdout)
            self.assertEqual((report["episodes"], report["updates"], report["model_spec"]),
                             (4, 12, {"width": 9, "blocks": 1}))
            config = TrainConfig(**torch.load(Path(temporary) / "latest-state.pt", weights_only=True)["config"])
            self.assertEqual((config.updates_per_episode, config.lr_decay_updates, config.torch_threads), (3, 12, 2))
            self.assertEqual(config.epsilons, (0.01, 0.05))
            self.assertEqual(config.margin_weight, 0.25)
            self.assertEqual(len((Path(temporary) / "metrics.jsonl").read_text().splitlines()), 2)

    def test_cli_uses_default_dimensions_for_omitted_width_or_blocks(self):
        for flags, expected in (
            ([], {"width": 128, "blocks": 3}),
            (["--width", "9"], {"width": 9, "blocks": 3}),
            (["--blocks", "1"], {"width": 128, "blocks": 1}),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary, mock.patch(
                "sys.stdout", new_callable=io.StringIO,
            ) as output:
                self.assertEqual(main([
                    "--output-dir", temporary, "--replay-capacity", "4", "--validation-games", "0",
                    "--max-episodes", "0", *flags,
                ]), 0)
                self.assertEqual(json.loads(output.getvalue())["model_spec"], expected)
                saved = torch.load(Path(temporary) / "latest-state.pt", weights_only=True)
                self.assertEqual(saved["spec"], expected)
                self.assertEqual(saved["version"], 12)

    def test_resume_rejects_old_state_versions_and_missing_reigning_champion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Trainer.open(root, self.overrides, initial=ModelSpec(9, 1))
            state = torch.load(root / "latest-state.pt", weights_only=True)
            for version in (8, 11):
                torch.save({**state, "version": version}, root / "latest-state.pt")
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "incompatible trainer state"):
                    Trainer.open(root)
            torch.save({**state, "version": STATE_VERSION}, root / "latest-state.pt")
            (root / "best-model.pt").unlink()
            with self.assertRaisesRegex(ValueError, "missing opponent checkpoint best-model.pt"):
                Trainer.open(root)


if __name__ == "__main__":
    unittest.main()
