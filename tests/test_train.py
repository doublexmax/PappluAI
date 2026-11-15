from __future__ import annotations

from collections import Counter
import copy
import io
import json
from pathlib import Path
import random
import runpy
import sys
import tempfile
import unittest
from unittest import mock

from src.checkpoints import evidence as gate_evidence
from src.checkpoints.io import file_sha256
from src.cli._runtime import MissingTrainingDependency
from src.cli.train import build_arg_parser, main as cli_main
from src.game.environment import ENCODING_VERSION, GameConfig, Observation, PappluEnv, discard_action
from src.game.multiplayer import MatchConfig, MultiplayerEnv
from src.game.outcome import SeatOutcome
from src.model.encoding import encode
from src.model.spec import ARCHITECTURE, ModelSpec

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.evaluation.promotion import GateInterrupted
    from src.model.network import QNetwork, save_checkpoint
    from src.training.train import (
        Confirming,
        Learner,
        MAX_OPPONENT_POOL,
        RANDOM_OPPONENT,
        Replay,
        Selecting,
        Step,
        TrainConfig,
        Trainer,
        Trajectory,
        curriculum_step,
        disjoint_seed_gap,
        epsilon_at,
        pfsp_probabilities,
    )


def _synthetic_gate_result(
    *,
    positive: bool,
    candidate_path: str,
    champion_path: str,
    seed: int,
    seed_blocks: int,
    solo_games: int,
    max_turns: int,
    player_counts: tuple[int, ...],
    samples: int,
    recycle_discard: bool,
    phase: str,
    **_unused: object,
) -> dict:
    rules = GameConfig(max_turns=max_turns, recycle_discard=recycle_discard)
    margin_value = 0.25 if positive else -0.25
    multiplayer = {}
    for count in player_counts:
        count_blocks = gate_evidence.blocks_for_count(seed_blocks, count)
        games = count_blocks * len(gate_evidence.balanced_seat_orders(count))
        margins = [margin_value] * count_blocks
        declarations = [0.10 if positive else -0.10] * count_blocks
        multiplayer[str(count)] = {
            "blocks": count_blocks,
            "games": games,
            "placement_margins": margins,
            "candidate_minus_champion": sum(margins) / count_blocks,
            "declaration_margins": declarations,
            "declaration_delta": sum(declarations) / count_blocks,
        }
    cand_outcomes = [1 if positive else 0] * solo_games
    champ_outcomes = [0 if positive else 1] * solo_games
    solo = {
        "games": solo_games,
        "candidate_outcomes": cand_outcomes,
        "champion_outcomes": champ_outcomes,
        "candidate_wins": sum(cand_outcomes),
        "champion_wins": sum(champ_outcomes),
    }
    metrics = gate_evidence.evidence_metrics(
        multiplayer, solo, seed_blocks, seed, samples, player_counts=player_counts,
    )
    accepted, failures = gate_evidence.assess_gate(
        phase, rules, multiplayer, metrics, seed_blocks, solo_games, player_counts=player_counts,
    )
    return {
        "gate_schema_version": gate_evidence.GATE_SCHEMA_VERSION,
        "encoding_version": ENCODING_VERSION,
        "architecture": ARCHITECTURE,
        "phase": phase,
        "accepted": accepted,
        "gate_failures": failures,
        "candidate_sha256": file_sha256(Path(candidate_path)),
        "champion_sha256": file_sha256(Path(champion_path)),
        "player_counts": list(player_counts),
        "rules": rules.to_dict(),
        "seed": seed,
        "seed_blocks": seed_blocks,
        "bootstrap_samples": samples,
        "multiplayer": multiplayer,
        "solo": solo,
        **metrics,
    }


class TestTrainCliBoundaries(unittest.TestCase):
    def test_help_and_missing_output_never_load_torch_runtime(self):
        parser = build_arg_parser()
        parsed = parser.parse_args(["--players", "1", "2", "--max-episodes", "4", "--output-dir", "out"])
        self.assertEqual(parsed.player_counts, [1, 2])
        self.assertEqual(parsed.max_episodes, 4)
        with mock.patch("src.cli.train.load_runtime", side_effect=AssertionError("runtime loaded")):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(cli_main(["--help"]), 0)
            with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(cli_main([]), 2)
                self.assertIn("--output-dir or --checkpoint", stderr.getvalue())
        with mock.patch.object(sys, "argv", ["src.cli.train", "--help"]), mock.patch(
            "sys.stdout", new_callable=io.StringIO,
        ):
            saved = sys.modules.pop("src.cli.train", None)
            try:
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_module("src.cli.train", run_name="__main__")
            finally:
                if saved is not None:
                    sys.modules["src.cli.train"] = saved
            self.assertEqual(exited.exception.code, 0)

    def test_missing_training_dependency_exits_with_code_two(self):
        with mock.patch(
            "src.cli.train.load_runtime",
            side_effect=MissingTrainingDependency("PyTorch is missing"),
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(cli_main(["--output-dir", "tmp"]), 2)
            self.assertIn("PyTorch is missing", stderr.getvalue())


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestReplayAndConfig(unittest.TestCase):
    def make_steps(self, players: int, count: int) -> tuple[Step, ...]:
        if players == 1:
            env = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1, max_turns=max(4, count)))
            obs = env.reset(seed=players + count)
            steps = []
            for idx in range(count):
                if obs.done:
                    obs = env.reset(seed=100 + idx)
                action = 0 if obs.phase.value == 0 else discard_action(next(i for i, n in enumerate(obs.hand) if n))
                steps.append(Step(encode(obs), action))
                obs = env.step(action)
            return tuple(steps)
        match = MultiplayerEnv(MatchConfig(GameConfig(cards_in_hand=3, required_sequences=1, max_turns=20), players))
        match.reset(players * 13 + count)
        steps = []
        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
            while len(steps) < count:
                obs = match.observe()
                seat = match.view().current_seat
                hands = match.private_hands(seat)[1:]
                action = 0 if obs.phase.value == 0 else discard_action(next(i for i, n in enumerate(obs.hand) if n))
                steps.append(Step(encode(obs), action, hands))
                match.step(action)
        return tuple(steps)

    def test_strict_step_capacity_and_player_count_stratified_sampling(self):
        replay = Replay(capacity=10)
        steps_2p = self.make_steps(2, 7)
        steps_3p = self.make_steps(3, 6)
        replay.append(Trajectory(2, steps_2p, SeatOutcome(0.5, True, 0)))
        replay.append(Trajectory(3, steps_3p, SeatOutcome(-0.5, False, 40)))
        self.assertEqual(len(replay), 10)
        self.assertEqual(replay.counts_by_players(), {2: 4, 3: 6})
        replay_multi = Replay(capacity=100)
        for players, count in ((1, 25), (2, 10), (4, 2)):
            replay_multi.append(Trajectory(players, self.make_steps(players, count), SeatOutcome(0.0, False, 10)))
        sampled = replay_multi.sample(random.Random(19), 30, n_step=3, gamma=0.9)
        buckets_seen = Counter()
        for item in sampled:
            for p, queue in replay_multi._buckets.items():
                if item.step in queue[0].steps:
                    buckets_seen[p] += 1
        self.assertEqual(buckets_seen, {1: 10, 2: 10, 4: 10})

    def test_n_step_bootstrap_stays_within_trajectory_and_contracts(self):
        steps = self.make_steps(1, 4)
        replay = Replay(capacity=10)
        replay.append(Trajectory(1, steps, SeatOutcome(1.0, True, 0)))
        seen = {}
        rng = random.Random(5)
        for _ in range(64):
            sample = replay.sample(rng, 1, n_step=2, gamma=0.9)[0]
            idx = steps.index(sample.step)
            seen[idx] = sample
        self.assertEqual(set(seen), {0, 1, 2, 3})
        self.assertEqual(seen[0].bootstrap, steps[2].features)
        self.assertAlmostEqual(seen[0].scale, 0.81)
        self.assertEqual(seen[1].bootstrap, steps[3].features)
        self.assertAlmostEqual(seen[1].scale, 0.81)
        self.assertIsNone(seen[2].bootstrap)
        self.assertAlmostEqual(seen[2].scale, 0.9)
        self.assertIsNone(seen[3].bootstrap)
        self.assertAlmostEqual(seen[3].scale, 1.0)

    def test_zlib_roundtrip_is_weights_only_safe_without_observation_objects(self):
        replay = Replay(capacity=32)
        replay.append(Trajectory(1, self.make_steps(1, 3), SeatOutcome(1.0, True, 0)))
        replay.append(Trajectory(3, self.make_steps(3, 4), SeatOutcome(-0.25, False, 30)))
        replay.sample(random.Random(3), 1)
        state = replay.state_dict()
        self.assertIsInstance(state["features_zlib"], bytes)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded_state = torch.load(buffer, weights_only=True)
        restored = Replay(capacity=32)
        restored.load_state_dict(loaded_state)
        self.assertEqual(restored.state_dict(), state)
        for queue in restored._buckets.values():
            for traj in queue:
                for step in traj.steps:
                    self.assertNotIsInstance(step, Observation)
                    self.assertFalse(hasattr(step, "observation"))
        before = restored.state_dict()
        for bad in (
            {**loaded_state, "version": 4},
            {**loaded_state, "features_zlib": loaded_state["features_zlib"][:-4]},
        ):
            with self.assertRaises(ValueError):
                restored.load_state_dict(bad)
            self.assertEqual(restored.state_dict(), before)
        tiny = Replay(capacity=2)
        with self.assertRaisesRegex(ValueError, "capacity"):
            tiny.load_state_dict(loaded_state)

    def test_train_config_validation_and_resume_overrides(self):
        base = TrainConfig(player_counts=(2, 1, 3))
        self.assertEqual(base.player_counts, (1, 2, 3))
        self.assertEqual(base.gate_counts, (2, 3))
        updated = TrainConfig.resolve(base, {"checkpoint_every": 8, "gate_workers": 3})
        self.assertEqual((updated.checkpoint_every, updated.gate_workers), (8, 3))
        with self.assertRaisesRegex(ValueError, "cannot change training config on resume"):
            TrainConfig.resolve(base, {"learning_rate": 1e-2})
        invalid_cases = (
            {"player_counts": ()},
            {"player_counts": (2, 2)},
            {"player_counts": (2,), "warm_start_fraction": 0.2},
            {"player_counts": (2,), "curriculum_distances": (3,)},
            {"pfsp_champion_share": 0.7, "pfsp_anchor_share": 0.5},
            {"promote": True, "player_counts": (1,)},
            {"promote": True, "player_counts": (2,), "cards_in_hand": 3, "required_sequences": 1},
            {"epsilon_start": 0.1, "epsilon_end": 0.2},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    def test_curriculum_and_pfsp_pure_rules(self):
        cfg = TrainConfig(
            player_counts=(1, 2),
            curriculum_distances=(3, 6),
            stage_passes=2,
            stage_margin=0.10,
            stage_floor=0.25,
            pfsp_champion_share=0.40,
            pfsp_anchor_share=0.10,
            pfsp_exponent=2.0,
            pfsp_prior_games=4.0,
        )
        self.assertAlmostEqual(epsilon_at(cfg, 0), cfg.epsilon_start)
        self.assertAlmostEqual(epsilon_at(cfg, cfg.epsilon_decay_episodes), cfg.epsilon_end)
        stage, passes = curriculum_step(cfg, 0, 0, greedy_rate=0.30, random_rate=0.15)
        self.assertEqual((stage, passes), (0, 1))
        stage, passes = curriculum_step(cfg, stage, passes, greedy_rate=0.20, random_rate=0.15)
        self.assertEqual((stage, passes), (0, 0))
        stage, passes = curriculum_step(cfg, 0, 1, greedy_rate=0.30, random_rate=0.15)
        self.assertEqual((stage, passes), (1, 0))
        probs = pfsp_probabilities(
            cfg,
            scores={(2, "weak"): (9.0, 10), (2, "strong"): (1.0, 10)},
            players=2,
            pool_ids=["champ", "weak", "strong", RANDOM_OPPONENT],
            champion_id="champ",
        )
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertAlmostEqual(probs[0], 0.40)
        self.assertAlmostEqual(probs[3], 0.10)
        self.assertGreater(probs[2], probs[1])


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestLearnerAndTrainerLifecycle(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_learner_updates_heads_and_roundtrips_state(self):
        cfg = TrainConfig(
            cards_in_hand=3,
            required_sequences=1,
            max_turns=6,
            player_counts=(1, 3),
            batch_size=4,
            learning_rate=1e-3,
        )
        net = QNetwork(ModelSpec(16, 2, 1))
        learner = Learner(net, cfg, seed=11)
        solo_env = PappluEnv(cfg.game)
        solo_obs = solo_env.reset(seed=3)
        match = MultiplayerEnv(MatchConfig(cfg.game, 3))
        match.reset(seed=5)
        mp_obs = match.observe()
        mp_hands = match.private_hands(match.view().current_seat)[1:]
        stats_solo = learner.learn([Trajectory(1, (Step(encode(solo_obs), 0),), SeatOutcome(1.0, True, 0))], updates=2)
        self.assertEqual(stats_solo.belief_loss, 0.0)
        stats_mp = learner.learn([Trajectory(3, (Step(encode(mp_obs), 0, mp_hands),), SeatOutcome(-0.5, False, 35))], updates=2)
        self.assertGreater(stats_mp.belief_loss, 0.0)
        for param in learner.online.parameters():
            self.assertTrue(torch.isfinite(param).all())
        saved = learner.state_dict()
        clone_a = Learner(QNetwork(ModelSpec(16, 2, 1)), cfg, seed=99)
        clone_a.load_state_dict(saved)
        out_orig = learner.update(2)
        out_clone = clone_a.update(2)
        self.assertAlmostEqual(out_orig.loss, out_clone.loss, places=7)
        for left, right in zip(learner.online.parameters(), clone_a.online.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_mixed_table_curriculum_pfsp_and_deterministic_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            overrides = {
                "cards_in_hand": 3,
                "required_sequences": 1,
                "max_turns": 2,
                "player_counts": (1, 2, 3),
                "curriculum_distances": (2,),
                "warm_start_fraction": 0.25,
                "round_episodes": 3,
                "updates_per_episode": 1,
                "batch_size": 4,
                "validate_every": 2,
                "validation_games": 2,
                "checkpoint_every": 3,
                "seed": 31,
            }
            spec = ModelSpec(16, 2, 1)
            part = Trainer.open(root / "split", overrides, initial=spec)
            self.assertEqual(part.run(max_episodes=3), "max_episodes")
            resumed = Trainer.open(root / "split")
            self.assertEqual(resumed.run(max_episodes=6), "max_episodes")
            control = Trainer.open(root / "control", overrides, initial=spec)
            self.assertEqual(control.run(max_episodes=6), "max_episodes")
            self.assertEqual(resumed.summary(), control.summary())
            for name, tensor in control.network.state_dict().items():
                torch.testing.assert_close(resumed.network.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertTrue((root / "split" / "best-model.pt").exists())
            self.assertIsNotNone(resumed.progress.best_score)
            self.assertEqual(set(resumed.learner.replay.counts_by_players()), {1, 2, 3})

    def test_promotion_interrupt_resume_and_rejection_rebase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            seed_model = root / "seed.pt"
            save_checkpoint(str(seed_model), QNetwork(ModelSpec(16, 2, 1)), GameConfig(max_turns=60))
            base_overrides = {
                "player_counts": (2, 3, 4),
                "max_turns": 60,
                "promote": True,
                "cycle_episodes": 2,
                "round_episodes": 2,
                "updates_per_episode": 1,
                "batch_size": 2,
                "checkpoint_every": 1,
                "rebase_after_rejections": 2,
                "seed": 44,
            }
            run_dir = root / "promote"
            calls = []

            def interrupted_then_accept(**kwargs):
                calls.append((kwargs["phase"], kwargs["seed"]))
                if len(calls) in (1, 3):
                    raise GateInterrupted("deadline")
                return _synthetic_gate_result(positive=True, **kwargs)

            with mock.patch(
                "src.training.train.evaluate_gate", side_effect=interrupted_then_accept,
            ):
                trainer = Trainer.open(run_dir, base_overrides, initial=seed_model)
                self.assertEqual(trainer.run(max_cycles=1), "budget")
                self.assertIsInstance(trainer.progress.gate, Selecting)
                trainer = Trainer.open(run_dir)
                self.assertIsInstance(trainer.progress.gate, Selecting)
                self.assertEqual(trainer.run(max_cycles=1), "budget")
                self.assertIsInstance(trainer.progress.gate, Confirming)
                trainer = Trainer.open(run_dir)
                self.assertIsInstance(trainer.progress.gate, Confirming)
                self.assertEqual(trainer.run(max_cycles=1), "max_cycles")
                self.assertEqual(trainer.progress.promotions, 1)
                self.assertEqual(trainer.progress.cycle, 1)
                self.assertIsNone(trainer.progress.gate)
                select_seed = calls[1][1]
                confirm_seed = calls[3][1]
                self.assertEqual(confirm_seed - select_seed, disjoint_seed_gap(trainer.config))

                promoted_champ_id = trainer.registry.champion().id
                with mock.patch(
                    "src.training.train.evaluate_gate",
                    side_effect=lambda **kw: _synthetic_gate_result(positive=False, **kw),
                ):
                    self.assertEqual(trainer.run(max_cycles=3), "max_cycles")
                self.assertEqual(trainer.progress.rejections, 0)
                champ_net = QNetwork(ModelSpec(16, 2, 1))
                champ_state = torch.load(
                    trainer.registry.model_path(promoted_champ_id), weights_only=True,
                )["model_state_dict"]
                champ_net.load_state_dict(champ_state)
                for name, tensor in champ_net.state_dict().items():
                    torch.testing.assert_close(
                        trainer.network.state_dict()[name], tensor, rtol=0, atol=0,
                    )

    def test_cli_one_shot_checkpoint_and_output_dir_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt = root / "fresh.pt"
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                code = cli_main([
                    "--checkpoint", str(ckpt),
                    "--max-episodes", "0",
                    "--width", "16", "--heads", "2", "--blocks", "1",
                ])
            self.assertEqual(code, 0)
            self.assertTrue(ckpt.exists())
            self.assertEqual(json.loads(stdout.getvalue())["episodes"], 0)

            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout_timed:
                timed_ckpt = root / "timed.pt"
                self.assertEqual(cli_main([
                    "--checkpoint", str(timed_ckpt),
                    "--cards-in-hand", "3",
                    "--required-sequences", "1",
                    "--max-turns", "1",
                    "--round-episodes", "2",
                    "--max-seconds", "0.2",
                    "--width", "16", "--heads", "2", "--blocks", "1",
                ]), 0)
            self.assertGreater(json.loads(stdout_timed.getvalue())["episodes"], 0)
            self.assertTrue(timed_ckpt.exists())

            with mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(cli_main([
                    "--checkpoint", str(root / "bad-cycles.pt"),
                    "--max-cycles", "1",
                    "--width", "16", "--heads", "2", "--blocks", "1",
                ]), 2)

            out_dir = root / "run"
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(cli_main([
                    "--output-dir", str(out_dir),
                    "--cards-in-hand", "3",
                    "--required-sequences", "1",
                    "--max-turns", "1",
                    "--max-episodes", "2",
                    "--round-episodes", "2",
                    "--width", "16", "--heads", "2", "--blocks", "1",
                ]), 0)
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout2:
                self.assertEqual(cli_main([
                    "--output-dir", str(out_dir),
                    "--max-episodes", "4",
                ]), 0)
            self.assertEqual(json.loads(stdout2.getvalue())["episodes"], 4)

    def test_pool_pruning_and_torch_rng_survive_resume_without_resurrection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game = GameConfig(num_decks=1, cards_in_hand=3, required_sequences=1, max_turns=2)
            opps: list[Path] = []
            for idx in range(MAX_OPPONENT_POOL + 3):
                torch.manual_seed(200 + idx)
                path = root / f"opp-{idx}.pt"
                save_checkpoint(str(path), QNetwork(ModelSpec(16, 2, 1)), config=game)
                opps.append(path)
            run_dir = root / "pool_run"
            trainer = Trainer.open(
                run_dir,
                {
                    "num_decks": 1,
                    "cards_in_hand": 3,
                    "required_sequences": 1,
                    "max_turns": 2,
                    "player_counts": (2,),
                    "round_episodes": 2,
                    "updates_per_episode": 0,
                    "validation_games": 0,
                },
                initial=ModelSpec(16, 2, 1),
                opponents=opps,
            )
            self.assertEqual(len(trainer.pool_order), MAX_OPPONENT_POOL)
            trainer.run(max_episodes=2)
            saved_order = list(trainer.pool_order)
            expected_next = torch.rand(1).item()
            resumed = Trainer.open(run_dir)
            self.assertEqual(resumed.pool_order, saved_order)
            self.assertAlmostEqual(torch.rand(1).item(), expected_next, places=7)


if __name__ == "__main__":
    unittest.main()
