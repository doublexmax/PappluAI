from __future__ import annotations

import copy
from pathlib import Path
import io
import tempfile
import unittest
from unittest import mock

from collections import Counter
from src.game.environment import GameConfig, Phase, discard_action
from src.game.multiplayer import MultiplayerEnv
from src.training.league import (
    LeagueConfig,
    LeagueSession,
    save_session_checkpoint,
)
from src.cli.league import main

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


def fake_match(
    policies, config, seed, record_trajectories=False, deadline=None, *, record_terminal=False,
):
    from types import SimpleNamespace

    env = MultiplayerEnv(config)
    env.reset(seed)
    trajectories = []
    for seat in range(config.players):
        env._current_seat = seat
        trajectories.append(((env.observe(), 0, 0.0),))
    return SimpleNamespace(
        seed=seed,
        winner=0,
        terminal_reason="winner",
        seat_turns=tuple(1 for _ in policies),
        action_count=len(policies),
        stock_remaining=100,
        trajectories=tuple(trajectories),
        terminal_snapshot=None,
    )


@unittest.skipIf(torch is None, TORCH_REASON)
class TestLeagueSession(unittest.TestCase):
    def make_model(self, path: Path, max_turns: int = 60) -> None:
        from src.model.network import QNetwork, save_checkpoint

        save_checkpoint(
            str(path),
            QNetwork(),
            GameConfig(max_turns=max_turns),
        )

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_match_updates_only_challenger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.pt"
            opponent = root / "opponent.pt"
            self.make_model(champion)
            self.make_model(opponent)
            session = LeagueSession(
                str(champion),
                [str(opponent)],
                config=LeagueConfig(
                    batch_size=1,
                    updates_per_match=1,
                ),
            )
            learner_before = copy.deepcopy(session.network.state_dict())
            frozen_before = [
                copy.deepcopy(entry.policy.network.state_dict())
                for entry in session._opponents
                if hasattr(entry.policy, "network")
            ]
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                result = session.train_match()
            self.assertTrue(result["updated"])
            self.assertTrue(
                any(
                    not torch.equal(learner_before[name], value)
                    for name, value in session.network.state_dict().items()
                )
            )
            frozen_after = [
                entry.policy.network.state_dict()
                for entry in session._opponents
                if hasattr(entry.policy, "network")
            ]
            for before, after in zip(frozen_before, frozen_after):
                self.assert_nested_equal(before, after)

    def test_resume_reproduces_next_match_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.pt"
            self.make_model(champion, max_turns=30)
            config = LeagueConfig(batch_size=1, updates_per_match=1)
            control = LeagueSession(str(champion), config=config, seed=91)
            split = LeagueSession(str(champion), config=config, seed=91)
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                for _ in range(5):
                    control.train_match()
                    split.train_match()
            path = root / "state.pt"
            torch.save(split.state_dict(), path)
            resumed = LeagueSession.load(
                str(path),
                str(champion),
                expected_config=config,
            )
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                control_result = control.train_match()
                resumed_result = resumed.train_match()
            self.assertEqual(control_result, resumed_result)
            self.assert_nested_equal(
                control.network.state_dict(), resumed.network.state_dict()
            )
            self.assert_nested_equal(
                control.optimizer.state_dict(),
                resumed.optimizer.state_dict(),
            )
            self.assertEqual(
                control.replay.state_dict(), resumed.replay.state_dict()
            )
            self.assert_nested_equal(
                control.state_dict()["rng_states"],
                resumed.state_dict()["rng_states"],
            )

    def test_interrupted_match_rewinds_without_update(self):
        from src.evaluation.arena import MatchInterrupted

        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), seed=12)
            before = session.state_dict()
            with mock.patch(
                "src.training.league.play_match",
                side_effect=MatchInterrupted("deadline"),
            ):
                with self.assertRaises(MatchInterrupted):
                    session.train_match(deadline=0.0)
            after = session.state_dict()
            self.assert_nested_equal(before, after)
            self.assertTrue(session.at_match_boundary)

    def test_match_without_challenger_action_is_counted_without_update(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion))

            def no_action(policies, config, seed, **kwargs):
                return SimpleNamespace(
                    seed=seed,
                    winner=1,
                    terminal_reason="winner",
                    trajectories=tuple(() for _ in policies),
                )

            with mock.patch("src.training.league.play_match", side_effect=no_action):
                result = session.train_match()
            self.assertFalse(result["updated"])
            self.assertEqual(session.completed_matches, 1)
            self.assertEqual(session.no_action_matches, 1)
            self.assertEqual(session.total_updates, 0)
            self.assertEqual(len(session.replay), 0)

    def test_match_outcome_overwrites_recorded_rewards_with_discounted_targets(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            for winner, expected in ((0, [0.5, 1.0]), (1, [0.0, 0.0]), (None, [0.0, 0.0])):
                with self.subTest(winner=winner):
                    session = LeagueSession(str(champion), config=LeagueConfig(
                        game=GameConfig(max_turns=1), player_counts=(2,),
                        batch_size=1, updates_per_match=1, gamma=0.5,
                    ))

                    def recorded_match(policies, config, seed, **kwargs):
                        env = MultiplayerEnv(config)
                        env.reset(seed)
                        trajectories = [[] for _ in policies]
                        with mock.patch("src.game.multiplayer.hand_reward", return_value=0.0):
                            while not env.view().done:
                                seat = env.view().current_seat
                                observation = env.observe()
                                action = (
                                    0 if observation.phase is Phase.DRAW
                                    else discard_action(next(face for face, count in enumerate(observation.hand) if count))
                                )
                                trajectories[seat].append((observation, action, 1.0))
                                env.step(action)
                        return SimpleNamespace(
                            seed=seed, winner=winner, trajectories=trajectories,
                            terminal_reason="turns_exhausted" if winner is None else "win",
                            seat_turns=tuple(config.game.max_turns - n for n in env.view().turns_remaining),
                            terminal_snapshot=None,
                        )

                    with mock.patch("src.training.league.play_match", side_effect=recorded_match):
                        result = session.train_match()
                    self.assertEqual(result["challenger_seat"], 0)
                    transitions = session.replay.state_dict()["episodes"][-1]["transitions"]
                    self.assertEqual([step[3] for step in transitions], expected)

    def test_changed_opponent_pool_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.pt"
            first = root / "first.pt"
            second = root / "second.pt"
            self.make_model(champion)
            self.make_model(first)
            self.make_model(second)
            original = LeagueSession(str(champion), [str(first)])
            path = root / "state.pt"
            torch.save(original.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "opponent pool"):
                LeagueSession.load(
                    str(path),
                    str(champion),
                    [str(second)],
                )

    def test_generation_refresh_freezes_new_pool_and_preserves_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.pt"
            first = root / "first.pt"
            second = root / "second.pt"
            self.make_model(champion)
            self.make_model(first)
            self.make_model(second)
            session = LeagueSession(
                str(champion),
                [str(first)],
                config=LeagueConfig(batch_size=1, updates_per_match=1),
            )
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                session.train_match()
            replay_before = session.replay.state_dict()
            hashes_before = session.opponent_hashes

            session.refresh_opponent_pool([str(second)])

            self.assertEqual(session.opponent_pool_generation, 1)
            self.assertNotEqual(session.opponent_hashes, hashes_before)
            self.assertEqual(session.replay.state_dict(), replay_before)
            state = session.state_dict()
            self.assertTrue(state["replay_preserved_across_pool_refresh"])
            self.assertEqual(state["opponent_pool_generation"], 1)

            session.refresh_opponent_pool([str(second)])
            self.assertEqual(session.opponent_pool_generation, 1)
            self.assertEqual(session.replay.state_dict(), replay_before)

    def test_invalid_config_resume_state_and_nonfinite_optimizer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.pt"
            from src.model.network import QNetwork, save_checkpoint

            with self.assertRaisesRegex(ValueError, "21 cards"):
                LeagueConfig(game=GameConfig(cards_in_hand=18))

            save_checkpoint(
                str(wrong),
                QNetwork(),
                GameConfig(cards_in_hand=18, required_sequences=5),
            )
            with self.assertRaisesRegex(ValueError, "21 cards"):
                LeagueSession(str(wrong))

            valid = root / "valid.pt"
            self.make_model(valid)
            session = LeagueSession(str(valid))
            state_path = root / "state.pt"
            torch.save(session.state_dict(), state_path)
            with self.assertRaisesRegex(ValueError, "requested config"):
                LeagueSession.load(
                    str(state_path),
                    str(valid),
                    expected_config=LeagueConfig(batch_size=1),
                )

            payload = session.state_dict()
            parameter = next(iter(payload["optimizer_state_dict"]["state"].values()), None)
            if parameter is None:
                with mock.patch("src.training.league.play_match", side_effect=fake_match):
                    session.train_match()
                payload = session.state_dict()
                parameter = next(
                    iter(payload["optimizer_state_dict"]["state"].values())
                )
            parameter["exp_avg"].fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                session.load_state_dict(payload)

    def test_invalid_replay_is_rejected_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(
                str(champion),
                config=LeagueConfig(
                    player_counts=(2,),
                    batch_size=1,
                    updates_per_match=1,
                ),
            )
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                session.train_match()
            network_before = copy.deepcopy(session.network.state_dict())
            payload = copy.deepcopy(session.state_dict())
            journal = payload["replay_state_dict"]["episodes"][0]
            episode = list(journal["transitions"])
            state, prefix, _action, target = episode[0]
            episode[0] = (state, prefix, 54, target)
            journal["transitions"] = tuple(episode)
            first_parameter = next(
                iter(payload["network_state_dict"].values())
            )
            first_parameter.add_(1.0)

            with self.assertRaisesRegex(ValueError, "0..53"):
                session.load_state_dict(payload)
            self.assert_nested_equal(
                session.network.state_dict(),
                network_before,
            )

    def test_checkpoint_exports_resumable_and_model_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion = root / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion))
            output = root / "output"
            save_session_checkpoint(session, output)
            self.assertTrue((output / "latest-state.pt").is_file())
            self.assertTrue((output / "latest-model.pt").is_file())
            resumed = LeagueSession.load(
                str(output / "latest-state.pt"),
                str(champion),
            )
            self.assertEqual(
                resumed.state_dict()["counters"],
                session.state_dict()["counters"],
            )

    def test_count_and_seat_schedules_are_independently_balanced(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(
                str(champion), config=LeagueConfig(batch_size=1, updates_per_match=1),
            )
            with mock.patch("src.training.league.play_match", side_effect=fake_match):
                results = [session.train_match() for _ in range(35)]
                self.assertEqual(session.count_matches, Counter(result["players"] for result in results))
                results.append(session.train_match())
            self.assertEqual(Counter(result["players"] for result in results), {2: 12, 3: 12, 4: 12})
            for players in (2, 3, 4):
                counts = Counter(result["challenger_seat"] for result in results if result["players"] == players)
                self.assertEqual(counts, {seat: 12 // players for seat in range(players)})
            self.assertEqual(session.count_matches, {2: 12, 3: 12, 4: 12})


class TestLeagueCli(unittest.TestCase):
    @unittest.skipIf(torch is None, TORCH_REASON)
    def test_resume_accepts_partial_matching_options_and_rejects_changes_before_publish(self):
        from src.model.network import QNetwork, save_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            save_checkpoint(str(model), QNetwork(), GameConfig())
            session = LeagueSession(str(model), config=LeagueConfig(
                game=GameConfig(max_turns=1), batch_size=2, updates_per_match=1,
            ), seed=73)
            output = root / "league"
            save_session_checkpoint(session, output)
            args = [
                "--initial-model", str(model), "--resume", str(output / "latest-state.pt"),
                "--output-dir", str(output), "--matches", "0", "--max-seconds", "0",
            ]
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main([
                    *args, "--players", "2", "3", "4", "--updates-per-match", "1",
                ]), 0)
            before = {name: (output / name).read_bytes() for name in ("latest-state.pt", "latest-model.pt")}
            for options in (
                ["--players", "2"], ["--players", "3", "2", "4"],
                ["--players", "2", "3", "3", "4"], ["--max-turns", "60"],
                ["--batch-size", "64"], ["--seed", "74"],
                ["--updates-per-match", "2"],
            ):
                with self.subTest(options=options), mock.patch(
                    "sys.stderr", new_callable=io.StringIO,
                ) as error:
                    self.assertEqual(main([*args, *options]), 2)
                    self.assertIn("resume cannot change", error.getvalue())
                    self.assertTrue(error.getvalue().startswith("error: "))
                self.assertEqual({name: (output / name).read_bytes() for name in before}, before)


if __name__ == "__main__":
    unittest.main()
