from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.environment import GameConfig, STATE_DIM
from src.league_train import (
    LeagueConfig,
    LeagueSession,
    build_arg_parser,
    save_session_checkpoint,
)

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


def fake_match(policies, config, seed, record_trajectories=False, deadline=None):
    from types import SimpleNamespace

    step = ((0.0,) * STATE_DIM, 0, 0.0)
    return SimpleNamespace(
        seed=seed,
        winner=0,
        terminal_reason="winner",
        seat_turns=tuple(1 for _ in policies),
        action_count=len(policies),
        stock_remaining=100,
        trajectories=tuple((step,) for _ in policies),
    )


@unittest.skipIf(torch is None, TORCH_REASON)
class TestLeagueSession(unittest.TestCase):
    def make_model(self, path: Path, max_turns: int = 60) -> None:
        from src.model import QNetwork, save_checkpoint

        save_checkpoint(
            str(path),
            QNetwork(architecture="suit_conv"),
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

    def test_config_is_full21_shared_deck_scope(self):
        config = LeagueConfig()
        self.assertEqual(config.game, GameConfig(max_turns=60))
        self.assertEqual(config.player_counts, (2, 3))
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.replay_capacity, 10_000)

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
            with mock.patch("src.arena.play_match", side_effect=fake_match):
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
            with mock.patch("src.arena.play_match", side_effect=fake_match):
                control.train_match()
                split.train_match()
            path = root / "state.pt"
            torch.save(split.state_dict(), path)
            resumed = LeagueSession.load(
                str(path),
                str(champion),
                expected_config=config,
            )
            with mock.patch("src.arena.play_match", side_effect=fake_match):
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
        from src.arena import MatchInterrupted

        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), seed=12)
            before = session.state_dict()
            with mock.patch(
                "src.arena.play_match",
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

            with mock.patch("src.arena.play_match", side_effect=no_action):
                result = session.train_match()
            self.assertFalse(result["updated"])
            self.assertEqual(session.completed_matches, 1)
            self.assertEqual(session.no_action_matches, 1)
            self.assertEqual(session.total_updates, 0)
            self.assertEqual(len(session.replay), 0)

    def test_opponent_win_overwrites_challenger_terminal_reward_with_zero(self):
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
            with mock.patch("src.arena.play_match", side_effect=fake_match):
                session.train_match()
                result = session.train_match()
            self.assertEqual(result["challenger_seat"], 1)
            newest = session.replay.state_dict()["episodes"][-1]
            self.assertEqual(newest[-1][2], 0.0)

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
            with mock.patch("src.arena.play_match", side_effect=fake_match):
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

    def test_invalid_rules_and_nonfinite_optimizer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.pt"
            from src.model import QNetwork, save_checkpoint

            save_checkpoint(
                str(wrong),
                QNetwork(architecture="suit_conv"),
                GameConfig(cards_in_hand=18, required_sequences=5),
            )
            with self.assertRaisesRegex(ValueError, "21 cards"):
                LeagueSession(str(wrong))

            valid = root / "valid.pt"
            self.make_model(valid)
            session = LeagueSession(str(valid))
            payload = session.state_dict()
            parameter = next(iter(payload["optimizer_state_dict"]["state"].values()), None)
            if parameter is None:
                with mock.patch("src.arena.play_match", side_effect=fake_match):
                    session.train_match()
                payload = session.state_dict()
                parameter = next(
                    iter(payload["optimizer_state_dict"]["state"].values())
                )
            parameter["exp_avg"].fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                session.load_state_dict(payload)

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

    def test_remote_actual_two_player_match_updates_complete_boundary(self):
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
            result = session.train_match()
            self.assertEqual(result["players"], 2)
            self.assertEqual(session.completed_matches, 1)
            self.assertTrue(session.at_match_boundary)


class TestLeagueCli(unittest.TestCase):
    def test_cli_exposes_bounded_resume_contract(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--initial-model",
                "champion.pt",
                "--opponent",
                "random",
                "--output-dir",
                "out",
                "--matches",
                "2",
                "--max-seconds",
                "10",
                "--resume",
                "state.pt",
            ]
        )
        self.assertEqual(args.matches, 2)
        self.assertEqual(args.opponent, ["random"])
        self.assertEqual(args.resume, "state.pt")


if __name__ == "__main__":
    unittest.main()
