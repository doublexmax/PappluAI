import copy
from collections import Counter
from dataclasses import replace
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import torch

from src.cli.league import main
from src.evaluate import HandPenalty
from src.evaluation.arena import MatchInterrupted, run_matches
from src.game.environment import GameConfig
from src.game.outcome import DealOutcome
from src.model.network import QNetwork, save_checkpoint
from src.model.search import TeacherConfig
from src.model.spec import ModelSpec
from src.training.league import LeagueConfig, LeagueSession, save_session_checkpoint


class TestLeagueSession(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_model(self, path, max_turns=60):
        save_checkpoint(str(path), QNetwork(ModelSpec(8, 1, 1)), GameConfig(max_turns=max_turns))

    def config(self, **overrides):
        return LeagueConfig(**{
            "game": GameConfig(max_turns=1), "batch_size": 2, "updates_per_match": 1,
            "teacher": TeacherConfig(fraction=0), **overrides,
        })

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

    def test_round_updates_only_challenger_and_batches_all_requested_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion, opponent = Path(temporary) / "champion.pt", Path(temporary) / "opponent.pt"
            self.make_model(champion)
            self.make_model(opponent)
            session = LeagueSession(str(champion), [str(opponent)], config=self.config())
            before = copy.deepcopy(session.network.state_dict())
            frozen = [
                copy.deepcopy(entry.policy.network.state_dict()) for entry in session._opponents
                if hasattr(entry.policy, "network")
            ]
            with mock.patch("src.training.league.run_matches", wraps=run_matches) as runner:
                results = session.train_round(4)
            self.assertEqual([r["match"] for r in results], [1, 2, 3, 4])
            self.assertEqual([r["players"] for r in results], [2, 3, 4, 2])
            self.assertEqual(len(runner.call_args.args[0]), 4)
            self.assertEqual((session.total_updates, session.update_matches), (4, 4))
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in session.network.state_dict().items()))
            after = [entry.policy.network.state_dict() for entry in session._opponents if hasattr(entry.policy, "network")]
            self.assert_nested_equal(frozen, after)
            self.assertEqual(sum(row[3] for row in session.scheduler.state_dict()["scores"]), 7)

    def test_resume_reproduces_next_round_target_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion, max_turns=30)
            config = self.config(teacher=TeacherConfig(max_children=8))
            control = LeagueSession(str(champion), config=config, seed=91)
            split = LeagueSession(str(champion), config=config, seed=91)
            self.assertEqual(control.train_round(5), split.train_round(5))
            path = Path(temporary) / "state.pt"
            torch.save(split.state_dict(), path)
            resumed = LeagueSession.load(str(path), str(champion), expected_config=config)
            self.assertEqual(control.train_round(4), resumed.train_round(4))
            self.assert_nested_equal(control.state_dict(), resumed.state_dict())

    def test_interrupted_round_rewinds_every_rng_without_learning_or_scheduling(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), config=self.config(), seed=12)
            before = session.state_dict()
            with self.assertRaises(MatchInterrupted):
                session.train_round(8, deadline=0.0)
            self.assert_nested_equal(session.state_dict(), before)
            self.assertTrue(session.at_match_boundary)

    def test_match_without_challenger_action_is_counted_without_an_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), config=self.config())

            def no_action(*args, **kwargs):
                return [
                    replace(result, trajectories=tuple(() for _ in result.trajectories))
                    for result in run_matches(*args, **kwargs)
                ]

            with mock.patch("src.training.league.run_matches", side_effect=no_action):
                result = session.train_match()
            self.assertFalse(result["updated"])
            self.assertEqual((session.completed_matches, session.no_action_matches, session.total_updates), (1, 1, 0))
            self.assertEqual(len(session.replay), 0)

    def test_terminal_placement_and_penalty_are_used_instead_of_sparse_declaration_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            for points, expected in (((10, 50), 1.0), ((50, 10), -1.0), ((30, 30), 0.0)):
                with self.subTest(points=points):
                    session = LeagueSession(str(champion), config=self.config(player_counts=(2,), gamma=0.5))
                    outcome = DealOutcome(None, "turns_exhausted", tuple(
                        HandPenalty(value, (), (0,) * 52) for value in points
                    ))

                    def placed(*args, **kwargs):
                        return [replace(result, outcome=outcome) for result in run_matches(*args, **kwargs)]

                    with mock.patch("src.training.league.run_matches", side_effect=placed):
                        result = session.train_match()
                    self.assertEqual(result["challenger_seat"], 0)
                    self.assertEqual(session.replay.state_dict()["trajectories"][0]["outcome"], (expected, False, points[0]))
                    samples = session.replay.sample(random.Random(3), 100, gamma=0.5)
                    self.assertEqual({s.outcome.utility for s in samples}, {expected})
                    self.assertEqual({s.scale for s in samples}, {0.5, 1.0})
                    self.assertTrue(all(s.bootstrap is None for s in samples))

    def test_changed_opponent_pool_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion, first, second = [root / name for name in ("champion.pt", "first.pt", "second.pt")]
            for path in (champion, first, second):
                self.make_model(path)
            original = LeagueSession(str(champion), [str(first)], config=self.config())
            path = root / "state.pt"
            torch.save(original.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "opponent pool"):
                LeagueSession.load(str(path), str(champion), [str(second)])

    def test_pool_refresh_preserves_replay_and_pairwise_scores_for_returning_opponents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            champion, first, second = [root / name for name in ("champion.pt", "first.pt", "second.pt")]
            for path in (champion, first, second):
                self.make_model(path)
            session = LeagueSession(str(champion), [str(first)], config=self.config())
            session.train_round(3)
            replay, scheduler = session.replay.state_dict(), session.scheduler.state_dict()
            hashes = session.opponent_hashes
            session.refresh_opponent_pool([str(second)])
            self.assertNotEqual(session.opponent_hashes, hashes)
            self.assertEqual(session.opponent_pool_generation, 1)
            self.assertEqual(session.replay.state_dict(), replay)
            self.assertEqual(session.scheduler.state_dict(), scheduler)
            session.refresh_opponent_pool([str(second)])
            self.assertEqual(session.opponent_pool_generation, 1)
            session.refresh_opponent_pool([str(first)])
            self.assertEqual(session.scheduler.state_dict(), scheduler)

    def test_rebase_adopts_online_and_target_and_resets_only_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion, external = Path(temporary) / "champion.pt", Path(temporary) / "external.pt"
            self.make_model(champion)
            self.make_model(external)
            session = LeagueSession(str(champion), config=self.config())
            session.train_match()
            replay, scheduler = session.replay.state_dict(), session.scheduler.state_dict()
            counters = session.state_dict()["counters"]
            self.assertTrue(session.follow_champion(str(external)))
            weights = torch.load(external, weights_only=True)["model_state_dict"]
            self.assert_nested_equal(session.learner.online.state_dict(), weights)
            self.assert_nested_equal(session.learner.target.state_dict(), weights)
            self.assertEqual(session.optimizer.state_dict()["state"], {})
            self.assertEqual(session.replay.state_dict(), replay)
            self.assertEqual(session.scheduler.state_dict(), scheduler)
            self.assertEqual(session.state_dict()["counters"], counters)

    def test_invalid_config_and_nonfinite_optimizer_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong, valid = Path(temporary) / "wrong.pt", Path(temporary) / "valid.pt"
            with self.assertRaisesRegex(ValueError, "21 cards"):
                LeagueConfig(game=GameConfig(cards_in_hand=18))
            save_checkpoint(str(wrong), QNetwork(ModelSpec(8, 1, 1)), GameConfig(cards_in_hand=18, required_sequences=5))
            with self.assertRaisesRegex(ValueError, "21 cards"):
                LeagueSession(str(wrong))
            self.make_model(valid)
            session = LeagueSession(str(valid), config=self.config())
            session.train_match()
            payload = session.state_dict()
            next(iter(payload["learner"]["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
            before = session.state_dict()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                session.load_state_dict(payload)
            self.assert_nested_equal(session.state_dict(), before)

    def test_invalid_replay_and_scheduler_are_rejected_before_model_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), config=self.config())
            session.train_match()
            before = session.state_dict()
            for field in ("replay", "scheduler"):
                payload = copy.deepcopy(before)
                next(iter(payload["learner"]["online"].values())).add_(1)
                if field == "replay":
                    journal = payload["learner"]["replay"]["trajectories"][0]
                    snapshot, prefix, _, hidden = journal["decisions"][0]
                    journal["decisions"] = ((snapshot, prefix, 54, hidden),)
                else:
                    payload["scheduler"]["scores"] = ((2, "random", 10.0, 1),)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    session.load_state_dict(payload)
                self.assert_nested_equal(session.state_dict(), before)

    def test_checkpoint_exports_resumable_and_model_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), config=self.config())
            output = Path(temporary) / "output"
            save_session_checkpoint(session, output)
            resumed = LeagueSession.load(str(output / "latest-state.pt"), str(champion))
            self.assert_nested_equal(resumed.state_dict(), session.state_dict())
            with self.assertRaisesRegex(ValueError, "requested config"):
                LeagueSession.load(str(output / "latest-state.pt"), str(champion), expected_config=self.config(batch_size=1))

    def test_count_and_seat_schedules_are_independently_balanced_across_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            champion = Path(temporary) / "champion.pt"
            self.make_model(champion)
            session = LeagueSession(str(champion), config=self.config())
            results = session.train_round(16) + session.train_round(16) + session.train_round(4)
            self.assertEqual(Counter(result["players"] for result in results), {2: 12, 3: 12, 4: 12})
            for players in (2, 3, 4):
                counts = Counter(result["challenger_seat"] for result in results if result["players"] == players)
                self.assertEqual(counts, {seat: 12 // players for seat in range(players)})
            self.assertEqual(session.count_matches, {2: 12, 3: 12, 4: 12})


class TestLeagueCli(unittest.TestCase):
    def test_resume_accepts_matching_options_and_rejects_changes_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            save_checkpoint(str(model), QNetwork(ModelSpec(8, 1, 1)), GameConfig())
            session = LeagueSession(str(model), config=LeagueConfig(
                game=GameConfig(max_turns=1), batch_size=2, updates_per_match=1, matches_per_round=8,
            ), seed=73)
            output = root / "league"
            save_session_checkpoint(session, output)
            args = [
                "--initial-model", str(model), "--resume", str(output / "latest-state.pt"),
                "--output-dir", str(output), "--matches", "0", "--max-seconds", "0",
            ]
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main([*args, "--players", "2", "3", "4", "--updates-per-match", "1",
                                       "--matches-per-round", "8"]), 0)
            before = {name: (output / name).read_bytes() for name in ("latest-state.pt", "latest-model.pt")}
            for options in (
                ["--players", "2"], ["--players", "3", "2", "4"], ["--players", "2", "3", "3", "4"],
                ["--max-turns", "60"], ["--batch-size", "64"], ["--seed", "74"],
                ["--updates-per-match", "2"], ["--matches-per-round", "4"],
            ):
                with self.subTest(options=options), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                    self.assertEqual(main([*args, *options]), 2)
                    self.assertIn("resume cannot change", error.getvalue())
                self.assertEqual({name: (output / name).read_bytes() for name in before}, before)


if __name__ == "__main__":
    unittest.main()
