from __future__ import annotations

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import src.training.improve as improve_runtime
from src.training.improve import (
    ALGORITHM,
    ImproveConfig,
    ImprovementController,
)
from src.cli.improve import build_arg_parser
from src.checkpoints.io import file_sha256

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


class TestMetricHistory(unittest.TestCase):
    def test_initial_evidence_requires_current_schema_and_matching_rules(self):
        from src.training.improve import _initial_evidence

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.pt"
            model.write_bytes(b"same frozen weights")
            rules = {
                "num_decks": 3, "cards_in_hand": 21, "required_sequences": 5,
                "max_turns": 60, "recycle_discard": True,
            }
            result = {
                "encoding_version": 2, "checkpoint_version": 4, "architecture": "axial_table",
                "checkpoint_sha256": sha256(model),
                "game_config": rules,
                "greedy": {"games": 2, "wins": 1, "outcomes": [1, 0]},
            }
            path = root / "final-best.json"
            path.write_text(json.dumps(result))
            evidence = _initial_evidence(model, current_recycle_discard=True)
            verified = evidence["current_solo_reference"]
            self.assertEqual(verified["win_rate"], 0.5)
            self.assertTrue(verified["recycle_discard"])
            for bad in (
                {**result, "encoding_version": 1},
                {**result, "game_config": {**rules, "recycle_discard": False}},
            ):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    _initial_evidence(model, current_recycle_discard=True)

    def test_cached_event_index_avoids_rescanning_the_entire_log(self):
        from src.training.improve import _append_metric_once, _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identifiers = set()
            with mock.patch("src.training.improve.json.loads", side_effect=AssertionError("unexpected rescan")):
                _append_metric_once(output, "first", {"value": 1}, identifiers)
                _append_metric_once(output, "second", {"value": 2}, identifiers)
                _append_metric_once(output, "first", {"value": 99}, identifiers)
            self.assertEqual(_read_metric_ids(output / ".active-metrics.jsonl"), {"first", "second"})
            self.assertEqual(len((output / ".active-metrics.jsonl").read_text().splitlines()), 2)

    def test_malformed_history_is_reported_instead_of_silently_skipped(self):
        from src.training.improve import _read_metric_ids

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            path.write_text("{incomplete", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed"):
                _read_metric_ids(path)


def sha256(path):
    return file_sha256(Path(path))


def gate_result(accepted):
    def evaluate(**kwargs):
        from src.checkpoints.evidence import balanced_seat_orders, evidence_metrics

        blocks = kwargs["seed_blocks"]
        multiplayer = {
            str(players): {
                "candidate_minus_champion": 0.5 if accepted else 0.0,
                "games": blocks * len(balanced_seat_orders(players)),
                "seed_block_margins": [0.5 if accepted else 0.0] * blocks,
            }
            for players in kwargs["player_counts"]
        }
        solo = {
            "seed": kwargs["seed"] + 100_000,
            "games": kwargs["solo_games"],
            "candidate_outcomes": [0] * kwargs["solo_games"],
            "champion_outcomes": [0] * kwargs["solo_games"],
            "candidate_wins": 0,
            "champion_wins": 0,
        }
        samples = max(100, kwargs["samples"])
        return {
            "gate_schema_version": 2,
            "encoding_version": 2,
            "architecture": "axial_table",
            "candidate_sha256": sha256(kwargs["candidate_path"]),
            "champion_sha256": sha256(kwargs["champion_path"]),
            "accepted": accepted,
            **evidence_metrics(
                multiplayer, solo, blocks, kwargs["seed"], samples,
                player_counts=kwargs["player_counts"],
            ),
            "bootstrap_samples": samples,
            "seed": kwargs["seed"],
            "seed_blocks": kwargs["seed_blocks"],
            "player_counts": list(kwargs["player_counts"]),
            "rules": {
                "num_decks": 3,
                "cards_in_hand": 21,
                "required_sequences": 5,
                "max_turns": kwargs["max_turns"],
                "recycle_discard": kwargs["recycle_discard"],
            },
            "pool_hashes": [
                sha256(path) for path in kwargs["opponent_paths"]
            ],
            "raw_outcomes": (),
            "multiplayer": multiplayer,
            "solo": solo,
            "scope": {"player_counts": kwargs["player_counts"]},
        }

    return evaluate


class TestImproveConfig(unittest.TestCase):
    def test_rebase_threshold_requires_a_positive_integer(self):
        for value in (0, -1, True, 1.5, "8", None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ImproveConfig(rebase_after_rejections=value)
        config = ImproveConfig(rebase_after_rejections=2)
        self.assertEqual(
            ImproveConfig.from_dict(config.to_dict()).rebase_after_rejections, 2,
        )

    def test_specialist_scopes_roundtrip_without_expanding_to_master(self):
        for counts in ((2,), (3,), (4,), (2, 4), (2, 3, 4)):
            with self.subTest(counts=counts):
                config = ImproveConfig(player_counts=counts)
                restored = ImproveConfig.from_dict(config.to_dict())
                self.assertEqual(restored.player_counts, counts)
                self.assertEqual(restored, config)

    def test_invalid_player_count_tuples_are_rejected(self):
        for counts in ((), (1,), (5,), (True,), (2.0,), ("2",), (2, 2), (3, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                ImproveConfig(player_counts=counts)
        for counts in (None, [2], "234", {2, 3, 4}):
            with self.subTest(counts=counts):
                with self.assertRaisesRegex(TypeError, "player_counts must be a tuple"):
                    ImproveConfig(player_counts=counts)
                raw = ImproveConfig().to_dict()
                raw["player_counts"] = counts
                with self.assertRaisesRegex(TypeError, "player_counts must be a tuple"):
                    ImproveConfig.from_dict(raw)


@unittest.skipIf(torch is None, TORCH_REASON)
class TestImprovementController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_torch_threads)

    def make_model(self, path, parameter_value=None):
        from src.game.environment import GameConfig
        from src.model.network import QNetwork, save_checkpoint
        from src.model.spec import ModelSpec

        network = QNetwork(ModelSpec(8, 1, 1))
        if parameter_value is not None:
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.fill_(parameter_value)
        save_checkpoint(
            str(path),
            network,
            GameConfig(max_turns=60),
            meta={"best_score": 164 / 256},
        )

    def make_controller(
        self, root, *, snapshots_dir=None, parameter_value=None, **config_overrides,
    ):
        model = root / "champion-source.pt"
        self.make_model(model, parameter_value=parameter_value)
        config = ImproveConfig(
            max_cycles=1,
            league_matches_per_cycle=1,
            selection_blocks=32,
            confirmation_blocks=128,
            solo_games=256,
            checkpoint_every=1,
            gate_samples=20,
        )
        return ImprovementController(
            model, root / "output", config=replace(config, **config_overrides),
            snapshots_dir=snapshots_dir,
        )

    def make_snapshot(self, directory):
        source = directory / "incoming.pt"
        self.make_model(source)
        destination = directory / ("model-" + sha256(source) + ".pt")
        source.rename(destination)
        return destination

    def prepare_distinct_candidate(self, controller):
        with torch.no_grad():
            next(iter(controller.league.network.parameters())).add_(0.01)
        controller.league_in_cycle = controller.config.league_matches_per_cycle
        controller._prepare_candidate()

    def replace_champion(self, controller, path):
        self.make_model(path, parameter_value=0.125)
        previous = controller.registry.champion()
        replacement = controller.registry.register(
            str(path), origin="frozen_opponent", parents=(previous.id,),
        )
        arguments = {
            "candidate_path": str(controller.registry.model_path(replacement.id)),
            "champion_path": str(controller.registry.model_path(previous.id)),
            "opponent_paths": controller._gate_opponent_paths(previous.id),
            "solo_games": controller.config.solo_games,
            "max_turns": controller.config.max_turns,
            "player_counts": controller.config.player_counts,
            "samples": controller.config.gate_samples,
            "recycle_discard": controller.config.recycle_discard,
        }
        selection = {
            **gate_result(True)(**arguments, seed=60_000_000, seed_blocks=32),
            "phase": "selection",
        }
        confirmation = {
            **gate_result(True)(**arguments, seed=60_500_000, seed_blocks=128),
            "phase": "confirmation",
            "selection_passed": True,
            "selection": selection,
            "selection_seed": 60_000_000,
        }
        return controller.registry.promote(
            replacement.id, expected_id=previous.id, evidence=confirmation,
            player_counts=controller.config.player_counts,
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

    def test_single_challenger_finetunes_gently(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            session = controller.league
            self.assertEqual(session.optimizer.param_groups[0]["lr"], 2e-5)
            self.assertEqual(session.current_epsilon, 0.04)
            session.epsilon_progress = session.config.epsilon_decay_matches
            self.assertAlmostEqual(session.current_epsilon, 0.01)
            self.assertFalse(hasattr(controller, "research"))
            self.assertIsNone(controller.best_score)

    def test_self_promotion_preserves_warm_adam_and_runs_each_gate_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary), max_cycles=2, player_counts=(2,),
            )
            original_id = controller.registry.champion().id
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            self.assertEqual(controller.league.completed_matches, 1)
            self.assertEqual(controller.league.total_updates, 4)
            self.assertEqual(controller.phase, "selection")
            candidate_id = controller.candidate_id
            candidate = controller.registry.get(candidate_id)
            self.assertEqual(candidate.origin, "league")
            self.assertEqual(candidate.parents, (original_id,))
            self.assertEqual(controller.league.last_exported_sha256, candidate.sha256)
            network = controller.league.network
            optimizer = controller.league.optimizer
            replay = controller.league.replay
            weights = copy.deepcopy(network.state_dict())
            moments = copy.deepcopy(optimizer.state_dict())
            replay_state = copy.deepcopy(replay.state_dict())
            rng_states = copy.deepcopy(controller.league.state_dict()["rng_states"])
            self.assertNotEqual(moments["state"], {})
            self.assertTrue(any(
                torch.count_nonzero(state["exp_avg"]).item()
                for state in moments["state"].values()
            ))
            with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(True)):
                self.assertTrue(controller.step(deadline))
                self.assertEqual(controller.phase, "confirmation")
                self.assertEqual(controller.registry.champion().id, original_id)
                self.assertTrue(controller.step(deadline))
                self.assertEqual(controller.registry.champion().id, original_id)
                self.assertTrue(controller.step(deadline))
            self.assertEqual(controller.registry.champion().id, candidate_id)
            self.assertEqual(controller.promotions, 1)
            self.assertEqual(controller.consecutive_rejections, 0)
            self.assertEqual(controller.league.parent_sha256, candidate.sha256)
            self.assertEqual(controller.league.opponent_pool_generation, 1)
            self.assertIn(candidate_id, controller.opponent_ids)
            self.assertFalse(controller.step(deadline=0))
            self.assertIs(controller.league.network, network)
            self.assertIs(controller.league.optimizer, optimizer)
            self.assertIs(controller.league.replay, replay)
            self.assert_nested_equal(controller.league.network.state_dict(), weights)
            self.assert_nested_equal(controller.league.optimizer.state_dict(), moments)
            self.assert_nested_equal(controller.league.replay.state_dict(), replay_state)
            self.assert_nested_equal(controller.league.state_dict()["rng_states"], rng_states)
            metrics = [
                json.loads(line)
                for line in (controller.output_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [(event["phase"], event["candidate_origin"]) for event in metrics if event["event"] == "gate"],
                [("selection", "league"), ("confirmation", "league")],
            )
            pointer = json.loads(controller.registry.champion_path.read_text())
            self.assertEqual(pointer["generation"], 1)
            self.assertEqual(len(pointer["history"]), 1)
            self.assertEqual(pointer["history"][0]["evidence"]["candidate_origin"], "league")
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"), controller.output_dir,
            )
            self.assertEqual(resumed.league.parent_sha256, candidate.sha256)
            self.assert_nested_equal(resumed.league.optimizer.state_dict(), moments)
            self.assertEqual(resumed.league.opponent_hashes, controller.league.opponent_hashes)

    def test_external_weights_rebase_once_and_preserve_replay_and_rngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.make_controller(root, player_counts=(2,))
            self.assertEqual(controller.league.train_match()["match"], 1)
            league = controller.league
            warm_optimizer = league.optimizer
            self.assertNotEqual(warm_optimizer.state_dict()["state"], {})
            replay = league.replay
            replay_state = copy.deepcopy(replay.state_dict())
            rng_states = copy.deepcopy(league.state_dict()["rng_states"])
            champion = root / "external.pt"
            self.make_model(champion, parameter_value=0.125)
            global_rng = torch.get_rng_state().clone()
            self.assertTrue(league.follow_champion(str(champion)))
            self.assertEqual(league.parent_sha256, sha256(champion))
            self.assertEqual(league._parent_path, str(champion))
            for parameter in league.network.parameters():
                self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)))
            for parameter in league.learner.target.parameters():
                self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)))
            self.assertIsNot(league.optimizer, warm_optimizer)
            self.assertEqual(league.optimizer.state_dict()["state"], {})
            self.assertEqual(league.optimizer.param_groups[0]["lr"], 2e-5)
            self.assertIs(league.replay, replay)
            self.assert_nested_equal(league.replay.state_dict(), replay_state)
            self.assert_nested_equal(league.state_dict()["rng_states"], rng_states)
            self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
            optimizer = league.optimizer
            weights = copy.deepcopy(league.network.state_dict())
            self.assertFalse(league.follow_champion(str(champion)))
            self.assertFalse(league.follow_champion(str(champion)))
            self.assertIs(league.optimizer, optimizer)
            self.assert_nested_equal(league.network.state_dict(), weights)
            self.assertEqual(league.optimizer.state_dict()["state"], {})

    def test_external_champion_at_cycle_start_rebases_and_resumes_from_its_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.make_controller(root, max_cycles=2, player_counts=(2,))
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(False)):
                self.assertTrue(controller.step(deadline))
            self.assertEqual(controller.cycle_index, 1)
            warm_optimizer = controller.league.optimizer
            replay = copy.deepcopy(controller.league.replay.state_dict())
            rngs = copy.deepcopy(controller.league.state_dict()["rng_states"])
            replacement = self.replace_champion(controller, root / "external.pt")
            self.assertFalse(controller.step(deadline=0))
            self.assertEqual(controller.league.parent_sha256, replacement.sha256)
            self.assertEqual(controller.consecutive_rejections, 0)
            self.assertIsNot(controller.league.optimizer, warm_optimizer)
            self.assertEqual(controller.league.optimizer.state_dict()["state"], {})
            for parameter in controller.league.network.parameters():
                self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)))
            self.assert_nested_equal(controller.league.replay.state_dict(), replay)
            self.assert_nested_equal(controller.league.state_dict()["rng_states"], rngs)
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"), controller.output_dir,
            )
            self.assert_nested_equal(resumed.state_dict(), controller.state_dict())
            self.assertEqual(resumed.registry.champion().id, replacement.id)

    def test_selection_and_confirmation_rejections_rebase_at_the_threshold(self):
        for rejected_phase in ("selection", "confirmation"):
            with self.subTest(phase=rejected_phase), tempfile.TemporaryDirectory() as temporary:
                controller = self.make_controller(
                    Path(temporary), max_cycles=3, player_counts=(2,),
                    rebase_after_rejections=2, parameter_value=0.125,
                )
                champion_id = controller.registry.champion().id
                deadline = time.monotonic() + 120

                def evaluate(**kwargs):
                    accepted = rejected_phase == "confirmation" and kwargs["seed_blocks"] == 32
                    return gate_result(accepted)(**kwargs)

                for cycle in range(2):
                    self.assertTrue(controller.step(deadline))
                    warm_optimizer = controller.league.optimizer
                    moments = copy.deepcopy(warm_optimizer.state_dict())
                    weights = copy.deepcopy(controller.league.network.state_dict())
                    replay = copy.deepcopy(controller.league.replay.state_dict())
                    rngs = copy.deepcopy(controller.league.state_dict()["rng_states"])
                    self.assertNotEqual(moments["state"], {})
                    with mock.patch("src.training.improve.evaluate_gate", side_effect=evaluate):
                        self.assertTrue(controller.step(deadline))
                        if rejected_phase == "confirmation":
                            self.assertTrue(controller.step(deadline))
                            self.assertTrue(controller.step(deadline))
                    self.assertEqual(controller.cycle_index, cycle + 1)
                    self.assertEqual(controller.registry.champion().id, champion_id)
                    self.assertEqual(controller.promotions, 0)
                    self.assert_nested_equal(controller.league.replay.state_dict(), replay)
                    self.assert_nested_equal(controller.league.state_dict()["rng_states"], rngs)
                    if cycle == 0:
                        self.assertEqual(controller.consecutive_rejections, 1)
                        self.assertIs(controller.league.optimizer, warm_optimizer)
                        self.assert_nested_equal(controller.league.optimizer.state_dict(), moments)
                        self.assert_nested_equal(controller.league.network.state_dict(), weights)
                    else:
                        self.assertEqual(controller.consecutive_rejections, 0)
                        self.assertIsNot(controller.league.optimizer, warm_optimizer)
                        self.assertEqual(controller.league.optimizer.state_dict()["state"], {})
                        for parameter in controller.league.network.parameters():
                            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)))
                champion_path = str(controller.registry.model_path(champion_id))
                self.assertFalse(controller.league.follow_champion(champion_path))
                self.assertFalse(controller.league.follow_champion(champion_path))
                self.assertEqual(controller.league.optimizer.state_dict()["state"], {})

    def test_load_recovers_a_crash_after_the_champion_pointer_was_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary), max_cycles=2, player_counts=(2,),
            )
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            candidate_id = controller.candidate_id
            candidate_sha256 = controller.registry.get(candidate_id).sha256
            with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(True)):
                self.assertTrue(controller.step(deadline))
                self.assertTrue(controller.step(deadline))
            moments = copy.deepcopy(controller.league.optimizer.state_dict())
            replay = copy.deepcopy(controller.league.replay.state_dict())
            promote = controller.registry.promote

            def promote_then_crash(*args, **kwargs):
                promote(*args, **kwargs)
                raise RuntimeError("injected crash after champion pointer write")

            with mock.patch.object(controller.registry, "promote", side_effect=promote_then_crash):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    controller.step(deadline)
            self.assertEqual(controller.registry.champion().id, candidate_id)
            self.assertEqual(controller.phase, "confirmation")
            self.assertEqual(controller.promotions, 0)
            state_path = controller.output_dir / "latest-state.pt"
            saved = torch.load(state_path, weights_only=True)
            self.assertEqual(saved["phase"], "confirmation")
            self.assertEqual(saved["evaluation"]["candidate_id"], candidate_id)
            for _ in range(2):
                resumed = ImprovementController.load(str(state_path), controller.output_dir)
                self.assertEqual(resumed.cycle_index, 1)
                self.assertEqual(resumed.phase, "training")
                self.assertEqual(resumed.promotions, 1)
                self.assertEqual(resumed.promoted_cycles, (0,))
                self.assertEqual(resumed.consecutive_rejections, 0)
                self.assertIsNone(resumed.candidate_id)
                self.assertEqual(resumed.league.parent_sha256, candidate_sha256)
                self.assert_nested_equal(resumed.league.optimizer.state_dict(), moments)
                self.assert_nested_equal(resumed.league.replay.state_dict(), replay)
            self.assertEqual(resumed.run(0), "budget")
            reloaded = ImprovementController.load(str(state_path), controller.output_dir)
            self.assertEqual(reloaded.cycle_index, 1)
            self.assertEqual(reloaded.promotions, 1)
            self.assert_nested_equal(reloaded.league.optimizer.state_dict(), moments)
            pointer = json.loads(reloaded.registry.champion_path.read_text())
            self.assertEqual(pointer["generation"], 1)
            self.assertEqual(len(pointer["history"]), 1)

    def test_load_discards_stale_selection_and_confirmation_without_counting_a_rejection(self):
        for phase in ("selection", "confirmation"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                controller = self.make_controller(root, max_cycles=2, player_counts=(2,))
                self.prepare_distinct_candidate(controller)
                if phase == "confirmation":
                    with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(True)):
                        controller._selection_step()
                controller.consecutive_rejections = 3
                controller.save_checkpoint()
                replacement = self.replace_champion(controller, root / "external.pt")
                resumed = ImprovementController.load(
                    str(controller.output_dir / "latest-state.pt"), controller.output_dir,
                )
                self.assertEqual(resumed.cycle_index, 1)
                self.assertEqual(resumed.phase, "training")
                self.assertEqual(resumed.league_in_cycle, 0)
                self.assertEqual(resumed.consecutive_rejections, 3)
                self.assertEqual(resumed.promotions, 0)
                self.assertIsNone(resumed.candidate_id)
                self.assertIsNone(resumed.selection_result)
                self.assertIsNone(resumed.confirmation_result)
                self.assertIsNone(resumed.champion_at_cycle_id)
                self.assertFalse(resumed.step(deadline=0))
                self.assertEqual(resumed.league.parent_sha256, replacement.sha256)
                self.assertEqual(resumed.consecutive_rejections, 0)
                self.assertEqual(resumed.league.optimizer.state_dict()["state"], {})
                for parameter in resumed.league.network.parameters():
                    self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.125)))

    def test_load_explicitly_rejects_other_controller_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.save_checkpoint()
            payload = controller.state_dict()
            path = controller.output_dir / "incompatible.pt"
            for version in (0, 1, 2, 4, 3.0, "3", None):
                with self.subTest(version=version):
                    torch.save({**payload, "improvement_state_version": version}, path)
                    with self.assertRaisesRegex(ValueError, "expected version 3"):
                        ImprovementController.load(str(path), controller.output_dir)
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"), controller.output_dir,
            )
            self.assertEqual(resumed.cycle_index, 0)
            self.assertEqual(resumed.phase, "training")

    def test_snapshot_ingestion_skips_bad_files_and_keeps_a_bounded_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            sources = [self.make_snapshot(snapshots) for _ in range(8)]
            corrupt = snapshots / "corrupt.pt"
            corrupt.write_bytes(b"not a torch checkpoint")
            corrupt_id = sha256(corrupt)[:20]
            corrupt.rename(snapshots / ("model-" + sha256(corrupt) + ".pt"))
            (snapshots / ("model-" + "0" * 64 + ".pt")).write_bytes(sources[0].read_bytes())
            (snapshots / "model-invalid.pt").write_bytes(sources[0].read_bytes())
            controller = self.make_controller(root, snapshots_dir=snapshots)
            expected_ids = tuple(sha256(path)[:20] for path in sorted(sources)[-6:])
            self.assertEqual(controller.opponent_ids, (controller.baseline_id, *expected_ids))
            self.assertEqual(controller.league.opponent_pool_generation, 1)
            for record_id in expected_ids:
                record = controller.registry.get(record_id)
                self.assertEqual(record.origin, "frozen_opponent")
                self.assertIn(record.sha256, controller.league.opponent_hashes)
            with self.assertRaisesRegex(ValueError, "Unknown model id"):
                controller.registry.get(corrupt_id)
            self.assertFalse(controller._ingest_snapshots())
            self.assertEqual(controller.opponent_ids, (controller.baseline_id, *expected_ids))
            self.assertEqual(controller.league.opponent_pool_generation, 1)

    def test_cycle_boundary_ingestion_is_saved_and_resumes_before_new_arrivals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            controller = self.make_controller(
                root, snapshots_dir=snapshots, max_cycles=2, player_counts=(2,),
                max_turns=1, selection_blocks=1, confirmation_blocks=1,
                solo_games=1, gate_samples=100,
            )
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            self.assertEqual(controller.phase, "selection")
            first = self.make_snapshot(snapshots)
            first_id = sha256(first)[:20]
            self.assertEqual(controller.opponent_ids, (controller.baseline_id,))
            while controller.cycle_index == 0:
                self.assertTrue(controller.step(deadline))
            self.assertEqual(controller.opponent_ids, (controller.baseline_id, first_id))
            self.assertEqual(controller.league.opponent_pool_generation, 1)
            saved_path = controller.output_dir / "latest-state.pt"
            saved = torch.load(saved_path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["cycle"], 1)
            self.assertEqual(saved["phase"], "training")
            self.assertEqual(saved["opponent_ids"], (controller.baseline_id, first_id))
            self.assertIn(sha256(first), saved["league_state"]["opponent_hashes"])
            second = self.make_snapshot(snapshots)
            second_id = sha256(second)[:20]
            resumed = ImprovementController.load(
                str(saved_path), controller.output_dir, snapshots_dir=snapshots,
            )
            self.assertEqual(resumed.opponent_ids, (controller.baseline_id, first_id))
            self.assertEqual(resumed.league.completed_matches, 1)
            self.assertFalse(resumed.step(deadline=0))
            self.assertEqual(resumed.opponent_ids, (controller.baseline_id, first_id, second_id))
            self.assertEqual(resumed.league.completed_matches, 1)
            updated = torch.load(saved_path, map_location="cpu", weights_only=True)
            self.assertEqual(updated["opponent_ids"], (controller.baseline_id, first_id, second_id))
            self.assertEqual(updated["cursors"]["league"], 0)
            self.assertIn(sha256(second), updated["league_state"]["opponent_hashes"])
            reloaded = ImprovementController.load(
                str(saved_path), controller.output_dir, snapshots_dir=snapshots,
            )
            self.assertEqual(reloaded.league.opponent_hashes, resumed.league.opponent_hashes)
            self.assertEqual(reloaded.config.player_counts, (2,))

    def test_midcycle_resume_defers_new_snapshots_until_the_next_cycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            first = self.make_snapshot(snapshots)
            first_id = sha256(first)[:20]
            controller = self.make_controller(
                root, snapshots_dir=snapshots, max_cycles=2, player_counts=(2,),
                max_turns=1, selection_blocks=1, confirmation_blocks=1,
                solo_games=1, gate_samples=100, league_matches_per_cycle=2,
            )
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            second = self.make_snapshot(snapshots)
            second_id = sha256(second)[:20]
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"),
                controller.output_dir, snapshots_dir=snapshots,
            )
            self.assertEqual(resumed.league_in_cycle, 1)
            self.assertEqual(resumed.opponent_ids, (controller.baseline_id, first_id))
            self.assertTrue(resumed.step(deadline))
            self.assertEqual(resumed.phase, "selection")
            self.assertEqual(resumed.opponent_ids, (controller.baseline_id, first_id))
            while resumed.cycle_index == 0:
                self.assertTrue(resumed.step(deadline))
            self.assertEqual(resumed.opponent_ids, (controller.baseline_id, first_id, second_id))
            self.assertEqual(resumed.league.opponent_pool_generation, 2)

    def test_specialist_promotion_preserves_its_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), player_counts=(3,))
            self.prepare_distinct_candidate(controller)
            with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(True)):
                controller._selection_step()
                candidate_id = controller.candidate_id
                controller._confirmation_step()
                controller._confirmation_step()
            self.assertEqual(controller.registry.champion().id, candidate_id)
            pointer = json.loads(controller.registry.champion_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["history"][0]["evidence"]["player_counts"], [3])
            self.assertEqual(controller.promotions, 1)

    def test_other_rule_reference_cannot_seed_current_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "champion-source.pt"
            self.make_model(model)
            (root / "final-best.json").write_text(
                json.dumps(
                    {
                        "encoding_version": 2,
                        "checkpoint_version": 4,
                        "architecture": "axial_table",
                        "checkpoint_sha256": sha256(model),
                        "game_config": {
                            "num_decks": 3,
                            "cards_in_hand": 21,
                            "required_sequences": 5,
                            "max_turns": 60,
                            "recycle_discard": False,
                        },
                        "greedy": {
                            "games": 2,
                            "wins": 1,
                            "outcomes": [1, 0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "rules must match"):
                ImprovementController(model, root / "output", config=ImproveConfig())

    def test_weak_selection_keeps_current_champion_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            original = controller.registry.champion()
            original_bytes = controller.registry.model_path(original.id).read_bytes()
            self.prepare_distinct_candidate(controller)
            with mock.patch(
                "src.training.improve.evaluate_gate",
                side_effect=gate_result(False),
            ):
                controller._selection_step()
            current = controller.registry.champion()
            self.assertEqual(current.id, original.id)
            self.assertEqual(controller.phase, "training")
            self.assertEqual(controller.promotions, 0)
            self.assertEqual(
                controller.registry.model_path(original.id).read_bytes(),
                original_bytes,
            )
            self.assertEqual(original.evidence, {})

    def test_only_fresh_confirmation_can_change_champion(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            original = controller.registry.champion()
            original_bytes = controller.registry.model_path(
                original.id
            ).read_bytes()
            self.prepare_distinct_candidate(controller)
            with mock.patch(
                "src.training.improve.evaluate_gate",
                side_effect=gate_result(True),
            ):
                controller._selection_step()
                candidate_id = controller.candidate_id
                self.assertEqual(controller.registry.champion().id, original.id)
                controller._confirmation_step()
                self.assertEqual(controller.registry.champion().id, original.id)
                controller._confirmation_step()
            self.assertEqual(controller.registry.champion().id, candidate_id)
            self.assertEqual(
                controller.registry.model_path(original.id).read_bytes(),
                original_bytes,
            )
            self.assertEqual(controller.promotions, 1)
            self.assertEqual(controller.promoted_cycles, (0,))
            self.assertEqual(controller.league.opponent_pool_generation, 1)

    def test_one_selection_bank_and_one_confirmation_bank_gate_the_same_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            self.prepare_distinct_candidate(controller)
            calls = []
            decisions = iter((True, False))

            def evaluate(**kwargs):
                calls.append(dict(kwargs))
                return gate_result(next(decisions))(**kwargs)

            with mock.patch(
                "src.training.improve.evaluate_gate",
                side_effect=evaluate,
            ):
                controller._selection_step()
                candidate_path = str(
                    controller.registry.model_path(controller.candidate_id)
                )
                controller._confirmation_step()
                self.assertEqual(controller.registry.champion().id, controller.baseline_id)
                controller._confirmation_step()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["seed"], 60_000_000)
            self.assertEqual(calls[1]["seed"], 60_500_000)
            self.assertEqual(calls[0]["seed_blocks"], 32)
            self.assertEqual(calls[1]["seed_blocks"], 128)
            self.assertEqual(calls[0]["candidate_path"], candidate_path)
            self.assertEqual(calls[1]["candidate_path"], candidate_path)
            self.assertEqual(controller.promotions, 0)
            self.assertEqual(controller.consecutive_rejections, 1)

    def test_v3_state_roundtrip_preserves_the_single_learner_and_rejection_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.make_controller(
                root, league_matches_per_cycle=2, max_turns=1, player_counts=(2,),
            )
            self.assertTrue(controller.step(time.monotonic() + 120))
            controller.consecutive_rejections = 3
            controller.save_checkpoint()
            before = copy.deepcopy(controller.state_dict())
            resumed = ImprovementController.load(
                str(controller.output_dir / "latest-state.pt"),
                controller.output_dir,
            )
            after = resumed.state_dict()
            self.assert_nested_equal(before, after)
            self.assertEqual(resumed.league.completed_matches, 1)
            self.assertEqual(resumed.consecutive_rejections, 3)
            self.assertEqual(after["improvement_state_version"], 3)
            self.assertEqual(
                after["league_state"]["last_exported_sha256"],
                sha256(controller.output_dir / "latest-model.pt"),
            )
            self.assertNotIn("research_state", after)
            resumed_dir = root / "resumed"
            shutil.copytree(controller.output_dir, resumed_dir)
            continued = ImprovementController.load(
                str(resumed_dir / "latest-state.pt"), resumed_dir,
            )
            deadline = time.monotonic() + 120
            self.assertTrue(controller.step(deadline))
            self.assertTrue(continued.step(deadline))
            self.assertEqual(controller.phase, "selection")
            self.assert_nested_equal(controller.state_dict(), continued.state_dict())

    def test_rounds_stop_at_checkpoint_and_cycle_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(
                Path(temporary), league_matches_per_cycle=7, matches_per_round=5,
                checkpoint_every=3, max_turns=1, player_counts=(2,),
            )
            deadline = time.monotonic() + 120
            with mock.patch.object(controller.league, "train_round", wraps=controller.league.train_round) as rounds:
                for expected in (3, 6, 7):
                    self.assertTrue(controller.step(deadline))
                    self.assertEqual(controller.league_in_cycle, expected)
                    saved = torch.load(controller.output_dir / "latest-state.pt", weights_only=True)
                    self.assertEqual(saved["cursors"]["league"], expected)
            self.assertEqual([call.args[0] for call in rounds.call_args_list], [3, 3, 1])
            self.assertEqual(controller.phase, "selection")
            self.assertEqual(controller.league.total_updates, 28)

    def test_resume_can_extend_runtime_cap_but_not_learning_settings(self):
        from src.cli.improve import _validate_resume_options

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            parser = build_arg_parser()
            arguments = [
                "--resume", "latest-state.pt", "--output-dir", "output",
                "--max-seconds", "60", "--max-cycles", "3",
            ]
            _validate_resume_options(
                improve_runtime,
                controller,
                parser.parse_args(arguments),
                arguments,
            )
            self.assertEqual(controller.config.max_cycles, 3)
            self.assertEqual(controller.config.league_matches_per_cycle, 1)
            for changed in (["--league-matches", "999"], ["--rebase-after-rejections", "2"], ["--matches-per-round", "4"]):
                invalid = arguments + changed
                with self.assertRaisesRegex(ValueError, "cannot change"):
                    _validate_resume_options(
                        improve_runtime,
                        controller,
                        parser.parse_args(invalid),
                        invalid,
                    )

    def test_resume_preserves_specialist_scope_unless_explicitly_changed(self):
        from src.cli.improve import _validate_resume_options

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary), player_counts=(3,))
            parser = build_arg_parser()
            base = [
                "--resume", "latest-state.pt", "--output-dir", "output",
                "--max-seconds", "60",
            ]
            for extra in ([], ["--players", "3"]):
                arguments = base + extra
                _validate_resume_options(
                    improve_runtime, controller, parser.parse_args(arguments), arguments,
                )
                self.assertEqual(controller.config.player_counts, (3,))
            for counts in (["2"], ["2", "3", "4"], ["3", "3"]):
                arguments = base + ["--players", *counts]
                with self.assertRaisesRegex(ValueError, "Resume cannot change --players"):
                    _validate_resume_options(
                        improve_runtime, controller, parser.parse_args(arguments), arguments,
                    )
                self.assertEqual(controller.config.player_counts, (3,))

    def test_controller_resume_rejects_nonfinite_league_optimizer_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            payload = controller.state_dict()
            payload["league_state"]["learner"]["optimizer"]["state"] = {
                0: {"exp_avg": torch.tensor([float("nan")])}
            }
            path = controller.output_dir / "invalid-state.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                ImprovementController.load(str(path), controller.output_dir)

    def test_zero_budget_exports_truthful_flat_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            reason = controller.run(0)
            self.assertEqual(reason, "budget")
            status = json.loads(
                (controller.output_dir / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["algorithm"], ALGORITHM)
            self.assertEqual(status["completed_cycles"], 0)
            self.assertEqual(status["completed_episodes"], 0)
            self.assertEqual(status["league_matches"], 0)
            self.assertEqual(status["updates"], 0)
            self.assertEqual(status["consecutive_rejections"], 0)
            self.assertNotIn("research_episodes", status)
            self.assertEqual(status["promotions"], 0)
            self.assertEqual(status["distance"], None)
            self.assertNotIn("genes", json.dumps(status).lower())
            required = {
                "latest-state.pt",
                "latest-model.pt",
                "best-model.pt",
                "baseline-model.pt",
                "metrics.jsonl",
                "registry.json",
                "champion.json",
            }
            self.assertTrue(required.issubset(status["files"]))
            for name, metadata in status["files"].items():
                path = controller.output_dir / name
                self.assertEqual(metadata["bytes"], path.stat().st_size)
                self.assertEqual(metadata["sha256"], sha256(path))

    def test_active_metrics_cannot_invalidate_published_snapshot(self):
        from src.training.improve import _append_metric_once

        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.save_checkpoint()
            status = json.loads((controller.output_dir / "status.json").read_text())
            metadata = status["files"]["metrics.jsonl"]
            _append_metric_once(controller.output_dir, "new-event", {"event": "test"}, controller.metric_ids)
            self.assertEqual(sha256(controller.output_dir / "metrics.jsonl"), metadata["sha256"])
            self.assertTrue((controller.output_dir / ".active-metrics.jsonl").read_text())
            controller.save_checkpoint()
            self.assertIn("new-event", (controller.output_dir / "metrics.jsonl").read_text())

    def test_gate_budget_exceptions_checkpoint_without_partial_promotion(self):
        from src.evaluation.arena import MatchInterrupted
        from src.evaluation.promotion import GateInterrupted

        interruptions = (
            GateInterrupted("deadline"),
            MatchInterrupted("deadline"),
            TimeoutError("deadline"),
        )
        for interruption in interruptions:
            with self.subTest(exception=type(interruption).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    controller = self.make_controller(Path(temporary))
                    self.prepare_distinct_candidate(controller)
                    champion_id = controller.registry.champion().id
                    before = controller.state_dict()
                    with mock.patch(
                        "src.training.improve.evaluate_gate",
                        side_effect=interruption,
                    ):
                        reason = controller.run(10)
                    self.assertEqual(reason, "budget")
                    self.assertEqual(controller.registry.champion().id, champion_id)
                    self.assertIsNone(controller.selection_result)
                    self.assertIsNone(controller.last_gate)
                    saved = torch.load(
                        controller.output_dir / "latest-state.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assert_nested_equal(
                        saved["league_state"], before["league_state"]
                    )
                    status = json.loads(
                        (controller.output_dir / "status.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(status["stop_reason"], "budget")
                    self.assertEqual(status["completed_cycles"], 0)
                    self.assertEqual(status["completed_episodes"], 0)

    def test_controller_state_uses_relative_registry_ids_not_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            controller.save_checkpoint()
            payload = torch.load(
                controller.output_dir / "latest-state.pt",
                map_location="cpu",
                weights_only=True,
            )
            encoded = repr(payload)
            self.assertNotIn("champion-source.pt", encoded)
            self.assertEqual(payload["baseline_id"], controller.baseline_id)
            self.assertIsInstance(payload["opponent_ids"], tuple)

    def test_actual_reduced_cap_cycle_keeps_champion_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "champion-source.pt"
            self.make_model(model)
            controller = ImprovementController(
                model,
                root / "output",
                config=ImproveConfig(
                    max_cycles=1,
                    league_matches_per_cycle=2,
                    selection_blocks=2,
                    confirmation_blocks=2,
                    solo_games=2,
                    checkpoint_every=2,
                    gate_samples=100,
                    max_turns=1,
                ),
            )
            original = controller.registry.champion().id
            reason = controller.run(300)
            self.assertEqual(reason, "max_cycles")
            self.assertEqual(controller.cycle_index, 1)
            self.assertEqual(controller.league.completed_matches, 2)
            self.assertEqual(controller.registry.champion().id, original)
            self.assertEqual(controller.promotions, 0)
            self.assertEqual(controller.league.config.game.max_turns, 1)
            self.assertEqual(controller.last_gate["max_turns"], 1)
            self.assertEqual(controller.last_gate["candidate_origin"], "league")
            metrics = [
                json.loads(line)
                for line in (controller.output_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in metrics],
                ["league_match", "league_match", "gate"],
            )
            self.assertEqual(metrics[-1]["phase"], "selection")


class TestImproveCli(unittest.TestCase):
    def test_removed_research_options_fail_before_loading_the_training_runtime(self):
        from src.cli.improve import main

        base = [
            "--initial-model", "champion.pt", "--output-dir", "out",
            "--max-seconds", "60",
        ]
        for option in ("--research-episodes", "--research-episodes-per-cycle"):
            with self.subTest(option=option), mock.patch(
                "src.cli.improve.load_runtime", side_effect=AssertionError("runtime loaded"),
            ), mock.patch("sys.stderr", new_callable=io.StringIO) as error:
                self.assertEqual(main(base + [option, "2"]), 2)
                self.assertIn("unrecognized arguments", error.getvalue())
                self.assertIn(option, error.getvalue())

    def test_cli_exposes_specialist_scopes_and_optional_snapshot_directory(self):
        parser = build_arg_parser()
        base = ["--initial-model", "champion.pt", "--output-dir", "out", "--max-seconds", "60"]
        defaults = parser.parse_args(base)
        self.assertIsNone(defaults.players)
        self.assertIsNone(defaults.snapshots)
        for counts in (["2"], ["3"], ["4"], ["2", "3", "4"]):
            with self.subTest(counts=counts):
                args = parser.parse_args(base + ["--players", *counts, "--snapshots", "snapshots"])
                self.assertEqual(args.players, [int(count) for count in counts])
                self.assertEqual(args.snapshots, Path("snapshots"))

    def test_cli_requires_budget_and_exposes_bounded_cycle_overrides(self):
        parser = build_arg_parser()
        max_seconds = next(
            action
            for action in parser._actions
            if action.dest == "max_seconds"
        )
        self.assertTrue(max_seconds.required)
        args = parser.parse_args(
            [
                "--initial-model",
                "champion.pt",
                "--output-dir",
                "out",
                "--max-seconds",
                "60",
                "--max-cycles",
                "1",
                "--league-matches-per-cycle",
                "2",
                "--matches-per-round",
                "4",
                "--rebase-after-rejections",
                "3",
                "--selection-blocks",
                "2",
                "--confirmation-blocks",
                "2",
                "--solo-games",
                "2",
            ]
        )
        self.assertEqual(args.max_cycles, 1)
        self.assertEqual(args.league_matches_per_cycle, 2)
        self.assertEqual(args.matches_per_round, 4)
        self.assertEqual(args.rebase_after_rejections, 3)
        self.assertEqual(args.selection_blocks, 2)
        self.assertEqual(args.confirmation_blocks, 2)
        self.assertEqual(args.solo_games, 2)


if __name__ == "__main__":
    unittest.main()
