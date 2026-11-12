from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

from src.training.improve import (
    FinetuneSchedule,
    ImproveConfig,
    ImprovementController,
)
from tests.improve_support import ControllerTestCase, gate_result, sha256, torch


class TestImprovementController(ControllerTestCase):
    def test_verified_solo_mastery_selects_defaults_and_resume_keeps_the_saved_schedule(self):
        for wins, score, expected in ((80, 0.8, (1e-4, 0.08, 0.02)), (95, 0.95, (2e-5, 0.04, 0.01))):
            with self.subTest(wins=wins), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / "initial.pt"
                self.make_model(model)
                self.write_solo_reference(model, wins)
                controller = ImprovementController(model, root / "output")
                league = controller.league
                self.assertEqual(
                    (league.optimizer.param_groups[0]["lr"], league.current_epsilon, league.config.epsilon_end),
                    expected,
                )
                self.assertEqual(controller.best_score, score)
                controller.save_checkpoint()
                self.write_solo_reference(model, 100 - wins)
                resumed = ImprovementController.load(
                    str(controller.output_dir / "latest-state.pt"), controller.output_dir,
                )
                self.assertEqual(
                    (resumed.league.optimizer.param_groups[0]["lr"], resumed.league.current_epsilon,
                     resumed.league.config.epsilon_end),
                    expected,
                )

    def test_explicit_finetune_schedule_overrides_verified_mastery_and_persists_decay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "initial.pt"
            self.make_model(model)
            self.write_solo_reference(model, 95)
            controller = ImprovementController(
                model, root / "output", finetune=FinetuneSchedule(0.00015, 0.12, 0.03),
                epsilon_decay_matches=77,
            )
            self.assertEqual(controller.league.optimizer.param_groups[0]["lr"], 0.00015)
            self.assertEqual(controller.league.current_epsilon, 0.12)
            controller.save_checkpoint()
            resumed = ImprovementController.load(str(controller.output_dir / "latest-state.pt"), controller.output_dir)
            resumed.league.epsilon_progress = 77
            self.assertAlmostEqual(resumed.league.current_epsilon, 0.03)
            self.assertEqual(resumed.league.config.epsilon_decay_matches, 77)

    def test_unmeasured_challenger_uses_the_partial_mastery_finetune_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.make_controller(Path(temporary))
            session = controller.league
            self.assertEqual(session.optimizer.param_groups[0]["lr"], 1e-4)
            self.assertEqual(session.current_epsilon, 0.08)
            session.epsilon_progress = session.config.epsilon_decay_matches
            self.assertAlmostEqual(session.current_epsilon, 0.02)
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
            with mock.patch("src.training.improve.evaluate_gate", side_effect=gate_result(True)) as gate:
                self.assertTrue(controller.step(deadline))
                self.assertEqual(controller.phase, "confirmation")
                self.assertEqual(controller.registry.champion().id, original_id)
                with mock.patch.object(controller, "save_checkpoint", wraps=controller.save_checkpoint) as checkpoints:
                    self.assertTrue(controller.step(deadline))
                self.assertEqual(checkpoints.call_count, 2)
                self.assertEqual(gate.call_count, 2)
                self.assertEqual(controller.phase, "training")
                self.assertEqual(controller.cycle_index, 1)
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
            self.assertEqual(league.optimizer.param_groups[0]["lr"], 1e-4)
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
                            with mock.patch.object(controller, "save_checkpoint", wraps=controller.save_checkpoint) as checkpoints:
                                self.assertTrue(controller.step(deadline))
                            self.assertEqual(checkpoints.call_count, 1)
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
            moments = copy.deepcopy(controller.league.optimizer.state_dict())
            replay = copy.deepcopy(controller.league.replay.state_dict())
            promote = controller.registry.promote

            def promote_then_crash(*args, **kwargs):
                promote(*args, **kwargs)
                raise RuntimeError("injected crash after champion pointer write")

            with mock.patch.object(controller.registry, "promote", side_effect=promote_then_crash), mock.patch(
                "src.training.improve.evaluate_gate", side_effect=gate_result(True),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    controller.step(deadline)
            self.assertEqual(controller.registry.champion().id, candidate_id)
            self.assertEqual(controller.phase, "confirmation")
            self.assertEqual(controller.promotions, 0)
            state_path = controller.output_dir / "latest-state.pt"
            saved = torch.load(state_path, weights_only=True)
            self.assertEqual(saved["phase"], "confirmation")
            self.assertEqual(saved["evaluation"]["candidate_id"], candidate_id)
            self.assertTrue(saved["evaluation"]["confirmation_result"]["accepted"])
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
            for version in (0, 1, 2, 3, 4.0, "4", None):
                with self.subTest(version=version):
                    torch.save({**payload, "improvement_state_version": version}, path)
                    with self.assertRaisesRegex(ValueError, "expected version 4"):
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
                self.assertEqual(controller.phase, "training")
                self.assertEqual(controller.cycle_index, 1)
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
            controller = self.make_controller(Path(temporary), gate_workers=2)
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
                self.assertEqual(controller.phase, "training")
                self.assertEqual(controller.cycle_index, 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["seed"], 60_000_000)
            self.assertEqual(calls[1]["seed"], 60_500_000)
            self.assertEqual(calls[0]["seed_blocks"], 32)
            self.assertEqual(calls[1]["seed_blocks"], 128)
            self.assertEqual([call["phase"] for call in calls], ["selection", "confirmation"])
            self.assertEqual([call["solo_games"] for call in calls], [256, 256])
            self.assertEqual(calls[0]["candidate_path"], candidate_path)
            self.assertEqual(calls[1]["candidate_path"], candidate_path)
            self.assertEqual([call["workers"] for call in calls], [2, 2])
            self.assertEqual(
                [call["journal_dir"] for call in calls], [controller.output_dir / "gates"] * 2,
            )
            self.assertEqual(controller.promotions, 0)
            self.assertEqual(controller.consecutive_rejections, 1)

    def test_v4_state_roundtrip_preserves_the_single_learner_and_rejection_count(self):
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
            self.assertEqual(after["improvement_state_version"], 4)
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
            self.assertEqual(controller.last_gate["gate_schema_version"], 4)
            self.assertEqual(controller.last_gate["candidate_origin"], "league")
            self.assertIn("rules", controller.last_gate["gate_failures"])
            journal, = (controller.output_dir / "gates").glob("*.jsonl")
            completed = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(
                [record["job"] for record in completed if record["job"][0] == "solo"],
                [["solo", 0, 2]],
            )
            metrics = [
                json.loads(line)
                for line in (controller.output_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in metrics],
                ["league_match", "league_match", "gate"],
            )
            self.assertEqual(metrics[-1]["phase"], "selection")


if __name__ == "__main__":
    unittest.main()
