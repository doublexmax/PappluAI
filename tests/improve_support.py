from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.checkpoints.io import file_sha256
from src.training.improve import ImproveConfig, ImprovementController

try:
    import torch
except ImportError:
    torch = None


TORCH_REASON = "PyTorch not installed (see requirements-training.txt)"


def sha256(path):
    return file_sha256(Path(path))


def gate_result(accepted):
    def evaluate(**kwargs):
        from src.checkpoints.evidence import balanced_seat_orders, blocks_for_count, evidence_metrics

        blocks = kwargs["seed_blocks"]
        multiplayer = {
            str(players): {
                "blocks": blocks_for_count(blocks, players),
                "candidate_minus_champion": 0.5 if accepted else 0.0,
                "games": blocks_for_count(blocks, players) * len(balanced_seat_orders(players)),
                "placement_margins": [0.5 if accepted else 0.0] * blocks_for_count(blocks, players),
                "declaration_margins": [0.0] * blocks_for_count(blocks, players),
                "declaration_delta": 0.0,
            }
            for players in kwargs["player_counts"]
        }
        solo = {
            "seed": kwargs["seed"] + 100_000, "games": kwargs["solo_games"],
            "candidate_outcomes": [0] * kwargs["solo_games"],
            "champion_outcomes": [0] * kwargs["solo_games"],
            "candidate_wins": 0, "champion_wins": 0,
        }
        samples = max(100, kwargs["samples"])
        return {
            "gate_schema_version": 4, "encoding_version": 2, "architecture": "axial_table",
            "candidate_sha256": sha256(kwargs["candidate_path"]),
            "champion_sha256": sha256(kwargs["champion_path"]),
            "accepted": accepted,
            "phase": kwargs["phase"],
            "gate_failures": [] if accepted else ["placement"],
            **evidence_metrics(
                multiplayer, solo, blocks, kwargs["seed"], samples,
                player_counts=kwargs["player_counts"],
            ),
            "bootstrap_samples": samples, "seed": kwargs["seed"],
            "seed_blocks": kwargs["seed_blocks"], "player_counts": list(kwargs["player_counts"]),
            "rules": {
                "num_decks": 3, "cards_in_hand": 21, "required_sequences": 5,
                "max_turns": kwargs["max_turns"], "recycle_discard": kwargs["recycle_discard"],
            },
            "pool_hashes": [sha256(path) for path in kwargs["opponent_paths"]],
            "raw_outcomes": (), "multiplayer": multiplayer, "solo": solo,
            "scope": {"player_counts": kwargs["player_counts"]},
        }

    return evaluate


@unittest.skipIf(torch is None, TORCH_REASON)
class ControllerTestCase(unittest.TestCase):
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
            str(path), network, GameConfig(max_turns=60), meta={"best_score": 164 / 256},
        )

    def make_controller(
        self, root, *, snapshots_dir=None, parameter_value=None, **config_overrides,
    ):
        model = root / "champion-source.pt"
        self.make_model(model, parameter_value=parameter_value)
        config = ImproveConfig(
            max_cycles=1, league_matches_per_cycle=1, selection_blocks=32,
            confirmation_blocks=128, solo_games=256, checkpoint_every=1, gate_samples=20,
        )
        return ImprovementController(
            model, root / "output", config=replace(config, **config_overrides),
            snapshots_dir=snapshots_dir,
        )

    def write_solo_reference(self, model, wins):
        (model.parent / "final-best.json").write_text(json.dumps({
            "encoding_version": 2, "checkpoint_version": 4, "architecture": "axial_table",
            "checkpoint_sha256": sha256(model),
            "game_config": {
                "num_decks": 3, "cards_in_hand": 21, "required_sequences": 5,
                "max_turns": 60, "recycle_discard": True,
            },
            "greedy": {"games": 100, "wins": wins, "outcomes": [1] * wins + [0] * (100 - wins)},
        }), encoding="utf-8")

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
            "solo_games": controller.config.solo_games, "max_turns": controller.config.max_turns,
            "player_counts": controller.config.player_counts, "samples": controller.config.gate_samples,
            "recycle_discard": controller.config.recycle_discard,
        }
        selection = {
            **gate_result(True)(**arguments, seed=60_000_000, seed_blocks=32, phase="selection"),
        }
        confirmation = {
            **gate_result(True)(**arguments, seed=60_500_000, seed_blocks=128, phase="confirmation"),
            "selection_passed": True,
            "selection": selection, "selection_seed": 60_000_000,
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
