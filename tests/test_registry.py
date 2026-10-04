from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.checkpoints import atomic_write_json
from src.promotion import evidence_metrics
from src.registry import ModelRecord, ModelRegistry


def install_record(registry, data, origin):
    digest = hashlib.sha256(data).hexdigest()
    record = ModelRecord(digest[:20], digest, "model-" + digest + ".pt", origin, {}, {}, ())
    (registry.root / record.filename).write_bytes(data)
    database = json.loads(registry.path.read_text(encoding="utf-8"))
    database["models"][record.id] = asdict(record)
    atomic_write_json(database, registry.path)
    return record


def gate_evidence(candidate, champion, blocks, seed):
    margins = [0.5] * (blocks // 4) + [0.0] * (
        blocks - blocks // 4
    )
    margin = sum(margins) / blocks
    multiplayer = {
        "2": {
            "candidate_minus_champion": margin,
            "seed_block_margins": margins,
            "games": blocks * 2,
        },
        "3": {
            "candidate_minus_champion": margin,
            "seed_block_margins": margins,
            "games": blocks * 6,
        },
    }
    solo = {
        "games": 256,
        "candidate_outcomes": [1] * 16 + [0] * 240,
        "champion_outcomes": [1] * 8 + [0] * 248,
        "candidate_wins": 16,
        "champion_wins": 8,
    }
    return {
        "accepted": True,
        "candidate_sha256": candidate.sha256,
        "champion_sha256": champion.sha256,
        "player_counts": [2, 3],
        "seed_blocks": blocks,
        "seed": seed,
        "bootstrap_samples": 100,
        "rules": {
            "num_decks": 3, "cards_in_hand": 21, "required_sequences": 5,
            "max_turns": 60, "recycle_discard": True,
        },
        **evidence_metrics(multiplayer, solo, blocks, seed, 100),
        "solo": solo,
        "multiplayer": multiplayer,
    }


def evidence(candidate, champion):
    selection = {
        **gate_evidence(candidate, champion, 32, 60_000_000),
        "phase": "selection",
    }
    return {
        **gate_evidence(candidate, champion, 128, 60_500_000),
        "selection_passed": True,
        "phase": "confirmation",
        "selection_seed": selection["seed"],
        "selection": selection,
    }


class TestChampionProtection(unittest.TestCase):
    def test_atomic_registry_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            with self.assertRaises(ValueError):
                atomic_write_json({"score": float("nan")}, path)
            self.assertFalse(path.exists())

    def test_promotion_changes_only_pointer_and_preserves_old_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"frozen champion", "initial")
            candidate = install_record(registry, b"challenger", "league")
            registry.initialize_champion(champion.id)
            before = registry.model_path(champion.id).read_bytes()
            registry.promote(candidate.id, champion.id, evidence(candidate, champion))
            self.assertEqual(registry.champion().id, candidate.id)
            self.assertEqual(registry.model_path(champion.id).read_bytes(), before)
            pointer = json.loads(registry.champion_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["generation"], 1)
            self.assertEqual(pointer["history"][0]["previous_id"], champion.id)

    def test_weak_or_stale_or_selection_only_evidence_cannot_promote(self):
        mutations = [
            {"accepted": False}, {"selection_passed": False}, {"phase": "selection"},
            {"multiplayer_gain": 0.02}, {"multiplayer_ci95": [-0.01, 0.1]},
            {"solo_gain": -0.01}, {"solo_ci95": [-0.04, 0.1]},
            {"seed_blocks": 2}, {"solo": {"games": 2}},
            {"selection_seed": 60_500_000}, {"candidate_sha256": "wrong"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"frozen", "initial")
            candidate = install_record(registry, b"new", "research")
            registry.initialize_champion(champion.id)
            for mutation in mutations:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    registry.promote(candidate.id, champion.id, {**evidence(candidate, champion), **mutation})
                self.assertEqual(registry.champion().id, champion.id)

    def test_changed_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"original", "initial")
            registry.initialize_champion(champion.id)
            (registry.root / champion.filename).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                registry.champion()

    def test_rejects_summary_metrics_not_supported_by_the_raw_games(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"original", "initial")
            candidate = install_record(registry, b"new", "league")
            registry.initialize_champion(champion.id)
            for field, value in (("solo_gain", 1.0), ("multiplayer_gain", 0.99)):
                bad = evidence(candidate, champion)
                bad[field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "raw evidence"):
                    registry.promote(candidate.id, champion.id, bad)
                self.assertEqual(registry.champion().id, champion.id)
            bad = evidence(candidate, champion)
            bad["solo"]["candidate_outcomes"] = [1, 0]
            with self.assertRaisesRegex(ValueError, "per game"):
                registry.promote(candidate.id, champion.id, bad)

    def test_confirmation_and_selection_have_independent_sample_floors(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"original", "initial")
            candidate = install_record(registry, b"new", "league")
            registry.initialize_champion(champion.id)

            short_confirmation = evidence(candidate, champion)
            replacement = gate_evidence(
                candidate,
                champion,
                127,
                short_confirmation["seed"],
            )
            short_confirmation.update(replacement)
            with self.assertRaisesRegex(ValueError, "confirmation requires"):
                registry.promote(
                    candidate.id,
                    champion.id,
                    short_confirmation,
                )

            short_selection = evidence(candidate, champion)
            short_selection["selection"] = {
                **gate_evidence(
                    candidate,
                    champion,
                    31,
                    short_selection["selection_seed"],
                ),
                "phase": "selection",
            }
            with self.assertRaisesRegex(ValueError, "selection requires"):
                registry.promote(
                    candidate.id,
                    champion.id,
                    short_selection,
                )

    def test_cannot_replace_initial_pin_or_use_outdated_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            champion = install_record(registry, b"original", "initial")
            candidate = install_record(registry, b"new", "league")
            registry.initialize_champion(champion.id)
            with self.assertRaises(ValueError):
                registry.initialize_champion(candidate.id)
            with self.assertRaises(ValueError):
                registry.promote(candidate.id, candidate.id, evidence(candidate, champion))


if __name__ == "__main__":
    unittest.main()
