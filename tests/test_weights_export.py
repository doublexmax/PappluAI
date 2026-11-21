import base64
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from src.game.environment import GameConfig, PappluEnv, Phase, discard_action
from src.game.information import EventKind
from src.model.encoding import encode, feature_values, unpack
from src.model.spec import ModelSpec
from tools.weights_export import (
    GOLDEN_CASES, golden_observation, main, observation_json, reference_layers, reference_q,
    require_sensitive, rows, tensor_shapes,
)

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def zero_tensors():
    return {name: [0.0] * math.prod(shape) for name, shape in tensor_shapes()}


class TestPortableExport(unittest.TestCase):
    def test_bootstrap_runs_with_torch_and_numpy_imports_blocked(self):
        script = """
import builtins, runpy, sys
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('bootstrap imported ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = ['tools/weights_export.py', '--bootstrap', '--output', sys.argv[1], '--golden', sys.argv[2]]
runpy.run_path('tools/weights_export.py', run_name='__main__')
"""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            model, golden = Path(temporary) / "model.json", Path(temporary) / "golden.json"
            result = subprocess.run(
                [sys.executable, "-c", script, str(model), str(golden)],
                cwd=ROOT, capture_output=True, text=True, timeout=180, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            document = json.loads(model.read_text())
            fixture = json.loads(golden.read_text())
            self.assertEqual(document["checkpointVersion"], 6)
            self.assertEqual(document["encodingVersion"], 8)
            self.assertEqual(document["architecture"], "card_mlp")
            self.assertEqual(document["modelSpec"], {"width": 80, "blocks": 3})
            self.assertEqual(document["source"]["kind"], "bootstrap")
            self.assertEqual(fixture["oracle"], "python-reference")
            self.assertEqual(len(fixture["cases"]), 15)
            self.assertTrue(all(name.startswith("policy.") for name in document["tensors"]))
            self.assertEqual(document["source"]["tensorsSha256"], fixture["tensorsSha256"])
            self.assertEqual(len(fixture["cases"][0]["features"]), 1433)
            self.assertEqual(fixture["featureLayout"], [
                ["card", [52, 25]], ["seat", [6, 9]], ["context", [24]], ["legal", [54]], ["top", [1]],
            ])
            self.assertEqual(
                sum(len(base64.b64decode(entry["data"])) // 4 for entry in document["tensors"].values()), 344244,
            )

    def test_legacy_heads_flag_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "-B", "tools/weights_export.py", "--bootstrap", "--heads", "4"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --heads 4", result.stderr)


class TestReferenceExport(unittest.TestCase):
    def test_dueling_values_center_only_on_legal_actions(self):
        tensors = zero_tensors()
        tensors["policy.value_out.bias"] = [0.25]
        tensors["policy.draw_out.bias"] = [1.0, 3.0]
        tensors["policy.face_out.bias"] = [7.0]
        layers = reference_layers(tensors)
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        for phase, expected in (
            (Phase.DRAW, [-0.75, 1.25] + [5.25] * 52),
            (Phase.DISCARD, [-5.75, -3.75] + [0.25] * 52),
            (Phase.TERMINAL, [1.25, 3.25] + [7.25] * 52),
        ):
            with self.subTest(phase=phase):
                self.assertEqual(reference_q(layers, encode(replace(obs, phase=phase))), expected)

    def test_insensitive_bootstrap_weights_are_rejected(self):
        feature = encode(PappluEnv().reset(17))
        with self.assertRaisesRegex(SystemExit, "not sensitive to the card feature field"):
            require_sensitive(zero_tensors(), [feature], [{"q": [0.0] * 54}])

    def test_observable_encoding_keeps_wild_counts_and_points_alongside_meld_flags(self):
        obs = PappluEnv(GameConfig(cards_in_hand=3, required_sequences=1)).reset(17)
        hand = tuple((5, 5, 7, 20).count(face) for face in range(52))
        features = unpack(encode(replace(obs, hand=hand, joker=7)))
        card_rows = rows(features["card"], 25)
        self.assertEqual(card_rows[5][1:4], (0.0, 1.0, 0.0))
        self.assertEqual(card_rows[7][6:9], (1.0, 1.0, 0.0))
        self.assertEqual(card_rows[20][6:9], (0.0, 1.0, 0.0))
        self.assertAlmostEqual(features["context"][14], 1 / 3, places=7)
        self.assertAlmostEqual(features["context"][15], 1 / 6, places=7)
        self.assertAlmostEqual(features["context"][16], 0.12, places=7)

    def test_removed_registry_source_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "-B", "tools/weights_export.py", "--registry", "retired"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --registry retired", result.stderr)


@unittest.skipIf(torch is None, "torch is not installed")
class TestCheckpointExport(unittest.TestCase):
    def test_real_checkpoint_export_matches_torch_and_portable_reference(self):
        from src.model.network import QNetwork, collate, save_checkpoint

        script = """
import {readFileSync} from 'node:fs';
import {encodeObservation, parseQModel, qValues} from './web/qnet.mjs';
const model = parseQModel(JSON.parse(readFileSync(process.argv[1], 'utf8')));
const observations = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify({
  q: qValues(model, observations).map(row => Array.from(row)),
  features: observations.map(obs => Array.from(encodeObservation(obs))),
}));
"""
        torch.set_num_threads(1)
        config = GameConfig(max_turns=60)
        for spec in (ModelSpec(80, 3), ModelSpec(128, 3), ModelSpec(9, 1)):
            with self.subTest(spec=spec), tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                torch.manual_seed(1729)
                network = QNetwork(spec).eval()
                checkpoint = Path(temporary) / "model.pt"
                output, golden = Path(temporary) / "model.json", Path(temporary) / "golden.json"
                save_checkpoint(str(checkpoint), network, config)
                self.assertEqual(main(["--checkpoint", str(checkpoint), "--output", str(output), "--golden", str(golden)]), 0)
                document, fixture = json.loads(output.read_text()), json.loads(golden.read_text())
                self.assertEqual(document["modelSpec"], spec.to_dict())
                self.assertEqual(document["source"]["kind"], "checkpoint")
                self.assertEqual(fixture["oracle"], "torch")
                tensors = {}
                for name, entry in document["tensors"].items():
                    data = base64.b64decode(entry["data"])
                    tensors[name] = list(struct.unpack("<%df" % (len(data) // 4), data))
                layers = reference_layers(tensors, spec)
                observations = [golden_observation(case) for case in GOLDEN_CASES]
                env = PappluEnv(config)
                observations.append(env.reset(917))
                observations.append(env.step(0))
                env.reset(918)
                taken = env.step(1)
                observations.append(taken)
                features = [encode(obs) for obs in observations]
                self.assertEqual([obs.phase for obs in observations[-3:]], [
                    Phase.DRAW, Phase.DISCARD, Phase.DISCARD,
                ])
                self.assertIs(taken.history[-1].kind, EventKind.TAKE_DISCARD)
                self.assertEqual(unpack(features[-1])["legal"][discard_action(taken.history[-1].face)], 0.0)
                with torch.inference_mode():
                    expected = network(collate(features, torch.device("cpu"))).tolist()
                for feature, entry, torch_q in zip(features[:15], fixture["cases"], expected):
                    self.assertEqual(list(feature_values(feature)), entry["features"])
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(entry["q"], torch_q)), 1e-6)
                    self.assertLess(max(abs(a - b) for a, b in zip(reference_q(layers, feature, spec), torch_q)), 1e-5)
                browser = subprocess.run(
                    ["node", "--input-type=module", "-e", script, str(output)],
                    input=json.dumps([observation_json(obs) for obs in observations]),
                    cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
                )
                self.assertEqual(browser.returncode, 0, browser.stdout + browser.stderr)
                actual = json.loads(browser.stdout)
                self.assertEqual(len(actual["q"]), 18)
                self.assertGreater(min(max(q) - min(q) for q in expected), 1e-3)
                for index, (feature, torch_q, node_q) in enumerate(zip(features, expected, actual["q"])):
                    self.assertEqual(len(node_q), 54)
                    self.assertEqual(actual["features"][index], list(feature_values(feature)))
                    self.assertLessEqual(max(abs(a - b) for a, b in zip(torch_q, node_q)), 1e-5)


if __name__ == "__main__":
    unittest.main()
