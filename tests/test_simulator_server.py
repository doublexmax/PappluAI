import http.client
import json
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.simulator import (
    EvaluationTimeout,
    SimulatorServer,
    evaluate_request,
    validate_payload,
)


def payload(cards=(16, 17, 7), required=1, size=3):
    hand = [0] * 52
    for card in cards:
        hand[card] += 1
    return {"hand": hand, "joker": 7, "required_sequences": required, "cards_in_hand": size}


class PayloadTests(unittest.TestCase):
    def test_defaults(self):
        request = {"hand": [0] * 52, "joker": 7}
        self.assertEqual(
            validate_payload(request),
            {**request, "required_sequences": 5, "cards_in_hand": 21},
        )

    def test_invalid_inputs(self):
        bad = [
            None, [], {}, {**payload(), "extra": 1},
            {**payload(), "hand": [0] * 51},
            {**payload(), "hand": [True] + [0] * 51},
            {**payload(), "hand": [-1] + [0] * 51},
            {**payload(), "hand": [7] + [0] * 51},
            {**payload(), "hand": [6] * 6 + [0] * 46},
            {**payload(), "joker": True}, {**payload(), "joker": 52},
            {**payload(), "cards_in_hand": 31},
            {**payload(), "cards_in_hand": 2},
            {**payload(), "required_sequences": 11},
            {**payload(), "required_sequences": 1.0},
        ]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_payload(request)

    def test_timeout_stops_owned_process(self):
        before = {process.pid for process in multiprocessing.active_children()}
        with self.assertRaises(EvaluationTimeout):
            evaluate_request(payload(), timeout=0)
        self.assertEqual(
            {process.pid for process in multiprocessing.active_children()}, before,
        )
        self.assertTrue(evaluate_request(payload())["is_valid"])


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = SimulatorServer(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=15,
        )
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, value):
        status, headers, body = self.request(
            "POST", "/api/evaluate", json.dumps(value),
            {"Content-Type": "application/json"},
        )
        return status, headers, json.loads(body)

    def test_exact_joker_result_and_witness(self):
        status, _, result = self.post(payload())
        self.assertEqual(status, 200)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(len(result["melds"]), 1)
        self.assertTrue(result["melds"][0]["is_pure"])
        self.assertCountEqual(result["melds"][0]["cards"], [16, 17, 7])

    def test_off_suit_joker_does_not_fill_required_sequence(self):
        status, _, result = self.post(payload((16, 17, 33)))
        self.assertEqual(status, 200)
        self.assertEqual(result, {"is_valid": False, "melds": [], "reward": 0.0})

    def test_incomplete_hand_is_not_a_request_error(self):
        status, _, result = self.post(payload((16, 17)))
        self.assertEqual(status, 200)
        self.assertFalse(result["is_valid"])

    def test_malformed_payload_is_not_a_losing_hand(self):
        status, _, result = self.post({"hand": [False] * 52, "joker": 7})
        self.assertEqual(status, 400)
        self.assertIn("error", result)
        self.assertNotIn("reward", result)

    def test_invalid_json(self):
        status, _, body = self.request(
            "POST", "/api/evaluate", "{", {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))

    def test_wrong_content_type_and_large_body(self):
        status, _, _ = self.request("POST", "/api/evaluate", "{}")
        self.assertEqual(status, 415)
        status, _, _ = self.request(
            "POST", "/api/evaluate", " " * 8193,
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 413)

    def test_fixed_asset_allowlist(self):
        for path in ("/README.md", "/src/evaluate.py", "/../.git/config", "/%2e%2e/.git/config"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)

    def test_static_assets_and_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("<h1>Papplu</h1>", encoding="utf-8")
            (root / "game.mjs").write_text("export const ready = true;", encoding="utf-8")
            with patch("src.simulator.WEB_ROOT", root):
                status, headers, body = self.request("GET", "/")
                self.assertEqual(status, 200)
                self.assertEqual(body, b"<h1>Papplu</h1>")
                self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
                self.assertEqual(headers["Cache-Control"], "no-store")
                status, headers, body = self.request("HEAD", "/game.mjs")
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], "text/javascript; charset=utf-8")
                self.assertGreater(int(headers["Content-Length"]), 0)
                self.assertEqual(body, b"")

    def test_foreign_origin_and_host_rejected(self):
        for headers in (
            {"Origin": "https://example.com"},
            {"Origin": "null"},
            {"Host": f"example.com:{self.server.server_port}"},
        ):
            with self.subTest(headers=headers):
                status, _, _ = self.request("GET", "/", headers=headers)
                self.assertEqual(status, 403)

    def test_busy_is_not_a_losing_hand(self):
        self.server.evaluation_slots.acquire()
        self.server.evaluation_slots.acquire()
        try:
            status, _, result = self.post(payload())
        finally:
            self.server.evaluation_slots.release()
            self.server.evaluation_slots.release()
        self.assertEqual(status, 503)
        self.assertIn("error", result)
        self.assertNotIn("is_valid", result)

    def test_timeout_is_not_a_losing_hand(self):
        with patch.object(self.server, "evaluate", side_effect=EvaluationTimeout("Too slow.")):
            status, _, result = self.post(payload())
        self.assertEqual(status, 503)
        self.assertEqual(result, {"error": "Too slow."})


if __name__ == "__main__":
    unittest.main()
