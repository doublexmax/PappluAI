from __future__ import annotations

import argparse
import json
import multiprocessing
import socket
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import urlsplit

from .evaluate import evaluate_hand


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/game.mjs": ("game.mjs", "text/javascript; charset=utf-8"),
    "/cards.mjs": ("cards.mjs", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 8192
EVALUATION_SECONDS = 10


class EvaluationTimeout(Exception):
    pass


class EvaluationFailure(Exception):
    pass


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("The request must be a JSON object.")
    allowed = {"hand", "joker", "required_sequences", "cards_in_hand"}
    if set(payload) - allowed:
        raise ValueError("Unknown evaluation fields.")
    hand = payload.get("hand")
    if not isinstance(hand, list) or len(hand) != 52:
        raise ValueError("hand must contain 52 card counts.")
    if any(type(count) is not int or not 0 <= count <= 6 for count in hand):
        raise ValueError("Each card count must be an integer from 0 through 6.")
    if sum(hand) > 31:
        raise ValueError("The simulator accepts at most 31 cards.")
    values = {
        "joker": (payload.get("joker"), 0, 51),
        "required_sequences": (payload.get("required_sequences", 5), 0, 10),
        "cards_in_hand": (payload.get("cards_in_hand", 21), 3, 30),
    }
    result = {"hand": hand}
    for name, (value, minimum, maximum) in values.items():
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} through {maximum}.")
        result[name] = value
    return result


def _evaluate_child(payload, connection):
    with connection:
        result = evaluate_hand(**payload)
        response = asdict(result)
        response["reward"] = float(result.is_valid)
        connection.send(response)


def evaluate_request(payload, timeout=EVALUATION_SECONDS):
    # A thread timeout cannot stop an expensive exact search.
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_evaluate_child, args=(payload, sender), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise EvaluationTimeout(
                f"Evaluation exceeded {timeout:g} seconds. "
                "Try fewer wildcards or a smaller hand. This is not a losing-hand verdict."
            )
        try:
            return receiver.recv()
        except EOFError as exc:
            raise EvaluationFailure("The evaluator stopped unexpectedly. See the server output.") from exc
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()


class SimulatorHandler(BaseHTTPRequestHandler):
    server: SimulatorServer

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _json(self, status, value):
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _allowed_request(self):
        port = self.server.server_port
        hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
        if self.headers.get("Host") not in hosts:
            self._json(403, {"error": "Only local simulator requests are accepted."})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{host}" for host in hosts}:
            self._json(403, {"error": "Cross-origin requests are not accepted."})
            return False
        return True

    def do_GET(self):
        if not self._allowed_request():
            return
        asset = ASSETS.get(urlsplit(self.path).path)
        if asset is None:
            self._json(404, {"error": "Not found."})
            return
        filename, content_type = asset
        try:
            body = (WEB_ROOT / filename).read_bytes()
        except FileNotFoundError:
            self._json(404, {"error": f"Simulator asset {filename} is missing."})
            return
        self._headers(200, content_type, len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if not self._allowed_request():
            return
        if urlsplit(self.path).path != "/api/evaluate":
            self._json(404, {"error": "Not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "Invalid Content-Length."})
            return
        if not 0 < length <= MAX_BODY_BYTES:
            self._json(413, {"error": "The request body must be between 1 and 8192 bytes."})
            return
        self.connection.settimeout(5)
        try:
            payload = validate_payload(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except socket.timeout:
            self._json(408, {"error": "Timed out reading the request."})
            return
        if not self.server.evaluation_slots.acquire(blocking=False):
            self._json(503, {"error": "The evaluator is busy. Try again shortly."})
            return
        try:
            result = self.server.evaluate(payload)
        except EvaluationTimeout as exc:
            self.log_error("%s", exc)
            self._json(503, {"error": str(exc)})
        except EvaluationFailure as exc:
            self.log_error("%s", exc)
            self._json(500, {"error": str(exc)})
        else:
            self._json(200, result)
        finally:
            self.server.evaluation_slots.release()


class SimulatorServer(ThreadingHTTPServer):
    def __init__(self, port=8765, evaluator=evaluate_request):
        super().__init__(("127.0.0.1", port), SimulatorHandler)
        self.evaluate = evaluator
        self.evaluation_slots = BoundedSemaphore(2)


def main():
    parser = argparse.ArgumentParser(description="Run the local Papplu hand builder and simulator.")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    try:
        server = SimulatorServer(arguments.port)
    except OSError as exc:
        parser.exit(2, f"Cannot start the simulator: {exc}\n")
    print(f"Papplu simulator: http://127.0.0.1:{server.server_port}", flush=True)
    print("Press Ctrl+C to stop. Game state stays in the browser and resets on reload.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
