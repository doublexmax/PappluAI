from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class BlobStore:
    def __init__(self, base: str, sas: str) -> None:
        if not base.startswith("https://"):
            raise ValueError("Artifact storage must use HTTPS")
        self.base = base.rstrip("/")
        self.sas = sas

    def request(self, name: str, data: Optional[bytes] = None, optional: bool = False):
        headers = {}
        if data is not None:
            headers = {
                "x-ms-blob-type": "BlockBlob",
                "Content-MD5": base64.b64encode(hashlib.md5(data).digest()).decode("ascii"),
            }
        request = urllib.request.Request(
            self.base + "/" + urllib.parse.quote(name, safe="/") + "?" + self.sas,
            data=data, headers=headers, method="GET" if data is None else "PUT",
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if optional and data is None and exc.code == 404:
                    return None
                if exc.code < 500 and exc.code not in (408, 429):
                    raise RuntimeError("Blob request failed for %s: HTTP %d" % (name, exc.code)) from None
                error = "HTTP %d" % exc.code
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = type(exc).__name__
            if attempt < 3:
                time.sleep(2 ** attempt)
        raise RuntimeError("Blob request failed for %s: %s" % (name, error))

    def get_json(self, name: str):
        data = self.request(name, optional=True)
        return None if data is None else json.loads(data)

    def put_json(self, name: str, value: dict) -> None:
        self.request(name, json.dumps(value, sort_keys=True).encode("utf-8"))


def verify(content: bytes, expected: dict) -> bool:
    return (
        len(content) == expected["bytes"]
        and hashlib.sha256(content).hexdigest() == expected["sha256"]
    )


def restore_snapshot(store: BlobStore, pointer: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    for name, metadata in pointer["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Unsafe snapshot filename")
        content = store.request(metadata["blob"])
        if not verify(content, metadata):
            raise RuntimeError("Cloud snapshot checksum mismatch: " + name)
        (output / name).write_bytes(content)
    status = store.request("checkpoints/%d/status.json" % pointer["slot"])
    if hashlib.sha256(status).hexdigest() != pointer["generation"]:
        raise RuntimeError("Cloud status checksum mismatch")
    (output / "status.json").write_bytes(status)
    return output / "latest-state.pt"


def training_required(
    pointer, episode_limit: int, deadline: float, now: float,
    counter: str = "completed_episodes",
) -> bool:
    if now >= deadline:
        return False
    return pointer is None or pointer["training_status"][counter] < episode_limit


def learner_contract(module: str) -> tuple:
    if module == "src.long_train":
        return "--max-episodes", "completed_episodes"
    if module == "src.improve":
        return "--max-cycles", "completed_cycles"
    raise ValueError("Unsupported cloud training module")


class Publisher:
    def __init__(self, store: BlobStore, run_id: str, source_hash: str) -> None:
        self.store = store
        self.run_id = run_id
        self.source_hash = source_hash
        self.pointer = store.get_json("latest.json")
        if self.pointer and (
            self.pointer["run_id"] != run_id
            or self.pointer["source_sha256"] != source_hash
        ):
            raise ValueError("Existing checkpoint belongs to a different run or source")

    def sync(self, output: Path) -> bool:
        status_path = output / "status.json"
        if not status_path.exists():
            return False
        status_bytes = status_path.read_bytes()
        status = json.loads(status_bytes)
        if status.get("status_schema_version") != 1:
            raise ValueError("Unsupported training status schema")
        files = status.get("files")
        if not files:
            return False
        contents = {}
        for name, metadata in files.items():
            if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError("Unsafe checkpoint filename")
            path = output / name
            if not path.exists():
                return False
            content = path.read_bytes()
            if not verify(content, metadata):
                return False
            contents[name] = content
        if status_path.read_bytes() != status_bytes:
            return False
        required_files = {"latest-state.pt", "latest-model.pt", "best-model.pt", "baseline-model.pt", "metrics.jsonl"}
        if not required_files <= files.keys():
            raise ValueError("Training status lacks a resumable state or required model export")
        generation = hashlib.sha256(status_bytes).hexdigest()
        if self.pointer and self.pointer["generation"] == generation:
            return True
        slot = 1 - self.pointer["slot"] if self.pointer else 0
        remote_files = {}
        for name, content in contents.items():
            blob = "checkpoints/%d/%s" % (slot, name)
            self.store.request(blob, content)
            remote_files[name] = {**files[name], "blob": blob}
        self.store.request("checkpoints/%d/status.json" % slot, status_bytes)
        pointer = {
            "run_id": self.run_id,
            "source_sha256": self.source_hash,
            "generation": generation,
            "slot": slot,
            "committed_utc": utc_now(),
            "files": remote_files,
            "training_status": status,
        }
        self.store.put_json("latest.json", pointer)
        self.pointer = pointer
        return True

    def heartbeat(self, phase: str, extra: Optional[dict] = None) -> None:
        status = self.pointer["training_status"] if self.pointer else {}
        event = {
            "run_id": self.run_id,
            "utc": utc_now(),
            "phase": phase,
            "checkpoint_utc": self.pointer["committed_utc"] if self.pointer else None,
            "checkpoint": {
                "episodes": status.get("completed_episodes"),
                "updates": status.get("updates"),
                "stage": status.get("stage"),
                "distance": status.get("distance"),
                "best_full_win_rate": status.get("best_score"),
            },
            **(extra or {}),
        }
        try:
            self.store.put_json("heartbeat.json", event)
        except RuntimeError as error:
            event["heartbeat_upload_failed"] = True
            print("warning: %s" % error, file=sys.stderr, flush=True)
        print(json.dumps(event), flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish_log(store: BlobStore, path: Path, name: str) -> None:
    if not path.exists():
        return
    try:
        store.request(name, path.read_bytes())
    except RuntimeError as error:
        print("warning: log upload failed: %s" % error, file=sys.stderr, flush=True)


def run_process(command, output: Path, publisher: Publisher, deadline: float, name: str):
    log_path = output.parent / (name + ".log")
    output.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while process.poll() is None:
                synced = publisher.sync(output)
                publisher.heartbeat(name, {"checkpoint_coherent": synced})
                if time.time() >= deadline:
                    raise TimeoutError(name + " reached the server-side deadline")
                time.sleep(15 if name == "smoke" else 30)
        if process.returncode:
            raise RuntimeError("%s exited with code %d" % (name, process.returncode))
        if not publisher.sync(output):
            raise RuntimeError(name + " ended without a coherent checkpoint")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        publisher.sync(output)
        publish_log(publisher.store, log_path, "logs/" + log_path.name)


def main() -> int:
    store = BlobStore(os.environ["BLOB_BASE_URL"], os.environ["BLOB_SAS"])
    run_id = os.environ["RUN_ID"]
    source_hash = os.environ["BUNDLE_SHA256"]
    deadline = datetime.fromisoformat(os.environ["DEADLINE_UTC"]).timestamp()
    root = Path("/work")
    root.mkdir(exist_ok=True)
    os.chdir(root)
    publisher = Publisher(store, run_id, source_hash)
    if time.time() >= deadline:
        publisher.heartbeat("deadline_reached")
        return 0
    publisher.heartbeat("installing")
    source = store.request("source.zip")
    if hashlib.sha256(source).hexdigest() != source_hash:
        raise RuntimeError("Training source checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        archive.extractall(root)
    install_log = root / "install.log"
    try:
        with install_log.open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir",
                 "--disable-pip-version-check", "torch==2.8.0",
                 "--index-url", "https://download.pytorch.org/whl/cpu"],
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=min(600, max(1, deadline - time.time())),
            )
    finally:
        publish_log(store, install_log, "logs/install.log")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    smoke = os.environ.get("SMOKE_TEST") == "1"
    if smoke and store.get_json("resume-proof.json") is not None:
        publisher.heartbeat("smoke_completed")
        return 0
    if (
        smoke
        or os.environ.get("REVALIDATE_SOURCE") == "1"
        or os.environ.get("VALIDATE_BEFORE_TRAIN") == "1" and publisher.pointer is None
    ):
        test_log = root / "tests.log"
        try:
            with test_log.open("w", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
                    stdout=log, stderr=subprocess.STDOUT, check=True,
                    timeout=min(600, max(1, deadline - time.time())),
                )
        finally:
            store.request("logs/tests.log", test_log.read_bytes())
    training_arguments = json.loads(os.environ["TRAINING_ARGUMENTS"])
    module = os.environ.get("TRAINING_MODULE", "src.long_train")
    limit_argument, work_counter = learner_contract(module)
    command = [
        sys.executable, "-u", "-m", module, "--output-dir", str(output),
        *training_arguments,
    ]
    if publisher.pointer:
        state = restore_snapshot(store, publisher.pointer, output)
        command.extend(["--resume", str(state)])
    else:
        command.extend(["--initial-model", os.environ["INITIAL_MODEL"]])
    train_deadline = min(
        deadline - 120,
        datetime.fromisoformat(os.environ["TRAIN_DEADLINE_UTC"]).timestamp(),
    )
    episode_limit = int(training_arguments[training_arguments.index(limit_argument) + 1])
    if training_required(publisher.pointer, episode_limit, train_deadline, time.time(), work_counter):
        command.extend(["--max-seconds", str(max(1, int(train_deadline - time.time())))])
        run_process(command, output, publisher, train_deadline + 60, "smoke" if smoke else "training")
    elif publisher.pointer is None:
        raise RuntimeError("Training window ended without a committed checkpoint")
    publisher.heartbeat("training_completed")
    if smoke:
        before = publisher.pointer
        resumed = root / "resumed"
        state = restore_snapshot(store, before, resumed)
        resume_arguments = json.loads(os.environ["TRAINING_ARGUMENTS"])
        limit_index = resume_arguments.index(limit_argument) + 1
        resume_arguments[limit_index] = str(int(resume_arguments[limit_index]) + 2)
        resume_command = [
            sys.executable, "-u", "-m", module,
            "--resume", str(state), "--output-dir", str(resumed),
            *resume_arguments, "--max-seconds", str(max(1, int(deadline - time.time() - 60))),
        ]
        run_process(resume_command, resumed, publisher, deadline - 30, "smoke")
        if (
            publisher.pointer["files"]["latest-state.pt"]["sha256"] == before["files"]["latest-state.pt"]["sha256"]
            or publisher.pointer["training_status"][work_counter]
            != before["training_status"][work_counter] + 2
        ):
            raise RuntimeError("Resume smoke did not advance the checkpoint")
        store.put_json("resume-proof.json", {
            "run_id": run_id,
            "restored_generation": before["generation"],
            "resumed_generation": publisher.pointer["generation"],
            "before": before["training_status"],
            "after": publisher.pointer["training_status"],
        })
        publisher.heartbeat("smoke_completed")
        return 0
    results = {}
    evaluation_records = {}
    for label in ("best", "baseline"):
        if deadline - time.time() < 300:
            incomplete = {"completed_controls": list(results), "utc": utc_now()}
            store.put_json("final/incomplete.json", incomplete)
            publisher.heartbeat("confirmation_incomplete", incomplete)
            return 2
        model_hash = publisher.pointer["files"][label + "-model.pt"]["sha256"]
        cached = store.get_json("final/" + label + ".json")
        if cached is not None and cached.get("checkpoint_sha256") == model_hash:
            result = cached
        else:
            result_path = output / ("final-" + label + ".json")
            evaluation_command = [
                sys.executable, "-u", "-m", "src.benchmark",
                "--checkpoint", str(output / (label + "-model.pt")),
                "--games", "256", "--seed", os.environ.get("FINAL_EVALUATION_SEED", "40000000"), "--warm-games", "0",
                "--output", str(result_path),
            ]
            run_process(evaluation_command, output, publisher, deadline - 30, "final-" + label)
            result = json.loads(result_path.read_bytes())
            result["checkpoint_sha256"] = model_hash
            store.put_json("final/" + label + ".json", result)
        evaluation_records[label] = result
        results[label] = {
            "games": result["greedy"]["games"],
            "wins": result["greedy"]["wins"],
            "random_wins": result["random"]["wins"],
        }
    from src.benchmark import paired_comparison

    best = evaluation_records["best"]
    baseline = evaluation_records["baseline"]
    versus_baseline = paired_comparison([best["greedy"]["outcomes"]], [baseline["greedy"]["outcomes"]])
    versus_random = paired_comparison([best["greedy"]["outcomes"]], [best["random"]["outcomes"]])
    gain = all(
        comparison["absolute_gain"] >= 0.05 and comparison["ci95"][0] > 0
        for comparison in (versus_baseline, versus_random)
    )
    confirmation = {
        "scores": results,
        "versus_baseline": versus_baseline,
        "versus_random": versus_random,
        "holdout_gain_observed": gain,
        "scope": "One training run and 256 unseen full21 deals, not multi-seed confirmation.",
    }
    store.put_json("final/summary.json", confirmation)
    publisher.heartbeat("completed_with_holdout_gain" if gain else "completed_no_confirmed_gain", {"confirmation": results})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        store = BlobStore(os.environ["BLOB_BASE_URL"], os.environ["BLOB_SAS"])
        store.put_json("heartbeat.json", {
            "run_id": os.environ["RUN_ID"],
            "utc": utc_now(),
            "phase": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
