from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
import os
import threading
import time
from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, Union


Job = TypeVar("Job")
Result = TypeVar("Result")


@dataclass(frozen=True)
class WorkerResult(Generic[Job, Result]):
    job: Job
    value: Result
    elapsed_seconds: float


@dataclass(frozen=True)
class WorkerFailure(Generic[Job]):
    job: Job
    kind: str
    detail: str
    elapsed_seconds: float


@dataclass
class _Worker:
    process: BaseProcess
    connection: Connection
    launched: float
    ready: bool = False
    assignment: Optional[tuple] = None
    started: Optional[float] = None


def _exit_with_parent(parent: BaseProcess) -> None:
    while parent.is_alive():
        time.sleep(1)
    os._exit(1)


def _worker_main(connection: Connection, function: Callable, initializer: Optional[Callable]) -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()
    try:
        if initializer is not None:
            initializer()
        connection.send(("ready", None, None))
        while True:
            try:
                assignment = connection.recv()
            except EOFError:
                return
            if assignment is None:
                return
            ordinal, job = assignment
            started = time.monotonic()
            connection.send(("started", ordinal, started))
            result = function(job)
            connection.send(("result", ordinal, (time.monotonic() - started, result)))
    finally:
        connection.close()


class BoundedWorkers(Generic[Job, Result]):
    def __init__(
        self,
        function: Callable[[Job], Result],
        workers: int = 4,
        task_seconds: float = 120,
        initializer: Optional[Callable[[], None]] = None,
        startup_seconds: float = 30,
    ) -> None:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        for name, value in (("task_seconds", task_seconds), ("startup_seconds", startup_seconds)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        self.function = function
        self.workers = workers
        self.task_seconds = task_seconds
        self.initializer = initializer
        self.startup_seconds = startup_seconds
        self._context = multiprocessing.get_context("spawn")
        self._slots: list[_Worker] = []
        self._ordinal = 0
        self._open = False

    def __enter__(self) -> "BoundedWorkers[Job, Result]":
        if self._open:
            raise RuntimeError("worker pool is already open")
        self._open = True
        opened = False
        try:
            for _ in range(self.workers):
                self._slots.append(self._spawn())
            opened = True
            return self
        finally:
            if not opened:
                self.close()

    def __exit__(self, *exception) -> None:
        self.close()

    @property
    def worker_count(self) -> int:
        return sum(slot.process.is_alive() for slot in self._slots)

    def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main,
            args=(child, self.function, self.initializer),
            daemon=True,
        )
        started = False
        try:
            process.start()
            started = True
        finally:
            child.close()
            if not started:
                parent.close()
        return _Worker(process, parent, time.monotonic())

    @staticmethod
    def _stop(slot: _Worker) -> None:
        slot.connection.close()
        if slot.process.is_alive():
            slot.process.terminate()
        slot.process.join(timeout=2)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(timeout=2)
        if slot.process.is_alive():
            raise RuntimeError("could not stop owned worker %s" % slot.process.pid)
        slot.process.close()

    def close(self) -> None:
        self._open = False
        failures = []
        slots, self._slots = self._slots, []
        for slot in slots:
            try:
                self._stop(slot)
            except (OSError, RuntimeError) as error:
                failures.append(error)
        if failures:
            raise RuntimeError("failed to stop an owned worker") from failures[0]

    def run(
        self,
        jobs: Iterable[Job],
        on_tick: Optional[Callable[[int], None]] = None,
    ) -> Iterator[Union[WorkerResult[Job, Result], WorkerFailure[Job]]]:
        if not self._open:
            raise RuntimeError("use BoundedWorkers as a context manager")
        while len(self._slots) < self.workers:
            self._slots.append(self._spawn())
        iterator = iter(jobs)
        exhausted = object()
        pending = next(iterator, exhausted)
        while True:
            if not self._open:
                raise RuntimeError("worker pool was closed during execution")
            for slot in tuple(self._slots):
                disconnected = False
                while True:
                    try:
                        if not slot.connection.poll():
                            break
                        kind, ordinal, value = slot.connection.recv()
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        disconnected = True
                        break
                    if kind == "ready":
                        if slot.ready or slot.assignment is not None:
                            raise RuntimeError("worker sent an unexpected ready message")
                        slot.ready = True
                    elif slot.assignment is None or ordinal != slot.assignment[0]:
                        raise RuntimeError("worker response does not match its assignment")
                    elif kind == "started":
                        slot.started = value
                    elif kind == "result":
                        elapsed, result = value
                        job = slot.assignment[1]
                        slot.assignment = None
                        slot.started = None
                        if elapsed > self.task_seconds:
                            yield WorkerFailure(job, "timeout", "worker returned after its deadline", elapsed)
                        else:
                            yield WorkerResult(job, result, elapsed)
                        if not self._open:
                            raise RuntimeError("worker pool was closed during execution")
                    else:
                        raise RuntimeError("unknown worker response")
                now = time.monotonic()
                dead = disconnected or not slot.process.is_alive()
                startup_expired = not slot.ready and now - slot.launched > self.startup_seconds
                timed_out = slot.started is not None and now - slot.started > self.task_seconds
                start_missing = (
                    slot.assignment is not None and slot.started is None
                    and now - slot.assignment[2] > self.startup_seconds
                )
                if dead or startup_expired or timed_out or start_missing:
                    assignment = slot.assignment
                    elapsed = now - (slot.started if slot.started is not None else slot.launched)
                    if dead:
                        slot.process.join(timeout=0.2)
                        code = slot.process.exitcode
                        detail = "worker exited with code %s" % code if code is not None else "worker closed its result channel"
                    else:
                        detail = "worker exceeded its deadline"
                    self._slots.remove(slot)
                    self._stop(slot)
                    if assignment is None:
                        raise RuntimeError("worker startup or idle failure: " + detail)
                    yield WorkerFailure(assignment[1], "worker_exit" if dead else "timeout", detail, elapsed)
                    if not self._open:
                        raise RuntimeError("worker pool was closed during execution")
                    if pending is not exhausted:
                        self._slots.append(self._spawn())
                    continue
                if slot.ready and slot.assignment is None and pending is not exhausted:
                    ordinal = self._ordinal
                    self._ordinal += 1
                    slot.assignment = (ordinal, pending, time.monotonic())
                    slot.started = None
                    slot.connection.send((ordinal, pending))
                    pending = next(iterator, exhausted)
            active = sum(slot.assignment is not None for slot in self._slots)
            if on_tick is not None:
                on_tick(active)
            if pending is exhausted and not active:
                return
            if not self._slots:
                self._slots.append(self._spawn())
            wait([slot.connection for slot in self._slots], timeout=0.05)
