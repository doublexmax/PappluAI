from __future__ import annotations

from collections import deque
import math
from numbers import Real
import random
from typing import Any, Deque, Dict, List, Sequence, Tuple, TYPE_CHECKING

from src.environment import NUM_ACTIONS, STATE_DIM

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from src.model import QNetwork

__all__ = (
    "Episode",
    "EpisodeReplay",
    "Transition",
    "discounted_returns",
    "train_batch",
)

Transition = Tuple[Tuple[float, ...], int, float]
Episode = Tuple[Transition, ...]


class EpisodeReplay:
    STATE_VERSION = 1

    def __init__(self, capacity: int) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity < 1
        ):
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._episodes: Deque[Episode] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def append(self, episode: Episode) -> None:
        stored = tuple(episode[-self._capacity:])
        if not stored:
            raise ValueError("cannot append an empty episode")
        self._episodes.append(stored)
        self._size += len(stored)

        excess = self._size - self._capacity
        while excess > 0:
            oldest = self._episodes[0]
            if len(oldest) <= excess:
                self._episodes.popleft()
                self._size -= len(oldest)
                excess -= len(oldest)
            else:
                self._episodes[0] = oldest[excess:]
                self._size -= excess
                excess = 0

    def sample(self, rng: random.Random, batch_size: int) -> List[Transition]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if not self._episodes:
            raise ValueError("cannot sample empty replay")
        batch: List[Transition] = []
        for _ in range(batch_size):
            episode = self._episodes[rng.randrange(len(self._episodes))]
            batch.append(episode[rng.randrange(len(episode))])
        return batch

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "capacity": self._capacity,
            "episodes": tuple(self._episodes),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("replay state must be a dict")
        if set(state) != {"version", "capacity", "episodes"}:
            raise ValueError("replay state has unexpected fields")
        if state["version"] != self.STATE_VERSION:
            raise ValueError(
                "unsupported replay state version %r" % (state["version"],)
            )
        capacity = state["capacity"]
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity != self._capacity
        ):
            raise ValueError(
                "replay capacity %r does not match expected %d"
                % (capacity, self._capacity)
            )
        raw_episodes = state["episodes"]
        if not isinstance(raw_episodes, tuple):
            raise TypeError("replay episodes must be a tuple")

        episodes: Deque[Episode] = deque()
        size = 0
        for episode_index, raw_episode in enumerate(raw_episodes):
            if not isinstance(raw_episode, tuple) or not raw_episode:
                raise ValueError(
                    "replay episode %d must be a nonempty tuple" % episode_index
                )
            episode: List[Transition] = []
            for transition_index, raw_transition in enumerate(raw_episode):
                if not isinstance(raw_transition, tuple) or len(raw_transition) != 3:
                    raise ValueError(
                        "replay transition %d:%d must be a three-item tuple"
                        % (episode_index, transition_index)
                    )
                raw_observation, raw_action, raw_target = raw_transition
                if (
                    not isinstance(raw_observation, tuple)
                    or len(raw_observation) != STATE_DIM
                ):
                    raise ValueError(
                        "replay observation %d:%d must contain %d values"
                        % (episode_index, transition_index, STATE_DIM)
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in raw_observation
                ):
                    raise ValueError(
                        "replay observation %d:%d must contain finite numbers"
                        % (episode_index, transition_index)
                    )
                if (
                    isinstance(raw_action, bool)
                    or not isinstance(raw_action, int)
                    or not 0 <= raw_action < NUM_ACTIONS
                ):
                    raise ValueError(
                        "replay action %d:%d must be in 0..%d"
                        % (
                            episode_index,
                            transition_index,
                            NUM_ACTIONS - 1,
                        )
                    )
                if (
                    isinstance(raw_target, bool)
                    or not isinstance(raw_target, Real)
                    or not math.isfinite(float(raw_target))
                ):
                    raise ValueError(
                        "replay target %d:%d must be finite"
                        % (episode_index, transition_index)
                    )
                episode.append(raw_transition)
            episodes.append(tuple(episode))
            size += len(episode)
        if size > self._capacity:
            raise ValueError(
                "replay contains %d transitions, capacity is %d"
                % (size, self._capacity)
            )

        self._episodes = episodes
        self._size = size


def discounted_returns(
    rewards: Sequence[float],
    gamma: float,
) -> List[float]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1], got %r" % (gamma,))
    returns: List[float] = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def train_batch(
    network: "QNetwork",
    optimizer: "torch.optim.Optimizer",
    loss_fn: "nn.Module",
    batch: Sequence[Transition],
    device: "torch.device",
    grad_clip: float,
) -> float:
    import torch

    states = torch.tensor(
        [transition[0] for transition in batch],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        [transition[1] for transition in batch],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [transition[2] for transition in batch],
        dtype=torch.float32,
        device=device,
    )

    network.train()
    selected = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = loss_fn(selected, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())
