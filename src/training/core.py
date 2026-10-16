from __future__ import annotations

from collections import deque
import math
from numbers import Real
import random
from typing import Any, Deque, Dict, List, Sequence, Tuple

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, NUM_ACTIONS,
    Observation, discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationHistory

__all__ = (
    "Episode",
    "EpisodeReplay",
    "Transition",
    "discounted_returns",
)

Transition = Tuple[Observation, int, float]
Episode = Tuple[Transition, ...]


class EpisodeReplay:
    STATE_VERSION = 2

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
        self._sample_cursor = 0

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
        by_count: Dict[int, List[Episode]] = {}
        for episode in self._episodes:
            by_count.setdefault(len(episode[0][0].players), []).append(episode)
        counts = sorted(by_count)
        batch: List[Transition] = []
        for index in range(batch_size):
            episodes = by_count[counts[(self._sample_cursor + index) % len(counts)]]
            episode = episodes[rng.randrange(len(episodes))]
            batch.append(episode[rng.randrange(len(episode))])
        self._sample_cursor += batch_size
        return batch

    def state_dict(self) -> Dict[str, Any]:
        episodes = []
        for episode in self._episodes:
            final = episode[-1][0]
            history = final.history
            transitions = []
            previous_prefix = 0
            for obs, action, target in episode:
                prefix = len(obs.history)
                if (
                    obs.config != final.config or obs.joker != final.joker
                    or len(obs.players) != len(final.players)
                    or obs.warm_start != final.warm_start
                ):
                    raise ValueError("replay episode changes journal configuration")
                if prefix < previous_prefix or obs.history != history[:prefix]:
                    raise ValueError("replay episode mixes seat or episode histories")
                transitions.append((obs.to_snapshot(), prefix, action, target))
                previous_prefix = prefix
            episodes.append({
                "config": final.config.to_dict(),
                "joker": final.joker,
                "player_count": len(final.players),
                "history": tuple(event.to_dict() for event in history),
                "transitions": tuple(transitions),
            })
        return {
            "version": self.STATE_VERSION,
            "capacity": self._capacity,
            "sample_cursor": self._sample_cursor,
            "episodes": tuple(episodes),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("replay state must be a dict")
        if set(state) != {"version", "capacity", "episodes", "sample_cursor"}:
            raise ValueError("replay state has unexpected fields")
        if type(state["version"]) is not int or state["version"] != self.STATE_VERSION:
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
        cursor = state["sample_cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("replay sample_cursor must be a nonnegative integer")

        episodes: Deque[Episode] = deque()
        size = 0
        for episode_index, raw_episode in enumerate(raw_episodes):
            if not isinstance(raw_episode, dict) or set(raw_episode) != {
                "config", "joker", "player_count", "history", "transitions",
            }:
                raise ValueError(
                    "replay episode %d must use the journal schema" % episode_index
                )
            config = GameConfig.from_dict(raw_episode["config"])
            history = InformationHistory.from_dict(
                raw_episode["history"], config, raw_episode["player_count"],
            )
            raw_transitions = raw_episode["transitions"]
            if not isinstance(raw_transitions, tuple) or not raw_transitions:
                raise ValueError("replay journal transitions must be a nonempty tuple")
            episode: List[Transition] = []
            previous_prefix = 0
            for transition_index, raw_transition in enumerate(raw_transitions):
                if not isinstance(raw_transition, tuple) or len(raw_transition) != 4:
                    raise ValueError(
                        "replay transition %d:%d must be a four-item tuple"
                        % (episode_index, transition_index)
                    )
                raw_observation, prefix, raw_action, raw_target = raw_transition
                observation = Observation.from_snapshot(
                    raw_observation, history=history, prefix=prefix,
                    joker=raw_episode["joker"],
                )
                if prefix < previous_prefix:
                    raise ValueError("replay history prefix lengths must be nondecreasing")
                if episode and observation.warm_start != episode[0][0].warm_start:
                    raise ValueError("replay episode changes journal configuration")
                previous_prefix = prefix
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
                if not legal_action_mask(observation)[raw_action]:
                    raise ValueError("replay contains an illegal action")
                if prefix < len(history.events):
                    event = history.events[prefix]
                    expected_action = (
                        ACTION_DRAW_STOCK if event.kind is EventKind.DRAW_STOCK
                        else ACTION_TAKE_DISCARD if event.kind is EventKind.TAKE_DISCARD
                        else discard_action(event.face) if event.kind is EventKind.DISCARD
                        else None
                    )
                    if event.actor != 0 or raw_action != expected_action:
                        raise ValueError("replay action does not match the next history event")
                if (
                    isinstance(raw_target, bool)
                    or not isinstance(raw_target, Real)
                    or not math.isfinite(float(raw_target))
                ):
                    raise ValueError(
                        "replay target %d:%d must be finite"
                        % (episode_index, transition_index)
                    )
                episode.append((observation, raw_action, float(raw_target)))
            episodes.append(tuple(episode))
            size += len(episode)
        if size > self._capacity:
            raise ValueError(
                "replay contains %d transitions, capacity is %d"
                % (size, self._capacity)
            )

        self._episodes = episodes
        self._size = size
        self._sample_cursor = cursor


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
