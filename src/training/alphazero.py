from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import json
import math
import random
import sys
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from src.evaluation.arena import play_match
from src.game.environment import GameConfig, NUM_ACTIONS, Observation, legal_action_mask
from src.game.multiplayer import MatchConfig
from src.model.encoding import collate_observations
from src.model.network import QNetwork, load_checkpoint
from src.model.policy_value import PolicyValueNet
from src.model.puct import PUCTConfig, PUCTPolicy, PUCTResult, puct_search
from src.training.core import TerminalShaping, terminal_reward


@dataclass(frozen=True)
class AlphaZeroSample:
    observation: Observation
    action: int
    policy_target: Tuple[Tuple[int, float], ...]
    search_value: float
    outcome_value: float

    def __post_init__(self) -> None:
        mask = legal_action_mask(self.observation)
        if type(self.action) is not int or not 0 <= self.action < NUM_ACTIONS or not mask[self.action]:
            raise ValueError("sample action must be legal")
        seen = set()
        total = 0.0
        for action, probability in self.policy_target:
            if (
                type(action) is not int or not 0 <= action < NUM_ACTIONS
                or not mask[action] or action in seen
            ):
                raise ValueError("policy target must contain distinct legal actions")
            if (
                type(probability) not in (int, float)
                or not math.isfinite(probability) or probability < 0
            ):
                raise ValueError("policy target probabilities must be finite and nonnegative")
            seen.add(action)
            total += probability
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("policy target probabilities must sum to one")
        for value in (self.search_value, self.outcome_value):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("sample values must be finite")


@dataclass(frozen=True)
class AlphaZeroLoss:
    total_loss: float
    value_loss: float
    policy_loss: float
    q_loss: float


class AlphaZeroReplay:
    STATE_VERSION = 1

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._samples: Deque[AlphaZeroSample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def append_samples(self, samples: Sequence[AlphaZeroSample]) -> None:
        stored = tuple(samples)
        if any(not isinstance(sample, AlphaZeroSample) for sample in stored):
            raise TypeError("replay accepts AlphaZeroSample items")
        self._samples.extend(stored)

    def sample(self, rng: random.Random, batch_size: int) -> Tuple[AlphaZeroSample, ...]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self._samples:
            raise ValueError("cannot sample empty replay")
        return tuple(rng.choices(tuple(self._samples), k=batch_size))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "capacity": self.capacity,
            "samples": tuple({
                "observation": sample.observation.to_dict(),
                "action": sample.action,
                "policy_target": sample.policy_target,
                "search_value": sample.search_value,
                "outcome_value": sample.outcome_value,
            } for sample in self._samples),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"version", "capacity", "samples"}:
            raise ValueError("replay state fields do not match")
        if type(state["version"]) is not int or state["version"] != self.STATE_VERSION:
            raise ValueError("unsupported replay state version")
        if type(state["capacity"]) is not int or state["capacity"] != self.capacity:
            raise ValueError("replay capacity does not match")
        if not isinstance(state["samples"], tuple) or len(state["samples"]) > self.capacity:
            raise ValueError("replay samples must fit the buffer")
        restored = []
        for raw in state["samples"]:
            values = dict(raw)
            values["observation"] = Observation.from_dict(values["observation"])
            restored.append(AlphaZeroSample(**values))
        self._samples = deque(restored, maxlen=self.capacity)


def _validate_losses(
    value_weight: float, policy_weight: float, q_weight: float,
    bootstrap_blend: float, grad_clip: float,
) -> None:
    for name, value in (
        ("value_weight", value_weight), ("policy_weight", policy_weight),
        ("q_weight", q_weight), ("bootstrap_blend", bootstrap_blend),
        ("grad_clip", grad_clip),
    ):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("%s must be finite and nonnegative" % name)
    if value_weight + policy_weight + q_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if bootstrap_blend > 1 or grad_clip == 0:
        raise ValueError("bootstrap_blend must be in [0, 1] and grad_clip must be positive")


def train_alphazero_batch(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[AlphaZeroSample],
    *,
    device: torch.device = torch.device("cpu"),
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    q_weight: float = 0.5,
    bootstrap_blend: float = 0.5,
    grad_clip: float = 1.0,
) -> AlphaZeroLoss:
    if not batch:
        raise ValueError("training batch must not be empty")
    _validate_losses(value_weight, policy_weight, q_weight, bootstrap_blend, grad_clip)
    observations = [sample.observation for sample in batch]
    if any(obs.config != model.game_config for obs in observations):
        raise ValueError("training observations must match the model game_config")
    states = collate_observations(observations, device)
    mask = torch.tensor(
        [legal_action_mask(obs) for obs in observations], dtype=torch.bool, device=device,
    )
    pi_target = torch.zeros((len(batch), NUM_ACTIONS), dtype=torch.float32, device=device)
    for row, sample in enumerate(batch):
        for action, probability in sample.policy_target:
            pi_target[row, action] = probability
    actions = torch.tensor([sample.action for sample in batch], dtype=torch.long, device=device)
    v_target = torch.tensor([
        (1 - bootstrap_blend) * sample.outcome_value + bootstrap_blend * sample.search_value
        for sample in batch
    ], dtype=torch.float32, device=device)
    model.train()
    out = model(states, mask)
    value_loss = F.smooth_l1_loss(out.position_value, v_target)
    policy_loss = (
        out.temperature.detach()
        * -(pi_target * out.log_policy.masked_fill(~mask, 0.0)).sum(dim=1)
    ).mean()
    q_loss = F.smooth_l1_loss(out.q.gather(1, actions[:, None]).squeeze(1), v_target)
    total = value_weight * value_loss + policy_weight * policy_loss + q_weight * q_loss
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return AlphaZeroLoss(
        float(total.detach()), float(value_loss.detach()),
        float(policy_loss.detach()), float(q_loss.detach()),
    )


@dataclass(frozen=True)
class AlphaZeroConfig:
    game: GameConfig = GameConfig()
    search: PUCTConfig = PUCTConfig(temperature=1.0)
    batch_size: int = 32
    replay_capacity: int = 5000
    learning_rate: float = 1e-3
    updates_per_match: int = 4
    exploration_turns: int = 12
    gamma: float = 0.99
    shaping: TerminalShaping = TerminalShaping()
    value_weight: float = 1.0
    policy_weight: float = 1.0
    q_weight: float = 0.5
    bootstrap_blend: float = 0.5
    grad_clip: float = 1.0
    opponent_paths: Tuple[str, ...] = ()
    seed: int = 41

    def __post_init__(self) -> None:
        if not isinstance(self.game, GameConfig) or not isinstance(self.search, PUCTConfig):
            raise TypeError("game and search must use their domain types")
        if not isinstance(self.shaping, TerminalShaping):
            raise TypeError("shaping must be a TerminalShaping")
        MatchConfig(self.game, players=2)
        for name, minimum in (
            ("batch_size", 1), ("replay_capacity", 1), ("updates_per_match", 0),
            ("exploration_turns", 0), ("seed", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        if (
            type(self.learning_rate) not in (int, float)
            or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if type(self.gamma) not in (int, float) or not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not isinstance(self.opponent_paths, tuple) or any(
            not isinstance(path, str) or not path for path in self.opponent_paths
        ):
            raise ValueError("opponent_paths must be a tuple of checkpoint paths")
        _validate_losses(
            self.value_weight, self.policy_weight, self.q_weight,
            self.bootstrap_blend, self.grad_clip,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlphaZeroConfig":
        if set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("alphazero config fields do not match")
        values = dict(raw)
        values["game"] = GameConfig.from_dict(values["game"])
        values["search"] = PUCTConfig(**values["search"])
        values["shaping"] = TerminalShaping.from_dict(values["shaping"])
        return cls(**values)


class _SelfPlayPolicy(PUCTPolicy):
    def __init__(self, model: PolicyValueNet, config: AlphaZeroConfig, device: torch.device) -> None:
        super().__init__(model, config=config.search, device=device)
        self.exploration_turns = config.exploration_turns
        self.trajectory: list[Tuple[Observation, PUCTResult]] = []

    def act(self, obs: Observation, rng: random.Random) -> int:
        early = obs.config.max_turns - obs.turns_remaining < self.exploration_turns
        search_config = self.config if early else replace(
            self.config, temperature=0.0, dirichlet_eps=0.0,
        )
        result = puct_search(self.network, obs, search_config, rng, self.device)
        self.trajectory.append((obs, result))
        return result.action


class AlphaZeroSession:
    STATE_VERSION = 1

    def __init__(
        self,
        model: Union[PolicyValueNet, str],
        config: Optional[AlphaZeroConfig] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(model, str):
            model = PolicyValueNet.from_checkpoint(model, device=device)
        self.device = next(model.parameters()).device if device is None else torch.device(device)
        self.model = model.to(self.device)
        self.network = self.model.qnet
        self.config = config if config is not None else AlphaZeroConfig(
            game=model.game_config, gamma=model.gamma, shaping=model.shaping,
        )
        if self.config.game != model.game_config:
            raise ValueError("session game must match the model game_config")
        self.model.gamma = self.config.gamma
        self.model.shaping = self.config.shaping
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.replay = AlphaZeroReplay(self.config.replay_capacity)
        self.rng = random.Random(self.config.seed)
        self.replay_rng = random.Random(self.rng.getrandbits(64))
        opponent_search = replace(self.config.search, temperature=0.0, dirichlet_eps=0.0)
        self.opponents = tuple(
            PUCTPolicy.from_checkpoint(path, opponent_search, device=self.device)
            for path in self.config.opponent_paths
        )
        for opponent in self.opponents:
            if opponent.game_config != self.config.game:
                raise ValueError("opponent game_config must match the session")
            opponent.network.eval().requires_grad_(False)
        self.completed_matches = 0
        self.total_updates = 0
        self.last_samples: Tuple[AlphaZeroSample, ...] = ()
        self.last_losses: Tuple[AlphaZeroLoss, ...] = ()

    def train_match(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None and type(seed) is not int:
            raise ValueError("match seed must be an integer")
        match_seed = self.rng.getrandbits(63) if seed is None else seed
        policies = [
            _SelfPlayPolicy(self.model, self.config, self.device),
            _SelfPlayPolicy(self.model, self.config, self.device),
        ]
        if self.opponents:
            policies[1 - self.completed_matches % 2] = self.rng.choice(self.opponents)
        result = play_match(
            policies, MatchConfig(self.config.game, players=2), match_seed,
            record_trajectories=True, record_terminal=True,
        )
        terminal = result.terminal_snapshot
        if terminal is None:
            raise RuntimeError("self-play did not return terminal hands")
        samples = []
        for seat, policy in enumerate(policies):
            if not isinstance(policy, _SelfPlayPolicy):
                continue
            outcome = terminal_reward(
                self.config.shaping, won=result.winner == seat,
                turns_remaining=self.config.game.max_turns - result.seat_turns[seat],
                max_turns=self.config.game.max_turns, hand=terminal.hands[seat],
                joker=terminal.joker, required_sequences=self.config.game.required_sequences,
            )
            for index, (obs, search) in enumerate(policy.trajectory):
                samples.append(AlphaZeroSample(
                    obs, search.action, search.policy_target, search.root_value,
                    outcome * self.config.gamma ** (len(policy.trajectory) - 1 - index),
                ))
        self.last_samples = tuple(samples)
        self.replay.append_samples(self.last_samples)
        losses = []
        for _ in range(self.config.updates_per_match):
            losses.append(train_alphazero_batch(
                self.model, self.optimizer,
                self.replay.sample(self.replay_rng, self.config.batch_size),
                device=self.device, value_weight=self.config.value_weight,
                policy_weight=self.config.policy_weight, q_weight=self.config.q_weight,
                bootstrap_blend=self.config.bootstrap_blend, grad_clip=self.config.grad_clip,
            ))
        self.last_losses = tuple(losses)
        self.total_updates += len(losses)
        self.completed_matches += 1
        return {
            "algorithm": "alphazero_puct",
            "matches": self.completed_matches,
            "seed": match_seed,
            "winner": result.winner,
            "terminal_reason": result.terminal_reason,
            "actions": result.action_count,
            "samples": len(samples),
            "replay": len(self.replay),
            "train_updates": self.total_updates,
            "loss": asdict(losses[-1]) if losses else None,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": self.config.to_dict(),
            "completed_matches": self.completed_matches,
            "total_updates": self.total_updates,
            "replay": self.replay.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.getstate(),
            "replay_rng": self.replay_rng.getstate(),
            "log_temperature": self.model.log_temperature.detach().cpu().clone(),
        }

    def save_checkpoint(self, path: str) -> str:
        return self.model.save(path, {"alphazero": self.state_dict()})

    @classmethod
    def from_checkpoint(
        cls, path: str, config: Optional[AlphaZeroConfig] = None, *,
        device: Optional[torch.device] = None,
    ) -> "AlphaZeroSession":
        model = PolicyValueNet.from_checkpoint(path, device=device)
        _, _, meta = load_checkpoint(path, device=device)
        state = meta.get("alphazero")
        if state is None:
            return cls(model, config, device=device)
        if type(state["version"]) is not int or state["version"] != cls.STATE_VERSION:
            raise ValueError("unsupported alphazero session version")
        saved_config = AlphaZeroConfig.from_dict(state["config"])
        if config is not None and config != saved_config:
            raise ValueError("resume config must match the saved session")
        session = cls(model, saved_config, device=device)
        for name in ("completed_matches", "total_updates"):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
            setattr(session, name, value)
        session.replay.load_state_dict(state["replay"])
        session.optimizer.load_state_dict(state["optimizer"])
        session.rng.setstate(state["rng"])
        session.replay_rng.setstate(state["replay_rng"])
        with torch.no_grad():
            session.model.log_temperature.copy_(state["log_temperature"])
        return session

    def export_model(self, path: str, meta: Optional[Mapping[str, Any]] = None) -> str:
        return self.model.save(path, {
            **dict(meta or {}), "algorithm": "alphazero_puct",
            "completed_matches": self.completed_matches, "total_updates": self.total_updates,
        })


def main(args: Any) -> int:
    try:
        if args.matches < 0 or args.torch_threads < 1:
            raise ValueError("matches must be nonnegative and torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
        device = torch.device(args.device)
        if args.resume:
            session = AlphaZeroSession.from_checkpoint(args.resume, device=device)
        else:
            torch.manual_seed(args.seed)
            settings = {
                name: getattr(args, name)
                for name in ("num_decks", "cards_in_hand", "required_sequences", "max_turns")
                if getattr(args, name) is not None
            }
            if args.load:
                model = PolicyValueNet.from_checkpoint(args.load, device=device)
                if any(getattr(model.game_config, name) != value for name, value in settings.items()):
                    raise ValueError("explicit game settings must match the loaded checkpoint")
            else:
                game = GameConfig(**settings)
                model = PolicyValueNet(QNetwork().to(device), game)
            config = AlphaZeroConfig(
                game=model.game_config,
                search=PUCTConfig(
                    simulations=args.simulations, max_stock_outcomes=args.stock_outcomes,
                    opponent_outcomes=args.opponent_outcomes, temperature=args.temperature,
                    dirichlet_eps=args.dirichlet_eps,
                ),
                batch_size=args.batch_size, replay_capacity=args.replay_capacity,
                learning_rate=args.lr, updates_per_match=args.updates_per_match,
                exploration_turns=args.exploration_turns,
                gamma=model.gamma if args.gamma is None else args.gamma,
                shaping=model.shaping, bootstrap_blend=args.bootstrap_blend,
                grad_clip=args.grad_clip, opponent_paths=tuple(args.opponent), seed=args.seed,
            )
            session = AlphaZeroSession(model, config, device=device)
        for _ in range(args.matches):
            print(json.dumps(session.train_match(), sort_keys=True), flush=True)
        summary = {
            "algorithm": "alphazero_puct", "matches": session.completed_matches,
            "train_updates": session.total_updates, "replay": len(session.replay),
        }
        if args.checkpoint:
            summary["checkpoint_sha256"] = session.save_checkpoint(args.checkpoint)
        if args.export:
            summary["export_sha256"] = session.export_model(args.export)
        print(json.dumps(summary, sort_keys=True))
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return 0
