"""Episodic Monte Carlo Q-value regression for solo Papplu.

Each episode records (state, action, reward). After the episode ends, discounted
returns G_t = r_t + gamma * G_{t+1} become regression targets for Q(s_t, a_t).
There is no next-state max, target network, or tree search. This is not DQN.

Optional --warm-start-fraction mixes evaluator-confirmed winning-hand-plus-extra
discard-phase starts with ordinary random deals. Warm-start metrics are labeled
separately and are not held-out full-game wins.
"""

from __future__ import annotations

from collections import deque
import json
import math
import random
from typing import Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import src.training.core as training_core
import src.training.optimization as training_optimization
from src.game.environment import (
    ENCODING_VERSION,
    GameConfig,
    Observation,
    PappluEnv,
    encode_observation,
)
from src.model.network import (
    CHECKPOINT_VERSION,
    QNetwork,
    load_checkpoint,
    save_checkpoint,
    select_action,
    select_greedy_action,
    select_random_legal,
)

EpisodeStep = Tuple[Tuple[float, ...], int, float]


def run_episode(
    env: PappluEnv,
    network: "QNetwork",
    epsilon: float,
    rng: random.Random,
    warm_start: bool = False,
    device: Optional["torch.device"] = None,
) -> Tuple[List[EpisodeStep], Observation]:
    obs = env.reset_warm_start() if warm_start else env.reset()
    steps: List[EpisodeStep] = []
    while not obs.done:
        action = select_action(network, obs, epsilon=epsilon, rng=rng, device=device)
        state = encode_observation(obs)
        obs = env.step(action)
        steps.append((state, action, float(obs.last_reward)))
    return steps, obs


def train(
    episodes: int = 100,
    seed: int = 0,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 32,
    replay_capacity: int = 5000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 200,
    warm_start_fraction: float = 0.0,
    updates_per_episode: int = 4,
    grad_clip: float = 1.0,
    device: str = "cpu",
    torch_threads: int = 1,
    config: Optional[GameConfig] = None,
    checkpoint_path: Optional[str] = None,
    load_path: Optional[str] = None,
    eval_episodes: int = 0,
    eval_seed: int = 12345,
    log_every: int = 10,
    architecture: Optional[str] = None,
    replay_sampling: str = "transition",
) -> Dict[str, object]:
    """Train Q via episodic Monte Carlo regression. Returns summary metrics."""
    for name, value, minimum in (
        ("episodes", episodes, 0),
        ("batch_size", batch_size, 1),
        ("replay_capacity", replay_capacity, 1),
        ("epsilon_decay_episodes", epsilon_decay_episodes, 0),
        ("updates_per_episode", updates_per_episode, 1),
        ("eval_episodes", eval_episodes, 0),
        ("log_every", log_every, 0),
        ("torch_threads", torch_threads, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("%s must be an integer >= %d" % (name, minimum))
    for name, value in (
        ("gamma", gamma),
        ("warm_start_fraction", warm_start_fraction),
        ("epsilon_start", epsilon_start),
        ("epsilon_end", epsilon_end),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be in [0, 1]" % name)
    for name, value in (("lr", lr), ("grad_clip", grad_clip)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("%s must be finite and positive" % name)
    if epsilon_end > epsilon_start:
        raise ValueError("epsilon_end cannot exceed epsilon_start")
    if eval_episodes and eval_seed == seed:
        raise ValueError("eval_seed must differ from the training seed")
    if replay_sampling not in ("transition", "episode"):
        raise ValueError("replay_sampling must be 'transition' or 'episode'")

    cfg = config if config is not None else GameConfig()
    torch_device = torch.device(device)
    seed_source = random.Random(seed)
    source_rng = random.Random(seed_source.getrandbits(64))
    action_rng = random.Random(seed_source.getrandbits(64))
    replay_rng = random.Random(seed_source.getrandbits(64))
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)

    if load_path:
        network, cfg, _ = load_checkpoint(
            load_path,
            device=torch_device,
            expected_config=config,
            expected_architecture=architecture,
        )
    else:
        network = QNetwork(
            architecture="mlp" if architecture is None else architecture
        ).to(torch_device)
    if warm_start_fraction and cfg.cards_in_hand < 3:
        raise ValueError("warm starts require at least three cards")
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    env = PappluEnv(cfg)
    env.seed(seed)
    replay: Union[
        Deque[training_core.Transition],
        training_core.EpisodeReplay,
    ]
    if replay_sampling == "transition":
        replay = deque(maxlen=replay_capacity)
    else:
        replay = training_core.EpisodeReplay(replay_capacity)

    reward_sum = 0.0
    win_count = 0
    warm_episodes = 0
    normal_episodes = 0
    warm_wins = 0
    normal_wins = 0
    loss_sum = 0.0
    loss_count = 0
    return_sum = 0.0
    train_steps = 0

    def epsilon_at(episode_index: int) -> float:
        if epsilon_decay_episodes <= 0:
            return epsilon_end
        t = min(1.0, episode_index / float(epsilon_decay_episodes))
        return epsilon_start + (epsilon_end - epsilon_start) * t

    for ep in range(episodes):
        use_warm = source_rng.random() < warm_start_fraction
        eps = epsilon_at(ep)
        steps, final_obs = run_episode(
            env,
            network,
            epsilon=eps,
            rng=action_rng,
            warm_start=use_warm,
            device=torch_device,
        )
        rewards = [r for _, _, r in steps]
        returns = training_core.discounted_returns(rewards, gamma)
        completed_episode = tuple(
            (state, action, ret)
            for (state, action, _reward), ret in zip(steps, returns)
        )
        if isinstance(replay, training_core.EpisodeReplay):
            replay.append(completed_episode)
        else:
            for transition in completed_episode:
                replay.append(transition)

        episode_reward = sum(rewards)
        reward_sum += episode_reward
        return_sum += returns[0] if returns else 0.0
        won = bool(final_obs.won)
        if won:
            win_count += 1
        if use_warm:
            warm_episodes += 1
            if won:
                warm_wins += 1
        else:
            normal_episodes += 1
            if won:
                normal_wins += 1

        network.train()
        for _ in range(updates_per_episode):
            if isinstance(replay, training_core.EpisodeReplay):
                batch = replay.sample(replay_rng, min(batch_size, len(replay)))
            else:
                batch = replay_rng.sample(
                    list(replay),
                    min(batch_size, len(replay)),
                )
            loss_val = training_optimization.train_batch(
                network, optimizer, loss_fn, batch, torch_device, grad_clip
            )
            loss_sum += loss_val
            loss_count += 1
            train_steps += 1

        if log_every and (ep + 1) % log_every == 0:
            avg_loss = loss_sum / loss_count if loss_count else 0.0
            print(
                json.dumps(
                    {
                        "episode": ep + 1,
                        "epsilon": round(eps, 4),
                        "reward_avg": round(reward_sum / (ep + 1), 4),
                        "win_rate": round(win_count / (ep + 1), 4),
                        "loss_avg": round(avg_loss, 6),
                        "warm_episodes": warm_episodes,
                        "warm_wins": warm_wins,
                        "normal_episodes": normal_episodes,
                        "normal_wins": normal_wins,
                        "replay": len(replay),
                        "replay_sampling": replay_sampling,
                    }
                ),
                flush=True,
            )

    summary: Dict[str, object] = {
        "algorithm": "episodic_monte_carlo_q_regression",
        "episodes": episodes,
        "seed": seed,
        "gamma": gamma,
        "epsilon_end": epsilon_at(max(0, episodes - 1)) if episodes else epsilon_end,
        "reward_avg": reward_sum / episodes if episodes else 0.0,
        "win_rate": win_count / episodes if episodes else 0.0,
        "wins": win_count,
        "loss_avg": loss_sum / loss_count if loss_count else 0.0,
        "return_avg": return_sum / episodes if episodes else 0.0,
        "train_updates": train_steps,
        "warm_episodes": warm_episodes,
        "warm_wins": warm_wins,
        "normal_episodes": normal_episodes,
        "normal_wins": normal_wins,
        "warm_start_fraction": warm_start_fraction,
        "encoding_version": ENCODING_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": network.architecture,
        "param_count": sum(parameter.numel() for parameter in network.parameters()),
        "replay_sampling": replay_sampling,
        "game_config": cfg.to_dict(),
        "torch_threads": torch_threads,
    }

    if eval_episodes > 0:
        summary["heldout"] = evaluate_policy(
            network,
            config=cfg,
            games=eval_episodes,
            seed=eval_seed,
            device=torch_device,
        )

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            network,
            cfg,
            meta={
                "algorithm": "episodic_monte_carlo_q_regression",
                "summary": {
                    k: summary[k]
                    for k in (
                        "episodes",
                        "seed",
                        "gamma",
                        "win_rate",
                        "reward_avg",
                        "warm_start_fraction",
                        "replay_sampling",
                    )
                },
            },
        )
        summary["checkpoint"] = checkpoint_path

    return summary


def evaluate_policy(
    network: "QNetwork",
    config: Optional[GameConfig] = None,
    games: int = 50,
    seed: int = 12345,
    device: Optional["torch.device"] = None,
    compare_random: bool = True,
) -> Dict[str, float]:
    """Held-out greedy play on standard random deals. No learning updates."""
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise ValueError("games must be a positive integer")
    cfg = config if config is not None else GameConfig()
    if device is None:
        device = next(network.parameters()).device

    was_training = network.training
    network.eval()
    env = PappluEnv(cfg)

    greedy_rewards = 0.0
    greedy_wins = 0
    for i in range(games):
        obs = env.reset(seed=seed + i)
        total_r = 0.0
        while not obs.done:
            action = select_greedy_action(network, obs, device=device)
            obs = env.step(action)
            total_r += obs.last_reward
        greedy_rewards += total_r
        if obs.won:
            greedy_wins += 1

    result: Dict[str, float] = {
        "games": float(games),
        "greedy_reward_avg": greedy_rewards / games if games else 0.0,
        "greedy_win_rate": greedy_wins / games if games else 0.0,
    }

    if compare_random:
        rand_rewards = 0.0
        rand_wins = 0
        for i in range(games):
            obs = env.reset(seed=seed + i)
            total_r = 0.0
            step_rng = random.Random(seed + 10_000 + i)
            while not obs.done:
                action = select_random_legal(obs, step_rng)
                obs = env.step(action)
                total_r += obs.last_reward
            rand_rewards += total_r
            if obs.won:
                rand_wins += 1
        result["random_reward_avg"] = rand_rewards / games if games else 0.0
        result["random_win_rate"] = rand_wins / games if games else 0.0

    if was_training:
        network.train()
    return result
