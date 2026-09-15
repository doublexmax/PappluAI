"""Episodic Monte Carlo Q-value regression for solo Papplu.

Each episode records (state, action, reward). After the episode ends, discounted
returns G_t = r_t + gamma * G_{t+1} become regression targets for Q(s_t, a_t).
There is no next-state max, target network, or tree search. This is not DQN.

Optional --warm-start-fraction mixes evaluator-confirmed winning-hand-plus-extra
discard-phase starts with ordinary random deals. Warm-start metrics are labeled
separately and are not held-out full-game wins.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import random
import sys
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Sequence, Tuple

from src.environment import (
    ENCODING_VERSION,
    GameConfig,
    Observation,
    PappluEnv,
    encode_observation,
)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from src.model import QNetwork


Transition = Tuple[Tuple[float, ...], int, float]
EpisodeStep = Tuple[Tuple[float, ...], int, float]


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    """Full-episode Monte Carlo returns, last step first in the recurrence."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1], got %r" % (gamma,))
    returns: List[float] = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def run_episode(
    env: PappluEnv,
    network: "QNetwork",
    epsilon: float,
    rng: random.Random,
    warm_start: bool = False,
    device: Optional["torch.device"] = None,
) -> Tuple[List[EpisodeStep], Observation]:
    from src.model import select_action

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
) -> Dict[str, object]:
    """Train Q via episodic Monte Carlo regression. Returns summary metrics."""
    import torch
    import torch.nn as nn
    from src.model import CHECKPOINT_VERSION, QNetwork, load_checkpoint, save_checkpoint

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

    cfg = config if config is not None else GameConfig()
    torch_device = torch.device(device)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    network = QNetwork().to(torch_device)

    if load_path:
        network, cfg, _ = load_checkpoint(
            load_path,
            device=torch_device,
            expected_config=config,
            network=network,
        )
    if warm_start_fraction and cfg.cards_in_hand < 3:
        raise ValueError("warm starts require at least three cards")
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    env = PappluEnv(cfg)
    env.seed(seed)
    replay: Deque[Transition] = deque(maxlen=replay_capacity)

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
        use_warm = rng.random() < warm_start_fraction
        eps = epsilon_at(ep)
        steps, final_obs = run_episode(
            env,
            network,
            epsilon=eps,
            rng=rng,
            warm_start=use_warm,
            device=torch_device,
        )
        rewards = [r for _, _, r in steps]
        returns = discounted_returns(rewards, gamma)
        for (state, action, _reward), ret in zip(steps, returns):
            replay.append((state, action, ret))

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
            batch = rng.sample(list(replay), min(batch_size, len(replay)))
            loss_val = _train_batch(
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
                    )
                },
            },
        )
        summary["checkpoint"] = checkpoint_path

    return summary


def _train_batch(
    network: "QNetwork",
    optimizer: "torch.optim.Optimizer",
    loss_fn: "nn.Module",
    batch: Sequence[Transition],
    device: "torch.device",
    grad_clip: float,
) -> float:
    import torch

    states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    targets = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)

    network.train()
    q_all = network(states)
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = loss_fn(q_sa, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def evaluate_policy(
    network: "QNetwork",
    config: Optional[GameConfig] = None,
    games: int = 50,
    seed: int = 12345,
    device: Optional["torch.device"] = None,
    compare_random: bool = True,
) -> Dict[str, float]:
    """Held-out greedy play on standard random deals. No learning updates."""
    from src.model import select_greedy_action, select_random_legal

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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a Papplu Q-network with episodic Monte Carlo return regression "
            "(not DQN). Rewards come from src.evaluate.hand_reward."
        )
    )
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--replay-capacity", type=int, default=5000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=int, default=200)
    p.add_argument(
        "--warm-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of episodes that start as winning-hand-plus-extra discard "
            "puzzles (default 0 = random deals)."
        ),
    )
    p.add_argument("--updates-per-episode", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-threads", type=int, default=1, help="CPU threads for the small network (default 1).")
    p.add_argument("--num-decks", type=int, help="Decks (default 3, or loaded checkpoint).")
    p.add_argument("--cards-in-hand", type=int, help="Hand size (default 21, or loaded checkpoint).")
    p.add_argument("--required-sequences", type=int, help="Required pure sequences (default 5, or loaded checkpoint).")
    p.add_argument(
        "--max-turns",
        type=int,
        help="Turn budget (default 40, or loaded checkpoint).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to write the trained checkpoint after training.",
    )
    p.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Load weights and game settings to fine-tune. Replay and optimizer "
            "start fresh. Explicit game settings must match the checkpoint."
        ),
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Held-out greedy vs random games after training (0 skips).",
    )
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 1

    try:
        game_settings = {
            name: getattr(args, name)
            for name in ("num_decks", "cards_in_hand", "required_sequences", "max_turns")
            if getattr(args, name) is not None
        }
        if args.load and game_settings:
            from src.model import load_checkpoint

            _, saved_config, _ = load_checkpoint(args.load)
            config = GameConfig(**{**saved_config.to_dict(), **game_settings})
        else:
            config = GameConfig(**game_settings) if not args.load else None
        summary = train(
            episodes=args.episodes,
            seed=args.seed,
            gamma=args.gamma,
            lr=args.lr,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            warm_start_fraction=args.warm_start_fraction,
            updates_per_episode=args.updates_per_episode,
            grad_clip=args.grad_clip,
            device=args.device,
            torch_threads=args.torch_threads,
            config=config,
            checkpoint_path=args.checkpoint or None,
            load_path=args.load or None,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            log_every=args.log_every,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        print("error: PyTorch is required. Run python -m pip install -r requirements-training.txt", file=sys.stderr)
        return 2
    except (TypeError, ValueError, OSError, RuntimeError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
