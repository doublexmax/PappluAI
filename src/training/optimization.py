from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.model.network import QNetwork
from src.training.core import Transition


def train_batch(
    network: QNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: Sequence[Transition],
    device: torch.device,
    grad_clip: float,
) -> float:
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
