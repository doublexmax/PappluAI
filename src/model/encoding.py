from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from src.evaluate import NUM_FACES
from src.game.environment import Observation

CONTEXT_FEATURES = 11
PLAYER_FEATURES = 6
EVENT_FEATURES = 8


@dataclass(frozen=True)
class ObservationBatch:
    cards: torch.Tensor
    initial_hand: torch.Tensor
    top: torch.Tensor
    context: torch.Tensor
    players: torch.Tensor
    player_mask: torch.Tensor
    next_player: torch.Tensor
    previous_player: torch.Tensor
    event_kind: torch.Tensor
    event_face: torch.Tensor
    offered_top: torch.Tensor
    event_features: torch.Tensor
    event_actor: torch.Tensor
    history_mask: torch.Tensor
    history_lengths: torch.Tensor


def collate_observations(
    observations: Sequence[Observation],
    device: torch.device,
) -> ObservationBatch:
    if not observations or any(not obs.history or not obs.players for obs in observations):
        raise ValueError("model input needs observations with complete history and player views")
    batch = len(observations)
    players = max(len(obs.players) for obs in observations)
    events = max(len(obs.history) for obs in observations)

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def indices(*shape: int, fill: int = 0) -> torch.Tensor:
        return torch.full(shape, fill, dtype=torch.long, device=device)

    cards = zeros(batch, NUM_FACES, 5)
    initial_hand = zeros(batch, NUM_FACES)
    top = indices(batch, fill=NUM_FACES)
    context = zeros(batch, CONTEXT_FEATURES)
    player_values = zeros(batch, players, PLAYER_FEATURES)
    player_mask = torch.zeros((batch, players), dtype=torch.bool, device=device)
    next_player = indices(batch, players)
    previous_player = indices(batch, players)
    event_kind = indices(batch, events)
    event_face = indices(batch, events, fill=NUM_FACES)
    offered_top = indices(batch, events, fill=NUM_FACES)
    event_features = zeros(batch, events, EVENT_FEATURES)
    event_actor = indices(batch, events)
    history_mask = torch.zeros((batch, events), dtype=torch.bool, device=device)
    lengths = []

    for row, obs in enumerate(observations):
        cfg = obs.config
        count = len(obs.players)
        history_length = len(obs.history)
        lengths.append(history_length)
        card_values = [[n / cfg.num_decks, 0.0, 0.0, 0.0, 0.0] for n in obs.hand]
        for face in obs.discard_pile:
            card_values[face][1] += 1.0 / cfg.num_decks
        if obs.discard_top is not None:
            top[row] = obs.discard_top
            card_values[obs.discard_top][2] = 1.0
        card_values[obs.joker][3] = 1.0
        for suit in range(4):
            card_values[suit * 13 + obs.joker % 13][4] = 1.0
        cards[row] = torch.tensor(card_values, device=device)
        initial_hand[row] = torch.tensor(obs.history[0].initial_hand, device=device) / cfg.num_decks
        phase = [float(obs.phase.value == value) for value in range(3)]
        context[row] = torch.tensor(phase + [
            obs.turns_remaining / cfg.max_turns,
            obs.stock_remaining / (cfg.num_decks * NUM_FACES),
            cfg.num_decks / 6.0, cfg.cards_in_hand / 30.0,
            cfg.required_sequences / 10.0, float(cfg.recycle_discard),
            count / 6.0, cfg.max_turns / 60.0,
        ], device=device)
        for relative, player in enumerate(obs.players):
            angle = 2.0 * math.pi * relative / count
            player_values[row, relative] = torch.tensor([
                float(relative == 0), player.hand_size / cfg.cards_in_hand,
                player.turns_remaining / cfg.max_turns, float(player.active),
                math.sin(angle), math.cos(angle),
            ], device=device)
            next_player[row, relative] = player.next_player
            previous_player[row, relative] = player.previous_player
        player_mask[row, :count] = True
        event_kind[row, :history_length] = torch.tensor(
            [int(event.kind) for event in obs.history], device=device,
        )
        event_face[row, :history_length] = torch.tensor([
            NUM_FACES if event.face is None else event.face for event in obs.history
        ], device=device)
        offered_top[row, :history_length] = torch.tensor([
            NUM_FACES if event.offered_top is None else event.offered_top for event in obs.history
        ], device=device)
        event_actor[row, :history_length] = torch.tensor(
            [event.actor for event in obs.history], device=device,
        )
        event_features[row, :history_length] = torch.tensor([
            [
                float(event.actor == 0),
                math.sin(2.0 * math.pi * event.actor / count),
                math.cos(2.0 * math.pi * event.actor / count),
                event.hand_size / cfg.cards_in_hand,
                event.turns_remaining / cfg.max_turns,
                event.stock_remaining / (cfg.num_decks * NUM_FACES),
                math.sin(2.0 * math.pi * event.next_player / count),
                math.cos(2.0 * math.pi * event.next_player / count),
            ] for event in obs.history
        ], device=device)
        history_mask[row, :history_length] = True
    return ObservationBatch(
        cards, initial_hand, top, context, player_values, player_mask,
        next_player, previous_player, event_kind, event_face, offered_top,
        event_features, event_actor, history_mask,
        torch.tensor(lengths, dtype=torch.long),
    )
