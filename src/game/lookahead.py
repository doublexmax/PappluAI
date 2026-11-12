from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

from src.evaluate import is_valid_hand, pure_sequence_upper_bound
from src.game.belief import BeliefState
from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, GameConfig, Observation, Phase,
    discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationEvent, _player_views


LEGAL_RUNS = tuple((start, start + 1, start + 2) for start in range(11)) + ((11, 12, 0),)


def can_declare_ub(hand: Sequence[int], joker: int, required_sequences: int) -> bool:
    return pure_sequence_upper_bound(hand, joker) >= required_sequences


def winning_discard(
    hand: Sequence[int], joker: int, required_sequences: int = 5, cards_in_hand: int = 21,
) -> Optional[int]:
    if sum(hand) != cards_in_hand + 1 or not can_declare_ub(hand, joker, required_sequences):
        return None
    remaining = list(hand)
    for face, copies in enumerate(hand):
        if copies <= 0:
            continue
        remaining[face] -= 1
        wins = is_valid_hand(
            remaining, joker, required_sequences, cards_in_hand,
        )
        remaining[face] += 1
        if wins:
            return face
    return None


def outs_1ply(hand: Sequence[int], belief: BeliefState, config: GameConfig) -> float:
    if not can_declare_ub(hand, belief.joker, config.required_sequences - 1):
        return 0.0
    drawn, probability = list(hand), 0.0
    for face, chance in belief.stock_probabilities():
        drawn[face] += 1
        if winning_discard(
            drawn, belief.joker, config.required_sequences, config.cards_in_hand,
        ) is not None:
            probability += chance
        drawn[face] -= 1
    return probability


def after_draw(obs: Observation, face: int, kind: EventKind) -> Observation:
    hand = list(obs.hand)
    hand[face] += 1
    take = kind is EventKind.TAKE_DISCARD
    stock = obs.stock_remaining - int(not take)
    player = replace(obs.players[0], hand_size=obs.players[0].hand_size + 1)
    event = InformationEvent(
        kind, 0, player.hand_size, obs.turns_remaining, stock,
        player.next_player, face=face, offered_top=obs.discard_top,
    )
    return replace(
        obs, hand=tuple(hand), discard_pile=obs.discard_pile[:-1] if take else obs.discard_pile,
        phase=Phase.DISCARD, stock_remaining=stock, players=(player, *obs.players[1:]),
        history=(*obs.history, event), last_reward=0.0,
    )


def after_discard(obs: Observation, face: int) -> Observation:
    hand = list(obs.hand)
    hand[face] -= 1
    turns = obs.turns_remaining - 1
    players = list(obs.players)
    players[0] = replace(
        players[0], hand_size=sum(hand), turns_remaining=turns, active=turns > 0,
    )
    sizes = [player.hand_size for player in players]
    budgets = [player.turns_remaining for player in players]
    views = _player_views(sizes, budgets, 0)
    event = InformationEvent(
        EventKind.DISCARD, 0, sum(hand), turns, obs.stock_remaining,
        views[0].next_player, face=face,
    )
    return replace(
        obs, hand=tuple(hand), discard_pile=(*obs.discard_pile, face), phase=Phase.DRAW,
        turns_remaining=turns, players=views, history=(*obs.history, event), last_reward=0.0,
    )


@dataclass(frozen=True)
class Declares:
    probability: float


@dataclass(frozen=True)
class Continues:
    probability: float
    observation: Observation


@dataclass(frozen=True)
class Terminates:
    probability: float
    value: float


Branch = Union[Declares, Continues, Terminates]


def expand(obs: Observation) -> dict[int, tuple[Branch, ...]]:
    if obs.done or not obs.players[0].active:
        return {}
    last = obs.history[-1]
    if obs.phase is Phase.DISCARD and last.actor != 0:
        return {}
    if obs.phase is Phase.DRAW and last.kind in (EventKind.DISCARD, EventKind.REFILL) and last.next_player != 0:
        return {}
    terminal_discard = obs.warm_start or (
        obs.turns_remaining <= 1 and all(player.turns_remaining <= 0 for player in obs.players[1:])
    )
    if obs.phase is Phase.DISCARD and terminal_discard and len(obs.players) > 1:
        return {}
    result = {}
    for action, allowed in enumerate(legal_action_mask(obs)):
        if not allowed:
            continue
        if action == ACTION_DRAW_STOCK:
            result[action] = tuple(
                Continues(probability, after_draw(obs, face, EventKind.DRAW_STOCK))
                for face, probability in BeliefState.from_observation(obs).stock_probabilities()
            )
        elif action == ACTION_TAKE_DISCARD:
            result[action] = (Continues(1.0, after_draw(obs, obs.discard_top, EventKind.TAKE_DISCARD)),)
        else:
            face = action - discard_action(0)
            remaining = list(obs.hand)
            remaining[face] -= 1
            result[action] = (
                Declares(1.0)
                if is_valid_hand(remaining, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand)
                else Terminates(1.0, -1.0) if terminal_discard
                else Continues(1.0, after_discard(obs, face)),
            )
    return result
