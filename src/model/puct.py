from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from src.game.environment import (
    ACTION_DRAW_STOCK, ACTION_TAKE_DISCARD, NUM_FACES,
    GameConfig, Observation, Phase, discard_action, legal_action_mask,
)
from src.game.information import EventKind, InformationEvent, _player_views
from src.model.encoding import collate_observations
from src.model.network import QNetwork
from src.model.policy_value import PolicyValueNet
from src.model.search import (
    BeliefState, _after_discard, _after_draw, _score_22_discards, outs_1ply,
    winning_discard,
)
from src.training.core import terminal_reward


@dataclass(frozen=True)
class PUCTConfig:
    simulations: int = 32
    c_puct: float = 1.25
    max_stock_outcomes: int = 12
    opponent_outcomes: int = 4
    afterstate_weight: float = 0.20
    meld_weight: float = 0.06
    outs_weight: float = 0.15
    feed_weight: float = 0.03
    dirichlet_alpha: float = 0.30
    dirichlet_eps: float = 0.25
    temperature: float = 0.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("simulations", 0), ("max_stock_outcomes", 1), ("opponent_outcomes", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError("%s must be an integer >= %d" % (name, minimum))
        for name in (
            "c_puct", "afterstate_weight", "meld_weight", "outs_weight",
            "feed_weight", "dirichlet_alpha", "dirichlet_eps", "temperature",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if self.dirichlet_alpha == 0:
            raise ValueError("dirichlet_alpha must be positive")
        if self.afterstate_weight > 1 or self.dirichlet_eps > 1:
            raise ValueError("afterstate_weight and dirichlet_eps must be in [0, 1]")


@dataclass(frozen=True)
class PUCTResult:
    action: int
    policy_target: Tuple[Tuple[int, float], ...]
    root_value: float
    visits: Tuple[Tuple[int, int], ...]
    rows: int
    reason: str


@dataclass(frozen=True)
class _Evaluation:
    q: Tuple[float, ...]
    prior: Tuple[float, ...]
    value: float


@dataclass
class _Outcome:
    probability: float
    observation: Observation
    estimate: float
    terminal: bool
    visits: int = 0
    node: Optional[_Node] = None


@dataclass
class _Edge:
    action: int
    prior: float
    initial_value: float
    enabled: bool = True
    visits: int = 0
    value_sum: float = 0.0
    outcomes: Optional[Tuple[_Outcome, ...]] = None

    @property
    def q(self) -> float:
        return (self.initial_value + self.value_sum) / (1 + self.visits)


@dataclass
class _Node:
    observation: Observation
    edges: Tuple[_Edge, ...]
    position_value: float


def _winning_action(obs: Observation) -> Optional[int]:
    if obs.phase is Phase.DISCARD:
        face = winning_discard(
            obs.hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
        )
        return None if face is None else discard_action(face)
    if obs.phase is Phase.DRAW and obs.discard_top is not None:
        hand = list(obs.hand)
        hand[obs.discard_top] += 1
        if winning_discard(
            hand, obs.joker, obs.config.required_sequences, obs.config.cards_in_hand,
        ) is not None:
            return ACTION_TAKE_DISCARD
    return None


def _normalize(weighted: Sequence[Tuple[int, float]]) -> Tuple[Tuple[int, float], ...]:
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        raise ValueError("chance outcomes must have positive mass")
    return tuple((item, weight / total) for item, weight in weighted)


def _refill(obs: Observation, actor: int) -> Observation:
    if obs.stock_remaining or not obs.config.recycle_discard or len(obs.discard_pile) < 2:
        return obs
    stock = len(obs.discard_pile) - 1
    player = obs.players[actor]
    event = InformationEvent(
        EventKind.REFILL, actor, player.hand_size, player.turns_remaining,
        stock, player.next_player,
    )
    return replace(
        obs, discard_pile=obs.discard_pile[-1:], stock_remaining=stock,
        history=(*obs.history, event),
    )


def _opponent_transition(obs: Observation, took: bool, face: int) -> Observation:
    opponent = obs.players[1]
    stock = obs.stock_remaining - int(not took)
    pile = obs.discard_pile[:-1] if took else obs.discard_pile
    draw = InformationEvent(
        EventKind.TAKE_DISCARD if took else EventKind.DRAW_STOCK,
        1, opponent.hand_size + 1, opponent.turns_remaining, stock,
        opponent.next_player, face=obs.discard_top if took else None,
        offered_top=obs.discard_top,
    )
    turns = [player.turns_remaining for player in obs.players]
    turns[1] -= 1
    players = _player_views([player.hand_size for player in obs.players], turns, 0)
    discard = InformationEvent(
        EventKind.DISCARD, 1, opponent.hand_size, turns[1], stock,
        players[1].next_player, face=face,
    )
    return _refill(replace(
        obs, discard_pile=(*pile, face), stock_remaining=stock, players=players,
        history=(*obs.history, draw, discard),
    ), 1)


class _Search:
    def __init__(
        self, network: PolicyValueNet, config: PUCTConfig,
        device: torch.device,
    ) -> None:
        self.network = network
        self.config = config
        self.device = device
        self.rows = 0
        self.cache: Dict[Observation, _Evaluation] = {}
        self.draw_value_cache: Dict[Observation, float] = {}
        self.temperatures = network.log_temperature.detach().exp().cpu().tolist()

    def evaluate(self, observations: Sequence[Observation]) -> Tuple[_Evaluation, ...]:
        missing = tuple(dict.fromkeys(obs for obs in observations if obs not in self.cache))
        for start in range(0, len(missing), 256):
            batch = missing[start:start + 256]
            out = self.network(
                collate_observations(batch, self.device),
                torch.tensor(
                    [legal_action_mask(obs) for obs in batch],
                    dtype=torch.bool, device=self.device,
                ),
            )
            q = out.q.cpu().tolist()
            prior = out.log_policy.exp().cpu().tolist()
            values = out.position_value.cpu().tolist()
            for row, obs in enumerate(batch):
                if not all(math.isfinite(value) for value in q[row]):
                    raise ValueError("network returned nonfinite Q values")
                self.cache[obs] = _Evaluation(tuple(q[row]), tuple(prior[row]), values[row])
            self.rows += len(batch)
        return tuple(self.cache[obs] for obs in observations)

    def draw_values(self, observations: Sequence[Observation]) -> Tuple[float, ...]:
        missing = tuple(dict.fromkeys(
            obs for obs in observations if obs not in self.draw_value_cache
        ))
        for start in range(0, len(missing), 256):
            batch = missing[start:start + 256]
            values = self.network.qnet(collate_observations(batch, self.device))[:, 0].cpu().tolist()
            for obs, value in zip(batch, values):
                if not math.isfinite(value):
                    raise ValueError("network returned a nonfinite position value")
                self.draw_value_cache[obs] = value
            self.rows += len(batch)
        return tuple(self.draw_value_cache[obs] for obs in observations)

    def stock_draws(self, obs: Observation) -> Tuple[Tuple[int, float], ...]:
        probabilities = BeliefState.from_observation(obs).stock_probabilities()
        return _normalize(sorted(
            probabilities, key=lambda item: (-item[1], item[0]),
        )[:self.config.max_stock_outcomes])

    def discard_edges(
        self, observations: Sequence[Observation],
        excluded: Sequence[Optional[int]],
        reference_pile: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Tuple[_Edge, ...], ...]:
        evaluations = self.evaluate(observations)
        plans = []
        afterstates = []
        for obs, forbidden in zip(observations, excluded):
            actions = tuple(action for action, ok in enumerate(legal_action_mask(obs)) if ok)
            enabled = tuple(action for action in actions if action != forbidden)
            if not enabled:
                enabled = actions
            win = _winning_action(obs)
            if win is not None:
                enabled = (win,)
            offset = len(afterstates)
            for action in enabled:
                after = _after_discard(obs, action - 2)
                afterstates.append(replace(
                    after,
                    discard_pile=obs.discard_pile if reference_pile is None else reference_pile,
                ))
            plans.append((actions, enabled, win, offset))
        after_values = self.draw_values(afterstates)
        result = []
        for obs, evaluation, plan in zip(observations, evaluations, plans):
            actions, enabled, win, offset = plan
            belief = BeliefState.from_observation(obs)
            faces = [action - 2 for action in enabled]
            meld = _score_22_discards(
                obs.hand, faces, evaluation.q[2:], obs.joker,
                obs.config.required_sequences, obs.config.cards_in_hand,
            ) if self.config.meld_weight else {}
            scores = {}
            for index, action in enumerate(enabled):
                face = action - 2
                after = afterstates[offset + index]
                scores[action] = (
                    (1 - self.config.afterstate_weight) * evaluation.q[action]
                    + self.config.afterstate_weight * after_values[offset + index]
                    - self.config.feed_weight * belief.feed_risk(face, 1)
                    + self.config.meld_weight * meld.get(face, 0.0)
                    + (self.config.outs_weight * outs_1ply(after.hand, belief, obs.config)
                       if self.config.outs_weight else 0.0)
                )
            if win is not None:
                scores[win] = 1.0
            floor = min(scores.values()) - 1.0
            priors = {
                action: max(evaluation.prior[action], 1e-30) if action in enabled else 1e-30
                for action in actions
            }
            total = sum(priors.values())
            result.append(tuple(
                _Edge(action, priors[action] / total, scores.get(action, floor), action in enabled)
                for action in actions
            ))
        return tuple(result)

    def expand(self, obs: Observation) -> _Node:
        evaluation = self.evaluate((obs,))[0]
        if obs.phase is Phase.DISCARD:
            last = obs.history[-1]
            excluded = (
                discard_action(last.face)
                if last.actor == 0 and last.kind is EventKind.TAKE_DISCARD else None
            )
            return _Node(obs, self.discard_edges((obs,), (excluded,))[0], evaluation.value)
        actions = tuple(action for action, ok in enumerate(legal_action_mask(obs)) if ok)
        children = []
        excluded = []
        branches = []
        for action in actions:
            draws = (
                ((obs.discard_top, 1.0),) if action == ACTION_TAKE_DISCARD
                else self.stock_draws(obs)
            )
            indices = []
            for face, probability in draws:
                indices.append((len(children), probability))
                kind = EventKind.DRAW_STOCK if obs.stock_remaining else EventKind.TAKE_DISCARD
                children.append(_after_draw(obs, face, kind))
                excluded.append(
                    discard_action(face) if action == ACTION_TAKE_DISCARD else None,
                )
            branches.append((action, indices))
        scores = self.discard_edges(children, excluded, obs.discard_pile)
        edges = tuple(
            _Edge(action, max(evaluation.prior[action], 1e-30), sum(
                probability * max(edge.initial_value for edge in scores[index] if edge.enabled)
                for index, probability in indices
            ))
            for action, indices in branches
        )
        return _Node(obs, edges, evaluation.value)

    def opponent_turn(self, post: Observation) -> Tuple[Tuple[Observation, float], ...]:
        post = _refill(post, 0)
        if len(post.players) < 2 or not post.players[1].active:
            return ((post, 1.0),)
        if self.config.opponent_outcomes == 0:
            if not post.stock_remaining:
                return ((post, 1.0),)
            return tuple(
                (_after_draw(post, face, EventKind.DRAW_STOCK), probability)
                for face, probability in self.stock_draws(post)
            )
        belief = BeliefState.from_observation(post)
        take = (
            belief.feed_risk(post.discard_top, 1) if post.stock_remaining else 1.0
        )
        weights = [
            (belief.unseen_counts[face] + belief.known_held[1][face])
            * (1.0 - belief.opp_wanted[1][face])
            for face in range(NUM_FACES)
        ]
        if not any(weights):
            weights = [
                belief.unseen_counts[face] + belief.known_held[1][face]
                for face in range(NUM_FACES)
            ]
        candidates = [
            (took, face, probability * weight)
            for took, probability in ((False, 1 - take), (True, take))
            for face, weight in enumerate(weights)
            if probability * weight > 0 and not (took and face == post.discard_top)
        ]
        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        candidates = candidates[:self.config.opponent_outcomes]
        total = sum(weight for _, _, weight in candidates)
        if total == 0:
            raise ValueError("belief has no possible opponent discard")
        return tuple(
            (_opponent_transition(post, took, face), weight / total)
            for took, face, weight in candidates
        )

    def terminal_value(self, obs: Observation) -> Optional[float]:
        if obs.turns_remaining <= 0 or obs.done:
            return terminal_reward(
                self.network.shaping, won=obs.won, turns_remaining=obs.turns_remaining,
                max_turns=obs.config.max_turns, hand=obs.hand, joker=obs.joker,
                required_sequences=obs.config.required_sequences,
            )
        if _winning_action(obs) is not None:
            return 1.0
        return None

    def outcomes(self, node: _Node, edge: _Edge) -> Tuple[_Outcome, ...]:
        obs = node.observation
        if obs.phase is Phase.DRAW:
            children = (
                ((_after_draw(obs, obs.discard_top, EventKind.TAKE_DISCARD), 1.0),)
                if edge.action == ACTION_TAKE_DISCARD else tuple(
                    (_after_draw(obs, face, EventKind.DRAW_STOCK), probability)
                    for face, probability in self.stock_draws(obs)
                )
            )
        else:
            post = _after_discard(obs, edge.action - 2)
            post = replace(post, players=_player_views(
                [player.hand_size for player in post.players],
                [player.turns_remaining for player in post.players], 0,
            ))
            children = (
                ((post, 1.0),) if post.turns_remaining <= 0
                else self.opponent_turn(post)
            )
        terminal = [self.terminal_value(child) for child, _ in children]
        self.evaluate([child for (child, _), value in zip(children, terminal) if value is None])
        return tuple(
            _Outcome(
                probability, child,
                self.cache[child].value if value is None else value,
                value is not None,
            )
            for (child, probability), value in zip(children, terminal)
        )

    def improved_policy(self, node: _Node) -> Tuple[float, ...]:
        tau = max(self.temperatures[int(node.observation.phase is Phase.DISCARD)], 1e-7)
        baseline = sum(edge.prior * edge.q for edge in node.edges)
        logits = [math.log(edge.prior) + (edge.q - baseline) / tau for edge in node.edges]
        peak = max(logits)
        weights = [math.exp(max(-60.0, logit - peak)) for logit in logits]
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def value(self, node: _Node) -> float:
        return sum(
            probability * edge.q
            for probability, edge in zip(self.improved_policy(node), node.edges)
        )

    def simulate(self, node: _Node) -> float:
        visits = sum(edge.visits for edge in node.edges)
        edge = max(
            (edge for edge in node.edges if edge.enabled),
            key=lambda edge: (
                edge.q + self.config.c_puct * edge.prior * math.sqrt(1 + visits)
                / (1 + edge.visits), -edge.action,
            ),
        )
        if edge.outcomes is None:
            edge.outcomes = self.outcomes(node, edge)
        count = sum(outcome.visits for outcome in edge.outcomes)
        outcome = max(
            edge.outcomes,
            key=lambda outcome: outcome.probability * (1 + count) - outcome.visits,
        )
        if not outcome.terminal:
            if outcome.node is None:
                outcome.node = self.expand(outcome.observation)
                outcome.estimate = outcome.node.position_value
            else:
                outcome.estimate = self.simulate(outcome.node)
        outcome.visits += 1
        value = self.network.gamma * sum(
            child.probability * child.estimate for child in edge.outcomes
        )
        edge.visits += 1
        edge.value_sum += value
        return self.value(node)


@torch.inference_mode()
def puct_search(
    network: Union[PolicyValueNet, QNetwork],
    obs: Observation,
    config: PUCTConfig = PUCTConfig(),
    rng: Optional[random.Random] = None,
    device: Optional[torch.device] = None,
) -> PUCTResult:
    legal = tuple(action for action, ok in enumerate(legal_action_mask(obs)) if ok)
    if not legal:
        raise ValueError("no legal actions")
    win = None if len(legal) == 1 else _winning_action(obs)
    if len(legal) == 1 or win is not None:
        action = legal[0] if win is None else win
        return PUCTResult(
            action, tuple((candidate, float(candidate == action)) for candidate in legal),
            0.0 if win is None else 1.0,
            tuple((candidate, 0) for candidate in legal), 0,
            "forced" if win is None else "exact_win",
        )
    model = network if isinstance(network, PolicyValueNet) else PolicyValueNet(network, obs.config)
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    source = rng if rng is not None else random.Random(0)
    tree = _Search(model, config, device)
    root = tree.expand(obs)
    enabled = [edge for edge in root.edges if edge.enabled]
    if config.temperature > 0 and config.dirichlet_eps > 0:
        noise = [source.gammavariate(config.dirichlet_alpha, 1.0) for _ in enabled]
        total = sum(noise)
        if total > 0:
            for edge, weight in zip(enabled, noise):
                edge.prior = (
                    (1 - config.dirichlet_eps) * edge.prior
                    + config.dirichlet_eps * weight / total
                )
    for _ in range(config.simulations):
        tree.simulate(root)
    probabilities = tree.improved_policy(root)
    if config.temperature == 0:
        chosen = max(enabled, key=lambda edge: (edge.visits, edge.q, -edge.action))
    else:
        logits = [
            (math.log(probability) + math.log1p(edge.visits)) / config.temperature
            for edge, probability in zip(root.edges, probabilities) if edge.enabled
        ]
        peak = max(logits)
        chosen = source.choices(
            enabled, weights=[math.exp(max(-60.0, logit - peak)) for logit in logits], k=1,
        )[0]
    return PUCTResult(
        chosen.action,
        tuple((edge.action, probability) for edge, probability in zip(root.edges, probabilities)),
        tree.value(root), tuple((edge.action, edge.visits) for edge in root.edges),
        tree.rows, "puct_search",
    )


class PUCTPolicy:
    def __init__(
        self, network: Union[PolicyValueNet, QNetwork],
        game_config: Optional[GameConfig] = None,
        config: PUCTConfig = PUCTConfig(), *, device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(network, QNetwork):
            network = PolicyValueNet(network, game_config or GameConfig())
        self.network = network
        self.game_config = network.game_config if game_config is None else game_config
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, path: str, config: PUCTConfig = PUCTConfig(), *,
        device: Optional[torch.device] = None,
    ) -> "PUCTPolicy":
        return cls(PolicyValueNet.from_checkpoint(path, device=device), config=config, device=device)

    def act(self, obs: Observation, rng: random.Random) -> int:
        return puct_search(self.network, obs, self.config, rng, self.device).action
