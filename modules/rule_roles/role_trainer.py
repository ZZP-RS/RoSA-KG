"""
Dynamic role trainer that maintains weights for rule structural roles and
produces selection decisions for virtual relation injection.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .role_schema import RoleDescriptor, RoleType, initial_role_weight
from ..rule_mining import RuleDefinition


@dataclass
class RoleUpdateSignal:
    rule_key: Tuple[Tuple[int, ...], int]
    position: int
    reward: float
    diversity_reward: float = 0.0
    usage: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoleState:
    descriptor: RoleDescriptor
    weight: float
    momentum: float
    relation_id: int
    usage_count: int = 0
    last_reward: float = 0.0
    role_entity_id: int = -1  # optional: when role is injected as an entity node
    suspended: bool = False
    negative_streak: int = 0
    div_score_ema: float = 0.0
    cooldown: int = 0

    def apply_update(self, signal: RoleUpdateSignal, lr: float, momentum_factor: float) -> None:
        delta = lr * signal.reward
        self.last_reward = signal.reward
        updated = momentum_factor * self.weight + (1.0 - momentum_factor) * (self.weight + delta)
        self.weight = max(1e-6, updated)
        self.usage_count += signal.usage


class RuleRoleManager:
    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.6,
        max_active_roles: int = 128,
        suspension_threshold: int = 3,
        diversity_coeff: float = 0.6,
        diversity_smoothing: float = 0.3,
        selection_temperature: float = 1.0,
        cooldown_steps: int = 2,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_active_roles = max_active_roles
        self.suspension_threshold = max(1, suspension_threshold)
        self.diversity_coeff = diversity_coeff
        self.diversity_smoothing = max(1e-3, min(0.9, diversity_smoothing))
        self.selection_temperature = max(1e-3, selection_temperature)
        self.cooldown_steps = max(0, cooldown_steps)
        self._roles: Dict[Tuple[Tuple[Tuple[int, ...], int], int], RoleState] = {}
        self._rule_lookup: Dict[Tuple[Tuple[int, ...], int], RuleDefinition] = {}
        self._next_relation_id: int = 0
        self._rng = random.Random(19)

    def bootstrap(self, rules: Iterable[RuleDefinition], base_relation_id: int) -> None:
        self._rule_lookup = {(rule.body_relations, rule.head_relation): rule for rule in rules}
        self._next_relation_id = base_relation_id
        self._roles.clear()
        for rule in rules:
            key = (rule.body_relations, rule.head_relation)
            body_length = len(rule.body_relations)
            for position, size in enumerate(rule.position_unique_counts):
                descriptor = RoleDescriptor(
                    rule_key=key,
                    position=position,
                    body_length=body_length,
                    size=size,
                    base_weight=initial_role_weight(rule.metrics, max(1, size)),
                    metrics=dict(rule.metrics),
                )
                relation_id = self._allocate_relation_id()
                state = RoleState(
                    descriptor=descriptor,
                    weight=max(descriptor.base_weight, 1e-4),
                    momentum=self.momentum,
                    relation_id=relation_id,
                )
                self._roles[(key, position)] = state

    def _allocate_relation_id(self) -> int:
        relation_id = self._next_relation_id
        self._next_relation_id += 1
        return relation_id

    def record_feedback(self, signals: Iterable[RoleUpdateSignal]) -> None:
        for signal in signals:
            if abs(signal.reward) < 1e-8:
                continue
            key = (signal.rule_key, signal.position)
            state = self._roles.get(key)
            if state is None:
                continue
            if signal.reward > 0:
                state.suspended = False
                state.negative_streak = 0
            else:
                state.negative_streak += signal.usage
                if state.negative_streak >= self.suspension_threshold:
                    state.suspended = True
                    state.weight = max(1e-6, state.weight * 0.5)
                    continue
            if signal.diversity_reward < 0.0:
                state.cooldown = max(state.cooldown, self.cooldown_steps)
            state.apply_update(signal, self.learning_rate, self.momentum)
            smooth = self.diversity_smoothing
            state.div_score_ema = (
                (1.0 - smooth) * state.div_score_ema + smooth * signal.diversity_reward
            )

    def select_active_roles(self, limit: Optional[int] = None) -> List[RoleState]:
        if limit is None:
            limit = self.max_active_roles
        if (limit is None or limit <= 0) and self._roles:
            limit = len(self._roles)
        if not self._roles or (limit is not None and limit <= 0):
            return []
        for state in self._roles.values():
            if state.cooldown > 0:
                state.cooldown -= 1
        candidates = [
            state for state in self._roles.values() if not state.suspended and state.cooldown <= 0
        ]
        if not candidates:
            candidates = [state for state in self._roles.values() if not state.suspended]
        if not candidates:
            return []
        scores = [
            (state.weight + self.diversity_coeff * state.div_score_ema)
            / self.selection_temperature
            for state in candidates
        ]
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        chosen: List[RoleState] = []
        pool = list(zip(candidates, weights))
        while pool and len(chosen) < limit:
            total = sum(weight for _, weight in pool)
            if total <= 0.0:
                chosen.extend(state for state, _ in pool[: limit - len(chosen)])
                break
            r = self._rng.random() * total
            cumulative = 0.0
            picked_index = 0
            for idx, (state, weight) in enumerate(pool):
                cumulative += weight
                if r <= cumulative:
                    picked_index = idx
                    break
            state, _ = pool.pop(picked_index)
            chosen.append(state)
        return chosen

    def count_available_roles(self, include_cooldown: bool = True) -> int:
        if include_cooldown:
            return sum(1 for state in self._roles.values() if not state.suspended)
        return sum(
            1 for state in self._roles.values() if not state.suspended and state.cooldown <= 0
        )

    @property
    def rule_lookup(self) -> Dict[Tuple[Tuple[int, ...], int], RuleDefinition]:
        return self._rule_lookup

    def get_rule(self, rule_key: Tuple[Tuple[int, ...], int]) -> Optional[RuleDefinition]:
        return self._rule_lookup.get(rule_key)

    def serialize(self) -> Dict:
        data = {
            "next_relation_id": self._next_relation_id,
            "roles": [],
        }
        for (rule_key, position), state in self._roles.items():
            payload = {
                "body_relations": list(rule_key[0]),
                "head_relation": rule_key[1],
                "position": position,
                "relation_id": state.relation_id,
                "weight": state.weight,
                "momentum": state.momentum,
                "usage_count": state.usage_count,
                "last_reward": state.last_reward,
                "div_score_ema": state.div_score_ema,
                "cooldown": state.cooldown,
            }
            data["roles"].append(payload)
        return data

    def load_state(self, payload: Dict) -> None:
        self._next_relation_id = payload.get("next_relation_id", self._next_relation_id)
        for role_payload in payload.get("roles", []):
            rule_key = (tuple(role_payload["body_relations"]), role_payload.get("head_relation"))
            position = role_payload.get("position", 0)
            state = self._roles.get((rule_key, position))
            if state is None:
                continue
            state.relation_id = role_payload.get("relation_id", state.relation_id)
            state.weight = role_payload.get("weight", state.weight)
            state.momentum = role_payload.get("momentum", state.momentum)
            state.usage_count = role_payload.get("usage_count", state.usage_count)
            state.last_reward = role_payload.get("last_reward", state.last_reward)
            state.div_score_ema = role_payload.get("div_score_ema", state.div_score_ema)
            state.cooldown = role_payload.get("cooldown", state.cooldown)
