"""
Post-processing helpers for mined rules.

These utilities mirror AMIE+'s filtering stages by enforcing quality thresholds,
deduplicating rules and ranking them using configurable scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .miner import RuleDefinition


@dataclass
class RuleFilterConfig:
    min_lift: float = 1.0
    min_conviction: float = 1.0
    min_role_entropy: float = 0.0
    max_rules: int = 2000
    score_key: str = "score"
    deduplicate: bool = True


class RulePostProcessor:
    def __init__(self, config: RuleFilterConfig):
        self.config = config

    def filter_and_rank(self, rules: Sequence[RuleDefinition]) -> List[RuleDefinition]:
        filtered: Dict[Tuple[Tuple[int, ...], int], RuleDefinition] = {}
        for rule in rules:
            if not self._passes_threshold(rule):
                continue
            key = (rule.body_relations, rule.head_relation)
            if self.config.deduplicate:
                existing = filtered.get(key)
                if existing is None or self._score(rule) > self._score(existing):
                    filtered[key] = rule
            else:
                filtered[key] = rule

        ranked = sorted(filtered.values(), key=self._score, reverse=True)
        if self.config.max_rules:
            ranked = ranked[: self.config.max_rules]
        return ranked

    def _passes_threshold(self, rule: RuleDefinition) -> bool:
        lift = rule.metrics.get("lift", 0.0)
        conviction = rule.metrics.get("conviction", 0.0)
        role_entropy = rule.metrics.get("role_entropy", 0.0)
        if lift < self.config.min_lift:
            return False
        if conviction < self.config.min_conviction:
            return False
        if role_entropy < self.config.min_role_entropy:
            return False
        return True

    def _score(self, rule: RuleDefinition) -> float:
        return float(rule.metrics.get(self.config.score_key, 0.0))
