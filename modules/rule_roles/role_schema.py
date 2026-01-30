"""
Role schema definitions used to translate mined rules into trainable structural roles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class RoleType(str, Enum):
    SOURCE = "source"
    CONNECTOR = "connector"
    TARGET = "target"
    CONDITION = "condition"
    SUPPRESSOR = "suppressor"

    @staticmethod
    def for_position(position: int, body_length: int) -> "RoleType":
        if position <= 0:
            return RoleType.SOURCE
        if position >= body_length:
            return RoleType.TARGET
        return RoleType.CONNECTOR


@dataclass
class RoleDescriptor:
    rule_key: Tuple[Tuple[int, ...], int]
    position: int
    body_length: int
    size: int
    base_weight: float
    metrics: Dict[str, float] = field(default_factory=dict)
    role_type: RoleType = field(init=False)

    def __post_init__(self) -> None:
        self.role_type = RoleType.for_position(self.position, self.body_length)


def initial_role_weight(metrics: Dict[str, float], position_size: int) -> float:
    confidence = float(metrics.get("confidence", 0.0))
    coverage = float(metrics.get("head_coverage", 0.0))
    lift = float(metrics.get("lift", 0.0))
    entropy = float(metrics.get("role_entropy", 0.0))
    normalization = max(1.0, position_size)
    weight = 0.4 * confidence + 0.3 * coverage + 0.2 * math.tanh(lift) + 0.1 * entropy
    return weight / normalization
