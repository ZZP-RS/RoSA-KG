"""
Utility components for diversity-aware structural role filtering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class DiversityControl:
    """Hyper-parameters controlling how diversity scoring is computed."""

    gamma_inv_pop: float = 0.6
    beta_exposure: float = 0.4
    reward_boost: float = 0.5
    exposure_decay: float = 0.98
    min_exposure: float = 1e-4
    clamp_score: float = 1e-3

    def update_from_feedback(self, acc_grad: float, div_grad: float) -> None:
        # Adjust knobs smoothly to avoid oscillations.
        scale = 0.05
        self.gamma_inv_pop = max(
            0.1, self.gamma_inv_pop * (1.0 + scale * div_grad - scale * 0.5 * acc_grad)
        )
        self.beta_exposure = min(
            0.95, max(0.05, self.beta_exposure * (1.0 - scale * div_grad))
        )
        self.reward_boost = max(
            0.1, self.reward_boost * (1.0 + scale * div_grad * 0.5)
        )


class ExposureTracker:
    """Lightweight exponential moving average tracker for item exposures."""

    def __init__(self, decay: float = 0.98) -> None:
        self.decay = float(decay)
        self._stats: Dict[int, float] = {}

    def register(self, entity_ids: Iterable[int]) -> None:
        seen: Dict[int, int] = {}
        for ent in entity_ids:
            ent = int(ent)
            seen[ent] = seen.get(ent, 0) + 1
        for ent, cnt in seen.items():
            prev = self._stats.get(ent, 0.0)
            self._stats[ent] = prev * self.decay + float(cnt)

    def exposure(self, entity_id: int) -> float:
        return self._stats.get(int(entity_id), 0.0)

    def exposures(self, entity_ids: Sequence[int]) -> np.ndarray:
        return np.array([self.exposure(ent) for ent in entity_ids], dtype=np.float32)


class RoleDiversityScorer:
    """Score candidate entities by combining reliability and diversity signals."""

    def __init__(
        self,
        control: DiversityControl,
        exposure_tracker: ExposureTracker,
        popularity: Mapping[int, float],
    ) -> None:
        self.control = control
        self.exposure_tracker = exposure_tracker
        self.popularity = popularity
        self._pop_cache: Dict[int, float] = {}

    def _inverse_popularity(self, entity_ids: Sequence[int]) -> np.ndarray:
        inv_values = []
        for ent in entity_ids:
            ent = int(ent)
            cached = self._pop_cache.get(ent)
            if cached is not None:
                inv_values.append(cached)
                continue
            freq = max(float(self.popularity.get(ent, 0.0)), 1.0)
            inv = 1.0 / math.sqrt(freq)
            self._pop_cache[ent] = inv
            inv_values.append(inv)
        return np.asarray(inv_values, dtype=np.float32)

    def score(
        self,
        entity_ids: Sequence[int],
        base_scores: Mapping[int, float],
        role_reward: float = 0.0,
    ) -> Dict[int, float]:
        if not entity_ids:
            return {}
        entity_ids = [int(e) for e in entity_ids]
        base = np.array(
            [float(base_scores.get(e, 1.0)) for e in entity_ids], dtype=np.float32
        )
        inv_pop = self._inverse_popularity(entity_ids)
        exposures = self.exposure_tracker.exposures(entity_ids)

        gamma = float(self.control.gamma_inv_pop)
        beta = float(self.control.beta_exposure)
        boost = 1.0 + float(self.control.reward_boost) * float(role_reward)

        score = base * (1.0 + gamma * inv_pop) * np.maximum(
            float(self.control.min_exposure),
            (1.0 - beta * exposures / (1.0 + exposures)),
        )
        score = score * max(self.control.clamp_score, boost)
        return {ent: float(max(self.control.clamp_score, sc)) for ent, sc in zip(entity_ids, score)}

