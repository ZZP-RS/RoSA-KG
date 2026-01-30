"""
Utility (MSRS component) for translating active roles into virtual relation triples
that can be ingested by the RoSA-KG GCN module.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .role_trainer import RoleState
from ..rule_mining import RuleDefinition


@dataclass
class VirtualRelationConfig:
    max_edges_per_role: int = 1024
    sample_with_replacement: bool = False
    random_state: int = 7


class VirtualRelationBuilder:
    def __init__(self, config: VirtualRelationConfig, triplets: Optional[np.ndarray] = None):
        self.config = config
        self._rng = random.Random(config.random_state)
        self._pair_cache: Dict[Tuple[Tuple[int, ...], int], np.ndarray] = {}
        self._position_cache: Dict[Tuple[Tuple[int, ...], int], List[np.ndarray]] = {}
        self._relation_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self._head_to_tails: Dict[int, Dict[int, Set[int]]] = {}
        self._head_tail_sets: Dict[int, Set[Tuple[int, int]]] = {}
        self._item_popularity: Dict[int, float] = {}
        self._sampling_eta: float = 0.8
        self._sampling_tau: float = 1.2
        self._sampling_role_coeff: float = 0.5
        self._min_weight: float = 1e-3
        if triplets is not None:
            self._build_indices(triplets)

    def set_item_popularity(self, popularity: Mapping[int, float]) -> None:
        self._item_popularity = dict(popularity)

    def update_sampling_params(self, eta: float, tau: float, role_coeff: float) -> None:
        self._sampling_eta = max(0.0, float(eta))
        self._sampling_tau = max(0.5, float(tau))
        self._sampling_role_coeff = float(role_coeff)

    def build_edges(
        self,
        state: RoleState,
        rule: RuleDefinition,
        limit: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int, int]], Dict]:
        pair_array = self._materialize_pairs(rule)
        if pair_array.size == 0:
            return [], {}
        if limit is None:
            limit = self.config.max_edges_per_role
        total_pairs = pair_array.shape[0]
        if total_pairs == 0:
            return [], {}
        limit = min(limit, total_pairs)
        if limit <= 0:
            return [], {}

        weights = self._compute_pair_weights(pair_array, state)
        selected, selected_weights = self._weighted_sample(pair_array, weights, limit)

        triples = [(int(head), state.relation_id, int(tail)) for head, tail in selected]
        metadata = {
            "rule_key": (list(rule.body_relations), rule.head_relation),
            "relation_id": state.relation_id,
            "role_type": state.descriptor.role_type.value,
            "weight": state.weight,
            "num_edges": len(triples),
            "avg_pair_weight": float(np.mean(weights)) if weights.size else 0.0,
            "edge_weights": selected_weights.tolist() if selected_weights.size else [],
            "div_reward": float(getattr(state, "div_score_ema", 0.0)),
        }
        return triples, metadata

    def build_batch(
        self,
        states: Sequence[RoleState],
        lookup: Dict[Tuple[Tuple[int, ...], int], RuleDefinition],
        limit_per_role: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int, int]], List[Dict]]:
        all_triples: List[Tuple[int, int, int]] = []
        metadata: List[Dict] = []
        try:
            from tqdm import tqdm  # best-effort progress
            iterator = tqdm(states, desc="[Phase] 角色生成虚拟边", total=len(states))
        except Exception:
            iterator = states
        for state in iterator:
            rule = lookup.get(state.descriptor.rule_key)
            if rule is None:
                continue
            triples, info = self.build_edges(state, rule, limit_per_role)
            if not triples:
                continue
            all_triples.extend(triples)
            metadata.append(info)
        return all_triples, metadata

    def _build_indices(self, triplets: np.ndarray) -> None:
        """Prepare per-relation adjacency caches for role entity materialisation."""
        rel_pairs: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        head_map: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
        # ensure we operate on plain Python ints for downstream speed
        for head, relation, tail in triplets:
            h = int(head)
            r = int(relation)
            t = int(tail)
            rel_pairs[r].add((h, t))
            head_map[r][h].add(t)
        self._relation_pairs = {rel: sorted(pairs) for rel, pairs in rel_pairs.items()}
        self._head_to_tails = {
            rel: {head: set(tails) for head, tails in mapping.items()}
            for rel, mapping in head_map.items()
        }
        self._head_tail_sets = {rel: set(pairs) for rel, pairs in rel_pairs.items()}

    def _compute_pair_weights(
        self, pair_array: np.ndarray, state: RoleState
    ) -> np.ndarray:
        if pair_array.size == 0:
            return np.empty((0,), dtype=np.float32)
        pop = self._item_popularity
        eta = self._sampling_eta
        tau = self._sampling_tau
        role_coeff = self._sampling_role_coeff
        role_term = max(0.1, 1.0 + role_coeff * state.div_score_ema)
        weights = []
        for head, tail in pair_array:
            freq_head = max(float(pop.get(int(head), 0.0)), 1.0)
            freq_tail = max(float(pop.get(int(tail), 0.0)), 1.0)
            mean_freq = 0.5 * (freq_head + freq_tail)
            inv_pop = 1.0 / math.sqrt(mean_freq)
            base = (1.0 + eta * inv_pop) ** tau
            weights.append(max(self._min_weight, base * role_term))
        return np.asarray(weights, dtype=np.float32)

    def _weighted_sample(
        self, pair_array: np.ndarray, weights: np.ndarray, limit: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if weights.size == 0:
            slice_array = pair_array[:limit]
            return slice_array, np.ones((slice_array.shape[0],), dtype=np.float32)
        if not self.config.sample_with_replacement and limit >= pair_array.shape[0]:
            return pair_array.copy(), weights[: pair_array.shape[0]].copy()
        selected_indices: List[int] = []
        available = list(range(pair_array.shape[0]))
        weight_list = weights.tolist()
        for _ in range(limit):
            if not available:
                break
            total = sum(weight_list[idx] for idx in available)
            if total <= 0.0:
                selected_indices.extend(available[: limit - len(selected_indices)])
                break
            r = self._rng.random() * total
            cumulative = 0.0
            pick_idx = available[0]
            for idx in available:
                cumulative += weight_list[idx]
                if r <= cumulative:
                    pick_idx = idx
                    break
            selected_indices.append(pick_idx)
            if self.config.sample_with_replacement:
                continue
            available.remove(pick_idx)
        selected_array = pair_array[selected_indices]
        selected_weights = weights[selected_indices]
        return selected_array, selected_weights

    def _materialize_position_entities_from_samples(self, rule: RuleDefinition) -> List[np.ndarray]:
        body_len = len(rule.body_relations)
        if body_len == 0:
            arrays = [np.empty((0,), dtype=np.int64)]
            self._position_cache[(rule.body_relations, rule.head_relation)] = arrays
            return arrays
        positions: List[Set[int]] = [set() for _ in range(body_len + 1)]
        for record in rule.sample_paths or []:
            path = record.get("path")
            if not path:
                continue
            if len(path) < body_len + 1:
                continue
            for idx in range(body_len + 1):
                positions[idx].add(int(path[idx]))
        if rule.cached_pairs:
            for head, tail in rule.cached_pairs:
                positions[0].add(int(head))
                positions[-1].add(int(tail))
        arrays = [
            np.array(sorted(nodes), dtype=np.int64) if nodes else np.empty((0,), dtype=np.int64)
            for nodes in positions
        ]
        return arrays

    def _materialize_position_entities(self, rule: RuleDefinition) -> List[np.ndarray]:
        key = (rule.body_relations, rule.head_relation)
        cached = self._position_cache.get(key)
        if cached is not None:
            return cached
        body = list(rule.body_relations)
        body_len = len(body)
        if body_len == 0:
            arrays = [np.empty((0,), dtype=np.int64)]
            self._position_cache[key] = arrays
            return arrays
        # if relation indices missing fallback to sample-based approximation
        if not self._relation_pairs or any(rel not in self._head_to_tails for rel in body):
            arrays = self._materialize_position_entities_from_samples(rule)
            self._position_cache[key] = arrays
            return arrays
        pair_array = self._materialize_pairs(rule)
        if pair_array.size == 0:
            arrays = [np.empty((0,), dtype=np.int64) for _ in range(body_len + 1)]
            self._position_cache[key] = arrays
            return arrays
        support_pairs = [(int(h), int(t)) for h, t in pair_array.tolist()]
        positions: List[Set[int]] = [set() for _ in range(body_len + 1)]
        head_rel_pairs = self._head_tail_sets.get(int(rule.head_relation), set())

        def dfs(idx: int, current: int, tail_target: int, visited: Set[Tuple[int, int]]) -> bool:
            state = (idx, current)
            if state in visited:
                return False
            visited.add(state)
            if idx == body_len:
                match = current == tail_target
                visited.remove(state)
                return match
            rel = int(body[idx])
            rel_map = self._head_to_tails.get(rel)
            if not rel_map:
                visited.remove(state)
                return False
            tails = rel_map.get(current)
            if not tails:
                visited.remove(state)
                return False
            success = False
            for nxt in tails:
                if dfs(idx + 1, nxt, tail_target, visited):
                    positions[idx + 1].add(int(nxt))
                    success = True
            visited.remove(state)
            return success

        for head, tail in support_pairs:
            if head_rel_pairs and (head, tail) not in head_rel_pairs:
                continue
            visited: Set[Tuple[int, int]] = set()
            if dfs(0, head, tail, visited):
                positions[0].add(int(head))
                positions[-1].add(int(tail))

        arrays = [
            np.array(sorted(nodes), dtype=np.int64) if nodes else np.empty((0,), dtype=np.int64)
            for nodes in positions
        ]
        if not any(arr.size for arr in arrays):
            arrays = self._materialize_position_entities_from_samples(rule)
        self._position_cache[key] = arrays
        return arrays

    def sample_role_entities(
        self,
        rule: RuleDefinition,
        position: int,
        limit: Optional[int] = None,
    ) -> np.ndarray:
        arrays = self._materialize_position_entities(rule)
        if position < 0 or position >= len(arrays):
            return np.empty((0,), dtype=np.int64)
        candidates = arrays[position]
        total = int(candidates.shape[0])
        if total == 0:
            return candidates
        if limit is None:
            limit = self.config.max_edges_per_role
        if limit is None or limit <= 0:
            return np.empty((0,), dtype=np.int64)
        if not self.config.sample_with_replacement:
            limit = min(limit, total)
            if limit >= total:
                selected_list = candidates.tolist()
            else:
                indices = self._rng.sample(range(total), limit)
                selected_list = [int(candidates[idx]) for idx in indices]
        else:
            indices = [self._rng.randrange(total) for _ in range(limit)]
            selected_list = [int(candidates[idx]) for idx in indices]
        if not selected_list:
            return np.empty((0,), dtype=np.int64)
        self._rng.shuffle(selected_list)
        return np.array(selected_list, dtype=np.int64)

    def get_position_entities(self, rule: RuleDefinition) -> List[np.ndarray]:
        """Expose cached position-specific entity sets for external quality filtering."""
        return self._materialize_position_entities(rule)

    def _materialize_pairs(self, rule: RuleDefinition) -> np.ndarray:
        key = (rule.body_relations, rule.head_relation)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        if rule.cached_pairs:
            array = np.array(rule.cached_pairs, dtype=np.int64)
            self._pair_cache[key] = array
            return array
        pairs: List[Tuple[int, int]] = []
        for sample in rule.sample_paths:
            head = sample.get("head", [None])[0]
            tail = sample.get("tail", [None])[0]
            if head is None or tail is None:
                continue
            pairs.append((head, tail))
        array = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
        self._pair_cache[key] = array
        return array
