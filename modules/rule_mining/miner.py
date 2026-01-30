"""
Rule mining module (RSRG component) that mirrors AMIE+ style Horn rule discovery
using Python tooling. It works on the preprocessed knowledge graph triplets and
produces rule definitions enriched with statistical indicators that are compatible
with downstream dynamic training.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from collections import defaultdict
from contextlib import suppress


@dataclass
class RuleMinerConfig:
    max_length: int = 3
    min_support: int = 10
    min_confidence: float = 0.2
    min_pca_confidence: float = 0.3
    topk_per_length: int = 500
    cache_matches: bool = True
    sample_size: int = 200
    max_body_evaluations: Optional[int] = None
    random_state: int = 42
    # Performance knobs
    sample_bodies: bool = False
    body_sample_rate: float = 0.3  # applied to length>=3 bodies (probability)
    use_gpu: bool = True  # best-effort; falls back to CPU if unavailable
    gpu_entity_threshold: int = 4_000  # conservative threshold for dense fallback safety
    use_torch_sparse: bool = True  # prefer torch-sparse on CUDA if available
    # CPU sparse acceleration (SciPy)
    use_scipy: bool = True
    max_intermediate_nnz: int = 10_000_000  # guardrail for spsp densification
    product_density_limit: float = 0.02     # if nnz/(n*n) exceeds, fallback


@dataclass
class RuleDefinition:
    body_relations: Tuple[int, ...]
    head_relation: int
    support: int
    body_support: int
    head_support: int
    metrics: Dict[str, float] = field(default_factory=dict)
    position_unique_counts: List[int] = field(default_factory=list)
    cached_pairs: Optional[List[Tuple[int, int]]] = None
    sample_paths: List[Dict[str, List[int]]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["body_relations"] = list(self.body_relations)
        return payload

    @staticmethod
    def from_dict(payload: Dict) -> "RuleDefinition":
        obj = RuleDefinition(
            body_relations=tuple(payload["body_relations"]),
            head_relation=payload["head_relation"],
            support=payload["support"],
            body_support=payload["body_support"],
            head_support=payload["head_support"],
            metrics=dict(payload.get("metrics", {})),
            position_unique_counts=list(payload.get("position_unique_counts", [])),
            cached_pairs=None,
            sample_paths=list(payload.get("sample_paths", [])),
        )
        cached = payload.get("cached_pairs")
        if cached is not None:
            obj.cached_pairs = [tuple(pair) for pair in cached]
        return obj

    def head_coverage(self) -> float:
        if self.head_support == 0:
            return 0.0
        return self.support / float(self.head_support)

    def confidence(self) -> float:
        if self.body_support == 0:
            return 0.0
        return self.support / float(self.body_support)

    def pca_confidence(self) -> float:
        return self.metrics.get("pca_confidence", 0.0)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RuleMiner:
    def __init__(self, triplets: np.ndarray, config: RuleMinerConfig):
        if triplets.ndim != 2 or triplets.shape[1] != 3:
            raise ValueError("Triplets must be a [n,3] numpy array.")
        self.triplets = triplets.astype(np.int64, copy=False)
        self.config = config
        self._rng = random.Random(config.random_state)
        self._setup_indices()
        self._torch_enabled = False
        self._scipy_enabled = False
        self._torch_sparse_enabled = False
        # Try enable GPU acceleration for body materialization (torch-sparse preferred)
        if self.config.use_gpu and self.config.use_torch_sparse:
            try:
                import torch  # type: ignore
                from torch_sparse import SparseTensor  # type: ignore
                if torch.cuda.is_available() and len(self.entities) <= self.config.gpu_entity_threshold:
                    self._torch = torch
                    self._SparseTensor = SparseTensor
                    self._device = torch.device("cuda")
                    self._build_sparse_mats_torch_sparse()
                    self._torch_sparse_enabled = True
                    print("[静态规则] torch-sparse 稀疏乘法加速开启 (GPU)")
            except Exception:
                self._torch_sparse_enabled = False
        # Fallback GPU using torch.sparse (small graphs only)
        if not self._torch_sparse_enabled and self.config.use_gpu:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available() and len(self.entities) <= self.config.gpu_entity_threshold:
                    self._torch = torch
                    self._device = torch.device("cuda")
                    self._build_sparse_mats_gpu()
                    self._torch_enabled = True
                    print("[静态规则] GPU加速开启 (torch.sparse，小图)")
            except Exception:
                self._torch_enabled = False
        # Try enable SciPy sparse multiplication path on CPU when GPU path not enabled
        if not self._torch_enabled and self.config.use_scipy:
            try:
                import scipy.sparse as sps  # type: ignore
                self._sps = sps
                self._build_sparse_mats_cpu()
                self._scipy_enabled = True
                print("[静态规则] SciPy稀疏乘法加速开启 (CPU)")
            except Exception:
                self._scipy_enabled = False

        # Choose materialization implementation
        if self._torch_sparse_enabled:
            self._materialize_impl = self._materialize_body_torch_sparse
        elif self._torch_enabled:
            self._materialize_impl = self._materialize_body_gpu
        elif self._scipy_enabled:
            self._materialize_impl = self._materialize_body_sparse
        else:
            self._materialize_impl = self._materialize_body_cpu

    def _setup_indices(self) -> None:
        self.relations: Set[int] = set(int(r) for r in np.unique(self.triplets[:, 1]))
        self.entities: Set[int] = set(int(e) for e in np.unique(self.triplets[:, [0, 2]]))
        self.relation_to_pairs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.head_to_tails: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.tail_to_heads: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

        for h, r, t in self.triplets:
            h = int(h)
            r = int(r)
            t = int(t)
            self.relation_to_pairs[r].append((h, t))
            self.head_to_tails[r][h].add(t)
            self.tail_to_heads[r][t].add(h)

        self.relation_pair_sets: Dict[int, Set[Tuple[int, int]]] = {
            r: set(pairs) for r, pairs in self.relation_to_pairs.items()
        }
        self.relation_support: Dict[int, int] = {
            r: len(pairs) for r, pairs in self.relation_to_pairs.items()
        }
        self.total_pairs = max(1, len(self.entities) * len(self.entities))

    def mine(self) -> List[RuleDefinition]:
        results: List[RuleDefinition] = []
        evaluations = 0
        rel_count = max(1, len(self.relations))
        tqdm = None
        with suppress(Exception):
            from tqdm import tqdm as _tqdm  # optional
            tqdm = _tqdm
        for length in range(1, self.config.max_length + 1):
            bodies = self._enumerate_bodies(length)
            body_candidates: List[RuleDefinition] = []
            est_total = rel_count ** length
            remaining = (
                (self.config.max_body_evaluations - evaluations)
                if self.config.max_body_evaluations
                else est_total
            )
            iterator = bodies
            if tqdm is not None:
                iterator = tqdm(bodies, total=min(est_total, max(1, int(remaining))),
                                desc=f"[静态规则] 枚举规则体 长度={length}")
            for body in iterator:
                if self.config.max_body_evaluations and evaluations >= self.config.max_body_evaluations:
                    break
                # 无损剪枝：上界支持度小于阈值则跳过（不可能达标）
                try:
                    upper_bound = min(self.relation_support.get(int(rel), 0) for rel in body)
                    if upper_bound < self.config.min_support:
                        continue
                except Exception:
                    pass
                # 可选采样：仅对 length>=3 的体做概率采样，避免遗漏长度1/2的体
                if length >= 3 and self.config.sample_bodies:
                    rate = max(0.0, min(1.0, self.config.body_sample_rate)) ** (length - 2)
                    if self._rng.random() > rate:
                        continue
                evaluations += 1
                discovered = self._evaluate_body(body)
                body_candidates.extend(discovered)
            body_candidates.sort(key=lambda rule: rule.metrics.get("score", 0.0), reverse=True)
            if self.config.topk_per_length:
                body_candidates = body_candidates[: self.config.topk_per_length]
            results.extend(body_candidates)
        return results

    def _enumerate_bodies(self, length: int) -> Iterable[Tuple[int, ...]]:
        relation_list = sorted(self.relations)
        if length == 1:
            for rel in relation_list:
                yield (rel,)
            return

        def recurse(current: Tuple[int, ...], depth: int):
            if depth == length:
                yield current
                return
            for rel in relation_list:
                yield from recurse(current + (rel,), depth + 1)

        for rel in relation_list:
            yield from recurse((rel,), 1)

    def _materialize_body_cpu(
        self, body: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[Dict[str, List[int]]]]:
        if not body:
            return [], [], []

        position_unique: List[Set[int]] = [set() for _ in range(len(body) + 1)]
        frontier: List[Tuple[int, List[int]]] = []

        first_rel = body[0]
        for head, tail in self.relation_to_pairs.get(first_rel, []):
            frontier.append((head, [head, tail]))

        if not frontier:
            return [], [0] * (len(body) + 1), []

        position_unique[0].update(head for head, _ in frontier)
        position_unique[1].update(path[-1] for _, path in frontier)

        for rel_index, rel in enumerate(body[1:], start=2):
            next_frontier: List[Tuple[int, List[int]]] = []
            rel_map = self.head_to_tails.get(rel, {})
            for head, node_path in frontier:
                last_node = node_path[-1]
                tails = rel_map.get(last_node)
                if not tails:
                    continue
                for tail in tails:
                    new_path = node_path + [tail]
                    next_frontier.append((head, new_path))
            frontier = next_frontier
            if not frontier:
                return [], [len(nodes) for nodes in position_unique], []
            position_unique[rel_index].update(path[-1] for _, path in frontier)

        support_pairs: List[Tuple[int, int]] = []
        seen_pairs: Set[Tuple[int, int]] = set()
        sample_paths: List[Dict[str, List[int]]] = []
        max_samples = max(0, int(self.config.sample_size))
        total_seen = 0

        for head, node_path in frontier:
            tail = node_path[-1]
            pair = (head, tail)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            support_pairs.append(pair)
            total_seen += 1
            if max_samples > 0:
                record = {"head": [head], "tail": [tail], "path": list(node_path)}
                if len(sample_paths) < max_samples:
                    sample_paths.append(record)
                else:
                    idx = self._rng.randrange(total_seen)
                    if idx < max_samples:
                        sample_paths[idx] = record

        position_counts = [len(nodes) for nodes in position_unique]
        return support_pairs, position_counts, sample_paths

    def _build_sparse_mats_torch_sparse(self) -> None:
        torch = self._torch
        SparseTensor = self._SparseTensor
        device = self._device
        n = int(max(self.entities) + 1)
        self._rel_adj_ts = {}
        for r, pairs in self.relation_to_pairs.items():
            if not pairs:
                continue
            arr = np.array(pairs, dtype=np.int64)
            rows = torch.from_numpy(arr[:, 0]).to(device)
            cols = torch.from_numpy(arr[:, 1]).to(device)
            A = SparseTensor(row=rows, col=cols, sparse_sizes=(n, n)).coalesce()
            self._rel_adj_ts[int(r)] = A

    def _build_sparse_mats_gpu(self) -> None:
        """Prepare per-relation sparse adjacency on GPU (boolean values).

        This is a best-effort accelerator. Only used for pair materialization.
        """
        torch = self._torch
        device = self._device
        n = int(max(self.entities) + 1)
        self._rel_adj_gpu = {}
        for r, pairs in self.relation_to_pairs.items():
            if not pairs:
                continue
            arr = np.array(pairs, dtype=np.int64)
            rows = torch.from_numpy(arr[:, 0]).to(device)
            cols = torch.from_numpy(arr[:, 1]).to(device)
            vals = torch.ones(rows.size(0), device=device, dtype=torch.float32)
            coo = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n))
        self._rel_adj_gpu[int(r)] = coo.coalesce()

    def _build_sparse_mats_cpu(self) -> None:
        """Prepare per-relation SciPy CSR adjacency (boolean)."""
        n = int(max(self.entities) + 1)
        sps = self._sps
        self._rel_adj_csr: Dict[int, "sps.csr_matrix"] = {}
        for r, pairs in self.relation_to_pairs.items():
            if not pairs:
                continue
            arr = np.array(pairs, dtype=np.int64)
            data = np.ones(arr.shape[0], dtype=np.float32)
            mat = sps.csr_matrix((data, (arr[:, 0], arr[:, 1])), shape=(n, n))
            mat.data[:] = 1.0
            self._rel_adj_csr[int(r)] = mat

    def _materialize_body_gpu(
        self, body: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[Dict[str, List[int]]]]:
        # Requires GPU sparse matrices prepared
        if not body:
            return [], [], []
        torch = self._torch
        device = self._device
        # First step adjacency
        A = self._rel_adj_gpu.get(int(body[0]))
        if A is None or A._nnz() == 0:
            return [], [0] * (len(body) + 1), []
        # position 0 heads and 1 tails from first relation
        pos_counts = []
        idx = A.indices()
        pos_counts.append(int(idx[0].unique().numel()))  # position 0
        pos_counts.append(int(idx[1].unique().numel()))  # position 1
        M = A
        for step, rel in enumerate(body[1:], start=2):
            B = self._rel_adj_gpu.get(int(rel))
            if B is None or B._nnz() == 0:
                return [], [len_ for len_ in pos_counts] + [0] * (len(body) + 1 - len(pos_counts)), []
            # Sparse @ Sparse -> approximate via dense fallback on small blocks is unsafe; try coalesce spsp via to_dense if tiny
            try:
                # Only allow dense fallback when the target dense is reasonably small
                rows, cols = M.size(0), B.size(1)
                if rows * cols > 25_000_000:  # ~100MB for float32
                    return self._materialize_body_cpu(body)
                M = torch.sparse.mm(M, B.to_dense()).to_sparse()
            except Exception:
                # Fall back to CPU materialization for this body
                return self._materialize_body_cpu(body)
            M = M.coalesce()
            if M._nnz() == 0:
                return [], [len_ for len_ in pos_counts] + [0] * (len(body) + 1 - len(pos_counts)), []
            idxM = M.indices()
            pos_counts.append(int(idxM[1].unique().numel()))
        # pairs from non-zeros of M
        idxM = M.indices()
        heads = idxM[0].detach().cpu().numpy()
        tails = idxM[1].detach().cpu().numpy()
        pairs = list({(int(h), int(t)) for h, t in zip(heads, tails)})
        # sampling of paths not available; return empty sample_paths to save time/mem
        return pairs, pos_counts, []

    def _materialize_body_torch_sparse(
        self, body: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[Dict[str, List[int]]]]:
        if not body:
            return [], [], []
        torch = self._torch
        # First relation tensor
        A = self._rel_adj_ts.get(int(body[0]))
        if A is None or A.nnz() == 0:
            return [], [0] * (len(body) + 1), []
        pos_counts: List[int] = []
        r0, c0, _ = A.coo()
        pos_counts.append(int(torch.unique(r0).numel()))
        pos_counts.append(int(torch.unique(c0).numel()))
        M = A
        for rel in body[1:]:
            B = self._rel_adj_ts.get(int(rel))
            if B is None or B.nnz() == 0:
                return [], pos_counts + [0] * (len(body) + 1 - len(pos_counts)), []
            # SparseTensor @ SparseTensor on CUDA
            try:
                M = M.matmul(B).coalesce()
            except Exception:
                return self._materialize_body_cpu(body)
            if M.nnz() == 0:
                return [], pos_counts + [0] * (len(body) + 1 - len(pos_counts)), []
            _, cM, _ = M.coo()
            pos_counts.append(int(torch.unique(cM).numel()))
        rM, cM, _ = M.coo()
        pairs = list({(int(h), int(t)) for h, t in zip(rM.tolist(), cM.tolist())})
        return pairs, pos_counts, []

    def _materialize_body_sparse(
        self, body: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[Dict[str, List[int]]]]:
        if not body:
            return [], [], []
        sps = self._sps
        n = int(max(self.entities) + 1)
        A = self._rel_adj_csr.get(int(body[0]))
        if A is None or A.nnz == 0:
            return [], [0] * (len(body) + 1), []
        pos_counts: List[int] = []
        pos_counts.append(int(np.count_nonzero(A.getnnz(axis=1))))
        pos_counts.append(int(np.count_nonzero(A.getnnz(axis=0))))
        M = A
        for rel in body[1:]:
            B = self._rel_adj_csr.get(int(rel))
            if B is None or B.nnz == 0:
                return [], pos_counts + [0] * (len(body) + 1 - len(pos_counts)), []
            if M.nnz > self.config.max_intermediate_nnz:
                return self._materialize_body_cpu(body)
            M = (M @ B).astype(bool)
            density = M.nnz / float(n * n)
            if density > self.config.product_density_limit:
                return self._materialize_body_cpu(body)
            pos_counts.append(int(np.count_nonzero(M.getnnz(axis=0))))
            if M.nnz == 0:
                return [], pos_counts + [0] * (len(body) + 1 - len(pos_counts)), []
        rows, cols = M.nonzero()
        pairs = list({(int(h), int(t)) for h, t in zip(rows.tolist(), cols.tolist())})
        return pairs, pos_counts, []

    def _evaluate_body(self, body: Sequence[int]) -> List[RuleDefinition]:
        support_pairs_list, position_counts, sample_paths_all = self._materialize_impl(body)
        if not support_pairs_list:
            return []
        body_pairs = set(support_pairs_list)
        body_support = len(body_pairs)
        if body_support < self.config.min_support:
            return []

        results: List[RuleDefinition] = []
        for head_relation, head_pairs in self.relation_pair_sets.items():
            support_pairs = body_pairs.intersection(head_pairs)
            support = len(support_pairs)
            if support < self.config.min_support:
                continue
            head_support = self.relation_support.get(head_relation, 0)
            confidence = support / float(body_support)
            if confidence < self.config.min_confidence:
                continue
            pca_denominator = 0
            head_tail_map = self.head_to_tails.get(head_relation, {})
            for head_entity, _ in support_pairs:
                candidate_tails = head_tail_map.get(head_entity)
                if candidate_tails:
                    pca_denominator += len(candidate_tails)
                else:
                    pca_denominator += 1
            if pca_denominator == 0:
                pca_denominator = len(support_pairs)
            pca_confidence = support / float(pca_denominator)
            if pca_confidence < self.config.min_pca_confidence:
                continue

            head_ratio = head_support / float(self.total_pairs)
            lift = confidence / head_ratio if head_ratio > 0 else 0.0
            conviction = ((1.0 - head_ratio) / (1.0 - confidence + 1e-9)) if confidence < 1.0 else float("inf")
            role_entropy = 0.0
            total_nodes = sum(position_counts)
            if total_nodes > 0:
                for count in position_counts:
                    if count > 0:
                        prob = count / float(total_nodes)
                        role_entropy -= prob * math.log(prob + 1e-12)

            metrics = {
                "confidence": confidence,
                "pca_confidence": pca_confidence,
                "head_coverage": support / float(head_support) if head_support else 0.0,
                "lift": lift,
                "conviction": conviction,
                "role_entropy": role_entropy,
                "score": support * confidence * (1.0 + lift),
            }

            cached_pairs = None
            if self.config.cache_matches:
                cached_pairs = sorted(support_pairs)
            if sample_paths_all:
                support_pair_set = set(support_pairs)
                sample_paths = [
                    record for record in sample_paths_all
                    if (record["head"][0], record["tail"][0]) in support_pair_set
                ]
            else:
                sample_paths = []
            rule = RuleDefinition(
                body_relations=tuple(body),
                head_relation=head_relation,
                support=support,
                body_support=body_support,
                head_support=head_support,
                metrics=metrics,
                position_unique_counts=position_counts,
                cached_pairs=cached_pairs,
                sample_paths=sample_paths,
            )
            results.append(rule)
        return results

    @staticmethod
    def save_rules(path: str, rules: Sequence[RuleDefinition]) -> None:
        payload = [rule.to_dict() for rule in rules]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    @staticmethod
    def load_rules(path: str) -> List[RuleDefinition]:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return [RuleDefinition.from_dict(item) for item in payload]
