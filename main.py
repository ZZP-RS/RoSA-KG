# RoSA-KG training script.
import os

import sys
import math
import random
import torch
import itertools
import numpy as np
import pickle as pkl
from math import log
from tqdm import tqdm
from time import time 
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import platform
import scipy.sparse as sp
from collections import Counter, defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Any, Sequence, Deque, Mapping
from datetime import datetime
import atexit

from utils.parser import parse_args
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from utils.data_loader import load_data
from sklearn.metrics.pairwise import cosine_similarity

from modules.KGR_model import KGR
from modules.rosa_kg_model import RoSAKGRecommender
from modules.pcgrad import PCGrad
from modules.rule_mining import RuleFilterConfig, RuleMinerConfig
from pipelines import DynamicRuleConfig, DynamicRuleTrainer
from modules.rule_roles import RoleType

from utils.evaluate import test
from utils.helper import early_stopping, _generate_candi_kg, _cal_npmi


cores = max(1, (multiprocessing.cpu_count() * 3) // 4)
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0
n_entities_real = 0  # real entities excluding dynamic role nodes
NEG_POOL_WORKERS = cores
NEG_POOL_USE_THREADS = platform.system().lower().startswith("win")
role_item_user_map: Dict[int, Set[int]] = defaultdict(set)


def _role_log(message: str, log_handle=None, flush: bool = False) -> None:
    print(message)
    if flush:
        sys.stdout.flush()


class _TeeStdout:
    def __init__(self, console_stream, log_stream):
        self.console = console_stream
        self.log_stream = log_stream

    def write(self, data):
        self.console.write(data)
        if self.log_stream:
            self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.console.flush()
        if self.log_stream:
            self.log_stream.flush()

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


ROLE_TYPE_TO_ID = {
    RoleType.SOURCE: 0,
    RoleType.CONNECTOR: 1,
    RoleType.TARGET: 2,
    RoleType.CONDITION: 3,
    RoleType.SUPPRESSOR: 4,
}


def _parse_reward_weights(spec: str):
    default_weights = {
        "recall": 1.0,
        "ndcg": 0.8,
        "precision": 0.5,
        "hit_ratio": 0.3,
        "kgat_ad": 0.05,
        "kgat_arp": -0.02,
        "kgc_acc": 0.35,
        "kgc_loss": -0.2,
    }
    weights = default_weights.copy()
    if not spec:
        return weights
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    for token in tokens:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        key = key.strip()
        try:
            weights[key] = float(value)
        except ValueError:
            continue
    return weights


def _initial_role_bridge_relation(rsrg_trainer, fallback_rel_id: int) -> int:
    if rsrg_trainer is None:
        return int(fallback_rel_id)
    try:
        snapshot = rsrg_trainer.role_manager.serialize()
    except Exception:
        return int(fallback_rel_id)
    next_rel = snapshot.get("next_relation_id", fallback_rel_id)
    try:
        next_rel_int = int(next_rel)
    except (TypeError, ValueError):
        return int(fallback_rel_id)
    return max(int(fallback_rel_id), next_rel_int)


def get_feed_data(train_entity_pairs, train_user_set):

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]

    return feed_dict

def _mulp_neg(triplet):
    h,r,t = triplet
    rand_value = random.random()
    while True:
        if rand_value < 0.33 and n_relations > 1:
            new_rel = random.randint(1, n_relations - 1)
            if new_rel != r:
                new_triplet = [h, new_rel, t]
                break
        elif rand_value>=0.33 and rand_value>0.66 and n_entities_real > 1:
            new_head = random.randint(0, n_entities_real - 1)
            if new_head != h:
                new_triplet = [new_head, r, t]
                break
        else:
            if n_entities_real > n_items:
                new_t = random.randint(n_items, n_entities_real - 1)
            else:
                new_t = t
            if new_t != t:
                new_triplet = [h, r, new_t]
                break
        rand_value = random.random()
    return new_triplet


def _neg_sampler_init(total_real_entities, total_relations, total_items):
    global n_entities_real, n_relations, n_items
    n_entities_real = total_real_entities
    n_relations = total_relations
    n_items = total_items

def _create_neg_pool(processes, initializer, initargs, use_threads):
    if processes <= 1:
        return None
    if use_threads:
        return ThreadPool(processes=processes, initializer=initializer, initargs=initargs)
    ctx = multiprocessing.get_context("spawn")
    return ctx.Pool(processes=processes, initializer=initializer, initargs=initargs)


def _get_KGC_neg_data(triplets, neg_times=4, sample_rate=1.0):
    def negative_sampling(sampled_triplets):
        _neg_sampler_init(n_entities_real, n_relations, n_items)
        data_list = sampled_triplets.tolist()
        total = len(data_list)
        if total == 0:
            return np.empty((0, 3), dtype=np.int64)
        workers = min(max(1, NEG_POOL_WORKERS), total)
        use_threads = NEG_POOL_USE_THREADS
        last_error = None
        while workers > 1:
            try:
                pool = _create_neg_pool(
                    workers,
                    _neg_sampler_init,
                    (n_entities_real, n_relations, n_items),
                    use_threads,
                )
                neg_triplets = list(
                    tqdm(
                        iterable=pool.imap(_mulp_neg, data_list),
                        total=total,
                    )
                )
                pool.close()
                if hasattr(pool, "join"):
                    pool.join()
                return np.array(neg_triplets)
            except (OSError, MemoryError) as exc:
                last_error = exc
                winerr = getattr(exc, "winerror", None)
                if winerr == 1455 or isinstance(exc, MemoryError):
                    workers = max(1, workers // 2)
                    if workers == 1 and use_threads:
                        break
                    continue
                raise
        # fallback sequential / minimal workers
        if last_error is not None and getattr(last_error, "winerror", None) == 1455:
            print("[NegPool] Falling back to sequential negative sampling due to limited virtual memory.")
        return np.array(
            list(tqdm(map(_mulp_neg, data_list), total=total))
        )

    if sample_rate < 1.0:
        sample_size = max(1, int(len(triplets) * sample_rate))
        idx = np.random.choice(len(triplets), size=sample_size, replace=False)
        triplets = triplets[idx]

    all_neg = []
    for _ in range(neg_times):
        neg_triplets = negative_sampling(triplets)
        all_neg.append(neg_triplets)
    return np.unique(np.concatenate(all_neg, axis=0), axis=0)


def _gather_dynamic_triplets(model) -> np.ndarray:
    def _tensor_to_np(edge_index, edge_type):
        if edge_index is None or edge_type is None:
            return None
        if edge_index.numel() == 0 or edge_type.numel() == 0:
            return None
        edge_index_cpu = edge_index.detach().cpu()
        edge_type_cpu = edge_type.detach().cpu()
        stacked = torch.stack([edge_index_cpu[0], edge_type_cpu, edge_index_cpu[1]], dim=1)
        return stacked.numpy().astype(np.int64, copy=False)

    triplet_list = []
    gcn = getattr(model, "gcn", None)
    if gcn is not None:
        aug_edges = _tensor_to_np(getattr(gcn, "aug_edge_index", None), getattr(gcn, "aug_edge_type", None))
        if aug_edges is not None:
            triplet_list.append(aug_edges)
        role_edges = _tensor_to_np(getattr(gcn, "role_edge_index", None), getattr(gcn, "role_edge_type", None))
        if role_edges is not None:
            triplet_list.append(role_edges)
    else:
        aug_edges = _tensor_to_np(getattr(model, "aug_edge_index", None), getattr(model, "aug_edge_type", None))
        if aug_edges is not None:
            triplet_list.append(aug_edges)
        role_edges = _tensor_to_np(getattr(model, "role_edge_index", None), getattr(model, "role_edge_type", None))
        if role_edges is not None:
            triplet_list.append(role_edges)
    if not triplet_list:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(triplet_list, axis=0), axis=0)


def _ensure_role_relations(model, relation_ids: np.ndarray):
    if relation_ids is None or relation_ids.size == 0:
        return
    rel_list = [int(r) for r in np.unique(relation_ids) if int(r) >= 0]
    if not rel_list:
        return
    max_rel = max(rel_list)
    if max_rel >= getattr(model, "n_relations", 0):
        model.ensure_relation_capacity(max_rel + 1)
    gcn_module = getattr(model, "gcn", None)
    if gcn_module is not None and hasattr(gcn_module, "register_role_relations"):
        gcn_module.register_role_relations(rel_list)


def train_kgr_model(model, kgr_optimizer, triplets, kg_mask=None, epochs=2, threshold=0.5,
                    neg_sample_rate=1.0):
    # neg_kg_data = torch.LongTensor(neg_kg_data[index])
    if kg_mask is None:
        kg_mask = []
    if kg_mask!=[]:
        triplets = triplets[kg_mask.reshape(-1)!=0.]
    dynamic_triplets = _gather_dynamic_triplets(model)
    if dynamic_triplets.size > 0:
        triplets = np.unique(np.concatenate([triplets, dynamic_triplets], axis=0), axis=0)

    global n_entities_real, n_relations
    n_entities_real = max(int(model.n_entities), n_entities_real)
    n_relations = max(int(model.n_relations), n_relations)

    kgr_batch = 1024

    pos_label = np.ones((triplets.shape[0],1))
    pos_data = np.concatenate([triplets,pos_label],axis=-1)
    index = np.arange(len(pos_data))
    np.random.shuffle(index)
    pos_data = pos_data[index]

    pos_valid_num = int(len(pos_data) * 0.05)
    pos_train = pos_data[:-pos_valid_num]
    pos_valid = pos_data[-pos_valid_num:]

    kgr_train_tensor = None
    kgr_valid_tensor = torch.LongTensor(pos_valid)
    if kgr_valid_tensor.numel() > 0:
        max_rel = int(kgr_valid_tensor[:, 1].max())
        if max_rel >= model.n_relations:
            model.ensure_relation_capacity(max_rel + 1)
        max_ent = int(kgr_valid_tensor[:, [0, 2]].max())
        if max_ent >= model.n_entities:
            model.ensure_entity_capacity(max_ent + 1)
    kgr_iter = 0
    kgr_val_iter = math.ceil(len(kgr_valid_tensor) / kgr_batch) if kgr_valid_tensor.numel() > 0 else 0

    model.train()
    acc = 0.0
    epoch_loss_history: List[float] = []
    for epoch in tqdm(range(epochs)):
        torch.cuda.empty_cache()
        if epoch % 4 == 0:
            kgc_neg_data = _get_KGC_neg_data(triplets, neg_times=4, sample_rate=neg_sample_rate)
            neg_label = np.zeros((kgc_neg_data.shape[0], 1))
            neg_data = np.concatenate([kgc_neg_data, neg_label], axis=-1)

            neg_train = neg_data[:-pos_valid_num]
            kgr_train_np = np.concatenate([pos_train, neg_train], axis=0)
            index = np.arange(len(kgr_train_np))
            np.random.shuffle(index)
            kgr_train_np = kgr_train_np[index]

            kgr_train_tensor = torch.LongTensor(kgr_train_np)
            if kgr_train_tensor.numel() > 0:
                max_rel = int(kgr_train_tensor[:, 1].max())
                if max_rel >= model.n_relations:
                    model.ensure_relation_capacity(max_rel + 1)
                max_ent = int(kgr_train_tensor[:, [0, 2]].max())
                if max_ent >= model.n_entities:
                    model.ensure_entity_capacity(max_ent + 1)
            kgr_iter = math.ceil(len(kgr_train_tensor) / kgr_batch) if kgr_train_tensor is not None and kgr_train_tensor.numel() > 0 else 0
            kgr_val_iter = math.ceil(len(kgr_valid_tensor) / kgr_batch) if kgr_valid_tensor.numel() > 0 else 0

        if kgr_train_tensor is None or kgr_train_tensor.numel() == 0:
            continue

        batch_kg = dict()
        total_kg_loss = 0.0
        for i in range(kgr_iter):
            batch_kg["hr_pair"] = kgr_train_tensor[i * kgr_batch:(i + 1) * kgr_batch].to(device)

            kgr_batch_loss = model(batch_kg, mode="kgc")
            total_kg_loss += kgr_batch_loss.item()

            kgr_optimizer.zero_grad()
            kgr_batch_loss.backward()
            kgr_optimizer.step()

        if kgr_iter > 0:
            epoch_loss_history.append(total_kg_loss / max(1, kgr_iter))

        if kgr_val_iter > 0:
            model.eval()
            pred_label = []
            true_label = []
            for i in range(kgr_val_iter):
                batch_kg["hr_pair"] = kgr_valid_tensor[i * kgr_batch:(i + 1) * kgr_batch].to(device)
                if batch_kg["hr_pair"].numel() == 0:
                    continue
                pre = model.gcn.kgc(batch_kg, eval=True)
                pre = (pre >= threshold).squeeze(-1).float()
                pre = pre.detach().cpu().numpy()
                label = batch_kg["hr_pair"][:, -1]
                label = label.cpu().numpy()
                pred_label.append(pre)
                true_label.append(label)

            if pred_label:
                pred_label = np.concatenate(pred_label, axis=0)
                true_label = np.concatenate(true_label, axis=0)
                acc = accuracy_score(pred_label, true_label)

            model.train()

    avg_loss = float(np.mean(epoch_loss_history)) if epoch_loss_history else 0.0
    return {"kgc_acc": float(acc), "kgc_loss": avg_loss}

def _process_kg_attr(canditate_kg,tripltes,kg_mask=None):
    
    canditate_kg = np.unique(canditate_kg,axis=0)
    if kg_mask!=None:
        tripltes = tripltes[kg_mask.reshape(-1)!=0]
        
    # attr_triplets = tripltes[tripltes[:,0]>n_items-1]
    attr_set = set(np.unique(canditate_kg[:,-1]))
    out_attr_kg = []
    for tirp in tqdm(tripltes):
        if tirp[0] in attr_set:
            out_attr_kg.append(tirp)
    out_attr_kg = np.asarray(out_attr_kg)
    if out_attr_kg.shape[0] > 0:
        all_candi_kg = np.concatenate([canditate_kg,out_attr_kg],axis=0)
    else:
        all_candi_kg = canditate_kg
    return np.unique(all_candi_kg,axis=0)


def _filter_role_entities(
    state,
    entity_ids: np.ndarray,
    item_pmi_dict: dict,
    item_embs,
    model,
    args,
    rsrg_trainer,
    n_items: int,
    log_handle=None,
):
    if entity_ids.size == 0 or rsrg_trainer is None:
        return entity_ids, {}
    rule = rsrg_trainer.role_manager.get_rule(state.descriptor.rule_key)
    if rule is None:
        return entity_ids, {}
    position_arrays = rsrg_trainer.virtual_builder.get_position_entities(rule)
    anchors: List[int] = []
    if position_arrays:
        anchors.extend(int(x) for x in position_arrays[0] if int(x) < n_items)
        anchors.extend(int(x) for x in position_arrays[-1] if int(x) < n_items)
    anchors = sorted(set(anchors))
    if not anchors:
        return entity_ids, {}
    npmi_threshold = float(getattr(args, "role_npmi_threshold", 0.4))
    cos_threshold = float(getattr(args, "role_cos_threshold", 0.8))
    min_keep = max(1, int(getattr(args, "role_filter_min_keep", 1)))
    topk_ratio = float(getattr(args, "role_topk_ratio", 0.3))

    has_item_emb = isinstance(item_embs, np.ndarray) and item_embs.size > 0
    item_emb_matrix = np.asarray(item_embs) if has_item_emb else None
    anchor_for_cos = [
        a for a in anchors if item_emb_matrix is not None and a < item_emb_matrix.shape[0]
    ]

    entity_ids = entity_ids.astype(np.int64, copy=False)
    original_order: Dict[int, int] = {}
    for idx, ent in enumerate(entity_ids.tolist()):
        ent = int(ent)
        if ent not in original_order:
            original_order[ent] = idx

    scored_items: Dict[int, float] = {}
    preserved_non_items: List[int] = []
    candidate_items: List[int] = []
    for ent in entity_ids.tolist():
        ent = int(ent)
        if ent >= n_items:
            preserved_non_items.append(ent)
        else:
            candidate_items.append(ent)

    if not candidate_items:
        unique_non_items = sorted(
            set(preserved_non_items), key=lambda x: original_order.get(x, 0)
        )
        return np.array(unique_non_items, dtype=np.int64), scored_items

    candidate_items_array = np.asarray(candidate_items, dtype=np.int64)

    best_npmi = np.full(candidate_items_array.shape[0], -1.0, dtype=np.float32)
    for anchor in anchors:
        forward_keys = [f"{item},{anchor}" for item in candidate_items_array]
        reverse_keys = [f"{anchor},{item}" for item in candidate_items_array]
        forward_scores = np.array(
            [item_pmi_dict.get(k, [0.0, -1.0])[1] for k in forward_keys],
            dtype=np.float32,
        )
        reverse_scores = np.array(
            [item_pmi_dict.get(k, [0.0, -1.0])[1] for k in reverse_keys],
            dtype=np.float32,
        )
        best_npmi = np.maximum(best_npmi, np.maximum(forward_scores, reverse_scores))

    npmi_mask = best_npmi >= npmi_threshold
    for item, score in zip(candidate_items_array[npmi_mask], best_npmi[npmi_mask]):
        scored_items[int(item)] = float(score)

    remaining_mask = ~npmi_mask
    if remaining_mask.any() and anchor_for_cos and item_emb_matrix is not None:
        remain_items = candidate_items_array[remaining_mask]
        remain_vectors = item_emb_matrix[remain_items]
        anchor_vectors = item_emb_matrix[anchor_for_cos]
        sims = cosine_similarity(remain_vectors, anchor_vectors, dense_output=True)
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        best_cos = np.max(sims, axis=1)
        cos_mask = best_cos >= cos_threshold
        for item, cos_score in zip(remain_items[cos_mask], best_cos[cos_mask]):
            previous = scored_items.get(int(item), -np.inf)
            scored_items[int(item)] = max(previous, float(cos_score))

    if not scored_items:
        if getattr(args, "progress_verbose", True):
            _role_log(
                f"[Role] role={state.descriptor.role_type.value} rule={state.descriptor.rule_key} "
                f"no scored entities after filters.",
                log_handle,
            )
        return np.array(preserved_non_items, dtype=np.int64), {}

    role_reward = float(getattr(state, "div_score_ema", 0.0))
    diversity_scorer = rsrg_trainer.get_diversity_scorer(role_reward)
    diversity_scores = diversity_scorer.score(
        list(scored_items.keys()),
        scored_items,
        role_reward=role_reward,
    )

    sorted_items = sorted(
        diversity_scores.items(),
        key=lambda kv: (-kv[1], original_order.get(kv[0], n_items + 1)),
    )

    if sorted_items and 0.0 < topk_ratio < 1.0:
        raw_limit = int(math.ceil(len(sorted_items) * topk_ratio))
        topk_limit = max(min_keep, raw_limit)
    else:
        topk_limit = max(min_keep, len(sorted_items))
    topk_limit = min(len(sorted_items), topk_limit)

    top_entities = [item for item, _ in sorted_items[:topk_limit]]

    if not top_entities:
        fallback_items = candidate_items[:min_keep]
        if getattr(args, "progress_verbose", True):
            _role_log(
                f"[Role] role={state.descriptor.role_type.value} rule={state.descriptor.rule_key} "
                f"no ranked entities, fallback {len(fallback_items)} / {len(candidate_items)}",
                log_handle,
            )
        top_entities = fallback_items
        for item in fallback_items:
            base = float(scored_items.get(int(item), npmi_threshold))
            diversity_scores[int(item)] = base
    elif len(top_entities) < min_keep:
        needed = min_keep - len(top_entities)
        for item in candidate_items:
            if item in top_entities:
                continue
            top_entities.append(item)
            diversity_scores[int(item)] = float(scored_items.get(int(item), npmi_threshold))
            needed -= 1
            if needed <= 0:
                break

    context_cap = max(0, int(getattr(args, "role_context_cap", 64)))
    final_entities: List[int] = []
    seen = set()
    for ent in top_entities:
        if ent not in seen:
            final_entities.append(ent)
            seen.add(ent)
    if context_cap > 0 and preserved_non_items:
        for ent in preserved_non_items[:context_cap]:
            if ent not in seen:
                final_entities.append(ent)
                seen.add(ent)
    if hasattr(rsrg_trainer, "register_entity_exposure"):
        rsrg_trainer.register_entity_exposure([ent for ent in top_entities if ent < n_items])

    if getattr(args, "progress_verbose", True):
        avg_score = (
            float(np.mean([diversity_scores.get(ent, 0.0) for ent in top_entities]))
            if top_entities
            else 0.0
        )
        _role_log(
            f"[Role] role={state.descriptor.role_type.value} rule={state.descriptor.rule_key} "
            f"anchors={len(anchors)} candidates={len(candidate_items)} kept={len(top_entities)} "
            f"avg_score={avg_score:.4f}",
            log_handle,
        )

    return np.array(final_entities, dtype=np.int64), diversity_scores


def _compute_role_threshold(base_threshold: float, state, rsrg_trainer) -> float:
    rule = rsrg_trainer.role_manager.get_rule(state.descriptor.rule_key)
    if rule is None or not getattr(rule, "metrics", None):
        return base_threshold
    confidence = float(rule.metrics.get("confidence", 0.5))
    confidence = min(max(confidence, 0.0), 1.0)
    lift = float(rule.metrics.get("lift", 1.0))
    lift = min(max(lift, 0.1), 5.0)
    adjustment = (1.0 - 0.2 * (confidence - 0.5)) / (1.0 + 0.1 * (lift - 1.0))
    threshold = base_threshold * adjustment
    return float(min(max(threshold, 0.05), 0.95))


def _merge_entity_weights(
    existing_entities: np.ndarray,
    existing_weights: np.ndarray,
    new_entities: np.ndarray,
    new_scores: Dict[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    weight_map: Dict[int, float] = {}
    if existing_entities is not None and existing_entities.size > 0:
        for ent, weight in zip(existing_entities.tolist(), existing_weights.tolist()):
            weight_map[int(ent)] = float(weight)
    for ent in new_entities.tolist():
        weight_map[int(ent)] = max(weight_map.get(int(ent), 0.0), float(new_scores.get(int(ent), 0.0)))
    entities = np.array(sorted(weight_map.keys()), dtype=np.int64)
    weights = np.array([weight_map[e] for e in entities], dtype=np.float32)
    if weights.size > 0:
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        else:
            weights.fill(1.0 / weights.size)
    return entities, weights


def _build_role_entity_base_weights(
    entity_ids: np.ndarray,
    score_map: Dict[int, float],
    rsrg_trainer,
    n_items: int,
) -> np.ndarray:
    if entity_ids.size == 0:
        return np.empty((0,), dtype=np.float32)
    pop_dict = getattr(rsrg_trainer, "item_popularity", {}) if rsrg_trainer else {}
    max_pop = float(max(pop_dict.values())) if pop_dict else 0.0
    weights: List[float] = []
    for ent in entity_ids.tolist():
        ent = int(ent)
        base_score = float(score_map.get(ent, 0.0))
        if base_score <= 0.0:
            base_score = 0.05
        if pop_dict and ent < n_items and max_pop > 0.0:
            pop = float(pop_dict.get(ent, 0.0)) / max_pop
            pop_component = 1.0 - pop
        else:
            pop_component = 0.5
        combined = 0.7 * base_score + 0.3 * pop_component
        weights.append(max(combined, 1e-4))
    arr = np.asarray(weights, dtype=np.float32)
    total = float(arr.sum())
    if total > 0:
        arr = arr / total
    return arr


def _build_item_user_map(train_user_set: Mapping[int, Sequence[int]]) -> Dict[int, Set[int]]:
    mapping: Dict[int, Set[int]] = defaultdict(set)
    if not train_user_set:
        return mapping
    for user, items in train_user_set.items():
        try:
            iterator = iter(items)
        except TypeError:
            continue
        for item in items:
            mapping[int(item)].add(int(user))
    return mapping


def _summarize_role_users(
    entity_ids: Sequence[int],
    coverage_cap: Optional[int] = None,
    embed_cap: Optional[int] = None,
):
    if entity_ids is None:
        return set(), None, None
    try:
        ent_array = np.asarray(entity_ids, dtype=np.int64).reshape(-1)
    except Exception:
        ent_array = np.array(list(entity_ids), dtype=np.int64)
    counter: Counter = Counter()
    for ent in ent_array.tolist():
        users = role_item_user_map.get(int(ent))
        if not users:
            continue
        for uid in users:
            counter[uid] += 1
    if not counter:
        return set(), None, None
    if coverage_cap is not None and coverage_cap > 0:
        coverage_items = counter.most_common(coverage_cap)
    else:
        coverage_items = list(counter.items())
    coverage_set = set(uid for uid, _ in coverage_items)
    if embed_cap is not None and embed_cap > 0:
        embed_items = counter.most_common(embed_cap)
    else:
        embed_items = list(counter.items())
    user_ids = (
        np.array([uid for uid, _ in embed_items], dtype=np.int64) if embed_items else None
    )
    user_weights = (
        np.array([float(freq) for _, freq in embed_items], dtype=np.float32)
        if embed_items
        else None
    )
    return coverage_set, user_ids, user_weights


def _blend_edge_scores_with_kgc(
    model,
    edges: np.ndarray,
    base_weights: np.ndarray,
    device,
    mix: float = 0.5,
) -> np.ndarray:
    if edges.size == 0 or base_weights.size == 0 or mix <= 0.0:
        return base_weights
    tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.gcn.kgc(tensor, eval=True, cf_train=True)
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().reshape(-1)
        else:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size != base_weights.size:
        return base_weights
    scores = scores.astype(np.float32)
    min_s = float(scores.min(initial=0.0))
    max_s = float(scores.max(initial=1.0))
    if max_s - min_s > 1e-6:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores.fill(0.5)
    mix = max(0.0, min(1.0, mix))
    blended = mix * base_weights + (1.0 - mix) * scores
    total = float(blended.sum())
    if total > 0:
        blended = blended / total
    return blended

def _filter_role_edges_by_score(
    model,
    edges: np.ndarray,
    device,
    threshold: float,
    top_ratio: float,
    min_keep: int,
    verbose: bool,
    label: str,
    base_weights: np.ndarray = None,
    auto_threshold: bool = False,
    threshold_quantile: float = 0.6,
    threshold_min: float = 0.05,
    log_handle=None,
):
    if edges.size == 0:
        empty_stats = {"total": 0, "kept": 0, "avg_score": 0.0, "fallback": False}
        return edges, empty_stats, np.empty((0,), dtype=np.float32), None
    edges = np.asarray(edges, dtype=np.int64)
    base_arr = None
    if base_weights is not None:
        base_arr = np.asarray(base_weights, dtype=np.float32).reshape(-1)
        if base_arr.shape[0] < edges.shape[0]:
            pad = edges.shape[0] - base_arr.shape[0]
            base_arr = np.pad(base_arr, (0, pad), mode="edge")
    if edges.ndim != 2 or edges.shape[1] < 3:
        empty_stats = {"total": 0, "kept": 0, "avg_score": 0.0, "fallback": False}
        return edges, empty_stats, np.empty((0,), dtype=np.float32), base_arr
    _, unique_idx = np.unique(edges, axis=0, return_index=True)
    order = np.sort(unique_idx)
    edges = edges[order]
    if base_arr is not None:
        base_arr = base_arr[order]
    # Proactively grow model capacity when new role nodes/relations appear.
    if edges.ndim == 2 and edges.shape[1] >= 3:
        try:
            max_rel = int(edges[:, 1].max())
        except ValueError:
            max_rel = -1
        if max_rel >= 0 and hasattr(model, "ensure_relation_capacity"):
            current_rel = int(getattr(model, "n_relations", max_rel + 1))
            if max_rel >= current_rel:
                model.ensure_relation_capacity(max_rel + 1)
        try:
            max_ent = int(edges[:, [0, 2]].max())
        except ValueError:
            max_ent = -1
        if max_ent >= 0 and hasattr(model, "ensure_entity_capacity"):
            current_ent = int(getattr(model, "n_entities", max_ent + 1))
            if max_ent >= current_ent:
                model.ensure_entity_capacity(max_ent + 1)
        # Guard against any negative or still-out-of-range indices that would break CUDA kernels.
        valid_mask = np.ones(edges.shape[0], dtype=bool)
        ent_limit_attr = getattr(model, "n_entities", None)
        if ent_limit_attr is None and hasattr(model, "gcn") and hasattr(model.gcn, "n_entity"):
            ent_limit_attr = getattr(model.gcn, "n_entity", None)
        if ent_limit_attr is not None:
            ent_limit = int(ent_limit_attr)
            ent_mask = (edges[:, 0] >= 0) & (edges[:, 0] < ent_limit) & (edges[:, 2] >= 0) & (edges[:, 2] < ent_limit)
            valid_mask &= ent_mask
        rel_limit_attr = getattr(model, "n_relations", None)
        if rel_limit_attr is None and hasattr(model, "gcn") and hasattr(model.gcn, "n_relations"):
            rel_limit_attr = getattr(model.gcn, "n_relations", None)
        if rel_limit_attr is not None:
            rel_limit = int(rel_limit_attr)
            rel_mask = (edges[:, 1] >= 0) & (edges[:, 1] < rel_limit)
            valid_mask &= rel_mask
        dropped = int(edges.shape[0] - valid_mask.sum())
        if dropped > 0:
            edges = edges[valid_mask]
            if base_arr is not None and base_arr.shape[0] >= valid_mask.size:
                base_arr = base_arr[valid_mask]
            if verbose and edges.size > 0:
                _role_log(f"[Role] edge-filter {label} dropped_invalid={dropped}", log_handle)
    if edges.size == 0:
        empty_stats = {"total": 0, "kept": 0, "avg_score": 0.0, "fallback": False}
        return edges, empty_stats, np.empty((0,), dtype=np.float32), None
    tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model.gcn.kgc(tensor, eval=True, cf_train=True).squeeze(-1)
    scores_cpu = scores.detach().cpu().numpy()
    order = np.argsort(-scores_cpu)
    eff_threshold = float(threshold)
    if auto_threshold and scores_cpu.size > 0:
        q = min(max(threshold_quantile, 0.0), 1.0)
        if q >= 1.0:
            quant_val = float(scores_cpu.max())
        elif q <= 0.0:
            quant_val = float(scores_cpu.min())
        else:
            quant_val = float(np.quantile(scores_cpu, q))
        eff_threshold = max(threshold_min, min(eff_threshold, quant_val))
    keep_mask = scores_cpu >= eff_threshold
    kept_edges = edges[keep_mask]
    kept_scores = scores_cpu[keep_mask]
    kept_base = base_arr[keep_mask] if base_arr is not None else None
    fallback = False
    if kept_edges.size == 0:
        fallback = True
        ratio = max(top_ratio, 0.0)
        topk = max(min_keep, int(math.ceil(edges.shape[0] * ratio)))
        topk = min(edges.shape[0], topk)
        select_idx = order[:topk]
        kept_edges = edges[select_idx]
        kept_scores = scores_cpu[select_idx]
        if base_arr is not None:
            kept_base = base_arr[select_idx]
    elif 0.0 < top_ratio < 1.0 and kept_edges.shape[0] > min_keep:
        topk = max(min_keep, int(math.ceil(kept_edges.shape[0] * top_ratio)))
        order_idx = np.argsort(-kept_scores)
        select_idx = order_idx[:topk]
        kept_edges = kept_edges[select_idx]
        kept_scores = kept_scores[select_idx]
        if kept_base is not None:
            kept_base = kept_base[select_idx]
    stats = {
        "total": int(edges.shape[0]),
        "kept": int(kept_edges.shape[0]),
        "avg_score": float(kept_scores.mean()) if kept_edges.size else 0.0,
        "fallback": fallback,
        "threshold": float(eff_threshold),
    }
    if verbose:
        tag = ' fallback' if fallback else ''
        _role_log(
            f"[Role] edge-filter {label} total={stats['total']} kept={stats['kept']} "
            f"thr={eff_threshold:.3f} avg={stats['avg_score']:.3f}{tag}",
            log_handle,
        )
    return kept_edges, stats, kept_scores, kept_base

import torch.nn.functional as F


def _compose_edge_gate(
    scores: np.ndarray,
    base_weights: Optional[np.ndarray],
) -> np.ndarray:
    if scores is None or scores.size == 0:
        return np.empty((0,), dtype=np.float32)
    gate = np.clip(scores.astype(np.float32), 1e-4, 1.0)
    if base_weights is not None:
        weights = np.clip(base_weights.astype(np.float32), 1e-4, 10.0)
        if weights.size == gate.size:
            gate *= weights
        elif weights.size > gate.size:
            gate *= weights[: gate.size]
        else:
            gate[: weights.size] *= weights
    return gate.astype(np.float32)


def _apply_gate_temperature(arr: Optional[np.ndarray], temperature: float) -> Optional[np.ndarray]:
    if arr is None or temperature <= 1.0:
        return arr
    if arr.size == 0:
        return arr
    scaled = np.clip(arr.astype(np.float32, copy=False), 1e-6, 1.0)
    scaled = np.power(scaled, 1.0 / max(1e-6, temperature))
    max_val = float(scaled.max())
    if max_val > 0:
        scaled /= max_val
    return scaled.astype(np.float32, copy=False)


def _evaluate_role_prescore(model, candidate: Dict[str, Any], args) -> Tuple[float, Dict[str, float]]:
    if not bool(getattr(args, "role_prescore_enable", True)):
        base_score = float(candidate.get("score", 0.0))
        base_score = max(0.0, min(1.0, base_score))
        return base_score, {"kgc": base_score, "cos": 0.5, "reward": 0.5}
    alpha = float(getattr(args, "role_prescore_alpha", 0.6))
    beta = float(getattr(args, "role_prescore_beta", 0.3))
    gamma = float(getattr(args, "role_prescore_gamma", 0.15))
    alpha = min(max(alpha, 0.0), 1.0)
    beta = min(max(beta, 0.0), 1.0)
    gamma = min(max(gamma, 0.0), 1.0)
    norm = alpha + beta + gamma
    if norm <= 0:
        alpha, beta, gamma = 1.0, 0.0, 0.0
        norm = 1.0
    alpha /= norm
    beta /= norm
    gamma /= norm
    kgc_score = float(candidate.get("score", 0.0))
    kgc_score = max(0.0, min(1.0, kgc_score))
    reward_raw = float(candidate.get("role_reward", 0.0))
    reward_norm = 0.5 + 0.5 * math.tanh(reward_raw)
    cos_sim = 0.5
    embed = getattr(model, "all_embed", None)
    if embed is not None and hasattr(embed, "data"):
        try:
            with torch.no_grad():
                rid = int(candidate.get("rid", -1))
                if 0 <= rid < embed.size(0):
                    role_vec = embed.data[rid].detach()
                    ent_ids = candidate.get("entities")
                    if ent_ids is not None:
                        ent_arr = np.asarray(ent_ids, dtype=np.int64).reshape(-1)
                        if ent_arr.size > 0:
                            sample = ent_arr[: min(ent_arr.shape[0], 20)]
                            ent_tensor = torch.as_tensor(sample, dtype=torch.long, device=embed.device)
                            mask = (ent_tensor >= 0) & (ent_tensor < embed.size(0))
                            ent_tensor = ent_tensor[mask]
                            if ent_tensor.numel() > 0:
                                ent_vec = embed.index_select(0, ent_tensor).detach()
                                role_vec_exp = role_vec.unsqueeze(0).expand_as(ent_vec)
                                cos_vals = F.cosine_similarity(role_vec_exp, ent_vec, dim=-1)
                                cos_sim = float((cos_vals.mean().clamp(-1, 1) + 1.0) * 0.5)
        except Exception:
            cos_sim = 0.5
    cos_div = 1.0 - cos_sim
    cos_div = max(0.0, min(1.0, cos_div))
    prescore = alpha * kgc_score + beta * cos_div + gamma * reward_norm
    prescore = max(0.0, min(1.0, prescore))
    return prescore, {"kgc": kgc_score, "cos_sim": cos_sim, "cos_div": cos_div, "reward": reward_norm}


def _select_retained_edges(
    edges: np.ndarray,
    gate: Optional[np.ndarray],
    top_ratio: float,
    min_score: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if edges is None or edges.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    if gate is None:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    gate_arr = np.asarray(gate, dtype=np.float32).reshape(-1)
    if gate_arr.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    min_score = float(min_score)
    mask = gate_arr >= min_score
    selected_idx = np.where(mask)[0]
    ratio = min(max(top_ratio, 0.0), 1.0)
    if ratio > 0.0:
        topk = max(1, int(math.ceil(gate_arr.size * ratio)))
        order = np.argsort(-gate_arr)
        top_idx = order[:topk]
        selected_idx = np.unique(np.concatenate([selected_idx, top_idx])) if selected_idx.size else top_idx
    if selected_idx.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    selected_edges = edges[selected_idx]
    selected_gate = gate_arr[selected_idx]
    return selected_edges.astype(np.int64, copy=False), selected_gate.astype(np.float32, copy=False)


def _store_retained_edges(
    store: Dict[int, Dict[str, Any]],
    rid: int,
    role_type_id: int,
    entity_edges: Optional[np.ndarray],
    entity_gate: Optional[np.ndarray],
    relation_edges: Optional[np.ndarray],
    relation_gate: Optional[np.ndarray],
    top_ratio: float,
    min_score: float,
    ttl: int,
    rule_key: Optional[Tuple[Tuple[int, ...], int]] = None,
    position: Optional[int] = None,
) -> None:
    if ttl <= 0:
        return
    if top_ratio <= 0.0 and min_score <= 0.0:
        return
    updated = False
    quality_hint = 0.0
    if entity_edges is not None and entity_gate is not None:
        entity_edges = np.asarray(entity_edges)
        entity_gate = np.asarray(entity_gate)
        if entity_edges.size > 0 and entity_gate.size == entity_edges.shape[0]:
            kept_edges, kept_gate = _select_retained_edges(entity_edges, entity_gate, top_ratio, min_score)
            if kept_edges.size > 0:
                entry = store.get(rid, {"role_type": role_type_id})
                entry["entity_edges"] = kept_edges.copy()
                entry["entity_gate"] = kept_gate.copy()
                entry["role_type"] = role_type_id
                entry["ttl"] = ttl
                if rule_key is not None:
                    entry["rule_key"] = rule_key
                if position is not None:
                    entry["position"] = int(position)
                if kept_gate.size > 0:
                    quality_hint = max(quality_hint, float(np.mean(kept_gate)))
                store[rid] = entry
                updated = True
    if relation_edges is not None and relation_gate is not None:
        relation_edges = np.asarray(relation_edges)
        relation_gate = np.asarray(relation_gate)
        if relation_edges.size > 0 and relation_gate.size == relation_edges.shape[0]:
            kept_edges, kept_gate = _select_retained_edges(relation_edges, relation_gate, top_ratio, min_score)
            if kept_edges.size > 0:
                entry = store.get(rid, {"role_type": role_type_id})
                entry["relation_edges"] = kept_edges.copy()
                entry["relation_gate"] = kept_gate.copy()
                entry["role_type"] = role_type_id
                entry["ttl"] = ttl
                if rule_key is not None:
                    entry["rule_key"] = rule_key
                if position is not None:
                    entry["position"] = int(position)
                if kept_gate.size > 0:
                    quality_hint = max(quality_hint, float(np.mean(kept_gate)))
                store[rid] = entry
                updated = True
    if updated and rid in store:
        store[rid]["quality"] = quality_hint
        store[rid]["ttl"] = max(1, store[rid].get("ttl", ttl))


def _round_robin_select_edges(
    dynamic_edges: np.ndarray,
    gate_arr: Optional[np.ndarray],
    role_arr: Optional[np.ndarray],
    reward_arr: Optional[np.ndarray],
    metadata: Optional[List[Dict]],
    fraction: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if dynamic_edges.size == 0 or fraction >= 1.0 or not metadata:
        return dynamic_edges, gate_arr, role_arr, reward_arr
    total_edges = int(dynamic_edges.shape[0])
    target = max(1, int(math.ceil(total_edges * max(0.0, fraction))))
    segments = []
    offset = 0
    for entry in metadata:
        count = int(entry.get("num_edges", 0))
        end = offset + max(0, count)
        if count > 0:
            segments.append(
                {
                    "edges": dynamic_edges[offset:end],
                    "gate": None if gate_arr is None else gate_arr[offset:end],
                    "role": None if role_arr is None else role_arr[offset:end],
                    "reward": None if reward_arr is None else reward_arr[offset:end],
                    "cursor": 0,
                }
            )
        offset = end
    if not segments:
        cap = min(target, total_edges)
        indices = slice(0, cap)
        return (
            dynamic_edges[indices],
            None if gate_arr is None else gate_arr[indices],
            None if role_arr is None else role_arr[indices],
            None if reward_arr is None else reward_arr[indices],
        )
    collected_edges: List[np.ndarray] = []
    collected_gate: Optional[List[np.ndarray]] = [] if gate_arr is not None else None
    collected_role: Optional[List[np.ndarray]] = [] if role_arr is not None else None
    collected_reward: Optional[List[np.ndarray]] = [] if reward_arr is not None else None
    while len(collected_edges) < target:
        progressed = False
        for segment in segments:
            cursor = segment["cursor"]
            if cursor >= segment["edges"].shape[0]:
                continue
            segment["cursor"] += 1
            collected_edges.append(segment["edges"][cursor])
            if collected_gate is not None:
                collected_gate.append(segment["gate"][cursor])
            if collected_role is not None:
                collected_role.append(segment["role"][cursor])
            if collected_reward is not None:
                collected_reward.append(segment["reward"][cursor])
            progressed = True
            if len(collected_edges) >= target:
                break
        if not progressed:
            break
    if not collected_edges:
        cap = min(target, total_edges)
        indices = slice(0, cap)
        return (
            dynamic_edges[indices],
            None if gate_arr is None else gate_arr[indices],
            None if role_arr is None else role_arr[indices],
            None if reward_arr is None else reward_arr[indices],
        )
    edges_out = np.stack(collected_edges, axis=0).astype(dynamic_edges.dtype, copy=False)
    gate_out = (
        None
        if collected_gate is None
        else np.asarray(collected_gate, dtype=np.float32)
    )
    role_out = (
        None
        if collected_role is None
        else np.asarray(collected_role, dtype=np.int64)
    )
    reward_out = (
        None
        if collected_reward is None
        else np.asarray(collected_reward, dtype=np.float32)
    )
    return edges_out, gate_out, role_out, reward_out


def _update_role_mix_slopes(model, diversity_scores: Dict[int, float], momentum: float = 0.5) -> None:
    if not diversity_scores:
        return
    if not hasattr(model, "gcn") or not hasattr(model.gcn, "update_role_mix"):
        return
    model.gcn.update_role_mix(diversity_scores, momentum=momentum)


def _adapt_role_edges(
    model,
    edge_optimizer,
    edges_tensor: torch.Tensor,
    args,
    device,
    override_steps: Optional[int] = None,
    gate_tensor: Optional[torch.Tensor] = None,
):
    steps = override_steps if override_steps is not None else max(0, int(getattr(args, "role_ft_steps", 0)))
    if steps <= 0 or edges_tensor.numel() == 0:
        return
    batch_size = max(1, int(getattr(args, "role_ft_batch", 256)))
    edges_tensor = edges_tensor.to(device=device, dtype=torch.long, non_blocking=True)
    total = edges_tensor.size(0)
    gate_tensor_device = None
    if gate_tensor is not None:
        gate_tensor_device = torch.as_tensor(gate_tensor, dtype=torch.float32, device=device).view(-1)
        if gate_tensor_device.size(0) > total:
            gate_tensor_device = gate_tensor_device[:total]
        elif gate_tensor_device.size(0) < total:
            padded = torch.ones(total, dtype=gate_tensor_device.dtype, device=device)
            padded[: gate_tensor_device.size(0)] = gate_tensor_device
            gate_tensor_device = padded
    neg_per_pos = max(1, int(getattr(args, "role_neg_per_pos", 1)))
    corrupt_ratio = float(getattr(args, "role_neg_corrupt_ratio", 0.5))
    corrupt_ratio = min(max(corrupt_ratio, 0.0), 1.0)
    updated_rel_ids = set()
    for _ in range(steps):
        if total <= batch_size:
            batch = edges_tensor
            gate_batch = gate_tensor_device
        else:
            idx = torch.randint(0, total, (batch_size,), device=device)
            batch = edges_tensor[idx]
            gate_batch = gate_tensor_device[idx] if gate_tensor_device is not None else None
        updated_rel_ids.update(batch[:, 1].detach().tolist())
        pos_scores = model.gcn.kgc(batch, eval=True, cf_train=True).squeeze(-1)
        if gate_batch is not None:
            pos_targets = gate_batch.clamp(1e-4, 1.0).to(device=pos_scores.device, dtype=pos_scores.dtype)
        else:
            pos_targets = torch.ones_like(pos_scores)

        neg_edges = batch.unsqueeze(1).repeat(1, neg_per_pos, 1).view(-1, batch.size(1))
        num_neg = neg_edges.size(0)
        if num_neg == 0:
            continue
        head_mask = torch.zeros(num_neg, dtype=torch.bool, device=device)
        if corrupt_ratio >= 1.0:
            head_mask.fill_(True)
        elif corrupt_ratio <= 0.0:
            head_mask.fill_(False)
        else:
            n_head = int(round(num_neg * corrupt_ratio))
            if n_head > 0:
                head_mask[:n_head] = True
                head_mask = head_mask[torch.randperm(num_neg, device=device)]
        tail_mask = ~head_mask

        if head_mask.any():
            random_heads = torch.randint(0, model.n_entities, (head_mask.sum(),), device=device)
            same_head = random_heads == neg_edges[head_mask, 0]
            if same_head.any():
                random_heads[same_head] = (random_heads[same_head] + 1) % model.n_entities
            neg_edges[head_mask, 0] = random_heads
        if tail_mask.any():
            random_tails = torch.randint(0, model.n_entities, (tail_mask.sum(),), device=device)
            same_tail = random_tails == neg_edges[tail_mask, 2]
            if same_tail.any():
                random_tails[same_tail] = (random_tails[same_tail] + 1) % model.n_entities
            neg_edges[tail_mask, 2] = random_tails

        neg_scores = model.gcn.kgc(neg_edges, eval=True, cf_train=True).squeeze(-1)
        neg_targets = torch.zeros_like(neg_scores)

        loss_pos = F.binary_cross_entropy(pos_scores, pos_targets)
        loss_neg = F.binary_cross_entropy(neg_scores, neg_targets)
        loss = loss_pos + loss_neg
        edge_optimizer.zero_grad()
        loss.backward()
        edge_optimizer.step()
    if updated_rel_ids and hasattr(model, "gcn") and hasattr(model.gcn, "sync_role_relation_weights"):
        model.gcn.sync_role_relation_weights(relation_ids=list(updated_rel_ids))


if __name__ == '__main__':
    """fix the random seed"""
    # seed = 1998
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
  
    """read args"""
    global args, device, train_user_set,kg_dict, item_lists_dict, ent_lists_dict
    args = parse_args()
    # Device selection with optional auto-pick
    if args.cuda and torch.cuda.is_available():
        gpu_id = int(getattr(args, 'gpu_id', 0))
        if gpu_id < 0:
            try:
                ndev = torch.cuda.device_count()
                best_id = 0
                best_free = -1
                for i in range(ndev):
                    free, total = torch.cuda.mem_get_info(i)
                    if free > best_free:
                        best_free = free
                        best_id = i
                gpu_id = best_id
            except Exception:
                gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        try:
            name = torch.cuda.get_device_name(device)
            if getattr(args, 'progress_verbose', True):
                print(f"[Device] 使用CUDA GPU: id={gpu_id}, name={name}")
        except Exception:
            pass
    else:
        if args.cuda and not torch.cuda.is_available() and getattr(args, 'progress_verbose', True):
            print("[Device] CUDA 不可用，使用CPU")
        device = torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, ui_sparse_graph, all_sparse_graph, item_rel_mask, triplets, kg_dict = load_data(args)
    # item_pmi_dict = pkl.load(open("item_pair_pmi.pkl","rb"))
    item_pmi_dict = _cal_npmi(user_dict['train_user_set'])
    # Persist NPMI cache under the dataset folder instead of CWD
    npmi_path = os.path.join(args.data_path, args.dataset, "item_pair_pmi.pkl")
    if getattr(args, 'progress_verbose', True):
        print(f"[Phase] 保存NPMI缓存 → {npmi_path}")
    pkl.dump(item_pmi_dict, open(npmi_path, "wb"))
    if getattr(args, 'progress_verbose', True):
        print(f"[Phase] NPMI缓存完成，大小≈{len(item_pmi_dict):,} 对")
    
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_entities_real = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    train_user_set = user_dict['train_user_set']
    item_popularity_counter = Counter()
    for items in train_user_set.values():
        for item in items:
            item_popularity_counter[int(item)] += 1
    role_item_user_map.clear()
    role_item_user_map.update(_build_item_user_map(train_user_set))

    rsrg_trainer = None
    # For entity-mode injection with per-rule relations
    role_entity_next_id = n_entities  # next entity id for role nodes
    rule_relation_map = {}  # ((body_relations, head_relation), position) -> (src_rel_id, dst_rel_id)
    rule_rel_next_id = n_relations
    req_workers = int(getattr(args, "neg_pool_workers", 0))
    if req_workers <= 0:
        NEG_POOL_WORKERS = cores
    else:
        NEG_POOL_WORKERS = max(1, min(cores, req_workers))
    force_threads = bool(getattr(args, "neg_pool_use_threads", False))
    if force_threads:
        NEG_POOL_USE_THREADS = True
    else:
        NEG_POOL_USE_THREADS = platform.system().lower().startswith("win")
    # Determine whether to enable role training (new flag) with backward compatibility
    enable_role_training = getattr(args, "enable_role_training", None)
    if enable_role_training is None:
        old_flag = getattr(args, "enable_dynamic_rules", None)
        enable_role_training = True if old_flag is None else bool(old_flag)
    if enable_role_training:
        reward_weights = _parse_reward_weights(getattr(args, "rule_reward_weights", ""))
        reward_weights.setdefault("kgc_acc", 0.2)
        reward_weights.setdefault("kgc_loss", -0.2)
        rules_dir = os.path.join(args.data_path, args.dataset, "rules")
        usage_log_path = os.path.join(rules_dir, f"rule_usage_history_{args.dataset}.jsonl")
        miner_config = RuleMinerConfig(
            max_length=args.rule_max_length,
            min_support=args.rule_min_support,
            min_confidence=args.rule_min_confidence,
            min_pca_confidence=args.rule_min_pca_confidence,
            topk_per_length=args.rule_topk_per_length,
            max_body_evaluations=args.rule_max_body_evaluations,
            sample_bodies=args.rule_sample_bodies,
            body_sample_rate=args.rule_body_sample_rate,
            use_gpu=args.rule_gpu_mining,
            gpu_entity_threshold=args.rule_gpu_entity_threshold,
            use_torch_sparse=args.rule_use_torch_sparse,
        )
        filter_config = RuleFilterConfig(
            min_lift=args.rule_min_lift,
            min_conviction=args.rule_min_conviction,
            max_rules=args.rule_max_rules,
        )
        dynamic_config = DynamicRuleConfig(
            rules_dir=rules_dir,
            usage_log=usage_log_path,
            refresh_interval=args.rule_refresh_interval,
            max_roles=args.rule_max_roles,
            max_edges_per_role=args.rule_edges_per_role,
            reward_weights=reward_weights,
            force_rebuild=args.rule_force_rebuild,
        )
        rsrg_trainer = DynamicRuleTrainer(
            triplets,
            n_relations,
            miner_config,
            filter_config,
            dynamic_config,
        )
        rsrg_trainer.bootstrap()
        rsrg_trainer.set_item_popularity(item_popularity_counter)
        rule_rel_next_id = _initial_role_bridge_relation(rsrg_trainer, n_relations)
        n_relations = max(n_relations, rule_rel_next_id)
        role_channel_synced = False

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    """define model"""
    model = RoSAKGRecommender(n_params, args, graph, ui_sparse_graph, item_rel_mask).to(device)
    # KGR_model = KGR(n_items, n_entities,n_relations,dim=256, p_norm=1,margin=1.).to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = PCGrad(optimizer)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step,gamma=args.lr_dc)  
    role_params: List[torch.nn.Parameter] = []
    _seen_params: Set[int] = set()

    def _add_role_param(param: Optional[torch.nn.Parameter]) -> None:
        if param is None or not isinstance(param, torch.nn.Parameter):
            return
        if not param.requires_grad:
            return
        pid = id(param)
        if pid in _seen_params:
            return
        _seen_params.add(pid)
        role_params.append(param)

    for p in model.gcn.kgc.parameters():
        _add_role_param(p)
    _add_role_param(model.gcn.n_relation_weight)
    for p in model.gcn.role_bridge_linear.parameters():
        _add_role_param(p)
    _add_role_param(model.gcn.role_type_gate)
    _add_role_param(model.gcn.reward_gate)
    _add_role_param(model.gcn.degree_gate)
    _add_role_param(getattr(model, "all_embed", None))

    if not role_params:
        role_params = list(model.parameters())
    kgr_optimizer = torch.optim.AdamW(role_params, lr=0.001)
    model.register_optimizer(optimizer.optimizer)
    model.register_optimizer(kgr_optimizer)
    
    cur_best = 0
    stopping_step = 0
    should_stop = False
    tracked_metrics = ["recall", "ndcg", "precision", "hit_ratio",
                       "kgat_recall", "kgat_precision", "kgat_ndcg", "kgat_hit_ratio", "kgat_mrr",
                       "kgat_coverage", "kgat_ad", "kgat_arp", "kgat_md", "kgat_mp", "kgat_hit", "kgat_ad2"]
    minimize_metrics = {"kgat_arp", "kgat_mp"}
    best_metric = {
        k: (float("inf") if k in minimize_metrics else float("-inf"))
        for k in tracked_metrics
    }
    best_epoch = {k: 0 for k in tracked_metrics}
    last_kgr_stats: Dict[str, float] = {"kgc_acc": 0.0, "kgc_loss": 0.0}
    recall_history: List[Tuple[int, float]] = []
    recall_history_cap = max(0, int(getattr(args, "role_metric_history", 200)))
    recall_drop_tolerance = float(getattr(args, "role_recall_drop_tolerance", 1e-5))
    recall_drop_tolerance = max(0.0, recall_drop_tolerance)
    recall_resume_tolerance = float(
        getattr(args, "role_recall_resume_tolerance", recall_drop_tolerance)
    )
    recall_resume_tolerance = max(0.0, recall_resume_tolerance)
    pause_new_roles = False
    pending_new_role_scales: Deque[float] = deque()
    new_role_pause_reason = ""
    role_penalties: Dict[int, Dict[str, int]] = {}
    last_injection_epoch: Optional[int] = None
    last_injection_new_roles: List[Dict[str, Any]] = []
    last_rollback_epoch: Optional[int] = None
    last_minor_rollback_epoch: Optional[int] = None
    recall_decline_since_injection = 0
    role_rollback_keep_ratio = float(getattr(args, "role_rollback_keep_ratio", 0.2))
    role_rollback_keep_ratio = min(max(role_rollback_keep_ratio, 0.0), 1.0)
    role_minor_rollback_keep_ratio = float(getattr(args, "role_minor_rollback_keep_ratio", 0.7))
    role_minor_rollback_keep_ratio = min(max(role_minor_rollback_keep_ratio, 0.0), 1.0)
    role_recall_decline_trigger = max(1, int(getattr(args, "role_recall_decline_trigger", 2)))
    
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)
    cl_batch = 512
    cl_iter = math.ceil(n_items / cl_batch)
    item_embs = []
    KG_mask = []
    persistent_role_cache: Dict[Tuple[Tuple[int, ...], int, int], Dict[str, np.ndarray]] = {}
    retained_role_edges: Dict[int, Dict[str, Any]] = {}
    role_active_rounds: Dict[int, int] = {}

    def _apply_role_penalty(
        rid: int,
        cache_key: Optional[Tuple[Tuple[int, ...], int, int]],
        score: float,
    ) -> None:
        pen = role_penalties.setdefault(rid, {"count": 0, "cooldown": 0})
        pen["count"] += 1
        required_cooldown = max(0, pen["count"] - 1)
        pen["cooldown"] = max(pen.get("cooldown", 0), required_cooldown)
        if cache_key is not None:
            persistent_role_cache.pop(cache_key, None)
        retained_role_edges.pop(rid, None)
        role_active_rounds.pop(rid, None)

    def _rollback_recent_new_roles(
        trigger_epoch: int,
        injection_records: List[Dict[str, Any]],
        keep_ratio: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        total = len(injection_records)
        if total == 0:
            return injection_records, 0
        ratio = role_rollback_keep_ratio if keep_ratio is None else min(max(keep_ratio, 0.0), 1.0)
        keep_count = max(1, int(math.ceil(total * ratio)))
        remove_count = max(0, total - keep_count)
        if remove_count <= 0:
            return injection_records, 0
        sorted_new = sorted(injection_records, key=lambda x: x.get("score", 0.0))
        victims = sorted_new[:remove_count]
        survivors = sorted_new[remove_count:]
        removed_ids = []
        for rec in victims:
            rid = rec["rid"]
            cache_key = rec.get("cache_key")
            score = rec.get("score", 0.0)
            _apply_role_penalty(rid, cache_key, score)
            removed_ids.append(rid)
        if getattr(args, 'progress_verbose', True):
            _role_log(
                f"[Role] rollback {len(removed_ids)} new roles from epoch {trigger_epoch} "
                f"(kept {len(survivors)}/{total})"
            )
        return survivors, len(removed_ids)
    
    refresh_interval = max(1, int(getattr(args, 'rule_refresh_interval', 3)))
    role_channel_synced = False
    select_frac_init = float(getattr(args, 'role_first_fraction', 0.08))
    select_frac_step = float(getattr(args, 'role_fraction_step', 0.04))
    entity_frac_init = float(getattr(args, 'role_entity_fraction', 0.03))
    entity_frac_step = float(getattr(args, 'role_entity_fraction_step', 0.008))
    role_fraction_learned_weight = float(getattr(args, 'role_fraction_learned_weight', 0.4))
    role_fraction_learned_weight = min(max(role_fraction_learned_weight, 0.0), 1.0)
    role_fraction_lr = float(getattr(args, 'role_fraction_lr', 0.35))
    role_entity_lr = float(getattr(args, 'role_entity_lr', 0.25))
    role_entity_frac_cap = float(getattr(args, 'role_entity_fraction_cap', 0.18))
    role_entity_frac_floor = float(getattr(args, 'role_entity_fraction_floor', 0.0))
    role_entity_reward_gain = float(getattr(args, 'role_entity_reward_gain', 0.6))
    role_entity_reward_decay = float(getattr(args, 'role_entity_reward_decay', 0.4))
    role_entity_gate_temp = float(getattr(args, 'role_entity_gate_temperature', 1.0))
    role_relation_gate_temp = float(getattr(args, 'role_relation_gate_temperature', 1.0))
    role_edge_auto_threshold = bool(getattr(args, 'role_edge_auto_threshold', True))
    role_edge_keep_quantile = float(getattr(args, 'role_edge_keep_quantile', 0.6))
    role_edge_min_threshold = float(getattr(args, 'role_edge_min_threshold', 0.05))
    role_mix_reward_gain = float(getattr(args, 'role_mix_reward_gain', 0.3))
    role_mix_floor = float(getattr(args, 'role_mix_floor', 0.0))
    role_retention_top_ratio = float(getattr(args, 'role_retention_top_ratio', 0.25))
    role_retention_min_score = float(getattr(args, 'role_retention_min_score', 0.65))
    role_retention_ttl = max(0, int(getattr(args, 'role_retention_ttl', 2)))
    role_item_mix_momentum = float(getattr(args, 'role_item_mix_momentum', 0.55))
    role_item_mix_momentum = min(max(role_item_mix_momentum, 0.0), 0.999)
    role_min_active = max(1, int(getattr(args, 'role_min_active', 1)))
    role_round_robin_fraction = float(getattr(args, 'role_round_robin_fraction', 1.0))
    role_round_robin_fraction = min(max(role_round_robin_fraction, 0.0), 1.0)
    role_bridge_lr = float(getattr(args, 'role_bridge_lr', getattr(model, 'role_bridge_lr', 0.15)))
    role_bridge_decay = float(getattr(args, 'role_bridge_decay', getattr(model, 'role_bridge_decay', 0.1)))
    role_phase_ratio_default = float(getattr(args, "role_inject_phase_ratio", 1.0))
    role_phase_ratio_default = min(max(role_phase_ratio_default, 0.0), 1.0)
    role_selection_floor = float(getattr(args, "role_selection_floor", 0.0))
    role_selection_floor = min(max(role_selection_floor, 0.0), 1.0)
    role_selection_cap = float(getattr(args, "role_selection_cap", 0.6))
    role_selection_cap = min(max(role_selection_cap, 1e-3), 1.0)
    role_negative_backoff = float(getattr(args, "role_negative_backoff", 0.4))
    role_negative_backoff = min(max(role_negative_backoff, 0.0), 1.0)
    role_metric_drop_threshold = max(0.0, float(getattr(args, "role_metric_drop_threshold", 0.0)))
    role_reward_floor = max(0.0, float(getattr(args, "role_reward_floor", 0.0)))
    role_prescore_enabled = bool(getattr(args, "role_prescore_enable", True))
    role_prescore_blend = float(getattr(args, "role_prescore_blend", 0.45))
    role_prescore_blend = min(max(role_prescore_blend, 0.0), 1.0)
    role_prescore_skip = float(getattr(args, "role_prescore_skip", 0.25))
    role_prescore_skip = min(max(role_prescore_skip, 0.0), 1.0)
    role_prescore_gate_floor = float(getattr(args, "role_prescore_gate_floor", 0.35))
    role_prescore_gate_floor = min(max(role_prescore_gate_floor, 0.0), 1.0)
    role_prescore_role_cap = float(getattr(args, "role_prescore_role_cap", 0.4))
    role_prescore_role_cap = min(max(role_prescore_role_cap, 0.0), 1.0)
    role_user_coverage_cap = max(0, int(getattr(args, "role_user_coverage_cap", 512)))
    role_user_embed_cap = max(0, int(getattr(args, "role_user_embed_cap", 128)))
    role_user_overlap_cap = max(0, int(getattr(args, "role_user_overlap_cap", 4096)))
    role_redundancy_weight = float(getattr(args, "role_redundancy_weight", 0.4))
    role_redundancy_weight = min(max(role_redundancy_weight, 0.0), 1.0)
    role_complementary_weight = float(getattr(args, "role_complementary_weight", 0.25))
    role_complementary_weight = min(max(role_complementary_weight, 0.0), 1.0)
    role_log_path_arg = getattr(args, "role_log_file", "")
    if role_log_path_arg is None:
        role_log_path_arg = ""
    role_log_default = os.path.join(
        "result",
        f"{args.dataset}_role_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    role_log_path_resolved = role_log_path_arg.strip() or role_log_default
    role_log_fp_global = None
    _original_stdout = sys.stdout
    _stdout_tee = None
    role_logging_enabled = False
    if role_log_path_resolved:
        log_dir = os.path.dirname(role_log_path_resolved)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        role_log_fp_global = open(role_log_path_resolved, "a", encoding="utf-8")
        _stdout_tee = _TeeStdout(sys.stdout, role_log_fp_global)
        sys.stdout = _stdout_tee
        role_logging_enabled = True

    def _close_role_log_streams():
        global role_log_fp_global, _stdout_tee
        if _stdout_tee is not None:
            sys.stdout = _original_stdout
            _stdout_tee = None
        if role_log_fp_global is not None:
            role_log_fp_global.close()
            role_log_fp_global = None

    if role_logging_enabled:
        atexit.register(_close_role_log_streams)
    role_log_path_arg = getattr(args, "role_log_file", "")
    if role_log_path_arg is None:
        role_log_path_arg = ""
    role_log_default = os.path.join(
        "result",
        f"{args.dataset}_role_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )
    role_log_path_resolved = role_log_path_arg.strip() or role_log_default

    def _safe_logit(p: float) -> float:
        p = max(1e-5, min(1.0 - 1e-5, p))
        return math.log(p / (1.0 - p))

    role_select_logit = _safe_logit(select_frac_init)
    role_entity_logit = _safe_logit(entity_frac_init)
    role_explore_epochs = max(0, int(getattr(args, "role_explore_epochs", 2)))
    role_explore_fraction = float(getattr(args, "role_explore_fraction", 0.15))
    explore_countdown = 0

    for epoch in range(100):
        torch.cuda.empty_cache()
        # Role injection curriculum: after warmup, adjust every N epochs
        try:
            warmup = int(getattr(args, 'role_warmup_epochs', 3))
            period = int(getattr(args, 'role_adjust_period', 2))
            mix_init = float(getattr(args, 'role_mix_coeff_init', 0.2))
            mix_final = float(getattr(args, 'role_mix_coeff_final', 1.0))
            thr_init = float(getattr(args, 'new_edge_threshold_init', 0.6))
            thr_final = float(getattr(args, 'new_edge_threshold_final', 0.3))
            total_epochs = int(getattr(args, 'epoch', 100))
            start_epoch = int(getattr(args, 'role_injection_start_epoch', 20))
            mix_smooth = float(getattr(args, 'role_mix_smooth', 0.6))
            frac_smooth = float(getattr(args, 'role_fraction_smooth', 0.6))
            bridge_init = float(getattr(args, 'role_bridge_coeff_init', getattr(model, 'role_bridge_coeff_init', 0.0)))
            bridge_final = float(getattr(args, 'role_bridge_coeff_final', getattr(model, 'role_bridge_coeff_max', 0.4)))
            bridge_cap = float(getattr(model, 'role_bridge_coeff_max', bridge_final))
            bridge_smooth = float(getattr(args, 'role_bridge_coeff_smooth', 0.7))
            bridge_floor = float(getattr(args, 'role_bridge_coeff_floor', 0.0))
            # compute progress fraction f in [0,1]
            if epoch < warmup:
                f = 0.0
                role_mix = 0.0
                new_thr = thr_init
                role_select_fraction = 0.0
                role_entity_fraction = 0.0
                bridge_coeff = 0.0
            else:
                steps_total = max(1, (total_epochs - warmup + period - 1) // period)
                steps_now = max(0, min(steps_total, (epoch - warmup) // period))
                f = steps_now / float(steps_total)
                role_mix = mix_init + (mix_final - mix_init) * f
                new_thr = thr_init + (thr_final - thr_init) * f
                bridge_coeff = bridge_init + (bridge_final - bridge_init) * f
                bridge_coeff = min(max(bridge_coeff, bridge_floor), bridge_cap)
                if epoch < start_epoch:
                    role_select_fraction = 0.0
                    role_entity_fraction = 0.0
                else:
                    frac_steps_total = max(1, (total_epochs - start_epoch + period - 1) // period)
                    frac_steps_now = max(0, min(frac_steps_total, (epoch - start_epoch) // period))
                    role_select_fraction = min(1.0, select_frac_init + select_frac_step * frac_steps_now)
                    ent_steps_total = frac_steps_total
                    ent_steps_now = frac_steps_now
                    role_entity_fraction = min(1.0, entity_frac_init + entity_frac_step * ent_steps_now)
                    if role_entity_fraction <= 0.0:
                        role_entity_fraction = 0.0
            # apply smoothing to avoid abrupt jumps
            if mix_smooth > 0 and hasattr(model, 'gcn') and hasattr(model.gcn, "role_mix_coeff"):
                prev_mix = float(getattr(model.gcn, "role_mix_coeff", role_mix))
                role_mix = mix_smooth * prev_mix + (1.0 - mix_smooth) * role_mix
            if frac_smooth > 0 and hasattr(model, 'gcn'):
                prev_frac = float(getattr(model.gcn, "current_role_selection_fraction", 0.0))
                prev_ent_frac = float(getattr(model.gcn, "current_role_fraction", 0.0))
                role_select_fraction = frac_smooth * prev_frac + (1.0 - frac_smooth) * role_select_fraction
                role_entity_fraction = frac_smooth * prev_ent_frac + (1.0 - frac_smooth) * role_entity_fraction
            if bridge_smooth > 0 and hasattr(model, 'gcn') and hasattr(model.gcn, "role_bridge_coeff"):
                prev_bridge = float(getattr(model.gcn, "role_bridge_coeff", bridge_coeff))
                bridge_coeff = bridge_smooth * prev_bridge + (1.0 - bridge_smooth) * bridge_coeff
            reward_enabled = epoch >= start_epoch
            acc_grad = 0.0
            div_grad = 0.0
            if rsrg_trainer is not None:
                last_vec = getattr(rsrg_trainer, "last_vector_reward", None)
                if last_vec is not None and reward_enabled:
                    acc_grad = float(last_vec[0])
                    div_grad = float(last_vec[1])
                    role_select_logit += role_fraction_lr * (0.4 * acc_grad + div_grad)
                    role_entity_logit += role_entity_lr * (acc_grad + 0.4 * div_grad)
                    role_mix = max(0.0, role_mix * (1.0 + 0.1 * acc_grad + 0.15 * div_grad))
                    role_mix += role_mix_reward_gain * (0.6 * acc_grad + 0.4 * div_grad)
                    new_thr = min(1.0, max(role_edge_min_threshold, new_thr - 0.05 * div_grad))
                    bridge_coeff = bridge_coeff + role_bridge_lr * div_grad
                    if acc_grad > 0.0 and div_grad <= 0.0:
                        bridge_coeff = bridge_coeff - role_bridge_decay * acc_grad
                    bridge_coeff = min(max(bridge_coeff, bridge_floor), bridge_cap)
                    if role_explore_epochs > 0 and div_grad < -1e-4:
                        explore_countdown = max(explore_countdown, role_explore_epochs)
                    elif role_explore_epochs > 0 and div_grad > 1e-4:
                        explore_countdown = max(0, explore_countdown - 1)
            learned_select = 1.0 / (1.0 + math.exp(-role_select_logit))
            learned_entity = 1.0 / (1.0 + math.exp(-role_entity_logit))
            role_select_fraction = (
                (1.0 - role_fraction_learned_weight) * role_select_fraction
                + role_fraction_learned_weight * learned_select
            )
            role_entity_fraction = (
                (1.0 - role_fraction_learned_weight) * role_entity_fraction
                + role_fraction_learned_weight * learned_entity
            )
            role_mix = min(mix_final, max(role_mix_floor, role_mix))
            role_select_fraction = min(1.0, max(0.0, role_select_fraction))
            role_select_fraction = min(role_selection_cap, role_select_fraction)
            role_entity_fraction = min(1.0, max(0.0, role_entity_fraction))
            if role_entity_fraction > 0.0:
                upper_cap = max(1e-6, role_entity_frac_cap)
                lower_cap = max(0.0, min(role_entity_frac_floor, upper_cap))
                role_entity_fraction = min(upper_cap, max(lower_cap, role_entity_fraction))
            new_thr = min(1.0, max(role_edge_min_threshold, new_thr))
            if explore_countdown > 0:
                role_select_fraction = max(role_select_fraction, role_explore_fraction)
                role_entity_fraction = max(role_entity_fraction, role_explore_fraction * 0.6)
                explore_countdown -= 1
            role_phase_ratio = role_phase_ratio_default
            metric_drop = 0.0
            last_reward_val = None
            if rsrg_trainer is not None:
                metric_drop = float(getattr(rsrg_trainer, "last_metric_drop", 0.0))
                last_reward_val = getattr(rsrg_trainer, "last_reward", None)
            backoff_triggered = False
            if (
                role_metric_drop_threshold > 0.0
                and metric_drop > role_metric_drop_threshold
                and role_negative_backoff > 0.0
            ):
                role_select_fraction *= role_negative_backoff
                role_entity_fraction *= role_negative_backoff
                role_phase_ratio *= role_negative_backoff
                backoff_triggered = True
            skip_roles = False
            skip_reason = ""
            if role_metric_drop_threshold > 0.0 and metric_drop > role_metric_drop_threshold:
                skip_roles = True
                skip_reason = f"metric_drop={metric_drop:.4f} > thr={role_metric_drop_threshold:.4f}"
            if (
                not skip_roles
                and role_reward_floor > 0.0
                and last_reward_val is not None
                and last_reward_val < -role_reward_floor
            ):
                skip_roles = True
                skip_reason = f"reward={last_reward_val:.4f} < -{role_reward_floor:.4f}"
            if skip_roles:
                role_select_fraction = 0.0
                role_entity_fraction = 0.0
                role_phase_ratio = 0.0
            else:
                if role_select_fraction > 0.0:
                    role_select_fraction = max(role_select_fraction, role_selection_floor)
                if role_entity_fraction > 0.0:
                    role_entity_fraction = max(role_entity_fraction, role_selection_floor)
            role_select_fraction = min(role_selection_cap, role_select_fraction)
            role_phase_ratio = min(1.0, max(0.0, role_phase_ratio))
            setattr(model, "_skip_role_injection", bool(skip_roles))
            setattr(model, "_skip_role_reason", skip_reason if skip_roles else "")
            # update GCN knobs
            if hasattr(model, 'gcn'):
                model.gcn.role_mix_coeff = float(role_mix)
                model.gcn.new_edge_threshold = float(new_thr)
                # fixed MMD weight from args
                model.gcn.mmd_weight = float(getattr(args, 'mmd_weight', 0.5))
                model.gcn.current_role_fraction = float(role_entity_fraction)
                setattr(model.gcn, "current_role_selection_fraction", float(role_select_fraction))
                setattr(model.gcn, "current_role_phase_ratio", float(role_phase_ratio))
                if hasattr(model.gcn, "set_role_bridge_coeff"):
                    model.gcn.set_role_bridge_coeff(float(min(max(bridge_coeff, bridge_floor), bridge_cap)))
            if getattr(args, 'progress_verbose', True) and epoch % max(1, period) == 0:
                if backoff_triggered:
                    print(
                        f"[Schedule] metric_drop={metric_drop:.4f} > thr={role_metric_drop_threshold:.4f} "
                        f"apply_backoff={role_negative_backoff:.3f}"
                    )
                if skip_roles and skip_reason:
                    print(f"[Schedule] skip role injection this epoch: {skip_reason}")
                print(
                    f"[Schedule] epoch={epoch} mix={role_mix:.3f} new_thr={new_thr:.3f} "
                    f"select_frac={role_select_fraction:.3f} entity_frac={role_entity_fraction:.3f} "
                    f"phase_ratio={role_phase_ratio:.3f} "
                    f"bridge={bridge_coeff:.3f} "
                    f"warmup={warmup} period={period}"
                )
        except Exception:
            pass
        if epoch % 10 == 0 or epoch==0:
        # if epoch == 100 or epoch == 0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}
            all_feed_data['pos_index'] = torch.LongTensor(index)

        if epoch % refresh_interval == 0:  # 控制候选KG/结构角色刷新的间隔 
            if getattr(args, 'progress_verbose', True):
                print(f"[Phase] 生成候选KG与动态边 | epoch={epoch}")
            inj_start_t = time()
            old_relations = int(getattr(model, 'n_relations', 0))
            old_entities = int(getattr(model, 'n_entities', 0))
            roles_active = 0
            cnt_candi = 0
            cnt_role_rel = 0
            cnt_role_entity = 0
            role_reports: List[Dict[str, Any]] = []
            candi_kg = _generate_candi_kg(item_pmi_dict,n_items,kg_dict,item_embeds=item_embs,pmi_threshold=0.6,cos_threshold=0.95)
            all_candi_kg = _process_kg_attr(candi_kg,triplets)
            if epoch == 0:
                kgr_epoch = max(0, int(getattr(args, "kgr_epoch_initial", 10)))
            else:
                kgr_epoch = max(0, int(getattr(args, "kgr_epoch_refresh", 1)))
            kgr_stats = train_kgr_model(
                model,
                kgr_optimizer,
                triplets,
                kg_mask=KG_mask,
                epochs=kgr_epoch,
                neg_sample_rate=getattr(args, 'kg_neg_sample_rate', 1.0),
            )
            if kgr_stats:
                last_kgr_stats = kgr_stats
            # fitered_kg = _generate_new_kgs(model,candi_kg,pre_rate=0.5)
            # fitered_kg = process_kg_attr(fitered_kg,triplets,kg_mask=KG_mask)
            # pkl.dump(fitered_kg,open(str(epoch)+"_filtered_kg","wb"))
            # fitered_kg = torch.LongTensor(fitered_kg).to(device)
            # 按批注入，避免一次性大张量导致内存/页面不足
            inj_chunk = max(1, int(getattr(args, 'dynamic_edge_chunk_size', 200000)))
            if not role_channel_synced and hasattr(model, 'all_embed'):
                dyn_channel = int(getattr(model, 'emb_size', getattr(args, 'dim', 0)))
                total_dim = int(model.all_embed.size(1)) if dyn_channel > 0 else 0
                dyn_offset = dyn_channel * 2
                if dyn_channel > 0 and total_dim >= dyn_offset + dyn_channel:
                    with torch.no_grad():
                        static_slice = model.all_embed.data[:, dyn_channel:dyn_channel * 2].clone()
                        model.all_embed.data[:, dyn_offset:dyn_offset + dyn_channel].copy_(static_slice)
                gcn_module = getattr(model, 'gcn', None)
                rel_weight = getattr(gcn_module, 'relation_weight', None)
                dyn_rel_weight = getattr(gcn_module, 'n_relation_weight', None)
                if rel_weight is not None and dyn_rel_weight is not None:
                    rel_rows = min(rel_weight.size(0), dyn_rel_weight.size(0))
                    with torch.no_grad():
                        dyn_rel_weight.data[:rel_rows].copy_(rel_weight.data[:rel_rows])
                        if dyn_rel_weight.size(0) > rel_rows:
                            dyn_rel_weight.data[rel_rows:].zero_()
                role_channel_synced = True
            if rsrg_trainer is not None and getattr(args, 'rule_reset_feedback_each_refresh', False):
                rsrg_trainer.clear_feedback_state()
            model.reset_dynamic_edges()
            if role_logging_enabled:
                _role_log(f"[RoleLog] ===== Epoch {epoch} =====", flush=True)
            if retained_role_edges:
                expired = []
                for rid, bundle in list(retained_role_edges.items()):
                    ttl = int(bundle.get("ttl", 1))
                    role_type_id = int(bundle.get("role_type", 0))
                    entity_edges = np.asarray(bundle.get("entity_edges", np.empty((0, 3), dtype=np.int64)))
                    entity_gate = np.asarray(bundle.get("entity_gate", np.empty((0,), dtype=np.float32)))
                    relation_edges = np.asarray(bundle.get("relation_edges", np.empty((0, 3), dtype=np.int64)))
                    relation_gate = np.asarray(bundle.get("relation_gate", np.empty((0,), dtype=np.float32)))
                    quality_thr = float(role_retention_min_score)
                    if entity_edges.size > 0 and entity_gate.size == entity_edges.shape[0]:
                        filtered_edges, _, kept_scores, kept_base = _filter_role_edges_by_score(
                            model,
                            entity_edges,
                            device,
                            threshold=quality_thr,
                            top_ratio=1.0,
                            min_keep=0,
                            verbose=False,
                            label=f"retain-entity:{rid}",
                            base_weights=entity_gate,
                            auto_threshold=False,
                            threshold_quantile=1.0,
                            threshold_min=quality_thr,
                        )
                        if filtered_edges.size > 0:
                            entity_gate = _compose_edge_gate(kept_scores, kept_base)
                            entity_edges = filtered_edges
                        else:
                            entity_gate = np.empty((0,), dtype=np.float32)
                            entity_edges = np.empty((0, 3), dtype=np.int64)
                    if relation_edges.size > 0 and relation_gate.size == relation_edges.shape[0]:
                        filtered_rel, _, rel_scores, rel_base = _filter_role_edges_by_score(
                            model,
                            relation_edges,
                            device,
                            threshold=quality_thr,
                            top_ratio=1.0,
                            min_keep=0,
                            verbose=False,
                            label=f"retain-rel:{rid}",
                            base_weights=relation_gate,
                            auto_threshold=False,
                            threshold_quantile=1.0,
                            threshold_min=quality_thr,
                        )
                        if filtered_rel.size > 0:
                            relation_gate = _compose_edge_gate(rel_scores, rel_base)
                            relation_edges = filtered_rel
                        else:
                            relation_gate = np.empty((0,), dtype=np.float32)
                            relation_edges = np.empty((0, 3), dtype=np.int64)
                    retained_role_edges[rid]["entity_edges"] = entity_edges
                    retained_role_edges[rid]["entity_gate"] = entity_gate
                    retained_role_edges[rid]["relation_edges"] = relation_edges
                    retained_role_edges[rid]["relation_gate"] = relation_gate
                    if entity_edges.size == 0 and relation_edges.size == 0:
                        expired.append(rid)
                        continue
                    injected = False
                    if entity_edges.size > 0 and entity_gate.size == entity_edges.shape[0]:
                        _ensure_role_relations(model, entity_edges[:, 1])
                        ent_tensor = torch.LongTensor(entity_edges)
                        gate_tensor = torch.as_tensor(entity_gate, dtype=torch.float32)
                        meta_payload = {
                            "gate": gate_tensor,
                            "role_type": torch.full((gate_tensor.size(0),), role_type_id, dtype=torch.long),
                            "reward": torch.zeros(gate_tensor.size(0), dtype=torch.float32),
                        }
                        model.append_dynamic_edges(ent_tensor, meta=meta_payload)
                        _adapt_role_edges(
                            model,
                            kgr_optimizer,
                            ent_tensor,
                            args,
                            device,
                            gate_tensor=gate_tensor,
                        )
                        injected = True
                    if relation_edges.size > 0 and relation_gate.size == relation_edges.shape[0]:
                        _ensure_role_relations(model, relation_edges[:, 1])
                        rel_tensor = torch.LongTensor(relation_edges)
                        rel_gate_tensor = torch.as_tensor(relation_gate, dtype=torch.float32)
                        rel_meta = {
                            "gate": rel_gate_tensor,
                            "role_type": torch.full((rel_gate_tensor.size(0),), role_type_id, dtype=torch.long),
                            "reward": torch.zeros(rel_gate_tensor.size(0), dtype=torch.float32),
                        }
                        model.append_dynamic_edges(rel_tensor, meta=rel_meta)
                        _adapt_role_edges(
                            model,
                            kgr_optimizer,
                            rel_tensor,
                            args,
                            device,
                            gate_tensor=rel_gate_tensor,
                        )
                        injected = True
                    if injected and rsrg_trainer is not None:
                        rule_key = bundle.get("rule_key")
                        if rule_key is not None:
                            role_reports.append(
                                {
                                    "rule_key": rule_key,
                                    "position": int(bundle.get("position", 0)),
                                    "edge_count": int(entity_edges.shape[0] + relation_edges.shape[0]),
                                    "quality": float(bundle.get("quality", 0.0)),
                                    "avg_score": float(bundle.get("quality", 0.0)),
                                    "retained": 1.0,
                                    "streak": float(bundle.get("ttl", 1)),
                                    "reward_hint": 0.0,
                                }
                            )
                    bundle["ttl"] = ttl if injected else 0
                    if bundle["ttl"] <= 0:
                        expired.append(rid)
                for rid in expired:
                    retained_role_edges.pop(rid, None)
            # 候选KG边
            if all_candi_kg.size > 0:
                cnt_candi = int(all_candi_kg.shape[0])
                for start in range(0, all_candi_kg.shape[0], inj_chunk):
                    end = min(start + inj_chunk, all_candi_kg.shape[0])
                    chunk_tensor = torch.LongTensor(all_candi_kg[start:end])
                    model.append_dynamic_edges(chunk_tensor)
                    _adapt_role_edges(model, kgr_optimizer, chunk_tensor, args, device)
                if getattr(args, 'progress_verbose', True):
                    edge_tensor = getattr(model, "aug_edge_index", None)
                    device_info = edge_tensor.device if edge_tensor is not None else "cpu"
                    print(f"[KG] injected={cnt_candi} device={device_info}")
            if rsrg_trainer is not None:
                # 实体模式优先（默认）：只走两跳角色注入；预热期内跳过结构角色注入
                start_epoch = int(getattr(args, 'role_injection_start_epoch', getattr(args, 'role_warmup_epochs', 3)))
                roles_enabled = epoch >= start_epoch
                role_entity_fraction = float(getattr(model.gcn, 'current_role_fraction', 0.0)) if hasattr(model, 'gcn') else 0.0
                role_selection_fraction = float(getattr(model.gcn, 'current_role_selection_fraction', role_entity_fraction)) if hasattr(model, 'gcn') else role_entity_fraction
                role_phase_ratio = float(getattr(model.gcn, "current_role_phase_ratio", role_phase_ratio_default)) if hasattr(model, 'gcn') else role_phase_ratio_default
                role_phase_ratio = min(max(role_phase_ratio, 0.0), 1.0)
                if role_selection_fraction > 0.0:
                    role_selection_fraction = max(role_selection_fraction, role_selection_floor)
                if role_entity_fraction > 0.0:
                    role_entity_fraction = max(role_entity_fraction, role_selection_floor)
                if hasattr(model, 'gcn'):
                    setattr(model.gcn, "current_relation_fraction", 0.0)
                skip_roles_flag = bool(getattr(model, "_skip_role_injection", False))
                skip_reason = getattr(model, "_skip_role_reason", "")
                if skip_roles_flag and getattr(args, 'progress_verbose', True):
                    note = skip_reason or "suppressed by scheduler"
                    _role_log(f"[Role] skip structural role injection (epoch={epoch}): {note}")
                roles_enabled = roles_enabled and not skip_roles_flag
                allow_new_role_candidates = True
                current_new_role_scale = 1.0
                new_role_gate_reason = ""
                if getattr(args, 'role_entity_mode', True) and roles_enabled and role_selection_fraction > 0.0 and role_entity_fraction > 0.0:
                    allow_new_role_candidates = True
                    current_new_role_scale = 1.0
                    new_role_gate_reason = ""
                    if recall_history:
                        last_epoch_idx, last_recall_val = recall_history[-1]
                        prev_recall_val = recall_history[-2][1] if len(recall_history) >= 2 else None
                        best_recall_epoch = best_epoch.get("recall", last_epoch_idx)
                        best_recall_val = best_metric.get("recall", float("-inf"))
                        if (
                            prev_recall_val is not None
                            and last_recall_val + recall_drop_tolerance < prev_recall_val
                        ):
                            allow_new_role_candidates = False
                            new_role_gate_reason = (
                                f"recall {last_recall_val:.4f} < prev {prev_recall_val:.4f}"
                            )
                        elif (
                            math.isfinite(best_recall_val)
                            and last_recall_val + recall_resume_tolerance < best_recall_val
                        ):
                            allow_new_role_candidates = False
                            new_role_gate_reason = (
                                f"recall {last_recall_val:.4f} < best {best_recall_val:.4f}"
                            )
                    if not allow_new_role_candidates:
                        if (not pause_new_roles) or (new_role_pause_reason != new_role_gate_reason):
                            if getattr(args, 'progress_verbose', True):
                                _role_log(
                                    f"[Role] pause new role injection (epoch={epoch}): {new_role_gate_reason}"
                                )
                        pause_new_roles = True
                        new_role_pause_reason = new_role_gate_reason
                        pending_new_role_scales.clear()
                        current_new_role_scale = 0.0
                    else:
                        if pause_new_roles:
                            pause_new_roles = False
                            new_role_pause_reason = ""
                            pending_new_role_scales.clear()
                            pending_new_role_scales.extend([1.0 / 3.0, 2.0 / 3.0, 1.0])
                            if getattr(args, 'progress_verbose', True):
                                _role_log(f"[Role] resume new role injection (epoch={epoch})")
                        if pending_new_role_scales:
                            current_new_role_scale = pending_new_role_scales.popleft()
                        else:
                            current_new_role_scale = 1.0
                    from tqdm import tqdm as _tqdm
                    current_injection_new_roles: List[Dict[str, Any]] = []
                    raw_role_pairs = rsrg_trainer.collect_role_pairs(args.rule_edges_per_role)
                    if not raw_role_pairs:
                        cached_pairs = rsrg_trainer.get_recent_role_pairs()
                        if cached_pairs:
                            raw_role_pairs = cached_pairs
                            if getattr(args, 'progress_verbose', True):
                                _role_log("[Role] reuse cached role pairs (sampler empty this round)")
                    item_emb_array = item_embs if isinstance(item_embs, np.ndarray) else np.asarray(item_embs)
                    role_candidates = []
                    total_entities_considered = 0
                    total_scored_entities = 0
                    for state, entity_ids in raw_role_pairs:
                        entities_arr = np.asarray(entity_ids, dtype=np.int64).reshape(-1)
                        if entities_arr.size == 0:
                            continue
                        filtered_entities, score_map = _filter_role_entities(
                            state,
                            entities_arr,
                            item_pmi_dict,
                            item_emb_array,
                            model,
                            args,
                            rsrg_trainer,
                            n_items,
                        )
                        if filtered_entities.size == 0:
                            continue
                        total_entities_considered += int(filtered_entities.shape[0])
                        total_scored_entities += len(score_map)
                        role_candidates.append((state, filtered_entities, score_map))
                    if getattr(args, 'progress_verbose', True):
                        last_reward = getattr(rsrg_trainer, "last_reward", None)
                        _role_log(
                            f"[Role] raw_pairs={len(raw_role_pairs)} candidates={len(role_candidates)} "
                            f"entities={total_entities_considered} scored={total_scored_entities}"
                        )
                        if last_reward is not None:
                            _role_log(f"[Role] last_reward={last_reward:.4f}")
                    selected_candidates = []
                    if not role_candidates:
                        roles_active = 0
                        role_active_rounds = {}
                    else:
                        new_roles = [st for st, _, _ in role_candidates if getattr(st, 'role_entity_id', -1) < 0]
                        if new_roles:
                            need = len(new_roles)
                            required_entities = role_entity_next_id + need
                            model.ensure_entity_capacity(required_entities)
                            for st in new_roles:
                                st.role_entity_id = role_entity_next_id
                                role_entity_next_id += 1
                            n_entities = model.n_entities
                        edge_threshold = float(
                            getattr(
                                args,
                                "role_edge_score_threshold",
                                getattr(getattr(model, "gcn", None), "new_edge_threshold", 0.5),
                            )
                        )
                        edge_top_ratio = float(getattr(args, "role_edge_top_ratio", 1.0))
                        edge_min_keep = max(1, int(getattr(args, "role_edge_min_keep", 2)))
                        edge_verbose = bool(getattr(args, "progress_verbose", True))
                        inject_score_threshold = float(getattr(args, "role_inject_score_threshold", 0.0))
                        inject_score_threshold = min(max(inject_score_threshold, 0.0), 1.0)
                        prepared_candidates = []
                        rejected_new_candidates: List[Dict[str, Any]] = []
                        for state, entities_arr, score_map in role_candidates:
                            rid = state.role_entity_id
                            if rid is None or rid < 0:
                                continue
                            penalty_info = role_penalties.get(rid)
                            if penalty_info and penalty_info.get("cooldown", 0) > 0:
                                if getattr(args, 'progress_verbose', True):
                                    _role_log(
                                        f"[Role] defer rid={rid} due to penalty cooldown={penalty_info['cooldown']}"
                                    )
                                continue
                            entities_arr = np.asarray(entities_arr, dtype=np.int64)
                            if entities_arr.size == 0:
                                continue
                            reward_signal = float(getattr(state, "div_score_ema", 0.0))
                            dyn_fraction = role_entity_fraction
                            if dyn_fraction > 0.0:
                                if reward_signal > 0.0 and role_entity_reward_gain > 0.0:
                                    dyn_fraction *= (1.0 + role_entity_reward_gain * reward_signal)
                                elif reward_signal < 0.0 and role_entity_reward_decay > 0.0:
                                    dyn_fraction /= (1.0 + role_entity_reward_decay * (-reward_signal))
                                dyn_fraction = min(
                                    max(1e-6, role_entity_frac_cap),
                                    max(role_entity_frac_floor, dyn_fraction),
                                )
                            entity_keep = max(1, int(entities_arr.shape[0] * max(dyn_fraction, 1e-6)))
                            if entity_keep < entities_arr.shape[0]:
                                entities_arr = entities_arr[:entity_keep]
                            weight_values = _build_role_entity_base_weights(
                                entities_arr, score_map, rsrg_trainer, n_items
                            )
                            if weight_values.size == 0:
                                continue
                            rule_key = state.descriptor.rule_key
                            relation_key = (rule_key, state.descriptor.position)
                            role_reward = float(getattr(state, "div_score_ema", 0.0))
                            role_type_id = ROLE_TYPE_TO_ID.get(state.descriptor.role_type, 0)
                            rel_pair = rule_relation_map.get(relation_key)
                            if rel_pair is None:
                                src_rel = rule_rel_next_id
                                dst_rel = rule_rel_next_id + 1
                                rule_rel_next_id += 2
                                model.ensure_relation_capacity(rule_rel_next_id)
                                n_relations = max(n_relations, rule_rel_next_id)
                                rule_relation_map[relation_key] = (src_rel, dst_rel)
                                model.gcn.register_role_relations([src_rel, dst_rel])
                            else:
                                src_rel, dst_rel = rel_pair
                            entity_to_role = np.column_stack(
                                (
                                    entities_arr,
                                    np.full((entities_arr.shape[0],), src_rel, dtype=np.int64),
                                    np.full((entities_arr.shape[0],), rid, dtype=np.int64),
                                )
                            )
                            role_to_entity = np.column_stack(
                                (
                                    np.full((entities_arr.shape[0],), rid, dtype=np.int64),
                                    np.full((entities_arr.shape[0],), dst_rel, dtype=np.int64),
                                    entities_arr,
                                )
                            )
                            merged_edges = np.vstack((entity_to_role, role_to_entity))
                            entity_base_weights = np.concatenate([weight_values, weight_values])
                            entity_base_weights = _blend_edge_scores_with_kgc(
                                model, merged_edges, entity_base_weights, device, mix=0.6
                            )
                            raw_entity_edges = merged_edges.copy()
                            raw_entity_base_weights = entity_base_weights.copy()
                            local_threshold = _compute_role_threshold(edge_threshold, state, rsrg_trainer)
                            local_threshold = max(role_edge_min_threshold, local_threshold)
                            filtered_edges, edge_stats, edge_scores, edge_base = _filter_role_edges_by_score(
                                model,
                                merged_edges,
                                device,
                                threshold=local_threshold,
                                top_ratio=edge_top_ratio,
                                min_keep=edge_min_keep,
                                verbose=edge_verbose,
                                label=f"entity:{state.descriptor.role_type.value}:{state.descriptor.rule_key}",
                                base_weights=entity_base_weights,
                                auto_threshold=role_edge_auto_threshold,
                                threshold_quantile=role_edge_keep_quantile,
                                threshold_min=role_edge_min_threshold,
                            )
                            entity_gate = _compose_edge_gate(edge_scores, edge_base)
                            entity_gate = _apply_gate_temperature(entity_gate, role_entity_gate_temp)
                            rule = rsrg_trainer.role_manager.get_rule(state.descriptor.rule_key)
                            rel_edges = np.empty((0, 3), dtype=np.int64)
                            rel_stats = {"total": 0, "kept": 0, "avg_score": 0.0, "fallback": False}
                            rel_scores = np.empty((0,), dtype=np.float32)
                            rel_base = None
                            relation_gate = np.empty((0,), dtype=np.float32)
                            raw_relation_edges: Optional[np.ndarray] = None
                            raw_relation_base_weights: Optional[np.ndarray] = None
                            if rule is not None:
                                triples, rel_meta = rsrg_trainer.virtual_builder.build_edges(
                                    state, rule, args.rule_edges_per_role
                                )
                                if triples:
                                    rel_edges = np.asarray(triples, dtype=np.int64)
                                    if rel_meta and rel_meta.get("edge_weights"):
                                        rel_base_weights = np.asarray(rel_meta["edge_weights"], dtype=np.float32)
                                    else:
                                        rel_base_weights = np.ones((rel_edges.shape[0],), dtype=np.float32)
                                    if rel_base_weights.size != rel_edges.shape[0]:
                                        rel_base_weights = np.resize(rel_base_weights, (rel_edges.shape[0],))
                                    total_rel_weight = float(rel_base_weights.sum())
                                    if total_rel_weight > 0:
                                        rel_base_weights = rel_base_weights / total_rel_weight
                                    rel_base_weights = _blend_edge_scores_with_kgc(
                                        model, rel_edges, rel_base_weights, device, mix=0.6
                                    )
                                    raw_relation_edges = rel_edges.copy()
                                    raw_relation_base_weights = rel_base_weights.copy()
                                    role_reward = float(rel_meta.get("div_reward", role_reward))
                                    rel_edges, rel_stats, rel_scores, rel_base = _filter_role_edges_by_score(
                                        model,
                                        rel_edges,
                                        device,
                                        threshold=local_threshold,
                                        top_ratio=edge_top_ratio,
                                        min_keep=edge_min_keep,
                                        verbose=edge_verbose,
                                        label=f"rel:{state.descriptor.role_type.value}:{state.descriptor.rule_key}",
                                        base_weights=rel_base_weights,
                                        auto_threshold=role_edge_auto_threshold,
                                        threshold_quantile=role_edge_keep_quantile,
                                        threshold_min=role_edge_min_threshold,
                                    )
                            relation_gate = _compose_edge_gate(rel_scores, rel_base)
                            relation_gate = _apply_gate_temperature(relation_gate, role_relation_gate_temp)
                            _store_retained_edges(
                                retained_role_edges,
                                rid,
                                role_type_id,
                                filtered_edges,
                                entity_gate,
                                rel_edges,
                                relation_gate,
                                role_retention_top_ratio,
                                role_retention_min_score,
                                role_retention_ttl,
                                state.descriptor.rule_key,
                                state.descriptor.position,
                            )
                            candidate_score = max(edge_stats.get("avg_score", 0.0), rel_stats.get("avg_score", 0.0))
                            candidate_payload = {
                                "state": state,
                                "rid": rid,
                                "entities": entities_arr,
                                "weights": weight_values,
                                "score_map": score_map,
                                "entity_edges": filtered_edges,
                                "entity_stats": edge_stats,
                                "entity_scores": edge_scores,
                                "entity_base_weights": edge_base,
                                "entity_gate": entity_gate,
                                "relation_edges": rel_edges,
                                "relation_stats": rel_stats,
                                "relation_scores": rel_scores,
                                "relation_base_weights": rel_base,
                                "relation_gate": relation_gate,
                                "score": candidate_score,
                                "threshold": local_threshold,
                                "role_type_id": role_type_id,
                                "role_reward": role_reward,
                                "raw_entity_edges": raw_entity_edges,
                                "raw_entity_base_weights": raw_entity_base_weights,
                                "raw_relation_edges": raw_relation_edges,
                                "raw_relation_base_weights": raw_relation_base_weights,
                            }
                            coverage_set, embed_ids, embed_weights = _summarize_role_users(
                                entities_arr,
                                coverage_cap=role_user_coverage_cap,
                                embed_cap=role_user_embed_cap,
                            )
                            candidate_payload["user_coverage"] = coverage_set
                            candidate_payload["avg_npmi"] = (
                                float(np.mean(list(score_map.values()))) if score_map else 0.0
                            )
                            if embed_ids is not None:
                                candidate_payload["user_embed_ids"] = embed_ids
                                candidate_payload["user_embed_weights"] = embed_weights
                            is_new_role = role_active_rounds.get(rid, 0) == 0
                            if candidate_score < inject_score_threshold or (
                                filtered_edges.size == 0 and rel_edges.size == 0
                            ):
                                if candidate_score < inject_score_threshold and edge_verbose:
                                    _role_log(
                                        f"[Role] skip candidate rule={state.descriptor.rule_key} "
                                        f"score={candidate_score:.3f} < inject_thr={inject_score_threshold:.3f}"
                                    )
                                if is_new_role:
                                    rejected_new_candidates.append(candidate_payload)
                                continue
                            prepared_candidates.append(candidate_payload)
                        prev_role_counts = role_active_rounds
                        retained_selected: List[Dict[str, Any]] = []
                        fresh_candidates: List[Dict[str, Any]] = []
                        selected_candidates: List[Dict[str, Any]] = []
                        keep_roles = 0
                        if prepared_candidates:
                            if role_prescore_enabled:
                                for cand in prepared_candidates:
                                    pre_score, diag = _evaluate_role_prescore(model, cand, args)
                                    cand["pre_score"] = pre_score
                                    cand["pre_diag"] = diag
                                    base_score = float(cand.get("score", 0.0))
                                    cand["combined_score"] = (
                                        (1.0 - role_prescore_blend) * base_score
                                        + role_prescore_blend * pre_score
                                    )
                            else:
                                for cand in prepared_candidates:
                                    base_score = float(cand.get("score", 0.0))
                                    cand["pre_score"] = base_score
                                    cand["pre_diag"] = {"kgc": base_score, "cos": 0.5, "reward": 0.5}
                                    cand["combined_score"] = base_score
                            prepared_candidates.sort(
                                key=lambda x: x.get("combined_score", x["score"]), reverse=True
                            )
                            def _relax_candidate_edges_for_explore(
                                payload: Dict[str, Any], relax_ratio: float = 0.1
                            ) -> None:
                                relaxed_threshold = max(
                                    role_edge_min_threshold, payload.get("threshold", 0.0) * (1.0 - relax_ratio)
                                )
                                relaxed_top_ratio = min(1.0, edge_top_ratio * (1.0 + relax_ratio))
                                raw_entity_edges = payload.get("raw_entity_edges")
                                raw_entity_base = payload.get("raw_entity_base_weights")
                                if isinstance(raw_entity_edges, np.ndarray) and raw_entity_edges.size > 0:
                                    entity_base = raw_entity_base
                                    if entity_base is None:
                                        entity_base = np.ones((raw_entity_edges.shape[0],), dtype=np.float32)
                                    filtered, stats, scores, base = _filter_role_edges_by_score(
                                        model,
                                        raw_entity_edges,
                                        device,
                                        threshold=relaxed_threshold,
                                        top_ratio=relaxed_top_ratio,
                                        min_keep=max(1, int(edge_min_keep * 0.9)),
                                        verbose=False,
                                        label="explore-entity",
                                        base_weights=entity_base,
                                        auto_threshold=role_edge_auto_threshold,
                                        threshold_quantile=role_edge_keep_quantile,
                                        threshold_min=role_edge_min_threshold,
                                    )
                                    payload["entity_edges"] = filtered
                                    payload["entity_stats"] = stats
                                    payload["entity_scores"] = scores
                                    payload["entity_base_weights"] = base
                                    payload["entity_gate"] = _apply_gate_temperature(
                                        _compose_edge_gate(scores, base), role_entity_gate_temp
                                    )
                                raw_relation_edges = payload.get("raw_relation_edges")
                                raw_relation_base = payload.get("raw_relation_base_weights")
                                if (
                                    isinstance(raw_relation_edges, np.ndarray)
                                    and raw_relation_edges.size > 0
                                    and raw_relation_base is not None
                                ):
                                    filtered_rel, stats_rel, scores_rel, base_rel = _filter_role_edges_by_score(
                                        model,
                                        raw_relation_edges,
                                        device,
                                        threshold=relaxed_threshold,
                                        top_ratio=relaxed_top_ratio,
                                        min_keep=max(1, int(edge_min_keep * 0.9)),
                                        verbose=False,
                                        label="explore-rel",
                                        base_weights=raw_relation_base,
                                        auto_threshold=role_edge_auto_threshold,
                                        threshold_quantile=role_edge_keep_quantile,
                                        threshold_min=role_edge_min_threshold,
                                    )
                                    payload["relation_edges"] = filtered_rel
                                    payload["relation_stats"] = stats_rel
                                    payload["relation_scores"] = scores_rel
                                    payload["relation_base_weights"] = base_rel
                                    payload["relation_gate"] = _apply_gate_temperature(
                                        _compose_edge_gate(scores_rel, base_rel), role_relation_gate_temp
                                    )
                            global_prescore = float(
                                np.mean([cand.get("pre_score", 0.0) for cand in prepared_candidates])
                            ) if prepared_candidates else 0.0
                            total_candidates = len(prepared_candidates)
                            keep_roles = min(
                                total_candidates,
                                max(
                                    role_min_active,
                                    int(math.ceil(total_candidates * role_selection_fraction)),
                                ),
                            )
                            phase_cap = max(
                                role_min_active,
                                int(math.ceil(total_candidates * max(role_phase_ratio, 1e-6))),
                            )
                            keep_roles = min(keep_roles, phase_cap)
                            if role_prescore_enabled:
                                quality_cap = min(role_prescore_role_cap, max(global_prescore, 1e-3))
                                keep_roles = min(
                                    keep_roles,
                                    max(
                                        role_min_active,
                                        int(math.ceil(total_candidates * quality_cap)),
                                    ),
                                )
                            keep_roles = max(role_min_active, keep_roles)
                            retained_in_pool = sum(
                                1 for cand in prepared_candidates if prev_role_counts.get(cand["rid"], 0) > 0
                            )
                            retained_rank_limit = 0
                            if retained_in_pool > 0 and total_candidates > 0:
                                rank_fraction = min(
                                    1.0, (retained_in_pool * 2.0) / max(float(total_candidates), 1.0)
                                )
                                retained_rank_limit = max(
                                    1, int(math.ceil(total_candidates * rank_fraction))
                                )
                            for idx, cand in enumerate(prepared_candidates):
                                rid = cand["rid"]
                                if prev_role_counts.get(rid, 0) > 0:
                                    if retained_rank_limit > 0 and idx < retained_rank_limit:
                                        retained_selected.append(cand)
                                    continue
                                fresh_candidates.append(cand)
                            max_new_cap = max(
                                0, int(math.ceil(keep_roles * max(current_new_role_scale, 0.0)))
                            )
                            if (
                                allow_new_role_candidates
                                and not pause_new_roles
                                and current_new_role_scale > 0.0
                                and not fresh_candidates
                                and rejected_new_candidates
                                and max_new_cap > 0
                            ):
                                explore_cap = max(1, int(math.ceil(max_new_cap * 0.25)))
                                rejected_new_candidates.sort(
                                    key=lambda x: x.get("score", 0.0), reverse=True
                                )
                                exploratory = rejected_new_candidates[:explore_cap]
                                for cand in exploratory:
                                    _relax_candidate_edges_for_explore(cand, relax_ratio=0.1)
                                fresh_candidates.extend(exploratory)
                                if getattr(args, 'progress_verbose', True):
                                    _role_log(
                                        f"[Role] explore extra new roles (cap={explore_cap}/{max_new_cap}) "
                                        f"due to empty fresh candidate pool"
                                    )
                            if fresh_candidates and (role_redundancy_weight > 0.0 or role_complementary_weight > 0.0):
                                baseline_users: Set[int] = set()
                                if role_redundancy_weight > 0.0 and retained_selected:
                                    for retained in retained_selected:
                                        cov = retained.get("user_coverage")
                                        if not cov:
                                            continue
                                        for uid in cov:
                                            baseline_users.add(uid)
                                            if role_user_overlap_cap > 0 and len(baseline_users) >= role_user_overlap_cap:
                                                break
                                        if role_user_overlap_cap > 0 and len(baseline_users) >= role_user_overlap_cap:
                                            break
                                for cand in fresh_candidates:
                                    score = float(cand.get("combined_score", cand.get("score", 0.0)))
                                    if score <= 0.0:
                                        continue
                                    adjust = 1.0
                                    cov = cand.get("user_coverage") or set()
                                    if baseline_users and cov and role_redundancy_weight > 0.0:
                                        overlap = len(baseline_users.intersection(cov))
                                        if overlap > 0:
                                            redundancy = overlap / max(1.0, len(cov))
                                            adjust *= max(0.05, 1.0 - role_redundancy_weight * redundancy)
                                    if role_complementary_weight > 0.0:
                                        avg_npmi = float(cand.get("avg_npmi", 0.0))
                                        avg_npmi = min(max(avg_npmi, 0.0), 1.0)
                                        adjust *= 1.0 + role_complementary_weight * (1.0 - avg_npmi)
                                    cand["combined_score"] = score * adjust
                                fresh_candidates.sort(
                                    key=lambda x: x.get("combined_score", x.get("score", 0.0)),
                                    reverse=True,
                                )
                            base_new_cap = min(len(fresh_candidates), keep_roles)
                            scaled_new_cap = base_new_cap
                            if not allow_new_role_candidates or current_new_role_scale <= 0.0:
                                scaled_new_cap = 0
                            elif current_new_role_scale < 0.999 and base_new_cap > 0:
                                scaled_new_cap = int(math.ceil(base_new_cap * current_new_role_scale))
                            selected_new = fresh_candidates[:scaled_new_cap]
                            if (
                                getattr(args, 'progress_verbose', True)
                                and base_new_cap > 0
                                and scaled_new_cap != base_new_cap
                            ):
                                if scaled_new_cap == 0:
                                    note = new_role_pause_reason or new_role_gate_reason or "policy guard"
                                    _role_log(f"[Role] skip new roles this round: {note}")
                                else:
                                    note = new_role_pause_reason or new_role_gate_reason
                                    if not note and current_new_role_scale < 1.0:
                                        note = "resume ramp"
                                    elif not note:
                                        note = "policy guard"
                                    _role_log(
                                        f"[Role] limit new roles to {scaled_new_cap}/{base_new_cap} "
                                        f"(scale={current_new_role_scale:.2f}, {note})"
                                    )
                        selected_candidates = retained_selected + selected_new
                        roles_active = len(selected_candidates)
                        current_rids = [cand["rid"] for cand in selected_candidates]
                        prev_counts = role_active_rounds
                        updated_counts: Dict[int, int] = {}
                        new_roles = 0
                        for rid in current_rids:
                            streak = prev_counts.get(rid, 0) + 1
                            updated_counts[rid] = streak
                            if streak == 1:
                                new_roles += 1
                        role_active_rounds = updated_counts
                        if getattr(args, 'progress_verbose', True) and current_rids:
                            streak_counter = Counter(updated_counts.values())
                            continuing = len(current_rids) - new_roles
                            summary_parts = [
                                f"{count} roles for {length} rounds"
                                for length, count in sorted(streak_counter.items(), reverse=True)
                            ]
                            summary = "; ".join(summary_parts)
                            _role_log(
                                f"[Role] streak summary: total={len(current_rids)} new={new_roles} continuing={continuing} | {summary}",
                            )
                        else:
                            roles_active = 0
                        if current_injection_new_roles:
                            last_injection_epoch = epoch
                            last_injection_new_roles = current_injection_new_roles
                            recall_decline_since_injection = 0
                            last_minor_rollback_epoch = None
                        elif last_injection_epoch == epoch:
                            last_injection_new_roles = []
                        if role_penalties:
                            for pen in role_penalties.values():
                                cooldown = pen.get("cooldown", 0)
                                if cooldown > 0:
                                    pen["cooldown"] = max(0, cooldown - 1)
                    next_role_cache: Dict[Tuple[Tuple[int, ...], int, int], Dict[str, np.ndarray]] = {}
                    pending_mix_updates: Dict[int, float] = {}
                    prescore_skipped = 0
                    for candidate in selected_candidates:
                        state = candidate["state"]
                        rid = candidate["rid"]
                        entities_arr = candidate["entities"]
                        weights = candidate["weights"]
                        score_map = candidate["score_map"]
                        entity_base = candidate.get("entity_base_weights")
                        entity_scores = candidate.get("entity_scores")
                        entity_gate = candidate.get("entity_gate")
                        relation_base = candidate.get("relation_base_weights")
                        relation_scores = candidate.get("relation_scores")
                        relation_gate = candidate.get("relation_gate")
                        role_type_id = candidate.get("role_type_id", 0)
                        role_reward = float(candidate.get("role_reward", 0.0))
                        acc_weight = float(getattr(state, "weight", 1.0))
                        diversity_boost = 1.0 + 0.2 * math.tanh(role_reward)
                        accuracy_gate = max(0.0, 1.0 + 0.5 * math.tanh(acc_weight))
                        pre_score = float(candidate.get("pre_score", candidate.get("score", 0.0)))
                        pre_score = max(0.0, min(1.0, pre_score))
                        if role_prescore_enabled and pre_score < role_prescore_skip:
                            prescore_skipped += 1
                            if getattr(args, 'progress_verbose', True):
                                diag = candidate.get("pre_diag", {})
                                _role_log(
                                    f"[Role] defer candidate rule={state.descriptor.rule_key} "
                                    f"pre_score={pre_score:.3f} skip_thr={role_prescore_skip:.3f} diag={diag}"
                                )
                            continue
                        if prev_counts.get(rid, 0) == 0:
                            current_injection_new_roles.append(
                                {
                                    "rid": rid,
                                    "score": float(candidate.get("combined_score", candidate.get("score", 0.0))),
                                    "cache_key": (state.descriptor.rule_key, state.descriptor.position, rid),
                                }
                            )
                        gate_scale = max(role_prescore_gate_floor, pre_score) if role_prescore_enabled else 1.0
                        state_key = state.descriptor.rule_key
                        cache_key = (state_key, state.descriptor.position, rid)
                        if score_map:
                            for item_id, val in score_map.items():
                                item_idx = int(item_id)
                                if 0 <= item_idx < n_items:
                                    val_f = float(val)
                                    existing = pending_mix_updates.get(item_idx)
                                    if existing is None or val_f > existing:
                                        pending_mix_updates[item_idx] = val_f
                        prev_entry = persistent_role_cache.get(cache_key)
                        if prev_entry is not None:
                            prev_entities = prev_entry.get("entities", np.empty((0,), dtype=np.int64))
                            prev_weights = prev_entry.get("weights", np.empty((0,), dtype=np.float32))
                            merged_entities, merged_weights = _merge_entity_weights(
                                prev_entities,
                                prev_weights,
                                entities_arr,
                                score_map,
                            )
                        else:
                            merged_entities = entities_arr
                            merged_weights = weights
                        merged_cov_set, merged_user_ids, merged_user_weights = _summarize_role_users(
                            merged_entities,
                            coverage_cap=role_user_coverage_cap,
                            embed_cap=role_user_embed_cap,
                        )
                        if merged_cov_set:
                            candidate["user_coverage"] = merged_cov_set
                        if merged_user_ids is not None:
                            candidate["user_embed_ids"] = merged_user_ids
                            candidate["user_embed_weights"] = merged_user_weights
                        model.initialize_role_entity_embedding(
                            rid,
                            merged_entities,
                            merged_weights,
                            user_ids=merged_user_ids,
                            user_weights=merged_user_weights,
                        )
                        entity_edges = candidate["entity_edges"]
                        if entity_base is None and entity_edges.size > 0:
                            entity_base = np.ones((entity_edges.shape[0],), dtype=np.float32)
                        if prev_entry is not None and prev_entry.get("entity_edges") is not None:
                            if prev_entry["entity_edges"].size > 0:
                                combined_edges = np.vstack((prev_entry["entity_edges"], entity_edges))
                                prev_base = prev_entry.get("entity_base_weights")
                                if prev_base is not None and prev_entry["entity_edges"].size > 0:
                                    prev_base_arr = np.asarray(prev_base, dtype=np.float32)
                                else:
                                    prev_base_arr = np.ones((prev_entry["entity_edges"].shape[0],), dtype=np.float32)
                                entity_base_arr = np.asarray(entity_base, dtype=np.float32) if entity_base is not None else np.ones((entity_edges.shape[0],), dtype=np.float32)
                                combined_base = np.concatenate((prev_base_arr, entity_base_arr))
                                entity_edges, candidate_entity_stats, entity_scores, entity_base = _filter_role_edges_by_score(
                                    model,
                                    combined_edges,
                                    device,
                                    threshold=candidate["threshold"],
                                    top_ratio=edge_top_ratio,
                                    min_keep=edge_min_keep,
                                    verbose=False,
                                    label=f"persist-entity:{state_key}",
                                    base_weights=combined_base,
                                    auto_threshold=role_edge_auto_threshold,
                                    threshold_quantile=role_edge_keep_quantile,
                                    threshold_min=role_edge_min_threshold,
                                )
                                candidate["entity_stats"] = candidate_entity_stats
                                candidate["entity_scores"] = entity_scores
                                candidate["entity_base_weights"] = entity_base
                                entity_gate = _compose_edge_gate(entity_scores, entity_base)
                                entity_gate = _apply_gate_temperature(entity_gate, role_entity_gate_temp)
                                candidate["entity_gate"] = entity_gate
                        gate_arr = (
                            entity_gate.astype(np.float32, copy=False)
                            if entity_gate is not None
                            else np.ones((entity_edges.shape[0],), dtype=np.float32)
                        )
                        final_gate_arr = gate_arr * accuracy_gate * diversity_boost * gate_scale
                        pretrain_steps = max(0, int(getattr(args, "role_pretrain_steps", 0)))
                        if entity_edges.size > 0:
                            if pretrain_steps > 0:
                                pre_tensor = torch.LongTensor(entity_edges)
                                _adapt_role_edges(
                                    model,
                                    kgr_optimizer,
                                    pre_tensor,
                                    args,
                                    device,
                                    override_steps=pretrain_steps,
                                    gate_tensor=torch.as_tensor(final_gate_arr, dtype=torch.float32),
                                )
                            _ensure_role_relations(model, entity_edges[:, 1])
                            for start in range(0, entity_edges.shape[0], inj_chunk):
                                end = min(start + inj_chunk, entity_edges.shape[0])
                                chunk_tensor = torch.LongTensor(entity_edges[start:end])
                                gate_slice = torch.as_tensor(final_gate_arr[start:end], dtype=torch.float32)
                                meta_payload = {
                                    "gate": gate_slice,
                                    "role_type": torch.full((gate_slice.size(0),), role_type_id, dtype=torch.long),
                                    "reward": torch.zeros(gate_slice.size(0), dtype=torch.float32),
                                }
                                model.append_dynamic_edges(chunk_tensor, meta=meta_payload)
                                _adapt_role_edges(
                                    model,
                                    kgr_optimizer,
                                    chunk_tensor,
                                    args,
                                    device,
                                    gate_tensor=gate_slice,
                                )
                            cnt_role_entity += int(entity_edges.shape[0])
                        relation_edges = candidate["relation_edges"]
                        if relation_base is None and relation_edges.size > 0:
                            relation_base = np.ones((relation_edges.shape[0],), dtype=np.float32)
                        if prev_entry is not None and prev_entry.get("relation_edges") is not None:
                            if prev_entry["relation_edges"].size > 0:
                                combined_rel = np.vstack((prev_entry["relation_edges"], relation_edges))
                                prev_rel_base = prev_entry.get("relation_base_weights")
                                if prev_rel_base is not None and prev_entry["relation_edges"].size > 0:
                                    prev_rel_base = np.asarray(prev_rel_base, dtype=np.float32)
                                else:
                                    prev_rel_base = np.ones((prev_entry["relation_edges"].shape[0],), dtype=np.float32)
                                rel_base_arr = np.asarray(relation_base, dtype=np.float32) if relation_base is not None else np.ones((relation_edges.shape[0],), dtype=np.float32)
                                combined_rel_base = np.concatenate((prev_rel_base, rel_base_arr))
                                relation_edges, candidate_relation_stats, relation_scores, relation_base = _filter_role_edges_by_score(
                                    model,
                                    combined_rel,
                                    device,
                                    threshold=candidate["threshold"],
                                    top_ratio=edge_top_ratio,
                                    min_keep=edge_min_keep,
                                    verbose=False,
                                    label=f"persist-rel:{state_key}",
                                    base_weights=combined_rel_base,
                                    auto_threshold=role_edge_auto_threshold,
                                    threshold_quantile=role_edge_keep_quantile,
                                    threshold_min=role_edge_min_threshold,
                                )
                                candidate["relation_stats"] = candidate_relation_stats
                                candidate["relation_scores"] = relation_scores
                                candidate["relation_base_weights"] = relation_base
                                relation_gate = _compose_edge_gate(relation_scores, relation_base)
                                relation_gate = _apply_gate_temperature(relation_gate, role_relation_gate_temp)
                                candidate["relation_gate"] = relation_gate
                        if entity_edges.size > 0:
                            _ensure_role_relations(model, entity_edges[:, 1])
                        if relation_edges.size > 0:
                            _ensure_role_relations(model, relation_edges[:, 1])
                        _store_retained_edges(
                            retained_role_edges,
                            rid,
                            role_type_id,
                            entity_edges,
                            entity_gate,
                            relation_edges,
                            relation_gate,
                            role_retention_top_ratio,
                            role_retention_min_score,
                            role_retention_ttl,
                            state.descriptor.rule_key,
                            state.descriptor.position,
                        )
                        rel_gate_arr = (
                            relation_gate.astype(np.float32, copy=False)
                            if relation_gate is not None
                            else np.ones((relation_edges.shape[0],), dtype=np.float32)
                        )
                        final_rel_gate_arr = rel_gate_arr * accuracy_gate * diversity_boost * gate_scale
                        if relation_edges.size > 0:
                            if pretrain_steps > 0:
                                pre_tensor = torch.LongTensor(relation_edges)
                                _adapt_role_edges(
                                    model,
                                    kgr_optimizer,
                                    pre_tensor,
                                    args,
                                    device,
                                    override_steps=pretrain_steps,
                                    gate_tensor=torch.as_tensor(final_rel_gate_arr, dtype=torch.float32),
                                )
                            _ensure_role_relations(model, relation_edges[:, 1])
                            max_rel_id = int(relation_edges[:, 1].max())
                            model.ensure_relation_capacity(max_rel_id + 1)
                            for start in range(0, relation_edges.shape[0], inj_chunk):
                                end = min(start + inj_chunk, relation_edges.shape[0])
                                rel_tensor = torch.LongTensor(relation_edges[start:end])
                                rel_gate_slice = torch.as_tensor(final_rel_gate_arr[start:end], dtype=torch.float32)
                                rel_meta = {
                                    "gate": rel_gate_slice,
                                    "role_type": torch.full((rel_gate_slice.size(0),), role_type_id, dtype=torch.long),
                                    "reward": torch.zeros(rel_gate_slice.size(0), dtype=torch.float32),
                                }
                                model.append_dynamic_edges(rel_tensor, meta=rel_meta)
                                _adapt_role_edges(
                                    model,
                                    kgr_optimizer,
                                    rel_tensor,
                                    args,
                                    device,
                                    gate_tensor=rel_gate_slice,
                                )
                            cnt_role_rel += int(relation_edges.shape[0])
                        next_role_cache[cache_key] = {
                            "entities": merged_entities,
                            "weights": merged_weights,
                            "entity_edges": entity_edges,
                            "relation_edges": relation_edges,
                            "entity_base_weights": entity_base,
                            "relation_base_weights": relation_base,
                            "entity_gate": final_gate_arr,
                            "relation_gate": final_rel_gate_arr,
                        }
                        if rsrg_trainer is not None:
                            entity_stats = candidate.get("entity_stats") or {}
                            relation_stats = candidate.get("relation_stats") or {}
                            score_val = float(candidate.get("score", 0.0))
                            entity_avg = float(entity_stats.get("avg_score", 0.0))
                            relation_avg = float(relation_stats.get("avg_score", 0.0))
                            quality = max(score_val, entity_avg, relation_avg)
                            role_reports.append(
                                {
                                    "rule_key": state.descriptor.rule_key,
                                    "position": state.descriptor.position,
                                    "edge_count": int(entity_edges.shape[0] + relation_edges.shape[0]),
                                    "quality": quality,
                                    "avg_score": quality,
                                    "retained": 1.0 if prev_entry is not None else 0.0,
                                    "streak": float(role_active_rounds.get(rid, 1)),
                                    "reward_hint": float(candidate.get("role_reward", 0.0)),
                                }
                            )
                    if prescore_skipped > 0 and getattr(args, 'progress_verbose', True):
                        _role_log(
                            f"[Role] deferred {prescore_skipped} candidates due to low pre-score."
                        )
                    if pending_mix_updates:
                        _update_role_mix_slopes(model, pending_mix_updates, momentum=role_item_mix_momentum)
                    persistent_role_cache = next_role_cache
                elif (not getattr(args, 'role_entity_mode', True)) and roles_enabled and role_selection_fraction > 0.0:
                    # 关系模式（可选）：为角色分配独立关系ID，按 (h, rel, t) 注入
                    dynamic_edges, rel_metadata = rsrg_trainer.build_virtual_edges(epoch, args.rule_edges_per_role)
                    if dynamic_edges.size > 0:
                        if hasattr(model, "gcn"):
                            setattr(model.gcn, "current_relation_fraction", float(role_selection_fraction))
                        keep_edges = max(1, int(dynamic_edges.shape[0] * role_selection_fraction))
                        gate_arr = np.ones((dynamic_edges.shape[0],), dtype=np.float32)
                        role_arr = np.zeros((dynamic_edges.shape[0],), dtype=np.int64)
                        reward_arr = np.zeros((dynamic_edges.shape[0],), dtype=np.float32)
                        offset = 0
                        for entry in rel_metadata or []:
                            count = int(entry.get("num_edges", 0))
                            if count <= 0:
                                continue
                            end = min(offset + count, gate_arr.size)
                            if end <= offset:
                                break
                            weights = entry.get("edge_weights") or []
                            weights_arr = np.asarray(weights, dtype=np.float32)
                            span = end - offset
                            if weights_arr.size == 0:
                                weights_arr = np.ones((span,), dtype=np.float32)
                            elif weights_arr.size < span:
                                pad_val = float(weights_arr[-1])
                                weights_arr = np.pad(weights_arr, (0, span - weights_arr.size), constant_values=pad_val)
                            else:
                                weights_arr = weights_arr[:span]
                            gate_arr[offset:end] = weights_arr
                            role_name = entry.get("role_type", RoleType.SOURCE.value)
                            try:
                                role_id = ROLE_TYPE_TO_ID.get(RoleType(role_name), 0)
                            except ValueError:
                                role_id = ROLE_TYPE_TO_ID.get(RoleType.SOURCE, 0)
                            role_arr[offset:end] = role_id
                            reward_arr[offset:end] = float(entry.get("div_reward", 0.0))
                            offset = end
                        dynamic_edges, gate_arr, role_arr, reward_arr = _round_robin_select_edges(
                            dynamic_edges,
                            gate_arr,
                            role_arr,
                            reward_arr,
                            rel_metadata,
                            role_round_robin_fraction,
                        )
                        if dynamic_edges.size == 0:
                            continue
                        keep_edges = max(1, int(dynamic_edges.shape[0] * role_selection_fraction))
                        keep_edges = min(dynamic_edges.shape[0], keep_edges)
                        dynamic_edges = dynamic_edges[:keep_edges]
                        gate_arr = gate_arr[:keep_edges]
                        role_arr = role_arr[:keep_edges]
                        reward_arr = reward_arr[:keep_edges]
                        pretrain_steps = max(0, int(getattr(args, "role_pretrain_steps", 0)))
                        if pretrain_steps > 0:
                            pre_tensor = torch.LongTensor(dynamic_edges)
                            _adapt_role_edges(
                                model,
                                kgr_optimizer,
                                pre_tensor,
                                args,
                                device,
                                override_steps=pretrain_steps,
                                gate_tensor=torch.as_tensor(gate_arr, dtype=torch.float32),
                            )
                        max_rel_id = int(dynamic_edges[:, 1].max())
                        model.ensure_relation_capacity(max_rel_id + 1)
                        n_relations = max(n_relations, max_rel_id + 1)
                        _ensure_role_relations(model, dynamic_edges[:, 1])
                        cnt_role_rel = int(dynamic_edges.shape[0])
                        for start in range(0, dynamic_edges.shape[0], inj_chunk):
                            end = min(start + inj_chunk, dynamic_edges.shape[0])
                            chunk_tensor = torch.LongTensor(dynamic_edges[start:end])
                            rel_gate_slice = torch.as_tensor(gate_arr[start:end], dtype=torch.float32)
                            rel_meta = {
                                "gate": rel_gate_slice,
                                "role_type": torch.as_tensor(role_arr[start:end], dtype=torch.long),
                                "reward": torch.as_tensor(reward_arr[start:end], dtype=torch.float32),
                            }
                            model.append_dynamic_edges(chunk_tensor, meta=rel_meta)
                            _adapt_role_edges(
                                model,
                                kgr_optimizer,
                                chunk_tensor,
                                args,
                                device,
                                gate_tensor=rel_gate_slice,
                            )
                else:
                    if getattr(args, 'progress_verbose', True):
                        print(f"[Schedule] 预热期内跳过结构角色注入 (epoch={epoch} < warmup)")
            inj_end_t = time()
            if getattr(args, 'progress_verbose', True):
                new_relations = int(getattr(model, 'n_relations', 0))
                new_entities = int(getattr(model, 'n_entities', 0))
                d_rel = new_relations - old_relations
                d_ent = new_entities - old_entities
                total_injected = cnt_candi + cnt_role_rel + cnt_role_entity
                summary_msg = (
                    f"[注入统计] epoch={epoch} 激活角色={roles_active} 候选KG={cnt_candi} "
                    f"角色关系边={cnt_role_rel} 角色实体两跳边={cnt_role_entity} 总注入={total_injected} "
                    f"扩容: 关系+{d_rel} 实体+{d_ent} 用时={inj_end_t - inj_start_t:.2f}s 批大小={inj_chunk}"
                )
                _role_log(summary_msg)
            
        """training"""
        model.train()
        loss = 0
        train_s_t = time()

        for i in tqdm(range(iter)):
        # 避免在批次循环内频繁清理显存造成额外开销
            optimizer.zero_grad()
            batch = dict()
            batch['pos_index'] = all_feed_data['pos_index'][i * args.batch_size:(i + 1) * args.batch_size].to(device)
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)

            batch_loss, batch_mmd_loss = model(batch)
            loss_terms = []
            if batch_mmd_loss.requires_grad:
                loss_terms.append(batch_mmd_loss)
            if batch_loss.requires_grad:
                loss_terms.append(batch_loss)
            if len(loss_terms) == 1:
                loss_terms[0].backward()
            else:
                optimizer.pc_backward(loss_terms)
            optimizer.step()
            loss += batch_loss.item()

        # if epoch > 8:
        #     for param_group in optimizer.param_groups:
        #             param_group['lr'] = 0.0001

        train_e_t = time()
        # scheduler.step()                                         
        item_embs, KG_mask = model.generate_embeddings(for_kgc=True)
        item_embs = item_embs.detach().cpu().numpy()
        KG_mask = KG_mask.detach().cpu().numpy()
        # if epoch > 4 :
        """testing"""
        model.eval()
        test_s_t = time()
        with torch.no_grad():
            ret = test(model, user_dict, n_params)
        test_e_t = time()
        ret["kgc_acc"] = float(last_kgr_stats.get("kgc_acc", 0.0))
        ret["kgc_loss"] = float(last_kgr_stats.get("kgc_loss", 0.0))
        recall_value = ret["recall"][0] if isinstance(ret["recall"], (list, tuple, np.ndarray)) else ret["recall"]
        recall_history.append((epoch, float(recall_value)))
        if recall_history_cap > 0 and len(recall_history) > recall_history_cap:
            recall_history.pop(0)
        if last_injection_epoch is not None:
            if epoch <= last_injection_epoch:
                recall_decline_since_injection = 0
            else:
                if len(recall_history) >= 2:
                    prev_recall = recall_history[-2][1]
                    if recall_value + recall_drop_tolerance < prev_recall:
                        recall_decline_since_injection += 1
                    else:
                        recall_decline_since_injection = 0
            if (
                recall_decline_since_injection >= role_recall_decline_trigger
                and last_rollback_epoch != last_injection_epoch
                and last_injection_new_roles
            ):
                last_injection_new_roles, removed = _rollback_recent_new_roles(
                    last_injection_epoch,
                    last_injection_new_roles,
                )
                if removed:
                    last_rollback_epoch = last_injection_epoch
                    recall_decline_since_injection = 0
                    pause_new_roles = True
                    new_role_pause_reason = "rollback_guard"
                    pending_new_role_scales.clear()
            elif (
                recall_decline_since_injection == 1
                and last_injection_new_roles
                and last_minor_rollback_epoch != last_injection_epoch
                and role_minor_rollback_keep_ratio < 1.0
            ):
                last_injection_new_roles, removed = _rollback_recent_new_roles(
                    last_injection_epoch,
                    last_injection_new_roles,
                    keep_ratio=role_minor_rollback_keep_ratio,
                )
                if removed:
                    last_minor_rollback_epoch = last_injection_epoch
                    if getattr(args, 'progress_verbose', True):
                        _role_log(
                            f"[Role] minor rollback {removed} new roles from epoch {last_injection_epoch} "
                            f"(keep={role_minor_rollback_keep_ratio:.2f})"
                        )
        for k in best_metric.keys():
            metric_val = ret[k][0] if isinstance(ret[k], (list, tuple, np.ndarray)) else ret[k]
            if (
                (k in minimize_metrics and metric_val < best_metric[k])
                or (k not in minimize_metrics and metric_val > best_metric[k])
            ):
                best_metric[k] = metric_val
                best_epoch[k] = epoch
        if rsrg_trainer is not None:
            rsrg_trainer.register_feedback(ret)
        train_res = PrettyTable()
        train_res.field_names = [
            "Epoch", "training time", "tesing time", "Loss",
            "recall", "ndcg", "precision", "hit_ratio",
            "kgat_recall", "kgat_ndcg", "kgat_precision", "kgat_hit_ratio", "kgat_mrr", "kgat_auc",
            "kgat_coverage", "kgat_ad", "kgat_arp", "kgat_md", "kgat_mp", "kgat_hit", "kgat_ad2"
        ]
        train_res.add_row(
            [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss,
             ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'],
             ret['kgat_recall'], ret['kgat_ndcg'], ret['kgat_precision'], ret['kgat_hit_ratio'],
             ret['kgat_mrr'], ret['kgat_auc'],
             ret['kgat_coverage'], ret['kgat_ad'], ret['kgat_arp'], ret['kgat_md'],
             ret['kgat_mp'], ret['kgat_hit'], ret['kgat_ad2']]
        )
        
        f = open('./result/{}_exp_cxks_kg_xr_v2.txt'.format(args.dataset), 'a+')
        f.write(str(best_metric)+ '\n')
        f.write(str(best_epoch)+ '\n')
        f.write(str(train_res) + '\n')
        f.write('\n')
        f.close()

        # *********************************************************
        cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=20)
        
        if should_stop:
            break
        """save model"""
        if ret['recall'][0] == cur_best and args.save:
            torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')
            
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))






