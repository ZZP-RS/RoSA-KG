import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from .metrics import (
    AUC,
    F1,
    ndcg_at_k,
    kgat_map_at_k,
    kgat_mrr_at_k,
)
from .parser import parse_args
from metricskgat import (
    calc_coverage_at_k,
    calc_ad_at_k,
    calc_arp_at_k,
    calc_md_at_k_batch,
    calc_mp_at_k_batch,
    calc_hit_at_k_batch,
    calc_ad2_at_k_batch,
)


args = parse_args()
Ks = list(eval(args.Ks))
device = torch.device(f"cuda:{args.gpu_id}") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag
MAX_K = max(Ks) if Ks else 0


def _prepare_user_sets(user_dict):
    train_dict = user_dict["train_user_set"]
    test_dict = user_dict["test_user_set"]
    train_sets = {u: set(items) for u, items in train_dict.items()}
    test_sets = {u: set(items) for u, items in test_dict.items()}
    return train_dict, test_dict, train_sets, test_sets


def _build_popularity(train_dict):
    item_popularity = defaultdict(int)
    train_item_dict = defaultdict(list)
    for user, items in train_dict.items():
        for item in items:
            item_popularity[item] += 1
            train_item_dict[item].append(user)
    return item_popularity, train_item_dict


def _gather_scores(model, entity_emb, user_emb_batch, n_items):
    if batch_test_flag:
        scores = torch.empty((len(user_emb_batch), n_items), device=device)
        num_batches = n_items // BATCH_SIZE + 1
        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, n_items)
            if start >= end:
                break
            item_idx = torch.arange(start, end, device=device)
            item_emb = entity_emb[item_idx]
            batch_scores = torch.matmul(user_emb_batch, item_emb.t())
            scores[:, start:end] = batch_scores
    else:
        scores = torch.matmul(user_emb_batch, entity_emb.t())
    return scores


def test(model, user_dict, n_params):
    result = {
        "precision": np.zeros(len(Ks), dtype=np.float32),
        "recall": np.zeros(len(Ks), dtype=np.float32),
        "ndcg": np.zeros(len(Ks), dtype=np.float32),
        "hit_ratio": np.zeros(len(Ks), dtype=np.float32),
        "auc": 0.0,
        "kgat_precision": np.zeros(len(Ks), dtype=np.float32),
        "kgat_recall": np.zeros(len(Ks), dtype=np.float32),
        "kgat_ndcg": np.zeros(len(Ks), dtype=np.float32),
        "kgat_hit_ratio": np.zeros(len(Ks), dtype=np.float32),
        "kgat_f1": np.zeros(len(Ks), dtype=np.float32),
        "kgat_map": np.zeros(len(Ks), dtype=np.float32),
        "kgat_mrr": np.zeros(len(Ks), dtype=np.float32),
        "kgat_auc": np.zeros(len(Ks), dtype=np.float32),
        "kgat_coverage": np.zeros(len(Ks), dtype=np.float32),
        "kgat_ad": np.zeros(len(Ks), dtype=np.float32),
        "kgat_arp": np.zeros(len(Ks), dtype=np.float32),
        "kgat_md": np.zeros(len(Ks), dtype=np.float32),
        "kgat_mp": np.zeros(len(Ks), dtype=np.float32),
        "kgat_hit": np.zeros(len(Ks), dtype=np.float32),
        "kgat_ad2": np.zeros(len(Ks), dtype=np.float32),
    }

    train_dict, test_dict, train_sets, test_sets = _prepare_user_sets(user_dict)
    item_popularity, train_item_dict = _build_popularity(train_dict)

    n_items = n_params["n_items"]

    entity_emb, user_emb = model.generate_embeddings()
    entity_emb = entity_emb.to(device)
    user_emb = user_emb.to(device)

    test_users = list(test_dict.keys())
    n_test_users = len(test_users)
    if n_test_users == 0 or MAX_K == 0:
        return result

    train_index_map = {
        u: torch.tensor(list(items), dtype=torch.long, device=device)
        for u, items in train_sets.items()
        if items
    }
    test_pos_arrays = {
        u: torch.tensor(list(items), dtype=torch.long, device=device)
        for u, items in test_sets.items()
    }

    precision_sum = np.zeros(len(Ks), dtype=np.float64)
    recall_sum = np.zeros(len(Ks), dtype=np.float64)
    recall_count = np.zeros(len(Ks), dtype=np.int64)
    ndcg_sum = np.zeros(len(Ks), dtype=np.float64)
    ndcg_count = np.zeros(len(Ks), dtype=np.int64)
    hit_ratio_sum = np.zeros(len(Ks), dtype=np.float64)
    auc_full_sum = 0.0
    auc_full_count = 0

    kgat_precision_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_recall_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_ndcg_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_hit_ratio_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_f1_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_map_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_mrr_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_auc_sum = np.zeros(len(Ks), dtype=np.float64)
    kgat_auc_count = np.zeros(len(Ks), dtype=np.int64)
    total_positive = 0.0

    unique_items_per_k = {K: set() for K in Ks}
    item_counts_per_k = {K: defaultdict(int) for K in Ks}
    pair_sum_per_k = {K: 0.0 for K in Ks}
    arp_sum_per_k = {K: 0.0 for K in Ks}
    mp_sum_per_k = {K: 0.0 for K in Ks}
    kgat_hit_sum_per_k = {K: 0.0 for K in Ks}
    total_test_items_count = 0.0

    for start in tqdm(range(0, n_test_users, BATCH_SIZE), desc="Evaluating"):
        end = min(start + BATCH_SIZE, n_test_users)
        user_batch = test_users[start:end]
        if not user_batch:
            continue
        user_tensor = torch.tensor(user_batch, dtype=torch.long, device=device)
        user_emb_batch = user_emb[user_tensor]

        scores_gpu = _gather_scores(model, entity_emb, user_emb_batch, n_items)

        for row, uid in enumerate(user_batch):
            train_idx = train_index_map.get(uid)
            if train_idx is not None and train_idx.numel() > 0:
                scores_gpu[row, train_idx] = float("-inf")

        top_scores_gpu, top_idx_gpu = torch.topk(scores_gpu, MAX_K, dim=1)
        top_scores_np = top_scores_gpu.cpu().numpy()
        top_idx_np = top_idx_gpu.cpu().numpy()

        for row, uid in enumerate(user_batch):
            positives_tensor = test_pos_arrays.get(uid)
            if positives_tensor is None or positives_tensor.numel() == 0:
                continue
            positives_set = test_sets[uid]
            pos_len = positives_tensor.numel()
            total_positive += pos_len
            total_test_items_count += pos_len

            top_items_gpu = top_idx_gpu[row]
            hits_tensor = torch.isin(top_items_gpu, positives_tensor)
            hits_row = hits_tensor.to(torch.float32).cpu().numpy()
            top_items = top_items_gpu.cpu().numpy()

            scores_row = scores_gpu[row].cpu().numpy()
            valid_mask = np.isfinite(scores_row)
            if valid_mask.any():
                valid_items = np.nonzero(valid_mask)[0]
                valid_scores = scores_row[valid_mask]
                positives_array = positives_tensor.cpu().numpy()
                valid_labels = np.isin(valid_items, positives_array, assume_unique=False).astype(np.int8)
                if 0 < valid_labels.sum() < valid_labels.size:
                    auc_full_sum += AUC(valid_labels.tolist(), valid_scores.tolist())
                else:
                    auc_full_sum += 0.0
            auc_full_count += 1

            hits_list = hits_row.tolist()
            for idx, K in enumerate(Ks):
                hits_k = hits_list[:K]
                hit_count = float(sum(hits_k))
                precision_val = hit_count / K
                precision_sum[idx] += precision_val
                kgat_precision_sum[idx] += precision_val
                hit_ratio_sum[idx] += 1.0 if hit_count > 0 else 0.0
                kgat_hit_ratio_sum[idx] += 1.0 if hit_count > 0 else 0.0
                kgat_hit_sum_per_k[K] += hit_count

                recall_val = hit_count / pos_len
                recall_sum[idx] += recall_val
                recall_count[idx] += 1
                kgat_recall_sum[idx] += recall_val

                ndcg_val = ndcg_at_k(hits_k, K, positives_set)
                ndcg_sum[idx] += ndcg_val
                ndcg_count[idx] += 1
                kgat_ndcg_sum[idx] += ndcg_val

                kgat_f1_sum[idx] += F1(precision_val, recall_val)
                kgat_map_sum[idx] += kgat_map_at_k(hits_list, K)
                kgat_mrr_sum[idx] += kgat_mrr_at_k(hits_list, K)

                kgat_auc_val = AUC(hits_k, top_scores_np[row, :K].tolist())
                kgat_auc_sum[idx] += kgat_auc_val
                kgat_auc_count[idx] += 1

                subset_items = top_items[:K]
                if subset_items.size:
                    subset_list = subset_items.tolist()
                    unique_items_per_k[K].update(subset_list)
                    for item in subset_list:
                        prev = item_counts_per_k[K][item]
                        new = prev + 1
                        item_counts_per_k[K][item] = new
                        pair_sum_per_k[K] += (new * (new - 1) / 2.0) - (prev * (prev - 1) / 2.0)
                    if K > 0:
                        arp_sum_per_k[K] += sum(item_popularity.get(item, 0) for item in subset_list) / K
                        mp_sum_per_k[K] += sum(len(train_item_dict.get(item, [])) for item in subset_list) / K

    for idx, K in enumerate(Ks):
        result["precision"][idx] = precision_sum[idx] / n_test_users
        result["hit_ratio"][idx] = hit_ratio_sum[idx] / n_test_users
        result["recall"][idx] = recall_sum[idx] / recall_count[idx] if recall_count[idx] else 0.0
        result["ndcg"][idx] = ndcg_sum[idx] / ndcg_count[idx] if ndcg_count[idx] else 0.0

        result["kgat_precision"][idx] = kgat_precision_sum[idx] / n_test_users
        result["kgat_hit_ratio"][idx] = kgat_hit_ratio_sum[idx] / n_test_users
        result["kgat_recall"][idx] = kgat_recall_sum[idx] / recall_count[idx] if recall_count[idx] else 0.0
        result["kgat_ndcg"][idx] = kgat_ndcg_sum[idx] / ndcg_count[idx] if ndcg_count[idx] else 0.0
        result["kgat_f1"][idx] = kgat_f1_sum[idx] / recall_count[idx] if recall_count[idx] else 0.0
        result["kgat_map"][idx] = kgat_map_sum[idx] / recall_count[idx] if recall_count[idx] else 0.0
        result["kgat_mrr"][idx] = kgat_mrr_sum[idx] / n_test_users
        result["kgat_auc"][idx] = kgat_auc_sum[idx] / kgat_auc_count[idx] if kgat_auc_count[idx] else 0.0
        result["kgat_hit"][idx] = kgat_hit_sum_per_k[K] / total_test_items_count if total_test_items_count else 0.0

        unique_items = unique_items_per_k[K]
        coverage = len(unique_items)
        result["kgat_coverage"][idx] = coverage / n_items if n_items > 0 else 0.0
        result["kgat_ad"][idx] = coverage / (n_test_users * K) if K > 0 and n_test_users > 0 else 0.0
        result["kgat_arp"][idx] = arp_sum_per_k[K] / n_test_users if n_test_users else 0.0
        result["kgat_mp"][idx] = mp_sum_per_k[K] / n_test_users if n_test_users else 0.0
        if n_test_users > 1 and K > 0:
            pair_sum = pair_sum_per_k[K]
            result["kgat_md"][idx] = 1 - (2 * pair_sum) / (K * n_test_users * (n_test_users - 1))
        else:
            result["kgat_md"][idx] = 1.0
        result["kgat_ad2"][idx] = float(coverage)

    result["auc"] = auc_full_sum / auc_full_count if auc_full_count else 0.0
    return result
