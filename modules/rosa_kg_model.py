# RoSA-KG main model (GCN backbone)
# This implementation is adapted from EditKG: Editing Knowledge Graph for Recommendation
# (SIGIR 2024) while integrating rule-driven structural roles and multi-view structural reliability.
# For the original EditKG method, please cite:
# @inproceedings{tang2024editkg,
#   title={Editkg: Editing knowledge graph for recommendation},
#   author={Tang, Gu and Gan, Xiaoying and Wang, Jinghe and Lu, Bin and Wu, Lyuwen and Fu, Luoyi and Zhou, Chenghu},
#   booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
#   pages={112--122},
#   year={2024}
# }


import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from typing import Sequence, Set, Dict, Optional, Mapping, List
from modules.rule_roles.role_schema import RoleType


def _copy_prefix(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy overlapping prefix (along each dimension) from src to dst."""
    slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape))
    dst[slices] = src[slices]
    return dst


def _replace_parameter(old_param: nn.Parameter, new_data: torch.Tensor,
                       optimizers: Sequence[Optimizer]) -> nn.Parameter:
    """Replace a parameter with new data while keeping optimizer states aligned."""
    device = old_param.device
    dtype = old_param.dtype
    new_param = nn.Parameter(new_data.to(device=device, dtype=dtype))
    new_param.requires_grad = old_param.requires_grad

    if old_param.grad is not None:
        new_grad = old_param.grad.new_zeros(new_param.shape)
        _copy_prefix(old_param.grad, new_grad)
        new_param.grad = new_grad

    for opt in optimizers or []:
        if not isinstance(opt, Optimizer):
            continue
        replaced = False
        for group in opt.param_groups:
            for idx, param in enumerate(group['params']):
                if param is old_param:
                    group['params'][idx] = new_param
                    replaced = True
        if not replaced:
            continue
        state = opt.state.pop(old_param, None)
        if state is None:
            opt.state[new_param] = {}
            continue
        new_state = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if value.dim() == 0:
                    new_state[key] = value.clone()
                else:
                    tensor = value.new_zeros(new_param.shape)
                    _copy_prefix(value, tensor)
                    new_state[key] = tensor
            else:
                new_state[key] = value
        opt.state[new_param] = new_state

    return new_param


def _expand_parameter(old_param: nn.Parameter, pad_rows: torch.Tensor,
                      optimizers: Sequence[Optimizer]) -> nn.Parameter:
    """Expand a parameter along dim-0 with pad_rows, returning the new Parameter."""
    if pad_rows is None or pad_rows.numel() == 0:
        return old_param
    with torch.no_grad():
        new_data = torch.cat(
            [old_param.data, pad_rows.to(device=old_param.device, dtype=old_param.dtype)],
            dim=0,
        )
    return _replace_parameter(old_param, new_data, optimizers)


def _reserve_capacity(current: int, required: int, growth: float = 0.2,
                      minimum_step: int = 128, align: int = 64) -> int:
    """Compute expanded capacity with modest reserve to reduce realloc frequency."""
    current = int(max(0, current))
    required = int(max(0, required))
    if required <= current:
        return current
    need = required - current
    reserve = max(int(math.ceil(need * growth)), minimum_step)
    target = current + need + reserve
    if align > 1:
        target = ((target + align - 1) // align) * align
    return target

class SelectAgent(nn.Module):
    def __init__(self, dim,temperature):
        super(SelectAgent,self).__init__()
        self.dim = dim
        # self.ln = nn.LayerNorm(256)
        self.temperature = temperature
        self.select_linear_1 = nn.Linear(self.dim,512)
        self.select_linear_2 = nn.Linear(512,256)
        self.select_linear_3 = nn.Linear(256,1)
        
    def forward(self,x,use_ln=False):
        x = torch.relu(self.select_linear_1(x))
        x =torch.relu(self.select_linear_2(x))
        # if use_ln:
        #     x = self.ln(x) + x
        x = self.select_linear_3(x)
        return x


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items, n_entity, n_relation, gamma, max_iter,
                 edge_chunk_size=131072, edge_sample_rate=1.0):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = gamma
        self.max_iter = int(max_iter)
        self.dim = 128

        # self.LN = nn.LayerNorm(64)
        self.activation = nn.LeakyReLU()
        self.edge_chunk_size = max(1, int(edge_chunk_size))
        self.edge_sample_rate = float(edge_sample_rate)

    def gumbel_process(self,action_prob, tau=1, dim=-1, hard=True):
        gumbels = (
            -torch.empty_like(action_prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)

        gumbels = (action_prob + 0.5 * gumbels)/tau

        y_soft = gumbels.softmax(dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(action_prob, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    
    def cosin_smi(self,a,b):
        if len(a.shape) != 3:
            a = a.unsqueeze(1)
            b = b.unsqueeze(1)
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-5)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-5)
        return torch.matmul(a_norm,b_norm.transpose(1,2))

    def _half_mask(self,a,b):
        ab_sig = torch.sigmoid(torch.rand(a.shape)).to(a.device)
        ab_mask = torch.bernoulli(ab_sig)
        ab_mask_rev = (ab_mask<1).float()
        mask_a = a * ab_mask
        mask_b = b * ab_mask_rev
        return mask_a, mask_b


    def forward(self, user_emb, item_emb, interact_mat, edge_sample_rate=None, edge_chunk_size=None):
        indices = interact_mat._indices()
        values = interact_mat._values()
        nnz = indices.size(1)

        sample_rate = self.edge_sample_rate if edge_sample_rate is None else float(edge_sample_rate)
        sample_rate = min(max(sample_rate, 0.0), 1.0)
        if sample_rate < 1.0 and nnz > 0:
            sample_size = max(1, int(nnz * sample_rate))
            perm = torch.randperm(nnz, device=indices.device)[:sample_size]
            indices = indices[:, perm]
            values = values[perm]

        rows = indices[0]
        cols = indices[1]
        if values.numel() == 0:
            weights = torch.ones(rows.size(0), device=item_emb.device, dtype=item_emb.dtype)
        else:
            weights = values.to(item_emb.dtype)

        chunk = self.edge_chunk_size if edge_chunk_size is None else max(1, int(edge_chunk_size))
        user_out = item_emb.new_zeros((self.n_users, item_emb.size(1)))
        item_out = user_emb.new_zeros((self.n_items, user_emb.size(1)))

        for start in range(0, rows.size(0), chunk):
            end = min(start + chunk, rows.size(0))
            r = rows[start:end]
            c = cols[start:end]
            w = weights[start:end].unsqueeze(-1)

            user_out.index_add_(0, r, item_emb[c] * w)
            item_out.index_add_(0, c, user_emb[r] * w)

        return user_out, item_out


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users,
                 n_items, n_entities, n_relations, interact_mat, gamma, max_iter,
                 device, node_dropout_rate=0.5, mess_dropout_rate=0.1,
                 kg_edge_chunk_size=1024, ui_edge_chunk_size=1024,
                 kg_edge_sample_rate=0.2, ui_edge_sample_rate=0.2,
                 mmd_estimator='linear', mmd_bandwidth_sample=1024, mmd_batch_size=1024,
                 role_mix_coeff=1.0, new_edge_threshold=0.1, new_kg_edge_sample_rate=None,
                 role_bridge_coeff=0.0, role_bridge_cap=1.0,
                 mmd_weight=1.0, role_relation_reg=1e-2, optimizer_refs=None,
                 kgc_align_weight=0.0):
        super(GraphConv, self).__init__()
        self.channel = channel
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.n_entity = n_entities
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.act_func = nn.LeakyReLU()
        self.Select_agent = SelectAgent(channel * 3 ,1)
        self.N_Select_agent = SelectAgent(channel * 3 ,1)

        self.bce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)
        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  
        self.relation_weight = nn.Parameter(relation_weight)  
        n_relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations, channel))  
        self.n_relation_weight = nn.Parameter(n_relation_weight)  
        
        self.kgc = KGC(n_items=self.n_items,num_ent=self.n_entity,num_rel=self.n_relations,dim=channel)
        for i in range(n_hops):
            self.convs.append(
                Aggregator(n_users=n_users,
                           n_items=n_items,
                           n_entity=n_entities,
                           n_relation=n_relations,
                           gamma=gamma,
                           max_iter=max_iter,
                           edge_chunk_size=ui_edge_chunk_size,
                           edge_sample_rate=ui_edge_sample_rate).to(self.device)
            )

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.kg_edge_chunk_size = max(1, int(kg_edge_chunk_size))
        self.ui_edge_chunk_size = max(1, int(ui_edge_chunk_size))
        self.kg_edge_sample_rate = min(max(float(kg_edge_sample_rate), 0.0), 1.0)
        self.ui_edge_sample_rate = min(max(float(ui_edge_sample_rate), 0.0), 1.0)
        self.mmd_estimator = str(mmd_estimator).lower()
        if self.mmd_estimator not in {"linear", "full"}:
            self.mmd_estimator = "linear"
        self.mmd_bandwidth_sample = max(2, int(mmd_bandwidth_sample))
        self.mmd_batch_size = max(2, int(mmd_batch_size))
        # role injection knobs (can be updated at runtime)
        self.role_mix_coeff = float(role_mix_coeff)
        self.new_edge_threshold = float(new_edge_threshold)
        self.base_edge_threshold = 0.1
        self.new_kg_edge_sample_rate = float(new_kg_edge_sample_rate) if new_kg_edge_sample_rate is not None else self.kg_edge_sample_rate
        self.mmd_weight = float(mmd_weight)
        self.role_bridge_cap = max(0.0, float(role_bridge_cap))
        self.role_bridge_coeff = float(min(max(role_bridge_coeff, 0.0), self.role_bridge_cap))
        self.kgc_align_weight = max(0.0, float(kgc_align_weight))
        self.base_relation_count = int(n_relations)
        self.role_relation_indices: Set[int] = set()
        self.role_relation_reg = float(role_relation_reg)
        self.current_role_fraction = 0.0
        self.current_relation_fraction = 0.0
        self._optimizer_refs = optimizer_refs if optimizer_refs is not None else []
        self.kgc.set_optimizer_refs(self._optimizer_refs)
        self.aug_edge_index: Optional[torch.Tensor] = None
        self.aug_edge_type: Optional[torch.Tensor] = None
        self.aug_edge_gate: Optional[torch.Tensor] = None
        self.role_edge_index: Optional[torch.Tensor] = None
        self.role_edge_type: Optional[torch.Tensor] = None
        self.role_edge_gate: Optional[torch.Tensor] = None
        self.role_edge_role: Optional[torch.Tensor] = None
        self.role_edge_reward: Optional[torch.Tensor] = None
        self.role_type_gate = nn.Parameter(torch.zeros(len(RoleType), dtype=torch.float32))
        self.reward_gate = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.degree_gate = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.gate_floor = 1e-4
        self.role_item_mix: Optional[torch.Tensor] = None
        if hasattr(self.kgc, "set_align_weight"):
            self.kgc.set_align_weight(self.kgc_align_weight)
        self.role_bridge_linear = nn.Linear(self.channel * 2, self.channel, bias=False)
        nn.init.xavier_uniform_(self.role_bridge_linear.weight)
        self.role_user_linear = nn.Linear(self.channel, self.channel * 3, bias=False)
        nn.init.xavier_uniform_(self.role_user_linear.weight)

    def expand_relation_capacity(self, required_relations: int):
        target = _reserve_capacity(
            self.n_relations,
            required_relations,
            growth=0.2,
            minimum_step=64,
            align=32,
        )
        if target <= self.n_relations:
            return
        device = self.relation_weight.device
        channel = self.relation_weight.size(1)
        extra = target - self.n_relations
        with torch.no_grad():
            add_main = torch.empty((extra, channel), device=device, dtype=self.relation_weight.dtype)
            nn.init.xavier_uniform_(add_main)
            add_new = torch.empty((extra, channel), device=device, dtype=self.n_relation_weight.dtype)
            nn.init.xavier_uniform_(add_new)
        self.relation_weight = _expand_parameter(self.relation_weight, add_main, self._optimizer_refs)
        self.n_relation_weight = _expand_parameter(self.n_relation_weight, add_new, self._optimizer_refs)
        self.n_relations = target
        self.kgc.expand_relations(target)

    def register_role_relations(self, indices: Sequence[int]):
        if not indices:
            return
        new_indices = [int(i) for i in indices if int(i) not in self.role_relation_indices]
        if not new_indices:
            return
        device = self.relation_weight.device
        base_count = max(1, min(self.base_relation_count, self.relation_weight.size(0)))
        with torch.no_grad():
            template_main = self.relation_weight[:base_count].mean(dim=0, keepdim=True)
            template_new = self.n_relation_weight[:base_count].mean(dim=0, keepdim=True)
            idx_tensor = torch.tensor(new_indices, dtype=torch.long, device=device)
            expand_main = template_main.expand(idx_tensor.size(0), -1)
            expand_new = template_new.expand(idx_tensor.size(0), -1)
            self.relation_weight.data[idx_tensor] = expand_main
            self.n_relation_weight.data[idx_tensor] = expand_new
            if hasattr(self.kgc, "rel_embeddings"):
                kgc_weight = self.kgc.rel_embeddings.weight.to(device=device, dtype=self.relation_weight.dtype)
                kgc_base_count = min(base_count, kgc_weight.size(0))
                if kgc_base_count > 0:
                    kgc_template = kgc_weight[:kgc_base_count].mean(dim=0, keepdim=True)
                    kgc_expand = kgc_template.expand(idx_tensor.size(0), -1)
                    self.kgc.rel_embeddings.weight.data[idx_tensor] = kgc_expand.to(self.kgc.rel_embeddings.weight.dtype, copy=False)
        self.role_relation_indices.update(new_indices)

    def expand_entity_capacity(self, required_entities: int):
        target = _reserve_capacity(
            self.n_entity,
            required_entities,
            growth=0.2,
            minimum_step=256,
            align=64,
        )
        if target <= self.n_entity:
            return
        # Update internal counters and expand KGC entity embeddings
        self.kgc.expand_entities(target)
        self.n_entity = target

    def update_role_mix(self, scores: Mapping[int, float], momentum: float = 0.5) -> None:
        if not scores:
            return
        momentum = float(min(max(momentum, 0.0), 0.999))
        device = self.relation_weight.device
        dtype = self.relation_weight.dtype
        if self.role_item_mix is None or self.role_item_mix.size(0) != self.n_items:
            base = torch.full((self.n_items, 1), self.role_mix_coeff, device=device, dtype=dtype)
            self.role_item_mix = base
        mix_tensor = self.role_item_mix
        keys = list(scores.keys())
        if not keys:
            return
        idx_tensor = torch.as_tensor(keys, dtype=torch.long, device=device)
        val_tensor = torch.as_tensor([scores[int(k)] for k in keys], dtype=dtype, device=device).unsqueeze(-1)
        valid_mask = (idx_tensor >= 0) & (idx_tensor < self.n_items)
        idx_tensor = idx_tensor[valid_mask]
        val_tensor = val_tensor[valid_mask]
        if idx_tensor.numel() == 0:
            return
        current = mix_tensor.index_select(0, idx_tensor)
        updated = momentum * current + (1.0 - momentum) * val_tensor
        mix_tensor.index_copy_(0, idx_tensor, updated)
        mix_tensor.clamp_(0.0, 2.0)

    def sync_role_relation_weights(self, relation_ids: Optional[Sequence[int]] = None) -> None:
        if self.role_relation_indices is None or not self.role_relation_indices:
            return
        device = self.relation_weight.device
        if relation_ids is None:
            relation_ids = sorted(self.role_relation_indices)
        if not relation_ids:
            return
        idx_tensor = torch.tensor([int(i) for i in relation_ids if int(i) < self.n_relations], dtype=torch.long, device=device)
        if idx_tensor.numel() == 0:
            return
        with torch.no_grad():
            kgc_weights = self.kgc.rel_embeddings.weight.to(device=self.relation_weight.device, dtype=self.relation_weight.dtype)
            self.n_relation_weight[idx_tensor] = kgc_weights[idx_tensor]

    def _apply_role_bridge(
        self,
        base_entity: torch.Tensor,
        role_entity: torch.Tensor,
        role_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        coeff = float(self.role_bridge_coeff)
        if coeff <= 1e-6:
            return base_entity
        merged = torch.cat([base_entity, role_entity], dim=-1)
        delta = self.role_bridge_linear(merged)
        if role_gate is not None:
            gate = torch.tanh(role_gate).clamp(0.0, 1.0)
            delta = delta * gate
        return base_entity + coeff * delta

    def _role_signal_for_users(
        self,
        role_gate: Optional[torch.Tensor],
        role_entities: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if role_gate is None or role_gate.numel() == 0:
            return None
        if role_entities.size(0) < self.n_items:
            return None
        item_gate = role_gate[: self.n_items]
        total = float(item_gate.sum().item())
        if total <= 1e-8:
            return None
        weighted = (role_entities[: self.n_items] * item_gate).sum(dim=0, keepdim=True) / total
        return self.role_user_linear(weighted)

    def set_role_bridge_coeff(self, value: float) -> None:
        value = float(min(max(value, 0.0), self.role_bridge_cap))
        self.role_bridge_coeff = value


    def set_optimizer_refs(self, refs: Sequence[Optimizer]) -> None:
        self._optimizer_refs = refs if refs is not None else []
        self.kgc.set_optimizer_refs(self._optimizer_refs)

    def _update_knowledge(self,two_hpo_kg):
        self.aug_edge_index = two_hpo_kg[:,[0,-1]].transpose(1,0)
        self.aug_edge_type = two_hpo_kg[:,1]

    # New: reset/append APIs to allow batched edge updates without building a giant tensor at once
    def reset_dynamic_edges(self):
        self.aug_edge_index = None
        self.aug_edge_type = None
        self.aug_edge_gate = None
        self.role_edge_index = None
        self.role_edge_type = None
        self.role_edge_gate = None
        self.role_edge_role = None
        self.role_edge_reward = None

    def append_dynamic_edges(self, two_hpo_kg_chunk: torch.Tensor, meta: Optional[Dict[str, torch.Tensor]] = None):
        if two_hpo_kg_chunk.numel() == 0:
            return
        device = self.relation_weight.device
        chunk = two_hpo_kg_chunk.to(device=device, dtype=torch.long, non_blocking=True)
        idx = chunk[:, [0, -1]].transpose(1, 0)
        typ = chunk[:, 1]
        edge_count = chunk.size(0)
        if meta is None:
            gate_tensor = torch.ones(edge_count, device=device, dtype=torch.float32)
            if self.aug_edge_index is None or self.aug_edge_type is None:
                self.aug_edge_index = idx
                self.aug_edge_type = typ
                self.aug_edge_gate = gate_tensor
            else:
                self.aug_edge_index = torch.cat([self.aug_edge_index, idx], dim=1)
                self.aug_edge_type = torch.cat([self.aug_edge_type, typ], dim=0)
                self.aug_edge_gate = (
                    torch.cat([self.aug_edge_gate, gate_tensor], dim=0)
                    if self.aug_edge_gate is not None
                    else gate_tensor
                )
        else:
            gate_tensor = meta.get("gate")
            if gate_tensor is None:
                gate_tensor = torch.ones(edge_count, device=device, dtype=torch.float32)
            else:
                gate_tensor = gate_tensor.to(device=device, dtype=torch.float32).reshape(-1)
            role_tensor = meta.get("role_type")
            if role_tensor is None:
                role_tensor = torch.zeros(edge_count, device=device, dtype=torch.long)
            else:
                role_tensor = role_tensor.to(device=device, dtype=torch.long).reshape(-1)
            reward_tensor = meta.get("reward")
            if reward_tensor is None:
                reward_tensor = torch.zeros(edge_count, device=device, dtype=torch.float32)
            else:
                reward_tensor = reward_tensor.to(device=device, dtype=torch.float32).reshape(-1)
            if self.role_edge_index is None or self.role_edge_type is None:
                self.role_edge_index = idx
                self.role_edge_type = typ
                self.role_edge_gate = gate_tensor
                self.role_edge_role = role_tensor
                self.role_edge_reward = reward_tensor
            else:
                self.role_edge_index = torch.cat([self.role_edge_index, idx], dim=1)
                self.role_edge_type = torch.cat([self.role_edge_type, typ], dim=0)
                self.role_edge_gate = (
                    torch.cat([self.role_edge_gate, gate_tensor], dim=0)
                    if self.role_edge_gate is not None
                    else gate_tensor
                )
                self.role_edge_role = (
                    torch.cat([self.role_edge_role, role_tensor], dim=0)
                    if self.role_edge_role is not None
                    else role_tensor
                )
                self.role_edge_reward = (
                    torch.cat([self.role_edge_reward, reward_tensor], dim=0)
                    if self.role_edge_reward is not None
                    else reward_tensor
                )
        
    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _edge_sampling_01(self, edge_index, edge_type, rate=0.5):
        n_edges = edge_index.shape[1]
        m = np.random.choice([0, 1], size=n_edges, p=[0.0, 1.0])
        return m

    def _maybe_sample_edges(
        self,
        edge_index,
        edge_type,
        sample_rate,
        edge_gate: Optional[torch.Tensor] = None,
        edge_role: Optional[torch.Tensor] = None,
        edge_reward: Optional[torch.Tensor] = None,
    ):
        if edge_index is None or edge_type is None:
            return None, None, None, None, None, None
        sample_rate = min(max(float(sample_rate), 0.0), 1.0)
        num_edges = edge_index.size(1)
        if sample_rate >= 1.0 or num_edges == 0:
            return edge_index, edge_type, edge_gate, edge_role, edge_reward, None
        sample_size = max(1, int(num_edges * sample_rate))
        perm = torch.randperm(num_edges, device=edge_index.device)[:sample_size]
        gate_slice = edge_gate[perm] if edge_gate is not None else None
        role_slice = edge_role[perm] if edge_role is not None else None
        reward_slice = edge_reward[perm] if edge_reward is not None else None
        return edge_index[:, perm], edge_type[perm], gate_slice, role_slice, reward_slice, perm

    def _kernel_sum(self, left, right, bandwidth, chunk):
        total = left.new_tensor(0.0)
        bw = bandwidth
        for start in range(0, left.size(0), chunk):
            end = min(start + chunk, left.size(0))
            left_chunk = left[start:end]
            diff = left_chunk.unsqueeze(1) - right.unsqueeze(0)
            k_vals = torch.exp(-diff.pow(2).sum(dim=2) / bw)
            total = total + k_vals.sum()
        return total

    def _estimate_bandwidths(self, samples, kernel_mul, kernel_num):
        total = samples
        sample_size = min(self.mmd_bandwidth_sample, total.size(0))
        if sample_size < 2:
            base = torch.tensor(1.0, device=total.device, dtype=total.dtype)
        else:
            perm = torch.randperm(total.size(0), device=total.device)[:sample_size]
            subset = total[perm]
            dist = torch.cdist(subset, subset, p=2).pow(2)
            denom = sample_size * sample_size - sample_size
            if denom <= 0:
                base = torch.tensor(1.0, device=total.device, dtype=total.dtype)
            else:
                base = dist.sum() / float(denom)
        base = base / (kernel_mul ** (kernel_num // 2))
        base = torch.clamp(base, min=1e-6)
        return [base * (kernel_mul ** i) for i in range(kernel_num)]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()                                       
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)        
        dropout_mask = torch.floor(random_tensor).type(torch.bool)   
        i = x._indices()                                             
        v = x._values()                                              

        i = i[:, dropout_mask]                                       
        v = v[dropout_mask]                                         

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)   
        # return out * (1. / (1 - rate))                              
        return out

    def Gumbel_process(self,action_prob, tau=1, dim=-1, hard=True, threshold=None):
        gumbels = (
            -torch.empty_like(action_prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        
        gumbels = (action_prob +  gumbels)/tau
        y_soft = torch.sigmoid(gumbels)
        thr = self.base_edge_threshold if threshold is None else float(threshold)
        y_hard = (y_soft > thr).float()

        return y_soft,y_hard
    
    def Dnoise_KG(
        self,
        edge_index,
        edge_type,
        entity_emb,
        relation_weight,
        is_gumble=True,
        new=False,
        edge_sample_rate=1.0,
        edge_gate: Optional[torch.Tensor] = None,
        edge_role: Optional[torch.Tensor] = None,
        edge_reward: Optional[torch.Tensor] = None,
    ):
        sampled_edge_index, sampled_edge_type, sampled_gate, sampled_role, sampled_reward, _ = self._maybe_sample_edges(
            edge_index,
            edge_type,
            edge_sample_rate if self.training else 1.0,
            edge_gate=edge_gate,
            edge_role=edge_role,
            edge_reward=edge_reward,
        )
        if sampled_edge_index is None:
            empty = entity_emb.new_zeros((0, 1))
            return empty, empty, edge_index, edge_type, sampled_gate, sampled_role, sampled_reward

        head, tail = sampled_edge_index
        num_edges = head.size(0)
        if num_edges == 0:
            empty = entity_emb.new_zeros((0, 1))
            return empty, empty, sampled_edge_index, sampled_edge_type, sampled_gate, sampled_role, sampled_reward

        agent = self.N_Select_agent if new else self.Select_agent
        chunk = self.kg_edge_chunk_size
        action_soft = entity_emb.new_empty((num_edges, 1))
        action_hard = entity_emb.new_empty((num_edges, 1))

        for start in range(0, num_edges, chunk):
            end = min(start + chunk, num_edges)
            head_emb = entity_emb[head[start:end]]
            tail_emb = entity_emb[tail[start:end]]
            rel_emb = relation_weight[sampled_edge_type[start:end]]
            h_r_t_emb = torch.cat([head_emb, rel_emb, tail_emb], dim=-1)
            h_r_t_emb = F.normalize(h_r_t_emb, dim=-1)
            action_prob = agent(h_r_t_emb)

            if is_gumble:
                # Use stricter or relaxed threshold depending on branch (new vs. base)
                thr = self.new_edge_threshold if new else self.base_edge_threshold
                soft, hard = self.Gumbel_process(action_prob, tau=1, dim=-1, hard=True, threshold=thr)
            else:
                soft = torch.sigmoid(action_prob)
                thr = self.new_edge_threshold if new else self.base_edge_threshold
                hard = (soft > thr).float()

            action_soft[start:end] = soft
            action_hard[start:end] = hard

        return (
            action_soft,
            action_hard,
            sampled_edge_index,
            sampled_edge_type,
            sampled_gate,
            sampled_role,
            sampled_reward,
        )

    def KG_forward(
        self,
        entity_emb,
        edge_index,
        edge_type,
        relation_weight,
        KG_drop_soft,
        KG_drop_hard,
        gate: Optional[torch.Tensor] = None,
        role: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):

        if edge_index is None or KG_drop_soft.size(0) == 0:
            zero_mask = entity_emb.new_zeros((entity_emb.size(0), 1))
            return entity_emb, zero_mask, zero_mask

        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        chunk = self.kg_edge_chunk_size

        if gate is not None:
            gate = gate.to(entity_emb.device, dtype=entity_emb.dtype).reshape(-1)
        if role is not None:
            role = role.to(entity_emb.device, dtype=torch.long).reshape(-1)
        if reward is not None:
            reward = reward.to(entity_emb.device, dtype=entity_emb.dtype).reshape(-1)

        entity_agg = entity_emb.new_zeros((n_entities, entity_emb.size(1)))
        score_agg = entity_emb.new_zeros((n_entities, 1))
        gate_agg = entity_emb.new_zeros((n_entities, 1))

        role_gate_values = torch.sigmoid(self.role_type_gate)

        for start in range(0, head.size(0), chunk):
            end = min(start + chunk, head.size(0))
            h_idx = head[start:end]
            t_idx = tail[start:end]
            rel_idx = edge_type[start:end]
            kg_soft = KG_drop_soft[start:end]
            kg_hard = KG_drop_hard[start:end]

            kg_score = kg_soft * kg_hard
            base_gate = torch.ones_like(kg_score)

            if gate is not None:
                gate_chunk = gate[start:end].unsqueeze(-1)
                base_gate = base_gate * gate_chunk
            if role is not None:
                role_chunk = role[start:end]
                type_gate = role_gate_values[role_chunk].unsqueeze(-1)
                base_gate = base_gate * type_gate
            if reward is not None:
                reward_chunk = reward[start:end].unsqueeze(-1)
                base_gate = base_gate * torch.sigmoid(self.reward_gate * reward_chunk)

            degree_counts = torch.bincount(h_idx, minlength=n_entities).float()
            degree_vals = degree_counts[h_idx]
            degree_gate = torch.exp(
                -torch.relu(self.degree_gate) * torch.log1p(degree_vals)
            ).unsqueeze(-1)

            total_gate = (base_gate * degree_gate).clamp_min(self.gate_floor)

            kg_score = kg_score * total_gate
            tail_emb = entity_emb[t_idx]
            rel_emb = relation_weight[rel_idx]
            neb_kg_emb = (tail_emb + rel_emb) * kg_score

            entity_agg.index_add_(0, h_idx, neb_kg_emb)
            score_agg.index_add_(0, h_idx, kg_hard * total_gate)
            gate_agg.index_add_(0, h_idx, total_gate)

        entity_agg = entity_agg / (score_agg + 1e-9)
        score_mask = (score_agg < 1).float()
        return entity_agg, score_mask, gate_agg.clamp(min=0.0)


    
    def split_kg(self,edge_index,edge_type,kg_mask_size=512):
        # topk_mask = np.zeros(edge_index.shape[0], dtype=bool)
        # topk_mask[topk_egde_id] = True
        # add another group of random mask
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(
            n_edges, size=kg_mask_size, replace=False)
        random_mask = np.zeros(edge_index.shape[1], dtype=bool)
        random_mask[random_indices] = True

        mask_edge_index = edge_index[:, random_mask]
        mask_edge_type = edge_type[random_mask]
        
        retain_edge_index = edge_index[:,~random_mask]
        retain_edge_type = edge_type[~random_mask]
        
        return retain_edge_index, retain_edge_type, mask_edge_index, mask_edge_type
    
    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        return mf_loss
    
    def create_bce_loss(self,head_emb,rel_emb,target_id,all_embed):
        merge = head_emb + rel_emb
        score = torch.matmul(merge, all_embed.transpose(1,0))
        bce_loss = self.bce_loss(score, target_id)
        return bce_loss
    
    def _cal_mmd(self, kg_drb, cf_drb, kernel_mul=2.0, kernel_num=5):
        if kg_drb is None or cf_drb is None:
            return torch.tensor(0.0, device=self.device)
        if kg_drb.numel() == 0 or cf_drb.numel() == 0:
            return kg_drb.new_tensor(0.0)

        if self.mmd_estimator == "linear":
            mean_diff = kg_drb.mean(dim=0) - cf_drb.mean(dim=0)
            return torch.sum(mean_diff ** 2)

        total = torch.cat([kg_drb, cf_drb], dim=0)
        bandwidths = self._estimate_bandwidths(total, kernel_mul, kernel_num)
        m = kg_drb.size(0)
        n = cf_drb.size(0)
        chunk = max(1, min(self.mmd_batch_size, max(m, n)))

        mmd_vals = []
        for bw in bandwidths:
            xx = self._kernel_sum(kg_drb, kg_drb, bw, chunk) / (m * m)
            yy = self._kernel_sum(cf_drb, cf_drb, bw, chunk) / (n * n)
            xy = self._kernel_sum(kg_drb, cf_drb, bw, chunk) / (m * n)
            mmd_vals.append(xx + yy - 2.0 * xy)
        return torch.stack(mmd_vals).mean()
    

    
    def forward(
        self,
        all_embed,
        all_embed_cf,
        edge_index,
        edge_type,
        aug_edge_index,
        aug_edge_type,
        role_edge_index,
        role_edge_type,
        interact_mat,
        mess_dropout=True,
        node_dropout=False,
        gumbel=True,
    ):
        """node dropout"""
        if node_dropout:
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)          

        user_embeds = all_embed[:self.n_users]
        item_embed = all_embed[self.n_users:self.n_users + self.n_items][:,:self.channel]
        entity_emb = all_embed[self.n_users:][:,self.channel:self.channel * 2]
        n_entity_emb = all_embed[self.n_users:][:,self.channel*2:]

        '''KG'''
        use_sampling = self.training and gumbel
        kg_sample_rate = self.kg_edge_sample_rate if use_sampling else 1.0
        ui_sample_rate = self.ui_edge_sample_rate if use_sampling else 1.0
        (
            KG_drop_soft,
            KG_drop_hard,
            kg_edge_index,
            kg_edge_type,
            _,
            _,
            _,
        ) = self.Dnoise_KG(
            edge_index,
            edge_type,
            entity_emb,
            self.relation_weight,
            is_gumble=gumbel,
            new=False,
            edge_sample_rate=kg_sample_rate,
        )
        (
            A_KG_drop_soft,
            A_KG_drop_hard,
            aug_edge_index_used,
            aug_edge_type_used,
            aug_gate_used,
            _,
            _,
        ) = self.Dnoise_KG(
            aug_edge_index,
            aug_edge_type,
            entity_emb,
            self.relation_weight,
            is_gumble=gumbel,
            new=False,
            edge_sample_rate=(self.kg_edge_sample_rate if use_sampling else 1.0),
            edge_gate=self.aug_edge_gate,
        )
        (
            N_KG_drop_soft,
            N_KG_drop_hard,
            role_edge_index_used,
            role_edge_type_used,
            role_gate_used,
            role_type_used,
            role_reward_used,
        ) = self.Dnoise_KG(
            role_edge_index,
            role_edge_type,
            n_entity_emb,
            self.n_relation_weight,
            is_gumble=gumbel,
            new=True,
            edge_sample_rate=(self.new_kg_edge_sample_rate if use_sampling else 1.0),
            edge_gate=self.role_edge_gate,
            edge_role=self.role_edge_role,
            edge_reward=self.role_edge_reward,
        )

        '''calculate mmd loss'''
        mmd_loss = all_embed.new_tensor(0.0)
        if role_edge_index_used is not None and N_KG_drop_soft.numel() > 0:
            kg_trip = torch.stack(
                [role_edge_index_used[0], role_edge_type_used, role_edge_index_used[1]], dim=1)
            sample_size = min(self.mmd_batch_size, kg_trip.size(0))
            if sample_size > 0:
                perm = torch.randint(low=0, high=kg_trip.size(0), size=(sample_size,), device=kg_trip.device)
                kg_bc_trip = kg_trip[perm]
                bc_n_kg_drop_soft = N_KG_drop_soft[perm]
                bc_kgc_soft = self.kgc(kg_bc_trip, eval=True, cf_train=True).detach()
                mmd_loss = self._cal_mmd(bc_kgc_soft, bc_n_kg_drop_soft) * self.mmd_weight

        role_reg = all_embed.new_tensor(0.0)
        if self.role_relation_indices:
            device = self.relation_weight.device
            idx_tensor = torch.tensor(sorted(self.role_relation_indices), dtype=torch.long, device=device)
            base_count = max(1, min(self.base_relation_count, self.relation_weight.size(0)))
            template_main = self.relation_weight[:base_count].mean(dim=0, keepdim=True)
            template_new = self.n_relation_weight[:base_count].mean(dim=0, keepdim=True)
            main_diff = self.relation_weight[idx_tensor] - template_main
            new_diff = self.n_relation_weight[idx_tensor] - template_new
            entity_frac = float(getattr(self, "current_role_fraction", 0.0))
            relation_frac = float(getattr(self, "current_relation_fraction", 0.0))
            reg_scale = 1.0 - max(entity_frac, relation_frac)
            reg_scale = min(1.0, max(0.2, reg_scale))
            role_reg = (main_diff.pow(2).mean() + new_diff.pow(2).mean()) * self.role_relation_reg * reg_scale
        mmd_loss = mmd_loss + role_reg

        entity_emb_res = entity_emb[:self.n_items]
        n_entity_emb_res = n_entity_emb[:self.n_items]
        for _ in range(len(self.convs)):
            main_updated = False
            role_updated = False
            role_gate_summary = None
            if kg_edge_index is not None and KG_drop_soft.size(0) > 0:
                entity_emb, _, _ = self.KG_forward(
                    entity_emb,
                    kg_edge_index,
                    kg_edge_type,
                    self.relation_weight,
                    KG_drop_soft,
                    KG_drop_hard,
                )
                main_updated = True
            if aug_edge_index_used is not None and A_KG_drop_soft.size(0) > 0:
                entity_emb, _, _ = self.KG_forward(
                    entity_emb,
                    aug_edge_index_used,
                    aug_edge_type_used,
                    self.relation_weight,
                    A_KG_drop_soft,
                    A_KG_drop_hard,
                    gate=aug_gate_used,
                )
                main_updated = True
            if role_edge_index_used is not None and N_KG_drop_soft.size(0) > 0:
                n_entity_emb, _, role_gate_summary = self.KG_forward(
                    n_entity_emb,
                    role_edge_index_used,
                    role_edge_type_used,
                    self.n_relation_weight,
                    N_KG_drop_soft,
                    N_KG_drop_hard,
                    gate=role_gate_used,
                    role=role_type_used,
                    reward=role_reward_used,
                )
                role_updated = True
            if role_gate_summary is not None:
                entity_emb = self._apply_role_bridge(entity_emb, n_entity_emb, role_gate_summary)
                user_role_delta = self._role_signal_for_users(role_gate_summary, n_entity_emb)
                if user_role_delta is not None:
                    user_embeds = user_embeds + user_role_delta
                main_updated = True
            if main_updated:
                entity_emb_res = entity_emb_res + F.normalize(entity_emb[:self.n_items])
            if role_updated:
                n_entity_emb_res = n_entity_emb_res + F.normalize(n_entity_emb[:self.n_items])

        # KG_drop_hard = None
        '''KG'''
        item_mix_tensor = None
        if self.role_item_mix is not None and self.role_item_mix.size(0) >= self.n_items:
            item_mix_tensor = self.role_item_mix[:self.n_items].to(n_entity_emb_res.device, dtype=n_entity_emb_res.dtype)
        bridge_gain = 1.0 + float(self.role_bridge_coeff)
        if item_mix_tensor is None:
            role_component = n_entity_emb_res * (self.role_mix_coeff * bridge_gain)
        else:
            role_component = n_entity_emb_res * item_mix_tensor.clamp(0.0, 2.0) * bridge_gain
        item_embeds = torch.cat([item_embed, entity_emb_res, role_component], dim=-1)

        user_embed_res = user_embeds
        item_embed_res = item_embeds
        for i in range(len(self.convs)-1):
            user_embeds, item_embeds = self.convs[i](user_embeds, item_embeds, self.interact_mat,
                                                     edge_sample_rate=ui_sample_rate,
                                                     edge_chunk_size=self.ui_edge_chunk_size)
            item_embed_res = item_embed_res + F.normalize(item_embeds)
            user_embed_res = user_embed_res + F.normalize(user_embeds)
        
        align_penalty = self.kgc.alignment_penalty() if hasattr(self.kgc, "alignment_penalty") else None
        if align_penalty is not None and isinstance(align_penalty, torch.Tensor):
            mmd_loss = mmd_loss + align_penalty
        bce_loss = 0
        return user_embed_res, item_embed_res, KG_drop_hard, mmd_loss



class RoSAKGRecommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ui_sp_graph, item_rel_mask):
        super(RoSAKGRecommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample
        
        self.gamma = args_config.gamma
        self.max_iter = args_config.max_iter
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        self.kg_edge_sample_rate = min(max(getattr(args_config, 'kg_edge_sample_rate', 1.0), 0.0), 1.0)
        self.ui_edge_sample_rate = min(max(getattr(args_config, 'ui_edge_sample_rate', 1.0), 0.0), 1.0)
        self.kg_edge_chunk_size = max(1, int(getattr(args_config, 'kg_edge_chunk_size', 65536)))
        self.ui_edge_chunk_size = max(1, int(getattr(args_config, 'ui_edge_chunk_size', 131072)))
        self.mmd_estimator = getattr(args_config, 'mmd_estimator', 'linear')
        self.mmd_bandwidth_sample = max(2, int(getattr(args_config, 'mmd_bandwidth_sample', 2048)))
        self.mmd_batch_size = max(2, int(getattr(args_config, 'mmd_batch_size', 8192)))
        self.role_mix_coeff_init = float(getattr(args_config, 'role_mix_coeff_init', 0.2))
        self.new_edge_threshold_init = float(getattr(args_config, 'new_edge_threshold_init', 0.6))
        self.new_kg_edge_sample_rate = float(getattr(args_config, 'new_kg_edge_sample_rate', self.kg_edge_sample_rate))
        self.mmd_weight_param = float(getattr(args_config, 'mmd_weight', 0.5))
        self.role_relation_reg = float(getattr(args_config, 'role_relation_reg', 1e-2))
        self.role_fraction_learned_weight = float(getattr(args_config, 'role_fraction_learned_weight', 0.4))
        self.role_fraction_lr = float(getattr(args_config, 'role_fraction_lr', 0.35))
        self.role_entity_lr = float(getattr(args_config, 'role_entity_lr', 0.25))
        self.role_gate_degree_coef = float(getattr(args_config, 'role_gate_degree_coef', 0.15))
        self.role_gate_reward_coef = float(getattr(args_config, 'role_gate_reward_coef', 0.2))
        self.role_bridge_coeff_init = float(getattr(args_config, 'role_bridge_coeff_init', 0.0))
        self.role_bridge_coeff_max = float(getattr(args_config, 'role_bridge_coeff_max', 1.0))
        self.role_bridge_lr = float(getattr(args_config, 'role_bridge_lr', 0.15))
        self.role_bridge_decay = float(getattr(args_config, 'role_bridge_decay', 0.1))
        self.role_alignment_weight = float(getattr(args_config, 'role_alignment_weight', 0.0))
        self.role_alignment_weight *= float(getattr(args_config, 'role_alignment_scale', 1.0))
        self.role_user_embed_blend = float(getattr(args_config, 'role_user_embed_blend', 0.65))
        self.role_user_embed_blend = min(max(self.role_user_embed_blend, 0.0), 1.0)
        self.role_memory_momentum = float(getattr(args_config, 'role_memory_momentum', 0.7))
        self.role_memory_momentum = min(max(self.role_memory_momentum, 0.0), 0.999)
        self.role_memory: Dict[int, torch.Tensor] = {}
        self.item_rel_mask = torch.FloatTensor(item_rel_mask).to(self.device)
        self.ui_sp_graph = ui_sp_graph

        self.edge_index, self.edge_type = self._get_edges(graph)
        
        self.aug_edge_index = None
        self.aug_edge_type = None
        self.aug_edge_gate = None
        self.role_edge_index = None
        self.role_edge_type = None
        self.role_edge_gate = None
        self.role_edge_role = None
        self.role_edge_reward = None

        self.cet_loss = nn.CrossEntropyLoss(label_smoothing=0.85)
        # self.ranking_loss = nn.MarginRankingLoss(margin=1)
        self._init_weight()
        self._init_loss_function()

        self._optimizer_refs = []
        self.gcn = self._init_model()
        if hasattr(self.gcn, "set_role_bridge_coeff"):
            self.gcn.set_role_bridge_coeff(self.role_bridge_coeff_init)
        if hasattr(self.gcn, "kgc") and hasattr(self.gcn.kgc, "set_shared_embeddings"):
            self.gcn.kgc.set_shared_embeddings(
                self.all_embed,
                self.gcn.n_relation_weight,
                self.n_users,
                self.emb_size * 2,
                self.emb_size,
            )
        self.gcn.set_optimizer_refs(self._optimizer_refs)
        with torch.no_grad():
            self.gcn.degree_gate.data.fill_(self.role_gate_degree_coef)
            self.gcn.reward_gate.data.fill_(self.role_gate_reward_coef)
            # Encourage neutral initial role weights
            self.gcn.role_type_gate.data.zero_()

        
        
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size * 3))
        self.all_embed = nn.Parameter(self.all_embed)

        # self.all_embed_cf = initializer(torch.empty(self.n_users + self.n_items, self.emb_size))
        # self.all_embed_cf = nn.Parameter(self.all_embed_cf)
        self.all_embed_cf = None
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.ui_sp_graph).to(self.device)   

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def ensure_relation_capacity(self, required_relations: int):
        required_relations = int(required_relations)
        if required_relations <= self.n_relations and required_relations <= getattr(self.gcn, "n_relations", 0):
            return
        self.gcn.expand_relation_capacity(required_relations)
        new_capacity = int(getattr(self.gcn, "n_relations", self.n_relations))
        if new_capacity > self.n_relations:
            self.n_relations = new_capacity
        if hasattr(self.gcn, "kgc") and hasattr(self.gcn.kgc, "set_shared_embeddings"):
            self.gcn.kgc.set_shared_embeddings(
                self.all_embed,
                self.gcn.n_relation_weight,
                self.n_users,
                self.emb_size * 2,
                self.emb_size,
            )

    def ensure_entity_capacity(self, required_entities: int):
        required_entities = int(required_entities)
        if required_entities <= self.n_entities and required_entities <= getattr(self.gcn, "n_entity", 0):
            return
        # First expand GCN/KGC side (includes buffer reservation)
        self.gcn.expand_entity_capacity(required_entities)
        target_entities = int(getattr(self.gcn, "n_entity", self.n_entities))
        if target_entities <= self.n_entities:
            return
        # Then expand node embeddings (users + entities share all_embed rows)
        device = self.all_embed.device
        channel = self.emb_size * 3
        extra = target_entities - self.n_entities
        with torch.no_grad():
            add = torch.empty((extra, channel), device=device, dtype=self.all_embed.dtype)
            nn.init.xavier_uniform_(add)
        self.all_embed = _expand_parameter(self.all_embed, add, self._optimizer_refs)
        self.n_entities = target_entities
        self.n_nodes = self.n_users + self.n_entities
        if hasattr(self.gcn, "kgc") and hasattr(self.gcn.kgc, "set_shared_embeddings"):
            self.gcn.kgc.set_shared_embeddings(
                self.all_embed,
                self.gcn.n_relation_weight,
                self.n_users,
                self.emb_size * 2,
                self.emb_size,
            )

    def initialize_role_entity_embedding(
        self,
        role_entity_id: int,
        source_entities: np.ndarray,
        weights: np.ndarray = None,
        user_ids: Optional[np.ndarray] = None,
        user_weights: Optional[np.ndarray] = None,
    ):
        if source_entities.size == 0 or role_entity_id < 0:
            return
        valid = source_entities[(source_entities >= 0) & (source_entities < self.n_entities)]
        if valid.size == 0:
            return
        device = self.all_embed.device
        idx_tensor = torch.as_tensor(valid, dtype=torch.long, device=device)
        base_rows = self.all_embed.data[self.n_users + idx_tensor]
        if weights is not None and weights.size == valid.size:
            weight_tensor = torch.as_tensor(weights[:valid.size], dtype=torch.float32, device=device)
            weight_sum = float(weight_tensor.sum().item())
            if weight_sum > 0:
                weight_tensor = weight_tensor / weight_sum
            else:
                weight_tensor.fill_(1.0 / max(1, valid.size))
            mean_embed = (base_rows * weight_tensor.unsqueeze(-1)).sum(dim=0, keepdim=False)
        else:
            mean_embed = base_rows.mean(dim=0, keepdim=False)
        role_embed = mean_embed
        memory = self.role_memory.get(role_entity_id)
        if memory is not None:
            mem_tensor = memory.to(device=device, dtype=mean_embed.dtype)
            role_embed = (
                self.role_memory_momentum * mem_tensor
                + (1.0 - self.role_memory_momentum) * role_embed
            )
        role_slice = slice(self.emb_size * 2, self.emb_size * 3)
        if user_ids is not None and user_ids.size > 0:
            user_ids = user_ids.astype(np.int64, copy=False)
            valid_users = user_ids[(user_ids >= 0) & (user_ids < self.n_users)]
            if valid_users.size > 0:
                user_tensor = torch.as_tensor(valid_users, dtype=torch.long, device=device)
                user_rows = self.all_embed.data[user_tensor, : self.emb_size]
                if user_weights is not None and user_weights.size >= valid_users.size:
                    user_weight_tensor = torch.as_tensor(
                        user_weights[:valid_users.size], dtype=torch.float32, device=device
                    )
                    user_weight_sum = float(user_weight_tensor.sum().item())
                    if user_weight_sum > 0:
                        user_weight_tensor = user_weight_tensor / user_weight_sum
                    else:
                        user_weight_tensor.fill_(1.0 / max(1, valid_users.size))
                    user_role_embed = (user_rows * user_weight_tensor.unsqueeze(-1)).sum(dim=0, keepdim=False)
                else:
                    user_role_embed = user_rows.mean(dim=0, keepdim=False)
                blend = float(getattr(self, "role_user_embed_blend", 0.5))
                blend = min(max(blend, 0.0), 1.0)
                role_embed = role_embed.clone()
                base_slice = role_embed[role_slice]
                user_role_embed = user_role_embed.to(device=device, dtype=base_slice.dtype)
                role_embed[role_slice] = (
                    (1.0 - blend) * base_slice + blend * user_role_embed
                )
        self.all_embed.data[self.n_users + role_entity_id] = role_embed
        kgc_weight = self.gcn.kgc.ent_embeddings.weight
        kgc_dim = kgc_weight.size(0)
        if role_entity_id < kgc_dim:
            kgc_valid = idx_tensor[idx_tensor < kgc_dim]
            if kgc_valid.numel() > 0:
                kgc_mean = kgc_weight.data[kgc_valid].mean(dim=0, keepdim=False)
                kgc_weight.data[role_entity_id] = kgc_mean
        self.role_memory[role_entity_id] = role_embed.detach().cpu()

    def reset_dynamic_edges(self):
        self.gcn.reset_dynamic_edges()
        self.aug_edge_index = self.gcn.aug_edge_index
        self.aug_edge_type = self.gcn.aug_edge_type
        self.aug_edge_gate = self.gcn.aug_edge_gate
        self.role_edge_index = self.gcn.role_edge_index
        self.role_edge_type = self.gcn.role_edge_type
        self.role_edge_gate = self.gcn.role_edge_gate
        self.role_edge_role = self.gcn.role_edge_role
        self.role_edge_reward = self.gcn.role_edge_reward

    def append_dynamic_edges(self, edges: torch.Tensor, meta: Optional[Dict[str, torch.Tensor]] = None):
        if edges.numel() == 0:
            return
        self.gcn.append_dynamic_edges(edges, meta=meta)
        self.aug_edge_index = self.gcn.aug_edge_index
        self.aug_edge_type = self.gcn.aug_edge_type
        self.aug_edge_gate = self.gcn.aug_edge_gate
        self.role_edge_index = self.gcn.role_edge_index
        self.role_edge_type = self.gcn.role_edge_type
        self.role_edge_gate = self.gcn.role_edge_gate
        self.role_edge_role = self.gcn.role_edge_role
        self.role_edge_reward = self.gcn.role_edge_reward

    def update_role_mix(self, scores: Mapping[int, float], momentum: float = 0.5) -> None:
        if hasattr(self, "gcn") and hasattr(self.gcn, "update_role_mix"):
            self.gcn.update_role_mix(scores, momentum=momentum)

    def sync_role_relation_weights(self, relation_ids: Optional[Sequence[int]] = None) -> None:
        if hasattr(self, "gcn") and hasattr(self.gcn, "sync_role_relation_weights"):
            self.gcn.sync_role_relation_weights(relation_ids=relation_ids)

    def register_optimizer(self, optimizer) -> None:
        """
        Track optimizers so that dynamic parameter growth keeps optimizer state consistent.
        """
        if optimizer is None:
            return
        base = getattr(optimizer, "optimizer", optimizer)
        if not isinstance(base, Optimizer):
            return
        if base not in self._optimizer_refs:
            self._optimizer_refs.append(base)
        self.gcn.set_optimizer_refs(self._optimizer_refs)


    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_items=self.n_items,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         gamma=self.gamma,
                         max_iter=self.max_iter,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         kg_edge_chunk_size=self.kg_edge_chunk_size,
                         ui_edge_chunk_size=self.ui_edge_chunk_size,
                         kg_edge_sample_rate=self.kg_edge_sample_rate,
                         ui_edge_sample_rate=self.ui_edge_sample_rate,
                         mmd_estimator=self.mmd_estimator,
                         mmd_bandwidth_sample=self.mmd_bandwidth_sample,
                         mmd_batch_size=self.mmd_batch_size,
                         role_mix_coeff=self.role_mix_coeff_init,
                         new_edge_threshold=self.new_edge_threshold_init,
                         new_kg_edge_sample_rate=self.new_kg_edge_sample_rate,
                         role_bridge_coeff=self.role_bridge_coeff_init,
                         role_bridge_cap=self.role_bridge_coeff_max,
                         mmd_weight=self.mmd_weight_param,
                         role_relation_reg=self.role_relation_reg,
                         optimizer_refs=self._optimizer_refs,
                         kgc_align_weight=self.role_alignment_weight)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  
        index = graph_tensor[:, :-1] 
        type = graph_tensor[:, -1] 
        return index.t().long().to(self.device), type.long().to(self.device)



    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def L2_norm(self,hidden,k=1):
        hidden_norm = torch.norm(hidden,p=2,dim=-1,keepdim=True)
        out_hidden = k * torch.div(hidden,hidden_norm+1e-8)
        return out_hidden

    def _contrastive_loss(self,user_kg,n_user_kg,pos_user_id, tau=1,small_batch = 256):
        if small_batch:
            small_id = torch.randint(low=0, high=pos_user_id.shape[0], size=(small_batch,)).to(user_kg.device)
            pos_user_id = pos_user_id[small_id]
            
        pos_user_kge = user_kg[pos_user_id]
        pos_n_user_kge = n_user_kg[pos_user_id]

        pos = torch.eye(pos_user_kge.shape[0]).to(pos_user_kge.device)
        z1_norm = torch.norm(pos_user_kge, dim=-1, keepdim=True)
        z2_norm = torch.norm(pos_n_user_kge, dim=-1, keepdim=True)
        dot_numerator = torch.mm(pos_user_kge, pos_n_user_kge.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / (dot_denominator + 1e-5) / tau)
        
        smi_sum = torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-5

        sim_matrix = sim_matrix/smi_sum
        assert sim_matrix.size(0) == sim_matrix.size(1)
        cl_loss = -torch.log(sim_matrix.mul(pos).sum(dim=-1)).mean()

        return cl_loss


    def gcn_forword(self, user, pos_item):
        user_all_emb, item_all_emb, KG_drop_hard,mmd_loss  = self.gcn(
            self.all_embed,
            self.all_embed_cf,
            self.edge_index,
            self.edge_type,
            self.aug_edge_index,
            self.aug_edge_type,
            self.role_edge_index,
            self.role_edge_type,
            self.interact_mat,
            mess_dropout=self.mess_dropout,
            node_dropout=self.node_dropout,
            gumbel=True,
        )

        user_emb = user_all_emb[user]
        score = torch.matmul(user_emb, item_all_emb.transpose(1,0))
        cet_loss = self.cet_loss(score,pos_item)
        

        pos_emb = item_all_emb[pos_item]
        regularizer = (torch.norm(user_emb) ** 2
                       + torch.norm(pos_emb) ** 2)
 
        return  cet_loss, mmd_loss 
    


    def forward(self,batch=None,mode="cf"):
        if mode == "cf":
            user = batch['users']                                                                          
            pos_item = batch['pos_items']                                                                  

            loss_network = self.gcn_forword(user, pos_item)
            return loss_network
        
        else:
            kgc_loss = self.gcn.kgc(batch)
            return kgc_loss


    def generate_embeddings(self, for_kgc: bool = False):
        user_all_emb, item_all_emb, KG_drop_hard, kg_loss = self.gcn(
            self.all_embed,
            self.all_embed_cf,
            self.edge_index,
            self.edge_type,
            self.aug_edge_index,
            self.aug_edge_type,
            self.role_edge_index,
            self.role_edge_type,
            self.interact_mat,
            mess_dropout=False,
            node_dropout=False,
            gumbel=False,
        )

        item_pred_emb = item_all_emb
        user_pred_emb = user_all_emb
        if for_kgc:
            return item_pred_emb, KG_drop_hard
        else:
            return item_pred_emb, user_pred_emb

    # Backwards-compatible alias used by legacy EditKG code paths.
    def generate(self, for_kgc: bool = False):
        return self.generate_embeddings(for_kgc=for_kgc)


    def rating(self, u_g_embeddings, i_g_embeddings,type="bpr"):
        if type == "bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        else:
            return torch.cosine_similarity(u_g_embeddings[:, :self.emb_size].unsqueeze(1),
                                           i_g_embeddings[:, :self.emb_size].unsqueeze(0), dim=2).detach().cpu() + \
                   torch.cosine_similarity(u_g_embeddings[:, self.emb_size:].unsqueeze(1),
                                           i_g_embeddings[:, self.emb_size:].unsqueeze(0), dim=2).detach().cpu()


    def create_contrastive_loss(self, u_e, pos_e, neg_e,loss_weight):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)

        ui_pos_loss1 = torch.relu(1 - torch.cosine_similarity(u_e, pos_e, dim=1))

        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1 > 0
        ui_neg_loss1 = torch.sum(ui_neg1, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = (ui_pos_loss1 + ui_neg_loss1)

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss
    
    
    
    
    
class KGC(nn.Module):
    def __init__(self, n_items,num_ent, num_rel, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super(KGC, self).__init__()
        self.n_items = n_items
        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.linear_1 = nn.Linear(self.dim * 2,512)
        # self.linear_2 = nn.Linear(1024,1024)
        # self.linear_2 = nn.Linear(1024,512)
        self.linear_2 = nn.Linear(512,256)
        # self.ln = nn.LayerNorm(256)
        self.linear_pre = nn.Linear(256,1)


        self.ent_embeddings = nn.Embedding(self.num_ent,self.dim)
        self.rel_embeddings = nn.Embedding(self.num_rel, self.dim)
        self._optimizer_refs: Sequence[Optimizer] = []
        self.all_embed_param: Optional[torch.Tensor] = None
        self.rel_embed_param: Optional[torch.Tensor] = None
        self.n_users_offset: int = 0
        self.entity_channel_offset: int = 0
        self.entity_channel_dim: int = self.dim
        self.align_weight: float = 0.0

        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean")
        # self.bce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.bce_loss = nn.BCELoss(reduction="mean")
    def __parameter_init(self,normalize=False):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        if normalize:
            self.normalization_rel_embedding()
            self.normalization_ent_embedding()

    def normalization_ent_embedding(self):
        norm = self.ent_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embeddings.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embeddings.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embeddings.weight.data.copy_(torch.from_numpy(norm))

    def expand_relations(self, required_relations: int):
        required_relations = int(required_relations)
        if required_relations <= self.num_rel:
            return
        device = self.rel_embeddings.weight.device
        extra = required_relations - self.num_rel
        with torch.no_grad():
            add = torch.empty((extra, self.dim), device=device, dtype=self.rel_embeddings.weight.dtype)
            nn.init.xavier_uniform_(add)
        self.rel_embeddings.weight = _expand_parameter(self.rel_embeddings.weight, add, self._optimizer_refs)
        self.rel_embeddings.num_embeddings = required_relations
        self.num_rel = required_relations
    
    def expand_entities(self, required_entities: int):
        required_entities = int(required_entities)
        if required_entities <= self.num_ent:
            return
        device = self.ent_embeddings.weight.device
        extra = required_entities - self.num_ent
        with torch.no_grad():
            add = torch.empty((extra, self.dim), device=device, dtype=self.ent_embeddings.weight.dtype)
            nn.init.xavier_uniform_(add)
        self.ent_embeddings.weight = _expand_parameter(self.ent_embeddings.weight, add, self._optimizer_refs)
        self.ent_embeddings.num_embeddings = required_entities
        self.num_ent = required_entities

    def set_optimizer_refs(self, refs: Sequence[Optimizer]) -> None:
        self._optimizer_refs = refs if refs is not None else []
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    def _distance(self, h, t, r,neg=False):
        if neg:
            score = (h + r).unsqueeze(1) - t
            score = torch.norm(score, p=self.p_norm, dim=-1).mean(dim=1)
        else:
            score = (h + r) - t
            score = torch.norm(score, p=self.p_norm, dim=1)
        return score

    def forward(self, data, eval=False,rate=0.5,cf_train=False):

        if eval:
            if cf_train:
                batch_triple = data
            else:
                batch_triple = data["hr_pair"]
            
            batch_h = batch_triple[:,0]
            batch_r = batch_triple[:,1]
            batch_t = batch_triple[:,2]
            # batch_label = batch_triple[:,-1]

            h = self.ent_embeddings(batch_h)
            t = self.ent_embeddings(batch_t)
            r = self.rel_embeddings(batch_r)
            
            # score = torch.sigmoid(torch.matmul(h,(r * t)))
            h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
            h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
            h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
            # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
            score = torch.sigmoid(self.linear_pre(h_r_t_emb))
            # return (score>=rate).squeeze(-1).float()
            return score
        
        batch_triple = data["hr_pair"]
        
        batch_h = batch_triple[:,0]
        batch_r = batch_triple[:,1]
        batch_t = batch_triple[:,2]
        batch_label = batch_triple[:,-1]

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_r_t_emb = F.normalize(torch.cat([h,r*t],dim=-1))
        h_r_t_emb = torch.relu(self.linear_1(h_r_t_emb))
        h_r_t_emb = torch.relu(self.linear_2(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_3(h_r_t_emb))
        # h_r_t_emb = torch.relu(self.linear_4(h_r_t_emb))
        score = torch.sigmoid(self.linear_pre(h_r_t_emb))

        loss = self.bce_loss(score.squeeze(-1), batch_label.float())
 
        return loss

    def regularization(self, data):

        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings[batch_h]
        t = self.ent_embeddings[batch_t]
        r = self.rel_embeddings[batch_r]
        regul = (torch.mean(h ** 2) + 
                    torch.mean(t ** 2) + 
                    torch.mean(r ** 2)) / 3
        return regul

    def set_shared_embeddings(
        self,
        all_embed_param: Optional[torch.Tensor],
        rel_embed_param: Optional[torch.Tensor],
        n_users_offset: int,
        entity_channel_offset: int,
        entity_channel_dim: int,
    ) -> None:
        self.all_embed_param = all_embed_param
        self.rel_embed_param = rel_embed_param
        self.n_users_offset = int(max(0, n_users_offset))
        self.entity_channel_offset = int(max(0, entity_channel_offset))
        self.entity_channel_dim = int(max(1, entity_channel_dim))

    def set_align_weight(self, weight: float) -> None:
        self.align_weight = max(0.0, float(weight))

    def alignment_penalty(self) -> torch.Tensor:
        if self.align_weight <= 0.0:
            return self.ent_embeddings.weight.new_tensor(0.0)
        penalties: List[torch.Tensor] = []
        if self.all_embed_param is not None and self.all_embed_param.size(0) >= self.n_users_offset + self.num_ent:
            ent_slice = self.all_embed_param[
                self.n_users_offset:self.n_users_offset + self.num_ent,
                self.entity_channel_offset:self.entity_channel_offset + self.entity_channel_dim,
            ]
            ent_weight = self.ent_embeddings.weight
            rows = min(ent_slice.size(0), ent_weight.size(0))
            cols = min(ent_slice.size(1), ent_weight.size(1))
            if rows > 0 and cols > 0:
                diff = ent_weight[:rows, :cols] - ent_slice[:rows, :cols].detach()
                penalties.append(diff.pow(2).mean())
        if self.rel_embed_param is not None:
            rel_reference = self.rel_embed_param.weight if isinstance(self.rel_embed_param, nn.Embedding) else self.rel_embed_param
            rel_weight = self.rel_embeddings.weight
            rel_rows = min(rel_reference.size(0), rel_weight.size(0))
            cols = min(rel_reference.size(1), rel_weight.size(1))
            if rel_rows > 0 and cols > 0:
                rel_diff = rel_weight[:rel_rows, :cols] - rel_reference[:rel_rows, :cols].detach()
                penalties.append(rel_diff.pow(2).mean())
        if not penalties:
            return self.ent_embeddings.weight.new_tensor(0.0)
        return sum(penalties) * self.align_weight
