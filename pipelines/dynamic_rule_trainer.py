"""
Dynamic rule trainer (RSRG core) orchestrating static rule mining and role-aware
updates during RoSA-KG training.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from modules.rule_mining import (
    RuleDefinition,
    RuleFilterConfig,
    RuleMiner,
    RuleMinerConfig,
    RulePostProcessor,
)
from modules.rule_roles import (
    RoleUpdateSignal,
    RuleRoleManager,
    VirtualRelationBuilder,
    VirtualRelationConfig,
)
from modules.rule_roles.role_trainer import RoleState
from modules.rule_roles.diversity import (
    DiversityControl,
    ExposureTracker,
    RoleDiversityScorer,
)


@dataclass
class DynamicRuleConfig:
    rules_dir: str
    rules_filename: str = "ruleset.json"
    usage_log: Optional[str] = None
    refresh_interval: int = 3
    max_roles: int = 128
    max_edges_per_role: int = 512
    max_roles_growth: int = 32
    max_roles_cap: int = 0
    reward_weights: Dict[str, float] = None
    force_rebuild: bool = False

    def __post_init__(self) -> None:
        if self.reward_weights is None:
            self.reward_weights = {
                "recall": 1.0,
                "ndcg": 0.4,
                "kgat_ad": 0.3,
                "kgat_arp": -0.25,
                "kgc_acc": 0.2,
                "kgc_loss": -0.1,
            }
        self.max_roles = max(0, int(self.max_roles))
        self.max_roles_growth = max(1, int(self.max_roles_growth))
        self.max_roles_cap = int(self.max_roles_cap)


class DynamicRuleTrainer:
    def __init__(
        self,
        triplets: np.ndarray,
        base_relation_count: int,
        miner_config: RuleMinerConfig,
        filter_config: RuleFilterConfig,
        dynamic_config: DynamicRuleConfig,
    ):
        self.triplets = triplets
        self.base_relation_count = int(base_relation_count)
        self.miner_config = miner_config
        self.filter_config = filter_config
        self.dynamic_config = dynamic_config
        self._role_limit = int(self.dynamic_config.max_roles)
        self._role_limit_growth = int(self.dynamic_config.max_roles_growth)
        cap_value = int(self.dynamic_config.max_roles_cap)
        self._role_limit_cap: Optional[int] = None if cap_value <= 0 else cap_value

        self.rules: List[RuleDefinition] = []
        self.role_manager = RuleRoleManager(max_active_roles=dynamic_config.max_roles)
        self.virtual_builder = VirtualRelationBuilder(
            VirtualRelationConfig(max_edges_per_role=dynamic_config.max_edges_per_role),
            self.triplets,
        )
        self._last_metrics: Optional[Dict[str, float]] = None
        self._last_active_states = []
        self.last_reward: Optional[float] = None
        self.last_vector_reward: Tuple[float, float] = (0.0, 0.0)
        self._metric_ema: Dict[str, float] = {}
        self._metric_ema_decay: float = 0.9
        self.diversity_control = DiversityControl()
        self.exposure_tracker = ExposureTracker(
            decay=self.diversity_control.exposure_decay
        )
        self.item_popularity: Dict[int, float] = {}
        self._diversity_keys = {"kgat_ad", "kgat_arp", "kgat_md"}
        self._recent_active_states: Deque[List[RoleState]] = deque(maxlen=4)
        self._recent_role_pairs: List[Tuple[RoleState, np.ndarray]] = []
        self._recent_role_reports: Dict[Tuple[Tuple[int, ...], int], Dict[str, float]] = {}
        self.last_metric_drop: float = 0.0
        os.makedirs(self.dynamic_config.rules_dir, exist_ok=True)

    @property
    def rules_path(self) -> str:
        return os.path.join(self.dynamic_config.rules_dir, self.dynamic_config.rules_filename)

    def bootstrap(self) -> List[RuleDefinition]:
        if os.path.exists(self.rules_path) and not self.dynamic_config.force_rebuild:
            print(f"[静态规则] 读取缓存: {self.rules_path}")
            self.rules = RuleMiner.load_rules(self.rules_path)
            print(f"[静态规则] 已加载规则数: {len(self.rules)}")
        else:
            print("[静态规则] 开始挖掘规则 ...")
            miner = RuleMiner(self.triplets, self.miner_config)
            mined = miner.mine()
            print(f"[静态规则] 原始候选数: {len(mined)}，开始后处理 ...")
            processed = RulePostProcessor(self.filter_config).filter_and_rank(mined)
            print(f"[静态规则] 过滤后保留: {len(processed)}，保存至 {self.rules_path}")
            RuleMiner.save_rules(self.rules_path, processed)
            self.rules = processed
        self.role_manager.bootstrap(self.rules, self.base_relation_count)
        print("[结构角色] 初始化完成")
        return self.rules

    def should_refresh(self, epoch: int) -> bool:
        return epoch % max(1, self.dynamic_config.refresh_interval) == 0

    def build_virtual_edges(
        self, epoch: int, limit_per_role: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        if not self.should_refresh(epoch):
            return np.empty((0, 3), dtype=np.int64), []
        active = self._select_active_roles()
        self._last_active_states = active
        self._record_active_states(active)
        triples, metadata = self.virtual_builder.build_batch(
            active, self.role_manager.rule_lookup, limit_per_role
        )
        if not triples:
            return np.empty((0, 3), dtype=np.int64), []
        array = np.array(triples, dtype=np.int64)
        if self.dynamic_config.usage_log:
            self._log_usage(epoch, metadata)
        return array, metadata

    def collect_role_pairs(
        self, limit_per_role: Optional[int] = None
    ) -> List[Tuple[RoleState, np.ndarray]]:
        """Collect (state, entity_ids) pairs for active roles with sampling control.

        Returns a list of tuples (RoleState, np.ndarray[[entity_id, ...]]).
        """
        active = self._select_active_roles()
        self._last_active_states = active
        self._record_active_states(active)
        results: List[Tuple[RoleState, np.ndarray]] = []
        for state in active:
            rule = self.role_manager.get_rule(state.descriptor.rule_key)
            if rule is None:
                continue
            entities = self.virtual_builder.sample_role_entities(
                rule, state.descriptor.position, limit_per_role
            )
            if entities is None or entities.size == 0:
                continue
            results.append((state, entities))
        if results:
            self._recent_role_pairs = [
                (state, np.array(entities, copy=True)) for state, entities in results
            ]
        return results

    def _select_active_roles(self) -> List[RoleState]:
        limit = None if self._role_limit <= 0 else self._role_limit
        active = self.role_manager.select_active_roles(limit)
        if self._role_limit <= 0:
            return active
        available = self.role_manager.count_available_roles(include_cooldown=True)
        if (
            available > self._role_limit
            and len(active) >= self._role_limit
            and (self._role_limit_cap is None or self._role_limit < self._role_limit_cap)
        ):
            new_limit = self._role_limit + self._role_limit_growth
            if self._role_limit_cap is not None:
                new_limit = min(new_limit, self._role_limit_cap)
            if new_limit > self._role_limit:
                self._role_limit = new_limit
                active = self.role_manager.select_active_roles(self._role_limit)
        return active

    def register_entity_exposure(self, entities: Sequence[int]) -> None:
        if entities:
            self.exposure_tracker.register(entities)

    def set_item_popularity(self, popularity: Mapping[int, float]) -> None:
        self.item_popularity = dict(popularity)
        self.virtual_builder.set_item_popularity(self.item_popularity)

    def record_role_reports(self, reports: Sequence[Mapping]) -> None:
        if not reports:
            return
        for rep in reports:
            rule_key = rep.get("rule_key")
            position = int(rep.get("position", 0))
            if rule_key is None:
                continue
            if isinstance(rule_key, tuple) and len(rule_key) == 2:
                body, head = rule_key
                if isinstance(body, (list, tuple)):
                    rule_key = (tuple(body), head)
            key = (rule_key, position)
            quality = float(rep.get("quality", rep.get("avg_score", 0.0)))
            if not np.isfinite(quality):
                quality = 0.0
            payload = {
                "quality": quality,
                "edge_count": float(rep.get("edge_count", 0.0)),
                "retained": float(rep.get("retained", 0.0)),
                "streak": float(rep.get("streak", 0.0)),
                "reward_hint": float(rep.get("reward_hint", 0.0)),
            }
            self._recent_role_reports[key] = payload

    def get_diversity_scorer(self, role_reward: float) -> RoleDiversityScorer:
        self.exposure_tracker.decay = self.diversity_control.exposure_decay
        return RoleDiversityScorer(
            control=self.diversity_control,
            exposure_tracker=self.exposure_tracker,
            popularity=self.item_popularity,
        )

    def register_feedback(self, metrics: Dict) -> None:
        current = self._extract_metric_snapshot(metrics)
        drop_signal = 0.0
        if not self._metric_ema:
            self._metric_ema = dict(current)
            self._last_metrics = current
            self.last_reward = None
            self.last_metric_drop = 0.0
            return
        if not self._last_active_states:
            decay = float(self._metric_ema_decay)
            for key, value in current.items():
                baseline = self._metric_ema.get(key, value)
                self._metric_ema[key] = decay * baseline + (1.0 - decay) * value
            self._last_metrics = current
            self.last_reward = None
            self.last_metric_drop = 0.0
            return
        if self._last_metrics is None:
            self._last_metrics = current
            self.last_reward = None
            self.last_metric_drop = 0.0
            return
        acc_reward = 0.0
        div_reward = 0.0
        decay = float(self._metric_ema_decay)
        for key, weight in self.dynamic_config.reward_weights.items():
            current_val = current.get(key, 0.0)
            baseline = self._metric_ema.get(key, current_val)
            delta = current_val - baseline
            self._metric_ema[key] = decay * baseline + (1.0 - decay) * current_val
            if weight * delta < 0:
                drop_signal = max(drop_signal, abs(delta))
            if key in self._diversity_keys:
                div_reward += weight * delta
            else:
                acc_reward += weight * delta
        if acc_reward * div_reward < 0:
            if abs(acc_reward) < abs(div_reward):
                acc_reward = 0.0
            else:
                div_reward = 0.0
        total_reward = acc_reward + div_reward
        self.last_reward = total_reward
        self.last_vector_reward = (acc_reward, div_reward)
        if total_reward < 0:
            drop_signal = max(drop_signal, -total_reward)
        self.last_metric_drop = drop_signal
        acc_grad = np.clip(acc_reward, -1.0, 1.0)
        div_grad = np.clip(div_reward, -1.0, 1.0)
        self.diversity_control.update_from_feedback(acc_grad, div_grad)
        eta = max(0.1, self.diversity_control.gamma_inv_pop)
        tau = 1.0 + max(0.0, div_grad) * 0.4
        role_coeff = 0.5 + 0.3 * np.tanh(div_reward)
        self.virtual_builder.update_sampling_params(eta, tau, role_coeff)
        if abs(total_reward) < 1e-8:
            self._last_metrics = current
            self._last_active_states = []
            self.last_metric_drop = drop_signal
            return
        reports = self._recent_role_reports or {}
        weighted_states: List[Tuple[RoleState, Dict[str, float], float]] = []
        total_weight = 0.0
        for state in self._last_active_states:
            key = (state.descriptor.rule_key, state.descriptor.position)
            report = reports.get(key, {})
            quality = float(report.get("quality", 0.0))
            if not np.isfinite(quality):
                quality = 0.0
            usage = max(1.0, float(report.get("edge_count", 1.0)))
            streak = max(0.0, float(report.get("streak", 0.0)))
            retained_bonus = 1.0 + 0.2 * float(report.get("retained", 0.0))
            weight = max(1e-3, quality + 1e-3) * usage * (1.0 + 0.1 * streak) * retained_bonus
            total_weight += weight
            weighted_states.append((state, report, weight))
        if total_weight <= 0.0 and weighted_states:
            total_weight = float(len(weighted_states))
        signals = [
            RoleUpdateSignal(
                rule_key=state.descriptor.rule_key,
                position=state.descriptor.position,
                reward=acc_reward * (weight / total_weight if total_weight > 0 else 1.0),
                diversity_reward=div_reward * (weight / total_weight if total_weight > 0 else 1.0),
                usage=max(1, int(report.get("edge_count", 1.0))) if report else 1,
                metrics=current,
            )
            for state, report, weight in weighted_states
        ]
        self.role_manager.record_feedback(signals)
        self._last_metrics = current
        self._last_active_states = []
        self._recent_role_reports.clear()

    def _extract_metric_snapshot(self, metrics: Dict) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for key in self.dynamic_config.reward_weights:
            value = metrics.get(key, 0.0)
            if isinstance(value, (list, tuple, np.ndarray)):
                snapshot[key] = float(value[0])
            else:
                snapshot[key] = float(value)
        return snapshot

    def clear_feedback_state(self, preserve_active_roles: bool = False) -> None:
        """
        Reset stored metric deltas so injection epochs provide a fresh baseline.

        When ``preserve_active_roles`` is False (default) the caller is expected to
        repopulate ``_last_active_states`` before the next feedback cycle.
        """
        self._last_metrics = None
        if not preserve_active_roles:
            self._last_active_states = []
        self.last_reward = None
        self.last_vector_reward = (0.0, 0.0)
        self.last_metric_drop = 0.0
        self._recent_role_reports.clear()
        # keep recent caches so schedulers can fall back if needed

    def _log_usage(self, epoch: int, metadata: Sequence[Dict]) -> None:
        if not metadata:
            return
        record = {
            "epoch": epoch,
            "entries": metadata,
        }
        with open(self.dynamic_config.usage_log, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_recent_role_pairs(self) -> List[Tuple[RoleState, np.ndarray]]:
        if not self._recent_role_pairs:
            return []
        return [
            (state, np.array(entities, copy=True))
            for state, entities in self._recent_role_pairs
        ]

    def get_recent_active_states(self, limit: Optional[int] = None) -> List[List[RoleState]]:
        snapshots = list(self._recent_active_states)
        if limit is not None:
            snapshots = snapshots[-int(max(1, limit)) :]
        return [list(states) for states in snapshots]

    def _record_active_states(self, states: Sequence[RoleState]) -> None:
        if not states:
            return
        self._recent_active_states.append([state for state in states])
