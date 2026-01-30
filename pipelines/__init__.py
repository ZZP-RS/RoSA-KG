from .dynamic_rule_trainer import DynamicRuleTrainer, DynamicRuleConfig

# Alias for RoSA-KG components:
#   - RSRG: Rule-driven Structural Roles Generator
#   - MSRS: Multi-view Structural Reliability Selector
RSRGConfig = DynamicRuleConfig
RSRG = DynamicRuleTrainer

__all__ = [
    "DynamicRuleTrainer",
    "DynamicRuleConfig",
    "RSRG",
    "RSRGConfig",
]
