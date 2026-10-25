"""
一个算策略 (One Quant Engine).
提供向量化因子计算、可组合调仓触发器、统一策略接口、T+1 回测引擎及每日信号流水线。
"""

from core.quant_engine.backtester import (
    BacktestEngine,
    BacktestResult,
    PerformanceAnalyzer,
)
from core.quant_engine.factors import (
    FACTOR_REGISTRY,
    FactorService,
    register_factor,
)
from core.quant_engine.pipeline import QuantPipeline
from core.quant_engine.strategies import (
    STRATEGY_REGISTRY,
    BaseStrategy,
    DualMomentumStrategy,
    DynamicAllocationStrategy,
    EqualWeightStrategy,
    SbeAllocationStrategy,
    StrategyDecision,
    ThresholdRotationStrategy,
    create_strategy,
    register_strategy,
)
from core.quant_engine.triggers import (
    CalendarTrigger,
    DriftTrigger,
    ExtremumTrigger,
)

__all__ = [
    "FACTOR_REGISTRY",
    "FactorService",
    "register_factor",
    "CalendarTrigger",
    "DriftTrigger",
    "ExtremumTrigger",
    "STRATEGY_REGISTRY",
    "BaseStrategy",
    "StrategyDecision",
    "DynamicAllocationStrategy",
    "DualMomentumStrategy",
    "ThresholdRotationStrategy",
    "SbeAllocationStrategy",
    "EqualWeightStrategy",
    "create_strategy",
    "register_strategy",
    "BacktestEngine",
    "BacktestResult",
    "PerformanceAnalyzer",
    "QuantPipeline",
]
