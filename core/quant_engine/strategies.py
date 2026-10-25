"""
统一策略插件引擎 (Unified Strategy Registry & Implementations).

涵盖全部 5 类核心策略：
1. `dynamic_engine` (DynamicAllocationStrategy): 估值分位驱动的多资产优先级动态配置（支持 linear / parabolic / power / fixed / residual）。
2. `etf_weekly_momentum` (DualMomentumStrategy): 全球/核心 ETF 双周/周度双动量（均线绝对趋势过滤 + N日相对动量最优胜出）。
3. `rotation` (ThresholdRotationStrategy): 多标的相对动量阈值轮动（含买入阈值、卖出阈值、持仓惯性、避险资产兜底）。
4. `sbe_notice` (SbeAllocationStrategy): 股债性价比(SBE)分位数驱动的股债动态配比策略。
5. `equal_weight` (EqualWeightStrategy): 基准等权再平衡策略。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd


@dataclass
class StrategyDecision:
    """单次策略决策输出"""

    signal_date: str
    weights: Dict[str, float]  # symbol -> weight (0.0 ~ 1.0, sum <= 1.0)
    asset_details: List[Dict[str, Any]] = field(default_factory=list)
    summary_reason: str = ""


class BaseStrategy(ABC):
    """统一策略抽象基类"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.strategy_id: str = config.get("strategy_id", "unknown")
        self.name: str = config.get("name", self.strategy_id)
        self.params: Dict[str, Any] = dict(config.get("strategy_params") or {})

    @abstractmethod
    def get_universe(self) -> List[str]:
        """返回该策略所需加载的标的列表"""

    @abstractmethod
    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        """在指定决策日 `dt` 计算目标权重及单资产诊断明细"""


TupleKey = tuple[str, str]  # (symbol, factor_id)

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(
    strategy_type: str,
) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        STRATEGY_REGISTRY[strategy_type] = cls
        return cls

    return decorator


def create_strategy(config: Dict[str, Any]) -> BaseStrategy:
    stype = config.get("type", "dynamic_engine")
    if stype not in STRATEGY_REGISTRY:
        raise ValueError(
            f"未知的策略类型 '{stype}'，已注册类型: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[stype](config)


# ==============================================================================
# 1. 估值分位数动态配置引擎 (DynamicAllocationStrategy -> type: "dynamic_engine")
# ==============================================================================
@register_strategy("dynamic_engine")
class DynamicAllocationStrategy(BaseStrategy):
    """
    优先级驱动的多资产动态仓位配置 (与原 `DynamicAllocationStrategy` 100% 公式对齐)：
    - 设估值分位数归一化值 p = pe_percentile / 100.0 ∈ [0, 1]：
      * parabolic: w = base_weight * max(0.0, 1.0 - (p ** c1))   (默认 c1 = 2.0)
      * power:     w = base_weight * ((1.0 - p) ** c1)           (默认 c1 = 1.5)
      * linear:    w = base_weight * max(0.0, 1.0 - c1 * p)      (默认 c1 = 1.0)
      * fixed:     w = base_weight
    - fixed 角色：固定目标权重 `base_weight` (或 `target_weight`)
    - residual 角色：承接剩余仓位 `max(0.0, 1.0 - sum_allocated)`
    - 缺失数据策略 (`missing_data_policy`): 默认为 `skip`（若某资产在当期尚未上市或无因子值，权重记为 0 并跳过）。
    """

    def get_universe(self) -> List[str]:
        assets = self.params.get("assets") or []
        return [str(a["symbol"]) for a in assets if a.get("symbol")]

    @staticmethod
    def _eval_curve(
        pct_0_to_100: Optional[float],
        strategy_name: str,
        base_weight: float,
        c1: Optional[float] = None,
        min_w: Optional[float] = None,
        max_w: Optional[float] = None,
    ) -> float:
        stype = (strategy_name or "linear").lower()
        if stype == "fixed":
            return float(base_weight)

        if pct_0_to_100 is None or pd.isna(pct_0_to_100):
            return 0.0

        p = float(np.clip(pct_0_to_100 / 100.0, 0.0, 1.0))

        # 若兼容配置了 min_weight / max_weight 且未配置 base_weight
        if base_weight <= 0.0 and max_w is not None:
            lo = float(min_w or 0.0)
            hi = float(max_w)
            coef = float(c1 if c1 is not None else 1.0)
            if stype == "parabolic":
                return float(np.clip(lo + (hi - lo) * (1.0 - (p ** coef)), lo, hi))
            if stype == "power":
                return float(np.clip(lo + (hi - lo) * ((1.0 - p) ** coef), lo, hi))
            return float(np.clip(lo + (hi - lo) * (1.0 - coef * p), lo, hi))

        if stype == "parabolic":
            coef = float(c1 if c1 is not None else 2.0)
            return float(base_weight * max(0.0, 1.0 - (p ** coef)))
        if stype == "power":
            coef = float(c1 if c1 is not None else 1.5)
            return float(base_weight * ((1.0 - p) ** coef))
        if stype == "linear":
            coef = float(c1 if c1 is not None else 1.0)
            return float(base_weight * max(0.0, 1.0 - coef * p))
        return float(base_weight * max(0.0, 1.0 - p))

    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        assets_cfg = sorted(
            self.params.get("assets") or [],
            key=lambda item: int(item.get("priority", 99)),
        )
        missing_policy = (self.params.get("missing_data_policy") or "skip").lower()
        weights: Dict[str, float] = {}
        details: List[Dict[str, Any]] = []
        remaining = 1.0

        for item in assets_cfg:
            sym = str(item["symbol"])
            name = item.get("name", sym)
            trade_sym = item.get("trade_symbol", sym)
            role = (item.get("role") or item.get("type") or "equity").lower()
            factor_id = item.get("factor_id")
            w_strat = item.get("weight_strategy") or item.get("curve_type") or "linear"
            base_w = float(
                item.get("base_weight")
                if item.get("base_weight") is not None
                else (item.get("target_weight") or 0.0)
            )
            c1_val = (
                item.get("c1")
                if item.get("c1") is not None
                else item.get("power")
            )

            # 获取截至 dt 该标的自身最新的价格与日期 (不因跨市场休市强行回退)
            latest_price, asset_data_date, latest_pe = self._extract_asset_snapshot(
                sym, dt, prices_df, raw_frames
            )

            factor_val: Optional[float] = None
            if factor_id:
                f_series = factors_map.get((sym, factor_id))
                if f_series is not None and not f_series.empty:
                    sub = f_series[f_series.index <= dt]
                    if not sub.empty:
                        factor_val = float(sub.iloc[-1])

            if remaining <= 0.0:
                weights[sym] = 0.0
                details.append(
                    {
                        "symbol": sym,
                        "name": name,
                        "trade_symbol": trade_sym,
                        "role": role,
                        "base_weight": base_w,
                        "weight_strategy": w_strat,
                        "c1": c1_val,
                        "weight": 0.0,
                        "data_date": asset_data_date,
                        "close": latest_price,
                        "pe": latest_pe,
                        "factor_id": factor_id,
                        "factor_value": round(factor_val, 2)
                        if factor_val is not None
                        else None,
                    }
                )
                continue

            if role == "residual":
                raw_w = max(0.0, remaining)
            elif role == "fixed":
                raw_w = base_w
            else:
                # 若当期该资产尚未上市或无因子数据，按 missing_data_policy 处理
                if latest_price is None or factor_val is None:
                    if missing_policy == "keep_previous" and current_weights:
                        raw_w = float(current_weights.get(sym, 0.0))
                    else:
                        raw_w = 0.0
                else:
                    raw_w = self._eval_curve(
                        pct_0_to_100=factor_val,
                        strategy_name=w_strat,
                        base_weight=base_w,
                        c1=float(c1_val) if c1_val is not None else None,
                        min_w=item.get("min_weight"),
                        max_w=item.get("max_weight"),
                    )

            alloc_w = round(min(max(0.0, raw_w), max(0.0, remaining)), 6)
            remaining = max(0.0, round(remaining - alloc_w, 6))
            weights[sym] = alloc_w

            details.append(
                {
                    "symbol": sym,
                    "name": name,
                    "trade_symbol": trade_sym,
                    "role": role,
                    "base_weight": base_w,
                    "weight_strategy": w_strat if role == "equity" else role,
                    "c1": c1_val,
                    "target_weight": round(raw_w, 6),
                    "weight": alloc_w,
                    "data_date": asset_data_date,
                    "close": latest_price,
                    "pe": latest_pe,
                    "factor_id": factor_id,
                    "factor_value": round(factor_val, 2)
                    if factor_val is not None
                    else None,
                }
            )

        return StrategyDecision(
            signal_date=dt.strftime("%Y-%m-%d"),
            weights=weights,
            asset_details=details,
            summary_reason="估值分位动态计算",
        )

    @staticmethod
    def _extract_asset_snapshot(
        sym: str,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        raw_frames: Optional[Dict[str, pd.DataFrame]],
    ) -> tuple[Optional[float], str, Optional[float]]:
        dt_str = dt.strftime("%Y-%m-%d")
        if raw_frames and sym in raw_frames:
            df = raw_frames[sym]
            sub = df[df["trade_date"] <= dt_str]
            if not sub.empty:
                last = sub.iloc[-1]
                pe_val = (
                    float(last["pe"])
                    if "pe" in last and pd.notna(last["pe"])
                    else None
                )
                return (
                    float(last["close"]),
                    str(last["trade_date"])[:10],
                    pe_val,
                )

        if sym in prices_df.columns:
            sub_s = prices_df[sym].loc[:dt].dropna()
            if not sub_s.empty:
                return float(sub_s.iloc[-1]), sub_s.index[-1].strftime("%Y-%m-%d"), None
        return None, dt_str, None


# ==============================================================================
# 2. 双动量/周度ETF轮动引擎 (DualMomentumStrategy -> type: "etf_weekly_momentum")
# ==============================================================================
@register_strategy("etf_weekly_momentum")
class DualMomentumStrategy(BaseStrategy):
    """
    ETF 双动量/周度相对动量轮动：
    1. 计算候选池各 ETF 过去 `momentum_window` 日收益率。
    2. 若 `enable_ma_filter=True`，要求当前收盘价 >= `ma_window` 日均线且动量 > 0。
    3. 按动量降序选前 `top_k` 个标的等权持有；若无标的通过过滤，则根據 `fallback_mode`
       转入现金 (`cash`, 权重全0) 或避险资产 (`safe_asset`)。
    """

    def get_universe(self) -> List[str]:
        syms = [
            str(a["symbol"])
            for a in (self.params.get("assets") or [])
            if a.get("symbol")
        ]
        safe_asset = self.params.get("safe_asset")
        if safe_asset and str(safe_asset) not in syms:
            syms.append(str(safe_asset))
        return syms

    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        mom_win = int(self.params.get("momentum_window", 12))
        ma_win = int(self.params.get("ma_window", mom_win))
        enable_ma = bool(self.params.get("enable_ma_filter", True))
        top_k = max(1, int(self.params.get("top_k", 1)))
        fallback_mode = (self.params.get("fallback_mode") or "cash").lower()
        safe_asset = self.params.get("safe_asset")

        candidates_cfg = self.params.get("assets") or []
        scored: List[Dict[str, Any]] = []

        for item in candidates_cfg:
            sym = str(item["symbol"])
            name = item.get("name", sym)
            if sym not in prices_df.columns:
                continue
            s = prices_df[sym].loc[:dt].dropna()
            if len(s) <= max(mom_win, ma_win):
                continue

            cur_price = float(s.iloc[-1])
            past_price = float(s.iloc[-1 - mom_win])
            mom_ret = (
                (cur_price / past_price - 1.0) if past_price > 0 else np.nan
            )
            ma_val = float(s.iloc[-ma_win:].mean())
            above_ma = bool(cur_price >= ma_val)
            passed = (above_ma and mom_ret > 0.0) if enable_ma else (not pd.isna(mom_ret))

            scored.append(
                {
                    "symbol": sym,
                    "name": name,
                    "trade_symbol": sym,
                    "close": round(cur_price, 4),
                    "ma_value": round(ma_val, 4),
                    "above_ma": above_ma,
                    "momentum_pct": round(float(mom_ret * 100.0), 3)
                    if not pd.isna(mom_ret)
                    else None,
                    "passed_filter": passed,
                    "data_date": s.index[-1].strftime("%Y-%m-%d"),
                }
            )

        # 按动量从高到低排序
        scored.sort(
            key=lambda x: (
                x["momentum_pct"] if x["momentum_pct"] is not None else -9999.0
            ),
            reverse=True,
        )

        qualified = [x for x in scored if x["passed_filter"]]
        winners = qualified[:top_k]

        weights: Dict[str, float] = {
            str(a["symbol"]): 0.0 for a in candidates_cfg if a.get("symbol")
        }
        if safe_asset:
            weights[str(safe_asset)] = 0.0

        if winners:
            w_each = round(1.0 / len(winners), 6)
            for win in winners:
                weights[win["symbol"]] = w_each
            reason = f"选中动量最优: {', '.join(w['name'] for w in winners)}"
        else:
            if fallback_mode == "safe_asset" and safe_asset:
                weights[str(safe_asset)] = 1.0
                reason = f"无标的通过趋势过滤，转入避险资产 {safe_asset}"
            else:
                reason = "无标的通过趋势过滤，空仓观望(Cash=100%)"

        for idx, item in enumerate(scored, start=1):
            item["rank"] = idx
            item["weight"] = weights.get(item["symbol"], 0.0)

        return StrategyDecision(
            signal_date=dt.strftime("%Y-%m-%d"),
            weights=weights,
            asset_details=scored,
            summary_reason=reason,
        )


# ==============================================================================
# 3. 阈值动量轮动引擎 (ThresholdRotationStrategy -> type: "rotation")
# ==============================================================================
@register_strategy("rotation")
class ThresholdRotationStrategy(BaseStrategy):
    """
    多标的动量阈值轮动（含持仓惯性与避险兜底）：
    - 计算各候选资产 `lookback_days` 动量收益率
    - 若最优资产动量 >= `buy_threshold`，切换/持有最优资产
    - 若当前持仓资产动量 >= `sell_threshold` 且高于其他候选，则维持原持仓（减少磨损）
    - 若全部低于阈值，则转入 `safe_asset`（若未配置则空仓）
    """

    def get_universe(self) -> List[str]:
        syms = [
            str(a["symbol"])
            for a in (self.params.get("assets") or [])
            if a.get("symbol")
        ]
        safe_asset = self.params.get("safe_asset")
        if safe_asset and str(safe_asset) not in syms:
            syms.append(str(safe_asset))
        return syms

    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        lookback = int(self.params.get("lookback_days", 20))
        buy_th = float(self.params.get("buy_threshold", 0.0))
        sell_th = float(self.params.get("sell_threshold", -0.01))
        safe_asset = self.params.get("safe_asset")
        candidates_cfg = self.params.get("assets") or []

        scored: List[Dict[str, Any]] = []
        for item in candidates_cfg:
            sym = str(item["symbol"])
            name = item.get("name", sym)
            if sym not in prices_df.columns:
                continue
            s = prices_df[sym].loc[:dt].dropna()
            if len(s) <= lookback:
                continue
            ret = float(s.iloc[-1] / s.iloc[-1 - lookback] - 1.0)
            scored.append(
                {
                    "symbol": sym,
                    "name": name,
                    "trade_symbol": sym,
                    "close": round(float(s.iloc[-1]), 4),
                    "momentum": ret,
                    "momentum_pct": round(ret * 100.0, 3),
                    "data_date": s.index[-1].strftime("%Y-%m-%d"),
                }
            )

        scored.sort(key=lambda x: x["momentum"], reverse=True)
        weights: Dict[str, float] = {
            str(a["symbol"]): 0.0 for a in candidates_cfg if a.get("symbol")
        }
        if safe_asset:
            weights[str(safe_asset)] = 0.0

        chosen_sym: Optional[str] = None
        if scored:
            best = scored[0]
            if best["momentum"] >= buy_th:
                chosen_sym = best["symbol"]
            elif current_weights:
                # 检查当前已持有的非避险资产是否仍在卖出阈值之上
                for item in scored:
                    if (
                        current_weights.get(item["symbol"], 0.0) > 0.5
                        and item["momentum"] >= sell_th
                    ):
                        chosen_sym = item["symbol"]
                        break

        if chosen_sym:
            weights[chosen_sym] = 1.0
            reason = f"轮动持有 {chosen_sym}"
        elif safe_asset:
            weights[str(safe_asset)] = 1.0
            reason = f"动量低于阈值，转入避险资产 {safe_asset}"
        else:
            reason = "动量低于阈值，空仓观望"

        for idx, item in enumerate(scored, start=1):
            item["rank"] = idx
            item["weight"] = weights.get(item["symbol"], 0.0)

        return StrategyDecision(
            signal_date=dt.strftime("%Y-%m-%d"),
            weights=weights,
            asset_details=scored,
            summary_reason=reason,
        )


# ==============================================================================
# 4. 股债性价比策略 (SbeAllocationStrategy -> type: "sbe_notice")
# ==============================================================================
@register_strategy("sbe_notice")
class SbeAllocationStrategy(BaseStrategy):
    """
    股债性价比动态配置策略：
    - 若数据源为韭圈儿 (`jiucaishuo`)：分位数 `v` 越高代表股票越便宜，`equity_w = (v / 100)^2`
    - 若数据源为蛋卷 (`danjuan`)：估值分位数 `v` 越低代表股票越便宜，`equity_w = ((100 - v) / 100)^2`
    - 若未采集到外部 SBE 指标，自动使用权益指数自身 `pe_percentile_10y` (`((100 - pe_pct)/100)^2`) 无缝兜底。
    """

    def get_universe(self) -> List[str]:
        syms: List[str] = []
        for key in ("sbe_symbol", "equity_symbol", "bond_symbol"):
            val = self.params.get(key)
            if val and str(val) not in syms:
                syms.append(str(val))
        return syms

    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        sbe_sym = str(self.params.get("sbe_symbol", "SBE.JCS"))
        sbe_source = (self.params.get("sbe_source") or "jiucaishuo").lower()
        eq_sym = str(self.params.get("equity_symbol", "000300.SH"))
        eq_name = self.params.get("equity_name", "股票仓位")
        bond_sym = str(self.params.get("bond_symbol", "CBA02701.CS"))
        bond_name = self.params.get("bond_name", "债券仓位")
        min_eq = float(self.params.get("min_equity_weight", 0.0))
        max_eq = float(self.params.get("max_equity_weight", 1.0))

        sbe_val: Optional[float] = None
        effective_source = sbe_source

        if sbe_sym in prices_df.columns:
            sub = prices_df[sbe_sym].loc[:dt].dropna()
            if not sub.empty:
                sbe_val = float(sub.iloc[-1])

        # 若无独立 SBE 序列，自动回退权益指数 10年 PE 分位数衍生计算
        if sbe_val is None:
            pe_series = factors_map.get((eq_sym, "pe_percentile_10y"))
            if pe_series is not None and not pe_series.empty:
                sub_pe = pe_series[pe_series.index <= dt]
                if not sub_pe.empty:
                    # 转换为股债性价比分位 (100 - PE分位)
                    sbe_val = float(100.0 - sub_pe.iloc[-1])
                    effective_source = "jiucaishuo"

        if sbe_val is None:
            eq_w = 0.5
        else:
            v = float(np.clip(sbe_val, 0.0, 100.0))
            if effective_source == "danjuan":
                raw_eq = ((100.0 - v) / 100.0) ** 2
            else:
                raw_eq = (v / 100.0) ** 2
            eq_w = float(np.clip(raw_eq, min_eq, max_eq))

        eq_w = round(eq_w, 4)
        bond_w = round(1.0 - eq_w, 4)
        weights = {eq_sym: eq_w, bond_sym: bond_w}

        details = [
            {
                "symbol": eq_sym,
                "name": eq_name,
                "role": "equity",
                "weight": eq_w,
                "sbe_quantile": round(sbe_val, 2) if sbe_val is not None else None,
                "sbe_source": effective_source,
            },
            {
                "symbol": bond_sym,
                "name": bond_name,
                "role": "bond",
                "weight": bond_w,
            },
        ]

        return StrategyDecision(
            signal_date=dt.strftime("%Y-%m-%d"),
            weights=weights,
            asset_details=details,
            summary_reason=f"SBE分位={sbe_val:.1f}% -> 股{eq_w:.1%}/债{bond_w:.1%}"
            if sbe_val is not None
            else "默认股债均衡配置",
        )


# ==============================================================================
# 5. 等权基准策略 (EqualWeightStrategy -> type: "equal_weight")
# ==============================================================================
@register_strategy("equal_weight")
class EqualWeightStrategy(BaseStrategy):
    """简单等权再平衡策略"""

    def get_universe(self) -> List[str]:
        return [
            str(a["symbol"])
            for a in (self.params.get("assets") or [])
            if a.get("symbol")
        ]

    def compute_decision(
        self,
        dt: pd.Timestamp,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        current_weights: Optional[Dict[str, float]] = None,
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyDecision:
        syms = [s for s in self.get_universe() if s in prices_df.columns]
        if not syms:
            return StrategyDecision(dt.strftime("%Y-%m-%d"), {}, [], "无有效资产")
        w = round(1.0 / len(syms), 6)
        weights = {s: w for s in syms}
        details = [{"symbol": s, "name": s, "weight": w} for s in syms]
        return StrategyDecision(
            signal_date=dt.strftime("%Y-%m-%d"),
            weights=weights,
            asset_details=details,
            summary_reason="等权配置",
        )
