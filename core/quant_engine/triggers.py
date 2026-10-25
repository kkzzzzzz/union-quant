"""
统一可组合调仓触发器 (Composable Rebalance Triggers).

包含三种正交触发器：
1. CalendarTrigger: 日历周期触发（季度初、月初、半年初、每周/双周指定星期几）
2. DriftTrigger: 权重漂移触发（组合总单边换手漂移 或 非兜底资产单项漂移 + 冷却期）
3. ExtremumTrigger: 历史极值突破触发（RSI_14 + LogBias_20 在过去 N 日滚动分位数突破 + 冷却期 + 最长不调仓兜底）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.quant_engine.factors import calc_log_bias, calc_rsi_14


# ==============================================================================
# 1. 日历周期触发器 (CalendarTrigger)
# ==============================================================================
class CalendarTrigger:
    """日历周期调仓判定"""

    def __init__(
        self,
        frequency: str = "quarter",
        weekday: int = 4,
        interval_weeks: int = 1,
    ) -> None:
        self.frequency = (frequency or "quarter").lower()
        self.weekday = int(weekday)
        self.interval_weeks = max(1, int(interval_weeks))

    def get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        if len(dates) == 0:
            return []
        dates = pd.DatetimeIndex(sorted(dates.unique()))
        selected: List[pd.Timestamp] = []

        if self.frequency == "daily":
            return list(dates)

        if self.frequency == "weekly":
            # 筛选目标星期几（如周五 weekday=4）；若当周该日休市，选当周最后一个交易日
            s = pd.Series(dates, index=dates)
            iso = dates.isocalendar()
            week_key = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
            weekly_candidates: List[pd.Timestamp] = []
            for _, grp in s.groupby(week_key.values, sort=False):
                exact = grp[grp.dt.weekday == self.weekday]
                if not exact.empty:
                    weekly_candidates.append(exact.iloc[-1])
                else:
                    before = grp[grp.dt.weekday <= self.weekday]
                    if not before.empty:
                        weekly_candidates.append(before.iloc[-1])
            return weekly_candidates[:: self.interval_weeks]

        if self.frequency in ("month", "monthly"):
            s = pd.Series(dates, index=dates)
            return [grp.iloc[-1] for _, grp in s.groupby([dates.year, dates.month])]

        if self.frequency in ("quarter", "quarterly"):
            s = pd.Series(dates, index=dates)
            q_dates = [
                grp.iloc[-1]
                for (_, m), grp in s.groupby([dates.year, dates.month])
                if m in (3, 6, 9, 12)
            ]
            return q_dates

        if self.frequency in ("half_year", "semi_annual", "semiannual"):
            s = pd.Series(dates, index=dates)
            hy_dates = [
                grp.iloc[-1]
                for (_, m), grp in s.groupby([dates.year, dates.month])
                if m in (6, 12)
            ]
            return hy_dates

        if self.frequency in ("year", "yearly", "annual"):
            s = pd.Series(dates, index=dates)
            return [grp.iloc[-1] for _, grp in s.groupby(dates.year)]

        # 若为 extremum_signal 等纯信号模式，至少包含首日建仓
        return [dates[0]]


# ==============================================================================
# 2. 权重漂移触发器 (DriftTrigger)
# ==============================================================================
class DriftTrigger:
    """组合漂移与单项漂移监控"""

    def __init__(
        self,
        enabled: bool = False,
        portfolio_drift_threshold: float = 0.10,
        asset_drift_threshold: float = 0.08,
        cooldown_days: int = 20,
        residual_symbols: Optional[List[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.portfolio_drift_threshold = float(portfolio_drift_threshold)
        self.asset_drift_threshold = float(asset_drift_threshold)
        self.cooldown_days = int(cooldown_days)
        self.residual_symbols = set(residual_symbols or [])

    def check(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        days_since_last_rebalance: int,
    ) -> Tuple[bool, str]:
        if not self.enabled:
            return False, ""
        if days_since_last_rebalance < self.cooldown_days:
            return False, ""

        all_syms = set(current_weights.keys()) | set(target_weights.keys())
        total_turnover = 0.5 * sum(
            abs(current_weights.get(s, 0.0) - target_weights.get(s, 0.0))
            for s in all_syms
        )
        if total_turnover >= self.portfolio_drift_threshold:
            return (
                True,
                f"组合总漂移({total_turnover:.1%})>={self.portfolio_drift_threshold:.1%}",
            )

        for sym in all_syms:
            if sym in self.residual_symbols:
                continue
            diff = abs(current_weights.get(sym, 0.0) - target_weights.get(sym, 0.0))
            if diff >= self.asset_drift_threshold:
                return (
                    True,
                    f"资产[{sym}]漂移({diff:.1%})>={self.asset_drift_threshold:.1%}",
                )

        return False, ""


# ==============================================================================
# 3. 历史极值触发器 (ExtremumTrigger)
# ==============================================================================
@dataclass
class ExtremumSignalInfo:
    triggered: bool
    reason: str
    rsi_value: Optional[float] = None
    bias_value: Optional[float] = None


class ExtremumTrigger:
    """
    基于 RSI + 对数乖离率历史滚动分位数突破的极值触发器。
    """

    def __init__(
        self,
        enabled: bool = False,
        monitor_symbols: Optional[List[str]] = None,
        lookback_days: int = 756,
        rsi_window: int = 14,
        bias_window: int = 20,
        quantile_high: float = 0.99,
        quantile_low: float = 0.01,
        cooldown_days: int = 20,
        max_days_without_rebalance: int = 90,
    ) -> None:
        self.enabled = bool(enabled)
        self.monitor_symbols = list(monitor_symbols or [])
        self.lookback_days = int(lookback_days)
        self.rsi_window = int(rsi_window)
        self.bias_window = int(bias_window)
        self.quantile_high = float(quantile_high)
        self.quantile_low = float(quantile_low)
        self.cooldown_days = int(cooldown_days)
        self.max_days_without_rebalance = int(max_days_without_rebalance)
        self._precomputed: Dict[str, pd.DataFrame] = {}

    def prepare(self, prices_df: pd.DataFrame) -> None:
        """预先向量化计算监控标的的 RSI、LogBias 及其历史分位阈值"""
        self._precomputed.clear()
        if not self.enabled or prices_df.empty:
            return

        targets = self.monitor_symbols or list(prices_df.columns)
        min_p = min(60, max(20, self.lookback_days // 4))

        for sym in targets:
            if sym not in prices_df.columns:
                continue
            sub = pd.DataFrame({"close": prices_df[sym]}).dropna()
            if sub.empty:
                continue
            rsi = calc_rsi_14(sub, window=self.rsi_window)
            bias = calc_log_bias(sub, window=self.bias_window)

            # 使用 shift(1) 的历史窗口分位数避免前视偏差
            rsi_hi = rsi.shift(1).rolling(self.lookback_days, min_periods=min_p).quantile(
                self.quantile_high
            )
            rsi_lo = rsi.shift(1).rolling(self.lookback_days, min_periods=min_p).quantile(
                self.quantile_low
            )
            bias_hi = (
                bias.shift(1)
                .rolling(self.lookback_days, min_periods=min_p)
                .quantile(self.quantile_high)
            )
            bias_lo = (
                bias.shift(1)
                .rolling(self.lookback_days, min_periods=min_p)
                .quantile(self.quantile_low)
            )

            self._precomputed[sym] = pd.DataFrame(
                {
                    "rsi": rsi,
                    "bias": bias,
                    "rsi_hi": rsi_hi,
                    "rsi_lo": rsi_lo,
                    "bias_hi": bias_hi,
                    "bias_lo": bias_lo,
                },
                index=sub.index,
            )

    def check(
        self, dt: pd.Timestamp, days_since_last_rebalance: int
    ) -> ExtremumSignalInfo:
        if not self.enabled:
            return ExtremumSignalInfo(False, "")

        if days_since_last_rebalance < self.cooldown_days:
            return ExtremumSignalInfo(False, "")

        for sym, df in self._precomputed.items():
            if dt not in df.index:
                continue
            row = df.loc[dt]
            rsi, bias = row["rsi"], row["bias"]
            if pd.isna(rsi) or pd.isna(bias):
                continue

            if (
                not pd.isna(row["rsi_hi"])
                and not pd.isna(row["bias_hi"])
                and rsi >= row["rsi_hi"]
                and bias >= row["bias_hi"]
            ):
                return ExtremumSignalInfo(
                    True,
                    f"极值超买触发[{sym}](RSI={rsi:.1f}, Bias={bias:.2f}%)",
                    float(rsi),
                    float(bias),
                )

            if (
                not pd.isna(row["rsi_lo"])
                and not pd.isna(row["bias_lo"])
                and rsi <= row["rsi_lo"]
                and bias <= row["bias_lo"]
            ):
                return ExtremumSignalInfo(
                    True,
                    f"极值超卖触发[{sym}](RSI={rsi:.1f}, Bias={bias:.2f}%)",
                    float(rsi),
                    float(bias),
                )

        if (
            self.max_days_without_rebalance > 0
            and days_since_last_rebalance >= self.max_days_without_rebalance
        ):
            return ExtremumSignalInfo(
                True,
                f"超期兜底调仓(距上次{days_since_last_rebalance}交易日)",
            )

        return ExtremumSignalInfo(False, "")
