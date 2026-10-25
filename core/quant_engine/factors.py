"""
统一向量化因子引擎 (Unified Vectorized Factor Engine).

设计亮点：
1. 淘汰按标的和因子硬编码的 40+ 个 JSON 配置文件，改用 `@register_factor(factor_id)` 纯函数注册表。
2. 所有因子均基于 `market_daily` 标准 DataFrame (`close`, `pe`, `pb`, `dividend_yield`) 向量化计算。
3. 支持批量计算落库至 `factor_daily` 窄表，也支持在回测/实盘信号计算时按需实时动态计算（On-the-fly Fallback）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from core.config import AppConfig
from core.db import DatabaseGateway

logger = logging.getLogger(__name__)

FactorFunc = Callable[[pd.DataFrame], pd.Series]
FACTOR_REGISTRY: Dict[str, FactorFunc] = {}


def register_factor(factor_id: str) -> Callable[[FactorFunc], FactorFunc]:
    """因子注册装饰器"""

    def decorator(func: FactorFunc) -> FactorFunc:
        FACTOR_REGISTRY[factor_id] = func
        return func

    return decorator


# ==============================================================================
# 核心向量化算子与内置因子定义
# ==============================================================================
def _rolling_percentile(
    series: pd.Series, window: int, min_periods: int = 60
) -> pd.Series:
    """
    计算滚动历史分位数 (0.0 ~ 100.0)。
    定义为窗口内小于等于当前值的有效样本比例 * 100。
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(index=series.index, dtype=float)

    eff_min = min(min_periods, max(10, len(valid) // 2))

    def _pct_rank(arr: np.ndarray) -> float:
        clean = arr[~np.isnan(arr)]
        n = len(clean)
        if n < eff_min:
            return np.nan
        cur = clean[-1]
        return float(np.sum(clean <= cur) / n * 100.0)

    out = valid.rolling(window=window, min_periods=eff_min).apply(
        _pct_rank, raw=True
    )
    return out.reindex(series.index)


@register_factor("pe_percentile_10y")
def calc_pe_percentile_10y(df: pd.DataFrame) -> pd.Series:
    """10年滚动 PE 分位数 (窗口=2430个交易日，返回 0~100)"""
    if "pe" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return _rolling_percentile(df["pe"], window=2430, min_periods=60).round(4)


@register_factor("pe_percentile_5y")
def calc_pe_percentile_5y(df: pd.DataFrame) -> pd.Series:
    """5年滚动 PE 分位数 (窗口=1215个交易日，返回 0~100)"""
    if "pe" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return _rolling_percentile(df["pe"], window=1215, min_periods=60).round(4)


@register_factor("pb_percentile_5y")
def calc_pb_percentile_5y(df: pd.DataFrame) -> pd.Series:
    """5年滚动 PB 分位数 (窗口=1215个交易日，返回 0~100)"""
    if "pb" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return _rolling_percentile(df["pb"], window=1215, min_periods=60).round(4)


@register_factor("dy_percentile_5y")
def calc_dy_percentile_5y(df: pd.DataFrame) -> pd.Series:
    """5年滚动股息率分位数 (窗口=1215个交易日，返回 0~100)"""
    if "dividend_yield" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return _rolling_percentile(
        df["dividend_yield"], window=1215, min_periods=60
    ).round(4)


@register_factor("rsi_14")
def calc_rsi_14(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder 14日相对强弱指标 RSI (返回 0~100)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi.round(4)


@register_factor("log_bias")
def calc_log_bias(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """对数价格偏离度 Log-Bias = (ln(Close) - EMA(ln(Close), 20)) * 100"""
    close = pd.to_numeric(df["close"], errors="coerce")
    log_p = np.log(close.where(close > 0))
    ema_log_p = log_p.ewm(span=window, min_periods=window, adjust=False).mean()
    return ((log_p - ema_log_p) * 100.0).round(4)


@register_factor("ret_20d")
def calc_ret_20d(df: pd.DataFrame) -> pd.Series:
    """20日动量收益率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close.pct_change(20) * 100.0).round(4)


@register_factor("ret_1m")
def calc_ret_1m(df: pd.DataFrame) -> pd.Series:
    """1个月(21日)收益率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close.pct_change(21) * 100.0).round(4)


@register_factor("ret_3m")
def calc_ret_3m(df: pd.DataFrame) -> pd.Series:
    """3个月(63日)收益率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close.pct_change(63) * 100.0).round(4)


@register_factor("ret_6m")
def calc_ret_6m(df: pd.DataFrame) -> pd.Series:
    """6个月(126日)收益率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close.pct_change(126) * 100.0).round(4)


@register_factor("ret_1y")
def calc_ret_1y(df: pd.DataFrame) -> pd.Series:
    """1年(252日)收益率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    return (close.pct_change(252) * 100.0).round(4)


@register_factor("vol_30d")
def calc_vol_30d(df: pd.DataFrame) -> pd.Series:
    """30日滚动年化波动率 (%)"""
    close = pd.to_numeric(df["close"], errors="coerce")
    ret = close.pct_change()
    return (ret.rolling(30, min_periods=15).std() * np.sqrt(252) * 100.0).round(4)


# ==============================================================================
# 统一因子服务 (FactorService)
# ==============================================================================
class FactorService:
    """负责因子的批量计算落库与策略运行时的按需加载"""

    def __init__(
        self,
        db: Optional[DatabaseGateway] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.db = db or DatabaseGateway(config=self.config)

    @staticmethod
    def list_registered_factors() -> List[str]:
        return sorted(FACTOR_REGISTRY.keys())

    @staticmethod
    def compute_single_factor(
        df_market: pd.DataFrame, factor_id: str
    ) -> pd.Series:
        """基于单标的日线 DataFrame 直接计算因子序列 (Index 为 DatetimeIndex)"""
        if factor_id not in FACTOR_REGISTRY:
            raise ValueError(f"未注册的因子 ID: {factor_id}")
        if df_market.empty:
            return pd.Series(dtype=float)

        work = df_market.copy()
        work["dt"] = pd.to_datetime(work["trade_date"])
        work = work.drop_duplicates(subset=["dt"], keep="last").sort_values("dt")
        work = work.set_index("dt")
        series = FACTOR_REGISTRY[factor_id](work)
        series.name = factor_id
        return series.dropna()

    def compute_and_store_all(
        self,
        symbols: Optional[Iterable[str]] = None,
        tail_n: Optional[int] = None,
    ) -> int:
        """根据 `configs/assets.yaml` 中为每个标的配置的因子列表计算并写入 `factor_daily`"""
        assets = self.config.load_assets()
        target_symbols = list(symbols) if symbols else list(assets.keys())
        total_rows = 0

        for sym in target_symbols:
            spec = assets.get(sym)
            if not spec or not spec.factors:
                continue
            df_market = self.db.fetch_market_daily(sym)
            if df_market.empty:
                continue

            rows_to_upsert: List[Dict[str, Any]] = []
            for factor_id in spec.factors:
                if factor_id not in FACTOR_REGISTRY:
                    continue
                s = self.compute_single_factor(df_market, factor_id)
                if tail_n and tail_n > 0:
                    s = s.tail(tail_n)
                for dt, val in s.items():
                    rows_to_upsert.append(
                        {
                            "symbol": sym,
                            "trade_date": dt.strftime("%Y-%m-%d"),
                            "factor_id": factor_id,
                            "value": float(val),
                        }
                    )
            if rows_to_upsert:
                total_rows += self.db.upsert_factor_daily(rows_to_upsert)

        return total_rows

    def get_factor_series(
        self,
        symbol: str,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        df_market_fallback: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        供策略引擎读取因子序列（严格遵循 `end_date` 截断）：
        1. 若传入了 `df_market_fallback` 或数据库中因子为空，自动基于行情实时计算（保证回测/单测零外部依赖）。
        2. 否则从 `factor_daily` / `factor_value` 读取。
        """
        s = self.db.fetch_factor_series(
            symbol, factor_id, start_date=start_date, end_date=end_date
        )
        if not s.empty:
            return s

        df_mkt = (
            df_market_fallback
            if df_market_fallback is not None
            else self.db.fetch_market_daily(symbol, end_date=end_date)
        )
        if df_mkt is None or df_mkt.empty:
            return pd.Series(dtype=float)

        if end_date:
            df_mkt = df_mkt[
                pd.to_datetime(df_mkt["trade_date"]).dt.strftime("%Y-%m-%d")
                <= str(end_date)[:10]
            ]

        s = self.compute_single_factor(df_mkt, factor_id)
        if start_date:
            s = s[s.index >= pd.to_datetime(start_date)]
        if end_date:
            s = s[s.index <= pd.to_datetime(end_date)]
        return s
