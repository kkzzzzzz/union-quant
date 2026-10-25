"""
统一回测与绩效评估引擎 (BacktestEngine & PerformanceAnalyzer).

核心规范：
1. 严格遵循 T+1 执行语义 (`next_day_close_proxy`)：
   在决策日 `T` 基于截至 `T` 的数据产生目标权重信号，在 `T+1` 交易日按收盘代理价成交并扣除手续费与滑点。
2. 支持可组合触发器：`CalendarTrigger` + `DriftTrigger` + `ExtremumTrigger`。
3. 输出完整的净值曲线 (`equity_curve`)、调仓日志 (`rebalance_log`) 以及标准化绩效指标 (`metrics`)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.quant_engine.strategies import BaseStrategy, StrategyDecision, TupleKey
from core.quant_engine.triggers import CalendarTrigger, DriftTrigger, ExtremumTrigger


@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    equity_curve: pd.DataFrame  # index=DatetimeIndex, cols=[nav, daily_return, drawdown, cash_weight, ...]
    rebalance_log: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    latest_decision: Optional[StrategyDecision] = None


class PerformanceAnalyzer:
    """组合风险收益指标计算器"""

    @staticmethod
    def evaluate(
        equity_curve: pd.DataFrame,
        rebalance_log: List[Dict[str, Any]],
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        if equity_curve.empty or "nav" not in equity_curve.columns:
            return {}

        nav = equity_curve["nav"].dropna()
        if len(nav) < 2:
            return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        rets = nav.pct_change().dropna()
        total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
        days = max(1, (nav.index[-1] - nav.index[0]).days)
        years = days / 365.25
        cagr = (
            float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
            if years > 0 and nav.iloc[0] > 0
            else total_ret
        )

        # 最大回撤
        cum_max = nav.cummax()
        dd = (nav - cum_max) / cum_max
        max_dd = float(dd.min())

        # 年化波动率、夏普、索提诺、卡玛
        ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = (
            float((cagr - risk_free_rate) / ann_vol) if ann_vol > 1e-8 else 0.0
        )

        downside = rets[rets < 0]
        downside_vol = (
            float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
        )
        sortino = (
            float((cagr - risk_free_rate) / downside_vol)
            if downside_vol > 1e-8
            else 0.0
        )
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0
        win_rate = float((rets > 0).mean()) if len(rets) > 0 else 0.0

        # 年度收益率分解
        yearly_nav = nav.resample("YE").last()
        annual_returns: Dict[str, float] = {}
        prev_val = float(nav.iloc[0])
        for dt, val in yearly_nav.items():
            yr = str(dt.year)
            annual_returns[yr] = round(float(val / prev_val - 1.0), 4)
            prev_val = float(val)

        return {
            "start_date": nav.index[0].strftime("%Y-%m-%d"),
            "end_date": nav.index[-1].strftime("%Y-%m-%d"),
            "trading_days": int(len(nav)),
            "total_return": round(total_ret, 4),
            "cagr": round(cagr, 4),
            "annual_volatility": round(ann_vol, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "daily_win_rate": round(win_rate, 4),
            "rebalance_count": len(rebalance_log),
            "final_nav": round(float(nav.iloc[-1]), 4),
            "annual_returns": annual_returns,
        }


class BacktestEngine:
    """统一向量化/事件混合回测引擎 (严格 T+1 执行)"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        exec_cfg = config.get("execution") or {}
        self.mode = exec_cfg.get("mode", "next_day_close_proxy")
        self.initial_capital = float(exec_cfg.get("initial_capital", 1_000_000.0))
        self.commission_rate = float(exec_cfg.get("commission_rate", 0.0005))
        self.slippage_rate = float(exec_cfg.get("slippage_rate", 0.0005))

        reb_cfg = config.get("rebalance") or {}
        self.frequency = (reb_cfg.get("frequency") or "quarter").lower()
        self.calendar_trigger = CalendarTrigger(
            frequency=self.frequency,
            weekday=int(reb_cfg.get("weekday", 4)),
            interval_weeks=int(reb_cfg.get("interval_weeks", 1)),
        )

        # 提取 residual 标的，供漂移触发器豁免
        params = config.get("strategy_params") or {}
        residuals = [
            str(a["symbol"])
            for a in (params.get("assets") or [])
            if (a.get("role") or "").lower() == "residual" and a.get("symbol")
        ]
        drift_cfg = reb_cfg.get("drift_trigger") or {}
        self.drift_trigger = DriftTrigger(
            enabled=bool(drift_cfg.get("enabled", False)),
            portfolio_drift_threshold=float(
                drift_cfg.get("portfolio_drift_threshold", 0.10)
            ),
            asset_drift_threshold=float(
                drift_cfg.get("asset_drift_threshold", 0.08)
            ),
            cooldown_days=int(drift_cfg.get("cooldown_days", 20)),
            residual_symbols=residuals,
        )

        ext_cfg = reb_cfg.get("extremum_trigger") or {}
        self.extremum_trigger = ExtremumTrigger(
            enabled=bool(
                ext_cfg.get("enabled", False)
                or self.frequency == "extremum_signal"
            ),
            monitor_symbols=ext_cfg.get("monitor_symbols"),
            lookback_days=int(ext_cfg.get("lookback_days", 756)),
            rsi_window=int(ext_cfg.get("rsi_window", 14)),
            bias_window=int(ext_cfg.get("bias_window", 20)),
            quantile_high=float(ext_cfg.get("quantile_high", 0.99)),
            quantile_low=float(ext_cfg.get("quantile_low", 0.01)),
            cooldown_days=int(ext_cfg.get("cooldown_days", 20)),
            max_days_without_rebalance=int(
                ext_cfg.get("max_days_without_rebalance", 90)
            ),
        )

    def run(
        self,
        strategy: BaseStrategy,
        prices_df: pd.DataFrame,
        factors_map: Dict[TupleKey, pd.Series],
        raw_frames: Optional[Dict[str, pd.DataFrame]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        if prices_df.empty:
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                strategy_name=strategy.name,
                equity_curve=pd.DataFrame(),
            )

        df_p = prices_df.sort_index().ffill()
        if start_date:
            df_p = df_p[df_p.index >= pd.to_datetime(start_date)]
        if end_date:
            df_p = df_p[df_p.index <= pd.to_datetime(end_date)]

        # 去掉所有交易标的尚未全部上市的前导空行（保留宏观指标如 SBE.JCS 可为空）
        trade_syms = [
            s
            for s in strategy.get_universe()
            if s in df_p.columns and not s.startswith("SBE.")
        ]
        if trade_syms:
            df_p = df_p.dropna(subset=trade_syms, how="all")

        if len(df_p) < 2:
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                strategy_name=strategy.name,
                equity_curve=pd.DataFrame(),
            )

        # 预计算极值指标（使用全量历史窗口，避免截断前视期）
        self.extremum_trigger.prepare(prices_df.sort_index().ffill())

        dates = df_p.index
        calendar_set = set(self.calendar_trigger.get_rebalance_dates(dates))
        daily_Asset_rets = df_p.pct_change().fillna(0.0)

        current_weights: Dict[str, float] = {s: 0.0 for s in df_p.columns}
        pending_order: Optional[tuple[StrategyDecision, str]] = None
        rebalance_log: List[Dict[str, Any]] = []
        nav = 1.0
        nav_records: List[Dict[str, Any]] = []
        days_since_rebalance = 999999
        has_initialized = False

        cost_rate = self.commission_rate + self.slippage_rate

        for i, dt in enumerate(dates):
            # ----------------------------------------------------------
            # 步骤 1: 若昨日 (T) 产生了调仓指令，在今日 (T+1) 执行并扣除交易成本
            # ----------------------------------------------------------
            if pending_order is not None:
                decision, trigger_reason = pending_order
                target_w = decision.weights
                all_syms = set(current_weights.keys()) | set(target_w.keys())
                one_way_turnover = 0.5 * sum(
                    abs(target_w.get(s, 0.0) - current_weights.get(s, 0.0))
                    for s in all_syms
                )
                # 仅当换手率 > 0.1% 或首次建仓时真正调仓扣费
                if one_way_turnover > 0.001 or not has_initialized:
                    trade_cost = one_way_turnover * 2.0 * cost_rate
                    nav *= max(0.0, 1.0 - trade_cost)
                    current_weights = {
                        s: float(target_w.get(s, 0.0))
                        for s in df_p.columns
                        if not s.startswith("SBE.")
                    }
                    rebalance_log.append(
                        {
                            "signal_date": decision.signal_date,
                            "execution_date": dt.strftime("%Y-%m-%d"),
                            "trigger_reason": trigger_reason,
                            "turnover": round(one_way_turnover, 4),
                            "weights": dict(current_weights),
                            "summary": decision.summary_reason,
                        }
                    )
                    days_since_rebalance = 0
                    has_initialized = True
                pending_order = None

            # ----------------------------------------------------------
            # 步骤 2: 计算今日组合收益与收盘后自然漂移后的实际持仓权重
            # ----------------------------------------------------------
            if i > 0 and has_initialized:
                row_ret = daily_Asset_rets.loc[dt]
                port_ret = sum(
                    w * float(row_ret.get(sym, 0.0))
                    for sym, w in current_weights.items()
                )
                nav *= 1.0 + port_ret
                # 更新收盘后各资产自然漂移权重
                denom = 1.0 + port_ret
                if denom > 0:
                    current_weights = {
                        sym: w * (1.0 + float(row_ret.get(sym, 0.0))) / denom
                        for sym, w in current_weights.items()
                    }
                days_since_rebalance += 1
            else:
                port_ret = 0.0

            cash_w = max(0.0, 1.0 - sum(current_weights.values()))
            rec: Dict[str, Any] = {
                "trade_date": dt,
                "nav": nav,
                "daily_return": port_ret,
                "cash_weight": round(cash_w, 4),
            }
            for sym, w in current_weights.items():
                rec[f"w_{sym}"] = round(w, 4)
            nav_records.append(rec)

            # ----------------------------------------------------------
            # 步骤 3: 今日收盘后 (T日)，检查是否触发调仓信号 (将在 T+1 执行)
            # ----------------------------------------------------------
            trigger_fired = False
            trigger_reason = ""

            if not has_initialized and i == 0:
                trigger_fired = True
                trigger_reason = "初始建仓"
            elif self.frequency != "extremum_signal" and dt in calendar_set:
                trigger_fired = True
                trigger_reason = f"定期调仓({self.frequency})"
            else:
                # 检查极值触发
                ext_info = self.extremum_trigger.check(
                    dt, days_since_rebalance
                )
                if ext_info.triggered:
                    trigger_fired = True
                    trigger_reason = ext_info.reason
                elif self.drift_trigger.enabled and has_initialized:
                    cand_decision = strategy.compute_decision(
                        dt,
                        df_p.loc[:dt],
                        factors_map,
                        current_weights=current_weights,
                        raw_frames=raw_frames,
                    )
                    drifted, d_reason = self.drift_trigger.check(
                        current_weights,
                        cand_decision.weights,
                        days_since_rebalance,
                    )
                    if drifted:
                        trigger_fired = True
                        trigger_reason = d_reason
                        pending_order = (cand_decision, trigger_reason)

            if trigger_fired and pending_order is None:
                decision = strategy.compute_decision(
                    dt,
                    df_p.loc[:dt],
                    factors_map,
                    current_weights=current_weights,
                    raw_frames=raw_frames,
                )
                if self.mode == "same_day_close":
                    # 同日收盘模式（仅用于对比实验）
                    current_weights = {
                        s: float(decision.weights.get(s, 0.0))
                        for s in df_p.columns
                        if not s.startswith("SBE.")
                    }
                    rebalance_log.append(
                        {
                            "signal_date": decision.signal_date,
                            "execution_date": dt.strftime("%Y-%m-%d"),
                            "trigger_reason": trigger_reason,
                            "turnover": 0.0,
                            "weights": dict(current_weights),
                            "summary": decision.summary_reason,
                        }
                    )
                    days_since_rebalance = 0
                    has_initialized = True
                else:
                    # 默认标准 T+1 模式：挂单到下一交易日执行
                    pending_order = (decision, trigger_reason)

        eq_df = pd.DataFrame(nav_records).set_index("trade_date")
        eq_df["drawdown"] = (eq_df["nav"] - eq_df["nav"].cummax()) / eq_df[
            "nav"
        ].cummax()

        # 计算最新截至末尾日期的实时决策快照
        latest_dt = dates[-1]
        latest_decision = strategy.compute_decision(
            latest_dt,
            df_p,
            factors_map,
            current_weights=current_weights,
            raw_frames=raw_frames,
        )

        metrics = PerformanceAnalyzer.evaluate(eq_df, rebalance_log)
        return BacktestResult(
            strategy_id=strategy.strategy_id,
            strategy_name=strategy.name,
            equity_curve=eq_df,
            rebalance_log=rebalance_log,
            metrics=metrics,
            latest_decision=latest_decision,
        )
