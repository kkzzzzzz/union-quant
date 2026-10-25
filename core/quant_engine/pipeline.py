"""
统一策略回测与每日信号流水线 (Unified Quant Pipeline).

核心规则：
1. 严格执行 T+1 数据截断准则 (`T_PLUS_1_DATA_ALIGNMENT.md`)：
   - 常规盘前/盘中信号计算：当运行日为 `run_date` (T+1) 时，物理截断 `end_date = run_date - 1 day` (`trade_date <= end_date`)，杜绝当日未收盘脏数据进入计算。
   - 周五收盘后双动量策略 (`signal_cutoff == "same_day_close"`)：`end_date = run_date`。
2. 跨市场资产展示原则：
   - 不强行对所有资产取 `min(latest_date)`，每个资产在 `asset_details` 中展示其截至 `end_date` 自身最新的真实行情日期与估值分位。
3. 单表 JSONB 信号持久化：
   - 将目标权重 `weights`、单资产明细 `asset_details`、回测指标 `metrics_snapshot`、配置快照 `config_snapshot` 一次性幂等写入 `strategy_signals` 单表。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import OUTPUT_DIR, AppConfig
from core.data_hub.sync import DataSyncService
from core.db import DatabaseGateway
from core.notifier import PushNotifier
from core.quant_engine.backtester import BacktestEngine, BacktestResult
from core.quant_engine.factors import FactorService
from core.quant_engine.strategies import TupleKey, create_strategy

logger = logging.getLogger(__name__)


class QuantPipeline:
    """统一量化流水线服务"""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db: Optional[DatabaseGateway] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.db = db or DatabaseGateway(config=self.config)
        self.data_service = DataSyncService(db=self.db, config=self.config)
        self.factor_service = FactorService(db=self.db, config=self.config)
        self.notifier = PushNotifier()

    @staticmethod
    def resolve_cutoff_date(
        strategy_cfg: Dict[str, Any], run_date: Optional[str] = None
    ) -> str:
        """
        根据 T+1 准则计算物理数据截断日期 `end_date`：
        - 默认策略 (盘前/每日运行)：`end_date = run_date - 1 天`
        - 配置了 `execution.signal_cutoff == "same_day_close"` 的策略（如周五盘后运行）：`end_date = run_date`
        """
        dt_run = (
            pd.to_datetime(run_date)
            if run_date
            else pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
        )
        exec_cfg = strategy_cfg.get("execution") or {}
        cutoff_mode = exec_cfg.get("signal_cutoff", "previous_day_close")
        if cutoff_mode == "same_day_close":
            return dt_run.strftime("%Y-%m-%d")
        return (dt_run - timedelta(days=1)).strftime("%Y-%m-%d")

    def _load_strategy_inputs(
        self,
        strategy_cfg: Dict[str, Any],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[TupleKey, pd.Series]]:
        strategy = create_strategy(strategy_cfg)
        universe = strategy.get_universe()

        prices_df, raw_frames = self.data_service.load_price_matrix(
            universe, start_date=None, end_date=end_date, ffill=True
        )

        # 加载策略所需的因子序列
        factors_map: Dict[TupleKey, pd.Series] = {}
        params = strategy_cfg.get("strategy_params") or {}
        for item in params.get("assets") or []:
            sym = item.get("symbol")
            fid = item.get("factor_id")
            if sym and fid:
                factors_map[(str(sym), str(fid))] = (
                    self.factor_service.get_factor_series(
                        str(sym),
                        str(fid),
                        end_date=end_date,
                        df_market_fallback=raw_frames.get(str(sym)),
                    )
                )

        # 若为 SBE 策略，预加载权益标的的 10年 PE 分位作为无 SBE 数据时的自动兜底
        if strategy_cfg.get("type") == "sbe_notice":
            eq_sym = str(params.get("equity_symbol", "000300.SH"))
            factors_map[(eq_sym, "pe_percentile_10y")] = (
                self.factor_service.get_factor_series(
                    eq_sym,
                    "pe_percentile_10y",
                    end_date=end_date,
                    df_market_fallback=raw_frames.get(eq_sym),
                )
            )

        return prices_df, raw_frames, factors_map

    # ------------------------------------------------------------------
    # 1. 运行策略回测
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        strategy_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_outputs: bool = True,
    ) -> BacktestResult:
        strategy_cfg = self.config.load_strategy_config(strategy_id)
        strategy = create_strategy(strategy_cfg)
        prices_df, raw_frames, factors_map = self._load_strategy_inputs(
            strategy_cfg, start_date=start_date, end_date=end_date
        )

        engine = BacktestEngine(strategy_cfg)
        result = engine.run(
            strategy=strategy,
            prices_df=prices_df,
            factors_map=factors_map,
            raw_frames=raw_frames,
            start_date=start_date,
            end_date=end_date,
        )

        if save_outputs and not result.equity_curve.empty:
            out_dir = OUTPUT_DIR / strategy.strategy_id
            out_dir.mkdir(parents=True, exist_ok=True)
            result.equity_curve.to_csv(out_dir / "equity_curve.csv")
            pd.DataFrame(result.rebalance_log).to_csv(
                out_dir / "rebalance_log.csv", index=False
            )

        return result

    # ------------------------------------------------------------------
    # 2. 运行每日实盘/演练信号流水线并入库 + 推送
    # ------------------------------------------------------------------
    def run_daily_signal(
        self,
        strategy_id: str,
        run_date: Optional[str] = None,
        run_mode: str = "live",
        notify: bool = True,
    ) -> Dict[str, Any]:
        """
        生成每日策略目标仓位信号：
        - `run_mode="live"`: 写入 `strategy_signals` 数据库表并触发通知推送
        - `run_mode="dry-run"`: 仅计算并输出本地快照，不污染生产表
        """
        strategy_cfg = self.config.load_strategy_config(strategy_id)
        cutoff_date = self.resolve_cutoff_date(strategy_cfg, run_date=run_date)

        bt_result = self.run_backtest(
            strategy_id=strategy_id,
            end_date=cutoff_date,
            save_outputs=False,
        )

        if bt_result.latest_decision is None:
            raise RuntimeError(
                f"策略 [{strategy_id}] 在截断日期 <= {cutoff_date} 下无可用行情数据，请先执行 sync"
            )

        decision = bt_result.latest_decision
        last_reb = (
            bt_result.rebalance_log[-1] if bt_result.rebalance_log else {}
        )
        is_rebalance_day = bool(
            last_reb and last_reb.get("signal_date") == decision.signal_date
        )
        trigger_reason = (
            last_reb.get("trigger_reason", decision.summary_reason)
            if is_rebalance_day
            else f"持仓观察({decision.summary_reason})"
        )
        cash_weight = round(
            max(0.0, 1.0 - sum(decision.weights.values())), 6
        )

        payload: Dict[str, Any] = {
            "strategy_id": strategy_cfg["strategy_id"],
            "strategy_name": strategy_cfg.get("name", strategy_cfg["strategy_id"]),
            "strategy_version": str(strategy_cfg.get("version", "2.0")),
            "signal_date": decision.signal_date,
            "execution_date": (
                run_date or datetime.now().strftime("%Y-%m-%d")
            )[:10],
            "run_mode": run_mode,
            "is_rebalance_day": is_rebalance_day,
            "trigger_reason": trigger_reason,
            "cash_weight": cash_weight,
            "weights": decision.weights,
            "asset_details": decision.asset_details,
            "metrics_snapshot": bt_result.metrics,
            "config_snapshot": strategy_cfg,
        }

        if run_mode == "live":
            self.db.upsert_strategy_signal(payload)

        markdown_msg = self.format_signal_markdown(payload, cutoff_date)
        payload["markdown_report"] = markdown_msg

        if notify:
            title = f"[{payload['strategy_name']}] 信号 ({payload['signal_date']})"
            self.notifier.send(title, markdown_msg)

        return payload

    @staticmethod
    def format_signal_markdown(
        payload: Dict[str, Any], cutoff_date: str
    ) -> str:
        lines: List[str] = [
            f"- **策略名称**: {payload['strategy_name']} (`{payload['strategy_id']}`)",
            f"- **信号基准日 (T)**: `{payload['signal_date']}` (数据截断 `<= {cutoff_date}`)",
            f"- **计划执行日 (T+1)**: `{payload['execution_date']}`",
            f"- **调仓状态**: {'**触发调仓**' if payload['is_rebalance_day'] else '维持观察'} ({payload['trigger_reason']})",
            "",
            "#### 目标仓位配置",
        ]

        for item in payload.get("asset_details") or []:
            w = float(item.get("weight", 0.0))
            name = item.get("name", item.get("symbol"))
            sym = item.get("symbol")
            dt_str = item.get("data_date", payload["signal_date"])
            extra_parts: List[str] = []
            if item.get("factor_value") is not None:
                extra_parts.append(
                    f"{item.get('factor_id')}={item['factor_value']}%"
                )
            if item.get("momentum_pct") is not None:
                extra_parts.append(f"动量={item['momentum_pct']}%")
            if item.get("sbe_quantile") is not None:
                extra_parts.append(f"SBE分位={item['sbe_quantile']}%")
            extra_info = f" ({', '.join(extra_parts)}, 数据日:{dt_str})" if extra_parts else f" (数据日:{dt_str})"
            lines.append(f"- **{name}** (`{sym}`): **{w:.1%}**{extra_info}")

        if payload.get("cash_weight", 0.0) > 0.001:
            lines.append(f"- **现金 (Cash)**: **{payload['cash_weight']:.1%}**")

        metrics = payload.get("metrics_snapshot") or {}
        if metrics:
            lines.extend(
                [
                    "",
                    "#### 历史回测表现快照",
                    f"- 年化收益 (CAGR): `{metrics.get('cagr', 0.0):.2%}` | 最大回撤: `{metrics.get('max_drawdown', 0.0):.2%}` | 夏普比率: `{metrics.get('sharpe_ratio', 0.0):.2f}`",
                ]
            )

        return "\n".join(lines)
