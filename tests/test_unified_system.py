"""
全链路单元测试 (Unified System End-to-End Tests).
验证：
1. 资产字典与策略配置加载 (`AppConfig`)
2. 本地缓存数据库读写与行情估值非空合并 (`DatabaseGateway`)
3. 全部向量化因子计算与按需回退 (`FactorService`)
4. 三类可组合调仓触发器 (`CalendarTrigger`, `DriftTrigger`, `ExtremumTrigger`)
5. 全部 5 类策略插件计算与 T+1 回测引擎 (`DynamicAllocationStrategy`, `DualMomentumStrategy`, `ThresholdRotationStrategy`, `SbeAllocationStrategy`, `EqualWeightStrategy`)
6. T+1 信号截断规则 (`resolve_cutoff_date`) 与跨市场日期独立保留规则 (`QuantPipeline`)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import AppConfig
from core.db import DatabaseGateway
from core.quant_engine.backtester import BacktestEngine
from core.quant_engine.factors import FACTOR_REGISTRY, FactorService
from core.quant_engine.pipeline import QuantPipeline
from core.quant_engine.strategies import create_strategy
from core.quant_engine.triggers import (
    CalendarTrigger,
    DriftTrigger,
    ExtremumTrigger,
)


@pytest.fixture()
def temp_env(tmp_path: Path):
    """构建带合成历史行情与估值数据的隔离测试环境"""
    cfg = AppConfig()
    db = DatabaseGateway(
        config=cfg, cache_dir=tmp_path / "cache", enable_remote=False
    )

    dates = pd.bdate_range("2022-01-03", "2024-06-28")
    rng = np.random.default_rng(42)

    symbols_specs = [
        ("000300.SH", 4000.0, 12.0, 0.0002, 0.012),
        ("NDX.GI", 13000.0, 28.0, 0.0006, 0.015),
        ("Au99.99.SGE", 380.0, None, 0.0004, 0.007),
        ("CBA02701.CS", 210.0, None, 0.00018, 0.0015),
        ("CBA06501.CS", 190.0, None, 0.00012, 0.0010),
        ("513100.SH", 1.1, None, 0.0006, 0.015),
        ("513500.SH", 1.3, None, 0.0005, 0.012),
        ("510300.SH", 3.8, None, 0.0002, 0.012),
        ("159915.SZ", 2.2, None, 0.0001, 0.018),
        ("512890.SH", 1.0, None, 0.0003, 0.009),
        ("518880.SH", 3.9, None, 0.0004, 0.007),
        ("511090.SH", 105.0, None, 0.0002, 0.003),
        ("511010.SH", 120.0, None, 0.00012, 0.001),
        ("512100.SH", 2.5, None, 0.00025, 0.016),
    ]

    all_rows = []
    for sym, base_p, base_pe, mu, sigma in symbols_specs:
        rets = rng.normal(mu, sigma, size=len(dates))
        closes = base_p * np.cumprod(1.0 + rets)
        for i, dt in enumerate(dates):
            # 模拟美股 NDX.GI 在最后一个交易日比 A 股滞后一天，以测试跨市场日期独立展示
            if sym == "NDX.GI" and i == len(dates) - 1:
                continue
            row = {
                "symbol": sym,
                "trade_date": dt.strftime("%Y-%m-%d"),
                "open": float(closes[i] * 0.998),
                "high": float(closes[i] * 1.005),
                "low": float(closes[i] * 0.995),
                "close": float(closes[i]),
                "volume": 1000000.0,
            }
            if base_pe is not None:
                row["pe"] = float(base_pe * (closes[i] / base_p))
                row["pb"] = float(1.8 * (closes[i] / base_p))
                row["dividend_yield"] = float(2.5 / (closes[i] / base_p))
            all_rows.append(row)

    # 写入 SBE.JCS 股债性价比分位数序列 (0 ~ 100)
    for i, dt in enumerate(dates):
        all_rows.append(
            {
                "symbol": "SBE.JCS",
                "trade_date": dt.strftime("%Y-%m-%d"),
                "close": float(50.0 + 35.0 * np.sin(i / 40.0)),
            }
        )

    db.upsert_market_daily(all_rows)
    return cfg, db


def test_config_and_factor_registry(temp_env):
    cfg, db = temp_env
    assets = cfg.load_assets()
    assert "NDX.GI" in assets
    assert "000300.SH" in assets
    assert "CBA02701.CS" in assets

    online_strategies = cfg.list_strategies(only_online=True)
    all_strategies = cfg.list_strategies(only_online=False)
    assert len(online_strategies) == 3
    assert len(all_strategies) >= 6

    factor_svc = FactorService(db=db, config=cfg)
    df_hs300 = db.fetch_market_daily("000300.SH")
    for fid in [
        "pe_percentile_10y",
        "pb_percentile_5y",
        "dy_percentile_5y",
        "rsi_14",
        "log_bias",
        "ret_20d",
        "vol_30d",
    ]:
        assert fid in FACTOR_REGISTRY
        s = factor_svc.compute_single_factor(df_hs300, fid)
        assert not s.empty


def test_partial_valuation_merge_does_not_clobber_price(temp_env):
    """验证分拆写入估值时不会覆盖已有的收盘价，反之亦然"""
    _, db = temp_env
    db.upsert_market_daily(
        [{"symbol": "TEST.IDX", "trade_date": "2024-06-20", "close": 100.0}]
    )
    db.upsert_market_daily(
        [{"symbol": "TEST.IDX", "trade_date": "2024-06-20", "pe": 15.5}]
    )
    df = db.fetch_market_daily("TEST.IDX")
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 100.0
    assert float(df.iloc[0]["pe"]) == 15.5


def test_triggers_behavior():
    dates = pd.bdate_range("2023-01-02", "2023-12-29")
    cal_q = CalendarTrigger(frequency="quarter").get_rebalance_dates(dates)
    assert len(cal_q) == 4

    cal_w2 = CalendarTrigger(
        frequency="weekly", weekday=4, interval_weeks=2
    ).get_rebalance_dates(dates)
    assert 24 <= len(cal_w2) <= 27

    drift = DriftTrigger(
        enabled=True,
        portfolio_drift_threshold=0.10,
        asset_drift_threshold=0.06,
        cooldown_days=10,
        residual_symbols=["CBA06501.CS"],
    )
    # 冷却期内不触发
    ok, _ = drift.check({"A": 0.5}, {"A": 0.3}, days_since_last_rebalance=5)
    assert not ok
    # 单项非 residual 资产漂移触发
    ok, reason = drift.check(
        {"A": 0.37, "CBA06501.CS": 0.63},
        {"A": 0.30, "CBA06501.CS": 0.70},
        days_since_last_rebalance=15,
    )
    assert ok and "A" in reason


def test_all_six_strategies_backtest_and_signals(temp_env):
    """验证全部 6 个内置策略均能顺利完成 T+1 回测与每日信号生成"""
    cfg, db = temp_env
    pipeline = QuantPipeline(config=cfg, db=db)

    for sid in cfg.list_strategies():
        bt = pipeline.run_backtest(strategy_id=sid, save_outputs=False)
        assert not bt.equity_curve.empty, f"策略 {sid} 净值曲线为空"
        assert bt.metrics["trading_days"] > 100
        assert bt.latest_decision is not None
        total_w = sum(bt.latest_decision.weights.values())
        assert 0.0 <= total_w <= 1.0001

        sig = pipeline.run_daily_signal(
            strategy_id=sid,
            run_date="2024-06-28",
            run_mode="live",
            notify=False,
        )
        assert sig["strategy_id"] == sid
        assert "markdown_report" in sig


def test_t_plus_1_cutoff_and_cross_market_date_preservation(temp_env):
    """
    验证：
    1. 盘前运行 `global_asset_allocation` 时，run_date=2024-06-29 自动截断 `end_date=2024-06-28`；
    2. 周五盘后运行 `etf_dual_momentum` (`signal_cutoff=same_day_close`) 时，run_date=2024-06-28 截断 `end_date=2024-06-28`；
    3. 跨市场资产在 `asset_details` 中保留各自真实的最新数据日期 (`000300.SH` 为 `2024-06-28`，`NDX.GI` 为 `2024-06-27`)。
    """
    cfg, db = temp_env
    pipeline = QuantPipeline(config=cfg, db=db)

    gaa_cfg = cfg.load_strategy_config("global_asset_allocation")
    assert (
        pipeline.resolve_cutoff_date(gaa_cfg, run_date="2024-06-29")
        == "2024-06-28"
    )

    etf_cfg = cfg.load_strategy_config("etf_dual_momentum")
    assert (
        pipeline.resolve_cutoff_date(etf_cfg, run_date="2024-06-28")
        == "2024-06-28"
    )

    sig = pipeline.run_daily_signal(
        strategy_id="global_asset_allocation",
        run_date="2024-06-29",
        run_mode="dry-run",
        notify=False,
    )
    details_by_sym = {d["symbol"]: d for d in sig["asset_details"]}
    assert details_by_sym["000300.SH"]["data_date"] == "2024-06-28"
    assert details_by_sym["NDX.GI"]["data_date"] == "2024-06-27"


def test_http_client_and_sbe_macro_provider():
    """验证统一 HTTP 会话管理与 SBE 宏观源容灾获取"""
    from core.data_hub.http_client import get_http_session, safe_request
    from core.data_hub.providers import SbeMacroProvider

    session = get_http_session()
    assert session is not None
    # 验证 safe_request 对异常/404 静默并返回响应或 None
    resp = safe_request("GET", "https://httpbin.org/status/404", timeout=3)
    # 若在离线 sandbox 环境 resp 为 None；若在线则返回 404，两者均不抛未捕获异常
    if resp is not None:
        assert resp.status_code == 404

    sbe = SbeMacroProvider()
    # 模拟从蛋卷或韭圈儿获取指标
    bar = sbe.fetch_latest("SBE.JCS", "jiucaishuo")
    # bar 允许为 None（在完全断网的测试容器中）或有效 MarketBar
    if bar is not None:
        assert bar.symbol == "SBE.JCS"
        assert bar.close is not None


def test_atomic_upsert_market_and_factor_daily(temp_env):
    """验证原子 Upsert 在高频交替写入时的字段互补与因子覆盖幂等性"""
    _, db = temp_env

    # 1. market_daily 原子互补测试
    db.upsert_market_daily([
        {"symbol": "TEST.STOCK", "trade_date": "2024-01-01", "close": 10.0, "volume": 1000.0},
    ])
    # 仅更新估值字段，close 和 volume 应完整保留
    db.upsert_market_daily([
        {"symbol": "TEST.STOCK", "trade_date": "2024-01-01", "pe": 8.5, "pb": 1.2},
    ])
    df = db.fetch_market_daily("TEST.STOCK")
    assert len(df) == 1
    assert float(df.iloc[0]["close"]) == 10.0
    assert float(df.iloc[0]["volume"]) == 1000.0
    assert float(df.iloc[0]["pe"]) == 8.5
    assert float(df.iloc[0]["pb"]) == 1.2

    # 2. factor_daily 原子幂等测试
    db.upsert_factor_daily([
        {"symbol": "TEST.STOCK", "trade_date": "2024-01-01", "factor_id": "pe_10y", "value": 45.0},
        {"symbol": "TEST.STOCK", "trade_date": "2024-01-01", "factor_id": "rsi_14", "value": 55.0},
    ])
    # 更新 pe_10y，值更新为 48.0
    db.upsert_factor_daily([
        {"symbol": "TEST.STOCK", "trade_date": "2024-01-01", "factor_id": "pe_10y", "value": 48.0},
    ])
    s_pe = db.fetch_factor_series("TEST.STOCK", "pe_10y")
    assert len(s_pe) == 1
    assert float(s_pe.iloc[0]) == 48.0


def test_fetch_market_daily_multi(temp_env):
    """验证多标的单次批量查询"""
    _, db = temp_env
    multi_df = db.fetch_market_daily_multi(
        ["000300.SH", "NDX.GI"],
        start_date="2023-01-01",
        end_date="2023-01-15",
    )
    assert not multi_df.empty
    symbols_found = set(multi_df["symbol"].unique())
    assert "000300.SH" in symbols_found
    assert "NDX.GI" in symbols_found

