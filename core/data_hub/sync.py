"""
统一数据同步与行情读取服务 (DataSyncService).

核心职责：
1. `sync_daily`: 每日增量同步所有配置标的的最新行情与最新估值（含价格与估值日期错位分拆与滞后检测）。
2. `sync_valuation_catchup`: 盘后/晚间专项重拉估值，补齐好买晚于收盘更新的 PE/PB/股息率。
3. `sync_history`: 首次建库或回溯补齐历史日线与历史估值序列。
4. `load_price_matrix`: 为量化引擎提供严格遵守 `trade_date <= end_date` 的收盘价矩阵与单标的行情字典。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.config import AppConfig, AssetConfig
from core.data_hub.models import MarketBar, SyncReport
from core.data_hub.providers import (
    AkshareMarketProvider,
    ChinaBondProvider,
    HowbuyValuationProvider,
    SbeMacroProvider,
    SinaQuoteProvider,
)
from core.db import DatabaseGateway

logger = logging.getLogger(__name__)


class DataSyncService:
    """统一数据中心服务"""

    def __init__(
        self,
        db: Optional[DatabaseGateway] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.db = db or DatabaseGateway(config=self.config)
        self.sina = SinaQuoteProvider()
        self.howbuy = HowbuyValuationProvider()
        self.chinabond = ChinaBondProvider()
        self.akshare = AkshareMarketProvider()
        self.sbe = SbeMacroProvider()

    # ------------------------------------------------------------------
    # 1. 元数据同步
    # ------------------------------------------------------------------
    def sync_asset_metadata(
        self, symbols: Optional[Iterable[str]] = None
    ) -> int:
        assets = self.config.load_assets()
        target_symbols = list(symbols) if symbols else list(assets.keys())
        rows: List[Dict[str, Any]] = []
        for sym in target_symbols:
            spec = assets.get(sym)
            if not spec:
                continue
            rows.append(
                {
                    "symbol": spec.symbol,
                    "name": spec.name,
                    "asset_type": spec.asset_type,
                    "market": spec.market,
                    "currency": spec.currency,
                    "trade_etf": spec.trade_etf,
                    "source_config": spec.sources,
                    "is_active": True,
                }
            )
        return self.db.upsert_asset_meta(rows)

    # ------------------------------------------------------------------
    # 2. 每日增量同步 (行情 + 估值 + 错位分拆)
    # ------------------------------------------------------------------
    def sync_daily(
        self,
        symbols: Optional[Iterable[str]] = None,
        lookback_days: int = 15,
    ) -> SyncReport:
        assets = self.config.load_assets()
        target_symbols = list(symbols) if symbols else list(assets.keys())
        report = SyncReport(run_date=datetime.now().strftime("%Y-%m-%d"))

        self.sync_asset_metadata(target_symbols)
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        for sym in target_symbols:
            spec = assets.get(sym)
            if not spec:
                continue
            try:
                rows = self._collect_daily_records_for_asset(
                    spec, start_date=start_date, report=report
                )
                if rows:
                    self.db.upsert_market_daily(rows)
                    report.upserted_rows += len(rows)
                    report.synced_symbols.append(sym)
            except Exception as exc:
                logger.error("同步标的 %s 失败: %s", sym, exc)
                report.failed_symbols[sym] = str(exc)

        return report

    def _collect_daily_records_for_asset(
        self,
        spec: AssetConfig,
        start_date: str,
        report: SyncReport,
    ) -> List[Dict[str, Any]]:
        src = spec.sources
        records_by_date: Dict[str, Dict[str, Any]] = {}

        def _merge_bar(bar: Optional[MarketBar]) -> None:
            if bar is None:
                return
            rec = bar.to_record()
            dt = rec["trade_date"]
            existing = records_by_date.setdefault(
                dt, {"symbol": spec.symbol, "trade_date": dt}
            )
            for k, v in rec.items():
                if v is not None:
                    existing[k] = v

        if spec.asset_type == "index":
            # 1) 近期日线兜底
            if src.get("ak_symbol"):
                for b in self.akshare.fetch_index_history(
                    spec.symbol,
                    src.get("ak_type", "cn"),
                    src["ak_symbol"],
                    start_date=start_date,
                ):
                    _merge_bar(b)
            # 2) 新浪实时最新行情
            price_bar = None
            if src.get("sina_code"):
                price_bar = self.sina.fetch_realtime_bar(
                    spec.symbol, src["sina_code"]
                )
                _merge_bar(price_bar)
            # 3) 好买估值（自动分拆日期，防止估值滞后污染当日行情或丢失）
            if src.get("howbuy_zsdm"):
                val_bar = self.howbuy.fetch_latest_valuation(
                    spec.symbol, src["howbuy_zsdm"]
                )
                _merge_bar(val_bar)
                if (
                    price_bar
                    and val_bar
                    and val_bar.trade_date < price_bar.trade_date
                ):
                    report.stale_valuations.append(
                        {
                            "symbol": spec.symbol,
                            "name": spec.name,
                            "price_date": price_bar.trade_date,
                            "valuation_date": val_bar.trade_date,
                        }
                    )

        elif spec.asset_type == "bond":
            for b in self.chinabond.fetch_bars(
                spec.symbol,
                src.get("chinabond_id", ""),
                ak_bond_name=src.get("ak_bond_name"),
                ak_bond_indicator=src.get("ak_bond_indicator", "财富指数"),
                start_date=start_date,
            ):
                _merge_bar(b)

        elif spec.asset_type == "gold":
            for b in self.akshare.fetch_gold_history(
                spec.symbol,
                sge_symbol=src.get("sge_symbol", "Au99.99"),
                start_date=start_date,
            ):
                _merge_bar(b)

        elif spec.asset_type == "etf":
            for b in self.akshare.fetch_etf_history(
                spec.symbol,
                etf_code=src.get("etf_code", spec.symbol.split(".")[0]),
                sina_etf_code=src.get("sina_etf_code"),
                start_date=start_date,
            ):
                _merge_bar(b)

        elif spec.asset_type == "macro":
            bar = self.sbe.fetch_latest(
                spec.symbol, src.get("sbe_source", "jiucaishuo")
            )
            _merge_bar(bar)

        return [records_by_date[d] for d in sorted(records_by_date.keys())]

    # ------------------------------------------------------------------
    # 3. 估值滞后专项补齐 (Valuation Catch-up)
    # ------------------------------------------------------------------
    def sync_valuation_catchup(
        self, symbols: Optional[Iterable[str]] = None
    ) -> SyncReport:
        """重新拉取指数近期估值历史，补齐晚间才更新的 PE/PB/股息率"""
        assets = self.config.load_assets()
        target_symbols = list(symbols) if symbols else list(assets.keys())
        report = SyncReport(run_date=datetime.now().strftime("%Y-%m-%d"))

        for sym in target_symbols:
            spec = assets.get(sym)
            if not spec or spec.asset_type != "index":
                continue
            zsdm = spec.sources.get("howbuy_zsdm")
            if not zsdm:
                continue
            try:
                bars = self.howbuy.fetch_history_valuation(
                    spec.symbol, zsdm, cycle="1m"
                )
                latest = self.howbuy.fetch_latest_valuation(spec.symbol, zsdm)
                if latest:
                    bars.append(latest)
                rows = [b.to_record() for b in bars]
                if rows:
                    self.db.upsert_market_daily(rows)
                    report.upserted_rows += len(rows)
                    report.synced_symbols.append(sym)
            except Exception as exc:
                report.failed_symbols[sym] = str(exc)

        return report

    # ------------------------------------------------------------------
    # 4. 历史全量同步
    # ------------------------------------------------------------------
    def sync_history(
        self,
        symbols: Optional[Iterable[str]] = None,
        start_date: str = "2014-01-01",
    ) -> SyncReport:
        assets = self.config.load_assets()
        target_symbols = list(symbols) if symbols else list(assets.keys())
        report = SyncReport(run_date=datetime.now().strftime("%Y-%m-%d"))
        self.sync_asset_metadata(target_symbols)

        for sym in target_symbols:
            spec = assets.get(sym)
            if not spec:
                continue
            try:
                records_by_date: Dict[str, Dict[str, Any]] = {}
                src = spec.sources
                if spec.asset_type == "index":
                    if src.get("ak_symbol"):
                        for b in self.akshare.fetch_index_history(
                            spec.symbol,
                            src.get("ak_type", "cn"),
                            src["ak_symbol"],
                            start_date=start_date,
                        ):
                            records_by_date[b.trade_date] = b.to_record()
                    if src.get("howbuy_zsdm"):
                        for b in self.howbuy.fetch_history_valuation(
                            spec.symbol, src["howbuy_zsdm"], cycle="10y"
                        ):
                            rec = b.to_record()
                            dt = rec["trade_date"]
                            if dt >= start_date[:10]:
                                existing = records_by_date.setdefault(
                                    dt,
                                    {"symbol": spec.symbol, "trade_date": dt},
                                )
                                for k, v in rec.items():
                                    if v is not None:
                                        existing[k] = v
                else:
                    for rec in self._collect_daily_records_for_asset(
                        spec, start_date=start_date, report=report
                    ):
                        records_by_date[rec["trade_date"]] = rec

                rows = [
                    records_by_date[d] for d in sorted(records_by_date.keys())
                ]
                if rows:
                    self.db.upsert_market_daily(rows)
                    report.upserted_rows += len(rows)
                    report.synced_symbols.append(sym)
            except Exception as exc:
                logger.error("历史全量同步 %s 失败: %s", sym, exc)
                report.failed_symbols[sym] = str(exc)

        return report

    # ------------------------------------------------------------------
    # 5. 供量化引擎读取的标准行情矩阵 (严格遵循 T+1 end_date 截断)
    # ------------------------------------------------------------------
    def load_price_matrix(
        self,
        symbols: Iterable[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ffill: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        加载多个标的的日线数据并对齐为收盘价矩阵 `prices_df` (index=DatetimeIndex, columns=symbols)。
        同时返回每个标的各自的原始未强行截断 DataFrame 字典 `raw_frames`，
        以确保跨市场资产在展示层保留各自真实的最新数据日期 (`<= end_date`)。
        """
        raw_frames: Dict[str, pd.DataFrame] = {}
        close_series_map: Dict[str, pd.Series] = {}

        for sym in symbols:
            df = self.db.fetch_market_daily(
                sym, start_date=start_date, end_date=end_date
            )
            if df.empty or "close" not in df.columns:
                continue
            df = df.dropna(subset=["close"]).copy()
            if df.empty:
                continue
            df["dt"] = pd.to_datetime(df["trade_date"])
            df = df.drop_duplicates(subset=["dt"], keep="last").sort_values(
                "dt"
            )
            raw_frames[sym] = df
            close_series_map[sym] = pd.Series(
                df["close"].values, index=df["dt"], name=sym
            )

        if not close_series_map:
            return pd.DataFrame(), raw_frames

        prices_df = pd.DataFrame(close_series_map).sort_index()
        if ffill:
            prices_df = prices_df.ffill()
        return prices_df, raw_frames
