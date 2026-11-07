"""
统一关系型数据库与 Supabase 网关 (Unified Relational SQL & Supabase Gateway).

核心特性：
1. 本地关系型 SQL 引擎 (SQLAlchemy)：
   - 默认使用本地关系型数据库 `sqlite:///data/union_quant.db`，也可通过 `.env` 中的 `DATABASE_URL`
     无缝切换为本地 MySQL (`mysql+pymysql://user:pass@localhost:3306/union_quant`) 或 PostgreSQL (`postgresql://...`)。
   - 自动初始化与 `sql/schema.sql` 完全对齐的 4 张核心表：`asset_meta`、`market_daily`、`factor_daily`、`strategy_signals`。
2. 远端 Supabase 双写与一键迁移：
   - 配置 `SUPABASE_URL` 与 `SUPABASE_SERVICE_ROLE_KEY` 后，支持同步双写远端。
   - 提供 `migrate_to_supabase()` 与 `export_postgres_seed_sql()`，方便本地验证无误后一键迁移到 Supabase。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from core.config import DATA_CACHE_DIR, DEFAULT_SQLITE_PATH, AppConfig

logger = logging.getLogger(__name__)


class DatabaseGateway:
    """统一关系型数据库访问层 (Local SQLite/MySQL/Postgres + Supabase)"""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        cache_dir: Path = DATA_CACHE_DIR,
        db_url: Optional[str] = None,
        enable_remote: bool = True,
    ) -> None:
        self.config = config or AppConfig()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 解析关系型数据库连接串：优先显式传入 -> 环境变量 DATABASE_URL -> 默认本地 SQLite 库文件
        if db_url:
            self.db_url = db_url
        elif self.config.database_url and cache_dir == DATA_CACHE_DIR:
            self.db_url = self.config.database_url
        else:
            db_file = (
                DEFAULT_SQLITE_PATH
                if cache_dir == DATA_CACHE_DIR
                else (cache_dir / "union_quant.db")
            )
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self.db_url = f"sqlite:///{db_file.as_posix()}"

        self.engine: Engine = create_engine(self.db_url, future=True)
        self._init_relational_tables()

        self._client: Any = None
        if enable_remote and self.config.supabase_url and self.config.supabase_key:
            try:
                from supabase import create_client

                self._client = create_client(
                    self.config.supabase_url, self.config.supabase_key
                )
            except Exception as exc:
                logger.warning("初始化 Supabase 客户端失败，仅使用本地关系库: %s", exc)

    @property
    def is_remote_enabled(self) -> bool:
        return self._client is not None

    def _init_relational_tables(self) -> None:
        """在本地关系型数据库 (SQLite / MySQL / PostgreSQL) 中初始化与 schema.sql 一致的表结构"""
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS asset_meta (
                symbol VARCHAR(64) PRIMARY KEY,
                name VARCHAR(128) NOT NULL,
                asset_type VARCHAR(32) NOT NULL,
                market VARCHAR(16) NOT NULL DEFAULT 'CN',
                currency VARCHAR(16) NOT NULL DEFAULT 'CNY',
                trade_etf VARCHAR(64),
                source_config TEXT,
                latest_price DOUBLE PRECISION,
                latest_date VARCHAR(16),
                latest_pe DOUBLE PRECISION,
                latest_pb DOUBLE PRECISION,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                updated_at VARCHAR(32)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_daily (
                symbol VARCHAR(64) NOT NULL,
                trade_date VARCHAR(16) NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                amount DOUBLE PRECISION,
                change_pct DOUBLE PRECISION,
                pe DOUBLE PRECISION,
                pb DOUBLE PRECISION,
                dividend_yield DOUBLE PRECISION,
                roe DOUBLE PRECISION,
                extra_json TEXT,
                updated_at VARCHAR(32),
                PRIMARY KEY (symbol, trade_date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS factor_daily (
                symbol VARCHAR(64) NOT NULL,
                trade_date VARCHAR(16) NOT NULL,
                factor_id VARCHAR(64) NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                updated_at VARCHAR(32),
                PRIMARY KEY (symbol, trade_date, factor_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_meta (
                strategy_id VARCHAR(64) PRIMARY KEY,
                strategy_name VARCHAR(128) NOT NULL,
                strategy_type VARCHAR(64) NOT NULL,
                version VARCHAR(32) NOT NULL DEFAULT '2.0',
                status VARCHAR(32) NOT NULL DEFAULT 'live',
                description TEXT,
                rebalance_frequency VARCHAR(32),
                execution_mode VARCHAR(64),
                signal_cutoff VARCHAR(64),
                commission_rate DOUBLE PRECISION,
                slippage_rate DOUBLE PRECISION,
                rebalance_config TEXT,
                strategy_params TEXT,
                assets_config TEXT,
                updated_at VARCHAR(32)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_signals (
                strategy_id VARCHAR(64) NOT NULL,
                strategy_name VARCHAR(128) NOT NULL,
                strategy_version VARCHAR(32) NOT NULL DEFAULT '2.0',
                signal_date VARCHAR(16) NOT NULL,
                execution_date VARCHAR(16),
                run_mode VARCHAR(32) NOT NULL DEFAULT 'live',
                is_rebalance_day BOOLEAN NOT NULL DEFAULT 0,
                trigger_reason TEXT,
                cash_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                weights TEXT NOT NULL,
                asset_details TEXT,
                metrics_snapshot TEXT,
                config_snapshot TEXT,
                updated_at VARCHAR(32),
                PRIMARY KEY (strategy_id, signal_date, run_mode)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_signal_asset_details (
                strategy_id VARCHAR(64) NOT NULL,
                signal_date VARCHAR(16) NOT NULL,
                run_mode VARCHAR(32) NOT NULL DEFAULT 'live',
                display_order INTEGER NOT NULL DEFAULT 1,
                asset_symbol VARCHAR(64) NOT NULL,
                asset_name VARCHAR(128) NOT NULL,
                trade_symbol VARCHAR(64),
                role VARCHAR(32),
                weight_strategy VARCHAR(64),
                base_weight DOUBLE PRECISION,
                param_c1 DOUBLE PRECISION,
                weight_strategy_params TEXT,
                target_weight DOUBLE PRECISION,
                remaining_before DOUBLE PRECISION,
                actual_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                factor_id VARCHAR(64),
                factor_value_raw DOUBLE PRECISION,
                factor_value_normalized DOUBLE PRECISION,
                factor_date VARCHAR(16),
                close_price DOUBLE PRECISION,
                valuation_type VARCHAR(32),
                valuation_value DOUBLE PRECISION,
                valuation_date VARCHAR(16),
                calculation_process TEXT NOT NULL,
                updated_at VARCHAR(32),
                PRIMARY KEY (strategy_id, signal_date, run_mode, asset_symbol)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategy_backtest_runs (
                run_id VARCHAR(96) PRIMARY KEY,
                strategy_id VARCHAR(64) NOT NULL,
                strategy_name VARCHAR(128) NOT NULL,
                run_mode VARCHAR(32) NOT NULL DEFAULT 'lab',
                status VARCHAR(32) NOT NULL DEFAULT 'verified',
                start_date VARCHAR(16),
                end_date VARCHAR(16),
                trading_days INTEGER,
                rebalance_frequency VARCHAR(32),
                commission_rate DOUBLE PRECISION,
                slippage_rate DOUBLE PRECISION,
                total_return DOUBLE PRECISION,
                cagr DOUBLE PRECISION,
                annual_volatility DOUBLE PRECISION,
                max_drawdown DOUBLE PRECISION,
                sharpe_ratio DOUBLE PRECISION,
                sortino_ratio DOUBLE PRECISION,
                calmar_ratio DOUBLE PRECISION,
                daily_win_rate DOUBLE PRECISION,
                rebalance_count INTEGER,
                final_nav DOUBLE PRECISION,
                config_snapshot TEXT,
                metrics_json TEXT,
                latest_details_json TEXT,
                rebalance_log_json TEXT,
                updated_at VARCHAR(32)
            )
            """,
        ]
        with self.engine.begin() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))

    # ------------------------------------------------------------------
    # 1. 标的元数据 (asset_meta)
    # ------------------------------------------------------------------
    def upsert_asset_meta(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.engine.begin() as conn:
            for r in rows:
                sym = str(r["symbol"])
                conn.execute(
                    text("DELETE FROM asset_meta WHERE symbol = :symbol"),
                    {"symbol": sym},
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO asset_meta
                        (symbol, name, asset_type, market, currency, trade_etf, source_config, is_active, updated_at)
                        VALUES (:symbol, :name, :asset_type, :market, :currency, :trade_etf, :source_config, :is_active, :updated_at)
                        """
                    ),
                    {
                        "symbol": sym,
                        "name": r.get("name", sym),
                        "asset_type": r.get("asset_type", "index"),
                        "market": r.get("market", "CN"),
                        "currency": r.get("currency", "CNY"),
                        "trade_etf": r.get("trade_etf"),
                        "source_config": json.dumps(
                            r.get("source_config") or {}, ensure_ascii=False
                        ),
                        "is_active": bool(r.get("is_active", True)),
                        "updated_at": now_str,
                    },
                )

        if self._client is not None:
            try:
                self._client.table("asset_meta").upsert(
                    rows, on_conflict="symbol"
                ).execute()
            except Exception as exc:
                logger.warning("写入远端 asset_meta 失败: %s", exc)
        return len(rows)

    # ------------------------------------------------------------------
    # 2. 日频行情与估值 (market_daily)
    # ------------------------------------------------------------------
    def upsert_market_daily(
        self, rows: List[Dict[str, Any]], chunk_size: int = 500
    ) -> int:
        """批量幂等更新日线行情与估值 (支持价格与估值非空字段增量合并，互不覆盖)"""
        if not rows:
            return 0

        df_new = pd.DataFrame(rows)
        df_new["trade_date"] = pd.to_datetime(df_new["trade_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "change_pct",
            "pe",
            "pb",
            "dividend_yield",
            "roe",
        ]

        with self.engine.begin() as conn:
            for symbol, group in df_new.groupby("symbol"):
                sym_str = str(symbol)
                min_dt = str(group["trade_date"].min())
                max_dt = str(group["trade_date"].max())

                existing_df = pd.read_sql(
                    text(
                        """
                        SELECT * FROM market_daily
                        WHERE symbol = :symbol AND trade_date >= :min_dt AND trade_date <= :max_dt
                        """
                    ),
                    conn,
                    params={"symbol": sym_str, "min_dt": min_dt, "max_dt": max_dt},
                )

                merged = self._merge_market_frames(existing_df, group)
                conn.execute(
                    text(
                        """
                        DELETE FROM market_daily
                        WHERE symbol = :symbol AND trade_date >= :min_dt AND trade_date <= :max_dt
                        """
                    ),
                    {"symbol": sym_str, "min_dt": min_dt, "max_dt": max_dt},
                )

                insert_records: List[Dict[str, Any]] = []
                for _, row in merged.iterrows():
                    rec: Dict[str, Any] = {
                        "symbol": sym_str,
                        "trade_date": str(row["trade_date"])[:10],
                        "updated_at": now_str,
                    }
                    for c in num_cols:
                        val = row.get(c)
                        rec[c] = (
                            None
                            if val is None or pd.isna(val)
                            else float(val)
                        )
                    extra = row.get("extra_json")
                    if isinstance(extra, dict):
                        rec["extra_json"] = json.dumps(extra, ensure_ascii=False)
                    elif isinstance(extra, str) and extra.strip():
                        rec["extra_json"] = extra
                    else:
                        rec["extra_json"] = None
                    insert_records.append(rec)

                if insert_records:
                    conn.execute(
                        text(
                            """
                            INSERT INTO market_daily
                            (symbol, trade_date, open, high, low, close, volume, amount, change_pct, pe, pb, dividend_yield, roe, extra_json, updated_at)
                            VALUES (:symbol, :trade_date, :open, :high, :low, :close, :volume, :amount, :change_pct, :pe, :pb, :dividend_yield, :roe, :extra_json, :updated_at)
                            """
                        ),
                        insert_records,
                    )

                # 顺带更新 asset_meta 表中的最新行情与估值快照
                latest_close_rows = merged.dropna(subset=["close"])
                if not latest_close_rows.empty:
                    last_r = latest_close_rows.iloc[-1]
                    last_pe_rows = (
                        merged.dropna(subset=["pe"])
                        if "pe" in merged.columns
                        else pd.DataFrame()
                    )
                    last_pb_rows = (
                        merged.dropna(subset=["pb"])
                        if "pb" in merged.columns
                        else pd.DataFrame()
                    )
                    conn.execute(
                        text(
                            """
                            UPDATE asset_meta
                            SET latest_price = :price,
                                latest_date = :dt,
                                latest_pe = COALESCE(:pe, latest_pe),
                                latest_pb = COALESCE(:pb, latest_pb),
                                updated_at = :updated_at
                            WHERE symbol = :symbol
                            """
                        ),
                        {
                            "symbol": sym_str,
                            "price": float(last_r["close"]),
                            "dt": str(last_r["trade_date"])[:10],
                            "pe": float(last_pe_rows.iloc[-1]["pe"])
                            if not last_pe_rows.empty
                            else None,
                            "pb": float(last_pb_rows.iloc[-1]["pb"])
                            if not last_pb_rows.empty
                            else None,
                            "updated_at": now_str,
                        },
                    )

        if self._client is not None:
            try:
                clean_rows = self._clean_records_for_json(rows)
                for i in range(0, len(clean_rows), chunk_size):
                    batch = clean_rows[i : i + chunk_size]
                    self._client.table("market_daily").upsert(
                        batch, on_conflict="symbol,trade_date"
                    ).execute()
            except Exception as exc:
                logger.warning("写入远端 market_daily 失败: %s", exc)

        return len(rows)

    @staticmethod
    def _merge_market_frames(
        df_old: pd.DataFrame, df_new: pd.DataFrame
    ) -> pd.DataFrame:
        if df_old is None or df_old.empty:
            return df_new.sort_values("trade_date").reset_index(drop=True)
        df_old = df_old.copy()
        df_new = df_new.copy()
        df_old["trade_date"] = pd.to_datetime(df_old["trade_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        df_new["trade_date"] = pd.to_datetime(df_new["trade_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        old_idx = df_old.drop_duplicates(subset=["trade_date"], keep="last").set_index(
            "trade_date"
        )
        new_idx = df_new.drop_duplicates(subset=["trade_date"], keep="last").set_index(
            "trade_date"
        )
        combined = new_idx.combine_first(old_idx).reset_index()
        return combined.sort_values("trade_date").reset_index(drop=True)

    def fetch_market_daily(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """从本地关系库（或远端 Supabase）读取单标的日线与估值序列"""
        clauses = ["symbol = :symbol"]
        params: Dict[str, Any] = {"symbol": symbol}
        if start_date:
            clauses.append("trade_date >= :start_date")
            params["start_date"] = str(start_date)[:10]
        if end_date:
            clauses.append("trade_date <= :end_date")
            params["end_date"] = str(end_date)[:10]

        sql = f"SELECT * FROM market_daily WHERE {' AND '.join(clauses)} ORDER BY trade_date ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)

        # 若在全新环境（如 GitHub Actions Runner）本地仅同步了最近几天的增量，自动从远端 Supabase 拉取完整历史
        if len(df) < 100 and self._client is not None and not start_date:
            remote_df = self._fetch_remote_market(symbol, start_date, end_date)
            if len(remote_df) > len(df):
                df = remote_df

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "change_pct",
                    "pe",
                    "pb",
                    "dividend_yield",
                    "roe",
                ]
            )

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "change_pct",
            "pe",
            "pb",
            "dividend_yield",
            "roe",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_values("trade_date").reset_index(drop=True)

    def _fetch_remote_market(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        for table_name, symbol_col in [
            ("market_daily", "symbol"),
            ("index_daily_bars", "index_code"),
        ]:
            try:
                all_rows: List[Dict[str, Any]] = []
                page_size = 1000
                offset = 0
                while True:
                    query = (
                        self._client.table(table_name)
                        .select("*")
                        .eq(symbol_col, symbol)
                        .order("trade_date", desc=False)
                        .range(offset, offset + page_size - 1)
                    )
                    if start_date:
                        query = query.gte("trade_date", str(start_date)[:10])
                    if end_date:
                        query = query.lte("trade_date", str(end_date)[:10])
                    resp = query.execute()
                    batch = resp.data or []
                    all_rows.extend(batch)
                    if len(batch) < page_size:
                        break
                    offset += page_size
                if all_rows:
                    df = pd.DataFrame(all_rows)
                    if symbol_col != "symbol" and symbol_col in df.columns:
                        df["symbol"] = df[symbol_col]
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. 因子窄表 (factor_daily)
    # ------------------------------------------------------------------
    def upsert_factor_daily(
        self, rows: List[Dict[str, Any]], chunk_size: int = 500
    ) -> int:
        if not rows:
            return 0

        df_new = pd.DataFrame(rows)
        df_new["trade_date"] = pd.to_datetime(df_new["trade_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        df_new = df_new.drop_duplicates(
            subset=["symbol", "trade_date", "factor_id"], keep="last"
        )
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.engine.begin() as conn:
            for (symbol, factor_id), group in df_new.groupby(["symbol", "factor_id"]):
                sym_str = str(symbol)
                fid_str = str(factor_id)
                min_dt = str(group["trade_date"].min())
                max_dt = str(group["trade_date"].max())
                conn.execute(
                    text(
                        """
                        DELETE FROM factor_daily
                        WHERE symbol = :symbol AND factor_id = :factor_id
                          AND trade_date >= :min_dt AND trade_date <= :max_dt
                        """
                    ),
                    {
                        "symbol": sym_str,
                        "factor_id": fid_str,
                        "min_dt": min_dt,
                        "max_dt": max_dt,
                    },
                )
                recs = [
                    {
                        "symbol": sym_str,
                        "trade_date": str(r["trade_date"])[:10],
                        "factor_id": fid_str,
                        "value": float(r["value"]),
                        "updated_at": now_str,
                    }
                    for _, r in group.iterrows()
                    if pd.notna(r["value"])
                ]
                if recs:
                    conn.execute(
                        text(
                            """
                            INSERT INTO factor_daily (symbol, trade_date, factor_id, value, updated_at)
                            VALUES (:symbol, :trade_date, :factor_id, :value, :updated_at)
                            """
                        ),
                        recs,
                    )

        if self._client is not None:
            try:
                clean_rows = self._clean_records_for_json(rows)
                for i in range(0, len(clean_rows), chunk_size):
                    batch = clean_rows[i : i + chunk_size]
                    self._client.table("factor_daily").upsert(
                        batch, on_conflict="symbol,trade_date,factor_id"
                    ).execute()
            except Exception as exc:
                logger.warning("写入远端 factor_daily 失败: %s", exc)

        return len(df_new)

    def fetch_factor_series(
        self,
        symbol: str,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        clauses = ["symbol = :symbol", "factor_id = :factor_id"]
        params: Dict[str, Any] = {"symbol": symbol, "factor_id": factor_id}
        if start_date:
            clauses.append("trade_date >= :start_date")
            params["start_date"] = str(start_date)[:10]
        if end_date:
            clauses.append("trade_date <= :end_date")
            params["end_date"] = str(end_date)[:10]

        sql = f"SELECT trade_date, value FROM factor_daily WHERE {' AND '.join(clauses)} ORDER BY trade_date ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)

        if len(df) < 100 and self._client is not None and not start_date:
            remote_df = self._fetch_remote_factor(
                symbol, factor_id, start_date, end_date
            )
            if len(remote_df) > len(df):
                df = remote_df

        if df.empty or "trade_date" not in df.columns or "value" not in df.columns:
            return pd.Series(dtype=float)

        df = df.drop_duplicates(subset=["trade_date"], keep="last").sort_values(
            "trade_date"
        )
        s = pd.Series(
            pd.to_numeric(df["value"], errors="coerce").values,
            index=pd.to_datetime(df["trade_date"]),
            name=factor_id,
        )
        return s.dropna()

    def _fetch_remote_factor(
        self,
        symbol: str,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        try:
            all_rows: List[Dict[str, Any]] = []
            page_size = 1000
            offset = 0
            while True:
                q = (
                    self._client.table("factor_daily")
                    .select("trade_date,value")
                    .eq("symbol", symbol)
                    .eq("factor_id", factor_id)
                    .order("trade_date", desc=False)
                    .range(offset, offset + page_size - 1)
                )
                if start_date:
                    q = q.gte("trade_date", str(start_date)[:10])
                if end_date:
                    q = q.lte("trade_date", str(end_date)[:10])
                resp = q.execute()
                batch = resp.data or []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
            return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. 策略元数据与参数配置表 (strategy_meta)
    # ------------------------------------------------------------------
    def sync_strategy_meta_from_configs(self) -> int:
        """从 configs/strategies/*.yaml 自动解析并同步策略参数配置表 strategy_meta"""
        rows: List[Dict[str, Any]] = []
        for sid in self.config.list_strategies():
            try:
                cfg = self.config.load_strategy_config(sid)
            except Exception as exc:
                logger.warning("读取策略配置 %s 失败: %s", sid, exc)
                continue
            exec_cfg = cfg.get("execution") or {}
            reb_cfg = cfg.get("rebalance") or {}
            params = dict(cfg.get("strategy_params") or {})
            assets_list = list(params.pop("assets", []) or [])
            rows.append(
                {
                    "strategy_id": str(cfg.get("strategy_id", sid)),
                    "strategy_name": str(cfg.get("name", sid)),
                    "strategy_type": str(cfg.get("type", "dynamic_engine")),
                    "version": str(cfg.get("version", "2.0")),
                    "status": str(cfg.get("status", "live")),
                    "description": str(cfg.get("description", "")),
                    "rebalance_frequency": str(reb_cfg.get("frequency", "quarter")),
                    "execution_mode": str(exec_cfg.get("mode", "next_day_close_proxy")),
                    "signal_cutoff": str(
                        exec_cfg.get("signal_cutoff", "previous_day_close")
                    ),
                    "commission_rate": float(exec_cfg.get("commission_rate", 0.0005)),
                    "slippage_rate": float(exec_cfg.get("slippage_rate", 0.0005)),
                    "rebalance_config": reb_cfg,
                    "strategy_params": params,
                    "assets_config": assets_list,
                }
            )
        return self.upsert_strategy_meta(rows)

    def upsert_strategy_meta(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.engine.begin() as conn:
            for r in rows:
                sid = str(r["strategy_id"])
                conn.execute(
                    text("DELETE FROM strategy_meta WHERE strategy_id = :sid"),
                    {"sid": sid},
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO strategy_meta
                        (strategy_id, strategy_name, strategy_type, version, status, description,
                         rebalance_frequency, execution_mode, signal_cutoff, commission_rate, slippage_rate,
                         rebalance_config, strategy_params, assets_config, updated_at)
                        VALUES (:strategy_id, :strategy_name, :strategy_type, :version, :status, :description,
                                :rebalance_frequency, :execution_mode, :signal_cutoff, :commission_rate, :slippage_rate,
                                :rebalance_config, :strategy_params, :assets_config, :updated_at)
                        """
                    ),
                    {
                        "strategy_id": sid,
                        "strategy_name": r.get("strategy_name", sid),
                        "strategy_type": r.get("strategy_type", "dynamic_engine"),
                        "version": str(r.get("version", "2.0")),
                        "status": str(r.get("status", "live")),
                        "description": r.get("description", ""),
                        "rebalance_frequency": r.get("rebalance_frequency", "quarter"),
                        "execution_mode": r.get(
                            "execution_mode", "next_day_close_proxy"
                        ),
                        "signal_cutoff": r.get(
                            "signal_cutoff", "previous_day_close"
                        ),
                        "commission_rate": float(r.get("commission_rate", 0.0005)),
                        "slippage_rate": float(r.get("slippage_rate", 0.0005)),
                        "rebalance_config": json.dumps(
                            r.get("rebalance_config") or {}, ensure_ascii=False
                        )
                        if not isinstance(r.get("rebalance_config"), str)
                        else r.get("rebalance_config"),
                        "strategy_params": json.dumps(
                            r.get("strategy_params") or {}, ensure_ascii=False
                        )
                        if not isinstance(r.get("strategy_params"), str)
                        else r.get("strategy_params"),
                        "assets_config": json.dumps(
                            r.get("assets_config") or [], ensure_ascii=False
                        )
                        if not isinstance(r.get("assets_config"), str)
                        else r.get("assets_config"),
                        "updated_at": now_str,
                    },
                )
        if self._client is not None:
            try:
                clean_rows = self._clean_records_for_json(rows)
                self._client.table("strategy_meta").upsert(
                    clean_rows, on_conflict="strategy_id"
                ).execute()
            except Exception as exc:
                logger.warning("写入远端 strategy_meta 失败: %s", exc)
        return len(rows)

    def fetch_strategy_meta(
        self, strategy_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM strategy_meta"
        params: Dict[str, Any] = {}
        if strategy_id:
            sql += " WHERE strategy_id = :strategy_id"
            params["strategy_id"] = strategy_id
        sql += " ORDER BY strategy_id ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            self.sync_strategy_meta_from_configs()
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params)
        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = dict(row)
            for jcol in ["rebalance_config", "strategy_params", "assets_config"]:
                if isinstance(item.get(jcol), str) and item[jcol]:
                    try:
                        item[jcol] = json.loads(item[jcol])
                    except Exception:
                        pass
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # 5. 策略信号主表 (strategy_signals) 与单资产计算过程表 (strategy_signal_asset_details)
    # ------------------------------------------------------------------
    def upsert_strategy_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        strategy_id = str(payload["strategy_id"])
        signal_date = str(payload["signal_date"])[:10]
        run_mode = str(payload.get("run_mode", "live"))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details_list = payload.get("asset_details") or []

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM strategy_signals
                    WHERE strategy_id = :strategy_id AND signal_date = :signal_date AND run_mode = :run_mode
                    """
                ),
                {
                    "strategy_id": strategy_id,
                    "signal_date": signal_date,
                    "run_mode": run_mode,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO strategy_signals
                    (strategy_id, strategy_name, strategy_version, signal_date, execution_date,
                     run_mode, is_rebalance_day, trigger_reason, cash_weight,
                     weights, asset_details, metrics_snapshot, config_snapshot, updated_at)
                    VALUES (:strategy_id, :strategy_name, :strategy_version, :signal_date, :execution_date,
                            :run_mode, :is_rebalance_day, :trigger_reason, :cash_weight,
                            :weights, :asset_details, :metrics_snapshot, :config_snapshot, :updated_at)
                    """
                ),
                {
                    "strategy_id": strategy_id,
                    "strategy_name": payload.get("strategy_name", strategy_id),
                    "strategy_version": str(payload.get("strategy_version", "2.0")),
                    "signal_date": signal_date,
                    "execution_date": str(payload.get("execution_date", ""))[:10]
                    or None,
                    "run_mode": run_mode,
                    "is_rebalance_day": bool(payload.get("is_rebalance_day", False)),
                    "trigger_reason": payload.get("trigger_reason", ""),
                    "cash_weight": float(payload.get("cash_weight", 0.0)),
                    "weights": json.dumps(
                        payload.get("weights") or {}, ensure_ascii=False
                    ),
                    "asset_details": json.dumps(details_list, ensure_ascii=False),
                    "metrics_snapshot": json.dumps(
                        payload.get("metrics_snapshot") or {}, ensure_ascii=False
                    ),
                    "config_snapshot": json.dumps(
                        payload.get("config_snapshot") or {}, ensure_ascii=False
                    ),
                    "updated_at": now_str,
                },
            )

            # 同步落库单资产计算过程与参数验算明细表 strategy_signal_asset_details
            conn.execute(
                text(
                    """
                    DELETE FROM strategy_signal_asset_details
                    WHERE strategy_id = :strategy_id AND signal_date = :signal_date AND run_mode = :run_mode
                    """
                ),
                {
                    "strategy_id": strategy_id,
                    "signal_date": signal_date,
                    "run_mode": run_mode,
                },
            )
            detail_rows_to_insert: List[Dict[str, Any]] = []
            for idx, d in enumerate(details_list, start=1):
                sym = str(d.get("symbol") or d.get("asset_symbol") or f"ASSET_{idx}")
                raw_target = d.get("target_weight")
                act_w = d.get("actual_weight", d.get("weight", 0.0))
                rem_before = d.get("remaining_before")
                f_raw = d.get("factor_value_raw", d.get("factor_value"))
                f_norm = d.get("factor_value_normalized")
                c1_v = d.get("c1")
                base_w = d.get("base_weight")
                close_v = d.get("close")
                val_v = d.get("valuation_value", d.get("pe", d.get("ma_value", close_v)))
                val_type = d.get("valuation_type") or (
                    "pe_ratio" if d.get("pe") is not None else "price"
                )
                w_params = d.get("weight_strategy_params") or {}
                calc_proc = str(
                    d.get("calculation_process")
                    or f"目标权重={float(act_w or 0.0)*100:.2f}%"
                )
                detail_rows_to_insert.append(
                    {
                        "strategy_id": strategy_id,
                        "signal_date": signal_date,
                        "run_mode": run_mode,
                        "display_order": int(d.get("display_order", d.get("rank", idx))),
                        "asset_symbol": sym,
                        "asset_name": str(d.get("name") or d.get("asset_name") or sym),
                        "trade_symbol": d.get("trade_symbol"),
                        "role": d.get("role"),
                        "weight_strategy": d.get("weight_strategy"),
                        "base_weight": float(base_w)
                        if base_w is not None and not pd.isna(base_w)
                        else None,
                        "param_c1": float(c1_v)
                        if c1_v is not None and not pd.isna(c1_v)
                        else None,
                        "weight_strategy_params": json.dumps(
                            w_params, ensure_ascii=False
                        ),
                        "target_weight": float(raw_target)
                        if raw_target is not None and not pd.isna(raw_target)
                        else float(act_w or 0.0),
                        "remaining_before": float(rem_before)
                        if rem_before is not None and not pd.isna(rem_before)
                        else None,
                        "actual_weight": float(act_w or 0.0),
                        "factor_id": d.get("factor_id"),
                        "factor_value_raw": float(f_raw)
                        if f_raw is not None and not pd.isna(f_raw)
                        else None,
                        "factor_value_normalized": float(f_norm)
                        if f_norm is not None and not pd.isna(f_norm)
                        else None,
                        "factor_date": str(
                            d.get("factor_date") or d.get("data_date") or signal_date
                        )[:10],
                        "close_price": float(close_v)
                        if close_v is not None and not pd.isna(close_v)
                        else None,
                        "valuation_type": val_type,
                        "valuation_value": float(val_v)
                        if val_v is not None and not pd.isna(val_v)
                        else None,
                        "valuation_date": str(
                            d.get("valuation_date") or d.get("data_date") or signal_date
                        )[:10],
                        "calculation_process": calc_proc,
                        "updated_at": now_str,
                    }
                )
            if detail_rows_to_insert:
                conn.execute(
                    text(
                        """
                        INSERT INTO strategy_signal_asset_details
                        (strategy_id, signal_date, run_mode, display_order, asset_symbol, asset_name,
                         trade_symbol, role, weight_strategy, base_weight, param_c1, weight_strategy_params,
                         target_weight, remaining_before, actual_weight, factor_id, factor_value_raw,
                         factor_value_normalized, factor_date, close_price, valuation_type, valuation_value,
                         valuation_date, calculation_process, updated_at)
                        VALUES (:strategy_id, :signal_date, :run_mode, :display_order, :asset_symbol, :asset_name,
                                :trade_symbol, :role, :weight_strategy, :base_weight, :param_c1, :weight_strategy_params,
                                :target_weight, :remaining_before, :actual_weight, :factor_id, :factor_value_raw,
                                :factor_value_normalized, :factor_date, :close_price, :valuation_type, :valuation_value,
                                :valuation_date, :calculation_process, :updated_at)
                        """
                    ),
                    detail_rows_to_insert,
                )

        if self._client is not None:
            try:
                clean = self._clean_records_for_json([payload])[0]
                self._client.table("strategy_signals").upsert(
                    clean, on_conflict="strategy_id,signal_date,run_mode"
                ).execute()
            except Exception as exc:
                logger.warning("写入远端 strategy_signals 失败: %s", exc)
            if detail_rows_to_insert:
                try:
                    clean_details = self._clean_records_for_json(detail_rows_to_insert)
                    self._client.table("strategy_signal_asset_details").upsert(
                        clean_details,
                        on_conflict="strategy_id,signal_date,run_mode,asset_symbol",
                    ).execute()
                except Exception as exc:
                    logger.warning(
                        "写入远端 strategy_signal_asset_details 失败: %s", exc
                    )

        return payload

    def fetch_all_strategy_signals(
        self, run_mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM strategy_signals"
        params: Dict[str, Any] = {}
        if run_mode:
            sql += " WHERE run_mode = :run_mode"
            params["run_mode"] = run_mode
        sql += " ORDER BY signal_date DESC, strategy_id ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            return []

        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = dict(row)
            for jcol in [
                "weights",
                "asset_details",
                "metrics_snapshot",
                "config_snapshot",
            ]:
                if isinstance(item.get(jcol), str) and item[jcol]:
                    try:
                        item[jcol] = json.loads(item[jcol])
                    except Exception:
                        pass
            item["is_rebalance_day"] = bool(item.get("is_rebalance_day"))
            results.append(item)
        return results

    def fetch_strategy_signal_asset_details(
        self,
        strategy_id: Optional[str] = None,
        signal_date: Optional[str] = None,
        run_mode: str = "live",
    ) -> List[Dict[str, Any]]:
        clauses = ["run_mode = :run_mode"]
        params: Dict[str, Any] = {"run_mode": run_mode}
        if strategy_id:
            clauses.append("strategy_id = :strategy_id")
            params["strategy_id"] = strategy_id
        if signal_date:
            clauses.append("signal_date = :signal_date")
            params["signal_date"] = str(signal_date)[:10]

        sql = f"SELECT * FROM strategy_signal_asset_details WHERE {' AND '.join(clauses)} ORDER BY signal_date DESC, strategy_id ASC, display_order ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            return []
        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = dict(row)
            if isinstance(item.get("weight_strategy_params"), str) and item["weight_strategy_params"]:
                try:
                    item["weight_strategy_params"] = json.loads(
                        item["weight_strategy_params"]
                    )
                except Exception:
                    pass
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # 6. 策略实验室回测实验记录表 (strategy_backtest_runs)
    # ------------------------------------------------------------------
    def upsert_strategy_backtest_run(
        self, run_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        run_id = str(run_record["run_id"])
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics = run_record.get("metrics") or {}
        row_params = {
            "run_id": run_id,
            "strategy_id": str(run_record["strategy_id"]),
            "strategy_name": str(
                run_record.get("strategy_name", run_record["strategy_id"])
            ),
            "run_mode": str(run_record.get("run_mode", "lab")),
            "status": str(run_record.get("status", "verified")),
            "start_date": run_record.get("start_date") or metrics.get("start_date"),
            "end_date": run_record.get("end_date") or metrics.get("end_date"),
            "trading_days": int(
                run_record.get("trading_days") or metrics.get("trading_days") or 0
            ),
            "rebalance_frequency": str(
                run_record.get("rebalance_frequency", "quarter")
            ),
            "commission_rate": float(run_record.get("commission_rate", 0.0005)),
            "slippage_rate": float(run_record.get("slippage_rate", 0.0005)),
            "total_return": float(metrics.get("total_return", 0.0)),
            "cagr": float(metrics.get("cagr", 0.0)),
            "annual_volatility": float(metrics.get("annual_volatility", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
            "sortino_ratio": float(metrics.get("sortino_ratio", 0.0)),
            "calmar_ratio": float(metrics.get("calmar_ratio", 0.0)),
            "daily_win_rate": float(metrics.get("daily_win_rate", 0.0)),
            "rebalance_count": int(metrics.get("rebalance_count", 0)),
            "final_nav": float(metrics.get("final_nav", 1.0)),
            "config_snapshot": json.dumps(
                run_record.get("config_snapshot") or {}, ensure_ascii=False
            ),
            "metrics_json": json.dumps(metrics, ensure_ascii=False),
            "latest_details_json": json.dumps(
                run_record.get("latest_details") or [], ensure_ascii=False
            ),
            "rebalance_log_json": json.dumps(
                run_record.get("rebalance_log") or [], ensure_ascii=False
            ),
            "updated_at": now_str,
        }

        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM strategy_backtest_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO strategy_backtest_runs
                    (run_id, strategy_id, strategy_name, run_mode, status, start_date, end_date,
                     trading_days, rebalance_frequency, commission_rate, slippage_rate,
                     total_return, cagr, annual_volatility, max_drawdown, sharpe_ratio,
                     sortino_ratio, calmar_ratio, daily_win_rate, rebalance_count, final_nav,
                     config_snapshot, metrics_json, latest_details_json, rebalance_log_json, updated_at)
                    VALUES (:run_id, :strategy_id, :strategy_name, :run_mode, :status, :start_date, :end_date,
                            :trading_days, :rebalance_frequency, :commission_rate, :slippage_rate,
                            :total_return, :cagr, :annual_volatility, :max_drawdown, :sharpe_ratio,
                            :sortino_ratio, :calmar_ratio, :daily_win_rate, :rebalance_count, :final_nav,
                            :config_snapshot, :metrics_json, :latest_details_json, :rebalance_log_json, :updated_at)
                    """
                ),
                row_params,
            )

        if self._client is not None:
            try:
                clean = self._clean_records_for_json([row_params])[0]
                self._client.table("strategy_backtest_runs").upsert(
                    clean, on_conflict="run_id"
                ).execute()
            except Exception as exc:
                logger.warning("写入远端 strategy_backtest_runs 失败: %s", exc)

        return row_params

    def fetch_strategy_backtest_runs(
        self,
        strategy_id: Optional[str] = None,
        run_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if strategy_id:
            clauses.append("strategy_id = :strategy_id")
            params["strategy_id"] = strategy_id
        if run_mode:
            clauses.append("run_mode = :run_mode")
            params["run_mode"] = run_mode
        sql = "SELECT * FROM strategy_backtest_runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, run_id ASC"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        if df.empty:
            return []
        results: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = dict(row)
            for jcol in [
                "config_snapshot",
                "metrics_json",
                "latest_details_json",
                "rebalance_log_json",
            ]:
                if isinstance(item.get(jcol), str) and item[jcol]:
                    try:
                        item[jcol] = json.loads(item[jcol])
                    except Exception:
                        pass
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # 7. 数据库概览与一键迁移到 Supabase
    # ------------------------------------------------------------------
    def get_database_overview(self) -> Dict[str, Any]:
        """统计本地关系型数据库中全部 7 张核心表的数据量与最新状态"""
        self.sync_strategy_meta_from_configs()
        with self.engine.connect() as conn:
            meta_df = pd.read_sql(
                text("SELECT * FROM asset_meta ORDER BY asset_type, symbol"),
                conn,
            )
            mkt_stats = pd.read_sql(
                text(
                    """
                    SELECT symbol,
                           COUNT(*) AS bar_count,
                           MIN(trade_date) AS min_date,
                           MAX(trade_date) AS max_date,
                           SUM(CASE WHEN pe IS NOT NULL THEN 1 ELSE 0 END) AS pe_count
                    FROM market_daily
                    GROUP BY symbol
                    ORDER BY symbol
                    """
                ),
                conn,
            )
            fac_stats = pd.read_sql(
                text(
                    """
                    SELECT symbol,
                           factor_id,
                           COUNT(*) AS row_count,
                           MIN(trade_date) AS min_date,
                           MAX(trade_date) AS max_date
                    FROM factor_daily
                    GROUP BY symbol, factor_id
                    ORDER BY symbol, factor_id
                    """
                ),
                conn,
            )
            strat_meta_cnt = conn.execute(
                text("SELECT COUNT(*) FROM strategy_meta")
            ).scalar()
            sig_count = conn.execute(
                text("SELECT COUNT(*) FROM strategy_signals")
            ).scalar()
            sig_detail_count = conn.execute(
                text("SELECT COUNT(*) FROM strategy_signal_asset_details")
            ).scalar()
            bt_run_count = conn.execute(
                text("SELECT COUNT(*) FROM strategy_backtest_runs")
            ).scalar()
            mkt_total = conn.execute(
                text("SELECT COUNT(*) FROM market_daily")
            ).scalar()
            fac_total = conn.execute(
                text("SELECT COUNT(*) FROM factor_daily")
            ).scalar()

        return {
            "db_url": self.db_url,
            "supabase_configured": self.is_remote_enabled,
            "counts": {
                "asset_meta": len(meta_df),
                "market_daily": int(mkt_total or 0),
                "factor_daily": int(fac_total or 0),
                "strategy_meta": int(strat_meta_cnt or 0),
                "strategy_signals": int(sig_count or 0),
                "strategy_signal_asset_details": int(sig_detail_count or 0),
                "strategy_backtest_runs": int(bt_run_count or 0),
            },
            "asset_meta": meta_df.to_dict(orient="records"),
            "strategy_meta": self.fetch_strategy_meta(),
            "market_summary": mkt_stats.to_dict(orient="records"),
            "factor_summary": fac_stats.to_dict(orient="records"),
        }

    def migrate_to_supabase(self, chunk_size: int = 1500) -> Dict[str, int]:
        """将本地关系库中的 7 张核心表一键全量推送到远端 Supabase"""
        if self._client is None:
            raise RuntimeError(
                "尚未配置 SUPABASE_URL 和 SUPABASE_SERVICE_ROLE_KEY，请先在 .env 中配置后再执行迁移。"
            )
        stats = {
            "asset_meta": 0,
            "market_daily": 0,
            "factor_daily": 0,
            "strategy_meta": 0,
            "strategy_signals": 0,
            "strategy_signal_asset_details": 0,
            "strategy_backtest_runs": 0,
        }
        with self.engine.connect() as conn:
            # 1. asset_meta
            df_meta = pd.read_sql(text("SELECT * FROM asset_meta"), conn)
            if not df_meta.empty:
                rows = self._clean_records_for_json(df_meta.to_dict("records"))
                for r in rows:
                    r["is_active"] = bool(r.get("is_active", True))
                    if isinstance(r.get("source_config"), str):
                        try:
                            r["source_config"] = json.loads(r["source_config"])
                        except Exception:
                            r["source_config"] = {}
                    elif r.get("source_config") is None:
                        r["source_config"] = {}
                self._client.table("asset_meta").upsert(
                    rows, on_conflict="symbol"
                ).execute()
                stats["asset_meta"] = len(rows)
                logger.info("已迁移 asset_meta: %d 行", len(rows))

            # 2. market_daily
            df_mkt = pd.read_sql(text("SELECT * FROM market_daily"), conn)
            if not df_mkt.empty:
                rows = self._clean_records_for_json(df_mkt.to_dict("records"))
                for r in rows:
                    if isinstance(r.get("extra_json"), str):
                        try:
                            r["extra_json"] = json.loads(r["extra_json"])
                        except Exception:
                            r["extra_json"] = {}
                    elif r.get("extra_json") is None:
                        r["extra_json"] = {}
                for i in range(0, len(rows), chunk_size):
                    self._client.table("market_daily").upsert(
                        rows[i : i + chunk_size], on_conflict="symbol,trade_date"
                    ).execute()
                stats["market_daily"] = len(rows)
                logger.info("已迁移 market_daily: %d 行", len(rows))

            # 3. factor_daily
            df_fac = pd.read_sql(text("SELECT * FROM factor_daily"), conn)
            if not df_fac.empty:
                rows = self._clean_records_for_json(df_fac.to_dict("records"))
                for i in range(0, len(rows), chunk_size):
                    self._client.table("factor_daily").upsert(
                        rows[i : i + chunk_size],
                        on_conflict="symbol,trade_date,factor_id",
                    ).execute()
                stats["factor_daily"] = len(rows)
                logger.info("已迁移 factor_daily: %d 行", len(rows))

            # 4. strategy_meta
            df_smeta = pd.read_sql(text("SELECT * FROM strategy_meta"), conn)
            if not df_smeta.empty:
                rows = self._clean_records_for_json(df_smeta.to_dict("records"))
                try:
                    self._client.table("strategy_meta").upsert(
                        rows, on_conflict="strategy_id"
                    ).execute()
                    stats["strategy_meta"] = len(rows)
                except Exception as exc:
                    logger.warning("迁移 strategy_meta 跳过: %s", exc)

            # 5. strategy_signals
            sigs = self.fetch_all_strategy_signals()
            if sigs:
                rows = self._clean_records_for_json(sigs)
                for r in rows:
                    r["is_rebalance_day"] = bool(r.get("is_rebalance_day", False))
                self._client.table("strategy_signals").upsert(
                    rows, on_conflict="strategy_id,signal_date,run_mode"
                ).execute()
                stats["strategy_signals"] = len(rows)
                logger.info("已迁移 strategy_signals: %d 行", len(rows))

            # 6. strategy_signal_asset_details
            df_sdet = pd.read_sql(
                text("SELECT * FROM strategy_signal_asset_details"), conn
            )
            if not df_sdet.empty:
                rows = self._clean_records_for_json(df_sdet.to_dict("records"))
                try:
                    self._client.table("strategy_signal_asset_details").upsert(
                        rows,
                        on_conflict="strategy_id,signal_date,run_mode,asset_symbol",
                    ).execute()
                    stats["strategy_signal_asset_details"] = len(rows)
                except Exception as exc:
                    logger.warning("迁移 strategy_signal_asset_details 跳过: %s", exc)

            # 7. strategy_backtest_runs
            df_btr = pd.read_sql(text("SELECT * FROM strategy_backtest_runs"), conn)
            if not df_btr.empty:
                rows = self._clean_records_for_json(df_btr.to_dict("records"))
                try:
                    self._client.table("strategy_backtest_runs").upsert(
                        rows, on_conflict="run_id"
                    ).execute()
                    stats["strategy_backtest_runs"] = len(rows)
                except Exception as exc:
                    logger.warning("迁移 strategy_backtest_runs 跳过: %s", exc)

        return stats

    @staticmethod
    def _clean_records_for_json(
        records: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for row in records:
            item: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    item[k] = None
                elif isinstance(v, float) and (pd.isna(v) or v != v):
                    item[k] = None
                elif isinstance(v, (pd.Timestamp, datetime)):
                    item[k] = v.strftime("%Y-%m-%d")
                elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                    item[k] = None
                else:
                    item[k] = v
            cleaned.append(item)
        return cleaned
