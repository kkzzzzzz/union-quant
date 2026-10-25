-- ============================================================================
-- 增量建表脚本（无 DROP 关键字 + 自带 RLS 安全策略，消除 Supabase 控制台警告）
-- 目标库：now_asset 所在的 Supabase 库 (rdvjeyohxyghqllqjiwe)
-- ============================================================================

-- 1. 统一标的元数据表 (asset_meta)
CREATE TABLE IF NOT EXISTS public.asset_meta (
    symbol           TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    asset_type       TEXT NOT NULL CHECK (asset_type IN ('index', 'etf', 'bond', 'gold', 'macro', 'cash')),
    market           TEXT NOT NULL DEFAULT 'CN',
    currency         TEXT NOT NULL DEFAULT 'CNY',
    trade_etf        TEXT,
    source_config    JSONB NOT NULL DEFAULT '{}'::jsonb,
    latest_price     DOUBLE PRECISION,
    latest_date      DATE,
    latest_pe        DOUBLE PRECISION,
    latest_pb        DOUBLE PRECISION,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. 统一日频行情与估值表 (market_daily)
CREATE TABLE IF NOT EXISTS public.market_daily (
    symbol           TEXT NOT NULL,
    trade_date       DATE NOT NULL,
    open             DOUBLE PRECISION,
    high             DOUBLE PRECISION,
    low              DOUBLE PRECISION,
    close            DOUBLE PRECISION,
    volume           DOUBLE PRECISION,
    amount           DOUBLE PRECISION,
    change_pct       DOUBLE PRECISION,
    pe               DOUBLE PRECISION,
    pb               DOUBLE PRECISION,
    dividend_yield   DOUBLE PRECISION,
    roe              DOUBLE PRECISION,
    extra_json       JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_daily_date
    ON public.market_daily (trade_date DESC);

-- 3. 统一衍生因子窄表 (factor_daily)
CREATE TABLE IF NOT EXISTS public.factor_daily (
    symbol           TEXT NOT NULL,
    trade_date       DATE NOT NULL,
    factor_id        TEXT NOT NULL,
    value            DOUBLE PRECISION NOT NULL,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, trade_date, factor_id)
);

CREATE INDEX IF NOT EXISTS idx_factor_daily_lookup
    ON public.factor_daily (symbol, factor_id, trade_date DESC);

-- 4. 统一策略信号单表 (strategy_signals)
CREATE TABLE IF NOT EXISTS public.strategy_signals (
    id               BIGSERIAL PRIMARY KEY,
    strategy_id      TEXT NOT NULL,
    strategy_name    TEXT NOT NULL,
    strategy_version TEXT NOT NULL DEFAULT '2.0',
    signal_date      DATE NOT NULL,
    execution_date   DATE,
    run_mode         TEXT NOT NULL DEFAULT 'live' CHECK (run_mode IN ('live', 'backtest', 'paper', 'dry-run')),
    is_rebalance_day BOOLEAN NOT NULL DEFAULT FALSE,
    trigger_reason   TEXT,
    cash_weight      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    weights          JSONB NOT NULL DEFAULT '{}'::jsonb,
    asset_details    JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    config_snapshot  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_strategy_signals_key UNIQUE (strategy_id, signal_date, run_mode)
);

CREATE INDEX IF NOT EXISTS idx_strategy_signals_latest
    ON public.strategy_signals (strategy_id, run_mode, signal_date DESC);

-- 5. 开启 RLS (Row Level Security)：后端 service_role 密钥默认拥有全部读写权限，
--    同时允许已登录/前端只读查询这 4 张公共行情与信号表
ALTER TABLE public.asset_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.factor_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.strategy_signals ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow read access to asset_meta"
    ON public.asset_meta FOR SELECT USING (true);

CREATE POLICY "Allow read access to market_daily"
    ON public.market_daily FOR SELECT USING (true);

CREATE POLICY "Allow read access to factor_daily"
    ON public.factor_daily FOR SELECT USING (true);

CREATE POLICY "Allow read access to strategy_signals"
    ON public.strategy_signals FOR SELECT USING (true);
