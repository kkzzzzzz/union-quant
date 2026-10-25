-- =============================================================================
-- Union Quant 统一数据库 Schema (Single Supabase PostgreSQL)
-- 包含四大模块：
--   1. 行情与估值核心表 (asset_meta, market_daily)
--   2. 轻量派生因子表 (factor_daily)
--   3. 统一策略信号表 (strategy_signals)
--   4. 多用户鉴权、合规与实盘组合账本表 (app_users, user_subscriptions, operation_records 等)
-- 幂等设计：支持重复执行 (CREATE TABLE / INDEX IF NOT EXISTS)
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 通用 updated_at 自动维护触发器函数
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

-- =============================================================================
-- 1. 行情与估值核心表 (Data Hub)
-- =============================================================================

-- 1.1 统一资产元数据字典表（合并原 index_info / etf_info / index_etf_mapping / fund_nav_cache）
CREATE TABLE IF NOT EXISTS public.asset_meta (
    symbol          VARCHAR(30) PRIMARY KEY,               -- 唯一代码，如 'NDX.GI', '000300.SH', '510300.SH', 'SBE.CSI800'
    name            VARCHAR(100) NOT NULL,                 -- 资产中文名，如 '纳斯达克100'
    asset_type      VARCHAR(20)  NOT NULL DEFAULT 'index'  -- 'index' | 'etf' | 'fund' | 'macro' | 'bond' | 'commodity'
                    CHECK (asset_type IN ('index', 'etf', 'fund', 'macro', 'bond', 'commodity')),
    category        VARCHAR(50),                           -- 资产大类标签，如 '美股', 'A股', '港股', '固收', '商品'
    trade_etf       VARCHAR(30),                           -- 指数默认映射的场内可交易 ETF 代码，如 '513100.SH'
    source_config   JSONB        NOT NULL DEFAULT '{}'::jsonb, -- 数据源参数 (sina_code, howbuy_zsdm, ak_symbol, chinabond_id 等)
    latest_price    DOUBLE PRECISION,                      -- 最新收盘价/单位净值缓存
    latest_date     DATE,                                  -- 最新价格对应交易日
    latest_pct      DOUBLE PRECISION,                      -- 最新日涨跌幅(%)
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 1.2 统一日频行情与基本面估值表（合并原 index_daily_bars / etf_daily_bars / index_fundamental_daily）
CREATE TABLE IF NOT EXISTS public.market_daily (
    symbol          VARCHAR(30) NOT NULL,                  -- 资产代码
    trade_date      DATE        NOT NULL,                  -- 交易日期
    name            VARCHAR(100),                          -- 资产名称冗余（便于直接查表）
    -- 行情字段
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,                      -- 收盘价 / 基金单位净值 / 宏观分位数
    adj_close       DOUBLE PRECISION,                      -- 前复权收盘价 / 累计复权净值（回测优先使用）
    pct_change      DOUBLE PRECISION,                      -- 日涨跌幅 (%)
    volume          BIGINT,                                -- 成交量
    amount          DOUBLE PRECISION,                      -- 成交额
    turnover_rate   DOUBLE PRECISION,                      -- 换手率 (%)
    -- 估值与基本面字段
    pe              DOUBLE PRECISION,                      -- 市盈率 (PE-TTM)
    pb              DOUBLE PRECISION,                      -- 市净率 (PB)
    dividend_yield  DOUBLE PRECISION,                      -- 股息率 (%)
    roe             DOUBLE PRECISION,                      -- 净资产收益率 (%)
    extra_json      JSONB       NOT NULL DEFAULT '{}'::jsonb, -- 扩展指标（如好买原生百分位、FED 风险溢价、折溢价率等）
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_daily_date ON public.market_daily (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_daily_symbol_date ON public.market_daily (symbol, trade_date DESC);

-- =============================================================================
-- 2. 轻量派生因子窄表 (Factor Engine)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.factor_daily (
    symbol          VARCHAR(30)  NOT NULL,                 -- 资产代码，如 '000300.SH'
    trade_date      DATE         NOT NULL,                 -- 交易日期
    factor_id       VARCHAR(64)  NOT NULL,                 -- 因子标识，如 'pe_percentile_10y', 'rsi_14', 'log_bias'
    value           DOUBLE PRECISION,                      -- 因子值
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, trade_date, factor_id)
);

CREATE INDEX IF NOT EXISTS idx_factor_daily_lookup ON public.factor_daily (factor_id, symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_factor_daily_date ON public.factor_daily (trade_date DESC);

-- =============================================================================
-- 3. 统一策略信号与回测快照表 (Strategy Engine)
--    合并原 strategy_result + strategy_result_asset_detail + strategy_result_factor_detail
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.strategy_signals (
    id                      BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    strategy_id             VARCHAR(100) NOT NULL,         -- 策略唯一标识，如 'global_asset_allocation'
    strategy_name           VARCHAR(100) NOT NULL,         -- 策略展示名称
    strategy_mode           VARCHAR(50)  NOT NULL DEFAULT 'allocation',
    run_date                DATE         NOT NULL,         -- 实际运行日 (T+1 日)
    target_date             DATE         NOT NULL,         -- 信号目标日 / 数据截止交易日 (T 日)
    effective_trade_date    DATE         NOT NULL,         -- 有效交易日 (T 日)
    target_holding           VARCHAR(100) NOT NULL,         -- 头号重仓资产或轮动目标 ('NDX.GI' / 'CASH' 等)
    weights                 JSONB        NOT NULL DEFAULT '{}'::jsonb, -- 目标权重字典 {"NDX.GI": 0.45, ...}
    asset_details           JSONB        NOT NULL DEFAULT '[]'::jsonb, -- 各资产明细列表（含 base/target/actual 权重、PE、因子值、验算公式）
    metrics_snapshot        JSONB        NOT NULL DEFAULT '{}'::jsonb, -- 回测绩效指标快照 (CAGR, MaxDD, Sharpe 等)
    config_snapshot         JSONB        NOT NULL DEFAULT '{}'::jsonb, -- 运行时策略配置快照
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_strategy_signals_daily UNIQUE (strategy_id, strategy_mode, target_date)
);

CREATE INDEX IF NOT EXISTS idx_strategy_signals_lookup
    ON public.strategy_signals (strategy_id, target_date DESC);

-- =============================================================================
-- 4. 多用户鉴权、合规与实盘组合账本表 (保留自 now_asset 核心规范)
-- =============================================================================

-- 4.1 应用层用户表
CREATE TABLE IF NOT EXISTS public.app_users (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name TEXT,
    role         TEXT NOT NULL DEFAULT 'user'   CHECK (role IN ('user', 'admin')),
    status       TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled', 'pending_deletion')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4.2 外部认证账号映射表 (IdP 解耦)
CREATE TABLE IF NOT EXISTS public.auth_accounts (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL REFERENCES public.app_users(id) ON DELETE CASCADE,
    issuer         TEXT NOT NULL,
    subject        TEXT NOT NULL,
    email_snapshot TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (issuer, subject)
);
CREATE INDEX IF NOT EXISTS idx_auth_accounts_user ON public.auth_accounts(user_id);

-- 4.3 法律与风险协议文档表
CREATE TABLE IF NOT EXISTS public.legal_documents (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type  TEXT NOT NULL CHECK (document_type IN ('terms', 'privacy', 'investment_risk')),
    version        TEXT NOT NULL,
    title          TEXT NOT NULL,
    content_url    TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    effective_at   TIMESTAMPTZ NOT NULL,
    is_current     BOOLEAN NOT NULL DEFAULT false,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_type, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_legal_current_type
    ON public.legal_documents(document_type) WHERE is_current = true;

-- 4.4 用户协议签署记录表
CREATE TABLE IF NOT EXISTS public.user_consents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES public.app_users(id) ON DELETE CASCADE,
    consent_type TEXT NOT NULL,
    version      TEXT NOT NULL,
    accepted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ip_hmac      TEXT,
    user_agent   TEXT,
    UNIQUE (user_id, consent_type, version),
    FOREIGN KEY (consent_type, version)
        REFERENCES public.legal_documents(document_type, version)
);

-- 4.5 策略目录上架表
CREATE TABLE IF NOT EXISTS public.strategy_catalog (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system       TEXT NOT NULL DEFAULT 'new' CHECK (source_system IN ('legacy', 'new')),
    source_key          TEXT NOT NULL,
    strategy_mode       TEXT NOT NULL DEFAULT 'allocation',
    name                TEXT NOT NULL,
    risk_level          TEXT NOT NULL DEFAULT '中',
    tagline             TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'disabled' CHECK (status IN ('active', 'disabled')),
    follow_latest       BOOLEAN NOT NULL DEFAULT true,
    current_version_id  TEXT,
    current_target_date DATE,
    display_order       INT NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_system, source_key, strategy_mode)
);

-- 4.6 用户策略订阅与派生持仓快照表
CREATE TABLE IF NOT EXISTS public.user_subscriptions (
    id                  UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID           NOT NULL REFERENCES public.app_users(id),
    catalog_id          UUID           NOT NULL REFERENCES public.strategy_catalog(id),
    strategy_id         VARCHAR(100)   NOT NULL,
    strategy_version_id TEXT,
    total_investment    NUMERIC(18, 2) NOT NULL DEFAULT 0,
    holdings            JSONB          NOT NULL DEFAULT '[]'::jsonb,
    status              TEXT           NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    ledger_revision     BIGINT         NOT NULL DEFAULT 0,
    snapshot_revision   BIGINT         NOT NULL DEFAULT 0,
    snapshot_status     TEXT           NOT NULL DEFAULT 'clean' CHECK (snapshot_status IN ('clean', 'dirty')),
    created_at          TIMESTAMPTZ    NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ    NOT NULL DEFAULT now(),
    CONSTRAINT user_subscriptions_id_user_key UNIQUE (id, user_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_user_active_catalog
    ON public.user_subscriptions(user_id, catalog_id) WHERE status = 'active';

-- 4.7 实盘交易流水批次表（幂等与并发控制）
CREATE TABLE IF NOT EXISTS public.operation_batches (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES public.app_users(id),
    subscription_id   UUID NOT NULL,
    client_request_id UUID NOT NULL,
    request_hash      TEXT NOT NULL,
    ledger_revision   BIGINT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('accepted', 'snapshot_pending', 'completed')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, client_request_id),
    UNIQUE (id, user_id, subscription_id),
    UNIQUE (subscription_id, ledger_revision),
    CONSTRAINT operation_batches_subscription_owner_fk
        FOREIGN KEY (subscription_id, user_id)
        REFERENCES public.user_subscriptions(id, user_id)
);

-- 4.8 实盘交易操作流水表（事件溯源唯一事实源）
CREATE TABLE IF NOT EXISTS public.operation_records (
    id                     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID        NOT NULL REFERENCES public.app_users(id),
    subscription_id        UUID        NOT NULL,
    batch_id               UUID        NOT NULL,
    strategy_name          TEXT        NOT NULL DEFAULT 'PE组合',
    data_source            TEXT        NOT NULL DEFAULT 'new',
    operation_type         TEXT        NOT NULL,
    operation_date         DATE        NOT NULL,
    fund_code              TEXT        NOT NULL,
    fund_name              TEXT,
    asset_alias            TEXT,
    amount                 NUMERIC     NOT NULL DEFAULT 0,
    shares_delta           NUMERIC     NOT NULL DEFAULT 0,
    nav                    NUMERIC,
    fee                    NUMERIC     NOT NULL DEFAULT 0,
    source                 TEXT,
    plan_id                TEXT,
    reason                 TEXT,
    signal_snapshot        JSONB,
    pre_deviation          NUMERIC,
    post_deviation         NUMERIC,
    cost_basis_delta       NUMERIC,
    benefit_amount         NUMERIC,
    benefit_rate           NUMERIC,
    benefit_as_of          DATE,
    allocation_improvement NUMERIC,
    execution_slippage     NUMERIC,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_operation_subscription_owner
        FOREIGN KEY (subscription_id, user_id)
        REFERENCES public.user_subscriptions(id, user_id),
    CONSTRAINT fk_operation_batch_owner
        FOREIGN KEY (batch_id, user_id, subscription_id)
        REFERENCES public.operation_batches(id, user_id, subscription_id)
);

CREATE INDEX IF NOT EXISTS idx_op_subscription_date
    ON public.operation_records(subscription_id, operation_date, created_at, id);
CREATE INDEX IF NOT EXISTS idx_op_user_subscription
    ON public.operation_records(user_id, subscription_id);

-- 4.9 变更幂等、数据合规申请、审计日志与限流桶
CREATE TABLE IF NOT EXISTS public.api_mutation_requests (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES public.app_users(id),
    client_request_id UUID NOT NULL,
    action            TEXT NOT NULL,
    resource_id       TEXT,
    request_hash      TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('accepted', 'completed')),
    response_json     JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ,
    UNIQUE (user_id, client_request_id)
);

CREATE TABLE IF NOT EXISTS public.account_data_requests (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID REFERENCES public.app_users(id) ON DELETE SET NULL,
    user_ref_snapshot TEXT NOT NULL,
    email_snapshot    TEXT,
    request_type      TEXT NOT NULL CHECK (request_type IN ('export', 'deletion')),
    status            TEXT NOT NULL DEFAULT 'received'
                      CHECK (status IN ('received', 'processing', 'completed', 'rejected')),
    requested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ,
    handled_by        TEXT,
    note              TEXT
);

CREATE TABLE IF NOT EXISTS public.admin_audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    target_user_id  UUID,
    target_resource TEXT,
    details         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION public.deny_audit_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  RAISE EXCEPTION 'admin_audit_logs is append-only (attempted %)', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_admin_audit_append_only ON public.admin_audit_logs;
CREATE TRIGGER trg_admin_audit_append_only
    BEFORE UPDATE OR DELETE ON public.admin_audit_logs
    FOR EACH ROW EXECUTE FUNCTION public.deny_audit_mutation();

CREATE TABLE IF NOT EXISTS public.api_rate_limit_buckets (
    user_id    UUID NOT NULL REFERENCES public.app_users(id) ON DELETE CASCADE,
    endpoint   TEXT NOT NULL,
    tokens     NUMERIC NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, endpoint)
);

-- 挂载 updated_at 自动更新触发器
DO $$
DECLARE t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'asset_meta',
    'market_daily',
    'factor_daily',
    'strategy_signals',
    'app_users',
    'strategy_catalog',
    'user_subscriptions',
    'operation_batches',
    'operation_records'
  ]
  LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS trg_%s_updated_at ON public.%I', t, t);
    EXECUTE format(
      'CREATE TRIGGER trg_%s_updated_at BEFORE UPDATE ON public.%I
         FOR EACH ROW EXECUTE FUNCTION public.set_updated_at()', t, t);
  END LOOP;
END;
$$;

COMMIT;
