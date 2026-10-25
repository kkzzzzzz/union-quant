"""
统一量化系统可视化核验看板 (Union Quant Verification Web Dashboard).

启动方式：
  python cli.py serve --port 8088
  或直接运行：
  python web_dashboard.py

功能一览：
1. 库表总览：实时查看本地关系型数据库 (`data/union_quant.db` 或 MySQL/Postgres) 四张核心表
   (`asset_meta`, `market_daily`, `factor_daily`, `strategy_signals`) 的数据量与字段明细。
2. 行情/估值/因子核验：按任意标的查看收盘价、PE/PB/股息率走势、向量化因子曲线及最新 SQL 行记录。
3. 策略回测与每日信号核验：查看全部 6 个策略的最新 T+1 仓位信号、单资产诊断明细、净值回撤曲线与历史调仓记录。
4. 在线触发：支持在网页上一键触发「拉取数据并重算因子」与「重新计算全部策略信号」。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from core.config import AppConfig
from core.data_hub.bootstrap import bootstrap_verification_data
from core.db import DatabaseGateway
from core.quant_engine.pipeline import QuantPipeline

app = FastAPI(title="Union Quant Verification Dashboard", version="2.0")


def _get_services() -> tuple[AppConfig, DatabaseGateway, QuantPipeline]:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    pipeline = QuantPipeline(config=cfg, db=db)
    return cfg, db, pipeline


@app.get("/api/overview")
def api_overview() -> Dict[str, Any]:
    cfg, db, _ = _get_services()
    overview = db.get_database_overview()
    signals = db.fetch_all_strategy_signals(run_mode="live")
    overview["signals"] = signals
    overview["registered_strategies"] = cfg.list_strategies()
    return overview


@app.get("/api/market/{symbol}")
def api_market_detail(
    symbol: str, tail: int = Query(default=500, ge=20, le=3000)
) -> Dict[str, Any]:
    cfg, db, _ = _get_services()
    spec = cfg.get_asset(symbol)
    df_mkt = db.fetch_market_daily(symbol)
    if df_mkt.empty:
        raise HTTPException(status_code=404, detail=f"未找到标的 {symbol} 的行情数据")

    df_sub = df_mkt.tail(tail).copy()
    factors_data: Dict[str, list] = {}
    factor_ids = spec.factors if spec else []
    for fid in factor_ids:
        s = db.fetch_factor_series(symbol, fid)
        if not s.empty:
            s_aligned = s.reindex(pd.to_datetime(df_sub["trade_date"]))
            factors_data[fid] = [
                None if pd.isna(v) else round(float(v), 3) for v in s_aligned.values
            ]

    # 最近 50 条表格明细（倒序）
    recent_df = df_sub.tail(50).iloc[::-1].copy()
    recent_rows = DatabaseGateway._clean_records_for_json(
        recent_df.to_dict(orient="records")
    )

    return {
        "symbol": symbol,
        "name": spec.name if spec else symbol,
        "asset_type": spec.asset_type if spec else "index",
        "market": spec.market if spec else "CN",
        "trade_etf": spec.trade_etf if spec else None,
        "dates": df_sub["trade_date"].tolist(),
        "close": [
            None if pd.isna(v) else round(float(v), 4) for v in df_sub["close"]
        ],
        "pe": [
            None if pd.isna(v) else round(float(v), 3)
            for v in (
                df_sub["pe"] if "pe" in df_sub.columns else [None] * len(df_sub)
            )
        ],
        "pb": [
            None if pd.isna(v) else round(float(v), 3)
            for v in (
                df_sub["pb"] if "pb" in df_sub.columns else [None] * len(df_sub)
            )
        ],
        "dividend_yield": [
            None if pd.isna(v) else round(float(v), 3)
            for v in (
                df_sub["dividend_yield"]
                if "dividend_yield" in df_sub.columns
                else [None] * len(df_sub)
            )
        ],
        "factors": factors_data,
        "recent_rows": recent_rows,
    }


@app.get("/api/strategy/{strategy_id}")
def api_strategy_detail(strategy_id: str) -> Dict[str, Any]:
    _, db, pipeline = _get_services()
    bt = pipeline.run_backtest(strategy_id=strategy_id, save_outputs=False)
    signals = [
        s
        for s in db.fetch_all_strategy_signals()
        if s["strategy_id"] == strategy_id
    ]
    latest_signal = signals[0] if signals else None
    if latest_signal is None:
        latest_signal = pipeline.run_daily_signal(
            strategy_id=strategy_id, run_mode="live", notify=False
        )

    eq = bt.equity_curve
    dates = [d.strftime("%Y-%m-%d") for d in eq.index] if not eq.empty else []
    nav = (
        [round(float(v), 4) for v in eq["nav"].values] if not eq.empty else []
    )
    dd = (
        [round(float(v * 100.0), 2) for v in eq["drawdown"].values]
        if not eq.empty
        else []
    )

    return {
        "strategy_id": bt.strategy_id,
        "strategy_name": bt.strategy_name,
        "metrics": bt.metrics,
        "signal": latest_signal,
        "chart": {
            "dates": dates,
            "nav": nav,
            "drawdown_pct": dd,
        },
        "rebalance_log": list(reversed(bt.rebalance_log[-60:])),
    }


@app.post("/api/bootstrap")
def api_trigger_bootstrap(try_live: bool = True) -> Dict[str, Any]:
    cfg, db, _ = _get_services()
    return bootstrap_verification_data(
        config=cfg, db=db, try_live_fetch=try_live
    )


@app.post("/api/run-signals")
def api_trigger_signals(date: Optional[str] = None) -> Dict[str, Any]:
    cfg, _, pipeline = _get_services()
    results = []
    for sid in cfg.list_strategies():
        sig = pipeline.run_daily_signal(
            strategy_id=sid, run_date=date, run_mode="live", notify=False
        )
        results.append(
            {
                "strategy_id": sid,
                "signal_date": sig["signal_date"],
                "execution_date": sig["execution_date"],
                "weights": sig["weights"],
            }
        )
    return {"status": "ok", "count": len(results), "signals": results}


# ==============================================================================
# 前端单页可视化核验界面 (HTML + Tailwind + ECharts)
# ==============================================================================
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Union Quant — 数据与策略信号核验工作台</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif; background: #0f172a; color: #e2e8f0; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; }
    .tab-btn.active { background: #2563eb; color: #fff; border-color: #3b82f6; }
    .strat-btn.active { background: #1d4ed8; border-color: #60a5fa; color: #fff; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- 顶部导航栏 -->
  <header class="border-b border-slate-800 bg-slate-900/90 sticky top-0 z-30 backdrop-blur">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-lg shadow">UQ</div>
        <div>
          <h1 class="text-base font-bold text-white flex items-center gap-2">
            Union Quant 统一量化核验台
            <span class="text-xs font-normal px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">一个库表 · 一个拉数据 · 一个算策略</span>
          </h1>
          <p id="db-url-banner" class="text-xs text-slate-400 font-mono">正在连接本地关系数据库...</p>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button onclick="switchTab('strategies')" id="tab-strategies" class="tab-btn active px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 transition">
          1. 策略信号与回测检验
        </button>
        <button onclick="switchTab('market')" id="tab-market" class="tab-btn px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition">
          2. 行情/估值/因子检验
        </button>
        <button onclick="switchTab('database')" id="tab-database" class="tab-btn px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition">
          3. 库表总览与 Supabase 迁移
        </button>
        <div class="h-5 w-px bg-slate-700 mx-1"></div>
        <button onclick="triggerRunSignals()" id="btn-run-signals" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition shadow">
          ⚡ 重算全部信号
        </button>
        <button onclick="triggerBootstrap()" id="btn-bootstrap" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition shadow">
          🔄 同步最新行情/估值
        </button>
      </div>
    </div>
  </header>

  <!-- 顶部 4 张核心表统计卡片 -->
  <section class="max-w-7xl mx-auto w-full px-4 pt-4">
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div class="card p-3.5 flex items-center justify-between">
        <div>
          <div class="text-xs text-slate-400">标的元数据表 (<code class="text-blue-400">asset_meta</code>)</div>
          <div id="cnt-asset-meta" class="text-2xl font-bold text-white mt-0.5">-</div>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">指数/债/金/ETF/SBE</span>
      </div>
      <div class="card p-3.5 flex items-center justify-between">
        <div>
          <div class="text-xs text-slate-400">日线行情估值表 (<code class="text-emerald-400">market_daily</code>)</div>
          <div id="cnt-market-daily" class="text-2xl font-bold text-white mt-0.5">-</div>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">K线 + PE/PB/股息率</span>
      </div>
      <div class="card p-3.5 flex items-center justify-between">
        <div>
          <div class="text-xs text-slate-400">向量化因子窄表 (<code class="text-amber-400">factor_daily</code>)</div>
          <div id="cnt-factor-daily" class="text-2xl font-bold text-white mt-0.5">-</div>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">12 个注册算子</span>
      </div>
      <div class="card p-3.5 flex items-center justify-between">
        <div>
          <div class="text-xs text-slate-400">策略信号单表 (<code class="text-purple-400">strategy_signals</code>)</div>
          <div id="cnt-strategy-signals" class="text-2xl font-bold text-white mt-0.5">-</div>
        </div>
        <span class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300">JSONB 权重+诊断</span>
      </div>
    </div>
  </section>

  <!-- 主内容区 -->
  <main class="max-w-7xl mx-auto w-full px-4 py-4 flex-1">
    <!-- ==================== TAB 1: 策略信号与回测检验 ==================== -->
    <div id="view-strategies" class="space-y-4">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <!-- 左侧策略选择列表 -->
        <div class="lg:col-span-3 card p-3 space-y-2">
          <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1 mb-2">选择要核验的策略 (6个)</div>
          <div id="strategy-list" class="space-y-1.5"></div>
        </div>

        <!-- 右侧策略信号与回测详情 -->
        <div class="lg:col-span-9 space-y-4">
          <!-- 最新每日信号头部卡片 -->
          <div class="card p-4">
            <div class="flex flex-wrap items-start justify-between gap-3 border-b border-slate-700/80 pb-3">
              <div>
                <div class="flex items-center gap-2">
                  <h2 id="st-title" class="text-lg font-bold text-white">-</h2>
                  <span id="st-rebalance-badge" class="text-xs px-2.5 py-0.5 rounded-full font-medium">-</span>
                </div>
                <p id="st-trigger-reason" class="text-xs text-slate-300 mt-1">-</p>
              </div>
              <div class="flex items-center gap-4 text-xs bg-slate-900/80 px-3 py-2 rounded-lg border border-slate-800">
                <div>
                  <span class="text-slate-400">信号基准日 (T):</span>
                  <span id="st-signal-date" class="font-mono font-bold text-emerald-400 ml-1">-</span>
                </div>
                <div>
                  <span class="text-slate-400">计划执行日 (T+1):</span>
                  <span id="st-exec-date" class="font-mono font-bold text-blue-400 ml-1">-</span>
                </div>
                <div>
                  <span class="text-slate-400">现金比例:</span>
                  <span id="st-cash-weight" class="font-mono font-bold text-amber-300 ml-1">-</span>
                </div>
              </div>
            </div>

            <!-- 目标仓位与单资产诊断明细表 -->
            <div class="mt-3">
              <div class="text-xs font-semibold text-slate-400 mb-2 flex items-center justify-between">
                <span>最新目标仓位与单资产因子诊断明细 (<code class="text-purple-300">strategy_signals.asset_details</code>)</span>
                <span class="text-slate-500">注：跨市场标的各自展示其截至 T 日的最新真实行情日期</span>
              </div>
              <div class="overflow-x-auto">
                <table class="w-full text-xs text-left border-collapse">
                  <thead>
                    <tr class="border-b border-slate-700 text-slate-400 bg-slate-800/50">
                      <th class="py-2 px-2.5">标的代码</th>
                      <th class="py-2 px-2.5">名称</th>
                      <th class="py-2 px-2.5">交易ETF</th>
                      <th class="py-2 px-2.5">角色/排名</th>
                      <th class="py-2 px-2.5">数据日期 (T)</th>
                      <th class="py-2 px-2.5 text-right">最新收盘价</th>
                      <th class="py-2 px-2.5 text-right">PE / 均线</th>
                      <th class="py-2 px-2.5">核心因子 / 动量指标</th>
                      <th class="py-2 px-2.5 w-44">目标配置权重</th>
                    </tr>
                  </thead>
                  <tbody id="st-asset-details-body" class="divide-y divide-slate-800"></tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- 回测核心指标栏 -->
          <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5" id="st-metrics-grid"></div>

          <!-- 净值与回撤图表 -->
          <div class="card p-4">
            <div class="text-xs font-semibold text-slate-300 mb-2">T+1 历史回测净值曲线与水下回撤图</div>
            <div id="chart-strategy-nav" style="width: 100%; height: 340px;"></div>
          </div>

          <!-- 历史调仓记录表 -->
          <div class="card p-4">
            <div class="text-xs font-semibold text-slate-300 mb-2">历史调仓执行日志 (T 日信号 -> T+1 日执行)</div>
            <div class="overflow-x-auto max-h-64">
              <table class="w-full text-xs text-left border-collapse">
                <thead class="sticky top-0 bg-slate-800 text-slate-400">
                  <tr>
                    <th class="py-2 px-2.5">信号日期 (T)</th>
                    <th class="py-2 px-2.5">执行日期 (T+1)</th>
                    <th class="py-2 px-2.5">触发原因</th>
                    <th class="py-2 px-2.5 text-right">单边换手率</th>
                    <th class="py-2 px-2.5">执行后目标权重分布</th>
                  </tr>
                </thead>
                <tbody id="st-rebalance-body" class="divide-y divide-slate-800"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 2: 行情、估值与因子检验 ==================== -->
    <div id="view-market" class="hidden space-y-4">
      <div class="card p-3.5 flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <label class="text-xs font-semibold text-slate-300">选择检验标的:</label>
          <select id="select-symbol" onchange="loadMarketSymbol(this.value)" class="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"></select>
          <span id="mkt-badge" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono"></span>
        </div>
        <div class="text-xs text-slate-400">
          数据源表: <code class="text-emerald-400">market_daily</code> + <code class="text-amber-400">factor_daily</code>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-300 mb-2">日线收盘价与 PE / PB 估值双轴走势</div>
          <div id="chart-market-price" style="width: 100%; height: 320px;"></div>
        </div>
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-300 mb-2">已绑定向量化因子历史序列 (<code class="text-amber-400">factor_daily</code>)</div>
          <div id="chart-market-factors" style="width: 100%; height: 320px;"></div>
        </div>
      </div>

      <div class="card p-4">
        <div class="text-xs font-semibold text-slate-300 mb-2">最近 50 个交易日数据库原始记录 (<code class="text-emerald-400">market_daily</code>)</div>
        <div class="overflow-x-auto max-h-80">
          <table class="w-full text-xs text-left border-collapse">
            <thead class="sticky top-0 bg-slate-800 text-slate-400">
              <tr>
                <th class="py-2 px-2.5">交易日期</th>
                <th class="py-2 px-2.5">代码</th>
                <th class="py-2 px-2.5 text-right">开盘</th>
                <th class="py-2 px-2.5 text-right">最高</th>
                <th class="py-2 px-2.5 text-right">最低</th>
                <th class="py-2 px-2.5 text-right">收盘价 (close)</th>
                <th class="py-2 px-2.5 text-right">涨跌幅%</th>
                <th class="py-2 px-2.5 text-right">PE (市盈率)</th>
                <th class="py-2 px-2.5 text-right">PB (市净率)</th>
                <th class="py-2 px-2.5 text-right">股息率%</th>
              </tr>
            </thead>
            <tbody id="mkt-recent-body" class="divide-y divide-slate-800 font-mono"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 3: 库表总览与 Supabase 迁移 ==================== -->
    <div id="view-database" class="hidden space-y-4">
      <div class="card p-4 border-l-4 border-l-blue-500">
        <h3 class="text-sm font-bold text-white">如何从当前本地关系数据库无缝迁移到 Supabase / MySQL？</h3>
        <div class="mt-2 text-xs text-slate-300 space-y-1.5 leading-relaxed">
          <p>1. <strong>当前本地关系库</strong>：所有表结构（<code>asset_meta</code>、<code>market_daily</code>、<code>factor_daily</code>、<code>strategy_signals</code>）已严格按照 <code class="text-blue-400">sql/schema.sql</code> 规范建模并存储在本地关系库中。</p>
          <p>2. <strong>切换本地 MySQL</strong>：只需在根目录 <code>.env</code> 设置 <code>DATABASE_URL=mysql+pymysql://root:密码@127.0.0.1:3306/union_quant</code>，系统无需改任何代码即可直接读写 MySQL。</p>
          <p>3. <strong>一键迁移到 Supabase</strong>：在 Supabase SQL Editor 执行完 <code>sql/schema.sql</code> 后，在 <code>.env</code> 填入 <code>SUPABASE_URL</code> 与 <code>SUPABASE_SERVICE_ROLE_KEY</code>，运行命令：<code class="bg-slate-900 px-2 py-0.5 rounded text-emerald-400">python cli.py migrate-supabase</code> 即可把本地验证好的全部历史数据与信号一键全量推送上云。</p>
        </div>
      </div>

      <div class="card p-4">
        <div class="text-xs font-semibold text-slate-300 mb-2">各标的日线与估值覆盖度统计 (<code class="text-blue-400">asset_meta</code> JOIN <code class="text-emerald-400">market_daily</code>)</div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left border-collapse">
            <thead class="bg-slate-800 text-slate-400">
              <tr>
                <th class="py-2 px-2.5">标的代码</th>
                <th class="py-2 px-2.5">名称</th>
                <th class="py-2 px-2.5">类别</th>
                <th class="py-2 px-2.5">市场</th>
                <th class="py-2 px-2.5">交易ETF映射</th>
                <th class="py-2 px-2.5 text-right">总日线行数</th>
                <th class="py-2 px-2.5 text-right">含PE估值行数</th>
                <th class="py-2 px-2.5">起始日期</th>
                <th class="py-2 px-2.5">最新日期</th>
                <th class="py-2 px-2.5 text-right">最新收盘价</th>
                <th class="py-2 px-2.5 text-right">最新PE</th>
              </tr>
            </thead>
            <tbody id="db-summary-body" class="divide-y divide-slate-800"></tbody>
          </table>
        </div>
      </div>
    </div>
  </main>

  <script>
    let globalOverview = null;
    let currentStrategyId = null;
    let navChart = null;
    let priceChart = null;
    let factorChart = null;

    function switchTab(tab) {
      ['strategies', 'market', 'database'].forEach(t => {
        document.getElementById('view-' + t).classList.toggle('hidden', t !== tab);
        document.getElementById('tab-' + t).classList.toggle('active', t === tab);
      });
      if (tab === 'strategies' && navChart) setTimeout(() => navChart.resize(), 50);
      if (tab === 'market') {
        if (!priceChart) loadMarketSymbol(document.getElementById('select-symbol').value);
        else setTimeout(() => { priceChart.resize(); factorChart && factorChart.resize(); }, 50);
      }
    }

    async function loadOverview() {
      const resp = await fetch('/api/overview');
      globalOverview = await resp.json();

      document.getElementById('db-url-banner').textContent =
        `当前关系库引擎: ${globalOverview.db_url} | Supabase 远端同步: ${globalOverview.supabase_configured ? '已开启' : '未配置(本地独立运行)'}`;

      document.getElementById('cnt-asset-meta').textContent = globalOverview.counts.asset_meta.toLocaleString();
      document.getElementById('cnt-market-daily').textContent = globalOverview.counts.market_daily.toLocaleString();
      document.getElementById('cnt-factor-daily').textContent = globalOverview.counts.factor_daily.toLocaleString();
      document.getElementById('cnt-strategy-signals').textContent = globalOverview.counts.strategy_signals.toLocaleString();

      // 填充策略列表
      const sigMap = {};
      (globalOverview.signals || []).forEach(s => { sigMap[s.strategy_id] = s; });

      const listEl = document.getElementById('strategy-list');
      listEl.innerHTML = '';
      const strategies = globalOverview.registered_strategies || [];
      strategies.forEach((sid, idx) => {
        const sig = sigMap[sid];
        const name = sig ? sig.strategy_name : sid;
        const dt = sig ? sig.signal_date : '-';
        const btn = document.createElement('button');
        btn.id = 'strat-btn-' + sid;
        btn.className = 'strat-btn w-full text-left p-2.5 rounded-lg border border-slate-700/80 hover:bg-slate-800 transition flex flex-col gap-1';
        btn.onclick = () => selectStrategy(sid);
        btn.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-semibold text-xs text-white">${name}</span>
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">${dt}</span>
          </div>
          <div class="text-[11px] font-mono text-slate-400">${sid}</div>
        `;
        listEl.appendChild(btn);
        if (idx === 0 && !currentStrategyId) currentStrategyId = sid;
      });

      // 填充标的下拉框
      const sel = document.getElementById('select-symbol');
      sel.innerHTML = '';
      (globalOverview.asset_meta || []).forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.symbol;
        opt.textContent = `[${a.asset_type.toUpperCase()}] ${a.symbol} - ${a.name}`;
        sel.appendChild(opt);
      });

      // 填充库表统计明细
      renderDbSummaryTable();

      if (currentStrategyId) {
        selectStrategy(currentStrategyId);
      }
    }

    function renderDbSummaryTable() {
      const mktMap = {};
      (globalOverview.market_summary || []).forEach(m => { mktMap[m.symbol] = m; });
      const tbody = document.getElementById('db-summary-body');
      tbody.innerHTML = '';
      (globalOverview.asset_meta || []).forEach(a => {
        const m = mktMap[a.symbol] || {};
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/60 cursor-pointer';
        tr.onclick = () => {
          document.getElementById('select-symbol').value = a.symbol;
          switchTab('market');
          loadMarketSymbol(a.symbol);
        };
        tr.innerHTML = `
          <td class="py-2 px-2.5 font-mono font-semibold text-blue-400">${a.symbol}</td>
          <td class="py-2 px-2.5 text-white">${a.name}</td>
          <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300">${a.asset_type}</span></td>
          <td class="py-2 px-2.5 font-mono">${a.market}</td>
          <td class="py-2 px-2.5 font-mono text-slate-400">${a.trade_etf || '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono">${(m.bar_count || 0).toLocaleString()}</td>
          <td class="py-2 px-2.5 text-right font-mono text-amber-300">${(m.pe_count || 0).toLocaleString()}</td>
          <td class="py-2 px-2.5 font-mono text-slate-400">${m.min_date || '-'}</td>
          <td class="py-2 px-2.5 font-mono text-emerald-400">${a.latest_date || m.max_date || '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono text-white">${a.latest_price != null ? Number(a.latest_price).toFixed(3) : '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono text-amber-300">${a.latest_pe != null ? Number(a.latest_pe).toFixed(2) : '-'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    async function selectStrategy(sid) {
      currentStrategyId = sid;
      document.querySelectorAll('.strat-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('strat-btn-' + sid);
      if (activeBtn) activeBtn.classList.add('active');

      const resp = await fetch('/api/strategy/' + sid);
      const data = await resp.json();
      const sig = data.signal || {};

      document.getElementById('st-title').textContent = `${data.strategy_name} (${data.strategy_id})`;
      const badge = document.getElementById('st-rebalance-badge');
      if (sig.is_rebalance_day) {
        badge.textContent = '今日触发调仓';
        badge.className = 'text-xs px-2.5 py-0.5 rounded-full font-medium bg-rose-500/20 text-rose-300 border border-rose-500/40';
      } else {
        badge.textContent = '持仓观察';
        badge.className = 'text-xs px-2.5 py-0.5 rounded-full font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
      }
      document.getElementById('st-trigger-reason').textContent = `决策摘要: ${sig.trigger_reason || '-'}`;
      document.getElementById('st-signal-date').textContent = sig.signal_date || '-';
      document.getElementById('st-exec-date').textContent = sig.execution_date || '-';
      document.getElementById('st-cash-weight').textContent = `${((sig.cash_weight || 0) * 100).toFixed(1)}%`;

      // 单资产诊断明细表
      const tbody = document.getElementById('st-asset-details-body');
      tbody.innerHTML = '';
      (sig.asset_details || []).forEach(item => {
        const wPct = ((item.weight || 0) * 100).toFixed(1);
        let factorDesc = '-';
        if (item.factor_id && item.factor_value != null) {
          factorDesc = `${item.factor_id}: <span class="text-amber-300 font-mono">${item.factor_value}%</span>`;
        } else if (item.momentum_pct != null) {
          const color = item.momentum_pct >= 0 ? 'text-emerald-400' : 'text-rose-400';
          factorDesc = `动量收益: <span class="${color} font-mono">${item.momentum_pct}%</span> ${item.passed_filter === false ? '(未过均线)' : ''}`;
        } else if (item.sbe_quantile != null) {
          factorDesc = `SBE分位(${item.sbe_source}): <span class="text-amber-300 font-mono">${item.sbe_quantile}%</span>`;
        }

        const roleOrRank = item.role ? item.role.toUpperCase() : (item.rank ? `Rank #${item.rank}` : '-');
        const peOrMa = item.pe != null ? `PE=${Number(item.pe).toFixed(2)}` : (item.ma_value != null ? `MA=${Number(item.ma_value).toFixed(3)}` : '-');

        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/40';
        tr.innerHTML = `
          <td class="py-2 px-2.5 font-mono font-semibold text-blue-400">${item.symbol}</td>
          <td class="py-2 px-2.5 text-white">${item.name || item.symbol}</td>
          <td class="py-2 px-2.5 font-mono text-slate-400">${item.trade_symbol || '-'}</td>
          <td class="py-2 px-2.5"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">${roleOrRank}</span></td>
          <td class="py-2 px-2.5 font-mono text-emerald-400">${item.data_date || sig.signal_date || '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono">${item.close != null ? Number(item.close).toFixed(3) : '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono text-slate-300">${peOrMa}</td>
          <td class="py-2 px-2.5">${factorDesc}</td>
          <td class="py-2 px-2.5">
            <div class="flex items-center gap-2">
              <div class="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
                <div class="bg-blue-500 h-full" style="width: ${Math.min(100, Math.max(0, wPct))}%"></div>
              </div>
              <span class="font-mono font-bold text-white">${wPct}%</span>
            </div>
          </td>
        `;
        tbody.appendChild(tr);
      });

      // 回测指标栏
      const m = data.metrics || {};
      const metricItems = [
        { label: '累计总收益', val: m.total_return != null ? (m.total_return * 100).toFixed(2) + '%' : '-', color: 'text-emerald-400' },
        { label: '年化收益(CAGR)', val: m.cagr != null ? (m.cagr * 100).toFixed(2) + '%' : '-', color: 'text-emerald-400' },
        { label: '最大回撤', val: m.max_drawdown != null ? (m.max_drawdown * 100).toFixed(2) + '%' : '-', color: 'text-rose-400' },
        { label: '年化波动率', val: m.annual_volatility != null ? (m.annual_volatility * 100).toFixed(2) + '%' : '-', color: 'text-slate-200' },
        { label: '夏普比率(Sharpe)', val: m.sharpe_ratio != null ? Number(m.sharpe_ratio).toFixed(2) : '-', color: 'text-blue-400' },
        { label: '索提诺(Sortino)', val: m.sortino_ratio != null ? Number(m.sortino_ratio).toFixed(2) : '-', color: 'text-blue-400' },
        { label: '卡玛比率(Calmar)', val: m.calmar_ratio != null ? Number(m.calmar_ratio).toFixed(2) : '-', color: 'text-amber-300' },
        { label: '调仓总次数', val: m.rebalance_count != null ? m.rebalance_count + ' 次' : '-', color: 'text-purple-300' },
      ];
      const mGrid = document.getElementById('st-metrics-grid');
      mGrid.innerHTML = metricItems.map(it => `
        <div class="card p-2.5">
          <div class="text-[11px] text-slate-400">${it.label}</div>
          <div class="text-base font-bold font-mono mt-0.5 ${it.color}">${it.val}</div>
        </div>
      `).join('');

      // 渲染净值与回撤图表
      renderStrategyNavChart(data.chart);

      // 渲染调仓日志
      const rBody = document.getElementById('st-rebalance-body');
      rBody.innerHTML = (data.rebalance_log || []).map(r => {
        const wTags = Object.entries(r.weights || {})
          .filter(([_, v]) => v > 0.001)
          .map(([k, v]) => `<span class="inline-block px-1.5 py-0.5 mr-1 mb-0.5 rounded bg-slate-800 font-mono text-[11px] text-blue-300">${k}: ${(v * 100).toFixed(1)}%</span>`)
          .join('');
        return `
          <tr class="hover:bg-slate-800/40">
            <td class="py-1.5 px-2.5 font-mono text-emerald-400">${r.signal_date}</td>
            <td class="py-1.5 px-2.5 font-mono text-blue-400">${r.execution_date}</td>
            <td class="py-1.5 px-2.5 text-slate-300">${r.trigger_reason}</td>
            <td class="py-1.5 px-2.5 text-right font-mono">${((r.turnover || 0) * 100).toFixed(1)}%</td>
            <td class="py-1.5 px-2.5">${wTags || '<span class="text-amber-300">Cash: 100%</span>'}</td>
          </tr>
        `;
      }).join('');
    }

    function renderStrategyNavChart(chartData) {
      const dom = document.getElementById('chart-strategy-nav');
      if (!navChart) navChart = echarts.init(dom);
      navChart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        legend: { data: ['策略净值 (NAV)', '回撤 (%)'], textStyle: { color: '#94a3b8', fontSize: 11 } },
        grid: [
          { left: 55, right: 30, top: 30, height: '55%' },
          { left: 55, right: 30, top: '72%', height: '20%' }
        ],
        xAxis: [
          { type: 'category', data: chartData.dates, gridIndex: 0, axisLabel: { color: '#64748b', fontSize: 10 } },
          { type: 'category', data: chartData.dates, gridIndex: 1, axisLabel: { show: false } }
        ],
        yAxis: [
          { type: 'value', gridIndex: 0, scale: true, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#94a3b8', fontSize: 10 } },
          { type: 'value', gridIndex: 1, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#f43f5e', fontSize: 10, formatter: '{value}%' } }
        ],
        series: [
          {
            name: '策略净值 (NAV)',
            type: 'line',
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: chartData.nav,
            showSymbol: false,
            lineStyle: { width: 2, color: '#3b82f6' },
            areaStyle: { color: 'rgba(59, 130, 246, 0.12)' }
          },
          {
            name: '回撤 (%)',
            type: 'line',
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: chartData.drawdown_pct,
            showSymbol: false,
            lineStyle: { width: 1, color: '#f43f5e' },
            areaStyle: { color: 'rgba(244, 63, 94, 0.2)' }
          }
        ]
      }, true);
    }

    async function loadMarketSymbol(symbol) {
      if (!symbol) return;
      const resp = await fetch('/api/market/' + encodeURIComponent(symbol));
      const data = await resp.json();

      document.getElementById('mkt-badge').textContent =
        `${data.name} (${data.symbol}) | 市场: ${data.market} | 类型: ${data.asset_type} | 交易标的: ${data.trade_etf || '-'}`;

      // 价格 + PE 图
      const pDom = document.getElementById('chart-market-price');
      if (!priceChart) priceChart = echarts.init(pDom);
      const hasPe = (data.pe || []).some(v => v != null);
      const seriesPrice = [
        {
          name: '收盘价 (close)',
          type: 'line',
          yAxisIndex: 0,
          data: data.close,
          showSymbol: false,
          lineStyle: { width: 2, color: '#10b981' }
        }
      ];
      if (hasPe) {
        seriesPrice.push({
          name: 'PE (市盈率)',
          type: 'line',
          yAxisIndex: 1,
          data: data.pe,
          showSymbol: false,
          lineStyle: { width: 1.5, type: 'dashed', color: '#f59e0b' }
        });
      }
      priceChart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#94a3b8', fontSize: 11 } },
        grid: { left: 55, right: 55, top: 35, bottom: 30 },
        xAxis: { type: 'category', data: data.dates, axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: [
          { type: 'value', scale: true, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#10b981', fontSize: 10 } },
          { type: 'value', scale: true, splitLine: { show: false }, axisLabel: { color: '#f59e0b', fontSize: 10 } }
        ],
        series: seriesPrice
      }, true);

      // 因子走势图
      const fDom = document.getElementById('chart-market-factors');
      if (!factorChart) factorChart = echarts.init(fDom);
      const fEntries = Object.entries(data.factors || {});
      factorChart.setOption({
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: '#94a3b8', fontSize: 11 } },
        grid: { left: 50, right: 30, top: 35, bottom: 30 },
        xAxis: { type: 'category', data: data.dates, axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#94a3b8', fontSize: 10 } },
        series: fEntries.map(([fid, vals]) => ({
          name: fid,
          type: 'line',
          data: vals,
          showSymbol: false,
          lineStyle: { width: 1.8 }
        }))
      }, true);

      // 最近 50 条 SQL 行记录
      const rBody = document.getElementById('mkt-recent-body');
      rBody.innerHTML = (data.recent_rows || []).map(r => `
        <tr class="hover:bg-slate-800/40">
          <td class="py-1.5 px-2.5 text-emerald-400">${r.trade_date}</td>
          <td class="py-1.5 px-2.5 text-blue-400">${r.symbol}</td>
          <td class="py-1.5 px-2.5 text-right">${r.open != null ? Number(r.open).toFixed(3) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right">${r.high != null ? Number(r.high).toFixed(3) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right">${r.low != null ? Number(r.low).toFixed(3) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right font-bold text-white">${r.close != null ? Number(r.close).toFixed(3) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right ${r.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${r.change_pct != null ? Number(r.change_pct).toFixed(2) + '%' : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-amber-300">${r.pe != null ? Number(r.pe).toFixed(2) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-amber-200">${r.pb != null ? Number(r.pb).toFixed(2) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-indigo-300">${r.dividend_yield != null ? Number(r.dividend_yield).toFixed(2) + '%' : '-'}</td>
        </tr>
      `).join('');
    }

    async function triggerRunSignals() {
      const btn = document.getElementById('btn-run-signals');
      btn.disabled = true;
      btn.textContent = '⏳ 正在重算信号...';
      try {
        await fetch('/api/run-signals', { method: 'POST' });
        await loadOverview();
      } finally {
        btn.disabled = false;
        btn.textContent = '⚡ 重算全部信号';
      }
    }

    async function triggerBootstrap() {
      const btn = document.getElementById('btn-bootstrap');
      btn.disabled = true;
      btn.textContent = '⏳ 正在拉取行情与计算因子...';
      try {
        await fetch('/api/bootstrap', { method: 'POST' });
        await loadOverview();
      } finally {
        btn.disabled = false;
        btn.textContent = '🔄 同步最新行情/估值';
      }
    }

    window.addEventListener('resize', () => {
      navChart && navChart.resize();
      priceChart && priceChart.resize();
      factorChart && factorChart.resize();
    });

    loadOverview();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    return DASHBOARD_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088)
