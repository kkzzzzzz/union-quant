"""
GitHub Pages 静态每日信号与行情看板生成器 (Static GitHub Pages Builder).

调用方式：
  python cli.py export-pages --out-dir site

生成内容：
- `site/data.json`: 包含全部 6 个策略的最新每日信号、单资产因子诊断明细、历史净值与回撤序列、调仓记录，
  以及全部 24 个标的的最新行情、估值与近期因子走势。
- `site/index.html`: 纯静态、自适应手机与电脑端的每日信号看板（无需后端服务器即可在 GitHub Pages 直接打开）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import ROOT_DIR, AppConfig
from core.db import DatabaseGateway
from core.quant_engine.pipeline import QuantPipeline


def build_static_pages(
    out_dir: Path = ROOT_DIR / "site",
    config: Optional[AppConfig] = None,
    db: Optional[DatabaseGateway] = None,
    market_tail: int = 250,
) -> Path:
    cfg = config or AppConfig()
    gateway = db or DatabaseGateway(config=cfg)
    pipeline = QuantPipeline(config=cfg, db=gateway)

    out_dir.mkdir(parents=True, exist_ok=True)
    overview = gateway.get_database_overview()
    signals_map = {
        s["strategy_id"]: s
        for s in gateway.fetch_all_strategy_signals(run_mode="live")
    }

    strategies_bundle: Dict[str, Any] = {}
    for sid in cfg.list_strategies():
        bt = pipeline.run_backtest(strategy_id=sid, save_outputs=False)
        sig = signals_map.get(sid)
        if sig is None:
            sig = pipeline.run_daily_signal(
                strategy_id=sid, run_mode="live", notify=False
            )
        eq = bt.equity_curve
        strategies_bundle[sid] = {
            "strategy_id": bt.strategy_id,
            "strategy_name": bt.strategy_name,
            "metrics": bt.metrics,
            "signal": DatabaseGateway._clean_records_for_json([sig])[0],
            "chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in eq.index]
                if not eq.empty
                else [],
                "nav": [round(float(v), 4) for v in eq["nav"].values]
                if not eq.empty
                else [],
                "drawdown_pct": [
                    round(float(v * 100.0), 2) for v in eq["drawdown"].values
                ]
                if not eq.empty
                else [],
            },
            "rebalance_log": list(reversed(bt.rebalance_log[-50:])),
        }

    markets_bundle: Dict[str, Any] = {}
    for sym, spec in cfg.load_assets().items():
        df_mkt = gateway.fetch_market_daily(sym)
        if df_mkt.empty:
            continue
        df_sub = df_mkt.tail(market_tail).copy()
        factors_data: Dict[str, List[Optional[float]]] = {}
        for fid in spec.factors:
            s = gateway.fetch_factor_series(sym, fid)
            if not s.empty:
                s_aligned = s.reindex(pd.to_datetime(df_sub["trade_date"]))
                factors_data[fid] = [
                    None if pd.isna(v) else round(float(v), 3)
                    for v in s_aligned.values
                ]

        recent_df = df_sub.tail(30).iloc[::-1].copy()
        markets_bundle[sym] = {
            "symbol": sym,
            "name": spec.name,
            "asset_type": spec.asset_type,
            "market": spec.market,
            "trade_etf": spec.trade_etf,
            "dates": df_sub["trade_date"].tolist(),
            "close": [
                None if pd.isna(v) else round(float(v), 4)
                for v in df_sub["close"]
            ],
            "pe": [
                None if pd.isna(v) else round(float(v), 3)
                for v in (
                    df_sub["pe"]
                    if "pe" in df_sub.columns
                    else [None] * len(df_sub)
                )
            ],
            "pb": [
                None if pd.isna(v) else round(float(v), 3)
                for v in (
                    df_sub["pb"]
                    if "pb" in df_sub.columns
                    else [None] * len(df_sub)
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
            "recent_rows": DatabaseGateway._clean_records_for_json(
                recent_df.to_dict(orient="records")
            ),
        }

    full_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": overview["counts"],
        "asset_meta": DatabaseGateway._clean_records_for_json(
            overview["asset_meta"]
        ),
        "market_summary": DatabaseGateway._clean_records_for_json(
            overview["market_summary"]
        ),
        "registered_strategies": cfg.list_strategies(),
        "strategies": strategies_bundle,
        "markets": markets_bundle,
    }

    data_json_path = out_dir / "data.json"
    data_json_path.write_text(
        json.dumps(full_payload, ensure_ascii=False), encoding="utf-8"
    )

    index_html_path = out_dir / "index.html"
    index_html_path.write_text(STATIC_PAGES_HTML, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return out_dir


STATIC_PAGES_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Union Quant — 每日策略信号与估值看板</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif; background: #0f172a; color: #e2e8f0; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; }
    .tab-btn.active { background: #2563eb; color: #fff; border-color: #3b82f6; }
    .strat-btn.active { background: #1d4ed8; border-color: #60a5fa; color: #fff; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- 顶部导航 -->
  <header class="border-b border-slate-800 bg-slate-900/95 sticky top-0 z-30 backdrop-blur">
    <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-3">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-lg shadow">UQ</div>
        <div>
          <h1 class="text-base font-bold text-white flex items-center gap-2">
            Union Quant 每日量化信号看板
            <span class="text-xs font-normal px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">GitHub Pages 自动更新</span>
          </h1>
          <p id="gen-time-banner" class="text-xs text-slate-400 font-mono">正在加载最新快照...</p>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <button onclick="switchTab('strategies')" id="tab-strategies" class="tab-btn active px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 transition">
          1. 每日策略信号
        </button>
        <button onclick="switchTab('market')" id="tab-market" class="tab-btn px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition">
          2. 指数估值与因子
        </button>
        <button onclick="switchTab('overview')" id="tab-overview" class="tab-btn px-3.5 py-1.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-300 hover:bg-slate-800 transition">
          3. 全部标的快照
        </button>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto w-full px-4 py-4 flex-1 space-y-4">
    <!-- ==================== TAB 1: 每日策略信号 ==================== -->
    <div id="view-strategies" class="space-y-4">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div class="lg:col-span-3 card p-3 space-y-2">
          <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1 mb-2">选择策略 (6个)</div>
          <div id="strategy-list" class="space-y-1.5"></div>
        </div>

        <div class="lg:col-span-9 space-y-4">
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

            <div class="mt-3">
              <div class="text-xs font-semibold text-slate-400 mb-2">目标仓位与单资产因子诊断明细</div>
              <div class="overflow-x-auto">
                <table class="w-full text-xs text-left border-collapse">
                  <thead>
                    <tr class="border-b border-slate-700 text-slate-400 bg-slate-800/50">
                      <th class="py-2 px-2.5">标的代码</th>
                      <th class="py-2 px-2.5">名称</th>
                      <th class="py-2 px-2.5">交易ETF</th>
                      <th class="py-2 px-2.5">角色/公式</th>
                      <th class="py-2 px-2.5">数据日 (T)</th>
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

          <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5" id="st-metrics-grid"></div>

          <div class="card p-4">
            <div class="text-xs font-semibold text-slate-300 mb-2">T+1 历史回测净值与水下回撤曲线</div>
            <div id="chart-strategy-nav" style="width: 100%; height: 340px;"></div>
          </div>

          <div class="card p-4">
            <div class="text-xs font-semibold text-slate-300 mb-2">最近历史调仓记录 (T 日信号 -> T+1 日执行)</div>
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

    <!-- ==================== TAB 2: 指数估值与因子 ==================== -->
    <div id="view-market" class="hidden space-y-4">
      <div class="card p-3.5 flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <label class="text-xs font-semibold text-slate-300">选择标的:</label>
          <select id="select-symbol" onchange="renderMarketSymbol(this.value)" class="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"></select>
          <span id="mkt-badge" class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono"></span>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-300 mb-2">近 250 交易日收盘价与 PE 估值走势</div>
          <div id="chart-market-price" style="width: 100%; height: 320px;"></div>
        </div>
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-300 mb-2">向量化因子走势</div>
          <div id="chart-market-factors" style="width: 100%; height: 320px;"></div>
        </div>
      </div>

      <div class="card p-4">
        <div class="text-xs font-semibold text-slate-300 mb-2">最近 30 交易日明细</div>
        <div class="overflow-x-auto max-h-80">
          <table class="w-full text-xs text-left border-collapse">
            <thead class="sticky top-0 bg-slate-800 text-slate-400">
              <tr>
                <th class="py-2 px-2.5">日期</th>
                <th class="py-2 px-2.5">代码</th>
                <th class="py-2 px-2.5 text-right">收盘价</th>
                <th class="py-2 px-2.5 text-right">涨跌幅%</th>
                <th class="py-2 px-2.5 text-right">PE</th>
                <th class="py-2 px-2.5 text-right">PB</th>
                <th class="py-2 px-2.5 text-right">股息率%</th>
              </tr>
            </thead>
            <tbody id="mkt-recent-body" class="divide-y divide-slate-800 font-mono"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 3: 全部标的快照 ==================== -->
    <div id="view-overview" class="hidden space-y-4">
      <div class="card p-4">
        <div class="text-xs font-semibold text-slate-300 mb-2">24 个监控标的的最新行情与估值总览（点击任意行查看走势图）</div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left border-collapse">
            <thead class="bg-slate-800 text-slate-400">
              <tr>
                <th class="py-2 px-2.5">代码</th>
                <th class="py-2 px-2.5">名称</th>
                <th class="py-2 px-2.5">类型</th>
                <th class="py-2 px-2.5">市场</th>
                <th class="py-2 px-2.5">交易ETF</th>
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
    let bundle = null;
    let currentStrategyId = null;
    let navChart = null;
    let priceChart = null;
    let factorChart = null;

    function switchTab(tab) {
      ['strategies', 'market', 'overview'].forEach(t => {
        document.getElementById('view-' + t).classList.toggle('hidden', t !== tab);
        document.getElementById('tab-' + t).classList.toggle('active', t === tab);
      });
      if (tab === 'strategies' && navChart) setTimeout(() => navChart.resize(), 50);
      if (tab === 'market') {
        const sym = document.getElementById('select-symbol').value;
        renderMarketSymbol(sym);
      }
    }

    async function init() {
      const resp = await fetch('./data.json?t=' + Date.now());
      bundle = await resp.json();
      document.getElementById('gen-time-banner').textContent =
        `最后更新: ${bundle.generated_at} | 监控标的: ${bundle.counts.asset_meta} | 策略数: ${bundle.registered_strategies.length}`;

      const listEl = document.getElementById('strategy-list');
      listEl.innerHTML = '';
      bundle.registered_strategies.forEach((sid, idx) => {
        const item = bundle.strategies[sid];
        const sig = item ? item.signal : {};
        const btn = document.createElement('button');
        btn.id = 'strat-btn-' + sid;
        btn.className = 'strat-btn w-full text-left p-2.5 rounded-lg border border-slate-700/80 hover:bg-slate-800 transition flex flex-col gap-1';
        btn.onclick = () => selectStrategy(sid);
        btn.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-semibold text-xs text-white">${item ? item.strategy_name : sid}</span>
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">${sig.signal_date || '-'}</span>
          </div>
          <div class="text-[11px] font-mono text-slate-400">${sid}</div>
        `;
        listEl.appendChild(btn);
        if (idx === 0) currentStrategyId = sid;
      });

      const sel = document.getElementById('select-symbol');
      sel.innerHTML = '';
      (bundle.asset_meta || []).forEach(a => {
        const opt = document.createElement('option');
        opt.value = a.symbol;
        opt.textContent = `[${a.asset_type.toUpperCase()}] ${a.symbol} - ${a.name}`;
        sel.appendChild(opt);
      });

      const tbody = document.getElementById('db-summary-body');
      tbody.innerHTML = '';
      (bundle.asset_meta || []).forEach(a => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/60 cursor-pointer';
        tr.onclick = () => {
          document.getElementById('select-symbol').value = a.symbol;
          switchTab('market');
        };
        tr.innerHTML = `
          <td class="py-2 px-2.5 font-mono font-semibold text-blue-400">${a.symbol}</td>
          <td class="py-2 px-2.5 text-white">${a.name}</td>
          <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300">${a.asset_type}</span></td>
          <td class="py-2 px-2.5 font-mono">${a.market}</td>
          <td class="py-2 px-2.5 font-mono text-slate-400">${a.trade_etf || '-'}</td>
          <td class="py-2 px-2.5 font-mono text-emerald-400">${a.latest_date || '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono text-white">${a.latest_price != null ? Number(a.latest_price).toFixed(3) : '-'}</td>
          <td class="py-2 px-2.5 text-right font-mono text-amber-300">${a.latest_pe != null ? Number(a.latest_pe).toFixed(2) : '-'}</td>
        `;
        tbody.appendChild(tr);
      });

      if (currentStrategyId) selectStrategy(currentStrategyId);
    }

    function selectStrategy(sid) {
      currentStrategyId = sid;
      document.querySelectorAll('.strat-btn').forEach(b => b.classList.remove('active'));
      const activeBtn = document.getElementById('strat-btn-' + sid);
      if (activeBtn) activeBtn.classList.add('active');

      const data = bundle.strategies[sid];
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

      const tbody = document.getElementById('st-asset-details-body');
      tbody.innerHTML = '';
      (sig.asset_details || []).forEach(item => {
        const wPct = ((item.weight || 0) * 100).toFixed(1);
        let factorDesc = '-';
        if (item.factor_id && item.factor_value != null) {
          factorDesc = `${item.factor_id}: <span class="text-amber-300 font-mono">${item.factor_value}%</span>`;
        } else if (item.momentum_pct != null) {
          const color = item.momentum_pct >= 0 ? 'text-emerald-400' : 'text-rose-400';
          factorDesc = `动量: <span class="${color} font-mono">${item.momentum_pct}%</span> ${item.passed_filter === false ? '(未过均线)' : ''}`;
        } else if (item.sbe_quantile != null) {
          factorDesc = `SBE分位: <span class="text-amber-300 font-mono">${item.sbe_quantile}%</span>`;
        }

        let roleDesc = item.role ? item.role.toUpperCase() : (item.rank ? `Rank #${item.rank}` : '-');
        if (item.weight_strategy && item.base_weight) {
          roleDesc = `${item.weight_strategy}(base=${item.base_weight}${item.c1 ? ',c1=' + item.c1 : ''})`;
        }
        const peOrMa = item.pe != null ? `PE=${Number(item.pe).toFixed(2)}` : (item.ma_value != null ? `MA=${Number(item.ma_value).toFixed(3)}` : '-');

        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/40';
        tr.innerHTML = `
          <td class="py-2 px-2.5 font-mono font-semibold text-blue-400">${item.symbol}</td>
          <td class="py-2 px-2.5 text-white">${item.name || item.symbol}</td>
          <td class="py-2 px-2.5 font-mono text-slate-400">${item.trade_symbol || '-'}</td>
          <td class="py-2 px-2.5"><span class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">${roleDesc}</span></td>
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
      document.getElementById('st-metrics-grid').innerHTML = metricItems.map(it => `
        <div class="card p-2.5">
          <div class="text-[11px] text-slate-400">${it.label}</div>
          <div class="text-base font-bold font-mono mt-0.5 ${it.color}">${it.val}</div>
        </div>
      `).join('');

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
          { type: 'category', data: data.chart.dates, gridIndex: 0, axisLabel: { color: '#64748b', fontSize: 10 } },
          { type: 'category', data: data.chart.dates, gridIndex: 1, axisLabel: { show: false } }
        ],
        yAxis: [
          { type: 'value', gridIndex: 0, scale: true, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#94a3b8', fontSize: 10 } },
          { type: 'value', gridIndex: 1, splitLine: { lineStyle: { color: '#1e293b' } }, axisLabel: { color: '#f43f5e', fontSize: 10, formatter: '{value}%' } }
        ],
        series: [
          { name: '策略净值 (NAV)', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: data.chart.nav, showSymbol: false, lineStyle: { width: 2, color: '#3b82f6' }, areaStyle: { color: 'rgba(59, 130, 246, 0.12)' } },
          { name: '回撤 (%)', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: data.chart.drawdown_pct, showSymbol: false, lineStyle: { width: 1, color: '#f43f5e' }, areaStyle: { color: 'rgba(244, 63, 94, 0.2)' } }
        ]
      }, true);

      document.getElementById('st-rebalance-body').innerHTML = (data.rebalance_log || []).map(r => {
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

    function renderMarketSymbol(symbol) {
      const data = bundle.markets[symbol];
      if (!data) return;
      document.getElementById('mkt-badge').textContent =
        `${data.name} (${data.symbol}) | 市场: ${data.market} | 交易标的: ${data.trade_etf || '-'}`;

      const pDom = document.getElementById('chart-market-price');
      if (!priceChart) priceChart = echarts.init(pDom);
      const hasPe = (data.pe || []).some(v => v != null);
      const seriesPrice = [
        { name: '收盘价 (close)', type: 'line', yAxisIndex: 0, data: data.close, showSymbol: false, lineStyle: { width: 2, color: '#10b981' } }
      ];
      if (hasPe) {
        seriesPrice.push({ name: 'PE (市盈率)', type: 'line', yAxisIndex: 1, data: data.pe, showSymbol: false, lineStyle: { width: 1.5, type: 'dashed', color: '#f59e0b' } });
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
        series: fEntries.map(([fid, vals]) => ({ name: fid, type: 'line', data: vals, showSymbol: false, lineStyle: { width: 1.8 } }))
      }, true);

      document.getElementById('mkt-recent-body').innerHTML = (data.recent_rows || []).map(r => `
        <tr class="hover:bg-slate-800/40">
          <td class="py-1.5 px-2.5 text-emerald-400">${r.trade_date}</td>
          <td class="py-1.5 px-2.5 text-blue-400">${r.symbol}</td>
          <td class="py-1.5 px-2.5 text-right font-bold text-white">${r.close != null ? Number(r.close).toFixed(3) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right ${r.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${r.change_pct != null ? Number(r.change_pct).toFixed(2) + '%' : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-amber-300">${r.pe != null ? Number(r.pe).toFixed(2) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-amber-200">${r.pb != null ? Number(r.pb).toFixed(2) : '-'}</td>
          <td class="py-1.5 px-2.5 text-right text-indigo-300">${r.dividend_yield != null ? Number(r.dividend_yield).toFixed(2) + '%' : '-'}</td>
        </tr>
      `).join('');
    }

    window.addEventListener('resize', () => {
      navChart && navChart.resize();
      priceChart && priceChart.resize();
      factorChart && factorChart.resize();
    });

    init();
  </script>
</body>
</html>
"""
