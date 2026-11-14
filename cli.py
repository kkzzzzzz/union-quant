#!/usr/bin/env python3
"""
统一量化项目中枢 CLI (Union Quant Unified CLI).

支持子命令：
  1. python cli.py list
     列出所有已注册资产、内置因子及策略配置。

  2. python cli.py sync [--mode daily|catchup|history] [--symbols SYM1,SYM2] [--compute-factors]
     统一拉取指数/债券/黄金/ETF/股债性价比行情与估值，并可选顺带更新因子库。

  3. python cli.py factors [--symbols SYM1,SYM2] [--tail N]
     统一计算并持久化向量化因子到 `factor_daily`。

  4. python cli.py backtest [--strategy STRATEGY_ID|--all] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
     运行 T+1 策略回测并输出绩效汇总。

  5. python cli.py signal [--strategy STRATEGY_ID|--all] [--date YYYY-MM-DD] [--mode live|dry-run] [--no-notify]
     计算每日策略目标仓位信号（严格遵守 T+1 数据物理截断），写入 `strategy_signals` 并推送通知。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from core.config import AppConfig
from core.data_hub.sync import DataSyncService
from core.db import DatabaseGateway
from core.notifier import PushNotifier
from core.quant_engine.factors import FactorService
from core.quant_engine.pipeline import QuantPipeline
from core.quant_engine.strategies import STRATEGY_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("union.cli")


def _parse_symbols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def cmd_list(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    assets = cfg.load_assets()
    strategies = cfg.list_strategies()
    factors = FactorService.list_registered_factors()

    # 自动同步策略元数据到 strategy_meta
    meta_rows = []
    for sid in strategies:
        scfg = cfg.load_strategy_config(sid)
        meta_rows.append({
            "strategy_id": sid,
            "name": scfg.get("name", sid),
            "category": scfg.get("type", "dynamic_allocation"),
            "status": "online",
            "version": str(scfg.get("version", "2.0")),
            "universe": [a["symbol"] for a in scfg.get("strategy_params", {}).get("assets", [])] or scfg.get("rebalance", {}).get("extremum_trigger", {}).get("monitor_symbols", []),
            "default_params": scfg.get("strategy_params") or {},
            "execution_config": scfg.get("execution") or {},
            "description": scfg.get("description", ""),
        })
    db.upsert_strategy_meta(meta_rows)

    print("=" * 72)
    print(f"1. 已配置资产字典 ({len(assets)} 个):")
    for sym, spec in assets.items():
        print(
            f"   - {sym:<14} | {spec.asset_type:<6} | {spec.market:<3} | {spec.name} (交易标的: {spec.trade_etf or '-'})"
        )

    print("\n" + "=" * 72)
    print(f"2. 已注册向量化因子 ({len(factors)} 个):")
    print("   " + ", ".join(factors))

    print("\n" + "=" * 72)
    print(f"3. 已注册策略引擎插件 ({len(STRATEGY_REGISTRY)} 类):")
    print("   " + ", ".join(sorted(STRATEGY_REGISTRY.keys())))

    print("\n" + "=" * 72)
    print(f"4. 策略元数据 (已同步到 strategy_meta 表, {len(strategies)} 个):")
    for sid in strategies:
        scfg = cfg.load_strategy_config(sid)
        print(
            f"   - {sid:<26} | type={scfg.get('type'):<20} | {scfg.get('name')}"
        )
    print("=" * 72)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    sync_svc = DataSyncService(db=db, config=cfg)
    symbols = _parse_symbols(args.symbols)

    if args.mode == "catchup":
        report = sync_svc.sync_valuation_catchup(symbols=symbols)
    elif args.mode == "history":
        report = sync_svc.sync_history(
            symbols=symbols, start_date=args.start or "2014-01-01"
        )
    else:
        report = sync_svc.sync_daily(
            symbols=symbols, lookback_days=args.lookback
        )

    print(
        f"[Sync 完成] 模式={args.mode} | 成功标的={len(report.synced_symbols)} | 写入行数={report.upserted_rows} | 失败={len(report.failed_symbols)}"
    )
    if report.stale_valuations:
        print(f"[估值滞后提示] 共 {len(report.stale_valuations)} 个指数估值日期晚于价格日期:")
        for item in report.stale_valuations:
            print(
                f"   - {item['name']}({item['symbol']}): 价格日={item['price_date']}, 估值日={item['valuation_date']}"
            )

    if args.compute_factors:
        factor_svc = FactorService(db=db, config=cfg)
        f_rows = factor_svc.compute_and_store_all(
            symbols=symbols, tail_n=args.factor_tail
        )
        print(f"[Factor 更新完成] 写入因子行数={f_rows}")

    if args.notify and (report.failed_symbols or report.stale_valuations):
        notifier = PushNotifier()
        lines = [
            f"- 同步模式: `{args.mode}`",
            f"- 成功标的数: `{len(report.synced_symbols)}` (写入 `{report.upserted_rows}` 行)",
        ]
        if report.stale_valuations:
            lines.append(
                f"- 估值待晚间补齐: `{', '.join(x['symbol'] for x in report.stale_valuations)}`"
            )
        if report.failed_symbols:
            lines.append(
                f"- 失败标的: `{json.dumps(report.failed_symbols, ensure_ascii=False)}`"
            )
        notifier.send(f"数据同步报告 ({report.run_date})", "\n".join(lines))

    return 0


def cmd_factors(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    factor_svc = FactorService(db=db, config=cfg)
    symbols = _parse_symbols(args.symbols)
    total = factor_svc.compute_and_store_all(symbols=symbols, tail_n=args.tail)
    print(f"[Factors 完成] 共计算并写入 {total} 条因子记录。")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    pipeline = QuantPipeline(config=cfg)
    target_ids = (
        cfg.list_strategies()
        if args.all or not args.strategy
        else [args.strategy]
    )

    for sid in target_ids:
        res = pipeline.run_backtest(
            strategy_id=sid,
            start_date=args.start,
            end_date=args.end,
            save_outputs=not args.no_save,
        )
        m = res.metrics
        print("-" * 72)
        print(f"策略回测: {res.strategy_name} ({res.strategy_id})")
        if not m:
            print("  [警告] 无足够行情数据生成回测结果，请先运行 `python cli.py sync --mode history`。")
            continue
        print(
            f"  区间: {m.get('start_date')} ~ {m.get('end_date')} ({m.get('trading_days')} 交易日, 调仓 {m.get('rebalance_count')} 次)"
        )
        print(
            f"  总收益: {m.get('total_return', 0.0):.2%} | 年化(CAGR): {m.get('cagr', 0.0):.2%} | 最大回撤: {m.get('max_drawdown', 0.0):.2%}"
        )
        print(
            f"  年化波动: {m.get('annual_volatility', 0.0):.2%} | 夏普(Sharpe): {m.get('sharpe_ratio', 0.0):.2f} | 卡玛(Calmar): {m.get('calmar_ratio', 0.0):.2f}"
        )
        if res.latest_decision:
            w_str = ", ".join(
                f"{k}:{v:.1%}"
                for k, v in res.latest_decision.weights.items()
                if v > 0.0001
            )
            print(
                f"  最新期末信号 ({res.latest_decision.signal_date}): {w_str or '空仓(Cash 100%)'}"
            )
    print("-" * 72)
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    pipeline = QuantPipeline(config=cfg)
    target_ids = (
        cfg.list_strategies()
        if args.all or not args.strategy
        else [args.strategy]
    )

    for sid in target_ids:
        payload = pipeline.run_daily_signal(
            strategy_id=sid,
            run_date=args.date,
            run_mode=args.mode,
            notify=not args.no_notify,
        )
        print("=" * 72)
        print(payload["markdown_report"])
    print("=" * 72)
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from core.data_hub.bootstrap import bootstrap_verification_data

    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    res = bootstrap_verification_data(
        config=cfg, db=db, try_live_fetch=not args.no_live
    )
    ov = res["db_overview"]
    print("=" * 72)
    print(f"[Bootstrap 完成] 关系库地址: {ov['db_url']}")
    print(
        f"  - 实时行情/估值成功拉取标的 ({len(res['live_fetched_symbols'])} 个): {', '.join(res['live_fetched_symbols']) or '使用锚定模拟'}"
    )
    print(f"  - asset_meta 表记录数      : {ov['counts']['asset_meta']}")
    print(f"  - market_daily 表记录数    : {ov['counts']['market_daily']}")
    print(f"  - factor_daily 表记录数    : {ov['counts']['factor_daily']}")
    print(f"  - strategy_signals 表记录数: {ov['counts']['strategy_signals']}")
    print("=" * 72)
    return 0


def cmd_migrate_supabase(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    stats = db.migrate_to_supabase()
    print("[Supabase 迁移完成] 同步行数统计:", json.dumps(stats, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from web_dashboard import app

    print(f"启动可视化核验看板: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_export_pages(args: argparse.Namespace) -> int:
    from pathlib import Path
    from core.pages_builder import build_static_pages

    out_path = build_static_pages(out_dir=Path(args.out_dir))
    print(f"[Vercel 静态看板站点生成完成] 输出目录: {out_path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="union-quant",
        description="一个项目 · 一个库表 · 一个拉数据 · 一个算策略 统一命令行入口",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="列出资产、因子与策略清单")
    p_list.set_defaults(func=cmd_list)

    # sync
    p_sync = sub.add_parser("sync", help="统一拉取行情与估值数据")
    p_sync.add_argument(
        "--mode",
        choices=["daily", "catchup", "history"],
        default="daily",
        help="同步模式: daily=日常增量, catchup=晚间估值补齐, history=全量历史初始化",
    )
    p_sync.add_argument("--symbols", help="逗号分隔的标的列表，默认全部")
    p_sync.add_argument("--start", default="2014-01-01", help="history 模式的起始日期")
    p_sync.add_argument("--lookback", type=int, default=15, help="daily 模式回溯天数")
    p_sync.add_argument(
        "--compute-factors",
        action="store_true",
        help="同步完成后顺带计算并更新因子表",
    )
    p_sync.add_argument(
        "--factor-tail", type=int, default=60, help="增量写入最近 N 条因子"
    )
    p_sync.add_argument("--notify", action="store_true", help="发送同步报告通知")
    p_sync.set_defaults(func=cmd_sync)

    # factors
    p_fac = sub.add_parser("factors", help="计算并持久化向量化因子")
    p_fac.add_argument("--symbols", help="逗号分隔的标的列表，默认全部")
    p_fac.add_argument("--tail", type=int, default=None, help="仅写入最近 N 条记录")
    p_fac.set_defaults(func=cmd_factors)

    # backtest
    p_bt = sub.add_parser("backtest", help="运行策略历史回测")
    p_bt.add_argument("--strategy", "-s", help="策略 ID 或 YAML 路径")
    p_bt.add_argument("--all", action="store_true", help="回测全部已配置策略")
    p_bt.add_argument("--start", help="回测起始日期 YYYY-MM-DD")
    p_bt.add_argument("--end", help="回测结束日期 YYYY-MM-DD")
    p_bt.add_argument("--no-save", action="store_true", help="不写入 outputs/ 目录")
    p_bt.set_defaults(func=cmd_backtest)

    # signal
    p_sig = sub.add_parser("signal", help="生成每日策略信号并入库/推送")
    p_sig.add_argument("--strategy", "-s", help="策略 ID 或 YAML 路径")
    p_sig.add_argument("--all", action="store_true", help="计算全部策略信号")
    p_sig.add_argument(
        "--date",
        help="运行日期 YYYY-MM-DD (默认今天；常规策略自动截断 <= date-1，周五盘后策略截断 <= date)",
    )
    p_sig.add_argument(
        "--mode",
        choices=["live", "dry-run", "prod", "sandbox", "backtest"],
        default="live",
        help="运行模式: live/prod=写入生产信号, dry-run/sandbox=仅沙箱验算, backtest=线下回测",
    )
    p_sig.add_argument("--no-notify", action="store_true", help="禁用消息推送")
    p_sig.set_defaults(func=cmd_signal)

    # bootstrap
    p_boot = sub.add_parser(
        "bootstrap", help="初始化数据库、拉取实时行情估值、补齐历史并计算全部因子与策略信号"
    )
    p_boot.add_argument("--no-live", action="store_true", help="跳过外部实时网络请求，仅生成锚定历史")
    p_boot.set_defaults(func=cmd_bootstrap)

    # migrate-supabase
    p_mig = sub.add_parser(
        "migrate-supabase", help="将外部关系库中的四张核心表全量迁移至远端 Supabase"
    )
    p_mig.set_defaults(func=cmd_migrate_supabase)

    # serve
    p_srv = sub.add_parser("serve", help="启动 Web 可视化核验看板")
    p_srv.add_argument("--host", default="127.0.0.1", help="监听地址")
    p_srv.set_defaults(func=cmd_serve)

    # export-pages
    p_exp = sub.add_parser(
        "export-pages",
        aliases=["export-site"],
        help="生成纯静态 Vercel 每日信号与行情看板 (site/)",
    )
    p_exp.add_argument("--out-dir", default="site", help="静态站点输出目录")
    p_exp.set_defaults(func=cmd_export_pages)

    # compare (新老系统双跑比对稽核)
    p_cmp = sub.add_parser("compare", aliases=["reconcile"], help="新老系统影子双跑并排比对与稽核报告")
    p_cmp.add_argument("--date", help="指定比对日期 YYYY-MM-DD (默认最新)")
    p_cmp.add_argument("--tolerance", type=float, default=0.001, help="允许的权重偏差容差 (默认 0.001 即 0.1%)")
    p_cmp.add_argument("--notify", action="store_true", help="发送比对报告通知至已配置的钉钉/飞书/微信等通道")
    p_cmp.set_defaults(func=cmd_compare)

    # shadow-pipeline (一键执行全套影子流程：拉数据 -> 算因子 -> 算影子策略 -> 推送 Diff 结果)
    p_pipe = sub.add_parser("shadow-pipeline", help="全套自动化双跑：自动拉行情估值、算因子、算影子策略并推送 Diff 报告")
    p_pipe.add_argument("--mode", choices=["daily", "catchup"], default="daily", help="数据同步模式: daily=日常增量, catchup=晚间估值追赶")
    p_pipe.add_argument("--no-notify", action="store_true", help="禁用消息通知推送")
    p_pipe.add_argument(
        "--no-sync-live",
        action="store_true",
        help="禁用将最新影子计算结果自动同步至线上生产表",
    )
    p_pipe.set_defaults(func=cmd_shadow_pipeline)

    # strategy (策略表管理、离线回测验证与一键推线上/下线)
    p_strat = sub.add_parser("strategy", help="策略表管理：离线验证、回测核验、推线上/下线与元数据同步")
    strat_sub = p_strat.add_subparsers(dest="strat_action", required=True)

    # strategy list
    p_s_list = strat_sub.add_parser("list", help="查看策略表全量元数据、上下线状态与指标")
    p_s_list.set_defaults(func=cmd_strategy_list)

    # strategy verify
    p_s_ver = strat_sub.add_parser("verify", help="离线回测验证策略（Lab 沙箱验证模式，不污染生产）")
    p_s_ver.add_argument("strategy_id", help="策略 ID")
    p_s_ver.add_argument("--start", help="回测起始日期 YYYY-MM-DD")
    p_s_ver.add_argument("--end", help="回测结束日期 YYYY-MM-DD")
    p_s_ver.set_defaults(func=cmd_strategy_verify)

    # strategy promote
    p_s_pro = strat_sub.add_parser("promote", help="推线上：将离线验证通过的策略晋升为 online 线上状态")
    p_s_pro.add_argument("strategy_id", help="策略 ID")
    p_s_pro.set_defaults(func=cmd_strategy_promote)

    # strategy demote
    p_s_dem = strat_sub.add_parser("demote", help="下线：将策略设为 offline 离线状态")
    p_s_dem.add_argument("strategy_id", help="策略 ID")
    p_s_dem.set_defaults(func=cmd_strategy_demote)

    # strategy sync
    p_s_sync = strat_sub.add_parser("sync", help="同步本地策略配置到 strategy_meta 策略表")
    p_s_sync.set_defaults(func=cmd_strategy_sync)

    return parser


def cmd_compare(args: argparse.Namespace) -> int:
    from core.reconcile import ReconciliationService

    svc = ReconciliationService()
    summary = svc.compare(
        target_date=args.date,
        tolerance=args.tolerance,
        notify=args.notify,
    )
    print("=" * 72)
    print(summary["markdown_report"])
    print("=" * 72)
    print(f"[比对报告已导出至] {summary['report_path']}")
    return 0 if summary["is_all_clean"] else 0


def cmd_shadow_pipeline(args: argparse.Namespace) -> int:
    from core.reconcile import ReconciliationService

    svc = ReconciliationService()
    sync_live = not getattr(args, "no_sync_live", False)
    summary = svc.run_daily_shadow_pipeline(
        sync_mode=args.mode,
        compute_factors=True,
        notify=not args.no_notify,
        sync_live=sync_live,
    )
    print("=" * 72)
    print(summary["markdown_report"])
    print("=" * 72)
    print(f"[双跑自动化流程执行完成，Diff 报告已导出至] {summary['report_path']}")
    if sync_live:
        print("[生产同步] 已自动将最新影子信号同步更新至线上生产表 (run_mode=live)")
    return 0 if summary["is_all_clean"] else 0




def _sync_strategy_meta_if_needed(cfg: AppConfig, db: DatabaseGateway) -> List[Dict[str, Any]]:
    metas = db.fetch_strategy_meta()
    if not metas:
        records = []
        for sid in cfg.list_strategies(only_online=False):
            try:
                scfg = cfg.load_strategy_config(sid)
                records.append({
                    "strategy_id": sid,
                    "name": scfg.get("name", sid),
                    "category": scfg.get("type", "dynamic_allocation"),
                    "status": scfg.get("status", "online"),
                    "version": scfg.get("version", "2.0"),
                    "universe": scfg.get("strategy_params", {}).get("assets", []),
                    "default_params": scfg.get("strategy_params") or {},
                    "execution_config": scfg.get("execution") or {},
                    "description": scfg.get("description", ""),
                })
            except Exception:
                pass
        if records:
            db.upsert_strategy_meta(records)
            metas = db.fetch_strategy_meta()
    return metas


def cmd_strategy_list(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    metas = _sync_strategy_meta_if_needed(cfg, db)

    print("=" * 88)
    print(f"{'策略 ID':<25} | {'策略名称':<28} | {'状态':<10} | {'CAGR':<8} | {'MaxDD':<8} | {'Sharpe':<6}")
    print("-" * 88)
    for m in metas:
        sid = m["strategy_id"]
        name = m.get("name", sid)[:26]
        status = m.get("status", "online").upper()
        status_str = f"🟢 {status}" if status == "ONLINE" else (f"⚪ {status}" if status == "OFFLINE" else f"🟡 {status}")
        raw_m = m.get("metrics_snapshot")
        metrics = raw_m if isinstance(raw_m, dict) else {}
        cagr_str = f"{metrics['cagr']:.1%}" if "cagr" in metrics else "-"
        mdd_str = f"{metrics['max_drawdown']:.1%}" if "max_drawdown" in metrics else "-"
        sr_str = f"{metrics['sharpe_ratio']:.2f}" if "sharpe_ratio" in metrics else "-"
        print(f"{sid:<25} | {name:<26} | {status_str:<12} | {cagr_str:<8} | {mdd_str:<8} | {sr_str:<6}")
    print("=" * 88)
    print("💡 提示:")
    print("  • 离线回测验证: python cli.py strategy verify <strategy_id>")
    print("  • 验证通过推线上: python cli.py strategy promote <strategy_id>")
    print("  • 策略停跑下线:   python cli.py strategy demote <strategy_id>")
    return 0


def cmd_strategy_verify(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    pipeline = QuantPipeline(config=cfg, db=db)
    sid = args.strategy_id

    print(f"正在对策略 [{sid}] 执行离线沙箱回测验证 (Lab 模式，零写入线上生产表)...")
    res = pipeline.run_backtest(
        strategy_id=sid,
        start_date=args.start,
        end_date=args.end,
        save_outputs=True,
    )
    m = res.metrics
    print("=" * 72)
    print(f"【策略离线验证结果】策略: {res.strategy_name} ({sid})")
    print(f"  回测区间: {m.get('start_date')} ~ {m.get('end_date')} (共 {m.get('trading_days', 0)} 交易日)")
    print(f"  总收益率: {m.get('total_return', 0.0):.2%} | 年化收益 (CAGR): {m.get('cagr', 0.0):.2%}")
    print(f"  最大回撤: {m.get('max_drawdown', 0.0):.2%} | 年化波动率: {m.get('annual_volatility', 0.0):.2%}")
    print(f"  夏普比率: {m.get('sharpe_ratio', 0.0):.2f} | 卡玛比率: {m.get('calmar_ratio', 0.0):.2f} | 胜率: {m.get('daily_win_rate', 0.0):.2%}")
    if res.latest_decision:
        w_str = ", ".join(f"{k}:{v:.1%}" for k, v in res.latest_decision.weights.items() if v > 0.0001)
        print(f"  最新决策持仓: {w_str or '现金/空仓'}")
    print("=" * 72)

    # 将验证指标缓存至策略元数据表
    db.update_strategy_metrics(sid, m)
    print(f"✅ 策略 [{sid}] 离线回测验证完成！绩效指标已入库留痕。")
    print(f"👉 若指标符合预期，可执行以下命令一键推线上:")
    print(f"   python cli.py strategy promote {sid}")
    return 0


def cmd_strategy_promote(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    sid = args.strategy_id

    # 1. 更新数据库表 strategy_meta
    db.update_strategy_status(sid, "online")
    # 2. 同步更新本地 YAML
    cfg.update_strategy_file_status(sid, "online")

    print("=" * 72)
    print(f"🚀 [推线上成功] 策略 [{sid}] 状态已晋升为: ONLINE (线上生产)")
    print("  • 策略元数据表 strategy_meta 已更新为 online")
    print("  • 日常自动化双跑 (shadow-pipeline) 与实盘调度已自动纳入该策略")
    print("=" * 72)
    return 0


def cmd_strategy_demote(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    sid = args.strategy_id

    # 1. 更新数据库表 strategy_meta
    db.update_strategy_status(sid, "offline")
    # 2. 同步更新本地 YAML
    cfg.update_strategy_file_status(sid, "offline")

    print("=" * 72)
    print(f"⏸ [下线成功] 策略 [{sid}] 状态已变更为: OFFLINE (已下线)")
    print("  • 策略元数据表 strategy_meta 已更新为 offline")
    print("  • 该策略已从日常调度与双跑比对中剔除")
    print("=" * 72)
    return 0


def cmd_strategy_sync(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    db = DatabaseGateway(config=cfg)
    records = []
    for sid in cfg.list_strategies(only_online=False):
        try:
            scfg = cfg.load_strategy_config(sid)
            records.append({
                "strategy_id": sid,
                "name": scfg.get("name", sid),
                "category": scfg.get("type", "dynamic_allocation"),
                "status": scfg.get("status", "online"),
                "version": scfg.get("version", "2.0"),
                "universe": scfg.get("strategy_params", {}).get("assets", []),
                "default_params": scfg.get("strategy_params") or {},
                "execution_config": scfg.get("execution") or {},
                "description": scfg.get("description", ""),
            })
        except Exception as exc:
            print(f"跳过异常配置 {sid}: {exc}")
    n = db.upsert_strategy_meta(records)
    print(f"✅ [策略元数据同步完成] 共同步 {n} 个策略到 strategy_meta 表")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
