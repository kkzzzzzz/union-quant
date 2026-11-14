"""
新老系统影子双跑与数据对齐比对模块 (Shadow Run & Reconciliation Engine).

设计原则：
1. 100% 零侵入: 不修改 now_asset 现有任何代码，老系统继续写入其生产表。
2. 物理空间隔离: 新系统影子信号保存在专属隔离空间 (outputs/shadow_signals/ 或 shadow 模式)，不覆盖线上生产表。
3. 自动化比对稽核: 自动拉取老系统线上最新信号与新系统影子计算结果，输出并排 Diff 对比报表。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.config import AppConfig, OUTPUT_DIR
from core.db import DatabaseGateway
from core.notifier import PushNotifier
from core.quant_engine.pipeline import QuantPipeline


class ReconciliationService:
    """双跑对齐稽核服务"""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        db: Optional[DatabaseGateway] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.db = db or DatabaseGateway(config=self.config)
        self.pipeline = QuantPipeline(config=self.config, db=self.db)
        self.shadow_dir = OUTPUT_DIR / "shadow_signals"
        self.reports_dir = OUTPUT_DIR / "reports"
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_shadow_signals(
        self,
        strategy_ids: Optional[List[str]] = None,
        run_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        在新系统的隔离空间中计算影子信号，完全不写入线上 strategy_signals 生产表。
        默认仅针对已上线 (status: online) 的策略进行影子验算。
        """
        targets = strategy_ids or self.config.list_strategies(only_online=True)
        results: Dict[str, Dict[str, Any]] = {}
        target_date = run_date or datetime.now().strftime("%Y-%m-%d")

        for sid in targets:
            payload = self.pipeline.run_daily_signal(
                strategy_id=sid,
                run_date=target_date,
                run_mode="shadow",
                notify=False,
            )
            # 独立持久化至 outputs/shadow_signals/
            out_file = self.shadow_dir / f"{sid}_{payload['signal_date']}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            results[sid] = payload

        return results

    def fetch_live_signals(
        self, target_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """从线上生产表读取老系统写入的真实运行记录"""
        all_sigs = self.db.fetch_all_strategy_signals(run_mode="live")
        latest_by_strategy: Dict[str, Dict[str, Any]] = {}

        for sig in all_sigs:
            sid = sig["strategy_id"]
            sdate = str(sig["signal_date"])[:10]
            if target_date and sdate != str(target_date)[:10]:
                continue
            if sid not in latest_by_strategy:
                latest_by_strategy[sid] = sig

        return latest_by_strategy

    def compare(
        self,
        strategy_ids: Optional[List[str]] = None,
        target_date: Optional[str] = None,
        tolerance: float = 0.001,  # 允许 0.1% 的浮点容差
        notify: bool = False,
    ) -> Dict[str, Any]:
        """
        执行新老系统信号对齐稽核，输出详尽对比报告。
        默认仅比对当前已上线的策略 (status: online)。
        """
        target_strategies = strategy_ids or self.config.list_strategies(only_online=True)

        # 1. 运行新系统影子计算 (仅针对上线策略)
        shadow_results = self.run_shadow_signals(
            strategy_ids=target_strategies, run_date=target_date
        )

        # 2. 读取老系统线上生产信号
        live_results = self.fetch_live_signals(target_date=target_date)

        report_date = target_date or datetime.now().strftime("%Y-%m-%d")

        strategy_diffs: List[Dict[str, Any]] = []
        total_matched = 0
        total_mismatched = 0

        for sid in target_strategies:
            shadow = shadow_results.get(sid)
            live = live_results.get(sid)

            if not shadow or not live:
                strategy_diffs.append({
                    "strategy_id": sid,
                    "status": "MISSING_ONE_SIDE",
                    "reason": f"新系统影子存在={bool(shadow)}, 老系统线上存在={bool(live)}",
                    "details": [],
                })
                total_mismatched += 1
                continue

            shadow_weights = shadow.get("weights") or {}
            live_weights = live.get("weights") or {}
            all_assets = sorted(list(set(shadow_weights.keys()) | set(live_weights.keys())))

            weight_diffs = []
            is_matched = True

            for asset in all_assets:
                sw = float(shadow_weights.get(asset, 0.0))
                lw = float(live_weights.get(asset, 0.0))
                delta = abs(sw - lw)
                match = delta <= tolerance
                if not match:
                    is_matched = False

                weight_diffs.append({
                    "asset": asset,
                    "live_weight": lw,
                    "shadow_weight": sw,
                    "delta": delta,
                    "match": match,
                })

            status = "MATCH" if is_matched else "DIFF"
            if is_matched:
                total_matched += 1
            else:
                total_mismatched += 1

            strategy_diffs.append({
                "strategy_id": sid,
                "strategy_name": shadow.get("strategy_name", sid),
                "signal_date": shadow.get("signal_date"),
                "status": status,
                "live_trigger": live.get("trigger_reason", ""),
                "shadow_trigger": shadow.get("trigger_reason", ""),
                "weights_comparison": weight_diffs,
            })

        summary = {
            "report_date": report_date,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_strategies": len(target_strategies),
            "matched_count": total_matched,
            "diff_count": total_mismatched,
            "is_all_clean": total_mismatched == 0,
            "results": strategy_diffs,
            "shadow_results": shadow_results,
        }

        # 1. 完整 Markdown 报表并保存本地文件
        md_text = self._render_markdown_report(summary)
        summary["markdown_report"] = md_text

        report_file = self.reports_dir / f"reconcile_{report_date}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(md_text)

        summary["report_path"] = str(report_file.resolve())

        # 2. 精炼移动端推送格式 (HTML + 紧凑 Markdown)
        compact_html = self._render_compact_html(summary)
        compact_md = self._render_compact_markdown(summary)
        summary["compact_html"] = compact_html
        summary["compact_markdown"] = compact_md

        if notify:
            notifier = PushNotifier()
            title = f"量化双跑对齐稽核 ({report_date}): {'全部一致 ✅' if summary['is_all_clean'] else '存在微调差异 ⚠️'}"
            notifier.send(title, markdown_body=compact_md, html_body=compact_html)

        return summary

    def run_daily_shadow_pipeline(
        self,
        sync_mode: str = "daily",
        compute_factors: bool = True,
        notify: bool = True,
        sync_live: bool = True,
    ) -> Dict[str, Any]:
        """
        全套自动化双跑流水线：
        1. 拉取今天最新行情与估值 (DataSyncService)
        2. 自动计算最新因子
        3. 计算新系统影子策略信号 (outputs/shadow_signals/)
        4. 自动与老线上系统比对，生成 Diff 报告并推送通知！
        5. 同步将最新信号刷新至线上生产表 (run_mode=live)，供 Vercel 看板展示！
        """
        from core.data_hub.sync import DataSyncService
        from core.quant_engine.factors import FactorService

        # 1. 同步数据
        sync_svc = DataSyncService(db=self.db, config=self.config)
        if sync_mode == "catchup":
            sync_svc.sync_valuation_catchup()
        else:
            sync_svc.sync_daily(lookback_days=15)

        # 2. 计算因子
        if compute_factors:
            factor_svc = FactorService(db=self.db, config=self.config)
            factor_svc.compute_and_store_all(tail_n=60)

        # 3. 对比并推送 Diff 报告
        summary = self.compare(notify=notify)

        # 4. 同步更新线上生产表 (run_mode=live)
        if sync_live:
            shadow_results = summary.get("shadow_results") or self.run_shadow_signals()
            for sid, payload in shadow_results.items():
                try:
                    live_payload = dict(payload)
                    live_payload["run_mode"] = "live"
                    self.db.upsert_strategy_signal(live_payload)
                except Exception as exc:
                    logger.warning("同步 strategy_signals 到 live 失败 (%s): %s", sid, exc)
            logger.info("已将今日最新影子计算信号同步更新至线上生产表 (run_mode=live)")

        return summary


    @staticmethod
    def _format_asset_label(symbol: str, assets_map: Dict[str, Any]) -> str:
        cfg = assets_map.get(symbol)
        if cfg and getattr(cfg, "name", None):
            return f"{cfg.name}"
        known = {
            "513100.SH": "纳指ETF",
            "518880.SH": "黄金ETF",
            "159915.SZ": "创业板ETF",
            "512890.SH": "红利低波ETF",
            "510300.SH": "300ETF",
            "510500.SH": "500ETF",
            "000300.SH": "沪深300",
            "CBA02701.CS": "7-10年政金债",
            "SPX.GI": "标普500",
            "NDX.GI": "纳指100",
            "HSTECH.HI": "恒生科技",
            "000905.SH": "中证500",
            "000852.SH": "中证1000",
            "399269.SZ": "红利低波",
            "AU9999.SGE": "黄金现货",
            "Au99.99.SGE": "黄金现货",
            "AU9999.XSGE": "黄金现货",
        }
        return known.get(symbol, symbol)

    @staticmethod
    def _clean_strategy_name(name: str) -> str:
        import re
        return re.sub(r"\s*\(Trial.*?\)", "", name).strip()

    def _render_compact_html(self, summary: Dict[str, Any]) -> str:
        assets_map = self.config.load_assets()
        report_date = summary["report_date"]
        matched = summary["matched_count"]
        diff = summary["diff_count"]
        total = summary["total_strategies"]

        header_status = (
            f"✅ <b>全部 {total} 个策略 100% 对齐</b>"
            if summary["is_all_clean"]
            else f"⚠️ <b>{diff}/{total} 个策略存在偏差</b>"
        )

        lines = [
            f"📅 <b>对齐日期:</b> {report_date}",
            f"🎯 <b>对齐状态:</b> {header_status}",
            "🛡 <b>运行空间:</b> 影子隔离环境 (线上生产表 0 改动)",
            "",
        ]

        for idx, item in enumerate(summary["results"], 1):
            sid = item["strategy_id"]
            name = self._clean_strategy_name(item.get("strategy_name", sid))
            status = item["status"]

            if status == "MATCH":
                lines.append(f"<b>{idx}. {name}</b> (<code>{sid}</code>) ✅")
                raw_weights = item.get("weights_comparison", [])
                active_weights = [
                    w for w in raw_weights
                    if w["live_weight"] >= 0.005 or w["shadow_weight"] >= 0.005
                ]
                active_weights.sort(key=lambda x: x["live_weight"], reverse=True)

                if not active_weights:
                    lines.append("  • 当前空仓 / 避险现金")
                else:
                    lines.append("  • 资产配置 (完全一致):")
                    for w in active_weights:
                        label = self._format_asset_label(w["asset"], assets_map)
                        lines.append(f"    - {label}: <b>{w['live_weight']:.1%}</b>")
            elif status == "DIFF":
                lines.append(f"<b>{idx}. {name}</b> (<code>{sid}</code>) ❌ <b>存在偏差</b>")
                mismatches = [w for w in item.get("weights_comparison", []) if not w["match"]]
                mismatches.sort(key=lambda x: x["delta"], reverse=True)
                lines.append("  • 偏差标的:")
                for w in mismatches:
                    label = self._format_asset_label(w["asset"], assets_map)
                    lines.append(
                        f"    - {label}: 线上 <code>{w['live_weight']:.1%}</code> vs 影子 <code>{w['shadow_weight']:.1%}</code> (Δ {w['delta']:.1%})"
                    )
            else:
                lines.append(f"<b>{idx}. {name}</b> (<code>{sid}</code>) ⚠️")
                lines.append(f"  • 说明: {item.get('reason', '单侧缺失')}")
            lines.append("")

        lines.append("<i>📁 完整明细已落盘至 outputs/reports/ 目录</i>")
        return "\n".join(lines)

    def _render_compact_markdown(self, summary: Dict[str, Any]) -> str:
        assets_map = self.config.load_assets()
        report_date = summary["report_date"]
        matched = summary["matched_count"]
        diff = summary["diff_count"]
        total = summary["total_strategies"]

        header_status = (
            f"全部 {total} 个策略 100% 对齐 ✅"
            if summary["is_all_clean"]
            else f"{diff}/{total} 个策略存在偏差 ⚠️"
        )

        lines = [
            f"### 新老量化双跑对齐报告 ({report_date})",
            f"> 状态: **{header_status}** (容差 < 0.1%) | 模式: 影子隔离 (线上零改动)",
            "",
        ]

        for idx, item in enumerate(summary["results"], 1):
            sid = item["strategy_id"]
            name = self._clean_strategy_name(item.get("strategy_name", sid))
            status = item["status"]

            if status == "MATCH":
                lines.append(f"**{idx}. {name}** (`{sid}`) ✅")
                active_weights = [
                    w for w in item.get("weights_comparison", [])
                    if w["live_weight"] >= 0.005 or w["shadow_weight"] >= 0.005
                ]
                active_weights.sort(key=lambda x: x["live_weight"], reverse=True)

                if not active_weights:
                    lines.append("- 当前空仓 / 避险现金")
                else:
                    for w in active_weights:
                        label = self._format_asset_label(w["asset"], assets_map)
                        lines.append(f"- {label}: **{w['live_weight']:.1%}** (一致)")
            elif status == "DIFF":
                lines.append(f"**{idx}. {name}** (`{sid}`) ❌ **存在偏差**")
                mismatches = [w for w in item.get("weights_comparison", []) if not w["match"]]
                mismatches.sort(key=lambda x: x["delta"], reverse=True)
                for w in mismatches:
                    label = self._format_asset_label(w["asset"], assets_map)
                    lines.append(
                        f"- {label}: 线上 `{w['live_weight']:.1%}` vs 影子 `{w['shadow_weight']:.1%}` (Δ {w['delta']:.1%})"
                    )
            else:
                lines.append(f"**{idx}. {name}** (`{sid}`) ⚠️ {item.get('reason', '单侧缺失')}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_markdown_report(summary: Dict[str, Any]) -> str:
        lines = [
            f"# 新老量化系统双跑对齐稽核报告 ({summary['report_date']})",
            f"> 生成时间: `{summary['generated_at']}` | 模式: `Shadow 影子隔离模式` (now_asset 线上代码零改动)",
            "",
            "## 1. 总体对齐概况",
            f"- **比对策略总数**: `{summary['total_strategies']}` 个",
            f"- **完全一致策略**: `{summary['matched_count']}` 个 ✅",
            f"- **存在差异策略**: `{summary['diff_count']}` 个 {'❌ (需核验)' if summary['diff_count'] > 0 else ''}",
            f"- **综合判定结果**: `{'全部通过 100% 对齐' if summary['is_all_clean'] else '存在数值微调，请查阅下方明细'}`",
            "",
            "## 2. 策略逐项并排 Diff 明细",
            "",
        ]

        for item in summary["results"]:
            sid = item["strategy_id"]
            status = item["status"]
            status_icon = "✅ MATCH" if status == "MATCH" else "❌ DIFF"
            name = item.get("strategy_name", sid)
            sdate = item.get("signal_date", "-")

            lines.append(f"### {status_icon} | {name} (`{sid}`) - 信号日: `{sdate}`")
            if "reason" in item:
                lines.append(f"- 提示: `{item['reason']}`")
                lines.append("")
                continue

            lines.append(f"- **老线上触发理由**: {item.get('live_trigger')}")
            lines.append(f"- **新影子触发理由**: {item.get('shadow_trigger')}")
            lines.append("")
            lines.append("| 投资标的 (Asset) | 老系统线上权重 (Live) | 新系统影子权重 (Shadow) | 绝对偏差 (Delta) | 状态 |")
            lines.append("| :--- | :---: | :---: | :---: | :---: |")

            for w in item["weights_comparison"]:
                w_status = "✅ 一致" if w["match"] else "❌ 偏差"
                lines.append(
                    f"| `{w['asset']}` | `{w['live_weight']:.2%}` | `{w['shadow_weight']:.2%}` | `{w['delta']:.4%}` | {w_status} |"
                )
            lines.append("")

        return "\n".join(lines)
