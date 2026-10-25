"""
统一数据源插件集合 (Unified Data Providers).

整合原 `index_fetcher/fetchers/*` 与 `strategy/rotation_strategy/data/data_sources.py` 的全部能力：
1. SinaQuoteProvider: 新浪实时行情接口 (`hq.sinajs.cn`)，支持美股 (`gb_*`)、港股 (`rt_hk*`)、A股 (`sh*`/`sz*`)
2. HowbuyValuationProvider: 好买基金指数估值接口 (`data.howbuy.com`)，支持免登录获取最新及历史 PE/PB/股息率/ROE
3. ChinaBondProvider: 中债登官网财富指数接口 (`yield.chinabond.com.cn`) + AkShare 债券指数兜底
4. AkshareMarketProvider: AkShare 指数历史日线、上海金 (`Au99.99`)、场内 ETF 前复权日线（东财+新浪双源兜底）
5. SbeMacroProvider: 股债利差温度计分位数采集（蛋卷 HTTP + 韭圈儿/历史衍生计算兜底）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from core.data_hub.models import MarketBar

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "" or val == "--":
        return None
    try:
        f = float(val)
        if pd.isna(f) or f != f:
            return None
        return f
    except (ValueError, TypeError):
        return None


# ==============================================================================
# 1. 新浪实时行情源 (SinaQuoteProvider)
# ==============================================================================
class SinaQuoteProvider:
    """新浪财经极速行情源 (美股 / 港股 / A股指数与ETF)"""

    SINA_URL = "http://hq.sinajs.cn/list={codes}"
    HEADERS = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    def fetch_realtime_bar(self, symbol: str, sina_code: str) -> Optional[MarketBar]:
        try:
            resp = requests.get(
                self.SINA_URL.format(codes=sina_code),
                headers=self.HEADERS,
                timeout=10,
            )
            resp.encoding = "gbk"
            text = resp.text.strip()
            match = re.search(r'="([^"]+)"', text)
            if not match:
                return None
            parts = match.group(1).split(",")
            if sina_code.startswith("gb_"):
                return self._parse_us(symbol, parts)
            if sina_code.startswith("rt_hk"):
                return self._parse_hk(symbol, parts)
            return self._parse_cn(symbol, parts)
        except Exception as exc:
            logger.warning("Sina 实时行情获取失败 [%s/%s]: %s", symbol, sina_code, exc)
            return None

    @staticmethod
    def _parse_us(symbol: str, parts: List[str]) -> Optional[MarketBar]:
        # gb_ndx: 0:名称, 1:现价, 2:涨跌幅%, 3:日期时间(YYYY-MM-DD HH:MM:SS), ..., 5:今开, 6:最高, 7:最低, ..., 10:成交量
        if len(parts) < 8:
            return None
        close = _safe_float(parts[1])
        if not close or close <= 0:
            return None
        dt_str = parts[3].split(" ")[0] if len(parts) > 3 and parts[3] else ""
        if not dt_str:
            dt_str = datetime.now().strftime("%Y-%m-%d")
        return MarketBar(
            symbol=symbol,
            trade_date=dt_str[:10],
            open=_safe_float(parts[5]),
            high=_safe_float(parts[6]),
            low=_safe_float(parts[7]),
            close=close,
            volume=_safe_float(parts[10]) if len(parts) > 10 else None,
            change_pct=_safe_float(parts[2]),
        )

    @staticmethod
    def _parse_hk(symbol: str, parts: List[str]) -> Optional[MarketBar]:
        # rt_hkHSTECH: 0:英文, 1:中文, 2:今开, 3:昨收, 4:最高, 5:最低, 6:现价, 7:涨跌额, 8:涨跌幅%, ..., 17:日期(YYYY/MM/DD)
        if len(parts) < 18:
            return None
        close = _safe_float(parts[6])
        if not close or close <= 0:
            return None
        raw_date = parts[17].replace("/", "-")
        return MarketBar(
            symbol=symbol,
            trade_date=raw_date[:10],
            open=_safe_float(parts[2]),
            high=_safe_float(parts[4]),
            low=_safe_float(parts[5]),
            close=close,
            volume=_safe_float(parts[12]) if len(parts) > 12 else None,
            amount=_safe_float(parts[11]) if len(parts) > 11 else None,
            change_pct=_safe_float(parts[8]),
        )

    @staticmethod
    def _parse_cn(symbol: str, parts: List[str]) -> Optional[MarketBar]:
        # sh000300: 0:名称, 1:今开, 2:昨收, 3:现价, 4:最高, 5:最低, ..., 8:成交量, 9:成交额, ..., 30:日期, 31:时间
        if len(parts) < 31:
            return None
        close = _safe_float(parts[3])
        prev_close = _safe_float(parts[2])
        if not close or close <= 0:
            return None
        change_pct = None
        if prev_close and prev_close > 0:
            change_pct = round((close - prev_close) / prev_close * 100.0, 4)
        return MarketBar(
            symbol=symbol,
            trade_date=parts[30][:10],
            open=_safe_float(parts[1]),
            high=_safe_float(parts[4]),
            low=_safe_float(parts[5]),
            close=close,
            volume=_safe_float(parts[8]),
            amount=_safe_float(parts[9]),
            change_pct=change_pct,
        )


# ==============================================================================
# 2. 好买基金估值源 (HowbuyValuationProvider)
# ==============================================================================
class HowbuyValuationProvider:
    """好买指数估值接口 (PE / PB / 股息率 / ROE 最新值与历史全量序列)"""

    DETAIL_URL = "https://data.howbuy.com/cgi/fund/zszq/zsgz/detail.json"
    HISTORY_URL = "https://data.howbuy.com/cgi/fund/zszq/zsgz/zst.json"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.howbuy.com/",
        "Origin": "https://www.howbuy.com",
    }

    def fetch_latest_valuation(self, symbol: str, zsdm: str) -> Optional[MarketBar]:
        """获取指数最新单日估值快照"""
        try:
            resp = requests.post(
                self.DETAIL_URL,
                data={"zsdm": zsdm},
                headers=self.HEADERS,
                timeout=10,
            )
            data = resp.json().get("data") or {}
            gz_date = data.get("gzDate") or data.get("rq")
            if not gz_date:
                # 回退从历史曲线取最后一条
                hist = self.fetch_history_valuation(symbol, zsdm, cycle="1m")
                return hist[-1] if hist else None

            raw_dt = str(gz_date).replace("/", "-")
            if len(raw_dt) == 8 and raw_dt.isdigit():
                raw_dt = f"{raw_dt[:4]}-{raw_dt[4:6]}-{raw_dt[6:8]}"

            return MarketBar(
                symbol=symbol,
                trade_date=raw_dt[:10],
                pe=_safe_float(data.get("pe") or data.get("peTtm")),
                pb=_safe_float(data.get("pb")),
                dividend_yield=_safe_float(data.get("gxl")),
                roe=_safe_float(data.get("roe")),
            )
        except Exception as exc:
            logger.warning("Howbuy 最新估值获取失败 [%s/%s]: %s", symbol, zsdm, exc)
            return None

    def fetch_history_valuation(
        self, symbol: str, zsdm: str, cycle: str = "10y"
    ) -> List[MarketBar]:
        """获取指数历史估值序列 (PE/PB/GXL 合并按日期对齐)"""
        series_map: Dict[str, Dict[str, float]] = {}
        # field_type: 1=PE, 2=PB, 3=股息率
        for field_code, attr_name in [("1", "pe"), ("2", "pb"), ("3", "dividend_yield")]:
            try:
                resp = requests.post(
                    self.HISTORY_URL,
                    data={"zsdm": zsdm, "type": field_code, "cycle": cycle},
                    headers=self.HEADERS,
                    timeout=15,
                )
                payload = resp.json().get("data") or {}
                items = payload.get("list") or payload.get("zstList") or []
                for item in items:
                    dt_raw = str(item.get("rq") or item.get("date") or "").strip()
                    val = _safe_float(item.get("val") or item.get("value") or item.get(attr_name))
                    if not dt_raw or val is None:
                        continue
                    if len(dt_raw) == 8 and dt_raw.isdigit():
                        dt_str = f"{dt_raw[:4]}-{dt_raw[4:6]}-{dt_raw[6:8]}"
                    else:
                        dt_str = dt_raw[:10]
                    series_map.setdefault(dt_str, {})[attr_name] = val
            except Exception as exc:
                logger.debug("Howbuy 历史字段 %s 获取异常 [%s]: %s", attr_name, symbol, exc)

        bars: List[MarketBar] = []
        for dt_str in sorted(series_map.keys()):
            vals = series_map[dt_str]
            bars.append(
                MarketBar(
                    symbol=symbol,
                    trade_date=dt_str,
                    pe=vals.get("pe"),
                    pb=vals.get("pb"),
                    dividend_yield=vals.get("dividend_yield"),
                )
            )
        return bars


# ==============================================================================
# 3. 中债登财富指数源 (ChinaBondProvider)
# ==============================================================================
class ChinaBondProvider:
    """中债登官网财富指数接口 + AkShare 兜底"""

    URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://yield.chinabond.com.cn/cbweb-mn/indices/single_index_query",
        "X-Requested-With": "XMLHttpRequest",
    }

    def fetch_bars(
        self,
        symbol: str,
        chinabond_id: str,
        ak_bond_name: Optional[str] = None,
        ak_bond_indicator: str = "财富指数",
        start_date: Optional[str] = None,
    ) -> List[MarketBar]:
        bars = self._fetch_from_chinabond_web(symbol, chinabond_id, start_date)
        if not bars and ak_bond_name:
            bars = self._fetch_from_akshare(
                symbol, ak_bond_name, ak_bond_indicator, start_date
            )
        return bars

    def _fetch_from_chinabond_web(
        self, symbol: str, index_id: str, start_date: Optional[str] = None
    ) -> List[MarketBar]:
        try:
            params = {
                "indexid": index_id,
                "qxlx": "00",
                "ltlx": "01",
                "zslx": "CFZS_00",
                "lx": "1",
                "locale": "zh_CN",
            }
            resp = requests.post(
                self.URL, params=params, headers=self.HEADERS, timeout=15
            )
            data = resp.json()
            cfzs = data.get("CFZS_00") or {}
            if not cfzs:
                return []

            rows: List[tuple[str, float]] = []
            for ts_ms, val in cfzs.items():
                price = _safe_float(val)
                if price is None:
                    continue
                dt_str = datetime.fromtimestamp(int(ts_ms) / 1000.0).strftime("%Y-%m-%d")
                if start_date and dt_str < start_date[:10]:
                    continue
                rows.append((dt_str, price))

            rows.sort(key=lambda x: x[0])
            bars: List[MarketBar] = []
            prev_close: Optional[float] = None
            for dt_str, close in rows:
                chg = (
                    round((close - prev_close) / prev_close * 100.0, 4)
                    if prev_close and prev_close > 0
                    else None
                )
                bars.append(
                    MarketBar(
                        symbol=symbol,
                        trade_date=dt_str,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        change_pct=chg,
                    )
                )
                prev_close = close
            return bars
        except Exception as exc:
            logger.warning("中债官网拉取失败 [%s]: %s", symbol, exc)
            return []

    @staticmethod
    def _fetch_from_akshare(
        symbol: str,
        bond_name: str,
        indicator: str = "财富指数",
        start_date: Optional[str] = None,
    ) -> List[MarketBar]:
        try:
            import akshare as ak

            df = ak.bond_composite_index_cbond(indicator=indicator, period=bond_name)
            if df is None or df.empty:
                return []
            df = df.sort_values("date").reset_index(drop=True)
            bars: List[MarketBar] = []
            for _, row in df.iterrows():
                dt_str = str(row["date"])[:10]
                if start_date and dt_str < start_date[:10]:
                    continue
                close = _safe_float(row["value"])
                if close is None:
                    continue
                bars.append(
                    MarketBar(
                        symbol=symbol,
                        trade_date=dt_str,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                    )
                )
            return bars
        except Exception as exc:
            logger.warning("AkShare 债券兜底拉取失败 [%s]: %s", symbol, exc)
            return []


# ==============================================================================
# 4. AkShare 历史日线源 (指数 + 上海金 + 场内 ETF 前复权)
# ==============================================================================
class AkshareMarketProvider:
    """AkShare 历史行情抓取器 (指数 / 上海金 / ETF)"""

    def fetch_index_history(
        self,
        symbol: str,
        ak_type: str,
        ak_symbol: str,
        start_date: Optional[str] = None,
    ) -> List[MarketBar]:
        try:
            import akshare as ak

            if ak_type == "us":
                df = ak.index_us_stock_sina(symbol=ak_symbol)
            elif ak_type == "hk":
                df = ak.stock_hk_index_daily_sina(symbol=ak_symbol)
            elif ak_type == "cn_em":
                df = ak.stock_zh_index_daily_em(symbol=ak_symbol)
            else:
                df = ak.stock_zh_index_daily(symbol=ak_symbol)

            return self._dataframe_to_bars(symbol, df, start_date)
        except Exception as exc:
            logger.warning("AkShare 指数历史获取失败 [%s/%s]: %s", symbol, ak_symbol, exc)
            return []

    def fetch_gold_history(
        self, symbol: str, sge_symbol: str = "Au99.99", start_date: Optional[str] = None
    ) -> List[MarketBar]:
        try:
            import akshare as ak

            df = ak.spot_hist_sge(symbol=sge_symbol)
            return self._dataframe_to_bars(symbol, df, start_date)
        except Exception as exc:
            logger.warning("AkShare 上海金获取失败 [%s]: %s", symbol, exc)
            return []

    def fetch_etf_history(
        self,
        symbol: str,
        etf_code: str,
        sina_etf_code: Optional[str] = None,
        start_date: Optional[str] = None,
    ) -> List[MarketBar]:
        """获取场内 ETF 日线 (优先东财前复权 qfq，失败回退新浪 ETF 日线)"""
        try:
            import akshare as ak

            start_param = (
                start_date.replace("-", "") if start_date else "20130101"
            )
            df = ak.fund_etf_hist_em(
                symbol=etf_code,
                period="daily",
                start_date=start_param,
                adjust="qfq",
            )
            if df is not None and not df.empty:
                df = df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "涨跌幅": "change_pct",
                    }
                )
                return self._dataframe_to_bars(symbol, df, start_date)
        except Exception as exc:
            logger.debug("东财 ETF 日线拉取失败，尝试新浪兜底 [%s]: %s", symbol, exc)

        if sina_etf_code:
            try:
                import akshare as ak

                df = ak.fund_etf_hist_sina(symbol=sina_etf_code)
                return self._dataframe_to_bars(symbol, df, start_date)
            except Exception as exc:
                logger.warning("新浪 ETF 日线兜底获取失败 [%s]: %s", symbol, exc)
        return []

    @staticmethod
    def _dataframe_to_bars(
        symbol: str, df: pd.DataFrame, start_date: Optional[str] = None
    ) -> List[MarketBar]:
        if df is None or df.empty:
            return []
        df = df.copy()
        date_col = "date" if "date" in df.columns else df.columns[0]
        df["trade_date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("trade_date").reset_index(drop=True)
        if "change_pct" not in df.columns and "close" in df.columns:
            df["change_pct"] = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100.0

        if start_date:
            df = df[df["trade_date"] >= start_date[:10]]

        bars: List[MarketBar] = []
        for _, row in df.iterrows():
            close = _safe_float(row.get("close"))
            if close is None or close <= 0:
                continue
            bars.append(
                MarketBar(
                    symbol=symbol,
                    trade_date=str(row["trade_date"])[:10],
                    open=_safe_float(row.get("open")),
                    high=_safe_float(row.get("high")),
                    low=_safe_float(row.get("low")),
                    close=close,
                    volume=_safe_float(row.get("volume")),
                    amount=_safe_float(row.get("amount")),
                    change_pct=_safe_float(row.get("change_pct")),
                )
            )
        return bars


# ==============================================================================
# 5. 股债性价比温度计采集源 (SbeMacroProvider)
# ==============================================================================
class SbeMacroProvider:
    """
    股债性价比分位数采集器 (`SBE.DJ` / `SBE.JCS`)。
    把 SBE 指标统一为 `market_daily.close` (0~100 分位数)，使 SBE 策略完全复用标准行情与策略链路。
    """

    DANJUAN_URL = "https://danjuanfunds.com/djapi/index_eva/dj"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://danjuanfunds.com/",
    }

    def fetch_latest(self, symbol: str, sbe_source: str) -> Optional[MarketBar]:
        if sbe_source == "danjuan":
            return self._fetch_danjuan(symbol)
        return None

    def _fetch_danjuan(self, symbol: str) -> Optional[MarketBar]:
        try:
            resp = requests.get(self.DANJUAN_URL, headers=self.HEADERS, timeout=10)
            data = resp.json().get("data") or {}
            items = data.get("items") or []
            if not items:
                return None
            # 计算全市场平均 PE 百分位作为温度计分位参考
            pe_pcts = [
                float(x["pe_percentile"]) * 100.0
                for x in items
                if x.get("pe_percentile") is not None
            ]
            if not pe_pcts:
                return None
            avg_pct = round(sum(pe_pcts) / len(pe_pcts), 2)
            dt_str = datetime.now().strftime("%Y-%m-%d")
            return MarketBar(
                symbol=symbol,
                trade_date=dt_str,
                open=avg_pct,
                high=avg_pct,
                low=avg_pct,
                close=avg_pct,
                extra_json={"source": "danjuan"},
            )
        except Exception as exc:
            logger.debug("蛋卷估值温度计获取失败: %s", exc)
            return None
