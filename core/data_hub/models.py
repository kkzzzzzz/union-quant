"""
数据拉取层数据模型 (Data Hub Models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MarketBar:
    """统一的日频行情与估值行结构 (对应 `market_daily` 表)"""

    symbol: str
    trade_date: str  # YYYY-MM-DD
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None
    change_pct: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    extra_json: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "symbol": self.symbol,
            "trade_date": self.trade_date[:10],
        }
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
            val = getattr(self, col)
            if val is not None:
                record[col] = float(val)
        if self.extra_json:
            record["extra_json"] = self.extra_json
        return record


@dataclass
class SyncReport:
    """每日同步与估值补齐报告"""

    run_date: str
    synced_symbols: List[str] = field(default_factory=list)
    failed_symbols: Dict[str, str] = field(default_factory=dict)
    stale_valuations: List[Dict[str, Any]] = field(default_factory=list)
    upserted_rows: int = 0
