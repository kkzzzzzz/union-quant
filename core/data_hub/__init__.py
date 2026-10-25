"""
一个拉数据 (One Data Hub).
负责将全球及A股指数行情、好买PE/PB/股息率估值、中债财富指数、上海金、场内ETF日线、股债利差温度计(SBE)
统一拉取并落入 `market_daily` 表及本地缓存。
"""

from core.data_hub.models import MarketBar, SyncReport
from core.data_hub.sync import DataSyncService

__all__ = ["MarketBar", "SyncReport", "DataSyncService"]
