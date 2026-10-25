"""
统一配置加载器 (Unified Config Loader).
支持加载根目录 `.env`、`configs/assets.yaml` 以及 `configs/strategies/*.yaml`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "configs"
STRATEGIES_DIR = CONFIG_DIR / "strategies"
DATA_DIR = ROOT_DIR / "data"
DATA_CACHE_DIR = DATA_DIR / "cache"
DEFAULT_SQLITE_PATH = DATA_DIR / "union_quant.db"
OUTPUT_DIR = ROOT_DIR / "outputs"

# 自动加载根目录 .env
load_dotenv(ROOT_DIR / ".env")


@dataclass
class AssetConfig:
    symbol: str
    name: str
    asset_type: str  # index | bond | gold | etf | macro
    market: str = "CN"
    currency: str = "CNY"
    trade_etf: Optional[str] = None
    sources: Dict[str, Any] = field(default_factory=dict)
    factors: List[str] = field(default_factory=list)


class AppConfig:
    """全局配置中心"""

    def __init__(self, config_dir: Path = CONFIG_DIR) -> None:
        self.config_dir = config_dir
        self.strategies_dir = config_dir / "strategies"
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
        )
        self._assets_cache: Optional[Dict[str, AssetConfig]] = None

    def load_assets(self, force_reload: bool = False) -> Dict[str, AssetConfig]:
        """加载 `configs/assets.yaml` 中的所有标的定义"""
        if self._assets_cache is not None and not force_reload:
            return self._assets_cache

        assets_path = self.config_dir / "assets.yaml"
        if not assets_path.exists():
            self._assets_cache = {}
            return self._assets_cache

        with open(assets_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        assets_raw = raw.get("assets", {})
        result: Dict[str, AssetConfig] = {}
        for symbol, item in assets_raw.items():
            result[str(symbol)] = AssetConfig(
                symbol=str(symbol),
                name=item.get("name", str(symbol)),
                asset_type=item.get("asset_type", "index"),
                market=item.get("market", "CN"),
                currency=item.get("currency", "CNY"),
                trade_etf=item.get("trade_etf"),
                sources=dict(item.get("sources") or {}),
                factors=list(item.get("factors") or []),
            )
        self._assets_cache = result
        return result

    def get_asset(self, symbol: str) -> Optional[AssetConfig]:
        return self.load_assets().get(symbol)

    def list_strategies(self) -> List[str]:
        """列出所有可用的策略 ID"""
        if not self.strategies_dir.exists():
            return []
        ids: List[str] = []
        for p in sorted(self.strategies_dir.glob("*.yaml")):
            ids.append(p.stem)
        return ids

    def load_strategy_config(self, strategy_id_or_path: str) -> Dict[str, Any]:
        """按策略 ID 或 YAML 路径加载策略配置"""
        candidate = Path(strategy_id_or_path)
        if not candidate.exists():
            candidate = self.strategies_dir / f"{strategy_id_or_path}.yaml"
        if not candidate.exists():
            raise FileNotFoundError(f"未找到策略配置文件: {strategy_id_or_path}")

        with open(candidate, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        cfg.setdefault("strategy_id", candidate.stem)
        cfg.setdefault("name", cfg["strategy_id"])
        cfg.setdefault("version", "2.0")
        return cfg
