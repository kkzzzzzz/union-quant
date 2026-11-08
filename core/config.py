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
        self.database_url = self._resolve_database_url()
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
        )
        self._assets_cache: Optional[Dict[str, AssetConfig]] = None

    def _resolve_database_url(self) -> str:
        """解析关系型数据库连接串 (支持 DATABASE_URL 或独立 DB_* 环境变量配置)"""
        url = os.getenv("DATABASE_URL", "").strip() or os.getenv("DB_URL", "").strip()
        if not url:
            host = os.getenv("DB_HOST", "").strip()
            if host:
                db_type = os.getenv("DB_TYPE", "postgresql").strip().lower()
                user = os.getenv("DB_USER", "").strip()
                password = os.getenv("DB_PASSWORD", "").strip()
                port = os.getenv("DB_PORT", "").strip()
                name = os.getenv("DB_NAME", "").strip() or "union_quant"

                auth = f"{user}:{password}@" if user or password else ""
                port_str = f":{port}" if port else ""
                if db_type in ("postgres", "postgresql"):
                    driver = "postgresql+psycopg2"
                    default_port = port_str or ":5432"
                    url = f"{driver}://{auth}{host}{default_port}/{name}"
                elif db_type in ("mysql", "mariadb"):
                    driver = "mysql+pymysql"
                    default_port = port_str or ":3306"
                    url = f"{driver}://{auth}{host}{default_port}/{name}"
                else:
                    url = f"{db_type}://{auth}{host}{port_str}/{name}"

        # 规范化 postgres:// 为 postgresql://
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        return url


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

    def list_strategies(self, only_online: bool = True) -> List[str]:
        """列出策略 ID (默认仅返回 status 为 online 的已上线策略)"""
        if not self.strategies_dir.exists():
            return []
        ids: List[str] = []
        for p in sorted(self.strategies_dir.glob("*.yaml")):
            sid = p.stem
            if only_online:
                try:
                    scfg = self.load_strategy_config(sid)
                    if str(scfg.get("status", "online")).lower() != "online":
                        continue
                except Exception:
                    pass
            ids.append(sid)
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
        cfg.setdefault("status", "online")
        return cfg

    def update_strategy_file_status(self, strategy_id: str, status: str) -> bool:
        """同步更新策略配置 YAML 文件中的 status 字段"""
        candidate = self.strategies_dir / f"{strategy_id}.yaml"
        if not candidate.exists():
            return False
        with open(candidate, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["status"] = status
        with open(candidate, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return True
