"""
统一 HTTP 请求客户端与连接池会话管理 (HTTP Session & Retry Utilities).

提供：
1. 自动重试机制 (指数退避，针对 429, 500, 502, 503, 504 及连接超时)
2. HTTP Keep-Alive 连接池复用
3. 统一默认超时与 User-Agent 配置
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_GLOBAL_SESSION: Optional[requests.Session] = None


def get_http_session(
    total_retries: int = 3,
    backoff_factor: float = 0.5,
    pool_connections: int = 15,
    pool_maxsize: int = 30,
) -> requests.Session:
    """获取或初始化全局共享的带重试机制的 HTTP Session"""
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is not None:
        return _GLOBAL_SESSION

    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    _GLOBAL_SESSION = session
    return _GLOBAL_SESSION


def safe_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Optional[requests.Response]:
    """
    带超时、重试与异常静默保护的统一安全 HTTP 请求
    """
    session = get_http_session()
    req_headers = headers or {}
    try:
        resp = session.request(
            method=method.upper(),
            url=url,
            headers=req_headers,
            params=params,
            data=data,
            timeout=timeout,
            **kwargs,
        )
        return resp
    except Exception as exc:
        logger.warning("HTTP 请求失败 [%s %s]: %s", method, url, exc)
        return None
