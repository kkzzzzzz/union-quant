"""
统一多通道消息推送模块 (Unified Multi-Channel Push Notifier).
同时服务于：
1. 数据同步与估值滞后提醒 (`DataSyncService`)
2. 每日策略信号与调仓推送 (`run_daily_signal_pipeline`)

支持通道（通过环境变量自动发现）：
- 钉钉机器人 (`DINGTALK_WEBHOOK`, `DINGTALK_SECRET`)
- 飞书机器人 (`FEISHU_WEBHOOK`)
- 企业微信机器人 (`WECOM_WEBHOOK`)
- Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)
- Server酱 (`SERVERCHAN_KEY`)
- PushPlus (`PUSHPLUS_TOKEN`)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class PushNotifier:
    """统一多通道通知器"""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
        self.dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK", "").strip()
        self.dingtalk_secret = os.getenv("DINGTALK_SECRET", "").strip()
        self.feishu_webhook = os.getenv("FEISHU_WEBHOOK", "").strip()
        self.wecom_webhook = os.getenv("WECOM_WEBHOOK", "").strip()
        self.tg_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.serverchan_key = os.getenv("SERVERCHAN_KEY", "").strip()
        self.pushplus_token = os.getenv("PUSHPLUS_TOKEN", "").strip()

    def configured_channels(self) -> List[str]:
        channels: List[str] = []
        if self.dingtalk_webhook:
            channels.append("dingtalk")
        if self.feishu_webhook:
            channels.append("feishu")
        if self.wecom_webhook:
            channels.append("wecom")
        if self.tg_bot_token and self.tg_chat_id:
            channels.append("telegram")
        if self.serverchan_key:
            channels.append("serverchan")
        if self.pushplus_token:
            channels.append("pushplus")
        return channels

    def send(self, title: str, markdown_body: str) -> Dict[str, bool]:
        """向所有已配置的通道推送 Markdown 消息"""
        results: Dict[str, bool] = {}
        if self.dingtalk_webhook:
            results["dingtalk"] = self._send_dingtalk(title, markdown_body)
        if self.feishu_webhook:
            results["feishu"] = self._send_feishu(title, markdown_body)
        if self.wecom_webhook:
            results["wecom"] = self._send_wecom(title, markdown_body)
        if self.tg_bot_token and self.tg_chat_id:
            results["telegram"] = self._send_telegram(title, markdown_body)
        if self.serverchan_key:
            results["serverchan"] = self._send_serverchan(title, markdown_body)
        if self.pushplus_token:
            results["pushplus"] = self._send_pushplus(title, markdown_body)

        if not results:
            logger.info("未配置任何推送通道环境变量，跳过外部推送: %s", title)
        return results

    def _send_dingtalk(self, title: str, content: str) -> bool:
        url = self.dingtalk_webhook
        if self.dingtalk_secret:
            timestamp = str(round(time.time() * 1000))
            string_to_sign = f"{timestamp}\n{self.dingtalk_secret}"
            hmac_code = hmac.new(
                self.dingtalk_secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
            sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}timestamp={timestamp}&sign={sign}"

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": f"### {title}\n\n{content}"},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("DingTalk 推送失败: %s", exc)
            return False

    def _send_feishu(self, title: str, content: str) -> bool:
        payload = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": [[{"tag": "text", "text": content}]],
                    }
                }
            },
        }
        try:
            resp = requests.post(
                self.feishu_webhook, json=payload, timeout=self.timeout
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Feishu 推送失败: %s", exc)
            return False

    def _send_wecom(self, title: str, content: str) -> bool:
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": f"### {title}\n{content}"},
        }
        try:
            resp = requests.post(
                self.wecom_webhook, json=payload, timeout=self.timeout
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("WeCom 推送失败: %s", exc)
            return False

    def _send_telegram(self, title: str, content: str) -> bool:
        url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendMessage"
        payload = {
            "chat_id": self.tg_chat_id,
            "text": f"*{title}*\n\n{content}",
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Telegram 推送失败: %s", exc)
            return False

    def _send_serverchan(self, title: str, content: str) -> bool:
        url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
        try:
            resp = requests.post(
                url, data={"title": title, "desp": content}, timeout=self.timeout
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("ServerChan 推送失败: %s", exc)
            return False

    def _send_pushplus(self, title: str, content: str) -> bool:
        url = "https://www.pushplus.plus/send"
        payload = {
            "token": self.pushplus_token,
            "title": title,
            "content": content,
            "template": "markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("PushPlus 推送失败: %s", exc)
            return False
