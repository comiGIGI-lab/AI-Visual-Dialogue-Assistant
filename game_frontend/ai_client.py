# -*- coding: utf-8 -*-
"""
外部 AI 客户端 (可选)
=====================

支持通过环境变量启用外部大模型生成更自然的回复。
provider=none 时使用本地规则，不调用云端。

环境变量:
    OFFICEFIT_AI_PROVIDER = none | openai | qwen
    OPENAI_API_KEY        = sk-... (openai 时必需)
    OPENAI_BASE_URL       = https://... (可选，默认 api.openai.com)
    OPENAI_MODEL          = gpt-4o-mini (可选)
    DASHSCOPE_API_KEY     = sk-... (qwen 时必需)
    DASHSCOPE_BASE_URL    = https://... (qwen 可选)
    OFFICEFIT_QWEN_MODEL  = qwen-plus (可选)

用法:
    from game_frontend.ai_client import create_ai_client
    client = create_ai_client()
    reply = client.generate_reply(user_text, user_state_dict)
"""

import os
from typing import Optional, Dict, Any


SYSTEM_PROMPT = (
    "你是 OfficeFit AI 久坐健康助手。"
    "你不能做医学诊断，只能根据用户反馈和视觉状态推荐轻量办公拉伸。"
    "回复要简短、友好、可执行（2-3句话）。"
    "不要要求用户做产品无法识别的精细动作。"
    "不要提及上传视频。"
)


class BaseAIClient:
    def __init__(self):
        self._source_name = "本地"

    @property
    def source_name(self) -> str:
        return self._source_name

    def is_available(self) -> bool:
        return False

    def generate_reply(self, user_text: str,
                       user_state: Optional[dict] = None) -> Optional[str]:
        return None

    def get_status(self) -> dict:
        return {"provider": "none", "available": False,
                "source": self._source_name}


class NoneAIClient(BaseAIClient):
    def get_status(self) -> dict:
        return {"provider": "none", "available": False, "source": "本地"}


class OpenAIClient(BaseAIClient):
    def __init__(self, api_key: str, base_url: str = "",
                 model: str = ""):
        super().__init__()
        self._api_key = api_key
        self._available = bool(api_key)
        self._base_url = base_url or "https://api.openai.com/v1"
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._last_error: str = ""

    def is_available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str:
        return self._last_error

    def generate_reply(self, user_text: str,
                       user_state: Optional[dict] = None) -> Optional[str]:
        if not self._available:
            return None
        try:
            import requests

            state = user_state or {}
            user_prompt = (
                f"用户说: {user_text}\n"
                f"检测意图: {state.get('detected_intents', [])}\n"
                f"疲劳部位: {state.get('fatigue_parts', [])}\n"
                f"推荐动作: {state.get('recommended_actions', [])}\n"
                f"当前模式: {state.get('current_mode', 'office')}\n"
                f"连续久坐: {state.get('sedentary_minutes', 0)} 分钟\n"
                f"相机: {state.get('camera_source', 'unknown')} "
                f"(深度: {'有' if state.get('depth_available') else '无'})"
            )

            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 120,
                    "temperature": 0.7,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                result = resp.json()
                choices = result.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"].strip()
            else:
                self._last_error = f"API 返回 {resp.status_code}"
            return None
        except Exception as e:
            self._last_error = str(e)
            return None

    def get_status(self) -> dict:
        return {
            "provider": "openai",
            "available": self._available,
            "source": self._source_name,
        }


def create_ai_client() -> BaseAIClient:
    provider = os.environ.get("OFFICEFIT_AI_PROVIDER", "").strip().lower()

    # 如果显式设置为 none，强制关闭
    if provider == "none":
        return NoneAIClient()

    # 自动检测：如果有 key 则默认启用对应 provider
    if not provider:
        if os.environ.get("DASHSCOPE_API_KEY", "").strip():
            provider = "qwen"
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            provider = "openai"
        else:
            return NoneAIClient()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("[AI] OFFICEFIT_AI_PROVIDER=openai 但 OPENAI_API_KEY 未设置")
            return NoneAIClient()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
        client = OpenAIClient(api_key, base_url=base_url, model=model)
        client._source_name = "外部 AI"
        if base_url:
            print(f"[AI] OpenAI ready ({model}, base={base_url})")
        else:
            print(f"[AI] OpenAI ready ({model})")
        return client

    if provider == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            print("[AI] OFFICEFIT_AI_PROVIDER=qwen 但 DASHSCOPE_API_KEY 未设置")
            return NoneAIClient()
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip()
        model = os.environ.get("OFFICEFIT_QWEN_MODEL", "qwen-plus").strip()
        client = OpenAIClient(api_key, base_url=base_url, model=model)
        client._source_name = "外部 AI (Qwen)"
        print(f"[AI] Qwen ready ({model})")
        return client

    if provider != "none":
        print(f"[AI] unknown provider={provider}, fallback local")

    return NoneAIClient()
