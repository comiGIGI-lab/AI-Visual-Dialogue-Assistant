#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 连通性测试
===========================

用于测试任何 OpenAI-compatible 接口（OpenAI / DeepSeek / Qwen / 自定义代理等）。

环境变量:
    OPENAI_API_KEY     (必需) API Key
    OPENAI_BASE_URL    默认 https://api.deepseek.com
    OPENAI_MODEL       默认 deepseek-chat

用法:
    # DeepSeek (默认)
    $env:OPENAI_API_KEY="sk-..."
    python scripts/test_openai_compatible_api.py

    # 自定义代理
    $env:OPENAI_BASE_URL="https://cpa.aqor.io/v1"
    $env:OPENAI_MODEL="gpt-4o-mini"
    python scripts/test_openai_compatible_api.py
"""

import os
import sys


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get(
        "OPENAI_BASE_URL", "https://api.deepseek.com"
    ).strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat").strip()

    print("=" * 50)
    print("  OpenAI 兼容 API 连通性测试")
    print("=" * 50)
    print(f"  Base URL: {base_url}")
    print(f"  Model:    {model}")

    if not api_key:
        print()
        print("[错误] 未设置 OPENAI_API_KEY")
        print()
        print("PowerShell:")
        print('  $env:OPENAI_API_KEY="sk-你的key"')
        print()
        print("可选: 自定义 endpoint")
        print('  $env:OPENAI_BASE_URL="https://api.deepseek.com"')
        print('  $env:OPENAI_MODEL="deepseek-chat"')
        sys.exit(1)

    masked = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"  API Key:  {masked}")
    print()

    # 检查 openai 包
    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] openai Python SDK 未安装")
        print("pip install openai")
        sys.exit(1)

    # 创建客户端
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"[错误] 创建客户端失败: {e}")
        sys.exit(1)

    # 发送测试请求
    print("  发送测试请求...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请只回复：外部 AI API 测试成功"}],
            max_tokens=50,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        print(f"  模型回复: {content}")
        print()
        print("[通过] API 测试成功!")
        print(f"  实际模型: {response.model}")
        if getattr(response, "usage", None):
            u = response.usage
            print(f"  Tokens:   prompt={u.prompt_tokens}, completion={u.completion_tokens}")

    except Exception as e:
        msg = str(e)
        print(f"[错误] 请求失败: {msg}")
        print()

        if "401" in msg or "Unauthorized" in msg or "invalid" in msg.lower():
            print("  诊断: API Key 无效或已过期")
            print("  请检查 OPENAI_API_KEY 是否正确")
        elif "404" in msg or "model" in msg.lower():
            print(f"  诊断: 模型 '{model}' 不存在或无权访问")
            print("  请检查 OPENAI_MODEL 是否正确")
        elif "timeout" in msg.lower() or "Connection" in msg:
            print("  诊断: 网络连接失败")
            print(f"  请检查 OPENAI_BASE_URL={base_url} 是否可达")
        elif "403" in msg or "Forbidden" in msg:
            print("  诊断: 访问被拒绝，请确认 API Key 有调用权限")
        else:
            print("  诊断: 未知错误，详细信息:")
            import traceback
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
