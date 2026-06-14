#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen / DashScope API 连通性测试
================================

使用 OpenAI 兼容模式调用阿里百炼 DashScope API。
不修改主程序逻辑，仅用于验证 API key 和网络连通性。

环境变量:
    DASHSCOPE_API_KEY      (必需) 阿里百炼 API Key
    DASHSCOPE_BASE_URL      默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    OFFICEFIT_QWEN_MODEL    默认 qwen-plus

用法:
    python scripts/test_qwen_api.py

首次使用请在 PowerShell 中设置:
    $env:DASHSCOPE_API_KEY="sk-..."
"""

import os
import sys
import traceback


def main():
    # ── 读取配置 ──
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()
    model = os.environ.get("OFFICEFIT_QWEN_MODEL", "qwen-plus").strip()

    print("=" * 50)
    print("  Qwen / DashScope API 连通性测试")
    print("=" * 50)
    print(f"  Base URL: {base_url}")
    print(f"  Model:    {model}")
    print()

    # ── API Key 检查 ──
    if not api_key:
        print("[错误] 未设置 DASHSCOPE_API_KEY")
        print()
        print("请在 PowerShell 中设置:")
        print('  $env:DASHSCOPE_API_KEY="sk-你的key"')
        print()
        print("然后重新运行: python scripts/test_qwen_api.py")
        sys.exit(1)

    print(f"  API Key:  {api_key[:8]}...{api_key[-4:]} (已隐藏中间部分)")
    print()

    # ── 检查 openai 包 ──
    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] openai Python SDK 未安装")
        print()
        print("请执行: pip install openai")
        sys.exit(1)

    # ── 创建客户端 ──
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"[错误] 创建 OpenAI 客户端失败: {e}")
        sys.exit(1)

    # ── 发送测试请求 ──
    print("  发送测试请求...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "请只回复：Qwen API 测试成功",
                },
            ],
            max_tokens=50,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        print(f"  模型回复: {content}")
        print()
        print("[通过] Qwen API 测试成功!")
        print(f"  模型: {response.model}")
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"  Tokens: prompt={usage.prompt_tokens}, "
                  f"completion={usage.completion_tokens}")

    except Exception as e:
        error_msg = str(e)
        print(f"[错误] 请求失败: {error_msg}")
        print()

        # 常见错误诊断
        if "401" in error_msg or "Unauthorized" in error_msg or "invalid" in error_msg.lower():
            print("  诊断: API Key 无效或已过期")
            print("  请检查 DASHSCOPE_API_KEY 是否正确")
        elif "404" in error_msg or "model" in error_msg.lower():
            print(f"  诊断: 模型 '{model}' 不存在或无权访问")
            print("  请检查 OFFICEFIT_QWEN_MODEL 是否正确")
            print("  可用模型: qwen-plus, qwen-turbo, qwen-max 等")
        elif "timeout" in error_msg.lower() or "ConnectionError" in error_msg:
            print("  诊断: 网络连接失败")
            print("  请检查网络和 DASHSCOPE_BASE_URL")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("  诊断: 访问被拒绝")
            print("  请确认 API Key 有调用权限")
        else:
            print("  诊断: 未知错误，详细如下:")
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
