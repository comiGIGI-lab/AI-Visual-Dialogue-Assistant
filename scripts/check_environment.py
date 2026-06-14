#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署环境检查脚本
=================

用途：在启动项目前检查所有依赖、模型文件、硬件是否就绪。

使用方法：
    python scripts/check_environment.py

输出：
    - 每一项输出 [PASS] / [WARN] / [FAIL]
    - 末尾给出总结和修复建议

退出码：
    0 — 全部通过（无 FAIL）
    1 — 存在 FAIL 项
"""

import sys
import os
from pathlib import Path

# 确保 Windows 控制台使用 UTF-8 编码，避免中文和特殊字符乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================================
# 工具函数
# ============================================================================

_ICON_PASS = "[PASS]"
_ICON_WARN = "[WARN]"
_ICON_FAIL = "[FAIL]"

_passes = []
_warnings = []
_failures = []


def _result(icon, label, detail=""):
    line = f"  {icon}  {label}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    if icon == _ICON_FAIL:
        _failures.append(label)
    elif icon == _ICON_WARN:
        _warnings.append(label)
    else:
        _passes.append(label)


def _check_import(module_name, import_stmt=None, optional=False, fail_detail=""):
    """尝试导入一个模块，返回 (success, detail_str)"""
    if import_stmt is None:
        import_stmt = f"import {module_name}"
    try:
        exec(import_stmt)
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ============================================================================
# 1. Python 版本
# ============================================================================

def check_python():
    print("\n" + "=" * 60)
    print("  Python 环境")
    print("=" * 60)

    version = sys.version
    major, minor = sys.version_info[:2]
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  可执行文件:  {sys.executable}")
    print(f"  架构:         {'64-bit' if sys.maxsize > 2**32 else '32-bit'}")

    if (major, minor) < (3, 9):
        _result(_ICON_FAIL, "Python 版本过低", f"需要 >= 3.9，当前 {major}.{minor}")
    elif (major, minor) > (3, 13):
        _result(_ICON_WARN, "Python 版本较新", f"当前 {major}.{minor}，未充分测试，建议 3.10~3.12")
    else:
        _result(_ICON_PASS, f"Python {major}.{minor} 版本兼容")

    if sys.maxsize <= 2**32:
        _result(_ICON_FAIL, "需要 64 位 Python", "32 位不支持 onnxruntime/mediapipe")
    else:
        _result(_ICON_PASS, "64 位 Python")


# ============================================================================
# 2. 核心依赖检查
# ============================================================================

def check_dependencies():
    print("\n" + "=" * 60)
    print("  依赖库检查")
    print("=" * 60)

    # --- PySide6 ---
    ok, detail = _check_import("PySide6")
    if ok:
        import PySide6
        _result(_ICON_PASS, "PySide6", f"版本 {PySide6.__version__}")
    else:
        _result(_ICON_FAIL, "PySide6", f"导入失败: {detail}\n          pip install PySide6")

    # --- cv2 (opencv-python) ---
    ok, detail = _check_import("cv2")
    if ok:
        import cv2
        _result(_ICON_PASS, "opencv-python (cv2)", f"版本 {cv2.__version__}")
    else:
        _result(_ICON_FAIL, "opencv-python (cv2)", f"导入失败: {detail}\n          pip install opencv-python")

    # --- numpy ---
    ok, detail = _check_import("numpy")
    if ok:
        import numpy as np
        _result(_ICON_PASS, "numpy", f"版本 {np.__version__}")
    else:
        _result(_ICON_FAIL, "numpy", f"导入失败: {detail}\n          pip install numpy")

    # --- onnxruntime ---
    ok, detail = _check_import("onnxruntime")
    if ok:
        try:
            import onnxruntime as ort
            version = ort.__version__
            providers = ort.get_available_providers()
            gpu_providers = [p for p in providers if p != 'CPUExecutionProvider']
            provider_str = ", ".join(providers)
            _result(_ICON_PASS, f"onnxruntime {version}", f"可用 Providers: {provider_str}")
            if not gpu_providers:
                _result(_ICON_WARN, "onnxruntime 仅 CPU 推理",
                        "安装 onnxruntime-directml 可启用 GPU 加速")
        except Exception as e:
            _result(_ICON_FAIL, "onnxruntime", f"检查 providers 失败: {e}")
    else:
        _result(_ICON_FAIL, "onnxruntime", f"导入失败: {detail}\n          pip install onnxruntime")

    # --- mediapipe ---
    ok, detail = _check_import("mediapipe")
    if ok:
        try:
            import mediapipe as mp
            _result(_ICON_PASS, "mediapipe", f"版本 {mp.__version__}")
        except Exception as e:
            _result(_ICON_FAIL, "mediapipe", f"初始化失败: {e}")
    else:
        _result(_ICON_FAIL, "mediapipe",
                f"导入失败: {detail}\n          pip install mediapipe\n"
                "          如需降级 protobuf: pip install protobuf==4.25.9")

    # --- PIL (pillow) ---
    ok, detail = _check_import("PIL")
    if ok:
        from PIL import __version__ as pil_version
        _result(_ICON_PASS, "pillow (PIL)", f"版本 {pil_version}")
    else:
        _result(_ICON_FAIL, "pillow (PIL)", f"导入失败: {detail}\n          pip install pillow")

    # --- pygame ---
    ok, detail = _check_import("pygame")
    if ok:
        import pygame
        _result(_ICON_PASS, "pygame", f"版本 {pygame.version.ver}")
    else:
        _result(_ICON_FAIL, "pygame", f"导入失败: {detail}\n          pip install pygame")

    # --- pyorbbecsdk ---
    # 注意：pip 包名为 pyorbbecsdk2，安装后 import pyorbbecsdk
    ok, detail = _check_import("pyorbbecsdk")
    if ok:
        _result(_ICON_PASS, "pyorbbecsdk (Orbbec SDK)")
    else:
        _result(_ICON_FAIL, "pyorbbecsdk (Orbbec SDK)",
                f"导入失败: {detail}\n"
                "          pip 包名是 pyorbbecsdk2，安装后 import pyorbbecsdk\n"
                "          pip install pyorbbecsdk2\n"
                "          参考: https://github.com/orbbec/pyorbbecsdk")

    # --- open3d (可选) ---
    ok, detail = _check_import("open3d")
    if ok:
        import open3d as o3d
        _result(_ICON_PASS, "open3d (可选)", f"版本 {o3d.__version__}")
    else:
        _result(_ICON_WARN, "open3d (可选)", "未安装 — 3D 骨架可视化不可用，不影响游戏\n"
                "          pip install open3d")


# ============================================================================
# 3. 资源文件检查
# ============================================================================

def _find_resource(filename, search_dirs):
    """在多个搜索目录中查找文件，返回第一个匹配的路径或 None"""
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def check_resources():
    print("\n" + "=" * 60)
    print("  资源文件检查")
    print("=" * 60)

    # 确定项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 搜索路径：models/ 目录 和 项目根目录
    search_dirs = [
        os.path.join(project_root, "models"),
        project_root,
    ]

    # 必需模型文件
    required_models = [
        ("yolo26n.onnx", "YOLOv26-nano 人体检测"),
        ("pose_landmarker_lite.task", "MediaPipe 姿态估计"),
        ("coco.names", "YOLO 80 类 COCO 标签"),
    ]

    # 可选模型文件
    optional_models = [
        ("yolo11n.onnx", "YOLOv11-nano 人体检测（备选）"),
    ]

    for filename, desc in required_models:
        found_path = _find_resource(filename, search_dirs)
        if found_path:
            size_kb = os.path.getsize(found_path) / 1024
            rel = os.path.relpath(found_path, project_root)
            _result(_ICON_PASS, f"{desc}", f"{rel} ({size_kb:.1f} KB)")
        else:
            searched = " 或 ".join(
                os.path.relpath(os.path.join(d, filename), project_root)
                for d in search_dirs
            )
            _result(_ICON_FAIL, f"{desc}",
                    f"未找到 {filename}，请在 {searched} 放置该文件")

    for filename, desc in optional_models:
        found_path = _find_resource(filename, search_dirs)
        if found_path:
            size_kb = os.path.getsize(found_path) / 1024
            rel = os.path.relpath(found_path, project_root)
            _result(_ICON_PASS, f"{desc}", f"{rel} ({size_kb:.1f} KB)")
        else:
            _result(_ICON_WARN, f"{desc}", f"未找到 {filename}（非必需）")

    # 检查中文字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    found_font = None
    for fp in font_paths:
        if os.path.isfile(fp):
            found_font = fp
            break
    if found_font:
        font_name = os.path.basename(found_font)
        _result(_ICON_PASS, "中文字体", font_name)
    else:
        _result(_ICON_WARN, "中文字体",
                "未找到微软雅黑/黑体/宋体，中文可能显示为方块")


# ============================================================================
# 3.5. 麦克风检查
# ============================================================================

def check_microphone():
    print("\n" + "=" * 60)
    print("  麦克风检查")
    print("=" * 60)

    mic_ok = False

    # speech_recognition
    ok, detail = _check_import("speech_recognition")
    if ok:
        _result(_ICON_PASS, "speech_recognition 已安装")
    else:
        _result(_ICON_WARN, "speech_recognition 未安装",
                "语音识别不可用，可使用模拟语音按钮\n"
                "          pip install SpeechRecognition")

    # pyaudio
    ok, detail = _check_import("pyaudio")
    if ok:
        _result(_ICON_PASS, "pyaudio 已安装")
    else:
        _result(_ICON_WARN, "pyaudio 未安装",
                "语音识别不可用，可使用模拟语音按钮\n"
                "          pip install pyaudio")

    # 列出麦克风设备
    try:
        import speech_recognition as sr
        sr_names = sr.Microphone.list_microphone_names()
        if sr_names:
            mic_ok = True
            print(f"  speech_recognition 麦克风 ({len(sr_names)} 个):")
            for i, name in enumerate(sr_names):
                print(f"    [{i}] {name}")
        else:
            print("  speech_recognition 未检测到麦克风")
    except Exception as e:
        print(f"  speech_recognition 麦克风列表获取失败: {e}")

    # PyAudio 输入设备
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        inputs = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                inputs.append((i, info.get("name", "")))
        pa.terminate()
        if inputs:
            mic_ok = True
            print(f"  PyAudio 输入设备 ({len(inputs)} 个):")
            for idx, name in inputs[:10]:
                print(f"    [{idx}] {name}")
        else:
            print("  PyAudio 未检测到输入设备")
    except Exception as e:
        print(f"  PyAudio 设备列表获取失败: {e}")

    # OFFICEFIT_MIC_INDEX 检查
    env_idx = os.environ.get("OFFICEFIT_MIC_INDEX", "").strip()
    if env_idx:
        try:
            idx = int(env_idx)
            if 0 <= idx < len(sr_names):
                _result(_ICON_PASS, f"OFFICEFIT_MIC_INDEX={idx} 有效",
                        f"设备: {sr_names[idx]}")
                mic_ok = True
            else:
                _result(_ICON_WARN,
                        f"OFFICEFIT_MIC_INDEX={idx} 超出范围",
                        f"有效编号: 0~{len(sr_names)-1}")
        except ValueError:
            _result(_ICON_WARN,
                    f"OFFICEFIT_MIC_INDEX={env_idx} 不是有效数字")

    if mic_ok:
        _result(_ICON_PASS, "至少有一个麦克风输入设备可用")
    else:
        _result(_ICON_WARN, "未检测到麦克风输入设备",
                "语音识别不可用，可使用模拟语音按钮\n"
                "          设置: $env:OFFICEFIT_MIC_INDEX='编号'")

    return mic_ok


# ============================================================================
# 4. 摄像头检查
# ============================================================================

def check_camera():
    print("\n" + "=" * 60)
    print("  摄像头检查")
    print("=" * 60)

    orbbec_ok = False
    webcam_ok = False

    # 检查 Orbbec 相机
    ok, detail = _check_import("pyorbbecsdk")
    if ok:
        try:
            from pyorbbecsdk import Context
            ctx = Context()
            device_list = ctx.query_devices()
            count = device_list.get_count()
            if count > 0:
                orbbec_ok = True
                names = []
                for i in range(count):
                    dev = device_list.get_device_by_index(i)
                    info = dev.get_device_info()
                    names.append(f"{info.get_name()} (SN:{info.get_serial_number()})")
                _result(_ICON_PASS, f"Orbbec 3D 相机: 检测到 {count} 台", "; ".join(names))
            else:
                _result(_ICON_WARN, "Orbbec 3D 相机: 未检测到设备",
                        "Orbbec 设备未连接，将尝试降级到 USB 摄像头")
        except Exception as e:
            _result(_ICON_WARN, "Orbbec 3D 相机: 查询失败", str(e))
    else:
        _result(_ICON_WARN, "Orbbec 3D 相机",
                f"pyorbbecsdk 未安装（pip 包名: pyorbbecsdk2）\n"
                "          Orbbec 不可用，将尝试降级到 USB 摄像头")

    # 检查普通 USB 摄像头
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                webcam_ok = True
                h, w = frame.shape[:2]
                _result(_ICON_PASS, f"USB 摄像头 (index=0): {w}x{h}",
                        "RGB 兼容模式可用")
            else:
                _result(_ICON_WARN, "USB 摄像头 (index=0): 已打开但无法读取帧",
                        "可能是虚拟摄像头或权限不足")
            cap.release()
        else:
            _result(_ICON_WARN, "USB 摄像头 (index=0): 未找到",
                    "无 USB 摄像头可用")
    except Exception as e:
        _result(_ICON_WARN, "USB 摄像头检查失败", str(e))

    # ── 相机模式总结 ──
    print()
    if orbbec_ok:
        print("  ── 相机运行模式: Full mode (Orbbec 3D 深度相机)")
        print("     支持: RGB + Depth、YOLO、MediaPipe、3D 骨架、活体检测")
    elif webcam_ok:
        print("  ── 相机运行模式: Fallback mode (USB 摄像头 RGB 兼容)")
        print("     支持: RGB 图像、YOLO、MediaPipe、互动游戏/放松流程")
        print("     不支持: 深度信息、3D 骨架、活体检测")
    else:
        print("  ── 相机运行模式: No camera available")
        print("     请连接 Orbbec 3D 相机或 USB 摄像头")

    return orbbec_ok, webcam_ok


# ============================================================================
# 5. 总结
# ============================================================================

def print_summary():
    print("\n")
    print("=" * 60)
    print("  检查总结")
    print("=" * 60)

    n_pass = len(_passes)
    n_warn = len(_warnings)
    n_fail = len(_failures)

    print(f"\n  [OK]  通过:   {n_pass} 项")
    if _passes:
        for item in _passes:
            print(f"         - {item}")

    print(f"\n  [!!]  警告:   {n_warn} 项")
    if _warnings:
        for item in _warnings:
            print(f"         - {item}")
    else:
        print(f"         (无)")

    print(f"\n  [XX]  失败:   {n_fail} 项")
    if _failures:
        for item in _failures:
            print(f"         - {item}")
    else:
        print(f"         (无)")

    print()

    if n_fail == 0:
        print("  [DONE] 所有必需项检查通过！可以启动游戏：")
        print()
        print("     python run_game_frontend.py")
        print()
    else:
        print("  [FIX] 存在未通过项，请按以下顺序修复：")
        print()
        idx = 1
        print(f"  {idx}. 确认 Python 依赖已安装:")
        print(f"     pip install -r requirements.txt")
        idx += 1
        if any("pyorbbecsdk" in f for f in _failures):
            print(f"  {idx}. 安装 Orbbec SDK: pip install pyorbbecsdk2（安装后 import pyorbbecsdk）")
            print(f"     参考: https://github.com/orbbec/pyorbbecsdk")
            idx += 1
        if any("人体检测" in f or "姿态估计" in f or "COCO" in f or "模型" in f for f in _failures):
            print(f"  {idx}. 将缺失的模型文件放入 models/ 目录或项目根目录")
            idx += 1
        if any("相机" in f for f in _failures):
            print(f"  {idx}. 未检测到任何可用摄像头 — 请连接 Orbbec 3D 相机或 USB 摄像头")
            idx += 1
        print()
        print(f"  修复后重新运行: python scripts/check_environment.py")
        print()

    return n_fail


# ============================================================================
# 入口
# ============================================================================

def main():
    print()
    print("=" * 60)
    print("  环境检查 —")
    print("  基于 Orbbec 3D 视觉与姿态识别的办公久坐放松 AI 视觉对话助手")
    print("=" * 60)

    check_python()
    check_dependencies()
    check_resources()
    mic_ok = check_microphone()
    orbbec_ok, webcam_ok = check_camera()

    # 调整相机相关 FAIL → 如果 webcam 可用，Orbbec 缺失不算硬失败
    # (check_camera 已经将 Orbbec 缺失改为 WARN，这里处理遗留的 FAIL)
    orbbec_fail_items = [f for f in _failures if "Orbbec" in f or "相机" in f]
    if orbbec_fail_items and webcam_ok:
        for item in orbbec_fail_items:
            _failures.remove(item)
            _warnings.append(item + "（已降级到 USB 摄像头兼容模式）")

    n_fail = print_summary()

    # 额外打印相机可用性提示
    if not mic_ok:
        print("  ──────────────────────────────────────────────")
        print("  [INFO] 麦克风不可用，可使用模拟语音按钮进行演示。")
        print()

    if not orbbec_ok:
        print("  ──────────────────────────────────────────────")
        if webcam_ok:
            print("  [INFO] 没有 Orbbec 相机时仍可运行，但深度、3D 骨架、活体检测不可用。")
            print("  启动命令: python run_game_frontend.py")
            print("  强制测试 webcam 模式:")
            print("    Windows PowerShell:")
            print('      $env:OFFICEFIT_CAMERA="webcam"; python run_game_frontend.py')
        else:
            print("  [INFO] 没有检测到任何可用摄像头，请连接 Orbbec 或 USB 摄像头。")
        print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
