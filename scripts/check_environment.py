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
# 4. 摄像头检查
# ============================================================================

def check_camera():
    print("\n" + "=" * 60)
    print("  摄像头检查")
    print("=" * 60)

    # 检查 Orbbec 相机
    ok, detail = _check_import("pyorbbecsdk")
    if ok:
        try:
            from pyorbbecsdk import Context
            ctx = Context()
            device_list = ctx.query_devices()
            count = device_list.get_count()
            if count > 0:
                names = []
                for i in range(count):
                    dev = device_list.get_device_by_index(i)
                    info = dev.get_device_info()
                    names.append(f"{info.get_name()} (SN:{info.get_serial_number()})")
                _result(_ICON_PASS, f"Orbbec 3D 相机: 检测到 {count} 台", "; ".join(names))
            else:
                _result(_ICON_FAIL, "Orbbec 3D 相机: 未检测到设备",
                        "请连接 Orbbec 相机并确认设备管理器中已识别")
        except Exception as e:
            _result(_ICON_FAIL, "Orbbec 3D 相机: 查询失败", str(e))
    else:
        _result(_ICON_FAIL, "Orbbec 3D 相机",
                "pyorbbecsdk 未正确安装（pip 包名: pyorbbecsdk2），无法查询设备")

    # 检查普通 USB 摄像头（webcam fallback 预览）
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                _result(_ICON_PASS, f"USB 摄像头 (index=0): {w}x{h}",
                        "普通摄像头可用（当前版本暂不使用，未来版本会启用）")
            else:
                _result(_ICON_WARN, "USB 摄像头 (index=0): 已打开但无法读取帧",
                        "可能是虚拟摄像头或权限不足")
            cap.release()
        else:
            _result(_ICON_WARN, "USB 摄像头 (index=0): 未找到",
                    "无普通摄像头可用（不影响 Orbbec 模式运行）")
    except Exception as e:
        _result(_ICON_WARN, "USB 摄像头检查失败", str(e))


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
            print(f"  {idx}. 连接 Orbbec 3D 相机并确认设备管理器已识别")
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
    check_camera()

    n_fail = print_summary()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
