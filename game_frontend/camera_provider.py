# -*- coding: utf-8 -*-
"""
统一相机输入层 — Orbbec 3D 主用 + USB 摄像头降级兼容
=====================================================

用法:
    from game_frontend.camera_provider import create_camera_provider

    # mode='auto' → 自动: Orbbec 优先, 失败 → webcam
    # mode='orbbec' → 强制 Orbbec, 失败则报错不降级
    # mode='webcam' → 强制 USB 摄像头
    provider = create_camera_provider(mode='auto')

    try:
        provider.start()
    except RuntimeError:
        # 启动失败处理
        ...

    while True:
        packet = provider.read(timeout_ms=100)
        ...

    provider.stop()

环境变量:
    OFFICEFIT_CAMERA=orbbec   强制 Orbbec (不降级)
    OFFICEFIT_CAMERA=webcam   强制 USB 摄像头
    未设置 → auto 模式
"""

import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# 确保可以 import 项目根目录的 utils
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)


# ═══════════════════════════════════════════════════════════════
# FramePacket
# ═══════════════════════════════════════════════════════════════

class FramePacket:
    """统一帧数据包"""

    __slots__ = (
        "color_bgr", "depth_mm", "source",
        "depth_available", "timestamp", "intrinsics",
    )

    def __init__(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
        source: str = "unknown",
        depth_available: bool = False,
        timestamp: Optional[float] = None,
        intrinsics: Optional[dict] = None,
    ):
        self.color_bgr = color_bgr
        self.depth_mm = depth_mm
        self.source = source
        self.depth_available = depth_available
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.intrinsics = intrinsics or {}

    def __repr__(self):
        h, w = self.color_bgr.shape[:2]
        return (
            f"FramePacket(source={self.source}, "
            f"color=({w}x{h}), depth={self.depth_available})"
        )


# ═══════════════════════════════════════════════════════════════
# BaseCameraProvider
# ═══════════════════════════════════════════════════════════════

class BaseCameraProvider(ABC):
    """相机抽象基类"""

    def __init__(self):
        self._fallback_reason: Optional[str] = None
        self._mode: str = "auto"

    @abstractmethod
    def start(self):
        """启动相机; 失败抛出 RuntimeError"""

    @abstractmethod
    def read(self, timeout_ms: int = 1000) -> Optional[FramePacket]:
        """读取一帧; 超时或无帧返回 None"""

    @abstractmethod
    def stop(self):
        """停止并释放资源"""

    @abstractmethod
    def is_opened(self) -> bool:
        """相机是否已打开"""

    @abstractmethod
    def get_status(self) -> dict:
        """
        返回:
          source, depth_available, resolution, intrinsics,
          fallback_reason, error_message, mode
        """

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @fallback_reason.setter
    def fallback_reason(self, value: str):
        self._fallback_reason = value

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        self._mode = value


# ═══════════════════════════════════════════════════════════════
# OrbbecCameraProvider
# ═══════════════════════════════════════════════════════════════

_ORBBEC_SDK_AVAILABLE = False
_ORBBEC_IMPORT_ERROR = None

try:
    from pyorbbecsdk import (           # noqa: E402
        Pipeline, Config, AlignFilter,
        OBSensorType, OBFormat, OBStreamType,
        Context,
    )
    _ORBBEC_SDK_AVAILABLE = True
except ImportError as e:
    _ORBBEC_IMPORT_ERROR = str(e)


class CameraUnavailableError(RuntimeError):
    """相机不可用异常 — 用于 auto 模式触发 fallback"""
    pass


def _orbbec_sdk_available() -> bool:
    return _ORBBEC_SDK_AVAILABLE


def _orbbec_import_error() -> Optional[str]:
    return _ORBBEC_IMPORT_ERROR


class OrbbecCameraProvider(BaseCameraProvider):
    """Orbbec 3D 深度相机 (RGB-D)"""

    def __init__(self):
        super().__init__()
        if not _ORBBEC_SDK_AVAILABLE:
            raise CameraUnavailableError(
                f"pyorbbecsdk 不可用: {_ORBBEC_IMPORT_ERROR}\n"
                f"请执行: pip install pyorbbecsdk2"
            )

        self._pipeline: Optional["Pipeline"] = None
        self._align_filter = None
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._cam_w: int = 0
        self._cam_h: int = 0
        self._opened: bool = False
        self._error_message: Optional[str] = None

    @staticmethod
    def _check_device_connected() -> int:
        """检测有多少台 Orbbec 设备已连接。不抛异常，返回 0 表示无设备。"""
        try:
            ctx = Context()
            device_list = ctx.query_devices()
            return device_list.get_count()
        except Exception:
            return 0

    # ── start ──────────────────────────────────────────────

    def start(self):
        self._error_message = None

        # ── 设备存在性检查（必须在 Pipeline.start() 之前，防止阻塞）──
        device_count = self._check_device_connected()
        if device_count == 0:
            self._error_message = "未检测到 Orbbec 3D 相机设备"
            raise CameraUnavailableError(self._error_message)
        print(f"[Camera] 检测到 {device_count} 台 Orbbec 设备")

        try:
            print("[Camera] Starting Orbbec 3D camera...")
            self._pipeline = Pipeline()
            config = Config()

            color_profiles = self._pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR)
            depth_profiles = self._pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR)

            config.enable_stream(
                color_profiles.get_default_video_stream_profile())
            config.enable_stream(
                depth_profiles.get_default_video_stream_profile())

            self._pipeline.start(config)

            cp = color_profiles.get_default_video_stream_profile()
            self._cam_w = cp.get_width()
            self._cam_h = cp.get_height()

            self._align_filter = AlignFilter(
                align_to_stream=OBStreamType.COLOR_STREAM)

            camera_param = self._pipeline.get_camera_param()
            rgb_intr = camera_param.rgb_intrinsic
            self._fx = rgb_intr.fx
            self._fy = rgb_intr.fy
            self._cx = rgb_intr.cx
            self._cy = rgb_intr.cy

            self._opened = True
            print(
                f"[Camera] Orbbec started ({self._cam_w}x{self._cam_h}), "
                f"fx={self._fx:.1f} fy={self._fy:.1f}"
            )
        except Exception as e:
            self._opened = False
            self._error_message = f"Orbbec 启动失败: {e}"
            raise RuntimeError(self._error_message) from e

    # ── read ───────────────────────────────────────────────

    def read(self, timeout_ms: int = 1000) -> Optional[FramePacket]:
        if not self._opened or self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                return None

            frames = self._align_filter.process(frames)
            if frames is None:
                return None

            fs = frames.as_frame_set()
            color_frame = fs.get_color_frame()
            depth_frame = fs.get_depth_frame()

            if color_frame is None:
                return None

            from utils import frame_to_bgr_image
            img_bgr = frame_to_bgr_image(color_frame)
            if img_bgr is None:
                return None

            # 深度帧
            depth_mm = None
            depth_available = False
            if depth_frame is not None:
                try:
                    depth_data = np.frombuffer(
                        depth_frame.get_data(), dtype=np.uint16
                    ).reshape((depth_frame.get_height(),
                               depth_frame.get_width()))
                    scale = depth_frame.get_depth_scale()
                    depth_mm = (depth_data.astype(np.float32) * scale).astype(
                        np.uint16)
                    depth_available = True
                except Exception:
                    pass

            return FramePacket(
                color_bgr=img_bgr,
                depth_mm=depth_mm,
                source="orbbec",
                depth_available=depth_available,
                timestamp=time.time(),
                intrinsics={
                    "fx": self._fx, "fy": self._fy,
                    "cx": self._cx, "cy": self._cy,
                },
            )
        except Exception as e:
            print(f"[Camera] Orbbec read error: {e}")
            return None

    # ── stop ───────────────────────────────────────────────

    def stop(self):
        self._opened = False
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
                self._pipeline = None
                print("[Camera] Orbbec pipeline stopped")
        except Exception as e:
            print(f"[Camera] Orbbec stop error: {e}")

    # ── helpers ────────────────────────────────────────────

    def is_opened(self) -> bool:
        return self._opened

    def get_status(self) -> dict:
        return {
            "source": "orbbec",
            "depth_available": self._opened,
            "resolution": (self._cam_w, self._cam_h),
            "intrinsics": {
                "fx": self._fx, "fy": self._fy,
                "cx": self._cx, "cy": self._cy,
            } if self._opened else None,
            "fallback_reason": self._fallback_reason,
            "error_message": self._error_message,
            "mode": self._mode,
        }


# ═══════════════════════════════════════════════════════════════
# WebcamCameraProvider
# ═══════════════════════════════════════════════════════════════

class WebcamCameraProvider(BaseCameraProvider):
    """USB 普通摄像头 (仅 RGB)"""

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self._camera_index = camera_index
        self._cap: Optional["cv2.VideoCapture"] = None
        self._opened: bool = False
        self._width: int = 0
        self._height: int = 0
        self._error_message: Optional[str] = None

    # ── start ──────────────────────────────────────────────

    def start(self):
        self._error_message = None
        print(f"[Camera] Starting USB webcam (index={self._camera_index})...")
        try:
            import cv2
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"无法打开摄像头 index={self._camera_index}")

            # 尝试设置 1280x720
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # 读取一帧确认分辨率
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._height, self._width = frame.shape[:2]
            else:
                self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self._width <= 0 or self._height <= 0:
                raise RuntimeError(
                    f"摄像头分辨率无效 ({self._width}x{self._height})")

            self._opened = True
            print(
                f"[Camera] Webcam started ({self._width}x{self._height})"
            )
        except Exception as e:
            self._opened = False
            self._error_message = f"USB 摄像头启动失败: {e}"
            raise RuntimeError(self._error_message) from e

    # ── read ───────────────────────────────────────────────

    def read(self, timeout_ms: int = 1000) -> Optional[FramePacket]:
        if not self._opened or self._cap is None:
            return None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None

            return FramePacket(
                color_bgr=frame,
                depth_mm=None,
                source="webcam",
                depth_available=False,
                timestamp=time.time(),
                intrinsics=None,
            )
        except Exception as e:
            print(f"[Camera] Webcam read error: {e}")
            return None

    # ── stop ───────────────────────────────────────────────

    def stop(self):
        self._opened = False
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                print("[Camera] Webcam released")
        except Exception as e:
            print(f"[Camera] Webcam stop error: {e}")

    # ── helpers ────────────────────────────────────────────

    def is_opened(self) -> bool:
        return self._opened

    def get_status(self) -> dict:
        return {
            "source": "webcam",
            "depth_available": False,
            "resolution": (self._width, self._height),
            "intrinsics": None,
            "fallback_reason": self._fallback_reason,
            "error_message": self._error_message,
            "mode": self._mode,
        }


# ═══════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════

def _read_env_mode() -> str:
    """读取 OFFICEFIT_CAMERA 环境变量, 返回 'auto' / 'orbbec' / 'webcam'"""
    env = os.environ.get("OFFICEFIT_CAMERA", "").strip().lower()
    if env in ("orbbec", "webcam"):
        return env
    if env:
        print(f"[Camera] 未知 OFFICEFIT_CAMERA={env}, 回退到 auto")
    return "auto"


def create_camera_provider(mode: str = "auto") -> BaseCameraProvider:
    """
    创建相机实例 (未 start)。

    mode 取值:
      'auto'    — 自动: 优先 Orbbec → 失败降级 webcam
                   OFFICEFIT_CAMERA 未设置时的默认行为
      'orbbec'  — 强制 Orbbec 3D 相机, 失败直接报错
      'webcam'  — 强制 USB 摄像头

    注意: 返回的 provider 尚未 start(), 调用方需自行 start()。
           auto 模式下的降级发生在 backend_thread.py 的 try/except 中。
    """
    # 环境变量覆盖 mode 参数
    env_mode = _read_env_mode()
    if env_mode != "auto":
        mode = env_mode
        print(f"[Camera] OFFICEFIT_CAMERA={mode} (环境变量强制)")

    # ── 强制 Orbbec ──
    if mode == "orbbec":
        print("[Camera] Mode: orbbec (强制, 不降级)")
        provider = OrbbecCameraProvider()
        provider.mode = "orbbec"
        return provider

    # ── 强制 webcam ──
    if mode == "webcam":
        print("[Camera] Mode: webcam (强制, 不使用 Orbbec)")
        provider = WebcamCameraProvider(camera_index=0)
        provider.mode = "webcam"
        return provider

    # ── auto 模式 ──
    print("[Camera] Mode: auto (优先 Orbbec → 降级 webcam)")
    if _ORBBEC_SDK_AVAILABLE:
        try:
            provider = OrbbecCameraProvider()
            provider.mode = "auto"
            print("[Camera] 返回 OrbbecCameraProvider, "
                  "start() 时将检测设备并连接")
            return provider
        except CameraUnavailableError as e:
            print(f"[Camera] Orbbec SDK 可用但初始化失败: {e}")
    else:
        print(f"[Camera] pyorbbecsdk 未安装: {_ORBBEC_IMPORT_ERROR}")

    # 降级到 webcam
    print("[Camera] Fallback to WebcamCameraProvider")
    provider = WebcamCameraProvider(camera_index=0)
    provider.mode = "auto"
    provider.fallback_reason = (
        f"Orbbec 不可用"
        + (f": {_ORBBEC_IMPORT_ERROR}" if _ORBBEC_IMPORT_ERROR else "")
        + " → 已降级到 USB 摄像头"
    )
    return provider
