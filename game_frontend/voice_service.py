# -*- coding: utf-8 -*-
"""
本地语音指令识别服务 (MVP)
===========================

使用 speech_recognition + pyaudio 实现轻量语音指令识别。
不依赖云端 API，所有识别在本机完成。

用法:
    from game_frontend.voice_service import VoiceService

    voice = VoiceService()
    if voice.is_available():
        voice.set_callback(on_command)  # on_command(text: str)
        voice.start_listening()
        ...
        voice.stop_listening()

支持的关键词:
    - 开始放松
    - 暂停
    - 继续
    - 换一个动作
    - 结束
    - 上半身
    - 全身
"""

import os
import threading
import time
from typing import Optional, Callable


# 关键词列表 + 别名映射
_KEYWORD_ALIASES = {
    "开始放松": [
        "开始", "开始训练", "开始放松一下", "开始运动",
        "开始锻炼", "开始吧", "ready", "开始活动", "开始标准放松",
    ],
    "暂停": ["暂停一下", "停一下", "休息一下", "pause", "停止"],
    "继续": ["继续吧", "继续训练", "恢复", "resume", "go on"],
    "换一个动作": [
        "换一个", "换个动作", "换动作", "下一个动作",
        "下一个", "刷一个动作", "skip", "换",
    ],
    "结束": ["结束训练", "停止训练", "退出", "finish", "stop"],
    "上半身": ["上半身放松", "上半身训练", "upper body", "上半身模式"],
    "全身": ["全身放松", "全身训练", "full body", "全身互动", "全身模式"],
    "我脖子酸": ["脖子酸", "颈部酸", "低头久了", "脖子不舒服"],
    "我肩膀酸": ["肩膀酸", "肩膀发麻", "肩颈僵硬", "肩膀不舒服"],
    "腰背僵硬": ["背酸", "腰酸", "腰背不适"],
    # 旧口令兼容：统一归一化到“开始放松”，不在 UI 中主动展示。
    "开始放松": ["开始游戏", "开始挑战", "体感挑战", "游戏模式", "start"],
    "开始办公": ["开始工作", "办公模式"],
    "返回办公模式": ["回办公", "回到办公"],
    "结束办公": ["退出办公", "停止办公"],
}

# 扁平关键词 → 标准名
_KEYWORD_MAP: dict = {}
for _std, _aliases in _KEYWORD_ALIASES.items():
    _KEYWORD_MAP[_std] = _std
    for _a in _aliases:
        _KEYWORD_MAP[_a] = _std

KEYWORDS = list(_KEYWORD_ALIASES.keys())


def _check_voice_imports() -> bool:
    """检测 speech_recognition 和 pyaudio 是否可导入"""
    try:
        import speech_recognition as sr  # noqa: F401
        import pyaudio                  # noqa: F401
        return True
    except ImportError:
        return False


def _match_keyword(text: str):
    """在文本中匹配关键词（含同义词），返回标准关键词或 None"""
    if not text:
        return None
    text_lower = text.strip().lower()
    # 精确匹配优先
    for keyword in KEYWORDS:
        if keyword in text_lower:
            return keyword
    # 别名匹配
    for alias, standard in _KEYWORD_MAP.items():
        if alias in text_lower and alias != standard:
            return standard
    return None


def _dedup_repeated_phrase(text: str) -> str:
    """去除语音识别中常见的重复/截断现象。

    「开始放松开始放松开」→「开始放松」
    「暂停暂停」→「暂停」
    「换一个动作换一个动作」→「换一个动作」
    """
    if not text:
        return text
    # 找最短的非空重复前缀
    n = len(text)
    for pat_len in range(2, n // 2 + 1):
        pat = text[:pat_len]
        # 检查 text 是否以 pat 重复开头（允许最后一段不完整）
        i = pat_len
        while i + pat_len <= n and text[i:i + pat_len] == pat:
            i += pat_len
        # 最后一段不完整但能匹配前缀 → 也是重复
        remaining = n - i
        if i > pat_len and remaining > 0 and remaining < pat_len:
            if pat[:remaining] == text[i:]:
                i = n
        if i >= n:
            return pat
    return text


def normalize_command(text: str) -> str:
    """识别文本归一化。

    1. 去除重复识别（如「开始放松开始放松开」→「开始放松」）
    2. 关键词匹配 → 标准控制指令
    3. 身体不适等非控制语句原样返回，交给上层 intent_recognizer 处理
    """
    if not text:
        return ""
    t = text.strip()
    # 先做重复短语去重
    t = _dedup_repeated_phrase(t)

    # 旧口令兼容：统一为开始放松
    if any(k in t for k in ("计分挑战", "进入挑战", "开始挑战", "挑战模式",
                             "开始游戏", "游戏模式", "体感挑战")):
        return "开始放松"
    if "开始放松" in t:
        return "开始放松"
    if any(k in t for k in ("换一个", "下一个", "刷一个动作", "换个动作", "换动作")):
        return "换一个动作"
    if "暂停" in t:
        return "暂停"
    if "继续" in t:
        return "继续"
    if ("结束" in t) or ("停止" in t):
        return "结束"
    if "上半身" in t:
        return "上半身"
    if "全身" in t:
        return "全身"
    if any(k in t for k in ("开始办公", "办公模式", "开始工作")):
        return "开始办公"
    if any(k in t for k in ("结束办公", "退出办公")):
        return "结束办公"
    # 其它（含身体不适）→ 原样返回，交给意图识别
    return t


class VoiceService:
    """本地语音指令识别服务

    在后台线程中单次监听麦克风（最长 N 秒）。
    支持录音中再次点击结束、音量回调。
    """

    def __init__(self, language: str = "zh-CN"):
        self._language = language
        self._callback: Optional[Callable[[str], None]] = None
        self._volume_callback: Optional[Callable[[float], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_requested = False
        self._last_text: str = ""
        self._last_raw_text: str = ""        # 原始识别文本（未归一化）
        self._last_normalized: str = ""      # 归一化后的业务指令/文本
        self._last_error: str = ""
        self._last_error_type: str = ""  # no_mic / timeout / no_speech / service_error / user_stop
        self._voice_available: bool = False
        self._mic_device_name: str = ""
        self._mic_device_index: Optional[int] = None

        self._recognizer = None
        self._microphone = None
        self._pa = None              # pyaudio instance (for volume monitoring)
        self._volume_stream = None   # raw pyaudio stream for RMS

        if not _check_voice_imports():
            self._last_error = (
                "speech_recognition 或 pyaudio 未安装\n"
                "pip install SpeechRecognition pyaudio"
            )
            self._last_error_type = "no_mic"
            return

        self._recognizer = self._mic_init()

    # ── 麦克风初始化 ─────────────────────────────────────

    def _list_input_devices(self):
        """用 pyaudio 列出有输入通道的设备，返回 [(index, name), ...]"""
        devices = []
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append((i, info.get("name", "")))
            pa.terminate()
        except Exception:
            pass
        return devices

    def _mic_init(self):
        """初始化麦克风，返回 Recognizer 或 None"""
        import speech_recognition as sr

        # 列出所有麦克风
        sr_names = sr.Microphone.list_microphone_names()
        pa_devices = self._list_input_devices()

        # 默认只在排障模式下打印完整设备列表，避免启动刷屏
        # 排障：$env:OFFICEFIT_VOICE_DEBUG="1"
        if os.environ.get("OFFICEFIT_VOICE_DEBUG", "").strip() == "1":
            print("[Voice] Available microphones (speech_recognition):")
            for i, name in enumerate(sr_names):
                print(f"  [{i}] {name}")
            if pa_devices:
                print("[Voice] Input-capable devices (PyAudio):")
                for idx, name in pa_devices:
                    print(f"  [{idx}] {name}")

        # 无输入设备 → 不可用
        if not pa_devices and not sr_names:
            self._last_error = "未检测到任何麦克风输入设备"
            self._last_error_type = "no_mic"
            self._voice_available = False
            return None

        # 确定使用的设备编号
        env_index = os.environ.get("OFFICEFIT_MIC_INDEX", "").strip()
        recognizer = sr.Recognizer()

        if env_index:
            # 用户指定编号
            try:
                idx = int(env_index)
                mic = sr.Microphone(device_index=idx)
                name = (sr_names[idx] if idx < len(sr_names)
                        else pa_devices[0][1] if pa_devices else f"device {idx}")
                self._microphone = mic
                self._mic_device_index = idx
                self._mic_device_name = name
                self._voice_available = True
                print(f"[Voice] Using microphone index: {idx}, name: {name} "
                      f"(OFFICEFIT_MIC_INDEX)")
                return recognizer
            except Exception as e:
                self._last_error = (
                    f"麦克风设备编号 {env_index} 不可用: {e}\n"
                    f"请检查 OFFICEFIT_MIC_INDEX 是否正确设置\n"
                    f"可用编号: {[d[0] for d in pa_devices]}"
                )
                self._last_error_type = "no_mic"
                self._voice_available = False
                return None

        # 自动模式：先试默认 → 再试第一个 pyaudio 输入设备
        # 尝试默认设备
        try:
            mic = sr.Microphone()
            self._microphone = mic
            self._mic_device_index = 0
            self._mic_device_name = sr_names[0] if sr_names else "default"
            self._voice_available = True
            print(f"[Voice] Using microphone index: {self._mic_device_index}, "
                  f"name: {self._mic_device_name} (default)")
            return recognizer
        except Exception as e:
            print(f"[Voice] 默认麦克风失败: {e}, 尝试 fallback...")

        # fallback: pyaudio 第一个输入设备
        for idx, name in pa_devices:
            try:
                mic = sr.Microphone(device_index=idx)
                self._microphone = mic
                self._mic_device_index = idx
                self._mic_device_name = name
                self._voice_available = True
                print(f"[Voice] Using microphone index: {idx}, name: {name} "
                      f"(fallback)")
                return recognizer
            except Exception:
                continue

        # 全部失败
        self._last_error = (
            "所有麦克风设备初始化失败\n"
            "可尝试手动指定: $env:OFFICEFIT_MIC_INDEX='编号'"
        )
        self._last_error_type = "no_mic"
        self._voice_available = False
        return None

    # ── 公开 API ──────────────────────────────────────────

    def is_available(self) -> bool:
        """语音识别是否可用"""
        return self._voice_available

    @property
    def mic_device_name(self) -> str:
        return self._mic_device_name

    @property
    def mic_device_index(self) -> Optional[int]:
        return self._mic_device_index

    @property
    def last_text(self) -> str:
        """最近一次识别到的原始文本"""
        return self._last_text

    @property
    def last_error(self) -> str:
        """最近一次错误信息"""
        return self._last_error

    @property
    def last_error_type(self) -> str:
        """最近一次错误类型: no_mic / timeout / no_speech / service_error / ''"""
        return self._last_error_type

    @property
    def last_raw_text(self) -> str:
        """最近一次原始识别文本（未归一化）"""
        return self._last_raw_text

    @property
    def last_normalized(self) -> str:
        """最近一次归一化后的业务指令/文本"""
        return self._last_normalized

    def set_callback(self, callback: Callable[[str], None]):
        """设置指令回调。callback 接收归一化后的指令字符串。"""
        self._callback = callback

    def set_volume_callback(self, callback: Callable[[float], None]):
        """设置音量回调。callback 接收 0.0~1.0 的 RMS 音量值。"""
        self._volume_callback = callback

    def start_listening(self):
        """启动单次监听（最长 5 秒，识别到或超时后自动停止）"""
        if not self._voice_available:
            self._last_error = "语音识别不可用，无法启动监听"
            self._last_error_type = "no_mic"
            return

        if self._running:
            return

        self._running = True
        self._stop_requested = False
        self._last_text = ""
        self._last_error = ""
        self._last_error_type = ""
        self._start_volume_monitor()
        self._thread = threading.Thread(target=self._listen_once, daemon=True)
        self._thread.start()
        print("[Voice] Listening started (max 5s)...")

    def request_stop(self):
        """请求主动结束当前录音（再次点击语音按钮时调用）"""
        self._stop_requested = True
        print("[Voice] Stop requested by user")

    def stop_listening(self):
        """停止监听"""
        self._running = False
        self._stop_volume_monitor()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[Voice] Listening stopped")

    # ── 音量监测 ──────────────────────────────────────────

    def _start_volume_monitor(self):
        """启动独立的 pyaudio 流用于实时音量监测"""
        self._stop_volume_monitor()
        try:
            import pyaudio
            self._pa = pyaudio.PyAudio()
            device_index = (self._mic_device_index
                           if self._mic_device_index is not None
                           else None)
            self._volume_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=512,
                stream_callback=self._volume_stream_callback,
            )
        except Exception as e:
            print(f"[Voice] Volume monitor init failed: {e}")
            self._volume_stream = None

    def _volume_stream_callback(self, in_data, frame_count, time_info, status):
        """pyaudio 回调：计算 RMS 音量 → 通过 volume_callback 发送"""
        try:
            import pyaudio
            import numpy as np
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            rms = np.sqrt(np.mean(audio_data ** 2)) if len(audio_data) > 0 else 0
            # 归一化到 0.0~1.0 (典型静音 ~50, 正常说话 ~2000, 裁剪到 4000)
            level = min(1.0, max(0.0, rms / 4000.0))
            if self._volume_callback:
                self._volume_callback(level)
            return (None, pyaudio.paContinue)
        except Exception:
            return (None, 0)  # pyaudio.paContinue is 0

    def _stop_volume_monitor(self):
        """停止音量监测流"""
        if self._volume_stream is not None:
            try:
                self._volume_stream.stop_stream()
                self._volume_stream.close()
            except Exception:
                pass
            self._volume_stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # ── 单次监听 ──────────────────────────────────────────

    def _listen_once(self):
        """单次监听：听一次 → 识别 → 自动停止
        支持 stop_requested：用户再次点击按钮时提前结束录音。
        """
        import speech_recognition as sr

        if not self._microphone or not self._recognizer:
            self._running = False
            self._stop_volume_monitor()
            return

        # 校准
        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as e:
            self._last_error = f"噪声校准失败: {e}"
            self._last_error_type = "no_mic"
            self._running = False
            self._stop_volume_monitor()
            return

        # 单次监听
        try:
            with self._microphone as source:
                audio = self._recognizer.listen(
                    source, timeout=5.0, phrase_time_limit=3.0)
            # 用户主动结束录音 → 仍尝试识别已捕获的音频
            if self._stop_requested:
                print("[Voice] User stopped recording, recognizing captured audio...")
            # 识别
            try:
                text = self._recognizer.recognize_google(
                    audio, language=self._language)
                self._last_error = ""
                self._last_error_type = ""
                self._last_raw_text = text
                command = normalize_command(text)
                self._last_normalized = command
                self._last_text = command
                print(f"[Voice] raw: {text}")
                if command != text:
                    print(f"[Voice] normalized: {command}")
                if command and self._callback:
                    self._callback(command)
            except sr.UnknownValueError:
                if self._stop_requested:
                    self._last_error = "录音已结束"
                    self._last_error_type = "user_stop"
                else:
                    self._last_error = "没有听清，请再试一次或使用快捷输入"
                    self._last_error_type = "no_speech"
                print(f"[Voice] {self._last_error}")
            except sr.RequestError as e:
                self._last_error = f"识别服务不可用: {e}"
                self._last_error_type = "service_error"
                print(f"[Voice] {self._last_error}")
            except OSError as e:
                self._last_error = f"麦克风被占用: {e}"
                self._last_error_type = "no_mic"
                print(f"[Voice] {self._last_error}")
        except sr.WaitTimeoutError:
            if self._stop_requested:
                self._last_error = "录音已结束"
                self._last_error_type = "user_stop"
            else:
                self._last_error = "未检测到语音，请再试一次"
                self._last_error_type = "timeout"
            print(f"[Voice] {self._last_error}")
        except OSError as e:
            self._last_error = f"麦克风异常: {e}"
            self._last_error_type = "no_mic"
            print(f"[Voice] {self._last_error}")
        except Exception as e:
            self._last_error = f"监听异常: {e}"
            self._last_error_type = "service_error"
            print(f"[Voice] {self._last_error}")

        self._running = False
        self._stop_volume_monitor()

    def simulate_command(self, text: str):
        """快捷输入（用于演示 / 无麦克风环境的备用输入）"""
        command = normalize_command(text)
        self._last_raw_text = text
        self._last_normalized = command
        self._last_text = command
        print(f"[Voice] Quick input raw: {text}")
        if command != text:
            print(f"[Voice] Quick input normalized: {command}")
        if command and self._callback:
            self._callback(command)
