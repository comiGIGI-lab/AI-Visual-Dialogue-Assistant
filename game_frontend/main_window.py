"""主窗口：QStackedWidget 管理 menu → difficulty → game 流程"""
import os
import datetime
import winsound

from PySide6.QtWidgets import (QMainWindow, QStackedWidget, QMessageBox,
                                QApplication)
from PySide6.QtCore import Qt, QTimer, Signal, QObject

from game_frontend.styles import GLOBAL_QSS
from game_frontend.menu_widgets import MenuPage, DifficultyPage, SettingsPage, GameOverPage
from game_frontend.game_widgets import GamePage
from game_frontend.leaderboard_dialog import LeaderboardDialog, load_leaderboard, save_leaderboard
from game_frontend.backend_thread import BackendThread

PAGE_MENU = 0
PAGE_DIFFICULTY = 1
PAGE_SETTINGS = 2
PAGE_GAME = 3


class _VoiceBridge(QObject):
    """将后台线程的语音回调安全桥接到 Qt 主线程"""
    command_received = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OfficeFit AI 视觉对话放松助手")
        self.setMinimumSize(1024, 768)
        self.resize(1366, 900)
        self.setStyleSheet(GLOBAL_QSS)
        # 默认窗口模式，F11 切换全屏

        self._difficulty = 'normal'
        self._sound_enabled = True
        self._backend = None
        self._prev_score = 0
        self._prev_combo = 0
        self._leaderboard_saved = False
        self._game_phase = 'idle'  # idle | loading | ready_waiting | countdown | playing | game_over
        self._start_requested = False  # 用户是否已通过语音/按钮请求开始
        self._workout_mode = 'upper_body'  # upper_body | full_body
        self._session_state = 'ready'  # ready | training | paused | summary

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu_page = MenuPage()
        self._difficulty_page = DifficultyPage()
        self._settings_page = SettingsPage()
        self._game_page = GamePage()
        self._game_over_page = GameOverPage()

        self._stack.addWidget(self._menu_page)
        self._stack.addWidget(self._difficulty_page)
        self._stack.addWidget(self._settings_page)
        self._stack.addWidget(self._game_page)

        # ── 语音 & 对话（必须在 _game_page 创建之后）──
        self._voice_service = None
        self._voice_bridge = _VoiceBridge()
        self._voice_bridge.command_received.connect(self._on_voice_command)
        self._dialog_manager = None
        self._init_voice()

        # ── AI 助手面板信号 ──
        panel = self._game_page.assistant_panel
        panel.listen_clicked.connect(self._on_listen_clicked)
        panel.simulate_command.connect(self._on_simulate_command)
        panel.mode_selected.connect(self._on_mode_selected)
        self._update_mic_status()

        # 菜单
        self._menu_page.office_clicked.connect(self._enter_office_mode)
        self._menu_page.start_clicked.connect(self._on_start_clicked)
        self._menu_page.leaderboard_clicked.connect(self._show_leaderboard)
        self._menu_page.guide_clicked.connect(self._show_usage_guide)
        self._menu_page.exit_clicked.connect(self.close)

        # 难度
        self._difficulty_page.difficulty_selected.connect(self._on_difficulty_selected)
        self._difficulty_page.workout_mode_selected.connect(self._on_workout_mode_page)
        self._difficulty_page.action_guide_clicked.connect(self._show_action_guide)
        self._difficulty_page.back_clicked.connect(lambda: self._stack.setCurrentIndex(PAGE_MENU))

        # 设置
        self._settings_page.back_clicked.connect(lambda: self._stack.setCurrentIndex(PAGE_MENU))
        self._settings_page.sound_toggled.connect(self._on_sound_toggled)

        # 放松完成覆盖层
        self._game_over_page.restart_clicked.connect(self._restart_game)
        self._game_over_page.menu_clicked.connect(self._back_to_menu)
        self._game_over_page.hide()

        # 倒计时完成
        self._game_page.countdown_finished.connect(self._on_countdown_finished)

        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- 语音初始化 ----------

    def _init_voice(self):
        """初始化语音服务、对话管理器、用户状态和 AI 客户端"""
        try:
            from game_frontend.voice_service import VoiceService
            from game_frontend.dialog_manager import DialogManager
            from game_frontend.user_state import UserState
            from game_frontend.ai_client import create_ai_client

            self._user_state = UserState()
            self._dialog_manager = DialogManager()
            self._dialog_manager.set_user_state(self._user_state)
            self._voice_service = VoiceService()
            self._voice_service.set_callback(self._on_voice_raw)
            self._voice_service.set_volume_callback(
                lambda lvl: self._game_page.assistant_panel.update_voice_level(lvl))
            self._game_page.set_dialog_manager(self._dialog_manager)

            # 外部 AI 客户端 (可选)
            self._ai_client = create_ai_client()
            ai_status = self._ai_client.get_status()
            print(f"[MainWindow] AI provider: {ai_status['provider']}, "
                  f"available: {ai_status['available']}")
            self._game_page.assistant_panel.set_ai_source(
                ai_status["source"])

            # 健康提醒定时器
            self._health_timer = QTimer(self)
            self._health_timer.timeout.connect(self._on_health_tick)
            self._health_timer.start(10000)  # 每 10 秒检查
            self._sedentary_seconds = 0
            self._water_seconds = 0
        except Exception as e:
            print(f"[MainWindow] 语音模块初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._dialog_manager = None
            self._voice_service = None
            self._ai_client = None
            self._user_state = None

    def _on_voice_raw(self, text: str):
        """语音后台线程回调 → 桥接到主线程"""
        self._voice_bridge.command_received.emit(text)

    def _on_voice_command(self, keyword: str):
        """主线程处理语音指令（keyword 已由 voice_service 归一化）"""
        raw = (self._voice_service.last_raw_text
               if self._voice_service else "") or keyword
        normalized = (self._voice_service.last_normalized
                      if self._voice_service else "") or keyword
        self._last_voice_raw = normalized  # UI 展示归一化后的文本
        # 终端日志同时打印 raw 和 normalized
        if raw != normalized:
            print(f"[Voice] raw: {raw}")
            print(f"[Voice] normalized: {normalized}")
        self._show_central_hint(
            f"我听到了：{normalized}", "正在理解你的需求...", "cyan")
        self._dispatch_command(keyword)

    def _on_listen_clicked(self):
        """点击「语音输入」→ 开始录音 / 再次点击 → 结束录音"""
        panel = self._game_page.assistant_panel
        if self._voice_service is None:
            panel.set_listen_error("语音服务未初始化")
            return
        if not self._voice_service.is_available():
            panel.set_listen_error(
                f"麦克风不可用: {self._voice_service.last_error}"
                if self._voice_service.last_error
                else "请检查麦克风连接或使用快捷输入按钮"
            )
            return

        # 正在录音中 → 再次点击结束录音
        if self._voice_service._running:
            self._voice_service.request_stop()
            panel.set_voice_understanding()
            self._show_central_hint(
                "正在结束录音...",
                "正在理解你说的话",
                "cyan")
            return

        self._voice_service.start_listening()
        self._session_state = 'listening'
        panel.set_listening_active(True)
        panel.set_session_state('listening')
        self._show_central_hint(
            "正在录音...",
            "请说：我肩膀酸 / 我脖子酸 / 开始放松",
            "cyan")
        self._voice_check_timer = QTimer(self)
        self._voice_check_timer.timeout.connect(self._check_voice_status)
        self._voice_check_timer.start(1000)

    def _check_voice_status(self):
        """定时检查语音识别状态并更新 UI"""
        if self._voice_service is None:
            return
        panel = self._game_page.assistant_panel
        is_running = self._voice_service._running
        if not is_running:
            panel.set_listening_active(False)
            if hasattr(self, '_voice_check_timer'):
                self._voice_check_timer.stop()
            if self._session_state == 'listening':
                self._session_state = 'ready'
                panel.set_session_state('ready')
                err = self._voice_service.last_error
                err_type = self._voice_service.last_error_type
                if err_type == "user_stop":
                    # 用户主动结束录音
                    pass  # 已在点击时显示"正在结束录音"
                elif err_type == "timeout":
                    self._show_central_hint(
                        "没有检测到语音",
                        "请靠近麦克风后再说一次，或点击下方快捷指令",
                        "yellow")
                    panel.set_listen_error("没有检测到语音，请再试一次")
                elif err_type == "no_speech":
                    self._show_central_hint(
                        "没有听清，请再说一次",
                        "或点击下方快捷指令按钮",
                        "yellow")
                    panel.set_listen_error("没有听清，请再试一次或使用快捷输入")
                elif err:
                    panel.set_listen_error(err)
                else:
                    # 识别成功 → 已在 _on_voice_command 中处理
                    pass

    def _dispatch_command(self, command: str):
        """统一指令分发：调用 dialog_manager → 外部 AI 润色 → 根据 action 执行"""
        panel = self._game_page.assistant_panel
        if self._dialog_manager is None:
            return
        # 用户已交互 → 停止 office 空闲态自动提示，避免覆盖意图/AI 回复提示
        self._office_auto_hint = False

        result = self._dialog_manager.handle_user_command(command)
        local_reply = result.get("reply", "")
        action = result.get("action", "none")

        # 如果本地未识别且有外部 AI，让 AI 理解非标准文本
        intent = result.get("intent", "")
        if (intent in ("unknown", "none") and action == "none"
                and hasattr(self, '_ai_client') and self._ai_client
                and self._ai_client.is_available()):
            try:
                ai_understanding = self._ai_client.generate_reply(
                    f"用户说了一句不在关键词表里的话:「{command}」。"
                    "请判断意图(symptom/control/chat/unknown)，"
                    "如果是身体不适，推荐已有动作并给出简短回复。"
                    "如果是控制指令，映射到 start/pause/resume/next/end。"
                    "如果无法处理，友好提示可用指令。",
                    self._user_state.__dict__ if self._user_state else {}
                )
                if ai_understanding:
                    local_reply = ai_understanding
                    ai_source = "外部 AI"
                    # AI 理解了 → 标记为已处理
                    action = "ai_understood"
            except Exception:
                pass

        # 尝试外部 AI 润色
        reply = local_reply
        ai_source = "本地规则"
        if (hasattr(self, '_ai_client') and self._ai_client
                and self._ai_client.is_available()):
            try:
                state_dict = (
                    self._user_state.__dict__
                    if hasattr(self._user_state, '__dict__')
                    else {}
                )
                ai_reply = self._ai_client.generate_reply(command, state_dict)
                if ai_reply:
                    reply = ai_reply
                    ai_source = self._ai_client.source_name
            except Exception:
                ai_source = "外部 AI 暂不可用，已使用本地规则回复"
        # 所有回复路径都经过清洗，确保 JSON/dict 不泄漏到 UI
        reply = self._clean_ai_reply(reply)
        local_reply_clean = self._clean_ai_reply(local_reply)
        panel.set_assistant_message(reply if reply else local_reply_clean)
        if ai_source != "本地规则":
            panel.set_ai_source(ai_source)

        intent = result.get("intent", "")
        parts = result.get("body_parts",
                self._user_state.fatigue_parts if self._user_state else [])
        plan = result.get("recommended_plan", [])

        # 左侧「最近输入」两行（我听到 / 理解为）+ 推荐动作链
        panel.set_recent_input(
            getattr(self, '_last_voice_raw', command),
            self._describe_understanding(command, result, parts))
        panel.set_recommended_actions(plan)

        # 中央动态提示（按 action / intent 选择文案）
        non_symptom = {
            "", "none", "unknown", "pause", "resume", "skip", "end",
            "start_relax", "start_challenge", "start_game", "start_office",
            "end_office", "set_mode", "blacklisted",
        }
        if action == "show_relax_plan":
            self._show_relax_plan_hint(plan)
        elif intent not in non_symptom:
            parts_text = "、".join(parts) if parts else "肩颈"
            plan_text = "、".join(plan[:4]) if plan else "上半身放松"
            self._show_central_hint(
                f"我理解你是：{parts_text}不适",
                f"建议先做：{plan_text}", "green")
        elif intent == "blacklisted":
            self._show_central_hint("已为你调整建议", reply[:40], "cyan")
        elif action == "start_session":
            self._show_central_hint("进入放松训练", "请跟随引导完成动作", "green")
        elif reply:
            self._show_central_hint("AI 建议", reply[:40], "cyan")

        # ── 执行 action ──
        if action == "show_relax_plan":
            # 引导放松：展示计划，不启动计分后端，停留在守护相机视图
            self._session_state = "relax_guidance"
            panel.set_session_state("relax_guidance")

        elif action == "start_session":
            self._workout_mode = "upper_body"
            self._session_state = "training"
            panel.set_session_state("training")
            self._start_requested = True
            self._game_page.set_workout_ready()
            # 重启后端为 training 模式
            self._stop_backend()
            self._start_backend(run_mode="training")

        elif action == "pause_session":
            self._session_state = "paused"
            panel.set_session_state("paused")
            if self._backend is not None and self._backend.isRunning():
                self._backend.stop()

        elif action == "resume_session":
            self._session_state = "training"
            panel.set_session_state("training")
            if self._backend is not None and not self._backend.isRunning():
                self._start_backend(run_mode="training")

        elif action == "end_session":
            self._session_state = "summary"
            panel.set_session_state("summary")
            if self._backend is not None and self._backend.isRunning():
                self._backend.stop()

        elif action == "next_action":
            pass

        elif action in ("set_mode_upper", "set_mode_full"):
            self._workout_mode = (
                "upper_body" if action == "set_mode_upper" else "full_body"
            )

        elif action == "enter_office":
            self._enter_office_mode()

    def _on_simulate_command(self, command: str):
        """快捷输入（无麦克风环境备用）。归一化后直接分发，避免经语音回调重复分发。"""
        from game_frontend.voice_service import normalize_command
        norm = normalize_command(command)
        self._last_voice_raw = norm  # UI 展示归一化后的文本
        if command != norm:
            print(f"[Voice] Quick input raw: {command}")
            print(f"[Voice] Quick input normalized: {norm}")
        self._show_central_hint(
            f"我听到了：{norm}", "正在理解你的需求...", "cyan")
        self._dispatch_command(norm)

    # ---------- 健康提醒 ----------

    def _on_health_tick(self):
        """每 10 秒检查健康提醒"""
        if self._user_state is None:
            return
        if self._user_state.current_mode != "office":
            self._sedentary_seconds = 0
            self._water_seconds = 0
            return

        # 加速模式（开发调试用）
        interval = 10 if os.environ.get("OFFICEFIT_DEMO_TIMER") == "1" else 1
        self._sedentary_seconds += interval
        self._water_seconds += interval
        self._user_state.sedentary_minutes = self._sedentary_seconds // 60

        # 久坐提醒 (35 分钟; DEV 模式 30 秒)
        sedentary_trigger = 30 if os.environ.get("OFFICEFIT_DEMO_TIMER") == "1" else 35 * 60
        if self._sedentary_seconds >= sedentary_trigger:
            self._sedentary_seconds = 0
            if self._dialog_manager:
                result = self._dialog_manager.handle_user_command("久坐了")
                self._game_page.assistant_panel.set_assistant_message(result["reply"])

        # 喝水提醒 (60 分钟; DEV 模式 60 秒)
        water_trigger = 60 if os.environ.get("OFFICEFIT_DEMO_TIMER") == "1" else 60 * 60
        if self._water_seconds >= water_trigger:
            self._water_seconds = 0
            self._game_page.assistant_panel.set_assistant_message(
                "💧 喝水提醒：你已经连续办公较长时间，建议喝杯水休息一下。")

    def _enter_office_mode(self):
        """进入办公守护模式"""
        if self._user_state:
            self._user_state.current_mode = "office"
        self._start_requested = False
        self._session_state = "office"
        # office 空闲态自动提示（办公守护中 / 请坐到画面中央）开关
        self._office_auto_hint = True
        self._office_hint_state = None
        self._game_page.reset_loading()
        self._stack.setCurrentIndex(PAGE_GAME)
        self._game_page.assistant_panel.set_session_state("office")
        self._game_page.assistant_panel.set_assistant_message(
            "我已进入 AI 办公守护模式，正在观察你的坐姿。\n"
            "你可以说：我肩膀酸、我脖子酸、腰背僵硬、开始放松、结束办公。")
        self._show_central_hint(
            "办公守护中",
            "你可以说「我肩膀酸 / 我脖子酸 / 开始放松」，按 ESC 返回首页",
            "white")
        self._start_backend(run_mode="office")

    def _update_office_center_hint(self, status: dict):
        """office 空闲态：根据是否入镜在「办公守护中 / 请坐到画面中央」间切换。
        仅在用户尚未交互且会话仍为 office 时管理，避免覆盖语音/意图/AI 提示。"""
        if not getattr(self, '_office_auto_hint', False):
            return
        if self._session_state != 'office':
            return
        person = status.get('person_present', False)
        desired = 'centered' if person else 'not_centered'
        if getattr(self, '_office_hint_state', None) == desired:
            return
        self._office_hint_state = desired
        if person:
            self._show_central_hint(
                "办公守护中",
                "你可以说「我肩膀酸 / 我脖子酸 / 开始放松」，按 ESC 返回首页",
                "white")
        else:
            self._show_central_hint(
                "请坐到画面中央",
                "我需要看到你的头部、肩膀和双手，按 ESC 返回首页",
                "cyan")

    def _clean_ai_reply(self, text) -> str:
        """外部 AI 可能返回 JSON/dict，只展示可读回复，绝不暴露原始结构。"""
        if not text:
            return ""
        # 如果输入已经是 dict，直接提取字段
        if isinstance(text, dict):
            for k in ("reply", "message", "content", "text"):
                v = text.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # 尝试打印 dict 用于调试
            try:
                print(f"[MainWindow] AI returned dict without reply field: "
                      f"{str(text)[:200]}")
            except Exception:
                pass
            return ""
        s = str(text).strip()
        if not (s.startswith("{") or s.startswith("[")):
            return s
        import json
        import ast
        import re
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(s)
            except Exception:
                continue
            if isinstance(data, dict):
                for k in ("reply", "message", "content", "text"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        m = re.search(r"['\"]reply['\"]\s*:\s*['\"](.+?)['\"]", s, re.S)
        if m:
            return m.group(1).strip()
        # 最后的 fallback: 如果无法解析 JSON，返回去括号的纯文本
        cleaned = s.strip("{}[] ").strip()
        if not cleaned or len(cleaned) < 2:
            return "好的，我来帮你安排一组放松。"
        return cleaned

    def _describe_understanding(self, command: str, result: dict,
                                parts: list) -> str:
        """把一次输入归纳为用户可读的「理解为」一行。"""
        intent = result.get("intent", "")
        cmd_labels = {
            "开始放松": "开始放松", "暂停": "暂停", "继续": "继续",
            "换一个动作": "换一个动作", "结束": "结束",
            "上半身": "切换上半身", "全身": "切换全身",
            "开始计分挑战": "进入标准放松", "进入挑战": "进入标准放松",
            "开始挑战": "进入标准放松", "开始游戏": "进入标准放松",
            "开始办公": "进入办公守护", "返回办公模式": "返回办公守护",
            "结束办公": "结束办公",
        }
        if command in cmd_labels:
            return cmd_labels[command]
        if intent == "blacklisted":
            return "暂不支持的不适，已改为肩颈放松"
        if parts:
            return "、".join(parts) + "不适"
        if intent in ("", "none", "unknown"):
            return "未识别为指令"
        return intent

    def _show_relax_plan_hint(self, plan: list):
        """中央展示本轮 AI 放松计划。"""
        names = plan[:5] if plan else [
            "扩胸打开", "颈部左转", "颈部右转", "左侧拉伸", "右侧拉伸"]
        chain = " → ".join(names)
        self._show_central_hint(
            "本轮 AI 放松计划",
            f"{chain}　|　说「开始放松」进入训练", "green")

    def _build_relax_plan(self) -> list:
        """生成本轮放松计划（用于进入训练前的 AI 推荐展示）。"""
        if self._user_state and self._user_state.recommended_plan_text:
            return self._user_state.recommended_plan_text.split("、")
        if self._workout_mode == "full_body":
            return ["左抬腿", "右抬腿", "蹲下"]
        return ["扩胸打开", "颈部左转", "颈部右转", "左侧拉伸", "右侧拉伸"]

    def _on_mode_selected(self, mode: str):
        """用户选择放松模式（左侧面板 上半身 / 全身 按钮）"""
        self._workout_mode = mode
        mode_name = "上半身放松" if mode == "upper_body" else "全身活力"
        if self._dialog_manager is not None:
            result = self._dialog_manager.handle_user_command(
                "上半身" if mode == "upper_body" else "全身")
            reply = (self._clean_ai_reply(result.get("reply", ""))
                     if isinstance(result, dict) else str(result))
            self._game_page.assistant_panel.set_assistant_message(
                f"已切换为{mode_name}模式\n{reply}"
            )

    def _update_mic_status(self):
        """更新 UI 中的麦克风/AI 状态"""
        panel = self._game_page.assistant_panel
        if hasattr(self, '_ai_client') and self._ai_client:
            panel.set_ai_source(self._ai_client.get_status()["source"])
        if self._voice_service is not None:
            panel.set_mic_available(self._voice_service.is_available())
            panel.set_mic_device(
                self._voice_service.mic_device_name,
                self._voice_service.mic_device_index or 0,
            )
        else:
            panel.set_mic_available(False)

    def _show_central_hint(self, title: str, subtitle: str = "",
                           color: str = "white"):
        """更新画面中央动态提示"""
        overlay = self._game_page._countdown_overlay
        overlay.show_dynamic_hint(title, subtitle, color)

    # ---------- 页面切换 ----------

    def _on_start_clicked(self):
        self._stack.setCurrentIndex(PAGE_DIFFICULTY)

    def _on_difficulty_selected(self, difficulty):
        self._difficulty = difficulty
        self._leaderboard_saved = False
        self._prev_score = 0
        self._prev_combo = 0
        self._start_requested = False
        self._session_state = 'ready'
        self._game_page.assistant_panel.set_session_state('ready')

        # 进入训练前由 AI 助手先展示本轮计划（先推荐 → 后训练）
        plan = self._build_relax_plan()
        panel = self._game_page.assistant_panel
        mode_name = {'practice': '轻松', 'normal': '标准', 'hard': '活力'}.get(
            self._difficulty, self._difficulty)
        panel.set_assistant_message(
            f"已为你生成{mode_name}模式下的 2 分钟放松计划。\n"
            f"请站到画面中央，跟随引导完成每个动作。")
        panel.set_recommended_actions(plan)

        # 重置 GamePage 到 loading 状态，然后立即显示
        self._game_page.reset_loading()
        self._stack.setCurrentIndex(PAGE_GAME)
        self._start_backend()

    def _on_workout_mode_page(self, mode: str):
        """难度选择页中的放松模式切换"""
        self._workout_mode = mode
        mode_name = "上半身放松" if mode == "upper_body" else "全身互动"
        print(f"[MainWindow] 放松模式切换: {mode_name}")

    def _show_usage_guide(self):
        """打开使用说明弹窗"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("使用说明")
        dlg.setMinimumSize(520, 480)
        dlg.setStyleSheet(
            "QDialog { background-color: #0a0a1a; }"
            "QLabel { color: #d0d0e0; font-family: 'Microsoft YaHei'; }")
        layout = QVBoxLayout(dlg)
        title = QLabel("OfficeFit 使用说明")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 22px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        text = QLabel(
            '<b>开始办公</b><br>'
            '适合日常办公时常驻运行。AI 会观察你的坐姿与活动状态，'
            '你可以说出身体不适，例如「我肩膀酸」「我脖子酸」「腰背僵硬」。<br><br>'
            '<b>开始 2 分钟放松</b><br>'
            '适合立即进行一组结构化放松训练，系统会推荐适合当前状态的动作。<br><br>'
            '<b>语音使用方式</b><br>'
            '点击「语音输入」，然后说一句话；再次点击可结束录音。'
            '可说的指令包括：「我肩膀酸」「我脖子酸」「腰背僵硬」「开始放松」「换一个动作」「结束」。'
            '<br><br>'
            '<b>快捷键</b><br>'
            'F11：切换全屏<br>'
            'ESC：返回首页<br>'
            'R：重新开始本轮放松<br>'
            'Q：结束本轮放松<br><br>'
            '<b>语音与 AI</b><br>'
            '默认本地规则识别关键词，可选配置外部 AI 理解自然表达；'
            '不上传视频、深度图或原始音频，只传文本和结构化状态摘要。'
            '若语音识别失败，可使用左侧快捷指令按钮。'
        )
        text.setWordWrap(True)
        text.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(text)
        btn = QPushButton("关闭")
        btn.setObjectName("smallBtn")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        dlg.exec()

    def _show_action_guide(self, mode: str):
        """打开动作说明弹窗"""
        from game_frontend.action_guide import ActionGuideDialog
        dlg = ActionGuideDialog(self, mode=mode)
        dlg.exec()

    def _on_countdown_finished(self):
        """准备引导结束，训练正式开始"""
        self._game_phase = 'playing'

    def _show_leaderboard(self):
        dlg = LeaderboardDialog(self)
        dlg.exec()

    def _on_sound_toggled(self, enabled):
        self._sound_enabled = enabled

    def _back_to_menu(self):
        self._stop_backend()
        self._game_over_page.hide()
        self._game_phase = 'idle'
        self._stack.setCurrentIndex(PAGE_MENU)

    def _end_session_by_user(self):
        """Q 键：结束本轮放松，显示结束覆盖层"""
        self._stop_backend()
        self._session_state = 'summary'
        self._game_page.assistant_panel.set_session_state('summary')
        print("[Training] Session ended by user")
        self._game_over_page.reset_title_style()
        self._game_over_page.set_session_ended()
        self._game_over_page.setParent(self._game_page)
        self._game_over_page.setGeometry(0, 0,
                                         self._game_page.width(),
                                         self._game_page.height())
        self._game_over_page.show()
        self._game_over_page.raise_()

    # ---------- 后端管理 ----------

    def _start_backend(self, run_mode: str = "training"):
        self._stop_backend()
        # 后端只识别 upper_body / full_body；AI 推荐等其它选项归一为 upper_body
        backend_mode = (self._workout_mode
                        if self._workout_mode in ("upper_body", "full_body")
                        else "upper_body")
        self._backend = BackendThread(
            mode=self._difficulty,
            sound_enabled=False,
            parent=self,
            workout_mode=backend_mode,
            run_mode=run_mode,
        )
        self._backend.status_updated.connect(self._on_status_updated)
        self._backend.start()

    def _stop_backend(self):
        if self._backend is None:
            return
        try:
            self._backend.status_updated.disconnect(self._on_status_updated)
        except (TypeError, RuntimeError):
            pass
        if self._backend.isRunning():
            self._backend.stop()
        self._backend = None

    def _restart_game(self):
        # 彻底移除结算覆盖层
        self._game_over_page.hide()
        self._game_over_page.setParent(None)
        self._leaderboard_saved = False
        self._prev_score = 0
        self._prev_combo = 0
        self._game_phase = 'idle'
        self._start_requested = False
        self._session_state = 'ready'
        self._game_page.assistant_panel.set_session_state('ready')
        self._game_page.reset_loading()
        self._start_backend()
        print("[Training] Restart current relaxation session")

    def closeEvent(self, event):
        self._stop_backend()
        if self._voice_service is not None:
            self._voice_service.stop_listening()
        super().closeEvent(event)

    # ---------- 状态更新 ----------

    def _on_status_updated(self, status: dict):
        phase = status.get('phase', 'idle')
        self._game_page.update_status(status)

        # 同步视觉状态到对话管理器
        if self._dialog_manager is not None:
            self._dialog_manager.update_visual_state(status)

        # office 空闲态中央提示（入镜 → 办公守护中 / 未入镜 → 请坐到画面中央）
        if phase == 'office':
            self._update_office_center_hint(status)

        # 仅在 playing 阶段处理后端分数变化
        if phase == 'playing':
            score = status.get('score', 0)
            combo = status.get('combo', 0)
            if score > self._prev_score and self._sound_enabled:
                if combo >= 3:
                    winsound.Beep(1200, 80)
                winsound.Beep(880, 100)
                winsound.Beep(1100, 120)
            self._prev_score = score
            self._prev_combo = combo

        # 训练完成
        if phase == 'game_over' and not self._leaderboard_saved:
            self._on_game_over(status)

    def _on_game_over(self, status: dict):
        self._leaderboard_saved = True
        score = status.get('score', 0)
        high_score = status.get('high_score', 0)
        mode = status.get('mode', 'practice')

        if score > 0:
            entries = load_leaderboard()
            entries.append({
                'score': score,
                'difficulty': mode.upper(),
                'date': datetime.date.today().isoformat(),
            })
            entries.sort(key=lambda x: x['score'], reverse=True)
            entries = entries[:20]
            save_leaderboard(entries)

        if self._sound_enabled:
            winsound.Beep(200, 500)

        self._game_over_page.set_results(score, high_score, mode)
        self._game_over_page.setParent(self._game_page)
        self._game_over_page.setGeometry(0, 0,
                                         self._game_page.width(),
                                         self._game_page.height())
        self._game_over_page.show()
        self._game_over_page.raise_()

    # ---------- 键盘快捷键 ----------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return

        if self._stack.currentIndex() == PAGE_GAME:
            if key == Qt.Key_R:
                self._restart_game()
            elif key == Qt.Key_Escape:
                self._back_to_menu()
            elif key == Qt.Key_Q:
                self._end_session_by_user()
        elif self._stack.currentIndex() == PAGE_MENU:
            if key == Qt.Key_Escape:
                self.close()

        super().keyPressEvent(event)
