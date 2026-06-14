# -*- coding: utf-8 -*-
"""
用户状态心智模型
=================

UserState 是所有 AI 回复、动作推荐、模式切换的唯一数据源。
dialog_manager 基于 UserState 返回结构化决策结果。
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UserState:
    """用户当前完整状态"""

    # ── 身体状态 ──
    posture: str = "unknown"          # unknown / seated / standing / upper_body_visible
    sedentary_minutes: int = 0         # 累计久坐分钟（由前端定时器递增）
    fatigue_parts: List[str] = field(default_factory=list)  # ["肩颈", "腰背"]

    # ── 意图识别结果 ──
    detected_intents: List[str] = field(default_factory=list)
    blacklisted_terms: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    recommended_plan_text: str = ""
    last_recommendation_reason: str = ""

    # ── 模式 ──
    current_mode: str = "office"       # office / ready / teaching / relax_guidance / training / paused / summary

    # ── 硬件 ──
    camera_source: str = "unknown"
    depth_available: bool = False
    voice_available: bool = False

    # ── 最后交互 ──
    last_user_text: str = ""
    last_intent: str = ""              # fatigue_neck / fatigue_shoulder / start_relax / ...
    assistant_message: str = ""

    # ── 会话 ──
    session_started_at: Optional[float] = None
    completed_actions: int = 0
    workout_mode: str = "upper_body"   # upper_body / full_body
    last_reminder_at: Optional[float] = None

    def reset_session(self):
        self.session_started_at = None
        self.completed_actions = 0
        self.current_mode = "office"
        self.last_intent = ""

    def start_session(self, mode: str = "upper_body"):
        self.session_started_at = time.time()
        self.completed_actions = 0
        self.workout_mode = mode

    def add_fatigue(self, part: str):
        if part not in self.fatigue_parts:
            self.fatigue_parts.append(part)

    def clear_fatigue(self):
        self.fatigue_parts.clear()
