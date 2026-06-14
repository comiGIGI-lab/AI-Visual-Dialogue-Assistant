# -*- coding: utf-8 -*-
"""
本地规则对话管理器 — 基于 UserState 的结构化决策
=================================================

返回格式:
    {
        "reply": str,           # AI 回复
        "intent": str,          # fatigue_neck / fatigue_shoulder / ...
        "next_mode": str,       # office / relax_guidance / game / ...
        "recommended_plan": [], # 推荐动作列表
        "action": str,          # 前端执行的动作
    }
"""

from typing import Dict, List, Optional
from game_frontend.user_state import UserState
from game_frontend.intent_recognizer import IntentRecognizer
from game_frontend.action_recommender import ActionRecommender


# ── 疲劳关键词 → (疲劳部位, 回复) ─────────────────────────

_FATIGUE_MAP: Dict[str, tuple] = {
    "我脖子酸": ("颈部", "检测到你想活动颈部，建议做颈部转动和肩部放松。"),
    "脖子酸": ("颈部", "颈部不适是很常见的久坐问题，我先带你活动脖子。"),
    "我肩膀酸": ("肩颈", "检测到你想进行肩颈放松，我会优先安排上半身动作。"),
    "肩膀酸": ("肩颈", "肩颈僵硬是久坐的首要问题，我们针对性活动。"),
    "腰背僵硬": ("腰背", "腰背不适建议做侧向拉伸和扩胸，久坐尤其需要。"),
    "背酸": ("腰背", "后背酸痛可以试试上肢伸展，缓解肌肉僵硬。"),
    "腰酸": ("腰背", "腰部不适可以先做轻量上半身活动，不建议弯腰。"),
    "久坐了": ("通用", "我们先做 2 分钟轻量放松，重点活动肩颈和上背。"),
    "活动一下": ("通用", "好的，为你安排轻量上半身放松。"),
}

# ── 动作推荐 ─────────────────────────────────────────────

_RECOMMEND_UPPER = [
    "颈部左转", "颈部右转",
    "双手上举", "左手上举", "右手上举",
    "扩胸打开", "左侧拉伸", "右侧拉伸",
]

_RECOMMEND_NECK = ["颈部左转", "颈部右转", "双手上举", "扩胸打开"]
_RECOMMEND_SHOULDER = ["双手上举", "扩胸打开", "左侧拉伸", "右侧拉伸"]
_RECOMMEND_BACK = ["扩胸打开", "左侧拉伸", "右侧拉伸", "左手上举", "右手上举"]


class DialogManager:
    """本地规则对话管理器，基于 UserState"""

    def __init__(self):
        self._state: Optional[UserState] = None
        self._person_present: bool = False
        self._recognizer = IntentRecognizer()
        self._recommender = ActionRecommender()

    def set_user_state(self, state: UserState):
        self._state = state

    def get_state(self) -> dict:
        if self._state is None:
            return {}
        from dataclasses import asdict
        return asdict(self._state)

    def update_visual_state(self, status: dict):
        """从 backend status 更新视觉信息"""
        if self._state is None:
            return
        self._state.camera_source = status.get("camera_source", "unknown")
        self._state.depth_available = status.get("depth_available", False)
        action = status.get("action_text", "")
        self._person_present = bool(action and action != "STANDING")

    # ── 主入口 ────────────────────────────────────────────

    def handle_user_command(self, text: str) -> dict:
        """处理用户输入，返回结构化结果"""
        if self._state is None:
            self._state = UserState()
        self._state.last_user_text = text

        # 1. 明确控制指令优先（精确匹配）
        mode_handlers = {
            "开始办公": self._handle_office_mode,
            "返回办公模式": self._handle_office_mode,
            "结束办公": self._handle_end_office,
            "开始放松": self._handle_start_relax,
            "开始计分挑战": self._handle_start_challenge,
            "进入挑战": self._handle_start_challenge,
            "开始挑战": self._handle_start_challenge,
            "开始游戏": self._handle_start_game,
            "暂停": self._handle_pause,
            "继续": self._handle_resume,
            "换一个动作": self._handle_skip,
            "结束": self._handle_end,
            "上半身": self._handle_upper,
            "全身": self._handle_full,
        }
        handler = mode_handlers.get(text)
        if handler:
            return handler()

        # 2. 非控制指令 → 统一走意图识别
        #    覆盖自由语音、复合症状、黑名单（含黑名单+有效症状混合）
        return self._handle_fatigue(text)

    # ── 疲劳 ──────────────────────────────────────────────

    def _handle_fatigue(self, text: str) -> dict:
        # 先用意图识别器分析
        result = self._recognizer.recognize(text)
        intents = result["intents"]
        body_parts = result["body_parts"]
        blacklisted = result["blacklisted_terms"]
        is_composite = result["is_composite"]

        self._state.detected_intents = intents
        self._state.blacklisted_terms = blacklisted

        # 纯黑名单
        if not intents and blacklisted:
            self._state.last_intent = "blacklisted"
            return {
                "reply": (
                    "这个症状暂时不适合用骨架动作检测处理，"
                    "我可以先带你做肩颈放松或开胸伸展。"
                ),
                "intent": "blacklisted",
                "next_mode": "ready",
                "recommended_plan": _RECOMMEND_UPPER,
                "action": "none",
            }

        # 无识别结果
        if not intents:
            fallback = _FATIGUE_MAP.get(text)
            if fallback:
                part, reply = fallback
                self._state.add_fatigue(part)
                return {
                    "reply": f"{reply}\n是否开始上半身放松？",
                    "intent": "fatigue_general",
                    "next_mode": "ready",
                    "recommended_plan": _RECOMMEND_UPPER,
                    "action": "none",
                }
            return self._fallback(text)

        # 有意图 → 推荐动作
        rec = self._recommender.recommend(intents, self._state.workout_mode)
        self._state.recommended_actions = rec["actions"]
        self._state.recommended_plan_text = "、".join(rec["display_names"])
        self._state.last_recommendation_reason = rec["reason"]
        self._state.detected_intents = intents
        self._state.last_intent = (
            intents[0] if len(intents) == 1 else "composite"
        )
        for p in body_parts:
            self._state.add_fatigue(p)

        parts_text = "、".join(body_parts)
        plan_text = self._state.recommended_plan_text
        if is_composite or len(intents) > 1:
            reply = (
                f"我识别到你主要是{parts_text}紧张。"
                f"建议先做{plan_text}。"
                f"我会为你安排一组放松流程。是否开始上半身放松？"
            )
        else:
            reply = (
                f"检测到{parts_text}不适。"
                f"建议做{plan_text}。"
                f"是否开始上半身放松？"
            )
        if blacklisted:
            reply += "\n（注意：部分症状超出体感检测范围，但我会聚焦可检测的动作）"

        return {
            "reply": reply,
            "intent": self._state.last_intent,
            "next_mode": "ready",
            "recommended_plan": rec["display_names"],
            "action": "none",
        }

    # ── 办公模式 ──────────────────────────────────────────

    def _handle_office_mode(self) -> dict:
        self._state.current_mode = "office"
        return {
            "reply": (
                "我已进入办公守护模式。\n"
                "你可以说：我脖子酸、我肩膀酸、开始放松、"
                "结束办公。"
            ),
            "intent": "start_office",
            "next_mode": "office",
            "recommended_plan": [],
            "action": "enter_office",
        }

    def _handle_end_office(self) -> dict:
        self._state.current_mode = "summary"
        return {
            "reply": "已退出办公模式，本次守护结束。建议起来走动一下。",
            "intent": "end_office",
            "next_mode": "summary",
            "recommended_plan": [],
            "action": "end_office",
        }

    # ── 放松 / 训练 ───────────────────────────────────────

    def _handle_start_relax(self) -> dict:
        # 「开始放松」先进入 AI 引导放松（展示计划 + 第一步），不直接训练。
        self._state.current_mode = "relax_guidance"
        names = (self._state.recommended_plan_text.split("、")
                 if self._state.recommended_plan_text else list(_RECOMMEND_UPPER))
        first = names[0] if names else "扩胸打开"
        return {
            "reply": (
                "好的，先做一组 AI 引导放松。\n"
                f"第一步：{first}，跟随提示自然呼吸。\n"
                "准备好后，可返回首页选择「开始 2 分钟放松」进入结构化训练。"
            ),
            "intent": "start_relax",
            "next_mode": "relax_guidance",
            "recommended_plan": names,
            "action": "show_relax_plan",
        }

    def _handle_start_challenge(self) -> dict:
        if not self._person_present:
            return {
                "reply": "请先坐到画面中央，确保完整入镜，再进入标准放松。",
                "intent": "start_challenge",
                "next_mode": "relax_guidance",
                "recommended_plan": [],
                "action": "none",
            }
        self._state.start_session(self._state.workout_mode or "upper_body")
        self._state.current_mode = "training"
        return {
            "reply": "进入标准放松。请跟随屏幕提示完成动作，注意动作稳定和呼吸节奏。",
            "intent": "start_challenge",
            "next_mode": "training",
            "recommended_plan": [],
            "action": "start_session",
        }

    def _handle_start_game(self) -> dict:
        if not self._person_present:
            return {
                "reply": "请先站到画面中央，确保完整入镜。",
                "intent": "start_session",
                "next_mode": "ready",
                "recommended_plan": [],
                "action": "none",
            }
        self._state.start_session("full_body")
        self._state.current_mode = "training"
        return {
            "reply": "进入放松训练模式！跟随屏幕引导完成每个动作。",
            "intent": "start_session",
            "next_mode": "training",
            "recommended_plan": [],
            "action": "start_session",
        }

    # ── 控制 ──────────────────────────────────────────────

    def _handle_pause(self) -> dict:
        self._state.current_mode = "paused"
        return {
            "reply": "已暂停，准备好后可以说「继续」。",
            "intent": "pause",
            "next_mode": "paused",
            "recommended_plan": [],
            "action": "pause_session",
        }

    def _handle_resume(self) -> dict:
        prev = self._state.current_mode
        self._state.current_mode = (
            "relax_guidance" if prev == "paused" else prev
        )
        return {
            "reply": "好的，继续当前训练。",
            "intent": "resume",
            "next_mode": self._state.current_mode,
            "recommended_plan": [],
            "action": "resume_session",
        }

    def _handle_skip(self) -> dict:
        return {
            "reply": "好的，切换到下一个动作。",
            "intent": "skip",
            "next_mode": self._state.current_mode,
            "recommended_plan": [],
            "action": "next_action",
        }

    def _handle_end(self) -> dict:
        self._state.current_mode = "summary"
        return {
            "reply": (
                "本轮放松结束。建议活动一下肩颈，喝点水。\n"
                "可以说「返回办公模式」继续守护。"
            ),
            "intent": "end",
            "next_mode": "summary",
            "recommended_plan": [],
            "action": "end_session",
        }

    def _handle_upper(self) -> dict:
        return {
            "reply": "已选择上半身放松模式。",
            "intent": "set_mode",
            "next_mode": self._state.current_mode,
            "recommended_plan": [],
            "action": "set_mode_upper",
        }

    def _handle_full(self) -> dict:
        return {
            "reply": "已选择全身互动模式。",
            "intent": "set_mode",
            "next_mode": self._state.current_mode,
            "recommended_plan": [],
            "action": "set_mode_full",
        }

    def _fallback(self, text: str) -> dict:
        return {
            "reply": (
                f"我还没有理解「{text}」。\n"
                "你可以试着说：我肩膀酸、我脖子酸、开始放松、换一个动作。"
            ),
            "intent": "unknown",
            "next_mode": self._state.current_mode,
            "recommended_plan": [],
            "action": "none",
        }
