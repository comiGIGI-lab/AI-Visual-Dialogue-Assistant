# -*- coding: utf-8 -*-
"""
动作推荐器
==========

将意图标签映射为大肢体动作组合。
"""

from typing import List
from game_frontend.health_intent_rules import (
    INTENT_TO_ACTIONS,
    ACTION_DISPLAY_NAMES,
)


class ActionRecommender:
    """基于意图推荐动作"""

    def recommend(self, intents: List[str],
                  workout_mode: str = "upper_body") -> dict:
        """
        返回:
          actions: list[str]         动作 ID 列表
          display_names: list[str]    动作中文名
          reason: str                 推荐原因
        """
        if not intents:
            return {"actions": [], "display_names": [], "reason": ""}

        action_set = set()
        for intent in intents:
            acts = INTENT_TO_ACTIONS.get(intent, [])
            action_set.update(acts)

        # 去重并按映射表中的顺序
        seen = set()
        actions = []
        for intent in intents:
            for a in INTENT_TO_ACTIONS.get(intent, []):
                if a not in seen:
                    seen.add(a)
                    actions.append(a)

        names = [ACTION_DISPLAY_NAMES.get(a, a) for a in actions]
        reason = f"根据你反馈的症状推荐 {len(actions)} 个动作"

        return {
            "actions": actions,
            "display_names": names,
            "reason": reason,
        }
