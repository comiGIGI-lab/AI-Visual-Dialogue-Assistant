# -*- coding: utf-8 -*-
"""
久坐症状 AI 意图识别器
=======================

从用户语音文本中识别意图标签、疲劳部位、推荐动作组合。
"""

from typing import List, Dict, Set
from game_frontend.health_intent_rules import (
    INTENT_KEYWORDS,
    BLACKLIST_KEYWORDS,
    COMPOSITE_TRIGGERS,
    COMPOSITE_INTENTS,
)


class IntentRecognizer:
    """本地关键词匹配意图识别器"""

    def recognize(self, text: str) -> dict:
        """
        返回:
          intents: list[str]
          body_parts: list[str]
          blacklisted_terms: list[str]
          is_composite: bool
        """
        if not text:
            return self._empty()

        # 黑名单检测
        blacklisted = [w for w in BLACKLIST_KEYWORDS if w in text]

        # 先检查复合
        composite_hit = None
        for trigger in COMPOSITE_TRIGGERS:
            if trigger in text:
                composite_hit = trigger
                break

        if composite_hit:
            intents = COMPOSITE_INTENTS.get(composite_hit, [])
            parts = self._body_parts_from_intents(intents)
            return {
                "intents": intents,
                "body_parts": parts,
                "blacklisted_terms": blacklisted,
                "is_composite": True,
            }

        # 单项匹配
        intents = []
        parts_set: Set[str] = set()
        for intent_id, info in INTENT_KEYWORDS.items():
            for kw in info["keywords"]:
                if kw in text:
                    intents.append(intent_id)
                    parts_set.add(info["body_part"])
                    break

        return {
            "intents": intents,
            "body_parts": list(parts_set),
            "blacklisted_terms": blacklisted,
            "is_composite": len(intents) > 1,
        }

    def _body_parts_from_intents(self, intents: List[str]) -> List[str]:
        parts_set: Set[str] = set()
        for i in intents:
            info = INTENT_KEYWORDS.get(i, {})
            part = info.get("body_part", "")
            if part:
                parts_set.add(part)
        return list(parts_set)

    def _empty(self) -> dict:
        return {
            "intents": [], "body_parts": [],
            "blacklisted_terms": [], "is_composite": False,
        }
