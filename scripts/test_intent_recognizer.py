#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IntentRecognizer + ActionRecommender 快速测试

每个用例带期望断言，最后输出 PASS/FAIL 汇总并以退出码反映结果。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_frontend.intent_recognizer import IntentRecognizer
from game_frontend.action_recommender import ActionRecommender

# (输入, 期望: 是否有意图, 期望包含的意图(可空), 是否复合, 是否命中黑名单)
CASES = [
    ("我脖子酸",          True,  "neck_stiff",    None,  False),
    ("肩膀很紧",          True,  "shoulder_tight", None, False),
    ("腰背僵硬",          True,  "waist_sore",    None,  False),
    ("脖子肩膀腰都酸",    True,  None,            True,  False),
    ("眼睛干",            False, None,            None,  True),
    ("眼睛干但是肩膀酸",  True,  "shoulder_sore", None,  True),
    ("坐久了全身僵硬",    True,  None,            True,  False),
]

r = IntentRecognizer()
a = ActionRecommender()

passed = 0
failed = 0

for text, want_intent, want_intent_id, want_composite, want_blacklist in CASES:
    result = r.recognize(text)
    rec = a.recommend(result["intents"])

    checks = []
    if want_intent:
        checks.append(("有意图", bool(result["intents"])))
    else:
        checks.append(("无意图", not result["intents"]))
    if want_intent_id is not None:
        checks.append((f"含{want_intent_id}", want_intent_id in result["intents"]))
    if want_composite is not None:
        checks.append(("复合", result["is_composite"] == want_composite))
    if want_blacklist:
        checks.append(("命中黑名单", bool(result["blacklisted_terms"])))

    ok = all(v for _, v in checks)
    passed += ok
    failed += (not ok)

    print(f"[{'PASS' if ok else 'FAIL'}] 输入: {text}")
    print(f"  意图: {result['intents']}")
    print(f"  部位: {result['body_parts']}")
    print(f"  黑名单: {result['blacklisted_terms']}")
    print(f"  复合: {result['is_composite']}")
    print(f"  动作: {rec['display_names']}")
    if not ok:
        bad = [name for name, v in checks if not v]
        print(f"  未通过检查: {bad}")
    print()

print(f"==== 汇总: {passed} 通过 / {failed} 失败 (共 {len(CASES)}) ====")
sys.exit(1 if failed else 0)
