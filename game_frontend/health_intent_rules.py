# -*- coding: utf-8 -*-
"""
久坐症状 AI 意图识别规则表
===========================

集中定义关键词映射、黑名单、推荐动作，避免散落各处。
"""

# ── 意图关键词 ───────────────────────────────────────────

INTENT_KEYWORDS = {
    # 颈部不适
    "neck_stiff": {
        "body_part": "颈部",
        "priority": 1,
        "keywords": [
            "脖子僵硬", "脖子发紧", "脖子不灵活",
            "脖子酸痛", "脖子疼", "后颈酸胀",
            "转头不舒服", "扭头受限", "脖子拉扯痛",
            "脖子沉重", "颈后发僵", "颈椎不舒服",
            "久坐脖子难受", "脖子紧绷酸胀",
            # 常见短语
            "脖子酸", "脖子紧", "脖子很紧",
            "颈椎酸", "颈椎紧", "坐久了脖子酸",
        ],
    },
    "neck_pain": {
        "body_part": "颈部",
        "priority": 1,
        "keywords": ["脖子痛", "颈痛", "颈部疼痛", "颈酸痛"],
    },
    "neck_tension": {
        "body_part": "颈部",
        "priority": 1,
        "keywords": ["脖子紧张", "颈紧张", "脖子绷着", "脖子不自觉耸起"],
    },
    # 肩膀不适
    "shoulder_sore": {
        "body_part": "肩膀",
        "priority": 2,
        "keywords": [
            "肩膀酸痛", "肩膀发紧", "肩膀沉重",
            "双肩僵硬", "肩膀酸胀", "肩紧不舒服",
            "含胸肩累", "圆肩难受", "肩膀打不开",
            "肩背沉重", "肩膀僵硬抬不动", "久坐肩膀很累",
            "肩部紧绷酸痛",
            # 常见短语
            "肩膀酸", "肩很酸", "坐久了肩膀酸",
        ],
    },
    "shoulder_tight": {
        "body_part": "肩膀",
        "priority": 2,
        "keywords": [
            "肩膀紧", "肩紧", "肩膀发硬",
            "肩膀很紧", "肩很紧",
        ],
    },
    "shoulder_droop": {
        "body_part": "肩膀",
        "priority": 2,
        "keywords": ["肩膀下垂", "溜肩", "耸肩", "塌肩膀"],
    },
    # 上背部
    "back_upper_sore": {
        "body_part": "上背",
        "priority": 3,
        "keywords": [
            "后背僵硬", "上背酸痛", "背部发紧",
            "肩胛骨疼", "后背两侧酸胀", "肩胛紧绷",
            "驼背累背", "胸口闷背", "后背打不开",
            "久坐后背难受", "背部牵拉酸胀", "胸椎僵硬",
            "后背沉重发酸",
            # 常见短语
            "后背酸", "后背紧", "背酸", "背紧",
        ],
    },
    "scapula_tension": {
        "body_part": "肩胛",
        "priority": 3,
        "keywords": ["肩胛痛", "肩胛酸", "蝴蝶骨区域紧"],
    },
    # 腰部
    "waist_sore": {
        "body_part": "腰部",
        "priority": 4,
        "keywords": [
            "腰酸", "腰痛", "腰部酸胀", "腰发僵",
            "腰椎不舒服", "久坐腰硬", "直不起腰",
            "侧腰紧绷", "一边腰酸痛", "腰部拉扯感",
            "起身腰疼", "久坐腰累", "腰背僵硬",
            "腰肌紧张", "腰部沉重发酸",
            # 常见短语
            "腰很酸", "腰紧", "腰很紧", "坐久了腰酸",
        ],
    },
    "lumbar_tension": {
        "body_part": "腰椎",
        "priority": 4,
        "keywords": ["腰椎紧", "腰骶酸", "尾骨不舒服"],
    },
    "low_back_pain": {
        "body_part": "下背",
        "priority": 4,
        "keywords": ["下背痛", "下背酸", "下背僵硬"],
    },
    # 臀部/大腿
    "hip_tight": {
        "body_part": "臀部",
        "priority": 5,
        "keywords": ["屁股发麻", "臀部僵硬", "坐久屁股疼"],
    },
    "thigh_stiff": {
        "body_part": "大腿",
        "priority": 5,
        "keywords": [
            "大腿后侧紧绷", "大腿拉扯酸胀", "臀腿发僵",
            "下半身沉重", "久坐腿紧",
        ],
    },
    "hip_sore": {
        "body_part": "髋部",
        "priority": 5,
        "keywords": ["胯部紧绷", "腹股沟拉扯不适"],
    },
    # 体态/胸腔
    "chest_tight": {
        "body_part": "胸腔",
        "priority": 6,
        "keywords": ["胸口发闷", "胸腔打不开"],
    },
    "posture_collapse": {
        "body_part": "体态",
        "priority": 6,
        "keywords": [
            "含胸难受", "体态驼背", "坐不直",
            "上身塌累", "整个人紧绷", "上身僵硬舒展不开",
        ],
    },
}

# ── 黑名单 ───────────────────────────────────────────────

BLACKLIST_KEYWORDS = [
    "手指麻", "手腕疼", "小臂酸", "手肘痛",
    "眼睛干", "眼疲劳", "头晕头痛", "犯困",
    "腹胀", "消化不良", "浑身乏力",
    "脚凉", "脚底痛", "小腿水肿",
]

# ── 意图 → 推荐动作 ──────────────────────────────────────

INTENT_TO_ACTIONS = {
    "neck_stiff":    ["neck_turn_left", "neck_turn_right",
                      "both_hands_up", "chest_open"],
    "neck_pain":     ["neck_turn_left", "neck_turn_right", "chest_open"],
    "neck_tension":  ["neck_turn_left", "neck_turn_right",
                      "both_hands_up"],
    "shoulder_sore":  ["both_hands_up", "chest_open",
                       "left_stretch", "right_stretch"],
    "shoulder_tight": ["both_hands_up", "chest_open",
                       "left_stretch", "right_stretch"],
    "shoulder_droop": ["both_hands_up", "chest_open", "left_stretch"],
    "back_upper_sore": ["chest_open", "both_hands_up",
                        "left_stretch", "right_stretch"],
    "scapula_tension": ["chest_open", "both_hands_up",
                        "left_stretch", "right_stretch"],
    "waist_sore":     ["left_stretch", "right_stretch",
                       "both_hands_up", "chest_open"],
    "lumbar_tension": ["left_stretch", "right_stretch",
                       "both_hands_up"],
    "low_back_pain":  ["left_stretch", "right_stretch", "chest_open"],
    "hip_tight":      ["both_hands_up", "chest_open",
                       "left_stretch", "right_stretch"],
    "thigh_stiff":    ["both_hands_up", "left_stretch", "right_stretch"],
    "hip_sore":       ["left_stretch", "right_stretch", "both_hands_up"],
    "chest_tight":    ["chest_open", "both_hands_up", "left_stretch",
                       "right_stretch"],
    "posture_collapse": ["both_hands_up", "chest_open",
                         "left_stretch", "right_stretch"],
}

# ── 动作显示名 ────────────────────────────────────────────

ACTION_DISPLAY_NAMES = {
    "neck_turn_left": "颈部左转",
    "neck_turn_right": "颈部右转",
    "both_hands_up": "双手上举",
    "left_hand_up": "左手上举",
    "right_hand_up": "右手上举",
    "chest_open": "扩胸打开",
    "left_stretch": "左侧拉伸",
    "right_stretch": "右侧拉伸",
    "shoulder_relax": "肩部放松",
    "torso_side_bend": "躯干侧弯",
    "back_stretch": "背部拉伸",
    "seated_back_stretch": "坐姿背部舒展",
    "torso_forward_relax": "躯干前倾放松",
    "gentle_full_body_stretch": "轻量全身舒展",
    "posture_reset": "体态重置",
}

# ── 复合话术触发词 ────────────────────────────────────────

COMPOSITE_TRIGGERS = [
    "坐久了全身僵硬", "上半身都很紧", "脖子肩膀腰都酸",
    "后背和腰很不舒服", "肩颈腰背整体紧绷", "整个人坐得很僵硬",
    "全身不舒服", "浑身僵", "哪都不舒服", "上下都酸",
    "又酸又僵", "浑身僵硬",
    "上半身很紧", "上半身僵",
]

# 复合 → 对应意图集
COMPOSITE_INTENTS = {
    "坐久了全身僵硬": ["neck_stiff", "shoulder_sore", "waist_sore",
                     "back_upper_sore"],
    "全身不舒服": ["neck_stiff", "shoulder_sore", "waist_sore",
                 "back_upper_sore"],
    "浑身僵": ["neck_stiff", "shoulder_sore", "waist_sore",
              "back_upper_sore", "chest_tight"],
    "脖子肩膀腰都酸": ["neck_stiff", "shoulder_sore", "waist_sore"],
    "上半身都很紧": ["neck_stiff", "shoulder_sore", "back_upper_sore",
                   "chest_tight"],
    "后背和腰很不舒服": ["back_upper_sore", "waist_sore", "lumbar_tension"],
    "肩颈腰背整体紧绷": ["neck_stiff", "shoulder_sore",
                       "back_upper_sore", "waist_sore"],
    "整个人坐得很僵硬": ["neck_stiff", "shoulder_sore",
                       "back_upper_sore", "waist_sore", "chest_tight"],
    "哪都不舒服": ["neck_stiff", "shoulder_sore", "back_upper_sore"],
    "上下都酸": ["neck_stiff", "shoulder_sore", "waist_sore"],
    "又酸又僵": ["neck_stiff", "shoulder_sore", "back_upper_sore"],
    "浑身僵硬": ["neck_stiff", "shoulder_sore", "back_upper_sore",
                "waist_sore", "chest_tight"],
    "上半身很紧": ["neck_stiff", "shoulder_sore", "back_upper_sore",
                 "chest_tight"],
    "上半身僵": ["neck_stiff", "shoulder_sore", "back_upper_sore",
                "chest_tight"],
}
