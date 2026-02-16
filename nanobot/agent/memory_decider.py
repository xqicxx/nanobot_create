"""记忆操作触发决策器 - 轻量同步决策"""

import re
from dataclasses import dataclass


@dataclass
class TriggerResult:
    """触发结果"""
    should_trigger: bool
    reason: str
    priority: int = 0
    strategy: str = "none"


class MemoryTriggerDecider:
    """记忆触发决策器 - 轻量同步决策"""

    # 短回复模式（不写入）
    SKIP_PATTERNS = [
        r"^好的?$", r"^收到$", r"^ok$", r"^OK$", r"^[\d\s,.]+$",
        r"^[\u4e00-\u9fa5]{1,2}$",  # 1-2个汉字
        r"^[\u4e00-\u9fa5]{1,2}[吗呢吧呀啊哦]$",  # 语气词结尾
        r"^嗯$", r"^哦$", r"^啊$", r"^嗨$",
    ]

    # 显式触发词
    EXPLICIT_MEMORIZE = [
        "记住", "请记住", "帮我记住", "memo", "不要忘了",
        "以后都", "每次都", "一直记住", "存一下", "收藏",
        "记下来", "记住这个", "把这个记住",
    ]

    # 偏好模式
    PREFERENCE_PATTERNS = [
        r"我喜欢", r"我讨厌", r"我偏好", r"我不.*[吃喝玩做学习]",
        r"我喜欢.*风格", r"我喜欢.*颜色", r"我喜欢.*音乐",
        r"我最爱", r"我最讨厌", r"我想要", r"我需要",
    ]

    # 个人信息模式
    PROFILE_PATTERNS = [
        r"我叫", r"我.*叫.*", r"我是.*人", r"我.*岁",
        r"我的.*是.*@", r"我的.*电话", r"我的.*邮箱",
        r"我在.*工作", r"我在.*上学", r"我是.*学生",
    ]

    # 检索触发词
    RETRIEVE_TRIGGERS = [
        "我记得", "之前", "上次", "以前", "过去", "之前说过",
        "我的.*呢", "帮我找", "有什么.*记忆", "查一下.*之前",
    ]

    # 疑问词
    QUESTION_WORDS = ["什么", "哪", "谁", "怎", "为何", "为什么", "多少", "几", "是不是", "有没有"]

    def should_memorize(self, message: str, conversation: list[dict] | None = None) -> TriggerResult:
        """判断是否应该写入记忆"""
        msg = message.strip()
        if not msg:
            return TriggerResult(False, "空消息跳过")

        # 1. 跳过短回复
        if any(re.match(p, msg) for p in self.SKIP_PATTERNS):
            return TriggerResult(False, "短回复跳过")

        # 2. 显式触发
        if any(t in msg for t in self.EXPLICIT_MEMORIZE):
            return TriggerResult(True, "显式触发", 10)

        # 3. 偏好模式
        if any(re.search(p, msg) for p in self.PREFERENCE_PATTERNS):
            return TriggerResult(True, "偏好信息", 8)

        # 4. 个人信息
        if any(re.search(p, msg) for p in self.PROFILE_PATTERNS):
            return TriggerResult(True, "个人信息", 9)

        # 5. 多轮对话（对话长度 >= 4）
        if conversation and len(conversation) >= 4:
            return TriggerResult(True, "多轮对话", 5)

        return TriggerResult(False, "无触发条件")

    def should_retrieve(self, query: str, history: list[dict] | None = None) -> TriggerResult:
        """判断是否应该检索记忆"""
        if not query:
            return TriggerResult(False, "空查询跳过")

        query = query.strip()

        # 1. 明确触发
        if any(t in query for t in self.RETRIEVE_TRIGGERS):
            return TriggerResult(True, "明确触发", strategy="full")

        # 2. 疑问句
        if any(q in query for q in self.QUESTION_WORDS):
            return TriggerResult(True, "疑问句", strategy="quick")

        # 3. 历史有记忆（检查最近3条）
        if history:
            recent_history = history[-3:] if len(history) > 3 else history
            if any("memory" in str(h).lower() or "记住" in str(h) for h in recent_history):
                return TriggerResult(True, "历史有记忆", strategy="refresh")

        return TriggerResult(False, "无需检索")

    def get_category_from_message(self, message: str) -> str:
        """从消息中推断记忆分类"""
        msg = message.strip()

        # 偏好类
        if any(re.search(p, msg) for p in self.PREFERENCE_PATTERNS):
            return "interest"

        # 个人档案类
        if any(re.search(p, msg) for p in self.PROFILE_PATTERNS):
            return "profile"

        # 提醒类
        if any(t in msg for t in ["提醒", "提醒我", "待办", "todo", "任务"]):
            return "reminder"

        # 事件类
        if any(t in msg for t in ["发生了", "今天", "昨天", "上次", "会议", "活动"]):
            return "event"

        # 默认活动记录
        return "activity"
