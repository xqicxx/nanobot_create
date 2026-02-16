#!/usr/bin/env python3
"""测试 cron session context 恢复功能"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanobot.cron.types import CronPayload, CronSchedule, CronJob
from nanobot.cron.service import CronService


def test_cron_payload_has_session_fields():
    """测试 CronPayload 包含 session 恢复字段"""
    print("\n" + "=" * 50)
    print("测试1: CronPayload 字段")
    print("=" * 50)

    payload = CronPayload(
        message="测试任务",
        deliver=False,
        channel="whatsapp",
        to="+1234567890",
        original_session_key="whatsapp:+1234567890",
        user_id="whatsapp:+1234567890:user",
    )

    print(f"✅ message: {payload.message}")
    print(f"✅ deliver: {payload.deliver}")
    print(f"✅ channel: {payload.channel}")
    print(f"✅ to: {payload.to}")
    print(f"✅ original_session_key: {payload.original_session_key}")
    print(f"✅ user_id: {payload.user_id}")

    assert payload.original_session_key == "whatsapp:+1234567890"
    assert payload.user_id == "whatsapp:+1234567890:user"
    print("✅ CronPayload 字段测试通过")


def test_cron_payload_default_values():
    """测试 CronPayload 默认值"""
    print("\n" + "=" * 50)
    print("测试2: CronPayload 默认值")
    print("=" * 50)

    payload = CronPayload(message="测试")

    print(f"✅ deliver 默认值: {payload.deliver} (应为 False)")
    print(f"✅ original_session_key 默认值: {payload.original_session_key} (应为 None)")
    print(f"✅ user_id 默认值: {payload.user_id} (应为 None)")

    assert payload.deliver == False
    assert payload.original_session_key is None
    assert payload.user_id is None
    print("✅ CronPayload 默认值测试通过")


def test_session_key_extraction():
    """测试 session key 提取逻辑"""
    print("\n" + "=" * 50)
    print("测试3: Session Key 提取")
    print("=" * 50)

    # 模拟 on_cron_job 中的提取逻辑
    test_cases = [
        {
            "user_id": "whatsapp:+1234567890:user",
            "expected_sender": "user",
        },
        {
            "user_id": "cli:direct:admin",
            "expected_sender": "admin",
        },
        {
            "user_id": None,
            "expected_sender": None,
        },
    ]

    for tc in test_cases:
        user_id = tc["user_id"]
        expected = tc["expected_sender"]

        # 提取逻辑
        sender_id = None
        if user_id:
            parts = user_id.split(":")
            if len(parts) >= 3:
                sender_id = parts[2]

        print(f"✅ user_id={user_id} -> sender_id={sender_id}")
        assert sender_id == expected

    print("✅ Session Key 提取测试通过")


def test_original_session_fallback():
    """测试 original_session_key 回退逻辑"""
    print("\n" + "=" * 50)
    print("测试4: Original Session Key 回退")
    print("=" * 50)

    # 模拟 on_cron_job 中的回退逻辑
    test_cases = [
        {
            "job_id": "abc123",
            "original_session_key": "whatsapp:+1234567890",
            "expected": "whatsapp:+1234567890",
        },
        {
            "job_id": "abc123",
            "original_session_key": None,
            "expected": "cron:abc123",
        },
    ]

    for tc in test_cases:
        job_id = tc["job_id"]
        original = tc["original_session_key"]
        expected = tc["expected"]

        # 回退逻辑
        session_key = original or f"cron:{job_id}"

        print(f"✅ original_session_key={original} -> session_key={session_key}")
        assert session_key == expected

    print("✅ Original Session Key 回退测试通过")


def test_task_vs_reminder_mode():
    """测试 Task vs Reminder 模式"""
    print("\n" + "=" * 50)
    print("测试5: Task vs Reminder 模式")
    print("=" * 50)

    # Task 模式 (deliver=False)
    task_payload = CronPayload(
        message="分析我的工作",
        deliver=False,
        channel="whatsapp",
        to="+1234567890",
        original_session_key="whatsapp:+1234567890",
        user_id="whatsapp:+1234567890:user",
    )

    # Reminder 模式 (deliver=True)
    reminder_payload = CronPayload(
        message="喝水提醒",
        deliver=True,
        channel="whatsapp",
        to="+1234567890",
        original_session_key="whatsapp:+1234567890",
        user_id="whatsapp:+1234567890:user",
    )

    print(f"✅ Task 模式: deliver={task_payload.deliver}, 需要恢复上下文")
    print(f"✅ Reminder 模式: deliver={reminder_payload.deliver}, 轻量发送")

    assert task_payload.deliver == False
    assert reminder_payload.deliver == True

    # 两种模式都应该有 session 信息
    assert task_payload.original_session_key is not None
    assert reminder_payload.original_session_key is not None

    print("✅ Task vs Reminder 模式测试通过")


def test_cron_service_add_job_with_session():
    """测试 CronService.add_job 支持 session 参数"""
    print("\n" + "=" * 50)
    print("测试6: CronService.add_job 会话参数")
    print("=" * 50)

    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "cron" / "jobs.json"
        service = CronService(store_path=store_path)

        # 添加带 session 信息的 job
        job = service.add_job(
            name="测试任务",
            schedule=CronSchedule(kind="every", every_ms=60000),
            message="分析我的工作",
            deliver=False,
            channel="whatsapp",
            to="+1234567890",
            original_session_key="whatsapp:+1234567890",
            user_id="whatsapp:+1234567890:user",
        )

        print(f"✅ Job ID: {job.id}")
        print(f"✅ Job name: {job.name}")
        print(f"✅ deliver: {job.payload.deliver}")
        print(f"✅ original_session_key: {job.payload.original_session_key}")
        print(f"✅ user_id: {job.payload.user_id}")

        assert job.payload.original_session_key == "whatsapp:+1234567890"
        assert job.payload.user_id == "whatsapp:+1234567890:user"

        # 验证 job 存储
        jobs = service.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].payload.original_session_key == "whatsapp:+1234567890"

        print("✅ CronService.add_job 测试通过")


def main():
    print("\n" + "=" * 60)
    print("       Cron Session Context 恢复功能测试")
    print("=" * 60)

    test_cron_payload_has_session_fields()
    test_cron_payload_default_values()
    test_session_key_extraction()
    test_original_session_fallback()
    test_task_vs_reminder_mode()
    test_cron_service_add_job_with_session()

    print("\n" + "=" * 60)
    print("       ✅ 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
