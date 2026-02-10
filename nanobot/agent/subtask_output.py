"""Subtask output parsing and formatting with Pydantic validation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


StatusType = Literal["成功", "失败", "高风险拦截"]
ErrorCauseType = Literal["权限不足", "路径不存在", "语法错误", "超时", "拦截", "依赖缺失", "未知"]

_LABELS = {
    "状态": "status",
    "错误归因": "error_cause",
    "结论": "conclusion",
    "证据": "evidence",
    "风险": "risk",
    "下一步": "next_step",
    "Exit Code": "exit_code",
    "ExitCode": "exit_code",
}


class SubtaskResult(BaseModel):
    status: StatusType
    error_cause: ErrorCauseType | None = None
    conclusion: str = Field(default_factory=str)
    evidence: str = Field(default_factory=str)
    risk: str = Field(default_factory=str)
    next_step: str = Field(default_factory=str)
    exit_code: int | None = None
    task_id: str | None = None

    @model_validator(mode="after")
    def _require_error_cause(self) -> "SubtaskResult":
        if self.status == "失败" and not self.error_cause:
            raise ValueError("error_cause is required when status=失败")
        return self


def _infer_status(raw: str, fallback: StatusType = "成功") -> StatusType:
    text = raw.lower()
    if "高风险" in raw and "拦截" in raw:
        return "高风险拦截"
    if "失败" in raw or "error" in text:
        return "失败"
    return fallback


def infer_error_cause(raw: str) -> ErrorCauseType:
    text = raw.lower()
    if "permission" in text or "权限" in raw:
        return "权限不足"
    if "not found" in text or "不存在" in raw:
        return "路径不存在"
    if "syntax" in text or "语法" in raw:
        return "语法错误"
    if "timeout" in text or "timed out" in text or "超时" in raw:
        return "超时"
    if "blocked" in text or "拦截" in raw:
        return "拦截"
    if "depend" in text or "依赖" in raw:
        return "依赖缺失"
    return "未知"


def parse_subtask_output(raw: str) -> dict[str, str]:
    data: dict[str, str] = {}
    current_key: str | None = None
    lines = raw.splitlines()
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if current_key:
                data[current_key] = (data.get(current_key, "") + "\n").rstrip("\n")
            continue
        match = re.match(r"^([^:：]+)[:：]\s*(.*)$", line_strip)
        if match:
            label = match.group(1).strip()
            value = match.group(2).strip()
            key = _LABELS.get(label)
            if key:
                current_key = key
                data[key] = value
                continue
        if current_key:
            data[current_key] = (data.get(current_key, "") + "\n" + line_strip).strip()
    return data


def format_subtask_output(
    raw: str,
    status_hint: StatusType | None = None,
    error_cause_hint: ErrorCauseType | None = None,
    evidence: str | None = None,
    risk: str | None = None,
    next_step: str | None = None,
    exit_code: int | None = None,
    task_id: str | None = None,
) -> SubtaskResult:
    parsed = parse_subtask_output(raw)
    status: StatusType = status_hint or parsed.get("status") or _infer_status(raw)
    error_cause = parsed.get("error_cause") or error_cause_hint
    if status == "失败" and not error_cause:
        error_cause = infer_error_cause(raw)

    conclusion = parsed.get("conclusion") or raw.strip() or "已完成"
    evidence_val = evidence or parsed.get("evidence") or "无"
    if task_id and "task_id=" not in evidence_val and "Task ID" not in evidence_val and "任务ID" not in evidence_val:
        evidence_val = f"task_id={task_id}; {evidence_val}" if evidence_val else f"task_id={task_id}"
    risk_val = risk or parsed.get("risk") or ("高风险命令需确认" if status == "高风险拦截" else "无")
    next_step_val = next_step or parsed.get("next_step") or (
        "请确认是否调整需求或提供更多信息" if status == "失败" else "无"
    )

    exit_code_val: int | None = exit_code
    if exit_code_val is None:
        raw_exit = parsed.get("exit_code")
        if raw_exit is not None and str(raw_exit).strip().isdigit():
            exit_code_val = int(str(raw_exit).strip())

    try:
        return SubtaskResult(
            status=status,
            error_cause=error_cause,
            conclusion=conclusion,
            evidence=evidence_val,
            risk=risk_val,
            next_step=next_step_val,
            exit_code=exit_code_val,
            task_id=task_id,
        )
    except ValidationError:
        # Repair to minimum valid structure
        if status == "失败" and not error_cause:
            error_cause = "未知"
        return SubtaskResult(
            status=status,
            error_cause=error_cause,
            conclusion=conclusion or "未提供结论",
            evidence=evidence_val,
            risk=risk_val,
            next_step=next_step_val,
            exit_code=exit_code_val,
            task_id=task_id,
        )


def render_subtask_result(result: SubtaskResult) -> str:
    lines = [
        f"状态：{result.status}",
        f"错误归因：{result.error_cause}" if result.status == "失败" else "错误归因：",
        f"结论：{result.conclusion}",
        f"证据：{result.evidence}",
        f"风险：{result.risk}",
        f"下一步：{result.next_step}",
        f"Exit Code：{result.exit_code if result.exit_code is not None else ''}",
    ]
    return "\n".join(lines)
