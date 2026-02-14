"""Command guardrails and classification utilities."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


_DANGEROUS_SHELL_TOKENS = ["|", ";", "&&", "||", ">", ">>", "<", "$(", "`"]

_READONLY_ALLOW_CMDS = {"ls", "pwd", "cat", "rg", "head", "tail", "grep"}
_READONLY_GIT_SUBCMDS = {"status", "diff"}

_SENSITIVE_KEYWORDS_RE = re.compile(r"(?i)(password|secret|token|key|private|密码|口令|密钥|私钥|令牌)")
_SENSITIVE_PATH_RE = re.compile(
    r"(?i)(\.ssh|\.env|/etc/|/var/|/root/|id_rsa|authorized_keys|known_hosts|\.aws)"
)

# Only direct disk write operations
_HIGH_RISK_PATTERNS = [
    r">\s*/dev/",
]


@dataclass(frozen=True)
class CommandCheck:
    """Classification of a shell command under guardrails."""

    readonly_allowed: bool
    high_risk: bool
    sensitive: bool
    dangerous_syntax: bool


def has_dangerous_shell_syntax(command: str) -> bool:
    cmd = command.strip()
    return any(token in cmd for token in _DANGEROUS_SHELL_TOKENS)


def has_sensitive_keywords(command: str) -> bool:
    return bool(_SENSITIVE_KEYWORDS_RE.search(command))


def has_sensitive_paths(command: str) -> bool:
    return bool(_SENSITIVE_PATH_RE.search(command))


def has_sensitive_tokens(command: str) -> bool:
    return bool(has_sensitive_keywords(command) or has_sensitive_paths(command))


def is_high_risk_command(command: str) -> bool:
    lower = command.strip().lower()
    for pattern in _HIGH_RISK_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def _is_readonly_find(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "find":
        return False
    if "-exec" in tokens or "-delete" in tokens:
        return False
    if "-type" not in tokens:
        return False
    idx = tokens.index("-type")
    if idx + 1 >= len(tokens) or tokens[idx + 1] != "f":
        return False
    # Disallow any other flags besides -type
    for tok in tokens[1:]:
        if tok.startswith("-") and tok != "-type":
            return False
    return True


def _split_pipeline(command: str) -> list[list[str]] | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if "|" not in tokens:
        return None
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok == "|":
            if not current:
                return None
            segments.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments if len(segments) >= 2 else None


def is_readonly_pipeline(command: str) -> bool:
    segments = _split_pipeline(command)
    if not segments:
        return False
    for seg in segments:
        cmd = " ".join(seg)
        if not is_readonly_command(cmd):
            return False
        if has_dangerous_shell_syntax(cmd):
            return False
    return True


def is_readonly_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    head = tokens[0]
    if head in _READONLY_ALLOW_CMDS:
        return True
    if head == "git" and len(tokens) >= 2 and tokens[1] in _READONLY_GIT_SUBCMDS:
        return True
    if head == "find" and _is_readonly_find(tokens):
        return True
    if is_readonly_pipeline(command):
        return True
    return False


def classify_command(command: str) -> CommandCheck:
    readonly_allowed = is_readonly_command(command)
    dangerous = has_dangerous_shell_syntax(command)
    return CommandCheck(
        readonly_allowed=readonly_allowed,
        high_risk=is_high_risk_command(command),
        sensitive=has_sensitive_tokens(command),
        dangerous_syntax=dangerous,
    )
