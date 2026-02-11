"""Confirmation store for high-risk commands."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class ConfirmationRecord:
    """Record binding a confirmation ID to a command and context."""

    id: str
    command: str
    arguments: dict[str, Any]
    working_dir: str | None
    expires_at: datetime
    used: bool = False


class ConfirmationStore:
    """In-memory confirmation store with TTL and one-time use."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._records: dict[str, ConfirmationRecord] = {}

    def create(self, command: str, arguments: dict[str, Any], working_dir: str | None) -> ConfirmationRecord:
        self._purge_expired()
        confirm_id = self._generate_id()
        now = datetime.now(timezone.utc)
        record = ConfirmationRecord(
            id=confirm_id,
            command=command,
            arguments=arguments,
            working_dir=working_dir,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
        )
        self._records[confirm_id] = record
        return record

    def consume(self, confirm_id: str) -> ConfirmationRecord | None:
        self._purge_expired()
        record = self._records.get(confirm_id)
        if not record or record.used:
            return None
        record.used = True
        return record

    def list_pending_ids(self) -> list[str]:
        self._purge_expired()
        return [cid for cid, rec in self._records.items() if not rec.used]

    def peek(self, confirm_id: str) -> ConfirmationRecord | None:
        self._purge_expired()
        record = self._records.get(confirm_id)
        if not record or record.used:
            return None
        return record

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [cid for cid, rec in self._records.items() if rec.expires_at <= now or rec.used]
        for cid in expired:
            self._records.pop(cid, None)

    def _generate_id(self) -> str:
        for _ in range(10):
            token = secrets.token_hex(2).upper()
            confirm_id = f"CONFIRM-{token}"
            if confirm_id not in self._records:
                return confirm_id
        # fallback: longer token
        token = secrets.token_hex(3).upper()
        return f"CONFIRM-{token}"
