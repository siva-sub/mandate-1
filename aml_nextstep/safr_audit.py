"""Append-only, hash-linked audit records for the local Mandate-1 demonstration.

The hash chain detects modification within its stated threat model. A JSONL file
on one machine is not a regulated immutable ledger and must not be described as
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .safr_contracts import SafrContractError, canonical_json, sha256_json

AUDIT_SCHEMA_VERSION = "mandate1-audit-entry-v1"
GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    recorded_at: datetime
    event_type: str
    subject_id: str
    payload: Mapping[str, Any]
    previous_hash: str
    entry_hash: str
    schema_version: str = AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise SafrContractError("AuditEntry.schema_version: unsupported version")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise SafrContractError("AuditEntry.sequence: expected a non-negative integer")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise SafrContractError("AuditEntry.recorded_at: expected timezone-aware datetime")
        if not self.event_type or not self.subject_id:
            raise SafrContractError("AuditEntry: event_type and subject_id are required")
        canonical_json(dict(self.payload))
        for name, value in (("previous_hash", self.previous_hash), ("entry_hash", self.entry_hash)):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise SafrContractError(f"AuditEntry.{name}: expected SHA-256 hex")

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "recorded_at": _iso(self.recorded_at),
            "event_type": self.event_type,
            "subject_id": self.subject_id,
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.content_dict(), "entry_hash": self.entry_hash}


class HashChainAuditLog:
    """Small local append-only log with deterministic verification."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[AuditEntry] = []
        if self.path is not None and self.path.exists():
            self._entries = list(read_entries(self.path))
            ok, reason = verify_chain(self._entries)
            if not ok:
                raise SafrContractError(f"existing audit chain invalid: {reason}")

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def append(
        self,
        *,
        event_type: str,
        subject_id: str,
        payload: Mapping[str, Any],
        recorded_at: datetime,
    ) -> AuditEntry:
        previous = self._entries[-1].entry_hash if self._entries else GENESIS_HASH
        content = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "sequence": len(self._entries),
            "recorded_at": _iso(recorded_at),
            "event_type": event_type,
            "subject_id": subject_id,
            "payload": dict(payload),
            "previous_hash": previous,
        }
        entry = AuditEntry(
            sequence=content["sequence"],
            recorded_at=recorded_at.astimezone(timezone.utc),
            event_type=event_type,
            subject_id=subject_id,
            payload=dict(payload),
            previous_hash=previous,
            entry_hash=sha256_json(content),
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(entry.to_dict()) + "\n")
                handle.flush()
        self._entries.append(entry)
        return entry


def read_entries(path: str | Path) -> Iterable[AuditEntry]:
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            recorded_at = datetime.fromisoformat(value["recorded_at"].replace("Z", "+00:00"))
            yield AuditEntry(
                sequence=value["sequence"],
                recorded_at=recorded_at,
                event_type=value["event_type"],
                subject_id=value["subject_id"],
                payload=value["payload"],
                previous_hash=value["previous_hash"],
                entry_hash=value["entry_hash"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SafrContractError(f"audit line {line_number}: invalid entry") from error


def verify_chain(entries: Iterable[AuditEntry]) -> tuple[bool, str | None]:
    previous = GENESIS_HASH
    for expected_sequence, entry in enumerate(entries):
        if entry.sequence != expected_sequence:
            return False, f"sequence {entry.sequence} should be {expected_sequence}"
        if entry.previous_hash != previous:
            return False, f"entry {entry.sequence} previous hash mismatch"
        expected = sha256_json(entry.content_dict())
        if entry.entry_hash != expected:
            return False, f"entry {entry.sequence} content hash mismatch"
        previous = entry.entry_hash
    return True, None


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SafrContractError("audit timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
