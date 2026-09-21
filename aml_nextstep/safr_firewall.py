"""Raise-only semantic action firewall for SAFR-shaped governance envelopes.

The deterministic floor authenticates identity, mandate, permissions, limits, and
replay state. This module represents the narrower learned job: identify semantic
exceptions in unstructured instructions, traces, and evidence, with citations.

A model result can raise a disposition to human review. It cannot authorize an
action, weaken a deterministic result, create authority, or issue a legal/AML
conclusion. Deployments start in ``shadow`` mode and graduate to ``raise_only``
only after outcome-based calibration on their own action stream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .safr_contracts import DispositionDecision, GovernanceEnvelope, SafrContractError

FIREWALL_SCHEMA_VERSION = "safr-semantic-firewall-v0.1"

NO_EXCEPTION = "no_semantic_exception"
EXCEPTION_ACTIONS = (
    "request_missing_evidence",
    "route_scope_review",
    "quarantine_untrusted_instruction",
    "resolve_trace_conflict",
    "request_instruction_clarification",
    "route_novel_context",
)
FIREWALL_ACTIONS = (NO_EXCEPTION, *EXCEPTION_ACTIONS)

ACTION_DESCRIPTIONS = {
    NO_EXCEPTION: "No supplied semantic exception requires an additional hold.",
    "request_missing_evidence": "Required evidence for the declared rationale or action is absent or inadequate.",
    "route_scope_review": "The relationship between the proposed action and the mandate is ambiguous or outside its natural-language boundaries.",
    "quarantine_untrusted_instruction": "Untrusted content appears to redirect, override, or manipulate the agent's action.",
    "resolve_trace_conflict": "The trusted trace and the declared action, target, or rationale materially disagree.",
    "request_instruction_clarification": "The proposed action conflicts with, or cannot be justified from, the principal's instruction.",
    "route_novel_context": "The action-context combination lacks a sufficiently close reviewed analogue.",
}


@dataclass(frozen=True)
class FirewallSpan:
    span_id: str
    source: str
    text: str
    trust: str = "trusted"

    def __post_init__(self) -> None:
        _identifier(self.span_id, "FirewallSpan.span_id")
        _identifier(self.source, "FirewallSpan.source")
        _text(self.text, "FirewallSpan.text", maximum=4_000)
        if self.trust not in {"trusted", "untrusted", "declared"}:
            raise SafrContractError("FirewallSpan.trust: expected trusted, untrusted, or declared")

    def to_dict(self) -> dict[str, str]:
        return {
            "span_id": self.span_id,
            "source": self.source,
            "trust": self.trust,
            "text": self.text,
        }


@dataclass(frozen=True)
class FirewallInput:
    case_id: str
    envelope_id: str
    envelope_digest: str
    action_id: str
    spans: tuple[FirewallSpan, ...]

    def __post_init__(self) -> None:
        _identifier(self.case_id, "FirewallInput.case_id")
        _identifier(self.envelope_id, "FirewallInput.envelope_id")
        _digest(self.envelope_digest, "FirewallInput.envelope_digest")
        _identifier(self.action_id, "FirewallInput.action_id")
        if not 4 <= len(self.spans) <= 24:
            raise SafrContractError("FirewallInput.spans: expected 4 to 24 spans")
        ids = [span.span_id for span in self.spans]
        if len(ids) != len(set(ids)):
            raise SafrContractError("FirewallInput.spans: duplicate span IDs")

    @property
    def span_ids(self) -> tuple[str, ...]:
        return tuple(span.span_id for span in self.spans)

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": FIREWALL_SCHEMA_VERSION,
            "case_id": self.case_id,
            "envelope_id": self.envelope_id,
            "envelope_digest": self.envelope_digest,
            "action_id": self.action_id,
            "spans": [span.to_dict() for span in self.spans],
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIREWALL_SCHEMA_VERSION,
            "case_id": self.case_id,
            "envelope_id": self.envelope_id,
            "envelope_digest": self.envelope_digest,
            "action_id": self.action_id,
            "spans": [span.to_dict() for span in self.spans],
            "input_digest": self.digest,
        }


@dataclass(frozen=True)
class FirewallFinding:
    control_action: str
    supporting_span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.control_action not in FIREWALL_ACTIONS:
            raise SafrContractError(f"FirewallFinding.control_action: unknown {self.control_action!r}")
        if not 1 <= len(self.supporting_span_ids) <= 4:
            raise SafrContractError("FirewallFinding.supporting_span_ids: expected 1 to 4 citations")
        for index, span_id in enumerate(self.supporting_span_ids):
            _identifier(span_id, f"FirewallFinding.supporting_span_ids[{index}]")
        if len(self.supporting_span_ids) != len(set(self.supporting_span_ids)):
            raise SafrContractError("FirewallFinding.supporting_span_ids: duplicate citations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_action": self.control_action,
            "supporting_span_ids": list(self.supporting_span_ids),
        }


@dataclass(frozen=True)
class FirewallAssessment:
    input_digest: str
    findings: tuple[FirewallFinding, ...]
    model_id: str
    model_revision: str
    inference_status: str = "valid"

    def __post_init__(self) -> None:
        _digest(self.input_digest, "FirewallAssessment.input_digest")
        _text(self.model_id, "FirewallAssessment.model_id", maximum=200)
        _text(self.model_revision, "FirewallAssessment.model_revision", maximum=200)
        if self.inference_status not in {"valid", "abstained", "malformed", "unavailable"}:
            raise SafrContractError("FirewallAssessment.inference_status: invalid status")
        if self.inference_status == "valid" and not self.findings:
            raise SafrContractError("FirewallAssessment.findings: a valid result requires at least one finding")
        actions = [finding.control_action for finding in self.findings]
        if len(actions) != len(set(actions)):
            raise SafrContractError("FirewallAssessment.findings: duplicate control actions")
        if NO_EXCEPTION in actions and len(actions) != 1:
            raise SafrContractError("no_semantic_exception cannot coexist with an exception")

    @property
    def has_exception(self) -> bool:
        return any(finding.control_action in EXCEPTION_ACTIONS for finding in self.findings)

    def validate_against(self, value: FirewallInput) -> None:
        if self.input_digest != value.digest:
            raise SafrContractError("firewall assessment does not bind to this input")
        allowed = set(value.span_ids)
        cited = {
            span_id
            for finding in self.findings
            for span_id in finding.supporting_span_ids
        }
        unknown = sorted(cited - allowed)
        if unknown:
            raise SafrContractError(f"firewall assessment cites unknown spans: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIREWALL_SCHEMA_VERSION,
            "input_digest": self.input_digest,
            "findings": [finding.to_dict() for finding in self.findings],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "inference_status": self.inference_status,
        }


@dataclass(frozen=True)
class FirewallPolicy:
    mode: str = "shadow"
    failure_disposition: str = "escalate"

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "raise_only"}:
            raise SafrContractError("FirewallPolicy.mode: expected shadow or raise_only")
        if self.failure_disposition not in {"escalate", "deny"}:
            raise SafrContractError("FirewallPolicy.failure_disposition: expected escalate or deny")


@dataclass(frozen=True)
class FirewallOutcome:
    disposition: str
    reason_codes: tuple[str, ...]
    model_raised_disposition: bool
    shadow_findings: tuple[str, ...]
    decided_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIREWALL_SCHEMA_VERSION,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "model_raised_disposition": self.model_raised_disposition,
            "shadow_findings": list(self.shadow_findings),
            "decided_at": self.decided_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


def apply_semantic_firewall(
    base_decision: DispositionDecision,
    envelope: GovernanceEnvelope,
    assessment: FirewallAssessment,
    firewall_input: FirewallInput,
    *,
    policy: FirewallPolicy = FirewallPolicy(),
    now: datetime | None = None,
) -> FirewallOutcome:
    """Combine a binding deterministic decision with a raise-only semantic result.

    ``base_decision`` is already binding. The firewall may preserve it or make it
    stricter. It never turns deny/escalate/observe into auto_execute and never
    treats malformed or unavailable inference as clearance.
    """
    if base_decision.envelope_id != envelope.envelope_id:
        raise SafrContractError("base decision and envelope IDs differ")
    if firewall_input.envelope_id != envelope.envelope_id or firewall_input.envelope_digest != envelope.digest:
        raise SafrContractError("firewall input does not bind to the decided envelope")
    assessment.validate_against(firewall_input)
    return combine_firewall_floor(
        base_disposition=base_decision.disposition,
        assessment=assessment,
        policy=policy,
        now=now,
    )


def combine_firewall_floor(
    *,
    base_disposition: str,
    assessment: FirewallAssessment,
    policy: FirewallPolicy = FirewallPolicy(),
    now: datetime | None = None,
) -> FirewallOutcome:
    """Pure disposition lattice used when projection binding is recorded elsewhere."""
    if base_disposition not in {"deny", "escalate", "observe", "auto_execute"}:
        raise SafrContractError("base_disposition: unknown SAFR outcome")
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise SafrContractError("now: expected timezone-aware datetime")
    actions = tuple(finding.control_action for finding in assessment.findings)
    if policy.mode == "shadow":
        return FirewallOutcome(
            disposition=base_disposition,
            reason_codes=("semantic_firewall_shadow_only",),
            model_raised_disposition=False,
            shadow_findings=actions,
            decided_at=timestamp,
        )
    if assessment.inference_status != "valid":
        return FirewallOutcome(
            disposition=_higher_disposition(base_disposition, policy.failure_disposition),
            reason_codes=(f"semantic_firewall_{assessment.inference_status}",),
            model_raised_disposition=base_disposition not in {"deny", policy.failure_disposition},
            shadow_findings=actions,
            decided_at=timestamp,
        )
    if assessment.has_exception:
        raised = _higher_disposition(base_disposition, "escalate")
        return FirewallOutcome(
            disposition=raised,
            reason_codes=tuple(f"semantic_exception:{action}" for action in actions),
            model_raised_disposition=raised != base_disposition,
            shadow_findings=(),
            decided_at=timestamp,
        )
    return FirewallOutcome(
        disposition=base_disposition,
        reason_codes=("semantic_firewall_no_exception",),
        model_raised_disposition=False,
        shadow_findings=(),
        decided_at=timestamp,
    )


def validate_finding_set(
    findings: Iterable[FirewallFinding],
    *,
    allowed_span_ids: Iterable[str],
) -> tuple[FirewallFinding, ...]:
    values = tuple(findings)
    if not values:
        raise SafrContractError("at least one firewall finding is required")
    assessment = FirewallAssessment(
        input_digest="0" * 64,
        findings=values,
        model_id="validation-only",
        model_revision="validation-only",
    )
    allowed = set(allowed_span_ids)
    unknown = sorted(
        span_id
        for finding in assessment.findings
        for span_id in finding.supporting_span_ids
        if span_id not in allowed
    )
    if unknown:
        raise SafrContractError(f"unknown firewall citations: {unknown}")
    return values


def _higher_disposition(left: str, right: str) -> str:
    rank = {"auto_execute": 0, "observe": 1, "escalate": 2, "deny": 3}
    return left if rank[left] >= rank[right] else right


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise SafrContractError(f"{name}: expected a 1 to 128 character identifier")
    if not all(character.isalnum() or character in "-_.:" for character in value):
        raise SafrContractError(f"{name}: invalid identifier")
    return value


def _text(value: Any, name: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SafrContractError(f"{name}: expected non-empty text up to {maximum} characters")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SafrContractError(f"{name}: expected a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SafrContractError(f"{name}: expected a SHA-256 hex digest") from error
    return value
