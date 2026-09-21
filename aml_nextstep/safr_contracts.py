"""Strict contracts for the synthetic Mandate-1 SAFR-shaped prototype.

The contracts deliberately separate:

* the runtime-authenticated Governance Envelope;
* deterministic hard-control results;
* advisory semantic findings; and
* the binding disposition produced by deterministic code.

This is an unofficial research implementation. It is not an implementation or
certification of any institution's policy and is not endorsed by MAS.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

SCHEMA_VERSION = "mandate1-envelope-v1"
SEMANTIC_SCHEMA_VERSION = "mandate1-semantic-findings-v1"
DECISION_SCHEMA_VERSION = "mandate1-disposition-v1"

SIDE_EFFECT_CLASSES = (
    "internal_read",
    "internal_write",
    "draft_external",
    "external_communication",
    "consequential",
)
SCOPE_ALIGNMENT = ("within", "ambiguous", "outside")
TRACE_CONSISTENCY = ("consistent", "incomplete", "contradictory", "suspicious")
EVIDENCE_SUFFICIENCY = ("none", "weak", "adequate", "strong")
INSTRUCTION_ALIGNMENT = ("supports", "unclear", "conflicts")
NOVELTY = ("routine", "unusual", "novel", "out_of_distribution")
DISPOSITIONS = ("deny", "escalate", "auto_execute", "observe")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class SafrContractError(ValueError):
    """A SAFR prototype record violates its frozen contract."""


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise SafrContractError(f"{name}: expected a stable identifier")
    return value


def _text(value: Any, name: str, *, maximum: int = 20_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SafrContractError(f"{name}: expected non-empty text")
    if len(value) > maximum:
        raise SafrContractError(f"{name}: exceeds {maximum} characters")
    return value


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SafrContractError(f"{name}: expected a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_RE.fullmatch(value):
        raise SafrContractError(f"{name}: expected a lowercase SHA-256 digest")
    return value


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise SafrContractError(f"{name}: expected one of {choices}")
    return str(value)


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SafrContractError(f"{name}: expected a JSON object")
    result = dict(value)
    canonical_json(result)
    return result


def canonical_json(value: Any) -> str:
    """Canonical JSON used for hashes, signatures, manifests, and audit entries."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise SafrContractError("record contains non-JSON or non-finite data") from error


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _serialise(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_serialise(item) for item in value]
    if isinstance(value, list):
        return [_serialise(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    agent_version: str
    principal_id: str
    registry_id: str

    def __post_init__(self) -> None:
        for name in ("agent_id", "agent_version", "principal_id", "registry_id"):
            _identifier(getattr(self, name), f"AgentIdentity.{name}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Mandate:
    mandate_id: str
    version: str
    principal_id: str
    purpose: str
    allowed_action_ids: tuple[str, ...]
    allowed_scopes: tuple[str, ...]
    valid_from: datetime
    valid_until: datetime
    max_workflow_steps: int

    def __post_init__(self) -> None:
        _identifier(self.mandate_id, "Mandate.mandate_id")
        _identifier(self.version, "Mandate.version")
        _identifier(self.principal_id, "Mandate.principal_id")
        _text(self.purpose, "Mandate.purpose", maximum=4_000)
        if not self.allowed_action_ids or len(set(self.allowed_action_ids)) != len(self.allowed_action_ids):
            raise SafrContractError("Mandate.allowed_action_ids: expected unique non-empty values")
        if not self.allowed_scopes or len(set(self.allowed_scopes)) != len(self.allowed_scopes):
            raise SafrContractError("Mandate.allowed_scopes: expected unique non-empty values")
        for index, value in enumerate(self.allowed_action_ids):
            _identifier(value, f"Mandate.allowed_action_ids[{index}]")
        for index, value in enumerate(self.allowed_scopes):
            _identifier(value, f"Mandate.allowed_scopes[{index}]")
        start = _utc(self.valid_from, "Mandate.valid_from")
        end = _utc(self.valid_until, "Mandate.valid_until")
        if start >= end:
            raise SafrContractError("Mandate: valid_from must precede valid_until")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_until", end)
        if isinstance(self.max_workflow_steps, bool) or not 1 <= self.max_workflow_steps <= 100:
            raise SafrContractError("Mandate.max_workflow_steps: expected an integer from 1 to 100")

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ProposedAction:
    action_id: str
    action_version: str
    scope: str
    side_effect_class: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.action_id, "ProposedAction.action_id")
        _identifier(self.action_version, "ProposedAction.action_version")
        _identifier(self.scope, "ProposedAction.scope")
        _choice(self.side_effect_class, "ProposedAction.side_effect_class", SIDE_EFFECT_CLASSES)
        object.__setattr__(self, "parameters", _json_mapping(self.parameters, "ProposedAction.parameters"))

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    captured_by: str
    tool_id: str | None = None
    data_ids: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        _identifier(self.event_id, "TraceEvent.event_id")
        _identifier(self.event_type, "TraceEvent.event_type")
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, "TraceEvent.occurred_at"))
        if self.captured_by != "trusted_runtime":
            raise SafrContractError("TraceEvent.captured_by: authoritative trace must come from trusted_runtime")
        if self.tool_id is not None:
            _identifier(self.tool_id, "TraceEvent.tool_id")
        if len(set(self.data_ids)) != len(self.data_ids):
            raise SafrContractError("TraceEvent.data_ids: duplicate identifiers")
        for index, value in enumerate(self.data_ids):
            _identifier(value, f"TraceEvent.data_ids[{index}]")
        if self.detail:
            _text(self.detail, "TraceEvent.detail", maximum=2_000)

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    evidence_type: str
    source: str
    summary: str
    observed_at: datetime
    available_at: datetime
    content_digest: str

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "EvidenceItem.evidence_id")
        _identifier(self.evidence_type, "EvidenceItem.evidence_type")
        _identifier(self.source, "EvidenceItem.source")
        _text(self.summary, "EvidenceItem.summary", maximum=4_000)
        observed = _utc(self.observed_at, "EvidenceItem.observed_at")
        available = _utc(self.available_at, "EvidenceItem.available_at")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "available_at", available)
        _digest(self.content_digest, "EvidenceItem.content_digest")

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


@dataclass(frozen=True)
class GovernanceEnvelope:
    envelope_id: str
    case_id: str
    case_revision: int
    workflow_step: int
    created_at: datetime
    information_cutoff: datetime
    nonce: str
    identity: AgentIdentity
    mandate: Mandate
    original_instruction: str
    proposed_action: ProposedAction
    trace: tuple[TraceEvent, ...]
    evidence: tuple[EvidenceItem, ...]
    policy_version: str
    case_state_digest: str
    parent_envelope_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SafrContractError(f"GovernanceEnvelope.schema_version: expected {SCHEMA_VERSION}")
        _identifier(self.envelope_id, "GovernanceEnvelope.envelope_id")
        _identifier(self.case_id, "GovernanceEnvelope.case_id")
        if self.parent_envelope_id is not None:
            _identifier(self.parent_envelope_id, "GovernanceEnvelope.parent_envelope_id")
            if self.parent_envelope_id == self.envelope_id:
                raise SafrContractError("GovernanceEnvelope.parent_envelope_id: self-reference")
        if isinstance(self.case_revision, bool) or self.case_revision < 0:
            raise SafrContractError("GovernanceEnvelope.case_revision: expected a non-negative integer")
        if isinstance(self.workflow_step, bool) or self.workflow_step < 1:
            raise SafrContractError("GovernanceEnvelope.workflow_step: expected a positive integer")
        created = _utc(self.created_at, "GovernanceEnvelope.created_at")
        cutoff = _utc(self.information_cutoff, "GovernanceEnvelope.information_cutoff")
        if cutoff > created:
            raise SafrContractError("GovernanceEnvelope.information_cutoff: cannot be after creation")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "information_cutoff", cutoff)
        _identifier(self.nonce, "GovernanceEnvelope.nonce")
        _text(self.original_instruction, "GovernanceEnvelope.original_instruction", maximum=8_000)
        _identifier(self.policy_version, "GovernanceEnvelope.policy_version")
        _digest(self.case_state_digest, "GovernanceEnvelope.case_state_digest")
        if self.identity.principal_id != self.mandate.principal_id:
            raise SafrContractError("GovernanceEnvelope: principal does not match mandate")
        trace_ids = [event.event_id for event in self.trace]
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(trace_ids) != len(set(trace_ids)):
            raise SafrContractError("GovernanceEnvelope.trace: duplicate event IDs")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise SafrContractError("GovernanceEnvelope.evidence: duplicate evidence IDs")
        late = [item.evidence_id for item in self.evidence if item.available_at > cutoff]
        if late:
            raise SafrContractError(f"GovernanceEnvelope.evidence: unavailable at cutoff: {late}")

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "parent_envelope_id": self.parent_envelope_id,
            "case_id": self.case_id,
            "case_revision": self.case_revision,
            "workflow_step": self.workflow_step,
            "created_at": _iso(self.created_at),
            "information_cutoff": _iso(self.information_cutoff),
            "nonce": self.nonce,
            "identity": self.identity.to_dict(),
            "mandate": self.mandate.to_dict(),
            "original_instruction": self.original_instruction,
            "proposed_action": self.proposed_action.to_dict(),
            "trace": [event.to_dict() for event in self.trace],
            "evidence": [item.to_dict() for item in self.evidence],
            "policy_version": self.policy_version,
            "case_state_digest": self.case_state_digest,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class SignedGovernanceEnvelope:
    envelope: GovernanceEnvelope
    key_id: str
    signature: str
    algorithm: str = "HMAC-SHA256-DEMO"

    def __post_init__(self) -> None:
        _identifier(self.key_id, "SignedGovernanceEnvelope.key_id")
        _digest(self.signature, "SignedGovernanceEnvelope.signature")
        if self.algorithm != "HMAC-SHA256-DEMO":
            raise SafrContractError("SignedGovernanceEnvelope.algorithm: unsupported demo algorithm")

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope": self.envelope.to_dict(),
            "authentication": {
                "key_id": self.key_id,
                "algorithm": self.algorithm,
                "signature": self.signature,
            },
        }


def sign_envelope(envelope: GovernanceEnvelope, *, key_id: str, secret: bytes) -> SignedGovernanceEnvelope:
    _identifier(key_id, "key_id")
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise SafrContractError("secret: demo HMAC key must contain at least 16 bytes")
    signature = hmac.new(secret, canonical_json(envelope.to_dict()).encode("utf-8"), hashlib.sha256).hexdigest()
    return SignedGovernanceEnvelope(envelope=envelope, key_id=key_id, signature=signature)


def verify_envelope_signature(signed: SignedGovernanceEnvelope, *, secret: bytes) -> bool:
    if not isinstance(secret, bytes) or len(secret) < 16:
        return False
    expected = hmac.new(
        secret,
        canonical_json(signed.envelope.to_dict()).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signed.signature)


@dataclass(frozen=True)
class SemanticFindings:
    scope_alignment: str
    trace_consistency: str
    evidence_sufficiency: str
    instruction_alignment: str
    novelty: str
    supporting_evidence_ids: tuple[str, ...]
    model_id: str
    model_revision: str
    input_digest: str
    abstained: bool = False
    schema_version: str = SEMANTIC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_SCHEMA_VERSION:
            raise SafrContractError(f"SemanticFindings.schema_version: expected {SEMANTIC_SCHEMA_VERSION}")
        _choice(self.scope_alignment, "SemanticFindings.scope_alignment", SCOPE_ALIGNMENT)
        _choice(self.trace_consistency, "SemanticFindings.trace_consistency", TRACE_CONSISTENCY)
        _choice(self.evidence_sufficiency, "SemanticFindings.evidence_sufficiency", EVIDENCE_SUFFICIENCY)
        _choice(self.instruction_alignment, "SemanticFindings.instruction_alignment", INSTRUCTION_ALIGNMENT)
        _choice(self.novelty, "SemanticFindings.novelty", NOVELTY)
        if len(set(self.supporting_evidence_ids)) != len(self.supporting_evidence_ids):
            raise SafrContractError("SemanticFindings.supporting_evidence_ids: duplicate identifiers")
        for index, value in enumerate(self.supporting_evidence_ids):
            _identifier(value, f"SemanticFindings.supporting_evidence_ids[{index}]")
        _identifier(self.model_id, "SemanticFindings.model_id")
        _identifier(self.model_revision, "SemanticFindings.model_revision")
        _digest(self.input_digest, "SemanticFindings.input_digest")
        if not isinstance(self.abstained, bool):
            raise SafrContractError("SemanticFindings.abstained: expected boolean")

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class HardControlResult:
    signature_valid: bool
    nonce_fresh: bool
    identity_valid: bool
    mandate_valid: bool
    action_permitted: bool
    scope_permitted: bool
    step_limit_satisfied: bool
    policy_available: bool
    action_class_consistent: bool
    requires_human_approval: bool
    applied_rule_ids: tuple[str, ...]
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "signature_valid",
            "nonce_fresh",
            "identity_valid",
            "mandate_valid",
            "action_permitted",
            "scope_permitted",
            "step_limit_satisfied",
            "policy_available",
            "action_class_consistent",
            "requires_human_approval",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SafrContractError(f"HardControlResult.{name}: expected boolean")
        if len(set(self.applied_rule_ids)) != len(self.applied_rule_ids):
            raise SafrContractError("HardControlResult.applied_rule_ids: duplicate identifiers")
        for index, value in enumerate(self.applied_rule_ids):
            _identifier(value, f"HardControlResult.applied_rule_ids[{index}]")
        for index, value in enumerate(self.failure_reasons):
            _text(value, f"HardControlResult.failure_reasons[{index}]", maximum=500)

    @property
    def passed(self) -> bool:
        return all(
            (
                self.signature_valid,
                self.nonce_fresh,
                self.identity_valid,
                self.mandate_valid,
                self.action_permitted,
                self.scope_permitted,
                self.step_limit_satisfied,
                self.policy_available,
                self.action_class_consistent,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


@dataclass(frozen=True)
class ReviewContract:
    reviewer_role: str
    deadline: datetime
    fallback: str

    def __post_init__(self) -> None:
        _identifier(self.reviewer_role, "ReviewContract.reviewer_role")
        object.__setattr__(self, "deadline", _utc(self.deadline, "ReviewContract.deadline"))
        _choice(self.fallback, "ReviewContract.fallback", ("deny", "escalate"))

    def to_dict(self) -> dict[str, Any]:
        return _serialise(asdict(self))


@dataclass(frozen=True)
class DispositionDecision:
    decision_id: str
    envelope_id: str
    disposition: str
    gate_version: str
    applied_rule_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    semantic_finding_digest: str | None
    created_at: datetime
    review_contract: ReviewContract | None = None
    binding: bool = True
    schema_version: str = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise SafrContractError(f"DispositionDecision.schema_version: expected {DECISION_SCHEMA_VERSION}")
        _identifier(self.decision_id, "DispositionDecision.decision_id")
        _identifier(self.envelope_id, "DispositionDecision.envelope_id")
        _choice(self.disposition, "DispositionDecision.disposition", DISPOSITIONS)
        _identifier(self.gate_version, "DispositionDecision.gate_version")
        for index, value in enumerate(self.applied_rule_ids):
            _identifier(value, f"DispositionDecision.applied_rule_ids[{index}]")
        for index, value in enumerate(self.reason_codes):
            _identifier(value, f"DispositionDecision.reason_codes[{index}]")
        if self.semantic_finding_digest is not None:
            _digest(self.semantic_finding_digest, "DispositionDecision.semantic_finding_digest")
        object.__setattr__(self, "created_at", _utc(self.created_at, "DispositionDecision.created_at"))
        if self.disposition == "escalate" and self.review_contract is None:
            raise SafrContractError("DispositionDecision.review_contract: required for escalation")
        if self.disposition != "escalate" and self.review_contract is not None:
            raise SafrContractError("DispositionDecision.review_contract: only allowed for escalation")
        if self.binding is not True:
            raise SafrContractError("DispositionDecision.binding: gate decisions are binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            **_serialise(asdict(self)),
            "review_contract": self.review_contract.to_dict() if self.review_contract else None,
        }


def decimal_parameter(value: Any, name: str) -> Decimal:
    """Parse a finite non-negative decimal action parameter without binary-float drift."""
    if isinstance(value, bool):
        raise SafrContractError(f"{name}: expected a decimal value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise SafrContractError(f"{name}: expected a decimal value") from error
    if not result.is_finite() or result < 0:
        raise SafrContractError(f"{name}: expected a finite non-negative decimal value")
    return result
