"""Deterministic hard controls and disposition engine for Mandate-1.

Mandate-1 findings are inputs to this module. They are never binding by
themselves and cannot override identity, authority, action-class, or policy
failures.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

from .safr_contracts import (
    AgentIdentity,
    DispositionDecision,
    GovernanceEnvelope,
    HardControlResult,
    ReviewContract,
    SafrContractError,
    SemanticFindings,
    SignedGovernanceEnvelope,
    verify_envelope_signature,
)

GATE_VERSION = "mandate1-gate-0.1.0"


@dataclass(frozen=True)
class RegisteredAgent:
    identity: AgentIdentity
    active: bool = True


@dataclass(frozen=True)
class OperatorPolicy:
    action_id: str
    expected_side_effect_class: str
    requires_human_approval: bool
    enabled: bool = True


@dataclass(frozen=True)
class GatePolicy:
    policy_version: str
    operators: Mapping[str, OperatorPolicy]
    reviewer_role: str = "authorised_reviewer"
    review_timeout_minutes: int = 120
    review_timeout_fallback: str = "deny"

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise SafrContractError("GatePolicy.policy_version: expected non-empty value")
        if not self.operators:
            raise SafrContractError("GatePolicy.operators: expected at least one operator")
        for action_id, policy in self.operators.items():
            if action_id != policy.action_id:
                raise SafrContractError("GatePolicy.operators: key/action mismatch")
        if not 1 <= self.review_timeout_minutes <= 10_080:
            raise SafrContractError("GatePolicy.review_timeout_minutes: expected 1..10080")
        if self.review_timeout_fallback not in {"deny", "escalate"}:
            raise SafrContractError("GatePolicy.review_timeout_fallback: expected deny or escalate")


def evaluate_hard_controls(
    signed: SignedGovernanceEnvelope,
    *,
    secret: bytes,
    registry: Mapping[str, RegisteredAgent],
    policy: GatePolicy,
    now: datetime,
    seen_nonces: set[str] | None = None,
) -> HardControlResult:
    """Evaluate controls whose truth is available to deterministic code.

    `seen_nonces` belongs to the trusted runtime. This function reads it but does
    not mutate it; callers should persist a nonce only after recording the gate
    decision, so a failed write cannot silently consume authority.
    """
    envelope = signed.envelope
    now = _as_utc(now)
    nonce_fresh = seen_nonces is None or envelope.nonce not in seen_nonces
    signature_valid = verify_envelope_signature(signed, secret=secret)

    registered = registry.get(envelope.identity.agent_id)
    identity_valid = bool(
        registered
        and registered.active
        and registered.identity == envelope.identity
    )
    mandate_valid = bool(
        envelope.mandate.valid_from <= now <= envelope.mandate.valid_until
        and envelope.identity.principal_id == envelope.mandate.principal_id
    )
    action_permitted = envelope.proposed_action.action_id in envelope.mandate.allowed_action_ids
    scope_permitted = envelope.proposed_action.scope in envelope.mandate.allowed_scopes
    step_limit_satisfied = envelope.workflow_step <= envelope.mandate.max_workflow_steps
    policy_available = envelope.policy_version == policy.policy_version
    operator = policy.operators.get(envelope.proposed_action.action_id)
    action_class_consistent = bool(
        operator
        and operator.enabled
        and operator.expected_side_effect_class == envelope.proposed_action.side_effect_class
    )
    requires_human_approval = bool(
        envelope.proposed_action.side_effect_class
        in {"draft_external", "external_communication", "consequential"}
        or (operator and operator.requires_human_approval)
    )

    checks = (
        ("HC-SIGNATURE-001", signature_valid, "envelope_signature_invalid"),
        ("HC-NONCE-001", nonce_fresh, "envelope_nonce_reused"),
        ("HC-IDENTITY-001", identity_valid, "agent_identity_invalid"),
        ("HC-MANDATE-001", mandate_valid, "mandate_invalid_or_expired"),
        ("HC-ACTION-001", action_permitted, "action_not_delegated"),
        ("HC-SCOPE-001", scope_permitted, "scope_not_delegated"),
        ("HC-STEPS-001", step_limit_satisfied, "workflow_step_limit_exceeded"),
        ("HC-POLICY-001", policy_available, "policy_version_unavailable"),
        ("HC-CLASS-001", action_class_consistent, "action_class_mismatch_or_disabled"),
    )
    return HardControlResult(
        signature_valid=signature_valid,
        nonce_fresh=nonce_fresh,
        identity_valid=identity_valid,
        mandate_valid=mandate_valid,
        action_permitted=action_permitted,
        scope_permitted=scope_permitted,
        step_limit_satisfied=step_limit_satisfied,
        policy_available=policy_available,
        action_class_consistent=action_class_consistent,
        requires_human_approval=requires_human_approval,
        applied_rule_ids=tuple(rule for rule, _passed, _reason in checks),
        failure_reasons=tuple(reason for _rule, passed, reason in checks if not passed),
    )


def decide_disposition(
    envelope: GovernanceEnvelope,
    hard: HardControlResult,
    semantic: SemanticFindings | None,
    *,
    now: datetime,
    policy: GatePolicy,
) -> DispositionDecision:
    """Return one binding outcome using an explicit, reviewed priority order."""
    now = _as_utc(now)
    rules = list(hard.applied_rule_ids)

    if not hard.passed:
        return _decision(
            envelope,
            "deny",
            now,
            rules + ["GATE-HARD-DENY-001"],
            hard.failure_reasons or ("hard_control_failed",),
            semantic,
            policy,
        )

    action_class = envelope.proposed_action.side_effect_class
    if action_class == "consequential":
        return _decision(
            envelope,
            "deny",
            now,
            rules + ["GATE-CONSEQUENTIAL-DENY-001"],
            ("consequential_action_not_model_executable",),
            semantic,
            policy,
        )

    if semantic is None or semantic.abstained:
        outcome = "escalate" if action_class in {"internal_read", "internal_write", "draft_external"} else "deny"
        return _decision(
            envelope,
            outcome,
            now,
            rules + ["GATE-SEMANTIC-MISSING-001"],
            ("semantic_assessment_unavailable",),
            semantic,
            policy,
        )

    unknown_evidence = sorted(set(semantic.supporting_evidence_ids) - set(envelope.evidence_ids))
    if semantic.input_digest != envelope.digest or unknown_evidence:
        reasons = ["semantic_input_digest_mismatch"] if semantic.input_digest != envelope.digest else []
        if unknown_evidence:
            reasons.append("semantic_unknown_evidence_reference")
        return _decision(
            envelope,
            "escalate" if action_class != "external_communication" else "deny",
            now,
            rules + ["GATE-SEMANTIC-INTEGRITY-001"],
            tuple(reasons),
            semantic,
            policy,
        )

    if semantic.scope_alignment == "outside" or semantic.instruction_alignment == "conflicts":
        return _decision(
            envelope,
            "deny",
            now,
            rules + ["SC-AUTHORITY-001"],
            ("semantic_authority_conflict",),
            semantic,
            policy,
        )

    if semantic.trace_consistency in {"contradictory", "suspicious"}:
        return _decision(
            envelope,
            "deny" if semantic.trace_consistency == "suspicious" else "escalate",
            now,
            rules + ["SC-TRACE-001"],
            (f"trace_{semantic.trace_consistency}",),
            semantic,
            policy,
        )

    if hard.requires_human_approval or action_class in {"draft_external", "external_communication"}:
        return _decision(
            envelope,
            "escalate",
            now,
            rules + ["GATE-HUMAN-APPROVAL-001"],
            ("human_approval_required",),
            semantic,
            policy,
        )

    if (
        semantic.scope_alignment == "ambiguous"
        or semantic.instruction_alignment == "unclear"
        or semantic.trace_consistency == "incomplete"
        or semantic.evidence_sufficiency in {"none", "weak"}
        or semantic.novelty == "out_of_distribution"
    ):
        return _decision(
            envelope,
            "escalate",
            now,
            rules + ["GATE-SEMANTIC-ESCALATE-001"],
            ("semantic_review_threshold_triggered",),
            semantic,
            policy,
        )

    if action_class != "internal_read":
        return _decision(
            envelope,
            "escalate",
            now,
            rules + ["GATE-NONREAD-ESCALATE-001"],
            ("pilot_only_auto_executes_internal_reads",),
            semantic,
            policy,
        )

    if semantic.novelty in {"unusual", "novel"}:
        return _decision(
            envelope,
            "observe",
            now,
            rules + ["GATE-NOVELTY-OBSERVE-001"],
            ("permitted_internal_read_requires_observation",),
            semantic,
            policy,
        )

    return _decision(
        envelope,
        "auto_execute",
        now,
        rules + ["GATE-INTERNAL-READ-001"],
        ("permitted_reversible_internal_read",),
        semantic,
        policy,
    )


def _decision(
    envelope: GovernanceEnvelope,
    disposition: str,
    now: datetime,
    rules: list[str],
    reasons: tuple[str, ...],
    semantic: SemanticFindings | None,
    policy: GatePolicy,
) -> DispositionDecision:
    seed = "|".join((envelope.envelope_id, disposition, _iso(now), GATE_VERSION))
    decision_id = "dec-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    review = None
    if disposition == "escalate":
        review = ReviewContract(
            reviewer_role=policy.reviewer_role,
            deadline=now + timedelta(minutes=policy.review_timeout_minutes),
            fallback=policy.review_timeout_fallback,
        )
    return DispositionDecision(
        decision_id=decision_id,
        envelope_id=envelope.envelope_id,
        disposition=disposition,
        gate_version=GATE_VERSION,
        applied_rule_ids=tuple(dict.fromkeys(rules)),
        reason_codes=tuple(dict.fromkeys(reasons)),
        semantic_finding_digest=semantic.digest if semantic else None,
        created_at=now,
        review_contract=review,
    )


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SafrContractError("now: expected a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
