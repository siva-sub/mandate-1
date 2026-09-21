import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from aml_nextstep.safr_contracts import (
    AgentIdentity,
    EvidenceItem,
    GovernanceEnvelope,
    Mandate,
    ProposedAction,
    SafrContractError,
    TraceEvent,
    sign_envelope,
    verify_envelope_signature,
)


class SafrContractTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)
        self.identity = AgentIdentity("agent-demo", "0.1.0", "principal-1", "registry-1")
        self.mandate = Mandate(
            mandate_id="mandate-1",
            version="1.0",
            principal_id="principal-1",
            purpose="Retrieve internal evidence for a synthetic case.",
            allowed_action_ids=("retrieve-record",),
            allowed_scopes=("internal-case",),
            valid_from=self.now - timedelta(days=1),
            valid_until=self.now + timedelta(days=1),
            max_workflow_steps=3,
        )
        summary = "A current internal record is available."
        self.evidence = EvidenceItem(
            evidence_id="ev-1",
            evidence_type="internal-record",
            source="system-1",
            summary=summary,
            observed_at=self.now - timedelta(hours=1),
            available_at=self.now - timedelta(minutes=5),
            content_digest=__import__("hashlib").sha256(summary.encode()).hexdigest(),
        )
        self.envelope = GovernanceEnvelope(
            envelope_id="env-1",
            case_id="case-1",
            case_revision=1,
            workflow_step=1,
            created_at=self.now,
            information_cutoff=self.now - timedelta(minutes=1),
            nonce="nonce-1",
            identity=self.identity,
            mandate=self.mandate,
            original_instruction="Retrieve only the internal record.",
            proposed_action=ProposedAction(
                action_id="retrieve-record",
                action_version="1.0",
                scope="internal-case",
                side_effect_class="internal_read",
                parameters={"record_id": "record-1"},
            ),
            trace=(
                TraceEvent(
                    event_id="trace-1",
                    event_type="instruction_received",
                    occurred_at=self.now - timedelta(minutes=2),
                    captured_by="trusted_runtime",
                ),
            ),
            evidence=(self.evidence,),
            policy_version="policy-1",
            case_state_digest="a" * 64,
        )

    def test_demo_signature_binds_the_complete_envelope(self):
        signed = sign_envelope(self.envelope, key_id="key-1", secret=b"0123456789abcdef")
        self.assertTrue(verify_envelope_signature(signed, secret=b"0123456789abcdef"))
        changed = replace(self.envelope, original_instruction="A changed instruction.")
        tampered = replace(signed, envelope=changed)
        self.assertFalse(verify_envelope_signature(tampered, secret=b"0123456789abcdef"))

    def test_unavailable_evidence_is_rejected_at_construction(self):
        late = replace(self.evidence, evidence_id="ev-late", available_at=self.now)
        with self.assertRaisesRegex(SafrContractError, "unavailable at cutoff"):
            replace(self.envelope, evidence=(late,))

    def test_agent_cannot_supply_the_authoritative_trace(self):
        with self.assertRaisesRegex(SafrContractError, "trusted_runtime"):
            TraceEvent(
                event_id="trace-agent",
                event_type="tool_result",
                occurred_at=self.now,
                captured_by="agent",
            )

    def test_principal_must_match_mandate(self):
        changed = replace(self.identity, principal_id="principal-2")
        with self.assertRaisesRegex(SafrContractError, "principal"):
            replace(self.envelope, identity=changed)


if __name__ == "__main__":
    unittest.main()
