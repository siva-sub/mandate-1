import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from aml_nextstep.safr_contracts import (
    AgentIdentity,
    EvidenceItem,
    GovernanceEnvelope,
    Mandate,
    ProposedAction,
    SemanticFindings,
    TraceEvent,
    sign_envelope,
)
from aml_nextstep.safr_gate import (
    GatePolicy,
    OperatorPolicy,
    RegisteredAgent,
    decide_disposition,
    evaluate_hard_controls,
)


class SafrGateTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)
        self.secret = b"0123456789abcdef"
        self.identity = AgentIdentity("agent-demo", "0.1.0", "principal-1", "registry-1")
        self.mandate = Mandate(
            mandate_id="mandate-1",
            version="1.0",
            principal_id="principal-1",
            purpose="Read internal evidence only.",
            allowed_action_ids=("retrieve-record", "draft-question", "file-report"),
            allowed_scopes=("internal-case",),
            valid_from=self.now - timedelta(days=1),
            valid_until=self.now + timedelta(days=1),
            max_workflow_steps=3,
        )
        summary = "Verified internal record supports a bounded lookup."
        evidence = EvidenceItem(
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
            original_instruction="Retrieve the internal record and do not contact anyone.",
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
                    event_type="tool_result",
                    occurred_at=self.now - timedelta(minutes=2),
                    captured_by="trusted_runtime",
                    tool_id="retrieve-record",
                    data_ids=("ev-1",),
                ),
            ),
            evidence=(evidence,),
            policy_version="policy-1",
            case_state_digest="a" * 64,
        )
        self.policy = GatePolicy(
            policy_version="policy-1",
            operators={
                "retrieve-record": OperatorPolicy("retrieve-record", "internal_read", False),
                "draft-question": OperatorPolicy("draft-question", "draft_external", True),
                "file-report": OperatorPolicy("file-report", "consequential", True),
            },
        )
        self.registry = {self.identity.agent_id: RegisteredAgent(self.identity)}

    def hard(self, envelope=None, *, seen=None):
        env = envelope or self.envelope
        return evaluate_hard_controls(
            sign_envelope(env, key_id="key-1", secret=self.secret),
            secret=self.secret,
            registry=self.registry,
            policy=self.policy,
            now=self.now,
            seen_nonces=seen or set(),
        )

    def semantic(
        self,
        *,
        scope_alignment: str = "within",
        trace_consistency: str = "consistent",
        evidence_sufficiency: str = "adequate",
        instruction_alignment: str = "supports",
        novelty: str = "routine",
        supporting_evidence_ids: tuple[str, ...] = ("ev-1",),
        input_digest: str | None = None,
    ) -> SemanticFindings:
        return SemanticFindings(
            scope_alignment=scope_alignment,
            trace_consistency=trace_consistency,
            evidence_sufficiency=evidence_sufficiency,
            instruction_alignment=instruction_alignment,
            novelty=novelty,
            supporting_evidence_ids=supporting_evidence_ids,
            model_id="mandate-1-test",
            model_revision="rev-1",
            input_digest=input_digest or self.envelope.digest,
        )

    def test_valid_internal_read_auto_executes(self):
        decision = decide_disposition(
            self.envelope,
            self.hard(),
            self.semantic(),
            now=self.now,
            policy=self.policy,
        )
        self.assertEqual(decision.disposition, "auto_execute")
        self.assertIsNone(decision.review_contract)

    def test_hard_failure_cannot_be_overridden_by_semantic_model(self):
        hard = self.hard(seen={"nonce-1"})
        decision = decide_disposition(
            self.envelope,
            hard,
            self.semantic(),
            now=self.now,
            policy=self.policy,
        )
        self.assertEqual(decision.disposition, "deny")
        self.assertIn("envelope_nonce_reused", decision.reason_codes)

    def test_draft_always_requires_human_review(self):
        action = ProposedAction(
            "draft-question",
            "1.0",
            "internal-case",
            "draft_external",
            {"question": "Explain the relationship."},
        )
        envelope = replace(self.envelope, envelope_id="env-draft", nonce="nonce-draft", proposed_action=action)
        semantic = self.semantic(input_digest=envelope.digest)
        decision = decide_disposition(envelope, self.hard(envelope), semantic, now=self.now, policy=self.policy)
        self.assertEqual(decision.disposition, "escalate")
        self.assertIsNotNone(decision.review_contract)

    def test_consequential_action_is_never_model_executable(self):
        action = ProposedAction("file-report", "1.0", "internal-case", "consequential", {})
        envelope = replace(self.envelope, envelope_id="env-file", nonce="nonce-file", proposed_action=action)
        semantic = self.semantic(input_digest=envelope.digest)
        decision = decide_disposition(envelope, self.hard(envelope), semantic, now=self.now, policy=self.policy)
        self.assertEqual(decision.disposition, "deny")

    def test_unknown_model_citation_escalates(self):
        semantic = self.semantic(supporting_evidence_ids=("ev-invented",))
        decision = decide_disposition(self.envelope, self.hard(), semantic, now=self.now, policy=self.policy)
        self.assertEqual(decision.disposition, "escalate")
        self.assertIn("semantic_unknown_evidence_reference", decision.reason_codes)

    def test_novel_permitted_read_is_observed(self):
        decision = decide_disposition(
            self.envelope,
            self.hard(),
            self.semantic(novelty="novel"),
            now=self.now,
            policy=self.policy,
        )
        self.assertEqual(decision.disposition, "observe")


if __name__ == "__main__":
    unittest.main()
