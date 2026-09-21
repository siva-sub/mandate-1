import unittest
from datetime import datetime, timezone

from aml_nextstep.safr_contracts import DispositionDecision, SafrContractError
from aml_nextstep.safr_firewall import (
    FirewallAssessment,
    FirewallFinding,
    FirewallInput,
    FirewallPolicy,
    FirewallSpan,
    apply_semantic_firewall,
    combine_firewall_floor,
)


class SafrFirewallTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)
        self.envelope_digest = "a" * 64
        self.firewall_input = FirewallInput(
            case_id="case-1",
            envelope_id="env-1",
            envelope_digest=self.envelope_digest,
            action_id="prepare_internal_note",
            spans=(
                FirewallSpan("mandate", "mandate", "Internal evidence review only."),
                FirewallSpan("instruction", "instruction", "Do not contact the customer."),
                FirewallSpan("action", "action", "Prepare an internal note.", trust="declared"),
                FirewallSpan("trace", "trace", "No outbound tool ran."),
            ),
        )

    def assessment(self, action: str, spans: tuple[str, ...], status: str = "valid") -> FirewallAssessment:
        findings = () if status != "valid" else (FirewallFinding(action, spans),)
        return FirewallAssessment(
            input_digest=self.firewall_input.digest,
            findings=findings,
            model_id="needle-test",
            model_revision="rev-1",
            inference_status=status,
        )

    def test_shadow_mode_never_changes_binding_outcome(self):
        result = combine_firewall_floor(
            base_disposition="auto_execute",
            assessment=self.assessment("request_instruction_clarification", ("instruction", "action")),
            policy=FirewallPolicy(mode="shadow"),
            now=self.now,
        )
        self.assertEqual(result.disposition, "auto_execute")
        self.assertFalse(result.model_raised_disposition)
        self.assertEqual(result.shadow_findings, ("request_instruction_clarification",))

    def test_raise_only_can_escalate_but_never_clear_a_deny(self):
        assessment = self.assessment("resolve_trace_conflict", ("action", "trace"))
        raised = combine_firewall_floor(
            base_disposition="auto_execute",
            assessment=assessment,
            policy=FirewallPolicy(mode="raise_only"),
            now=self.now,
        )
        denied = combine_firewall_floor(
            base_disposition="deny",
            assessment=self.assessment("no_semantic_exception", ("action", "trace")),
            policy=FirewallPolicy(mode="raise_only"),
            now=self.now,
        )
        self.assertEqual(raised.disposition, "escalate")
        self.assertTrue(raised.model_raised_disposition)
        self.assertEqual(denied.disposition, "deny")

    def test_unavailable_model_fails_toward_review(self):
        result = combine_firewall_floor(
            base_disposition="auto_execute",
            assessment=self.assessment("no_semantic_exception", ("action",), status="unavailable"),
            policy=FirewallPolicy(mode="raise_only"),
            now=self.now,
        )
        self.assertEqual(result.disposition, "escalate")

    def test_clean_result_cannot_coexist_with_exception(self):
        with self.assertRaisesRegex(SafrContractError, "cannot coexist"):
            FirewallAssessment(
                input_digest=self.firewall_input.digest,
                findings=(
                    FirewallFinding("no_semantic_exception", ("action",)),
                    FirewallFinding("route_scope_review", ("mandate", "action")),
                ),
                model_id="needle-test",
                model_revision="rev-1",
            )

    def test_unknown_citation_and_wrong_projection_are_rejected(self):
        unknown = FirewallAssessment(
            input_digest=self.firewall_input.digest,
            findings=(FirewallFinding("resolve_trace_conflict", ("missing-span",)),),
            model_id="needle-test",
            model_revision="rev-1",
        )
        with self.assertRaisesRegex(SafrContractError, "unknown spans"):
            unknown.validate_against(self.firewall_input)

    def test_apply_binds_base_decision_envelope_and_projection(self):
        from dataclasses import replace
        from aml_nextstep.safr_contracts import (
            AgentIdentity,
            GovernanceEnvelope,
            Mandate,
            ProposedAction,
        )
        from datetime import timedelta

        identity = AgentIdentity("agent-1", "1", "principal-1", "registry-1")
        mandate = Mandate(
            "mandate-1",
            "1",
            "principal-1",
            "Internal review.",
            ("prepare_internal_note",),
            ("internal",),
            self.now - timedelta(days=1),
            self.now + timedelta(days=1),
            3,
        )
        envelope = GovernanceEnvelope(
            "env-1",
            "case-1",
            1,
            1,
            self.now,
            self.now,
            "nonce-1",
            identity,
            mandate,
            "Review internally.",
            ProposedAction("prepare_internal_note", "1", "internal", "internal_read", {}),
            (),
            (),
            "policy-1",
            "b" * 64,
        )
        bound_input = replace(self.firewall_input, envelope_digest=envelope.digest)
        assessment = replace(self.assessment("no_semantic_exception", ("action",)), input_digest=bound_input.digest)
        decision = DispositionDecision(
            "dec-1",
            "env-1",
            "auto_execute",
            "gate-1",
            ("rule-1",),
            ("reason-1",),
            None,
            self.now,
        )
        result = apply_semantic_firewall(
            decision,
            envelope,
            assessment,
            bound_input,
            policy=FirewallPolicy(mode="raise_only"),
            now=self.now,
        )
        self.assertEqual(result.disposition, "auto_execute")


if __name__ == "__main__":
    unittest.main()
