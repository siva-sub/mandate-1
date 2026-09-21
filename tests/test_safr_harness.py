import unittest
from dataclasses import replace
from datetime import datetime, timezone

from aml_nextstep.safr_audit import HashChainAuditLog
from aml_nextstep.safr_contracts import SemanticFindings, sign_envelope
from aml_nextstep.safr_harness import SandboxHarness, demo_cases, parse_judgement


class SafrHarnessTest(unittest.TestCase):
    def test_permitted_read_is_audited_before_execution(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        case = demo_cases(now)[0]
        audit = HashChainAuditLog()
        secret = b"synthetic-test-signing-key"

        def judge(envelope):
            return SemanticFindings(
                "within", "consistent", "adequate", "supports", "routine",
                ("record-1",), "fake", "test", envelope.digest,
            ), {"source": "test double"}

        harness = SandboxHarness(secret=secret, judge=judge, audit=audit)
        result = harness.run(sign_envelope(case.envelope, key_id="demo", secret=secret), now=now)
        self.assertEqual(result["decision"]["disposition"], "auto_execute")
        self.assertEqual(result["execution"]["status"], "executed_synthetic_read")
        self.assertEqual([entry.event_type for entry in audit.entries], ["decision", "execution"])

    def test_hard_failures_never_call_the_judge(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        secret = b"synthetic-test-signing-key"
        called = []
        def judge(envelope):
            called.append(envelope.envelope_id)
            raise AssertionError("must not call model")
        cases = demo_cases(now)
        signed = [sign_envelope(case.envelope, key_id="demo", secret=secret) for case in cases[-2:]]
        signed.append(replace(sign_envelope(cases[0].envelope, key_id="demo", secret=secret), signature="0"*64))
        for item in signed:
            with self.subTest(case=item.envelope.envelope_id):
                result = SandboxHarness(secret=secret, judge=judge, audit=HashChainAuditLog()).run(item, now=now)
                self.assertEqual(result["decision"]["disposition"], "deny")
                self.assertEqual(result["execution"]["status"], "not_executed")
        self.assertEqual(called, [])

    def test_model_failure_holds_instead_of_executing(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        secret = b"synthetic-test-signing-key"
        for error in (TimeoutError, ValueError):
            def judge(envelope):
                raise error("must not appear in audit")
            log = HashChainAuditLog()
            result = SandboxHarness(secret=secret, judge=judge, audit=log).run(
                sign_envelope(demo_cases(now)[0].envelope, key_id="demo", secret=secret), now=now)
            self.assertEqual(result["decision"]["disposition"], "escalate")
            self.assertEqual(result["execution"]["status"], "not_executed")
            self.assertNotIn("must not appear", str(result))

    def test_unknown_citations_or_stale_findings_cannot_release_a_read(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        secret = b"synthetic-test-signing-key"
        for refs, digest in ((('invented',), None), (('record-1',), 'a'*64), ((), None)):
            def judge(envelope):
                return SemanticFindings("within", "consistent", "adequate", "supports", "routine", refs,
                                        "fake", "test", digest or envelope.digest), {}
            result = SandboxHarness(secret=secret, judge=judge, audit=HashChainAuditLog()).run(
                sign_envelope(demo_cases(now)[0].envelope, key_id="demo", secret=secret), now=now)
            self.assertEqual(result["decision"]["disposition"], "escalate")
            self.assertEqual(result["execution"]["status"], "not_executed")

    def test_replay_remains_denied_when_harness_is_recreated(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        secret = b"synthetic-test-signing-key"
        audit = HashChainAuditLog()
        calls = []
        def judge(envelope):
            calls.append(envelope.envelope_id)
            return SemanticFindings("within", "consistent", "adequate", "supports", "routine",
                                    ("record-1",), "fake", "test", envelope.digest), {}
        signed = sign_envelope(demo_cases(now)[0].envelope, key_id="demo", secret=secret)
        SandboxHarness(secret=secret, judge=judge, audit=audit).run(signed, now=now)
        result = SandboxHarness(secret=secret, judge=judge, audit=audit).run(signed, now=now)
        self.assertEqual(result["decision"]["disposition"], "deny")
        self.assertEqual(len(calls), 1)

    def test_audit_failure_stops_before_execution(self):
        class BrokenAudit(HashChainAuditLog):
            def append(self, **kwargs):
                raise OSError("disk full")
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        secret = b"synthetic-test-signing-key"
        def judge(envelope):
            return SemanticFindings("within", "consistent", "adequate", "supports", "routine",
                                    ("record-1",), "fake", "test", envelope.digest), {}
        log = BrokenAudit()
        with self.assertRaises(OSError):
            SandboxHarness(secret=secret, judge=judge, audit=log).run(
                sign_envelope(demo_cases(now)[0].envelope, key_id="demo", secret=secret), now=now)
        self.assertEqual(log.entries, ())

    def test_json_parser_enforces_schema_and_citations(self):
        now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        envelope = demo_cases(now)[0].envelope
        value = dict(scope_alignment="within", trace_consistency="consistent", evidence_sufficiency="adequate",
                     instruction_alignment="supports", novelty="routine", supporting_evidence_ids=["record-1"], abstained=False)
        self.assertEqual(parse_judgement(value, envelope, model_id="fake").input_digest, envelope.digest)
        for patch in ({"disposition": "auto_execute"}, {"scope_alignment": "sure"},
                      {"supporting_evidence_ids": ["invented"]}, {"supporting_evidence_ids": []}, {"abstained": "false"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                parse_judgement({**value, **patch}, envelope, model_id="fake")

    def test_demonstration_covers_all_four_dispositions(self):
        cases = demo_cases(datetime(2026, 9, 22, tzinfo=timezone.utc))
        self.assertEqual({case.expected for case in cases}, {"auto_execute", "observe", "escalate", "deny"})
