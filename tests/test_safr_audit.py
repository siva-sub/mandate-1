import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from aml_nextstep.safr_audit import HashChainAuditLog, read_entries, verify_chain


class SafrAuditTest(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        now = datetime(2026, 9, 21, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            log = HashChainAuditLog(path)
            first = log.append(
                event_type="governance_envelope_received",
                subject_id="env-1",
                payload={"digest": "a" * 64},
                recorded_at=now,
            )
            second = log.append(
                event_type="disposition_recorded",
                subject_id="dec-1",
                payload={"disposition": "auto_execute"},
                recorded_at=now,
            )
            self.assertEqual(second.previous_hash, first.entry_hash)
            entries = list(read_entries(path))
            self.assertEqual(verify_chain(entries), (True, None))

            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("auto_execute", "observe"), encoding="utf-8")
            tampered = list(read_entries(path))
            ok, reason = verify_chain(tampered)
            self.assertFalse(ok)
            self.assertIn("content hash mismatch", reason or "")


if __name__ == "__main__":
    unittest.main()
