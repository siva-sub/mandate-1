#!/usr/bin/env python3
"""Run a bounded live DeepSeek Flash SAFR sandbox demonstration (seven API calls).

Requires --live. Sends only authored synthetic envelopes, never credentials or
real financial records. No retries, no student training, no external business tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aml_nextstep.safr_audit import HashChainAuditLog, verify_chain
from aml_nextstep.safr_contracts import SemanticFindings, sign_envelope
from aml_nextstep.safr_harness import DeepSeekJudge, SandboxHarness, demo_cases, parse_judgement
from scripts.evaluate_deepseek_safr_teacher import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicitly permit up to seven charged API calls")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("Use --live to permit the bounded synthetic API demonstration")
    if args.out.exists():
        raise SystemExit("Output exists; use a fresh directory to preserve evidence")
    load_dotenv(ROOT / ".env")
    judge = DeepSeekJudge(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                         base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    args.out.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    secret = secrets.token_bytes(32)
    audit = HashChainAuditLog(args.out / "audit.jsonl")
    runtime = SandboxHarness(secret=secret, judge=judge, audit=audit)
    cases = demo_cases(now)
    results = []
    for case in cases:
        signed = sign_envelope(case.envelope, key_id="ephemeral-demo", secret=secret)
        result = runtime.run(signed, now=now)
        results.append({"name": case.name, "expected": case.expected, "explanation": case.explanation,
                        "source": "live" if result["judge"]["called"] else "deterministic", **result})
        print(json.dumps({"case": case.name, "expected": case.expected,
                          "actual": result["decision"]["disposition"], "execution": result["execution"]["status"],
                          "judge": result["judge"].get("status", "skipped")}), flush=True)

    # Hard-fault probes reuse the real adapter, but must never call it.
    signed = sign_envelope(cases[0].envelope, key_id="ephemeral-demo", secret=secret)
    for name, candidate in (("replayed-envelope", signed), ("tampered-signature", replace(signed, signature="0"*64))):
        result = runtime.run(candidate, now=now)
        if result["judge"]["called"]:
            raise RuntimeError("hard failure reached teacher")
        results.append({"name": name, "expected": "deny", "explanation": "Hard controls stop this before inference.",
                        "source": "deterministic", **result})

    # Failure injection is offline and labelled separately from live provider behavior.
    def unavailable(envelope):
        raise TimeoutError("injected timeout")
    def malformed(envelope):
        return parse_judgement({"disposition": "auto_execute"}, envelope, model_id="injected"), {}
    def unknown_citation(envelope):
        return SemanticFindings("within", "consistent", "adequate", "supports", "routine",
                                ("invented-record",), "injected", "test", envelope.digest), {}
    for name, fake in (("model-timeout", unavailable), ("malformed-json-schema", malformed), ("unknown-citation", unknown_citation)):
        envelope = replace(cases[0].envelope, envelope_id="fault-"+name, nonce="nonce-"+name)
        fault_runtime = SandboxHarness(secret=secret, judge=fake, audit=audit)
        result = fault_runtime.run(sign_envelope(envelope, key_id="ephemeral-demo", secret=secret), now=now)
        results.append({"name": name, "expected": "escalate", "explanation": "Injected model failure must hold the read.",
                        "source": "offline_fault_injection", **result})
    valid, reason = verify_chain(audit.entries)
    if not valid:
        raise RuntimeError(f"audit chain invalid: {reason}")
    matched = sum(row["expected"] == row["decision"]["disposition"] for row in results)
    live = [row for row in results if row["source"] == "live"]
    receipt = {
        "schema": "safr-live-harness-v1", "observed_at": now.isoformat(), "model": "deepseek-flash",
        "thinking": "disabled", "max_live_requests": 7, "live_requests": len(live),
        "successful_live_responses": sum(row["judge"].get("status") == "valid" for row in live),
        "expected_disposition_matches": matched, "cases": len(results), "audit_chain_valid": valid,
        "audit_sha256": hashlib.sha256((args.out/"audit.jsonl").read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT/"aml_nextstep/safr_harness.py", ROOT/"aml_nextstep/safr_gate.py"]},
        "results": results,
        "limitations": [
            "Authored synthetic demonstration, not an independent statistical benchmark or AML detection evaluation.",
            "Provider model alias is mutable; returned API model/id and usage are recorded per request.",
            "The seven-way Laya student does not implement this richer five-primitive judge interface.",
            "Local HMAC and serial hash-linked log are demonstration mechanisms, not production key management or immutable storage.",
            "Citation IDs are validated for existence, not independently for relevance.",
            "Sandbox effects are only in-memory reads of synthetic inventory. No bank, customer, filing, or payment endpoint exists.",
        ],
    }
    (args.out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"receipt":str(args.out/"receipt.json"),"matches":matched,"cases":len(results),
                      "live_requests":len(live),"audit_chain_valid":valid}), flush=True)


if __name__ == "__main__":
    main()
