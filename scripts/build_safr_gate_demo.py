#!/usr/bin/env python3
"""Build the standalone SAFR Gate contrast demonstration from recorded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aml_nextstep.safr_firewall_eval import (
    WordNgramNaiveBayes,
    keyword_baseline,
    load_dataset_rows,
)

TEACHER_RECEIPT = Path("research-evidence/safr-deepseek-flash-teacher.json")
BASELINE_RECEIPT = Path("research-evidence/safr-firewall-cheap-baselines.json")
STUDENT_RECEIPT = Path(
    "research-evidence/safr-laya-template-overfit.json"
)
DISTILLED_RECEIPT = Path(
    "research-evidence/safr-laya-distilled-calibration-fixed/training-receipt.json"
)
AUGMENTED_NGRAM_RECEIPT = Path("research-evidence/safr-firewall-teacher-augmented-ngram.json")
RELEASE = Path("data/safr-semantic-firewall-v0.1")
TEMPLATE = Path("demo/safr-gate.template.html")
OUTPUT = Path("artifacts/safr-gate-demo.html")

SCENARIOS = (
    {
        "prefix": "fw-0302",
        "kicker": "AUTHORITY",
        "title": "One affiliate. Two different mandate states.",
        "summary": "The words are nearly identical. The controlling fact is whether the signed mandate enumerates the affiliate.",
        "question": "Is the proposed subject inside the current delegated scope?",
    },
    {
        "prefix": "fw-0303",
        "kicker": "UNTRUSTED INPUT",
        "title": "The same hostile sentence can be data—or control.",
        "summary": "A keyword rule sees the same imperative twice. The semantic check asks whether telemetry shows that it changed the action.",
        "question": "Did untrusted content influence the target or execution trace?",
    },
    {
        "prefix": "fw-0304",
        "kicker": "TRACE INTEGRITY",
        "title": "A completed lookup is not the same as the right lookup.",
        "summary": "The proposal says the lookup completed. Only the runtime subject and the superseding event reveal whether that claim is supported.",
        "question": "Does trusted telemetry support the declared action and target?",
    },
)

LABEL_COPY = {
    "no_semantic_exception": {
        "short": "No semantic hold",
        "detail": "No material semantic exception was found in this envelope.",
        "gate": "ELIGIBLE",
        "gate_detail": "May proceed only if identity, signature, mandate and action-class controls also pass.",
    },
    "route_scope_review": {
        "short": "Route scope review",
        "detail": "The proposed subject is outside or ambiguously beyond the active mandate.",
        "gate": "ESCALATE",
        "gate_detail": "A reviewer must resolve authority before the proposed read can run.",
    },
    "quarantine_untrusted_instruction": {
        "short": "Quarantine instruction",
        "detail": "Untrusted content appears to have changed the action or target.",
        "gate": "DENY",
        "gate_detail": "Block this proposal, isolate the content and create a fresh envelope.",
    },
    "resolve_trace_conflict": {
        "short": "Resolve trace conflict",
        "detail": "Trusted runtime telemetry does not support the declared action or subject.",
        "gate": "DENY",
        "gate_detail": "Do not rely on the declaration; reconcile the trace before any retry.",
    },
    "request_missing_evidence": {
        "short": "Request evidence",
        "detail": "A material premise lacks adequate current support.",
        "gate": "ESCALATE",
        "gate_detail": "Obtain or verify the evidence before relying on the premise.",
    },
    "request_instruction_clarification": {
        "short": "Clarify instruction",
        "detail": "The proposal conflicts with or is not resolved by the current instruction.",
        "gate": "ESCALATE",
        "gate_detail": "Clarify the principal's instruction before the action proceeds.",
    },
    "route_novel_context": {
        "short": "Route novel context",
        "detail": "The operator-context combination lacks a sufficiently close reviewed analogue.",
        "gate": "ESCALATE",
        "gate_detail": "A reviewer should assess this unfamiliar combination.",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_map(receipt: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    result = {}
    for batch in receipt["evaluations"][split]["batch_receipts"]:
        for prediction in batch["predictions"]:
            result[str(prediction["example_id"])] = {
                **prediction,
                "api_id": batch["api_id"],
                "api_model": batch["api_model"],
            }
    return result


def format_prediction(prediction: Any) -> dict[str, Any]:
    return {
        "label": prediction.actions[0]
        if getattr(prediction, "actions", ())
        else "no_semantic_exception",
        "citations": list(getattr(prediction, "citations", ())),
        "valid": bool(getattr(prediction, "valid", True)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ood_rows = load_dataset_rows(RELEASE / "dataset" / "ood.jsonl")
    train_rows = load_dataset_rows(RELEASE / "dataset" / "train.jsonl")
    teacher_receipt = json.loads(TEACHER_RECEIPT.read_text(encoding="utf-8"))
    baseline_receipt = json.loads(BASELINE_RECEIPT.read_text(encoding="utf-8"))
    student_receipt = json.loads(STUDENT_RECEIPT.read_text(encoding="utf-8"))
    distilled_receipt = json.loads(DISTILLED_RECEIPT.read_text(encoding="utf-8"))
    augmented_ngram = json.loads(AUGMENTED_NGRAM_RECEIPT.read_text(encoding="utf-8"))
    distilled = {
        str(value["example_id"]): value
        for value in distilled_receipt["evaluations"]["ood"]["predictions"]
    }

    teacher = prediction_map(teacher_receipt, "ood")
    keywords = keyword_baseline(ood_rows)
    ngram_model = WordNgramNaiveBayes()
    ngram_model.fit(train_rows)
    ngrams = ngram_model.predict(ood_rows)
    student = {
        str(value["example_id"]): value
        for value in student_receipt["evaluations"]["ood"]["predictions"]
    }
    rows_by_id = {str(row["example_id"]): row for row in ood_rows}

    scenarios = []
    for config in SCENARIOS:
        members = []
        for suffix in ("a", "b"):
            example_id = f"{config['prefix']}-{suffix}"
            row = rows_by_id[example_id]
            teacher_prediction = teacher[example_id]
            label = str(teacher_prediction["label"])
            members.append(
                {
                    "example_id": example_id,
                    "polarity": row["polarity"],
                    "state_label": "Exception present"
                    if row["polarity"] == "exception"
                    else "One fact resolved",
                    "input_digest": row["input"]["input_digest"],
                    "spans": row["input"]["spans"],
                    "gold": row["targets"][0]["control_action"],
                    "teacher": {
                        **teacher_prediction,
                        **LABEL_COPY[label],
                    },
                    "keyword": format_prediction(keywords[example_id]),
                    "ngram": format_prediction(ngrams[example_id]),
                    "augmented_ngram": {
                        "label": augmented_ngram["results"]["ood"]["predictions"][example_id][0],
                    },
                    "distilled": {
                        "label": distilled[example_id]["predicted_action"],
                        **LABEL_COPY[distilled[example_id]["predicted_action"]],
                        "citations": [],
                        "confidence": distilled[example_id]["confidence"],
                        "latency_seconds": distilled[example_id]["latency_seconds"],
                    },
                    "student": {
                        "label": student[example_id]["predicted_action"],
                        "confidence": student[example_id]["confidence"],
                        "latency_seconds": student[example_id]["latency_seconds"],
                    },
                }
            )
        scenarios.append({**config, "members": members})

    def model_metrics(source: dict[str, Any]) -> dict[str, float]:
        return {
            "exact": float(source["exact_set_accuracy"]),
            "pair": float(source["contrast_pair_accuracy"]),
            "false_clear": float(source["false_clear_rate"]),
            "false_hold": float(source["false_hold_rate"]),
        }

    payload = {
        "schema": "safr-gate-visible-proof-v2",
        "scenarios": scenarios,
        "metrics": {
            "keyword_rules": {
                "name": "Keyword rules",
                "test": model_metrics(
                    baseline_receipt["results"]["test"]["keyword_rules"]
                ),
                "ood": model_metrics(
                    baseline_receipt["results"]["ood"]["keyword_rules"]
                ),
            },
            "ngram": {
                "name": "Word n-gram",
                "test": model_metrics(
                    baseline_receipt["results"]["test"]["word_unigram_bigram_naive_bayes"]
                ),
                "ood": model_metrics(
                    baseline_receipt["results"]["ood"]["word_unigram_bigram_naive_bayes"]
                ),
            },
            "laya_full": {
                "name": "Laya full tune",
                "test": model_metrics(
                    student_receipt["evaluations"]["test"]["metrics"]
                ),
                "ood": model_metrics(
                    student_receipt["evaluations"]["ood"]["metrics"]
                ),
            },
            "augmented_ngram": {
                "name": "Word n-gram + teacher data",
                "test": model_metrics(augmented_ngram["results"]["test"]["metrics"]),
                "ood": model_metrics(augmented_ngram["results"]["ood"]["metrics"]),
            },
            "laya_distilled": {
                "name": "Laya + teacher data (shadow)",
                "test": model_metrics(distilled_receipt["evaluations"]["test"]["metrics"]),
                "ood": model_metrics(distilled_receipt["evaluations"]["ood"]["metrics"]),
            },
            "deepseek_teacher": {
                "name": "DeepSeek teacher",
                "test": model_metrics(
                    teacher_receipt["evaluations"]["test"]["metrics"]
                ),
                "ood": model_metrics(
                    teacher_receipt["evaluations"]["ood"]["metrics"]
                ),
            },
        },
        "evidence": {
            "release_manifest_sha256": sha256(RELEASE / "manifest.json"),
            "teacher_receipt_sha256": sha256(TEACHER_RECEIPT),
            "baseline_receipt_sha256": sha256(BASELINE_RECEIPT),
            "student_receipt_sha256": sha256(STUDENT_RECEIPT),
            "teacher_observed_at": teacher_receipt["observed_at"],
            "teacher_model": teacher_receipt["teacher"]["requested_model"],
            "teacher_thinking": "disabled",
            "student_weights_sha256": student_receipt["model"]["weights_sha256"],
            "distilled_receipt_sha256": sha256(DISTILLED_RECEIPT),
            "distilled_weights_sha256": distilled_receipt["model"]["weights_sha256"],
            "augmented_ngram_receipt_sha256": sha256(AUGMENTED_NGRAM_RECEIPT),
            "synthetic_examples": {
                "test": teacher_receipt["evaluations"]["test"]["metrics"]["examples"],
                "ood": teacher_receipt["evaluations"]["ood"]["metrics"]["examples"],
            },
        },
        "warnings": [
            "Synthetic programmatic labels; no practitioner-reviewed ground truth.",
            "The DeepSeek teacher is a mutable hosted API alias and is not the public model artifact.",
            "The teacher-augmented Laya student improves discrimination but still misses trace conflicts; it remains shadow-only and does not produce citations.",
            "This is recorded inference replay with an illustrative policy mapping, not live model inference or an execution gateway.",
            "The v0.1 test has been observed across prior experiments; a fresh independently authored challenge is still required.",
            "No score authorises an AML/CFT decision. Hard controls must remain binding in the runtime gateway.",
            "This viewer shows one exception at a time and does not establish production robustness.",
        ],
    }
    template = args.template.read_text(encoding="utf-8")
    marker = "__SAFR_DEMO_DATA__"
    if template.count(marker) != 1:
        raise ValueError(f"template must contain exactly one {marker} marker")
    rendered = template.replace(
        marker,
        json.dumps(payload, sort_keys=True, ensure_ascii=False).replace("</", "<\\/"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
                "scenarios": len(scenarios),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
