#!/usr/bin/env python3
"""Reassess a recorded teacher receipt with separated semantic/evidence gates.

This performs no API calls and never rewrites the original immutable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_deepseek_safr_teacher import gate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("research-evidence/safr-deepseek-flash-teacher.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research-evidence/safr-deepseek-flash-teacher-gates-v2.json"),
    )
    args = parser.parse_args()
    source = json.loads(args.receipt.read_text(encoding="utf-8"))
    evaluations = source["evaluations"]
    gates = {
        split: gate(value["metrics"], ood=split == "ood")
        for split, value in evaluations.items()
        if split in {"test", "ood"}
    }
    result = {
        "schema": "safr-tiered-model-gate-assessment-v2",
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_receipt": str(args.receipt),
        "source_receipt_sha256": sha256(args.receipt),
        "teacher": source["teacher"],
        "gates": gates,
        "interpretation": {
            "semantic_candidate": (
                "Moderate research gate over label discrimination and false-clear behavior."
            ),
            "evidence_alignment": (
                "Characterization against non-exhaustive programmatic citation targets; "
                "kept separate from semantic eligibility."
            ),
            "execution_candidate": (
                "Unavailable until citations and behavior are independently reviewed on "
                "externally representative data."
            ),
        },
        "publish_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
