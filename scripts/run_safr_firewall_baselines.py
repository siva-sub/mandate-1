#!/usr/bin/env python3
"""Run frozen keyword and word-ngram baselines on the SAFR firewall release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aml_nextstep.safr_firewall_eval import (
    WordNgramNaiveBayes,
    evaluate_predictions,
    keyword_baseline,
    load_dataset_rows,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/safr-semantic-firewall-v0.1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research-evidence/safr-firewall-cheap-baselines.json"),
    )
    args = parser.parse_args()

    dataset = args.release / "dataset"
    train_path = dataset / "train.jsonl"
    train = load_dataset_rows(train_path)
    ngram = WordNgramNaiveBayes().fit(train)
    results = {}
    input_hashes = {"train": sha256(train_path)}
    for split in ("dev", "calibration", "test", "ood"):
        path = dataset / f"{split}.jsonl"
        rows = load_dataset_rows(path)
        input_hashes[split] = sha256(path)
        results[split] = {
            "keyword_rules": evaluate_predictions(rows, keyword_baseline(rows)),
            "word_unigram_bigram_naive_bayes": evaluate_predictions(rows, ngram.predict(rows)),
        }

    receipt = {
        "schema": "safr-firewall-cheap-baselines-v1",
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "release_manifest_sha256": sha256(args.release / "manifest.json"),
        "input_sha256": input_hashes,
        "baselines": {
            "keyword_rules": "Frozen regex patterns; no fitted parameters.",
            "word_unigram_bigram_naive_bayes": "Multinomial NB fitted only on dataset/train.jsonl with alpha=1.",
        },
        "results": results,
        "interpretation": (
            "These cheap baselines are intended to expose lexical shortcuts. Test and OOD are held-out "
            "contrast renderings; a useful semantic model must improve contrast-pair accuracy without an "
            "unacceptable false-clear rate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
