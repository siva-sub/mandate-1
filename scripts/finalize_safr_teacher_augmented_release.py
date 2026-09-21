#!/usr/bin/env python3
"""Finalize a train-only teacher-augmented SAFR release after self-consistency checks.

Generated contrast pairs are retained only when the independent classification
request assigns the inherited label to both members. This is still weak,
non-practitioner-reviewed supervision and is never treated as test truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aml_nextstep.safr_gliner import write_gliner_dataset

BASE_SPLITS = ("dev", "calibration", "test", "ood")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prediction_index(receipt: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    evaluation = receipt.get("evaluations", {}).get(split)
    if not isinstance(evaluation, dict):
        raise ValueError(f"verification receipt has no evaluation for {split!r}")
    predictions: dict[str, dict[str, Any]] = {}
    for batch in evaluation.get("batch_receipts", []):
        for prediction in batch.get("predictions", []):
            example_id = str(prediction["example_id"])
            if example_id in predictions:
                raise ValueError(f"duplicate verification prediction for {example_id}")
            predictions[example_id] = {
                **prediction,
                "api_id": batch.get("api_id"),
                "api_model": batch.get("api_model"),
                "prompt_sha256": batch.get("prompt_sha256"),
            }
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("data/safr-semantic-firewall-v0.1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/safr-semantic-firewall-v0.1-teacher-augmented"),
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path(
            "data/safr-semantic-firewall-v0.1-teacher-augmented/teacher-self-consistency.json"
        ),
    )
    parser.add_argument("--verification-split", default="train-teacher-raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.output / "dataset" / "train-teacher-raw.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"missing generated data: {raw_path}")
    if not args.verification.exists():
        raise SystemExit(f"missing verification receipt: {args.verification}")

    base_rows = read_rows(args.base / "dataset" / "train.jsonl")
    generated_rows = read_rows(raw_path)
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    predictions = prediction_index(verification, args.verification_split)
    expected_ids = {str(row["example_id"]) for row in generated_rows}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))[:10]
        extra = sorted(set(predictions) - expected_ids)[:10]
        raise ValueError(f"verification coverage mismatch; missing={missing}, extra={extra}")

    parent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        parent_group_id = str(row["augmentation"]["parent_contrast_group_id"])
        parent_groups[parent_group_id].append(row)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for parent_group_id in sorted(parent_groups):
        members = parent_groups[parent_group_id]
        reasons = []
        variant_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            variant_groups[str(row["contrast_group_id"])].append(row)
        if len(members) != 4 or len(variant_groups) != 2:
            reasons.append("expected_two_complete_contrast_variants")
        for variant_group_id, variant_members in sorted(variant_groups.items()):
            if len(variant_members) != 2 or {
                row["polarity"] for row in variant_members
            } != {"exception", "hard_negative"}:
                reasons.append(f"{variant_group_id}:invalid_pair_shape")
        for row in members:
            example_id = str(row["example_id"])
            expected_label = str(row["targets"][0]["control_action"])
            predicted = predictions[example_id]
            if predicted.get("label") != expected_label:
                reasons.append(
                    f"{example_id}:expected={expected_label}:predicted={predicted.get('label')}"
                )
        if reasons:
            rejected.append(
                {"parent_contrast_group_id": parent_group_id, "reasons": reasons}
            )
            continue
        for row in members:
            example_id = str(row["example_id"])
            predicted = predictions[example_id]
            row["label_status"] = "teacher_generated_self_consistent_unreviewed"
            row["augmentation"]["self_consistency"] = {
                "requested_model": verification["teacher"]["requested_model"],
                "api_model": predicted.get("api_model"),
                "api_id": predicted.get("api_id"),
                "prompt_sha256": predicted.get("prompt_sha256"),
                "predicted_label": predicted.get("label"),
                "citations": predicted.get("citations", []),
            }
            accepted.append(row)

    accepted.sort(key=lambda row: (str(row["contrast_group_id"]), str(row["polarity"])))
    train_augmentations = [
        row for row in accepted if str(row["example_id"]).endswith("-t01")
    ]
    paraphrase_validation = [
        row for row in accepted if str(row["example_id"]).endswith("-t02")
    ]
    if len(train_augmentations) != len(paraphrase_validation):
        raise ValueError("accepted teacher variants are not balanced between train and validation")
    combined = sorted(
        [*base_rows, *train_augmentations],
        key=lambda row: (str(row["contrast_group_id"]), str(row["polarity"])),
    )
    dataset_dir = args.output / "dataset"
    train_path = dataset_dir / "train.jsonl"
    accepted_path = dataset_dir / "train-teacher-accepted.jsonl"
    validation_path = dataset_dir / "paraphrase_validation.jsonl"
    write_rows(train_path, combined)
    write_rows(accepted_path, accepted)
    write_rows(validation_path, paraphrase_validation)
    for split in BASE_SPLITS:
        source = args.base / "dataset" / f"{split}.jsonl"
        target = dataset_dir / f"{split}.jsonl"
        shutil.copyfile(source, target)

    gliner_manifest = write_gliner_dataset(
        args.output,
        splits=("train", "dev", "calibration", "paraphrase_validation", "test", "ood"),
    )
    files = {}
    for path in sorted(
        [
            train_path,
            accepted_path,
            validation_path,
            raw_path,
            args.verification,
            *(dataset_dir / f"{split}.jsonl" for split in BASE_SPLITS),
            *(
                args.output / "gliner25" / f"{split}.jsonl"
                for split in ("train", "paraphrase_validation", *BASE_SPLITS)
            ),
            args.output / "gliner25" / "manifest.json",
            args.output / "generation-manifest.json",
        ]
    ):
        if path.exists():
            files[str(path.relative_to(args.output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    accepted_counts = Counter(
        str(row["targets"][0]["control_action"])
        for row in train_augmentations
        if row["polarity"] == "exception"
    )
    manifest = {
        "schema": "safr-semantic-firewall-teacher-augmented-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_release": str(args.base),
        "base_train_sha256": sha256(args.base / "dataset" / "train.jsonl"),
        "augmentation_method": "DeepSeek Flash train-only pair paraphrase followed by a separate label self-consistency request",
        "teacher_alias_mutable": True,
        "source_train_examples": len(base_rows),
        "generated_examples": len(generated_rows),
        "accepted_generated_examples": len(accepted),
        "rejected_generated_examples": len(generated_rows) - len(accepted),
        "accepted_parent_worlds": len(accepted) // 4,
        "rejected_parent_worlds": rejected,
        "train_generated_examples": len(train_augmentations),
        "paraphrase_validation_examples": len(paraphrase_validation),
        "accepted_train_exception_counts": dict(sorted(accepted_counts.items())),
        "combined_train_examples": len(combined),
        "split_policy": {
            "teacher_variant_t01": "gradient updates",
            "teacher_variant_t02": "paraphrase robustness validation only",
            "base_dev": "template-matched development characterization",
            "base_calibration": "post-selection temperature fitting only",
            "base_test": "frozen final evaluation only",
            "base_ood": "frozen near-OOD final evaluation only",
        },
        "gliner_examples": gliner_manifest["examples"],
        "test_and_ood_copied_unchanged": True,
        "files": files,
        "label_status": "teacher_generated_self_consistent_unreviewed",
        "warnings": [
            "Self-consistency by the generating model is not independent validation.",
            "No generated label or text is practitioner reviewed.",
            "The teacher model identifier is a mutable hosted API alias.",
            "The paraphrase validation split shares latent source worlds with base train; it measures wording robustness, not world-level generalization.",
            "Test and OOD are synthetic programmatic labels and were not used for generation, checkpoint selection, or training.",
            "This release is for distillation research, not autonomous AML/CFT decisions.",
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card = f"""# SAFR Semantic Firewall — teacher-augmented training release

This research-only derivative combines the original {len(base_rows)} train rows with
{len(train_augmentations)} accepted DeepSeek Flash training paraphrases. Another
{len(paraphrase_validation)} accepted paraphrases are reserved for wording-robustness
validation. The generator saw **train only**. Dev, calibration, test, and OOD were
copied byte-for-byte from the frozen v0.1 release.

A generated contrast pair is accepted only when a separate classification request
returns both inherited labels. This is a self-consistency filter, **not independent
or practitioner validation**. The hosted model name is a mutable API alias.

- Accepted parent worlds (both pair variants passed): {len(accepted) // 4}
- Rejected parent worlds: {len(rejected)}
- Teacher-generated train rows: {len(train_augmentations)}
- Paraphrase-validation rows: {len(paraphrase_validation)}
- Combined source train rows: {len(combined)}
- Laya train sequences after positive balancing: {gliner_manifest['examples']['train']}
- Operational or autonomous use: prohibited

See `manifest.json`, `generation-manifest.json`, and
`teacher-self-consistency.json` for hashes, provenance, failures, and API receipts.
"""
    (args.output / "README.md").write_text(card, encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
