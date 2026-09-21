#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""Teacher-diversified Laya distillation with paraphrase checkpoint selection."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_PATH = globals().get("__file__")
if os.environ.get("SAFR_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["SAFR_PROJECT_ROOT"]).resolve()
elif _SCRIPT_PATH:
    PROJECT_ROOT = Path(str(_SCRIPT_PATH)).resolve().parents[1]
elif Path("/content/safr-job").is_dir():
    PROJECT_ROOT = Path("/content/safr-job")
else:
    PROJECT_ROOT = Path.cwd()
RELEASE = PROJECT_ROOT / "data" / "safr-semantic-firewall-v0.1-teacher-augmented"
WORK_ROOT = Path(
    os.environ.get(
        "SAFR_WORK_ROOT",
        "/kaggle/working" if Path("/kaggle/working").is_dir() else "/content",
    )
)
OUTPUT_ROOT = WORK_ROOT / "safr-laya-distilled"
ITEMS_PATH = WORK_ROOT / "safr-laya-distill-train-items.json"
VALIDATION_ITEMS_PATH = WORK_ROOT / "safr-laya-distill-validation-items.json"
MODEL_REPO = "convaiinnovations/laya-typed-decisions"
MODEL_REVISION = "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2"
LAYA_REVISION = "42626c348753fbb17572a813127df2278a1ec527"
LAYA_WHEEL_SHA256 = "1883529eafb06168a604d8f218703db7f205f2db209b8b68b3cba4a9334b1818"
SEED = 42_017


def run(command: list[str], **kwargs: Any) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    count = len(items)
    sequence_length = max(len(item["ids"]) for item in items)
    option_count = max(len(item["markers"]) for item in items)
    input_ids = torch.full((count, sequence_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((count, sequence_length), dtype=torch.long)
    marker_pos = torch.zeros((count, option_count), dtype=torch.long)
    marker_mask = torch.zeros((count, option_count), dtype=torch.bool)
    for index, item in enumerate(items):
        input_ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        attention_mask[index, : len(item["ids"])] = 1
        marker_pos[index, : len(item["markers"])] = torch.tensor(item["markers"])
        marker_mask[index, : len(item["markers"])] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.tensor([item["qtype"] for item in items]),
        "label": torch.tensor([item["label"] for item in items]),
    }


def build_items(
    path: Path,
    tokenizer: Any,
    config: dict[str, Any],
    question: dict[str, Any],
    labels: tuple[str, ...],
    *,
    shuffle_options: bool,
    seed: int,
    label_smoothing: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from laya.common import QTYPES, build_sequence

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    rng = random.Random(seed)
    items = []
    lengths = []
    counts = {label: 0 for label in labels}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        classification = record["output"]["classifications"][0]
        gold = str(classification["true_label"][0])
        canonical_index = labels.index(gold)
        order = list(range(len(labels)))
        if shuffle_options:
            rng.shuffle(order)
        sequence, markers = build_sequence(
            tokenizer,
            record["input"],
            {
                "t": "choice",
                "ins": question["instructions"],
                "crit": question["criteria"],
            },
            config["max_len"],
            config["head_max_len"],
            option_order=order,
        )
        if len(markers) != len(labels):
            raise ValueError("semantic-finding options exceeded Laya's head token budget")
        label = order.index(canonical_index)
        off_target = label_smoothing / (len(labels) - 1)
        target = [off_target] * len(labels)
        target[label] = 1.0 - label_smoothing
        items.append(
            {
                "ids": sequence,
                "markers": markers,
                "qtype": QTYPES["choice"],
                "target": target,
                "label": label,
                "option_order": order,
            }
        )
        lengths.append(len(sequence))
        counts[gold] += 1
    return items, {
        "examples": len(items),
        "tokens_min": min(lengths),
        "tokens_max": max(lengths),
        "tokens_mean": sum(lengths) / len(lengths),
        "label_counts": counts,
        "label_smoothing": label_smoothing,
    }


def fit_temperature(
    model: Any,
    items: list[dict[str, Any]],
    pad_id: int,
    device: Any,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    logits_parts = []
    label_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), 8):
            batch = collate(items[start : start + 8], pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits_parts.append(logits.float().cpu())
            label_parts.append(batch["label"])
    logits_all = torch.cat(logits_parts)
    labels_all = torch.cat(label_parts)
    uncalibrated_nll = float(functional.cross_entropy(logits_all, labels_all).item())
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> Any:
        optimizer.zero_grad()
        loss = functional.cross_entropy(logits_all / log_temperature.exp(), labels_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp(0.5, 5.0).item())
    calibrated_nll = float(
        functional.cross_entropy(logits_all / temperature, labels_all).item()
    )
    return {
        "temperature": temperature,
        "uncalibrated_nll": uncalibrated_nll,
        "calibrated_nll": calibrated_nll,
        "examples": len(items),
    }


def apply_choice_temperature(config: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Replace inherited choice buckets so Laya actually uses our fitted scalar.

    Laya Agent prefers temperature_by_options over the per-type temperature.
    Retaining a base-model choice bucket silently bypasses domain calibration.
    No other question type is calibrated by this training job.
    """
    if not math.isfinite(temperature) or not 0.5 <= temperature <= 5.0:
        raise ValueError("choice temperature must be finite and within [0.5, 5.0]")
    temperatures = list(config.get("temperature", [1.0, 1.0, 1.0]))
    if len(temperatures) != 3:
        raise ValueError("expected three question-type temperatures")
    temperatures[0] = temperature
    buckets = {
        key: value
        for key, value in config.get("temperature_by_options", {}).items()
        if not key.startswith("choice:")
    }
    return {**config, "temperature": temperatures, "temperature_by_options": buckets}


def ece(confidences: list[float], correct: list[float], bins: int = 15) -> float:
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, confidence in enumerate(confidences)
            if confidence > lower and confidence <= upper
        ]
        if selected:
            mean_confidence = sum(confidences[position] for position in selected) / len(selected)
            mean_correct = sum(correct[position] for position in selected) / len(selected)
            result += len(selected) / len(confidences) * abs(mean_confidence - mean_correct)
    return result


def evaluate(agent: Any, split: str, question: dict[str, Any]) -> dict[str, Any]:
    from aml_nextstep.safr_firewall_eval import (
        FirewallPrediction,
        evaluate_predictions,
        load_dataset_rows,
    )
    from aml_nextstep.safr_gliner import (
        FINDING_TASK,
        NO_FINDING,
        SEMANTIC_PRIMITIVES,
        render_envelope,
        semantic_finding_label,
    )

    from aml_nextstep.safr_firewall import NO_EXCEPTION

    label_to_action = {
        NO_FINDING: NO_EXCEPTION,
        **{
            item.exception_label: item.control_action
            for item in SEMANTIC_PRIMITIVES
        },
    }

    rows = load_dataset_rows(RELEASE / "dataset" / f"{split}.jsonl")
    predictions = {}
    receipts = []
    confidences = []
    correct = []
    brier = []
    negative_log_likelihood = []
    started = time.monotonic()
    for row in rows:
        inference_started = time.monotonic()
        result = agent.predict(render_envelope(row), {FINDING_TASK: question})
        latency = time.monotonic() - inference_started
        answer = result["answers"][FINDING_TASK]
        label = str(answer["choice"])
        probabilities = {
            str(key): float(value) for key, value in answer["probabilities"].items()
        }
        gold = semantic_finding_label(row)
        action = label_to_action[label]
        prediction = FirewallPrediction(
            example_id=str(row["example_id"]),
            actions=(action,),
            citations={},
            status="valid",
            latency_seconds=latency,
        )
        predictions[prediction.example_id] = prediction
        is_correct = float(label == gold)
        confidence = max(probabilities.values())
        confidences.append(confidence)
        correct.append(is_correct)
        brier.append(
            sum(
                (probability - float(candidate == gold)) ** 2
                for candidate, probability in probabilities.items()
            )
        )
        negative_log_likelihood.append(
            -math.log(max(probabilities.get(gold, 0.0), 1e-12))
        )
        receipts.append(
            {
                "example_id": prediction.example_id,
                "gold_finding": gold,
                "predicted_finding": label,
                "predicted_action": action,
                "confidence": confidence,
                "probabilities": probabilities,
                "input_tokens": int(result["usage"]["input_tokens"]),
                "latency_seconds": latency,
            }
        )
    elapsed = time.monotonic() - started
    return {
        "metrics": evaluate_predictions(rows, predictions),
        "probability_metrics": {
            "brier_multiclass": sum(brier) / len(brier),
            "negative_log_likelihood": sum(negative_log_likelihood)
            / len(negative_log_likelihood),
            "ece_15_bin": ece(confidences, correct),
            "mean_confidence": sum(confidences) / len(confidences),
        },
        "runtime": {
            "examples": len(rows),
            "elapsed_seconds": elapsed,
            "throughput_examples_per_second": len(rows) / elapsed,
        },
        "predictions": receipts,
    }


def order_robustness(
    agent: Any,
    split: str,
    question: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    from aml_nextstep.safr_firewall_eval import load_dataset_rows
    from aml_nextstep.safr_gliner import FINDING_TASK, render_envelope, semantic_finding_label

    rows = load_dataset_rows(RELEASE / "dataset" / f"{split}.jsonl")
    canonical_by_id = {
        item["example_id"]: item for item in canonical["predictions"]
    }
    reversed_question = {
        **question,
        "criteria": dict(reversed(list(question["criteria"].items()))),
    }
    consistent = 0
    correct = 0
    comparisons = []
    for row in rows:
        example_id = str(row["example_id"])
        result = agent.predict(render_envelope(row), {FINDING_TASK: reversed_question})
        reversed_label = str(result["answers"][FINDING_TASK]["choice"])
        canonical_label = canonical_by_id[example_id]["predicted_finding"]
        consistent += int(reversed_label == canonical_label)
        correct += int(reversed_label == semantic_finding_label(row))
        comparisons.append(
            {
                "example_id": example_id,
                "canonical": canonical_label,
                "reversed": reversed_label,
                "same": reversed_label == canonical_label,
            }
        )
    return {
        "examples": len(rows),
        "consistency": consistent / len(rows),
        "reversed_order_accuracy": correct / len(rows),
        "comparisons": comparisons,
    }


def gate(
    metrics: dict[str, Any],
    robustness: dict[str, Any],
    *,
    ood: bool = False,
) -> dict[str, Any]:
    limits = {
        "valid_output_rate": 1.0,
        "exact_set_accuracy": 0.75 if ood else 0.85,
        "contrast_pair_accuracy": 0.60 if ood else 0.75,
        "macro_f1": 0.70 if ood else 0.80,
        "false_clear_rate_max": 0.15 if ood else 0.10,
        "order_consistency": 0.90,
    }
    checks = {
        "valid_output_rate": metrics["valid_output_rate"] >= limits["valid_output_rate"],
        "exact_set_accuracy": metrics["exact_set_accuracy"] >= limits["exact_set_accuracy"],
        "contrast_pair_accuracy": metrics["contrast_pair_accuracy"] >= limits["contrast_pair_accuracy"],
        "macro_f1": metrics["macro_f1"] >= limits["macro_f1"],
        "false_clear_rate": metrics["false_clear_rate"] <= limits["false_clear_rate_max"],
        "order_consistency": robustness["consistency"] >= limits["order_consistency"],
    }
    return {"passed": all(checks.values()), "thresholds": limits, "checks": checks}


def write_model_card(receipt: dict[str, Any]) -> None:
    evaluations = receipt["evaluations"]
    gates = receipt["release_gates"]
    training = receipt["training"]
    model = receipt["model"]
    lines = [
        "---",
        "license: other",
        "library_name: transformers",
        f"base_model: {MODEL_REPO}",
        "tags:",
        "  - laya",
        "  - typed-decisions",
        "  - agentic-finance",
        "  - runtime-governance",
        "  - semantic-routing",
        "  - research",
        "---",
        "",
        "# SAFR Laya Semantic Firewall — teacher-distilled v0.2",
        "",
        "> **Private research checkpoint — not approved for publication or production use.**",
        "",
        "This checkpoint adapts Laya to emit one advisory semantic finding from a synthetic, authenticated Governance-Envelope-like record. It does not issue SAFR's binding disposition, authorize execution, detect money laundering, assess a customer, or make a regulatory decision.",
        "",
        "## Intended architecture role",
        "",
        "The checkpoint is a probabilistic sensor for AI-specific semantic controls. A deterministic runtime gateway must still verify identity and envelope integrity, enforce capability/mandate syntax and categorical limits, map findings to `Deny` / `Escalate` / `Auto-Execute` / `Observe`, gate execution, and append the audit record. A nominal model prediction never overrides a deterministic failure.",
        "",
        "## Base model and training",
        "",
        f"- Base: `{MODEL_REPO}` at revision `{MODEL_REVISION}`",
        f"- Laya code revision: `{LAYA_REVISION}`",
        f"- Parameters: {model['parameters']:,}; all parameters adapted",
        f"- Hardware: {training['world_size']} × {training['gpu']}",
        f"- Objective: {training['objective']}",
        f"- Epochs: best {training['best_epoch']} of {training['epochs_completed']} completed / {training['epochs_requested']} maximum; effective batch: {training['effective_batch']}",
        f"- Encoder LR: {training['encoder_learning_rate']}; head LR: {training['head_learning_rate']}",
        f"- Training time: {training['elapsed_seconds']:.1f} seconds",
        f"- Model SHA-256: `{model['weights_sha256']}`",
        "",
        f"Training uses {receipt['data']['train']['examples']:,} balanced typed-decision sequences from original train rows plus teacher-generated `t01` paraphrases. Complete option menus are randomly permuted and targets use 0.05 smoothing. The paired `t02` paraphrases ({receipt['data']['paraphrase_validation']['examples']:,} rows) select the checkpoint but receive no gradient. Calibration is separate. Test/OOD labels are excluded from generation, optimization, checkpoint selection, and temperature fitting.",
        "",
        "## Evaluation",
        "",
        "| Split | Exact accuracy | Pair accuracy | Macro-F1 | False clear | p50 latency | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("dev", "paraphrase_validation", "calibration", "test", "ood"):
        metrics = evaluations[split]["metrics"]
        gate_text = "n/a" if split not in gates else ("pass" if gates[split]["passed"] else "fail")
        lines.append(
            "| "
            + split
            + f" | {metrics['exact_set_accuracy']:.3f}"
            + f" | {metrics['contrast_pair_accuracy']:.3f}"
            + f" | {metrics['macro_f1']:.3f}"
            + f" | {metrics['false_clear_rate']:.3f}"
            + f" | {metrics['latency_seconds']['p50'] * 1000:.1f} ms"
            + f" | {gate_text} |"
        )
    lines.extend(
        [
            "",
            "Metrics are on programmatically labelled synthetic contrasts. They are engineering evidence, not AML/CFT accuracy. v0.1 test has been observed during earlier architecture experiments and is not a pristine permanent holdout; an independently authored v0.2 challenge set is required before any external claim.",
            "",
            "## Decision labels",
            "",
            "The single `choice` output is one of: `no material semantic exception`, `insufficient evidence`, `outside mandate`, `influencing action`, `trace contradictory`, `instruction conflicting`, or `no reviewed analogue`. Deterministic code maps that semantic fact to a raise-only advisory action.",
            "",
            "## Limitations",
            "",
            "- Synthetic programmatic labels; no practitioner-reviewed ground truth.",
            "- One primary exception per envelope; multi-finding composition is unsupported.",
            "- No trained span-citation output; citation quality remains a release blocker.",
            "- English only and a narrow set of fictional internal-note workflows.",
            "- Declared trust metadata is assumed; the model does not authenticate provenance.",
            "- Generator/template learning can inflate held-out scores.",
            "- Calibration can drift under new institutions, policies, tools, or action types.",
            "- Never use the model output to expand authority or bypass deterministic controls.",
            "",
            "## Release status",
            "",
            "`publish_allowed: false`. Public licensing, independent review, citation support, multi-exception evaluation, red-team testing, and a fresh final holdout are unresolved. No upload to Hugging Face is performed by the training job.",
            "",
            "## Reproducibility",
            "",
            "See `training-receipt.json`, `ddp-training-state.json`, the dataset `manifest.json`, and `gliner25/manifest.json` for pinned revisions, hashes, split provenance, environment, optimization history, calibration, per-example predictions, robustness checks, and gate decisions.",
            "",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.environ["USE_TF"] = "0"
    sys.path.insert(0, str(PROJECT_ROOT))
    vendor_wheels = sorted((PROJECT_ROOT / "vendor").glob("laya-*.whl"))
    if len(vendor_wheels) != 1:
        raise SystemExit(f"expected one vendored Laya wheel, found {vendor_wheels}")
    laya_wheel = vendor_wheels[0]
    if sha256(laya_wheel) != LAYA_WHEEL_SHA256:
        raise SystemExit("vendored Laya wheel failed its pinned SHA-256 check")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            str(laya_wheel),
            "huggingface_hub<2",
            "safetensors",
        ]
    )
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    laya = importlib.import_module("laya")
    laya_common = importlib.import_module("laya.common")
    from aml_nextstep.safr_firewall_eval import load_dataset_rows
    from aml_nextstep.safr_gliner import (
        FINDING_DESCRIPTIONS,
        FINDING_INSTRUCTION,
        FINDING_LABELS,
    )

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise SystemExit("CUDA is required for teacher-diversified adaptation")
    world_size = min(gpu_count, 2)
    gradient_accumulation = 4 if world_size == 2 else 8
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RELEASE / "gliner25" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, details in manifest["files"].items():
        path = RELEASE / relative
        if not path.is_file() or sha256(path) != details["sha256"]:
            raise SystemExit(f"data integrity check failed for {path}")

    model_dir = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION))
    base_config = json.loads(
        (model_dir / "rl_agent_config.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    question = {
        "type": "choice",
        "instructions": FINDING_INSTRUCTION,
        "criteria": FINDING_DESCRIPTIONS,
    }
    labels = tuple(FINDING_LABELS)
    train_items, train_stats = build_items(
        RELEASE / "gliner25" / "train.jsonl",
        tokenizer,
        base_config,
        question,
        labels,
        shuffle_options=True,
        seed=SEED,
        label_smoothing=0.05,
    )
    validation_items, validation_stats = build_items(
        RELEASE / "gliner25" / "paraphrase_validation.jsonl",
        tokenizer,
        base_config,
        question,
        labels,
        shuffle_options=False,
        seed=SEED,
    )
    validation_rows = load_dataset_rows(
        RELEASE / "dataset" / "paraphrase_validation.jsonl"
    )
    if len(validation_items) != len(validation_rows):
        raise SystemExit("paraphrase validation item alignment failed")
    for item, row in zip(validation_items, validation_rows, strict=True):
        item["example_id"] = row["example_id"]
        item["contrast_group_id"] = row["contrast_group_id"]
        item["polarity"] = row["polarity"]
    calibration_items, calibration_stats = build_items(
        RELEASE / "gliner25" / "calibration.jsonl",
        tokenizer,
        base_config,
        question,
        labels,
        shuffle_options=False,
        seed=SEED,
    )
    ITEMS_PATH.write_text(
        json.dumps(train_items, separators=(",", ":")), encoding="utf-8"
    )
    VALIDATION_ITEMS_PATH.write_text(
        json.dumps(validation_items, separators=(",", ":")), encoding="utf-8"
    )

    worker = PROJECT_ROOT / "scripts" / "kaggle_laya_safr_distill_worker.py"
    run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={world_size}",
            str(worker),
            "--model-dir",
            str(model_dir),
            "--items",
            str(ITEMS_PATH),
            "--validation-items",
            str(VALIDATION_ITEMS_PATH),
            "--output-dir",
            str(OUTPUT_ROOT),
            "--epochs",
            "6",
            "--micro-batch",
            "8",
            "--gradient-accumulation",
            str(gradient_accumulation),
        ]
    )

    device = torch.device("cuda:0")
    config_path = OUTPUT_ROOT / "rl_agent_config.json"
    tuned_config = json.loads(config_path.read_text(encoding="utf-8"))
    calibrated_model = laya_common.build_model(
        tuned_config, encoder_dir=str(OUTPUT_ROOT / "encoder")
    )
    calibrated_model.load_state_dict(
        load_file(str(OUTPUT_ROOT / "model.safetensors")), strict=True
    )
    calibrated_model.to(device)
    calibration = fit_temperature(
        calibrated_model,
        calibration_items,
        tokenizer.pad_token_id,
        device,
    )
    tuned_config = apply_choice_temperature(tuned_config, calibration["temperature"])
    config_path.write_text(
        json.dumps(tuned_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del calibrated_model
    torch.cuda.empty_cache()

    agent = laya.Agent(str(OUTPUT_ROOT), device="cuda:0")
    evaluations = {
        split: evaluate(agent, split, question)
        for split in ("dev", "paraphrase_validation", "calibration", "test", "ood")
    }
    robustness = {
        split: order_robustness(agent, split, question, evaluations[split])
        for split in ("test", "ood")
    }
    gates = {
        "test": gate(
            evaluations["test"]["metrics"], robustness["test"], ood=False
        ),
        "ood": gate(evaluations["ood"]["metrics"], robustness["ood"], ood=True),
    }
    training_state = json.loads(
        (OUTPUT_ROOT / "ddp-training-state.json").read_text(encoding="utf-8")
    )
    weights_path = OUTPUT_ROOT / "model.safetensors"
    receipt = {
        "schema": "safr-laya-distillation-training-receipt-v2",
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "architecture": {
            "role": "probabilistic semantic finding sensor",
            "binding_disposition": False,
            "deterministic_floor_required": True,
            "safr_alignment": (
                "AI-specific semantic controls feed a deterministic Disposition Engine; "
                "identity, mandate syntax, limits, rate limits, execution, and audit remain deterministic."
            ),
        },
        "base_model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "weights_sha256": sha256(model_dir / "model.safetensors"),
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in agent.model.parameters()),
            "all_parameters_adapted": True,
            "checkpoint_selected_before_test": True,
            "weights_bytes": weights_path.stat().st_size,
            "weights_sha256": sha256(weights_path),
        },
        "code": {
            "laya_revision": LAYA_REVISION,
            "laya_wheel": laya_wheel.name,
            "laya_wheel_sha256": LAYA_WHEEL_SHA256,
        },
        "data": {
            "manifest_sha256": sha256(manifest_path),
            "manifest": manifest,
            "train": train_stats,
            "paraphrase_validation": validation_stats,
            "calibration": calibration_stats,
            "teacher_variant_policy": {
                "t01": "gradient updates",
                "t02": "checkpoint selection only",
            },
            "test_labels_used_for_training_or_calibration": False,
            "ood_labels_used_for_training_or_calibration": False,
        },
        "training": training_state,
        "calibration": calibration,
        "evaluations": evaluations,
        "order_robustness": robustness,
        "release_gates": gates,
        "semantic_gate_passed": gates["test"]["passed"] and gates["ood"]["passed"],
        "publish_allowed": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "laya": laya.__version__,
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
        "warnings": [
            "Synthetic controlled contrasts are not institutional AML/CFT ground truth.",
            "The model emits a semantic fact; deterministic code owns binding SAFR disposition.",
            "Teacher-generated examples are self-consistent weak supervision, not practitioner-reviewed truth.",
            "Paraphrase validation shares latent source worlds with base train and measures wording robustness only.",
            "Citation extraction is not trained in this checkpoint and remains a release blocker.",
            "The benchmark currently contains one primary exception per envelope.",
            "No Hugging Face publication is permitted without explicit owner approval and independent validation.",
        ],
    }
    (OUTPUT_ROOT / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_model_card(receipt)
    shutil.copy2(RELEASE / "README.md", OUTPUT_ROOT / "DATASET_CARD.md")
    evidence_dir = WORK_ROOT / "safr-laya-distillation-evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)
    for name in (
        "README.md",
        "DATASET_CARD.md",
        "training-receipt.json",
        "ddp-training-state.json",
        "rl_agent_config.json",
    ):
        shutil.copy2(OUTPUT_ROOT / name, evidence_dir / name)
    shutil.make_archive(
        str(evidence_dir),
        "gztar",
        root_dir=evidence_dir,
        base_dir=".",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_ROOT),
                "weights": receipt["model"],
                "calibration": calibration,
                "metrics": {
                    split: evaluations[split]["metrics"]
                    for split in ("dev", "paraphrase_validation", "calibration", "test", "ood")
                },
                "robustness": robustness,
                "gates": gates,
                "publish_allowed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
