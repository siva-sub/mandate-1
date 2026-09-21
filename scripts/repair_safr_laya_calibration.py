#!/usr/bin/env python3
"""Correct inherited Laya temperature overrides without training or overwriting evidence.

Run in the original training environment. Uses only the previously fitted
calibration scalar, not test/OOD labels, to repair the saved inference config.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists; refusing to overwrite experiment evidence")
    os.environ["USE_TF"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import kaggle_train_laya_safr_distill as trainer
    from aml_nextstep.safr_gliner import FINDING_DESCRIPTIONS, FINDING_INSTRUCTION
    laya = importlib.import_module("laya")

    receipt_path = args.source / "training-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    weights_hash = trainer.sha256(args.source / "model.safetensors")
    if weights_hash != receipt["model"]["weights_sha256"]:
        raise SystemExit("source weights do not match training receipt")
    config_path = args.source / "rl_agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fitted = receipt["calibration"]["temperature"]
    updated = trainer.apply_choice_temperature(config, fitted)
    shutil.copytree(args.source, args.output)
    (args.output / "rl_agent_config.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    agent = laya.Agent(str(args.output), device="cuda:0")
    effective = agent.temperature_by_options.get("choice:6-10", agent.temperature[0])
    if effective != fitted:
        raise RuntimeError("Laya ignored the fitted temperature after repair")
    question = {
        "type": "choice",
        "instructions": FINDING_INSTRUCTION,
        "criteria": FINDING_DESCRIPTIONS,
    }
    prior_evaluations = receipt["evaluations"]
    evaluations = {}
    for split in ("dev", "paraphrase_validation", "calibration", "test", "ood"):
        result = trainer.evaluate(agent, split, question)
        previous = {
            item["example_id"]: item["predicted_finding"]
            for item in prior_evaluations[split]["predictions"]
        }
        changed = [
            item["example_id"] for item in result["predictions"]
            if previous[item["example_id"]] != item["predicted_finding"]
        ]
        if changed:
            raise RuntimeError(f"calibration unexpectedly changed ranking: {split} {changed}")
        evaluations[split] = result
        print(json.dumps({"split": split, "labels_unchanged": True,
                          "metrics": result["metrics"],
                          "probability_metrics": result["probability_metrics"]}), flush=True)
    robustness = {
        split: trainer.order_robustness(agent, split, question, evaluations[split])
        for split in ("test", "ood")
    }
    gates = {
        split: trainer.gate(evaluations[split]["metrics"], robustness[split], ood=split == "ood")
        for split in ("test", "ood")
    }
    if trainer.sha256(args.output / "model.safetensors") != weights_hash:
        raise RuntimeError("weights changed during calibration-only correction")
    receipt.update({
        "schema": "safr-laya-distillation-training-receipt-v2-calibration-fixed",
        "evaluations": evaluations,
        "order_robustness": robustness,
        "release_gates": gates,
        "semantic_gate_passed": all(value["passed"] for value in gates.values()),
        "calibration_correction": {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source_receipt_sha256": trainer.sha256(receipt_path),
            "source_config_sha256": trainer.sha256(config_path),
            "corrected_config_sha256": trainer.sha256(args.output / "rl_agent_config.json"),
            "effective_choice_temperature": effective,
            "old_effective_choice_temperature": config.get("temperature_by_options", {}).get(
                "choice:6-10", config["temperature"][0]
            ),
            "weights_unchanged": True,
            "canonical_labels_unchanged_all_splits": True,
            "fitting_or_training_repeated": False,
            "test_used_to_choose_correction": False,
            "reason": "Inherited option-count bucket silently overrode fitted per-type temperature.",
        },
    })
    receipt["warnings"].extend([
        "Calibration-set NLL is a fit diagnostic, not an independent generalization result.",
        "The v0.1 evaluation set has been observed in prior experiments; this is exploratory, not a pristine final holdout.",
        "Training stop logic was revised after an interrupted exploratory run; no confirmatory trial is claimed.",
    ])
    (args.output / "training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer.OUTPUT_ROOT = args.output
    trainer.write_model_card(receipt)
    evidence_dir = args.output.with_name(args.output.name + "-evidence")
    evidence_dir.mkdir(exist_ok=False)
    for name in ("README.md", "DATASET_CARD.md", "training-receipt.json", "ddp-training-state.json", "rl_agent_config.json"):
        shutil.copy2(args.output / name, evidence_dir / name)
    for source in (Path(__file__), Path(trainer.__file__), Path(__file__).with_name("kaggle_laya_safr_distill_worker.py")):
        shutil.copy2(source, evidence_dir / source.name)
    shutil.make_archive(str(evidence_dir), "gztar", root_dir=evidence_dir)
    shutil.make_archive(str(args.output), "gztar", root_dir=args.output)
    print(json.dumps({"output": str(args.output), "weights_sha256": weights_hash,
                      "effective_temperature": effective, "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
