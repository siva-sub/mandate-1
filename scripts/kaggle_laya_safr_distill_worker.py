#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""DDP worker for teacher-diversified Laya SAFR distillation.

The upstream RLCD/proper-scoring objective is retained. Checkpoint selection is
performed only on a reserved train-world paraphrase variant; frozen test and OOD
rows are not loaded by this worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from laya.common import build_model, proper_reward
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

SEED = 42_017
ENCODER_LR = 2.5e-5
HEAD_LR = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.08
GROUP_SIZE = 4
SIGMA_START = 0.4
SIGMA_END = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--validation-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    return parser.parse_args()


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    count = len(items)
    sequence_length = max(len(item["ids"]) for item in items)
    option_count = max(len(item["markers"]) for item in items)
    input_ids = torch.full((count, sequence_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((count, sequence_length), dtype=torch.long)
    marker_pos = torch.zeros((count, option_count), dtype=torch.long)
    marker_mask = torch.zeros((count, option_count), dtype=torch.bool)
    target = torch.zeros((count, option_count), dtype=torch.float32)
    for index, item in enumerate(items):
        input_ids[index, : len(item["ids"])] = torch.tensor(item["ids"])
        attention_mask[index, : len(item["ids"])] = 1
        marker_pos[index, : len(item["markers"])] = torch.tensor(item["markers"])
        marker_mask[index, : len(item["markers"])] = True
        target[index, : len(item["target"])] = torch.tensor(
            item["target"], dtype=torch.float32
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
        "label": torch.tensor([item["label"] for item in items]),
    }


def validation_metrics(
    model: Any,
    items: list[dict[str, Any]],
    *,
    pad_id: int,
    device: torch.device,
    micro_batch: int,
) -> dict[str, float | int]:
    """Evaluate the reserved t02 paraphrases without changing the checkpoint."""
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []
    nll_total = 0.0
    with torch.no_grad():
        for start in range(0, len(items), micro_batch):
            chunk = items[start : start + micro_batch]
            batch = collate(chunk, pad_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float().masked_fill(~batch["marker_mask"].to(device), -1e4)
            labels = batch["label"].to(device)
            nll_total += float(
                functional.cross_entropy(logits, labels, reduction="sum").item()
            )
            predictions.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch["label"].tolist())
    model.train()

    groups: dict[str, list[bool]] = defaultdict(list)
    false_clears = 0
    exceptions = 0
    false_holds = 0
    hard_negatives = 0
    correct = 0
    for item, predicted, expected in zip(items, predictions, gold, strict=True):
        is_correct = predicted == expected
        correct += int(is_correct)
        groups[str(item["contrast_group_id"])].append(is_correct)
        if item["polarity"] == "exception":
            exceptions += 1
            false_clears += int(predicted == 0)
        elif item["polarity"] == "hard_negative":
            hard_negatives += 1
            false_holds += int(predicted != 0)
        else:
            raise ValueError(f"unknown validation polarity: {item['polarity']}")
    malformed_groups = [group for group, values in groups.items() if len(values) != 2]
    if malformed_groups:
        raise ValueError(f"validation contrast groups are incomplete: {malformed_groups[:5]}")
    exact = correct / max(1, len(items))
    pair = sum(all(values) for values in groups.values()) / max(1, len(groups))
    false_clear = false_clears / max(1, exceptions)
    false_hold = false_holds / max(1, hard_negatives)
    return {
        "examples": len(items),
        "contrast_groups": len(groups),
        "exact_set_accuracy": exact,
        "contrast_pair_accuracy": pair,
        "false_clear_rate": false_clear,
        "false_hold_rate": false_hold,
        "nll": nll_total / max(1, len(items)),
    }


def selection_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    """Safety-aware lexicographic checkpoint preference, not a deployment score."""
    return (
        float(metrics["contrast_pair_accuracy"]),
        -float(metrics["false_clear_rate"]),
        float(metrics["exact_set_accuracy"]),
        -float(metrics["false_hold_rate"]),
        -float(metrics["nll"]),
    )


def half_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().half().contiguous().cpu()
        for name, value in model.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise SystemExit("epochs and patience must be positive")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)

    config = json.loads(
        (args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8")
    )
    config["gradient_checkpointing"] = True
    config["max_tokens_per_batch"] = 4096
    config["max_len"] = 1024
    config["head_max_len"] = 256
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer")
    model = build_model(config, encoder_dir=str(args.model_dir / "encoder"))
    model.load_state_dict(
        load_file(str(args.model_dir / "model.safetensors")), strict=True
    )
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    all_items = json.loads(args.items.read_text(encoding="utf-8"))
    validation_items = json.loads(args.validation_items.read_text(encoding="utf-8"))
    local_items = all_items[rank::world_size]
    encoder_parameters = [
        parameter
        for name, parameter in ddp_model.named_parameters()
        if "encoder." in name
    ]
    head_parameters = [
        parameter
        for name, parameter in ddp_model.named_parameters()
        if "encoder." not in name
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LR},
            {"params": head_parameters, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    updates_per_epoch = math.ceil(
        math.ceil(len(local_items) / args.micro_batch) / args.gradient_accumulation
    )
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(1, round(total_updates * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "maximum_epochs"
    best_checkpoint = args.output_dir / "best-checkpoint.safetensors"

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "training_started",
                    "world_size": world_size,
                    "train_items": len(all_items),
                    "validation_items": len(validation_items),
                    "items_per_rank": len(local_items),
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "effective_batch": (
                        args.micro_batch * args.gradient_accumulation * world_size
                    ),
                    "warmup_updates": warmup_updates,
                    "total_updates": total_updates,
                    "test_or_ood_loaded": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(args.epochs):
        epoch_rng = random.Random(SEED + epoch + rank)
        epoch_rng.shuffle(local_items)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_reward = 0.0
        batches = 0
        accumulation_index = 0
        progress = epoch / max(1, args.epochs - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for start in range(0, len(local_items), args.micro_batch):
            chunk = local_items[start : start + args.micro_batch]
            if not chunk:
                continue
            batch = collate(chunk, tokenizer.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                logits, activation = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            option_count = mask.sum(-1, keepdim=True).float()
            noise = (
                torch.randn((GROUP_SIZE,) + tuple(logits.shape), device=device)
                * sigma
                * mask
            )
            noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask
            sampled_logits = logits.detach().unsqueeze(0) + noise
            sampled_probabilities = torch.softmax(
                sampled_logits.masked_fill(~mask, -1e4), dim=-1
            )
            with torch.no_grad():
                reward = proper_reward(
                    sampled_probabilities,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            log_probability = -(
                ((sampled_logits - logits.unsqueeze(0)) ** 2) * mask
            ).sum(-1) / (2 * sigma**2)
            reinforcement_loss = -(advantage * log_probability).mean()
            cross_entropy_loss = -(
                target
                * torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
            ).sum(-1).mean()
            loss = (
                reinforcement_loss + cross_entropy_loss + 0.0 * activation.sum()
            ) / args.gradient_accumulation
            scaler.scale(loss).backward()
            accumulation_index += 1
            if (
                accumulation_index % args.gradient_accumulation == 0
                or start + args.micro_batch >= len(local_items)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.item()) * args.gradient_accumulation
            epoch_reward += float(reward.mean().item())
            batches += 1

        rank_loss = torch.tensor(
            [epoch_loss, epoch_reward, float(batches)], device=device
        )
        dist.all_reduce(rank_loss, op=dist.ReduceOp.SUM)
        dist.barrier()
        stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0:
            combined_batches = max(float(rank_loss[2].item()), 1.0)
            metrics = validation_metrics(
                model,
                validation_items,
                pad_id=tokenizer.pad_token_id,
                device=device,
                micro_batch=args.micro_batch,
            )
            current_key = selection_key(metrics)
            improved = best_key is None or current_key > best_key
            if improved:
                best_key = current_key
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_file(half_state(model), str(best_checkpoint))
            else:
                epochs_without_improvement += 1
            epoch_record = {
                "epoch": epoch + 1,
                "loss": float(rank_loss[0].item()) / combined_batches,
                "proper_reward": float(rank_loss[1].item()) / combined_batches,
                "sigma": sigma,
                "encoder_learning_rate": scheduler.get_last_lr()[0],
                "head_learning_rate": scheduler.get_last_lr()[1],
                "paraphrase_validation": metrics,
                "checkpoint_improved": improved,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(epoch_record)
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
            perfect_semantics = (
                metrics["contrast_pair_accuracy"] == 1.0
                and metrics["exact_set_accuracy"] == 1.0
                and metrics["false_clear_rate"] == 0.0
                and metrics["false_hold_rate"] == 0.0
            )
            if epoch + 1 >= 2 and perfect_semantics:
                stop_reason = "perfect_paraphrase_semantics_after_minimum_epochs"
                stop_tensor.fill_(1)
            elif epoch + 1 >= 3 and epochs_without_improvement >= args.patience:
                stop_reason = "paraphrase_validation_patience_exhausted"
                stop_tensor.fill_(1)
        dist.broadcast(stop_tensor, src=0)
        dist.barrier()
        if int(stop_tensor.item()) == 1:
            break

    dist.barrier()
    if rank == 0:
        if not best_checkpoint.exists():
            raise RuntimeError("no best checkpoint was written")
        final_weights = args.output_dir / "model.safetensors"
        shutil.copyfile(best_checkpoint, final_weights)
        best_checkpoint.unlink()
        model.encoder.config.save_pretrained(args.output_dir / "encoder")
        tokenizer.save_pretrained(args.output_dir / "tokenizer")
        config["fine_tuned"] = True
        config["model_name"] = "safr-laya-semantic-firewall-distilled-v0.2"
        config["temperature"] = [1.0, 1.0, 1.0]
        (args.output_dir / "rl_agent_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        training_state = {
            "schema": "safr-laya-distillation-training-state-v2",
            "completed_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "seed": SEED,
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(local_rank),
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "micro_batch_per_gpu": args.micro_batch,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch": (
                args.micro_batch * args.gradient_accumulation * world_size
            ),
            "optimizer": "AdamW",
            "encoder_learning_rate": ENCODER_LR,
            "head_learning_rate": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_updates": warmup_updates,
            "objective": "RLCD proper reward + soft cross entropy",
            "target_policy": "0.05 label smoothing on train; hard labels on validation",
            "checkpoint_selection": {
                "split": "teacher paraphrase variant t02",
                "test_or_ood_loaded": False,
                "lexicographic_order": [
                    "maximize contrast_pair_accuracy",
                    "minimize false_clear_rate",
                    "maximize exact_set_accuracy",
                    "minimize false_hold_rate",
                    "minimize nll",
                ],
                "patience": args.patience,
                "minimum_epochs": 2,
                "stop_on_perfect_semantic_metrics": True,
            },
            "gradient_checkpointing": True,
            "history": history,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes_rank_0": torch.cuda.max_memory_allocated(local_rank),
        }
        (args.output_dir / "ddp-training-state.json").write_text(
            json.dumps(training_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "model_saved",
                    "output": str(args.output_dir),
                    "best_epoch": best_epoch,
                    "stop_reason": stop_reason,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
