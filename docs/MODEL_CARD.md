---
license: apache-2.0
language:
- en
base_model: convaiinnovations/laya-typed-decisions
base_model_relation: finetune
datasets:
- sivasub987/mandate-1-safr-data
pipeline_tag: text-classification
inference: false
tags:
- laya
- typed-decisions
- safr
- runtime-governance
- research
- shadow-only
- agentic-ai
- ai-safety
- fintech
- financial-services
- aml-cft
- system-1
- small-language-model
- synthetic-data
- model-evaluation
---

# Mandate-1 Laya — research checkpoint v0.2

> **Experimental / shadow-only. Published to document what was tried, including failures. Not approved for autonomous execution or financial decision-making.**

A 421,293,827-parameter Laya adaptation returning one advisory semantic finding from a fictional governance record. It does not detect money laundering, assess customers, issue regulatory decisions, or grant SAFR execution permission.

[Try the demo](https://siva-sub.github.io/mandate-1/) · [Code and reference harness](https://github.com/siva-sub/mandate-1) · [Dataset](https://huggingface.co/datasets/sivasub987/mandate-1-safr-data) · [Kaggle model](https://www.kaggle.com/models/sivasub987/mandate-1-laya)

## Role and interface

Output: one seven-way `choice` plus option scores. Labels: `no material semantic exception`, `insufficient evidence`, `outside mandate`, `influencing action`, `trace contradictory`, `instruction conflicting`, `no reviewed analogue`.

It has **no citation head, no multi-exception output, and no validated deployment abstention policy**. It is not a drop-in replacement for the five-primitive DeepSeek judge in the reference harness. Model scores never override identity, authority, categorical limits or human-review requirements.

## Training and provenance

- Base model revision: `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`.
- Laya code revision: `42626c348753fbb17572a813127df2278a1ec527`.
- Full-model adaptation on one NVIDIA A100-SXM4-40GB; RLCD proper reward + soft cross entropy; option permutation; 0.05 label smoothing.
- 2,520 typed training sequences after exception oversampling. Underlying canonical train: 720 examples.
- Best epoch 1; 3 epochs completed out of 6 maximum; effective batch 64; encoder/head learning rates 2.5e-5 / 1e-4; seed 42017.
- Training loop: 222.3 seconds, excluding setup/download/evaluation. This is not an end-to-end cost claim.
- Checkpoint selection: 360 t02 teacher paraphrases; gradient updates use original train + t01. Selection worlds overlap train worlds.
- Separate 60-example temperature calibration. An inherited option-count temperature override was corrected without changing weights or labels; correction is recorded in the receipt.
- Weights SHA-256: `7d9106c9d2b30d3f66368bb20c1c831ebebcf920d5a31dc3b2d18d09a971cdd5`.

## Measured results

| Split | Accuracy | Complete-pair accuracy | Macro-F1 | False-clear rate |
|---|---:|---:|---:|---:|
| dev | 100.0% | 100.0% | 1.000 | 0.0% |
| paraphrase validation | 99.2% | 98.3% | 0.992 | 0.0% |
| calibration | 100.0% | 100.0% | 1.000 | 0.0% |
| test (120 examples) | 84.2% | 68.3% | 0.804 | 18.3% |
| near-OOD (60 examples) | 91.7% | 83.3% | 0.846 | 16.7% |

False clear means an exception example was predicted as having no semantic exception. It is not a money-laundering false-negative rate. Complete-pair accuracy requires both members of a contrast to be correct. Recorded GPU p50 inference latency was about 36 ms per example in the training environment; CPU and end-to-end production costs are unestablished.

The model misses important trace contradictions and did not meet the declared model-use gates. The synthetic test has been repeatedly observed during earlier experiments; results are exploratory, not evidence from a pristine final holdout. Stronger structured-rule/NLI baselines and practitioner-reviewed, world-disjoint challenges remain future work.

## Run the checkpoint

The custom Laya loader is required; this is not an `AutoModelForSequenceClassification` checkpoint. From the companion source repository:

```bash
python -m venv .venv
. .venv/bin/activate
pip install vendor/laya-0.3.4-py3-none-any.whl 'huggingface_hub<2' safetensors
python scripts/predict_safr_laya.py --repo sivasub987/mandate-1-laya --device cpu
```

This downloads approximately 846 MB and runs one fictional example. Use a suitable CUDA PyTorch installation and `--device cuda:0` for GPU inference. Upstream dependency resolution can change; the exact observed training environment is recorded in `training-receipt.json`. It is not a promise of bitwise retraining reproducibility.

Kaggle variation: [research-v02, version 1](https://www.kaggle.com/models/sivasub987/mandate-1-laya/PyTorch/research-v02/1). The bundle retains the nested tokenizer/encoder paths; Kaggle expands the uploaded ZIP into `model-bundle/`.

```bash
kaggle models instances versions download sivasub987/mandate-1-laya/pyTorch/research-v02/1 --untar -p model-download
python scripts/predict_safr_laya.py --local-model model-download/model-bundle --device cpu
```

The downloaded Kaggle bundle's SHA-256 checksums have been verified against the published files.

## Limits and release status

Synthetic, English-only, unreviewed labels; weak supervision by a mutable hosted teacher; train-world-overlapping validation; one primary exception; no evidence relevance verification. The model assumes declared trust metadata and cannot authenticate it. Never use it to close accounts, contact customers, submit filings, move money, expand authority, or silently clear a control.

Historical training receipts retain `publish_allowed: false` and failed gates unchanged. Owner-authorised **research publication** is recorded separately in `release-manifest.json`; it does not retroactively change those findings or approve deployment.

## License

Apache-2.0 for this adaptation and upstream Laya model/software; preserve `LICENSE` and `NOTICE`. Dataset: separate CC BY 4.0. Independent research inspired by [MAS SAFR](https://www.mas.gov.sg/-/media/mas-media-library/development/fintech/ai-safr/safr.pdf); no endorsement or regulatory certification.
