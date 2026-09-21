---
license: cc-by-4.0
language:
- en
pretty_name: Mandate-1 SAFR Synthetic Semantic Contrasts
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- synthetic
- weak-supervision
- runtime-governance
- safr
- aml-cft
- agentic-ai
- ai-safety
- fintech
- financial-services
- contrastive-evaluation
- model-evaluation
- system-1
configs:
- config_name: default
  data_files:
  - split: train
    path: dataset/train.jsonl
  - split: validation
    path: dataset/paraphrase_validation.jsonl
  - split: development
    path: dataset/dev.jsonl
  - split: calibration
    path: dataset/calibration.jsonl
  - split: test
    path: dataset/test.jsonl
  - split: ood
    path: dataset/ood.jsonl
- config_name: typed
  data_files:
  - split: train
    path: gliner25/train.jsonl
  - split: validation
    path: gliner25/paraphrase_validation.jsonl
  - split: development
    path: gliner25/dev.jsonl
  - split: calibration
    path: gliner25/calibration.jsonl
  - split: test
    path: gliner25/test.jsonl
  - split: ood
    path: gliner25/ood.jsonl
---

# Mandate-1 SAFR synthetic semantic contrasts

**Fictional research data, not transaction data, customer records, or practitioner-reviewed AML ground truth.** These controlled contrasts test semantic relationships in Governance-Envelope-like records: scope, instruction, trace, evidence and reviewed context.

[Try the demo](https://siva-sub.github.io/mandate-1/) · [Code](https://github.com/siva-sub/mandate-1) · [Model](https://huggingface.co/sivasub987/mandate-1-laya) · [Kaggle mirror](https://www.kaggle.com/datasets/sivasub987/mandate-1-safr-data)

## Contents and intended use

Each contrast pairs an exception with a similar-looking resolved/non-exception case. The input contains labelled synthetic text spans and an input digest. Targets contain an advisory control action and citation references. Seven categories cover no material semantic exception, insufficient evidence, outside mandate, influencing action, trace contradiction, instruction conflict and no reviewed analogue.

Use for prototyping semantic sensors, controlled-contrast evaluation, weak-supervision experiments and reproducible model comparisons. Never treat the target as authority to execute a financial action.

| File split | Canonical rows | Typed training/evaluation rows | Role |
|---|---:|---:|---|
| train | 720 | 2,520 | 360 original examples + 360 teacher t01 paraphrases; exception oversampling only in typed form |
| paraphrase_validation | 360 | 360 | Teacher t02 paraphrases; checkpoint selection, no gradient updates |
| dev | 60 | 210 | Template-matched development; typed exceptions oversampled |
| calibration | 60 | 60 | Temperature fitting after checkpoint selection |
| test | 120 | 120 | Exploratory frozen synthetic evaluation |
| ood | 60 | 60 | Authored near-OOD, not arbitrary world-level generalisation |

`dataset/` is the canonical format. `gliner25/` is a historical directory name for the seven-way typed format used to train Laya; it does not imply that GLiNER powers the released model. `manifest.json` records original construction hashes; `SHA256SUMS` covers files actually included in this distribution. Raw teacher generation batches are excluded. The GitHub release also retains the base pre-augmentation split for baseline reproduction.

## Generation and labels

The base examples were programmatically authored from fictional contrast worlds. DeepSeek Flash, thinking disabled, paraphrased train-only pairs; a separate request to the same teacher checked label self-consistency. This is **not independent validation**. The teacher alias is mutable; prompt hashes, API metadata and generation/self-consistency receipts document the process. No practitioner review has occurred.

The selected t01 variants receive gradient updates; t02 variants select checkpoints. **Validation shares the same latent training worlds** and therefore measures wording robustness, not world-disjoint generalisation. Some t02 rows retain the original `split: train` provenance field; use the containing filename/Hugging Face split for evaluation membership, not that inherited field.

Test and near-OOD rows were unchanged during this distillation run and not loaded for generation, gradient updates, selection or calibration. They have nevertheless been observed in earlier experiments. Do not describe them as a pristine, independently sealed final benchmark.

## Loading

```python
from datasets import load_dataset
rows = load_dataset("sivasub987/mandate-1-safr-data", split="test")
print(rows[0]["input"], rows[0]["targets"])
```

For exact provenance and hashes, download the JSONL files directly. Standard-library loading:

```python
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("dataset/test.jsonl").read_text().splitlines()]
```

Kaggle distribution: `kaggle datasets download -d sivasub987/mandate-1-safr-data --unzip -p data-download`. Kaggle expands the uploaded archive into `data-download/data-bundle/`. Run the standard-library example from that directory (or use `data-download/data-bundle/dataset/test.jsonl` directly). The Hugging Face layout starts at `dataset/` without the `data-bundle/` prefix.

## Limitations and misuse

English only; narrow fictional financial operations; one primary exception per example; template/lexical leakage possible; no genuine institutional policies, transaction distributions or investigator outcomes. A citation label is synthetic annotation, not independent evidence of truth. This dataset cannot establish AML/CFT detection quality, compliance, fairness across customers, money saved or time saved.

## License and attribution

CC BY 4.0. Attribute **siva-sub / Mandate-1**, link the repository, retain notices and indicate modifications. Third-party code/models have separate licenses; the MAS SAFR paper is referenced, not included.

Reference: [MAS SAFR](https://www.mas.gov.sg/-/media/mas-media-library/development/fintech/ai-safr/safr.pdf). Independent research; no MAS endorsement.
