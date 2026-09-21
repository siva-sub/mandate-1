# Mandate-1 · Before the agent acts

**A SAFR-inspired research harness that lets an authorised evidence read complete—and shows why another request must stop.**

An AML/CFT investigation assistant can propose a useful next check without receiving permission to execute arbitrary actions. Mandate-1 separates semantic judgement from authority: a model interprets the instruction and evidence; deterministic code owns the final disposition.

**[Try the recorded interactive demo](https://siva-sub.github.io/mandate-1/)** · [Dataset](https://huggingface.co/datasets/sivasub987/mandate-1-safr-data) · [Research checkpoint](https://huggingface.co/sivasub987/mandate-1-laya) · [Kaggle dataset](https://www.kaggle.com/datasets/sivasub987/mandate-1-safr-data) · [Kaggle model](https://www.kaggle.com/models/sivasub987/mandate-1-laya)

> Experimental, synthetic-data proof of concept. Not an AML detector, bank policy implementation, compliance certification, or MAS-endorsed product. The Laya checkpoint is **shadow-only** and does not power the live reference harness.

## Why I built it

I'm **Sivasubramanian Ramanathan (Siva)**, an independent consultant previously at the BIS Innovation Hub, exploring AI, tokenization and payments and open to opportunities. Jev's System-1 idea made me ask: *could a fast semantic model help with SAFR-style controls?* I tried custom Needle 3 and GLiNER 2.5 fine-tunes, then Laya, and built a working DeepSeek Flash reference path.

The distinction is **dangerous actions versus dangerous words**. The goal is not another keyword blocklist: it is to interpret the relationship between the proposal, authority and runtime evidence—without letting a model grant itself permission. This is independent work, not endorsed by a former employer.

[My motivation and portfolio](docs/PORTFOLIO.md) · [All experiments, including failures](docs/EXPERIMENTS.md)

## See the value first

Open `demo/index.html` locally; no installation, API key or web server is needed. Select a scenario and click **Show recorded outcome**.

- **Authorised read:** the sandbox returns the requested fictional treasury agreement.
- **Quoted injection:** dangerous wording is treated as data when telemetry shows it was not followed.
- **Contradictory trace:** the claimed lookup and actual lookup disagree; the read is held.
- **New but authorised context:** the read proceeds under `observe`.
- **Consequential action, expired authority, replay or tampering:** no model can override the block.
- **Timeout, malformed output or invented citation:** no execution is released.

The page replays stored receipts; it does **not** call a model when clicked. `demo/contrast.html` separately compares keyword rules, n-grams, student checkpoints and the hosted teacher on controlled contrasts.

## What was actually run

| Evidence | Observation | What it does not establish |
|---|---|---|
| Live reference harness | 7 DeepSeek Flash requests, thinking disabled; 7 valid typed responses | General reliability on independent production traffic |
| End-to-end demonstration | 14/14 expected dispositions, including deterministic and injected-failure checks | A statistically meaningful safety rate |
| Distilled Laya, synthetic test | 84.2% accuracy; 68.3% complete-pair accuracy; 18.3% false-clear rate | Readiness to authorise financial actions |
| Distilled Laya, near-OOD | 91.7% accuracy; 83.3% pair accuracy; 16.7% false-clear rate | Robustness to arbitrary new institutions or workflows |

The compact student improved over the earlier template-trained checkpoint but still misses important contradictions. We publish that negative result rather than lower the safety bar to call it production-ready. See [model card](docs/MODEL_CARD.md) and the unchanged per-example receipts in `research-evidence/`.

## Architecture

```text
untrusted proposal → signed synthetic envelope → hard authority checks
  → advisory semantic judge → deterministic disposition → decision audit
  → permitted in-memory read → execution receipt
```

The gate supports `deny`, `escalate`, `auto_execute` and `observe`. DeepSeek returns five typed semantic findings and evidence IDs—not a command or permission token. Laya emits a separate seven-way advisory classification; it is not a drop-in replacement for this richer interface.

[Architecture and SAFR mapping](ARCHITECTURE.md) · [Dataset card](docs/DATASET_CARD.md) · [Reproduce the experiments](docs/REPRODUCIBILITY.md)

## Run locally

Python 3.11+; the runtime and dataset builders use the standard library.

```bash
git clone https://github.com/siva-sub/mandate-1.git
cd mandate-1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest -q tests
PYTHONPATH=. python scripts/build_safr_harness_demo.py --output demo/index.html
PYTHONPATH=. python scripts/build_safr_gate_demo.py --output demo/contrast.html
```

To perform **a new charged live run**, set `DEEPSEEK_API_KEY` and optionally `DEEPSEEK_BASE_URL=https://api.deepseek.com` in your environment or local `.env`. Never commit credentials.

```bash
python scripts/run_safr_harness.py --live --out research-evidence/my-new-run
```

This makes at most seven requests, with thinking disabled and no retries. Output directories must be new. Only fictional envelopes are sent; the executor cannot contact a bank, customer or payment endpoint.

## Release layout

| Directory | Contents |
|---|---|
| `aml_nextstep/` | SAFR contracts, controls, audit, semantic data/evaluation and sandbox runtime |
| `tests/` | Focused SAFR regression tests |
| `scripts/` | Dataset construction, training, calibration repair, evaluation and demo builders |
| `data/` | Frozen base and teacher-augmented synthetic splits with provenance |
| `research-evidence/` | Live receipts, per-example model evaluations and baseline comparisons |
| `demo/` | Ready-to-open offline replays and their HTML templates |
| `docs/` | Cards, reproducibility instructions and release/launch notes |
| `vendor/` | Hash-pinned Apache-2.0 Laya wheel used for the experiment |

Model weights live on Hugging Face and Kaggle, not in Git history. Superseded experiments, failed checkpoint binaries, old product plans, local agent skills, caches, credentials and the MAS PDF are excluded from the public release. They remain in the local-only archive.

## Limits and next work

The data are fictional, English-only, programmatically labelled or teacher-generated, and not practitioner-reviewed. Paraphrase validation shares underlying training worlds. The synthetic test has been observed during previous experiments and is not a pristine permanent holdout. Citation existence is checked, but relevance is not independently verified. HMAC and hash-linked local logs illustrate contracts; they are not production identity management or immutable audit infrastructure.

Next: independent world-disjoint challenges, stronger structured-rule/NLI baselines, citation and multi-exception evaluation, measured CPU cost, and practitioner assessment. There is no demonstrated AML detection benefit or investigator time saving yet.

## Licensing and attribution

Code and model adaptation: [Apache-2.0](LICENSE). Original synthetic dataset: [CC BY 4.0](DATA_LICENSE.txt). Attribute the dataset to **siva-sub / Mandate-1** and retain provenance. Upstream models/software retain their respective notices; see [NOTICE](NOTICE).

Inspired by MAS's [SAFR framework](https://www.mas.gov.sg/-/media/mas-media-library/development/fintech/ai-safr/safr.pdf). The framework is cited, not redistributed. The released model adapts [Laya Typed-Decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions).
