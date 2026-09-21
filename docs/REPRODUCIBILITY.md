# Reproducing the research release

## 1. Offline harness and regression tests

Use Python 3.11+ and the source repository README commands. No model packages or credentials are needed for the SAFR unit tests or rebuilding either recorded HTML replay. The replay embeds evidence; it is not a fresh API evaluation.

The source release includes only SAFR tests. The earlier 278-test count included superseded non-SAFR experiments archived locally; do not use that count for this focused release.

## 2. New hosted-judge run

Set `DEEPSEEK_API_KEY`, optionally `DEEPSEEK_BASE_URL`, then run:

```bash
python scripts/run_safr_harness.py --live --out research-evidence/new-run
```

Maximum seven charged requests, `deepseek-flash`, thinking disabled, JSON mode, no retries. Only synthetic input is sent. Inspect `receipt.json` and `audit.jsonl`; non-matching cases are evidence, not grounds to silently relabel a test. The API alias is mutable. The existing receipt hashes refer to exact historical files, including code before later packaging-only changes.

## 3. Data

Both frozen base data and teacher-augmented canonical/typed splits are included under `data/`. Each has a provenance manifest. The release intentionally omits raw generation batches and unused adapters; historical manifests may therefore mention files outside the distributed subset. The release-level `SHA256SUMS` is the authority for files actually distributed.

The teacher-augmented dataset is also on Hugging Face and Kaggle. Canonical rows total 1,380 across six files. Paraphrase validation overlaps training worlds; its inherited row `split` field may say `train`. Evaluate according to the file boundary. Typed train/dev apply explicit exception oversampling; never treat duplicates as independent observations.

## 4. Student inference

Install the bundled Laya wheel and Hugging Face dependencies as shown in the model card. The wheel's expected SHA-256 is `1883529eafb06168a604d8f218703db7f205f2db209b8b68b3cba4a9334b1818`. The loader is custom and needs `encoder/`, `tokenizer/`, `rl_agent_config.json` and `model.safetensors` together.

```bash
python scripts/predict_safr_laya.py --repo sivasub987/mandate-1-laya --device cpu
```

`--local-model /path/to/extracted/model` avoids a download. Output is advisory only. The script deliberately does not invoke the execution harness.

## 5. Retraining (optional, GPU compute)

Run in a disposable environment on a CUDA GPU with enough memory. The recorded run used one A100 40 GB, Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1 and Laya 0.3.4. Package/hardware differences can change results. The original Kaggle GPU request was not provisioned; the completed run was on Colab.

```bash
export SAFR_PROJECT_ROOT="$PWD"
export SAFR_WORK_ROOT=/path/to/fresh-writable-work
python scripts/kaggle_train_laya_safr_distill.py
```

This installs the hash-pinned wheel, downloads the pinned upstream model, trains for up to six epochs, selects on teacher t02 paraphrases and calibrates on the separate calibration split. It can use one or two CUDA GPUs. Review the script before starting: training incurs compute use. Test/OOD are evaluated only after checkpoint selection in this run, but have been observed in earlier experiments.

The calibration repair script and regression tests document the inherited option-count override. The released weights are the original selected weights; only the effective temperature config was corrected. Receipts are retained unchanged.

## 6. Release and provenance

`python scripts/prepare_research_release.py --out dist/mandate-1-v0.2` creates fresh, allowlisted GitHub, Hugging Face and Kaggle bundles. It refuses an existing output directory. It never publishes, reads credential values or calls a model. Upload only these prepared folders—not the workspace root.

GitHub contains source, data, cards, evidence and offline demos, but no trained weights. Hugging Face and Kaggle host the selected checkpoint. Code/model are Apache-2.0; original data are CC BY 4.0. Public dissemination and model deployment approval are explicitly separate.
