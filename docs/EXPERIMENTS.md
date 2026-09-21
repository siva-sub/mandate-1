# Experiment ledger — keep the unsuccessful attempts visible

This portfolio began with the Jev/System-1 idea: can a compact typed model assess semantic relationships in a SAFR-style control workflow? The experiments below are related research, not interchangeable model contracts or a controlled architecture bake-off.

| Candidate | What was actually tried | Evidence and conclusion |
|---|---|---|
| Needle 3 | Custom LoRA fine-tunes, including long-prompt, short-prompt and positive-label formulations; rank 16, alpha 32, 2–5 epochs; exported 8-/20-layer runtime artifacts | Training receipts, adapter diagnostics and available evaluation preserved under `research-evidence/exploratory/mandate1-firewall-*`. Not selected for the reference runtime. The confidence head was not fine-tuned and must not be represented as calibrated. |
| GLiNER 2.5 small | Custom LoRA pilots: imbalanced, focused/balanced, mixed and classification-only formulations; approximately 74M base parameters | Training/configuration and independent evaluation receipts under `research-evidence/exploratory/mandate1-gliner25-*`. Some pilots were interrupted. The evaluated variants did not pass declared model-use gates. |
| Laya, template training | Full-model typed-decision adaptation on authored templates | `research-evidence/safr-laya-template-overfit.json`; poor generalisation despite strong template-matched results. |
| Laya, teacher-diversified | Full-model adaptation using original examples + DeepSeek train paraphrases, separate paraphrase selection and temperature calibration | Selected 421M checkpoint is released. Test accuracy 84.2%, complete-pair accuracy 68.3%, false clear 18.3%. Shadow-only; no trained citation head. |
| DeepSeek Flash | Thinking disabled; typed JSON findings; deterministic hard checks and execution gate | Seven live requests and fourteen total authored integration checks recorded in `research-evidence/safr-live-harness-v1/`. Demonstrates a working reference path, not production safety or independent benchmark accuracy. |
| Keyword and word n-gram baselines | Rules and lexical classifiers on frozen/augmented synthetic contrasts | Published comparison receipts and `demo/contrast.html`. Useful baselines, not an exhaustive comparison with structured rule engines, NLI or other safety systems. |

## What is and is not shipped

The selected Laya checkpoint is the only trained model binary published. Earlier experimental weights remain archived locally; selected JSON receipts preserve evidence without flooding the release with failed binaries. The earlier training recipes are historical experiments rather than guaranteed turnkey paths. Current Laya training, calibration, dataset and live-harness scripts are in the main source release.

Historical flags such as `publish_allowed: false` remain untouched. Permission to publish research is recorded separately; failed model-use gates still mean failed model-use gates. No trial establishes AML/CFT detection effectiveness or a return on investment.

Needle source model: `Cactus-Compute/needle3` at `b274efcb211a9eef48c9a88da4b43bd569696a39`.
GLiNER source model: `fastino/gliner2.5-small-v1` at `7e6f537f10337497069276892a5ef435028252ce`.
Laya source/model revisions are in the model card. Their respective software/model licenses remain separate; this release does not redistribute Needle or GLiNER weights.
