# Mandate-1 public research release plan

Owner decision: Apache-2.0 for code and model adaptations; CC BY 4.0 for the synthetic dataset. Publication requested on GitHub, Hugging Face, and Kaggle. Preserve upstream Laya attribution and separate scientific limitations from license restrictions.

## Order of work

1. Verify a larger-model path before further student training: DeepSeek Flash, thinking disabled, JSON output, authenticated synthetic envelopes, hard controls first, typed semantic findings, fail-closed errors, decision audit before a sandbox read.
2. Exercise all four dispositions plus tampered signatures, replay, expired authority, malformed output, missing/unknown citations, model timeout, and audit failure. Measure useful allowed work as well as prevented actions; an all-deny harness is not evidence of usefulness.
3. Freeze the current student and receipts. Keep v0.1 results explicitly exploratory; do not advertise the repeatedly observed benchmark as a pristine holdout.
4. Build explicit-allowlist source, dataset, and model bundles. Include cards, readmes, source hashes, dataset splits, per-example evaluation evidence, trained model config/tokenizer, upstream notices, and an offline demo.
5. Validate schemas, contrast completeness, grouping/provenance, tokenizer budgets, hashes, secret exclusion, clean-install paths, tests, and demo behavior before public uploads.
6. Publish source/demo on GitHub, dataset mirrors on Hugging Face/Kaggle, and research checkpoint on Hugging Face/Kaggle. Verify public metadata and file checksums and record URLs/version IDs.

## Publication is not deployment approval

Historical receipts say `publish_allowed: false` because approval had not yet been given. Preserve them unchanged. A new release manifest will distinguish owner-authorized research publication from model suitability: this checkpoint is shadow-only, has no citation head, and still misses trace contradictions. No AML effectiveness, institutional policy compliance, or MAS endorsement claim is permitted.

## Exclusions

No `.env`, credentials, local session logs, private unrelated files, assistant skills, raw teacher reasoning, or MAS PDF in a release. Cite the MAS source rather than redistributing it. Do not publish earlier degenerate all-deny envelope fixtures as a useful evaluation benchmark; the live harness must show both correct allows and correct holds.

## After the first honest research release

Add independently authored, world-disjoint validation/challenge data; compare structured rules, balanced word/character logistic regression, NLI and CE-only training against RLCD; test CPU latency and end-to-end cost. Do not raise claims or change deployment thresholds just to declare a model successful.
