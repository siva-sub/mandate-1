# Mandate-1 architecture

## Implemented reference path

This release is a single-process, serial synthetic sandbox inspired by the separation of identity, controls, disposition, execution and traceability in MAS's [SAFR paper](https://www.mas.gov.sg/-/media/mas-media-library/development/fintech/ai-safr/safr.pdf). It is not a complete SAFR deployment or institutional compliance assessment.

1. **Proposal and envelope:** a fictional agent proposes a registered operator. The envelope includes identity, mandate validity, instruction, runtime trace, evidence inventory, policy version, case digest and nonce.
2. **Authentication and hard controls:** a local HMAC authenticates the envelope. Deterministic checks verify identity, authority, supported action, time bounds and replay state. Consequential actions cannot execute in this sandbox.
3. **Semantic assessment:** DeepSeek Flash, thinking disabled, assesses scope alignment, trace consistency, evidence sufficiency, instruction alignment and novelty. It returns strict JSON and supplied evidence IDs. It cannot return an executable command or binding disposition.
4. **Disposition:** versioned code maps hard controls and findings to `deny`, `escalate`, `auto_execute` or `observe`. Missing, stale, malformed and unverifiable findings hold execution.
5. **Audit then execution:** the decision is appended before a permitted read. The only executor retrieves a record already in the authenticated synthetic inventory. It has no filesystem-path, arbitrary network, bank, payment, customer-contact or filing capability. An execution receipt follows.

## Trust boundaries

| Boundary | Mechanism | Limitation |
|---|---|---|
| Proposal → authority | Signed envelope and registered operator policy | Runtime-created envelope and signing key are trusted |
| Model → gate | Typed findings, input digest, citation existence and policy code | Citation relevance is not independently checked |
| Gate → executor | Exact allowlisted internal-read operator and fixed parameters | In-memory demonstration, not an OS/process isolation boundary |
| Decision → evidence | Hash-linked decision and execution records | Not immutable storage; a privileged writer can rewrite a whole chain |
| Replay → next action | Seen nonces reconstructed from audit entries | Serial runtime only; no distributed concurrency or durable nonce transaction |

The signing secret is ephemeral and never published. Public receipts support checking the hash chain and inspected inputs, not re-verifying the original secret HMAC. An audit error before execution propagates and prevents the read. A process crash or audit failure after the read is not a transactional execution guarantee.

## Two model interfaces, not one

**Reference judge:** five semantic primitives plus evidence IDs, supplied by the live hosted model. The provider's alias is mutable, so each receipt records API identity, prompt hash, usage and latency. Only the allowlisted synthetic requests are sent externally.

**Experimental student:** a 421M-parameter Laya adaptation returning one of seven semantic labels. It has no trained citation head and cannot represent multiple simultaneous exceptions. It remains a shadow-only classifier and is not used to release actions in the live harness.

A compact typed model is worth pursuing only if it offers a measured cost/latency advantage at acceptable failure rates versus structured rules, lexical baselines and larger models. This release demonstrates feasibility of the hosted-judge integration, not that the compact model is already superior.

## Demonstration versus benchmark

The 14 authored harness checks exercise all four dispositions, safe reads, hard failures and injected model failures. They are integration examples, not an independent accuracy sample. Older envelope fixture generation code remains only for contract/schema regression testing; its all-deny behavior is not used as evidence of useful governance.

The separately released contrast dataset probes changing semantic relationships while keeping wording similar. It is synthetic and partly weakly supervised. Per-example results and the student's false clears are published so success is not defined as merely blocking everything.

## Production requirements not implemented

Institutional policy review; verified runtime trace capture; protected key management and rotation; distributed replay protection; capability isolation; privacy controls for actual case data; immutable external audit anchors; transactional execution/recovery; independent semantic/citation evaluation; red teaming; concurrency/rate-limit enforcement; reviewer workflows; operational monitoring; rollback and incident response. None is implied by this research release.
