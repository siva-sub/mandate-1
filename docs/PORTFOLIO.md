# Why I built Mandate-1

**Sivasubramanian Ramanathan (Siva)** · Independent consultant · Previously at the BIS Innovation Hub

I experiment across AI, tokenization and payments, and I am open to opportunities where that mix is useful. This is an independent portfolio project—not work attributed to or endorsed by a former employer.

## The question

The System-1 idea behind Jev made me ask: could a fast, bounded model do useful work in a framework like SAFR?

Not replace the rules. Not decide whether someone is laundering money. Something narrower: look at an agent's proposed action, its instruction and its evidence, and spot a semantic exception before code decides whether the action may proceed.

There is a difference between **dangerous words** and **a dangerous action**. A record might quote “ignore the instructions” without the agent following it. Conversely, an innocent-looking request can be wrong because it targets the wrong entity or conflicts with the analyst's instruction. Matching keywords alone does not resolve those relationships.

## What I tried

I ran custom Needle 3 LoRA fine-tuning pilots and smaller GLiNER 2.5 fine-tuning experiments. They exposed issues with generalisation, formulation and evaluation. I then adapted Laya's typed-decision model and used teacher-diversified synthetic contrasts. The student improved, but its false clears remain too high to trust for releasing actions.

The clearest proof came from a larger model: DeepSeek Flash with thinking disabled and a constrained JSON interface. It returned typed semantic findings; deterministic code checked authority, applied the disposition, wrote the decision record and released only an allowed synthetic read.

That end-to-end path worked on the authored demonstration cases. We can now show useful work completing, a contradiction being held, and failures that cannot turn into permission. The [experiment ledger](EXPERIMENTS.md) and original receipts show both the successful integration and the unsuccessful compact-model attempts.

## Why this approach is interesting

Compared with a keyword-only filter, a semantic sensor can examine the relationship between an instruction, a proposal and a trace. Compared with asking an unconstrained LLM to make the whole decision, a small typed interface is easier to validate and leaves authority in deterministic code. The current demonstrations illustrate those distinctions; they do not prove superiority over every rules engine or safety system.

A sufficiently capable compact local model could eventually offer privacy, offline operation and lower latency or cost. Those are hypotheses to measure, not benefits established by this release. Laya and other local-model approaches—including future Qwen-derived candidates—are worth testing; this work does not claim they share the same architecture or that untested models already meet the bar.

## What the portfolio demonstrates

- Turning a policy/framework idea into executable contracts and an observable workflow.
- Building contrast data, checking provenance and separating model selection from final evaluation.
- Training and diagnosing small models without concealing failed results.
- Integrating a hosted model behind a bounded, fail-closed execution path.
- Shipping an inspectable demo, dataset, model card, source and reproducibility evidence.

I am looking for applied AI, fintech, payments and tokenization opportunities that value this kind of hands-on experimentation and careful engineering.

**Explore:** https://github.com/siva-sub/mandate-1
