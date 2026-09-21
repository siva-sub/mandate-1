# Mandate-1 launch copy

## LinkedIn — draft for Siva

I read Kenneth See and the team's SAFR paper—Safeguards for Agentic Finance at Runtime—and thought: hey, why not try building something inspired by this?

Jev is getting plenty of attention. For me, it brings a ChatGPT-like “aha” moment for models that choose from a set of answers, without needing fresh training for every task. That makes me optimistic about how quickly this area could improve.

But what could we use that idea for?

SAFR puts a checkpoint between an agent's decision and its action. I wondered whether a small, fast model could help with one part of that check: does the request actually fit the instructions and evidence?

Having permission doesn't mean the agent understood the task.

“Get the agreement” sounds harmless—until the evidence points to the wrong company. “Ignore the instructions” sounds dangerous—but it might just be a quote inside a document. The words alone don't tell the story.

My experiment, Mandate-1, follows a simple flow:

Agent asks → check permission → check the meaning → rules decide → record and act, or ask a person.

I tried custom fine-tunes of Needle 3 and GLiNER 2.5, then Laya, an open model that can also be fine-tuned. The smaller-model experiments taught me a lot, including where they still fall short.

The clearest working demonstration came from DeepSeek Flash with thinking off and a tightly limited set of answers. In the demo, an authorised read completes; a request based on the wrong company's record is held. The model helps check the request. It does not give itself permission.

DeepSeek publishes the Flash model weights too, so running it on your own suitable hardware is an option. My experiment used the hosted API—not a self-hosted deployment.

For me, this is an encouraging proof of concept. As small models improve, could we bring this kind of check closer to the work itself? That's what I'd like to keep exploring.

Thanks to Kenneth See and the SAFR team for sharing the paper. I'd love to hear your thoughts on this attempt.

Try it: https://siva-sub.github.io/mandate-1/
Code, experiments, data and model: https://github.com/siva-sub/mandate-1

I'm Sivasubramanian Ramanathan (Siva), previously at the BIS Innovation Hub and now an independent consultant. I experiment across AI, tokenization and payments, and I'm open to my next role. If you're building at those intersections, let's talk.

#AgenticAI #SAFR #FinTech #AppliedAI #OpenSource

## Attribution and factual notes — not part of the post

- Mention **Kenneth See, FCCA** using LinkedIn's mention picker: https://www.linkedin.com/in/kennethseedehui/
- His announcement: https://www.linkedin.com/feed/update/urn:li:activity:7478661943331823616/
- SAFR source: https://www.mas.gov.sg/-/media/mas-media-library/development/fintech/ai-safr/safr.pdf — also inspected from the owner's `/tmp/mas-safr.pdf`, physical PDF pages 8–12. Its components are Agent Identity, Controls Repository, Disposition Engine and Audit Log, connected by the Governance Envelope. The video simplifies that vocabulary; the source architecture document gives the detailed mapping.
- Jev source: https://typesafe.ai/blog/introducing-system-one-models-and-jev — the ChatGPT comparison and optimism are Siva's personal interpretation, not a benchmark result or a claim that Jev is a language model clone. “Without fresh training for every task” explains the zero-shot idea without requiring readers to know the term.
- DeepSeek's official announcement maps `deepseek-flash` to DeepSeek-V4.1-Flash: https://api-docs.deepseek.com/news/news260910
- Official weights and MIT license: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash — free-to-download weights do not mean free inference, a free hosted API, or suitability for an ordinary laptop. The published model is large and needs appropriate hardware and inference software. The experiment used the hosted alias; no local DeepSeek deployment was tested.
- The project adapts Laya; it does not claim Jev, Laya, Needle and GLiNER share an architecture. Fine-tuning receipts and failures remain in the linked experiment ledger.
- Credit Kenneth and the team without implying endorsement, official SAFR implementation or a financial compliance certification.
- No LinkedIn post, tag notification, comment or message has been sent. Detailed limitations remain in the cards rather than overwhelming this exploratory launch story.

## Short caption

I read the SAFR paper and thought: hey, why not try building something inspired by this? Jev's System-1 idea suggested a quick check before an agent acts—and Mandate-1 is my attempt to explore it.

Try it: https://siva-sub.github.io/mandate-1/

## Video direction

A plain-language, roughly 30-second /brag explainer with brighter original rhythmic music and no narration. Start with reading the SAFR paper and spell out its full name. Show where a quick model check fits before an agent acts, why permission and keywords alone are not enough, then Siva's proof of concept. Model names and training detail stay brief on screen; this post carries the fuller account. The current modular preview is 34 seconds. Render only after revised-preview approval.
