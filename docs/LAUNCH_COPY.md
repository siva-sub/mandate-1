# Mandate-1 launch copy

## LinkedIn — draft for Siva

Jev playing Doom got me thinking about a rather different setting: financial services.

Could that “System 1” idea—fast, bounded model decisions—help with something like SAFR?

Reading Kenneth See and the team's Safeguards for Agentic Finance at Runtime paper, I thought: why not try it?

That became Mandate-1.

The question I wanted to explore was simple: can a model tell the difference between dangerous words and a dangerous action?

An internal document might quote “ignore the instructions” without the agent following it. Another request might look perfectly ordinary but point to the wrong company or contradict what the analyst asked. The relationship matters, not just the words.

I experimented with custom Needle 3 and GLiNER 2.5 fine-tunes, then adapted Laya's typed-decision model. Some attempts worked better than others; the experiments and results are there to inspect.

The clearest demonstration came from DeepSeek Flash with thinking off and a constrained output format. In the synthetic workflow, it could assess the proposal and evidence, while code handled authority and the final decision. You can watch an authorised record read complete—and see a contradictory request held before execution.

For me, that's an encouraging proof of concept. The larger-model path already shows what is possible. As compact local models improve, I'd love to explore how much of that semantic checking could move closer to the workflow itself.

Thanks to Kenneth See and the SAFR team for putting the framework out there. I'd be interested in your thoughts on this small experiment.

Demo: https://siva-sub.github.io/mandate-1/
Code, data and model links: https://github.com/siva-sub/mandate-1

I'm Sivasubramanian Ramanathan (Siva), previously at the BIS Innovation Hub and now an independent consultant. I enjoy experimenting across AI, tokenization and payments, and I'm exploring my next role. If you're building at those intersections, let's talk.

#AgenticAI #SAFR #FinTech #AppliedAI #OpenSource

## Attribution / tagging notes — not part of the post

- Select **Kenneth See, FCCA** in LinkedIn's mention picker: https://www.linkedin.com/in/kennethseedehui/
- His announcement: https://www.linkedin.com/feed/update/urn:li:activity:7478661943331823616/
- The announcement describes SAFR as a reference approach for governing agentic actions before execution. Thank Kenneth and the SAFR team collectively; do not invent individual authors or imply endorsement.
- This file is a draft. No LinkedIn post, comment, tag notification or message has been sent.
- Detailed model limitations and failure rates belong in the linked cards and experiment ledger, rather than overwhelming this exploratory launch story.

## Short caption

Jev's System-1 idea got me wondering: could fast, bounded model decisions help with SAFR? I tried Needle, GLiNER and Laya, then built a working DeepSeek Flash demonstration around a simple question: dangerous words—or a dangerous action?

Explore Mandate-1: https://github.com/siva-sub/mandate-1

## Video direction

Polished 20-second landscape /brag, no narration. Hook: “Dangerous words. Or a dangerous action?” Show the actual interface returning an authorised record, then holding a contradictory trace. Close on “An idea worth trying.”, Mandate-1, Siva's name and the repository URL. Persistent small qualifier: “Synthetic research demo”. Focus on curiosity and visible possibility—not production certification or a parade of metrics.
