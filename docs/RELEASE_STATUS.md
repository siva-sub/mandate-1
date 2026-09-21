# Release links and discoverability

Verified on 2026-09-22 (Asia/Singapore). These are live links, not placeholders. Availability checks are a point-in-time observation, not a guarantee against future platform changes.

| Resource | Public link |
|---|---|
| Interactive recorded demo | https://siva-sub.github.io/mandate-1/ |
| Source and portfolio | https://github.com/siva-sub/mandate-1 |
| Release downloads | https://github.com/siva-sub/mandate-1/releases/latest |
| 34-second explainer, 1080p | https://github.com/siva-sub/mandate-1/releases/download/v0.2.0/brag.mp4 |
| Full-resolution poster | https://github.com/siva-sub/mandate-1/releases/download/v0.2.0/brag.jpg |
| Hugging Face dataset | https://huggingface.co/datasets/sivasub987/mandate-1-safr-data |
| Hugging Face checkpoint | https://huggingface.co/sivasub987/mandate-1-laya |
| Kaggle dataset | https://www.kaggle.com/datasets/sivasub987/mandate-1-safr-data |
| Kaggle checkpoint | https://www.kaggle.com/models/sivasub987/mandate-1-laya |
| Pinned Kaggle variation | https://www.kaggle.com/models/sivasub987/mandate-1-laya/PyTorch/research-v02/1 |

## Verification

- Public project/demo/Hub/mirror pages returned HTTP 200. The Kaggle variation was also opened signed out: its model card, version 1, license, file explorer and download controls were visible.
- The Kaggle dataset and checkpoint were downloaded, and all distributed SHA-256 manifests were checked, including nested `data-bundle/` and `model-bundle/` paths.
- Model weights match `7d9106c9d2b30d3f66368bb20c1c831ebebcf920d5a31dc3b2d18d09a971cdd5`.
- GitHub Pages reports a successful build. Hugging Face reports both repositories as public and enabled.
- The MAS paper and upstream Laya model links resolve. Kenneth's announcement resolves; LinkedIn blocks automated HTTP checks of his profile (999), so this is not treated as a dead link. His profile and announcement were read using the LinkedIn CLI.

## Platform metadata

**GitHub:** `agentic-ai`, `ai-safety`, `aml-cft`, `deepseek`, `explainable-ai`, `financial-services`, `fintech`, `gliner`, `laya`, `model-evaluation`, `natural-language-processing`, `needle`, `portfolio`, `python`, `runtime-governance`, `safr`, `small-language-models`, `synthetic-data`, `system-1`, `text-classification`.

**Hugging Face:** native `text-classification` task metadata, English language, correct licenses, dataset/base-model relationships, and focused research tags including `safr`, `runtime-governance`, `agentic-ai`, `ai-safety`, `fintech`, `aml-cft`, `system-1` and `model-evaluation`. Model-specific tags include `laya`, `typed-decisions` and `small-language-model`. Hosted inference is explicitly disabled because the checkpoint requires the custom Laya loader.

**Kaggle dataset:** verified native categories `finance`, `nlp`, `classification`.

**Kaggle model:** native PyTorch variation, Apache 2.0 license, fine-tunable/external-base-model metadata and a searchable SAFR/agentic-finance/text-classification description. Kaggle's model CLI does not expose arbitrary model tags; do not confuse the README's Hugging Face YAML tags with applied Kaggle taxonomy tags.

These choices improve the metadata available for discovery; they do not promise a search ranking or immediate indexing.

## Release hygiene

Publication uses the explicit allowlist in `scripts/prepare_research_release.py`, not the working directory. Local `.pi`, `.agents`, `.ruff_cache`, `.pytest_cache`, `.env`, credentials, archives and agent/session configuration are excluded. A secret-value scan and checksum/archive checks run before uploads. Hugging Face creates its own standard `.gitattributes`; that is repository/LFS metadata, not leaked local configuration.

Historical experiment receipts remain unchanged. Research publication does not change the compact model's measured failures or approve it for deployment. LinkedIn launch copy remains a draft; no post, mention notification or message has been sent.
