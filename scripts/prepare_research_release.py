#!/usr/bin/env python3
"""Prepare an allowlisted multi-platform research release. Never uploads."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'paraphrase_validation', 'dev', 'calibration', 'test', 'ood')
DATA = ROOT / 'data/safr-semantic-firewall-v0.1-teacher-augmented'
MODEL = ROOT / 'artifacts/mandate1-laya-distilled-calibration-fixed'


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Expected regular file: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def checksums(folder: Path) -> None:
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (folder/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(folder).as_posix()}\n' for p in files))


def bundle(source: Path, destination: Path) -> None:
    # Stable paths/timestamps. Stored, not recompressed: safetensors is already large binary data.
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as archive:
        for p in sorted(source.rglob('*')):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(source).as_posix(), (2026, 9, 22, 0, 0, 0))
                info.external_attr = 0o644 << 16
                with p.open('rb') as incoming, archive.open(info, 'w', force_zip64=True) as outgoing:
                    shutil.copyfileobj(incoming, outgoing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Use a new release directory; never reuse an upload staging folder')
    github, dataset, model, kgdata, kgmodel, kginstance = [out/p for p in (
        'github', 'huggingface/dataset', 'huggingface/model', 'kaggle/dataset', 'kaggle/model', 'kaggle/variation')]
    for folder in (github, dataset, model, kgdata, kgmodel, kginstance):
        folder.mkdir(parents=True)
    common = {
        'release': 'v0.2.0', 'research_publication_authorized': True,
        'deployment_approved': False, 'model_status': 'experimental-shadow-only',
        'code_model_license': 'Apache-2.0', 'dataset_license': 'CC-BY-4.0',
        'historical_receipts': 'Unchanged. Failed model-use gates are not overridden by publication permission.',
        'excluded': ['credentials', 'archive', 'agent skills', 'MAS PDF', 'raw generation batches', 'failed weights', 'unrelated experiments'],
        'upstream_model': 'convaiinnovations/laya-typed-decisions',
        'upstream_revision': 'f9ab0b228f0fc0f14d873dbc99038f135c2da1b2',
    }
    # Source surface is deliberately narrow; no recursive workspace copy.
    for directory, pattern in (('aml_nextstep', 'safr_*.py'), ('tests', 'test_safr_*.py'), ('scripts', '*.py')):
        for p in sorted((ROOT/directory).glob(pattern)):
            copy(p, github/p.relative_to(ROOT))
    for name in ('aml_nextstep/__init__.py', 'README.md', 'ARCHITECTURE.md', 'LICENSE', 'DATA_LICENSE.txt', 'NOTICE', 'requirements-test.txt',
                 'docs/DATASET_CARD.md', 'docs/MODEL_CARD.md', 'docs/REPRODUCIBILITY.md', 'docs/RELEASE_PLAN.md', 'docs/LAUNCH_COPY.md', 'docs/PORTFOLIO.md', 'docs/EXPERIMENTS.md',
                 'demo/safr-gate.template.html', 'demo/safr-harness.template.html', 'vendor/laya-0.3.4-py3-none-any.whl'):
        copy(ROOT/name, github/name)
    # Mirror ready-to-open demos without putting unrelated artifacts in Git.
    copy(ROOT/'artifacts/safr-harness-demo.html', github/'demo/index.html')
    copy(ROOT/'artifacts/safr-harness-demo.html', github/'docs/index.html')
    copy(ROOT/'artifacts/safr-gate-demo.html', github/'demo/contrast.html')
    for name in ('safr-deepseek-flash-teacher.json', 'safr-deepseek-flash-teacher-gates-v2.json', 'safr-firewall-cheap-baselines.json',
                 'safr-firewall-teacher-augmented-ngram.json', 'safr-laya-template-overfit.json',
                 'safr-live-harness-v1/receipt.json', 'safr-live-harness-v1/audit.jsonl',
                 'safr-laya-distilled-calibration-fixed/training-receipt.json', 'safr-laya-distilled-calibration-fixed/ddp-training-state.json'):
        copy(ROOT/'research-evidence'/name, github/'research-evidence'/name)
    for p in sorted((ROOT/'research-evidence/exploratory').rglob('*.json')):
        copy(p, github/p.relative_to(ROOT))
    copy(ROOT/'research-evidence/laya-cpu-smoke.json', github/'research-evidence/laya-cpu-smoke.json')
    for sub in ('dataset', 'gliner25'):
        for split in SPLITS:
            copy(DATA/sub/f'{split}.jsonl', dataset/sub/f'{split}.jsonl')
        manifest = DATA/sub/'manifest.json'
        if manifest.exists():
            copy(manifest, dataset/sub/'manifest.json')
    for name in ('manifest.json', 'generation-manifest.json', 'teacher-self-consistency.json'):
        copy(DATA/name, dataset/name)
    for p in sorted(dataset.rglob('*')):
        if p.is_file():
            copy(p, github/'data/safr-semantic-firewall-v0.1-teacher-augmented'/p.relative_to(dataset))
    base = ROOT/'data/safr-semantic-firewall-v0.1'
    for sub in ('dataset', 'gliner25'):
        for p in sorted((base/sub).glob('*.jsonl')):
            copy(p, github/'data/safr-semantic-firewall-v0.1'/sub/p.name)
    for name in ('manifest.json', 'gliner25/manifest.json'):
        if (base/name).exists():
            copy(base/name, github/'data/safr-semantic-firewall-v0.1'/name)
    copy(ROOT/'docs/DATASET_CARD.md', dataset/'README.md')
    copy(ROOT/'DATA_LICENSE.txt', dataset/'LICENSE')
    copy(ROOT/'NOTICE', dataset/'NOTICE')
    for name in ('model.safetensors', 'rl_agent_config.json', 'encoder/config.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'training-receipt.json', 'ddp-training-state.json'):
        copy(MODEL/name, model/name)
    expected = '7d9106c9d2b30d3f66368bb20c1c831ebebcf920d5a31dc3b2d18d09a971cdd5'
    if sha(model/'model.safetensors') != expected:
        raise ValueError('Wrong selected model weights')
    copy(ROOT/'docs/MODEL_CARD.md', model/'README.md')
    copy(ROOT/'docs/DATASET_CARD.md', model/'DATASET_CARD.md')
    for name in ('LICENSE', 'NOTICE'):
        copy(ROOT/name, model/name)
    for folder in (github, dataset, model):
        (folder/'release-manifest.json').write_text(json.dumps(common, indent=2)+'\n')
        checksums(folder)
    # Kaggle single archives preserve exact nested loader paths and data bytes.
    bundle(dataset, kgdata/'data-bundle.zip')
    copy(dataset/'README.md', kgdata/'README.md')
    copy(dataset/'LICENSE', kgdata/'LICENSE')
    bundle(model, kginstance/'model-bundle.zip')
    copy(model/'README.md', kginstance/'README.md')
    copy(model/'LICENSE', kginstance/'LICENSE')
    data_meta = {'id':'sivasub987/mandate-1-safr-data', 'title':'Mandate-1 SAFR Semantic Contrasts',
                 'subtitle':'Synthetic governance contrasts and transparent research provenance',
                 'description': (ROOT/'docs/DATASET_CARD.md').read_text().split('---',2)[-1].strip(),
                 'licenses':[{'name':'CC-BY-4.0'}], 'keywords':['finance','nlp','classification','deep-learning','artificial-intelligence']}
    (kgdata/'dataset-metadata.json').write_text(json.dumps(data_meta,indent=2)+'\n')
    model_meta = {'ownerSlug':'sivasub987','title':'Mandate-1 Laya','slug':'mandate-1-laya','isPrivate':False,
                  'subtitle':'Experimental shadow-only SAFR semantic classifier',
                  'description': (ROOT/'docs/MODEL_CARD.md').read_text().split('---',2)[-1].strip(),
                  'provenanceSources':'https://huggingface.co/convaiinnovations/laya-typed-decisions'}
    (kgmodel/'model-metadata.json').write_text(json.dumps(model_meta,indent=2)+'\n')
    instance = {'ownerSlug':'sivasub987','modelSlug':'mandate-1-laya','instanceSlug':'research-v02','framework':'pyTorch',
                'overview':'Teacher-distilled 421M Laya checkpoint. Synthetic research only; failed model-use gates disclosed.',
                'usage':model_meta['description'],'licenseName':'Apache 2.0','fineTunable':True,
                'trainingData':['https://www.kaggle.com/datasets/sivasub987/mandate-1-safr-data'],
                'externalBaseModelUrl':'https://huggingface.co/convaiinnovations/laya-typed-decisions'}
    (kginstance/'model-instance-metadata.json').write_text(json.dumps(instance,indent=2)+'\n')
    for folder in (kgdata, kginstance):
        checksums(folder)
    print(json.dumps({'out':str(out), 'github_files':len(list(github.rglob('*'))), 'weights_sha256':expected,
                      'model_bundle_bytes':(kginstance/'model-bundle.zip').stat().st_size},indent=2))


if __name__ == '__main__':
    main()
