#!/usr/bin/env python3
"""One shadow-only Laya classification. Never executes a proposed action."""
import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aml_nextstep.safr_gliner import FINDING_DESCRIPTIONS, FINDING_INSTRUCTION, FINDING_TASK, render_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='sivasub987/mandate-1-laya')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--local-model', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--input', type=Path, default=Path('data/safr-semantic-firewall-v0.1-teacher-augmented/dataset/test.jsonl'))
    args = parser.parse_args()
    if args.local_model:
        model = args.local_model
    else:
        from huggingface_hub import snapshot_download
        model = Path(snapshot_download(args.repo, revision=args.revision))
    laya = importlib.import_module('laya')
    row = json.loads(args.input.read_text().splitlines()[0])
    agent = laya.Agent(str(model), device=args.device)
    result = agent.predict(render_envelope(row), {FINDING_TASK: {
        'type': 'choice', 'instructions': FINDING_INSTRUCTION, 'criteria': FINDING_DESCRIPTIONS,
    }})
    print(json.dumps({'shadow_only': True, 'execution_permitted': False,
                      'example_id': row['example_id'], 'prediction': result}, indent=2, default=str))


if __name__ == '__main__':
    main()
