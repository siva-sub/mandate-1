#!/usr/bin/env python3
"""Embed a live harness receipt in a self-contained, offline HTML replay."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=Path('research-evidence/safr-live-harness-v1/receipt.json'))
    parser.add_argument('--template', type=Path, default=Path('demo/safr-harness.template.html'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/safr-harness-demo.html'))
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text())
    if data.get('schema') != 'safr-live-harness-v1' or not data.get('results'):
        raise ValueError('invalid live harness receipt')
    template = args.template.read_text()
    marker = '__SAFR_HARNESS_DATA__'
    if template.count(marker) != 1:
        raise ValueError('template must contain exactly one data marker')
    html = template.replace(marker, json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'cases': len(data['results'])}))


if __name__ == '__main__':
    main()
