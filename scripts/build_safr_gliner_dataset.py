#!/usr/bin/env python3
"""Build GLiNER2.5 semantic-primitive data from a frozen SAFR firewall release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aml_nextstep.safr_gliner import write_gliner_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/safr-semantic-firewall-v0.1"),
    )
    parser.add_argument("--output-name", default="gliner25")
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name:
        raise SystemExit("--output-name must be one directory name")
    manifest = write_gliner_dataset(args.release, output_name=args.output_name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
