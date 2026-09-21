#!/usr/bin/env python3
"""Build and validate the SAFR semantic-firewall contrast-set release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aml_nextstep.safr_firewall_data import (
    DEFAULT_SEED,
    SPLIT_WORLD_COUNTS,
    validate_firewall_release,
    write_firewall_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/safr-semantic-firewall-v0.1"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exception-training-permutations", type=int, default=6)
    parser.add_argument("--clean-training-permutations", type=int, default=1)
    args = parser.parse_args()

    manifest = write_firewall_release(
        args.output,
        seed=args.seed,
        split_world_counts=SPLIT_WORLD_COUNTS,
        exception_training_permutations=args.exception_training_permutations,
        clean_training_permutations=args.clean_training_permutations,
    )
    validation = validate_firewall_release(args.output)
    if not validation["ok"]:
        raise SystemExit(json.dumps(validation, indent=2))
    print(
        json.dumps(
            {
                "status": "validated",
                "output": str(args.output),
                "example_count": manifest["example_count"],
                "contrast_group_count": manifest["contrast_group_count"],
                "needle_examples": manifest["needle_examples"],
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
