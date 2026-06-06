from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rubic_rl.experiments import run_experiment_sweep


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a sequence of tracked RL experiment configs and rank the results."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Sweep name used for the summary output directory.",
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        type=Path,
        help="Path to an RL experiment JSON config. Repeat for multiple configs.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit path for the sweep summary JSON.",
    )
    args = parser.parse_args(argv)

    result = run_experiment_sweep(
        args.config,
        sweep_name=args.name,
        summary_out=args.summary_out,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
