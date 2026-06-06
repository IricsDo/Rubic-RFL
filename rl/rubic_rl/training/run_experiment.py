from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rubic_rl.experiments import load_experiment_spec, run_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a configured baseline, ADI, or DAVI experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to an RL experiment JSON config.",
    )
    args = parser.parse_args(argv)

    spec = load_experiment_spec(args.config)
    result = run_experiment(spec, spec_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
