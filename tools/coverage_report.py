"""Run backend and RL tests under coverage and write CI-friendly reports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
RCFILE = ROOT / ".coveragerc"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def coverage_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "coverage", *args]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate combined coverage reports for backend and RL Python code."
    )
    parser.add_argument(
        "--out-dir",
        default="reports",
        help="Directory for coverage.xml and coverage.json. Defaults to reports.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Optional minimum total coverage percentage.",
    )
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_file = out_dir / f".coverage.{os.getpid()}.{uuid4().hex}"

    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(coverage_file)

    try:
        run(coverage_command("--version"), cwd=ROOT, env=env)
    except subprocess.CalledProcessError as exc:
        print(
            "coverage is not installed. Install dev dependencies with "
            'python -m pip install -e "./backend[dev]" -e "./rl[dev]".',
            file=sys.stderr,
        )
        return exc.returncode

    run(coverage_command("erase", "--rcfile", str(RCFILE)), cwd=ROOT, env=env)

    run(
        coverage_command(
            "run",
            "--rcfile",
            str(RCFILE),
            "--source",
            str(ROOT / "backend" / "app"),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "backend/tests",
        ),
        cwd=ROOT,
        env=env,
    )

    run(
        coverage_command(
            "run",
            "--rcfile",
            str(RCFILE),
            "--append",
            "--source",
            str(ROOT / "rl" / "rubic_rl"),
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "tests",
        ),
        cwd=ROOT / "rl",
        env=env,
    )

    report_command = coverage_command("report", "--rcfile", str(RCFILE))
    if args.fail_under is not None:
        report_command.extend(["--fail-under", str(args.fail_under)])
    run(report_command, cwd=ROOT, env=env)

    run(
        coverage_command("xml", "--rcfile", str(RCFILE), "-o", str(out_dir / "coverage.xml")),
        cwd=ROOT,
        env=env,
    )
    run(
        coverage_command("json", "--rcfile", str(RCFILE), "-o", str(out_dir / "coverage.json")),
        cwd=ROOT,
        env=env,
    )

    print(f"Coverage reports written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
