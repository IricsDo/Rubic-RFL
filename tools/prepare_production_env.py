from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_deployment import parse_env_file, validate_deployment


DEFAULT_TEMPLATE = "deploy/production.env.example"
DEFAULT_OUTPUT = "deploy/production.env"


@dataclass(frozen=True)
class PreparedProductionEnv:
    schema_version: str
    output_file: str
    image_namespace: str
    image_tag: str
    dirty_worktree: bool
    warnings: list[str]
    validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_file": self.output_file,
            "image_namespace": self.image_namespace,
            "image_tag": self.image_tag,
            "dirty_worktree": self.dirty_worktree,
            "warnings": self.warnings,
            "validation": self.validation,
        }


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _infer_image_namespace(repo_root: Path) -> str:
    remote = _run_git(repo_root, "remote", "get-url", "origin")
    github_prefix = "https://github.com/"
    ssh_prefix = "git@github.com:"

    if remote.startswith(github_prefix):
        repo = remote[len(github_prefix) :]
    elif remote.startswith(ssh_prefix):
        repo = remote[len(ssh_prefix) :]
    else:
        raise ValueError(
            "Could not infer GHCR namespace from origin remote; pass --image-namespace explicitly."
        )

    if repo.endswith(".git"):
        repo = repo[:-4]
    if "/" not in repo:
        raise ValueError(
            "Could not infer GHCR namespace from origin remote; pass --image-namespace explicitly."
        )
    return f"ghcr.io/{repo.lower()}"


def _infer_image_tag(repo_root: Path) -> str:
    return _run_git(repo_root, "rev-parse", "--short=12", "HEAD")


def _is_dirty(repo_root: Path) -> bool:
    try:
        return bool(_run_git(repo_root, "status", "--porcelain"))
    except subprocess.CalledProcessError:
        return False


def _random_secret() -> str:
    return secrets.token_urlsafe(32)


def _rewrite_env_template(template_file: Path, output_file: Path, replacements: dict[str, str]) -> None:
    seen: set[str] = set()
    output_lines: list[str] = []

    for raw_line in template_file.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _value = stripped.split("=", 1)
            key = key.strip()
            if key in replacements:
                output_lines.append(f"{key}={replacements[key]}")
                seen.add(key)
                continue
        output_lines.append(raw_line)

    missing_keys = [key for key in replacements if key not in seen]
    if missing_keys:
        if output_lines and output_lines[-1]:
            output_lines.append("")
        output_lines.append("# Added by tools/prepare_production_env.py.")
        for key in missing_keys:
            output_lines.append(f"{key}={replacements[key]}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def prepare_production_env(
    *,
    repo_root: Path,
    template_file: Path,
    output_file: Path,
    image_namespace: str,
    image_tag: str,
    postgres_password: str,
    grafana_password: str,
    force: bool,
    require_model_file: bool,
) -> PreparedProductionEnv:
    if output_file.exists() and not force:
        raise FileExistsError(f"{output_file} already exists; pass --force to replace it")
    if not template_file.exists():
        raise FileNotFoundError(f"template file does not exist: {template_file}")

    template_values = parse_env_file(template_file)
    dirty_worktree = _is_dirty(repo_root)
    warnings: list[str] = []
    if dirty_worktree:
        warnings.append(
            "Git worktree has uncommitted changes; publish images from the committed revision before deploying."
        )

    replacements = {
        "RUBIC_FRONTEND_IMAGE": f"{image_namespace}/frontend:{image_tag}",
        "RUBIC_BACKEND_IMAGE": f"{image_namespace}/backend:{image_tag}",
        "POSTGRES_PASSWORD": postgres_password,
        "GRAFANA_ADMIN_PASSWORD": grafana_password,
    }

    for key in (
        "FRONTEND_PORT",
        "BACKEND_PORT",
        "PROMETHEUS_PORT",
        "GRAFANA_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "GRAFANA_ADMIN_USER",
        "GRAFANA_ANONYMOUS_ENABLED",
        "RUBIC_SESSION_STORAGE_BACKEND",
        "RUBIC_RL_MODEL_FILE",
        "RUBIC_RL_POLICY_TYPE",
        "RUBIC_RL_POLICY_DEVICE",
        "RUBIC_RL_MAX_STEPS",
        "RUBIC_RL_SEARCH_WIDTH",
        "RUBIC_RL_SEARCH_TOP_K",
        "RUBIC_RL_SEARCH_TRACE_LIMIT",
    ):
        value = template_values.get(key)
        if value is not None:
            replacements[key] = value

    _rewrite_env_template(template_file, output_file, replacements)
    validation = validate_deployment(
        env_file=output_file,
        compose_file=repo_root / "deploy" / "compose.production.yml",
        repo_root=repo_root,
        strict=True,
        require_model_file=require_model_file,
    ).to_dict()

    return PreparedProductionEnv(
        schema_version="rubic-rfl-production-env-v1",
        output_file=str(output_file),
        image_namespace=image_namespace,
        image_tag=image_tag,
        dirty_worktree=dirty_worktree,
        warnings=warnings,
        validation=validation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a strict-valid Rubic RFL production env file.")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="Template env file to copy from.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Output env file. Defaults to deploy/production.env.")
    parser.add_argument("--image-namespace", help="Image namespace, for example ghcr.io/owner/rubic-rfl.")
    parser.add_argument("--image-tag", help="Immutable image tag, usually the 12-character commit SHA.")
    parser.add_argument("--postgres-password", help="Production PostgreSQL password.")
    parser.add_argument("--grafana-password", help="Production Grafana admin password.")
    parser.add_argument(
        "--generate-secrets",
        action="store_true",
        help="Generate random PostgreSQL and Grafana passwords when explicit values are omitted.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists.")
    parser.add_argument(
        "--skip-model-file-check",
        action="store_true",
        help="Do not require RUBIC_RL_MODEL_FILE to exist under checkpoints/ during validation.",
    )
    parser.add_argument("--json", action="store_true", help="Write JSON output.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    try:
        image_namespace = (args.image_namespace or _infer_image_namespace(repo_root)).rstrip("/")
        image_tag = args.image_tag or _infer_image_tag(repo_root)

        postgres_password = args.postgres_password
        grafana_password = args.grafana_password
        if args.generate_secrets:
            postgres_password = postgres_password or _random_secret()
            grafana_password = grafana_password or _random_secret()
        if not postgres_password or not grafana_password:
            raise ValueError(
                "Pass --postgres-password and --grafana-password, or use --generate-secrets."
            )

        result = prepare_production_env(
            repo_root=repo_root,
            template_file=(repo_root / args.template).resolve(),
            output_file=(repo_root / args.out).resolve(),
            image_namespace=image_namespace,
            image_tag=image_tag,
            postgres_password=postgres_password,
            grafana_password=grafana_password,
            force=args.force,
            require_model_file=not args.skip_model_file_check,
        )
    except (subprocess.CalledProcessError, OSError, ValueError, FileExistsError) as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = result.to_dict()
    status = result.validation["status"]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Prepared production env: {status}")
        print(f"Output: {result.output_file}")
        print(f"Frontend image: {image_namespace}/frontend:{image_tag}")
        print(f"Backend image: {image_namespace}/backend:{image_tag}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        for error in result.validation["errors"]:
            print(f"ERROR: {error}")
    return 1 if status != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
