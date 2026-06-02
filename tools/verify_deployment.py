from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_ENV_KEYS = (
    "RUBIC_FRONTEND_IMAGE",
    "RUBIC_BACKEND_IMAGE",
    "FRONTEND_PORT",
    "BACKEND_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "RUBIC_SESSION_STORAGE_BACKEND",
    "RUBIC_RL_MODEL_FILE",
    "RUBIC_RL_POLICY_TYPE",
    "RUBIC_RL_POLICY_DEVICE",
    "RUBIC_RL_MAX_STEPS",
    "RUBIC_RL_SEARCH_WIDTH",
    "RUBIC_RL_SEARCH_TOP_K",
    "RUBIC_RL_SEARCH_TRACE_LIMIT",
)

OPTIONAL_PORT_KEYS = ("PROMETHEUS_PORT", "GRAFANA_PORT")
POSITIVE_INT_KEYS = (
    "RUBIC_RL_MAX_STEPS",
    "RUBIC_RL_SEARCH_WIDTH",
    "RUBIC_RL_SEARCH_TOP_K",
    "RUBIC_RL_SEARCH_TRACE_LIMIT",
)
POLICY_TYPES = {"auto", "linear", "mlp", "torch"}
POLICY_DEVICES = {"cpu", "cuda"}
SESSION_BACKENDS = {"auto", "files", "postgres"}
PLACEHOLDER_PARTS = (
    "replace_me",
    "replace_with",
    "changeme",
    "example",
    "owner/",
    "/repo/",
    "owner/repo",
)


@dataclass(frozen=True)
class DeploymentReport:
    schema_version: str
    env_file: str
    compose_file: str
    strict: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "env_file": self.env_file,
            "compose_file": self.compose_file,
            "strict": self.strict,
            "status": "failed" if self.errors else "passed",
            "errors": self.errors,
            "warnings": self.warnings,
        }


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} is not KEY=VALUE syntax")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"{path}:{line_number} has an empty key")
        values[key] = value
    return values


def _is_placeholder(value: str) -> bool:
    normalized = value.lower()
    return any(part in normalized for part in PLACEHOLDER_PARTS)


def _has_pinned_tag(image: str) -> bool:
    if "@" in image:
        return True
    last_segment = image.rsplit("/", 1)[-1]
    return ":" in last_segment


def _is_latest_tag(image: str) -> bool:
    return image.lower().endswith(":latest")


def _validate_port(key: str, value: str, errors: list[str]) -> None:
    try:
        port = int(value)
    except ValueError:
        errors.append(f"{key} must be an integer port, got {value!r}")
        return
    if not 1 <= port <= 65535:
        errors.append(f"{key} must be between 1 and 65535, got {port}")


def _validate_positive_int(key: str, value: str, errors: list[str]) -> None:
    try:
        number = int(value)
    except ValueError:
        errors.append(f"{key} must be an integer, got {value!r}")
        return
    if number < 1:
        errors.append(f"{key} must be at least 1, got {number}")


def validate_deployment(
    *,
    env_file: Path,
    compose_file: Path,
    repo_root: Path,
    strict: bool,
    require_model_file: bool,
) -> DeploymentReport:
    errors: list[str] = []
    warnings: list[str] = []

    if not env_file.exists():
        errors.append(f"env file does not exist: {env_file}")
        env_values: dict[str, str] = {}
    else:
        try:
            env_values = parse_env_file(env_file)
        except ValueError as exc:
            errors.append(str(exc))
            env_values = {}

    if not compose_file.exists():
        errors.append(f"compose file does not exist: {compose_file}")

    for key in REQUIRED_ENV_KEYS:
        if not env_values.get(key):
            errors.append(f"{key} is required")

    for key in ("FRONTEND_PORT", "BACKEND_PORT", *OPTIONAL_PORT_KEYS):
        value = env_values.get(key)
        if value:
            _validate_port(key, value, errors)

    frontend_port = env_values.get("FRONTEND_PORT")
    backend_port = env_values.get("BACKEND_PORT")
    if frontend_port and backend_port and frontend_port == backend_port:
        errors.append("FRONTEND_PORT and BACKEND_PORT must be different")

    for key in POSITIVE_INT_KEYS:
        value = env_values.get(key)
        if value:
            _validate_positive_int(key, value, errors)

    policy_type = env_values.get("RUBIC_RL_POLICY_TYPE")
    if policy_type and policy_type not in POLICY_TYPES:
        errors.append(f"RUBIC_RL_POLICY_TYPE must be one of {sorted(POLICY_TYPES)}, got {policy_type!r}")

    policy_device = env_values.get("RUBIC_RL_POLICY_DEVICE")
    if policy_device and policy_device not in POLICY_DEVICES:
        errors.append(f"RUBIC_RL_POLICY_DEVICE must be one of {sorted(POLICY_DEVICES)}, got {policy_device!r}")

    storage_backend = env_values.get("RUBIC_SESSION_STORAGE_BACKEND")
    if storage_backend and storage_backend not in SESSION_BACKENDS:
        errors.append(
            "RUBIC_SESSION_STORAGE_BACKEND must be one of "
            f"{sorted(SESSION_BACKENDS)}, got {storage_backend!r}"
        )

    for key in ("RUBIC_FRONTEND_IMAGE", "RUBIC_BACKEND_IMAGE"):
        image = env_values.get(key, "")
        if image and not _has_pinned_tag(image):
            errors.append(f"{key} must include an explicit tag or digest")
        if strict and image:
            if _is_placeholder(image):
                errors.append(f"{key} still contains an example image value: {image}")
            if _is_latest_tag(image):
                errors.append(f"{key} must use an immutable release tag or digest, not :latest")
            if image.endswith(":local"):
                errors.append(f"{key} must not point at a local-only image in strict mode")

    if strict:
        password = env_values.get("POSTGRES_PASSWORD", "")
        if len(password) < 20:
            errors.append("POSTGRES_PASSWORD must be at least 20 characters in strict mode")
        if _is_placeholder(password):
            errors.append("POSTGRES_PASSWORD still contains a placeholder value")

        grafana_password = env_values.get("GRAFANA_ADMIN_PASSWORD", "")
        if grafana_password:
            if grafana_password == "admin" or _is_placeholder(grafana_password):
                errors.append("GRAFANA_ADMIN_PASSWORD must be changed before production monitoring is exposed")
            if len(grafana_password) < 12:
                errors.append("GRAFANA_ADMIN_PASSWORD must be at least 12 characters when set")

    model_file = env_values.get("RUBIC_RL_MODEL_FILE")
    if require_model_file and model_file:
        model_path = repo_root / "checkpoints" / model_file
        if not model_path.exists():
            errors.append(f"configured checkpoint does not exist: {model_path}")
    elif model_file:
        model_path = repo_root / "checkpoints" / model_file
        if not model_path.exists():
            warnings.append(f"configured checkpoint was not found locally: {model_path}")

    for path in (
        repo_root / "backend" / "app" / "database" / "migrations",
        repo_root / "monitoring" / "prometheus",
        repo_root / "monitoring" / "grafana" / "provisioning",
        repo_root / "monitoring" / "grafana" / "dashboards",
    ):
        if not path.exists():
            errors.append(f"required deployment path is missing: {path}")

    return DeploymentReport(
        schema_version="rubic-rfl-deployment-readiness-v1",
        env_file=str(env_file),
        compose_file=str(compose_file),
        strict=strict,
        errors=errors,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Rubic RFL production deployment settings.")
    parser.add_argument("--env-file", default="deploy/production.env", help="Production env file to validate.")
    parser.add_argument(
        "--compose-file",
        default="deploy/compose.production.yml",
        help="Production Compose file to validate for existence.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on placeholders, :latest tags, and weak secrets.")
    parser.add_argument(
        "--require-model-file",
        action="store_true",
        help="Fail if RUBIC_RL_MODEL_FILE is not present under checkpoints/.",
    )
    parser.add_argument("--json", action="store_true", help="Write a JSON report to stdout.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    report = validate_deployment(
        env_file=(repo_root / args.env_file).resolve(),
        compose_file=(repo_root / args.compose_file).resolve(),
        repo_root=repo_root,
        strict=args.strict,
        require_model_file=args.require_model_file,
    )
    payload = report.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Deployment readiness: {payload['status']}")
        for error in report.errors:
            print(f"ERROR: {error}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")

    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
