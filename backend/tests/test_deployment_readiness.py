import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_production_env import prepare_production_env
from tools.verify_deployment import parse_env_file, validate_deployment


def _tmp_env_file() -> Path:
    directory = ROOT / ".tmp" / "deployment-readiness-tests"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid.uuid4().hex}.env"


def _write_env(path: Path, *, password: str, image_tag: str = "20260602abcdef") -> None:
    path.write_text(
        "\n".join(
            [
                f"RUBIC_FRONTEND_IMAGE=ghcr.io/acme/rubic-rfl/frontend:{image_tag}",
                f"RUBIC_BACKEND_IMAGE=ghcr.io/acme/rubic-rfl/backend:{image_tag}",
                "FRONTEND_PORT=80",
                "BACKEND_PORT=8000",
                "PROMETHEUS_PORT=9090",
                "GRAFANA_PORT=3000",
                "POSTGRES_DB=rubic_rfl",
                "POSTGRES_USER=rubic_rfl",
                f"POSTGRES_PASSWORD={password}",
                "GRAFANA_ADMIN_USER=admin",
                "GRAFANA_ADMIN_PASSWORD=another-long-random-password",
                "GRAFANA_ANONYMOUS_ENABLED=false",
                "RUBIC_SESSION_STORAGE_BACKEND=postgres",
                "RUBIC_RL_MODEL_FILE=mlp-baseline-depth-1-2-3.npz",
                "RUBIC_RL_POLICY_TYPE=auto",
                "RUBIC_RL_POLICY_DEVICE=cpu",
                "RUBIC_RL_MAX_STEPS=30",
                "RUBIC_RL_SEARCH_WIDTH=5",
                "RUBIC_RL_SEARCH_TOP_K=5",
                "RUBIC_RL_SEARCH_TRACE_LIMIT=200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_production_env_example_passes_schema_validation():
    report = validate_deployment(
        env_file=ROOT / "deploy" / "production.env.example",
        compose_file=ROOT / "deploy" / "compose.production.yml",
        repo_root=ROOT,
        strict=False,
        require_model_file=True,
    )

    assert report.errors == []


def test_strict_validation_rejects_placeholders_and_latest_tags():
    env_file = _tmp_env_file()
    _write_env(env_file, password="replace_with_a_long_random_password", image_tag="latest")
    report = validate_deployment(
        env_file=env_file,
        compose_file=ROOT / "deploy" / "compose.production.yml",
        repo_root=ROOT,
        strict=True,
        require_model_file=False,
    )

    assert any("POSTGRES_PASSWORD still contains a placeholder" in error for error in report.errors)
    assert any("RUBIC_FRONTEND_IMAGE must use an immutable release tag" in error for error in report.errors)
    assert any("RUBIC_BACKEND_IMAGE must use an immutable release tag" in error for error in report.errors)


def test_strict_validation_accepts_release_values():
    env_file = _tmp_env_file()
    _write_env(env_file, password="long-random-production-password")
    report = validate_deployment(
        env_file=env_file,
        compose_file=ROOT / "deploy" / "compose.production.yml",
        repo_root=ROOT,
        strict=True,
        require_model_file=True,
    )

    assert report.errors == []


def test_prepare_production_env_writes_strict_valid_env():
    output_file = _tmp_env_file()
    result = prepare_production_env(
        repo_root=ROOT,
        template_file=ROOT / "deploy" / "production.env.example",
        output_file=output_file,
        image_namespace="ghcr.io/acme/rubic-rfl",
        image_tag="20260602abcdef",
        postgres_password="long-random-production-password",
        grafana_password="another-long-random-password",
        force=False,
        require_model_file=True,
    )
    values = parse_env_file(output_file)

    assert result.validation["status"] == "passed"
    assert values["RUBIC_FRONTEND_IMAGE"] == "ghcr.io/acme/rubic-rfl/frontend:20260602abcdef"
    assert values["RUBIC_BACKEND_IMAGE"] == "ghcr.io/acme/rubic-rfl/backend:20260602abcdef"


def test_prepare_production_env_refuses_overwrite_without_force():
    output_file = _tmp_env_file()
    output_file.write_text("existing=true\n", encoding="utf-8")

    try:
        prepare_production_env(
            repo_root=ROOT,
            template_file=ROOT / "deploy" / "production.env.example",
            output_file=output_file,
            image_namespace="ghcr.io/acme/rubic-rfl",
            image_tag="20260602abcdef",
            postgres_password="long-random-production-password",
            grafana_password="another-long-random-password",
            force=False,
            require_model_file=False,
        )
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("prepare_production_env should refuse accidental overwrite")
