# Deployment

This folder contains a baseline single-host deployment path for Phase 11.

## Publish Images

Run the `Publish Images` GitHub Actions workflow manually or push a `v*` tag.
It publishes:

- `ghcr.io/<owner>/<repo>/backend:<sha>`
- `ghcr.io/<owner>/<repo>/frontend:<sha>`
- `ghcr.io/<owner>/<repo>/backend:latest`
- `ghcr.io/<owner>/<repo>/frontend:latest`

Use the `<sha>` tags for production. The workflow also uploads a
`published-images` artifact containing the immutable backend and frontend image
references.

## VPS or VM Deployment

On the host, install Docker and Docker Compose, then generate the production env
file with immutable image tags and strong local secrets:

```bash
python tools/prepare_production_env.py --generate-secrets --force
```

Pass `--image-namespace ghcr.io/<owner>/<repo>` and `--image-tag <sha>` when
the host is not a Git checkout with the correct origin and published commit.
Edit `deploy/production.env` only for public ports, checkpoint settings, or
managed service connection details.

Validate release settings before starting the stack:

```bash
python tools/verify_deployment.py --env-file deploy/production.env --strict --require-model-file
docker compose -f deploy/compose.production.yml --env-file deploy/production.env config
docker compose -f deploy/compose.production.yml --env-file deploy/production.env --profile monitoring config
```

Start the stack:

```bash
docker compose -f deploy/compose.production.yml --env-file deploy/production.env up -d
```

Verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
curl "http://127.0.0.1:8000/solve/rl/status?load=true"
```

Add `--profile monitoring` to the compose command to run Prometheus and
Grafana from the production compose file. Prometheus scrapes the backend over
the private Compose network, and Grafana loads the checked-in Rubic RFL
dashboard automatically.

Use a TLS reverse proxy in front of the frontend and backend before exposing a
public production instance.
