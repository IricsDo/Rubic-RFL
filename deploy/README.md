# Deployment

This folder contains a baseline single-host deployment path for Phase 11.

## Publish Images

Run the `Publish Images` GitHub Actions workflow manually or push a `v*` tag.
It publishes:

- `ghcr.io/<owner>/<repo>/backend:<sha>`
- `ghcr.io/<owner>/<repo>/frontend:<sha>`
- `ghcr.io/<owner>/<repo>/backend:latest`
- `ghcr.io/<owner>/<repo>/frontend:latest`

## VPS or VM Deployment

On the host, install Docker and Docker Compose, then create the production env
file:

```bash
cp deploy/production.env.example deploy/production.env
```

Edit `deploy/production.env` with real image names, a strong PostgreSQL
password, and the ports exposed by your reverse proxy.

Start the stack:

```bash
docker compose -f deploy/compose.production.yml --env-file deploy/production.env up -d
```

Verify:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Use a TLS reverse proxy in front of the frontend and backend before exposing a
public production instance.
