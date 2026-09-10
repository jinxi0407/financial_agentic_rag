# Deployment Guide

This document describes a safe, portable deployment shape. Replace all placeholders with environment-specific values outside Git.

## Topology

```text
Internet
  -> Cloudflare / HTTPS domain
  -> React / Vite frontend
  -> FastAPI application
  -> GPU-backed RAG and Agent runtime
  -> Milvus / Redis / Redis Stack / MySQL
```

| Role | Technology | Configuration boundary |
|---|---|---|
| Frontend | React / Vite | `VITE_API_BASE_URL`, `VITE_GITHUB_URL` |
| API | FastAPI | `HOST`, `PORT`, allowed origins |
| RAG / Agent | Python, BGE-M3, reranker, Qwen-compatible LLM | local model paths and LLM settings |
| FAQ | MySQL + BM25 | `MYSQL_*` |
| Memory | Redis and Redis Stack | `REDIS_*`, `AGENT_REDIS_*` |
| Retrieval | Milvus | `MILVUS_*` |

Use placeholders such as `<GPU_HOST>`, `<SSH_PORT>`, `<DOMAIN>` and `<API_DOMAIN>` in operational runbooks. Never commit a real endpoint, credential, tunnel token, key file or credential JSON path.

## Configuration

```bash
cp .env.example .env
cp config.example.ini config.ini
cp frontend/.env.example frontend/.env
```

Keep `.env`, `.env.*` and `config.ini` local. Configure the following categories only in environment-specific secret management or local files:

- LLM endpoint, model and API key
- MySQL / Redis / Milvus endpoints and credentials
- BGE-M3 and reranker model paths
- Agent Redis Stack endpoint and namespace
- frontend API base URL and optional GitHub URL

The canonical frozen retrieval settings are `RETRIEVAL_K=30` and `CANDIDATE_M=3`.

## Startup Order

1. Start MySQL and verify the FAQ database is reachable.
2. Start Redis for FAQ/cache and Redis Stack for Agent checkpoints; keep their DBs or prefixes isolated.
3. Start Milvus and verify the configured database and collection are available.
4. Make BGE-M3 and reranker model paths available to the GPU runtime.
5. Start FastAPI.
6. Build and serve the frontend, configured to reach `<API_DOMAIN>`.
7. Route HTTPS traffic through Cloudflare or an equivalent reverse-proxy/tunnel layer.

## Health Check

```bash
curl -fsS https://<API_DOMAIN>/health
```

Validate the Agent streaming endpoint separately with a non-sensitive request. Confirm the frontend reports connection failures rather than silently displaying fabricated market or news data.

## Restart and Shutdown

1. Stop new frontend traffic or enable maintenance mode at the proxy layer.
2. Stop the FastAPI process gracefully so in-flight WebSocket requests can close.
3. Leave Milvus, MySQL and Redis data volumes intact unless an explicit data-maintenance procedure is being performed.
4. Restart dependencies first, then FastAPI, then frontend/proxy routing.
5. Re-run `/health` and a small smoke query after restart.

## GPU Replacement

1. Provision a compatible GPU host as `<GPU_HOST>` with the required Python environment and model files.
2. Restore only non-secret configuration through the deployment secret mechanism.
3. Point the replacement runtime to existing or restored Milvus, MySQL and Redis services.
4. Verify model-path access, Milvus connectivity and Agent Redis checkpoint setup.
5. Run a small health/smoke check before moving `<DOMAIN>` traffic.

Do not copy `.env`, SSH keys, Cloudflare credentials, database passwords or Redis passwords into the repository during replacement.
