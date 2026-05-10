# Singularity Memory Provider

Postgres-backed memory with BM25 + vector + RRF fusion retrieval and optional
cross-encoder reranking. One server instance is shared across Hermes, OpenClaw,
and Claude Code — memories persist and move with the user, not the tool.

## Requirements

- Singularity Memory server running (see `singularity-memory` repo, `go/` directory)
- Postgres 18 + VectorChord (`vchord`) on the server side

## Setup

```bash
hermes memory setup  # select "singularity_memory"
```

Or manually:

```bash
hermes config set memory.provider singularity_memory
# add to $HERMES_HOME/singularity-memory.json:
echo '{"server_url": "http://singularity-memory.centralcloud-ops.svc:8888"}' \
  > ~/.hermes/singularity-memory.json
```

## Config

| Key | Env var | Default | Description |
|-----|---------|---------|-------------|
| `server_url` | `SINGULARITY_SERVER_URL` | `http://localhost:8888` | Server base URL |
| `api_key` | `SINGULARITY_API_KEY` | — | API key if server requires auth |
| `bank_id` | `SINGULARITY_BANK_ID` | `hermes` or `hermes.<profile>` | Memory bank ID |

Config file: `$HERMES_HOME/singularity-memory.json`

## Tools

| Tool | Description |
|------|-------------|
| `sm_recall` | BM25 + vector search with optional reranking |
| `sm_remember` | Explicitly persist a fact to long-term memory |
| `sm_forget` | Delete a specific memory by ID |

## Bank isolation

Each Hermes profile gets its own bank (`hermes.<profile>`). Set `SINGULARITY_BANK_ID`
to share a bank across profiles or with other tools.
