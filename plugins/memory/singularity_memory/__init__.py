"""Singularity Memory plugin — MemoryProvider backed by Singularity Memory server.

Postgres-backed memory with BM25 + vector + RRF fusion retrieval and optional
cross-encoder reranking. One server instance is shared across Hermes, OpenClaw,
Claude Code, and any MCP-aware client — memories persist and move with the user.

Config ($HERMES_HOME/singularity-memory.json or env vars):
  SINGULARITY_SERVER_URL  — server base URL (default: http://localhost:8888)
  SINGULARITY_API_KEY     — API key if server requires auth (optional)
  SINGULARITY_BANK_ID     — bank ID override (default: hermes or hermes.<profile>)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_CONFIG_FILE = "singularity-memory.json"
_DEFAULT_URL = "http://localhost:8888"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "sm_recall",
    "description": (
        "Search long-term memory using BM25 + vector + RRF fusion retrieval. "
        "Use to recall past conversations, facts, decisions, or context shared "
        "before. Prefer this over guessing what the user has previously said. "
        'Triggered by phrases like "Remember when..." or "What did we decide about..."'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "budget": {
                "type": "string",
                "enum": ["low", "mid", "high"],
                "description": "Result size — low (~1k tokens), mid (~3k, default), high (~6k).",
            },
            "rerank": {
                "type": "boolean",
                "description": "Cross-encoder reranking for higher precision (slower). Default: false.",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "sm_remember",
    "description": (
        "Explicitly persist a fact, decision, preference, or insight to long-term memory. "
        "Use for things that should survive beyond this session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or insight to store."},
            "context": {
                "type": "string",
                "description": "Optional context label (e.g. 'infra', 'preferences', 'decision').",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "sm_forget",
    "description": "Delete a specific memory by ID (IDs are returned by sm_recall results).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class _APIError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        super().__init__(f"HTTP {status}: {body[:200]}")


def _request(method: str, url: str, body: Any = None, api_key: str = "") -> Any:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise _APIError(e.code, e.read().decode(errors="replace"))
    except urllib.error.URLError as e:
        raise _APIError(0, str(e.reason))


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class SingularityMemoryProvider(MemoryProvider):

    def __init__(self):
        self._server_url = _DEFAULT_URL
        self._api_key = ""
        self._bank_id = "hermes"
        self._session_id = ""
        self._agent_context = "primary"

        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._retain_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "singularity_memory"

    # -- Availability --------------------------------------------------------

    def is_available(self) -> bool:
        cfg = self._load_config()
        url = os.environ.get("SINGULARITY_SERVER_URL") or cfg.get("server_url") or _DEFAULT_URL
        return bool(url)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = self._load_config()

        self._server_url = (
            os.environ.get("SINGULARITY_SERVER_URL")
            or cfg.get("server_url")
            or _DEFAULT_URL
        ).rstrip("/")

        self._api_key = os.environ.get("SINGULARITY_API_KEY") or cfg.get("api_key") or ""

        agent_identity = kwargs.get("agent_identity") or kwargs.get("agent_workspace") or ""
        self._bank_id = (
            os.environ.get("SINGULARITY_BANK_ID")
            or cfg.get("bank_id")
            or (f"hermes.{agent_identity}" if agent_identity else "hermes")
        )

        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")

        try:
            _request("PUT", self._bank_url(), body={}, api_key=self._api_key)
        except Exception as e:
            logger.warning("singularity_memory: bank init failed (server may not be up): %s", e)

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._retain_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- System prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            f"Long-term memory is active via Singularity Memory (bank: {self._bank_id}). "
            "Use `sm_recall` before answering questions about past conversations, decisions, "
            "or user preferences. Use `sm_remember` to store anything worth keeping beyond "
            "this session. Memory IDs in recall results can be passed to `sm_forget`."
        )

    # -- Prefetch ------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            try:
                result = self._do_recall(query, budget="low", rerank=False)
                with self._prefetch_lock:
                    self._prefetch_result = result
            except Exception as e:
                logger.debug("singularity_memory: prefetch failed: %s", e)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="sm-prefetch")
        self._prefetch_thread.start()

    # -- Turn sync -----------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._agent_context != "primary":
            return

        items = []
        if user_content.strip():
            items.append({
                "content": user_content,
                "context": "user",
                "document_id": f"turn:{self._session_id}:u",
            })
        if assistant_content.strip():
            items.append({
                "content": assistant_content,
                "context": "assistant",
                "document_id": f"turn:{self._session_id}:a",
            })
        if not items:
            return

        def _run():
            try:
                _request(
                    "POST",
                    self._bank_url("/memories"),
                    body={"items": items, "async": True},
                    api_key=self._api_key,
                )
            except Exception as e:
                logger.debug("singularity_memory: sync_turn failed: %s", e)

        if self._retain_thread and self._retain_thread.is_alive():
            self._retain_thread.join(timeout=2.0)
        self._retain_thread = threading.Thread(target=_run, daemon=True, name="sm-retain")
        self._retain_thread.start()

    # -- Hooks ---------------------------------------------------------------

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        self._session_id = new_session_id
        if reset:
            with self._prefetch_lock:
                self._prefetch_result = ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not messages or self._agent_context != "primary":
            return
        last_assistant = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"),
            "",
        )
        if not last_assistant or len(last_assistant) < 50:
            return
        try:
            _request(
                "POST",
                self._bank_url("/memories"),
                body={
                    "items": [{
                        "content": last_assistant[:2000],
                        "context": "session_end",
                        "document_id": f"session_end:{self._session_id}",
                        "tags": ["session-end"],
                    }],
                    "async": False,
                },
                api_key=self._api_key,
            )
        except Exception as e:
            logger.warning("singularity_memory: on_session_end retain failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if action not in ("add", "replace") or self._agent_context != "primary":
            return

        def _run():
            try:
                _request(
                    "POST",
                    self._bank_url("/memories"),
                    body={
                        "items": [{
                            "content": content,
                            "context": f"builtin:{target}",
                            "tags": ["builtin-memory"],
                        }],
                        "async": True,
                    },
                    api_key=self._api_key,
                )
            except Exception as e:
                logger.debug("singularity_memory: on_memory_write failed: %s", e)

        threading.Thread(target=_run, daemon=True, name="sm-mirror").start()

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "sm_recall":
            return self._tool_recall(args)
        if tool_name == "sm_remember":
            return self._tool_remember(args)
        if tool_name == "sm_forget":
            return self._tool_forget(args)
        raise NotImplementedError(tool_name)

    def _do_recall(self, query: str, budget: str = "mid", rerank: bool = False) -> str:
        resp = _request(
            "POST",
            self._bank_url("/memories/recall"),
            body={"query": query, "budget": budget, "max_tokens": 3000, "rerank": rerank},
            api_key=self._api_key,
        )
        results = resp.get("results", [])
        if not results:
            return ""
        lines = []
        for r in results:
            mid = r.get("id", "")[:8]
            text = r.get("text", "")
            ctx = r.get("context", "")
            prefix = f"[{ctx}] " if ctx else ""
            lines.append(f"• [{mid}] {prefix}{text}")
        return "\n".join(lines)

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("query is required")
        try:
            result = self._do_recall(
                query,
                budget=args.get("budget", "mid"),
                rerank=bool(args.get("rerank", False)),
            )
            return result or "No memories found for that query."
        except _APIError as e:
            logger.warning("sm_recall failed: %s", e)
            return tool_error(str(e))
        except Exception as e:
            logger.warning("sm_recall failed: %s", e)
            return tool_error(str(e))

    def _tool_remember(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("content is required")
        context = args.get("context", "explicit") or "explicit"
        try:
            _request(
                "POST",
                self._bank_url("/memories"),
                body={
                    "items": [{"content": content, "context": context, "tags": ["explicit"]}],
                    "async": False,
                },
                api_key=self._api_key,
            )
            return json.dumps({"status": "remembered", "preview": content[:120]})
        except _APIError as e:
            return tool_error(str(e))
        except Exception as e:
            return tool_error(str(e))

    def _tool_forget(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return tool_error("memory_id is required")
        try:
            _request("DELETE", self._bank_url(f"/memories/{memory_id}"), api_key=self._api_key)
            return json.dumps({"status": "deleted", "memory_id": memory_id})
        except _APIError as e:
            return tool_error(str(e))
        except Exception as e:
            return tool_error(str(e))

    # -- Config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "server_url",
                "description": "Singularity Memory server URL",
                "default": _DEFAULT_URL,
                "required": False,
            },
            {
                "key": "api_key",
                "description": "API key (if server requires auth)",
                "secret": True,
                "required": False,
                "env_var": "SINGULARITY_API_KEY",
            },
            {
                "key": "bank_id",
                "description": "Memory bank ID (default: hermes or hermes.<profile>)",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / _CONFIG_FILE
        path.write_text(json.dumps(values, indent=2))

    # -- Helpers -------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        try:
            path = get_hermes_home() / _CONFIG_FILE
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
        return {}

    def _bank_url(self, suffix: str = "") -> str:
        return f"{self._server_url}/v1/default/banks/{self._bank_id}{suffix}"


def register(ctx) -> None:
    ctx.register_memory_provider(SingularityMemoryProvider())
