"""Offline self-evolution request handling for Hermes.

This module intentionally does not import DSPy, GEPA, or the external
``hermes-agent-self-evolution`` package. Hermes owns the request contract and
artifact layout; the optimizer runs out-of-process and returns reviewable git
diffs.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


VALID_TARGETS = ("skill", "tool-description", "prompt-section", "code")
VALID_EVAL_SOURCES = ("sessiondb", "synthetic", "golden", "auto")
DEFAULT_FORGEJO_REPO = "centralcloud/hermes-agent-self-evolution"
DEFAULT_FORGEJO_CLONE_URL = (
    "ssh://git@git.infra.centralcloud.com:2222/"
    "centralcloud/hermes-agent-self-evolution.git"
)
UPSTREAM_REPO_URL = "https://github.com/NousResearch/hermes-agent-self-evolution"


@dataclass(frozen=True)
class EvolutionRequest:
    """Machine-readable request for an offline Hermes optimization run."""

    request_id: str
    target: str
    name: str
    reason: str
    eval_source: str
    iterations: int
    hermes_agent_repo: str
    external_runner_repo: str
    external_runner_source_url: str
    forgejo_repo: str
    status: str
    created_at: int
    source: str = "hermes-cli"
    dry_run_only: bool = True


def evolution_home(base_dir: Path | None = None) -> Path:
    """Return the profile-local directory for self-evolution queue state."""

    return base_dir if base_dir is not None else get_hermes_home() / "evolution"


def default_external_runner_repo() -> str:
    """Resolve the external optimizer checkout path without requiring it."""

    return os.environ.get(
        "HERMES_SELF_EVOLUTION_REPO",
        str(get_hermes_home() / "hermes-agent-self-evolution"),
    )


def default_forgejo_repo() -> str:
    """Return the Forgejo repository used for offline optimizer work."""

    return os.environ.get("HERMES_EVOLUTION_FORGEJO_REPO", DEFAULT_FORGEJO_REPO)


def default_external_runner_source_url() -> str:
    """Return the internal source repository for the external optimizer."""

    return os.environ.get(
        "HERMES_SELF_EVOLUTION_REPO_URL",
        DEFAULT_FORGEJO_CLONE_URL,
    )


def make_request_id(target: str, name: str, now: int | None = None) -> str:
    """Create a stable, readable request id for queue files and directories."""

    ts = now if now is not None else int(time.time())
    safe_target = _slug(target)
    safe_name = _slug(name)
    return f"hev-{ts}-{safe_target}-{safe_name}"[:96]


def build_request(
    *,
    target: str,
    name: str,
    reason: str,
    eval_source: str = "sessiondb",
    iterations: int = 10,
    hermes_agent_repo: str | None = None,
    external_runner_repo: str | None = None,
    external_runner_source_url: str | None = None,
    forgejo_repo: str | None = None,
    request_id: str | None = None,
    created_at: int | None = None,
) -> EvolutionRequest:
    """Validate user input and build a self-evolution request."""

    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of: {', '.join(VALID_TARGETS)}")
    if not name.strip():
        raise ValueError("name is required")
    if not reason.strip():
        raise ValueError("reason is required")
    if eval_source not in VALID_EVAL_SOURCES:
        raise ValueError(
            f"eval_source must be one of: {', '.join(VALID_EVAL_SOURCES)}"
        )
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    now = created_at if created_at is not None else int(time.time())
    repo = hermes_agent_repo or str(Path.cwd())
    return EvolutionRequest(
        request_id=request_id or make_request_id(target, name, now),
        target=target,
        name=name.strip(),
        reason=reason.strip(),
        eval_source=eval_source,
        iterations=iterations,
        hermes_agent_repo=str(Path(repo).expanduser().resolve()),
        external_runner_repo=external_runner_repo or default_external_runner_repo(),
        external_runner_source_url=(
            external_runner_source_url or default_external_runner_source_url()
        ),
        forgejo_repo=forgejo_repo or default_forgejo_repo(),
        status="queued",
        created_at=now,
    )


def write_request_artifacts(
    request: EvolutionRequest,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Persist request JSON and a human runbook under HERMES_HOME/evolution."""

    root = evolution_home(base_dir)
    request_dir = root / "requests" / request.request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    (request_dir / "request.json").write_text(
        json.dumps(asdict(request), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (request_dir / "RUNBOOK.md").write_text(render_runbook(request), encoding="utf-8")
    return request_dir


def append_queue(request: EvolutionRequest, *, base_dir: Path | None = None) -> Path:
    """Append the request to the profile-local JSONL queue."""

    root = evolution_home(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    queue_path = root / "queue.jsonl"
    with queue_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(request), sort_keys=True) + "\n")
    return queue_path


def load_request(request_id: str, *, base_dir: Path | None = None) -> EvolutionRequest:
    """Load a request by id from the artifact directory."""

    request_path = evolution_home(base_dir) / "requests" / request_id / "request.json"
    data = json.loads(request_path.read_text(encoding="utf-8"))
    return EvolutionRequest(**data)


def external_command(request: EvolutionRequest) -> list[str]:
    """Return the external optimizer command for a supported request."""

    if request.target == "skill":
        return [
            "python",
            "-m",
            "evolution.skills.evolve_skill",
            "--skill",
            request.name,
            "--iterations",
            str(request.iterations),
            "--eval-source",
            request.eval_source,
        ]

    raise ValueError(
        f"{request.target} evolution is queued but not executable by the current "
        "external runner; keep it as a PR-gated request."
    )


def render_runbook(request: EvolutionRequest) -> str:
    """Render a concise review/runbook for a queued evolution request."""

    command = _shell_join(external_command(request)) if request.target == "skill" else ""
    lines = [
        f"# Hermes Evolution Request: {request.request_id}",
        "",
        f"- Target: `{request.target}`",
        f"- Name: `{request.name}`",
        f"- Reason: {request.reason}",
        f"- Eval source: `{request.eval_source}`",
        f"- Iterations: `{request.iterations}`",
        f"- Hermes repo: `{request.hermes_agent_repo}`",
        f"- External runner repo: `{request.external_runner_repo}`",
        f"- External runner source: `{request.external_runner_source_url}`",
        f"- Forgejo repo: `{request.forgejo_repo}`",
        "",
        "## Contract",
        "",
        "- Run offline only.",
        "- Never patch an active conversation or running agent process.",
        "- Produce a git branch or PR with metrics, diffs, and holdout results.",
        "- Reject variants that fail tests, size limits, semantic preservation, or benchmark gates.",
        "",
    ]
    if command:
        lines.extend(
            [
                "## Runner Command",
                "",
                "```bash",
                f"cd {_shell_join([request.external_runner_repo])}",
                f"export HERMES_AGENT_REPO={_shell_join([request.hermes_agent_repo])}",
                command,
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Runner Command",
                "",
                "No executable runner exists yet for this target. Keep this request queued until the external optimizer supports it.",
                "",
            ]
        )
    return "\n".join(lines)


def run_external(request: EvolutionRequest, *, execute: bool = False) -> dict[str, Any]:
    """Dry-run or execute the external self-evolution command."""

    command = external_command(request)
    cwd = Path(request.external_runner_repo).expanduser()
    env = os.environ.copy()
    env["HERMES_AGENT_REPO"] = request.hermes_agent_repo
    env["HERMES_EVOLUTION_FORGEJO_REPO"] = request.forgejo_repo

    if not execute:
        return {
            "request_id": request.request_id,
            "execute": False,
            "cwd": str(cwd),
            "command": command,
            "env": {
                "HERMES_AGENT_REPO": request.hermes_agent_repo,
                "HERMES_EVOLUTION_FORGEJO_REPO": request.forgejo_repo,
            },
        }

    if not cwd.exists():
        raise FileNotFoundError(
            f"external runner repo not found: {cwd}. Clone "
            f"{request.external_runner_source_url or UPSTREAM_REPO_URL} "
            "or set HERMES_SELF_EVOLUTION_REPO."
        )

    proc = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "request_id": request.request_id,
        "execute": True,
        "cwd": str(cwd),
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def evolve_command(args) -> None:
    """Argparse entry point for ``hermes evolve``."""

    action = getattr(args, "evolve_action", None)
    if action in (None, "queue", "plan"):
        request = build_request(
            target=args.target,
            name=args.name,
            reason=args.reason,
            eval_source=args.eval_source,
            iterations=args.iterations,
            hermes_agent_repo=args.hermes_agent_repo,
            external_runner_repo=args.external_runner_repo,
            external_runner_source_url=args.external_runner_source_url,
            forgejo_repo=args.forgejo_repo,
        )
        request_dir = write_request_artifacts(request)
        queue_path = append_queue(request)
        payload = {
            "request": asdict(request),
            "request_dir": str(request_dir),
            "queue": str(queue_path),
            "external_command": external_command(request)
            if request.target == "skill"
            else None,
        }
        _print_payload(payload, json_output=args.json)
        return

    if action == "run":
        request = load_request(args.request_id)
        payload = run_external(request, execute=args.execute)
        _print_payload(payload, json_output=args.json)
        if payload.get("execute") and payload.get("returncode", 0) != 0:
            raise SystemExit(int(payload["returncode"]))
        return

    if action == "status":
        payload = queue_status()
        _print_payload(payload, json_output=args.json)
        return

    raise SystemExit(f"unknown evolve action: {action}")


def queue_status(*, base_dir: Path | None = None) -> dict[str, Any]:
    """Return a small machine-readable queue summary."""

    root = evolution_home(base_dir)
    queue_path = root / "queue.jsonl"
    requests: list[dict[str, Any]] = []
    if queue_path.exists():
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            requests.append(json.loads(line))
    return {
        "queue": str(queue_path),
        "count": len(requests),
        "requests": requests[-20:],
    }


def _print_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if "request" in payload:
        req = payload["request"]
        print(f"queued {req['request_id']}")
        print(f"request_dir: {payload['request_dir']}")
        print(f"queue: {payload['queue']}")
        if payload.get("external_command"):
            print("external_command: " + _shell_join(payload["external_command"]))
        return
    if "count" in payload:
        print(f"queue: {payload['queue']}")
        print(f"count: {payload['count']}")
        for req in payload["requests"]:
            print(f"- {req['request_id']} {req['target']}:{req['name']} {req['status']}")
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _slug(value: str) -> str:
    safe = []
    for ch in value.lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in ("-", "_", ".", "/"):
            safe.append("-")
    return "".join(safe).strip("-") or "request"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)
