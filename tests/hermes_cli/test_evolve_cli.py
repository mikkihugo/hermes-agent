"""Tests for the offline Hermes self-evolution queue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import evolve


def test_build_request_for_skill_defaults_to_pr_gated_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    request = evolve.build_request(
        target="skill",
        name="github-code-review",
        reason="review sessions missed planted auth bugs",
        eval_source="sessiondb",
        iterations=7,
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        created_at=1234,
    )

    assert request.request_id == "hev-1234-skill-github-code-review"
    assert request.target == "skill"
    assert request.dry_run_only is True
    assert request.forgejo_repo == "centralcloud/hermes-agent-self-evolution"
    assert request.external_runner_source_url.startswith(
        "ssh://git@git.infra.centralcloud.com:2222/centralcloud/"
    )
    assert request.external_runner_repo.endswith("hermes-agent-self-evolution")


def test_queue_writes_jsonl_request_and_runbook(tmp_path):
    request = evolve.build_request(
        target="skill",
        name="github-code-review",
        reason="missed tool misuse findings",
        eval_source="synthetic",
        iterations=3,
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        external_runner_repo=str(tmp_path / "runner"),
        request_id="hev-test",
        created_at=1234,
    )

    request_dir = evolve.write_request_artifacts(request, base_dir=tmp_path / "evolution")
    queue_path = evolve.append_queue(request, base_dir=tmp_path / "evolution")

    saved = json.loads((request_dir / "request.json").read_text())
    queued = json.loads(queue_path.read_text().strip())
    runbook = (request_dir / "RUNBOOK.md").read_text()

    assert saved["request_id"] == "hev-test"
    assert queued["request_id"] == "hev-test"
    assert "python -m evolution.skills.evolve_skill" in runbook
    assert "External runner source" in runbook
    assert "Never patch an active conversation" in runbook


def test_external_command_for_skill_matches_external_runner_contract(tmp_path):
    request = evolve.build_request(
        target="skill",
        name="arxiv",
        reason="paper lookup traces are weak",
        eval_source="golden",
        iterations=5,
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        request_id="hev-skill",
    )

    assert evolve.external_command(request) == [
        "python",
        "-m",
        "evolution.skills.evolve_skill",
        "--skill",
        "arxiv",
        "--iterations",
        "5",
        "--eval-source",
        "golden",
    ]


def test_run_external_without_execute_is_dry_run(tmp_path):
    request = evolve.build_request(
        target="skill",
        name="arxiv",
        reason="paper lookup traces are weak",
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        external_runner_repo=str(tmp_path / "missing-runner"),
        request_id="hev-dry",
    )

    result = evolve.run_external(request, execute=False)

    assert result["execute"] is False
    assert result["request_id"] == "hev-dry"
    assert result["env"]["HERMES_AGENT_REPO"].endswith("hermes-agent")


def test_non_skill_targets_are_queue_only(tmp_path):
    request = evolve.build_request(
        target="prompt-section",
        name="MEMORY_GUIDANCE",
        reason="sessions forgot to use memory",
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        request_id="hev-prompt",
    )

    with pytest.raises(ValueError, match="not executable"):
        evolve.external_command(request)


def test_queue_status_reads_recent_requests(tmp_path):
    base_dir = tmp_path / "evolution"
    request = evolve.build_request(
        target="skill",
        name="github-code-review",
        reason="missed review findings",
        hermes_agent_repo=str(tmp_path / "hermes-agent"),
        request_id="hev-status",
    )
    evolve.append_queue(request, base_dir=base_dir)

    status = evolve.queue_status(base_dir=base_dir)

    assert status["count"] == 1
    assert status["requests"][0]["request_id"] == "hev-status"
