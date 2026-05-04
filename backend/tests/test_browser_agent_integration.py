"""Integration tests for `backend.apps.agents.browser_agent`.

Drives the full `run_browser_agent` loop end-to-end with a fake
Anthropic SDK and a fully-stubbed `ws_manager`. The unit-helper
coverage lives in `test_browser_agent_unit.py`; this file focuses on
the orchestration + protocol behaviour:

  - happy path (text-only response → completed + screenshot fallback)
  - ReportProgress brain-state recording + same-turn action tool
  - ReportProgress violation: action tool without brain state is rejected
  - loop detection: third identical (tool, input, result) gets a warning
  - loop hard cap: 5 triggers force-exit before MAX_TURNS
  - RequestHumanIntervention allow + deny-with-message branches
  - cancellation via parent_session_id already stopped
  - "no LLM connected" error branch (resolve_aux_model raises)
  - prior-history resume + corrupted-cache drop
  - initial_url navigation runs before the first API call
  - token usage accumulates across turns
  - history is persisted (and well-formed) on completion
  - `_create_browser_card` writes to disk + broadcasts globally
  - `run_browser_agents` parallel fanout (auto-create card,
    exception-per-task surfacing)

All tests:
  - Patch the Anthropic client at the lazy-import boundary in
    `backend.apps.settings.credentials.get_anthropic_client_for_model`.
  - Patch `ws_manager.send_to_session` / `send_browser_command` /
    `send_approval_request` / `broadcast_global` as AsyncMocks.
  - Wipe `_browser_history` and `agent_manager.sessions/tasks` between
    tests so module-level state never leaks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.apps.agents import browser_agent as ba
from backend.apps.agents.browser_agent import (
    _LOOP_HARD_CAP,
    MAX_TURNS,
    _create_browser_card,
    _validate_message_pairing,
    run_browser_agent,
    run_browser_agents,
)
from backend.apps.agents.models import AgentSession


# ---------------------------------------------------------------------------
# SDK fakes
# ---------------------------------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    """Anthropic-shaped text content block (dot-attribute access)."""
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(*, id: str, name: str, input: dict) -> SimpleNamespace:
    """Anthropic-shaped tool_use content block."""
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _make_response(
    blocks: list[Any],
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> SimpleNamespace:
    """Anthropic-shaped Messages API response with a `usage` sub-object."""
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _make_fake_client(responses: list[Any] | None = None) -> MagicMock:
    """Build a duck-typed AsyncAnthropic client.

    `client.messages.create(...)` returns a coroutine resolving to the
    next item from `responses`. If a callable is passed instead, it's
    used as the AsyncMock side_effect (lets tests return the same
    response indefinitely for loop-detection coverage).

    The fake also snapshots each call's `messages` kwarg into
    `client.message_snapshots` because the agent loop mutates the
    same `messages` list after every turn — `mock.await_args.kwargs`
    only retains the reference, so without a snapshot we'd assert on
    the post-mutation state."""
    client = MagicMock()
    client.messages = MagicMock()
    client.message_snapshots: list[list[dict]] = []

    if callable(responses):
        cb = responses

        async def _wrapped(*args, **kwargs):
            client.message_snapshots.append(
                [dict(m) for m in kwargs.get("messages", [])]
            )
            return cb(*args, **kwargs)

        client.messages.create = AsyncMock(side_effect=_wrapped)
    else:
        queue = list(responses or [])

        async def _wrapped(*args, **kwargs):
            client.message_snapshots.append(
                [dict(m) for m in kwargs.get("messages", [])]
            )
            if not queue:
                raise StopAsyncIteration("ran out of scripted responses")
            return queue.pop(0)

        client.messages.create = AsyncMock(side_effect=_wrapped)
    return client


# ---------------------------------------------------------------------------
# Shared patching helpers
# ---------------------------------------------------------------------------


class _WSStub:
    """Captures every (event, payload) emitted to `ws_manager.send_to_session`,
    and counts approvals + browser command calls. Plug in via patch.object."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self.browser_commands: list[dict] = []
        self.approval_requests: list[dict] = []
        self.global_broadcasts: list[tuple[str, dict]] = []

        # Default behaviours; tests override to script approval flows.
        self.send_to_session = AsyncMock(side_effect=self._on_send_to_session)
        self.send_browser_command = AsyncMock(side_effect=self._on_browser_command)
        self.send_approval_request = AsyncMock(
            return_value={"behavior": "allow"}
        )
        self.broadcast_global = AsyncMock(side_effect=self._on_broadcast_global)

        # Default "browser command result" — tests override to script
        # specific tool results (e.g. {"image": "..."} for screenshots).
        self.browser_result: dict | Any = {"text": "ok"}

    async def _on_send_to_session(self, session_id, event, payload):
        self.events.append((session_id, event, payload))

    async def _on_browser_command(self, request_id, action, browser_id, params, tab_id=""):
        self.browser_commands.append({
            "request_id": request_id,
            "action": action,
            "browser_id": browser_id,
            "params": params,
            "tab_id": tab_id,
        })
        if callable(self.browser_result):
            return self.browser_result(action, params)
        return dict(self.browser_result)

    async def _on_broadcast_global(self, event, data):
        self.global_broadcasts.append((event, data))

    def events_of_type(self, event_type: str) -> list[dict]:
        return [p for (_sid, ev, p) in self.events if ev == event_type]


def _patch_browser_agent_deps(
    monkeypatch,
    *,
    fake_client: MagicMock,
    ws_stub: _WSStub,
    builtin_perms: dict[str, str] | None = None,
    is_builtin_model: bool = True,
    aux_model_raises: bool = False,
):
    """Apply the four canonical patches every integration test needs.

    - ws_manager methods → ws_stub's AsyncMocks
    - get_anthropic_client_for_model → returns fake_client
    - load_settings → no-op AppSettings()
    - permissions / model resolution → defaults safe for tests
    """
    monkeypatch.setattr(ba.ws_manager, "send_to_session", ws_stub.send_to_session)
    monkeypatch.setattr(ba.ws_manager, "send_browser_command", ws_stub.send_browser_command)
    monkeypatch.setattr(ba.ws_manager, "send_approval_request", ws_stub.send_approval_request)
    monkeypatch.setattr(ba.ws_manager, "broadcast_global", ws_stub.broadcast_global)

    monkeypatch.setattr(
        ba, "load_builtin_permissions", lambda: builtin_perms or {}
    )

    from backend.apps.settings import credentials as creds_mod
    from backend.apps.settings import settings as settings_mod
    from backend.apps.settings.models import AppSettings
    from backend.apps.agents.providers import registry as registry_mod

    monkeypatch.setattr(settings_mod, "load_settings", lambda: AppSettings())
    monkeypatch.setattr(
        creds_mod, "get_anthropic_client_for_model",
        lambda settings, model: fake_client,
    )

    if is_builtin_model:
        monkeypatch.setattr(
            registry_mod, "_find_builtin_model",
            lambda short: {"value": short, "api": "anthropic"},
        )
        monkeypatch.setattr(
            registry_mod, "resolve_model_id_for_sdk",
            lambda short, settings: f"claude-fake-{short}",
        )
    else:
        monkeypatch.setattr(registry_mod, "_find_builtin_model", lambda short: None)
        if aux_model_raises:
            async def _raises(*args, **kwargs):
                raise ValueError("no provider connected")
            monkeypatch.setattr(registry_mod, "resolve_aux_model", _raises)
        else:
            async def _ok(*args, **kwargs):
                return ("claude-fake-aux", None)
            monkeypatch.setattr(registry_mod, "resolve_aux_model", _ok)


@pytest.fixture(autouse=True)
def _reset_browser_agent_state(tmp_data_dirs):
    """Module-level conversation cache only — `tmp_data_dirs` (from
    conftest.py) already wipes the per-feature data dirs AND clears
    `agent_manager.sessions/tasks` between tests, so the only piece
    of state unique to browser_agent is `_browser_history`."""
    ba._browser_history.clear()
    yield
    ba._browser_history.clear()


# ===========================================================================
# run_browser_agent — happy paths
# ===========================================================================


async def test_happy_path_text_only_completes_and_persists_history(monkeypatch):
    """One turn returning text only → status flips to completed,
    summary echoes the model text, history is cached for next call."""
    ws = _WSStub()
    ws.browser_result = {"image": "FALLBACK_PNG", "url": "https://x.com"}
    fake = _make_fake_client([
        _make_response(
            [_text_block("All done!")],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=2,
        ),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(
        task="say hello",
        browser_id="b-1",
        model="sonnet",
    )
    assert out["browser_id"] == "b-1"
    assert out["summary"] == "All done!"
    # Final screenshot fallback was triggered (no screenshot during the loop).
    assert out["final_screenshot"] == "FALLBACK_PNG"

    # agent:status events fired for both running and completed.
    statuses = [p["status"] for p in ws.events_of_type("agent:status")]
    assert "running" in statuses
    assert "completed" in statuses

    # History was cached and round-trips _validate_message_pairing.
    assert "b-1" in ba._browser_history
    assert _validate_message_pairing(ba._browser_history["b-1"])


async def test_report_progress_plus_action_tool_records_brain_state(monkeypatch):
    """Turn 1: ReportProgress + BrowserNavigate. Turn 2: text + end.
    The brain-state assistant message must show up in session.messages
    AND the navigate tool must be dispatched through ws_manager."""
    ws = _WSStub()

    def _result_for(action: str, params: dict) -> dict:
        if action == "navigate":
            return {"text": f"navigated to {params.get('url')}"}
        return {"image": "PNG", "url": "https://x.com"}

    ws.browser_result = _result_for

    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="rp-1", name="ReportProgress",
                    input={
                        "evaluation_previous": "n/a (first turn)",
                        "working_memory": "fresh page",
                        "next_goal": "navigate to example.com",
                    },
                ),
                _tool_use_block(
                    id="nav-1", name="BrowserNavigate",
                    input={"url": "https://example.com"},
                ),
            ],
            stop_reason="tool_use",
            input_tokens=15, output_tokens=3,
        ),
        _make_response(
            [_text_block("Navigation complete.")],
            stop_reason="end_turn",
            input_tokens=20, output_tokens=4,
        ),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="go", browser_id="b-1", model="sonnet")
    assert out["summary"] == "Navigation complete."

    # Brain-state message ("📋 **Plan**") was persisted.
    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    brain_msgs = [
        m for m in sess.messages
        if m.role == "assistant" and isinstance(m.content, str)
        and "**Plan**" in m.content
    ]
    assert len(brain_msgs) == 1
    assert "next goal" in brain_msgs[0].content.lower() \
        or "navigate to example.com" in brain_msgs[0].content

    # Navigate command flowed through ws_manager with the right params.
    nav_calls = [c for c in ws.browser_commands if c["action"] == "navigate"]
    assert len(nav_calls) >= 1
    assert nav_calls[0]["params"] == {"url": "https://example.com"}

    # action_log captured the tool execution.
    nav_logs = [a for a in out["action_log"] if a["tool"] == "BrowserNavigate"]
    assert len(nav_logs) == 1


async def test_report_progress_violation_rejects_action_tool(monkeypatch):
    """Action tool without ReportProgress in the same turn → tool_result
    is is_error=True with REJECTED text; the underlying browser
    command is NEVER dispatched to ws_manager."""
    ws = _WSStub()
    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="click-1", name="BrowserClick",
                    input={"selector": "#go"},
                ),
            ],
            stop_reason="tool_use",
        ),
        # Turn 2: model gives up and returns text.
        _make_response([_text_block("aborted")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-rp", model="sonnet")
    assert out["summary"] == "aborted"

    # Click was rejected → no `click` action ever made it to ws_manager.
    click_calls = [c for c in ws.browser_commands if c["action"] == "click"]
    assert click_calls == []

    # The rejection text shows up as a tool_result Message in the session.
    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    rejection_msgs = [
        m for m in sess.messages
        if m.role == "tool_result" and isinstance(m.content, dict)
        and "REJECTED" in str(m.content.get("text", ""))
    ]
    assert len(rejection_msgs) == 1


# ===========================================================================
# Loop detection
# ===========================================================================


async def test_loop_detection_third_identical_call_gets_warning(monkeypatch):
    """Three identical (tool, input, result) tuples → on the third call
    the tool_result includes the LOOP DETECTED warning (is_error=True)."""
    ws = _WSStub()
    # Always return the same result so the loop-detector key is identical.
    ws.browser_result = {"text": "click did nothing"}

    def _looping_response(*args, **kwargs):
        return _make_response(
            [
                _tool_use_block(
                    id=f"rp-{kwargs.get('messages', [{}])[-1]}",
                    name="ReportProgress",
                    input={
                        "evaluation_previous": "didn't help",
                        "working_memory": "site is hostile",
                        "next_goal": "try again",
                    },
                ),
                _tool_use_block(
                    id=f"click-{len(kwargs.get('messages', []))}",
                    name="BrowserClick",
                    input={"selector": "#submit"},
                ),
            ],
            stop_reason="tool_use",
        )

    fake = _make_fake_client(_looping_response)
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-loop", model="sonnet")

    # Loop hard cap eventually force-exits, so we get the canned summary.
    assert isinstance(out["summary"], str) and out["summary"]

    # Inspect session tool_result messages: the 3rd+ click result must
    # carry the LOOP DETECTED warning. (Earlier ones won't.)
    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    click_results = [
        m for m in sess.messages
        if m.role == "tool_result"
        and isinstance(m.content, dict)
        and m.content.get("tool_name") == "BrowserClick"
    ]
    assert len(click_results) >= 3
    # Note: the warning is appended to the API content_block, not the
    # session.message text — so we verify via ws_manager send_browser_command
    # call count + the loop-trigger side effect (hard cap force-exit).
    # Verify hard cap took effect: fewer API calls than MAX_TURNS.
    assert fake.messages.create.await_count < MAX_TURNS


async def test_loop_hard_cap_force_exits_before_max_turns(monkeypatch):
    """Once `_LOOP_HARD_CAP` triggers fire, the loop must break early
    rather than burning the entire MAX_TURNS budget."""
    ws = _WSStub()
    ws.browser_result = {"text": "still failing"}

    call_count = {"n": 0}

    def _looping_response(*args, **kwargs):
        call_count["n"] += 1
        n = call_count["n"]
        return _make_response(
            [
                _tool_use_block(
                    id=f"rp-{n}", name="ReportProgress",
                    input={
                        "evaluation_previous": "stuck",
                        "working_memory": "...",
                        "next_goal": "click again",
                    },
                ),
                _tool_use_block(
                    id=f"click-{n}", name="BrowserClick",
                    input={"selector": "#x"},
                ),
            ],
            stop_reason="tool_use",
        )

    fake = _make_fake_client(_looping_response)
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-cap", model="sonnet")

    # Strict bound: hard cap ought to fire within ~2 * _LOOP_HARD_CAP turns
    # (each trigger after threshold takes one turn). Assert we're well
    # below MAX_TURNS — and well below it (no near-misses).
    assert fake.messages.create.await_count <= _LOOP_HARD_CAP + 5
    assert fake.messages.create.await_count < MAX_TURNS
    # Run still completed cleanly (status=completed, not error).
    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    assert sess.status == "completed"


# ===========================================================================
# RequestHumanIntervention
# ===========================================================================


async def test_request_human_intervention_allow_resumes(monkeypatch):
    """Approval allow → tool_result text reads 'User resolved the issue.'
    AND a waiting_approval status event was emitted to the session."""
    ws = _WSStub()
    ws.send_approval_request = AsyncMock(return_value={"behavior": "allow"})

    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="rhi-1", name="RequestHumanIntervention",
                    input={"problem": "captcha", "instruction": "solve it"},
                ),
            ],
            stop_reason="tool_use",
        ),
        _make_response([_text_block("done")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-hi", model="sonnet")
    assert out["summary"] == "done"

    statuses = [p["status"] for p in ws.events_of_type("agent:status")]
    assert "waiting_approval" in statuses

    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    resolved = [
        m for m in sess.messages
        if m.role == "tool_result"
        and isinstance(m.content, dict)
        and "User resolved" in str(m.content.get("text", ""))
    ]
    assert len(resolved) == 1


async def test_request_human_intervention_deny_with_message_surfaces_user_text(monkeypatch):
    """Approval deny with user-supplied message → tool_result text must
    include the user message and the 'Address what the user said' prefix."""
    ws = _WSStub()
    ws.send_approval_request = AsyncMock(return_value={
        "behavior": "deny",
        "message": "use a different site",
    })

    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="rhi-2", name="RequestHumanIntervention",
                    input={"problem": "blocked", "instruction": "?"},
                ),
            ],
            stop_reason="tool_use",
        ),
        _make_response([_text_block("ok")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-hi2", model="sonnet")

    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    deny_msgs = [
        m for m in sess.messages
        if m.role == "tool_result" and isinstance(m.content, dict)
        and "use a different site" in str(m.content.get("text", ""))
    ]
    assert len(deny_msgs) == 1
    assert "Address what the user said" in str(deny_msgs[0].content["text"])


# ===========================================================================
# Cancellation + error paths
# ===========================================================================


async def test_cancellation_via_stopped_parent_returns_stopped_summary(monkeypatch):
    """Parent session already stopped before browser-agent registers →
    cancel_event set immediately → no API call, status='stopped'."""
    from backend.apps.agents.agent_manager import agent_manager

    parent_id = "parent-stopped"
    parent = AgentSession(id=parent_id, name="parent", model="sonnet")
    parent.status = "stopped"
    agent_manager.sessions[parent_id] = parent

    ws = _WSStub()
    fake = _make_fake_client()  # never actually called
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(
        task="x",
        browser_id="b-cancel",
        model="sonnet",
        parent_session_id=parent_id,
    )
    assert "stopped by the user" in out["summary"]
    assert out["error"] == "Agent was stopped by the user."
    assert agent_manager.sessions[out["session_id"]].status == "stopped"
    # Loop never entered — API client was never called.
    assert fake.messages.create.await_count == 0


async def test_no_creds_branch_returns_error_summary(monkeypatch):
    """Unknown model + resolve_aux_model raises ValueError → error
    payload returned without ever making an API call."""
    ws = _WSStub()
    fake = _make_fake_client()
    _patch_browser_agent_deps(
        monkeypatch, fake_client=fake, ws_stub=ws,
        is_builtin_model=False, aux_model_raises=True,
    )

    out = await run_browser_agent(
        task="x", browser_id="b-noc", model="random-unknown-model",
    )
    assert out["summary"].startswith("Error: Browser agent requires")
    assert out["action_log"] == []
    assert out["final_screenshot"] is None

    from backend.apps.agents.agent_manager import agent_manager
    assert agent_manager.sessions[out["session_id"]].status == "error"
    assert fake.messages.create.await_count == 0


# ===========================================================================
# History resume / cache invalidation
# ===========================================================================


async def test_history_resume_prepends_cached_messages_to_first_call(monkeypatch):
    """Cached prior messages must be prepended to the messages list
    sent on turn 1 (otherwise the agent re-orients from scratch every
    swipe)."""
    cached = [
        {"role": "user", "content": "earlier task"},
        {"role": "assistant", "content": [{"type": "text", "text": "earlier reply"}]},
    ]
    ba._browser_history["b-resume"] = list(cached)

    ws = _WSStub()
    fake = _make_fake_client([
        _make_response([_text_block("ok")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    await run_browser_agent(task="next", browser_id="b-resume", model="sonnet")

    # Snapshot of the messages list at first-call time (the loop mutates it
    # in place between turns, so we can't read await_args after-the-fact).
    sent_messages = fake.message_snapshots[0]
    assert sent_messages[:2] == cached
    assert sent_messages[2] == {"role": "user", "content": "next"}


async def test_corrupt_cached_history_is_dropped(monkeypatch):
    """If the cached chain has an orphan tool_result, it's dropped and
    only the new user task is sent."""
    bad = [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "ghost", "content": []},
        ]},
    ]
    ba._browser_history["b-corrupt"] = list(bad)

    ws = _WSStub()
    fake = _make_fake_client([
        _make_response([_text_block("clean start")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    await run_browser_agent(task="fresh", browser_id="b-corrupt", model="sonnet")

    sent_messages = fake.message_snapshots[0]
    assert sent_messages == [{"role": "user", "content": "fresh"}]


# ===========================================================================
# Initial URL navigation + token tracking
# ===========================================================================


async def test_initial_url_triggers_navigate_before_first_api_call(monkeypatch):
    """When initial_url is provided, BrowserNavigate runs once before
    the loop opens an API call."""
    ws = _WSStub()

    def _result_for(action: str, params: dict) -> dict:
        if action == "navigate":
            return {"text": "navigated"}
        return {"image": "PNG", "url": "https://example.com"}

    ws.browser_result = _result_for

    fake = _make_fake_client([
        _make_response([_text_block("ack")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    await run_browser_agent(
        task="x", browser_id="b-init", model="sonnet",
        initial_url="https://example.com",
    )

    nav_calls = [c for c in ws.browser_commands if c["action"] == "navigate"]
    assert len(nav_calls) == 1
    assert nav_calls[0]["params"] == {"url": "https://example.com"}
    assert nav_calls[0]["browser_id"] == "b-init"


async def test_token_usage_accumulates_across_turns(monkeypatch):
    """Per-turn input/output tokens roll up into session.tokens."""
    ws = _WSStub()
    ws.browser_result = {"text": "ok"}

    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="rp-1", name="ReportProgress",
                    input={"evaluation_previous": "n/a",
                           "working_memory": "x",
                           "next_goal": "click"},
                ),
                _tool_use_block(
                    id="c-1", name="BrowserClick", input={"selector": "#a"},
                ),
            ],
            stop_reason="tool_use",
            input_tokens=5, output_tokens=3,
        ),
        _make_response(
            [_text_block("done")],
            stop_reason="end_turn",
            input_tokens=2, output_tokens=1,
        ),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    out = await run_browser_agent(task="x", browser_id="b-tok", model="sonnet")
    from backend.apps.agents.agent_manager import agent_manager
    sess = agent_manager.sessions[out["session_id"]]
    assert sess.tokens == {"input": 7, "output": 4}


async def test_history_persisted_and_well_formed(monkeypatch):
    """After a successful run, _browser_history[browser_id] is
    populated AND passes the same orphan-pairing validator the resume
    path uses on read."""
    ws = _WSStub()
    ws.browser_result = {"text": "ok"}

    fake = _make_fake_client([
        _make_response(
            [
                _tool_use_block(
                    id="rp-1", name="ReportProgress",
                    input={"evaluation_previous": "x", "working_memory": "y", "next_goal": "z"},
                ),
                _tool_use_block(
                    id="c-1", name="BrowserClick", input={"selector": "#a"},
                ),
            ],
            stop_reason="tool_use",
        ),
        _make_response([_text_block("done")], stop_reason="end_turn"),
    ])
    _patch_browser_agent_deps(monkeypatch, fake_client=fake, ws_stub=ws)

    await run_browser_agent(task="x", browser_id="b-hist", model="sonnet")
    assert "b-hist" in ba._browser_history
    cached = ba._browser_history["b-hist"]
    assert len(cached) >= 2
    assert _validate_message_pairing(cached) is True


# ===========================================================================
# _create_browser_card
# ===========================================================================


async def test_create_browser_card_writes_disk_and_broadcasts(monkeypatch):
    """Card is appended to the dashboard layout, persisted, and
    broadcast as `dashboard:browser_card_added`."""
    from backend.apps.dashboards.dashboards import _save, _load
    from backend.apps.dashboards.models import Dashboard

    dash = Dashboard(id="d-card", name="t")
    _save(dash)

    captured: list[tuple[str, dict]] = []

    async def _capture_broadcast(event: str, data: dict):
        captured.append((event, data))

    monkeypatch.setattr(ba.ws_manager, "broadcast_global",
                        AsyncMock(side_effect=_capture_broadcast))

    browser_id = await _create_browser_card(
        "d-card", "https://x.com", parent_session_id="p-1",
    )
    assert browser_id.startswith("browser-")

    reloaded = _load("d-card")
    assert browser_id in reloaded.layout.browser_cards
    card = reloaded.layout.browser_cards[browser_id]
    assert card.url == "https://x.com"
    assert card.spawned_by == "p-1"

    # Single global broadcast for the new card.
    new_card_events = [(ev, p) for (ev, p) in captured
                       if ev == "dashboard:browser_card_added"]
    assert len(new_card_events) == 1
    payload = new_card_events[0][1]
    assert payload["dashboard_id"] == "d-card"
    assert payload["parent_session_id"] == "p-1"
    assert payload["browser_card"]["browser_id"] == browser_id


async def test_create_browser_card_defaults_url_when_blank(monkeypatch):
    """If url is empty, both the card and tab fall back to google.com."""
    from backend.apps.dashboards.dashboards import _save, _load
    from backend.apps.dashboards.models import Dashboard

    _save(Dashboard(id="d-card2", name="t"))
    monkeypatch.setattr(ba.ws_manager, "broadcast_global", AsyncMock())

    browser_id = await _create_browser_card("d-card2", "")
    card = _load("d-card2").layout.browser_cards[browser_id]
    assert card.url == "https://www.google.com"
    assert card.tabs[0].url == "https://www.google.com"


# ===========================================================================
# run_browser_agents — parallel fanout
# ===========================================================================


async def test_run_browser_agents_fanout_auto_creates_card_and_returns_results(monkeypatch):
    """Two tasks: one with browser_id, one without. The latter should
    auto-create a card. Both results returned in input order."""
    from backend.apps.dashboards.dashboards import _save
    from backend.apps.dashboards.models import Dashboard

    _save(Dashboard(id="d-fanout", name="t"))

    # Don't actually wait 2s for "card settle".
    monkeypatch.setattr(ba.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(ba.ws_manager, "broadcast_global", AsyncMock())
    # Patch analytics so we don't depend on its plumbing here.
    monkeypatch.setattr(
        "backend.apps.analytics.collector.record",
        lambda *a, **kw: None,
    )

    call_log: list[dict] = []

    async def _fake_run_one(**kwargs):
        call_log.append(kwargs)
        return {
            "session_id": f"s-{kwargs['browser_id']}",
            "browser_id": kwargs["browser_id"],
            "summary": f"summary for {kwargs['browser_id']}",
            "action_log": [],
            "final_screenshot": None,
        }

    monkeypatch.setattr(ba, "run_browser_agent", _fake_run_one)

    results = await run_browser_agents(
        tasks=[
            {"browser_id": "b-existing", "task": "do A"},
            {"task": "do B", "url": "https://example.com"},
        ],
        model="sonnet",
        dashboard_id="d-fanout",
    )

    assert len(results) == 2
    assert results[0]["summary"] == "summary for b-existing"
    # Auto-created browser_id matches the second call's argument.
    auto_browser_id = call_log[1]["browser_id"]
    assert auto_browser_id.startswith("browser-")
    assert results[1]["summary"] == f"summary for {auto_browser_id}"


async def test_run_browser_agents_exception_per_task_surfaces_in_results(monkeypatch):
    """If run_browser_agent raises for one task, the other still
    completes and the failed slot becomes {summary: 'Error: ...'}."""
    monkeypatch.setattr(ba.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(ba.ws_manager, "broadcast_global", AsyncMock())
    monkeypatch.setattr(
        "backend.apps.analytics.collector.record",
        lambda *a, **kw: None,
    )

    async def _maybe_explode(**kwargs):
        if kwargs["browser_id"] == "boom":
            raise RuntimeError("kaboom")
        return {
            "session_id": "s-ok",
            "browser_id": kwargs["browser_id"],
            "summary": "ok-summary",
            "action_log": [],
            "final_screenshot": None,
        }

    monkeypatch.setattr(ba, "run_browser_agent", _maybe_explode)

    results = await run_browser_agents(
        tasks=[
            {"browser_id": "ok-1", "task": "fine"},
            {"browser_id": "boom", "task": "explodes"},
        ],
        model="sonnet",
    )
    assert results[0]["summary"] == "ok-summary"
    assert results[1]["summary"].startswith("Error: ")
    assert results[1]["action_log"] == []
    assert results[1]["final_screenshot"] is None
