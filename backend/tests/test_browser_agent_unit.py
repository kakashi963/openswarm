"""Unit tests for `backend.apps.agents.browser_agent` pure helpers.

The browser-agent runner is a hefty module, but most of the cleverness is
in small helpers that can be exercised in isolation:

  - `_hash_tool_call` / `_detect_loop` — sliding-window loop detector.
  - `_validate_message_pairing` — orphan-`tool_result` guard for cached
    history (the last-line-of-defense against a 400 from Anthropic).
  - `_is_fresh_user_message` / `_summarize_messages` /
    `_trim_history_by_turns` — programmatic compaction of long
    conversations.
  - `_format_tool_result` — translate `ws_manager` result dicts into
    Anthropic content blocks.
  - `clear_browser_history` / `execute_browser_tool` — small async + state
    helpers.

The integration-level tests for `run_browser_agent` /
`run_browser_agents` / `_create_browser_card` live in the sister file
`test_browser_agent_integration.py`. This file deliberately contains NO
real API calls — it asserts only on pure-Python behaviour.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.apps.agents import browser_agent as ba
from backend.apps.agents.browser_agent import (
    _LOOP_HARD_CAP,
    _LOOP_REPEAT_THRESHOLD,
    _LOOP_WARNING_TEXT,
    _LOOP_WINDOW_SIZE,
    _detect_loop,
    _format_tool_result,
    _hash_tool_call,
    _is_fresh_user_message,
    _summarize_messages,
    _trim_history_by_turns,
    _validate_message_pairing,
    clear_browser_history,
    execute_browser_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_browser_history_module_state():
    """Wipe the module-level conversation cache between tests so a leaked
    entry from a prior test never bleeds into the next."""
    ba._browser_history.clear()
    yield
    ba._browser_history.clear()


# ---------------------------------------------------------------------------
# clear_browser_history
# ---------------------------------------------------------------------------


def test_clear_browser_history_drops_only_target_entry():
    ba._browser_history["b-1"] = [{"role": "user", "content": "hi"}]
    ba._browser_history["b-2"] = [{"role": "user", "content": "yo"}]

    clear_browser_history("b-1")

    assert "b-1" not in ba._browser_history
    assert "b-2" in ba._browser_history


def test_clear_browser_history_missing_id_is_noop():
    """Calling clear on an unknown id must not raise (it's used in
    cleanup paths where the cache may already be empty)."""
    clear_browser_history("never-existed")
    assert ba._browser_history == {}


# ---------------------------------------------------------------------------
# _hash_tool_call
# ---------------------------------------------------------------------------


def test_hash_tool_call_deterministic_for_same_inputs():
    """Same (tool, input, result) → same key, every time. This is what
    lets the sliding-window loop detector compare across turns."""
    a = _hash_tool_call("BrowserClick", {"selector": "#x"}, {"text": "ok"})
    b = _hash_tool_call("BrowserClick", {"selector": "#x"}, {"text": "ok"})
    assert a == b
    # Tuple shape: (tool_name, json_input, truncated_json_result)
    assert isinstance(a, tuple) and len(a) == 3
    assert a[0] == "BrowserClick"


def test_hash_tool_call_input_key_uses_sorted_keys():
    """Dict ordering must NOT change the key — otherwise a model that
    happens to emit input keys in different orders would dodge the
    loop guard."""
    a = _hash_tool_call("BrowserType", {"selector": "#i", "text": "a"}, {})
    b = _hash_tool_call("BrowserType", {"text": "a", "selector": "#i"}, {})
    assert a == b


def test_hash_tool_call_different_results_yield_different_keys():
    """If the result changes (e.g. BrowserScroll on a long feed), the
    key must change so legitimate progress isn't mistaken for a loop."""
    a = _hash_tool_call("BrowserScroll", {"direction": "down"}, {"text": "scrolled 500px"})
    b = _hash_tool_call("BrowserScroll", {"direction": "down"}, {"text": "scrolled 250px"})
    assert a != b


def test_hash_tool_call_truncates_long_result_to_300():
    """The result key is sliced at 300 chars to keep huge image blobs
    (BrowserScreenshot returns base64 PNG) out of the in-memory key."""
    big = {"image": "A" * 5000}
    key = _hash_tool_call("BrowserScreenshot", {}, big)
    assert len(key[2]) <= 300


def test_hash_tool_call_falls_back_to_repr_on_serialization_failure():
    """Self-referential dicts trip json.dumps even with default=str.
    The except branch must still return a (str, str, str) tuple."""

    class _Boom:
        def __repr__(self) -> str:
            return "<Boom>"

    boom = _Boom()
    # Self-referential dict — json.dumps raises ValueError with default=str
    bad_input: dict = {"x": 1}
    bad_input["self"] = bad_input

    bad_result: dict = {}
    bad_result["self"] = bad_result

    key = _hash_tool_call("BrowserClick", bad_input, bad_result)
    assert isinstance(key, tuple) and len(key) == 3
    assert key[0] == "BrowserClick"
    # Both fallbacks should be strings (repr-derived), not raise.
    assert isinstance(key[1], str) and key[1]
    assert isinstance(key[2], str) and len(key[2]) <= 300


# ---------------------------------------------------------------------------
# _detect_loop
# ---------------------------------------------------------------------------


def _key(name: str, suffix: str = "") -> tuple[str, str, str]:
    """Convenience for building loop-detector keys without having to
    re-derive the JSON input/result every time."""
    return (name, "{}", suffix or "{}")


def test_detect_loop_excluded_tool_never_triggers():
    """Read-only / idempotent tools (Screenshot, GetText, ReportProgress…)
    are exempt — the model is allowed to call them in tight loops."""
    same = _key("BrowserScreenshot")
    history = [same, same, same, same]  # already at threshold
    assert _detect_loop(history, same) is False


def test_detect_loop_third_repeat_triggers():
    """Threshold is 3 occurrences within the sliding window. Two
    repeats is fine; the third (counting the new call) is the loop."""
    same = _key("BrowserClick")
    different = _key("BrowserType", suffix='{"text":"x"}')

    # 1st & 2nd occurrence in history; the new call is the 3rd.
    assert _detect_loop([same, same, different], same) is True

    # Only 1 prior occurrence + the new = 2 total → not yet a loop.
    assert _detect_loop([same], same) is False


def test_detect_loop_two_identical_plus_one_different_in_window_is_false():
    """Two repeats interleaved with other calls don't add up to a loop."""
    same = _key("BrowserClick")
    other = _key("BrowserScroll", suffix='{"text":"a"}')
    # Window: [same, other, same]. New call=same → 3 total inc. new.
    # Wait — that IS at threshold. Use a lower count instead.
    assert _detect_loop([same, other], same) is False


def test_detect_loop_window_size_caps_lookback():
    """`_LOOP_WINDOW_SIZE - 1 = 4` past entries are considered, plus the
    new call (= 5 total). Anything older falls outside the window."""
    same = _key("BrowserClick")
    other = _key("BrowserScroll")
    # 5 oldest matches but they're outside the window of 4 prior + 1 new.
    history = [same] * 5 + [other, other, other, other]
    # New call=same. Window = last 4 history + new → [other, other, other, other, same].
    # Only 1 match for `same` → not a loop.
    assert _detect_loop(history, same) is False


def test_detect_loop_constants_match_expectations():
    """Sanity-check the module constants the integration tests rely on
    so a refactor that loosens the threshold can't silently invalidate
    coverage downstream."""
    assert _LOOP_WINDOW_SIZE == 5
    assert _LOOP_REPEAT_THRESHOLD == 3
    assert _LOOP_HARD_CAP == 5
    assert "LOOP DETECTED" in _LOOP_WARNING_TEXT


# ---------------------------------------------------------------------------
# _validate_message_pairing
# ---------------------------------------------------------------------------


def test_validate_message_pairing_empty_is_valid():
    assert _validate_message_pairing([]) is True


def test_validate_message_pairing_string_content_skipped():
    """User messages with plain string content carry no tool_result
    blocks, so they have nothing to validate."""
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    assert _validate_message_pairing(msgs) is True


def test_validate_message_pairing_proper_pairing_passes():
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "t1", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": []},
        ]},
    ]
    assert _validate_message_pairing(msgs) is True


def test_validate_message_pairing_orphan_tool_result_fails():
    """tool_result whose tool_use_id was never declared by a prior
    assistant tool_use → invalid (Anthropic API would 400)."""
    msgs = [
        {"role": "user", "content": "do it"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t-orphan", "content": []},
        ]},
    ]
    assert _validate_message_pairing(msgs) is False


def test_validate_message_pairing_multiple_tool_uses_one_orphan_fails():
    """Even with several valid pairings, a single orphan must invalidate
    the whole history (we drop the cache wholesale, not partially)."""
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": []},
            {"type": "tool_result", "tool_use_id": "t-missing", "content": []},
        ]},
    ]
    assert _validate_message_pairing(msgs) is False


# ---------------------------------------------------------------------------
# _is_fresh_user_message
# ---------------------------------------------------------------------------


def test_is_fresh_user_message_string_content():
    assert _is_fresh_user_message({"role": "user", "content": "hi"}) is True


def test_is_fresh_user_message_text_only_list_is_fresh():
    msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert _is_fresh_user_message(msg) is True


def test_is_fresh_user_message_with_tool_result_is_not_fresh():
    """A user message containing a tool_result is mid-turn — cutting
    here would orphan the prior assistant tool_use."""
    msg = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": []},
    ]}
    assert _is_fresh_user_message(msg) is False


def test_is_fresh_user_message_assistant_role_is_never_fresh():
    msg = {"role": "assistant", "content": "world"}
    assert _is_fresh_user_message(msg) is False


def test_is_fresh_user_message_empty_list_is_fresh():
    """A degenerate empty content list contains no tool_result → still
    a safe cut point under the helper's contract."""
    msg = {"role": "user", "content": []}
    assert _is_fresh_user_message(msg) is True


# ---------------------------------------------------------------------------
# _summarize_messages
# ---------------------------------------------------------------------------


def test_summarize_messages_empty_returns_empty_string():
    assert _summarize_messages([]) == ""


def test_summarize_messages_extracts_initial_user_task():
    msgs = [
        {"role": "user", "content": "swipe right ten times"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    summary = _summarize_messages(msgs)
    assert 'Original task: "swipe right ten times"' in summary
    assert "Summary of earlier browser-agent activity" in summary


def test_summarize_messages_truncates_initial_task_at_300_chars():
    long_task = "x" * 600
    msgs = [{"role": "user", "content": long_task}]
    summary = _summarize_messages(msgs)
    # The quoted task substring must be at most 300 chars.
    start = summary.index('Original task: "') + len('Original task: "')
    end = summary.index('"', start)
    assert end - start <= 300


def test_summarize_messages_counts_tool_calls_with_key_param():
    """Tool calls are bucketed by name; key params are extracted in the
    priority order index → key → url → selector → direction → text."""
    msgs = [
        {"role": "user", "content": "do stuff"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "BrowserClick",
             "input": {"selector": "#a"}},
            {"type": "tool_use", "id": "t2", "name": "BrowserClick",
             "input": {"selector": "#b"}},
            {"type": "tool_use", "id": "t3", "name": "BrowserNavigate",
             "input": {"url": "https://example.com"}},
        ]},
    ]
    summary = _summarize_messages(msgs)
    assert "Actions taken (3 total):" in summary
    # BrowserClick appears twice → "(×2)" suffix on its bucket line.
    assert "BrowserClick(selector=#b) (×2)" in summary
    assert "BrowserNavigate(url=https://example.com)" in summary


def test_summarize_messages_includes_recent_brain_states():
    """ReportProgress.next_goal entries surface under 'Recent intents:'
    (last 5 only)."""
    blocks = []
    for i in range(7):
        blocks.append({
            "type": "tool_use", "id": f"r{i}", "name": "ReportProgress",
            "input": {
                "evaluation_previous": "ok",
                "working_memory": "...",
                "next_goal": f"goal-{i}",
            },
        })
    msgs = [{"role": "assistant", "content": blocks}]
    summary = _summarize_messages(msgs)
    assert "Recent intents:" in summary
    # Last 5 of 7 → goal-2..goal-6 included; goal-0/goal-1 dropped.
    for i in (2, 3, 4, 5, 6):
        assert f"goal-{i}" in summary
    assert "goal-0" not in summary
    assert "goal-1" not in summary


def test_summarize_messages_includes_last_assistant_text():
    msgs = [
        {"role": "assistant", "content": [{"type": "text", "text": "first thought"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "final thought"}]},
    ]
    summary = _summarize_messages(msgs)
    assert "Last update from assistant: final thought" in summary


def test_summarize_messages_truncates_last_assistant_text_at_400():
    long = "y" * 1000
    msgs = [{"role": "assistant", "content": [{"type": "text", "text": long}]}]
    summary = _summarize_messages(msgs)
    line = next(l for l in summary.split("\n") if l.startswith("Last update"))
    payload = line[len("Last update from assistant: "):]
    assert len(payload) <= 400


# ---------------------------------------------------------------------------
# _trim_history_by_turns
# ---------------------------------------------------------------------------


def test_trim_history_under_cap_returns_copy_unchanged():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    out = _trim_history_by_turns(msgs, max_messages=10)
    assert out == msgs
    # Returns a new list (not the same object) — defensive copy.
    assert out is not msgs


def test_trim_history_compacts_when_clean_cut_exists():
    """The classic case: long history, but there's a fresh user-text
    message later in the list → compact prefix into a single summary."""
    msgs = [
        {"role": "user", "content": "task one"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "BrowserClick", "input": {"selector": "#a"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": []},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "done one"}]},
        # Clean boundary: a brand-new user-text message
        {"role": "user", "content": "task two"},
        {"role": "assistant", "content": [{"type": "text", "text": "done two"}]},
    ]
    out = _trim_history_by_turns(msgs, max_messages=3)
    # First message is the synthetic summary, then the verbatim tail.
    assert out[0]["role"] == "user"
    assert isinstance(out[0]["content"], str)
    assert "Summary of earlier browser-agent activity" in out[0]["content"]
    assert "task one" in out[0]["content"]
    assert out[1:] == msgs[4:]


def test_trim_history_no_clean_cut_returns_original():
    """If every user message after [0] is a tool_result (no clean
    boundary anywhere), the helper returns the input unchanged rather
    than corrupting the conversation."""
    msgs = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": []},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": []},
        ]},
    ]
    out = _trim_history_by_turns(msgs, max_messages=2)
    assert out == msgs


def test_trim_history_falls_back_to_latest_cut_when_current_turn_too_big():
    """If the only clean cut leaves a tail that still exceeds the cap,
    we still take that latest cut — better some compaction than none."""
    msgs = [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        # Clean boundary at index 2
        {"role": "user", "content": "second task"},
        # The "current" turn alone is now bigger than the cap.
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": []},
        ]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": "BrowserClick", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": []},
        ]},
    ]
    out = _trim_history_by_turns(msgs, max_messages=3)
    # Even though tail (5 msgs) exceeds cap (3), we still compact
    # everything before the latest clean boundary at index 2.
    assert out[0]["role"] == "user" and isinstance(out[0]["content"], str)
    assert "first task" in out[0]["content"]
    assert out[1:] == msgs[2:]


# ---------------------------------------------------------------------------
# _format_tool_result
# ---------------------------------------------------------------------------


def test_format_tool_result_error_returns_text_block():
    out = _format_tool_result({"error": "no browser connected"}, "BrowserClick")
    assert out == [{"type": "text", "text": "Error: no browser connected"}]


def test_format_tool_result_screenshot_with_image_emits_image_and_text():
    """BrowserScreenshot results carry a base64 PNG; the formatter
    must wrap it in an Anthropic `image` content block."""
    out = _format_tool_result(
        {"image": "AAAA==", "url": "https://example.com"},
        "BrowserScreenshot",
    )
    assert len(out) == 2
    assert out[0]["type"] == "image"
    assert out[0]["source"]["type"] == "base64"
    assert out[0]["source"]["media_type"] == "image/png"
    assert out[0]["source"]["data"] == "AAAA=="
    assert out[1]["type"] == "text"
    assert "https://example.com" in out[1]["text"]


def test_format_tool_result_screenshot_without_image_falls_back_to_text():
    """If a screenshot tool somehow returns no image, fall through to
    the standard text-block path rather than emitting an empty image."""
    out = _format_tool_result({"text": "no image captured"}, "BrowserScreenshot")
    assert out == [{"type": "text", "text": "no image captured"}]


def test_format_tool_result_text_field_passes_through():
    out = _format_tool_result({"text": "clicked successfully"}, "BrowserClick")
    assert out == [{"type": "text", "text": "clicked successfully"}]


def test_format_tool_result_no_text_falls_back_to_json_dump():
    """Last-resort path: arbitrary dict → JSON-stringified text block."""
    out = _format_tool_result({"some": "shape", "value": 1}, "BrowserEvaluate")
    assert len(out) == 1
    assert out[0]["type"] == "text"
    # Should be a JSON dump containing both keys.
    payload = json.loads(out[0]["text"])
    assert payload == {"some": "shape", "value": 1}


# ---------------------------------------------------------------------------
# execute_browser_tool (async)
# ---------------------------------------------------------------------------


async def test_execute_browser_tool_unknown_tool_returns_error():
    """Tools not in ACTION_MAP must short-circuit — the frontend has
    no handler for them and would silently drop the command."""
    out = await execute_browser_tool("BrowserNotARealTool", {}, "b-1", "")
    assert out == {"error": "Unknown browser tool: BrowserNotARealTool"}


async def test_execute_browser_tool_dispatches_through_ws_manager():
    """Known tool → ws_manager.send_browser_command awaited with the
    mapped action, browser_id, params copy, and a tab_id."""
    captured: dict = {}

    async def _fake_send(request_id, action, browser_id, params, tab_id=""):
        captured.update({
            "request_id": request_id,
            "action": action,
            "browser_id": browser_id,
            "params": params,
            "tab_id": tab_id,
        })
        return {"text": "ok"}

    with patch.object(ba.ws_manager, "send_browser_command",
                      AsyncMock(side_effect=_fake_send)) as mock_send:
        result = await execute_browser_tool(
            "BrowserNavigate", {"url": "https://x.com"}, "b-42", "tab-7",
        )
    assert result == {"text": "ok"}
    mock_send.assert_awaited_once()

    # Tool name → mapped action via ACTION_MAP
    assert captured["action"] == "navigate"
    assert captured["browser_id"] == "b-42"
    assert captured["tab_id"] == "tab-7"
    assert captured["params"] == {"url": "https://x.com"}
    # request_id is a hex uuid4 → 32-char lowercase hex
    assert isinstance(captured["request_id"], str)
    assert len(captured["request_id"]) == 32
    assert all(c in "0123456789abcdef" for c in captured["request_id"])


async def test_execute_browser_tool_each_action_map_entry_resolves():
    """Smoke-cover ACTION_MAP: every declared schema tool that's mapped
    must dispatch with its canonical action string."""
    seen: list[str] = []

    async def _fake_send(request_id, action, browser_id, params, tab_id=""):
        seen.append(action)
        return {"text": "ok"}

    with patch.object(ba.ws_manager, "send_browser_command",
                      AsyncMock(side_effect=_fake_send)):
        for tool_name, action in ba.ACTION_MAP.items():
            seen.clear()
            await execute_browser_tool(tool_name, {}, "b-x")
            assert seen == [action], f"{tool_name} routed to {seen}, expected {[action]}"
