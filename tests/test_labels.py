"""Episode labeler helpers, exercised on synthetic episodes only (no private data)."""
from __future__ import annotations

from collections import Counter

import pytest

from sys1.datasets import label_episodes as le


@pytest.fixture(scope="module")
def loaded():
    return le.load_opcodes()


def new_stats():
    return {"steps": 0, "loop": 0, "retry": 0, "src": Counter(), "outcome": Counter(),
            "unmapped": Counter(), "partial_cap": Counter(), "terminal": Counter(), "ops": Counter()}


def test_static_tables_reference_existing_ops(loaded):
    mnemonics, dsco_map = loaded
    static = set(le.HAND.values()) | set(dsco_map.values()) | {le.GENERIC}
    static |= {m for d in le.ACTION.values() for m in d.values()}
    static |= {m for _, m in le.MCP_EMAIL_RULES + le.MCP_DST_RULES}
    assert not sorted(static - mnemonics)


def test_tool_op_routes(loaded):
    _, dsco_map = loaded
    assert le.tool_op("bash", "{}", dsco_map) == ("ENV.INVOKE.SHELL_COMMAND", "hand")
    assert le.tool_op("read_file", {"path": "x"}, dsco_map)[0] == "ENV.READ.FILE"
    assert le.tool_op("no_such_tool_xyz", "{}", dsco_map) == (le.GENERIC, "generic")


def test_outcome_op():
    assert le.outcome_op({"ok": True, "result": "fine"}) == "ACT.RESULT_OK"
    assert le.outcome_op({"ok": False}) == "ACT.RESULT_ERR"
    assert le.outcome_op({"ok": None}) == "ACT.RESULT_ERR"
    assert le.outcome_op({"failure": "timeout", "ok": False}) == "ACT.RESULT_TIMEOUT"
    assert le.outcome_op({"ok": True, "result": "x [spilled: 1]"}) == "ACT.RESULT_PARTIAL"
    capped = "a" * le.EXPORT_RESULT_MAX + "…[+10 bytes]"
    assert le.outcome_op({"ok": True, "result": capped}) == "ACT.RESULT_OK"
    assert le.outcome_op({"ok": True, "result": "short…[+10 bytes]"}) == "ACT.RESULT_PARTIAL"


def test_label_episode_retry_loop_terminal(loaded):
    mnemonics, dsco_map = loaded
    same = {"tool": "grep_files", "args": '{"q": "x"}', "ok": True, "result": "r"}
    ep = {"session": "s1", "goal": "g" * 1000, "end": "final_answer", "steps": [
        {"tool": "bash", "args": '{"cmd": "ls"}', "ok": False},
        {"tool": "bash", "args": '{"cmd": "ls"}', "ok": True, "result": "ok"},
        same, same, same,
    ]}
    stats = new_stats()
    lab = le.label_episode(ep, dsco_map, stats)
    assert lab["n_steps"] == 5
    assert len(lab["goal"]) == le.GOAL_PREVIEW_CHARS
    assert lab["steps"][1]["ops"][0] == "ACT.RETRY.TOOL_CALL"
    assert "CTRL.LOOP_DETECTED" in lab["steps"][4]["ops"]
    assert lab["ops"][-1] == "CTRL.FINAL"
    assert set(lab["ops"]) <= mnemonics
    assert stats["retry"] == 1 and stats["loop"] == 1 and stats["steps"] == 5
    lab2 = le.label_episode({**ep, "end": "max_turns"}, dsco_map, new_stats())
    assert lab2["ops"][-1] == "CTRL.STOP"
