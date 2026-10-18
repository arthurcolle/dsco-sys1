#!/usr/bin/env python3
"""Deterministic rule-based op-code labeling of agent episodes.

Reads data/episodes.jsonl + opcodes.json, writes
data/labels/labeled_episodes.jsonl and data/labels/report.json.
No network, no LLM. Every emitted mnemonic is asserted to exist in opcodes.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from sys1.paths import DATA, OPCODES_JSON

EPISODES = DATA / "episodes.jsonl"
OPCODES = OPCODES_JSON
OUT_DIR = DATA / "labels"

GENERIC = "ACT.CALL"
LOOP_MIN_REPEATS = 3
TRUNC_RE = re.compile(r"…\[\+\d+ bytes\]\s*$")
EXPORT_RESULT_MAX = 768  # export_episodes.py RESULT_MAX: its blob() cap appends the same marker
GOAL_PREVIEW_CHARS = 300

# ---------------------------------------------------------------------------
# Hand table: tool name -> op mnemonic. Takes precedence over the dsco field.
# ---------------------------------------------------------------------------
HAND = {
    # shell / process
    "bash": "ENV.INVOKE.SHELL_COMMAND", "Bash": "ENV.INVOKE.SHELL_COMMAND",
    "run_command": "ENV.INVOKE.SHELL_COMMAND", "ssh_command": "ENV.INVOKE.SHELL_COMMAND",
    "sandbox_run": "ENV.INVOKE.SHELL_COMMAND", "kitty_remote": "ENV.INVOKE.SHELL_COMMAND",
    "BashOutput": "ENV.AWAIT.PROCESS", "spawn_bg": "ENV.CREATE.PROCESS",
    "ps": "ENV.LIST.PROCESS", "process_tree": "ENV.LIST.PROCESS", "lsof": "ENV.INSPECT.PROCESS",
    "python": "ENV.INVOKE.SHELL_COMMAND", "dsco-python-3x": "ENV.INVOKE.SHELL_COMMAND",
    "python_exec": "ENV.INVOKE.SHELL_COMMAND", "eval": "ENV.INVOKE.SHELL_COMMAND",
    "ipc": "SWARM.SEND.MESSAGE_BUS",
    # files
    "read_file": "ENV.READ.FILE", "Read": "ENV.READ.FILE", "head_tail": "ENV.READ.FILE",
    "view_pdf": "PERC.READ.DOCUMENT", "view_image": "CTX.CAPTURE.IMAGE_BLOCK",
    "edit_file": "ENV.UPDATE.FILE", "MultiEdit": "ENV.UPDATE.FILE",
    "str_replace_based_edit_tool": "ENV.UPDATE.FILE", "append_file": "ENV.UPDATE.FILE",
    "ast_edit": "CODE.REFACTOR.FUNCTION",
    "write_file": "ENV.WRITE.FILE", "delete_file": "ENV.DELETE.FILE",
    "mkdir": "ENV.CREATE.DIRECTORY", "download_file": "ENV.FETCH.HTTP_ENDPOINT",
    "list_directory": "ENV.LIST.DIRECTORY", "file_tree": "ENV.LIST.DIRECTORY",
    "find_files": "ENV.LIST.DIRECTORY", "file_info": "ENV.STAT.FILE",
    "file_hash": "ENV.STAT.FILE", "sha256": "ENV.STAT.FILE", "md5": "ENV.STAT.FILE",
    "word_count": "ENV.STAT.FILE", "diff": "ENV.DIFF.FILE", "text_diff": "ENV.DIFF.FILE",
    "cwd": "ENV.CWD",
    # code search / analysis
    "grep_files": "KNOW.SEARCH.CODEBASE", "Grep": "KNOW.SEARCH.CODEBASE",
    "code_search": "KNOW.SEARCH.CODEBASE", "github_search": "KNOW.SEARCH.CODEBASE",
    "code_index": "KNOW.INDEX.CODEBASE", "ast_index": "KNOW.INDEX.CODEBASE",
    "ast_insights": "KNOW.SUMMARIZE.CODEBASE", "api_outline": "CODE.READ.API_CONTRACT",
    "symbol_def": "CODE.SEARCH.SYMBOL", "symbol_refs": "CODE.FIND_REFERENCES.SYMBOL",
    "git_blame": "KNOW.TRACE.CODEBASE", "contract_search": "KNOW.SEARCH.CODEBASE",
    "lint": "CODE.LINT.MODULE", "compile": "CODE.COMPILE", "diagnostics": "CODE.COMPILE",
    # repo / build / verification
    "git": "ENV.INSPECT.REPOSITORY", "make_build": "ENV.BUILD", "test_run": "VERIF.RUN_TESTS",
    "review_diff": "VERIF.AUDIT.CHANGE", "apply_patch": "CODE.CREATE.PATCH",
    "secret_scan": "GOV.AUDIT.SECRET", "env_scan": "GOV.AUDIT.SECRET",
    "dsco_doctor_integrations": "OBS.CHECK.HEALTH_CHECK", "self_inspect": "META.INSPECT.SELF_MODEL",
    # http / network
    "http_request": "ENV.FETCH.HTTP_ENDPOINT", "curl_raw": "ENV.FETCH.HTTP_ENDPOINT",
    "network": "ENV.INSPECT.HTTP_ENDPOINT", "net": "ENV.INSPECT.HTTP_ENDPOINT",
    "net_probe": "ENV.INSPECT.HTTP_ENDPOINT", "port_check": "ENV.INSPECT.HTTP_ENDPOINT",
    "url_parse": "DATA.PARSE.JSON_DOCUMENT",
    # system info
    "hostname": "ENV.PLATFORM", "sysinfo": "ENV.PLATFORM", "system_profiler": "ENV.PLATFORM",
    "timestamp": "TIME.NOW", "date": "TIME.NOW", "cron_parse": "TIME.VALIDATE.CRON_RULE",
    # context management
    "context_recall": "CTX.UNMASK.OBSERVATION_BLOCK", "context_status": "CTX.MEASURE.CONTEXT",
    "context_evict": "CTX.EVICT.MESSAGE", "context_compact": "CTX.COMPRESS.CONTEXT",
    "conversation_context": "CTX.INSPECT.CONTEXT", "token_audit": "BUD.MEASURE.TOKEN_BUDGET",
    "evict_tools": "ACT.EVICT.TOOLSET",
    # goals / plans
    "get_goal": "GOAL.INSPECT.GOAL", "goal_queue": "GOAL.INSPECT.GOAL_QUEUE",
    "update_goal": "GOAL.REVISE.GOAL", "persistent_directive": "GOAL.SET.CONSTRAINT",
    "TodoWrite": "PLAN.CREATE.PLAN", "EnterPlanMode": "PLAN.CREATE.PLAN",
    "ExitPlanMode": "PLAN.COMMIT.PLAN", "workflow": "PLAN.CREATE.PLAN",
    "StartOfLoopConstruct": "CTRL.CREATE.SUBROUTINE", "EndOfLoopConstruct": "CTRL.CHECK.ITERATION",
    "ooda": "CTRL.INSPECT.LOOP", "strategic_assess": "META.REFLECT.STRATEGY",
    "executive_decision": "UNC.CHOICE", "risk_gate": "UNC.THRESHOLD.RISK",
    "prediction": "UNC.QUANTIFY.PROBABILITY", "self_assess": "META.CALIBRATE.SELF_MODEL",
    "self_analyze": "META.INSPECT.SELF_MODEL", "self_improve": "META.REVISE.STRATEGY",
    "meta_optimize": "META.REVISE.STRATEGY", "research_probe": "PLAN.EXPLORE",
    # tool discovery
    "discover_tools": "ACT.DISCOVER.TOOL", "load_tools": "ACT.LOAD.TOOLSET",
    "invoke_tool": "ACT.INVOKE.TOOL", "discover_integrations": "ACT.DISCOVER.MCP_SERVER",
    # swarm
    "agent": "SWARM.SPAWN.SUBAGENT", "Agent": "SWARM.SPAWN.SUBAGENT", "Task": "SWARM.DELEGATE.SUBAGENT",
    "dispatch_agent": "SWARM.DELEGATE.SUBAGENT", "hermes_agent": "SWARM.DELEGATE.PEER_AGENT",
    "swarm": "SWARM.FAN_OUT", "agent_wait": "SWARM.AWAIT.SUBAGENT",
    "talons": "SWARM.ROLE_ASSIGN", "pheromone": "SWARM.WRITE.SHARED_BLACKBOARD",
    "wings_talons_status": "SWARM.MEASURE.AGENT_GROUP", "vos_status": "DUR.INSPECT.SUPERVISOR",
    "ol_call": "SWARM.DELEGATE.PEER_AGENT", "graphsub": "SWARM.DELEGATE.WORKER",
    # web / knowledge
    "jina_search": "KNOW.SEARCH.WEB", "jina_ai_search": "KNOW.SEARCH.WEB",
    "parallel_search": "KNOW.SEARCH.WEB", "parallel_ai_search": "KNOW.SEARCH.WEB",
    "tavily_search": "KNOW.SEARCH.WEB", "WebSearch": "KNOW.SEARCH.WEB",
    "jina_read": "KNOW.FETCH.WEB", "jina_ai_reader": "KNOW.FETCH.WEB", "WebFetch": "KNOW.FETCH.WEB",
    "parallel_ai_extract": "KNOW.FETCH.WEB",
    "jina_ai_research": "KNOW.MULTI_HOP", "parallel_ai_research": "KNOW.MULTI_HOP",
    "jina_ai_constellation": "KNOW.MULTI_HOP", "parallel_ai_constellation": "KNOW.MULTI_HOP",
    "parallel_ai_task_create": "TIME.SCHEDULE.JOB", "parallel_ai_task_result": "TIME.INSPECT.JOB",
    "parallel_ai_chat": "ROUTE.ROUTE.REQUEST", "jina_ai_chat": "ROUTE.ROUTE.REQUEST",
    "jina_ai_live_kb": "KNOW.SEARCH.KNOWLEDGE_BASE", "knowledge_base": "KNOW.SEARCH.KNOWLEDGE_BASE",
    "jina_ai_embed": "MEM.INDEX.VECTOR_INDEX", "jina_ai_batch_embed_submit": "MEM.INDEX.VECTOR_INDEX",
    "jina_ai_models_list": "ROUTE.LIST.MODEL", "openrouter_models": "ROUTE.LIST.MODEL",
    "openrouter_lanes": "ROUTE.LIST.PROVIDER", "avian": "ROUTE.ROUTE.REQUEST",
    "browser": "WEB.READ.PAGE",
    # domain data sources -> dataset search
    "weather": "KNOW.SEARCH.DATASET", "nws": "KNOW.SEARCH.DATASET",
    "weather_batch": "KNOW.SEARCH.DATASET", "synoptic": "KNOW.SEARCH.DATASET",
    "polymarket": "KNOW.SEARCH.DATASET", "kalshi": "KNOW.SEARCH.DATASET",
    "alpha_vantage": "KNOW.SEARCH.DATASET", "sqlite": "DATA.LOAD.TABLE",
    "signal_process": "DATA.TRANSFORM.RECORD_SET", "plot": "DATA.SUMMARIZE.RECORD_SET",
    "value_ledger": "ECON.READ.REVENUE_EVENT",
    # data / text utilities
    "jq": "DATA.TRANSFORM.JSON_DOCUMENT", "json_format": "DATA.NORMALIZE.JSON_DOCUMENT",
    "template_render": "DATA.TRANSFORM.JSON_DOCUMENT", "string_transform": "DATA.NORMALIZE.TEXT_CORPUS",
    "regex_match": "DATA.FILTER.TEXT_CORPUS", "base64_tool": "DATA.TRANSFORM.RECORD_SET",
    "calc": "REASON.ESTIMATE.QUANTITY", "semver_compare": "CODE.INSPECT.DEPENDENCY",
    "random_bytes": "SEC.ENCRYPT.PAYLOAD",
    # memory / learning
    "scratchpad": "MEM.WRITE.SCRATCHPAD", "ctx_put": "MEM.STORE.BLOB_STORE",
    "ctx_get": "MEM.RECALL.BLOB_STORE", "session_memory": "MEM.RECALL.EPISODIC_MEMORY",
    "memory_tier": "MEM.MEASURE.WORKING_MEMORY", "playbook_add": "LEARN.EXTRACT.LESSON",
    "playbook_search": "LEARN.RECALL.LESSON", "bg_learn": "LEARN.EXTRACT.LESSON",
    # governance
    "governance": "GOV.ENFORCE.POLICY",
    # UI
    "computer": "UI.FOREGROUND_CONTROL", "app_control": "UI.BACKGROUND_CONTROL",
    "clipboard": "UI.READ.TEXT_FIELD", "openai_image_generate": "COMM.CREATE.REPORT",
    "pets": "UI.OPEN.APPLICATION",
    # user dialog
    "AskUserQuestion": "COMM.ASK.USER",
}

# Arg-sensitive refinements: tool -> {action: mnemonic}.
ACTION = {
    "agent": {"status": "SWARM.INSPECT.SUBAGENT", "output": "SWARM.COLLECT.SUBAGENT",
              "spawn": "SWARM.SPAWN.SUBAGENT", "kill": "SWARM.TERMINATE.SUBAGENT",
              "wait": "SWARM.AWAIT.SUBAGENT"},
    "swarm": {"status": "SWARM.MEASURE.AGENT_GROUP", "inspect": "SWARM.MEASURE.AGENT_GROUP",
              "collect": "SWARM.COLLECT.AGENT_GROUP", "budget": "SWARM.MEASURE.AGENT_GROUP",
              "map_reduce": "SWARM.MAP_REDUCE", "create": "SWARM.SPAWN.AGENT_GROUP",
              "spawn_executor": "SWARM.SPAWN.WORKER"},
    "goal_queue": {"decompose": "GOAL.DECOMPOSE.GOAL", "complete": "GOAL.COMPLETE.TASK",
                   "status": "GOAL.INSPECT.GOAL_QUEUE", "block": "GOAL.BLOCKED",
                   "checkpoint": "GOAL.SNAPSHOT.GOAL_QUEUE", "fail": "GOAL.ABANDON.TASK"},
    "git": {"diff": "ENV.DIFF.REPOSITORY", "add": "ENV.COMMIT.REPOSITORY",
            "commit": "ENV.COMMIT.REPOSITORY", "status": "ENV.INSPECT.REPOSITORY",
            "log": "ENV.INSPECT.REPOSITORY"},
    "knowledge_base": {"ingest": "KNOW.STORE.KNOWLEDGE_BASE", "arxiv_ingest": "KNOW.STORE.KNOWLEDGE_BASE",
                       "get": "KNOW.CITE.KNOWLEDGE_BASE", "list": "KNOW.INDEX.KNOWLEDGE_BASE"},
    "scratchpad": {"read": "MEM.READ.SCRATCHPAD", "get": "MEM.READ.SCRATCHPAD",
                   "list": "MEM.READ.SCRATCHPAD", "status": "MEM.READ.SCRATCHPAD",
                   "clear": "MEM.CLEAR.SCRATCHPAD"},
    "computer": {"screenshot": "PERC.CAPTURE.SCREEN", "zoom": "UI.ZOOM.SCREEN_REGION"},
    "app_control": {"press": "UI.PRESS.BUTTON", "type": "UI.TYPE_INTO.TEXT_FIELD",
                    "screenshot": "UI.CAPTURE.WINDOW"},
    "governance": {"status": "GOV.INSPECT.POLICY", "audit": "GOV.AUDIT.POLICY",
                   "authorize": "GOV.CHECK.APPROVAL", "checkpoint": "DUR.SNAPSHOT.AGENT_STATE"},
    "workflow": {"checkpoint": "CTRL.CREATE.CHECKPOINT", "plan": "PLAN.CREATE.PLAN"},
    "talons": {"goal": "GOAL.SET.GOAL", "advance": "GOAL.NEXT_BEST_ACTION"},
    "ooda": {"observe": "PERC.READ.OBSERVATION", "orient": "STATE.UPDATE.BELIEF_STATE",
             "decide": "PLAN.SELECT.OPTION", "complete": "GOAL.DONE_CHECK"},
}

# MCP families: (regex on the part after mcp__<server>__, mnemonic). First match wins.
MCP_EMAIL_SERVERS = ("cloudmail-mcp", "email-remote", "email")
MCP_EMAIL_RULES = [
    (r"^(send_|reply|forward|cancel_send)", "COMM.NOTIFY.USER"),
    (r"(search|thread|by_date|uid_search|sort|similar)", "KNOW.SEARCH.DATASET"),
    (r"^(read_email|get_email_body|get_mime|get_attachment|save_all_attachments|uid_fetch|get_headers)", "PERC.READ.DOCUMENT"),
    (r"^(inbox|list_|folder_tree|status_folder|count_unread)", "KNOW.INSPECT.DATASET"),
    (r"^(top_|folder_stats|email_activity|uid_summary|extract_contacts|cost_)", "KNOW.SUMMARIZE.DATASET"),
    (r"^(flag_|set_vip|mark_|store_flags)", "STATE.UPDATE.ENTITY"),
    (r"^(rules_|safe_mode)", "GOV.INSPECT.POLICY"),
    (r"(index|pool_|periodic_)", "OBS.CHECK.HEALTH_CHECK"),
]
MCP_DST_RULES = [  # distributed-systems-tools and misc MCP servers
    (r"^(search_tools|query_tools|get_tool|batch_get_tools|list_tool|get_registry_stats|search_all)", "ACT.DISCOVER.TOOL"),
    (r"^(execute_tool|batch_execute|unit_execute)", "ACT.INVOKE.TOOL"),
    (r"^(learn_skill)", "LEARN.EXTRACT.SKILL"),
    (r"^(apply_skill)", "SKILL.INVOKE.SKILL_INVOCATION"),
    (r"^(get_job_status|await_run|get_job_result)", "TIME.INSPECT.JOB"),
    (r"^(submit_job|run_task|research_task)", "TIME.SCHEDULE.JOB"),
    (r"(deep_?search|parallel_search|multi_search|youtube_social_search|search_arxiv|^search$)", "KNOW.SEARCH.WEB"),
    (r"^(read_url|primer)", "KNOW.FETCH.WEB"),
    (r"(weather)", "KNOW.SEARCH.DATASET"),
    (r"^(sendgrid_email_list|sendgrid_email_stats)", "KNOW.INSPECT.DATASET"),
    (r"^(sendgrid_email_validate)", "DATA.VALIDATE.RECORD_SET"),
    (r"^(process_video|list_videos)", "PERC.READ.DOCUMENT"),
    (r"^(process_document)", "PERC.READ.DOCUMENT"),
    (r"^(record_episode)", "MEM.STORE.EPISODIC_MEMORY"),
    (r"^(update_execution_follow_up_status)", "GOAL.UPDATE.TASK"),
    (r"^(get_execution_policy_decisions)", "GOV.INSPECT.POLICY"),
    (r"^(chat_completion)", "ROUTE.ROUTE.REQUEST"),
    (r"^(list_models)", "ROUTE.LIST.MODEL"),
    (r"^(browser_take_screenshot)", "WEB.CAPTURE.PAGE"),
]


def load_opcodes():
    ops = json.loads(OPCODES.read_text())["ops"]
    mnemonics = {o["mnemonic"] for o in ops}
    dsco_map = {}
    for o in ops:
        d = o.get("dsco")
        if not d:
            continue
        m = re.fullmatch(r"([A-Za-z_][\w-]*)(?:\((\w+)\))?", d.strip())
        if not m:
            continue  # event names like tool.call.created / llm.response.*
        tool, action = m.group(1), m.group(2)
        if action:
            ACTION.setdefault(tool, {}).setdefault(action, o["mnemonic"])
        else:
            dsco_map.setdefault(tool, o["mnemonic"])
    return mnemonics, dsco_map


def parse_args(raw):
    try:
        a = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    return a if isinstance(a, dict) else {}


def tool_op(tool, args, dsco_map):
    """Return (mnemonic, source) for a tool call."""
    a = parse_args(args)
    action = a.get("action") or a.get("operation")
    if tool == "context_recall" and "key" not in a and "list" in a:
        return "CTX.SEARCH.ARCHIVE_PAGE", "action"
    if tool in ACTION and isinstance(action, str) and action in ACTION[tool]:
        return ACTION[tool][action], "action"
    if tool in HAND:
        return HAND[tool], "hand"
    if tool in dsco_map:
        return dsco_map[tool], "dsco"
    if tool.startswith("mcp__"):
        parts = tool.split("__", 2)
        if len(parts) == 3:
            server, name = parts[1], parts[2]
            rules = MCP_EMAIL_RULES if server in MCP_EMAIL_SERVERS else MCP_DST_RULES
            for pat, mn in rules:
                if re.search(pat, name):
                    return mn, "mcp"
    return GENERIC, "generic"


def outcome_op(step):
    result = step.get("result") or ""
    if step.get("failure") == "timeout":
        return "ACT.RESULT_TIMEOUT"
    if step.get("ok") is not True:  # False, or None (call never completed)
        return "ACT.RESULT_ERR"
    if "[spilled:" in result:  # dsco paged the tool output out of context
        return "ACT.RESULT_PARTIAL"
    if TRUNC_RE.search(result):
        # export_episodes.py cuts every result at EXPORT_RESULT_MAX chars with the
        # same marker; that is an export artifact, not a truncated tool result.
        prefix = result.rsplit("…[+", 1)[0]
        if len(prefix) != EXPORT_RESULT_MAX:
            return "ACT.RESULT_PARTIAL"
    return "ACT.RESULT_OK"


def label_episode(ep, dsco_map, stats):
    steps_out, flat = [], []
    prev_key, run_len, prev_failed, prev_tool = None, 0, False, None
    for i, s in enumerate(ep.get("steps") or []):
        tool = s.get("tool") or ""
        ops = []
        if prev_failed and tool == prev_tool:
            ops.append("ACT.RETRY.TOOL_CALL")
            stats["retry"] += 1
        mn, src = tool_op(tool, s.get("args"), dsco_map)
        ops.append(mn)
        out = outcome_op(s)
        ops.append(out)
        key = (tool, s.get("args"))
        run_len = run_len + 1 if key == prev_key else 1
        if run_len == LOOP_MIN_REPEATS:
            ops.append("CTRL.LOOP_DETECTED")
            stats["loop"] += 1
        prev_key, prev_tool = key, tool
        prev_failed = out in ("ACT.RESULT_ERR", "ACT.RESULT_TIMEOUT")

        stats["steps"] += 1
        stats["src"][src] += 1
        stats["outcome"][out] += 1
        if out == "ACT.RESULT_PARTIAL":
            prefix = (s.get("result") or "").rsplit("…[+", 1)[0]
            stats["partial_cap"]["export_cap_768" if len(prefix) == EXPORT_RESULT_MAX else "other"] += 1
        if mn == GENERIC:
            stats["unmapped"][tool] += 1
        steps_out.append({"i": i, "tool": tool, "ops": ops})
        flat.extend(ops)
    term = "CTRL.FINAL" if ep.get("end") == "final_answer" else "CTRL.STOP"
    flat.append(term)
    stats["terminal"][term] += 1
    for o in flat:
        stats["ops"][o] += 1
    goal = ep.get("goal") or ""
    return {
        "session": ep.get("session"),
        "goal": goal[:GOAL_PREVIEW_CHARS],
        "model": ep.get("model"),
        "end": ep.get("end"),
        "n_steps": len(steps_out),
        "steps": steps_out,
        "ops": flat,
    }


def main():
    mnemonics, dsco_map = load_opcodes()
    # Static check: every mnemonic in our tables must exist.
    static = set(HAND.values()) | set(dsco_map.values()) | {GENERIC}
    static |= {m for d in ACTION.values() for m in d.values()}
    static |= {m for _, m in MCP_EMAIL_RULES + MCP_DST_RULES}
    static |= {"ACT.RESULT_OK", "ACT.RESULT_ERR", "ACT.RESULT_TIMEOUT", "ACT.RESULT_PARTIAL",
               "ACT.RETRY.TOOL_CALL", "CTRL.LOOP_DETECTED", "CTRL.FINAL", "CTRL.STOP",
               "CTX.SEARCH.ARCHIVE_PAGE"}
    bad = sorted(static - mnemonics)
    assert not bad, f"mnemonics missing from opcodes.json: {bad}"

    stats = {"steps": 0, "loop": 0, "retry": 0, "src": Counter(), "outcome": Counter(),
             "unmapped": Counter(), "partial_cap": Counter(), "terminal": Counter(), "ops": Counter()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples, n_eps = [], 0
    with EPISODES.open() as fin, (OUT_DIR / "labeled_episodes.jsonl").open("w") as fout:
        for line in fin:
            ep = json.loads(line)
            lab = label_episode(ep, dsco_map, stats)
            for o in lab["ops"]:
                assert o in mnemonics, o
            fout.write(json.dumps(lab, ensure_ascii=False) + "\n")
            n_eps += 1
            examples.append(lab)

    # 3 deterministic examples: short, with a retry/loop, and an interrupted one.
    def pick(pred):
        for e in examples:
            if pred(e):
                return e
    chosen = [
        pick(lambda e: 3 <= e["n_steps"] <= 6 and e["end"] == "final_answer"),
        pick(lambda e: 4 <= e["n_steps"] <= 12 and "ACT.RETRY.TOOL_CALL" in e["ops"]
             and "CTRL.LOOP_DETECTED" in e["ops"]) or
        pick(lambda e: 4 <= e["n_steps"] <= 12 and "ACT.RETRY.TOOL_CALL" in e["ops"]),
        pick(lambda e: 2 <= e["n_steps"] <= 8 and e["end"] != "final_answer"),
    ]
    ex_out = []
    for e in chosen:
        if e is None:
            continue
        ex_out.append({"session": e["session"], "goal": e["goal"][:120], "end": e["end"],
                       "n_steps": e["n_steps"], "steps": e["steps"][:10], "terminal": e["ops"][-1]})

    total = stats["steps"]
    specific = total - sum(stats["unmapped"].values())
    report = {
        "episodes": n_eps,
        "total_steps": total,
        "specific_steps": specific,
        "specific_share": round(specific / total, 6) if total else 0.0,
        "distinct_tools_unmapped": len(stats["unmapped"]),
        "top_unmapped_tools": stats["unmapped"].most_common(30),
        "mapping_source": dict(stats["src"]),
        "top30_ops": stats["ops"].most_common(30),
        "distinct_ops_emitted": len(stats["ops"]),
        "outcome_distribution": dict(stats["outcome"]),
        "partial_breakdown": dict(stats["partial_cap"]),
        "terminal_distribution": dict(stats["terminal"]),
        "loop_detected": stats["loop"],
        "retry_tool_call": stats["retry"],
        "rules": {
            "outcome_priority": "timeout > err (ok false/None) > partial (dsco spill marker, or a truncation marker not produced by the export cap) > ok",
            "loop": f"same tool+raw args {LOOP_MIN_REPEATS}x consecutively; emitted once per run at the 3rd step",
            "retry": "previous step ERR/TIMEOUT and current step uses same tool",
        },
        "examples": ex_out,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("episodes", "total_steps", "specific_share",
                                              "top_unmapped_tools", "mapping_source",
                                              "outcome_distribution", "partial_breakdown", "terminal_distribution",
                                              "loop_detected", "retry_tool_call")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
