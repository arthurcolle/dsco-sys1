"""Synthetic agent-format corpus for training the sys1 BPE base.

Generates text in the shapes sys1 reads at inference time, so frequent agent
fragments (JSON keys, role markers, op-code prose, policy phrasing) earn merges:

  opcodes      every op-code definition from sys1/vocab/opcodes.json rendered as
               prose, JSON, doc comments, Q&A and compositional decompositions
  decisions    sys1.jevmr situations (all fractal variants) rendered as state +
               options + op-code tokens, in control-token, chat and JSON forms
  transcripts  role-marked tool-call transcripts using the role/control tokens of
               sys1/tokenizer/spec.py, with plausible synthetic JSON arguments and results
  schemas      JSON schemas for tools and op-code signatures in several tool-spec dialects

Everything is synthetic. Tool *names* come from the reserved tool region of the spec;
no arguments, results or text are read from private logs.

Output: data/corpus/agent/{opcodes,decisions,transcripts,schemas}.txt, heldout.txt
(5% of samples drawn uniformly across categories) and MANIFEST.json. One sample per
line, newlines escaped as the two characters backslash-n.

  python -m sys1.tokenizer.corpus_agent [--total-mb 190] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import time
import zlib

from sys1.paths import COMPOSITIONAL_JSON, DATA, OPCODES_JSON
from sys1.tokenizer import spec

OUT = DATA / "corpus" / "agent"
CAP_MB = 200
HELDOUT = 0.05
SHARES = {"opcodes": 0.11, "decisions": 0.24, "transcripts": 0.52, "schemas": 0.13}
VERSION = 1

OPS_LIST = json.loads(OPCODES_JSON.read_text())["ops"]
OPS = {o["mnemonic"]: o for o in OPS_LIST}
DECOMP = json.loads(COMPOSITIONAL_JSON.read_text())["decomposition"]
TOOL_TOKENS = {t[len("<|tool:"):-2] for t in spec._tools()}
TOOL_TO_OP = {}
for _o in OPS_LIST:
    if _o["dsco"]:
        TOOL_TO_OP.setdefault(re.split(r"[(.]", _o["dsco"])[0], _o["mnemonic"])

# ---------------------------------------------------------------- word pools
WORDS = ("config parser cache queue worker session token router handler client server buffer index "
         "schema record ledger invoice account user order payment report metric event stream batch job "
         "task plan goal agent model prompt context memory snapshot checkpoint retry backoff limit quota "
         "budget policy rule gate audit trace span log alert monitor health probe deploy release build test "
         "fixture mock adapter bridge plugin module package bundle asset image upload download sync merge "
         "diff patch commit branch tag review label score rank filter query search fetch store table column "
         "row key value hash lock mutex channel socket request response header payload codec frame packet "
         "shard replica leader follower lease timer clock window cursor page offset vector embedding graph "
         "node edge path route map reduce sort split join parse render format escape encode decode").split()
DIRS = ["src", "lib", "pkg", "internal", "app", "tests", "scripts", "cmd", "core", "api", "services/auth",
        "services/billing", "web/components", "tools", "docs", "src/net", "src/storage", "include", "migrations",
        "examples", "bench", "server/handlers", "client/src", "infra/terraform", "config"]
LANGS = {"py": "python", "c": "c", "h": "c", "go": "go", "rs": "rust", "ts": "typescript", "js": "javascript",
         "rb": "ruby", "java": "java", "sql": "sql", "yaml": "yaml", "toml": "toml", "md": "markdown", "sh": "bash"}
EXTS = list(LANGS)
HOSTS = ["example.com", "api.example.com", "docs.example.org", "status.example.net", "registry.example.io",
         "pypi.org", "github.com", "developer.mozilla.org", "en.wikipedia.org", "arxiv.org", "crates.io",
         "pkg.go.dev", "npmjs.com", "docs.python.org", "localhost:8080", "127.0.0.1:9000"]
PEOPLE = ["alice", "bob", "carol", "dmitri", "eve", "farah", "gus", "hana", "ivan", "jun", "kemal", "lena"]
ERRORS = [
    ("timeout", "operation timed out after {n}s"),
    ("rate_limited", "HTTP 429 Too Many Requests; retry-after {n}"),
    ("not_found", "No such file or directory: '{path}'"),
    ("invalid_arguments", "argument '{word}' is required"),
    ("permission_denied", "capability policy denies fs_write for this tier"),
    ("connection_reset", "connection reset by peer"),
    ("exit_code", "command exited with status {n}"),
    ("parse_error", "JSON parse error at line {n} column {m}"),
]
TASKS = [
    "fix the failing {word} test in {path}", "add a --{word} flag to the CLI", "rename {ident} to {ident2} across the repo",
    "find out why the {word} {word2} is slow", "add retry with exponential backoff to the {word} client",
    "write a migration that adds a {word}_id column to the {word2} table", "summarize the open issues about {word}",
    "check whether {pkg} has a newer release", "add input validation to {ident}", "document the {word} module",
    "profile the {word} benchmark and report the hot path", "remove the unused {word} helper",
    "make {ident} thread-safe", "port {path} from callbacks to async", "add a health endpoint for the {word} service",
    "investigate the {word} crash reported in CI", "update the README section on {word}",
    "convert the {word} config from JSON to TOML", "deduplicate rows in the {word} export",
    "split {path} into smaller modules", "audit the repo for leaked secrets", "upgrade {pkg} to the latest minor version",
]
PKGS = ["requests", "numpy", "serde", "tokio", "express", "pydantic", "fastapi", "lodash", "react", "sqlalchemy",
        "zod", "clap", "anyhow", "pytest", "httpx", "rich", "orjson", "uvloop", "jsonschema", "protobuf"]


def word(r):
    return r.choice(WORDS)


def ident(r, style=None):
    parts = [word(r) for _ in range(r.randint(1, 3))]
    style = style or r.choice(["snake", "snake", "camel", "pascal"])
    if style == "snake":
        return "_".join(parts)
    if style == "camel":
        return parts[0] + "".join(p.title() for p in parts[1:])
    return "".join(p.title() for p in parts)


def path(r, ext=None):
    ext = ext or r.choice(EXTS[:9])
    return f"{r.choice(DIRS)}/{ident(r, 'snake')}.{ext}"


def hexid(r, n=8):
    return "".join(r.choice("0123456789abcdef") for _ in range(n))


def url(r):
    return f"https://{r.choice(HOSTS[:14])}/{'/'.join(word(r) for _ in range(r.randint(1, 3)))}"


def ts(r):
    return (f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:"
            f"{r.randint(0, 59):02d}:{r.randint(0, 59):02d}Z")


def phrase(r, lo=3, hi=9):
    return " ".join(word(r) for _ in range(r.randint(lo, hi)))


def sentence(r):
    return phrase(r).capitalize() + "."


def fill(r, template):
    return template.format(word=word(r), word2=word(r), path=path(r), ident=ident(r), ident2=ident(r),
                           pkg=r.choice(PKGS), n=r.randint(1, 120), m=r.randint(1, 80))


def code_snippet(r, lang):
    f, a, b = ident(r, "snake"), word(r), word(r)
    if lang == "python":
        return (f"def {f}({a}, {b}=None):\n    \"\"\"{sentence(r)}\"\"\"\n    if {b} is None:\n        {b} = {{}}\n"
                f"    for k, v in {a}.items():\n        {b}[k] = v\n    return {b}\n")
    if lang == "c":
        return (f"static int {f}(const char *{a}, size_t {b}) {{\n    if (!{a}) return -1;\n"
                f"    for (size_t i = 0; i < {b}; i++) {{\n        if ({a}[i] == '\\0') return (int)i;\n    }}\n"
                f"    return 0;\n}}\n")
    if lang == "go":
        return (f"func {ident(r, 'pascal')}({a} []string) (map[string]int, error) {{\n\tout := make(map[string]int)\n"
                f"\tfor _, s := range {a} {{\n\t\tout[s]++\n\t}}\n\treturn out, nil\n}}\n")
    if lang == "rust":
        return (f"pub fn {f}({a}: &[u8]) -> Result<usize, Error> {{\n    let n = {a}.iter().filter(|b| **b == b'\\n').count();\n"
                f"    Ok(n)\n}}\n")
    if lang in ("typescript", "javascript"):
        return (f"export async function {ident(r, 'camel')}({a}{': string' if lang == 'typescript' else ''}) {{\n"
                f"  const res = await fetch(`/api/{b}/${{{a}}}`);\n  if (!res.ok) throw new Error(res.statusText);\n"
                f"  return res.json();\n}}\n")
    if lang == "sql":
        return f"ALTER TABLE {a}s ADD COLUMN {b}_id BIGINT;\nCREATE INDEX idx_{a}s_{b}_id ON {a}s ({b}_id);\n"
    return f"# {sentence(r)}\n{a}: {b}\n{word(r)}: {r.randint(1, 99)}\n"


# ---------------------------------------------------------------- operand / return values
def operand_value(r, name, typ):
    if typ == "id":
        return f"{name if name not in ('ref', 'a', 'b') else word(r)}_{hexid(r, 6)}"
    if typ == "int":
        return r.choice([1, 2, 3, 5, 8, 10, 16, 32, 64, 100, 256, 1024, 4096, 8000])
    if typ == "float":
        return round(r.uniform(0.01, 50), 2)
    if typ == "time":
        return ts(r)
    if typ == "ref":
        return f"{r.choice(['file', 'msg', 'obs', 'run', 'doc', 'tool'])}:{hexid(r)}"
    if typ == "json":
        return {word(r): r.choice([word(r), r.randint(0, 99), True, None]) for _ in range(r.randint(1, 3))}
    if typ == "list":
        return [word(r) for _ in range(r.randint(1, 4))]
    if typ == "blob":
        return "".join(r.choice(string.ascii_letters + string.digits) for _ in range(24)) + "=="
    if typ == "bool":
        return r.random() < 0.5
    if name == "url":
        return url(r)
    if name in ("query", "claim", "description", "criterion", "rubric", "predicate", "reason"):
        return phrase(r, 3, 8)
    if name == "scope":
        return r.choice([path(r), r.choice(DIRS), "repo", "session", "turn"])
    return ident(r, "snake")


def parse_operand(spec_str):
    name, _, typ = spec_str.partition(":")
    return name, typ or "str"


def op_args(r, op):
    return {n: operand_value(r, n, t) for n, t in map(parse_operand, op["operands"])}


def field_value(r, f):
    f = f.strip()
    if f in ("confidence", "p", "score", "value"):
        return round(r.uniform(0.05, 0.99), 2)
    if f in ("pass", "ok"):
        return r.random() < 0.7
    if f in ("tokens", "bytes", "count"):
        return r.randint(10, 90000)
    if f in ("text", "preview", "summary"):
        return phrase(r, 4, 12)
    if f in ("evidence", "key", "winner", "id"):
        return f"{word(r)}:{hexid(r, 6)}"
    if f in ("before", "after"):
        return hexid(r, 12)
    return phrase(r, 1, 3)


def return_value(r, ret):
    many = ret.endswith("[]") or "[]{" in ret
    base, _, fields = ret.replace("[]", "").partition("{")
    fields = [x for x in fields.rstrip("}").split(",") if x.strip()]

    def one():
        if fields:
            return {x.strip(): field_value(r, x) for x in fields}
        if base == "Ack":
            return {"ack": True}
        if base in ("Ref", "SnapshotId", "Key"):
            return {"id": f"{word(r)}_{hexid(r, 8)}"}
        if base == "Observation":
            return {"observed_at": ts(r), "value": phrase(r, 3, 10)}
        if base == "Answer":
            return {"answer": sentence(r)}
        if base == "Diff":
            return {"diff": f"--- a/{path(r)}\n+++ b/{path(r)}\n@@ -1,3 +1,4 @@\n+{phrase(r)}"}
        return {"result": phrase(r, 2, 6), "status": "ok"}
    return [one() for _ in range(r.randint(1, 3))] if many else one()


# ---------------------------------------------------------------- tool library (synthetic schemas)
def P(typ, desc, kind=None, enum=None):
    d = {"type": typ, "description": desc}
    if enum:
        d["enum"] = enum
    if kind:
        d["x-kind"] = kind
    return d


TOOLS = {
    "bash": ("Run a shell command in the workspace and return stdout, stderr and the exit code.",
             {"command": P("string", "Shell command to execute.", "command"),
              "timeout_ms": P("integer", "Hard timeout in milliseconds.", "timeout")}, ["command"]),
    "read_file": ("Read a text file, optionally a line range.",
                  {"path": P("string", "Path relative to the workspace root.", "path"),
                   "offset": P("integer", "First line to read (1-based).", "line"),
                   "limit": P("integer", "Maximum number of lines.", "small")}, ["path"]),
    "write_file": ("Create or overwrite a file with the given content.",
                   {"path": P("string", "Destination path.", "path"),
                    "content": P("string", "Full file content.", "code")}, ["path", "content"]),
    "edit_file": ("Replace an exact string in a file.",
                  {"path": P("string", "File to edit.", "path"), "old_string": P("string", "Exact text to replace.", "code_line"),
                   "new_string": P("string", "Replacement text.", "code_line"),
                   "replace_all": P("boolean", "Replace every occurrence.", "bool")}, ["path", "old_string", "new_string"]),
    "append_file": ("Append text to the end of a file.",
                    {"path": P("string", "File to append to.", "path"), "content": P("string", "Text to append.", "text")},
                    ["path", "content"]),
    "grep_files": ("Search file contents with a regular expression.",
                   {"pattern": P("string", "Regular expression.", "pattern"), "path": P("string", "Directory to search.", "dir"),
                    "glob": P("string", "Optional file glob filter.", "glob"),
                    "max_results": P("integer", "Cap on returned matches.", "small")}, ["pattern"]),
    "find_files": ("Find files whose names match a glob.",
                   {"glob": P("string", "Glob pattern such as **/*.py.", "glob"), "path": P("string", "Root directory.", "dir")},
                   ["glob"]),
    "list_directory": ("List entries in a directory.",
                       {"path": P("string", "Directory path.", "dir"), "depth": P("integer", "Recursion depth.", "tiny")}, ["path"]),
    "file_info": ("Return size, type and modification time of a path.", {"path": P("string", "Path to inspect.", "path")}, ["path"]),
    "git": ("Run a read-only or local git subcommand.",
            {"subcommand": P("string", "git subcommand.", None, ["status", "diff", "log", "show", "blame", "branch"]),
             "args": P("array", "Extra arguments.", "gitargs")}, ["subcommand"]),
    "http_request": ("Send an HTTP request and return status, headers and body.",
                     {"method": P("string", "HTTP method.", None, ["GET", "POST", "PUT", "DELETE", "HEAD"]),
                      "url": P("string", "Absolute URL.", "url"), "headers": P("object", "Request headers.", "headers"),
                      "body": P("string", "Request body.", "jsonbody"), "timeout_ms": P("integer", "Timeout.", "timeout")},
                     ["method", "url"]),
    "python": ("Execute a Python snippet in a sandbox and return its output.",
               {"code": P("string", "Python source to run.", "pycode")}, ["code"]),
    "test_run": ("Run the project's test suite or a subset.",
                 {"target": P("string", "Test path or selector.", "testpath"),
                  "fail_fast": P("boolean", "Stop at first failure.", "bool")}, []),
    "make_build": ("Build the project with make.", {"target": P("string", "Make target.", None, ["all", "test", "release", "debug", "lint"])}, []),
    "apply_patch": ("Apply a unified diff to the workspace.", {"patch": P("string", "Unified diff text.", "diff")}, ["patch"]),
    "review_diff": ("Review the pending diff and report risks.", {"base": P("string", "Base ref.", "ref")}, []),
    "diagnostics": ("Return compiler or linter diagnostics for a file.", {"path": P("string", "File to check.", "path")}, ["path"]),
    "jina_search": ("Search the public web.", {"query": P("string", "Search query.", "query"), "num": P("integer", "Result count.", "small")}, ["query"]),
    "jina_read": ("Fetch a web page and return readable text.", {"url": P("string", "Page URL.", "url")}, ["url"]),
    "WebSearch": ("Search the web and return ranked results.", {"query": P("string", "Search query.", "query")}, ["query"]),
    "WebFetch": ("Fetch a URL and answer a prompt about it.", {"url": P("string", "URL.", "url"), "prompt": P("string", "What to extract.", "query")}, ["url", "prompt"]),
    "knowledge_base": ("Query the local knowledge base.", {"query": P("string", "Question or keywords.", "query"), "k": P("integer", "Top-k.", "small")}, ["query"]),
    "scratchpad": ("Read or write the agent scratchpad.",
                   {"action": P("string", "Operation.", None, ["read", "write", "append", "clear"]), "text": P("string", "Text to store.", "text")}, ["action"]),
    "context_status": ("Report context window usage.", {}, []),
    "context_evict": ("Evict messages from context by id.", {"ids": P("array", "Message ids.", "msgids")}, ["ids"]),
    "context_compact": ("Summarize old turns to free context.", {"keep_last": P("integer", "Turns to keep verbatim.", "tiny")}, []),
    "context_recall": ("Restore a masked observation by key.", {"key": P("string", "Placeholder key.", "obskey")}, ["key"]),
    "goal_queue": ("Inspect or modify the persistent goal queue.",
                   {"action": P("string", "Operation.", None, ["list", "push", "pop", "decompose", "complete"]),
                    "goal": P("string", "Goal text.", "task")}, ["action"]),
    "get_goal": ("Return the current goal and its acceptance checks.", {}, []),
    "TodoWrite": ("Replace the task list.", {"todos": P("array", "Items with content and status.", "todos")}, ["todos"]),
    "swarm": ("Fan subtasks out to parallel agents.", {"tasks": P("array", "Subtask descriptions.", "subtasks"),
                                                       "max_parallel": P("integer", "Concurrency cap.", "tiny")}, ["tasks"]),
    "agent": ("Spawn a sub-agent with a prompt.", {"prompt": P("string", "Instructions for the sub-agent.", "task"),
                                                   "model": P("string", "Model id.", "model")}, ["prompt"]),
    "agent_wait": ("Wait for a sub-agent to finish.", {"agent_id": P("string", "Sub-agent id.", "agentid"),
                                                       "timeout_ms": P("integer", "Timeout.", "timeout")}, ["agent_id"]),
    "secret_scan": ("Scan paths for credentials.", {"path": P("string", "Root path.", "dir")}, []),
    "governance": ("Query or enforce governance policy.", {"action": P("string", "Operation.", None, ["status", "check", "explain"])}, ["action"]),
    "discover_tools": ("Search the tool catalog.", {"query": P("string", "Capability description.", "query")}, ["query"]),
    "load_tools": ("Load a toolset into the session.", {"names": P("array", "Tool names.", "toolnames")}, ["names"]),
    "playbook_add": ("Record a lesson in the playbook.", {"lesson": P("string", "Lesson text.", "text"), "tags": P("array", "Tags.", "tags")}, ["lesson"]),
    "weather": ("Current weather for a location.", {"location": P("string", "City or coordinates.", "city")}, ["location"]),
    "cwd": ("Return the working directory.", {}, []),
    "computer": ("Control the screen: screenshot, click, type, zoom.",
                 {"action": P("string", "Action.", None, ["screenshot", "click", "type", "zoom", "scroll"]),
                  "x": P("integer", "X coordinate.", "coord"), "y": P("integer", "Y coordinate.", "coord")}, ["action"]),
}
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Toronto", "Oslo", "Austin", "Seoul", "Lyon", "Perth"]


def generic_schema(name):
    """Schema for a known tool token without a curated definition, inferred from its name."""
    low = name.lower()
    props, req = {}, []
    if any(k in low for k in ("search", "query", "find", "lookup", "research")):
        props["query"] = P("string", "Search query.", "query"); req.append("query")
        props["limit"] = P("integer", "Maximum results.", "small")
    elif any(k in low for k in ("read", "get", "fetch", "status", "info", "show")):
        props["id"] = P("string", "Identifier of the item.", "anyid"); req.append("id")
    elif any(k in low for k in ("write", "add", "create", "put", "set", "save", "send")):
        props["name"] = P("string", "Name of the item.", "ident"); props["content"] = P("string", "Content.", "text")
        req += ["name", "content"]
    else:
        props["input"] = P("string", "Primary input.", "text"); req.append("input")
    readable = name.split("__")[-1].replace("_", " ").replace("-", " ")
    return (f"Synthetic schema for the {readable} tool.", props, req)


def tool_def(name):
    return TOOLS.get(name) or generic_schema(name)


def kind_value(r, kind, ctx):
    k = kind
    if k == "command":
        return r.choice([f"ls -la {r.choice(DIRS)}", f"rg -n '{ident(r)}' {r.choice(DIRS)}", "make -j8", "pytest -q",
                         f"cat {ctx['path']}", "git status --short", f"wc -l {ctx['path']}", "cargo test --quiet",
                         f"python -m {ident(r, 'snake')} --help", "npm run build", f"head -n 40 {ctx['path']}",
                         "go test ./...", f"du -sh {r.choice(DIRS)}", "uname -a", f"jq '.{word(r)}' data.json"])
    if k == "timeout":
        return r.choice([5000, 10000, 30000, 60000, 120000])
    if k == "path":
        return ctx["path"] if r.random() < 0.7 else path(r)
    if k == "dir":
        return r.choice(DIRS + ["."])
    if k == "line":
        return r.randint(1, 400)
    if k == "small":
        return r.choice([5, 10, 20, 50, 100])
    if k == "tiny":
        return r.randint(1, 6)
    if k == "code":
        return code_snippet(r, ctx["lang"])
    if k == "code_line":
        return code_snippet(r, ctx["lang"]).splitlines()[r.randint(0, 2)]
    if k == "bool":
        return r.random() < 0.5
    if k == "pattern":
        return r.choice([ctx["ident"], f"def {ctx['ident']}", f"{word(r)}\\(", f"TODO|FIXME", f"class {ident(r, 'pascal')}"])
    if k == "glob":
        return f"**/*.{ctx['ext']}"
    if k == "gitargs":
        return r.choice([[], ["--stat"], ["-n", "5"], ["--", ctx["path"]], ["HEAD~1"]])
    if k == "url":
        return url(r)
    if k == "headers":
        return {"Accept": "application/json", "User-Agent": "dsco/1.0"}
    if k == "jsonbody":
        return json.dumps({word(r): word(r), "limit": r.randint(1, 50)})
    if k == "pycode":
        return f"import json\ndata = json.load(open('{word(r)}.json'))\nprint(len(data), sorted(data)[:5])"
    if k == "testpath":
        return r.choice([f"tests/test_{ctx['word']}.py", f"tests/test_{ctx['word']}.py::test_{ident(r, 'snake')}", "", "./..."])
    if k == "diff":
        return f"--- a/{ctx['path']}\n+++ b/{ctx['path']}\n@@ -10,3 +10,4 @@\n {phrase(r)}\n+{phrase(r)}\n"
    if k == "ref":
        return r.choice(["HEAD", "main", "origin/main", "HEAD~3"])
    if k == "query":
        return r.choice([f"{ctx['pkg']} latest release", f"how to {phrase(r, 2, 5)}", f"{ctx['word']} {word(r)} error",
                         f"{ctx['pkg']} changelog {word(r)}", phrase(r, 2, 6)])
    if k == "text":
        return sentence(r)
    if k == "obskey":
        return f"obs:{hexid(r, 6)}"
    if k == "msgids":
        return [f"msg_{hexid(r, 6)}" for _ in range(r.randint(1, 4))]
    if k == "task":
        return ctx["task"]
    if k == "todos":
        return [{"content": phrase(r, 3, 6), "status": r.choice(["pending", "in_progress", "completed"])}
                for _ in range(r.randint(2, 5))]
    if k == "subtasks":
        return [f"{r.choice(['audit', 'test', 'document', 'benchmark'])} {path(r)}" for _ in range(r.randint(3, 6))]
    if k == "model":
        return r.choice(["local-small", "local-large", "router-default"])
    if k == "agentid":
        return f"agent_{hexid(r, 6)}"
    if k == "toolnames":
        return r.sample(sorted(TOOLS), 2)
    if k == "tags":
        return [word(r) for _ in range(r.randint(1, 3))]
    if k == "city":
        return r.choice(CITIES)
    if k == "coord":
        return r.randint(0, 1920)
    if k == "anyid":
        return f"{word(r)}_{hexid(r, 6)}"
    if k == "ident":
        return ident(r)
    return phrase(r, 2, 6)


def tool_args(r, name, ctx):
    _, props, req = tool_def(name)
    args = {}
    for p, d in props.items():
        if p not in req and r.random() < 0.5:
            continue
        args[p] = r.choice(d["enum"]) if "enum" in d else kind_value(r, d.get("x-kind"), ctx)
    return args


def tool_result(r, name, args, ctx):
    if name == "bash":
        code = 0 if r.random() < 0.85 else r.choice([1, 2, 127])
        out = "\n".join(f"{word(r)} {r.randint(0, 9999)} {path(r)}" for _ in range(r.randint(1, 5)))
        return {"exit_code": code, "stdout": out if code == 0 else "", "stderr": "" if code == 0 else fill(r, "error: {word} not found")}
    if name == "read_file":
        return {"path": args.get("path"), "lines": r.randint(20, 900), "content": code_snippet(r, ctx["lang"])}
    if name in ("write_file", "append_file"):
        return {"path": args.get("path"), "bytes_written": r.randint(40, 9000)}
    if name == "edit_file":
        return {"path": args.get("path"), "replacements": 1 if not args.get("replace_all") else r.randint(2, 9)}
    if name == "grep_files":
        return {"matches": [{"path": path(r, ctx["ext"]) if r.random() < 0.5 else ctx["path"], "line": r.randint(1, 500),
                             "text": f"{ctx['ident']}({word(r)})"} for _ in range(r.randint(0, 5))]}
    if name == "find_files":
        return {"files": [path(r, ctx["ext"]) for _ in range(r.randint(1, 6))]}
    if name == "list_directory":
        return {"entries": [{"name": f"{ident(r, 'snake')}.{r.choice(EXTS)}", "type": r.choice(["file", "file", "dir"]),
                             "size": r.randint(0, 50000)} for _ in range(r.randint(2, 7))]}
    if name == "file_info":
        return {"path": args.get("path"), "size": r.randint(10, 400000), "type": "file", "modified": ts(r)}
    if name == "git":
        sub = args.get("subcommand")
        if sub == "log":
            return {"commits": [{"sha": hexid(r, 7), "author": r.choice(PEOPLE), "subject": phrase(r, 3, 7)} for _ in range(3)]}
        if sub == "status":
            return {"branch": r.choice(["main", f"fix/{word(r)}", f"feat/{word(r)}"]),
                    "modified": [path(r) for _ in range(r.randint(0, 3))], "untracked": []}
        return {"output": f"diff --git a/{ctx['path']} b/{ctx['path']}\n+{phrase(r)}"}
    if name == "http_request":
        st = r.choice([200, 200, 200, 201, 204, 404, 500, 429])
        return {"status": st, "headers": {"content-type": "application/json"},
                "body": json.dumps({"id": hexid(r), word(r): word(r), "count": r.randint(0, 999)})}
    if name == "python":
        return {"stdout": f"{r.randint(1, 5000)} {[word(r) for _ in range(3)]}", "stderr": ""}
    if name == "test_run":
        total = r.randint(8, 400)
        failed = 0 if r.random() < 0.6 else r.randint(1, 4)
        res = {"passed": total - failed, "failed": failed, "skipped": r.randint(0, 3), "duration_s": round(r.uniform(0.4, 90), 1)}
        if failed:
            res["failures"] = [{"test": f"test_{ident(r, 'snake')}", "message": f"AssertionError: expected {r.randint(0, 9)} got {r.randint(10, 99)}"}]
        return res
    if name == "make_build":
        return {"ok": r.random() < 0.85, "warnings": r.randint(0, 6), "elapsed_s": round(r.uniform(1, 120), 1)}
    if name == "apply_patch":
        return {"applied": True, "files": [ctx["path"]]}
    if name == "review_diff":
        return {"files_changed": r.randint(1, 6), "risks": [phrase(r, 4, 9) for _ in range(r.randint(0, 2))], "verdict": r.choice(["ok", "ok", "needs_changes"])}
    if name == "diagnostics":
        return {"diagnostics": [{"line": r.randint(1, 300), "severity": r.choice(["error", "warning"]), "message": phrase(r, 3, 7)} for _ in range(r.randint(0, 3))]}
    if name in ("jina_search", "WebSearch", "knowledge_base", "discover_tools"):
        return {"results": [{"title": phrase(r, 3, 7).title(), "url": url(r), "snippet": sentence(r)} for _ in range(r.randint(1, 4))]}
    if name in ("jina_read", "WebFetch"):
        return {"url": args.get("url"), "title": phrase(r, 2, 6).title(), "text": " ".join(sentence(r) for _ in range(r.randint(2, 5)))}
    if name == "scratchpad":
        return {"ok": True, "length": r.randint(0, 4000)}
    if name == "context_status":
        used = r.randint(5000, 190000)
        return {"used_tokens": used, "window": 200000, "pct": round(used / 2000, 1), "masked_blocks": r.randint(0, 12)}
    if name in ("context_evict", "context_compact"):
        return {"freed_tokens": r.randint(1000, 60000)}
    if name == "context_recall":
        return {"key": args.get("key"), "content": phrase(r, 6, 14)}
    if name in ("goal_queue", "get_goal"):
        return {"goal": ctx["task"], "status": r.choice(["active", "pending"]), "checks": [phrase(r, 3, 6) for _ in range(2)]}
    if name == "TodoWrite":
        return {"ok": True, "count": len(args.get("todos", []))}
    if name == "swarm":
        return {"dispatched": len(args.get("tasks", [])), "run_id": f"swarm_{hexid(r, 8)}"}
    if name == "agent":
        return {"agent_id": f"agent_{hexid(r, 6)}", "status": "running"}
    if name == "agent_wait":
        return {"status": "completed", "summary": sentence(r)}
    if name == "secret_scan":
        return {"findings": r.choice([[], [{"path": path(r, "yaml"), "rule": "generic-api-key", "line": r.randint(1, 90)}]])}
    if name == "governance":
        return {"tier": r.choice(["trust", "standard", "restricted"]), "denied": r.randint(0, 3)}
    if name == "weather":
        return {"location": args.get("location"), "temp_c": r.randint(-10, 35), "conditions": r.choice(["clear", "rain", "cloudy", "snow"])}
    if name == "cwd":
        return {"cwd": f"/workspace/{ident(r, 'snake')}"}
    if name == "computer":
        return {"ok": True, "screenshot": f"img:{hexid(r)}"} if args.get("action") == "screenshot" else {"ok": True}
    if name == "playbook_add":
        return {"id": f"lesson_{hexid(r, 6)}"}
    if name == "load_tools":
        return {"loaded": args.get("names", [])}
    return {"ok": True, "result": phrase(r, 3, 9)}


# ---------------------------------------------------------------- category: opcodes
def op_prose(r, op):
    lines = [f"{op['token']} {op['mnemonic']}: {op['desc']}", f"Signature: {op['signature']}",
             f"When: {op['when']}"]
    if op["pre"]:
        lines.append("Preconditions: " + "; ".join(op["pre"]) + ".")
    if op["post"]:
        lines.append("Postconditions: " + "; ".join(op["post"]) + ".")
    lines.append(f"Effect: {op['effect']}. Archetype: {op['archetype']}. Reversibility: {op['reversibility']}.")
    if op["inverse"]:
        lines.append(f"Inverse: {OPS[op['inverse']]['token'] if op['inverse'] in OPS else op['inverse']}.")
    if op["failure_ops"]:
        lines.append("On failure: " + ", ".join(OPS[f]["token"] if f in OPS else f for f in op["failure_ops"]) + ".")
    if op["dsco"]:
        lines.append(f"Runtime binding: {op['dsco']}.")
    return "\n".join(lines) if r.random() < 0.5 else " ".join(lines)


def op_json(r, op):
    keys = list(op)
    if r.random() < 0.5:
        keys = [k for k in keys if r.random() < 0.8 or k in ("mnemonic", "token")]
    d = {k: op[k] for k in keys}
    return json.dumps(d, indent=r.choice([None, None, 2, 1]), ensure_ascii=False)


def op_doc(r, op):
    args = ", ".join(op["operands"]) or "none"
    style = r.choice(["c", "py", "md"])
    if style == "c":
        body = "\n".join([f"/* {op['mnemonic']} ({op['token']})", f" * {op['desc']}", f" * Operands: {args}",
                          f" * Returns: {op['returns']}", f" * Emit: {op['when']}", " */"])
        return body + f"\n#define OP_{op['mnemonic'].replace('.', '_')} {op['code']}"
    if style == "py":
        fn = op["mnemonic"].lower().replace(".", "_")
        params = ", ".join(a.replace(":", ": ") for a in op["operands"])
        return (f"def {fn}({params}) -> \"{op['returns']}\":\n    \"\"\"{op['desc']}\n\n    {op['when']}\n"
                f"    Pre: {'; '.join(op['pre'])}\n    Post: {'; '.join(op['post'])}\n    \"\"\"\n    return emit(\"{op['token']}\")")
    rows = ["| field | value |", "|---|---|"] + [f"| {k} | {v if not isinstance(v, list) else ', '.join(v)} |"
                                                  for k, v in op.items() if v not in (None, [])]
    return f"### {op['mnemonic']}\n\n" + "\n".join(rows)


def op_qa(r, op):
    when = op["when"].removeprefix("Emit ").rstrip(".")
    forms = [f"Q: Which op-code should the agent emit {when}? A: {op['token']} ({op['mnemonic']}).",
             f"Q: What does {op['token']} do? A: {op['desc']}",
             f"Q: What does {op['mnemonic']} return? A: {op['returns']}, per {op['signature']}.",
             f"Q: Is {op['mnemonic']} reversible? A: It is {op['reversibility']}"
             + (f"; the inverse is {op['inverse']}." if op["inverse"] else "."),
             f"Q: What must hold before {op['token']}? A: " + ("; ".join(op["pre"]) or "nothing") + "."]
    return " ".join(r.sample(forms, r.randint(2, len(forms))))


def op_decomp(r, op):
    parts = DECOMP.get(op["token"])
    if not parts:
        return f"{op['token']} is a single op of family {op['family']}: {op['desc']}"
    return f"{op['token']} = {' '.join(parts)} -> {op['mnemonic']}: {op['desc']}"


def op_call(r, op):
    args = op_args(r, op)
    ret = return_value(r, op["returns"])
    return (f"{op['token']}<|json|>{json.dumps(args)}<|/json|> [RESULT] <|json|>{json.dumps(ret)}<|/json|> "
            f"# {op['desc']}")


def op_family_list(r, fam_ops):
    chunk = r.sample(fam_ops, min(len(fam_ops), r.randint(4, 12)))
    fam = chunk[0]["family"]
    head = r.choice([f"Family {fam} op-codes:", f"## {fam}", f"Op-codes in the {fam} family", f"{fam} reference"])
    return head + "\n" + "\n".join(f"- {o['token']} {o['signature']} — {o['desc']}" for o in chunk)


def gen_opcodes(r):
    by_fam = {}
    for o in OPS_LIST:
        by_fam.setdefault(o["family"], []).append(o)
    fams = list(by_fam.values())
    renders = [op_prose, op_json, op_doc, op_qa, op_decomp, op_call]
    while True:
        order = OPS_LIST[:]
        r.shuffle(order)
        for op in order:
            yield r.choice(renders)(r, op)
            if r.random() < 0.08:
                yield op_family_list(r, r.choice(fams))


# ---------------------------------------------------------------- category: decisions
def gen_decisions(r):
    import sys1.jevmr.fractal  # noqa: F401  registers every fractal variant
    from sys1.jevmr.families import FAMILIES
    names = sorted(FAMILIES)
    seed = 0
    while True:
        for name in names:
            fam = FAMILIES[name]
            for it in fam.items(random.Random(seed * 7919 + zlib.crc32(name.encode()))):
                yield render_decision(r, fam, it)
        seed += 1


def render_decision(r, fam, it):
    opts = list(fam.options)
    r.shuffle(opts)
    gold = it["gold"]
    style = r.random()
    if style < 0.35:
        o = " ".join(f"[OPT] {OPS[m]['token']} {m}: {OPS[m]['desc']}" for m in opts)
        s = f"[BOS][GOAL] {fam.question} [STATE] {it['state']} [OPTS] {o} [DECIDE] {OPS[gold]['token']} [EOS]"
    elif style < 0.75:
        o = "\n".join(f"{i + 1}. {OPS[m]['token']} — {OPS[m]['desc']}" for i, m in enumerate(opts))
        think = ""
        if r.random() < 0.6:
            think = (f"<|think|>The situation is about {it['family'].split(':')[-1].replace('+', ', ').replace('_', ' ')}. "
                     f"Applying the stated conditions in order selects {gold}: {OPS[gold]['when']}<|/think|>")
        conf = f" <|confidence|> {round(r.uniform(0.7, 0.99), 2)}" if r.random() < 0.3 else ""
        s = (f"<|system|>You are the sys1 decision head. Answer with exactly one op-code token.<|end|>"
             f"<|user|>{it['state']}\n\n{fam.question}\n{o}<|end|>"
             f"<|assistant|>{think}<|choice|> {OPS[gold]['token']}{conf}<|end|>")
    else:
        req = fam.request(it, "sys1")
        answers = {"op": {"choice": gold, "token": OPS[gold]["token"]}}
        for k, (_, v) in it["probes"].items():
            answers[k] = {"noul": bool(v)}
        s = (f"<|json|>{json.dumps(req, ensure_ascii=False)}<|/json|> [ANSWER] "
             f"<|json|>{json.dumps(answers)}<|/json|>")
    if it["probes"] and r.random() < 0.3:
        s += " " + " ".join(f"<|noul|> {q} {'yes' if v else 'no'}" for q, v in it["probes"].values())
    return s


# ---------------------------------------------------------------- category: transcripts
SCENARIOS = {
    "bugfix": ["get_goal?", "grep_files", "read_file", "edit_file", "test_run", "review_diff?"],
    "feature": ["list_directory", "read_file", "write_file", "edit_file?", "make_build?", "test_run"],
    "research": ["jina_search|WebSearch", "jina_read|WebFetch", "scratchpad?", "jina_read?"],
    "repo_survey": ["cwd?", "list_directory", "find_files", "file_info?", "read_file", "git"],
    "ops": ["bash", "http_request", "bash?", "governance?"],
    "data": ["read_file", "python", "python?", "write_file?"],
    "context": ["context_status", "context_evict|context_compact|context_recall", "read_file?"],
    "swarm": ["goal_queue", "swarm", "agent_wait", "agent_wait?"],
    "plan": ["TodoWrite", "read_file", "edit_file", "test_run", "TodoWrite?"],
    "security": ["secret_scan", "read_file?", "edit_file?", "git"],
    "lookup": ["knowledge_base", "discover_tools?", "weather?"],
}
SYSTEMS = ["You are dsco, a local-first coding agent. Use tools to act; keep answers short and factual.",
           "You are an autonomous software agent working in a sandboxed repository. Every tool call goes through the capability gate.",
           "You are a careful assistant with shell and file tools. Verify changes with tests before reporting success.",
           "Agent runtime: tools are listed below. Emit exactly one op-code before each tool call."]
THINKS = ["I need to locate where {ident} is defined before editing.", "The last result shows what I needed; next I will {verb}.",
          "Tests have not been run since the edit, so I should run them.", "That call failed; a different approach is safer than repeating it.",
          "The context is getting full, so I will free some space first.", "These subtasks are independent, so they can run in parallel.",
          "I have enough information to answer.", "Let me confirm the current state before changing anything."]


def expand(r, steps):
    out = []
    for s in steps:
        opt = s.endswith("?")
        s = s.rstrip("?")
        if opt and r.random() < 0.5:
            continue
        out.append(r.choice(s.split("|")))
    return out


def gen_transcripts(r):
    extra = sorted(TOOL_TOKENS - set(TOOLS))
    while True:
        x = r.random()
        if x < 0.62:
            yield transcript_chat(r, extra)
        elif x < 0.82:
            yield transcript_control(r)
        else:
            yield transcript_ops(r)


def new_ctx(r):
    ext = r.choice(EXTS[:9])
    ctx = {"ext": ext, "lang": LANGS[ext], "path": path(r, ext), "ident": ident(r, "snake"), "word": word(r),
           "pkg": r.choice(PKGS)}
    ctx["task"] = fill(r, r.choice(TASKS)).replace("{", "").replace("}", "")
    return ctx


def tool_marker(name):
    return f"<|tool:{name}|>" if name in TOOL_TOKENS else f"<|tool|>{name}"


def transcript_chat(r, extra):
    ctx = new_ctx(r)
    steps = expand(r, SCENARIOS[r.choice(list(SCENARIOS))])
    if extra and r.random() < 0.15:
        steps.insert(r.randint(0, len(steps)), r.choice(extra))
    parts = [f"<|system|>{r.choice(SYSTEMS)}"]
    if r.random() < 0.35:
        listed = sorted(set(steps) | set(r.sample(sorted(TOOLS), 3)))
        parts.append(" Tools: " + json.dumps([{"name": n, "description": tool_def(n)[0]} for n in listed]))
    parts.append("<|end|>")
    parts.append(f"<|user|>{ctx['task'][0].upper() + ctx['task'][1:]}.<|end|>")
    for name in steps:
        args = tool_args(r, name, ctx)
        a = "<|assistant|>"
        if r.random() < 0.5:
            a += "<|think|>" + r.choice(THINKS).format(ident=ctx["ident"], verb=r.choice(["edit the file", "run the tests", "read the source", "search the code"])) + "<|/think|>"
        op = TOOL_TO_OP.get(name, "ACT.CALL")
        call_id = f"call_{hexid(r, 10)}"
        a += f"{OPS[op]['token']}{tool_marker(name)}<|json|>{json.dumps({'id': call_id, 'name': name, 'arguments': args}, ensure_ascii=False)}<|/json|><|end|>"
        parts.append(a)
        if r.random() < 0.1:
            kind, msg = r.choice(ERRORS)
            res = {"id": call_id, "ok": False, "error": {"type": kind, "message": fill(r, msg)}}
            parts.append(f"<|tool|>{OPS['ACT.RESULT_ERR']['token']}<|json|>{json.dumps(res)}<|/json|><|end|>")
            follow = {"timeout": "ACT.RETRY.TOOL_CALL", "connection_reset": "ACT.RETRY.TOOL_CALL", "rate_limited": "CTRL.BACKOFF",
                      "permission_denied": "GOV.NO_AUTHORITY"}.get(kind, "ERR.BOUNDED_ALTERNATIVE")
            parts.append(f"<|assistant|>{OPS[follow]['token']} {OPS[follow]['desc']}<|end|>")
            if follow == "GOV.NO_AUTHORITY":
                break
            continue
        res = {"id": call_id, "ok": True, "result": tool_result(r, name, args, ctx)}
        parts.append(f"<|tool|>{OPS['ACT.RESULT_OK']['token']}<|json|>{json.dumps(res, ensure_ascii=False)}<|/json|><|end|>")
    final = r.choice([f"Done. I {r.choice(['updated', 'checked', 'fixed', 'reviewed'])} {ctx['path']} and the {r.choice(['tests pass', 'build is green', 'change is in place'])}.",
                      f"Summary: {sentence(r)} {sentence(r)}", f"The {ctx['word']} issue is resolved; {sentence(r).lower()}",
                      f"I could not finish: {phrase(r, 3, 7)}. Next step: {phrase(r, 3, 6)}."])
    parts.append(f"<|assistant|>{OPS['CTRL.FINAL']['token']}{final}<|end|>[EOS]")
    return "\n".join(parts) if r.random() < 0.5 else "".join(parts)


def transcript_control(r):
    ctx = new_ctx(r)
    steps = expand(r, SCENARIOS[r.choice(list(SCENARIOS))])
    out = [f"[BOS][GOAL] {ctx['task']}"]
    for i, name in enumerate(steps):
        args = tool_args(r, name, ctx)
        op = TOOL_TO_OP.get(name, "ACT.CALL")
        out.append(f"[STEP] {i + 1} [ACT] {OPS[op]['token']} {tool_marker(name)} <|json|>{json.dumps(args)}<|/json|>")
        out.append(f"[OBS] <|json|>{json.dumps(tool_result(r, name, args, ctx))}<|/json|>")
    out.append(f"[DECIDE] {OPS['CTRL.FINAL']['token']} [ANSWER] {sentence(r)} [EOS]")
    return " ".join(out)


def transcript_ops(r):
    """Op-level traces: the agent emits op tokens with operand JSON and receives typed returns."""
    fam = r.choice(sorted({o["family"] for o in OPS_LIST}))
    pool = [o for o in OPS_LIST if o["family"] == fam] + r.sample(OPS_LIST, 6)
    out = [f"<|system|>Op-code mode. Each step is one op token plus operands as JSON.<|end|>",
           f"<|user|>{fill(r, r.choice(TASKS))}<|end|>", f"<|assistant|>{OPS['CTRL.START']['token']}<|end|>"]
    for _ in range(r.randint(2, 6)):
        op = r.choice(pool)
        out.append(f"<|assistant|>{op['token']}<|json|>{json.dumps(op_args(r, op))}<|/json|><|end|>")
        out.append(f"<|tool|>[RESULT] <|json|>{json.dumps(return_value(r, op['returns']))}<|/json|><|end|>")
    out.append(f"<|assistant|>{OPS['CTRL.FINAL']['token']}{sentence(r)}<|end|>")
    return "".join(out)


# ---------------------------------------------------------------- category: schemas
JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "json": "object", "list": "array", "bool": "boolean",
              "id": "string", "ref": "string", "time": "string", "blob": "string"}


def clean_props(props):
    return {k: {kk: vv for kk, vv in v.items() if kk != "x-kind"} for k, v in props.items()}


def op_schema(op):
    props = {}
    for n, t in map(parse_operand, op["operands"]):
        d = {"type": JSON_TYPES.get(t, "string"), "description": f"{n} operand ({t})"}
        if t == "time":
            d["format"] = "date-time"
        if t == "list":
            d["items"] = {"type": "string"}
        props[n] = d
    return (op["desc"], props, [parse_operand(x)[0] for x in op["operands"]])


def render_schema(r, name, desc, props, req):
    schema = {"type": "object", "properties": clean_props(props), "required": req}
    if r.random() < 0.4:
        schema["additionalProperties"] = False
    dialect = r.random()
    if dialect < 0.3:
        d = {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
    elif dialect < 0.55:
        d = {"name": name, "description": desc, "input_schema": schema}
    elif dialect < 0.75:
        d = {"name": name, "description": desc, "inputSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}}
    elif dialect < 0.9:
        d = {"$id": f"https://schemas.example.com/tools/{name}.json", "title": name, "description": desc, **schema}
    else:
        sig = ", ".join(f"{k}{'' if k in req else '?'}: {v['type']}" for k, v in props.items())
        return f"tool {name}({sig}) // {desc}"
    return json.dumps(d, indent=r.choice([None, None, 2]), ensure_ascii=False)


def gen_schemas(r):
    names = sorted(set(TOOLS) | TOOL_TOKENS)
    while True:
        x = r.random()
        if x < 0.45:
            n = r.choice(names)
            yield render_schema(r, n, *tool_def(n))
        elif x < 0.85:
            op = r.choice(OPS_LIST)
            name = op["mnemonic"].lower().replace(".", "_") if r.random() < 0.6 else op["token"]
            s = render_schema(r, name, *op_schema(op))
            if r.random() < 0.3:
                s += f" returns: {op['returns']}"
            yield s
        else:
            chosen = r.sample(names, r.randint(3, 8))
            tools = [{"name": n, "description": tool_def(n)[0],
                      "input_schema": {"type": "object", "properties": clean_props(tool_def(n)[1]),
                                       "required": tool_def(n)[2]}} for n in chosen]
            wrap = r.choice(["mcp", "list", "system"])
            if wrap == "mcp":
                yield json.dumps({"jsonrpc": "2.0", "id": r.randint(1, 999), "result": {"tools": tools}})
            elif wrap == "list":
                yield json.dumps(tools, indent=r.choice([None, 2]))
            else:
                yield f"<|system|>You can call these tools:<|json|>{json.dumps(tools)}<|/json|><|end|>"


# ---------------------------------------------------------------- driver
GENERATORS = {"opcodes": gen_opcodes, "decisions": gen_decisions, "transcripts": gen_transcripts, "schemas": gen_schemas}


def escape(s):
    return s.replace("\r", "").replace("\n", "\\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--total-mb", type=float, default=190.0, help=f"total output size incl. heldout (cap {CAP_MB})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    total = int(min(a.total_mb, CAP_MB) * 1_000_000)
    from pathlib import Path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held_f = open(out / "heldout.txt", "w", encoding="utf-8")
    held_h = hashlib.sha256()
    split = random.Random(a.seed + 1)
    manifest = {"version": VERSION, "seed": a.seed, "heldout_ratio": HELDOUT, "cap_mb": CAP_MB, "files": {},
                "heldout_by_category": {}, "format": "one sample per line; newlines escaped as \\n",
                "provenance": "synthetic; op-codes and jevmr families from sys1 sources; tool names from spec reserved region",
                "vocab_size": spec.VOCAB_SIZE, "opcodes": len(OPS_LIST)}
    held_bytes = held_lines = dups = 0
    seen = set()  # exact-line dedup across all files, so heldout never leaks into train
    t0 = time.time()
    for cat, share in SHARES.items():
        budget = int(total * share)
        rng = random.Random(f"{a.seed}:{cat}")
        h = hashlib.sha256()
        nbytes = nlines = hb = hl = streak = 0
        with open(out / f"{cat}.txt", "w", encoding="utf-8") as f:
            for sample in GENERATORS[cat](rng):
                line = (escape(sample) + "\n").encode("utf-8")
                d = hashlib.blake2b(line, digest_size=12).digest()
                if d in seen:
                    dups += 1; streak += 1
                    if streak > 50_000:
                        print(f"{cat}: generator exhausted unique samples", flush=True)
                        break
                    continue
                seen.add(d); streak = 0
                if split.random() < HELDOUT:
                    held_f.write(line.decode()); held_h.update(line)
                    hb += len(line); hl += 1
                else:
                    f.write(line.decode()); h.update(line)
                    nbytes += len(line); nlines += 1
                if nbytes + hb >= budget:
                    break
        held_bytes += hb; held_lines += hl
        manifest["files"][f"{cat}.txt"] = {"bytes": nbytes, "lines": nlines, "sha256": h.hexdigest()}
        manifest["heldout_by_category"][cat] = {"bytes": hb, "lines": hl}
        print(f"{cat}: {nlines} lines {nbytes / 1e6:.1f} MB (+{hl} heldout) {time.time() - t0:.0f}s", flush=True)
    held_f.close()
    manifest["files"]["heldout.txt"] = {"bytes": held_bytes, "lines": held_lines, "sha256": held_h.hexdigest()}
    manifest["duplicates_skipped"] = dups
    manifest["total_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    assert manifest["total_bytes"] <= CAP_MB * 1_000_000
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("total_bytes", "elapsed_s")}))


if __name__ == "__main__":
    main()
