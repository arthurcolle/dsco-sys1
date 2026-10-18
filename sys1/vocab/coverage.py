#!/usr/bin/env python3
"""Op-code coverage and gap analysis.

    python -m sys1.vocab.coverage [--max-candidates 150]

Measures how well the op-code table (sys1/vocab/opcodes.json) covers real
agent behaviour recorded locally:

  (a) op usage in data/labels/labeled_episodes.jsonl: distinct ops used,
      rank/frequency (Zipf) fit, head concentration, families never used.
  (b) activities in data/episodes.jsonl that no op names specifically:
      shell programs and subcommands (tool "bash" and shell aliases, args
      JSON "command"), tools reached only through generic dispatch
      (invoke_tool inner names, tools mapped to generic ops), and recurring
      goal intents (keyword/regex clustering).
  (c) up to N new candidate op-codes written as data to
      runs/coverage/candidates.json for maintainer review. opcodes.py is
      never modified.

Everything stays local. Stdout shows aggregate counts and short
anonymized examples only. runs/coverage/report.json holds the full stats.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPCODES = ROOT / "sys1" / "vocab" / "opcodes.json"
LABELED = ROOT / "data" / "labels" / "labeled_episodes.jsonl"
EPISODES = ROOT / "data" / "episodes.jsonl"
OUT_DIR = ROOT / "runs" / "coverage"

SHELL_TOOLS = {"bash", "Bash", "run_command", "shell", "exec", "terminal"}
GENERIC_OPS = {"ACT.INVOKE.TOOL", "ACT.DISCOVER.TOOL", "ENV.INVOKE.SHELL_COMMAND"}
EXAMPLE_LEN = 72

# ---------------------------------------------------------------- anonymize
_SECRET_RES = [
    (re.compile(r"(?i)\b(sk|pk|rk|ghp|gho|xox[abp]|hf|AKIA)[-_A-Za-z0-9]{8,}"), "<secret>"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|auth|bearer)(\s*[=:]\s*|\s+)\S+"), r"\1=<redacted>"),
    (re.compile(r"\b[A-Z][A-Z0-9_]{2,}=\S+"), "<VAR>=<v>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"(~|/Users|/home|/tmp|/private|/var|/opt|/etc|/Volumes)(/[^\s'\"|;&)]*)?"), "<path>"),
    (re.compile(r"(?<![\w<])(\.{0,2}/)?[\w.-]+(/[\w.-]+){1,}"), "<path>"),
    (re.compile(r"\b[0-9a-f]{12,}\b"), "<hex>"),
    (re.compile(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b"), "<ip>"),
    (re.compile(r"\b[\w-]+\.(local|lan|internal)\b"), "<host>"),
]


_LOCAL_USER = re.compile(r"(?i)\b" + re.escape(Path.home().name) + r"\b") if Path.home().name else None


def anonymize(text: str, n: int = EXAMPLE_LEN) -> str:
    s = " ".join(str(text).split())
    if _LOCAL_USER:
        s = _LOCAL_USER.sub("<user>", s)
    for rx, rep in _SECRET_RES:
        s = rx.sub(rep, s)
    return s if len(s) <= n else s[: n - 3] + "..."


# ---------------------------------------------------------------- op table
def load_ops() -> tuple[list[dict], set[str], set[str]]:
    ops = json.loads(OPCODES.read_text())["ops"]
    mnemonics = {o["mnemonic"] for o in ops}
    parts: set[str] = set()
    for m in mnemonics:
        for p in m.lower().split(".")[1:]:
            parts.add(p)
            parts.update(p.split("_"))
    return ops, mnemonics, parts


# ------------------------------------------------------------- (a) op usage
def op_usage(ops: list[dict]) -> dict:
    freq: Counter = Counter()
    ep_freq: Counter = Counter()
    n_eps = 0
    with LABELED.open() as fh:
        for line in fh:
            e = json.loads(line)
            n_eps += 1
            seq = e.get("ops") or []
            freq.update(seq)
            ep_freq.update(set(seq))
    known = {o["mnemonic"]: o["family"] for o in ops}
    used = [m for m in freq if m in known]
    unknown = {m: c for m, c in freq.items() if m not in known}
    total = sum(freq[m] for m in used)
    ranked = sorted((freq[m] for m in used), reverse=True)
    # Zipf exponent: least squares on log(rank) vs log(freq)
    xs = [math.log(r) for r in range(1, len(ranked) + 1)]
    ys = [math.log(c) for c in ranked]
    s = r2 = float("nan")
    if len(xs) > 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        s = -sxy / sxx
        r2 = (sxy * sxy) / (sxx * syy) if syy else float("nan")

    def head_share(k: int) -> float:
        return round(sum(ranked[:k]) / total, 4) if total else 0.0

    fam_total = Counter(known.values())
    fam_used = Counter(known[m] for m in used)
    fam_tokens = Counter()
    for m in used:
        fam_tokens[known[m]] += freq[m]
    never = sorted(f for f in fam_total if fam_used[f] == 0)
    curve_points = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    return {
        "episodes": n_eps,
        "op_tokens": total,
        "ops_in_table": len(known),
        "ops_used": len(used),
        "ops_used_share": round(len(used) / len(known), 4),
        "ops_used_ge10": sum(1 for c in ranked if c >= 10),
        "ops_used_once": sum(1 for c in ranked if c == 1),
        "unknown_ops_in_labels": unknown,
        "zipf_exponent": round(s, 3),
        "zipf_r2": round(r2, 3),
        "head_share": {f"top{k}": head_share(k) for k in (1, 5, 10, 50)},
        "rank_freq": {str(r): ranked[r - 1] for r in curve_points if r <= len(ranked)},
        "top_ops": freq.most_common(25),
        "families_total": len(fam_total),
        "families_used": len(fam_used),
        "families_never_used": never,
        "family_usage": {
            f: {"ops": fam_total[f], "used": fam_used[f], "tokens": fam_tokens[f]}
            for f in sorted(fam_total, key=lambda f: -fam_tokens[f])
        },
    }


# ----------------------------------------------------- (b) shell programs
WRAPPERS = {"sudo", "env", "time", "timeout", "gtimeout", "nohup", "exec", "command",
            "nice", "caffeinate", "stdbuf", "xargs", "builtin", "then", "do", "else", "!"}
SUBCOMMAND_PROGS = {"git", "gh", "make", "docker", "npm", "pnpm", "yarn", "cargo", "uv",
                    "pip", "pip3", "brew", "kubectl", "go", "launchctl", "tmux", "kitty",
                    "systemctl", "ollama", "huggingface-cli", "hf", "modal", "fly",
                    "gcloud", "aws", "az", "vercel", "wrangler", "conda", "poetry",
                    "dsco", "zig", "swift", "xcrun", "osascript", "open", "defaults",
                    "codesign", "security", "mix", "rebar3", "erl", "sqlite3", "psql",
                    "ssh", "rsync", "scp", "tailscale", "nvidia-smi", "lsof", "ps"}
SHELL_KEYWORDS = {"if", "fi", "for", "while", "done", "case", "esac", "until", "function",
                  "{", "}", "[[", "]]", "[", "]", "(", ")", "local", "export", "set",
                  "unset", "return", "true", "false", "test", "source", ".", "in", "exit",
                  "import", "from", "def", "class", "print", "elif", "with", "try",
                  "except", "finally", "pass", "break", "continue", "read", "wait",
                  "trap", "shift", "eval", "alias", "declare", "readonly", "let"}


def split_segments(cmd: str) -> list[str]:
    """Split a shell line on unquoted | || && ; newline $( and backticks."""
    out, cur, q, i, n = [], [], None, 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if q:
            if ch == "\\" and q == '"' and i + 1 < n:
                cur.append(cmd[i:i + 2]); i += 2; continue
            if ch == q:
                q = None
            cur.append(ch); i += 1; continue
        if ch in "'\"":
            q = ch; cur.append(ch); i += 1; continue
        if ch == "\\" and i + 1 < n:
            cur.append(cmd[i:i + 2]); i += 2; continue
        if ch in "|;&\n`" or (ch == "$" and cmd[i + 1:i + 2] == "("):
            if ch == "&" and (cmd[i - 1:i] in (">", "<") or cmd[i + 1:i + 2] == ">"):
                cur.append(ch); i += 1; continue  # redirection like 2>&1 or &>
            out.append("".join(cur)); cur = []
            i += 2 if cmd[i:i + 2] in ("||", "&&", "$(") else 1
            continue
        cur.append(ch); i += 1
    out.append("".join(cur))
    return [seg.strip().strip(")").strip() for seg in out if seg.strip().strip(")").strip()]


def _program_and_sub(segment: str) -> tuple[str, str] | None:
    seg = segment.strip().lstrip("({ ").strip()
    if not seg or seg.startswith("#"):
        return None
    try:
        toks = shlex.split(seg, posix=True)
    except ValueError:
        toks = seg.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t) or t in WRAPPERS:
            i += 1
            if t in ("timeout", "gtimeout") and i < len(toks) and re.match(r"^\d", toks[i]):
                i += 1
            continue
        if t.startswith("-") and i > 0 and toks[i - 1] in WRAPPERS:
            i += 1
            continue
        break
    if i >= len(toks):
        return None
    if re.match(r"^[A-Z][A-Z0-9_]+$", toks[i]):
        return None  # env-var name used as a word, not a program
    prog = toks[i].rsplit("/", 1)[-1].lower()
    if (prog in SHELL_KEYWORDS or not re.match(r"^[a-z][\w.+-]*$", prog)
            or prog.startswith(("test_", "<")) or prog.endswith((".md", ".json", ".txt", ":"))):
        return None
    if re.match(r"^python[0-9.]*$", prog) or prog in ("py", "ipython"):
        rest = toks[i + 1:]
        if len(rest) >= 2 and rest[0] == "-m":
            return "python", "-m " + rest[1].split(".")[0].lower()
        if rest and rest[0] == "-c":
            return "python", "-c"
        if rest and rest[0].endswith(".py"):
            return "python", "script"
        return "python", ""
    if re.match(r"^\./", toks[i]):
        return "./local_binary", ""
    sub = ""
    if prog in SUBCOMMAND_PROGS:
        for t in toks[i + 1:]:
            if t.startswith("-"):
                continue
            if prog in ("ssh", "scp", "rsync", "open", "psql", "sqlite3", "lsof", "ps",
                        "nvidia-smi", "osascript", "codesign"):
                break
            if re.match(r"^[a-z][\w:-]*$", t):
                sub = t.lower()
            break
    return prog, sub


def iter_steps():
    with EPISODES.open() as fh:
        for line in fh:
            yield json.loads(line)


# --------------------------------------------------------- goal intents
STOP = set("""a an the and or of to in on for with from by at as is are be this that these
those it its i me my we our you your please can could would should will just now then
also into about all any some up out over do does did make get let lets so not no yes if
what how why when where which who there here than via using use run it's i'm dont don't
ok okay hey hi hello thanks thank more one two new other like need want see look check
have but try was them they know something actually yeah dude com bytes been were had has
got going really think sure good still only even very much many every back same done
again because while after before into onto them they their there's""".split())
INTENT_RES = {
    # intent id: regex over the lowercased goal
    "weather_forecast": r"\b(weather|forecast|temperature|rain|snow)\b",
    "email_triage": r"\b(e-?mails?|inbox|gmail|unsubscribe)\b",
    "market_quote": r"\b(stock|ticker|quote|price|prices|market|crypto|btc|eth)\b",
    "pull_request": r"\b(pr|pull request|prs)\b",
    "git_commit_push": r"\b(commit|push|rebase|merge|branch)\b",
    "build_compile": r"\b(build|compile|make|link)\b",
    "test_suite": r"\b(tests?|pytest|ctest)\b",
    "benchmark_perf": r"\b(benchmark|bench|perf|latency|throughput|optimi[sz]e)\b",
    "deploy_release": r"\b(deploy|release|ship|publish)\b",
    "debug_fix": r"\b(fix|bug|broken|crash|error|debug)\b",
    "research_web": r"\b(research|search|find out|look up|news)\b",
    "summarize_doc": r"\b(summari[sz]e|summary|tl;?dr)\b",
    "code_review": r"\b(review|audit)\b",
    "refactor_code": r"\b(refactor|cleanup|clean up|simplify)\b",
    "swarm_agents": r"\b(swarm|agents?|subagents?|parallel)\b",
    "gpu_compute": r"\b(gpu|cuda|h100|a100|vast|runpod|modal)\b",
    "model_training": r"\b(train|training|fine-?tune|finetune|lora|rl)\b",
    "inference_serving": r"\b(llama|vllm|llamacpp|inference|serve|serving)\b",
    "ui_tui_render": r"\b(tui|ui|render|kitty|terminal|screenshot)\b",
    "docs_writing": r"\b(docs?|readme|documentation|write up|writeup)\b",
    "data_pipeline": r"\b(dataset|csv|parquet|etl|scrape|ingest)\b",
    "database_query": r"\b(sql|sqlite|postgres|database|db)\b",
    "calendar_schedule": r"\b(calendar|meeting|schedule|remind)\b",
    "finance_capital": r"\b(revenue|invoice|stripe|cash|capital|pricing|ledger)\b",
    "outreach_gtm": r"\b(outreach|prospect|lead|cold email|customer|sales)\b",
    "screen_desktop_control": r"\b(screen|click|desktop|window|app|browser)\b",
    "install_setup": r"\b(install|setup|set up|configure|config)\b",
    "status_report": r"\b(status|progress|report|what happened|where are we)\b",
    "continue_resume": r"^\s*(continue|go on|keep going|proceed|resume|next)\b",
    "approval_yes": r"^\s*(yes|y|do it|go|approved?|sure)\b",
    "file_cleanup": r"\b(delete|remove|clean|untracked|disk)\b",
}
INTENT_RES = {k: re.compile(v) for k, v in INTENT_RES.items()}

# ----------------------------------------------------- candidate templates
# key: "prog sub" or "prog" or "tool:<name>" or "intent:<id>"
# value: (family, verb, object, description)
T: dict[str, tuple[str, str, str, str]] = {
    # git
    "git": ("CODE", "invoke", "git", "Run a git command against the working repository."),
    "git status": ("CODE", "inspect", "worktree_status", "Report modified, staged and untracked files in the working tree."),
    "git diff": ("CODE", "diff", "worktree", "Show uncommitted changes in the working tree or index."),
    "git log": ("CODE", "read", "commit_history", "Read recent commit history of a branch."),
    "git show": ("CODE", "inspect", "commit", "Show the content and metadata of one commit."),
    "git add": ("CODE", "stage", "change", "Stage working-tree changes for the next commit."),
    "git commit": ("CODE", "create", "commit", "Record staged changes as a new commit."),
    "git push": ("CODE", "push", "branch", "Publish local commits to a remote branch."),
    "git pull": ("CODE", "pull", "branch", "Fetch and integrate remote commits into the local branch."),
    "git fetch": ("CODE", "fetch", "remote_refs", "Download remote refs without changing the working tree."),
    "git checkout": ("CODE", "switch", "branch", "Switch the working tree to another branch or revision."),
    "git switch": ("CODE", "switch", "branch", "Switch the working tree to another branch or revision."),
    "git branch": ("CODE", "list", "git_branch", "List or create local branches."),
    "git stash": ("CODE", "stash", "change", "Shelve uncommitted changes for later restoration."),
    "git rev-parse": ("CODE", "resolve", "revision", "Resolve a ref, path or repo root to a concrete identifier."),
    "git ls-files": ("CODE", "list", "tracked_file", "List files tracked or ignored by version control."),
    "git worktree": ("CODE", "create", "worktree", "Create or manage an isolated git worktree."),
    "git rebase": ("CODE", "rebase", "branch", "Replay local commits onto a new base."),
    "git merge": ("CODE", "merge", "branch", "Merge another branch into the current one."),
    "git reset": ("CODE", "reset", "index", "Move HEAD or unstage changes."),
    "git restore": ("CODE", "restore", "worktree_file", "Discard working-tree changes to a file."),
    "git remote": ("CODE", "inspect", "remote", "List or configure git remotes."),
    "git blame": ("CODE", "blame", "line", "Attribute each line of a file to the commit that last changed it."),
    "git grep": ("KNOW", "search", "tracked_file", "Search tracked files with git grep."),
    "git tag": ("CODE", "tag", "release_tag", "Create or list release tags."),
    "git clone": ("CODE", "clone", "repository", "Clone a remote repository locally."),
    "git cherry-pick": ("CODE", "cherry_pick", "commit", "Apply a specific commit onto the current branch."),
    "git config": ("CODE", "configure", "git", "Read or set git configuration."),
    "git rm": ("CODE", "untrack", "file", "Remove a file from version control."),
    "git mv": ("CODE", "move", "tracked_file", "Rename a tracked file."),
    "git clean": ("CODE", "clean", "untracked_file", "Delete untracked files from the working tree."),
    "gh": ("COLLAB", "invoke", "github_cli", "Run a GitHub CLI command."),
    "gh pr": ("COLLAB", "manage", "pull_request_gh", "Create, view or merge a pull request via the GitHub CLI."),
    "gh run": ("CI", "inspect", "workflow_run", "Inspect a CI workflow run and its logs."),
    "gh api": ("API", "call", "github_api", "Call the GitHub REST/GraphQL API."),
    "gh repo": ("COLLAB", "inspect", "github_repo", "View or create a GitHub repository."),
    "gh issue": ("COLLAB", "manage", "github_issue", "Create, view or comment on a GitHub issue."),
    # file and text utilities
    "cat": ("ENV", "dump", "file", "Print an entire file to stdout via the shell."),
    "head": ("ENV", "read", "file_head", "Read the first lines of a file."),
    "tail": ("ENV", "read", "file_tail", "Read the last lines of a file or follow a log."),
    "sed": ("TEXT", "stream_edit", "text", "Filter or print a line range with sed."),
    "awk": ("TEXT", "extract", "column", "Extract or aggregate columns from text with awk."),
    "grep": ("KNOW", "grep", "text", "Search text or files with a regular expression from the shell."),
    "rg": ("KNOW", "grep", "text", "Search text or files with a regular expression from the shell."),
    "find": ("ENV", "find", "file", "Find files by name, type or age in a directory tree."),
    "ls": ("ENV", "list", "directory_shell", "List directory contents from the shell."),
    "wc": ("TEXT", "count", "line", "Count lines, words or bytes."),
    "sort": ("TEXT", "sort", "line", "Sort lines of text."),
    "uniq": ("TEXT", "count", "unique_line", "Collapse or count repeated lines."),
    "cut": ("TEXT", "extract", "field", "Extract delimited fields from lines."),
    "tr": ("TEXT", "transliterate", "char", "Translate or delete characters in a stream."),
    "jq": ("DATA", "query", "json", "Query or transform JSON with jq."),
    "echo": ("ENV", "print", "message", "Print a literal message or variable to stdout."),
    "printf": ("ENV", "print", "message", "Print a formatted message to stdout."),
    "cd": ("ENV", "change", "cwd", "Change the shell working directory."),
    "pwd": ("ENV", "read", "cwd", "Print the current working directory."),
    "mkdir": ("ENV", "create", "directory", "Create a directory."),
    "rm": ("ENV", "delete", "path", "Delete files or directories."),
    "cp": ("ENV", "copy", "path", "Copy files or directories."),
    "mv": ("ENV", "move", "path", "Move or rename files or directories."),
    "chmod": ("ENV", "chmod", "file", "Change file permissions."),
    "touch": ("ENV", "touch", "file", "Create an empty file or update its timestamp."),
    "ln": ("ENV", "link", "path", "Create a symbolic or hard link."),
    "du": ("ENV", "measure", "disk_usage", "Measure disk usage of a path."),
    "df": ("ENV", "inspect", "disk_space", "Report free disk space per volume."),
    "stat": ("ENV", "stat", "file", "Report file metadata (size, mtime, mode)."),
    "file": ("ENV", "identify", "file_type", "Identify a file's type."),
    "tree": ("ENV", "list", "directory_tree", "Render a directory tree."),
    "diff": ("TEXT", "diff", "file", "Compare two files line by line."),
    "tee": ("ENV", "tee", "output", "Duplicate output to a file while printing it."),
    "xxd": ("ENV", "hexdump", "file", "Show a binary file as hex."),
    "strings": ("ENV", "extract", "binary_strings", "Extract printable strings from a binary."),
    "tar": ("ENV", "archive", "tarball", "Create or extract a tar archive."),
    "unzip": ("ENV", "extract", "zip_archive", "Extract a zip archive."),
    "basename": ("TEXT", "parse", "path", "Parse a path into its components."),
    "dirname": ("TEXT", "parse", "path", "Parse a path into its components."),
    "realpath": ("ENV", "resolve", "path", "Resolve a path to its canonical absolute form."),
    "which": ("ENV", "locate", "executable", "Locate an executable on PATH."),
    "command": ("ENV", "locate", "executable", "Locate an executable on PATH."),
    "type": ("ENV", "locate", "executable", "Locate an executable on PATH."),
    "date": ("TIME", "read", "clock", "Read the system clock or format a date."),
    "sleep": ("TIME", "sleep", "shell", "Pause the shell for a fixed duration."),
    "md5": ("SEC", "hash", "file", "Compute a file digest."),
    "shasum": ("SEC", "hash", "file", "Compute a file digest."),
    "sha256sum": ("SEC", "hash", "file", "Compute a file digest."),
    # build and languages
    "make": ("CI", "run", "make_target", "Run a Makefile target."),
    "cmake": ("CI", "configure", "cmake_build", "Configure or build a CMake project."),
    "ninja": ("CI", "run", "ninja_build", "Run an incremental ninja build."),
    "cc": ("CODE", "compile", "c_unit", "Compile C sources with the system compiler."),
    "clang": ("CODE", "compile", "c_unit", "Compile C sources with the system compiler."),
    "gcc": ("CODE", "compile", "c_unit", "Compile C sources with the system compiler."),
    "zig": ("CODE", "invoke", "zig_toolchain", "Build or cross-compile with the zig toolchain."),
    "swift": ("CODE", "invoke", "swift_toolchain", "Build or run Swift code."),
    "swiftc": ("CODE", "compile", "swift_unit", "Compile Swift sources."),
    "xcrun": ("CODE", "invoke", "xcode_tool", "Run an Xcode developer tool."),
    "cargo": ("CODE", "invoke", "cargo", "Build, test or run a Rust crate."),
    "rustc": ("CODE", "compile", "rust_unit", "Compile Rust sources."),
    "go": ("CODE", "invoke", "go_toolchain", "Build, test or run Go code."),
    "node": ("CODE", "run", "node_script", "Run JavaScript with node."),
    "npx": ("PKG", "run", "npx_package", "Run a package binary via npx."),
    "npm": ("PKG", "invoke", "npm", "Install or run scripts with npm."),
    "pnpm": ("PKG", "invoke", "pnpm", "Install or run scripts with pnpm."),
    "bun": ("CODE", "run", "bun_script", "Run or install JavaScript with bun."),
    "deno": ("CODE", "run", "deno_script", "Run TypeScript with deno."),
    "python": ("CODE", "run", "python", "Run the Python interpreter."),
    "python -c": ("CODE", "eval", "python_snippet", "Evaluate an inline Python snippet."),
    "python script": ("CODE", "run", "python_script", "Run a Python script file."),
    "python -m pytest": ("CI", "run", "pytest", "Run the Python test suite with pytest."),
    "python -m pip": ("PKG", "install", "python_package", "Install a Python package with pip."),
    "python -m json": ("DATA", "format", "json", "Pretty-print or validate JSON."),
    "python -m http": ("INFRA", "serve", "static_http", "Serve a directory over local HTTP."),
    "pytest": ("CI", "run", "pytest", "Run the Python test suite with pytest."),
    "pip": ("PKG", "install", "python_package", "Install a Python package with pip."),
    "pip3": ("PKG", "install", "python_package", "Install a Python package with pip."),
    "uv": ("PKG", "invoke", "uv", "Manage Python environments and packages with uv."),
    "uv run": ("CODE", "run", "uv_script", "Run a command inside a uv-managed environment."),
    "uv pip": ("PKG", "install", "python_package", "Install a Python package with pip."),
    "brew": ("PKG", "invoke", "homebrew", "Install or query packages via Homebrew."),
    "erl": ("CODE", "run", "erlang_vm", "Run code on the Erlang VM."),
    "mix": ("CODE", "invoke", "elixir_mix", "Build or test an Elixir project with mix."),
    "./local_binary": ("CODE", "run", "local_binary", "Run a binary built in the current repository."),
    "dsco": ("CTRL", "invoke", "dsco_cli", "Invoke the dsco CLI itself as a subprocess."),
    # network and infra
    "curl": ("WEB", "request", "http_shell", "Make an HTTP request from the shell."),
    "wget": ("WEB", "download", "file", "Download a file over HTTP."),
    "ssh": ("INFRA", "exec", "remote_host", "Run a command on a remote host over SSH."),
    "scp": ("INFRA", "copy", "remote_file", "Copy files to or from a remote host."),
    "rsync": ("INFRA", "sync", "remote_dir", "Synchronize a directory with a remote host."),
    "docker": ("INFRA", "invoke", "docker", "Build or run containers with docker."),
    "kubectl": ("INFRA", "invoke", "kubernetes", "Query or change a Kubernetes cluster."),
    "tailscale": ("INFRA", "inspect", "tailnet", "Inspect tailnet peers and status."),
    "nc": ("INFRA", "probe", "tcp_port", "Probe a TCP port."),
    "ping": ("INFRA", "probe", "host_reachability", "Check that a host is reachable."),
    "lsof": ("ENV", "inspect", "open_port", "List processes holding files or ports."),
    "ps": ("ENV", "list", "os_process", "List running OS processes."),
    "pgrep": ("ENV", "find", "os_process", "Find running processes by name."),
    "kill": ("ENV", "kill", "os_process", "Send a signal to an OS process."),
    "pkill": ("ENV", "kill", "os_process", "Send a signal to an OS process."),
    "top": ("PERF", "sample", "system_load", "Sample CPU and memory load."),
    "vm_stat": ("PERF", "sample", "memory_pressure_os", "Sample virtual memory statistics."),
    "sysctl": ("ENV", "read", "kernel_param", "Read OS/kernel parameters."),
    "uname": ("ENV", "read", "platform_info", "Read OS and architecture identity."),
    "nvidia-smi": ("COMPUTE", "inspect", "gpu_device", "Inspect GPU utilization and memory."),
    "launchctl": ("INFRA", "manage", "launchd_service", "Load, list or restart a launchd service."),
    "tmux": ("ENV", "manage", "tmux_session", "Create or drive a tmux session."),
    "kitty": ("UI", "control", "kitty_terminal", "Drive the kitty terminal via remote control."),
    "osascript": ("UI", "run", "applescript", "Run AppleScript to control a macOS app."),
    "open": ("UI", "open", "app_or_file", "Open a file, URL or app with the OS handler."),
    "screencapture": ("PERC", "capture", "screenshot", "Capture a screenshot of the screen or a window."),
    "defaults": ("ENV", "read", "macos_defaults", "Read or write macOS preference domains."),
    "security": ("SEC", "read", "keychain_item", "Read a keychain item or certificate."),
    "codesign": ("SEC", "sign", "binary", "Sign or verify a binary's code signature."),
    "sqlite3": ("DB", "query", "sqlite", "Query a local SQLite database."),
    "psql": ("DB", "query", "postgres", "Query a Postgres database."),
    "ollama": ("COMPUTE", "invoke", "ollama", "Manage or run local models with ollama."),
    "modal": ("COMPUTE", "invoke", "modal", "Deploy or run jobs on Modal."),
    "hf": ("DATA", "invoke", "huggingface_hub", "Upload or download Hugging Face Hub artifacts."),
    "huggingface-cli": ("DATA", "invoke", "huggingface_hub", "Upload or download Hugging Face Hub artifacts."),
    "ffmpeg": ("MEDIA", "transcode", "video", "Transcode or extract media with ffmpeg."),
    "sips": ("MEDIA", "transform", "image", "Resize or convert an image."),
    "say": ("MEDIA", "speak", "text", "Speak text aloud via TTS."),
    "pbcopy": ("UI", "write", "clipboard", "Copy text to the clipboard."),
    "pbpaste": ("UI", "read", "clipboard", "Read text from the clipboard."),
    "base64": ("TEXT", "encode", "base64", "Encode or decode base64."),
    "bash": ("ENV", "run", "shell_script", "Run a shell script in a subshell."),
    "sh": ("ENV", "run", "shell_script", "Run a shell script in a subshell."),
    "zsh": ("ENV", "run", "shell_script", "Run a shell script in a subshell."),
    "export": ("ENV", "set", "environment_variable", "Set an environment variable."),
    "cat >": ("ENV", "write", "heredoc_file", "Write a file through a shell heredoc."),
}
INTENT_T: dict[str, tuple[str, str, str, str]] = {
    "weather_forecast": ("WEB", "fetch", "weather_forecast", "Fetch current conditions or a forecast for a place."),
    "email_triage": ("COMM", "triage", "inbox", "Search, read and sort messages in an email inbox."),
    "market_quote": ("ECON", "fetch", "market_quote", "Fetch a quote or price series for a ticker or asset."),
    "pull_request": ("COLLAB", "prepare", "pull_request", "Assemble changes, summary and checks into a pull request."),
    "git_commit_push": ("CODE", "publish", "commit", "Commit and push a reviewed change set."),
    "benchmark_perf": ("PERF", "run", "benchmark", "Run a benchmark and record timing results."),
    "deploy_release": ("INFRA", "ship", "release", "Cut and publish a release artifact."),
    "summarize_doc": ("TEXT", "summarize", "document", "Summarize a document or thread for the user."),
    "gpu_compute": ("COMPUTE", "rent", "gpu_instance", "Rent or reserve a GPU instance for a job."),
    "model_training": ("LEARN", "launch", "training_run", "Launch a model training or fine-tuning run."),
    "inference_serving": ("COMPUTE", "serve", "model_endpoint", "Start or route to a model inference server."),
    "ui_tui_render": ("UI", "verify", "terminal_render", "Render the TUI and verify its visual output."),
    "calendar_schedule": ("CAL", "schedule", "calendar_event", "Create or move a calendar event."),
    "finance_capital": ("ECON", "record", "ledger_entry", "Record an evidenced financial ledger entry."),
    "outreach_gtm": ("COMM", "draft", "outreach_message", "Draft an outreach message for a prospect (no send)."),
    "screen_desktop_control": ("UI", "drive", "desktop_app", "Drive a desktop app through screen observation and input."),
    "status_report": ("COLLAB", "report", "status", "Report progress and blockers on current work."),
    "continue_resume": ("GOAL", "continue", "previous_goal", "Resume the prior goal after a terse 'continue' instruction."),
    "approval_yes": ("GOAL", "accept", "proposal", "Treat a terse affirmative as approval of the last proposal."),
    "file_cleanup": ("ENV", "clean", "workspace", "Remove untracked or stale files from a workspace after review."),
    "docs_writing": ("DOCW", "write", "doc_page", "Write or update a documentation page."),
    "data_pipeline": ("DATA", "ingest", "dataset", "Ingest or scrape records into a dataset."),
    "research_web": ("RESEARCH", "survey", "web_sources", "Survey web sources and synthesize findings."),
    "swarm_agents": ("SWARM", "launch", "agent_team", "Launch a team of parallel agents on sub-tasks."),
}
PROG_FAMILY_FALLBACK = "ENV"

# Activities whose meaning an existing op already names. These are labeler
# gaps (the shell call is labeled ENV.INVOKE.SHELL_COMMAND although a specific
# op exists), not vocabulary gaps. Entries whose target op is absent from the
# table are ignored at runtime.
EQUIV: dict[str, str] = {
    "cat": "ENV.READ.FILE", "head": "ENV.READ.FILE", "tail": "ENV.READ.FILE",
    "ls": "ENV.LIST.DIRECTORY", "tree": "ENV.LIST.DIRECTORY",
    "cd": "ENV.CWD", "pwd": "ENV.CWD",
    "grep": "KNOW.SEARCH.CODEBASE", "rg": "KNOW.SEARCH.CODEBASE", "git grep": "KNOW.SEARCH.CODEBASE",
    "ps": "ENV.LIST.PROCESS", "pgrep": "ENV.LIST.PROCESS",
    "kill": "ENV.KILL.PROCESS", "pkill": "ENV.KILL.PROCESS",
    "mkdir": "ENV.CREATE.DIRECTORY", "rm": "ENV.DELETE.FILE", "cp": "ENV.COPY.FILE",
    "mv": "ENV.MOVE.FILE", "stat": "ENV.STAT.FILE", "diff": "ENV.DIFF.FILE",
    "git diff": "ENV.DIFF.REPOSITORY", "git commit": "ENV.COMMIT.REPOSITORY",
    "make": "ENV.BUILD", "cmake": "ENV.BUILD", "ninja": "ENV.BUILD",
    "make test": "VERIF.RUN_TESTS", "pytest": "VERIF.RUN_TESTS", "python -m pytest": "VERIF.RUN_TESTS",
    "pip": "ENV.INSTALL", "pip3": "ENV.INSTALL", "uv pip": "ENV.INSTALL", "python -m pip": "ENV.INSTALL",
    "brew install": "ENV.INSTALL", "npm install": "ENV.INSTALL",
    "curl": "ENV.FETCH.HTTP_ENDPOINT", "wget": "ENV.FETCH.HTTP_ENDPOINT",
    "df": "ENV.DISK_SPACE", "nvidia-smi": "ENV.GPU_STATUS", "vm_stat": "ENV.MEMORY_PRESSURE",
    "uname": "ENV.PLATFORM", "gh pr": "CODE.CREATE.PULL_REQUEST",
    "screencapture": "PERC.CAPTURE.SCREEN", "sleep": "TIME.SLEEP.AGENT_PROCESS",
    "tool:WebSearch": "KNOW.SEARCH.WEB", "tool:WebFetch": "WEB.READ.PAGE",
    "tool:Grep": "KNOW.SEARCH.CODEBASE", "tool:grep_files": "KNOW.SEARCH.CODEBASE",
    "tool:read_file": "ENV.READ.FILE", "tool:code_read": "CODE.READ.MODULE",
    "tool:list_directory": "ENV.LIST.DIRECTORY", "tool:swarm": "SWARM.SPAWN.SUBAGENT",
    "tool:agent": "SWARM.DELEGATE.SUBAGENT", "tool:goal_queue": "GOAL.LIST.GOAL_QUEUE",
    "tool:get_goal": "GOAL.INSPECT.GOAL", "tool:view_image": "MEDIA.READ.IMAGE",
    "tool:jina_search": "KNOW.SEARCH.WEB", "tool:jina_ai_search": "KNOW.SEARCH.WEB",
    "tool:parallel_search": "KNOW.SEARCH.WEB", "tool:parallel_ai_search": "KNOW.SEARCH.WEB",
    "tool:jina_read": "WEB.READ.PAGE", "tool:jina_ai_reader": "WEB.READ.PAGE",
    "tool:http_request": "ENV.FETCH.HTTP_ENDPOINT", "tool:curl_raw": "ENV.FETCH.HTTP_ENDPOINT",
    "tool:native_window": "UI.INSPECT.WINDOW", "tool:computer": "PERC.CAPTURE.SCREEN",
    "tool:desktop": "PERC.CAPTURE.SCREEN",
    "intent:pull_request": "CODE.CREATE.PULL_REQUEST", "intent:calendar_schedule": "CAL.CREATE.CALENDAR_EVENT",
    "intent:deploy_release": "CI.CREATE.RELEASE", "intent:summarize_doc": "PERC.SUMMARIZE.DOCUMENT",
    "intent:swarm_agents": "SWARM.SPAWN.AGENT_GROUP", "intent:benchmark_perf": "EVAL.BENCHMARK.BENCHMARK_SUITE",
}


def tool_template(name: str) -> tuple[str, str, str, str]:
    base = re.sub(r"^mcp__[\w-]+?__", "", name)
    base = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    fam = "API"
    if re.search(r"search|jina|parallel|web|fetch|browser|http|curl", base):
        fam = "WEB"
    elif re.search(r"email|mail|ipc|send|message", base):
        fam = "COMM"
    elif re.search(r"window|desktop|computer|screen|surface|kitty|buffer|view_image|native", base):
        fam = "UI"
    elif re.search(r"swarm|agent|blackboard", base):
        fam = "SWARM"
    elif re.search(r"weather|nws", base):
        fam = "WEB"
    elif re.search(r"goal", base):
        fam = "GOAL"
    elif re.search(r"governance|killswitch", base):
        fam = "GOV"
    return fam, "call", base, f"Call the '{base}' tool directly (currently routed through generic dispatch)."


# ---------------------------------------------------------------- analysis
def mine(parts: set[str], mnemonics: set[str]) -> dict:
    prog_n: Counter = Counter()
    progsub_n: Counter = Counter()
    prog_eps: dict[str, set] = defaultdict(set)
    progsub_eps: dict[str, set] = defaultdict(set)
    examples: dict[str, str] = {}
    shell_calls = 0
    inner_tool_n: Counter = Counter()
    inner_tool_eps: dict[str, set] = defaultdict(set)
    tool_n: Counter = Counter()
    goal_lead: Counter = Counter()
    goal_uni: Counter = Counter()
    goal_bi: Counter = Counter()
    intent_n: Counter = Counter()
    intent_ex: dict[str, str] = {}
    n_eps = 0
    goals_seen: set[str] = set()
    private_terms: set[str] = set()
    heredoc = re.compile(r"\bcat\s*>{1,2}\s*\S+\s*<<")
    for ei, e in enumerate(iter_steps()):
        n_eps += 1
        goal = (e.get("goal") or "").strip()
        gl = goal.lower()
        gkey = " ".join(gl.split())[:80]
        for u in re.findall(r"/(?:users|home)/([\w.-]+)", gl):
            private_terms.add(u)
        if gl and gkey not in goals_seen:
            goals_seen.add(gkey)
            words = re.findall(r"[a-z][a-z0-9'_-]+", gl)
            content = [w for w in words if w not in STOP and len(w) > 2 and "'" not in w]
            if words:
                goal_lead[words[0]] += 1
            goal_uni.update(set(content))
            goal_bi.update({f"{a} {b}" for a, b in zip(content, content[1:])})
            for iid, rx in INTENT_RES.items():
                if rx.search(gl):
                    intent_n[iid] += 1
                    intent_ex.setdefault(iid, anonymize(goal, 60))
        for s in e.get("steps") or []:
            tool = s.get("tool") or ""
            tool_n[tool] += 1
            args = s.get("args")
            if tool == "invoke_tool":
                try:
                    a = json.loads(args) if isinstance(args, str) else args
                    nm = a.get("name") if isinstance(a, dict) else None
                except Exception:
                    nm = None
                if nm:
                    inner_tool_n[nm] += 1
                    inner_tool_eps[nm].add(ei)
                continue
            if tool not in SHELL_TOOLS:
                continue
            try:
                a = json.loads(args) if isinstance(args, str) else args
                cmd = a.get("command") if isinstance(a, dict) else None
            except Exception:
                cmd = None
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            shell_calls += 1
            seen_here = set()
            if heredoc.search(cmd):
                seen_here.add(("cat >", ""))
            body = re.sub(r"<<-?\s*'?(\w+)'?.*?\n\1\b", " ", cmd, flags=re.S)
            for seg in split_segments(body):
                ps = _program_and_sub(seg)
                if ps:
                    seen_here.add(ps)
            for prog, sub in seen_here:
                prog_n[prog] += 1
                prog_eps[prog].add(ei)
                key = f"{prog} {sub}".strip()
                if sub:
                    progsub_n[key] += 1
                    progsub_eps[key].add(ei)
                if key not in examples:
                    examples[key] = anonymize(cmd)
                if prog not in examples:
                    examples[prog] = anonymize(cmd)

    def named(key: str) -> str | None:
        """Existing op whose mnemonic names this activity, if any."""
        if EQUIV.get(key) in mnemonics:
            return EQUIV[key]
        if key in T:
            fam, verb, obj, _ = T[key]
            m = f"{fam}.{verb}.{obj}".upper()
            if m in mnemonics:
                return m
        toks = re.split(r"[\s_.:-]+", key.lower().replace("-m ", ""))
        toks = [t for t in toks if t and len(t) > 2 and t not in ("python", "local", "binary")]
        hits = [t for t in toks if t in parts]
        # program-level coverage needs the program itself named; subcommand
        # coverage needs both the program and the verb (e.g. "git commit").
        if key.split()[0] in parts and (len(toks) <= 1 or all(t in parts for t in toks)):
            return "~" + "+".join(hits)
        return None

    prog_rows = []
    for prog, c in prog_n.most_common():
        prog_rows.append({"key": prog, "calls": c, "episodes": len(prog_eps[prog]),
                          "named_by": named(prog), "example": examples.get(prog, "")})
    sub_rows = []
    for key, c in progsub_n.most_common():
        sub_rows.append({"key": key, "calls": c, "episodes": len(progsub_eps[key]),
                         "named_by": named(key), "example": examples.get(key, "")})
    inner_rows = []
    for nm, c in inner_tool_n.most_common():
        base = re.sub(r"^mcp__[\w-]+?__", "", nm).lower()
        eq = EQUIV.get(f"tool:{nm}")
        nb = eq if eq in mnemonics else (("~" + base) if base in parts else None)
        inner_rows.append({"key": f"tool:{nm}", "calls": c, "episodes": len(inner_tool_eps[nm]),
                           "named_by": nb})
    term_rows = []
    for term, c in goal_uni.most_common(400):
        if (term in parts or term.rstrip("s") in parts or any(p in term for p in private_terms)
                or (_LOCAL_USER and _LOCAL_USER.search(term))):
            continue
        term_rows.append({"term": term, "goals": c})
    bi_rows = [{"bigram": b, "goals": c} for b, c in goal_bi.most_common(80)
               if not all(w in parts for w in b.split())
               and not any(p in b for p in private_terms)]
    intent_rows = []
    for k, c in intent_n.most_common():
        eq = EQUIV.get(f"intent:{k}")
        intent_rows.append({"intent": k, "goals": c, "example": intent_ex.get(k, ""),
                            "named_by": eq if eq in mnemonics else None})
    return {
        "episodes": n_eps, "distinct_goals": len(goals_seen),
        "shell_calls": shell_calls, "distinct_programs": len(prog_n),
        "programs": prog_rows, "subcommands": sub_rows,
        "inner_tools": inner_rows,
        "goal_lead_verbs": goal_lead.most_common(40),
        "goal_terms_unnamed": term_rows[:80],
        "goal_bigrams_unnamed": bi_rows[:40],
        "goal_intents": intent_rows,
        "tool_calls": sum(tool_n.values()),
    }


def build_candidates(m: dict, mnemonics: set[str], limit: int) -> list[dict]:
    pool: dict[str, dict] = {}

    def add(key: str, tmpl, evidence: int, eps: int, example: str, source: str):
        fam, verb, obj, desc = tmpl
        mn = f"{fam}.{verb}.{obj}".upper()
        if mn in mnemonics:
            return
        cur = pool.get(mn)
        if cur:
            cur["evidence_count"] += evidence
            cur["episodes"] += eps
            cur["sources"].append(key)
            return
        pool[mn] = {"mnemonic": mn, "family": fam, "verb": verb, "object": obj,
                    "desc": desc, "evidence_count": evidence, "episodes": eps,
                    "example": example, "source": source, "sources": [key]}

    # subcommands first (more specific), then programs not already explained
    for r in m["subcommands"]:
        if r["named_by"] or r["calls"] < 15 or r["episodes"] < 5:
            continue
        tmpl = T.get(r["key"])
        if tmpl is None and re.match(r"^make (test|smoke|bench)[-_]", r["key"]):
            tmpl = ("CI", "run", "make_test_target",
                    "Run a named Makefile test/smoke/bench target.")
        if tmpl is None:
            prog, sub = r["key"].split(" ", 1)
            if prog not in T or prog == "make":
                continue
            fam = T[prog][0]
            obj = re.sub(r"[^a-z0-9]+", "_", f"{prog}_{sub}".lower()).strip("_")
            tmpl = (fam, "run", obj, f"Run '{prog} {sub}' from the shell.")
        add(r["key"], tmpl, r["calls"], r["episodes"], r["example"], "shell_subcommand")
    for r in m["programs"]:
        if r["named_by"] or r["calls"] < 15 or r["episodes"] < 5:
            continue
        tmpl = T.get(r["key"])
        if tmpl is None and r["episodes"] < 15:
            continue
        if tmpl is None:
            obj = re.sub(r"[^a-z0-9]+", "_", r["key"].lower()).strip("_")
            tmpl = (PROG_FAMILY_FALLBACK, "run", obj, f"Run the '{r['key']}' program from the shell.")
        add(r["key"], tmpl, r["calls"], r["episodes"], r["example"], "shell_program")
    for r in m["inner_tools"]:
        if r["named_by"] or r["calls"] < 10:
            continue
        name = r["key"][5:]
        add(r["key"], tool_template(name), r["calls"], r["episodes"],
            f"invoke_tool(name={anonymize(name, 40)})", "generic_dispatch")
    for r in m["goal_intents"]:
        if r["named_by"] or r["intent"] not in INTENT_T or r["goals"] < 10:
            continue
        add(f"intent:{r['intent']}", INTENT_T[r["intent"]], r["goals"], r["goals"],
            r["example"], "goal_intent")
    cands = sorted(pool.values(), key=lambda c: -c["evidence_count"])[:limit]
    for c in cands:
        c["token"] = "<|op:" + c["mnemonic"].lower() + "|>"
        c["sources"] = c["sources"][:6]
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-candidates", type=int, default=150)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    ops, mnemonics, parts = load_ops()
    usage = op_usage(ops)
    mined = mine(parts, mnemonics)
    cands = build_candidates(mined, mnemonics, args.max_candidates)

    shell_steps = mined["shell_calls"]
    gap_calls = sum(r["calls"] for r in mined["programs"] if not r["named_by"])
    labeler_gap = sum(r["calls"] for r in mined["programs"]
                      if r["named_by"] and not r["named_by"].startswith("~"))
    prog_calls = sum(r["calls"] for r in mined["programs"])
    summary = {
        "ops_used": f"{usage['ops_used']}/{usage['ops_in_table']}",
        "families_never_used": len(usage["families_never_used"]),
        "zipf_exponent": usage["zipf_exponent"],
        "shell_calls_one_generic_op": shell_steps,
        "shell_program_uses_unnamed_share": round(gap_calls / prog_calls, 4) if prog_calls else 0,
        "shell_program_uses_existing_op_unlabeled_share": round(labeler_gap / prog_calls, 4) if prog_calls else 0,
        "candidates": len(cands),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps({
        "version": 1, "note": "Proposed op-codes for maintainer review; not part of opcodes.py.",
        "count": len(cands), "candidates": cands}, indent=1) + "\n")
    (args.out / "report.json").write_text(json.dumps({
        "summary": summary, "usage": usage, "mined": mined}, indent=1) + "\n")

    u = usage
    print(f"(a) ops used {u['ops_used']}/{u['ops_in_table']} ({u['ops_used_share']:.1%}); "
          f">=10 uses {u['ops_used_ge10']}; once {u['ops_used_once']}; tokens {u['op_tokens']}")
    print(f"    zipf s={u['zipf_exponent']} r2={u['zipf_r2']} head {u['head_share']}")
    print(f"    rank->freq {u['rank_freq']}")
    print(f"    families used {u['families_used']}/{u['families_total']}; never: "
          f"{', '.join(u['families_never_used'])}")
    print(f"(b) shell calls {shell_steps} (all -> ENV.INVOKE.SHELL_COMMAND), "
          f"distinct programs {mined['distinct_programs']}, unnamed program uses "
          f"{summary['shell_program_uses_unnamed_share']:.1%}, existing-op-but-unlabeled "
          f"{summary['shell_program_uses_existing_op_unlabeled_share']:.1%}")
    print("    labeler gaps (op exists, shell call still generic):")
    for r in [r for r in mined["programs"] if r["named_by"] and not r["named_by"].startswith("~")][:10]:
        print(f"      {r['calls']:6d}  {r['key']:<10} -> {r['named_by']}")
    print("    top unnamed programs:")
    for r in [r for r in mined["programs"] if not r["named_by"]][:20]:
        print(f"      {r['calls']:6d} {r['episodes']:5d}ep  {r['key']:<16} {r['example'][:50]}")
    print("    top unnamed subcommands:")
    for r in [r for r in mined["subcommands"] if not r["named_by"]][:15]:
        print(f"      {r['calls']:6d}  {r['key']}")
    print("    top generic-dispatch tools:")
    for r in [r for r in mined["inner_tools"] if not r["named_by"]][:12]:
        print(f"      {r['calls']:6d}  {r['key']}")
    print(f"    goal intents (distinct goals {mined['distinct_goals']}):")
    for r in mined["goal_intents"][:18]:
        print(f"      {r['goals']:6d}  {r['intent']:<24} {r['named_by'] or 'UNNAMED'}")
    print("    unnamed goal terms: " + ", ".join(
        f"{r['term']}({r['goals']})" for r in mined["goal_terms_unnamed"][:25]))
    print(f"(c) {len(cands)} candidates -> {args.out / 'candidates.json'}")
    for c in cands[:20]:
        print(f"      {c['evidence_count']:6d}  {c['mnemonic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
