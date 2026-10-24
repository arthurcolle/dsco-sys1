"""Stream a permissively licensed, language-balanced source-code corpus.

Sources, in order (each file is kept only if its license metadata is permissive):

  1. codeparrot/github-code-clean (HF, ungated, parquet). Per-file ``license``
     column; kept: mit, apache-2.0, bsd-3-clause, bsd-2-clause. Supplies Python,
     C, C++, JavaScript, TypeScript, Go, Rust, Shell and Markdown.
  2. substratusai/the-stack-yaml-k8s (HF, ungated mirror of a The Stack YAML
     subset). Per-file ``max_stars_repo_licenses``; every listed license must be
     permissive. The Stack terms of use apply (recorded in the manifest).
  3. Local fallback for formats the HF sources lack or run short on (JSON, TOML,
     Rust top-up): packages already on this machine, each package's declared
     license recorded (npm package.json, Cargo.toml, Python dist metadata).

Output in ``data/corpus/code/``:

  code-00000.txt ...  one file per line: ``<lang=python> `` + escaped source
  heldout.txt         ~5% of accepted files (content-hash selected), same format
  MANIFEST.json       datasets, revisions, terms, license filter, per-language
                      counts/bytes, local packages with licenses, sha256 per file

Escapes match corpus_en: backslash -> ``\\\\``, LF -> ``\\n``, CR -> ``\\r``.

Run:  python -m sys1.tokenizer.corpus_code [--cap-mb 600] [--max-shards 60]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from importlib import metadata as imeta
from pathlib import Path

from sys1.paths import DATA

OUT_DIR = DATA / "corpus" / "code"
SHARD_BYTES = 64 * 1000 * 1000
HELDOUT_FRACTION = 0.05
MIN_CHARS = 64
MAX_FILE_BYTES = 200_000

GH_DATASET = "codeparrot/github-code-clean"
YAML_DATASET = "substratusai/the-stack-yaml-k8s"
GH_PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"}
SPDX_PERMISSIVE = {"MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "0BSD", "ISC"}
THE_STACK_TERMS = ("The Stack terms of use (https://huggingface.co/datasets/bigcode/the-stack#terms-of-use-for-the-stack): "
                   "use must abide by each file's original license incl. attribution; honor data-removal "
                   "updates; redistribution must include the terms. Mirror is ungated; files filtered to "
                   "permissive licenses here.")
GH_TERMS = ("Dataset card license: apache-2.0 (collection). Files carry their own repository license; "
            "this corpus keeps only per-file license in {mit, apache-2.0, bsd-3-clause, bsd-2-clause}.")

GH_LANG = {"Python": "python", "C": "c", "C++": "cpp", "JavaScript": "javascript", "TypeScript": "typescript",
           "GO": "go", "Rust": "rust", "Shell": "shell", "Markdown": "markdown"}
# Byte quotas as a fraction of the cap: 8 balanced groups of 1/8 each.
GROUP_SHARE = {"python": 1, "c": .5, "cpp": .5, "javascript": .5, "typescript": .5, "go": 1, "rust": 1,
               "shell": 1, "markdown": 1, "yaml": .45, "json": .45, "toml": .10}
LOCAL_EXT = {".json": "json", ".toml": "toml", ".rs": "rust", ".yaml": "yaml", ".yml": "yaml"}


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def unescape(line: str) -> str:
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            nxt = line[i + 1]
            out.append({"n": "\n", "r": "\r", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def spdx_permissive(expr: str | None) -> bool:
    """True if some OR-alternative of an SPDX-ish expression is entirely permissive."""
    if not expr:
        return False
    expr = expr.replace("(", " ").replace(")", " ").replace("/", " OR ")
    for alt in re.split(r"\s+OR\s+", expr, flags=re.I):
        ids = [t.strip() for t in re.split(r"\s+AND\s+", alt, flags=re.I) if t.strip()]
        if ids and all(_norm_spdx(i) in SPDX_PERMISSIVE for i in ids):
            return True
    return False


def _norm_spdx(s: str) -> str:
    s = s.strip().strip('"\'')
    low = s.lower().replace(" license", "").replace("license", "").strip(" -")
    table = {"mit": "MIT", "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache2": "Apache-2.0",
             "apache software": "Apache-2.0", "apache-2": "Apache-2.0", "apache 2": "Apache-2.0",
             "bsd": "BSD-3-Clause", "bsd-3-clause": "BSD-3-Clause", "new bsd": "BSD-3-Clause",
             "3-clause bsd": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause", "simplified bsd": "BSD-2-Clause",
             "isc": "ISC", "0bsd": "0BSD", "mit-0": "MIT-0"}
    return table.get(low, s)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _avail_gb() -> float:
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        pages = sum(int(re.search(rf"Pages {k}:\s+(\d+)", out).group(1)) for k in ("free", "inactive", "speculative"))
        return pages * 16384 / 1e9
    except Exception:
        return 99.0


class Writer:
    def __init__(self, out_dir: Path, cap: int):
        self.out_dir, self.cap = out_dir, cap
        self.quota = {k: int(cap * v / 8) for k, v in GROUP_SHARE.items()}
        self.used = {k: 0 for k in GROUP_SHARE}
        self.stats: dict[str, dict] = {}
        self.seen: set[bytes] = set()
        self.total = 0
        self.shards: list[Path] = []
        self.fh = None
        self.cur = 0
        self.held = (out_dir / "heldout.txt").open("wb")
        self.held_n = self.held_b = 0
        self.train_n = 0

    def full(self, lang: str) -> bool:
        return self.used[lang] >= self.quota[lang] or self.total >= self.cap

    def add(self, lang: str, text: str, source: str) -> bool:
        if self.full(lang) or "\x00" in text:
            return False
        text = text.replace("\r\n", "\n")
        if len(text) < MIN_CHARS or len(text) > MAX_FILE_BYTES or not text.strip():
            return False
        key = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest()
        if key in self.seen:
            return False
        self.seen.add(key)
        line = (f"<lang={lang}> " + escape(text) + "\n").encode("utf-8", "replace")
        n = len(line)
        heldout = int.from_bytes(key[:4], "big") % 10_000 < HELDOUT_FRACTION * 10_000
        if heldout:
            self.held.write(line)
            self.held_n += 1
            self.held_b += n
        else:
            if self.fh is None or self.cur + n > SHARD_BYTES:
                if self.fh:
                    self.fh.close()
                p = self.out_dir / f"code-{len(self.shards):05d}.txt"
                self.shards.append(p)
                self.fh = p.open("wb")
                self.cur = 0
            self.fh.write(line)
            self.cur += n
            self.train_n += 1
        self.used[lang] += n
        self.total += n
        s = self.stats.setdefault(lang, {"files": 0, "bytes": 0, "heldout_files": 0, "by_source": {}})
        s["files"] += 1
        s["bytes"] += n
        s["heldout_files"] += int(heldout)
        bs = s["by_source"].setdefault(source, {"files": 0, "bytes": 0})
        bs["files"] += 1
        bs["bytes"] += n
        return True

    def close(self):
        if self.fh:
            self.fh.close()
        self.held.close()


def log(w: Writer, msg: str):
    used = " ".join(f"{k}={w.used[k] / 1e6:.0f}" for k in GROUP_SHARE)
    print(f"[corpus_code] {msg} total={w.total / 1e6:.0f}MB | {used}", file=sys.stderr, flush=True)


def _revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def _iter_row_groups(fs, path: str, columns: list[str], retries: int = 4):
    import pyarrow.parquet as pq
    rg = 0
    n_rg = None
    while n_rg is None or rg < n_rg:
        for attempt in range(retries):
            try:
                with fs.open(path, block_size=8 << 20) as f:
                    pf = pq.ParquetFile(f)
                    n_rg = pf.metadata.num_row_groups
                    while rg < n_rg:
                        yield pf.read_row_group(rg, columns=columns)
                        rg += 1
                break
            except Exception as e:  # network hiccup: reopen and resume at this row group
                if attempt == retries - 1:
                    print(f"[corpus_code] giving up on {path} rg={rg}: {e}", file=sys.stderr)
                    return
                time.sleep(5 * (attempt + 1))


def stream_github(w: Writer, max_shards: int, budget_s: float) -> dict:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{GH_DATASET}/data/train-*.parquet"))
    langs = set(GH_LANG.values())
    used_shards = 0
    t0 = time.time()
    stop = "quotas filled"
    for path in files[:max_shards]:
        if all(w.full(l) for l in langs):
            break
        if time.time() - t0 > budget_s:
            stop = f"time budget {budget_s:.0f}s reached"
            break
        while _avail_gb() < 2.0:
            print("[corpus_code] low memory, waiting", file=sys.stderr, flush=True)
            time.sleep(20)
        for t in _iter_row_groups(fs, path, ["code", "language", "license"]):
            for code, lang, lic in zip(t["code"].to_pylist(), t["language"].to_pylist(), t["license"].to_pylist()):
                tag = GH_LANG.get(lang)
                if tag and lic in GH_PERMISSIVE:
                    w.add(tag, code, GH_DATASET)
            del t
        used_shards += 1
        log(w, f"gh shard {used_shards}/{max_shards} {path.rsplit('/', 1)[-1]}")
    else:
        stop = f"max_shards {max_shards} reached"
    return {"shards_read": used_shards, "shards_total": len(files), "stopped": stop,
            "seconds": round(time.time() - t0, 1)}


def stream_yaml(w: Writer) -> dict:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{YAML_DATASET}/data/train-*.parquet"))
    used = 0
    for path in files:
        if w.full("yaml"):
            break
        for t in _iter_row_groups(fs, path, ["content", "max_stars_repo_licenses"]):
            for code, lics in zip(t["content"].to_pylist(), t["max_stars_repo_licenses"].to_pylist()):
                if lics and all(l in SPDX_PERMISSIVE for l in lics):
                    w.add("yaml", code, YAML_DATASET)
            if w.full("yaml"):
                break
        used += 1
        log(w, f"yaml shard {used}/{len(files)}")
    return {"shards_read": used, "shards_total": len(files)}


# ---------------------------------------------------------------- local fallback

def _npm_packages(root: Path):
    """Yield (pkg_dir, name, license) for every package.json under root."""
    for dp, dn, fn in os.walk(root):
        if "package.json" in fn:
            try:
                j = json.loads((Path(dp) / "package.json").read_text(errors="replace"))
            except Exception:
                j = None
            if isinstance(j, dict) and j.get("name"):
                lic = j.get("license")
                if isinstance(lic, dict):
                    lic = lic.get("type")
                if not lic and isinstance(j.get("licenses"), list) and j["licenses"]:
                    l0 = j["licenses"][0]
                    lic = l0.get("type") if isinstance(l0, dict) else l0
                yield Path(dp), f"{j['name']}@{j.get('version', '?')}", lic if isinstance(lic, str) else None
        dn[:] = [d for d in dn if d not in (".git",)]


def _cargo_crates(root: Path):
    for reg in sorted(root.glob("*")):
        for crate in sorted(reg.glob("*")):
            ct = crate / "Cargo.toml"
            if not ct.is_file():
                continue
            m = re.search(r'^\s*license\s*=\s*"([^"]+)"', ct.read_text(errors="replace"), re.M)
            yield crate, crate.name, m.group(1) if m else None


def _py_dists(site: Path):
    for d in imeta.distributions(path=[str(site)]):
        try:
            name = d.metadata["Name"]
        except Exception:
            continue
        lic = d.metadata.get("License-Expression") or d.metadata.get("License") or ""
        classifiers = [c for c in (d.metadata.get_all("Classifier") or []) if c.startswith("License ::")]
        if not spdx_permissive(lic) or len(lic) > 80:
            lic2 = None
            for c in classifiers:
                tail = c.split("::")[-1].strip()
                if "MIT" in tail:
                    lic2 = "MIT"
                elif "Apache" in tail:
                    lic2 = "Apache-2.0"
                elif "BSD" in tail:
                    lic2 = "BSD-3-Clause"
                else:
                    lic2 = None
                    break
            lic = lic2 or lic[:80]
        files = {str(Path(f).parts[0]) for f in (d.files or []) if f.parts and not f.parts[0].endswith((".dist-info", ".egg-info", ".."))}
        for top in files:
            p = site / top
            if p.is_dir():
                yield p, f"{name}=={d.version}", lic


def _walk_files(pkg_dir: Path, exts: set[str], nested_stop: str | None):
    for dp, dn, fn in os.walk(pkg_dir):
        if nested_stop:
            dn[:] = [d for d in dn if d != nested_stop and not d.startswith(".")]
        else:
            dn[:] = [d for d in dn if not d.startswith(".")]
        # nested packages declare their own license and are visited separately
        dn[:] = [d for d in dn if not (Path(dp, d, "package.json").is_file() or Path(dp, d, "Cargo.toml").is_file())]
        for f in sorted(fn):
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                yield Path(dp) / f, LOCAL_EXT[ext]


def stream_local(w: Writer, wanted: set[str]) -> dict:
    exts = {e for e, l in LOCAL_EXT.items() if l in wanted}
    packages: list[dict] = []
    roots = []
    nm = Path("/opt/homebrew/lib/node_modules")
    if nm.is_dir():
        roots.append(("npm", nm, _npm_packages(nm), "node_modules"))
    cargo = Path.home() / ".cargo" / "registry" / "src"
    if cargo.is_dir():
        roots.append(("cargo", cargo, _cargo_crates(cargo), None))
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site.is_dir():
        roots.append(("pypi", site, _py_dists(site), "__pycache__"))
    for eco, root, pkgs, stop in roots:
        for pkg_dir, name, lic in pkgs:
            if all(w.full(l) for l in wanted):
                break
            ok = spdx_permissive(lic)
            rec = {"ecosystem": eco, "package": name, "license": lic, "accepted": ok, "files": 0, "bytes": 0}
            if ok:
                for path, lang in _walk_files(pkg_dir, exts, stop):
                    if w.full(lang):
                        continue
                    try:
                        if path.stat().st_size > MAX_FILE_BYTES:
                            continue
                        text = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    before = w.total
                    if w.add(lang, text, f"local:{eco}"):
                        rec["files"] += 1
                        rec["bytes"] += w.total - before
            if ok and rec["files"] or not ok:
                packages.append(rec)
        log(w, f"local {eco} done")
    used = [p for p in packages if p["files"]]
    return {"roots": [str(r[1]) for r in roots], "packages_used": used,
            "packages_rejected_license": sorted({f"{p['package']} ({p['license']})" for p in packages if not p["accepted"]})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap-mb", type=float, default=600)
    ap.add_argument("--max-shards", type=int, default=60, help="github-code-clean shards to stream at most")
    ap.add_argument("--gh-seconds", type=float, default=3000,
                    help="time budget for github-code-clean streaming; shortfall is topped up locally where possible")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in list(args.out.glob("code-*.txt")) + [args.out / "heldout.txt"]:
        old.unlink(missing_ok=True)
    cap = int(args.cap_mb * 1e6)
    w = Writer(args.out, cap)
    t0 = time.time()
    gh = stream_github(w, args.max_shards, args.gh_seconds)
    short = {l for l in ("json", "toml", "rust") if not w.full(l)}
    local = stream_local(w, short) if short else {"packages_used": []}
    # The config group (yaml+json+toml) keeps its 1/8 share: YAML absorbs JSON/TOML shortfall.
    for l in ("json", "toml"):
        w.quota["yaml"] += max(0, w.quota[l] - w.used[l])
        w.quota[l] = w.used[l]
    ym = stream_yaml(w)
    w.close()

    def entry(p: Path) -> dict:
        return {"path": p.name, "bytes": p.stat().st_size, "sha256": _sha256(p)}

    shard_entries = [entry(p) for p in w.shards]
    manifest = {
        "name": "sys1 tokenizer code corpus",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "format": "one source file per line: '<lang=TAG> ' + escaped text; escapes: \\\\ -> backslash, "
                  "\\n -> LF, \\r -> CR (see unescape()); CRLF normalized to LF",
        "cap_bytes": cap,
        "total_bytes": w.total,
        "quota_bytes": w.quota,
        "datasets": [
            {"id": GH_DATASET, "revision": _revision(GH_DATASET), "gated": False, "terms": GH_TERMS,
             "license_filter": {"column": "license", "keep": sorted(GH_PERMISSIVE)},
             "languages": GH_LANG, **gh},
            {"id": YAML_DATASET, "revision": _revision(YAML_DATASET), "gated": False, "terms": THE_STACK_TERMS,
             "license_filter": {"column": "max_stars_repo_licenses", "keep_if_all_in": sorted(SPDX_PERMISSIVE)},
             **ym},
        ],
        "local_fallback": {
            "reason": "github-code-clean has no JSON/TOML/YAML; JSON and TOML (and any shortfall) come from "
                      "permissively licensed packages already installed on this machine",
            "license_rule": "package's declared license (package.json / Cargo.toml / Python metadata) must have an "
                            f"OR-alternative entirely within {sorted(SPDX_PERMISSIVE)}",
            **local,
        },
        "filters": {"min_chars": MIN_CHARS, "max_file_bytes": MAX_FILE_BYTES, "dedup": "exact blake2b of content",
                    "drop": "files containing NUL"},
        "languages": dict(sorted(w.stats.items())),
        "train": {"files": w.train_n, "bytes": sum(e["bytes"] for e in shard_entries), "shards": shard_entries},
        "heldout": {**entry(args.out / "heldout.txt"), "files": w.held_n, "fraction": HELDOUT_FRACTION,
                    "selection": "blake2b(content)[:4] mod 10000 < 500"},
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {k: {"files": v["files"], "MB": round(v["bytes"] / 1e6, 1)} for k, v in sorted(w.stats.items())}
    print(json.dumps({"total_MB": round(w.total / 1e6, 1), "train_files": w.train_n, "heldout_files": w.held_n,
                      "languages": summary}, indent=1))


if __name__ == "__main__":
    main()
