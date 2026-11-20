"""Immutable snapshots of every pipeline input and output, one manifest per run.

Blobs go to archive/objects/<sha256[:2]>/<sha256> (gitignored; identical files are stored once).
Manifests go to archive/runs/<run_id>.json (small, committed). A manifest names every file by hash,
so any past run can be restored byte-for-byte after results/ has been overwritten.

Usage: ./run.sh archive [snapshot [label]] | list | restore <run_id> <dest_dir>
"""
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, package_versions, sha256_file

RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z(-[A-Za-z0-9_-]+)?")
SHA256 = re.compile(r"[0-9a-f]{64}")

TRACKED = ["config.yaml", "uv.lock", "pyproject.toml", "README.md", "DEVIATIONS.md", "RESULTS.md",
           "data", "results", "logs", "dashboard/index.html"]


def _tracked_files(root):
    out = []
    for item in TRACKED:
        p = root / item
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".") and "__pycache__" not in f.parts]
    return sorted(out)


def _git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def git_state(root):
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {"commit": head.strip() if head else None,
            "uncommitted": status.splitlines() if status is not None else None}


def snapshot(label="", root=ROOT):
    utc = datetime.now(timezone.utc)
    run_id = utc.strftime("%Y%m%dT%H%M%SZ") + (f"-{re.sub(r'[^A-Za-z0-9_-]', '-', label)}" if label else "")
    manifest_path = root / "archive" / "runs" / f"{run_id}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    objects = root / "archive" / "objects"
    files = {}
    for f in _tracked_files(root):
        digest = sha256_file(f)
        blob = objects / digest[:2] / digest
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_name(blob.name + ".tmp")
            shutil.copyfile(f, tmp)
            os.replace(tmp, blob)
        files[f.relative_to(root).as_posix()] = {"sha256": digest, "bytes": f.stat().st_size}
    rec = {"run_id": run_id, "utc": utc.isoformat(timespec="seconds"), "label": label,
           "python": sys.version.split()[0], "packages": package_versions(),
           "config_sha256": sha256_file(root / "config.yaml"), "git": git_state(root), "files": files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return run_id


def _check_run_id(run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"bad run_id: {run_id!r}")


def load_manifest(run_id, root=ROOT):
    _check_run_id(run_id)
    return json.loads((root / "archive" / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))


def list_runs(root=ROOT):
    lines = []
    for p in sorted((root / "archive" / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        commit = (r["git"]["commit"] or "no-git")[:7]
        dirty = " dirty" if r["git"]["uncommitted"] else ""
        lines.append(f"{r['run_id']}  {commit}{dirty}  files={len(r['files'])}  config={r['config_sha256'][:8]}")
    return lines


def restore(run_id, dest, root=ROOT):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; restore into a new directory")
    files = load_manifest(run_id, root)["files"]
    base = dest.resolve()
    plan = []  # validate every entry before writing anything
    for rel, e in files.items():
        if not SHA256.fullmatch(e["sha256"]):
            raise ValueError(f"bad sha256 for {rel!r}")
        out = (dest / rel).resolve()
        if out == base or not out.is_relative_to(base):
            raise ValueError(f"manifest path escapes restore dir: {rel!r}")
        blob = root / "archive" / "objects" / e["sha256"][:2] / e["sha256"]
        if sha256_file(blob) != e["sha256"]:
            raise ValueError(f"blob for {rel} is corrupt: {blob}")
        plan.append((blob, out))
    for blob, out in plan:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(blob, out)
    return len(plan)


def main(argv):
    cmd = argv[0] if argv else "snapshot"
    if cmd == "snapshot":
        print("archived run", snapshot(argv[1] if len(argv) > 1 else ""))
    elif cmd == "list":
        print("\n".join(list_runs()))
    elif cmd == "restore" and len(argv) == 3:
        print(f"restored {restore(argv[1], argv[2])} files to {argv[2]}")
    else:
        raise SystemExit("usage: archive [snapshot [label] | list | restore <run_id> <dest_dir>]")


if __name__ == "__main__":
    main(sys.argv[1:])
