"""v2.0 evaluation window. The start comes from the committer date of the commit tagged prereg-v2.0
(US Eastern calendar date, then the first NYSE trading day after it). The end and the download cut-off
come from data_v2/window.json, written by `./run.sh v2-all` once the stopping rule has fired."""
import json
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

from .nyse import add_trading_days, next_trading_day_after

ET = ZoneInfo("America/New_York")


def tag_commit_iso(tag, repo):
    """Committer date (ISO 8601 with offset) of the commit that `tag` points to. Refuses if the tag is missing."""
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{tag}^{{commit}}"], cwd=repo,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tag {tag} not found: v2.0 cannot run until the freeze commit is tagged {tag}")
    sha = r.stdout.strip()
    return subprocess.run(["git", "log", "-1", "--format=%cI", sha], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


def window_start(commit_iso):
    """First NYSE trading day strictly after the US Eastern calendar date of the commit."""
    return next_trading_day_after(datetime.fromisoformat(commit_iso).astimezone(ET).date())


def window_record(start, end, stop, tag, commit_iso, extra=None):
    """The frozen window: end T is the stopping day; sample_end is T + 5 trading days (last day needed by a 5-day target)."""
    return {"start": start.isoformat(), "end": end.isoformat(), "stop": stop, "tag": tag,
            "tag_commit_iso": commit_iso, "sample_end": add_trading_days(end, 5).isoformat(), **(extra or {})}


def resolve_window(cfg, data_dir, repo):
    """Fill cfg['windows']['primary'] and cfg['data']['sample_end'] for a v2.0 config."""
    tag = cfg["stopping"]["tag"]
    commit = tag_commit_iso(tag, repo)
    start = window_start(commit)
    w = cfg["windows"]["primary"]
    w["start"], w["end"] = start.isoformat(), None
    cfg["v2"] = {"tag": tag, "tag_commit_iso": commit, "window_start": start.isoformat(), "stop": None}
    f = data_dir / "window.json"
    if f.exists():
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["start"] != start.isoformat():
            raise SystemExit(f"{f} records start {rec['start']} but tag {tag} gives {start}: refusing to continue")
        w["end"], cfg["data"]["sample_end"] = rec["end"], rec["sample_end"]
        cfg["v2"]["stop"] = rec["stop"]
    return cfg
