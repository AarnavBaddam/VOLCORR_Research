import json

import pytest

from volcorr.archive import list_runs, restore, snapshot


def make_repo(root):
    (root / "results" / "stats").mkdir(parents=True)
    (root / "config.yaml").write_text("a: 1\n")
    (root / "results" / "x.csv").write_text("v\n1\n")
    (root / "results" / ".DS_Store").write_text("junk")


def blobs(root):
    return [p for p in (root / "archive" / "objects").rglob("*") if p.is_file()]


def test_snapshot_dedups_and_restores_old_version(tmp_path):
    root = tmp_path / "repo"
    make_repo(root)
    first = snapshot("first", root=root)
    assert len(blobs(root)) == 2  # config.yaml + x.csv

    (root / "results" / "x.csv").write_text("v\n2\n")  # the overwrite that used to lose the old run
    second = snapshot("second", root=root)
    assert len(blobs(root)) == 3  # config.yaml blob reused, not stored again

    manifest = json.loads((root / "archive" / "runs" / f"{first}.json").read_text())
    assert ".DS_Store" not in "".join(manifest["files"])
    assert manifest["config_sha256"] == manifest["files"]["config.yaml"]["sha256"]
    assert manifest["git"] == {"commit": None, "uncommitted": None}  # tmp dir is not a git repo

    dest = tmp_path / "restored"
    assert restore(first, dest, root=root) == 2
    assert (dest / "results" / "x.csv").read_text() == "v\n1\n"
    assert restore(second, tmp_path / "restored2", root=root) == 2
    assert (tmp_path / "restored2" / "results" / "x.csv").read_text() == "v\n2\n"

    assert [line.split()[0] for line in list_runs(root)] == sorted([first, second])


def test_restore_detects_corrupt_blob(tmp_path):
    root = tmp_path / "repo"
    make_repo(root)
    run = snapshot("dup", root=root)
    for b in blobs(root):
        b.write_text("tampered")
    with pytest.raises(ValueError):
        restore(run, tmp_path / "out", root=root)
