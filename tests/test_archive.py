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


@pytest.mark.parametrize("bad_rel", ["../escape.txt", "results/../../escape.txt", "/tmp/escape-abs.txt"])
def test_restore_rejects_escaping_paths_before_writing(tmp_path, bad_rel):
    root = tmp_path / "repo"
    make_repo(root)
    run = snapshot("evil", root=root)
    mpath = root / "archive" / "runs" / f"{run}.json"
    m = json.loads(mpath.read_text())
    m["files"][bad_rel] = m["files"]["config.yaml"]
    mpath.write_text(json.dumps(m))
    dest = tmp_path / "out"
    with pytest.raises(ValueError):
        restore(run, dest, root=root)
    assert not (tmp_path / "escape.txt").exists()
    assert not dest.exists()  # nothing written, not even the good entries


def test_restore_rejects_bad_sha_and_run_id(tmp_path):
    root = tmp_path / "repo"
    make_repo(root)
    run = snapshot("ok", root=root)
    mpath = root / "archive" / "runs" / f"{run}.json"
    m = json.loads(mpath.read_text())
    m["files"]["config.yaml"]["sha256"] = "../../etc/passwd"
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        restore(run, tmp_path / "out", root=root)
    for bad in ["../" + run, "20261119T231709Z-x/../../y", "not-a-run"]:
        with pytest.raises(ValueError):
            restore(bad, tmp_path / "out2", root=root)
