#!/usr/bin/env python3
"""Download the YuE2 models, printing JSON progress lines: {"bytes": done, "total": total, "rate_mbps": r}.

Arguments are Hub repos, optionally pinned as ``repo@revision``.  A repo whose config.json
names a parent (``base_model_name_or_path`` + ``base_model_revision``, as SheetSage2 does for
its MERT-v2 backbone) has that parent downloaded too, at the pinned revision; ``total`` grows
when such a parent is discovered, so readers must take it from every line.

Uses huggingface_hub's tqdm_class hook so byte counts come from the downloader itself.
"""
import json, os, sys, threading, time
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"      # the Xet engine reports no progress until a file completes; plain HTTPS reports per chunk
from huggingface_hub import HfApi, snapshot_download
import importlib
hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")   # the package attribute shadows the module

DEFAULT_REPOS = ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
state = {"bars": {}, "last": 0.0, "samples": [], "total": 0}
lock = threading.Lock()


def parse_repo(arg):
    """``repo`` or ``repo@revision`` -> (repo, revision or None); the last '@' splits."""
    repo, at, revision = arg.rpartition("@")
    return (repo, revision) if at and repo and revision else (arg, None)


def pinned_parent(snapshot):
    """The (parent, revision) a snapshot's config.json pins, or None.

    SheetSage2 ships LoRA adapters over m-a-p/MERT-v2-FullSong and its remote code
    resolves that parent at ``base_model_revision`` only, so the parent's ``main``
    (which moved on after SheetSage2 was published) is no use to it offline.
    """
    try:
        config = json.loads(Path(snapshot, "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    parent, revision = config.get("base_model_name_or_path"), config.get("base_model_revision")
    if not (isinstance(parent, str) and isinstance(revision, str) and parent and revision):
        return None
    if os.path.isdir(parent):           # a local directory, nothing to fetch
        return None
    return parent, revision


def report(force=False):
    now = time.time()
    if not force and now - state["last"] < 0.5:
        return
    state["last"] = now
    done = sum(state["bars"].values())
    state["samples"].append((now, done))
    state["samples"] = [s for s in state["samples"] if now - s[0] <= 15]
    rate = 0.0
    if len(state["samples"]) > 1 and now - state["samples"][0][0] >= 2:
        rate = (done - state["samples"][0][1]) / (now - state["samples"][0][0]) / 1e6
    print(json.dumps({"bytes": done, "total": state["total"], "rate_mbps": round(rate, 1)}), flush=True)


class Progress(hf_tqdm.tqdm):
    """Silent tqdm that accumulates byte progress across all files.

    huggingface_hub builds its per-file byte bars from its own ``utils.tqdm.tqdm``
    (``tqdm_class`` only reaches the outer file counter), so that class is replaced.
    """
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        self._bytes = kwargs.get("unit") == "B"
        self._count = kwargs.get("initial", 0) or 0
        if self._bytes:
            with lock:
                state["bars"][id(self)] = self._count

    def update(self, n=1):
        if self._bytes:
            self._count += n
            with lock:
                state["bars"][id(self)] = self._count
                report()
        return super().update(n)


def main(argv):
    hf_tqdm.tqdm = Progress
    api = HfApi()

    def size(repo, revision):
        return sum((f.size or 0) for f in api.model_info(repo, revision=revision, files_metadata=True).siblings)

    repos = [parse_repo(a) for a in (argv or DEFAULT_REPOS)]
    state["total"] = sum(size(*r) for r in repos)
    print(json.dumps({"bytes": 0, "total": state["total"], "rate_mbps": 0.0}), flush=True)
    i = 0
    while i < len(repos):                           # the list grows as pinned parents are found
        repo, revision = repos[i]
        i += 1
        snapshot = snapshot_download(repo, revision=revision, max_workers=16)
        parent = pinned_parent(snapshot)
        if parent and parent not in repos:
            repos.append(parent)
            with lock:
                state["total"] += size(*parent)
                report(force=True)
    with lock:
        report(force=True)
    print(json.dumps({"done": True, "total": state["total"]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
