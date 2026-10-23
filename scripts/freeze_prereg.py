#!/usr/bin/env python3
"""
Freeze the revision's pre-registration before any new run is read.

Writes <prereg>.lock.json beside the pre-registration with the SHA-256 of its
canonical JSON (sorted keys, no whitespace), a UTC timestamp and, if the repo
is a git checkout, the current HEAD. Then tells you exactly what to commit.

The commit is what makes "pre-specified" checkable by someone who was not
there, the same way commit 2bcfd96 did for the original analysis. The report
script refuses to label any result as pre-registered unless the lock exists and
the pre-registration still hashes to it.

    python scripts/freeze_prereg.py                     # freeze
    python scripts/freeze_prereg.py --check             # verify, exit 1 on mismatch
    python scripts/freeze_prereg.py --amend "reason"    # re-freeze, logging why

An amendment does not erase the earlier lock: it appends the old hash and the
reason to the lock's history, so the paper can (and must) report it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "prereg" / "two_tail_prereg.json"


def canonical_sha256(path):
    obj = json.load(open(path))
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def lock_path(prereg):
    p = Path(prereg)
    return p.with_name(p.stem + ".lock.json")


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def verify(prereg):
    """(ok, message). Used by two_tail_report.py."""
    lp = lock_path(prereg)
    if not lp.exists():
        return False, f"no lock file ({lp.name}); the pre-registration is not frozen"
    lock = json.load(open(lp))
    now = canonical_sha256(prereg)
    if now != lock["sha256"]:
        return False, (f"{Path(prereg).name} has changed since it was frozen "
                       f"({now[:12]} != {lock['sha256'][:12]})")
    return True, f"frozen {lock['frozen_utc']}  sha256 {lock['sha256'][:12]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg", default=str(DEFAULT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--amend", metavar="REASON")
    ap.add_argument("--no-git", action="store_true", help="skip the git hint")
    a = ap.parse_args()

    if a.check:
        ok, msg = verify(a.prereg)
        print(("OK: " if ok else "NOT OK: ") + msg)
        return 0 if ok else 1

    lp = lock_path(a.prereg)
    sha = canonical_sha256(a.prereg)
    history = []
    if lp.exists():
        old = json.load(open(lp))
        if old["sha256"] == sha:
            print(f"already frozen at {old['frozen_utc']} ({sha[:12]}); nothing to do")
            return 0
        if not a.amend:
            raise SystemExit(
                f"{lp.name} exists with a different hash. The pre-registration "
                f"was edited after freezing. If that is deliberate, re-run with "
                f"--amend \"why\" and report the amendment in the paper.")
        history = old.get("history", []) + [{
            "sha256": old["sha256"], "frozen_utc": old["frozen_utc"],
            "git_head": old.get("git_head"), "superseded_because": a.amend}]

    lock = {"prereg": Path(a.prereg).name, "sha256": sha,
            "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_head": None if a.no_git else git_head(), "history": history}
    json.dump(lock, open(lp, "w"), indent=2)
    print(f"froze {Path(a.prereg).name}  sha256 {sha}")
    print(f"wrote {lp}")
    if not a.no_git:
        rel = Path(a.prereg).resolve().relative_to(REPO)
        print("\nNow commit it BEFORE running any new experiment:\n"
              f"  git add {rel} {rel.with_name(lp.name)}\n"
              f"  git commit -m \"Freeze two-tail pre-registration ({sha[:12]})\"\n"
              "and quote that commit hash in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
