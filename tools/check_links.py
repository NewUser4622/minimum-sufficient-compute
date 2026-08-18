#!/usr/bin/env python3
"""
check_links.py -- every document reference in the repo must resolve.

The docs cross-reference each other about 120 times ("see `09_LAB_NOTEBOOK.md`
§4"). Those references are load-bearing: `20_IN100_PORT_PLAN.md` opens by
telling the reader to read four other files in order, and the reading order is
the onboarding path for anyone picking this project up.

Nothing checked them. Before the docs were reorganised into `docs/cifar100/`
and `docs/imagenet100/`, one reference was already dead -- `24_IN100_RESULTS.md`,
a file that has never existed (the file is `24_IN100_STATUS.md`). A moved file
would have silently broken all of them at once.

Checks two things:
  * bare `NN_NAME.md` mentions in prose resolve to a real file somewhere;
  * markdown links `[text](path)` resolve relative to the containing file.

Usage
    python tools/check_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BARE = re.compile(r"`?\b([0-9]{2}_[A-Z0-9_]+\.md)`?")
LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

SKIP_DIRS = {".git", "node_modules", "__pycache__", "scratch",
             "msc_data", "msc_results", "notebooks", "notebooks_in100"}


def main() -> int:
    md = [p for p in ROOT.rglob("*.md")
          if not any(part in SKIP_DIRS for part in p.parts)]
    index = {p.name: p for p in md}
    problems = []

    for p in md:
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = p.relative_to(ROOT)

        for name in set(BARE.findall(text)):
            if name not in index:
                near = [n for n in index if n[:3] == name[:3]]
                problems.append(
                    f"{rel}: references '{name}', which does not exist"
                    + (f" -- did you mean {near[0]}?" if near else ""))

        for label, target in LINK.findall(text):
            t = target.split("#")[0].strip()
            if not t or t.startswith(("http://", "https://", "mailto:")):
                continue
            if not (p.parent / t).resolve().exists():
                problems.append(f"{rel}: link [{label}]({target}) does not resolve")

    for msg in sorted(set(problems)):
        print(f"  [FAIL] {msg}")
    if problems:
        print(f"\n  {len(set(problems))} broken reference(s).")
        return 1
    print(f"  all document references resolve ({len(md)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
