#!/usr/bin/env python3
"""CI guardrail: fail when a class name is defined in two or more files under ``src/``.

Why src-only? Test scaffolding legitimately reuses class names across modules — per-module
pytest test-cases (``TestModel``, ``TestIntegration``) and mocks/fakes (``MockAgent``,
``MockLLM``) are independent by design and never clash at runtime. Defensive import-fallback
stubs (e.g. ``PackageNotFoundError``) also reuse a name across src and tests. None of these
is a real conflict. This check therefore scans ONLY ``src/`` and fails only on a genuine
production-namespace collision.

As of the 1.1 cleanup that count is 0 (the agent-output Pydantic/TypedDict collision was
resolved: ``RefinerOutput`` etc. are Pydantic models, ``RefinerState`` etc. are the TypedDict
state schemas). So this is a **ratchet against regressions**, not a cleanup tool.

Self-contained: standard library only, no dependency on the (untracked) ``.claude/`` registry
tooling, so it runs anywhere — CI, pre-commit, or ``make check-class-conflicts``.
"""

from __future__ import annotations

import ast
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))


def _scan(root: str) -> dict[str, set[str]]:
    """Map each class name to the set of files under ``root`` that define it."""
    found: dict[str, set[str]] = defaultdict(set)
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found[node.name].add(os.path.relpath(path))
    return found


def main() -> int:
    classes = _scan(_SRC)
    total = sum(len(v) for v in classes.values())
    collisions = {
        name: sorted(files) for name, files in classes.items() if len(files) > 1
    }

    if collisions:
        print(
            f"❌ {len(collisions)} src-vs-src class-name collision(s) (of {total} classes):"
        )
        for name, files in sorted(collisions.items()):
            print(f"   {name}:")
            for f in files:
                print(f"       {f}")
        print(
            "\nEach production class name must be unique. Rename one side "
            "(see .claude/CLASS_RENAMING_STRATEGY.md), e.g. a domain prefix or a "
            "context suffix."
        )
        return 1

    print(f"✅ src namespace clean: {total} classes, all names unique.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
