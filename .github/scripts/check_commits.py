"""Check the PR title and commit subjects against the commit convention (CONTRIBUTING.md).

Header format: <type>(<scope>): <subject>, scope optional, subject at most 50
characters and not ending with a period. Merge and revert commits are skipped.

Usage (CI): PR_TITLE=... BASE=<sha> HEAD=<sha> python3 .github/scripts/check_commits.py
"""
import os
import re
import subprocess
import sys

TYPES = ("feat", "fix", "hotfix", "docs", "style", "refactor", "test", "chore")
HEADER = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9._/-]+\))?!?: (?P<subject>.+)$")
MAX_SUBJECT = 50


def problems(header):
    """List what is wrong with one commit header or PR title (empty if fine)."""
    m = HEADER.match(header)
    if m is None:
        return [f"형식은 <타입>(<범위>): <제목>, 타입은 {', '.join(TYPES)} 중 하나"]
    out = []
    subject = m.group("subject")
    if len(subject) > MAX_SUBJECT:
        out.append(f"제목이 {MAX_SUBJECT}자를 넘음 ({len(subject)}자)")
    if subject.endswith("."):
        out.append("제목 끝에 마침표")
    return out


def main():
    """Print a GitHub annotation per violation and exit 1 if there is any."""
    checks = [("PR 제목", os.environ["PR_TITLE"])]
    log = subprocess.run(
        ["git", "log", "--no-merges", "--format=%h %s", f"{os.environ['BASE']}..{os.environ['HEAD']}"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in log.splitlines():
        sha, header = line.split(" ", 1)
        if not header.startswith("Revert "):
            checks.append((f"커밋 {sha}", header))

    failed = False
    for where, header in checks:
        for p in problems(header):
            print(f"::error::{where} \"{header}\": {p}")
            failed = True
    print(f"{len(checks)}개 확인, {'실패' if failed else '통과'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
