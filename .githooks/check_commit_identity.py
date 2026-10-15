#!/usr/bin/env python3
"""Refuse a commit whose author or committer email is not a public address.

AGENTS.md ("Keep this repository public-safe") allows exactly two public
addresses, mpei@vibe-qc.com and project@vibe-qc.com, and keeps private
addresses out of everything the repository records, commit metadata
included. A clone without a local identity falls back to the global git
identity, and every commit it makes then carries that address into the
public mirror. This hook catches that at commit time, before any bytes
are recorded.

Fix for a clone:

    git config --local user.email mpei@vibe-qc.com

Another contributor adds their own public address locally (repeatable):

    git config --local --add privacy.commitEmail you@example.org

Diagnostics never repeat the rejected value.
"""
import re
import subprocess
import sys

PUBLIC_EMAILS = frozenset({"mpei@vibe-qc.com", "project@vibe-qc.com"})
_EMAIL = re.compile(r"<([^>]*)>")


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout


def identity_email(var):
    """Email of ``git var GIT_AUTHOR_IDENT`` or ``GIT_COMMITTER_IDENT``."""
    match = _EMAIL.search(_git("var", var))
    return match.group(1).strip().lower() if match else ""


def allowed_emails():
    extra = {
        line.strip().lower()
        for line in _git("config", "--get-all", "privacy.commitEmail").splitlines()
        if line.strip()
    }
    return PUBLIC_EMAILS | extra


def main():
    allowed = allowed_emails()
    failed = False
    for var, role in (
        ("GIT_AUTHOR_IDENT", "author"),
        ("GIT_COMMITTER_IDENT", "committer"),
    ):
        email = identity_email(var)
        if email in allowed:
            continue
        failed = True
        print(
            f"commit identity check: the {role} email "
            f"([redacted, {len(email)} characters]) is not a public address.",
            file=sys.stderr,
        )
    if failed:
        print(
            "Set this clone's identity with: "
            "git config --local user.email mpei@vibe-qc.com",
            file=sys.stderr,
        )
        print(
            "Another public address: "
            "git config --local --add privacy.commitEmail you@example.org",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
