#!/usr/bin/env python3
"""Fast privacy preflight for staged or committed text; no history rewriting.

Private site terms are supplied by the existing external-policy interface.
Exact reviewed public examples are bound to path, rule, match and source line.
Binary assets and full publication history still have a separate release gate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize

from private_terms import load_terms

REVIEW_FILE = ".githooks/public_content_review.json"
PUBLIC_EMAILS = {"project@vibe-qc.com", "mpei@vibe-qc.com"}
PATTERNS = {
    "email": re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\.[A-Za-z]{2,}(?![\w.-])"),
    "personal-home": re.compile(r"/(?:Users|home)/[A-Za-z][A-Za-z0-9_.-]*", re.I),
    "private-ipv4": re.compile(r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"),
    "internal-host": re.compile(r"\b[A-Za-z0-9][A-Za-z0-9.-]{0,251}\.(?:internal|local|lan)(?![A-Za-z0-9_.-])", re.I),
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def source_files(root, tree=None, staged=False):
    """Read Git objects, never unstaged worktree content or external symlink targets."""
    if staged:
        records = git(root, "ls-files", "--stage", "-z").split(b"\0")
        entries = []
        for record in records:
            if not record:
                continue
            meta, path = record.split(b"\t", 1)
            mode, oid, stage = meta.split()
            if stage != b"0":
                raise ValueError("unmerged index; resolve conflicts before privacy preflight")
            entries.append((mode, oid, path))
        changed = set(git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z").split(b"\0"))
        entries = [item for item in entries if item[2] in changed or item[2].decode() == REVIEW_FILE]
    else:
        records = git(root, "ls-tree", "-rz", "--full-tree", tree or "HEAD").split(b"\0")
        entries = []
        for record in records:
            if record:
                meta, path = record.split(b"\t", 1)
                mode, kind, oid = meta.split()
                if kind != b"blob":
                    raise ValueError("submodule needs independent publication review")
                entries.append((mode, oid, path))
    # Batch avoids a subprocess per file on the full-tree CI path.
    request = b"".join(oid + b"\n" for _, oid, _ in entries)
    output = b"" if not entries else subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"], input=request,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(output)
    for _, oid, path in entries:
        header = stream.readline().split()
        if len(header) != 3 or header[0] != oid or header[1] != b"blob":
            raise ValueError("cannot read source object")
        data = stream.read(int(header[2]))
        if len(data) != int(header[2]) or stream.read(1) != b"\n":
            raise ValueError("truncated source object")
        yield path.decode("utf-8"), data


def key(path, rule, match, line):
    payload = json.dumps([path, rule, match, line], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def python_operators(path, text):
    if not path.endswith(".py"):
        return set(), set()
    result, attributes = set(), set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.OP and token.string == "@":
                result.add(token.start)
            if token.type == tokenize.NAME and token.string in {"local", "internal", "lan"}:
                attributes.add(token.end)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # An incomplete file must not gain email exceptions from partial tokens.
        return set(), set()
    return result, attributes


def findings(path, text, terms=()):
    operators = attributes = None
    # Most scientific fixtures contain none of these signals. Filter in C regex
    # before allocating and checking millions of numeric lines in Python.
    marker = re.compile(r"@|/(?:Users|home)/|\.(?:internal|local|lan)\b|(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.|192\.168\.\d{1,3}\.|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.)", re.I)
    positions = {match.start() for match in marker.finditer(text)}
    folded_text = text.casefold() if terms else ""
    for term in terms:
        start = 0
        while (position := folded_text.find(term, start)) >= 0:
            # Casefold can change offsets; use original lines for private terms.
            if len(folded_text) != len(text):
                positions.update(match.start() for match in re.finditer(r"(?m)^", text))
                break
            positions.add(position)
            start = position + len(term)
    starts = sorted({text.rfind("\n", 0, position) + 1 for position in positions})
    number, previous = 1, 0
    for start in starts:
        number += text.count("\n", previous, start)
        previous = start
        end = text.find("\n", start)
        line = text[start:end if end >= 0 else len(text)].rstrip("\r")
        for rule, pattern in PATTERNS.items():
            if rule == "email" and "@" not in line:
                continue
            if rule == "personal-home" and not any(value in line.casefold() for value in ("/users/", "/home/")):
                continue
            if rule == "private-ipv4" and not any(value in line for value in ("10.", "172.", "192.168.")):
                continue
            if rule == "internal-host" and not any(value in line.casefold() for value in (".internal", ".local", ".lan")):
                continue
            for match in pattern.finditer(line):
                value = match.group()
                if rule == "email":
                    address = value.casefold()
                    domain = address.rsplit("@", 1)[1]
                    if address in PUBLIC_EMAILS or domain in {"example.com", "example.org", "example.net"} or domain.endswith((".example", ".invalid", ".test")):
                        continue
                    if operators is None:
                        operators, attributes = python_operators(path, text)
                    if (number, match.start() + value.index("@")) in operators:
                        continue
                if rule == "internal-host":
                    if attributes is None:
                        operators, attributes = python_operators(path, text)
                    if (number, match.end()) in attributes:
                        continue
                if rule == "personal-home" and value.rsplit("/", 1)[1] in {"USER", "Shared", "runner", "root", "user"}:
                    continue
                yield {"path": path, "line": number, "rule": rule, "key": key(path, rule, value, line)}
        folded = line.casefold()
        for term in terms:
            if term in folded:
                yield {"path": path, "line": number, "rule": "private-term", "key": key(path, "private-term", term, line)}


def scan(files, terms=()):
    files = dict(files)
    reviewed = json.loads(files.get(REVIEW_FILE, b'{"reviews": []}'))
    accepted = {row["key"] for row in reviewed["reviews"] if row.get("reason", "").strip()}
    pending, count = [], 0
    for path, data in files.items():
        # File names are public even for binary assets.
        pending.extend(item for item in findings(path, path, terms) if item["key"] not in accepted)
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # Legacy text gets checked too; genuine binary is reviewed at publication.
            text = data.decode("latin-1")
        count += 1
        pending.extend(item for item in findings(path, text, terms) if item["key"] not in accepted)
    return pending, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--tree", default=None, help="committed revision (default HEAD)")
    parser.add_argument("--json", action="store_true", help="redacted findings for exact review")
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
        terms = load_terms(root)
        if os.environ.get("VIBE_PRIVACY_REQUIRE_TERMS") == "1" and not terms:
            raise ValueError("required external privacy policy is missing")
        pending, count = scan(source_files(root, args.tree, args.staged), terms)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ERROR: privacy preflight could not complete ({type(error).__name__}); values and paths redacted.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"text_files": count, "findings": pending}, indent=2))
    elif pending:
        for item in pending:
            # Paths themselves can contain private identifiers; do not echo them.
            print(f"ERROR: {item['rule']} at line {item['line']}, review key {item['key']} [values and paths redacted]", file=sys.stderr)
        print("Inspect staged/source content locally. Fix private data in the original source; do not bypass the gate.", file=sys.stderr)
    else:
        print(f"Privacy preflight passed ({count} text files).")
    return 1 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
