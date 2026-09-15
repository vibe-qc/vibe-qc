#!/usr/bin/env python3
"""Reject staged Markdown under docs/ that contains em (U+2014) or en
(U+2013) dashes in PROSE.

The public style rule in CONTRIBUTING.md ("Documentation prose") bans
em and en dashes from documentation prose. This hook enforces it at
commit time so parallel contributors cannot silently reintroduce them.

Exempt from the check (dashes there are real content, not prose style):
  * fenced code blocks (``` / ~~~), which carry program output and code,
  * inline-code spans (`like this`),
  * inline math ($...$), where '-' is a minus sign anyway.
  * provenance-marked QVF snapshots under docs/qvf/_vendored/, whose
    upstream bytes must be preserved when re-vendoring.

The check looks at the full staged content of each touched docs/*.md
file, so a file that still carries a legacy dash must be scrubbed before
it can be committed again. Bypass a reviewed exception with
`git commit --no-verify` and explain it in the commit message.
"""
import re
import subprocess
import sys

_INLINE_CODE = re.compile(r'(`+)(.+?)\1')
_FENCE = re.compile(r'^[ \t]*(```|~~~)')
_INLINE_MATH = re.compile(r'\$[^$]*\$')
_DASHES = ('—', '–')  # em dash, en dash
_QVF_PROVENANCE = re.compile(
    r'\A<!-- VENDORED from the qvf repository, \S+ at \S+ \([0-9a-f]{12,40}\)\.\n'
)


def prose_dash_hits(content):
    inside_code = False
    hits = []
    for lineno, line in enumerate(content.split('\n'), 1):
        if _FENCE.match(line):
            inside_code = not inside_code
            continue
        if inside_code:
            continue
        stripped = _INLINE_CODE.sub('', line)
        stripped = _INLINE_MATH.sub('', stripped)
        if any(d in stripped for d in _DASHES):
            hits.append((lineno, line.strip()[:88]))
    return hits


def main():
    staged = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'],
        capture_output=True, text=True, check=True).stdout.split()
    md = [f for f in staged if f.startswith('docs/') and f.endswith('.md')]

    bad = []
    for f in md:
        content = subprocess.run(
            ['git', 'show', f':{f}'],
            capture_output=True, text=True).stdout
        if f.startswith('docs/qvf/_vendored/') and _QVF_PROVENANCE.match(content):
            continue
        for lineno, text in prose_dash_hits(content):
            bad.append(f'  {f}:{lineno}: {text}')

    if bad:
        sys.stderr.write(
            'ERROR: staged docs Markdown contains em or en dashes in prose '
            '(the project no-dash style rule):\n\n'
            + '\n'.join(bad)
            + '\n\nReplace them with commas, colons, semicolons, or hyphens. '
            'Code fences, inline code, and math are exempt. If a use is '
            'intentional and reviewed, bypass with:\n  git commit --no-verify\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
