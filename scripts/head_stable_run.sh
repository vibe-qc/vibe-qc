#!/usr/bin/env bash
# scripts/head_stable_run.sh — run a build+measure cycle and refuse to
# report its result if HEAD moved underneath it.
#
# Usage:
#   scripts/head_stable_run.sh [--repo DIR] [--label NAME] -- CMD [ARG...]
#
#   --repo DIR    repository to guard (default: the repo containing this
#                 script). CMD runs with DIR as its working directory.
#   --label NAME  tag for the status lines, so interleaved logs from
#                 several legs stay attributable (default: the basename
#                 of CMD).
#
# Exit codes:
#   98   HEAD moved while CMD ran. The measurement is VOID and CMD's own
#        exit status is deliberately discarded, so a green-looking run
#        cannot be mistaken for a valid one.
#   99   usage or setup error, including a repository that is mid-rebase,
#        mid-merge, mid-cherry-pick or mid-bisect (see below).
#   *    otherwise CMD's own exit status, with HEAD verified stable
#        across the whole run.
#
# Why this exists
# ---------------
# The editable install has scikit-build auto-rebuild OFF and pins
# `vibeqc` to whichever checkout was installed, so a build and the
# measurement that follows it are meaningful only as a pair: the
# compiled core and the Python sources under `python/vibeqc/` have to
# come from the same commit. Anything that moves HEAD between the two --
# a concurrent `git checkout`, a `pull --rebase`, a second agent in a
# shared clone (CLAUDE.md paragraph 3) -- silently breaks that pairing.
#
# The loud failure is an ImportError for a symbol that moved between the
# two commits. The quiet failure is worse, and
# `tests/conftest.py::pytest_sessionstart` spells it out: a stale core
# "can silently produce WRONG NUMBERS, so the symptom looks like a
# physics regression rather than a build problem". That hook runs only
# under pytest. Bisect legs, convergence probes and benchmark drivers
# are bare `python` invocations and get no warning at all -- this
# wrapper is that warning, for anything with a command line.
#
# The consumers of these numbers are exactly the tasks that cannot
# tolerate a wrong one: `git bisect` verdicts and cross-code parity
# checks. A voided leg costs a rebuild; a leg that completes on a
# mismatched pair sends a bisect after a regression that never existed.
#
# 2026-08-27, issue #129 is the incident this generalises: a leg was
# checked out in a shared clone, and a concurrent agent ran `checkout
# main` + `pull --rebase` during its ~25-minute native rebuild.
#
# Refusing an in-progress git operation (LEARNINGS.md L121)
# ---------------------------------------------------------
# Mid-rebase, mid-merge, mid-cherry-pick and mid-bisect, `git rev-parse
# HEAD` does not fail -- it returns the last successfully applied commit,
# which during a rebase is *upstream's* tip and belongs to somebody else's
# work entirely. Both stamps would then read that same unrelated SHA,
# compare equal, and this wrapper would certify a measurement taken
# against a tree it never described. A guard that can be fooled is worse
# than no guard, because its "ok" line is evidence. So the run is refused
# up front rather than guarded: finish or abort the operation first.

set -uo pipefail

die() { printf '%s: %s\n' "${0##*/}" "$1" >&2; exit 99; }

repo=""
label=""
while [ $# -gt 0 ]; do
    case "$1" in
        --repo)  [ $# -ge 2 ] || die "--repo needs an argument"; repo="$2"; shift 2 ;;
        --label) [ $# -ge 2 ] || die "--label needs an argument"; label="$2"; shift 2 ;;
        --)      shift; break ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)       die "unknown option: $1 (did you forget the '--' before CMD?)" ;;
    esac
done

[ $# -gt 0 ] || die "no command given; usage: $0 [--repo DIR] [--label NAME] -- CMD [ARG...]"

if [ -z "$repo" ]; then
    repo="$(cd "$(dirname "$0")/.." && pwd)" || die "cannot resolve the default repo"
fi
[ -d "$repo" ] || die "not a directory: $repo"
cd "$repo" || die "cannot cd into $repo"
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository: $repo"

[ -n "$label" ] || label="${1##*/}"

# An in-progress operation makes HEAD unreliable rather than unreadable --
# see the L121 note above. Detect via the git-dir so this is independent of
# porcelain wording and of the caller's locale.
gitdir="$(git rev-parse --git-dir 2>/dev/null)" || die "cannot resolve the git dir"
for marker in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG; do
    if [ -e "$gitdir/$marker" ]; then
        die "repository is mid-operation ($marker present in $gitdir).
HEAD is not a reliable stamp until it finishes -- it reads back as the last
applied commit, which may belong to another lane entirely (LEARNINGS.md L121).
Finish or abort the operation, then re-run the whole build-and-measure cycle."
    fi
done

head_now() { git rev-parse HEAD 2>/dev/null; }

before="$(head_now)" || die "cannot read HEAD in $repo"
[ -n "$before" ] || die "cannot read HEAD in $repo"

printf 'HEAD_STABLE %s: start head=%s repo=%s\n' "$label" "$before" "$repo"

"$@"
status=$?

after="$(head_now)"
if [ -z "$after" ]; then
    printf 'HEAD_STABLE %s: VOID -- HEAD unreadable after the run (was %s)\n' \
        "$label" "$before" >&2
    exit 98
fi

if [ "$after" != "$before" ]; then
    cat >&2 <<EOF
HEAD_STABLE $label: VOID -- HEAD moved while the command ran
    repo:     $repo
    at start: $before
    at end:   $after
    command:  $*

Something moved HEAD underneath this run: a concurrent checkout or
pull, or another agent working in this clone (CLAUDE.md paragraph 3 --
one clone per chat). The compiled core and the Python sources are no
longer guaranteed to come from the same commit, so this measurement
proves nothing and its numbers must not be recorded.

The command's own exit status ($status) is discarded on purpose.

Re-run the whole build-and-measure cycle in a clone no other chat is
working in.
EOF
    exit 98
fi

printf 'HEAD_STABLE %s: ok head=%s exit=%d\n' "$label" "$after" "$status"
exit "$status"
