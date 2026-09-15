# scripts/_verify_source.sh — supply-chain integrity helpers.
#
# Sourced (NOT executed) from each build_*.sh that fetches third-party
# source. Two helpers, both idempotent and side-effect-free on success:
#
#   verify_tarball_sha256 <file> <expected_sha256>
#       Compute the SHA-256 of <file> and compare it to <expected_sha256>.
#       Bail with a clear diff on mismatch. Uses `shasum -a 256` (BSD +
#       macOS) with a `sha256sum` (Linux/coreutils) fallback.
#
#   git_clone_pinned <url> <tag> <expected_sha> <dest>
#       Produce a shallow checkout of <url> at <tag> in <dest> whose HEAD
#       is <expected_sha>, or fail. Catches the case where a maintainer
#       force-moves a tag upstream (tags are mutable in git); the build
#       refuses to proceed against unverified source.
#
#       Where the source comes from (#241):
#         1. A local repository, when VIBEQC_GIT_REFERENCE_DIR names one
#            or more directories (colon-separated) to search. A candidate
#            is used only if it already holds <expected_sha>, so a local
#            source can never substitute a different commit; the HEAD
#            check below still runs on the result. Point it at another
#            vibe-qc checkout's third_party/ or at a directory of mirrors
#            named after the upstream repository (libint, libint.git,
#            libint/src, libint-src are all recognised).
#         2. Otherwise a network clone, at most VIBEQC_GIT_CLONE_ATTEMPTS
#            times (default 3), each attempt under a wall-clock deadline
#            of VIBEQC_GIT_CLONE_TIMEOUT seconds (default 1800) that ends
#            in SIGKILL, not just SIGTERM: a wedged clone has been seen
#            to ignore SIGTERM for 34 hours. Stalled HTTP transfers are
#            additionally cut by git's own low-speed limit (1 KiB/s over
#            60 s unless GIT_HTTP_LOW_SPEED_LIMIT / _TIME are set).
#
# Why this exists: vibe-qc vendors libint / libxc / spglib / fftw /
# libecpint / openblas as our entire native-ABI surface. Without
# checksum + commit pinning, a compromised upstream or a moved tag
# would silently change what our binaries link against. The .system
# manifest claims a specific vendored ABI; this helper makes that claim
# enforceable, not aspirational.
#
# Version + checksum bumps: when a build_*.sh updates a pinned VERSION,
# the corresponding SHA in that script MUST be updated in the same
# commit. The recipe for resolving the new value is in the comment
# block at the top of each script. Re-using a stale SHA after a
# version bump will fail the verification step loudly, which is the
# intended behaviour.
#
# This file is not part of any build_*.sh recipe hash (see
# _native_stamp.sh), so changes here do not trigger fleet rebuilds.

set -euo pipefail

# --- SHA-256 sum: shasum on macOS+BSD, sha256sum on coreutils Linux. -----
_sha256() {
    local file="$1"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    else
        echo "_verify_source.sh: need shasum or sha256sum on PATH" >&2
        return 1
    fi
}

verify_tarball_sha256() {
    local file="$1"
    local expected="$2"
    if [ ! -f "$file" ]; then
        echo "verify_tarball_sha256: file not found: $file" >&2
        return 1
    fi
    local got
    got="$(_sha256 "$file")"
    if [ "$got" != "$expected" ]; then
        cat >&2 <<MSG
verify_tarball_sha256: checksum mismatch for $file
    expected: $expected
    got:      $got

The upstream source has changed since this script's checksum was
pinned. This is either a moved/replaced tarball, a network-level
tamper, or a stale checksum after an upstream re-roll. Do NOT
override silently — investigate and update the script's pinned
SHA in the same commit as the version bump.
MSG
        return 1
    fi
}

# --- Bounded subprocesses -------------------------------------------------

# Send <signal> to <pid> and every descendant, children first.
_vqc_kill_tree() {
    local sig="$1" pid="$2" child
    if command -v pgrep >/dev/null 2>&1; then
        for child in $(pgrep -P "$pid" 2>/dev/null || true); do
            _vqc_kill_tree "$sig" "$child"
        done
    fi
    kill "-$sig" "$pid" 2>/dev/null || true
}

# _vqc_run_with_deadline <limit_s> <grace_s> <cmd...>
#
# Run <cmd...> under a wall-clock deadline: SIGTERM to its whole process
# tree after <limit_s>, SIGKILL <grace_s> later. Returns the command's exit
# status, or 124 when the deadline fired. GNU timeout(1) is absent on macOS
# and `timeout N` without --kill-after stops at SIGTERM, which is how a
# wedged clone once outlived its guard by 34 hours (#241). Ctrl-C and
# SIGTERM to the calling script still reach the command: the script's
# own INT/TERM dispositions are restored and the signal re-delivered.
# One command at a time; the relay state below is not re-entrant.
_VQC_DEADLINE_PID=""
_VQC_DEADLINE_WATCHDOG=""
_VQC_DEADLINE_FIRED=""
_VQC_DEADLINE_OLD_INT=""
_VQC_DEADLINE_OLD_TERM=""

_vqc_run_with_deadline() {
    local limit="$1" grace="$2"
    shift 2
    _VQC_DEADLINE_FIRED="$(mktemp "${TMPDIR:-/tmp}/vibeqc-deadline.XXXXXX")"

    "$@" &
    _VQC_DEADLINE_PID=$!
    (
        trap 'kill "$sleeper" 2>/dev/null; exit 0' TERM
        sleep "$limit" &
        sleeper=$!
        wait "$sleeper" 2>/dev/null || true
        echo "_vqc_run_with_deadline: no completion after ${limit}s; sending SIGTERM, then SIGKILL after ${grace}s: $*" >&2
        echo fired > "$_VQC_DEADLINE_FIRED"
        _vqc_kill_tree TERM "$_VQC_DEADLINE_PID"
        sleep "$grace"
        _vqc_kill_tree KILL "$_VQC_DEADLINE_PID"
    ) &
    _VQC_DEADLINE_WATCHDOG=$!

    _VQC_DEADLINE_OLD_INT="$(trap -p INT)"
    _VQC_DEADLINE_OLD_TERM="$(trap -p TERM)"
    trap '_vqc_deadline_relay INT' INT
    trap '_vqc_deadline_relay TERM' TERM

    local rc=0
    wait "$_VQC_DEADLINE_PID" || rc=$?
    kill -TERM "$_VQC_DEADLINE_WATCHDOG" 2>/dev/null || true
    wait "$_VQC_DEADLINE_WATCHDOG" 2>/dev/null || true

    _vqc_deadline_restore_traps
    if [ -s "$_VQC_DEADLINE_FIRED" ]; then
        rc=124
    fi
    rm -f "$_VQC_DEADLINE_FIRED"
    _VQC_DEADLINE_PID=""
    _VQC_DEADLINE_WATCHDOG=""
    _VQC_DEADLINE_FIRED=""
    return "$rc"
}

_vqc_deadline_restore_traps() {
    eval "${_VQC_DEADLINE_OLD_INT:-trap - INT}"
    eval "${_VQC_DEADLINE_OLD_TERM:-trap - TERM}"
}

# Signal relay for _vqc_run_with_deadline: on INT/TERM to the calling
# script, take the command and the watchdog down, restore the script's
# previous dispositions, and re-deliver the signal so the script ends the
# way it would have without the deadline wrapper.
_vqc_deadline_relay() {
    local sig="$1"
    [ -n "$_VQC_DEADLINE_PID" ] && _vqc_kill_tree TERM "$_VQC_DEADLINE_PID"
    [ -n "$_VQC_DEADLINE_WATCHDOG" ] && _vqc_kill_tree KILL "$_VQC_DEADLINE_WATCHDOG"
    [ -n "$_VQC_DEADLINE_FIRED" ] && rm -f "$_VQC_DEADLINE_FIRED"
    _vqc_deadline_restore_traps
    kill "-$sig" "$$" 2>/dev/null || true
    case "$sig" in
        INT) exit 130 ;;
        *) exit 143 ;;
    esac
}

# --- Pinned git checkouts -------------------------------------------------

# True when <dir> is the top level of a git work tree, or a bare repository.
# A directory merely *inside* a repository (third_party/ of a checkout, say)
# is not a source: `git -C` would silently walk up to the enclosing repo.
_vqc_is_repo_root() {
    local dir="$1" top
    if [ "$(git -C "$dir" rev-parse --is-bare-repository 2>/dev/null)" = "true" ]; then
        return 0
    fi
    top="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)" || return 1
    [ "$top" = "$(cd "$dir" && pwd -P)" ]
}

# _vqc_find_local_source <url> <expected_sha>
#
# Print the first repository under VIBEQC_GIT_REFERENCE_DIR that holds
# <expected_sha>, or return 1. Candidates are derived from the URL's
# repository name (`libint` for .../evaleev/libint.git), in the layouts
# vibe-qc itself produces (third_party/<dep>/src, libecpint's <dep>-src)
# and the usual mirror spellings.
_vqc_find_local_source() {
    local url="$1" expected="$2"
    local dirs="${VIBEQC_GIT_REFERENCE_DIR:-}"
    [ -n "$dirs" ] || return 1
    local name lower
    name="${url%/}"
    name="${name##*/}"
    name="${name%.git}"
    lower="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
    local dir cand candidates
    local old_ifs="$IFS"
    IFS=':'
    # shellcheck disable=SC2206
    local dir_list=($dirs)
    IFS="$old_ifs"
    for dir in "${dir_list[@]}"; do
        [ -n "$dir" ] || continue
        candidates=(
            "$dir/$name" "$dir/$name.git" "$dir/$name/src" "$dir/$name-src" "$dir/$name/$name-src"
        )
        if [ "$lower" != "$name" ]; then
            candidates+=(
                "$dir/$lower" "$dir/$lower.git" "$dir/$lower/src" "$dir/$lower-src" "$dir/$lower/$lower-src"
            )
        fi
        candidates+=("$dir")
        for cand in "${candidates[@]}"; do
            [ -d "$cand" ] || continue
            _vqc_is_repo_root "$cand" || continue
            if git -C "$cand" cat-file -e "$expected^{commit}" 2>/dev/null; then
                printf '%s\n' "$cand"
                return 0
            fi
            echo "git_clone_pinned: local source $cand does not contain $expected; ignoring it" >&2
        done
    done
    return 1
}

# _vqc_clone_from_local <source_repo> <url> <tag> <expected_sha> <dest>
#
# Shallow-fetch <expected_sha> from a local repository into a fresh <dest>
# and check it out detached. The upstream <url> is recorded as `origin` and
# the tag carried over when the source has it, so the result reads like an
# ordinary `git clone --depth 1 --branch <tag>`.
_vqc_clone_from_local() {
    local source="$1" url="$2" tag="$3" expected="$4" dest="$5"
    rm -rf "$dest"
    git init -q "$dest" || return 1
    if ! git -C "$dest" fetch -q --depth 1 "$source" "$expected" 2>/dev/null; then
        # A server that refuses a bare object id in `want` still serves refs.
        git -C "$dest" fetch -q --depth 1 "$source" "refs/tags/$tag" || return 1
    fi
    git -c advice.detachedHead=false -C "$dest" checkout -q --detach FETCH_HEAD || return 1
    git -C "$dest" remote add origin "$url" 2>/dev/null || true
    git -C "$dest" fetch -q --depth 1 "$source" "refs/tags/$tag:refs/tags/$tag" 2>/dev/null || true
}

# _vqc_verify_clone_head <url> <tag> <expected_sha> <dest>
_vqc_verify_clone_head() {
    local url="$1" tag="$2" expected="$3" dest="$4"
    local got
    got="$(git -C "$dest" rev-parse HEAD)"
    if [ "$got" != "$expected" ]; then
        cat >&2 <<MSG
git_clone_pinned: commit-SHA mismatch for $url @ $tag
    expected: $expected
    got:      $got

Tags are mutable in git. The upstream tag $tag now points at a
different commit than when this script's SHA was pinned. Do NOT
override silently — verify what changed upstream and update the
pinned SHA in the same commit as the version bump.
MSG
        rm -rf "$dest"
        return 1
    fi
}

git_clone_pinned() {
    local url="$1"
    local tag="$2"
    local expected="$3"
    local dest="$4"
    local attempts="${VIBEQC_GIT_CLONE_ATTEMPTS:-3}"
    local limit="${VIBEQC_GIT_CLONE_TIMEOUT:-1800}"
    local grace="${VIBEQC_GIT_CLONE_KILL_GRACE:-10}"
    local delay="${VIBEQC_GIT_CLONE_RETRY_DELAY:-10}"

    # 1. A local source that already holds the pinned commit needs no network.
    local source
    if source="$(_vqc_find_local_source "$url" "$expected")"; then
        echo "git_clone_pinned: taking $url @ $tag from local source $source"
        if _vqc_clone_from_local "$source" "$url" "$tag" "$expected" "$dest" \
           && _vqc_verify_clone_head "$url" "$tag" "$expected" "$dest"; then
            return 0
        fi
        echo "git_clone_pinned: local source $source did not yield $expected; falling back to $url" >&2
        rm -rf "$dest"
    fi

    # 2. Upstream, bounded in attempts and in wall time per attempt.
    local attempt=1 rc
    while :; do
        rm -rf "$dest"
        if _vqc_run_with_deadline "$limit" "$grace" \
             env GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-1024}" \
                 GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-60}" \
                 git -c advice.detachedHead=false clone --depth 1 --branch "$tag" "$url" "$dest"; then
            break
        else
            rc=$?
        fi
        if [ "$attempt" -ge "$attempts" ]; then
            cat >&2 <<MSG
git_clone_pinned: giving up on $url @ $tag after $attempts attempt(s) (last exit $rc).

If this machine already holds a clone of that repository at commit
$expected, point VIBEQC_GIT_REFERENCE_DIR at the directory holding it
(another vibe-qc checkout's third_party/, for example) and rerun; the
commit pin is still verified. VIBEQC_GIT_CLONE_ATTEMPTS and
VIBEQC_GIT_CLONE_TIMEOUT (seconds per attempt) widen the retry budget.
MSG
            rm -rf "$dest"
            return 1
        fi
        echo "git_clone_pinned: attempt $attempt of $attempts for $url failed (exit $rc); retrying in ${delay}s" >&2
        sleep "$delay"
        attempt=$((attempt + 1))
    done
    _vqc_verify_clone_head "$url" "$tag" "$expected" "$dest"
}
