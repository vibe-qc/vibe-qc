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
#       Shallow-clone <url> at <tag> into <dest>, then verify HEAD's
#       commit SHA matches <expected_sha>. Catches the case where a
#       maintainer force-moves a tag upstream (tags are mutable in git);
#       the build refuses to proceed against unverified source.
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
        cat >&2 <<EOF
verify_tarball_sha256: checksum mismatch for $file
    expected: $expected
    got:      $got

The upstream source has changed since this script's checksum was
pinned. This is either a moved/replaced tarball, a network-level
tamper, or a stale checksum after an upstream re-roll. Do NOT
override silently — investigate and update the script's pinned
SHA in the same commit as the version bump.
EOF
        return 1
    fi
}

git_clone_pinned() {
    local url="$1"
    local tag="$2"
    local expected="$3"
    local dest="$4"
    git clone --depth 1 --branch "$tag" "$url" "$dest"
    local got
    got="$(git -C "$dest" rev-parse HEAD)"
    if [ "$got" != "$expected" ]; then
        cat >&2 <<EOF
git_clone_pinned: commit-SHA mismatch for $url @ $tag
    expected: $expected
    got:      $got

Tags are mutable in git. The upstream tag $tag now points at a
different commit than when this script's SHA was pinned. Do NOT
override silently — verify what changed upstream and update the
pinned SHA in the same commit as the version bump.
EOF
        rm -rf "$dest"
        return 1
    fi
}
