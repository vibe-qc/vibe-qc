# scripts/_native_stamp.sh — native-dep build provenance stamp.
#
# Sourced by setup_native_deps.sh (writer) and by install.sh / update.sh
# / doctor.sh (reader). Centralises the format so the producer and
# consumers can't drift.
#
# Why this exists
# ---------------
# Each per-dep `build_*.sh` short-circuits if `third_party/<dep>/install/`
# already exists. That's correct in the common case (idempotent rebuilds
# after a vanilla `git pull`), but silently wrong when:
#
#   * a vendored library version bumped between two checkouts (the libxc
#     7.0.0 → 7.0.1 case in updating.md § "Vendored library version
#     bumps");
#   * the build flags inside `build_libint.sh` changed but the version
#     pin didn't — the libint ERI3 footgun in updating.md
#     § "libint stale-install footgun".
#
# The stamp captures **what the install/ tree was built from**:
#
#   one line per dep:  <dep-name> <VERSION> <legacy-hash> v2:<recipe-hash>
#   (recipe-hash = sha256 of build_<dep>.sh's build-relevant content)
#
# The v2 hash pins the **build-relevant** content of `build_*.sh` (its
# configure flags + cmake arguments + source/version pins). Full-line
# comments, blank lines, and explicitly delimited lifecycle-only blocks
# are excluded (see _vibeqc_recipe_hash), so locking/orchestration changes
# cannot trigger an expensive native rebuild while any unmarked executable
# change remains rebuild-relevant by default. The legacy hash keeps older
# three-field stamps and readers compatible during migration. Readers compare
# the stamp against the currently-checked-out `build_*.sh`: update.sh rebuilds
# the drifted dep(s) automatically and targeted; `--rebuild-native-deps`
# forces a full rebuild of every dep; doctor.sh just reports.
#
# The stamp is local-only (third_party/ is gitignored), regenerated on
# every successful `setup_native_deps.sh` run.

VIBEQC_STAMP_PATH="third_party/.build-stamp"

# libint's max_am is chosen by an environment variable rather than by the text
# of build_libint.sh, so the recipe hash has to be told about it — see
# _vibeqc_recipe_hash below. Resolve this helper's own directory rather than
# assuming CWD: the rest of this file expects CWD = repo root, but doctor.sh
# and the tests source it from elsewhere.
# shellcheck source=_libint_max_am.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_libint_max_am.sh"

# List of dep-name : build-script pairs we stamp. setup_native_deps.sh
# orchestrates these in the same order; keep them in sync.
_vibeqc_stamp_pairs() {
    cat <<'EOF'
libint    scripts/build_libint.sh
libxc     scripts/build_libxc.sh
spglib    scripts/build_spglib.sh
fftw      scripts/build_fftw.sh
libecpint scripts/build_libecpint.sh
EOF
    # WITH_OPENBLAS=1 is opt-in. Once selected, its stamp line is also part of
    # the checkout's native configuration: keep checking it even if the whole
    # install tree was wiped so a normal update can restore it.
    if [ "${WITH_OPENBLAS:-0}" = "1" ] || \
       [ -d "third_party/openblas/install" ] || \
       { [ -f "$VIBEQC_STAMP_PATH" ] && \
         awk '$1 == "openblas" { found=1 } END { exit !found }' \
             "$VIBEQC_STAMP_PATH"; }; then
        echo "openblas  scripts/build_openblas.sh"
    fi
}

# Hash a string using the cross-platform tool already required by the original
# stamp format. Command substitution strips trailing newlines; adding exactly
# one here preserves the historical canonicalisation.
_vibeqc_sha256_text() {
    local content="$1" result
    if command -v shasum >/dev/null 2>&1; then
        result=$(printf '%s\n' "$content" | shasum -a 256 | awk '{print $1}') || return
    elif command -v sha256sum >/dev/null 2>&1; then
        result=$(printf '%s\n' "$content" | sha256sum | awk '{print $1}') || return
    else
        echo "_vibeqc_sha256_text: neither shasum nor sha256sum on PATH." >&2
        return 1
    fi
    [ -n "$result" ] || return 1
    printf '%s\n' "$result"
}

# Original v1 recipe hash: every non-comment, non-blank line in the builder.
# New writers retain it as field 3 so a v0.15.121-or-older reader can still
# consume a new stamp, and new readers use it to accept a three-field legacy
# stamp once before the next normal setup pass upgrades the row to v2.
_vibeqc_legacy_recipe_hash() {
    local stripped
    stripped=$(grep -vE '^[[:space:]]*(#.*)?$' "$1" || true)
    _vibeqc_sha256_text "$stripped"
}

# Emit only build-relevant builder content. The exclusion is deliberately
# explicit and narrow: code is ignored only between a matched pair of exact
# lifecycle-only markers. Everything else executable remains included by
# default. Nested, orphaned, or unterminated markers fail closed rather than
# silently hiding the remainder of a build recipe.
_vibeqc_recipe_content() {
    awk -v source="$1" '
        function fail(message) {
            print "_vibeqc_recipe_hash: " source ": " message > "/dev/stderr"
            invalid = 1
            exit 2
        }
        /^[[:space:]]*#[[:space:]]*vibeqc-recipe-hash:[[:space:]]*lifecycle-only-begin[[:space:]]*$/ {
            if (excluded) {
                fail("nested lifecycle-only-begin marker")
            }
            excluded = 1
            next
        }
        /^[[:space:]]*#[[:space:]]*vibeqc-recipe-hash:[[:space:]]*lifecycle-only-end[[:space:]]*$/ {
            if (!excluded) {
                fail("lifecycle-only-end marker without a matching begin")
            }
            excluded = 0
            next
        }
        excluded { next }
        /^[[:space:]]*(#.*)?$/ { next }
        { print }
        END {
            if (excluded && !invalid) {
                fail("lifecycle-only-begin marker without a matching end")
            }
        }
    ' "$1"
}

# sha256 of a build_<dep>.sh's *build-relevant* content. Full-line comments,
# blank lines, and the explicitly marked lifecycle-only block are stripped.
# Editing locking mechanics therefore does NOT register as drift, while any
# change to a configure flag, cmake argument, source/version pin, compiler
# selection, or other unmarked command does. Inline trailing comments are NOT
# stripped (rare in these scripts), erring on the safe side.
#
# One recipe carries build-relevant state that is NOT in its own text:
# build_libint.sh reads its per-derivative-order max_am from
# VIBEQC_LIBINT_MAX_AM (scripts/_libint_max_am.sh). Without folding that in,
# switching 5_4_4 <-> 5_4_3 would leave this hash — and therefore update.sh's
# drift check — completely blind to a differently-configured libint, which is
# the same stale-install failure this whole file exists to prevent. The salt is
# appended ONLY for that recipe, so every other dep's hash stays byte-identical
# to what pre-existing stamps recorded and nobody gets a spurious rebuild out
# of it. The legacy (v1) hash is deliberately left unsalted: it is a
# fixed-format compatibility artifact for readers that predate the option.
_vibeqc_recipe_hash() {
    local stripped salt=""
    if ! stripped="$(_vibeqc_recipe_content "$1")"; then
        return 1
    fi
    case "$(basename "$1")" in
        build_libint.sh) salt="max_am=$(vibeqc_libint_max_am_spec)" ;;
    esac
    if [ -n "$salt" ]; then
        stripped="$stripped
$salt"
    fi
    _vibeqc_sha256_text "$stripped"
}

# Exact, reviewed compatibility aliases for three-field stamps whose builder
# changed only in lifecycle code that predated the v2 exclusion markers. Each
# entry binds the builder path, the complete historical v1 SHA-256, and the
# complete current v2 SHA-256. The current hash is computed only after marker
# validation by _vibeqc_recipe_hash, so malformed markers fail before this
# whitelist is consulted. Any source pin, configure flag, CMake argument, or
# other unmarked recipe change also changes the current hash and invalidates
# the alias.
#
# There is currently NO entry. The one that existed — scripts/build_libint.sh
# at 8d3443c65f0c2480a0688d928a133da48488c815, whose only change was the lock
# source/acquire pair that 4caf8b2a7 moved and 0255fec94 marked lifecycle-only
# — was retired on 2026-08-06, when the max_am change (deriv-2 tier 3 -> 4,
# plus --libint-max-am) made build_libint.sh differ in genuinely
# build-relevant content. Those legacy-stamped trees were compiled at
# max_am 5;4;3 and really do need recompiling, so declining to migrate them is
# the correct, fail-closed answer.
#
# Do NOT "repair" this by re-pinning the alias to the new current hash. That
# would tell a legacy-stamped install it matches a recipe it was not built
# from, skipping the rebuild and leaving analytic Hessians silently unable to
# handle g-function bases — exactly the stale-install class this file exists
# to catch.
_vibeqc_reviewed_legacy_recipe_matches() {
    local script="$1" legacy_stamped="$2" current="$3"
    case "$script" in
        *) return 1 ;;
    esac
}

# Compare a stamp row's recipe fields with the current script. Four-field v2
# rows use the build-relevant fingerprint; three-field legacy rows use the
# original all-executable-lines fingerprint. This is intentionally read-only:
# setup_native_deps.sh rewrites a matching legacy row to v2 later in the same
# normal update, without rebuilding the dependency.
_vibeqc_stamp_recipe_matches() {
    local script="$1" legacy_stamped="$2" v2_stamped="${3:-}"
    local current legacy_current
    if [ -n "$v2_stamped" ]; then
        case "$v2_stamped" in
            v2:*)
                if ! current="$(_vibeqc_recipe_hash "$script")"; then
                    return 1
                fi
                [ "$v2_stamped" = "v2:$current" ]
                ;;
            *) return 1 ;;
        esac
    else
        # Validate the v2 markers even while consuming a legacy row. Otherwise
        # an unmatched marker (a comment under v1 rules) could appear current
        # until the later stamp rewrite failed.
        if ! current="$(_vibeqc_recipe_hash "$script")"; then
            return 1
        fi
        if ! legacy_current="$(_vibeqc_legacy_recipe_hash "$script")"; then
            return 1
        fi
        if [ "$legacy_stamped" = "$legacy_current" ]; then
            return 0
        fi
        # A pre-locking v1 builder has no lifecycle block, so its all-lines
        # hash is exactly the current v2 build-relevant hash after locking is
        # added inside an excluded block. Accepting that equality is safe: it
        # proves every old executable line matches current build-relevant
        # content, while avoiding a rebuild caused solely by the new lock.
        if [ "$legacy_stamped" = "$current" ]; then
            return 0
        fi
        _vibeqc_reviewed_legacy_recipe_matches \
            "$script" "$legacy_stamped" "$current"
    fi
}

# Extract the pinned version literal from a build_<dep>.sh — e.g.
# LIBINT_VERSION="v2.13.1" / FFTW_VERSION="3.3.10" / OPENBLAS_VERSION=...
_vibeqc_extract_version() {
    awk -F'=' '
        /^[A-Z_]+_VERSION=/ {
            v = $2
            gsub(/^"|"$/, "", v)
            print v
            exit
        }
    ' "$1"
}

# ---------------------------------------------------------------------------
# Native-library artifact health (the 2026-06-13 fleet wiped-.so failure)
# ---------------------------------------------------------------------------
# The per-dep build_<dep>.sh short-circuits on the presence of its CMake
# *config* file (lib/cmake/<Dep>/...Config.cmake), and the stamp records a
# pinned version + recipe hash — but NONE of those notice when the actual
# shared library is deleted out from under a built install/ tree. A managed
# venv whose third_party/libint/install/lib/libint2.so was wiped then fails
# at `import vibeqc` (ImportError: libint2.so: cannot open shared object
# file) yet never self-heals through a normal `vq admin update`: git is
# current (no drift), the stamp says "built", and the CMake config still
# exists, so every layer short-circuits the rebuild. The two helpers below
# add the one check those layers miss — is the library actually on disk? —
# so the drift/rebuild logic treats a wiped artifact like a missing install.

# Basename of the primary shared library each dep installs, i.e. the file
# whose absence breaks `import vibeqc`. Note spglib's library is libsymspg
# (not libspglib) and fftw's is libfftw3. Returns non-zero for an unknown
# dep so callers can distinguish "known-and-missing" from "don't know".
_vibeqc_dep_lib_basename() {
    case "$1" in
        libint)    echo "libint2" ;;
        libxc)     echo "libxc" ;;
        spglib)    echo "libsymspg" ;;
        fftw)      echo "libfftw3" ;;
        libecpint) echo "libecpint" ;;
        openblas)  echo "libopenblas" ;;
        *)         return 1 ;;
    esac
}

# Returns 0 if dep's primary shared library is present under
# third_party/<dep>/install/lib*/ — covers lib/ vs lib64/, an ELF
# .so[.MAJOR...] or a Mach-O .dylib, and the bare-or-versioned symlink every
# correct CMake install lays down. Returns 1 when the install tree exists
# but the library is gone (the wiped-.so case build_<dep>.sh's CMake-config
# short-circuit misses). Returns 2 for a dep whose library name we don't
# know, so a caller can treat that as "can't tell", not "missing". Expects
# CWD = repo root, like the other helpers here. The unquoted globs below
# are intentional: with no match bash leaves the pattern literal and the
# `[ -e ]` test is simply false (same idiom setup_native_deps.sh uses for
# its BLAS probe), so no nullglob dependency.
_vibeqc_dep_artifact_present() {
    local dep="$1" base f
    base="$(_vibeqc_dep_lib_basename "$dep")" || return 2
    for f in "third_party/$dep/install"/lib*/"$base".so* \
             "third_party/$dep/install"/lib*/"$base".dylib; do
        if [ -e "$f" ]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# vibeqc_stamp_write
#
# Writes third_party/.build-stamp from the current build_*.sh files.
# Caller invokes this *after* a successful setup_native_deps.sh run.
# ---------------------------------------------------------------------------
vibeqc_stamp_write() {
    local stamp="$VIBEQC_STAMP_PATH"
    local dir tmp dep script
    dir=$(dirname "$stamp")
    [ -d "$dir" ] || mkdir -p "$dir"

    # Validate first so a broken/partial install cannot replace a previously
    # good stamp. Missing installs are omitted (partial targeted builds are
    # legitimate); present installs must carry their primary shared library.
    while read -r dep script; do
        [ -z "$dep" ] && continue
        [ -d "third_party/$dep/install" ] || continue
        if ! _vibeqc_dep_artifact_present "$dep"; then
            echo "vibeqc_stamp_write: refusing to stamp $dep without its primary library." >&2
            return 1
        fi
    done < <(_vibeqc_stamp_pairs)

    tmp="$(mktemp "${stamp}.tmp.XXXXXX")" || return

    if ! {
        echo "# Auto-generated by scripts/setup_native_deps.sh — do not edit by hand."
        echo "# Each line:  <dep-name>  <pinned-version>  <legacy-v1-hash>  v2:<recipe-hash>"
        echo "# Used by install.sh / update.sh / doctor.sh to detect drift between"
        echo "# the vendored install/ trees and the currently-checked-out build_*.sh."
        echo "# Regenerated on every successful setup_native_deps.sh run."
        echo "#"
        echo "# Written at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$tmp"; then
        rm -f "$tmp"
        return 1
    fi

    local version sha legacy_sha
    while read -r dep script; do
        [ -z "$dep" ] && continue
        if [ ! -f "$script" ]; then
            continue
        fi
        # Never stamp a recipe for a dependency that has no verified
        # installed artifact. A partial/fresh targeted build may produce
        # only some deps; omitted lines correctly remain drift/missing.
        if [ ! -d "third_party/$dep/install" ]; then
            continue
        fi
        if ! version="$(_vibeqc_extract_version "$script")"; then
            rm -f "$tmp"
            return 1
        fi
        if ! sha="$(_vibeqc_recipe_hash "$script")" || [ -z "$sha" ]; then
            rm -f "$tmp"
            return 1
        fi
        if ! legacy_sha="$(_vibeqc_legacy_recipe_hash "$script")" || \
           [ -z "$legacy_sha" ]; then
            rm -f "$tmp"
            return 1
        fi
        if ! printf '%-12s %-12s %s v2:%s\n' \
            "$dep" "${version:-unknown}" "$legacy_sha" "$sha" >> "$tmp"; then
            rm -f "$tmp"
            return 1
        fi
    done < <(_vibeqc_stamp_pairs)
    if ! mv "$tmp" "$stamp"; then
        rm -f "$tmp"
        return 1
    fi

    echo "==> Wrote $stamp"
}

# ---------------------------------------------------------------------------
# vibeqc_stamp_check
#
# Compares third_party/.build-stamp against the current build_*.sh
# files. Prints a per-dep status table and a recovery hint if any
# drift is detected. Returns:
#   0 — stamp matches, no action needed
#   1 — drift detected (caller may choose to warn or to force a rebuild)
#   2 — stamp missing (third_party/install trees exist but were built
#       before this stamp logic landed; we can't tell whether they're
#       current)
#
# Pass --quiet as first arg to suppress the "all current" success line
# (used by doctor.sh which already has its own header).
# ---------------------------------------------------------------------------
vibeqc_stamp_check() {
    local quiet=""
    if [ "${1:-}" = "--quiet" ]; then
        quiet=1
        shift
    fi

    local stamp="$VIBEQC_STAMP_PATH"

    # Nothing to check if no install/ tree exists at all.
    local any_install=""
    local d
    for d in libint libxc spglib fftw libecpint openblas; do
        if [ -d "third_party/$d/install" ]; then
            any_install=1
            break
        fi
    done
    if [ -z "$any_install" ]; then
        [ -z "$quiet" ] && echo "==> No third_party/<dep>/install/ trees yet — nothing to check."
        return 0
    fi

    if [ ! -f "$stamp" ]; then
        cat <<EOF
==> No build stamp at $stamp.
    A third_party/ install tree exists but was built before the stamp
    helper landed (or by hand). Drift against the currently-checked-out
    build_*.sh cannot be detected. If you suspect the install is stale:
        ./scripts/update.sh --rebuild-native-deps
EOF
        return 2
    fi

    local drift=0
    local dep script ver_now stamp_line ver_stamped legacy_stamped v2_stamped status base
    local am_have am_want
    local report=()

    while read -r dep script; do
        [ -z "$dep" ] && continue
        if [ ! -f "$script" ]; then
            continue
        fi
        # A wiped shared library (install/ tree + CMake config survive, but
        # the .so/.dylib itself is gone) is the most severe form of drift,
        # and the one neither the stamp nor build_<dep>.sh's short-circuit
        # can see. Flag it before the version/sha comparison — which would
        # otherwise report "ok" — so the report names the missing artifact
        # and the rebuild is forced.
        if [ -d "third_party/$dep/install" ] && ! _vibeqc_dep_artifact_present "$dep"; then
            base="$(_vibeqc_dep_lib_basename "$dep" 2>/dev/null || echo "$dep")"
            report+=("  $(printf '%-12s %s' "$dep" "MISSING LIBRARY ($base.{so,dylib} gone → forcing rebuild)")")
            drift=1
            continue
        fi
        ver_now=$(_vibeqc_extract_version "$script")
        # First field-match for $dep in the stamp file.
        stamp_line=$(awk -v d="$dep" '$1 == d { print; exit }' "$stamp")
        if [ -z "$stamp_line" ]; then
            status="MISSING (built without stamp / opted-in after the fact)"
            drift=1
        else
            ver_stamped=$(echo "$stamp_line" | awk '{print $2}')
            legacy_stamped=$(echo "$stamp_line" | awk '{print $3}')
            v2_stamped=$(echo "$stamp_line" | awk '{print $4}')
            if [ "$ver_now" != "$ver_stamped" ]; then
                status="VERSION DRIFT ($ver_stamped → $ver_now)"
                drift=1
            elif ! _vibeqc_stamp_recipe_matches \
                    "$script" "$legacy_stamped" "$v2_stamped"; then
                # Distinguish "the recipe file changed" from "an out-of-file
                # build setting changed". For libint the latter is a
                # --libint-max-am switch, and reporting that as "$script
                # changed" would send the reader hunting through a diff that
                # does not exist. The installed marker is what the tree was
                # actually built with.
                am_have=""
                am_want=""
                if [ "$dep" = "libint" ]; then
                    am_have="$(vibeqc_libint_installed_spec)"
                    am_want="$(vibeqc_libint_max_am_spec)"
                fi
                if [ -n "$am_have" ] && [ "$am_have" != "unknown" ] \
                   && [ "$am_have" != "$am_want" ]; then
                    status="BUILD-SETTING DRIFT (max_am=$am_have → max_am=$am_want)"
                else
                    status="BUILD-FLAGS DRIFT (same version, $script changed)"
                fi
                drift=1
            elif [ -z "$v2_stamped" ]; then
                status="ok ($ver_now; legacy stamp upgrades on next setup)"
            else
                status="ok ($ver_now)"
            fi
        fi
        report+=("  $(printf '%-12s %s' "$dep" "$status")")
    done < <(_vibeqc_stamp_pairs)

    if [ "$drift" -eq 0 ]; then
        if [ -z "$quiet" ]; then
            echo "==> Native-dep build stamp: all current."
        fi
        return 0
    fi

    cat <<EOF
==> Native-dep build stamp: drift detected.
EOF
    local line
    for line in "${report[@]}"; do
        echo "$line"
    done
    cat <<EOF

    The vendored install/ trees no longer match what the current
    build_*.sh would produce. Force a clean rebuild with:
        ./scripts/update.sh --rebuild-native-deps
EOF
    return 1
}

# ---------------------------------------------------------------------------
# vibeqc_stamp_drifted_deps
#
# Echoes (one per line, in _vibeqc_stamp_pairs order) the name of every
# dep that needs rebuilding to match the current build_<dep>.sh:
#   * its third_party/<dep>/install/ tree is missing, OR
#   * its install/ tree exists but the primary shared library was wiped
#     (libint2.so / libxc.so / ... gone while lib/cmake/ + headers
#     remain) — checked regardless of the stamp, since this breaks
#     `import vibeqc` yet build_<dep>.sh's CMake-config short-circuit
#     would skip the rebuild, OR
#   * the stamp exists and the dep's pinned version or build-relevant
#     recipe fingerprint differs from what the stamp recorded.
# A dep whose install exists, whose library is present, but for which no
# stamp exists is NOT emitted — recipe drift is undetectable without a
# stamp, so the caller should fall back to an explicit full rebuild if it
# suspects staleness.
#
# Pure: stdout only, no side effects. This is the targeted counterpart
# to vibeqc_stamp_check (which prints a human report + return code);
# update_native_deps.sh consumes this to rebuild only what changed,
# instead of the all-or-nothing wipe in `update.sh --rebuild-native-deps`.
# Like the other helpers here it expects the caller's CWD to be the repo
# root (third_party/<dep>/install is resolved relative to it).
# ---------------------------------------------------------------------------
vibeqc_stamp_drifted_deps() {
    local stamp="$VIBEQC_STAMP_PATH"
    local dep script ver_now stamp_line ver_stamped legacy_stamped v2_stamped
    while read -r dep script; do
        [ -z "$dep" ] && continue
        [ -f "$script" ] || continue
        # Missing install → must (re)build regardless of stamp.
        if [ ! -d "third_party/$dep/install" ]; then
            echo "$dep"
            continue
        fi
        # Install tree survived but the primary shared library was wiped
        # (libint2.so deleted while lib/cmake/ + headers remain). The
        # CMake-config short-circuit in build_<dep>.sh would skip the
        # rebuild and leave `import vibeqc` broken, so force this dep
        # regardless of the stamp.
        if ! _vibeqc_dep_artifact_present "$dep"; then
            echo "$dep"
            continue
        fi
        # Install present but no stamp → drift undetectable; skip.
        [ -f "$stamp" ] || continue
        ver_now=$(_vibeqc_extract_version "$script")
        stamp_line=$(awk -v d="$dep" '$1 == d { print; exit }' "$stamp")
        if [ -z "$stamp_line" ]; then
            echo "$dep"          # install present but never stamped
            continue
        fi
        ver_stamped=$(echo "$stamp_line" | awk '{print $2}')
        legacy_stamped=$(echo "$stamp_line" | awk '{print $3}')
        v2_stamped=$(echo "$stamp_line" | awk '{print $4}')
        if [ "$ver_now" != "$ver_stamped" ] || \
           ! _vibeqc_stamp_recipe_matches \
               "$script" "$legacy_stamped" "$v2_stamped"; then
            echo "$dep"
        fi
    done < <(_vibeqc_stamp_pairs)
    return 0
}
