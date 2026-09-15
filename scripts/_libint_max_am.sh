# scripts/_libint_max_am.sh — resolve libint's per-derivative-order max_am.
#
# ONE setting, several consumers:
#
#   * build_libint.sh       turns it into cmake arguments, and refuses to
#                           short-circuit on an install/ tree that was
#                           built with a different one;
#   * _native_stamp.sh      folds it into libint's recipe hash, so changing
#                           the setting registers as build drift and
#                           update.sh rebuilds libint on its own;
#   * install.sh /          validate the --libint-max-am flag at parse
#     update.sh /           time, before a multi-hour build starts.
#     update_native_deps.sh
#
# The setting travels as the environment variable VIBEQC_LIBINT_MAX_AM,
# spelled N_N_N: the max angular momentum for derivative orders 0, 1 and 2
# in that order. The default "5_4_4" means energies to h functions,
# gradients to g, second derivatives to g for the orbital centers.
# The two/three-center fitting limits are independently fixed at 6_6_6
# in build_libint.sh, so physical i-function auxiliary shells remain usable
# without raising the expensive four-center orbital kernel limit.
#
# Why it is worth configuring at all: libint's generated-kernel count grows
# roughly as max_am^4 per derivative order, and the deriv-2 stratum
# dominates the build.
#
#   5_4_3   what vibe-qc shipped before 2026-08-06. Energies to h, but
#           Hessians limited to f functions -- libint throws on a
#           g-function basis. Budget ~30 min. Pick this if you do not
#           run analytic Hessians, or do not run them on large bases.
#   5_4_4   default. Adds analytic Hessians on g-function basis sets
#           (def2-TZVPP heavy atoms, cc-pVQZ). Budget ~1.5-2 h.
#   6_5_4   adds i-function (L = 6) energies and h-function gradients,
#           for cc-pV6Z-class work. Substantially longer again -- the
#           deriv-0 stratum alone grows by ~(6/5)^4.
#   6_6_6   everything the cart_to_sph table can express. Very long
#           build; only worth it if you actually need i-function
#           second derivatives.
#
# Nothing here alters a computed number. The spec only decides which
# basis sets each derivative order can accept -- above its limit libint
# throws rather than returning something wrong. Derivative orders 0, 1
# and 2 are ALWAYS generated: the spec sets the angular momentum at each
# order, not which orders exist. The analytic Hessian (Phase 17b) needs
# deriv order 2 to be present at all, so there is deliberately no way to
# switch that off here.
#
# CEILING: 6, and it is not libint's. vibe-qc's periodic AO-pair Fourier
# transform (ao_pair_fourier_transform_bloch) transforms Cartesian to
# spherical through the generated table in
# cpp/include/vibeqc/cart_to_sph_data.hpp, whose kMaxL is 6, and throws
# on any shell above it. Building libint higher would produce integrals
# the periodic stack cannot consume -- a half-working configuration.
# Raising the ceiling means bumping MAX_L in
# scripts/codegen_cart_to_sph.py, regenerating the header, and
# re-validating the fit precision (its own comment warns the fit gets
# rank-revealing past L = 6 and needs a larger sample set) -- not just
# relaxing this check.

VIBEQC_LIBINT_MAX_AM_DEFAULT="5_4_4"

# Highest angular momentum any part may request. Bounded by
# vibeqc::cart_to_sph_data::kMaxL -- see the CEILING note above. Keep the
# two in lockstep.
VIBEQC_LIBINT_MAX_AM_CEILING=6

# Echo the effective spec, normalised to N_N_N (';' and ',' are accepted as
# separators too, since the cmake-list spelling is the obvious thing to
# type). Never fails and never validates: _native_stamp.sh calls this from
# command substitution under `set -euo pipefail`, where a non-zero return
# would abort the caller. Validation is vibeqc_libint_max_am_resolve's job.
vibeqc_libint_max_am_spec() {
    local raw
    raw="${VIBEQC_LIBINT_MAX_AM:-$VIBEQC_LIBINT_MAX_AM_DEFAULT}"
    printf '%s\n' "$raw" | tr ';,' '__'
}

# Print the "here is how to spell it" block. Separate from the resolver so
# the wrapper scripts' --help text can reuse it.
vibeqc_libint_max_am_hint() {
    cat >&2 <<'EOF'
    Spell it N_N_N: max angular momentum for derivative orders 0, 1, 2.
    Each part must be 1..6.
        5_4_3   f-function Hessians only, energies to h  (~30 min build)
        5_4_4   default; g-function Hessians             (~1.5-2 h build)
        6_5_4   + i-function energies, h-function gradients   (longer)
        6_6_6   everything the cart_to_sph table allows       (longest)
    Pass it as --libint-max-am to install.sh / update.sh /
    update_native_deps.sh, or export VIBEQC_LIBINT_MAX_AM.
EOF
}

# Validate the effective spec and publish the derived values:
#
#   VIBEQC_LIBINT_AM_SPEC    normalised spec  (e.g. "5_4_4")
#   VIBEQC_LIBINT_AM_LIST    cmake list form  (e.g. "5;4;4")
#   VIBEQC_LIBINT_AM_GLOBAL  LIBINT2_MAX_AM   (the largest of the three;
#                                              libint requires the global
#                                              cap to cover every stratum)
#
# Returns 1 with a message on stderr if the spec is malformed.
vibeqc_libint_max_am_resolve() {
    local spec part max=0 n=0

    spec="$(vibeqc_libint_max_am_spec)"

    local _oldifs="$IFS"
    IFS='_'
    # Word-splitting on '_' is the point here.
    # shellcheck disable=SC2086
    set -- $spec
    IFS="$_oldifs"

    if [ "$#" -ne 3 ]; then
        echo "Error: invalid libint max_am spec '$spec' -- expected exactly" >&2
        echo "       three parts (derivative orders 0, 1, 2), got $#." >&2
        vibeqc_libint_max_am_hint
        return 1
    fi

    for part in "$@"; do
        n=$((n + 1))
        case "$part" in
            ''|*[!0-9]*)
                echo "Error: invalid libint max_am spec '$spec' -- part $n" >&2
                echo "       ('$part') is not a non-negative integer." >&2
                vibeqc_libint_max_am_hint
                return 1
                ;;
        esac
        if [ "$part" -lt 1 ] || [ "$part" -gt "$VIBEQC_LIBINT_MAX_AM_CEILING" ]; then
            echo "Error: invalid libint max_am spec '$spec' -- part $n ($part) is" >&2
            echo "       out of range; each part must be 1..$VIBEQC_LIBINT_MAX_AM_CEILING." >&2
            if [ "$part" -gt "$VIBEQC_LIBINT_MAX_AM_CEILING" ]; then
                cat >&2 <<EOF
       The upper bound is vibe-qc's, not libint's. The periodic AO-pair
       FT transforms through cpp/include/vibeqc/cart_to_sph_data.hpp,
       whose kMaxL is $VIBEQC_LIBINT_MAX_AM_CEILING, and throws on any shell above it -- a higher
       libint would emit integrals nothing downstream can consume.
       Raising it means bumping MAX_L in scripts/codegen_cart_to_sph.py,
       regenerating that header, re-validating the fit precision, and
       moving VIBEQC_LIBINT_MAX_AM_CEILING in lockstep.
EOF
            fi
            vibeqc_libint_max_am_hint
            return 1
        fi
        # if/then, NOT `[ ... ] && max=...`: a false test at the end of the
        # loop body would return 1 and trip the caller's `set -e`.
        if [ "$part" -gt "$max" ]; then
            max="$part"
        fi
    done

    VIBEQC_LIBINT_AM_SPEC="${1}_${2}_${3}"
    VIBEQC_LIBINT_AM_LIST="${1};${2};${3}"
    VIBEQC_LIBINT_AM_GLOBAL="$max"

    # Angular momentum rising with derivative order is almost certainly a
    # typo, and is wasteful either way. Warn rather than refuse -- there is
    # no correctness reason to forbid it, and an unforeseen use is cheaper
    # to allow than to argue with.
    if [ "$2" -gt "$1" ] || [ "$3" -gt "$2" ]; then
        echo "Warning: libint max_am spec '$VIBEQC_LIBINT_AM_SPEC' rises with" >&2
        echo "         derivative order; the convention is non-increasing" >&2
        echo "         (e.g. 5_4_4). Building it anyway." >&2
    fi

    return 0
}

# Validate the spec a wrapper script was just handed and export it for the
# build scripts further down the chain. Usage, from an argv parser:
#
#     --libint-max-am)
#         vibeqc_libint_max_am_accept "$2" || exit 1
#         shift 2 ;;
#
# Fails fast and loudly: the whole point is to reject "5_4_x" now rather
# than after install.sh has already spent 20 minutes on libxc.
vibeqc_libint_max_am_accept() {
    if [ -z "${1:-}" ]; then
        echo "Error: --libint-max-am requires an argument (e.g. 5_4_3)." >&2
        vibeqc_libint_max_am_hint
        return 1
    fi
    VIBEQC_LIBINT_MAX_AM="$1"
    export VIBEQC_LIBINT_MAX_AM
    vibeqc_libint_max_am_resolve || return 1
    # Re-export the normalised form so every child sees one spelling.
    VIBEQC_LIBINT_MAX_AM="$VIBEQC_LIBINT_AM_SPEC"
    export VIBEQC_LIBINT_MAX_AM
    return 0
}

# Path of the marker build_libint.sh writes into a completed install tree,
# recording the spec that tree was actually built with. Relative to the
# repo root, like the other helpers here.
VIBEQC_LIBINT_MARKER="third_party/libint/install/.vibeqc-max-am"

# Echo the spec an existing libint install was built with, or "unknown" if
# the tree predates the marker (or no tree exists). Never fails.
vibeqc_libint_installed_spec() {
    local marker="${1:-$VIBEQC_LIBINT_MARKER}"
    if [ -r "$marker" ]; then
        # First non-empty, non-comment line.
        awk 'NF && $0 !~ /^[[:space:]]*#/ { print $1; exit }' "$marker"
    else
        echo "unknown"
    fi
}
