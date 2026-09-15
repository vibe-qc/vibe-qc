# Updating vibe-qc

For an existing checkout you've already used. If this is your first
install, see [installation](installation.md) instead.

This page covers the native vibe-qc engine. For the separate viewer and queue,
and the basis driver retained in this checkout, see
[Install and maintain the vibe toolset](toolset_lifecycle.md).

## Moving from the archived monorepo

The old `mpei/vibeqc` repository (project 19) is archived. The split projects
have fresh histories: do not change an old checkout's remote and rebase it
onto the new `main`. Keep it for reproducibility and create a fresh clone:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/install.sh --dev --extras test --venv .venv \
    --python /opt/homebrew/bin/python3.14
```

That is the tested macOS route; select an installed supported Python on other
platforms. Preserve calculation files and old environments. Install viewer
and queue from their own repositories; see [toolset lifecycle](toolset_lifecycle.md).
Historical tags and commit IDs belong to the archived repository unless
explicitly present in the new one. `vibe-basis/` stays with the core.

The repository published the `v0.16.0` tag and the `release` branch on
2026-09-08, so the release-specific examples below work; `--dev` follows
`main`. They still apply only to refs actually published in the selected
repository -- historical monorepo tags are not among them.

## The easy button

With no branch flag, or with `--release`, the updater follows origin's
`release` branch. Repositories publishing only main and release tags instead
select the newest stable `vX.Y.Z` tag advertised by origin, at a detached HEAD.
Local-only tags and prereleases are excluded. `--branch` and `--ref` remain
exact requests. A dry run reports this selection policy without fetching.

```sh
./scripts/update.sh --dev
```

That command pulls the current `main` development source, rebuilds any native
dependencies whose source changed, refreshes the Python package
inside your `.venv/`, and prints the new banner so you can confirm
the version flipped. About 30 seconds wall-time on a vanilla
`git pull` (no native-dep changes); 5-15 minutes if a vendored
library version bumped between releases.

After the script finishes you should see a banner like:

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Release v0.12.0 "Knuth's Beaver"  --  Quantum chemistry for molecules and solids                                           ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com                                                                 ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7 (vendored, MAX_L=5) · fftw3 3.3.10 · blas Accelerate ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

With the optional ``[dispersion]`` extra installed:

```
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7 (vendored, MAX_L=5) · fftw3 3.3.10 · blas Accelerate ║
║ dispersion: dftd3 1.4.0 · dftd4 4.2.0                                                                                     ║
```

The first line tells you which release you're on. If it shows the
old version you didn't expect, see [common issues](#common-issues)
below.

## Variants

```sh
./scripts/update.sh --release            # latest tagged release (default; flag is for symmetry)
./scripts/update.sh --dev                # bleeding-edge main (X.Y.devN banner)
./scripts/update.sh --branch main        # verbose form of --dev
./scripts/update.sh --branch <existing-branch> # select a published branch
./scripts/update.sh --branch <existing-tag>      # pin to a specific tag
./scripts/update.sh --ref <existing-tag>         # older spelling of --branch, kept for back-compat
./scripts/update.sh --dev --rebuild-native-deps   # see "Vendored library version bumps" below
./scripts/update.sh --dev --recreate-venv      # atomic venv replacement with rollback
./scripts/update.sh --dev --recreate-venv --python python3.13  # intentionally change Python
./scripts/update.sh --dev --with-openblas      # add or retain vendored OpenBLAS
./scripts/update.sh --dev --clean              # native sources/builds + venv + build cache
./scripts/update.sh --dev --dry-run            # preview: branch / drift / venv state, no changes
./scripts/update.sh --dev --extras dev         # developer profile; install the separate viewer explicitly
./scripts/update.sh --dev --venv .venv-bsd     # explicit venv (for side-by-side per-branch venvs)
./scripts/update.sh --dev --rebuild-native-deps    # combine flags
```

`--release` / `--dev` are named shortcuts; `--branch NAME` accepts
any branch or tag (`--ref NAME` is kept as the older spelling).
The branch-selection flags are mutually exclusive, pick one. The
default targets `release`; use it only after that branch is published in
the new repository. **`--dev`** switches to `main` for previewing
in-flight features before they ship.

`--recreate-venv` is the targeted fix for a broken venv: stale
`pip` shebang after copying a venv across trees, hybrid
`pyvenv.cfg` after `python3 -m venv` ran against a different
interpreter, or just "import vibeqc segfaults and I don't know
why". The prior environment waits in a same-parent backup until its
replacement installs and verifies; any failure restores it. By default the
replacement preserves the old venv's base interpreter. Use `--python` only
when you intentionally want to change it.

Replacement also requires this checkout's lifecycle ownership marker. For an
installation created before markers existed, add `--adopt-legacy`; the script
then reads PEP 610 metadata with a trusted external Python and proceeds only
when that metadata points to this exact checkout. A foreign marker is never
adopted. Ordinary in-place updates do not claim ownership of shared venvs.

`--clean` is the nuclear option: combines `--rebuild-native-deps`
+ `--recreate-venv`, removes native source/build/install trees, and wipes
Python build caches. It never removes the tracked basis library.
Takes 15-40 min on a clean run but is guaranteed reproducible,
use after a known-bad build state, before publishing benchmark
numbers, or when bisecting a regression.

`--dry-run` previews everything without touching anything: the
target commit, build-stamp drift against the currently-checked-out
`build_*.sh` files, venv health. Safe to run during an in-flight
calculation.

An update requires an existing usable venv. If none is found, it exits before
fetching or rebuilding anything and directs you to `install.sh`; it never
reports a partial update as success. For environment-only lifecycle work, use
`reinstall.sh` or `uninstall.sh`.

If you keep two checkouts side-by-side (one for production runs,
one for dev), use `update.sh` (or `--release`) inside the release
tree and `update.sh --dev` inside the dev tree. The lighter-weight
zsh [`vibe-update`](#maintaining-dev--release--experimental-side-by-side)
helper later in this page knows about both trees and refreshes them
in one shot.

`-h` / `--help` prints the same option list inline.

## Build-pressure safety

On Linux, `update.sh` (and `setup_native_deps.sh`, and each
`build_*.sh`) self-re-execs under `nice -n 19 ionice -c 3` and
sets a memory-safe default for `CMAKE_BUILD_PARALLEL_LEVEL`. The
first line of output on a fresh build will look like:

```
==> CMAKE_BUILD_PARALLEL_LEVEL=8 (nproc=32, mem=128721MB; cap@8 / 15GB-per-worker)
```

The cap is `min(nproc, max(2, mem_mb // 15000), 8)`, which keeps
`ninja` from firing more cc1plus workers than the host has RAM
for. Each cc1plus on vibe-qc's template-heavy translation units
(libint integrals, `periodic_*.cpp`, `gradient.cpp`) peaks at
8-10 GB resident; the formula budgets 15 GB/worker with a hard
ceiling of 8, above 8, `ninja` serializes on link/IO contention
and extra parallelism just thrashes the page cache. The
`nice -n 19 ionice -c 3` prefix keeps the foreground shell
responsive during long builds.

This exists because unbounded `ninja` parallelism on compute-reference
(32 threads / 125 GB) triggered global-OOM and required a hard
reset twice on 2026-05-16. The helper script
(`scripts/_safe_build_env.sh`) is sourced by every entry-point
script that triggers heavy compilation, so the cap propagates
through the full chain.

**Overrides** (set in your env *before* invoking `update.sh`):

```sh
CMAKE_BUILD_PARALLEL_LEVEL=12 ./scripts/update.sh --dev
# → pin parallelism explicitly (skips the formula)

VIBEQC_BUILD_NICED=1 ./scripts/update.sh --dev
# → skip the nice/ionice re-exec (full CPU + IO priority)

VIBEQC_BUILD_ENV_QUIET=1 ./scripts/update.sh --dev
# → silence the cap-announcement banner
```

**macOS**: both side-effects skip cleanly (no `/proc/meminfo`,
no `ionice`). Set `CMAKE_BUILD_PARALLEL_LEVEL` manually if your
dev box is RAM-constrained.

See [`operations.md`](https://github.com/vibe-qc/vibe-queue/blob/main/docs/operations.md) in the
vibe-queue tree for the full post-mortem of the compute-reference
incident and the recovery procedures for related failure modes
(zombie user-systemd, stale daemon-in-memory after `pip install
-e .`, etc.).

## Advanced diagnostic outline

The managed updater also holds cross-platform checkout and environment locks,
checks ownership and build provenance, targets only drifted dependencies,
verifies the installed extension, and rolls back a replaced environment on
failure. The commands below are only a diagnostic outline for understanding an
update; they are not a safe drop-in equivalent for normal maintenance:

```sh
# 1. Refuse if working tree is dirty (silently stashing user work
#    is worse than failing loud)
git diff --quiet && git diff --cached --quiet || \
    { echo "uncommitted changes -- commit or stash"; exit 1; }

# 2. Fetch and check out the target ref (release branch by default)
git fetch origin --tags
git checkout release
git pull --ff-only origin release        # only on branches; tags don't pull

# 3. Rebuild any native deps whose source changed
./scripts/setup_native_deps.sh

# 4. Refresh the Python package (rebuilds the C++ extension)
.venv/bin/pip install -e '.[test]'

# 5. Confirm the new banner
.venv/bin/python -c "import vibeqc; vibeqc.print_banner()"
```

Step 3 is the heavy one if anything changed. The managed build stamps and
artifact checks skip current dependencies and rebuild only drifted or damaged
ones, so a vanilla update usually takes seconds.

## Switching between releases

Sometimes you want to drop to a previous release temporarily, to
reproduce an old result, or to confirm a regression bisects to a
specific tag.

```sh
./scripts/update.sh --ref <existing-tag>
# ... run the calculation, compare ...
./scripts/update.sh --ref release         # back to current
```

The on-disk state is fully consistent at each step. Outputs from
your previous calculations survive, only the vibe-qc code changes.

## Going to bleeding-edge `main`

```sh
./scripts/update.sh --ref main
```

`main` is where active development lands. Banner reads
``dev X.Y.devN (main @ <sha>)``, the SHA pins the build for
reproducibility. **Use this** when you need a feature that's
already merged but not yet released; **don't use this** for
publication-quality numbers (subtle bugs may be undetected until
the next regression-test pass).

To go back to the published release:

```sh
./scripts/update.sh --ref release
```

## Common issues

### Banner still shows the old version after update

You ran `git pull` but didn't re-install the Python package. The
C++ extension is built once at `pip install` time and doesn't auto-
rebuild from `git pull` alone, `update.sh` does both, hand-rolled
flows need the explicit step.

**Fix**: re-run `./scripts/update.sh --dev`, or manually:

```sh
.venv/bin/pip install -e '.[test]' --force-reinstall --no-deps
```

### Vendored library version bumps

Each per-dep build script's idempotency check looks at "does
`third_party/<dep>/install/lib/cmake/X/XConfig.cmake` exist?", and
skips if yes. If a release bumps libxc 7.0.0 → 7.0.1 (or libint /
spglib / FFTW / libecpint), the user's stale install silently sticks
around unless the orchestrator reconciles provenance first.

**Diagnostic + auto-fix**: `update.sh` checks the `third_party/.build-stamp`
file (written on every successful native-deps build) against the
currently-checked-out `scripts/build_*.sh` files. If the version pinned
in `build_libxc.sh` differs from the stamp, or the *build-relevant*
content of `build_libxc.sh` changed (source pin, compiler selection,
configure flags, or CMake arguments), `update.sh` **rebuilds the drifted dep(s)
automatically and targeted**, via `scripts/update_native_deps.sh`: only
the changed dep's pinned source/build/install trees are removed and rebuilt,
not all of them. `setup_native_deps.sh` performs the same reconciliation
before its idempotent build calls, so it cannot relabel an old binary with a
new recipe hash. Legacy install trees without a stamp stay explicitly
uncertified until a full rebuild. Pass
`--rebuild-native-deps` to force a full rebuild of every dep instead.
`./scripts/doctor.sh` runs the same drift check on demand and only
reports (never rebuilds). You can also invoke the targeted rebuild
directly: `./scripts/update_native_deps.sh` (`--dep NAME` / `--all` /
`--dry-run`).

The recipe fingerprint is conservative: every executable builder line counts
unless it is inside the narrowly marked lifecycle-only locking block. Full-line
comments, blank lines, and changes confined to that lock block do not change
the native artifact, so they do not trigger a costly libint/libxc rebuild.
Malformed or unmatched lifecycle markers fail closed and report recipe drift.
Build stamps written before this distinction have three fields; the next
normal update accepts a matching legacy fingerprint and upgrades it to the
four-field format without rebuilding the dependency.

One libint builder from immediately before the unified lifecycle move placed
the same lock acquisition later in the script, so its legacy fingerprint does
not equal either current fingerprint even though its native recipe is
unchanged. That reviewed case is admitted through an exact compatibility
triple: builder path, full historical SHA-256, and full current build-relevant
SHA-256 must all match. This is not a prefix or fuzzy migration. Unknown or
partial fingerprints, malformed lifecycle markers, and any change to the
source pin, configure inputs, CMake arguments, or other unmarked command still
report drift and rebuild libint.

The same targeted auto-rebuild also fires when a dep's `install/` tree is
present but its actual shared library was **deleted**, the drift check
looks for the real `libint2.so` / `libxc.so` / `libsymspg.so` / … on disk,
not just the surviving CMake config file. So a normal `update.sh` (and
`vq admin update`) now repairs a wiped library on its own; you no longer
need the `--rebuild-native-deps` force flag for that case. See *Wiped
shared library* below.

**Diagnostic, by hand**: the banner's
`linked: libint X.Y.Z · libxc A.B.C ...` line should match the
versions pinned in the various `scripts/build_<dep>.sh`. If it
doesn't, the stale-install case has bitten.

**Fix**:

```sh
./scripts/update.sh --dev --rebuild-native-deps
```

That removes every pinned native source/build/install tree (and the stamp)
before re-running the native-deps orchestrator, forcing each dependency to
fetch and rebuild its currently pinned version. An existing vendored OpenBLAS
selection is retained automatically, even if its install tree is damaged,
instead of silently falling back to system BLAS.

(libint-stale-install-footgun-build-flags-changed)=
### Rebuild one native dependency

Current build stamps include the pinned version and recipe fingerprint. A
changed libint build recipe, missing artifact, or stale source is detected
instead of trusting an existing directory. To force only libint through the
locked native updater:

```sh
./scripts/update_native_deps.sh --dep libint
./scripts/update.sh --dev
```

Use `./scripts/update.sh --dev --rebuild-native-deps` when every vendored dependency
needs a clean rebuild. Do not remove native trees manually while another
lifecycle or build command may be active.

**You normally don't have to do this by hand.** `update.sh`
hashes each `build_<dep>.sh`'s build-relevant lines into the
native-dep stamp, so a changed cmake flag registers as drift
and triggers a rebuild of just that dep. The manual
`rm -rf` recipe above is for installs that pre-date the stamp
(`update.sh` says "Legacy install (no build stamp)"), or when
you've bypassed `update.sh` entirely.

Concretely, the 2026-08-06 bump of the 2nd-derivative
`max_am` tier from 3 to 4 (analytic Hessians on g-function
basis sets) is a flag change of exactly this kind: running
`./scripts/update.sh --dev` after pulling it rebuilds libint
automatically. Budget 1.5-2 h for that rebuild, since the
generated-kernel count for the deriv-2 stratum roughly
triples.

During that rebuild the terminal shows one heartbeat line per
minute rather than the raw codegen/compile stream; the full
output is in `third_party/libint/build.log` (`tail -f` it, or
set `VIBEQC_BUILD_VERBOSE=1` to stream). A failed build
replays the last 200 log lines automatically.

### Choosing libint's `max_am` (`--libint-max-am`)

That rebuild is optional. The spec is three numbers: the
max angular momentum at derivative orders 0, 1 and 2:

```bash
./scripts/update.sh --dev --libint-max-am 5_4_3
```

| Spec | Build | Energies | Gradients | Hessians |
|---|---|---|---|---|
| `5_4_3` | ~30 min | h (L=5) | g (L=4) | f (L=3) |
| `5_4_4` (default) | ~1.5-2 h | h | g | g |
| `6_5_4` | longer | i (L=6) | h | g |
| `6_6_6` | longest | i | i | i |

Each part may be 1..6. Above a stratum's limit libint
throws rather than returning a wrong number, so the spec
only decides which basis sets each derivative order
accepts: no computed value changes. Most users want
`5_4_3` or `5_4_4`; the `6_*` specs are for cc-pV6Z-class
work.

**6 is the ceiling, and it is vibe-qc's rather than
libint's.** The periodic AO-pair Fourier transform
transforms Cartesian to spherical through the generated
table in `cpp/include/vibeqc/cart_to_sph_data.hpp`
(`kMaxL = 6`) and throws on any shell above it, so a
higher libint would emit integrals the periodic stack
cannot consume. Raising it means bumping `MAX_L` in
`scripts/codegen_cart_to_sph.py`, regenerating that
header, re-validating the fit precision (it gets
rank-revealing past L = 6), and moving
`VIBEQC_LIBINT_MAX_AM_CEILING` in lockstep.

The setting is not stored in `build_libint.sh`, so the build
stamp folds it into libint's recipe hash on purpose, and
`build_libint.sh` refuses to short-circuit on an install
tree built with a different spec. Both directions therefore
work as a plain update, with no force flags:

```bash
./scripts/update.sh --dev --libint-max-am 5_4_4
```

`doctor.sh` reports the installed spec, and drift shows up
as `BUILD-SETTING DRIFT (max_am=5_4_4 → max_am=5_4_3)`
rather than as a phantom change to `build_libint.sh`. The
same flag is on `install.sh` and `update_native_deps.sh`;
`VIBEQC_LIBINT_MAX_AM` is the underlying environment
variable if you need to set it once for a shell.

An install tree built before this option landed carries no
marker. It is assumed to match the default, which is correct
for anything built from a 2026-08-06-or-later checkout; if
you need certainty, `rm -rf third_party/libint/install` and
let it rebuild.

### Wiped shared library (install tree present, `.so` gone)

**Symptom**: `import vibeqc` fails with
`ImportError: libint2.so: cannot open shared object file: No such file or
directory` (or `libxc.so` / `libsymspg.so` / …), and
`ldd .../_vibeqc_core*.so` shows the dep as `=> not found`, even though
`third_party/<dep>/install/` still exists with its headers and
`lib/cmake/` config. The shared library itself was deleted out from under
a built checkout (seen on the fleet when a managed venv's
`third_party/libint/install/lib/libint2.so*` was wiped).

This used to be a silent dead-end: git was current (no drift), the build
stamp said "built", and `build_<dep>.sh` short-circuits on the CMake
config file, so a normal `update.sh` kept skipping the rebuild and the
only escape was the force flag.

**Fix**: it now self-heals. A normal update detects the missing library
and rebuilds just that dep:

```sh
./scripts/update.sh --dev          # or: vq admin update <env>
```

You'll see a line naming the missing artifact, e.g.
`native lib libint2 missing for libint … → forcing rebuild of libint` (via
`setup_native_deps.sh`) or a `MISSING LIBRARY` row in the build-stamp drift
table (via `update.sh` → `update_native_deps.sh`). `./scripts/doctor.sh`
reports the same `MISSING LIBRARY` status without rebuilding. The
`--rebuild-native-deps` force flag is still available but no longer
required for this case.

### `OSError: cannot load library 'libxc.so.7'`

A vendored dep's install/ tree was wiped (or never built). Re-run:

```sh
./scripts/update.sh --dev --rebuild-native-deps
```

### CMake complains about a stale build cache after update

Rare but possible if scikit-build-core's build cache went stale:

```sh
./scripts/update.sh --dev --clean
```

`--clean` removes the native source/build/install trees and Python build caches
under the shared lifecycle and native-build locks, then recreates and verifies
the environment with rollback.

### Tests fail after update

```sh
.venv/bin/python -m pytest tests/
```

If a few tests fail right after a major update, that's a real signal
worth investigating, please open an issue with the banner output and
the pytest tail. **Don't** ignore the failures and run "real"
calculations on the build until the regression is understood.

## Shell aliases for daily workflow

If you activate the venv and run updates many times a day, a few zsh
aliases save typing. Adapt the paths to your install.

```sh
# --- vibeqc helpers (~/.zshrc) ---
alias vibe-up='cd /path/to/vibeqc && source .venv/bin/activate'
alias vibe-up-experimental='cd /path/to/vibeqc-experimental && source .venv/bin/activate'
alias vibe-down='deactivate 2>/dev/null; cd ~'
```

`vibe-up` jumps to the repo and activates the venv; `vibe-down`
deactivates and returns home. `vibe-up-experimental` activates a
separate venv tracking an experimental feature branch (e.g.
`feature/v0.7-pyscf-pbc-parity` while the periodic-SCF bug-fix work
is in flight), see the next section for what an experimental tree
is for.

### Maintaining dev + release (+ experimental) side-by-side

If you keep two checkouts, one tracking `main` for development, one
tracking `release` for production runs, the function below pulls
both, reinstalls the Python package, and verifies each tree by
importing the freshly-installed package.

The function also knows about an optional **`experimental`** tree:
a third checkout tracking a feature branch you want to keep current
without disrupting your day-to-day dev / release setup. Typical
uses: previewing in-flight bug-fix branches (e.g. the
`feature/v0.7-pyscf-pbc-parity` periodic-SCF work), evaluating a
breaking-change branch before it merges to `main`, or running
side-by-side comparisons against the dev tree on the same data.
The experimental tree is **opt-in**, if `/path/to/vibeqc-experimental`
doesn't exist, `vibe-update` silently skips it.

```sh
vibe-update() {
  local target="${1:-all}"
  local errors=0
  local repo  # NOT 'path' -- see footgun note below

  case "$target" in
    dev|release|experimental|all|both) ;;
    *) echo "vibe-update: unknown target '$target' -- try dev / release / experimental / all"
       return 2 ;;
  esac

  local GIT
  for candidate in /opt/homebrew/bin/git /usr/local/bin/git /usr/bin/git; do
    [[ -x "$candidate" ]] && { GIT="$candidate"; break; }
  done
  [[ -n "$GIT" ]] || { echo "vibe-update: no git found"; return 1; }

  local PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  export PATH  # zsh's `local PATH` does not auto-export to subprocesses

  for tree in dev release experimental; do
    # Selection rules:
    #   target=all          → every configured tree
    #   target=both         → dev + release (back-compat alias for the
    #                         pre-experimental default)
    #   target=<tree-name>  → just that one
    case "$target" in
      all|"$tree") ;;
      both) [[ "$tree" == experimental ]] && continue ;;
      *) continue ;;
    esac

    case "$tree" in
      dev)          repo=/path/to/vibeqc-dev ;;
      release)      repo=/path/to/vibeqc-release ;;
      experimental) repo=/path/to/vibeqc-experimental ;;
    esac

    # Experimental is opt-in: if the directory doesn't exist and the
    # user didn't ask for it explicitly, skip silently rather than
    # erroring out -- most users won't have an experimental checkout.
    if [[ ! -d "$repo" ]]; then
      [[ "$tree" == experimental && "$target" != experimental ]] && continue
      echo; echo "=== $tree ($repo) ==="
      echo "  ✗ no checkout at $repo -- skipping"; ((errors++))
      continue
    fi

    echo; echo "=== Updating $tree ($repo) ==="
    [[ -d "$repo/.venv" ]] || { echo "  ✗ no .venv at $repo -- skipping"; ((errors++)); continue; }
    pushd "$repo" >/dev/null || { ((errors++)); continue; }

    "$GIT" fetch origin && "$GIT" pull --ff-only \
      || { echo "  ✗ pull failed"; ((errors++)); popd >/dev/null; continue; }

    # Keep pip itself current -- without this, every install prints a
    # "[notice] A new release of pip is available" pestering the user
    # with the `python -m pip install --upgrade pip` command they could
    # run to silence it. Doing it here suppresses the notice and keeps
    # the venv aligned with whatever pip version `pip install -e` wants.
    "$repo/.venv/bin/python" -m pip install --quiet --upgrade pip \
      || { echo "  ✗ pip self-upgrade failed"; ((errors++)); popd >/dev/null; continue; }

    "$repo/.venv/bin/pip" install -e "$repo" --no-deps --quiet \
      || { echo "  ✗ pip install failed"; ((errors++)); popd >/dev/null; continue; }

    "$repo/.venv/bin/python" -c \
      "import vibeqc; print(f'  ✓ vibe-qc {vibeqc.__version__} from {vibeqc.__file__}')"

    popd >/dev/null
  done

  echo
  (( errors > 0 )) && { echo "vibe-update: $errors tree(s) failed"; return 1; }
  echo "vibe-update: done"
}
```

Usage:
- `vibe-update`, every configured tree (dev + release, plus
  experimental if you've checked one out).
- `vibe-update dev` / `vibe-update release` / `vibe-update experimental`
  - that one tree only.
- `vibe-update both`, dev + release only, skipping experimental.
  Back-compat alias from the pre-experimental version of this
  function.

#### Setting up the experimental tree

```sh
git clone https://github.com/vibe-qc/vibe-qc.git /path/to/vibeqc-experimental
cd /path/to/vibeqc-experimental
git checkout feature/v0.7-pyscf-pbc-parity     # or whichever branch
./scripts/setup_native_deps.sh                  # if it differs from your dev tree
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

After that, `vibe-update` (no args) will keep all three trees current
on every invocation; `vibe-up-experimental` activates its venv when
you want to actually use it.

This is a lighter-weight cousin of `./scripts/update.sh --dev`, it skips
the native-deps rebuild and the banner print, so it's only safe when
no vendored library version bumped between pulls. For release-day
updates or after a known native-dep bump, prefer `update.sh`.

### Two zsh footguns baked into the function above

Both bit during the original write-up; documenting them here so the
next reader doesn't rediscover them the hard way.

**Don't name the variable `path`.** zsh ties the lowercase `path`
array to the uppercase `PATH` scalar via `typeset -T`. Assigning
`path=/some/dir` silently overwrites `$PATH`, after which git can no
longer find `ssh` and the pull fails with
`error: cannot run ssh: No such file or directory`. The function uses
`repo` for this reason.

**`local PATH=...` needs an explicit `export PATH`.** Inside a zsh
function, `local` declares but does not re-export the variable to
forked subprocesses. Without `export`, the localized PATH stays
inside the shell and git's child processes (in particular `ssh`)
don't see it. Same symptom as the `path`/`PATH` tie above, different
cause.

Bash users can adapt the same function, `local` semantics differ
slightly, but the `path`/`PATH` collision is a zsh-only issue.

## Long-running calculations across an update

If you have a long calculation in flight and want to update vibe-qc
without disturbing it, **don't**: in-flight `python` processes hold
references to the loaded `_vibeqc_core.so`, so an update mid-flight
doesn't break the running job, but starting a new calculation in
the same shell will pick up the new code immediately. Cleanest is:

1. Wait for the running job to finish (or pause it via `Ctrl-Z` and
   `bg` if you must).
2. Update.
3. Start subsequent jobs.

For background-run patterns (`tmux` + `tee`), see the
[running guide](running.md).

## Read-only health check

If you only want to know "is my install OK?" without triggering
anything, run:

```sh
./scripts/doctor.sh
```

It reports on:

- working-tree state (clean / dirty, current branch + SHA);
- which `third_party/<dep>/install/` trees exist;
- build-stamp drift against the current `build_*.sh` files
  (the same check `update.sh` runs automatically);
- venv health, broken python, stale pip shebang, hybrid
  `pyvenv.cfg` / runtime-python mismatch;
- the banner (which library versions are actually linked).

No builds. No installs. No git operations. Exit code is 0 if every
check passes, 1 if anything emitted a warning, so it's scriptable:

```sh
if ./scripts/doctor.sh >/dev/null; then
    echo "install OK"
fi
```

## Where to go next

- [Quickstart](quickstart.md), confirm the new install with a
  smoke-test calculation.
- [Release process](release_process.md), the upstream side of the
  release flow that drives what you're updating to.
- [Changelog](changelog.md), what shipped in the release you just
  installed.
