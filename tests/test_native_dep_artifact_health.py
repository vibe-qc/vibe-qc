"""Native-dep *artifact* health: a wiped ``.so`` forces a rebuild.

Regression guard for the 2026-06-13 compute-study/compute-medium fleet failure. A managed
``vibeqc-dev`` venv whose ``third_party/libint/install/lib/libint2.so`` had
been wiped (so ``import vibeqc`` throws ``ImportError: libint2.so: cannot
open shared object file``) did **not** self-heal through a normal
``vq admin update``: the git tree was current (no drift), the build stamp
said "built", and ``build_libint.sh`` short-circuits on the surviving CMake
*config* file — so every layer skipped the rebuild and the venv stayed
broken until an operator knew the magic ``update.sh --rebuild-native-deps``
force flag.

The fix adds an artifact-existence check (``scripts/_native_stamp.sh`` ::
``_vibeqc_dep_artifact_present``) to the drift/rebuild decision so a wiped
library is treated like a missing install:

* ``vibeqc_stamp_drifted_deps`` emits the dep — this is the list
  ``scripts/update_native_deps.sh`` rebuilds, i.e. the targeted rebuild a
  normal ``vq admin update`` performs once drift is detected.
* ``vibeqc_stamp_check`` returns drift (rc 1) and names the missing
  library — this is the gate ``update.sh`` uses to *trigger* that rebuild,
  and the line ``update.sh --dry-run`` / ``doctor.sh`` now print honestly.

These tests drive the *real* shipped ``scripts/_native_stamp.sh`` through a
throwaway tree (mirroring ``test_release_version_guard.py``'s style of
pinning a shell contract), so the decision logic is locked regardless of
the collect-only CI. The actual compile is the existing ``build_*.sh``
path — too slow to run here — but flipping these decisions IS the
mechanism by which a normal update rebuilds the wiped dep.

Update this test in lockstep with the artifact helpers in
``scripts/_native_stamp.sh`` if the per-dep library names or the
drift/check contract change.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_STAMP = REPO_ROOT / "scripts" / "_native_stamp.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not NATIVE_STAMP.exists(),
    reason="bash not available or scripts/_native_stamp.sh missing",
)

# Path of libint's primary shared library inside a built install tree,
# relative to the repo root — the file the repro wipes.
_LIBINT_SO = "third_party/libint/install/lib/libint2.so"

# Reviewed from the v0.15.121 release tree
# (SHA 4f7173ae60741ad35e7b352afbf62f0174c75c13)
# with the shipped v1 rule: sha256 of every non-comment, non-blank builder
# line. The v2 marker comments must leave these exact hashes intact so fleet
# stamps written by v0.15.121 migrate for dependencies whose numerical
# recipes have not changed. Libint and libecpint explicitly require rebuilds.
_V015121_LEGACY_RECIPE_HASHES = (('scripts/build_openblas.sh',
  '821eaeb47a6bc6f56c6f91fdb8fbfeecbbf83ad5d2eb99e1d1ceda1be208e723'),)

# f8fcea27f added unmarked executable pruning code. Under the shipped
# policy, every such command is recipe-relevant until explicitly reviewed
# as lifecycle-only. Preserve historical hashes and require a rebuild; do
# not silently repin compatibility aliases to the new recipes.
_V015121_CHANGED_RECIPE_HASHES = (('scripts/build_libxc.sh',
  'dae25a2599bb30ca7e0d236afcd61901cd252ca85423be45561044f848db6769'),
 ('scripts/build_spglib.sh',
  '0076c812f34e0ac8df4b784863e3721bbd4b8a68ed2230da78ebe08faf612edb'),
 ('scripts/build_fftw.sh',
  '2e2adb6a3ecc40e178b306db6f1a5aecfec6bec0e27d29b5605653a6abfd573f'))

# Exact pre-unified-lifecycle libint builder provenance, produced by commit
# 8d3443c65f0c2480a0688d928a133da48488c815. It used to back a reviewed
# compatibility alias in scripts/_native_stamp.sh that migrated such stamps
# without recompiling, on the premise that only lifecycle lines had changed.
# The 2026-08-06 max_am change broke that premise, the alias was retired, and
# this hash is now kept to prove the retirement holds.
_PRE_UNIFIED_LIBINT_V1_HASH = (
    "f79c3c9fbd8949f066f4ea84a5aaf6acfc0401bcfed6e07bbad160a4d2c56016"
)
# libint's v0.15.121 fleet stamp hash. Retired from the migrate-without-
# rebuild tuple on 2026-08-06: the max_am change made build_libint.sh differ
# in build-relevant content, so this fingerprint now correctly reports drift.
_V015121_LIBINT_LEGACY_HASH = (
    "4ffe973a425c59eb7cf3eb730db26fb64d4bb392db6715bd36fff4ca62c9dc24"
)


def _bash(tree: Path, snippet: str) -> subprocess.CompletedProcess:
    """Run ``snippet`` with CWD = ``tree`` and the real _native_stamp.sh
    sourced (so the test pins the as-shipped helper, not a copy)."""
    script = (
        f"cd {shlex.quote(str(tree))}\n. {shlex.quote(str(NATIVE_STAMP))}\n{snippet}\n"
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def _recipe_hash(tree: Path, function: str = "_vibeqc_recipe_hash") -> str:
    result = _bash(tree, f"{function} scripts/build_libint.sh")
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write_libint_legacy_stamp(tree: Path, recipe_hash: str) -> None:
    stamp = tree / "third_party" / ".build-stamp"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        f"libint v2.13.1 {recipe_hash}\n",
        encoding="utf-8",
    )


def _relabel_stand_in_builder(
    tree: Path, dep: str, version_var: str, version: str
) -> None:
    """Re-point ``built_tree``'s stand-in builder at a different dependency.

    ``built_tree`` is libint-shaped, but libint is the one dep whose recipe
    fingerprint depends on state outside the builder file. Tests of the
    generic stamp mechanics use this to run on a dep without that property.
    """
    soname = {"libxc": "libxc", "spglib": "libsymspg", "fftw": "libfftw3"}[dep]
    (tree / "scripts" / f"build_{dep}.sh").write_text(
        f'{version_var}="{version}"\n'
        "# pretend configure flags — body is what the recipe hash pins\n"
        "echo build\n",
        encoding="utf-8",
    )
    (tree / "scripts" / "build_libint.sh").unlink()
    shutil.rmtree(tree / "third_party" / "libint")
    libdir = tree / "third_party" / dep / "install" / "lib"
    libdir.mkdir(parents=True)
    (libdir / f"{soname}.so").write_text("", encoding="utf-8")
    res = _bash(tree, "vibeqc_stamp_write >/dev/null")
    assert res.returncode == 0, res.stderr


@pytest.fixture
def built_tree(tmp_path: Path) -> Path:
    """A minimal 'already built' libint install: the CMake config file and
    the ``.so`` both present, plus a build stamp whose pinned version +
    recipe hash match ``scripts/build_libint.sh`` — so the ONLY thing that
    can flag libint for rebuild is the artifact check, not recipe drift."""
    (tmp_path / "scripts").mkdir()
    # A stand-in build_libint.sh carrying a parseable *_VERSION line; the
    # stamp helpers hash this file's build-relevant content.
    (tmp_path / "scripts" / "build_libint.sh").write_text(
        'LIBINT_VERSION="v2.13.1"\n'
        "# pretend configure flags — body is what the recipe hash pins\n"
        "echo build\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "third_party" / "libint" / "install" / "lib" / "cmake" / "libint2"
    cfg.mkdir(parents=True)
    (cfg / "libint2-config.cmake").write_text("", encoding="utf-8")
    (tmp_path / _LIBINT_SO).write_text("", encoding="utf-8")

    # Write the stamp from the present tree (no drift baseline).
    res = _bash(tmp_path, "vibeqc_stamp_write >/dev/null")
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "third_party" / ".build-stamp").exists()
    return tmp_path


@pytest.fixture
def pre_unified_libint_tree(tmp_path: Path) -> Path:
    """Current builder plus an installed artifact carrying the exact v1
    fingerprint written by the reviewed pre-unified-lifecycle source."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(
        REPO_ROOT / "scripts" / "build_libint.sh",
        scripts / "build_libint.sh",
    )
    libint_so = tmp_path / _LIBINT_SO
    libint_so.parent.mkdir(parents=True)
    libint_so.write_text("", encoding="utf-8")
    _write_libint_legacy_stamp(tmp_path, _PRE_UNIFIED_LIBINT_V1_HASH)
    return tmp_path


# ---------------------------------------------------------------------------
# Core fix: the drift/rebuild decision flips when the library is wiped.
# ---------------------------------------------------------------------------


def test_drifted_deps_empty_when_library_present(built_tree: Path):
    res = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == [], (
        "a fully-built, stamp-matching tree must report nothing drifted:\n"
        f"{res.stdout!r}"
    )


def test_drifted_deps_flags_wiped_library(built_tree: Path):
    """The list scripts/update_native_deps.sh rebuilds must include libint
    once its .so is gone, even though git/stamp/CMake-config are unchanged."""
    (built_tree / _LIBINT_SO).unlink()
    res = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["libint"], (
        f"wiped libint2.so must flag libint for rebuild:\n{res.stdout!r}"
    )


def test_stamp_check_present_is_clean(built_tree: Path):
    res = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=0" in res.stdout, res.stdout + res.stderr


def test_stamp_check_reports_and_returns_drift_on_wiped_library(built_tree: Path):
    """update.sh's gate: stamp_check must return drift (rc 1) AND name the
    missing artifact so the rebuild fires and the report is honest."""
    (built_tree / _LIBINT_SO).unlink()
    res = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in res.stdout, res.stdout + res.stderr
    assert "MISSING LIBRARY" in res.stdout, res.stdout
    assert "libint" in res.stdout, res.stdout


def test_stamp_writer_omits_dependencies_that_are_not_installed(
    built_tree: Path,
):
    stamp = (built_tree / "third_party" / ".build-stamp").read_text(encoding="utf-8")
    data_lines = [line for line in stamp.splitlines() if not line.startswith("#")]
    assert len(data_lines) == 1
    fields = data_lines[0].split()
    assert fields[0] == "libint"
    assert len(fields) == 4
    assert fields[3].startswith("v2:")


def test_failed_stamp_write_preserves_previous_stamp(built_tree: Path):
    stamp_path = built_tree / "third_party" / ".build-stamp"
    previous = stamp_path.read_bytes()
    (built_tree / _LIBINT_SO).unlink()
    res = _bash(built_tree, "vibeqc_stamp_write")
    assert res.returncode != 0
    assert "refusing to stamp libint" in res.stderr
    assert stamp_path.read_bytes() == previous


def test_recipe_hash_failure_preserves_previous_stamp(built_tree: Path):
    stamp_path = built_tree / "third_party" / ".build-stamp"
    previous = stamp_path.read_bytes()
    res = _bash(
        built_tree,
        "_vibeqc_recipe_hash() { return 1; }; vibeqc_stamp_write",
    )
    assert res.returncode != 0
    assert stamp_path.read_bytes() == previous
    assert not list((built_tree / "third_party").glob(".build-stamp.tmp.*"))


# ---------------------------------------------------------------------------
# Recipe identity: lifecycle locking is not a native rebuild trigger.
# ---------------------------------------------------------------------------


# scripts/build_libint.sh is deliberately absent from the tuple above. The
# 2026-08-06 max_am change (deriv-2 tier 3 -> 4, plus --libint-max-am) altered
# its build-relevant content, so a v0.15.121 fleet stamp for libint no longer
# describes what the current recipe produces and MUST rebuild. That is asserted
# directly by test_v015121_libint_fleet_stamp_now_requires_a_rebuild below.
@pytest.mark.parametrize("builder,expected", _V015121_LEGACY_RECIPE_HASHES)
def test_v015121_fleet_stamp_migrates_without_rebuild(
    builder: str, expected: str
):
    hashed = _bash(
        REPO_ROOT,
        f"_vibeqc_legacy_recipe_hash {shlex.quote(builder)}",
    )
    assert hashed.returncode == 0, hashed.stderr
    assert hashed.stdout.strip() == expected

    matched = _bash(
        REPO_ROOT,
        (
            f"_vibeqc_stamp_recipe_matches {shlex.quote(builder)} "
            f"{shlex.quote(expected)} ''; echo RC=$?"
        ),
    )
    assert matched.returncode == 0, matched.stderr
    assert "RC=0" in matched.stdout


@pytest.mark.parametrize("builder,historical", _V015121_CHANGED_RECIPE_HASHES)
@pytest.mark.parametrize("stamp_format", ["legacy", "v2"])
def test_unmarked_builder_changes_require_recipe_refresh(builder, historical, stamp_format):
    current = _bash(REPO_ROOT, f"_vibeqc_recipe_hash {shlex.quote(builder)}")
    assert current.returncode == 0, current.stderr
    assert current.stdout.strip() != historical
    stamped_v2 = f"v2:{historical}" if stamp_format == "v2" else ""
    matched = _bash(REPO_ROOT, (
        f"_vibeqc_stamp_recipe_matches {shlex.quote(builder)} "
        f"{historical} {shlex.quote(stamped_v2)}; echo RC=$?"
    ))
    assert "RC=1" in matched.stdout, matched.stdout + matched.stderr


def test_v015121_libint_fleet_stamp_now_requires_a_rebuild(built_tree: Path):
    """libint's v0.15.121 fleet stamp must NOT migrate silently any more.

    Until 2026-08-06 the libint builder had changed only in lifecycle code,
    so its fleet stamp was migrated without recompiling. The max_am change
    (deriv-2 tier 3 -> 4, plus --libint-max-am) is a real recipe change:
    those trees were compiled at max_am 5;4;3 and their analytic Hessians
    cannot handle a g-function basis. Accepting them would be exactly the
    silent stale-install failure this module guards.
    """
    stamp = built_tree / "third_party" / ".build-stamp"
    stamp.write_text(
        f"libint v2.13.1 {_V015121_LIBINT_LEGACY_HASH}\n", encoding="utf-8"
    )
    drifted = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"], drifted.stdout

    checked = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in checked.stdout, checked.stdout + checked.stderr


@pytest.mark.parametrize("stamp_format", ["legacy", "v2"])
def test_unpatched_libecpint_stamp_requires_radial_rebuild(tmp_path, stamp_format):
    """The #758 numerical patch cannot inherit the old no-rebuild identity."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(
        REPO_ROOT / "scripts" / "build_libecpint.sh",
        scripts / "build_libecpint.sh",
    )
    libdir = tmp_path / "third_party" / "libecpint" / "install" / "lib"
    libdir.mkdir(parents=True)
    (libdir / "libecpint.so").touch()
    historical = "e63d9d7dff66acb05be18fc99835d69542c3bd02073a50db3328b518fa02a7af"
    suffix = f" v2:{historical}" if stamp_format == "v2" else ""
    (tmp_path / "third_party" / ".build-stamp").write_text(
        f"libecpint v1.0.7 {historical}{suffix}\n", encoding="utf-8"
    )
    drifted = _bash(tmp_path, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libecpint"]
    checked = _bash(tmp_path, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in checked.stdout, checked.stdout + checked.stderr


def test_pre_unified_libint_stamp_now_requires_a_rebuild(
    pre_unified_libint_tree: Path,
):
    """Same for the older pre-unified-lifecycle stamp: the reviewed alias
    that used to migrate it was retired when the recipe genuinely changed,
    so this now reports drift and asks for a rebuild."""
    checked = _bash(
        pre_unified_libint_tree,
        "vibeqc_stamp_check; echo RC=$?",
    )
    assert "RC=1" in checked.stdout, checked.stdout + checked.stderr

    drifted = _bash(pre_unified_libint_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"], drifted.stdout


def test_no_reviewed_legacy_alias_admits_libint(tmp_path: Path):
    """The retired alias must stay retired. Re-pinning it to whatever the
    current hash happens to be would tell a legacy install it matches a
    recipe it was not built from."""
    res = _bash(
        tmp_path,
        "_vibeqc_reviewed_legacy_recipe_matches scripts/build_libint.sh "
        f"{_PRE_UNIFIED_LIBINT_V1_HASH} anything; echo RC=$?",
    )
    assert "RC=1" in res.stdout, res.stdout


@pytest.mark.parametrize(
    "recipe_hash",
    [
        _PRE_UNIFIED_LIBINT_V1_HASH[:8],
        _PRE_UNIFIED_LIBINT_V1_HASH[:-1],
        _PRE_UNIFIED_LIBINT_V1_HASH[:-1] + "0",
    ],
)
def test_pre_unified_libint_alias_requires_exact_full_sha256(
    recipe_hash: str, pre_unified_libint_tree: Path
):
    _write_libint_legacy_stamp(pre_unified_libint_tree, recipe_hash)
    drifted = _bash(pre_unified_libint_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            'LIBINT_COMMIT_SHA="5fb07b4862f219749c51d59ee08073d20a56d506"',
            'LIBINT_COMMIT_SHA="0000000000000000000000000000000000000001"',
            id="source-pin",
        ),
        pytest.param(
            "-DLIBINT2_ENABLE_ERI3=2",
            "-DLIBINT2_ENABLE_ERI3=1",
            id="cmake-option",
        ),
        pytest.param(
            'cmake --build "$BUILD_DIR" --target install',
            'cmake --build "$BUILD_DIR" --target all',
            id="unmarked-build-command",
        ),
    ],
)
def test_pre_unified_libint_alias_rejects_real_recipe_drift(
    old: str,
    new: str,
    pre_unified_libint_tree: Path,
):
    recipe = pre_unified_libint_tree / "scripts" / "build_libint.sh"
    source = recipe.read_text(encoding="utf-8")
    assert source.count(old) == 1
    recipe.write_text(source.replace(old, new), encoding="utf-8")

    drifted = _bash(pre_unified_libint_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"]

    checked = _bash(
        pre_unified_libint_tree,
        "vibeqc_stamp_check; echo RC=$?",
    )
    assert "RC=1" in checked.stdout
    assert "BUILD-FLAGS DRIFT" in checked.stdout


def test_pre_unified_libint_alias_requires_valid_current_markers(
    pre_unified_libint_tree: Path,
):
    recipe = pre_unified_libint_tree / "scripts" / "build_libint.sh"
    source = recipe.read_text(encoding="utf-8")
    end_marker = "# vibeqc-recipe-hash: lifecycle-only-end\n"
    # Three balanced pairs since 2026-08-13: the build lock plus the two
    # quiet-build-logging blocks around `cmake --build`. Dropping the LAST
    # end marker leaves its begin unmatched, which is the fail-closed shape
    # this test pins ("without a matching end").
    assert source.count(end_marker) == 3
    head, _, tail = source.rpartition(end_marker)
    recipe.write_text(head + tail, encoding="utf-8")

    drifted = _bash(pre_unified_libint_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0
    assert drifted.stdout.split() == ["libint"]
    assert "without a matching end" in drifted.stderr


def test_recipe_hash_ignores_only_explicit_lifecycle_lock_block(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    recipe = scripts / "build_libint.sh"
    recipe.write_text(
        'LIBINT_VERSION="v2.13.1"\n'
        'LIBINT_COMMIT_SHA="source-a"\n'
        "# vibeqc-recipe-hash: lifecycle-only-begin\n"
        '. "$SCRIPT_DIR/_build_lock.sh"\n'
        "vibeqc_acquire_build_lock\n"
        "# vibeqc-recipe-hash: lifecycle-only-end\n"
        'CC="${CC:-cc}"\n'
        "cmake -DLIBINT2_MAX_AM=6\n",
        encoding="utf-8",
    )
    v2_before = _recipe_hash(tmp_path)
    legacy_before = _recipe_hash(tmp_path, "_vibeqc_legacy_recipe_hash")

    source = recipe.read_text(encoding="utf-8")
    recipe.write_text(
        source.replace(
            '. "$SCRIPT_DIR/_build_lock.sh"\n'
            "vibeqc_acquire_build_lock\n",
            '. "$SCRIPT_DIR/_new_lifecycle_lock.sh"\n'
            'vibeqc_acquire_build_lock --wait "$LOCK_WAIT"\n',
        ),
        encoding="utf-8",
    )

    assert _recipe_hash(tmp_path) == v2_before
    assert _recipe_hash(tmp_path, "_vibeqc_legacy_recipe_hash") != legacy_before


@pytest.mark.parametrize(
    "old,new",
    [
        ('LIBINT_VERSION="v2.13.1"', 'LIBINT_VERSION="v2.14.0"'),
        ('LIBINT_COMMIT_SHA="source-a"', 'LIBINT_COMMIT_SHA="source-b"'),
        ('CC="${CC:-cc}"', 'CC="${CC:-clang}"'),
        ("-DLIBINT2_MAX_AM=6", "-DLIBINT2_MAX_AM=7"),
    ],
)
def test_recipe_hash_keeps_real_build_inputs(old: str, new: str, tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    recipe = scripts / "build_libint.sh"
    recipe.write_text(
        'LIBINT_VERSION="v2.13.1"\n'
        'LIBINT_COMMIT_SHA="source-a"\n'
        "# vibeqc-recipe-hash: lifecycle-only-begin\n"
        '. "$SCRIPT_DIR/_build_lock.sh"\n'
        "vibeqc_acquire_build_lock\n"
        "# vibeqc-recipe-hash: lifecycle-only-end\n"
        'CC="${CC:-cc}"\n'
        "cmake -DLIBINT2_MAX_AM=6\n",
        encoding="utf-8",
    )
    before = _recipe_hash(tmp_path)
    recipe.write_text(
        recipe.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    assert _recipe_hash(tmp_path) != before


@pytest.mark.parametrize(
    "markers",
    [
        "# vibeqc-recipe-hash: lifecycle-only-begin\n",
        "# vibeqc-recipe-hash: lifecycle-only-end\n",
        (
            "# vibeqc-recipe-hash: lifecycle-only-begin\n"
            "# vibeqc-recipe-hash: lifecycle-only-begin\n"
            "# vibeqc-recipe-hash: lifecycle-only-end\n"
        ),
    ],
)
def test_recipe_hash_rejects_malformed_lifecycle_markers(
    markers: str, tmp_path: Path
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_libint.sh").write_text(
        'LIBINT_VERSION="v2.13.1"\n' + markers + "echo build\n",
        encoding="utf-8",
    )
    result = _bash(tmp_path, "_vibeqc_recipe_hash scripts/build_libint.sh")
    assert result.returncode != 0
    assert "_vibeqc_recipe_hash:" in result.stderr


@pytest.mark.parametrize("legacy_stamp", [False, True])
def test_malformed_recipe_markers_fail_closed_as_drift(
    legacy_stamp: bool, built_tree: Path
):
    if legacy_stamp:
        stamp_path = built_tree / "third_party" / ".build-stamp"
        lines = stamp_path.read_text(encoding="utf-8").splitlines()
        stamp_path.write_text(
            "\n".join(
                " ".join(line.split()[:3])
                if line and not line.startswith("#")
                else line
                for line in lines
            )
            + "\n",
            encoding="utf-8",
        )
    recipe = built_tree / "scripts" / "build_libint.sh"
    recipe.write_text(
        recipe.read_text(encoding="utf-8")
        + "# vibeqc-recipe-hash: lifecycle-only-begin\n",
        encoding="utf-8",
    )
    drifted = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0
    assert drifted.stdout.split() == ["libint"]

    checked = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in checked.stdout
    assert "BUILD-FLAGS DRIFT" in checked.stdout


def test_valid_v2_stamp_drifts_on_unmarked_recipe_change(built_tree: Path):
    recipe = built_tree / "scripts" / "build_libint.sh"
    recipe.write_text(
        recipe.read_text(encoding="utf-8") + "cmake -DLIBINT2_MAX_AM=7\n",
        encoding="utf-8",
    )
    drifted = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"]

    checked = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in checked.stdout
    assert "BUILD-FLAGS DRIFT" in checked.stdout


@pytest.mark.parametrize(
    "fourth_field",
    ["v2:", "v2:not-a-sha256", "v3:unknown-schema", "unknown"],
)
def test_malformed_or_unknown_stamp_fingerprint_drifts(
    fourth_field: str, built_tree: Path
):
    stamp_path = built_tree / "third_party" / ".build-stamp"
    lines = stamp_path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        if line and not line.startswith("#"):
            fields = line.split()
            fields[3] = fourth_field
            line = " ".join(fields)
        rewritten.append(line)
    stamp_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    drifted = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0, drifted.stderr
    assert drifted.stdout.split() == ["libint"]

    checked = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=1" in checked.stdout
    assert "BUILD-FLAGS DRIFT" in checked.stdout


def test_three_field_legacy_stamp_upgrades_without_rebuild(built_tree: Path):
    # Exercised on a dep with no out-of-file build state. libint cannot carry
    # this case any more: its recipe fingerprint includes VIBEQC_LIBINT_MAX_AM
    # (scripts/_libint_max_am.sh), which by construction is absent from the
    # all-executable-lines v1 hash, so the pre-locking v1 == current v2
    # shortcut cannot apply to it. Renaming the stand-in builder keeps this
    # testing the generic legacy-upgrade mechanism rather than libint's.
    _relabel_stand_in_builder(built_tree, "libxc", "LIBXC_VERSION", "7.0.0")
    stamp_path = built_tree / "third_party" / ".build-stamp"
    lines = stamp_path.read_text(encoding="utf-8").splitlines()
    legacy_lines = [
        " ".join(line.split()[:3]) if line and not line.startswith("#") else line
        for line in lines
    ]
    stamp_path.write_text("\n".join(legacy_lines) + "\n", encoding="utf-8")

    # Simulate the v0.15.119 -> v0.15.121 transition: the legacy stamp was
    # written before the builder acquired its own lifecycle lock. Adding only
    # an explicitly excluded lock block must not turn that old binary stale.
    recipe = built_tree / "scripts" / "build_libxc.sh"
    original = recipe.read_text(encoding="utf-8")
    recipe.write_text(
        original.replace(
            "echo build\n",
            "# vibeqc-recipe-hash: lifecycle-only-begin\n"
            '. "$SCRIPT_DIR/_build_lock.sh"\n'
            "vibeqc_acquire_build_lock\n"
            "# vibeqc-recipe-hash: lifecycle-only-end\n"
            "echo build\n",
        ),
        encoding="utf-8",
    )

    checked = _bash(built_tree, "vibeqc_stamp_check; echo RC=$?")
    assert "RC=0" in checked.stdout
    drifted = _bash(built_tree, "vibeqc_stamp_drifted_deps")
    assert drifted.returncode == 0
    assert drifted.stdout.split() == []

    rewritten = _bash(built_tree, "vibeqc_stamp_write >/dev/null")
    assert rewritten.returncode == 0, rewritten.stderr
    data_line = next(
        line
        for line in stamp_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    fields = data_line.split()
    assert len(fields) == 4
    assert fields[3].startswith("v2:")


def test_openblas_stamp_remembers_selection_after_install_tree_is_wiped(
    tmp_path: Path,
):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "build_openblas.sh").write_text(
        'OPENBLAS_VERSION="0.3.33"\necho build\n', encoding="utf-8"
    )
    third_party = tmp_path / "third_party"
    third_party.mkdir()
    (third_party / ".build-stamp").write_text(
        "openblas 0.3.33 placeholder-hash\n", encoding="utf-8"
    )
    res = _bash(tmp_path, "vibeqc_stamp_drifted_deps")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["openblas"]


def test_functions_are_errexit_safe(built_tree: Path):
    """The real callers (update.sh, update_native_deps.sh) run under
    ``set -euo pipefail`` and consume these via command substitution /
    ``|| rc=$?``. A wiped library (helper returns non-zero) must not abort
    them — the artifact check lives in conditional context for exactly this
    reason. Guards the set -e footgun."""
    (built_tree / _LIBINT_SO).unlink()
    snippet = (
        "set -euo pipefail\n"
        'drifted="$(vibeqc_stamp_drifted_deps | tr "\\n" " ")"\n'
        "rc=0; vibeqc_stamp_check >/dev/null 2>&1 || rc=$?\n"
        'echo "drifted=[${drifted}] rc=${rc}"\n'
    )
    res = _bash(built_tree, snippet)
    assert res.returncode == 0, f"errexit aborted the helpers:\n{res.stderr}"
    assert "drifted=[libint ] rc=1" in res.stdout, res.stdout


# ---------------------------------------------------------------------------
# Artifact probe robustness: real-world install layouts + name gotchas.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dep,relpaths,expect_present",
    [
        # Linux bare .so / lib64 + versioned-only .so / macOS bare + versioned .dylib
        ("libint", ["third_party/libint/install/lib/libint2.so"], True),
        ("spglib", ["third_party/spglib/install/lib64/libsymspg.so.2.7.0"], True),
        (
            "libxc",
            [
                "third_party/libxc/install/lib/libxc.dylib",
                "third_party/libxc/install/lib/libxc.15.dylib",
            ],
            True,
        ),
        ("fftw", ["third_party/fftw/install/lib/libfftw3.so"], True),
        ("libecpint", ["third_party/libecpint/install/lib/libecpint.dylib"], True),
        # Wiped: install dir exists but no matching library file.
        ("libint", ["third_party/libint/install/lib/.keep"], False),
        # A sibling library must NOT be mistaken for the primary one
        # (libint2-cxx.{so,dylib} is not libint2).
        ("libint", ["third_party/libint/install/lib/libint2-cxx.dylib"], False),
    ],
)
def test_artifact_present_layouts(tmp_path: Path, dep, relpaths, expect_present):
    for rel in relpaths:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")
    res = _bash(tmp_path, f"_vibeqc_dep_artifact_present {dep}; echo RC=$?")
    assert res.returncode == 0, res.stderr
    want = "RC=0" if expect_present else "RC=1"
    assert want in res.stdout, (
        f"{dep} {relpaths} -> expected {want}, got:\n{res.stdout}"
    )


@pytest.mark.parametrize(
    "dep,expected",
    [
        ("libint", "libint2"),
        ("libxc", "libxc"),
        ("spglib", "libsymspg"),  # NOT libspglib — the easy-to-get-wrong one
        ("fftw", "libfftw3"),
        ("libecpint", "libecpint"),
        ("openblas", "libopenblas"),
    ],
)
def test_lib_basename_mapping(tmp_path: Path, dep, expected):
    res = _bash(tmp_path, f"_vibeqc_dep_lib_basename {dep}")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected


def test_lib_basename_unknown_is_nonzero(tmp_path: Path):
    res = _bash(tmp_path, "_vibeqc_dep_lib_basename nope; echo RC=$?")
    assert "RC=1" in res.stdout, res.stdout
