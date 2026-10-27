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
        f"cd {shlex.quote(str(tree))}\n. {shlex.quote(str(NATIVE_STAMP))}\n"
        # These legacy stand-ins test stamp protocol, not loaded-image admission.
        # Real admission/caller controls below use _fftw_shell without this stub.
        f"vibeqc_verify_cached_libint() {{ :; }}\n{snippet}\n"
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
    soname = {"libxc": "libxc", "spglib": "libsymspg", "fftw": "libfftw3", "openblas": "libopenblas", "libecpint": "libecpint"}[dep]
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
    # The immutable historical row must still be accepted. Admission-only
    # code can now change the v1 text; exact reviewed aliases preserve reuse.
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
    # Admission, lock and logging blocks are balanced. Dropping the LAST
    # end marker leaves its begin unmatched, which is the fail-closed shape
    # this test pins ("without a matching end").
    assert source.count(end_marker) == source.count("# vibeqc-recipe-hash: lifecycle-only-begin\n")
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
    _relabel_stand_in_builder(built_tree, "libecpint", "LIBECPINT_VERSION", "1.0.7")
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
    recipe = built_tree / "scripts" / "build_libecpint.sh"
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


# These controls inspect library metadata only; they never compile or run FFTs.
def _fftw_cache_tree(tmp_path: Path, *, real_library=None) -> Path:
    import os

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for source in (REPO_ROOT / "scripts").glob("*.sh"):
        shutil.copy2(source, scripts / source.name)
    for name in ("attest_fftw_image.py", "attest_native_image.py", "patch_libxc_spin_fractions.py"):
        helper = REPO_ROOT / "scripts" / name
        if helper.exists():
            shutil.copy2(helper, scripts / helper.name)
    # Prevent pressure/lock setup from obscuring the cache contract in fixtures.
    (scripts / "_safe_build_env.sh").write_text(":\n")
    (scripts / "_build_lock.sh").write_text("vibeqc_acquire_build_lock() { :; }\n")
    install = tmp_path / "third_party/fftw/install"
    config = install / "lib/cmake/fftw3/FFTW3Config.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# cache metadata alone is not artifact evidence\n")
    library = install / "lib" / (Path(real_library).name if real_library else "libfftw3.so")
    if real_library:
        shutil.copy2(real_library, library)
        link_name = "libfftw3.dylib" if library.name.endswith(".dylib") else "libfftw3.so"
        if library.name != link_name:
            (library.parent / link_name).symlink_to(library.name)
    else:
        library.write_bytes(b"not a shared library\n")
    (tmp_path / "third_party/.build-stamp").write_text("previous stamp: preserve me\n")
    # If reconciliation reaches a builder, fail cheaply and leave a witness.
    for dep in ("libint", "libxc", "spglib", "libecpint", "openblas"):
        (scripts / f"build_{dep}.sh").write_text(
            f'#!/bin/bash\n{dep.upper()}_VERSION="1"\n'
            'touch "$(dirname "$0")/../BUILD_ATTEMPTED"\nexit 99\n'
        )
        os.chmod(scripts / f"build_{dep}.sh", 0o755)
    return install


def _fftw_shell(tree: Path, command: str):
    import os

    env = dict(os.environ, VIBEQC_SKIP_PREFLIGHT="1", OMP_NUM_THREADS="2")
    guard = tree / ".no-native-tools"
    if guard.is_dir():
        env["PATH"] = str(guard) + os.pathsep + env.get("PATH", "")
        for name, tool in (("CC", "cc"), ("CXX", "c++"), ("FC", "gfortran"),
                           ("CMAKE_MAKE_PROGRAM", "ninja")):
            env[name] = str(guard / tool)
    return subprocess.run(["/bin/bash", "-c", command], cwd=tree, env=env,
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("command", [
    "/bin/bash scripts/build_fftw.sh",
    ". scripts/_native_stamp.sh; vibeqc_stamp_write",
    "/bin/bash scripts/update_native_deps.sh",
    "/bin/bash scripts/setup_native_deps.sh",
])
def test_fftw_unloadable_cache_refused_without_mutation(tmp_path, command):
    install = _fftw_cache_tree(tmp_path)
    stamp = tmp_path / "third_party/.build-stamp"
    before = {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}
    result = _fftw_shell(tmp_path, command)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "identity unavailable" in result.stderr
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()
    assert stamp.read_text() == "previous stamp: preserve me\n"
    assert before == {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}


def test_fftw_cache_admission_is_separate_from_recipe_drift(tmp_path):
    _fftw_cache_tree(tmp_path)
    result = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_stamp_check")
    assert result.returncode == 3, result.stdout + result.stderr


@pytest.mark.parametrize("args", ["--all", "--dep fftw"])
def test_fftw_explicit_rebuild_preview_preserves_invalid_cache(tmp_path, args):
    install = _fftw_cache_tree(tmp_path)
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run {args}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fftw" in result.stdout
    assert install.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


@pytest.mark.parametrize("v2", ["", "v2:1aaac633983c4a69f5304deb3997734c259028e6ce33fe2e90871c12e53717cd"])
def test_fftw_admission_preserves_exact_qualified_recipe(tmp_path, v2):
    _fftw_cache_tree(tmp_path)
    predicate = ('. scripts/_native_stamp.sh; _vibeqc_stamp_recipe_matches '
                 'scripts/build_fftw.sh '
                 '5e4c72c10655584c2dd511080e982a4177c5afaf81633de3df8a4aecd1e61fd2 '
                 + shlex.quote(v2))
    assert _fftw_shell(tmp_path, predicate).returncode == 0
    builder = tmp_path / "scripts/build_fftw.sh"
    builder.write_text(builder.read_text().replace('-DENABLE_OPENMP=ON', '-DENABLE_OPENMP=OFF'))
    assert _fftw_shell(tmp_path, predicate).returncode != 0


def test_fftw_real_image_admission_and_loader_substitution(tmp_path):
    import hashlib
    import json
    import os
    import sys

    real = os.environ.get("VIBEQC_TEST_FFTW_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_FFTW_LIBRARY to an existing qualified library; no build")
    install = _fftw_cache_tree(tmp_path, real_library=real)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_fftw.sh")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[0])
    assert record["source_build_provenance"] == "not_attested"
    artifact = record["artifacts"][0]
    local = install / "lib" / Path(real).name
    assert artifact["requested_path"] == artifact["loaded_path"] == str(local.resolve())
    assert artifact["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert "source/build provenance not attested" in result.stdout
    # Real artifact + exact qualified historical stamps stay on the reuse path.
    for dep in ("libint", "libxc", "spglib", "libecpint", "openblas"):
        (tmp_path / "scripts" / f"build_{dep}.sh").unlink()
    stamp = tmp_path / "third_party/.build-stamp"
    legacy = "fftw 3.3.10 5e4c72c10655584c2dd511080e982a4177c5afaf81633de3df8a4aecd1e61fd2"
    for row in (legacy, legacy + " v2:1aaac633983c4a69f5304deb3997734c259028e6ce33fe2e90871c12e53717cd"):
        stamp.write_text(row + "\n")
        checked = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_stamp_check")
        assert checked.returncode == 0, checked.stderr + checked.stdout
        preview = _fftw_shell(tmp_path, "/bin/bash scripts/update_native_deps.sh --dry-run")
        assert preview.returncode == 0, preview.stderr
        assert "all current" in preview.stdout
        assert stamp.read_text() == row + "\n"
    # The same basename in an override directory must never qualify a fake file.
    local.write_bytes(b"not a loadable image")
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(real).parent), LD_LIBRARY_PATH=str(Path(real).parent))
    helper = tmp_path / "scripts/attest_fftw_image.py"
    for args in ([str(install)], ["--attest", str(local)]):
        negative = subprocess.run([sys.executable, str(helper), *args], env=env,
                                  capture_output=True, text=True, timeout=30)
        assert negative.returncode != 0, negative.stdout


def _libxc_cache_tree(tmp_path, *, real_library=None, config_only=False):
    _fftw_cache_tree(tmp_path)
    shutil.rmtree(tmp_path / "third_party/fftw")
    shutil.copy2(REPO_ROOT / "scripts/build_libxc.sh", tmp_path / "scripts/build_libxc.sh")
    install = tmp_path / "third_party/libxc/install"
    config = install / "lib/cmake/Libxc/LibxcConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# config alone is not loaded artifact identity\n")
    if not config_only:
        library = install / "lib" / (Path(real_library).name if real_library else "libxc.so")
        if real_library:
            shutil.copy2(real_library, library)
            bare = "libxc.dylib" if library.name.endswith(".dylib") else "libxc.so"
            if library.name != bare:
                (library.parent / bare).symlink_to(library.name)
        else:
            library.write_bytes(b"not a shared library\n")
    return install


def test_libxc_spin_fraction_patch_is_idempotent_and_preserves_unknown_sources(tmp_path, monkeypatch):
    import hashlib
    import runpy

    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    patch = runpy.run_path(str(REPO_ROOT / "scripts/patch_libxc_spin_fractions.py"))
    transform = patch["patched_text"]
    old, new = patch["TARGETS"][0][2][0]
    source = "/* MPL notice retained */\n" + (old + "\n") * 5
    digest = hashlib.sha256(source.encode()).hexdigest()
    replacements = ((old, new),)
    fixed = transform(source, digest, replacements)
    assert fixed.startswith("/* MPL notice retained */\n")
    assert fixed.count(new) == 5
    assert transform(fixed, digest, replacements) == fixed
    with pytest.raises(ValueError, match="refusing to overwrite"):
        transform(source + "/* local edit */\n", digest, replacements)

    # A later unknown target must not leave the earlier valid file modified.
    good = tmp_path / "good.c"
    bad = tmp_path / "bad.c"
    good.write_text(source)
    bad.write_text(source + "/* local edit */\n")
    patch_source = patch["patch_source"]
    monkeypatch.setitem(patch_source.__globals__, "TARGETS", (
        ("good.c", digest, replacements), ("bad.c", digest, replacements),
    ))
    with pytest.raises(ValueError, match="bad.c"):
        patch_source(tmp_path)
    assert good.read_text() == source
    assert bad.read_text().endswith("/* local edit */\n")


@pytest.mark.parametrize("config_only", [False, True])
@pytest.mark.parametrize("command", [
    "/bin/bash scripts/build_libxc.sh",
    ". scripts/_native_stamp.sh; vibeqc_stamp_write",
    ". scripts/_native_stamp.sh; vibeqc_stamp_check",
    "/bin/bash scripts/update_native_deps.sh",
    "/bin/bash scripts/setup_native_deps.sh",
])
def test_libxc_invalid_cache_refused_without_mutation(tmp_path, command, config_only):
    install = _libxc_cache_tree(tmp_path, config_only=config_only)
    stamp = tmp_path / "third_party/.build-stamp"
    before = {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}
    result = _fftw_shell(tmp_path, command)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "identity unavailable" in result.stderr
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()
    assert stamp.read_text() == "previous stamp: preserve me\n"
    assert before == {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}


@pytest.mark.parametrize("args", ["--all", "--dep libxc"])
def test_libxc_explicit_rebuild_preview_preserves_invalid_cache(tmp_path, args):
    install = _libxc_cache_tree(tmp_path)
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run {args}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "libxc" in result.stdout
    assert install.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


@pytest.mark.parametrize("v2", ["", "v2:ba791910b05b03d672db9b327dfdb0d11f4bd3b59b63183d980f461bec8c008f"])
def test_libxc_spin_fraction_patch_invalidates_unpatched_recipe(tmp_path, v2):
    _libxc_cache_tree(tmp_path)
    predicate = ('. scripts/_native_stamp.sh; _vibeqc_stamp_recipe_matches '
                 'scripts/build_libxc.sh '
                 '4bae92e0e9cfbb4dce430145fa20b20772657f2995f721db90afc32d2310fc59 '
                 + shlex.quote(v2))
    assert _fftw_shell(tmp_path, predicate).returncode != 0
    builder = tmp_path / "scripts/build_libxc.sh"
    builder.write_text(builder.read_text().replace('-DENABLE_FORTRAN=OFF', '-DENABLE_FORTRAN=ON'))
    assert _fftw_shell(tmp_path, predicate).returncode != 0


def test_libxc_real_image_admission_and_loader_substitution(tmp_path):
    import hashlib
    import json
    import os
    import sys

    real = os.environ.get("VIBEQC_TEST_LIBXC_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_LIBXC_LIBRARY to an existing library; no build")
    install = _libxc_cache_tree(tmp_path, real_library=real)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_libxc.sh")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[0])
    assert record["source_build_provenance"] == "not_attested"
    artifact = record["artifacts"][0]
    local = install / "lib" / Path(real).name
    assert artifact["requested_path"] == artifact["loaded_path"] == str(local.resolve())
    assert artifact["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert "source/build provenance not attested" in result.stdout
    for dep in ("libint", "fftw", "spglib", "libecpint", "openblas"):
        (tmp_path / "scripts" / f"build_{dep}.sh").unlink()
    stamp = tmp_path / "third_party/.build-stamp"
    legacy = "libxc 7.0.0 4bae92e0e9cfbb4dce430145fa20b20772657f2995f721db90afc32d2310fc59"
    for row in (legacy, legacy + " v2:ba791910b05b03d672db9b327dfdb0d11f4bd3b59b63183d980f461bec8c008f"):
        stamp.write_text(row + "\n")
        checked = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_stamp_check")
        assert checked.returncode == 1, checked.stderr + checked.stdout
        preview = _fftw_shell(tmp_path, "/bin/bash scripts/update_native_deps.sh --dry-run")
        assert preview.returncode == 0, preview.stderr
        assert "libxc" in preview.stdout
        assert stamp.read_text() == row + "\n"
    local.write_bytes(b"not a loadable image")
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(real).parent), LD_LIBRARY_PATH=str(Path(real).parent))
    helper = tmp_path / "scripts/attest_native_image.py"
    for args in ([str(install)], ["--attest", str(local)]):
        negative = subprocess.run([sys.executable, str(helper), "libxc", *args], env=env,
                                  capture_output=True, text=True, timeout=30)
        assert negative.returncode != 0, negative.stdout


@pytest.mark.parametrize("selection", ["fftw", "libxc"])
def test_explicit_rebuild_cannot_replace_another_invalid_cache(tmp_path, selection):
    libxc = _libxc_cache_tree(tmp_path)
    fftw = tmp_path / "third_party/fftw/install/lib"
    fftw.mkdir(parents=True)
    (fftw / "libfftw3.so").write_bytes(b"not a shared library")
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run --dep {selection}")
    assert result.returncode == 3, result.stdout + result.stderr
    assert libxc.exists() and fftw.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


def test_libxc_rejects_real_library_without_version_entry_point(tmp_path):
    import os

    wrong_library = os.environ.get("VIBEQC_TEST_FFTW_LIBRARY")
    if not wrong_library:
        pytest.skip("set VIBEQC_TEST_FFTW_LIBRARY for a real wrong-library control")
    install = _libxc_cache_tree(tmp_path)
    fake = install / "lib/libxc.so"
    fake.unlink()
    name = "libxc.dylib" if wrong_library.endswith(".dylib") else "libxc.so"
    shutil.copy2(wrong_library, install / "lib" / name)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_libxc.sh")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "xc_version" in result.stderr
    assert "identity unavailable" in result.stderr


def _spglib_cache_tree(tmp_path, *, real_library=None, config_only=False):
    _libxc_cache_tree(tmp_path, config_only=True)
    shutil.rmtree(tmp_path / "third_party/libxc")
    shutil.copy2(REPO_ROOT / "scripts/build_spglib.sh", tmp_path / "scripts/build_spglib.sh")
    install = tmp_path / "third_party/spglib/install"
    config = install / "lib/cmake/Spglib/SpglibConfig.cmake"
    config.parent.mkdir(parents=True)
    config.write_text("# config alone is not loaded artifact identity\n")
    if not config_only:
        library = install / "lib" / (Path(real_library).name if real_library else "libsymspg.so")
        if real_library:
            shutil.copy2(real_library, library)
            bare = "libsymspg.dylib" if library.name.endswith(".dylib") else "libsymspg.so"
            if library.name != bare:
                (library.parent / bare).symlink_to(library.name)
        else:
            library.write_bytes(b"not a shared library\n")
    return install


@pytest.mark.parametrize("config_only", [False, True])
@pytest.mark.parametrize("command", [
    "/bin/bash scripts/build_spglib.sh",
    ". scripts/_native_stamp.sh; vibeqc_stamp_write",
    ". scripts/_native_stamp.sh; vibeqc_stamp_check",
    "/bin/bash scripts/update_native_deps.sh",
    "/bin/bash scripts/setup_native_deps.sh",
])
def test_spglib_invalid_cache_refused_without_mutation(tmp_path, command, config_only):
    install = _spglib_cache_tree(tmp_path, config_only=config_only)
    stamp = tmp_path / "third_party/.build-stamp"
    before = {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}
    result = _fftw_shell(tmp_path, command)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "identity unavailable" in result.stderr
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()
    assert stamp.read_text() == "previous stamp: preserve me\n"
    assert before == {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}


@pytest.mark.parametrize("args", ["--all", "--dep spglib"])
def test_spglib_explicit_rebuild_preview_preserves_invalid_cache(tmp_path, args):
    install = _spglib_cache_tree(tmp_path)
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run {args}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "spglib" in result.stdout
    assert install.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


@pytest.mark.parametrize("v2", ["", "v2:5d0e3001218aa57c4a7a15a7213fbf276bbae08b60cd2a74baacd1f9d8239179"])
def test_spglib_admission_preserves_exact_qualified_recipe(tmp_path, v2):
    _spglib_cache_tree(tmp_path)
    predicate = ('. scripts/_native_stamp.sh; _vibeqc_stamp_recipe_matches '
                 'scripts/build_spglib.sh '
                 'e07be4baf5d5adee173497d6250badcedb9f9662cc3b525c07fc411b0422bf16 '
                 + shlex.quote(v2))
    assert _fftw_shell(tmp_path, predicate).returncode == 0
    builder = tmp_path / "scripts/build_spglib.sh"
    builder.write_text(builder.read_text().replace('-DSPGLIB_WITH_TESTS=OFF', '-DSPGLIB_WITH_TESTS=ON'))
    assert _fftw_shell(tmp_path, predicate).returncode != 0


def test_spglib_real_image_admission_and_loader_substitution(tmp_path):
    import hashlib
    import json
    import os
    import sys

    real = os.environ.get("VIBEQC_TEST_SPGLIB_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_SPGLIB_LIBRARY to an existing library; no build")
    install = _spglib_cache_tree(tmp_path, real_library=real)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_spglib.sh")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[0])
    assert record["source_build_provenance"] == "not_attested"
    artifact = record["artifacts"][0]
    local = install / "lib" / Path(real).name
    assert artifact["requested_path"] == artifact["loaded_path"] == str(local.resolve())
    assert artifact["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert "source/build provenance not attested" in result.stdout
    for dep in ("libint", "fftw", "libxc", "libecpint", "openblas"):
        (tmp_path / "scripts" / f"build_{dep}.sh").unlink()
    stamp = tmp_path / "third_party/.build-stamp"
    legacy = "spglib v2.7.0 e07be4baf5d5adee173497d6250badcedb9f9662cc3b525c07fc411b0422bf16"
    for row in (legacy, legacy + " v2:5d0e3001218aa57c4a7a15a7213fbf276bbae08b60cd2a74baacd1f9d8239179"):
        stamp.write_text(row + "\n")
        checked = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_stamp_check")
        assert checked.returncode == 0, checked.stderr + checked.stdout
        preview = _fftw_shell(tmp_path, "/bin/bash scripts/update_native_deps.sh --dry-run")
        assert preview.returncode == 0, preview.stderr
        assert "all current" in preview.stdout
        assert stamp.read_text() == row + "\n"
    local.write_bytes(b"not a loadable image")
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(real).parent), LD_LIBRARY_PATH=str(Path(real).parent))
    helper = tmp_path / "scripts/attest_native_image.py"
    for args in ([str(install)], ["--attest", str(local)]):
        negative = subprocess.run([sys.executable, str(helper), "spglib", *args], env=env,
                                  capture_output=True, text=True, timeout=30)
        assert negative.returncode != 0, negative.stdout




@pytest.mark.parametrize("selection", ["fftw", "libxc", "spglib"])
def test_spglib_mixed_explicit_rebuild_preserves_other_invalid_cache(tmp_path, selection):
    install = _spglib_cache_tree(tmp_path)
    fftw = tmp_path / "third_party/fftw/install/lib"
    fftw.mkdir(parents=True)
    (fftw / "libfftw3.so").write_bytes(b"not a shared library")
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run --dep {selection}")
    assert result.returncode == 3, result.stdout + result.stderr
    assert install.exists() and fftw.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


def test_spglib_rejects_real_library_without_version_entry_point(tmp_path):
    import os

    wrong_library = os.environ.get("VIBEQC_TEST_FFTW_LIBRARY")
    if not wrong_library:
        pytest.skip("set VIBEQC_TEST_FFTW_LIBRARY for a real wrong-library control")
    install = _spglib_cache_tree(tmp_path)
    (install / "lib/libsymspg.so").unlink()
    name = "libsymspg.dylib" if wrong_library.endswith(".dylib") else "libsymspg.so"
    shutil.copy2(wrong_library, install / "lib" / name)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_spglib.sh")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "spg_get_version" in result.stderr
    assert "identity unavailable" in result.stderr


def _openblas_cache_tree(tmp_path, *, real_library=None, header_only=False):
    _spglib_cache_tree(tmp_path, config_only=True)
    shutil.rmtree(tmp_path / "third_party/spglib")
    shutil.copy2(REPO_ROOT / "scripts/build_openblas.sh", tmp_path / "scripts/build_openblas.sh")
    install = tmp_path / "third_party/openblas/install"
    header = install / "include/lapacke.h"
    header.parent.mkdir(parents=True)
    header.write_text("/* A header alone is not loaded backend identity. */\n")
    if not header_only:
        library = install / "lib" / (Path(real_library).name if real_library else "libopenblas.so")
        library.parent.mkdir()
        if real_library:
            shutil.copy2(real_library, library)
            bare = "libopenblas.dylib" if library.name.endswith(".dylib") else "libopenblas.so"
            if library.name != bare:
                (library.parent / bare).symlink_to(library.name)
        else:
            library.write_bytes(b"not a shared library\n")
    return install


@pytest.mark.parametrize("header_only", [False, True])
@pytest.mark.parametrize("command", [
    "/bin/bash scripts/build_openblas.sh",
    ". scripts/_native_stamp.sh; vibeqc_stamp_write",
    ". scripts/_native_stamp.sh; vibeqc_stamp_check",
    "/bin/bash scripts/update_native_deps.sh",
    "/bin/bash scripts/setup_native_deps.sh",
])
def test_openblas_invalid_cache_refused_without_mutation(tmp_path, command, header_only):
    install = _openblas_cache_tree(tmp_path, header_only=header_only)
    stamp = tmp_path / "third_party/.build-stamp"
    before = {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}
    result = _fftw_shell(tmp_path, command)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "identity unavailable" in result.stderr
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()
    assert stamp.read_text() == "previous stamp: preserve me\n"
    assert before == {str(p.relative_to(install)): p.read_bytes() for p in install.rglob("*") if p.is_file()}


@pytest.mark.parametrize("args", ["--all", "--dep openblas"])
def test_openblas_explicit_rebuild_preview_preserves_invalid_cache(tmp_path, args):
    install = _openblas_cache_tree(tmp_path)
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run {args}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "openblas" in result.stdout
    assert install.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


@pytest.mark.parametrize("v2", ["", "v2:51acb78acc510a3c19e9ab0394cd572c220d7fcdc378f868af95b64c7b1027f3"])
def test_openblas_admission_preserves_exact_qualified_recipe(tmp_path, v2):
    _openblas_cache_tree(tmp_path)
    predicate = ('. scripts/_native_stamp.sh; _vibeqc_stamp_recipe_matches '
                 'scripts/build_openblas.sh '
                 '821eaeb47a6bc6f56c6f91fdb8fbfeecbbf83ad5d2eb99e1d1ceda1be208e723 '
                 + shlex.quote(v2))
    assert _fftw_shell(tmp_path, predicate).returncode == 0
    builder = tmp_path / "scripts/build_openblas.sh"
    builder.write_text(builder.read_text().replace('USE_LAPACKE=1', 'USE_LAPACKE=0'))
    assert _fftw_shell(tmp_path, predicate).returncode != 0


def test_openblas_real_image_admission_and_loader_substitution(tmp_path):
    import hashlib
    import json
    import os
    import sys

    real = os.environ.get("VIBEQC_TEST_OPENBLAS_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_OPENBLAS_LIBRARY to an existing library; no build")
    install = _openblas_cache_tree(tmp_path, real_library=real)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_openblas.sh")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.splitlines()[0])
    assert record["source_build_provenance"] == "not_attested"
    artifact = record["artifacts"][0]
    local = install / "lib" / Path(real).name
    assert artifact["requested_path"] == artifact["loaded_path"] == str(local.resolve())
    assert artifact["sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert "source/build provenance not attested" in result.stdout
    for dep in ("libint", "fftw", "libxc", "libecpint", "spglib"):
        (tmp_path / "scripts" / f"build_{dep}.sh").unlink()
    stamp = tmp_path / "third_party/.build-stamp"
    legacy = "openblas 0.3.33 821eaeb47a6bc6f56c6f91fdb8fbfeecbbf83ad5d2eb99e1d1ceda1be208e723"
    for row in (legacy, legacy + " v2:51acb78acc510a3c19e9ab0394cd572c220d7fcdc378f868af95b64c7b1027f3"):
        stamp.write_text(row + "\n")
        checked = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_stamp_check")
        assert checked.returncode == 0, checked.stderr + checked.stdout
        preview = _fftw_shell(tmp_path, "/bin/bash scripts/update_native_deps.sh --dry-run")
        assert preview.returncode == 0, preview.stderr
        assert "all current" in preview.stdout
        assert stamp.read_text() == row + "\n"
    local.chmod(0o644)  # Only the local copy; installed controls may be read-only.
    local.write_bytes(b"not a loadable image")
    env = dict(os.environ, DYLD_LIBRARY_PATH=str(Path(real).parent), LD_LIBRARY_PATH=str(Path(real).parent))
    helper = tmp_path / "scripts/attest_native_image.py"
    for args in ([str(install)], ["--attest", str(local)]):
        negative = subprocess.run([sys.executable, str(helper), "openblas", *args], env=env,
                                  capture_output=True, text=True, timeout=30)
        assert negative.returncode != 0, negative.stdout






@pytest.mark.parametrize("selection", ["fftw", "libxc", "spglib", "openblas"])
def test_openblas_mixed_explicit_rebuild_preserves_other_invalid_cache(tmp_path, selection):
    install = _openblas_cache_tree(tmp_path)
    fftw = tmp_path / "third_party/fftw/install/lib"
    fftw.mkdir(parents=True)
    (fftw / "libfftw3.so").write_bytes(b"not a shared library")
    result = _fftw_shell(tmp_path, f"/bin/bash scripts/update_native_deps.sh --dry-run --dep {selection}")
    assert result.returncode == 3, result.stdout + result.stderr
    assert install.exists() and fftw.exists()
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


def test_openblas_absence_does_not_opt_in(tmp_path):
    _openblas_cache_tree(tmp_path)
    shutil.rmtree(tmp_path / "third_party/openblas")
    checked = _fftw_shell(tmp_path, ". scripts/_native_stamp.sh; vibeqc_verify_cached_openblas; _vibeqc_stamp_pairs")
    assert checked.returncode == 0, checked.stderr
    assert "openblas" not in checked.stdout
    assert not (tmp_path / "third_party/openblas").exists()


def test_openblas_explicit_rebuild_reaches_builder_without_admitting_cache(tmp_path):
    install = _openblas_cache_tree(tmp_path, header_only=True)
    # Stop immediately at the compiler prerequisite, before any cold-build work.
    builder = tmp_path / "scripts/build_openblas.sh"
    source = builder.read_text()
    source = source.replace('if ! command -v gfortran >/dev/null 2>&1; then',
                            'echo EXPLICIT_REBUILD_REACHED; exit 42\nif false; then')
    builder.write_text(source)
    result = _fftw_shell(tmp_path, "REBUILD=1 /bin/bash scripts/build_openblas.sh")
    assert result.returncode == 42, result.stdout + result.stderr
    assert "EXPLICIT_REBUILD_REACHED" in result.stdout
    assert "cache admitted" not in result.stdout
    assert install.exists()


@pytest.mark.parametrize("configuration,admitted", [
    ("OpenBLAS 0.3.34 USE_OPENMP MAX_THREADS=56", True),
    ("OpenBLAS 0.3.33 DYNAMIC_ARCH MAX_THREADS=128", True),
    ("OpenBLAS 0.3.34 USE64BITINT DYNAMIC_ARCH", False),
    ("OpenBLAS 0.3.34 NO_LAPACK", False),
    ("OpenBLAS 0.3.34 NO_LAPACKE", False),
])
def test_openblas_configuration_keeps_abi_and_backend_distinctions(configuration, admitted):
    import importlib.util

    spec = importlib.util.spec_from_file_location("native_identity", REPO_ROOT / "scripts/attest_native_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if admitted:
        assert module.parse_openblas_configuration(configuration) in ("0.3.33", "0.3.34")
    else:
        with pytest.raises(ValueError):
            module.parse_openblas_configuration(configuration)


def test_openblas_rejects_real_library_without_identification_entry_point(tmp_path):
    import os

    wrong_library = os.environ.get("VIBEQC_TEST_FFTW_LIBRARY")
    if not wrong_library:
        pytest.skip("set VIBEQC_TEST_FFTW_LIBRARY for a real wrong-library control")
    install = _openblas_cache_tree(tmp_path)
    (install / "lib/libopenblas.so").unlink()
    name = "libopenblas.dylib" if wrong_library.endswith(".dylib") else "libopenblas.so"
    shutil.copy2(wrong_library, install / "lib" / name)
    result = _fftw_shell(tmp_path, "/bin/bash scripts/build_openblas.sh")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "openblas_get_config" in result.stderr


def test_openblas_missing_lapacke_export_is_refused_before_configuration_call(monkeypatch):
    import importlib.util
    import os

    real = os.environ.get("VIBEQC_TEST_OPENBLAS_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_OPENBLAS_LIBRARY; no build")
    spec = importlib.util.spec_from_file_location("native_identity", REPO_ROOT / "scripts/attest_native_image.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.ctypes.CDLL

    class MissingLapacke:
        def __init__(self, library):
            self.library = library

        def __getattr__(self, name):
            if name == "LAPACKE_dsyev":
                raise AttributeError("missing required LAPACKE_dsyev")
            return getattr(self.library, name)

    def controlled_loader(path):
        library = original(path)
        return MissingLapacke(library) if path is not None else library

    monkeypatch.setattr(module.ctypes, "CDLL", controlled_loader)
    with pytest.raises(AttributeError, match="LAPACKE_dsyev"):
        module.probe_identity(real, "openblas")


def test_setup_routes_explicit_openblas_rebuild_through_drift_reconciliation(tmp_path):
    _openblas_cache_tree(tmp_path, header_only=True)
    updater = tmp_path / "scripts/update_native_deps.sh"
    updater.write_text('#!/bin/bash\nprintf "REQUEST:%s\\n" "$@"\nexit 42\n')
    result = _fftw_shell(tmp_path, "REBUILD=1 WITH_OPENBLAS=1 /bin/bash scripts/setup_native_deps.sh")
    assert result.returncode == 42, result.stdout + result.stderr
    assert "REQUEST:--dep\nREQUEST:openblas" in result.stdout
    assert not (tmp_path / "BUILD_ATTEMPTED").exists()


def _libint_cache_tree(tmp_path, *, unloaded=False, real_prefix=None):
    _fftw_cache_tree(tmp_path)
    guard = tmp_path / '.no-native-tools'
    guard.mkdir()
    for tool in ('git', 'curl', 'wget', 'cmake', 'make', 'ninja', 'cc', 'c++',
                 'gcc', 'g++', 'clang', 'clang++', 'gfortran', 'configure'):
        stub = guard / tool
        stub.write_text('#!/bin/sh\necho "REFUSED_NATIVE_TOOL:$0" >&2\nexit 77\n')
        stub.chmod(0o755)
    shutil.rmtree(tmp_path / 'third_party/fftw/install')
    builder = (REPO_ROOT / 'scripts/build_libint.sh').read_text()
    boundary = '# --- Source ---------------------------------------------------------------'
    assert boundary in builder
    builder = builder.replace(boundary,
        '# vibeqc-recipe-hash: lifecycle-only-begin\n'
        'if [ "${VIBEQC_FETCH_ONLY:-0}" != "1" ]; then\n'
        'touch "$(dirname "$0")/../BUILD_ATTEMPTED"; exit 99\nfi\n'
        '# vibeqc-recipe-hash: lifecycle-only-end\n' + boundary, 1)
    (tmp_path / 'scripts/build_libint.sh').write_text(builder)
    install = tmp_path / 'third_party/libint/install'
    config = install / 'lib/cmake/libint2/libint2-config.cmake'
    config.parent.mkdir(parents=True)
    config.write_text('# recipe metadata is not an artifact identity\n')
    (install / '.vibeqc-max-am').write_text('5_4_4\n')
    (install / '.vibeqc-fitting-max-am').write_text('6_6_6\n')
    header = install / 'include/libint2/boys.h'
    header.parent.mkdir(parents=True)
    header.write_text(
        '#include <algorithm>\n'
        'class FmEval_Chebyshev7 {\n    if (x > T_crit) {\n    }\n  }  // eval()\n};\n'
        'struct erfx_coulomb_gm_eval\n    : private scratch_base {\n'
        '  void operator()(Real* Gm, Real rho, Real T, int mmax, Real omega, Real lambda,\n'
        '                  Real sigma) {\n'
        '    evaluate(&base_type::Fm_[0], T, mmax);\n'
        '    Gm[m] = sigma * base_type::Fm_[m];\n'
        '    evaluate(&base_type::Fm_[0], scaled_T, mmax);\n'
        '    Gm[m] -= factor * base_type::Fm_[m];\n'
        '  }\n};\n}  // namespace os_core_ints\n')
    if unloaded:
        (install / 'lib/libint2.so').write_bytes(b'not a loadable library')
        (install / 'lib/libint2-cxx.so').write_bytes(b'not a loadable library')
    if real_prefix:
        for name in ('libint2.dylib', 'libint2-cxx.dylib'):
            shutil.copyfile(Path(real_prefix) / 'lib' / name, install / 'lib' / name)
    return install


@pytest.mark.parametrize('unloaded', [False, True])
@pytest.mark.parametrize('command', [
    '/bin/bash scripts/build_libint.sh',
    '. scripts/_native_stamp.sh; vibeqc_stamp_write',
    '. scripts/_native_stamp.sh; vibeqc_stamp_check',
    '/bin/bash scripts/update_native_deps.sh',
    '/bin/bash scripts/setup_native_deps.sh',
])
def test_libint_invalid_cache_refused_before_header_or_stamp_mutation(tmp_path, command, unloaded):
    install = _libint_cache_tree(tmp_path, unloaded=unloaded)
    header = install / 'include/libint2/boys.h'
    original = header.read_bytes()
    stamp = tmp_path / 'third_party/.build-stamp'
    old_stamp = stamp.read_bytes()
    result = _fftw_shell(tmp_path, command)
    assert result.returncode == 3, result.stdout + result.stderr
    assert 'identity unavailable' in result.stderr
    assert header.read_bytes() == original
    assert stamp.read_bytes() == old_stamp
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


def test_libint_real_c_image_precedes_cached_header_patch(tmp_path):
    import json
    import os
    real = os.environ.get('VIBEQC_TEST_LIBINT_LIBRARY')
    if not real:
        pytest.skip('set VIBEQC_TEST_LIBINT_LIBRARY; no build')
    install = _libint_cache_tree(tmp_path, real_prefix=Path(real).parent.parent)
    # The core links the header-only C++ interface to the C shared library.
    # libint2-cxx is deliberately absent, not an inferred required artifact.
    (install / 'lib/libint2-cxx.dylib').unlink()
    result = _fftw_shell(tmp_path, '/bin/bash scripts/build_libint.sh')
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout.splitlines()[0])['artifacts'][0]
    assert record['version'] == record['runtime_capabilities'] == 'not_attested'
    assert record['requested_path'] == record['loaded_path'] == str((install / 'lib/libint2.dylib').resolve())
    header = install / 'include/libint2/boys.h'
    assert 'x >= T_crit' in header.read_text()
    assert 'std::array<Real, 32> local' in header.read_text()
    original = header.stat().st_mtime_ns
    again = _fftw_shell(tmp_path, '/bin/bash scripts/build_libint.sh')
    assert again.returncode == 0, again.stderr
    assert header.stat().st_mtime_ns == original
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()
    for dep in ('fftw', 'libxc', 'spglib', 'openblas', 'libecpint'):
        (tmp_path / 'scripts' / f'build_{dep}.sh').unlink()
    stamp = tmp_path / 'third_party/.build-stamp'
    old = 'libint v2.13.1 15c929d322009784daff6981454e439dbead4b5518be2d43e032e795d86757c5'
    for row in (old, old + ' v2:5bf93e44e0ffedba0604cd0ab7fb3b1bbd64248a8ca9e2483ee746e07568a5e7'):
        stamp.write_text(row + '\n')
        checked = _fftw_shell(tmp_path, '. scripts/_native_stamp.sh; vibeqc_stamp_check')
        assert checked.returncode == 0, checked.stdout + checked.stderr
        preview = _fftw_shell(tmp_path, '/bin/bash scripts/update_native_deps.sh --dry-run')
        assert preview.returncode == 0 and 'all current' in preview.stdout, preview.stderr
        assert stamp.read_text() == row + '\n'
    builder = tmp_path / 'scripts/build_libint.sh'
    builder.write_text(builder.read_text().replace('-DBUILD_SHARED_LIBS=ON', '-DBUILD_SHARED_LIBS=OFF'))
    changed = _fftw_shell(tmp_path, '. scripts/_native_stamp.sh; _vibeqc_stamp_recipe_matches scripts/build_libint.sh 15c929d322009784daff6981454e439dbead4b5518be2d43e032e795d86757c5')
    assert changed.returncode != 0
    # Loader search overrides cannot rescue the invalid requested image.
    local = install / 'lib/libint2.dylib'
    local.write_bytes(b'not a loadable image')
    helper = tmp_path / 'scripts/attest_native_image.py'
    for arg in (str(install), '--attest ' + str(local)):
        negative = _fftw_shell(tmp_path, 'DYLD_LIBRARY_PATH=' + str(Path(real).parent) +
                              ' LD_LIBRARY_PATH=' + str(Path(real).parent) +
                              ' python3 ' + str(helper) + ' libint ' + arg)
        assert negative.returncode != 0, negative.stdout


@pytest.mark.parametrize('selection,expected', [('--all', 0), ('--dep libint', 0), ('--dep fftw', 3)])
def test_libint_explicit_rebuild_preview_stays_scoped(tmp_path, selection, expected):
    install = _libint_cache_tree(tmp_path, unloaded=True)
    original = (install / 'include/libint2/boys.h').read_bytes()
    result = _fftw_shell(tmp_path, '/bin/bash scripts/update_native_deps.sh --dry-run ' + selection)
    assert result.returncode == expected, result.stdout + result.stderr
    assert (install / 'include/libint2/boys.h').read_bytes() == original
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


def test_libint_fetch_only_does_not_admit_or_patch_invalid_install(tmp_path):
    install = _libint_cache_tree(tmp_path, unloaded=True)
    header = install / 'include/libint2/boys.h'
    original = header.read_bytes()
    src = tmp_path / 'third_party/libint/src'
    (src / 'include/libint2').mkdir(parents=True)
    (src / 'CMakeLists.txt').write_text('# pre-staged fixture\n')
    (src / 'include/libint2/boys.h').write_bytes(original)
    result = _fftw_shell(tmp_path, 'VIBEQC_FETCH_ONLY=1 /bin/bash scripts/build_libint.sh')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'source staged' in result.stdout
    assert 'already installed' not in result.stdout
    assert header.read_bytes() == original
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


@pytest.mark.parametrize('kind', ['wrong', 'ambiguous', 'escape', 'static'])
def test_libint_unsupported_images_refused_without_header_mutation(tmp_path, kind):
    import os
    real = os.environ.get('VIBEQC_TEST_LIBINT_LIBRARY')
    wrong = os.environ.get('VIBEQC_TEST_FFTW_LIBRARY')
    if not real or not wrong:
        pytest.skip('set existing libint and FFTW controls; no build')
    install = _libint_cache_tree(tmp_path)
    primary = install / 'lib/libint2.dylib'
    if kind == 'wrong':
        shutil.copyfile(wrong, primary)
    elif kind == 'escape':
        primary.symlink_to(real)
    elif kind == 'static':
        (install / 'lib/libint2.a').write_bytes(b'not a shared image')
    else:
        shutil.copyfile(real, primary)
        shutil.copyfile(real, install / 'lib/libint2.99.dylib')
    header = install / 'include/libint2/boys.h'
    original = header.read_bytes()
    result = _fftw_shell(tmp_path, '/bin/bash scripts/build_libint.sh')
    assert result.returncode == 3, result.stdout + result.stderr
    assert header.read_bytes() == original
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


def test_libint_selection_does_not_hide_other_invalid_cache(tmp_path):
    _libint_cache_tree(tmp_path, unloaded=True)
    other = tmp_path / 'third_party/fftw/install/lib'
    other.mkdir(parents=True)
    (other / 'libfftw3.so').write_bytes(b'invalid other dependency')
    result = _fftw_shell(tmp_path, '/bin/bash scripts/update_native_deps.sh --dry-run --dep libint')
    assert result.returncode == 3
    assert 'fftw artifact identity unavailable' in result.stderr
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


@pytest.mark.parametrize('command', [
    '/bin/bash scripts/build_libint.sh',
    '. scripts/_native_stamp.sh; vibeqc_stamp_write',
    '. scripts/_native_stamp.sh; vibeqc_stamp_check',
    '/bin/bash scripts/setup_native_deps.sh',
])
def test_libint_missing_config_never_triggers_cache_repair(tmp_path, command):
    install = _libint_cache_tree(tmp_path)
    (install / 'lib/cmake/libint2/libint2-config.cmake').unlink()
    original = (install / 'include/libint2/boys.h').read_bytes()
    result = _fftw_shell(tmp_path, command)
    assert result.returncode == 3, result.stdout + result.stderr
    assert 'lacks libint2-config.cmake' in result.stderr
    assert (install / 'include/libint2/boys.h').read_bytes() == original
    assert not (tmp_path / 'BUILD_ATTEMPTED').exists()


@pytest.mark.parametrize("dep", ["fftw", "libxc", "spglib"])
@pytest.mark.parametrize("mode", ["unset", "enabled", "failure", "fetch", "cache", "inside", "relative"])
def test_optional_native_collector_build_boundary(tmp_path, dep, mode):
    """Exercise public argv plumbing with all native/network work replaced."""
    import json
    import os
    import sys

    tree = tmp_path / "product with spaces"
    scripts = tree / "scripts"
    scripts.mkdir(parents=True)
    for name in (f"build_{dep}.sh", "_native_collector.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts / name)
    # The lock and cache probes have their own actual-image/lifecycle controls.
    # This fixture isolates the observer boundary, never a compiler or loader.
    (scripts / "_safe_build_env.sh").write_text(":\n")
    (scripts / "_build_lock.sh").write_text("vibeqc_acquire_build_lock() { :; }\n")
    (scripts / "_verify_source.sh").write_text(
        "git_clone_pinned() { exit 77; }\nverify_tarball_sha256() { :; }\n"
    )
    for name in ("attest_fftw_image.py", "attest_native_image.py", "patch_libxc_spin_fractions.py"):
        (scripts / name).write_text("raise SystemExit(0)\n")
    source = tree / "third_party" / dep / "src"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("# staged fixture; never configured\n")
    build = source.parent / "build"
    install = source.parent / "install"
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "commands.jsonl"
    observer_log = tmp_path / "observer.jsonl"
    # PATH and explicit compiler variables both refuse native/network tools.
    refusal = "#!/bin/sh\nexit 77\n"
    for name in ("git", "curl", "wget", "ninja", "make", "cc", "c++", "gcc", "g++", "clang", "clang++", "gfortran", "configure", "brew"):
        tool = binary / name
        tool.write_text(refusal)
        tool.chmod(0o755)
    cmake = binary / "cmake"
    cmake.write_text(f"#!{sys.executable}\n" + "import json, os, sys\nwith open(os.environ['COLLECTOR_TEST_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    cmake.chmod(0o755)
    observer = tmp_path / "external collector"
    observer.write_text(f"#!{sys.executable}\n" + "import json, os, subprocess, sys\nwith open(os.environ['COLLECTOR_TEST_OBSERVER'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nif os.environ['COLLECTOR_TEST_MODE'] == 'failure': raise SystemExit(19)\nraise SystemExit(subprocess.call(sys.argv[sys.argv.index('--')+1:]))\n")
    observer.chmod(0o755)
    receipts = tmp_path / "external receipts"
    receipts.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("VIBEQC_")}
    env.update(PATH=str(binary) + os.pathsep + os.environ["PATH"],
               CC=str(binary / "cc"), CXX=str(binary / "c++"), FC=str(binary / "gfortran"),
               CMAKE_BUILD_PARALLEL_LEVEL="1", COLLECTOR_TEST_LOG=str(log),
               COLLECTOR_TEST_OBSERVER=str(observer_log), COLLECTOR_TEST_MODE=mode)
    if mode != "unset":
        env.update(VIBEQC_NATIVE_BUILD_COLLECTOR=str(observer), VIBEQC_NATIVE_RECEIPT_DIR=str(receipts))
    if mode == "fetch":
        env["VIBEQC_FETCH_ONLY"] = "1"
    if mode == "cache":
        suffix = {"fftw": "fftw3/FFTW3Config.cmake", "libxc": "Libxc/LibxcConfig.cmake", "spglib": "Spglib/SpglibConfig.cmake"}[dep]
        config = install / "lib/cmake" / suffix
        config.parent.mkdir(parents=True)
        config.write_text("# cache route fixture\n")
    if mode == "inside":
        alias = tmp_path / "receipt alias"
        alias.symlink_to(tree, target_is_directory=True)
        env["VIBEQC_NATIVE_RECEIPT_DIR"] = str(alias)
    if mode == "relative":
        env["VIBEQC_NATIVE_BUILD_COLLECTOR"] = "external collector"
    result = subprocess.run(["/bin/bash", str(scripts / f"build_{dep}.sh")], cwd=tree,
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == (19 if mode == "failure" else 3 if mode in ("inside", "relative") else 0), result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    observed = [json.loads(line) for line in observer_log.read_text().splitlines()] if observer_log.exists() else []
    if mode in ("cache", "fetch", "inside", "relative"):
        assert commands == observed == []
        assert not build.exists()
        return
    assert commands[0][:4] == ["-S", str(source), "-B", str(build)]
    assert ("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" in commands[0]) == (mode != "unset")
    if mode == "unset":
        assert observed == []
    else:
        expected = ["record", "--source", f"{dep}={source}"]
        for name in (f"build_{dep}.sh", "_native_collector.sh", "_safe_build_env.sh", "_verify_source.sh", "_build_lock.sh"):
            expected += ["--recipe", str(scripts / name)]
        expected += ["--cmake-build", str(build), "--install", str(install),
                     "--artifact-manifest", str(build / "install_manifest.txt"),
                     "--cwd", str(tree), "--output", str(receipts / f"{dep}.json"),
                     "--", "cmake", "--build", str(build), "--target", "install"]
        assert observed == [expected]
    assert commands[1:] == ([] if mode == "failure" else [["--build", str(build), "--target", "install"]])


# This stdlib-only probe also runs under the source-staging interpreter, before
# the project environment (and its newer Python/pytest requirements) exists.
_BOOTSTRAP_HELPER_CHECK = r'''
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, sys.argv[1])
import attest_native_image as image
import patch_libxc_spin_fractions as patch

passed = []
def refuses(name, operation):
    try:
        operation()
    except ValueError:
        passed.append(name)
    else:
        raise AssertionError(name + " accepted invalid evidence")

with tempfile.TemporaryDirectory(prefix="native-bootstrap-") as directory:
    root = Path(directory).resolve()
    source = root / "source"
    source.mkdir()
    old, new = patch.TARGETS[0][2][0]
    original = "/* preserve upstream notice */\n" + (old + "\n") * 5
    digest = hashlib.sha256(original.encode()).hexdigest()
    good = source / "good.c"
    bad = source / "bad.c"
    good.write_text(original)
    bad.write_text(original + "/* local edit */\n")
    targets = (("good.c", digest, ((old, new),)), ("bad.c", digest, ((old, new),)))
    with mock.patch.object(patch, "TARGETS", targets):
        refuses("all targets checked before mutation", lambda: patch.patch_source(source))
    assert good.read_text() == original
    with mock.patch.object(patch, "TARGETS", targets[:1]):
        patch.patch_source(source)
        fixed = good.read_bytes()
        patch.patch_source(source)
        assert good.read_bytes() == fixed and fixed.decode().count(new) == 5
    passed.append("source patch and idempotence")
    refuses("modified source rejected", lambda: patch.patched_text(original + "edit", digest, ((old, new),)))
    marker = patch.patched_text(original, digest, ())
    assert patch.patched_text(marker, digest, ()) == marker
    passed.append("marker idempotence")

    install = root / "install"
    library = install / "lib/libxc.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"identity fixture")
    assert image.artifact_paths(install, "libxc") == [library]
    passed.append("contained artifact discovery")
    outside = root / "outside.so"
    outside.write_bytes(b"outside")
    escape = library.parent / "libxc.so.escaped"
    escape.symlink_to(outside)
    refuses("resolved symlink escape rejected", lambda: image.artifact_paths(install, "libxc"))
    escape.unlink()
    refuses("missing artifact rejected", lambda: image.artifact_paths(install, "fftw"))
    owner = image.loaded_symbol_path(ctypes.cast(ctypes.pythonapi.Py_GetVersion, ctypes.c_void_p).value)
    assert owner.is_file()
    passed.append("actual loaded symbol ownership")

    # Real bounded subprocesses exercise the old interpreter API. The child is
    # synthetic; separate loaded-image controls cover actual dependency symbols.
    child = root / "probe.py"
    child_source = "import hashlib,json,pathlib,sys\np=pathlib.Path(sys.argv[-1])\n"
    child_source += "print(json.dumps({'requested_path':str(p),'loaded_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))\n"
    child.write_text(child_source)
    with mock.patch.object(image, "__file__", str(child)):
        assert image.attest_install(install, "libxc")[0]["requested_path"] == str(library)
        passed.append("attestation subprocess succeeds")
        child.write_text(child_source.replace("'loaded_path':str(p)", "'loaded_path':'wrong'"))
        refuses("child substituted path rejected", lambda: image.attest_install(install, "libxc"))
        child.write_text(child_source.replace("hashlib.sha256(p.read_bytes()).hexdigest()", "'wrong'"))
        refuses("child digest mismatch rejected", lambda: image.attest_install(install, "libxc"))
        child.write_text("raise SystemExit(9)\n")
        refuses("failed attestation child rejected", lambda: image.attest_install(install, "libxc"))
    child.write_text("raise SystemExit(0)\n")
    with mock.patch.object(patch, "__file__", str(child)):
        patch.check_install(install)
        passed.append("patch install subprocess succeeds")
        child.write_text("raise SystemExit(9)\n")
        refuses("missing patch child rejected", lambda: patch.check_install(install))
print(json.dumps({"python": sys.version, "executable": sys.executable, "passed": passed}))
'''


def test_bootstrap_helpers_with_selected_python(tmp_path):
    import json
    import os
    import sys

    python = os.environ.get("VIBEQC_BOOTSTRAP_PYTHON", sys.executable)
    result = subprocess.run([python, "-B", "-c", _BOOTSTRAP_HELPER_CHECK, str(REPO_ROOT / "scripts")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert len(report["passed"]) == 14


@pytest.mark.parametrize("mode", ["valid", "wrong-pin", "unsupported", "collector-failure"])
def test_collector_authenticated_archive_wiring(tmp_path, mode):
    """Exercise the real hook with a recorder fixture, never a compiler."""
    import hashlib
    import json
    import os
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"synthetic archive: no compilation")
    receipt = tmp_path / "receipts"
    receipt.mkdir()
    product = tmp_path / "product"
    product.mkdir()
    recorder = tmp_path / "collector"
    recorder.write_text("#!/usr/bin/env python3\nimport json, pathlib, sys\n"
                        "pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps(sys.argv[1:]))\n"
                        + ("raise SystemExit(7)\n" if mode == "collector-failure" else ""))
    recorder.chmod(0o755)
    env = dict(os.environ, VIBEQC_NATIVE_BUILD_COLLECTOR=str(recorder),
               VIBEQC_NATIVE_RECEIPT_DIR=str(receipt), VIBEQC_NATIVE_SOURCE_ARCHIVE=str(archive),
               REPO_ROOT=str(product), SRC_DIR=str(product / "src"),
               BUILD_DIR=str(product / "build"), INSTALL_DIR=str(product / "install"),
               SCRIPT_DIR=str(REPO_ROOT / "scripts"), FFTW_VERSION="3.3.10",
               FFTW_SHA256="0" * 64 if mode == "wrong-pin" else hashlib.sha256(archive.read_bytes()).hexdigest())
    dependency = "libxc" if mode == "unsupported" else "fftw"
    result = subprocess.run(["bash", "-c",
        'source "$SCRIPT_DIR/_native_collector.sh"; vibeqc_enable_native_collector "$1" || exit $?; '
        'cmake --build "$BUILD_DIR" --target install', "hook-test", dependency],
        env=env, capture_output=True, text=True, timeout=10)
    output = receipt / "fftw.json"
    if mode in ("wrong-pin", "unsupported"):
        assert result.returncode == 3
        assert not output.exists()
        assert not (product / "build").exists()
    else:
        assert result.returncode == (7 if mode == "collector-failure" else 0), result.stderr
        argv = json.loads(output.read_text())
        spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_text())
        assert spec["authenticated_sources"]["fftw"]["sha256"] == env["FFTW_SHA256"]
        assert spec["authenticated_sources"]["fftw"]["prefix"] == "fftw-3.3.10"
        assert spec["sources"] == {"fftw": env["SRC_DIR"]}
        assert argv[argv.index("--") + 1:] == ["cmake", "--build", env["BUILD_DIR"], "--target", "install"]
