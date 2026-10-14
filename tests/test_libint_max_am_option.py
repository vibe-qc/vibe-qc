"""The ``--libint-max-am`` build option: parsing, drift, and short-circuit.

``scripts/build_libint.sh`` reads libint's per-derivative-order ``max_am``
from ``VIBEQC_LIBINT_MAX_AM`` (default ``5_4_4``) rather than from a literal
in the recipe, so users can trade the ~1.5-2 h g-function-Hessian build for
the ~30 min ``5_4_3`` one. See ``scripts/_libint_max_am.sh``.

Making a build flag live outside the recipe file re-opens the exact footgun
``docs/updating.md`` § "libint stale-install footgun" documents: the flags
change while the file on disk does not, so every existing staleness check
looks straight through it. Two mechanisms close it, and both are pinned
here because neither is exercised by anything cheaper than a real build:

* **The stamp salt.** ``_vibeqc_recipe_hash`` folds the effective spec into
  libint's recipe hash, so switching specs is drift and ``update.sh``
  rebuilds libint. Critically it is folded in for *that recipe only* —
  every other dep's hash must stay byte-identical to what already-written
  stamps recorded, or upgrading to this commit would force a spurious
  full rebuild of all five deps on every checkout in the fleet.
* **The install marker.** ``build_libint.sh`` writes the spec it built into
  ``third_party/libint/install/.vibeqc-max-am`` and refuses to short-circuit
  on a tree carrying a different one.

These drive the *real* shipped shell, mirroring
``test_native_dep_artifact_health.py``'s style. The build_libint.sh cases
stop at an injected pre-configure fixture boundary, which exits right
after the source check and well after the short-circuit decision under
test. One optional native check compiles a small evaluator probe against an
existing vendored installation; it does not build or modify Libint.

Update this file in lockstep with ``scripts/_libint_max_am.sh``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
HELPER = SCRIPTS / "_libint_max_am.sh"
NATIVE_STAMP = SCRIPTS / "_native_stamp.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or not HELPER.exists(),
    reason="bash not available or scripts/_libint_max_am.sh missing",
)

DEFAULT_SPEC = "5_4_4"
# The pre-2026-08-06 recipe, and the documented "make it fast again" value.
FAST_SPEC = "5_4_3"

# Every dep whose recipe hash must be unaffected by the libint salt.
OTHER_DEPS = ["libxc", "spglib", "fftw", "libecpint", "openblas"]


def _bash(snippet: str, *, cwd: Path | None = None, env: dict | None = None,
          source: Path = HELPER) -> subprocess.CompletedProcess:
    # Stamp fixtures below are metadata stand-ins; actual artifact admission
    # is exercised by test_native_dep_artifact_health's real-image controls.
    protocol_stub = "vibeqc_verify_cached_libint() { :; }\n" if source == NATIVE_STAMP else ""
    script = f". {shlex.quote(str(source))}\n{protocol_stub}{snippet}\n"
    full_env = dict(os.environ)
    # Never inherit a developer's own setting into the assertions.
    full_env.pop("VIBEQC_LIBINT_MAX_AM", None)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=full_env,
    )


# ---------------------------------------------------------------------------
# Spec resolution.
# ---------------------------------------------------------------------------

_RESOLVE = (
    "vibeqc_libint_max_am_resolve || exit 1\n"
    'echo "$VIBEQC_LIBINT_AM_SPEC|$VIBEQC_LIBINT_AM_LIST|$VIBEQC_LIBINT_AM_GLOBAL"'
)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, f"{DEFAULT_SPEC}|5;4;4|5"),      # unset → documented default
        ("5_4_4", "5_4_4|5;4;4|5"),
        ("5_4_3", "5_4_3|5;4;3|5"),
        # The cmake-list and comma spellings are the obvious things to type;
        # both normalise to the underscore form so one spelling reaches the
        # stamp (else 5;4;3 and 5_4_3 would hash differently and each switch
        # between them would look like drift).
        ("5;4;3", "5_4_3|5;4;3|5"),
        ("5,4,3", "5_4_3|5;4;3|5"),
        # LIBINT2_MAX_AM must cover every stratum → it is the max, not [0].
        ("4_4_4", "4_4_4|4;4;4|4"),
        # l=6 work (cc-pV6Z-class i functions) must be reachable: it is the
        # documented top of the range, not an accident of the check.
        ("6_5_4", "6_5_4|6;5;4|6"),
        ("6_6_6", "6_6_6|6;6;6|6"),
    ],
)
def test_resolve_valid_specs(spec, expected):
    env = {} if spec is None else {"VIBEQC_LIBINT_MAX_AM": spec}
    res = _bash(_RESOLVE, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == expected, res.stdout + res.stderr


@pytest.mark.parametrize(
    "spec",
    [
        "5_4",        # too few parts
        "5_4_4_3",    # too many
        "5_4_x",      # not an integer
        "5__4",       # empty middle part
        "0_4_4",      # below range
        "7_4_4",      # one past the cart_to_sph ceiling
        "9_4_4",      # well above range
        "fast",       # word, not a spec
    ],
)
def test_resolve_rejects_malformed_specs(spec):
    res = _bash(_RESOLVE, env={"VIBEQC_LIBINT_MAX_AM": spec})
    assert res.returncode == 1, f"{spec!r} was accepted:\n{res.stdout}"
    assert "invalid libint max_am spec" in res.stderr, res.stderr
    # The error must show the caller how to spell it — this is the message a
    # user sees instead of a 2 h build that produces the wrong thing.
    assert "N_N_N" in res.stderr, res.stderr


def test_ceiling_matches_the_cart_to_sph_table():
    """The upper bound is vibe-qc's, not libint's: the periodic AO-pair FT
    throws on any shell past ``cart_to_sph_data::kMaxL``, so a libint built
    above it emits integrals nothing downstream can transform. If the table
    is ever regenerated deeper, this check has to move with it — that is
    what this test is for."""
    header = (
        REPO_ROOT / "cpp" / "include" / "vibeqc" / "cart_to_sph_data.hpp"
    ).read_text(encoding="utf-8")
    kmaxl = next(
        int(ln.split("=")[1].strip().rstrip(";"))
        for ln in header.splitlines()
        if "constexpr int kMaxL" in ln
    )
    res = _bash('echo "$VIBEQC_LIBINT_MAX_AM_CEILING"')
    assert res.returncode == 0, res.stderr
    assert int(res.stdout.strip()) == kmaxl, (
        f"_libint_max_am.sh caps max_am at {res.stdout.strip()} but "
        f"cart_to_sph_data.hpp covers L = 0..{kmaxl}"
    )


def test_ceiling_rejection_explains_itself():
    """A user asking for l=7 must learn why it is refused — otherwise the
    obvious next move is to edit the range check and build something the
    periodic stack rejects at runtime."""
    res = _bash(_RESOLVE, env={"VIBEQC_LIBINT_MAX_AM": "7_5_4"})
    assert res.returncode == 1, res.stdout
    assert "cart_to_sph" in res.stderr, res.stderr


def test_rising_spec_warns_but_builds():
    """Angular momentum rising with derivative order is almost always a typo,
    but there is no correctness reason to refuse it."""
    res = _bash(_RESOLVE, env={"VIBEQC_LIBINT_MAX_AM": "3_4_5"})
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "3_4_5|3;4;5|5"
    assert "rises with" in res.stderr, res.stderr


def test_spec_accessor_is_errexit_safe():
    """_native_stamp.sh calls vibeqc_libint_max_am_spec from command
    substitution under `set -euo pipefail`; a non-zero return there would
    abort update.sh mid-run. It must never fail, even on a bogus spec."""
    snippet = (
        "set -euo pipefail\n"
        'got="$(vibeqc_libint_max_am_spec)"\n'
        'echo "got=${got}"\n'
    )
    res = _bash(snippet, env={"VIBEQC_LIBINT_MAX_AM": "utter_nonsense_here"})
    assert res.returncode == 0, f"errexit aborted the accessor:\n{res.stderr}"
    assert "got=utter_nonsense_here" in res.stdout, res.stdout


# ---------------------------------------------------------------------------
# Stamp integration: switching the spec is drift, for libint and nothing else.
# ---------------------------------------------------------------------------


@pytest.fixture
def stamp_tree(tmp_path: Path) -> Path:
    """Stamp-only stand-ins for deps whose admission is presence-based.

    FFTW/libxc/spglib loaded-image admission is covered by test_native_dep_artifact_health;
    an empty file is deliberately no longer a valid native cache fixture.
    """
    (tmp_path / "scripts").mkdir()
    libs = {
        "libint": ("LIBINT_VERSION", "v2.13.1", "libint2"),
        "libecpint": ("LIBECPINT_VERSION", "1.0.7", "libecpint"),
    }
    for dep, (var, ver, soname) in libs.items():
        (tmp_path / "scripts" / f"build_{dep}.sh").write_text(
            f'{var}="{ver}"\necho build\n', encoding="utf-8"
        )
        libdir = tmp_path / "third_party" / dep / "install" / "lib"
        libdir.mkdir(parents=True)
        (libdir / f"{soname}.so").write_text("", encoding="utf-8")

    res = _bash("vibeqc_stamp_write >/dev/null", cwd=tmp_path, source=NATIVE_STAMP)
    assert res.returncode == 0, res.stderr
    return tmp_path


def test_stamp_row_format_is_unchanged(stamp_tree: Path):
    """The spec rides inside the hash, NOT as an extra stamp column. The row
    stays exactly the four fields the stamp format defines
    (<dep> <version> <legacy-hash> v2:<hash>), so readers that predate this
    option keep parsing it."""
    stamp = (stamp_tree / "third_party" / ".build-stamp").read_text(encoding="utf-8")
    libint_line = next(
        ln for ln in stamp.splitlines() if ln.startswith("libint ")
    )
    fields = libint_line.split()
    assert len(fields) == 4, libint_line
    assert fields[0] == "libint" and fields[3].startswith("v2:"), libint_line


def test_same_spec_is_not_drift(stamp_tree: Path):
    res = _bash("vibeqc_stamp_drifted_deps", cwd=stamp_tree, source=NATIVE_STAMP)
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == [], res.stdout


def test_switching_spec_drifts_libint_only(stamp_tree: Path):
    """The whole point: `update.sh --libint-max-am 5_4_3` must rebuild
    libint, and must NOT drag the other four deps along with it."""
    res = _bash(
        "vibeqc_stamp_drifted_deps",
        cwd=stamp_tree,
        source=NATIVE_STAMP,
        env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC},
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["libint"], res.stdout


def test_switching_spec_reports_as_a_setting_change(stamp_tree: Path):
    """A max_am switch must not be reported as "build_libint.sh changed" —
    that sends the reader hunting through a diff that does not exist. The
    installed marker is what makes the precise message possible."""
    marker = (
        stamp_tree / "third_party" / "libint" / "install" / ".vibeqc-max-am"
    )
    marker.write_text(f"{DEFAULT_SPEC}\n", encoding="utf-8")
    res = _bash(
        "vibeqc_stamp_check; echo RC=$?",
        cwd=stamp_tree,
        source=NATIVE_STAMP,
        env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC},
    )
    assert "RC=1" in res.stdout, res.stdout
    assert "BUILD-SETTING DRIFT" in res.stdout, res.stdout
    assert f"max_am={DEFAULT_SPEC}" in res.stdout, res.stdout
    assert f"max_am={FAST_SPEC}" in res.stdout, res.stdout


def test_switching_spec_without_a_marker_still_drifts(stamp_tree: Path):
    """Without the marker the message falls back to the generic wording, but
    the drift itself — the part that triggers the rebuild — must not depend
    on the marker being there."""
    res = _bash(
        "vibeqc_stamp_check; echo RC=$?",
        cwd=stamp_tree,
        source=NATIVE_STAMP,
        env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC},
    )
    assert "RC=1" in res.stdout, res.stdout
    assert "libint" in res.stdout, res.stdout


def test_switching_back_is_also_drift(stamp_tree: Path):
    """Symmetry: someone who built 5_4_3 must be able to get 5_4_4 back the
    same way, without knowing about --rebuild-native-deps."""
    res = _bash(
        "vibeqc_stamp_write >/dev/null; "
        "VIBEQC_LIBINT_MAX_AM=" + DEFAULT_SPEC + " vibeqc_stamp_drifted_deps",
        cwd=stamp_tree,
        source=NATIVE_STAMP,
        env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC},
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["libint"], res.stdout


@pytest.mark.parametrize("dep", OTHER_DEPS)
def test_other_recipe_hashes_are_untouched_by_the_salt(dep: str):
    """Regression guard against a fleet-wide spurious rebuild.

    The salt must be folded into build_libint.sh's hash ONLY. If it leaked
    into every hash (e.g. by unconditionally appending a line), every
    already-written third_party/.build-stamp in the fleet would mismatch on
    upgrade and every checkout would rebuild all five native deps for
    nothing. Asserted as invariance under the setting rather than by
    recomputing the digest here, so this stays valid when the hashing rules
    themselves evolve (they did, in 0255fec94's lifecycle-only exclusion).
    """
    script = SCRIPTS / f"build_{dep}.sh"
    if not script.exists():
        pytest.skip(f"{script} not present")
    snippet = f"_vibeqc_recipe_hash {shlex.quote(str(script))}"
    a = _bash(snippet, source=NATIVE_STAMP,
              env={"VIBEQC_LIBINT_MAX_AM": DEFAULT_SPEC})
    b = _bash(snippet, source=NATIVE_STAMP,
              env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC})
    c = _bash(snippet, source=NATIVE_STAMP)  # setting unset entirely
    assert a.returncode == 0 and b.returncode == 0 and c.returncode == 0
    assert a.stdout.strip(), a.stderr
    assert a.stdout.strip() == b.stdout.strip() == c.stdout.strip(), (
        f"build_{dep}.sh's recipe hash moved with VIBEQC_LIBINT_MAX_AM; the "
        "salt leaked past libint and would force a full native rebuild"
    )


def test_libint_hash_actually_changes_with_the_spec():
    """The converse of the test above: the salt has to do something."""
    script = SCRIPTS / "build_libint.sh"
    a = _bash(f"_vibeqc_recipe_hash {shlex.quote(str(script))}",
              source=NATIVE_STAMP, env={"VIBEQC_LIBINT_MAX_AM": DEFAULT_SPEC})
    b = _bash(f"_vibeqc_recipe_hash {shlex.quote(str(script))}",
              source=NATIVE_STAMP, env={"VIBEQC_LIBINT_MAX_AM": FAST_SPEC})
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout.strip() and a.stdout.strip() != b.stdout.strip()


# ---------------------------------------------------------------------------
# The install marker + build_libint.sh's short-circuit.
# ---------------------------------------------------------------------------


@pytest.fixture
def libint_tree(tmp_path: Path) -> Path:
    """A fake repo root holding a 'built' libint install and enough of
    scripts/ for build_libint.sh to run its decision logic."""
    (tmp_path / "scripts").mkdir()
    for name in (
        "build_libint.sh",
        "attest_native_image.py",
        "_safe_build_env.sh",
        "_verify_source.sh",
        "_build_lock.sh",
        "_libint_max_am.sh",
    ):
        shutil.copy(SCRIPTS / name, tmp_path / "scripts" / name)
    real = os.environ.get("VIBEQC_TEST_LIBINT_LIBRARY")
    if not real:
        pytest.skip("set VIBEQC_TEST_LIBINT_LIBRARY to an existing C library; no build")
    library = tmp_path / "third_party/libint/install/lib" / Path(real).name
    library.parent.mkdir(parents=True)
    shutil.copyfile(real, library)
    builder = tmp_path / "scripts/build_libint.sh"
    text = builder.read_text()
    boundary = "# --- Platform-aware prefix detection"
    assert boundary in text
    builder.write_text(text.replace(boundary, "echo 'FIXTURE_BUILD_BOUNDARY'; exit 0\n" + boundary, 1))
    guard = tmp_path / ".no-native-tools"
    guard.mkdir()
    for tool in ("git", "curl", "wget", "cmake", "make", "ninja", "cc", "c++", "gcc", "g++", "clang", "clang++", "gfortran", "configure"):
        stub = guard / tool
        stub.write_text('#!/bin/sh\necho REFUSED_NATIVE_TOOL >&2\nexit 77\n')
        stub.chmod(0o755)
    # Pre-staged source so the fetch step is a no-op (no network, no clone).
    src = tmp_path / "third_party" / "libint" / "src"
    src.mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("", encoding="utf-8")
    cfg = tmp_path / "third_party" / "libint" / "install" / "lib" / "cmake" / "libint2"
    cfg.mkdir(parents=True)
    (cfg / "libint2-config.cmake").write_text("", encoding="utf-8")
    for prefix in (src, tmp_path / "third_party/libint/install"):
        header = prefix / "include/libint2/boys.h"
        header.parent.mkdir(parents=True)
        header.write_text(
            "#include <algorithm>\n"
            "class FmEval_Chebyshev7 {\n"
            "    if (x > T_crit) {\n    }\n  }  // eval()\n};\n"
            "struct erfx_coulomb_gm_eval\n    : private scratch_base {\n"
            "  void operator()(Real* Gm, Real rho, Real T, int mmax, Real omega, Real lambda,\n"
            "                  Real sigma) {\n"
            "    evaluate(&base_type::Fm_[0], T, mmax);\n"
            "    Gm[m] = sigma * base_type::Fm_[m];\n"
            "    evaluate(&base_type::Fm_[0], scaled_T, mmax);\n"
            "    Gm[m] -= factor * base_type::Fm_[m];\n"
            "  }\n};\n}  // namespace os_core_ints\n"
        )
    (tmp_path / "third_party/libint/install/.vibeqc-fitting-max-am").write_text("6_6_6\n")
    return tmp_path


def _marker(tree: Path) -> Path:
    return tree / "third_party" / "libint" / "install" / ".vibeqc-max-am"


def _run_build_libint(tree: Path, spec: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VIBEQC_LIBINT_MAX_AM", None)
    # Stop before configure/compile; the decision under test happens earlier.
    env["VIBEQC_FETCH_ONLY"] = "0"
    guard = tree / ".no-native-tools"
    env["PATH"] = str(guard) + os.pathsep + env.get("PATH", "")
    for key, tool in (("CC", "cc"), ("CXX", "c++"), ("FC", "gfortran"), ("CMAKE_MAKE_PROGRAM", "ninja")):
        env[key] = str(guard / tool)
    # Skip the Linux nice/ionice self-re-exec so the test is deterministic.
    env["VIBEQC_BUILD_NICED"] = "1"
    if spec is not None:
        env["VIBEQC_LIBINT_MAX_AM"] = spec
    return subprocess.run(
        ["bash", str(tree / "scripts" / "build_libint.sh")],
        capture_output=True, text=True, env=env,
    )


def test_boys_endpoint_patch_applies_before_the_installed_short_circuit(libint_tree):
    _marker(libint_tree).write_text(DEFAULT_SPEC + "\n")
    header = libint_tree / "third_party/libint/install/include/libint2/boys.h"
    first = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert first.returncode == 0, first.stderr
    assert "if (x >= T_crit)" in header.read_text()
    changed_at = header.stat().st_mtime_ns
    second = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert second.returncode == 0, second.stderr
    assert header.stat().st_mtime_ns == changed_at


def test_boys_endpoint_patch_updates_fresh_source_and_rejects_unknown_code(libint_tree):
    shutil.rmtree(libint_tree / "third_party/libint/install")
    header = libint_tree / "third_party/libint/src/include/libint2/boys.h"
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode == 0, result.stderr
    assert "if (x >= T_crit)" in header.read_text()
    header.write_text(header.read_text().replace("x >= T_crit", "x > unknown_limit"))
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode != 0
    assert "unexpected evaluator" in result.stderr


@pytest.mark.parametrize("installed", [False, True])
def test_erfc_scratch_patch_is_idempotent_and_preserves_arithmetic(libint_tree, installed):
    _marker(libint_tree).write_text(DEFAULT_SPEC + "\n")
    if not installed:
        shutil.rmtree(libint_tree / "third_party/libint/install")
    prefix = "install" if installed else "src"
    header = libint_tree / f"third_party/libint/{prefix}/include/libint2/boys.h"
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode == 0, result.stderr
    text = header.read_text()
    assert "std::array<Real, 32> local;" in text
    assert ": base_type(-1), fm_eval_(other.fm_eval_) {}" in text
    assert "base_type::Fm_.resize(mmax + 1);" in text
    assert "Gm[m] = sigma * scratch[m];" in text
    assert "Gm[m] -= factor * scratch[m];" in text
    changed_at = header.stat().st_mtime_ns
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode == 0, result.stderr
    assert header.read_text() == text
    assert header.stat().st_mtime_ns == changed_at


@pytest.mark.parametrize("already_patched", [False, True])
def test_erfc_scratch_patch_refuses_unrecognized_or_partial_header(libint_tree, already_patched):
    _marker(libint_tree).write_text(DEFAULT_SPEC + "\n")
    header = libint_tree / "third_party/libint/install/include/libint2/boys.h"
    if already_patched:
        result = _run_build_libint(libint_tree, DEFAULT_SPEC)
        assert result.returncode == 0, result.stderr
        text = header.read_text().replace("std::array<Real, 32>", "std::array<Real, 31>")
    else:
        # Apply the unrelated endpoint repair first, so this check can assert
        # the scratch patch leaves an unrecognized header completely intact.
        text = header.read_text().replace("x > T_crit", "x >= T_crit")
        text = text.replace("factor * base_type::Fm_[m]", "factor * unknown[m]")
    header.write_text(text)
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode != 0
    assert "libint erfc scratch patch: unexpected" in result.stderr
    assert header.read_text() == text


def test_erfc_scratch_native_copy_allocations_and_high_order_fallback(tmp_path):
    prefix = REPO_ROOT / "third_party/libint/install"
    upstream = prefix / "include/libint2/boys.h"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if os.name == "nt" or not upstream.is_file() or not shutil.which(compiler[0]):
        pytest.skip("requires a POSIX C++ compiler and existing vendored Libint")
    if not any((prefix / "lib" / f"libint2-cxx{suffix}").is_file()
               for suffix in (".so", ".dylib")):
        pytest.skip("vendored Libint C++ shared library is unavailable")
    header = tmp_path / "include/libint2/boys.h"
    header.parent.mkdir(parents=True)
    shutil.copyfile(upstream, header)
    recipe = (SCRIPTS / "build_libint.sh").read_text()
    function = "patch_erfc_scratch() {" + recipe.split("patch_erfc_scratch() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    patched = subprocess.run(
        ["bash", "-c", function + '\npatch_erfc_scratch "$1"', "patch", str(header)],
        capture_output=True, text=True, timeout=15,
    )
    assert patched.returncode == 0, patched.stderr
    source = tmp_path / "scratch.cpp"
    source.write_text(r'''
#include <libint2/boys.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <thread>
static std::atomic<unsigned> allocations{0};
void* operator new(std::size_t n) {
    ++allocations;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
using Eval = libint2::GenericGmEval<libint2::os_core_ints::erfx_coulomb_gm_eval<double>>;
int main() {
    for (int order : {0, 8, 31, 32, 40}) {
        const Eval eval(order, 1e-15);
        std::array<double, 43> value;
        value.fill(-123.);
        allocations = 0;
        for (int i = 0; i < 200; ++i) {
            eval.eval(value.data() + 1, 1., 0., order, 0., 0., 1.);
            for (int m = 0; m <= order; ++m)
                if (std::abs(value[m + 1] - 1. / (2 * m + 1)) > 1e-14) return 1;
        }
        if (order < 32 ? allocations != 0 : allocations != 200) return 2;
        if (value.front() != -123. || value[order + 2] != -123.) return 3;
    }
    const Eval shared(16, 1e-15);
    std::array<std::array<double, 17>, 4> values;
    std::array<std::thread, 4> workers;
    for (int j = 0; j < 4; ++j) workers[j] = std::thread([&, j] {
        for (int i = 0; i < 1000; ++i)
            shared.eval(values[j].data(), 1. + j, .1 * j, 16, .4, .2, .8);
    });
    for (auto& worker : workers) worker.join();
    for (int j = 0; j < 4; ++j) {
        std::array<double, 17> expected;
        shared.eval(expected.data(), 1. + j, .1 * j, 16, .4, .2, .8);
        if (values[j] != expected) return 4;
    }
}
''')
    executable = tmp_path / "scratch"
    command = compiler + [
        "-std=c++17", "-O2", "-pthread", "-I", str(tmp_path / "include"),
        "-I", str(prefix / "include"), str(source), "-L", str(prefix / "lib"),
        "-lint2-cxx", "-lint2", f"-Wl,-rpath,{prefix / 'lib'}", "-o", str(executable),
    ]
    built = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_installed_spec_reads_the_marker(libint_tree: Path):
    _marker(libint_tree).write_text("# header\n5_4_3\n", encoding="utf-8")
    res = _bash("vibeqc_libint_installed_spec", cwd=libint_tree)
    assert res.stdout.strip() == "5_4_3", res.stdout


def test_installed_spec_is_unknown_without_a_marker(libint_tree: Path):
    res = _bash("vibeqc_libint_installed_spec", cwd=libint_tree)
    assert res.stdout.strip() == "unknown", res.stdout


def test_matching_marker_short_circuits(libint_tree: Path):
    _marker(libint_tree).write_text(f"{DEFAULT_SPEC}\n", encoding="utf-8")
    res = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert res.returncode == 0, res.stderr
    assert "already installed" in res.stdout, res.stdout
    assert "Configuring" not in res.stdout, res.stdout


def test_differing_marker_forces_a_rebuild(libint_tree: Path):
    """The core guarantee: a tree built at 5_4_4 must not be silently reused
    when 5_4_3 was asked for."""
    _marker(libint_tree).write_text(f"{DEFAULT_SPEC}\n", encoding="utf-8")
    res = _run_build_libint(libint_tree, FAST_SPEC)
    assert res.returncode == 0, res.stderr
    assert "rebuilding" in res.stdout, res.stdout
    assert "already installed" not in res.stdout, res.stdout


def test_markerless_tree_rebuilds_even_for_the_default(libint_tree: Path):
    """Independent fitting support requires evidence of both generated limits."""
    res = _run_build_libint(libint_tree, None)
    assert res.returncode == 0, res.stderr
    assert "rebuilding" in res.stdout, res.stdout


def test_markerless_tree_rebuilds_for_a_non_default_request(libint_tree: Path):
    """...but an explicit non-default request cannot be satisfied by a tree
    whose spec is unprovable, so it rebuilds."""
    res = _run_build_libint(libint_tree, FAST_SPEC)
    assert res.returncode == 0, res.stderr
    assert "rebuilding" in res.stdout, res.stdout


def test_build_refuses_a_malformed_spec_before_doing_anything(libint_tree: Path):
    res = _run_build_libint(libint_tree, "5_4_9")
    assert res.returncode == 1, res.stdout
    assert "invalid libint max_am spec" in res.stderr, res.stderr
    assert "already installed" not in res.stdout, res.stdout


# ---------------------------------------------------------------------------
# The wrapper flag, on every script that exposes it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script", ["install.sh", "update.sh", "update_native_deps.sh"]
)
def test_wrapper_rejects_a_malformed_spec(script: str, tmp_path: Path):
    """Fail at parse time — before the wrapper has checked out a branch,
    touched a venv, or started a multi-hour build."""
    env = dict(os.environ)
    env.pop("VIBEQC_LIBINT_MAX_AM", None)
    env["VIBEQC_BUILD_NICED"] = "1"
    res = subprocess.run(
        ["bash", str(SCRIPTS / script), "--libint-max-am", "5_4_x"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert res.returncode != 0, res.stdout
    assert "invalid libint max_am spec" in res.stderr, res.stderr


@pytest.mark.parametrize(
    "script", ["install.sh", "update.sh", "update_native_deps.sh"]
)
def test_wrapper_requires_an_argument(script: str, tmp_path: Path):
    env = dict(os.environ)
    env.pop("VIBEQC_LIBINT_MAX_AM", None)
    env["VIBEQC_BUILD_NICED"] = "1"
    res = subprocess.run(
        ["bash", str(SCRIPTS / script), "--libint-max-am"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert res.returncode != 0, res.stdout
    # install.sh/update.sh route through their own require_value helper;
    # update_native_deps.sh has its own inline check. Both must refuse.
    assert "requires" in res.stderr and "--libint-max-am" in res.stderr, res.stderr


@pytest.mark.parametrize(
    "script", ["install.sh", "update.sh", "update_native_deps.sh"]
)
def test_wrapper_documents_the_flag(script: str):
    text = (SCRIPTS / script).read_text(encoding="utf-8")
    assert "--libint-max-am" in text
    help_block = text.split("set -euo pipefail")[0]
    assert "--libint-max-am" in help_block, (
        f"{script} accepts --libint-max-am but its --help text does not "
        "mention it"
    )


@pytest.mark.parametrize("fitting_spec", [None, "5_4_4"])
def test_orbital_marker_cannot_certify_missing_i_fitting_kernels(libint_tree, fitting_spec):
    _marker(libint_tree).write_text(DEFAULT_SPEC + "\n")
    marker = libint_tree / "third_party/libint/install/.vibeqc-fitting-max-am"
    if fitting_spec is None:
        marker.unlink()
    else:
        marker.write_text(fitting_spec + "\n")
    result = _run_build_libint(libint_tree, DEFAULT_SPEC)
    assert result.returncode == 0, result.stderr
    assert "rebuilding" in result.stdout
    assert "already installed" not in result.stdout
