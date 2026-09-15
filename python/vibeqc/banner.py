"""Runtime banner and linked-library version reporter.

Printed at the top of any serious vibe-qc text output so a user can always
identify exactly which build of the program produced a given log.

Programmatic API
----------------

``banner()``
    Return the full multi-line banner as a string.

``print_banner()``
    Convenience: ``print(banner(), **kwargs)``.

``library_versions()``
    Dict of the versions of linked native dependencies (libint, libxc,
    spglib) plus the package version. Useful in tests and bug reports.

``build_info()``
    Dict describing the current source build: git branch, short commit
    SHA, working-tree dirty flag, and exact-tag (if any). Empty when
    running from an installed wheel without a ``.git/`` available.

``enforce_runtime_pin_from_env()``
    Optional release-validation guard. When ``VIBEQC_EXPECT_*``
    environment variables are set, fail early if the running version,
    git SHA, or branch does not match the requested pin.

The ``format_scf_trace()`` helper in :mod:`vibeqc.scf_log` prepends the
banner automatically so a persisted SCF run log carries its provenance --
critical for matching a calculation result back to the exact build that
produced it (see ``docs/release_process.md``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping, TextIO

# Package version -- reads the installed metadata so it tracks pyproject.toml
# automatically at build/install time. Falls back to an explicit label if the
# package isn't found (e.g. running from a source tree without installation).
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
except ImportError:  # pragma: no cover -- Python 3.11+ always ships this
    _pkg_version = None
    PackageNotFoundError = Exception  # type: ignore[misc, assignment]


__all__ = [
    "RELEASE_CODENAMES",
    "VIBEQC_VERSION",
    "banner",
    "build_info",
    "codename_for_version",
    "enforce_runtime_pin_from_env",
    "library_versions",
    "print_banner",
]


# --- Release codename catalog --------------------------------------------
#
# Every major + minor release carries a "Scientist's Animal" codename --
# see docs/roadmap.md Sec. "Release codenames" for the full policy. The
# catalog is keyed by the X.Y.Z release version; minor releases (X.Y.0)
# are always present, and patch releases (X.Y.Z, Z > 0) MAY add their
# own override here if they have a distinct theme worth surfacing
# separately (e.g. v0.7.1 "Pulay's Triangle", v0.7.2 "Boys' Crucible",
# v0.7.3 "Whitten's Bridge" all override v0.7.0 "Löwdin's Compass").
# Patch versions WITHOUT an entry inherit the parent minor's codename
# via the fallback in :func:`codename_for_version`. Dev builds
# (.devN / aN / bN / rcN) inherit the codename of the upcoming release
# after PEP-440 suffix stripping.
#
# When cutting a new MINOR release (X.Y.0), add the entry here in the
# same commit that bumps ``pyproject.toml``. Patches need an entry
# here only if they get a distinct codename; otherwise they inherit.
#
# This catalog is also imported by ``docs/conf.py`` to render the
# landing-page "Current release" tip and the codename in MyST
# substitutions ({{codename}}). One source of truth, two consumers.
RELEASE_CODENAMES: dict[str, str] = {
    "0.1.0": "Roothaan's Raven",  # Clemens Roothaan (1918-2019),
    # Roothaan-Hall equations (1951):
    # the matrix-eigenvalue formulation of
    # HF that every molecular code rests on.
    # Raven: the first bird, intelligent,
    # lays the foundation everything else
    # builds on.
    "0.2.0": "Ewald's Eagle",  # Paul Peter Ewald (1888-1985),
    # Ewald summation (1921) for
    # conditionally convergent lattice
    # sums. Eagle: soars above the
    # periodic landscape with a Fourier
    # perspective.
    "0.2.5": "Wigner's Wolf",  # Eugene Wigner (1902-1995),
    # Nobel 1963, group theory in quantum
    # mechanics. The Wigner D-matrices
    # are the mathematical core of the
    # symmetry exploitation phase. Wolf:
    # hunts in packs (group theory).
    "0.3.0": "Mulliken's Magpie",  # Robert S. Mulliken (1896-1986),
    # Nobel 1966, molecular orbital
    # theory and population analysis.
    # Magpie: collects shiny things
    # (properties, visualizations).
    "0.4.0": "Hellmann's Hedgehog",  # Hans Hellmann (1903-1938),
    # Hellmann-Feynman theorem (1937)
    # makes analytic gradients tractable.
    # Hedgehog: well-protected by spines
    # (ECPs replace core electrons).
    "0.5.0": "Wilson's Otter",  # vibrations / Hessian flagship
    # (E. B. Wilson 1955)
    "0.6.0": "Pulay's Owl",  # periodic atomic gradients --
    # Pulay correction terms (1969)
    # are the conceptual core of
    # this release; Pulay also gave
    # us DIIS (1980) which underpins
    # the SCF stack the gradients
    # build on. Owl: nocturnal,
    # sees what others miss
    # (sub-meV displacement-derived
    # forces).
    "0.7.0": "Löwdin's Compass",  # periodic SCF correctness
    # flagship: linear-dependence
    # diagnostic + screening
    # optimiser + Lehtola pivoted
    # Cholesky + transparent
    # primitive filtering. Per-Olov
    # Löwdin gave us canonical
    # orthogonalisation (Adv. Quantum
    # Chem. 5, 185, 1970) and the
    # symmetric S^{-1/2} that anchors
    # this release's overlap-matrix
    # work. Compass: navigates the
    # basis-set / cutoff space and
    # finds the loosest PSD-keeping
    # configuration without silent
    # truncation. Aligns with
    # CRYSTAL's "refuse to silently
    # truncate; push the problem to
    # basis-set design" philosophy.
    "0.7.1": "Pulay's Triangle",  # cross-code parity layer:
    # automated regression suite
    # (examples/regression/) extended
    # to molecules + DF + open-shell
    # + MP2 + B3LYP/G; vibe-qc, PySCF
    # and ORCA driven from a single
    # spec and Δ-compared per case.
    # Peter Pulay invented DIIS (1980)
    # which makes molecular SCF
    # converge tightly enough for
    # µHa cross-code parity, the
    # Pulay-Vosko RI / density-fitting
    # framework (1990) that anchors
    # the DF layer, and the analytical-
    # gradient theory the suite will
    # exercise next. Triangle: the
    # three-way reference geometry
    # vibe-qc / PySCF / ORCA.
    "0.7.3": "Whitten's Bridge",  # bridge the v0.7.2 DF feature
    # into the molecular regression
    # runner: `density_fit=True` +
    # `aux_basis=...` (with autodetect
    # via `default_aux_basis_for`)
    # threaded through the SCF and
    # post-SCF MP2 paths. John L.
    # Whitten introduced
    # density-fitting / RI in 1973
    # (J. Chem. Phys. 58, 4496),
    # giving the field the trick that
    # makes def2-style RI auxiliary
    # bases tractable. Bridge: the
    # (case, code) wiring from spec to
    # vibe-qc DF kernel that this
    # patch closes -- and the bug
    # discovered by closing it
    # (def2-svp DF SCF SIGSEGVs in
    # `run_rhf`, surfaced cleanly via
    # the v0.7.2 subprocess-isolation
    # layer rather than bricking the
    # whole suite).
    "0.8.0": "Grimme's Gecko",  # molecular methods polish:
    # B2PLYP + DSD-PBEP86-D4 double
    # hybrids, D4 dispersion via dftd4,
    # EEQ atomic-charge model
    # (Caldeweyher/Grimme D4a-i,
    # Apache-2.0 port), direct SCF for
    # all four molecular drivers,
    # RIJCOSX performance closure
    # vs ORCA, Lebedev-Laikov angular
    # grids + Stratmann-Scuseria-Frisch
    # atomic partition, B3LYP-VWN5
    # convention alignment, libecpint
    # banner integration, vqfetch
    # external-data Phase 1. Stefan
    # Grimme's group's lineage runs
    # through every major v0.8.0
    # methods deliverable: D3(BJ) -> D4
    # -> EEQ; the B2PLYP family (2006);
    # DSD-PBEP86 (Kozuch/Martin/Grimme);
    # composite "3c" methods (queued
    # for v0.9.0). Gecko: small, fast,
    # surfaces-stick -- the mol-methods
    # arc that ships an entire class
    # of well-validated post-HF methods
    # built on a clean dispersion +
    # double-hybrid foundation.
    "0.7.2": "Boys' Crucible",  # density-fitting flagship:
    # 2c + 3c integral kernels +
    # DensityFitting object + def2 /
    # cc-pV*Z / Pople-RIFIT aux
    # libraries + DF wired into RHF /
    # UHF / RKS / UKS / MP2 / UMP2
    # energy and J / K analytic
    # gradients. S. F. Boys (1911-1972)
    # introduced Gaussian-type orbitals
    # in 1950 (Proc. R. Soc. A 200, 542),
    # the foundation that makes RI /
    # density-fitting numerically
    # tractable: the exp(-a r^2)
    # primitives admit Gaussian-product
    # 2- and 3-centre overlap and
    # Coulomb integrals in closed form.
    # Crucible: per-case subprocess
    # isolation in the regression
    # suite -- every (case, code) pair
    # runs in its own contained
    # subprocess so a C-level crash
    # (PySCF segfault, ORCA non-zero
    # exit, libxc abort) is captured
    # as a single error row instead of
    # bricking the whole dispatcher.
    "0.9.0": "Knowles's Kingfisher",  # wavefunction-methods cycle:
    # MP2 -> MP3 -> MP4 -> CCD -> CCSD
    # canonical-correlation
    # scaffolding + the full
    # semiempirical-DFTB stack
    # (GFN2-xTB, PM6, OM1/OM2/OM3) +
    # basissetdev integration. Peter
    # J. Knowles is a coupled-cluster
    # / configuration-interaction
    # methods author (the MOLPRO
    # MRCI / direct-CI machinery,
    # Knowles & Handy 1984 FCI
    # algorithm, the Werner-Knowles
    # CASSCF). Kingfisher: precise,
    # patient, strikes once -- the
    # post-HF correlation ladder
    # that v0.14.0+ will build on.
    # Retroactively added to the
    # catalog post-v0.10.0; v0.9.0
    # cut went to a side branch and
    # the entry never landed on main.
    "0.10.0": "Pisani's Penguin",  # periodic-SCF maturity cycle:
    # multi-k EDIIS + Python
    # accelerator family, SYM6
    # crystal symmetry, BIPOLE
    # L_max>=2, GPW M2-full iterative
    # SCF, CRYSTAL-a / Ewald-w gauge
    # unification (cluster-A1 fix at
    # 859efe0a closes +19 Ha
    # translation-invariance break
    # on LiH rocksalt), full DFTB
    # stack + DFT+U Increments 1/2/
    # 4a/4b/4d-light, NEB, PWPB95,
    # periodic D3-BJ. Cesare Pisani
    # founded the CRYSTAL code at
    # Turin (1976-) -- the periodic-
    # SCF lineage vibe-qc inherits.
    # Penguin: lives at extremes
    # (solid-state, structured,
    # low-T). Periodic GDF deferred
    # to v0.11.0 (fourth cycle:
    # v0.8.0 -> v0.9.0 -> v0.10.0 ->
    # v0.11.0).
    "0.11.0": "Sun's Stingray",  # periodic-GDF chain closure:
    # compcell + multi-k + AFT --
    # fourth-cycle defer finally
    # closes (v0.8.0 -> v0.9.0 ->
    # v0.10.0 -> v0.11.0). The
    # Sun-Berkelbach RSGDF
    # (range-separated Gaussian
    # density fitting, Sun et al.
    # JCP 147, 164119, 2017) is the
    # algorithm vibe-qc actually
    # implements: short-range Lpq
    # via Coulomb-metric GDF +
    # long-range via analytic-FT
    # auxiliary plane-wave terms.
    # Qiming Sun also authored
    # PySCF (the µHa-parity
    # reference). Stingray: glides
    # over the periodic landscape
    # via a Fourier transform --
    # k-space agile.
    "0.12.0": "Knuth's Beaver",  # algorithmic density release:
    # DLPNO-CCSD, TDDFT, CASCI/
    # CASSCF, IC-CASPT2, CIS/TDA
    # gradients, MSINDO periodic
    # CCM, metadynamics, MD, MACE
    # MLIP, GPW production surface,
    # BIPOLE analytic gradient
    # gate, MPI parallelisation.
    # Donald Knuth authored The
    # Art of Computer Programming
    # -- algorithmic rigour at the
    # foundation of DLPNO pair-
    # classification, integral
    # transforms, and per-pair
    # solvers. Beaver: industrious
    # builder (nine headline
    # features in a single minor
    # cycle).
    "0.13.0": "Wisesa's Fox",  # generalised regular (GR)
    # k-grids headline; ROHF/ROKS;
    # periodic local correlation
    # (megacell MP2 + DLPNO-CCSD(T));
    # Gilat-Raubenheimer metallic DOS;
    # MSINDO to Xe; BIPOLE<->CRYSTAL
    # parity. Pandu Wisesa: GR-grid
    # first author (2016). Fox:
    # nimble + efficient (~2x MP).
    "0.14.0": "Bartlett's Goose",  # canonical CCSD(T)
    # headline (R/U/ROHF + (T), FNO)
    # + DLPNO-MP2/CCSD(T) (R+U) +
    # VV10/double-hybrids + ROHF/ROKS
    # + periodic guesses/mixers.
    # Rodney Bartlett: CCSD for
    # quantum chemistry (Purvis-
    # Bartlett 1982). PRE-STAGED
    # (decided 2026-06-20); renders
    # only on a 0.14.x build, locked
    # for real at the v0.14.0 cut.
    "0.15.0": "Neese's Cheetah",  # DLPNO,
    # local correlation headline
    # (molecular DLPNO-MP2/CCSD(T)) +
    # eigensolver framework + CASSCF
    # analytic gradients/geom-opt +
    # COOP/COHP + QTAIM + dense-crystal
    # periodic-XC fix. Frank Neese:
    # DLPNO-CCSD(T) (Riplinger-Neese
    # 2013). Cheetah: the speed local
    # correlation buys. PRE-STAGED
    # (decided 2026-06-25); renders
    # only on a 0.15.x build, locked
    # at the v0.15.0 cut.
    "0.16.0": "Pople's Puffin",  # the
    # repository split: one project
    # became vibe-qc, vibe-view,
    # vibe-queue and qvf, each
    # adoptable on its own. John
    # Pople (1925-2004), Nobel 1998:
    # ab initio methods made usable
    # by anyone, which is what
    # splitting these components into
    # adoptable projects is for.
    # Locked at the v0.16.0 cut
    # (2026-09-08).
    "0.17.0": "Tew's Tern",  # the
    # symmetry and periodic
    # foundations the native BvK
    # local-correlation route will
    # consume: shared group/space
    # infrastructure, audited BIPOLE
    # Seitz supports, exact-mesh HF
    # long-range contractions; plus
    # the COSMO molecule-fixed cavity
    # frame and the cc-pVnZ-PP sets.
    # David P. Tew, periodic local
    # correlation (Nejad-Zhu-Tew
    # 2025). Tern: navigates by the
    # lattice it crosses. The route
    # itself is v0.18.0; this tag is
    # the groundwork under it.
    # Locked at the v0.17.0 cut.
}


def _strip_dev_suffix(version: str) -> str:
    """Return the release version a dev build is heading toward, by
    stripping any ``.devN``, ``aN``, ``bN``, ``rcN`` (PEP 440)
    pre-release suffix. ``"0.5.0.dev0"`` -> ``"0.5.0"``."""
    for sep in (".dev", "a", "b", "rc"):
        idx = version.find(sep)
        if idx > 0 and version[idx - 1].isdigit():
            return version[:idx]
    return version


def codename_for_version(version: str) -> str | None:
    """Return the codename for a given vibe-qc release version, or
    ``None`` if no codename is assigned.

    Resolution order:

      1. Strip ``.devN`` / ``aN`` / ``bN`` / ``rcN`` pre-release
         suffix (PEP 440 dev builds inherit the upcoming release's
         codename).
      2. Direct lookup of the resulting ``X.Y.Z`` in
         :data:`RELEASE_CODENAMES`.
      3. Fall back to the parent minor's codename: look up
         ``f"{X}.{Y}.0"``. So v0.4.1 / v0.4.2 / ... automatically
         inherit v0.4.0's codename (matches the policy in
         ``docs/roadmap.md Sec. Release codenames``: "Patch releases
         inherit their parent minor's codename").
      4. ``None`` if no codename is registered for the minor.

    Examples
    --------
    >>> codename_for_version("0.5.0")
    "Wilson's Otter"
    >>> codename_for_version("0.5.0.dev0")
    "Wilson's Otter"
    >>> codename_for_version("0.4.1")          # patch inherits v0.4.0
    "Hellmann's Hedgehog"
    >>> codename_for_version("99.99.99") is None
    True
    """
    stripped = _strip_dev_suffix(version)
    if stripped in RELEASE_CODENAMES:
        return RELEASE_CODENAMES[stripped]
    parts = stripped.split(".")
    if len(parts) >= 2:
        anchor = f"{parts[0]}.{parts[1]}.0"
        if anchor in RELEASE_CODENAMES:
            return RELEASE_CODENAMES[anchor]
    return None


def _compute_version() -> str:
    """Read the installed distribution metadata. The distribution is
    named ``vibe-qc`` in ``pyproject.toml`` since the rename; older
    local installs may still be listed as the legacy ``vibeqc``, so we
    try both."""
    if _pkg_version is None:
        return "0.0.0+unknown"
    for dist_name in ("vibe-qc", "vibeqc"):
        try:
            return _pkg_version(dist_name)
        except PackageNotFoundError:
            continue
    return "0.0.0+unknown"


VIBEQC_VERSION: str = _compute_version()


def _blas_backend_label(libraries: str) -> str:
    """Map the raw BLAS_LIBRARIES string from CMake's FindBLAS to a
    short human-readable backend label for the banner.

    Examples:
        "-framework Accelerate"            -> "Accelerate"
        "/usr/lib/libopenblas.so"          -> "OpenBLAS"
        "/usr/lib/libblas.so.3.12.0"       -> "netlib BLAS"
        "/opt/intel/.../libmkl_rt.so"      -> "MKL"
        ""                                 -> "none"

    The label is informational -- it falls back to the raw library
    name when no pattern matches, so a build against an unexpected
    BLAS still gets a meaningful banner line.
    """
    if not libraries:
        return "none"
    haystack = libraries.lower()
    # Order matters: more specific patterns first.
    if "accelerate" in haystack:
        return "Accelerate"
    if "mkl" in haystack:
        return "MKL"
    if "openblas" in haystack:
        return "OpenBLAS"
    if "blis" in haystack:
        return "BLIS"
    if "atlas" in haystack:
        return "ATLAS"
    if "libblas" in haystack or "/blas" in haystack:
        return "netlib BLAS"
    # Unknown backend -- return the basename so the user sees *something*
    # concrete in the banner instead of "unknown".
    first = libraries.split(";")[0].strip()
    if first.startswith("-framework "):
        return first[len("-framework ") :]
    return os.path.basename(first) or first


def library_versions() -> Mapping[str, str]:
    """Return the version strings of vibe-qc itself and its native
    dependencies. Keys are always present; values are ``"unknown"``
    if a probe fails, ``"none"`` when the dependency was not linked
    at build / install time.

    Core (always linked at build time): ``libint``, ``libxc``,
    ``spglib``, ``libecpint``, ``fftw3``, ``blas``.

    Optional (linked at runtime only when the ``[dispersion]``
    extra is installed): ``dftd3``, ``dftd4``. Each is set to
    ``"none"`` when the corresponding PyPI package isn't installed
    in the user's environment -- distinguishes "not installed" from
    "probe failed" (the latter would be ``"unknown"``).

    The ``"vibe-qc"`` key carries the project version. The legacy
    ``"vibeqc"`` key is kept as an alias so code that predates the
    rename keeps working without change.

    The ``"blas"`` key carries a short backend label (Accelerate /
    OpenBLAS / MKL / netlib BLAS / ...) decorated with ``" +LAPACKE"``
    when EIGEN_USE_LAPACKE is enabled, so the banner conveys both
    the linkage and whether dense solvers delegate to LAPACK.
    """
    versions: dict[str, str] = {
        "vibe-qc": VIBEQC_VERSION,
        "vibeqc": VIBEQC_VERSION,  # alias, retained for back-compat
    }

    # Probe the compiled extension. Each of the libraries exposes a
    # version accessor; every version is known at link time.
    try:
        from . import _vibeqc_core as _core
    except ImportError:
        versions.update(
            {
                "libint": "unknown",
                "libxc": "unknown",
                "spglib": "unknown",
                "fftw3": "unknown",
                "blas": "unknown",
            }
        )
        return versions

    try:
        versions["libint"] = _core.libint_version()
    except Exception:
        versions["libint"] = "unknown"

    try:
        versions["libxc"] = getattr(_core, "libxc_version", lambda: "unknown")()
    except Exception:
        versions["libxc"] = "unknown"

    try:
        versions["spglib"] = _core.spglib_version()
    except Exception:
        versions["spglib"] = "unknown"

    # libecpint version probe -- added with the v0.8.0 banner / libecpint
    # coupled fix (prep doc Sec. 488-501). libecpint has been bundled since
    # v0.4.0 (commit add8163) but the banner didn't surface its version
    # until v0.8.0. Defensive: an older _core lacking libecpint_version
    # falls back to "unknown" rather than failing the whole library
    # probe.
    try:
        versions["libecpint"] = getattr(_core, "libecpint_version", lambda: "unknown")()
    except Exception:
        versions["libecpint"] = "unknown"

    # FFTW3 version probe -- FFTW3 has been a required build dep since
    # the FFT-Poisson Ewald long-range Hartree solver landed; the GAPW
    # route (v0.10.x) is the second consumer. Banner coverage added per
    # CLAUDE.md Sec. 6 alongside the M1 PW-basis infrastructure. Defensive
    # getattr: an older _core lacking fftw3_version falls back to
    # "unknown".
    try:
        versions["fftw3"] = getattr(_core, "fftw3_version", lambda: "unknown")()
    except Exception:
        versions["fftw3"] = "unknown"

    # BLAS info was added with the BLAS+LAPACK linkage commit; an
    # older C extension predating that change won't have blas_info.
    # Treat that as "unknown" so the banner remains parseable across
    # builds.
    try:
        _info = _core.blas_info()
        if not _info.get("blas_enabled"):
            versions["blas"] = "none"
        else:
            label = _blas_backend_label(_info.get("libraries", ""))
            if _info.get("lapacke_enabled"):
                label = f"{label} +LAPACKE"
            versions["blas"] = label
    except Exception:
        versions["blas"] = "unknown"

    # Optional dispersion backends (dftd3, dftd4). Both ship as PyPI
    # wheels with bundled binary libraries (libdftd3 / libdftd4) that
    # vibe-qc links to at runtime via ctypes/cffi when the user calls
    # vibeqc.compute_d3 / vibeqc.compute_d4. CLAUDE.md Sec. 6 mandates
    # banner coverage for every linked native dep; these are linked
    # *optionally* (extra `[dispersion]`) so the value is "none" when
    # the package is absent rather than "unknown" -- distinguishes
    # "not installed" from "failed to probe".
    for pkg in ("dftd3", "dftd4"):
        try:
            module = __import__(pkg)
            versions[pkg] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[pkg] = "none"
        except Exception:
            versions[pkg] = "unknown"

    # Optional MLIP stack (the `[mace]` extra): PyTorch + ACEsuit MACE +
    # e3nn. Not vendored native libs, but heavy runtime deps that produce
    # user-facing energies via method="mace", so the banner records their
    # versions for provenance (CLAUDE.md Sec. 6). Query distribution metadata
    # instead of importing these packages: importing torch only to render the
    # banner can load an OpenMP runtime before the MACE adapter installs its
    # macOS process safeguard. The import name ``mace`` belongs to the
    # ``mace-torch`` distribution.
    mlip_distributions = {
        "torch": "torch",
        "mace": "mace-torch",
        "e3nn": "e3nn",
    }
    for pkg, distribution in mlip_distributions.items():
        try:
            versions[pkg] = str(_pkg_version(distribution))
        except PackageNotFoundError:
            versions[pkg] = "none"
        except Exception:
            versions[pkg] = "unknown"

    return versions


# --- Build-provenance probe -----------------------------------------------
#
# When vibe-qc runs from an editable install or directly from a source
# checkout, we want the banner to record exactly which git revision is
# in play -- branch name, short SHA, dirty-flag, and exact-tag (for
# tagged releases). When running from a wheel that has no ``.git/``
# alongside it, all of these probes return None and the banner falls
# back to the package version alone.
#
# This matters because users will run calculations against specific
# branches for testing (``release``, ``main``, or topic branches), and
# the only way to match a misbehaving SCF log to the build that
# produced it is to record the SHA in the log itself.
#
# Probes are cached at module-import time to avoid repeated subprocess
# calls; ``build_info.cache_clear()`` is available for tests.


def _find_git_root(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a ``.git`` directory or
    ``.git`` worktree-link file. Returns the parent directory of the
    first match, or None if we hit the filesystem root."""
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return None


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command in ``repo`` and return stripped stdout, or
    None on any failure (no git installed, broken repo, command
    rejected). Never raises."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return out.strip() or None
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        return None


@lru_cache(maxsize=1)
def build_info() -> Mapping[str, str | bool | None]:
    """Return git provenance for the current build, or empty mapping if
    no ``.git/`` is available (e.g. installed wheel).

    Keys (always present, may be None / False):

    - ``branch``  : current branch name, or ``"HEAD"`` for detached.
    - ``sha``     : 7-character abbreviated commit SHA.
    - ``sha_full``: full 40-character SHA.
    - ``dirty``   : True if the working tree has uncommitted changes.
    - ``tag``     : exact tag at HEAD, e.g. ``"v0.4.1"``; None if not
                    on a tag.
    - ``is_release``: True if the current commit is exactly tagged AND
                      the working tree is clean. This is the only state
                      where the banner displays as "Release X.Y.Z".

    Empty dict (``{}``) when no git context exists at all.
    """
    repo = _find_git_root(Path(__file__).parent)
    if repo is None:
        return {}

    # Allow CI / packaging to override these values without spawning git.
    # Useful when building wheels from a tarball where .git is absent.
    env_branch = os.environ.get("VIBEQC_BUILD_BRANCH")
    env_sha = os.environ.get("VIBEQC_BUILD_SHA")
    env_tag = os.environ.get("VIBEQC_BUILD_TAG")

    sha_full = env_sha or _git(repo, "rev-parse", "HEAD")
    sha = sha_full[:7] if sha_full else None

    branch = env_branch or _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        # Detached HEAD: try to identify by tag or short ref name.
        branch = _git(repo, "describe", "--all", "--exact-match") or "HEAD"

    tag = env_tag
    if tag is None:
        # ``--exact-match`` returns non-zero when the current commit isn't
        # tagged; _git swallows that and returns None, which is the
        # correct "not a release" signal.
        tag = _git(repo, "describe", "--tags", "--exact-match")

    dirty_status = _git(repo, "status", "--porcelain")
    dirty = bool(dirty_status)

    is_release = bool(tag) and not dirty

    return {
        "branch": branch,
        "sha": sha,
        "sha_full": sha_full,
        "dirty": dirty,
        "tag": tag,
        "is_release": is_release,
    }


_EXPECT_VERSION_ENV = "VIBEQC_EXPECT_VERSION"
_EXPECT_SHA_ENV = "VIBEQC_EXPECT_GIT_SHA"
_EXPECT_BRANCH_ENV = "VIBEQC_EXPECT_GIT_BRANCH"


def _pin_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _sha_pin_matches(expected: str, actual_full: object, actual_short: object) -> bool:
    exp = expected.strip().lower()
    if len(exp) < 7:
        return False
    for candidate in (actual_full, actual_short):
        if not candidate:
            continue
        cand = str(candidate).strip().lower()
        if not cand or cand == "unknown":
            continue
        if len(exp) <= len(cand) and cand.startswith(exp):
            return True
    return False


def enforce_runtime_pin_from_env() -> None:
    """Fail early when the current build does not match an opt-in pin.

    The guard is intentionally environment-driven so release-paper and
    validation queues can pin an exact runtime without changing the public
    ``run_job`` / ``run_periodic_job`` signatures. If none of the pin
    variables are set, this function is a no-op.

    Supported variables:

    - ``VIBEQC_EXPECT_VERSION``: exact package version, e.g. ``0.15.1``.
    - ``VIBEQC_EXPECT_GIT_SHA``: commit prefix of at least seven characters.
    - ``VIBEQC_EXPECT_GIT_BRANCH``: exact branch name, e.g. ``release``.
    """
    expected_version = _pin_env(_EXPECT_VERSION_ENV)
    expected_sha = _pin_env(_EXPECT_SHA_ENV)
    expected_branch = _pin_env(_EXPECT_BRANCH_ENV)
    if not (expected_version or expected_sha or expected_branch):
        return

    info = build_info()
    actual_branch = info.get("branch") or "unknown" if info else "unknown"
    actual_sha = info.get("sha") or "unknown" if info else "unknown"
    actual_sha_full = info.get("sha_full") or "unknown" if info else "unknown"

    mismatches: list[str] = []
    if expected_version is not None and expected_version != VIBEQC_VERSION:
        mismatches.append(
            f"{_EXPECT_VERSION_ENV}={expected_version!r}, "
            f"runtime version is {VIBEQC_VERSION!r}"
        )
    if expected_branch is not None and expected_branch != actual_branch:
        mismatches.append(
            f"{_EXPECT_BRANCH_ENV}={expected_branch!r}, "
            f"runtime branch is {actual_branch!r}"
        )
    if expected_sha is not None:
        if len(expected_sha.strip()) < 7:
            mismatches.append(
                f"{_EXPECT_SHA_ENV} must contain at least 7 characters "
                f"(got {expected_sha!r})"
            )
        elif not _sha_pin_matches(expected_sha, actual_sha_full, actual_sha):
            mismatches.append(
                f"{_EXPECT_SHA_ENV}={expected_sha!r}, "
                f"runtime git SHA is {actual_sha!r}"
            )
        if info and bool(info.get("dirty")):
            mismatches.append(
                f"{_EXPECT_SHA_ENV} is set but the runtime checkout is dirty"
            )

    if mismatches:
        detail = "; ".join(mismatches)
        raise RuntimeError(
            "vibe-qc runtime pin mismatch: "
            f"{detail}. Refusing to start the calculation; unset or correct "
            "the VIBEQC_EXPECT_* environment variables for this job."
        )


_COPYRIGHT_LINE = "© Michael F. Peintinger . MPL 2.0  .  https://vibe-qc.com"


def _version_descriptor() -> str:
    """Compose the line that identifies the current build:

    * Tagged release, clean tree    ->  "Release v0.5.0 \"Wilson's Otter\""
    * Dev build, branch + sha       ->  "dev 0.5.0.dev0 \"Wilson's Otter\" (main @ abc1234)"
    * Dev build, dirty tree         ->  "dev 0.5.0.dev0 \"Wilson's Otter\" (main @ abc1234, dirty)"
    * Wheel install (no git info)   ->  "vibe-qc 0.5.0 \"Wilson's Otter\""
    * Pre-codename releases         ->  no quoted codename suffix
      (e.g. "Release v0.4.1", "dev 0.4.0.dev0 (main @ abc1234)")

    The codename is bound to the release version; dev builds against
    an upcoming release inherit its codename so the banner is fun
    even on a working tree. See ``_RELEASE_CODENAMES``.

    Generated example artifacts are no longer committed, so release
    labeling relies only on the exact tag / clean tree state reported
    by :func:`build_info`.
    """
    info = build_info()
    codename = codename_for_version(VIBEQC_VERSION)
    cn_suffix = f' "{codename}"' if codename else ""

    if not info:
        return f"vibe-qc {VIBEQC_VERSION}{cn_suffix}"

    if info.get("is_release"):
        return f"Release {info['tag']}{cn_suffix}"

    branch = info.get("branch") or "HEAD"
    sha = info.get("sha") or "unknown"
    suffix = ", dirty" if info.get("dirty") else ""
    return f"dev {VIBEQC_VERSION}{cn_suffix} ({branch} @ {sha}{suffix})"


def banner(*, width: int = 72) -> str:
    """Return the full banner as a single string, no trailing newline.

    ``width`` controls the horizontal extent; defaults to a terminal-
    friendly 72 characters. The banner is pure ASCII + box-drawing
    characters so it renders cleanly in any modern UTF-8 terminal.

    .. note::

       ``vibeqc/__init__.py`` re-exports this function, so on the package
       root the name ``banner`` is **this function**, not the
       :mod:`vibeqc.banner` submodule. ``from vibeqc import banner``
       followed by ``getattr(banner, "build_info")`` therefore finds
       nothing rather than raising -- a silent failure that has cost real
       runtime-identity probes. Every public name of this module is also
       re-exported on the package root, so prefer ``vibeqc.build_info()``
       over reaching for the submodule; use
       ``importlib.import_module("vibeqc.banner")`` only if you genuinely
       need the module object itself.
    """
    versions = library_versions()
    libs = (
        f"libint {versions['libint']} . "
        f"libxc {versions['libxc']} . "
        f"spglib {versions['spglib']} . "
        f"libecpint {versions['libecpint']} . "
        f"fftw3 {versions['fftw3']} . "
        f"blas {versions['blas']}"
    )
    # Optional dispersion backends. Hidden when both are absent --
    # users without the `[dispersion]` extra installed see the
    # unchanged classic banner. When at least one is installed the
    # banner gains a second linkage line.
    disp_parts = []
    if versions.get("dftd3", "none") != "none":
        disp_parts.append(f"dftd3 {versions['dftd3']}")
    if versions.get("dftd4", "none") != "none":
        disp_parts.append(f"dftd4 {versions['dftd4']}")
    disp_line = " . ".join(disp_parts) if disp_parts else None

    # Optional MLIP stack (the `[mace]` extra). Hidden when absent -- users
    # without `[mace]` see the unchanged banner.
    mlip_parts = []
    if versions.get("mace", "none") != "none":
        mlip_parts.append(f"mace {versions['mace']}")
    if versions.get("torch", "none") != "none":
        mlip_parts.append(f"torch {versions['torch']}")
    if versions.get("e3nn", "none") != "none":
        mlip_parts.append(f"e3nn {versions['e3nn']}")
    mlip_line = " . ".join(mlip_parts) if mlip_parts else None

    descriptor = _version_descriptor()
    header = f"{descriptor}  --  Quantum chemistry for molecules and solids"
    libs_line = f"linked: {libs}"

    # Adjust width if any line is wider than requested -- banner should
    # never truncate content. Compose each row's full text BEFORE the
    # width calc so prefixes ("linked: ", "dispersion: ") are
    # accounted for. Before v0.8.1: width used `len(libs) + 4` but the
    # actual row is f"linked: {libs}" (8 chars longer). When libs was
    # the longest source, width was undersized by 8 -> _row pad went
    # negative -> closing ║ jammed up against the last char on the libs
    # row. Surfaced once libecpint joined the linked: line and OpenBLAS
    # +LAPACKE pushed libs past header in length on typical builds.
    row_widths = [
        len(header) + 4,
        len(_COPYRIGHT_LINE) + 4,
        len(libs_line) + 4,
    ]
    if disp_line is not None:
        row_widths.append(len(f"dispersion: {disp_line}") + 4)
    if mlip_line is not None:
        row_widths.append(len(f"mlip: {mlip_line}") + 4)
    width = max(width, *row_widths)

    bar = "═" * (width - 2)
    top = f"╔{bar}╗"
    bot = f"╚{bar}╝"

    def _row(text: str) -> str:
        pad = (width - 2) - len(text)
        return f"║ {text}{' ' * (pad - 1)}║"

    rows = [
        top,
        _row(header),
        _row(_COPYRIGHT_LINE),
        _row(libs_line),
    ]
    if disp_line is not None:
        rows.append(_row(f"dispersion: {disp_line}"))
    if mlip_line is not None:
        rows.append(_row(f"mlip: {mlip_line}"))
    rows.append(bot)
    return "\n".join(rows)


def print_banner(*, file: TextIO = sys.stdout, width: int = 72) -> None:
    """Print the banner to ``file`` (default: stdout)."""
    print(banner(width=width), file=file)
