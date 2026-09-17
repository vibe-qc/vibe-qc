"""vibe-qc -- quantum chemistry code.

Milestone 1 scope: restricted Hartree-Fock total energy for a single
closed-shell molecule. The cyclic cluster model for periodic systems is
the longer-term target.
"""

from __future__ import annotations

import logging as _logging
import os as _os
import sys as _sys
import time as _time
import types as _types
from pathlib import Path as _Path

import numpy as np  # used by the +U gradient wrappers (compute_gradient*)

# Attach a NullHandler to the top-level logger so the library is silent
# unless the user explicitly configures logging. Standard practice.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())


def _pin_blas_threads() -> None:
    """Default BLAS-internal threading to 1 to avoid nested
    oversubscription with vibe-qc's OpenMP parallelism.

    Why: the C++ core threads its inner loops with OpenMP (controlled
    by ``OMP_NUM_THREADS``). If the linked BLAS (OpenBLAS-pthreads,
    MKL, BLIS, Apple Accelerate) *also* threads internally, you get
    NxK threads contending for cores -- measurably worse than either
    layer alone. Standard practice in mixed OpenMP+BLAS codes is to
    keep BLAS single-threaded and let the outer parallelism drive
    occupancy. These env vars are read at library-init time, so we
    set them *before* anything imports the C extension.

    User overrides are respected: ``os.environ.setdefault`` only fills
    a key that isn't already set, so an explicit
    ``OPENBLAS_NUM_THREADS=4`` from the user's shell wins.

    The set of probed vars covers the BLAS implementations we care
    about: OpenBLAS, MKL, Apple Accelerate (vecLib), and BLIS.
    """
    for _key in (
        "OPENBLAS_NUM_THREADS",  # OpenBLAS (Arch / Fedora / Debian / Homebrew)
        "MKL_NUM_THREADS",  # Intel MKL
        "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (vecLib)
        "BLIS_NUM_THREADS",  # BLIS
    ):
        _os.environ.setdefault(_key, "1")


_pin_blas_threads()


def _resolve_basis_library(package_dir: _Path | None = None) -> _Path | None:
    """Pick the on-disk basis-library directory libint should use.

    Implements branches 2-4 of the resolution order documented in
    :func:`_install_basis_library` (branch 1, the user's own
    ``$LIBINT_DATA_PATH``, is the caller's concern). Returns ``None``
    when no candidate has the standard bundled payload. ``package_dir``
    exists so tests can point the resolution at a synthetic layout.
    """
    here = package_dir if package_dir is not None else _Path(__file__).resolve().parent
    # Build-output overlay: populate writes here; preferred when it exists.
    build_dir = here.parent.parent / "build" / "basis_library"
    if _basis_library_has_standard_payload(build_dir):
        _warn_if_basis_overlay_is_stale(build_dir, here / "basis_library")
        return build_dir
    # Bundled with the wheel / package.
    bundled = here / "basis_library"
    if _basis_library_has_standard_payload(bundled):
        return bundled
    # Legacy editable-install layout (basis_library/ at the repo root).
    legacy = here.parent.parent / "basis_library"
    if _basis_library_has_standard_payload(legacy):
        return legacy
    return None


def _basis_library_has_standard_payload(root: _Path | str | None) -> bool:
    """Return True when ``root`` has vibe-qc's full basis-library payload."""
    if root is None:
        return False
    try:
        basis_dir = _Path(root) / "basis"
    except TypeError:
        return False
    # ``def2-svp`` is a release-paper canary for the bundled libint +
    # custom-library payload; ``6-311+g3df2p`` and ``pob-tzvp-rev2``
    # keep stale build/deploy overlays from shadowing newer committed
    # custom bases.
    required = ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94")
    return all((basis_dir / name).is_file() for name in required)


def _warn_if_basis_overlay_is_stale(overlay: _Path, bundled: _Path) -> None:
    """Warn when the build overlay is older than the committed library.

    ``setup_native_deps.sh`` assembles ``build/basis_library`` once from
    libint's payload plus ``custom/``, and nothing invalidates it when the
    committed library changes -- ``pip install -e .`` does not regenerate
    it. Because the overlay is *preferred* by the resolution above, a clone
    whose overlay predates a basis-data commit silently resolves the older
    library. That is how ``def2-svp`` lost silver on a loop-runner clone
    after 6664da63d (#88) added it: eleven guess tests died in BasisSet
    construction, and the seven negative ECP-contract cases among them were
    indistinguishable from a real refusal regression.

    Two checks, because they catch different failures, reported together in
    one warning so that neither hides the other.

    *Inventory*: a basis the overlay simply does not have. A newly added
    basis changes no existing file, so a content comparison cannot see it --
    which is exactly what happened when the sixteen cc-pVnZ-PP sets landed
    (2026-09-08) and every clone's overlay kept resolving without them. The
    symptom is ``BasisSet: no shells loaded for basis '...'`` naming a basis
    that is plainly committed. Only *missing* names are reported: the overlay
    legitimately merges libint's payload with ``custom/``, so extras are not
    evidence of staleness.

    *Content*: the same file present in both but different, over three
    orbital canaries and every committed ``.ecp`` sidecar. Compared by
    bytes, not size: #207 (8826f98) corrected the radial powers in
    ``pob-tzvp-rev2.ecp`` without changing its 27,983-byte length, so a
    size comparison kept a 40.9 Ha-wrong ECP operator in a stale overlay
    with no warning (#235). Sizes are still checked first, so bytes are
    read only for files whose sizes agree, and a pair that cannot be read
    is skipped on its own rather than ending the pass for the files after
    it.

    Both are cheap enough for import: two directory listings, then about
    1.5 MB read from each tree when everything matches (about 3 ms cold
    against 1.3 ms for the size-only check, of a ~460 ms import). The
    overlay is still *used* -- the bundled tree alone may not carry every
    libint base, so silently switching could lose bases rather than gain
    them. The defect being fixed is the silence, not the preference.
    """
    try:
        overlay_basis = overlay / "basis"
        bundled_basis = bundled / "basis"
        if not (overlay_basis.is_dir() and bundled_basis.is_dir()):
            return
        have = {q.name for q in overlay_basis.iterdir()}
        want = {q.name for q in bundled_basis.iterdir()}
    except OSError:
        return
    absent = sorted(want - have)

    names = ["def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94"]
    # Every ECP sidecar: an in-place correction to one of them is the #235
    # blind spot, and the whole set is about 1.3 MB.
    names += sorted(n for n in want if n.endswith(".ecp") and n not in names)
    differing: list[str] = []
    for name in names:
        if name not in have or name not in want:
            continue  # the inventory line covers what is missing
        a = overlay_basis / name
        b = bundled_basis / name
        try:
            if not a.is_file() or not b.is_file():
                continue
            if a.stat().st_size == b.stat().st_size:
                if a.read_bytes() == b.read_bytes():
                    continue
        except OSError:
            # One unreadable pair says nothing about the others; a stale
            # sidecar sorted after it must still be found.
            continue
        differing.append(name)

    if not absent and not differing:
        return
    import warnings as _warnings

    def _shown(items: list[str]) -> str:
        more = f" (+{len(items) - 4} more)" if len(items) > 4 else ""
        return ", ".join(items[:4]) + more

    lines = [
        "vibe-qc basis library: the build overlay is out of date and is "
        "being used in preference to the committed one.",
        f"  overlay (used):    {overlay_basis}",
        f"  committed (newer): {bundled_basis}",
    ]
    if absent:
        lines.append(
            f"  {len(absent)} file(s) missing from the overlay: {_shown(absent)}"
        )
    if differing:
        lines.append(
            f"  {len(differing)} file(s) present in both with different "
            f"content: {_shown(differing)}"
        )
        lines.append(f"    overlay (used):    {overlay_basis / differing[0]}")
        lines.append(f"    committed (newer): {bundled_basis / differing[0]}")
    lines.append("Regenerate with:  bash scripts/setup_basis_library.sh")
    lines.append(
        "Until then, missing bases do not resolve at all (the error names "
        "the basis rather than the stale overlay), and bases the overlay "
        "does have may be older than the committed ones. Nothing else warns "
        "about this."
    )
    _warnings.warn("\n".join(lines), RuntimeWarning, stacklevel=3)


def _basis_library_has_basis_dir(root: _Path | str | None) -> bool:
    if root is None:
        return False
    try:
        return (_Path(root) / "basis").is_dir()
    except TypeError:
        return False


def _install_basis_library() -> None:
    """Point libint at vibe-qc's bundled basis library.

    libint looks at ``$LIBINT_DATA_PATH/basis/<name>.g94`` at BasisSet
    construction time. The bundled basis library ships *inside* the
    Python package (``python/vibeqc/basis_library/``) so a fresh
    ``pip install -e .`` from the vibe-qc checkout already has every
    standard libint basis plus vibe-qc's custom additions (pob-tzvp,
    pob-dzvp-rev2, pob-tzvp-rev2) immediately -- no separate setup
    step, no ``LIBINT_DATA_PATH`` fiddling.

    Resolution order:

      1. ``$LIBINT_DATA_PATH`` already set by user and carrying the
         standard vibe-qc payload -> respect it (user override wins).
         An existing but incomplete libint-stock ``basis/`` directory
         is ignored so it does not shadow the bundled def2/cc/pob
         libraries. A nonexistent/bogus path remains user-owned so
         BasisSet construction can raise with that path in the message.
      2. ``build/basis_library/basis/`` -- populate output from
         ``scripts/setup_basis_library.sh`` (build-time overlay,
         gitignored). Preferred over the committed copy when
         present because it reflects the current libint install
         + custom/ state.
      3. ``python/vibeqc/basis_library/basis/`` (bundled with the
         wheel) -> fallback.
      4. ``<repo-root>/basis_library/basis/`` (legacy editable-install
         layout, kept for backwards compatibility while users migrate).
    """
    if "LIBINT_DATA_PATH" in _os.environ:
        current = _os.environ.get("LIBINT_DATA_PATH")
        if (
            _basis_library_has_standard_payload(current)
            or not _basis_library_has_basis_dir(current)
        ):
            return
    chosen = _resolve_basis_library()
    if chosen is not None:
        _os.environ["LIBINT_DATA_PATH"] = str(chosen)


_install_basis_library()


def _resolve_bundled_ecp_share_dir() -> str:
    """Locate the directory containing libecpint's XML basis library.

    libecpint's :func:`ECPIntegrator::set_ecp_basis_from_library` takes
    a ``share_dir`` argument and internally appends ``"/xml/" + name +
    ".xml"`` -- so the path returned here is the parent of ``xml/``,
    not ``xml/`` itself.

    Resolution order:

      1. ``$VIBEQC_ECP_SHARE_DIR`` if set (user override).
      2. ``python/vibeqc/ecp_library/`` bundled with the wheel --
         contains the libecpint 1.0.7 XML library
         (ecp10mdf, ecp28mdf, ecp46mdf, ecp60mdf, ecp78mdf, lanl2dz).
      3. ``<repo-root>/third_party/libecpint/install/share/libecpint``
         (legacy editable-install path, kept while we migrate).
      4. Empty string -- caller must pass ``share_dir`` explicitly.

    Returned in plain string form for direct hand-off to
    :func:`compute_ecp_matrix`.
    """
    if "VIBEQC_ECP_SHARE_DIR" in _os.environ:
        return _os.environ["VIBEQC_ECP_SHARE_DIR"]
    here = _Path(__file__).resolve().parent
    bundled = here / "ecp_library"
    if (bundled / "xml" / "ecp10mdf.xml").is_file():
        return str(bundled)
    legacy = (
        here.parent.parent
        / "third_party"
        / "libecpint"
        / "install"
        / "share"
        / "libecpint"
    )
    if (legacy / "xml" / "ecp10mdf.xml").is_file():
        return str(legacy)
    return ""


_VIBEQC_ECP_SHARE_DIR = _resolve_bundled_ecp_share_dir()

# Tell the C++ side where to find the bundled XML library too --
# ``run_rhf`` / ``run_uhf`` / ``run_rks`` paths call
# ``compute_ecp_matrix`` internally with an empty ``share_dir`` and
# rely on the build-time ``VIBEQC_LIBECPINT_SHARE_DIR`` macro.
# That macro doesn't exist on a wheel install (the path was relative
# to the build host's third_party/libecpint/install/), so the C++
# side reads ``$VIBEQC_ECP_SHARE_DIR`` at runtime as the bundled
# fallback. Setting it from Python before any C++ ECP entry runs
# is the simplest cross-language plumbing.
if _VIBEQC_ECP_SHARE_DIR and "VIBEQC_ECP_SHARE_DIR" not in _os.environ:
    _os.environ["VIBEQC_ECP_SHARE_DIR"] = _VIBEQC_ECP_SHARE_DIR

# Re-register the compiled core's pybind11 submodules in ``sys.modules``
# on every package (re-)import. ``def_submodule`` registers the dotted
# names (``vibeqc._vibeqc_core.semiempirical`` plus its ``.nddo`` /
# ``.xtb`` / ``.indo`` children) only while the extension's one-time
# ``PyInit`` runs; CPython caches the single-phase-init extension, so
# after any ``sys.modules['vibeqc*']`` purge (test-isolation shims, e.g.
# tests/basisset_dev/) a plain re-import would never restore them and
# the dotted import form
# (``from vibeqc._vibeqc_core.semiempirical import nddo``) would raise
# ModuleNotFoundError for the rest of the process.
#
# The plain ``from . import`` form below also (re-)binds the package
# ATTRIBUTE ``vibeqc._vibeqc_core``: when a purge kept the extension's
# ``sys.modules`` entry, the cache-hit re-import skips the loader step
# that normally setattrs the submodule onto the new package object, so
# without this binding attribute-chain access
# (``vibeqc._vibeqc_core.compute_eri``) would raise AttributeError even
# though from-imports still resolve via the sys.modules fallback.
from . import _vibeqc_core


def _register_compiled_submodules(_mod: _types.ModuleType) -> None:
    for _value in vars(_mod).values():
        if isinstance(_value, _types.ModuleType) and _value.__name__.startswith(
            _mod.__name__ + "."
        ):
            _sys.modules.setdefault(_value.__name__, _value)
            _register_compiled_submodules(_value)


_register_compiled_submodules(_vibeqc_core)

# Semiempirical module is imported lazily to avoid circular imports.
# Use ``from vibeqc import semiempirical`` or ``import vibeqc.semiempirical``.
# The module is attached below after all names are bound.
from ._vibeqc_core import (
    ADIIS,
    DIIS,
    DIISDepthPolicy,
    EDIIS,
    KDIIS,
    Atom,
    AtomicGridProfile,
    BandDiag,
    BasisSet,
    BlochKMesh,
    CCSDIteration,
    CCSDResult,
    CosxVariant,
    CoulombMethod,
    Crystal,
    D3BJParams,
    DavidsonOptions,
    DipoleIntegrals,
    DispersionResult,
    ECPCenter,
    ECPPrimitiveBlock,
    EwaldOptions,
    ExternalChargeGradient,
    Functional,
    GradientOptions,
    Grid,
    GridOptions,
    InitialGuess,
    IrreducibleKMesh,
    JKBuilder,
    JKMatrices,
    LatticeCell,
    LatticeMatrixSet,
    LatticeSumOptions,
    LOBPCGOptions,
    Molecule,
    MP2Options,
    MP2Result,
    OpenTrustRegionOptions,
    has_opentrustregion,
    NewtonOptions,
    PairJKContribution,
    PeriodicKSOptions,
    PeriodicKSResult,
    PeriodicRHFOptions,
    PeriodicRHFResult,
    PeriodicSCFOptions,
    PeriodicSystem,
    PeriodicXCDensityDomain,
    PeriodicUKSXCContribution,
    PeriodicXCContribution,
    RHFOptions,
    RHFResult,
    RKSOptions,
    RKSResult,
    SCFAccelerator,
    SCFIteration,
    SCFMode,
    ShellInfo,
    SOSCFOptions,
    SCFRestartOptions,
    SpaceGroup,
    SpinlockMode,
    SymmetryOp,
    TRAHOptions,
    UHFOptions,
    UHFResult,
    UHFXCKernelBuilder,
    UKSOptions,
    UKSResult,
    UMP2Options,
    UMP2Result,
    XCKernelBuilder,
    XCKind,
    analyze_symmetry,
    attach_symmetry,
    bloch_sum,
    build_cosx_q,
    build_cosx_schwarz,
    build_fock_2e_real_space,
    build_grid,
    build_jk_2e_real_space,
    build_jk_2e_real_space_explicit,
    build_jk_gamma_molecular_limit,
    build_jk_gamma_molecular_limit_explicit,
    build_jk_pair_contributions,
    build_xc_periodic,
    build_xc_periodic_uks,
    compute_2c_eri,
    compute_2c_eri_gradient_contribution,
    compute_2c_eri_gradient_weighted,
    compute_2c_eri_lattice,
    compute_2c_eri_lattice_blocks,
    compute_2c_eri_lattice_sr,
    compute_3c_eri,
    compute_3c_eri_gradient_contribution,
    compute_3c_eri_gradient_weighted,
    compute_3c_eri_lattice,
    compute_3c_eri_lattice_blocks,
    compute_3c_eri_lattice_sr,
    compute_b97m_semilocal_exc,
    compute_b97m_semilocal_vxc,
    compute_cosx_k,
    compute_cosx_k_gradient_contribution,
    compute_df_j_gradient,
    compute_df_jk_gradient,
    compute_df_k_gradient,
    compute_dipole,
    compute_eri,
    compute_external_charge_density_gradient,
    compute_kinetic,
    compute_kinetic_lattice,
    compute_kinetic_lattice_explicit,
    compute_nuclear,
    compute_nuclear_erfc_lattice,
    compute_nuclear_lattice,
    compute_nuclear_lattice_ewald,
    compute_nuclear_lattice_explicit,
    compute_nuclear_with_charges,
    compute_overlap,
    compute_overlap_lattice,
    compute_overlap_lattice_explicit,
    compute_overlap_two_basis,
    compute_robust_cosx_k,
    compute_vv10,
    cosx_basis_cardinality_from_name,
    cosx_grid_options_for_level,
    cosx_grid_stages_for_level,
    d3_coordination_numbers,
    d3_r2r4,
    d3_rcov,
    diagonalize_bloch,
    direct_lattice_cells,
    pair_complete_lattice_cells,
    eri_exponent_derivative,
    evaluate_ao,
    evaluate_ao_with_gradient,
    evaluate_ao_with_hessian,
    evaluate_bloch_ao,
    ewald_2d_point_charge_energy,
    ewald_2d_point_charge_energy_with_background,
    ewald_2d_point_charge_gradient_with_background,
    ewald_2d_point_charge_potential,
    ewald_2d_point_charge_potential_with_background,
    ewald_nuclear_potential,
    ewald_nuclear_repulsion,
    ewald_point_charge_energy,
    ewald_point_charge_potential,
    get_num_threads,
    hartree_energy_on_grid,
    hello,
    irreducible_kpoints,
    kinetic_exponent_derivative,
    libecpint_version,
    libint_version,
    libxc_version,
    make_cosx_jk_builder,
    make_df_jk_builder,
    make_direct_jk_builder,
    make_batched_polarised_xc_kernel_builder,
    make_four_index_jk_builder,
    make_periodic_gamma_jk_builder,
    make_polarised_gga_xc_kernel_builder,
    make_polarised_lda_xc_kernel_builder,
    make_polarised_xc_kernel_builder,
    make_unpolarised_xc_kernel_builder,
    nuclear_exponent_derivative,
    nuclear_repulsion_per_cell,
    nuclear_repulsion_slab_ewald_2d_gradient,
    overlap_exponent_derivative,
    real_space_density_from_kpoints,
    resolve_cosx_grid_level,
    resolve_cosx_variant,
    rhf_result_from_rks,
    run_mp2,
    run_rhf,
    run_rhf_periodic,
    run_rhf_periodic_gamma,
    run_rhf_scf_with_jk,
    run_rks,
    run_rks_periodic,
    run_rks_scf_with_jk,
    run_uhf,
    run_uhf_scf_with_jk,
    run_uks,
    run_uks_scf_with_jk,
    run_ump2,
    sad_density,
    set_num_threads,
    solve_poisson_coulomb,
    solve_poisson_erf_screened,
    spglib_version,
    to_primitive,
)
from ._vibeqc_core import (
    compute_ecp_gradient_contribution as _compute_ecp_gradient_native,
)
from ._vibeqc_core import (
    compute_ecp_gradient_contribution_from_primitives,
)
from ._vibeqc_core import (
    compute_ecp_matrix as _compute_ecp_matrix_native,
)
from ._vibeqc_core import (
    compute_ecp_matrix_from_primitives as _compute_ecp_matrix_from_primitives_native,
)
from ._vibeqc_core import (
    compute_gradient as _compute_gradient_core,
)
from ._vibeqc_core import (
    compute_gradient_rks as _compute_gradient_rks_core,
)
from ._vibeqc_core import (
    compute_gradient_uhf as _compute_gradient_uhf_core,
)
from ._vibeqc_core import (
    compute_gradient_uks as _compute_gradient_uks_core,
)
from ._vibeqc_core import (
    define_functional as _define_functional_core,
)
from ._vibeqc_core import (
    define_external_functional as _define_external_functional_core,
)
# Register the immutable SKALA names with a lazy callback.  Importing vibeqc
# does not import PyTorch or touch the network; both are deferred until a
# calculation actually evaluates SKALA.
from .skala import (
    CANONICAL_FUNCTIONAL as SKALA_FUNCTIONAL,
    MODEL_REVISION as SKALA_MODEL_REVISION,
    MODEL_SHA256 as SKALA_MODEL_SHA256,
    ensure_skala_model,
    model_provenance as skala_model_provenance,
    register_with_core as _register_skala_with_core,
)

# The documentation build (scripts/build_site.sh, docs/conf.py) intentionally
# replaces ``vibeqc._vibeqc_core`` with a Sphinx autodoc mock: every attribute
# is a callable stand-in and every call returns another stand-in, so the
# registry query cannot answer ``None`` or a record and the strict adoption
# check reads the stand-in as a collision (#559).  A mock has no registry to
# adopt or define; skip registration there and leave the callback unset.  A
# real core that cannot satisfy registration still fails loudly here, on
# purpose: a stale or mismatched native extension must not import quietly.
if getattr(_vibeqc_core, "__sphinx_mock__", False):
    _skala_callback = None
else:
    _skala_callback = _register_skala_with_core(_vibeqc_core)
from ._vibeqc_core import (
    ecp_core_electrons as _ecp_core_electrons_native,
)
from ._vibeqc_core import (
    ecp_effective_charges as _ecp_effective_charges_native,
)
from ._vibeqc_core import (
    monkhorst_pack as _monkhorst_pack_native,
)
from ._vibeqc_core import level_shift_at_iter
from .level_shift_schedule import LevelShiftSchedule
# NOTE: the ``banner`` FUNCTION imported here shadows the ``vibeqc.banner``
# SUBMODULE on the package root, so ``from vibeqc import banner`` yields the
# function and ``getattr(banner, "build_info")`` silently finds nothing. The
# shadowing is load-bearing for backwards compatibility (``vq.banner()`` is
# long-standing public API), so instead of renaming it we re-export every
# public name of :mod:`vibeqc.banner` here -- callers never need the submodule.
from .banner import (
    RELEASE_CODENAMES,
    VIBEQC_VERSION,
    banner,
    build_info,
    codename_for_version,
    enforce_runtime_pin_from_env,
    library_versions,
    print_banner,
)
from .guess_fragmo import Fragment, PeriodicFragment

# Public spelling compatibility: the implemented guess is named HUECKEL in the
# C++ enum, but many input decks and feature runners spell the same method
# HUCKEL. Rebuilt wheels bind the alias in C++; this keeps editable installs
# compatible before the native module is rebuilt.
if not hasattr(InitialGuess, "HUCKEL"):
    InitialGuess.HUCKEL = InitialGuess.HUECKEL

# ---------------------------------------------------------------------------
# SCFAccelerator string parsing -- runner / config / user-input helper.
# ---------------------------------------------------------------------------

_SCF_ACCELERATOR_FROM_STRING = {
    "diis": SCFAccelerator.DIIS,
    "kdiis": SCFAccelerator.KDIIS,
    "ediis": SCFAccelerator.EDIIS,
    "ediis_diis": SCFAccelerator.EDIIS_DIIS,
    "ediis+diis": SCFAccelerator.EDIIS_DIIS,
    "adiis": SCFAccelerator.ADIIS,
    "adiis_diis": SCFAccelerator.ADIIS_DIIS,
    "adiis+diis": SCFAccelerator.ADIIS_DIIS,
    # Adaptive-depth commutator-DIIS (Chupin, Dupuy, Legendre & Séré,
    # ESAIM: M2AN 55, 2785 (2021)). R_CDIIS = restarted; AD_CDIIS =
    # adaptive-depth. Several ergonomic spellings resolve to each.
    "r_cdiis": SCFAccelerator.R_CDIIS,
    "rcdiis": SCFAccelerator.R_CDIIS,
    "r-cdiis": SCFAccelerator.R_CDIIS,
    "restarted_cdiis": SCFAccelerator.R_CDIIS,
    "restarted_diis": SCFAccelerator.R_CDIIS,
    "ad_cdiis": SCFAccelerator.AD_CDIIS,
    "adcdiis": SCFAccelerator.AD_CDIIS,
    "ad-cdiis": SCFAccelerator.AD_CDIIS,
    "adaptive_cdiis": SCFAccelerator.AD_CDIIS,
    "adaptive_diis": SCFAccelerator.AD_CDIIS,
}

# Density-space mixers that share a user-facing name with the Fock-space
# ``SCFAccelerator`` enum but are a *different axis*: they extrapolate the
# density matrix, not the Fock matrix, so they have no ``SCFAccelerator`` value.
# These used to resolve *silently* to DIIS -- a wrong-method-without-warning trap
# (a user asking for Anderson got DIIS with no indication). They now raise a
# clear error pointing at the real entry point: the periodic multi-k RKS driver
# accepts ``density_mixer="anderson"`` / ``"broyden"`` (roadmap D4, v0.10.x,
# ``periodic_density_mixing.py``).
_SCF_ACCELERATOR_UNIMPLEMENTED = {
    "anderson": "Anderson real-space density mixing (Anderson 1965)",
    "broyden": "modified-Broyden / Johnson density mixing (Johnson 1988)",
}


def scf_accelerator_from_string(name: str) -> "SCFAccelerator":
    """Resolve a user-facing string to an SCFAccelerator enum value.

    Accepts case-insensitive ``"diis"``, ``"kdiis"``, ``"ediis"``,
    ``"adiis"``, ``"ediis_diis"`` and the ergonomic alias ``"ediis+diis"``,
    plus the adaptive-depth commutator-DIIS variants of Chupin et al.
    2021: ``"r_cdiis"`` (restarted) and ``"ad_cdiis"`` (adaptive-depth),
    each with a few ergonomic spellings.

    Use this in input parsers / config-driven runners so users can
    write ``scf.accelerator = "ediis_diis"`` in YAML and have it
    resolve cleanly.

    Raises :class:`NotImplementedError` for a density-space mixer name
    (``"anderson"`` / ``"broyden"``), which is not an ``SCFAccelerator``
    (Fock-space) value -- the message points at the periodic
    ``density_mixer=`` entry point -- and ``ValueError`` for an unrecognised
    name. Each message names the accepted accelerators.
    """
    key = str(name).strip().lower()
    if key in _SCF_ACCELERATOR_FROM_STRING:
        return _SCF_ACCELERATOR_FROM_STRING[key]
    supported = ", ".join(
        sorted(set(v.name for v in _SCF_ACCELERATOR_FROM_STRING.values()))
    )
    if key in _SCF_ACCELERATOR_UNIMPLEMENTED:
        raise NotImplementedError(
            f"scf_accelerator={name!r} selects "
            f"{_SCF_ACCELERATOR_UNIMPLEMENTED[key]}, a density-space mixer -- "
            "not a Fock-space SCFAccelerator (it has no enum value and is never "
            "silently treated as DIIS). Pass it to the periodic multi-k RKS "
            f"driver instead: density_mixer={key!r} "
            "(run_rks_periodic_multi_k_ewald3d / run_rks_periodic_scf). For a "
            f"Fock accelerator use one of {supported}."
        )
    raise ValueError(
        f"Unknown SCFAccelerator name {name!r}; expected one of {supported}"
    )


# Attach as a class method on the enum for ergonomic call sites
# (``vibeqc.SCFAccelerator.from_string("kdiis")``).
SCFAccelerator.from_string = staticmethod(scf_accelerator_from_string)


def monkhorst_pack(system, mesh, is_shift=None, use_symmetry=False, *, symmetry=None):
    """Build a legacy :class:`BlochKMesh` Monkhorst-Pack mesh.

    This public compatibility wrapper keeps the historical
    gamma-centred default of the C++ binding (``is_shift=(0, 0, 0)``)
    while accepting the same low-dimensional active-axis shorthand as
    :class:`KPoints`: ``[n]`` for 1D and ``[n1, n2]`` for 2D. Inactive
    axes are pinned to Γ with mesh size 1 and shift 0.

    Use :meth:`KPoints.monkhorst_pack` for the richer Python API with
    classical Monkhorst-Pack auto-shifts.

    .. warning::

       **This is the Gamma-centred convention, and it is not what
       ASE/GPAW's identically-named function returns.** vibe-qc samples
       ``{0, 1/N, ..., (N-1)/N}`` along each axis (Gamma is always in the
       set); ``ase.dft.kpoints.monkhorst_pack`` returns the classical
       mesh, offset by half a step.

       For **odd** N the half-step offset wraps onto the same lattice and
       the two point sets are identical. For **even** N it lands exactly
       between vibe-qc's points and the sets are *disjoint* -- measured
       2026-08-02, ``(3,3,3)`` and ``(5,5,5)`` share all their points while
       ``(2,2,2)``, ``(4,4,4)`` and ``(6,6,6)`` share none. A cross-code
       comparison validated at an odd mesh will therefore pass and still
       be sampling a different Brillouin zone at an even one.

       Pass ``is_shift`` explicitly to build the shifted mesh.
    """
    from .kpoints import _mesh_tuple_for_system, _shift_tuple_for_system

    if symmetry is not None:
        use_symmetry = bool(symmetry)
    mesh_t = _mesh_tuple_for_system(system, mesh)
    if is_shift is None:
        shift_t = (0, 0, 0)
    else:
        shift_t = _shift_tuple_for_system(system, is_shift)
    return _monkhorst_pack_native(
        system,
        list(mesh_t),
        list(shift_t),
        bool(use_symmetry),
    )


def compute_ecp_matrix(basis, ecp_centers, library_name="ecp10mdf", share_dir=""):
    """Compute the AO-basis ECP matrix V_ECP_{muν} = <chi_mu|V_ECP|chi_ν>
    via libecpint's built-in XML library (ecp10mdf, ecp28mdf,
    ecp46mdf, ecp60mdf, ecp78mdf, lanl2dz). Output is in spherical
    (real-solid-harmonic) basis.

    ``share_dir`` defaults to the XML library bundled inside the
    vibe-qc Python package (``python/vibeqc/ecp_library/``) -- no
    setup, no LIBECPINT_SHARE_DIR fiddling, no Homebrew dependency.
    Override by passing an explicit path, or by setting
    ``$VIBEQC_ECP_SHARE_DIR`` before ``import vibeqc``.

    Note on the share-dir convention: libecpint's
    :func:`ECPIntegrator::set_ecp_basis_from_library` internally
    appends ``"/xml/" + name + ".xml"``, so ``share_dir`` is the
    directory CONTAINING ``xml/`` (i.e. ``.../share/libecpint``),
    not ``xml/`` itself. The libecpint header doc is misleading on
    this; api.cpp:73 is the authority.
    """
    if not share_dir:
        share_dir = _VIBEQC_ECP_SHARE_DIR
    return _compute_ecp_matrix_native(
        basis,
        ecp_centers,
        library_name,
        share_dir,
    )


def compute_ecp_matrix_from_primitives(basis, center_xyz, primitives):
    """Compute V_ECP from inline primitive blocks.

    This is the public Phase-14g companion to :func:`compute_ecp_matrix`.
    ``center_xyz`` is a flat ``[x1, y1, z1, ...]`` list and ``primitives`` is
    a list of :class:`ECPPrimitiveBlock` objects, one per center.
    """

    return _compute_ecp_matrix_from_primitives_native(basis, center_xyz, primitives)


def compute_ecp_gradient_contribution(
    basis, mol, ecp_centers, D, library_name="ecp10mdf", share_dir=""
):
    """S_muν D . (dV_ECP/dR) contraction per atom: the ECP
    correction to the analytic nuclear gradient.

    Returns an (n_atoms, 3) Eigen matrix in Hartree/bohr. Empty
    ``ecp_centers`` gives a zero matrix. ``share_dir`` defaults to
    the bundled XML library (same convention as
    :func:`compute_ecp_matrix`).
    """
    if not share_dir:
        share_dir = _VIBEQC_ECP_SHARE_DIR
    return _compute_ecp_gradient_native(
        basis,
        mol,
        ecp_centers,
        D,
        library_name,
        share_dir,
    )


def ecp_core_electrons(charges, library_name, share_dir=""):
    """Return ``{Z: n_core}`` for requested elements present in an ECP.

    libecpint's bulk loader raises an opaque ``stoi: no conversion`` when
    even one requested element is absent from the XML library.  Filter the
    request against the XML element tags first so the public contract really
    does skip absent elements, while still leaving XML parsing and ncore
    validation to libecpint for every element that is present.
    """
    from xml.etree import ElementTree

    from .basis_crystal import _ELEMENT_SYMBOLS

    resolved_share_dir = share_dir or _VIBEQC_ECP_SHARE_DIR
    xml_path = _Path(resolved_share_dir) / "xml" / f"{library_name}.xml"
    if not xml_path.is_file():
        # Preserve the native loader's directive error for missing libraries.
        return _ecp_core_electrons_native(
            list(charges), library_name, resolved_share_dir
        )

    root = ElementTree.parse(xml_path).getroot()
    available = {child.tag.strip().lower() for child in root}
    present = []
    for charge in charges:
        z = int(charge)
        symbol = _ELEMENT_SYMBOLS[z] if 0 < z < len(_ELEMENT_SYMBOLS) else ""
        if symbol.lower() in available and z not in present:
            present.append(z)
    if not present:
        return {}
    return _ecp_core_electrons_native(present, library_name, resolved_share_dir)


def ecp_effective_charges(mol, ecp_centers, library_name="ecp10mdf", share_dir=""):
    """Per-atom effective charges ``Z_eff = Z - n_core`` (bare ``Z``
    for atoms without an ECP), in ``mol.atoms()`` order. Matches the
    SCF's nuclear-attraction / nuclear-repulsion convention when ECPs
    replace core electrons.
    """
    if not share_dir:
        share_dir = _VIBEQC_ECP_SHARE_DIR
    return _ecp_effective_charges_native(
        mol,
        ecp_centers,
        library_name,
        share_dir,
    )


from ._vibeqc_core import (
    EEQOptions,
    EEQResult,
    chg_max_supported_Z,
    compute_chg_dispersion,
    eeq_charges,
    eeq_coordination_numbers,
)
from ._vibeqc_core import _HubbardSiteCxx as _HubbardSiteCxx  # internal
from .composites import (
    Availability as CompositeAvailability,
)
from .composites import (
    CompositeRecipe,
    CompositeUnavailable,
    ShortRangeCorrection,
    list_composites,
    resolve_composite,
)
from .density_fitting import (
    AuxiliaryBasisCoverageError,
    DensityFitting,
    DensityFittingInfo,
    aux_basis_coverage_gaps,
    check_aux_basis_coverage,
    default_aux_basis_for,
    make_checked_aux_basis,
)

# Wrap the C++ SCF drivers to give users a clean ``dft_plus_u=`` kwarg
# surface. The C++ side carries two parallel arrays
# (``Options.dft_plus_u_sites`` + ``dft_plus_u_ao_groups``); the
# wrapper does the eV->Hartree conversion + ao_group_indices walk once
# at SCF setup so the SCF loop sees only the precomputed C++ structs.
# The actual translation helper lives in ``vibeqc.dft_plus_u`` so
# ``runner._run_single_point`` can reuse it without a circular import.
from .dft_plus_u import (
    HubbardSite,
    _apply_dft_plus_u_to_options,
    ao_group_indices,
    compute_dudarev_energy,
    compute_occupation_matrices,
)

_run_rhf_cxx = run_rhf  # capture the bare C++ binding before reassignment
_run_rks_cxx = run_rks  # capture the bare C++ binding before reassignment
_run_uhf_cxx = run_uhf
_run_uks_cxx = run_uks
_run_rhf_periodic_cxx = run_rhf_periodic
_run_rhf_periodic_gamma_cxx = run_rhf_periodic_gamma
_run_rks_periodic_cxx = run_rks_periodic


def _fragmo_cross_check(options, fragments, read_from=None):
    """Guard the FRAGMO <-> ``fragments=`` pairing: a partition passed without
    FRAGMO is almost certainly a mistake (and would be silently ignored)."""
    from ._initial_guess import coerce_initial_guess

    kind = coerce_initial_guess(options.initial_guess)
    if read_from is not None and kind != InitialGuess.READ:
        raise ValueError("read_from requires initial_guess=READ")
    is_fragmo = kind == InitialGuess.FRAGMO
    if fragments is not None and not is_fragmo:
        raise ValueError(
            "fragments=... was supplied but options.initial_guess is "
            f"{options.initial_guess.name}, not FRAGMO; set "
            "options.initial_guess = InitialGuess.FRAGMO to use the "
            "fragment-superposition guess."
        )
    return is_fragmo


def _coerce_molecular_basis(molecule, basis):
    """Accept the documented basis-name shorthand on direct SCF wrappers."""
    if isinstance(basis, str):
        return BasisSet(molecule, basis)
    return basis


def _refuse_otr_native_multiguess(options):
    if (getattr(options, "orbital_optimizer", "native") == "opentrustregion"
            and getattr(options, "multi_guess_seeds", [])):
        raise ValueError(
            "OpenTrustRegion does not use native multi_guess_seeds or restart "
            "schedules; run separately chosen starting densities, or select native"
        )


def _attach_inline_ecp_from_basis_sidecar(options, molecule, basis) -> None:
    """Populate molecular SCF options from the basis ECP sidecar."""

    from .ecp_metadata import attach_inline_ecp_options_from_basis_sidecar

    attach_inline_ecp_options_from_basis_sidecar(options, molecule, basis)


def _guard_scf_aux_basis_coverage(options, molecule, basis, *, route) -> None:
    """Refuse a density-fitted SCF whose aux basis skips an element (#480).

    The SCF J/K fit is built inside the C++ driver, which constructs the
    auxiliary ``BasisSet`` itself -- so the Python-side ``DensityFitting``
    guard never sees it. Checking here, before dispatch, makes the SCF JK
    route and the post-SCF RI route refuse identically.

    See :func:`vibeqc.density_fitting.guard_aux_coverage_for_options` for
    the permissiveness contract: this may only *add* the partial-coverage
    refusal, never change which error a caller already saw.
    """
    from .density_fitting import guard_aux_coverage_for_options

    guard_aux_coverage_for_options(options, molecule, basis, route=route)


def run_rhf(
    molecule, basis, options=None, *, dft_plus_u=None, read_from=None, fragments=None
):
    """Restricted Hartree-Fock SCF on a closed-shell molecule.

    Parameters
    ----------
    molecule, basis
        The Molecule and a BasisSet or bundled basis name.
    options
        Optional RHFOptions. A fresh one is constructed if not given.
    read_from
        Prior SCF result to restart from when
        ``options.initial_guess == InitialGuess.READ``. Alternatively set
        ``options.read_path`` to a ``.qvf`` / ``.molden`` file (including
        ORCA's ``.molden.input`` suffix). The prior
        density is projected onto the current basis (identity when basis and
        geometry already match). Ignored unless the guess is READ.
    fragments
        Fragment partition for the FRAGMO guess
        (``options.initial_guess == InitialGuess.FRAGMO``): a list of
        :class:`vibeqc.guess_fragmo.Fragment` specs, or plain atom-index
        sequences (e.g. ``[[0, 1, 2], [3, 4, 5]]``, each a neutral
        closed-shell fragment). Each fragment is converged on its own and the
        densities are assembled block-diagonally into the supersystem guess.
        Ignored unless the guess is FRAGMO.
    dft_plus_u
        Optional iterable of :class:`HubbardSite` objects (eV-input
        user dataclass from :mod:`vibeqc.dft_plus_u`). When non-empty,
        the SCF Fock builder adds the Dudarev rotationally-invariant
        per-spin potential
        ``V_U^A = U_eff (1/2 d - n^A_l)`` on each (atom, l) channel; the
        Dudarev energy contribution ``E_U = 2 S_A (U_eff/2) (tr n -
        tr n^2)`` (closed-shell, summed over both spins) appears as
        ``result.e_dft_plus_u`` and is included in ``result.energy``.

    Returns
    -------
    RHFResult
        Same shape as the C++ binding, with the additional
        ``e_dft_plus_u`` field populated when +U was active.
    """
    if options is None:
        options = RHFOptions()
    require_nonnegative_max_iter(options.max_iter, route="run_rhf")
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(options, molecule, basis)
    _guard_scf_aux_basis_coverage(options, molecule, basis, route="run_rhf")
    if dft_plus_u:
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    if _fragmo_cross_check(options, fragments, read_from):
        from .guess_fragmo import resolve_fragmo_density_closed

        options.read_density = resolve_fragmo_density_closed(
            options, molecule, basis, fragments
        )
    elif options.initial_guess == InitialGuess.READ:
        from .guess_read import resolve_read_density_closed

        options.read_density = resolve_read_density_closed(
            options, molecule, basis, read_from
        )
    return _run_rhf_cxx(molecule, basis, options)


def run_rks(
    molecule,
    basis,
    options=None,
    *,
    dft_plus_u=None,
    read_from=None,
    fragments=None,
    grid_level="orca-defgrid3",
    _allow_double_hybrid_scf=False,
):
    """Restricted Kohn-Sham SCF on a closed-shell molecule.

    Same shape as :func:`run_rhf`; ``options`` is :class:`RKSOptions`
    (carries the XC functional name) and the result type is
    :class:`RKSResult` (carries the KS energy decomposition plus the
    optional ``e_dft_plus_u`` field). ``read_from`` / ``options.read_path``
    drive the READ restart guess as in :func:`run_rhf`; ``fragments=`` drives
    the FRAGMO fragment-superposition guess (each fragment converged with the
    same functional).

    ``grid_level`` selects the DFT integration grid when ``options`` is
    ``None``.  See :func:`run_job` for the accepted values.  This low-level
    wrapper uses a supplied ``options`` object as given, grid included; the
    entry points that take a ``grid_level`` (:func:`run_job`, the ASE
    calculator, the geometry-optimization provider) apply it to any options
    whose grid is untouched, so passing ``RKSOptions()`` to them does not
    select the legacy grid (GitLab #663).
    """
    if options is None:
        options = RKSOptions()
        from .runner import _apply_grid_level
        _apply_grid_level(options.grid, grid_level)
    require_nonnegative_max_iter(options.max_iter, route="run_rks")
    from .roks import _refuse_plain_double_hybrid

    _refuse_plain_double_hybrid(
        Functional(options.functional or "lda"),
        route="RKS",
        allow_double_hybrid_scf=_allow_double_hybrid_scf,
    )
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(options, molecule, basis)
    _guard_scf_aux_basis_coverage(options, molecule, basis, route="run_rks")
    if dft_plus_u:
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    if _fragmo_cross_check(options, fragments, read_from):
        from .guess_fragmo import resolve_fragmo_density_closed

        options.read_density = resolve_fragmo_density_closed(
            options, molecule, basis, fragments
        )
    elif options.initial_guess == InitialGuess.READ:
        from .guess_read import resolve_read_density_closed

        options.read_density = resolve_read_density_closed(
            options, molecule, basis, read_from
        )
    return _run_rks_cxx(molecule, basis, options)


def run_uhf(
    molecule, basis, options=None, *, dft_plus_u=None, read_from=None, fragments=None
):
    """Unrestricted Hartree-Fock SCF on an open-shell molecule.

    Open-shell counterpart of :func:`run_rhf`. The Dudarev +U formula
    applies natively per-spin -- the SCF Fock builder calls the kernel
    once per spin with that spin's density matrix and adds the
    per-spin V_U to the per-spin Fock; ``result.e_dft_plus_u`` is the
    sum of the two per-spin contributions (no factor of 2 needed --
    the per-spin formula already gives one term per spin).

    ``read_from`` / ``options.read_path`` drive the READ restart guess; the
    prior alpha/beta densities are projected onto the current basis.
    ``fragments=`` drives the FRAGMO fragment-superposition guess (each
    fragment's a/b densities feed the corresponding supersystem spin blocks).
    """
    if options is None:
        options = UHFOptions()
    _refuse_otr_native_multiguess(options)
    require_nonnegative_max_iter(options.max_iter, route="run_uhf")
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(options, molecule, basis)
    _guard_scf_aux_basis_coverage(options, molecule, basis, route="run_uhf")
    if dft_plus_u:
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    if _fragmo_cross_check(options, fragments, read_from):
        from .guess_fragmo import resolve_fragmo_densities_open

        da, db = resolve_fragmo_densities_open(options, molecule, basis, fragments)
        options.read_density_alpha = da
        options.read_density_beta = db
    elif options.initial_guess == InitialGuess.READ:
        from .guess_read import resolve_read_densities_open

        da, db = resolve_read_densities_open(options, molecule, basis, read_from)
        options.read_density_alpha = da
        options.read_density_beta = db

    result = _run_uhf_cxx(molecule, basis, options)

    # ---- multi-guess loop (BUG 88) ------------------------------------
    extra_seeds = list(getattr(options, "multi_guess_seeds", []) or [])
    if extra_seeds and result.converged:
        from .state_fingerprint import compute_state_fingerprint
        best = result
        seen: set[str] = set()
        _ne = molecule.n_electrons() - int(
            getattr(best, "ecp_total_ncore", 0) or 0
        )
        _mult = molecule.multiplicity
        _na = (_ne + _mult - 1) // 2
        _nb = (_ne - _mult + 1) // 2
        _s2 = float(getattr(best, "s_squared", 0.0))
        seen.add(compute_state_fingerprint(
            energy=float(best.energy), n_alpha=int(_na), n_beta=int(_nb),
            s_squared=_s2, molecule_symbols=[a.symbol for a in molecule.atoms]))
        for seed in extra_seeds:
            # Value-copy the complete native option surface. A whitelist can
            # silently change the Hamiltonian: in particular, dropping ECP
            # centers/primitive data would compare an ECP base result with an
            # all-electron alternate and erase its ecp_total_ncore provenance.
            opts_alt = UHFOptions(options)
            opts_alt.restart_opts.enabled = True
            opts_alt.restart_opts.seed = int(seed)
            try:
                alt = _run_uhf_cxx(molecule, basis, opts_alt)
            except RuntimeError:
                continue
            if not alt.converged:
                continue
            _fp = compute_state_fingerprint(
                energy=float(alt.energy), n_alpha=int(_na), n_beta=int(_nb),
                s_squared=float(getattr(alt, "s_squared", 0.0)),
                molecule_symbols=[a.symbol for a in molecule.atoms])
            if _fp in seen:
                continue
            if getattr(alt, "internal_instability", False):
                continue
            # GitLab #566: internal_instability is the fail-closed conjunction
            # ``stability_checked && converged && lambda < -tol``, so False
            # means EITHER certified-stable OR no verdict at all -- a seed
            # whose stability Davidson exhausted its budget looks identical
            # here to one that passed. Ranking such a seed above a
            # verdict-carrying one would let an unexamined state win the
            # multi-guess selection on energy alone.
            #
            # Three states, not two. ``stability_checked`` is what keeps this
            # from disabling multi-guess entirely: when the analysis was never
            # requested the flag is False and the seed stays eligible, exactly
            # as before. Only a seed that WAS examined and returned no verdict
            # is set aside.
            if getattr(alt, "stability_checked", False) and not getattr(
                alt, "stability_analysis_converged", False
            ):
                continue
            # Only an eligible, verdict-carrying state participates in
            # duplicate suppression.  Recording a rejected no-verdict seed
            # here would hide a later seed that converged to the same SCF
            # state *and* obtained a stability verdict (issue #566, gen 2).
            seen.add(_fp)
            if alt.energy < best.energy - 1e-8:
                best = alt
        return best
    return result


def run_uks(
    molecule,
    basis,
    options=None,
    *,
    dft_plus_u=None,
    read_from=None,
    fragments=None,
    grid_level="orca-defgrid3",
    _allow_double_hybrid_scf=False,
):
    """Unrestricted Kohn-Sham SCF on an open-shell molecule.

    Open-shell counterpart of :func:`run_rks`. Same per-spin
    convention as :func:`run_uhf`; ``options`` is :class:`UKSOptions`
    and the result is :class:`UKSResult`. ``read_from`` /
    ``options.read_path`` drive the READ restart guess; ``fragments=`` drives
    the FRAGMO fragment-superposition guess (each fragment converged with the
    same functional, a/b densities feeding the supersystem spin blocks).

    ``grid_level`` selects the DFT integration grid when ``options`` is
    ``None``.  See :func:`run_job` for the accepted values.  As for
    :func:`run_rks`, a supplied ``options`` object is used as given here,
    while the high-level entry points apply ``grid_level`` to an untouched
    grid (GitLab #663).
    """
    if options is None:
        options = UKSOptions()
        from .runner import _apply_grid_level
        _apply_grid_level(options.grid, grid_level)
    _refuse_otr_native_multiguess(options)
    require_nonnegative_max_iter(options.max_iter, route="run_uks")
    from .roks import _refuse_plain_double_hybrid

    _refuse_plain_double_hybrid(
        Functional(options.functional or "lda", 2),
        route="UKS",
        allow_double_hybrid_scf=_allow_double_hybrid_scf,
    )
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(options, molecule, basis)
    _guard_scf_aux_basis_coverage(options, molecule, basis, route="run_uks")
    if dft_plus_u:
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    if _fragmo_cross_check(options, fragments, read_from):
        from .guess_fragmo import resolve_fragmo_densities_open

        da, db = resolve_fragmo_densities_open(options, molecule, basis, fragments)
        options.read_density_alpha = da
        options.read_density_beta = db
    elif options.initial_guess == InitialGuess.READ:
        from .guess_read import resolve_read_densities_open

        da, db = resolve_read_densities_open(options, molecule, basis, read_from)
        options.read_density_alpha = da
        options.read_density_beta = db

    result = _run_uks_cxx(molecule, basis, options)

    # ---- multi-guess loop (BUG 88) ----------------------------------
    extra_seeds = list(getattr(options, "multi_guess_seeds", []) or [])
    if extra_seeds and result.converged:
        from .state_fingerprint import compute_state_fingerprint
        best = result
        seen: set[str] = set()
        _ne = molecule.n_electrons() - int(
            getattr(best, "ecp_total_ncore", 0) or 0
        )
        _mult = molecule.multiplicity
        _na = (_ne + _mult - 1) // 2
        _nb = (_ne - _mult + 1) // 2
        _s2 = float(getattr(best, "s_squared", 0.0))
        seen.add(compute_state_fingerprint(
            energy=float(best.energy), n_alpha=int(_na), n_beta=int(_nb),
            s_squared=_s2, molecule_symbols=[a.symbol for a in molecule.atoms]))
        for seed in extra_seeds:
            # Value-copy the complete native option surface. Besides avoiding
            # omissions such as DFT+U, this preserves whether stability=True
            # is the implicit default: assigning through its Python property
            # would turn an automatic unsupported-route skip into an explicit
            # request that raises and silently discards every alternate.
            opts_alt = UKSOptions(options)
            opts_alt.restart_opts.enabled = True
            opts_alt.restart_opts.seed = int(seed)
            try:
                alt = _run_uks_cxx(molecule, basis, opts_alt)
            except RuntimeError:
                continue
            if not alt.converged:
                continue
            _fp = compute_state_fingerprint(
                energy=float(alt.energy), n_alpha=int(_na), n_beta=int(_nb),
                s_squared=float(getattr(alt, "s_squared", 0.0)),
                molecule_symbols=[a.symbol for a in molecule.atoms])
            if _fp in seen:
                continue
            if getattr(alt, "internal_instability", False):
                continue
            # GitLab #566: internal_instability is the fail-closed conjunction
            # ``stability_checked && converged && lambda < -tol``, so False
            # means EITHER certified-stable OR no verdict at all -- a seed
            # whose stability Davidson exhausted its budget looks identical
            # here to one that passed. Ranking such a seed above a
            # verdict-carrying one would let an unexamined state win the
            # multi-guess selection on energy alone.
            #
            # Three states, not two. ``stability_checked`` is what keeps this
            # from disabling multi-guess entirely: when the analysis was never
            # requested the flag is False and the seed stays eligible, exactly
            # as before. Only a seed that WAS examined and returned no verdict
            # is set aside.
            if getattr(alt, "stability_checked", False) and not getattr(
                alt, "stability_analysis_converged", False
            ):
                continue
            # Keep rejected no-verdict/unstable seeds out of the dedupe set:
            # a later seed may reach the same state with a real verdict.
            seen.add(_fp)
            if alt.energy < best.energy - 1e-8:
                best = alt
        return best
    return result


def _run_native_periodic_source(run, system, basis, options, kmesh=None, read_from=None, fragments=None):
    import numpy as np
    from .guess_read import resolve_periodic_read_density_closed, resolve_periodic_read_density_k_closed
    if fragments is not None or options.initial_guess == InitialGuess.FRAGMO:
        if options.initial_guess != InitialGuess.FRAGMO or read_from is not None:
            raise ValueError("fragments requires FRAGMO and cannot be combined with read_from")
        from .guess_fragmo import resolve_periodic_fragmo_source
        from .guess import GuessSelection
        source = resolve_periodic_fragmo_source(
            options, system, basis, kmesh if kmesh is not None else monkhorst_pack(system, [1,1,1]), fragments)
        options.initial_guess = InitialGuess.READ
        try:
            result = _run_native_periodic_source(run, system, basis, options, kmesh, source)
        finally:
            options.initial_guess = InitialGuess.FRAGMO
        result.guess_selection = GuessSelection(InitialGuess.FRAGMO, InitialGuess.FRAGMO, InitialGuess.READ)
        return result
    if read_from is None:
        return run(system, basis, options) if kmesh is None else run(system, basis, kmesh, options)
    if options.initial_guess != InitialGuess.READ:
        raise ValueError("read_from requires initial_guess=InitialGuess.READ")
    if kmesh is None:
        field = "read_density"
        density = resolve_periodic_read_density_closed(basis, read_from=read_from)
        saved = np.array(options.read_density, copy=True)
    else:
        field = "read_density_k"
        density = resolve_periodic_read_density_k_closed(
            read_from=read_from, basis=basis, system=system, kmesh=kmesh)
        saved = [np.array(d, copy=True) for d in options.read_density_k]
    try:
        setattr(options, field, density)
        return run(system, basis, options) if kmesh is None else run(system, basis, kmesh, options)
    finally:
        setattr(options, field, saved)


def run_rhf_periodic(system, basis, kmesh, options=None, *, read_from=None, fragments=None):
    """Native multi-k RHF with optional result/QVF READ source."""
    from .kpoints import as_bloch_kmesh
    if options is None:
        options = PeriodicSCFOptions()
    return _run_native_periodic_source(
        _run_rhf_periodic_cxx, system, basis, options,
        as_bloch_kmesh(kmesh), read_from, fragments)


def run_rhf_periodic_gamma(system, basis, options=None, *, dft_plus_u=None, read_from=None, fragments=None):
    """Γ-only closed-shell RHF for a periodic system (molecular-limit
    regime).

    The Γ-only periodic SCF carries a single real ``Eigen::MatrixXd``
    density and AO overlap -- same shape as the molecular case -- so
    the +U kernel is reused unchanged. ``options`` is a
    :class:`PeriodicRHFOptions`; ``dft_plus_u`` is a list of
    :class:`HubbardSite` objects (per-atom (atom_index, l, U_ev)
    channels). For multi-k +U use :func:`run_rks_periodic` (closed-shell
    KS, Increment 4c) or ``run_periodic_job`` — its auto jk_method runs
    closed-shell multi-k +U natively on the GDF drivers.
    """
    if options is None:
        options = PeriodicRHFOptions()
    if dft_plus_u:
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    return _run_native_periodic_source(
        _run_rhf_periodic_gamma_cxx, system, basis, options, read_from=read_from, fragments=fragments)


def run_rks_periodic(system, basis, kmesh, options=None, *, dft_plus_u=None, read_from=None, fragments=None):
    """Multi-k closed-shell periodic Kohn-Sham SCF (DIRECT_TRUNCATED).

    Optional ``dft_plus_u=[HubbardSite(...)]`` adds the Dudarev
    rotationally-invariant Hubbard correction (Increments 4b + 4c).
    The multi-k path (Increment 4c) computes the k-averaged AO
    occupation matrix
    ``n^A_l = (1/2) Σ_k w_k Re[(S(k)P(k)S(k))_{(A,l)}]`` and adds
    the per-k Fock contribution ``S(k) V_AO S(k)`` to each diagonalised
    F(k).  Single-Γ ``[1,1,1]`` and arbitrary k-meshes are both
    supported; there is no Γ-only restriction.
    """
    # K7 boundary: KPoints -> BlochKMesh, BlochKMesh passes through.
    from .kpoints import as_bloch_kmesh

    bm = as_bloch_kmesh(kmesh)
    if options is None:
        options = PeriodicKSOptions()
    if dft_plus_u:
        # Multi-k +U landed in Increment 4c -- the C++ driver now
        # computes ``n = S_k w_k Re[(S(k) P(k) S(k))_{(A,l)}]`` and
        # adds the k-dependent ``S(k) V_AO S(k)`` to each F(k) in
        # the diagonalisation loop. Single ``[1,1,1]`` and arbitrary
        # k-mesh are both supported.
        _apply_dft_plus_u_to_options(options, basis, dft_plus_u)
    return _run_native_periodic_source(
        _run_rks_periodic_cxx, system, basis, options, bm, read_from, fragments)


from .dispersion import (
    compute_d3bj,
    d3bj_params_for,
    dftd3_available,
)
from .dispersion_d4 import (
    D4Result,
    compute_d4,
    dftd4_available,
)
from .dispersion_d4_model import (
    D4_WEIGHTING_FACTOR,
    cn_gaussian_weights,
)
from .dispersion_d4_refdata import (
    casimir_polder_c6,
    coupled_polarizability_imag_freq,
    imaginary_frequency_grid,
    london_c6_single_pole,
    molecular_c6,
    uncoupled_polarizability_imag_freq,
)
from .dispersion_d4_reference_systems import (
    ReferenceSystem,
    all_reference_elements,
    all_reference_systems,
    reference_systems_for,
)
from .dispersion_d4_reference_systems import (
    build_molecule as build_reference_molecule,
)

# v0.9.0 composite 3c methods + geometric counterpoise.
from .gcp import (
    GCPDataMissing,
    GCPParams,
    GCPResult,
    compute_gcp,
    gcp_params_for,
)
from .gcp import (
    available_basis_sets as available_gcp_basis_sets,
    available_variants as available_gcp_variants,
)

# ---------------------------------------------------------------------
# Gradient kernel -- thin Python wrappers over the C++ bindings.
#
# History: v0.7.3 had a silent bug in the direct 4-index ERI gradient
# kernel on multi-heavy-atom systems with f-shells (def2-tzvp+). The
# v0.7.4 "Fix C" (canonical 1/8 shell loop + l-canonical reorder
# before libint) lives in ``cpp/src/gradient.cpp ::
# two_electron_gradient_contribution`` and resolves the bug to machine
# precision (H2CO/def2-tzvp Cs-zero ~1e-13, vs PySCF ~5e-11).
# Earlier branches shipped a "Fix D" auto-route through DF on f-shell
# bases -- that workaround is removed here because the direct kernel is
# now correct. The ``_auto_df_for_fshell`` keyword survives as a no-op
# for one minor release for backwards compatibility; it will be
# removed in v0.8.0.
# ---------------------------------------------------------------------


from .gradient_terms import (  # noqa: E402
    functional_gradient_terms_missing,
    require_complete_analytic_gradient,
)
from .ecp_metadata import (  # noqa: E402
    validate_gradient_ecp_contract as _validate_gradient_ecp_contract,
)


def compute_gradient(
    mol,
    basis,
    result,
    options=None,
    *,
    dft_plus_u=None,
    _auto_df_for_fshell=False,
):
    """RHF analytic nuclear gradient (Ha/bohr, per atom).

    ``_auto_df_for_fshell`` is accepted but ignored -- kept for one
    minor-version backwards compatibility after v0.7.3's f-shell
    auto-route was retired.

    ``dft_plus_u`` (optional): list of :class:`HubbardSite` the SCF
    was run with. When set, the +U explicit ``dE_U/dR`` is added to
    the HF gradient. The orbital-response (Pulay) piece of
    ``dE_U/dR`` is automatically captured by the standard
    energy-weighted-density Pulay term because the converged
    ``result.mo_energies`` already include the +U shift (the C++
    SCF iterates the variational Fock ``F + S V_AO S``, see
    :mod:`vibeqc.dft_plus_u`). Verified against FD on H2O/STO-3G
    at 6.9x10⁻¹¹ Ha/bohr.
    """
    del _auto_df_for_fshell  # no-op (v0.7.4); will be removed in v0.8.0.
    if options is None:
        options = GradientOptions()
    _validate_gradient_ecp_contract(
        mol, result, options, route="compute_gradient"
    )
    # #480: the native DF gradient driver builds its own auxiliary
    # BasisSet from options.aux_basis, so the DensityFitting guard
    # never sees it. Refuse an uncovered element before any fitted
    # derivative work.
    _guard_scf_aux_basis_coverage(options, mol, basis, route="gradient")
    native_result = getattr(result, "_vibeqc_native_result", result)
    grad = np.asarray(
        _compute_gradient_core(mol, basis, native_result, options)
    )
    if dft_plus_u:
        from ._vibeqc_core import compute_overlap
        from .dft_plus_u import _compute_dft_plus_u_gradient

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        grad = grad + _compute_dft_plus_u_gradient(
            basis,
            mol,
            S,
            dft_plus_u,
            P_total=P,
        )
    return grad


def compute_gradient_rks(
    mol,
    basis,
    result,
    grid_options=None,
    options=None,
    *,
    dft_plus_u=None,
    allow_incomplete=False,
    _auto_df_for_fshell=False,
):
    """RKS analytic nuclear gradient (Ha/bohr, per atom).

    Note: the C++ signature is
    ``compute_gradient_rks(mol, basis, result, grid_options, options)``;
    keep the positional order or pass by keyword.

    ``dft_plus_u``: same convention as :func:`compute_gradient`.

    ``allow_incomplete``: the kernel omits the long-range exact-exchange
    gradient of a range-separated hybrid and the VV10 nonlocal gradient
    (GitLab #571; see :func:`functional_gradient_terms_missing`). For
    such a ``result.functional`` this wrapper raises
    :class:`NotImplementedError` naming the missing terms, because the
    returned array would not be the derivative of ``result.energy``
    (hse06 5.6e-3, vv10 7.0e-4, wb97x-v 5.6e-2 Ha/bohr off finite
    differences on O/H/H STO-3G). Pass ``allow_incomplete=True`` to
    receive the partial analytic gradient knowingly.
    """
    del _auto_df_for_fshell  # no-op (v0.7.4); will be removed in v0.8.0.
    if not allow_incomplete:
        require_complete_analytic_gradient(
            result, route="compute_gradient_rks", spin=1
        )
    if options is None:
        options = GradientOptions()
    _validate_gradient_ecp_contract(
        mol, result, options, route="compute_gradient_rks"
    )
    # #480: the native DF gradient driver builds its own auxiliary
    # BasisSet from options.aux_basis, so the DensityFitting guard
    # never sees it. Refuse an uncovered element before any fitted
    # derivative work.
    _guard_scf_aux_basis_coverage(options, mol, basis, route="gradient")
    if grid_options is None:
        # Match the C++ default (vibeqc::GridOptions{}).
        grid_options = GridOptions()
    grad = np.asarray(
        _compute_gradient_rks_core(
            mol,
            basis,
            getattr(result, "_vibeqc_native_result", result),
            grid_options,
            options,
        )
    )
    if dft_plus_u:
        from ._vibeqc_core import compute_overlap
        from .dft_plus_u import _compute_dft_plus_u_gradient

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        grad = grad + _compute_dft_plus_u_gradient(
            basis,
            mol,
            S,
            dft_plus_u,
            P_total=P,
        )
    return grad


def compute_gradient_uhf(
    mol,
    basis,
    result,
    options=None,
    *,
    dft_plus_u=None,
    _auto_df_for_fshell=False,
):
    """UHF analytic nuclear gradient (Ha/bohr, per atom).

    ``dft_plus_u``: open-shell variant -- per-spin densities are
    pulled from ``result.density_alpha`` / ``density_beta``.
    """
    del _auto_df_for_fshell  # no-op (v0.7.4); will be removed in v0.8.0.
    if options is None:
        options = GradientOptions()
    _validate_gradient_ecp_contract(
        mol, result, options, route="compute_gradient_uhf"
    )
    # #480: the native DF gradient driver builds its own auxiliary
    # BasisSet from options.aux_basis, so the DensityFitting guard
    # never sees it. Refuse an uncovered element before any fitted
    # derivative work.
    _guard_scf_aux_basis_coverage(options, mol, basis, route="gradient")
    native_result = getattr(result, "_vibeqc_native_result", result)
    grad = np.asarray(
        _compute_gradient_uhf_core(mol, basis, native_result, options)
    )
    if dft_plus_u:
        from ._vibeqc_core import compute_overlap
        from .dft_plus_u import _compute_dft_plus_u_gradient

        S = np.asarray(compute_overlap(basis))
        grad = grad + _compute_dft_plus_u_gradient(
            basis,
            mol,
            S,
            dft_plus_u,
            P_alpha=np.asarray(result.density_alpha),
            P_beta=np.asarray(result.density_beta),
        )
    return grad


def compute_gradient_uks(
    mol,
    basis,
    result,
    grid_options=None,
    options=None,
    *,
    dft_plus_u=None,
    allow_incomplete=False,
    _auto_df_for_fshell=False,
):
    """UKS analytic nuclear gradient (Ha/bohr, per atom).

    Note: the C++ signature is
    ``compute_gradient_uks(mol, basis, result, grid_options, options)``;
    keep the positional order or pass by keyword.

    ``dft_plus_u``: same convention as :func:`compute_gradient_uhf`.

    ``allow_incomplete``: same contract as :func:`compute_gradient_rks`
    (GitLab #571): a range-separated or VV10-paired
    ``result.functional`` raises :class:`NotImplementedError` unless the
    caller opts in to the partial analytic gradient.
    """
    del _auto_df_for_fshell  # no-op (v0.7.4); will be removed in v0.8.0.
    if not allow_incomplete:
        require_complete_analytic_gradient(
            result, route="compute_gradient_uks", spin=2
        )
    if options is None:
        options = GradientOptions()
    _validate_gradient_ecp_contract(
        mol, result, options, route="compute_gradient_uks"
    )
    # #480: the native DF gradient driver builds its own auxiliary
    # BasisSet from options.aux_basis, so the DensityFitting guard
    # never sees it. Refuse an uncovered element before any fitted
    # derivative work.
    _guard_scf_aux_basis_coverage(options, mol, basis, route="gradient")
    if grid_options is None:
        grid_options = GridOptions()
    grad = np.asarray(
        _compute_gradient_uks_core(
            mol,
            basis,
            getattr(result, "_vibeqc_native_result", result),
            grid_options,
            options,
        )
    )
    if dft_plus_u:
        from ._vibeqc_core import compute_overlap
        from .dft_plus_u import _compute_dft_plus_u_gradient

        S = np.asarray(compute_overlap(basis))
        grad = grad + _compute_dft_plus_u_gradient(
            basis,
            mol,
            S,
            dft_plus_u,
            P_alpha=np.asarray(result.density_alpha),
            P_beta=np.asarray(result.density_beta),
        )
    return grad


# ---------------------------------------------------------------------
# Scaled-spin-component MP2 convenience wrappers.
#
# All of these default ``density_fit=True`` (RI-MP2) -- for double-hybrid
# correlations the MP2 step is cost-dominant and canonical MP2 is
# untractable past def2-TZVP on a workstation. Pass ``density_fit=False``
# for an explicit canonical-MP2 parity run; ``aux_basis`` is then
# ignored. With the default RI path and ``aux_basis=""``, the per-zeta
# RIfit aux is auto-resolved from the orbital basis name via
# ``default_aux_basis_for(name, kind="ri")``.
#
# References:
#   * SCS-MP2 -- Grimme, J. Chem. Phys. 118, 9095 (2003).
#   * SOS-MP2 -- Jung, Lochan, Dutoi, Head-Gordon, J. Chem. Phys.
#               121, 9793 (2004).
# ---------------------------------------------------------------------


def _resolve_mp2_aux_basis(basis, density_fit, aux_basis):
    if not density_fit:
        return ""
    if aux_basis:
        return aux_basis
    return default_aux_basis_for(basis.name, kind="ri")


def _emit_mp2_citations(output, basis, *, method_key, frozen_core):
    """Common citation-sibling emitter for the SCS / SOS MP2 wrappers."""
    if output is None:
        return
    try:
        from .correlation_conventions import (
            _PUBLISHED_FROZEN_CORE_CITATION_KEY,
            _uses_published_frozen_core_selector,
        )
        from .output.citations import emit_citations

        emit_citations(
            output,
            method=method_key,
            basis=basis.name,
            extra_entries=(
                (_PUBLISHED_FROZEN_CORE_CITATION_KEY,)
                if _uses_published_frozen_core_selector(frozen_core)
                else ()
            ),
        )
    except Exception:
        pass


def run_scs_mp2(
    mol,
    basis,
    rhf_result,
    *,
    density_fit=True,
    aux_basis="",
    c_os=6.0 / 5.0,
    c_ss=1.0 / 3.0,
    report_ri_residual=False,
    frozen_core=None,
    output=None,
):
    """SCS-MP2 (Grimme JCP 118, 9095 (2003)) on a closed-shell RHF
    reference. Defaults to RI-MP2 with Grimme's c_os = 6/5 and
    c_ss = 1/3.

    ``frozen_core=None`` selects the published ORCA 6.1 Table 2.69
    count-only convention. Pass ``False`` for an all-electron calculation
    or a non-negative integer for an explicit frozen spatial-orbital count.

    ``report_ri_residual=True`` (opt-in, default off) additionally
    builds the canonical four-index (ia|jb) and reports the per-bucket
    RI fit residual on ``MP2Result.{e_os_ri_residual,
    e_ss_ri_residual}`` -- doubles the cost, intended for verifying
    aux-basis adequacy. No-op when ``density_fit=False``.

    ``output`` -- when set, writes ``{output}.bibtex`` and
    ``{output}.references`` siblings carrying the SCS-MP2 citation
    bundle (Moller-Plesset 1934 + Grimme 2003 + Feyereisen 1993 when
    ``density_fit=True``), plus the official frozen-core table source when
    the published selector is used.
    """
    opts = MP2Options()
    opts.n_frozen_core = resolve_frozen_core(mol, frozen_core, reference=rhf_result)
    opts.density_fit = density_fit
    opts.aux_basis = _resolve_mp2_aux_basis(basis, density_fit, aux_basis)
    opts.c_os = c_os
    opts.c_ss = c_ss
    opts.report_ri_residual = report_ri_residual
    result = run_mp2(mol, basis, rhf_result, opts)
    _emit_mp2_citations(
        output,
        basis,
        method_key=("scs-ri-mp2" if density_fit else "scs-mp2"),
        frozen_core=frozen_core,
    )
    return result


def run_sos_mp2(
    mol,
    basis,
    rhf_result,
    *,
    density_fit=True,
    aux_basis="",
    c_os=1.3,
    c_ss=0.0,
    report_ri_residual=False,
    frozen_core=None,
    output=None,
):
    """SOS-MP2 (Jung-Lochan-Dutoi-Head-Gordon, JCP 121, 9793 (2004))
    on a closed-shell RHF reference. Defaults to RI-MP2 with the
    original c_os = 1.3, c_ss = 0.

    ``frozen_core`` -- see :func:`run_scs_mp2`.

    ``report_ri_residual`` -- see :func:`run_scs_mp2`.

    ``output`` -- see :func:`run_scs_mp2`.
    """
    opts = MP2Options()
    opts.n_frozen_core = resolve_frozen_core(mol, frozen_core, reference=rhf_result)
    opts.density_fit = density_fit
    opts.aux_basis = _resolve_mp2_aux_basis(basis, density_fit, aux_basis)
    opts.c_os = c_os
    opts.c_ss = c_ss
    opts.report_ri_residual = report_ri_residual
    result = run_mp2(mol, basis, rhf_result, opts)
    _emit_mp2_citations(
        output,
        basis,
        method_key=("sos-ri-mp2" if density_fit else "sos-mp2"),
        frozen_core=frozen_core,
    )
    return result


def run_scs_ump2(
    mol,
    basis,
    uhf_result,
    *,
    density_fit=True,
    aux_basis="",
    c_os=6.0 / 5.0,
    c_ss=1.0 / 3.0,
    report_ri_residual=False,
    frozen_core=None,
    output=None,
):
    """SCS-UMP2 on a UHF reference. Same scaling as ``run_scs_mp2`` but
    applied per Grimme channel definition to the ab (opposite-spin)
    and aa+bb (same-spin) UMP2 channels.

    ``frozen_core=None`` selects the published ORCA 6.1 Table 2.69
    count-only convention. Pass ``False`` for an all-electron calculation
    or a non-negative integer for an explicit frozen spatial-orbital count.

    ``report_ri_residual=True`` (opt-in, default off) additionally
    reports the per-channel RI fit residual on
    ``UMP2Result.{e_aa_ri_residual, e_bb_ri_residual,
    e_ab_ri_residual}`` -- doubles the cost, for aux-basis
    verification. No-op when ``density_fit=False``.
    """
    opts = UMP2Options()
    opts.n_frozen_core = resolve_frozen_core(mol, frozen_core, reference=uhf_result)
    opts.density_fit = density_fit
    opts.aux_basis = _resolve_mp2_aux_basis(basis, density_fit, aux_basis)
    opts.c_os = c_os
    opts.c_ss = c_ss
    opts.report_ri_residual = report_ri_residual
    result = run_ump2(mol, basis, uhf_result, opts)
    _emit_mp2_citations(
        output,
        basis,
        method_key=("scs-ri-mp2" if density_fit else "scs-mp2"),
        frozen_core=frozen_core,
    )
    return result


def run_sos_ump2(
    mol,
    basis,
    uhf_result,
    *,
    density_fit=True,
    aux_basis="",
    c_os=1.3,
    c_ss=0.0,
    report_ri_residual=False,
    frozen_core=None,
    output=None,
):
    """SOS-UMP2 on a UHF reference. Same scaling as ``run_sos_mp2``.

    ``frozen_core`` -- see :func:`run_scs_ump2`.

    ``report_ri_residual`` -- see :func:`run_scs_ump2`.

    ``output`` -- see :func:`run_scs_mp2`.
    """
    opts = UMP2Options()
    opts.n_frozen_core = resolve_frozen_core(mol, frozen_core, reference=uhf_result)
    opts.density_fit = density_fit
    opts.aux_basis = _resolve_mp2_aux_basis(basis, density_fit, aux_basis)
    opts.c_os = c_os
    opts.c_ss = c_ss
    opts.report_ri_residual = report_ri_residual
    result = run_ump2(mol, basis, uhf_result, opts)
    _emit_mp2_citations(
        output,
        basis,
        method_key=("sos-ri-mp2" if density_fit else "sos-mp2"),
        frozen_core=frozen_core,
    )
    return result


# ---------------------------------------------------------------------
# Double-hybrid DFT dispatcher (B2PLYP -- first end-to-end double hybrid).
#
# Double hybrids partition the XC energy into an SCF-stage hybrid (HF +
# GGA exchange + GGA correlation) plus a post-SCF scaled-MP2 correction
# on the converged KS orbitals:
#
#   E_DH = E_RKS[xc_scf_piece] + c_os . E_os^MP2 + c_ss . E_ss^MP2
#
# B2PLYP (Grimme 2006): xc_scf = 0.53.HF + 0.47.B88 + 0.73.LYP and
# c_os = c_ss = 0.27. ``Functional("b2plyp")`` resolves to the
# combined recipe -- ``hf_exchange_fraction`` carries the 0.53,
# ``mp2_c_os`` / ``mp2_c_ss`` carry the 0.27. RI-MP2 is the default
# for the MP2 correction (the cost-dominant step at production
# basis sizes); pass ``density_fit_mp2=False`` for a canonical-MP2
# parity run.
# ---------------------------------------------------------------------


class DoubleHybridResult:
    """Result of a double-hybrid run: the SCF (hybrid-DFT) result, the
    post-SCF MP2 correction result, optionally a dispersion correction,
    and the combined total energy.

    Attributes
    ----------
    rks : RKSResult
        The converged hybrid-DFT SCF step.
    mp2 : MP2Result
        The post-SCF MP2 correction. ``mp2.e_correlation`` is already
        scaled by the functional's ``mp2_c_os`` / ``mp2_c_ss``; the
        unscaled per-component energies are in ``mp2.e_os`` / ``mp2.e_ss``.
    dispersion : DispersionResult or D4Result or None
        The dispersion correction. ``DispersionResult`` for D3(BJ),
        :class:`D4Result` for D4, or ``None`` when the dispatcher was
        called without a ``dispersion`` kwarg. Both result types
        expose ``.energy`` (Hartree).
    e_total : float
        Published total energy in Hartree:
        ``rks.energy + mp2.e_correlation + (dispersion.energy if dispersion else 0)``.
        With ``dispersion=None`` this is the un-dispersed XC + MP2
        total; with ``dispersion="d3bj"`` / ``"d4"`` it is the
        published ``X-D3BJ`` / ``X-D4`` total (e.g. B2PLYP-D3BJ,
        B2PLYP-D4, DSD-PBEP86-D3BJ, DSD-PBEP86-D4).
    functional : str
        Name passed to the dispatcher (e.g. "b2plyp").
    """

    __slots__ = ("rks", "mp2", "dispersion", "e_total", "functional")

    def __init__(self, rks, mp2, e_total, functional, dispersion=None):
        self.rks = rks
        self.mp2 = mp2
        self.dispersion = dispersion
        self.e_total = e_total
        self.functional = functional

    def __repr__(self):
        disp_str = (
            f", e_dispersion={self.dispersion.energy:.10f}"
            if self.dispersion is not None
            else ""
        )
        return (
            f"DoubleHybridResult(functional={self.functional!r}, "
            f"e_total={self.e_total:.10f}, "
            f"e_rks={self.rks.energy:.10f}, "
            f"e_mp2_corr={self.mp2.e_correlation:.10f}"
            f"{disp_str})"
        )


def _double_hybrid_citation_rows(refs) -> list[dict[str, object]]:
    """Flatten assembled citations into ``.system`` manifest rows."""
    rows: list[dict[str, object]] = []
    for c in getattr(refs, "citations", ()) or ():
        year = getattr(c, "year", None)
        rows.append(
            {
                "key": str(getattr(c, "key", "")),
                "kind": str(getattr(c, "kind", "")),
                "authors": "; ".join(getattr(c, "authors", ()) or ()),
                "title": str(getattr(c, "title", "")),
                "journal": str(getattr(c, "journal", None) or ""),
                "year": year if isinstance(year, int) else str(year or ""),
                "doi": str(getattr(c, "doi", None) or ""),
                "license": str(getattr(c, "license", None) or ""),
                "print": bool(getattr(c, "print", True)),
            }
        )
    return rows


def _double_hybrid_dispersion_key(dispersion) -> str | None:
    if dispersion is None:
        return None
    name = str(dispersion).lower()
    if name in {"d3bj", "d4"}:
        return name
    return None


def _double_hybrid_bond_orders_data(population_summary, mol):
    if not getattr(population_summary, "mayer_bonds", None):
        return None
    import math

    positions = [(a.xyz[0], a.xyz[1], a.xyz[2]) for a in mol.atoms]
    return {
        "method": "mayer",
        "pairs": [
            {
                "i": int(i),
                "j": int(j),
                "order": float(order),
                "symbol_i": str(si),
                "symbol_j": str(sj),
                "distance_ang": float(
                    math.dist(positions[i], positions[j]) * 0.529177210903
                ),
            }
            for (i, j, si, sj, order) in population_summary.mayer_bonds
        ],
    }


def _double_hybrid_dipole_data(population_summary):
    dipole = getattr(population_summary, "dipole", None)
    if dipole is None:
        return None
    return {
        "total_debye": float(dipole["total_debye"]),
        "vector_debye": [
            float(dipole["x_ebohr"]) * 2.541746473,
            float(dipole["y_ebohr"]) * 2.541746473,
            float(dipole["z_ebohr"]) * 2.541746473,
        ],
        "origin": [float(value) for value in dipole["origin_bohr"]],
    }


def _write_double_hybrid_outputs(
    *,
    mol,
    basis,
    result: DoubleHybridResult,
    functional_name: str,
    output,
    citations: bool,
    write_molden_file: bool,
    write_xyz_file: bool,
    write_population_file: bool,
    output_qvf: bool,
    dispersion,
    perf_log_path,
    structured_log_path,
    wall_seconds: float,
    nuclear_charges=None,
):
    """Emit standard sidecars for standalone double-hybrid dispatchers.

    The MP2 correction changes the energy but does not create a new
    post-SCF density in this dispatcher, so wavefunction-like artefacts
    deliberately use the converged SCF half while the log records the
    double-hybrid total and component breakdown.
    """
    stem = _Path(_os.fspath(output))
    stem.parent.mkdir(parents=True, exist_ok=True)
    basis_name = str(getattr(basis, "name", basis))

    from .banner import banner as _banner
    # Imported inside the function, not at module scope: vibeqc.output
    # reaches back into vibeqc for banner/properties, so a top-level
    # import here would be circular.
    from .output import (
        HeaderlessBlock,
        HeaderlessColumn,
        OutputChannel,
        OutputPlan,
        OutputWriter,
        Quantity,
        active_policy,
        write,
    )
    from .output._stem_paths import stem_sibling

    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method=functional_name,
        basis=basis_name,
        functional=functional_name,
        write_molden_file=write_molden_file,
        write_xyz=write_xyz_file,
        citations=citations,
        write_population=write_population_file,
        crash_dump=False,
        job_kind="post_scf",
        perf_log=perf_log_path or False,
        structured_log=structured_log_path or False,
        output_qvf=output_qvf,
    )
    writer = OutputWriter(plan)
    writer.update_run_fields(
        {
            "frozen_core_orbitals": 0,
            "frozen_core_convention": "all-electron-explicit",
        }
    )

    citation_block = ""
    bibtex_content = None
    if citations:
        try:
            from .output.citations import emit_citations, format_bibtex

            refs, bib_path, ref_path, citation_block = emit_citations(
                stem,
                method=functional_name,
                basis=basis_name,
                functional=functional_name,
                dispersion=_double_hybrid_dispersion_key(dispersion),
            )
            writer.set_citations(_double_hybrid_citation_rows(refs))
            writer.record(bib_path)
            writer.record(ref_path)
            bibtex_content = format_bibtex(refs)
        except Exception as exc:  # noqa: BLE001 - citation siblings are optional
            citation_block = (
                f"(warning: citation emission failed: "
                f"{type(exc).__name__}: {exc})\n"
            )

    log_path = stem_sibling(stem, ".out")
    disp_energy = (
        float(result.dispersion.energy) if result.dispersion is not None else 0.0
    )
    with OutputChannel.to_file(log_path):
        write(_banner())
        write("\n\n")
        write("Double-hybrid DFT dispatcher\n")
        write(f"  Functional:        {functional_name}\n")
        write(f"  Basis:             {basis_name}\n")
        write(f"  Charge:            {int(getattr(mol, 'charge', 0))}\n")
        write(f"  Multiplicity:      {int(getattr(mol, 'multiplicity', 1))}\n")
        write("  MP2 frozen core:   0 (explicit all-electron model component)\n")
        # The same content-sized headerless primitive owns every molecular
        # energy-components block. Values and the unit-bearing title follow
        # the active FormatPolicy; no caller guesses a rule width.
        energy_block = HeaderlessBlock(
            "Energy components",
            [
                HeaderlessColumn("<", min_width=32),
                HeaderlessColumn(">", min_width=18),
            ],
            gutter=1,
            unit_for="energy",
        )
        energy_block.add_row(
            "RKS reference", Quantity(float(result.rks.energy), "energy")
        )
        energy_block.add_row(
            "MP2 correlation",
            Quantity(float(result.mp2.e_correlation), "energy"),
        )
        if result.dispersion is not None:
            energy_block.add_row(
                "Dispersion", Quantity(disp_energy, "energy")
            )
        energy_block.add_row(
            "Total energy", Quantity(float(result.e_total), "energy")
        )
        energy_block.divider()
        write("\n" + energy_block.render(active_policy()) + "\n")
        write("\nSCF status\n")
        write(f"  Converged:         {bool(getattr(result.rks, 'converged', False))}\n")
        write(f"  Iterations:        {int(getattr(result.rks, 'n_iter', 0))}\n")
        write("\nSidecars\n")
        for pf in plan.files:
            if pf.role == "manifest":
                continue
            write(f"  {pf.role:12s} {pf.path.name}\n")
        if citation_block:
            from .output.citations import write_references_block

            # Same first-class citation printer the runners use; trailing
            # is conditional to keep the historical byte layout (the block
            # from emit_citations already ends in a newline, a warning
            # string may not).
            write_references_block(
                block=citation_block,
                leading="\n",
                trailing="" if citation_block.endswith("\n") else "\n",
            )
    writer.record(log_path)

    if write_molden_file:
        writer.dispatch_role(
            "orbitals",
            result=result.rks,
            basis=basis,
            molecule=mol,
            title=stem.name,
        )

    population_summary = None
    if write_population_file:
        from .output.formats.population import (
            compute_population_summary,
            write_population_summary,
        )

        population_summary = compute_population_summary(
            result.rks, basis, mol, nuclear_charges=nuclear_charges
        )
        pop_txt, pop_json = write_population_summary(stem, population_summary)
        writer.record(pop_txt)
        writer.record(pop_json)

    if write_xyz_file:
        writer.dispatch_role(
            "geometry",
            molecule=mol,
            energy_ha=float(result.e_total),
        )

    if output_qvf:
        from .output.formats.qvf import (
            assemble_run_record,
            qvf_wf_data,
            terminal_run_status,
            write_qvf,
        )

        wf_data = qvf_wf_data(result.rks, basis, mol) if write_molden_file else None
        run_record = assemble_run_record(plan, wall_seconds=wall_seconds)
        qvf_path = write_qvf(
            stem,
            plan,
            molecule=mol,
            result=result.rks,
            method=functional_name,
            functional=functional_name,
            basis=basis_name,
            bibtex_content=bibtex_content,
            population_summary=population_summary,
            wf_data=wf_data,
            wall_seconds=wall_seconds,
            run_record=run_record,
            run_status=terminal_run_status(result.rks),
            bond_orders_data=(
                _double_hybrid_bond_orders_data(population_summary, mol)
                if population_summary is not None
                else None
            ),
            dipole_moment_data=(
                _double_hybrid_dipole_data(population_summary)
                if population_summary is not None
                else None
            ),
        )
        writer.record(qvf_path)

    return writer


def _double_hybrid_log_target(option, stem, suffix: str, env_name: str):
    """Resolve one standalone-dispatcher telemetry destination."""
    # Local import: vibeqc.output reaches back into vibeqc, so a module
    # scope import here would be circular (see _write_double_hybrid_outputs).
    from .output._stem_paths import stem_sibling

    if option is True:
        return stem_sibling(stem, suffix) if stem is not None else None
    if option is False:
        return None
    if option is None:
        env_value = _os.environ.get(env_name, "").strip()
        return _Path(env_value) if env_value else None
    return _Path(_os.fspath(option))


def _double_hybrid_memory_snapshot(label: str, slog) -> None:
    """Record the same RSS sample in the perf and structured logs."""
    tracker = active_tracker()
    if tracker is None:
        return
    tracker.snapshot_memory(label)
    sample = tracker.memory_snapshots[-1]
    slog.emit(
        "memory_snapshot",
        label=str(sample["label"]),
        rss_mb=float(sample["rss_mb"]),
        elapsed_wall_s=float(sample["t_s"]),
        phase="double_hybrid",
    )


def _record_double_hybrid_scf_telemetry(result, functional_name: str, slog) -> None:
    """Replay the native molecular SCF trace into both telemetry streams."""
    tracker = active_tracker()
    for step in getattr(result, "scf_trace", ()) or ():
        # wall_s == 0.0 is the C++ "unmeasured" sentinel; None renders
        # as "--" in the perf table (BUG87-A truthful timers).
        _step_wall = float(getattr(step, "wall_s", 0.0))
        row = {
            "iter": int(step.iter),
            "energy": float(step.energy),
            "dE": float(step.delta_e) if step.iter > 1 else None,
            "grad": float(step.grad_norm),
            "diis": int(step.diis_subspace),
            "wall_s": _step_wall if _step_wall > 0.0 else None,
        }
        if tracker is not None:
            tracker.add_scf_iter(**row)
        slog.emit(
            "scf_iter",
            iter=row["iter"],
            energy=row["energy"],
            dE=row["dE"],
            grad_norm=row["grad"],
            diis_subspace=row["diis"],
            phase="double_hybrid_scf",
            functional=functional_name,
        )


def _run_double_hybrid_core(
    mol,
    basis,
    functional,
    functional_name,
    *,
    density_fit,
    aux_basis,
    density_fit_mp2,
    aux_basis_mp2,
    dispersion,
    rks_options,
    slog,
    grid_level="orca-defgrid3",
):
    """Run the SCF, perturbative correlation, and dispersion phases."""
    from .runner import apply_ks_grid_default
    open_shell = mol.multiplicity > 1
    t_scf = _time.perf_counter()
    _double_hybrid_memory_snapshot("start_of_scf", slog)
    with PerfScope("double_hybrid.reference_scf"):
        if open_shell:
            from .roks import ROKSOptions
            from .roks import run_roks as _run_roks

            roks_options = ROKSOptions()
            if rks_options is not None:
                # The public reference options apply to either spin family.
                # Copy common controls, including the complete guess payload.
                from dataclasses import fields
                for field in fields(roks_options):
                    if hasattr(rks_options, field.name):
                        setattr(roks_options, field.name,
                                getattr(rks_options, field.name))
            apply_ks_grid_default(roks_options, grid_level)
            roks_options.functional = functional_name
            roks_options.density_fit = density_fit
            if density_fit:
                roks_options.aux_basis = aux_basis or default_aux_basis_for(
                    basis.name, kind="jk"
                )
            elif aux_basis:
                roks_options.aux_basis = aux_basis
            rks_result = _run_roks(
                mol,
                basis,
                roks_options,
                _allow_double_hybrid_scf=True,
            )
        else:
            if rks_options is None:
                rks_options = RKSOptions()
            apply_ks_grid_default(rks_options, grid_level)
            rks_options.functional = functional_name
            rks_options.density_fit = density_fit
            if density_fit:
                rks_options.aux_basis = aux_basis or default_aux_basis_for(
                    basis.name, kind="jk"
                )
            elif aux_basis:
                rks_options.aux_basis = aux_basis
            rks_result = run_rks(
                mol,
                basis,
                rks_options,
                _allow_double_hybrid_scf=True,
            )
    scf_wall_s = _time.perf_counter() - t_scf
    _record_double_hybrid_scf_telemetry(rks_result, functional_name, slog)
    _double_hybrid_memory_snapshot("end_of_scf", slog)
    slog.emit(
        "scf_converged",
        method="roks" if open_shell else "rks",
        functional=functional_name,
        converged=bool(rks_result.converged),
        n_iter=int(rks_result.n_iter),
        energy=float(rks_result.energy),
        wall_s=float(scf_wall_s),
    )
    if not rks_result.converged:
        reference = "open-shell ROKS" if open_shell else "hybrid SCF"
        options_hint = (
            "ROKSOptions (max_iter, level_shift)"
            if open_shell
            else "RKSOptions (max_iter, accelerator)"
        )
        raise RuntimeError(
            f"run_double_hybrid({functional_name!r}): {reference} step "
            f"did not converge (n_iter={rks_result.n_iter}, "
            f"energy={rks_result.energy:.6f}). Tighten "
            f"{options_hint} and retry."
        )

    t_mp2 = _time.perf_counter()
    with PerfScope("double_hybrid.mp2"):
        if open_shell:
            from .dlpno._ccsd_ref import RefMP2Result

            rohf_mp2 = run_rohf_mp2(
                mol,
                basis,
                rks_result,
                n_frozen_core=0,
                aux_basis=(aux_basis_mp2 or None),
            )
            e_mp2_corr = (
                functional.mp2_c_os * rohf_mp2.e_os
                + functional.mp2_c_ss * rohf_mp2.e_ss
            )
            mp2_result = RefMP2Result(
                e_corr=e_mp2_corr,
                e_singles=0.0,
                e_doubles=rohf_mp2.e_doubles,
                e_os=rohf_mp2.e_os,
                e_ss=rohf_mp2.e_ss,
                e_hf=float(rks_result.energy),
                e_total=float(rks_result.energy) + e_mp2_corr,
            )
            mp2_density_fit = True
            mp2_aux_basis = str(aux_basis_mp2 or "")
        else:
            mp2_opts = MP2Options()
            # vibe-qc's existing double-hybrid contract uses an all-electron
            # scaled-MP2 component. Keep that model definition explicit when
            # standalone MP2 adopts the published frozen-core default (#140).
            mp2_opts.n_frozen_core = 0
            mp2_opts.density_fit = density_fit_mp2
            mp2_opts.aux_basis = _resolve_mp2_aux_basis(
                basis, density_fit_mp2, aux_basis_mp2
            )
            mp2_opts.c_os = functional.mp2_c_os
            mp2_opts.c_ss = functional.mp2_c_ss
            rhf_ref = rhf_result_from_rks(rks_result)
            mp2_result = run_mp2(mol, basis, rhf_ref, mp2_opts)
            mp2_density_fit = bool(mp2_opts.density_fit)
            mp2_aux_basis = str(mp2_opts.aux_basis or "")
    mp2_wall_s = _time.perf_counter() - t_mp2
    _double_hybrid_memory_snapshot("end_of_mp2", slog)
    slog.emit(
        "double_hybrid_mp2_done",
        functional=functional_name,
        open_shell=bool(open_shell),
        density_fit=mp2_density_fit,
        aux_basis=mp2_aux_basis,
        n_frozen_core=0,
        frozen_core_convention="all-electron-explicit",
        c_os=float(functional.mp2_c_os),
        c_ss=float(functional.mp2_c_ss),
        e_os=float(mp2_result.e_os),
        e_ss=float(mp2_result.e_ss),
        e_correlation=float(mp2_result.e_correlation),
        e_total=float(mp2_result.e_total),
        solver_iterations=None,
        iteration_contract="direct_noniterative",
        wall_s=float(mp2_wall_s),
    )

    disp_result = None
    dispersion_wall_s = 0.0
    if dispersion is not None:
        t_dispersion = _time.perf_counter()
        with PerfScope("double_hybrid.dispersion"):
            disp_name = dispersion.lower()
            if disp_name == "d4":
                disp_result = compute_d4(
                    mol,
                    functional_name,
                    charge=float(mol.charge),
                )
            elif disp_name == "d3bj":
                disp_result = compute_d3bj(mol, functional_name)
            else:
                raise ValueError(
                    f"run_double_hybrid: dispersion={dispersion!r} not "
                    "recognised. Supported: None (un-dispersed), "
                    "'d3bj', 'd4'."
                )
        dispersion_wall_s = _time.perf_counter() - t_dispersion
        _double_hybrid_memory_snapshot("end_of_dispersion", slog)
        slog.emit(
            "double_hybrid_dispersion_done",
            functional=functional_name,
            model=str(dispersion).lower(),
            energy=float(disp_result.energy),
            solver_iterations=None,
            iteration_contract="not_defined",
            wall_s=float(dispersion_wall_s),
        )

    e_total = rks_result.energy + mp2_result.e_correlation
    if disp_result is not None:
        e_total += disp_result.energy
    result = DoubleHybridResult(
        rks_result,
        mp2_result,
        e_total,
        functional_name,
        dispersion=disp_result,
    )
    scaled_os = float(functional.mp2_c_os) * float(mp2_result.e_os)
    scaled_ss = float(functional.mp2_c_ss) * float(mp2_result.e_ss)
    slog.emit(
        "double_hybrid_post_scf",
        functional=functional_name,
        reference_scf_energy=float(rks_result.energy),
        raw_os_correlation=float(mp2_result.e_os),
        raw_ss_correlation=float(mp2_result.e_ss),
        os_scale=float(functional.mp2_c_os),
        ss_scale=float(functional.mp2_c_ss),
        scaled_os_correlation=scaled_os,
        scaled_ss_correlation=scaled_ss,
        mp2_correlation=float(mp2_result.e_correlation),
        n_frozen_core=0,
        frozen_core_convention="all-electron-explicit",
        dispersion_energy=(
            float(disp_result.energy) if disp_result is not None else 0.0
        ),
        assembled_total_energy=float(result.e_total),
        reference_scf_wall_s=float(scf_wall_s),
        mp2_wall_s=float(mp2_wall_s),
        dispersion_wall_s=float(dispersion_wall_s),
        reference_scf_iterations=int(rks_result.n_iter),
        solver_iterations=None,
        iteration_contract="direct_noniterative",
    )
    return result


def run_double_hybrid(
    mol,
    basis,
    functional_name,
    *,
    density_fit=True,
    aux_basis="",
    density_fit_mp2=True,
    aux_basis_mp2="",
    dispersion=None,
    rks_options=None,
    grid_level="orca-defgrid3",
    output=None,
    citations=True,
    output_qvf=True,
    write_molden_file=True,
    write_xyz_file=True,
    write_population_file=True,
    perf_log=None,
    structured_log=False,
):
    """Generic double-hybrid dispatcher.

    The ``grid_level`` preset (default ``"orca-defgrid3"``) applies to
    absent or untouched reference options. Custom grid fields win; pass
    ``grid_level="legacy"`` to reproduce the old reference grid.

    Resolves ``functional_name`` to a :class:`Functional`, runs a
    hybrid-DFT SCF step with that functional's SCF piece, runs an MP2
    correction on the converged KS orbitals with the functional's
    ``mp2_c_os`` / ``mp2_c_ss`` scaling, optionally adds a dispersion
    correction, and returns a :class:`DoubleHybridResult` carrying all
    pieces and the combined total energy.

    ``functional_name`` must resolve to an ``is_double_hybrid``
    functional (currently ``"b2plyp"``, ``"dsd-pbep86"``, or
    ``"pwpb95"``); a ``ValueError`` is raised otherwise. The thin
    wrappers :func:`run_b2plyp`, :func:`run_dsd_pbep86`, and
    :func:`run_pwpb95` are the name-specific entry points most callers
    will use. ``"pwpb95"`` resolves to a meta-GGA functional, so its
    SCF step runs through the tau-dependent Kohn-Sham path; the
    dispatcher handles that transparently.

    Parameters
    ----------
    mol, basis : Molecule, BasisSet
        The molecular system. Closed-shell runs RKS + RMP2; open-shell
        (multiplicity > 1) runs the spin-pure ROKS SCF half + a semicanonical
        ROHF-MP2 doubles correction.
    functional_name : str
        Name of a double-hybrid functional registered in vibe-qc's
        :class:`Functional` resolver.
    density_fit : bool, default True
        Density-fit the RKS hybrid SCF step (J via RI-J, K via RI-K
        since double hybrids in this line all have a_HF > 0).
    aux_basis : str, default ""
        Aux basis for the SCF DF. Empty -> auto-resolve a JKfit aux
        from the orbital basis name.
    density_fit_mp2 : bool, default True
        Use RI-MP2 (recommended) for the post-SCF correction. False
        falls back to canonical MP2 -- only useful for parity
        validation against canonical-MP2 reference codes.
    aux_basis_mp2 : str, default ""
        Aux basis for the MP2 DF. Empty -> auto-resolve a per-zeta
        RIfit aux from the orbital basis name. Ignored when
        ``density_fit_mp2=False``.
    dispersion : str or None, default None
        Optional dispersion correction folded into
        :attr:`DoubleHybridResult.e_total`. Supported:

        * ``None`` -- un-dispersed XC + MP2 total (the published
          "no-D" energy).
        * ``"d3bj"`` -- Grimme-Antony-Ehrlich-Krieg 2010 D3 with
          Becke-Johnson damping, parameters for the functional read
          from the existing :func:`vibeqc.compute_d3bj` framework
          (prefers the optional ``dftd3`` backend when installed,
          falls back to the vibe-qc native stub).
        * ``"d4"`` -- Caldeweyher-Bannwarth-Grimme 2019 D4 via the
          optional ``dftd4`` package; see :func:`vibeqc.compute_d4`.

        The dispersion energy is stored on
        :attr:`DoubleHybridResult.dispersion` as either a
        :class:`DispersionResult` (D3-BJ) or a :class:`D4Result` (D4);
        both expose ``.energy`` so consumers can treat them
        uniformly.
    rks_options : RKSOptions, optional
        Caller-provided options for the SCF step. The functional
        field is overridden with ``functional_name`` regardless;
        everything else is honoured. If ``None``, a default
        :class:`RKSOptions` is used with ``density_fit`` /
        ``aux_basis`` applied per the keyword arguments above.
    output : str or PathLike, optional
        Output stem. When provided, the standalone dispatcher writes
        ``.out`` + ``.system`` and the same default-on molecular
        sidecars as :func:`run_job`: Molden orbitals, XYZ geometry,
        population/property JSON/TXT, citation siblings, and a QVF
        archive unless the corresponding opt-out keyword is false.
    citations : bool, default True
        Emit ``.bibtex`` / ``.references`` siblings and record the
        assembled citation provenance in ``.system``.
    output_qvf : bool, default True
        Emit ``{output}.qvf``. The archive contains the converged SCF
        wavefunction and SCF-density properties plus provenance naming
        the double-hybrid dispatcher; the MP2 correction contributes to
        the reported total energy but does not define a separate
        post-SCF density.
    write_molden_file, write_xyz_file, write_population_file : bool, default True
        Opt out of the matching sidecars when ``output`` is provided.
    perf_log, structured_log : bool or path-like, optional
        Opt in to standalone-dispatcher performance and NDJSON telemetry.
        ``True`` writes ``{output}.perf`` and
        ``{output}.scf.jsonl``; a path-like selects an explicit destination.
        ``False`` disables that artifact, and ``None`` defers to
        ``VIBEQC_PERFLOG`` / ``VIBEQC_STRUCTURED_LOG``.

    Returns
    -------
    DoubleHybridResult
    """
    # ECP references are consumed like everywhere else since 2026-09: the
    # SCF wrappers attach the basis sidecar, the ROKS driver applies the
    # operator for open shells, and the MP2 kernels partition the valence
    # count (this route's scaled MP2 runs all-electron on that count).
    basis = _coerce_molecular_basis(mol, basis)
    functional = Functional(functional_name)
    if not functional.is_double_hybrid:
        raise ValueError(
            f"run_double_hybrid: {functional_name!r} resolves to a "
            f"non-double-hybrid functional (mp2_c_os = "
            f"{functional.mp2_c_os}, mp2_c_ss = {functional.mp2_c_ss}). "
            "Use run_rks for plain hybrid / GGA / LDA functionals."
        )

    stem = _Path(_os.fspath(output)) if output is not None else None
    perf_target = _double_hybrid_log_target(
        perf_log,
        stem,
        ".perf",
        "VIBEQC_PERFLOG",
    )
    structured_target = _double_hybrid_log_target(
        structured_log,
        stem,
        ".scf.jsonl",
        "VIBEQC_STRUCTURED_LOG",
    )

    from .output.formats.perf import perf_log as _perf_log_context
    from .output.formats.structured_log import (
        structured_log as _structured_log_context,
    )
    from .output._stem_paths import stem_sibling
    from contextlib import nullcontext as _nullcontext

    t_job = _time.perf_counter()
    writer = None
    perf_context = (
        _nullcontext(PerfTracker())
        if perf_log is False
        else _perf_log_context(perf_target)
    )
    with (
        perf_context as tracker,
        _structured_log_context(
            structured_target,
            enabled=structured_target is not None,
        ) as slog,
        PerfScope("double_hybrid.total"),
    ):
        libs = library_versions()
        slog.emit(
            "banner",
            vibeqc_version=VIBEQC_VERSION,
            libint=libs.get("libint", "unknown"),
            libxc=libs.get("libxc", "unknown"),
            spglib=libs.get("spglib", "unknown"),
            run_fingerprint=run_fingerprint(
                method="double-hybrid",
                basis=str(getattr(basis, "name", basis)),
                functional=functional_name,
                molecule=mol,
            ),
        )
        slog.emit(
            "job_start",
            method="double-hybrid",
            basis=str(getattr(basis, "name", basis)),
            functional=functional_name,
            optimize=False,
            threads=int(get_num_threads()),
            n_atoms=int(len(list(mol.atoms))),
            charge=int(mol.charge),
            multiplicity=int(mol.multiplicity),
            n_electrons=int(mol.n_electrons()),
            output_stem=str(stem) if stem is not None else None,
        )
        _double_hybrid_memory_snapshot("job_start", slog)
        result = _run_double_hybrid_core(
            mol,
            basis,
            functional,
            functional_name,
            density_fit=density_fit,
            aux_basis=aux_basis,
            density_fit_mp2=density_fit_mp2,
            aux_basis_mp2=aux_basis_mp2,
            dispersion=dispersion,
            rks_options=rks_options,
            slog=slog,
            grid_level=grid_level,
        )
        if stem is not None:
            with PerfScope("double_hybrid.output"):
                writer = _write_double_hybrid_outputs(
                    mol=mol,
                    basis=basis,
                    result=result,
                    functional_name=functional_name,
                    output=stem,
                    citations=citations,
                    write_molden_file=write_molden_file,
                    write_xyz_file=write_xyz_file,
                    write_population_file=write_population_file,
                    output_qvf=output_qvf,
                    dispersion=dispersion,
                    perf_log_path=perf_target,
                    structured_log_path=structured_target,
                    wall_seconds=_time.perf_counter() - t_job,
                    # ECP systems: Z - n_core in every electrostatic property
                    # (#642); bare Z when the RKS options carry no ECP.
                    nuclear_charges=effective_nuclear_charges(mol, rks_options),
                )
        _double_hybrid_memory_snapshot("job_end", slog)
        total_wall_s = _time.perf_counter() - t_job
        slog.emit(
            "job_end",
            total_wall_s=float(total_wall_s),
            reference_scf_wall_s=float(
                tracker.phases.get("double_hybrid.reference_scf").wall_s
                if "double_hybrid.reference_scf" in tracker.phases
                else 0.0
            ),
            n_iter=int(result.rks.n_iter),
            converged=bool(result.rks.converged),
            energy=float(result.e_total),
            energy_kind="assembled_total",
            e_scf=float(result.rks.energy),
            e_mp2_correlation=float(result.mp2.e_correlation),
            e_dispersion=(
                float(result.dispersion.energy)
                if result.dispersion is not None
                else 0.0
            ),
            out_path=(str(stem_sibling(stem, ".out")) if stem is not None else None),
        )

    if writer is not None:
        for telemetry_path in (structured_target, perf_target):
            if telemetry_path is not None:
                writer.record(telemetry_path)
        writer.finish(wall_seconds=_time.perf_counter() - t_job)
    return result


def run_b2plyp(mol, basis, **kwargs):
    """B2PLYP (Grimme, *J. Chem. Phys.* **124**, 034108 (2006)).

    Thin wrapper around :func:`run_double_hybrid` with
    ``functional_name="b2plyp"``. Runs the hybrid SCF step
    (0.53.HF + 0.47.B88 + 0.73.LYP) and the RI-MP2 correction
    (c_os = c_ss = 0.27) on the converged KS orbitals. See
    :func:`run_double_hybrid` for the keyword arguments and
    :class:`DoubleHybridResult` for the return value.
    """
    return run_double_hybrid(mol, basis, "b2plyp", **kwargs)


def run_dsd_pbep86(mol, basis, **kwargs):
    """DSD-PBEP86 (Kozuch & Martin, *Phys. Chem. Chem. Phys.* **13**,
    20104 (2011)) -- no D dispersion.

    Thin wrapper around :func:`run_double_hybrid` with
    ``functional_name="dsd-pbep86"``. Runs the hybrid SCF step
    (0.70.HF + 0.30.PBE + 0.43.P86(VWN5)) and the RI-MP2 correction with
    asymmetric scaling (c_os = 0.53, c_ss = 0.25) on the converged
    KS orbitals. These are the final recommended D3(BJ)-fit
    parameters of the 2011 paper (p. 20106) -- the same set ORCA's
    ``DSD-PBEP86`` keyword uses; the 2013 JCC revision
    (doi:10.1002/jcc.23391) re-optimized every coefficient and is a
    *different* parameterization that vibe-qc deliberately does not
    map to this name (see the ``dsd-pbep86`` entry in
    ``cpp/src/xc.cpp``).

    The published DSD-PBEP86 method usually pairs with the
    Grimme-Antony-Ehrlich-Krieg D3(BJ) dispersion correction. This
    dispatcher returns the un-dispersed XC + MP2 total -- add the
    dispersion correction via :func:`compute_d3bj` post-hoc when you
    need the published "with-D" energy. A dedicated D4 parameter set
    for DSD-PBEP86 is a separate upcoming item.

    See :func:`run_double_hybrid` for keyword arguments and
    :class:`DoubleHybridResult` for the return value.
    """
    return run_double_hybrid(mol, basis, "dsd-pbep86", **kwargs)


def run_revdsd_pbep86(mol, basis, **kwargs):
    """revDSD-PBEP86-D4 (Santra, Sylvetsky & Martin, *J. Phys. Chem. A*
    **123**, 5129 (2019)) -- the GMTKN55-retrained revision of
    DSD-PBEP86.

    Thin wrapper around :func:`run_double_hybrid` with
    ``functional_name="revdsd-pbep86"``. Runs the hybrid SCF step
    (0.69.HF + 0.31.PBE-X + 0.4210.P86-C) and the RI-MP2 correction
    with asymmetric scaling (c_os = 0.5922, c_ss = 0.0636) on the
    converged KS orbitals.

    The published method is **revDSD-PBEP86-D4**: it pairs with the
    Caldeweyher-Bannwarth-Grimme D4 dispersion correction (the
    coefficients above were jointly fit *with* D4). Pass
    ``dispersion="d4"`` to fold the D4 term into
    :attr:`DoubleHybridResult.e_total` for the published total; the
    default (``dispersion=None``) returns the un-dispersed XC + MP2
    energy. The D4 damping (s6 = 0.5132, a1 = 0.44, a2 = 3.60) is read
    automatically from :mod:`vibeqc.dispersion_d4_parameters` via the
    ``"revdsdpbep86"`` key.

    Note the -D3BJ variant uses a *different* XC/MP2 fit and is not
    reachable through this alias (it is the D4 member specifically).

    See :func:`run_double_hybrid` for keyword arguments and
    :class:`DoubleHybridResult` for the return value.
    """
    return run_double_hybrid(mol, basis, "revdsd-pbep86", **kwargs)


def run_pwpb95(mol, basis, **kwargs):
    """PWPB95 (Goerigk & Grimme, *J. Chem. Theory Comput.* **7**, 291
    (2011)) -- spin-opposite-scaled double-hybrid meta-GGA.

    Thin wrapper around :func:`run_double_hybrid` with
    ``functional_name="pwpb95"``. Runs the hybrid meta-GGA SCF step
    (0.50.HF + 0.50.PW6-modified-mPW exchange + 0.731.B95 correlation)
    and the post-SCF MP2 correction on the converged KS orbitals.

    PWPB95 is the **SOS** member of the double-hybrid line: the
    same-spin MP2 coefficient is zero, so the correction is
    opposite-spin MP2 only (c_os = 0.269, c_ss = 0.0). It is also the
    first meta-GGA double hybrid in vibe-qc -- the SCF step runs through
    the t-dependent Kohn-Sham path because the B95 correlation
    component is a meta-GGA. Unlike B2PLYP / DSD-PBEP86, PWPB95 mixes
    *reparametrised* libxc components (the "PW6" mPW91 exchange and a
    re-tuned B95 correlation); those parameters are baked into the
    ``Functional("pwpb95")`` resolver.

    The published method is **PWPB95-D3(BJ)** -- Goerigk-Grimme pair it
    with Grimme's D3 Becke-Johnson dispersion. Pass ``dispersion="d3bj"``
    (or ``"d4"`` for the D4 re-parameterisation) to fold the dispersion
    correction into :attr:`DoubleHybridResult.e_total`; the default
    (``dispersion=None``) returns the un-dispersed XC + MP2 total.

    See :func:`run_double_hybrid` for keyword arguments and
    :class:`DoubleHybridResult` for the return value.
    """
    return run_double_hybrid(mol, basis, "pwpb95", **kwargs)


# ---------------------------------------------------------------------
# wB97X-D -- range-separated hybrid + Chai-Head-Gordon empirical
# dispersion. Like the double-hybrid dispatchers above, the complete
# wB97X-D energy is "SCF + an additive correction": the libxc XC part
# (a range-separated hybrid GGA, id 471) plus the geometry-only "-D"
# dispersion term. ``functional="wb97x-d"`` through plain run_rks /
# run_uks gives the XC/SCF piece only; ``run_wb97x_d`` adds the
# dispersion for the published total -- the same alias-vs-dispatcher
# split as ``b2plyp`` / ``run_b2plyp``.
# ---------------------------------------------------------------------


class WB97XDResult:
    """Result of an wB97X-D run: the range-separated-hybrid SCF result,
    the Chai-Head-Gordon empirical dispersion correction, and the
    combined total energy.

    Attributes
    ----------
    scf : RKSResult or UKSResult
        The converged wB97X-D range-separated-hybrid SCF step (the
        libxc XC_HYB_GGA_XC_WB97X_D functional). ``RKSResult`` for a
        closed-shell system, ``UKSResult`` for open-shell.
    dispersion : DispersionResult
        The Chai-Head-Gordon ("-D") dispersion correction. Exposes
        ``.energy`` (Hartree) and ``.gradient``.
    e_total : float
        Published wB97X-D total energy (Hartree):
        ``scf.energy + dispersion.energy``.
    functional : str
        Always ``"wb97x-d"``.
    """

    __slots__ = ("scf", "dispersion", "e_total", "functional")

    def __init__(self, scf, dispersion, e_total):
        self.scf = scf
        self.dispersion = dispersion
        self.e_total = e_total
        self.functional = "wb97x-d"

    def __repr__(self):
        return (
            f"WB97XDResult(e_total={self.e_total:.10f}, "
            f"e_scf={self.scf.energy:.10f}, "
            f"e_dispersion={self.dispersion.energy:.10f})"
        )


def run_wb97x_d(mol, basis, options=None, *, grid_level="orca-defgrid3", output=None):
    """wB97X-D (Chai & Head-Gordon, *Phys. Chem. Chem. Phys.* **10**,
    6615 (2008)) -- the full functional, XC + dispersion.

    wB97X-D is a range-separated hybrid GGA (libxc
    ``XC_HYB_GGA_XC_WB97X_D``) carrying an intrinsic empirical
    dispersion correction of the older "DFT-D2" family. The two pieces
    are independent -- the dispersion is a geometry-only additive term
    with no SCF coupling -- so this dispatcher:

    1. runs the range-separated-hybrid SCF (``run_rks`` for a
       closed-shell system, ``run_uks`` for open-shell, picked from
       ``mol.multiplicity()``) with ``functional="wb97x-d"``;
    2. computes the Chai-Head-Gordon dispersion
       (:func:`vibeqc.compute_chg_dispersion`);
    3. returns a :class:`WB97XDResult` with
       ``e_total = scf.energy + dispersion.energy``.

    Range-separated hybrids run via direct SCF (the erf-attenuated
    exchange is built on the fly); ``density_fit`` is not supported and
    raises inside the SCF step.

    Parameters
    ----------
    mol, basis : Molecule, BasisSet
        The molecular system. Closed- or open-shell.
    options : RKSOptions or UKSOptions, optional
        Options for the SCF step. The ``functional`` field is
        overridden with ``"wb97x-d"``. If ``None``, a default is
        created matching the system's shell type. For open-shell
        wB97X-D on orbital-near-degenerate radicals a level shift
        (``options.level_shift ≈ 0.5``) may be needed -- see the
        wB97X note in ``docs/user_guide/functionals.md``.

    grid_level : str
        Grid preset for absent or untouched options (``"orca-defgrid3"``).
        Custom grid fields win; choose ``"legacy"`` for the old defaults.

    Returns
    -------
    WB97XDResult
    """
    closed_shell = mol.multiplicity == 1
    if options is None:
        options = RKSOptions() if closed_shell else UKSOptions()
    from .runner import apply_ks_grid_default

    apply_ks_grid_default(options, grid_level)
    options.functional = "wb97x-d"

    if closed_shell:
        scf = run_rks(mol, basis, options)
    else:
        scf = run_uks(mol, basis, options)
    if not scf.converged:
        raise RuntimeError(
            f"run_wb97x_d: wB97X-D SCF did not converge "
            f"(n_iter={scf.n_iter}, energy={scf.energy:.6f}). For "
            "open-shell radicals try options.level_shift = 0.5."
        )

    disp = compute_chg_dispersion(mol)
    if output is not None:
        try:
            from .output.citations import emit_citations

            emit_citations(
                output,
                method="wb97x-d",
                basis=basis.name,
                functional="wb97x-d",
            )
        except Exception:
            pass
    return WB97XDResult(scf, disp, scf.energy + disp.energy)


# Implicit solvation (CPCM / COSMO) -- v0.9.0 headline feature. Coupled
# self-consistently with the molecular SCF via macro-iteration over an
# apparent surface charge on a Lebedev-tessellated Bondi cavity. Wired
# through ``run_job(solvent=...)`` and the ASE calculator.
from . import solvation

# Non-mean-field solvers (v0.9.0)
from . import solvers as solvers  # noqa: F401

# The ASE GPW Calculator wrapper (vibeqc.ase_periodic_gpw) hard-requires ASE.
# ASE is a declared runtime dependency for normal installs, but the guard keeps
# source-tree / --no-deps imports usable until the user touches an ASE surface.
try:
    from .ase_periodic_gpw import VibeqcGPW
except ImportError:  # pragma: no cover - ASE optional extra not installed
    pass
from .bands import (
    BandStructure,
    DensityOfStates,
    FrontierBands,
    KPath,
    ProjectedDensityOfStates,
    ao_groups_per_atom,
    ao_groups_per_atom_l,
    band_structure,
    band_structure_hcore,
    density_of_states,
    density_of_states_hcore,
    density_of_states_projected,
    density_of_states_projected_hcore,
    frontier_bands,
    kpath_from_segments,
    per_k_band_energies,
)
from .basis_filter import (
    BasisFilterReport,
    DroppedPrimitive,
    clear_filtered_basis_cache,
    filter_basis_by_exponent,
    format_basis_filter_report,
    make_basis,
)
from .basis_registry import (
    BasisRecord,
    EcpRequirement,
    basis_replaces_core,
    canonical_basis_name,
    default_aux_basis,
    ecp_requirements,
    element_coverage,
)
from .basis_registry import lookup as basis_registry_lookup
from .bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_rks,
    compute_bipole_gradient_uhf,
    compute_bipole_gradient_uks,
    compute_stress_tensor,
)
from .bipole_optimize import (
    OptimizeResult,
    relax_atoms,
    relax_cell,
    relax_cell_gradient,
    relax_full,
)
from .build import (
    BULK_LATTICE_CONSTANTS,
    SlabInfo,
    place_adsorbate,
    slab,
    slab_2d,
    synthesize_slab_a3,
)
from .build import (
    molecule as build_molecule,
)
from .cc import (
    BruecknerCCDResult,
    BruecknerIteration,
    CCSDOptions,
    chemical_core_orbital_count,
    run_bccd,
    run_ccsd,
    run_rohf_ccsd,
    run_rohf_mp2,
    run_uccsd,
    run_uccsd_from_mos,
)
from .correlation_conventions import (
    PUBLISHED_FROZEN_CORE_CONVENTION,
    published_frozen_core_orbital_count,
    resolve_frozen_core,
    resolve_frozen_core_count,
)
from .cif import read_cif
from .coop_cohp import (
    COOPCOHPResult,
    ao_pairs_per_atom_pair,
    compute_coop_cohp,
    periodic_mayer_bond_orders,
)
from .cphf import (
    CPHFConvergenceError,
    CPHFOptions,
    cphf_solve_rhf,
    dipole_polarizability_rhf,
)
from .crash_dump import (
    active_crash_dump_stem,
    classify_failure,
    crash_dump_context,
    dump_on_failure,
    load_dump,
)
from .cube import (
    CubeGrid,
    make_uniform_grid,
    write_cube_density,
    write_cube_mo,
    write_cube_mos,
)
from .dimer import (
    DimerResult,
    run_dimer,
)
from .dispersion_periodic import (
    PeriodicDispersionResult,
    compute_d3bj_periodic,
)
from .ecp_metadata import (
    attach_inline_ecp_options_from_basis_sidecar,
    auto_ecp_centers,
    basis_sidecar_replaces_core,
    ecp_centers_from_dict,
    EcpHeader,
    effective_nuclear_charges,
    inline_ecp_data_for,
    InlineECPRecord,
    is_ecp_paired_basis,
    library_for,
    molecular_options_replace_core,
    parse_inline_ecp_sidecar,
    parse_sidecar_path,
    sidecar_path_for,
    validate_ecp_required,
)
from .eigs_preflight import (
    DisambiguationReport,
    EIGSReport,
    TruncationOptimizationReport,
    disambiguate_critical_overlap,
    eigs_preflight,
    format_disambiguation_report,
    format_eigs_report,
    format_truncation_optimization_report,
    optimize_truncation,
)
from .ewald_composed import (
    auto_ewald_alpha,
    build_j_ewald_3d,
    compute_j_ewald_3d_ft_gamma,
    compute_j_ewald_3d_ft_lattice,
    makov_payne_coefficient_cubic,
)
from .ewald_j import auto_grid, build_j_long_range
from .geomopt import (
    CartesianCoordinates,
    CellStrainCoordinates,
    ConvergencePolicy,
    ConvergenceReport,
    CoordinateRepresentation,
    DelocalizedInternalCoordinates,
    EnergyGradientProvider,
    FractionalCoordinates,
    GeomOptResult,
    HessianProvider,
    HistoryManager,
    HistoryRecord,
    MolecularHessianFDProvider,
    MolecularSCFProvider,
    OptimizerState,
    PeriodicSCFProvider,
    RestartSerializer,
    bfgs,
    conjugate_gradient,
    lbfgs,
    resolve_optimizer,
    run_geomopt,
    run_periodic_geomopt,
    steepest_descent,
)
from .hessian import (
    HessianFDOptions,
    HessianResult,
    compute_hessian_fd,
    ir_intensities,
)
from .hessian_analytic import compute_hessian_rhf_analytic
from .hessian_analytic_rks import compute_hessian_rks_analytic
from .hessian_analytic_uhf import compute_hessian_uhf_analytic
from .hessian_analytic_uks import compute_hessian_uks_analytic
from .output.formats.trexio import (
    TrexioData,
    TrexioMOBlock,
    TrexioSparse,
    read_trexio,
    read_trexio_fields,
    write_trexio,
    write_trexio_fields,
)
from .qcschema import (
    molecule_from_qcschema,
    molecule_to_qcschema,
    read_qcschema,
    run_qcschema,
)
from .output.formats.qcschema import write_qcschema
from .io import (
    normal_mode_trajectory,
    write_molden,
    write_opt_trajectory,
    write_orca_hess,
    write_xyz_trajectory,
)
from .irc import (
    IRCResult,
    run_irc,
)
from .kpoints import KPointConvergence, KPoints, as_bloch_kmesh
from .lattice_screening import (
    RcutStrategy,
    estimate_rcut_pyscf,
    estimate_rcut_pyscf_per_shell,
    make_lattice_opts,
)
from .linear_dependence import (
    DEFAULT_ERROR_THRESHOLD,
    DEFAULT_NEGATIVE_THRESHOLD,
    DEFAULT_WARN_THRESHOLD,
    LinearDependenceError,
    LinearDependenceOffender,
    LinearDependenceReport,
    check_linear_dependence,
    check_overlap_matrix,
    format_linear_dependence_report,
    raise_if_severe,
    scf_preflight_overlap_check,
)
from .madelung import (
    apply_madelung_correction,
    apply_madelung_correction_per_k,
    cell_electron_charge,
    cell_net_charge,
    cell_nuclear_charge,
    cubic_cell_edge,
    madelung_alpha,
    madelung_correction_scalar,
)
from .memory import (
    BatchMemoryPlan,
    InsufficientMemoryError,
    MemoryEstimate,
    available_memory_bytes,
    check_memory,
    estimate_memory,
    estimate_neb_memory,
    estimate_semiempirical_memory,
    estimate_periodic_gpw_gapw,
    estimate_periodic_xc_gradient,
    format_memory_report,
    plan_batch_memory,
)
from .molecular_optimize import (
    MolecularOptimizeResult,
    brent_minimize_1d,
    optimize_molecule,
    optimize_molecule_brent,
)
from .molecule import from_xyz
from .mpi import (
    GridSlab,
    KPointPartition,
    LatticeOutputPartition,
    MPIWorld,
    ShellPairRange,
    distribute_shell_pairs,
    distribute_tasks,
    grid_slab_partition,
    mpi_allgather,
    mpi_allreduce,
    mpi_available,
    mpi_barrier,
    mpi_bcast,
    mpi_gather,
    mpi_rank,
    mpi_reduce_sum,
    mpi_size,
    mpi_world,
)
from .neb import (
    NEBImage,
    NEBImageSCFError,
    NEBPath,
    NEBResult,
    interpolate_idpp,
    interpolate_linear,
    run_neb,
)
from .orthogonalisation import (
    OrthogonalisationInfo,
    canonical_orth,
    orthogonalise_overlap,
    pivoted_cholesky_orth,
    symmetric_orth,
)
from .pbc_bipole import (
    PBCBipoleEnergyComponents,
    PBCBipoleRHFResult,
    run_pbc_bipole_rhf,
)
from .pbc_bipole_rks import (
    PBCBipoleRKSResult,
    run_pbc_bipole_rks,
)
from .pbc_bipole_uhf import (
    PBCBipoleUHFResult,
    run_pbc_bipole_uhf,
)
from .pbc_bipole_uks import (
    PBCBipoleUKSResult,
    run_pbc_bipole_uks,
)
from .pbc_gdf import (
    PBCExxDiv,
    PBCGDFResult,
    PBCGDFUHFResult,
    PBCGDFUKSResult,
    PBCMethod,
    run_pbc_gdf_rhf,
    run_pbc_gdf_rks,
    run_pbc_gdf_uhf,
    run_pbc_gdf_uks,
)
from .perf import (
    PerfScope,
    PerfTracker,
    active_tracker,
    format_perf_report,
    perf_log,
)
from .periodic_density import (
    build_j_long_range_periodic,
    evaluate_periodic_density_on_grid,
)
from .periodic_fock_multi_k import (
    build_fock_2e_ewald3d_blocks,
    build_periodic_fock_ewald3d_k,
    ewald_3d_j_blocks,
)
from .periodic_gapw_atomic_grid import (
    AtomicRadialGrid,
    default_alpha_for_element,
    lebedev_supported_orders,
)
from .periodic_gapw_augment import (
    GapwAugmentation,
    GapwJBuilder,
    GapwScfResult,
    GapwUhfScfResult,
    GapwUksScfResult,
    run_periodic_rhf_gapw,
    run_periodic_rks_gapw,
    run_periodic_rks_gapw_multi_k,
    run_periodic_uhf_gapw,
    run_periodic_uks_gapw,
)
from .periodic_gapw_cpp_host import (
    PyGpwJKBuilder,
    run_rhf_scf_gpw_cpp,
)

# v0.10.x GAPW route -- M1 + M2 infrastructure. Experimental;
# raises GAPWExperimentalWarning until the M3 full SCF wiring
# lands. See docs/design_periodic_gapw.md.
# v0.10.x GAPW route -- M1 + M2 infrastructure. Experimental;
# raises GAPWExperimentalWarning until the M3 full SCF wiring
# lands. See docs/design_periodic_gapw.md.
from .periodic_gapw_cube import write_cube_density_periodic
from .periodic_gapw_gradient import (
    GapwGradientReport,
    compute_gradient_gapw,
    compute_gradient_gpw,
)
from .periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
    collocate_point_charges_on_grid,
    make_grid,
    nx_for_axis,
    point_charge_self_energy,
    recommend_cutoff_from_basis,
    round_up_fft_friendly,
)
from .periodic_gapw_hessian import (
    compute_hessian_gpw,
    compute_vibrational_frequencies,
)
from .periodic_gapw_j import (
    GpwEnergyBreakdown,
    GpwJBuilder,
    GpwMultiKScfResult,
    GpwScfResult,
    collocate_density_on_grid,
    compute_j_via_gpw,
    evaluate_gpw_energy,
    project_potential_to_ao,
    run_periodic_rhf_gpw,
    run_periodic_rks_gpw,
    run_periodic_rks_gpw_multi_k,
)

# ASE-driven GAPW molecular dynamics (vibeqc.periodic_gapw_md) uses ASE's MD
# engines and hard-requires ASE. Normal installs include it; this import guard
# preserves no-deps/source-tree imports until an ASE MD route is requested.
try:
    from .periodic_gapw_md import (
        run_md,
        run_npt,
        run_nve,
        run_nvt,
    )
except ImportError:  # pragma: no cover - ASE optional extra not installed
    pass
from .periodic.chi.scf import (
    AICCM2026DevBBackend,
    AICCM2026DevBDiagnostics,
    AICCM2026DevBExactExchangeAssembly,
    AICCM2026DevBExperimentalWarning,
    AICCM2026DevBFiniteTorusConvention,
    AICCM2026DevBLatticeExtension,
    AICCM2026DevBResidueTransformExecution,
    WignerSeitzRepresentative,
    cyclic_gamma_mesh,
    cyclic_lattice_extension,
    inverse_bloch_transform,
    pair_wigner_seitz_representatives,
    rhf_idempotency_error,
    run_aiccm2026dev_b_rhf,
    run_aiccm2026dev_b_rks,
    run_aiccm2026dev_b_uhf,
    run_aiccm2026dev_b_uks,
    uhf_idempotency_error,
    wigner_seitz_representatives,
)
from .periodic.chi.gradient import (
    AICCM2026DevBEwaldElectrostaticEnergyComponents,
    AICCM2026DevBEwaldElectrostaticGradientComponents,
    AICCM2026DevBGradientStatus,
    aiccm2026dev_b_gradient_status,
    compute_aiccm2026dev_b_ewald_electrostatic_energy_components,
    compute_aiccm2026dev_b_ewald_electrostatic_gradient_components,
    compute_aiccm2026dev_b_ewald_electron_nuclear_gradient,
    compute_aiccm2026dev_b_ewald_nuclear_gradient,
    compute_aiccm2026dev_b_fixed_density_kinetic_energy,
    compute_aiccm2026dev_b_fixed_density_kinetic_gradient,
    compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy,
    compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient,
    compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy,
    compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient,
    compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice,
    compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice,
    compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient,
    compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian,
    compute_aiccm2026dev_b_gradient,
    compute_aiccm2026dev_b_scf_density_lattice,
    compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components,
    compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components,
    run_aiccm2026dev_b_gradient,
)
from .periodic.chi.localization import (
    AICCM2026DevBLocalizationMethod,
    AICCM2026DevBLocalizationResult,
    AICCM2026DevBLocalizationWarning,
    AICCM2026DevBUnrestrictedLocalizationResult,
    localize_aiccm2026dev_b_occupied,
    localize_aiccm2026dev_b_occupied_blocks,
    localize_aiccm2026dev_b_unrestricted_occupied,
)
from .periodic.chi.pno import (
    AICCM2026DevBPairOrbit,
    AICCM2026DevBPAOSpace,
    AICCM2026DevBPNOSpace,
    build_pair_natural_orbitals,
    enumerate_pair_orbits,
    projected_atomic_orbitals,
)
from .periodic.chi.posthf import (
    AICCM2026DevBDLPNOCCSDResult,
    AICCM2026DevBDLPNOMP2Result,
    AICCM2026DevBLocalCorrelationSpace,
    AICCM2026DevBMP2Result,
    AICCM2026DevBUCCSDResult,
    AICCM2026DevBUMP2Result,
    run_aiccm2026dev_b_ccsd,
    run_aiccm2026dev_b_ccsd_t,
    run_aiccm2026dev_b_dlpno_ccsd,
    run_aiccm2026dev_b_dlpno_ccsd_t,
    run_aiccm2026dev_b_dlpno_mp2,
    run_aiccm2026dev_b_dlpno_uccsd,
    run_aiccm2026dev_b_dlpno_uccsd_t,
    run_aiccm2026dev_b_dlpno_ump2,
    run_aiccm2026dev_b_mp2,
    run_aiccm2026dev_b_uccsd,
    run_aiccm2026dev_b_uccsd_t,
    run_aiccm2026dev_b_ump2,
)
from .periodic.chi.properties import (
    AICCM2026DevBBondAnalysis,
    AICCM2026DevBBondOrder,
    AICCM2026DevBSCFProperties,
    aiccm2026dev_b_band_structure,
    aiccm2026dev_b_mayer_bond_orders,
    aiccm2026dev_b_one_electron_expectation,
    derive_aiccm2026dev_b_scf_properties,
)
from .periodic.chi.symmetry import (
    AICCM2026DevBSymmetryDiagnostics,
    AICCM2026DevBSymmetryMode,
    AICCM2026DevBSymmetryOperation,
    AICCM2026DevBSymmetryPlan,
    build_aiccm2026dev_b_symmetry_plan,
    gamma_matrix_symmetry_residual,
    shell_pair_orbits,
    shell_quartet_orbits,
    symmetrize_gamma_ao_matrix,
)
from .periodic.ccm.experimental import AICCM2026DevAExperimentalWarning
from .periodic_gapw_open_shell import (
    GpwRohfScfResult,
    GpwRoksMultiKScfResult,
    GpwRoksScfResult,
    GpwUhfScfResult,
    GpwUksScfResult,
    run_periodic_rohf_gpw,
    run_periodic_roks_gpw,
    run_periodic_roks_gpw_multi_k,
    run_periodic_uhf_gpw,
    run_periodic_uks_gpw,
)
from .periodic_gapw_ot import (
    OrbitalTransformation,
    run_ot_rhf_gapw,
    run_ot_rks_gapw,
)
from .periodic_gapw_phonon import (
    PhononCalculator,
    compute_dynamical_matrix_fd,
    phonon_dos,
    phonon_eigenvalues,
)
from .periodic_gapw_postscf import (
    band_path_eigenvalues,
    compute_dos_from_result,
    compute_homo_lumo,
    gaussian_dos,
)
from .periodic_gapw_qvf import qvf_density_data_periodic
from .periodic_gapw_range_sep import (
    RsGapwJBuilder,
    RsJBuilder,
    omega_for_functional,
    run_periodic_rhf_rsgapw,
    run_periodic_rks_rsgapw,
)
from .periodic_gapw_restart import (
    describe_gpw_result,
    load_gpw_result,
    save_gpw_result,
)
from .periodic_gapw_smearing import (
    GaussianMultipoleCompensator,
    SmearedNuclearCharges,
    alpha_to_sigma,
    build_electronic_compensator,
    compute_atom_local_multipoles,
    default_smearing_alpha_from_grid,
    monopole_compensator_for_charge,
    sigma_to_alpha,
    smeared_nuclear_density_on_grid,
    smeared_pairwise_overlap_energy,
    smeared_self_energy,
    smeared_v_ne_erfc_short_matrix,
    smeared_v_ne_full_gamma,
    smeared_v_ne_long_matrix,
)
from .periodic_gapw_stress import (
    compute_stress_gapw,
    compute_stress_gpw,
)
from .periodic_gdf_blocks import (
    bloch_sum_2c_eri_blocks,
    bloch_sum_3c_eri_blocks,
    gdf_block_phases,
)
from .periodic_gdf_gradient import compute_gdf_gradient
from .periodic_gradient import compute_gradient_periodic_rhf_gamma
from .periodic_gradient_fd import compute_gradient_periodic_rhf_fd
from .periodic_gradient_multi_k import (
    compute_gradient_periodic_rhf_multi_k,
    compute_gradient_periodic_rks_multi_k,
)
from .periodic_gradient_open_shell import (
    compute_gradient_periodic_uks_multi_k,
)
from .periodic_gradient_rks import compute_gradient_periodic_rks_gamma
from .lattice_convention import (
    CRYSTAL_SYSTEMS,
    PLANE_LATTICE_SYSTEMS,
    CellParameters,
    LatticeDeclarationError,
    NearestNeighbour,
    cell_parameters,
    check_crystal_system,
    check_space_group,
    lattice_from_vectors,
    lattice_vectors,
    nearest_neighbour_distance,
)
from .periodic_grid import (
    build_periodic_becke_grid,
    extended_partition_atoms,
)
from .periodic_jk_direct import jk_via_direct
from .periodic_jk_method import (
    PeriodicJKMethod,
    describe_jk_method,
    is_orthorhombic,
    pick_jk_method,
    validate_jk_method,
)
from .periodic_k_gdf import (
    PeriodicKRHFGDFResult,
    PeriodicKRKSGDFResult,
    PeriodicKUHFGDFResult,
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
    run_kuks_periodic_gdf,
)
from .periodic_rohf_gdf import (
    PeriodicKROHFGDFResult,
    run_krohf_periodic_gdf,
)
from .periodic_ks_dispatch import (
    run_rks_periodic_gamma_scf,
    run_rks_periodic_scf,
)
from .periodic_orbitals import (
    PrimitiveCellGrid,
    evaluate_bloch_orbital,
    make_primitive_cell_grid,
    write_cube_mo_periodic,
    write_xsf_density,
    write_xsf_mo,
)
from .periodic_rhf_dispatch import (
    run_rhf_periodic_gamma_scf,
    run_rhf_periodic_scf,
)
from .periodic_rhf_ewald import (
    PeriodicRHFEwaldResult,
    run_rhf_periodic_gamma_ewald2d,
    run_rhf_periodic_gamma_ewald3d,
)
from .periodic_rhf_gdf import (
    PeriodicRHFGDFResult,
    run_rhf_periodic_gamma_gdf,
    run_rks_periodic_gamma_gdf,
)
from .periodic_rhf_multi_k_ewald import (
    PeriodicRHFMultiKEwaldResult,
    run_rhf_periodic_multi_k_ewald3d,
)
from .periodic_rks_ewald import (
    PeriodicRKSEwaldResult,
    run_rks_periodic_gamma_ewald2d,
    run_rks_periodic_gamma_ewald3d,
)
from .periodic_rks_multi_k_ewald import (
    PeriodicRKSMultiKEwaldResult,
    run_rks_periodic_multi_k_ewald3d,
)
from .periodic_rohf_ewald import (
    PeriodicROHFEwaldResult,
    run_rohf_periodic_gamma_ewald3d,
)
from .periodic_rohf_multi_k_ewald import (
    PeriodicROHFMultiKEwaldResult,
    run_rohf_periodic_multi_k_ewald3d,
)
from .periodic_roks_multi_k_ewald import (
    PeriodicROKSMultiKEwaldResult,
    run_roks_periodic_multi_k_ewald3d,
)
from .periodic_runner import run_periodic_job
from .periodic_symmetrize import (
    SymmetriseReport,
    SymmetriseResult,
    detect_spacegroup,
    symmetrise,
)
from .periodic_uhf_ewald import (
    PeriodicUHFEwaldResult,
    run_uhf_periodic_gamma_ewald3d,
)
from .periodic_uhf_multi_k_ewald import (
    PeriodicUHFMultiKEwaldResult,
    run_uhf_periodic_multi_k_ewald3d,
)
from .periodic_uks_ewald import (
    PeriodicUKSEwaldResult,
    run_uks_periodic_gamma_ewald2d,
    run_uks_periodic_gamma_ewald3d,
)
from .periodic_uks_multi_k_ewald import (
    PeriodicUKSMultiKEwaldResult,
    run_uks_periodic_multi_k_ewald3d,
)
from .poscar import read_poscar, write_poscar
from .progress import ProgressLogger, resolve_progress
from .properties import (
    DipoleMoment,
    HirshfeldResult,
    NaturalOrbitals,
    center_of_mass,
    dipole_moment,
    hirshfeld_charges,
    idempotency_deviation,
    loewdin_charges,
    mayer_bond_orders,
    mulliken_charges,
    natural_orbitals,
    prominent_bonds,
)
from .iao import (
    IAOReference,
    IBOAnalysis,
    analyse_ibo,
    build_iaos,
    iao_charges,
    iao_reference,
    iao_unsupported_reason,
)
from .qtaim import (
    BondPath,
    CriticalPoint,
    QTAIMResult,
    qtaim_analysis,
    qtaim_result_to_qvf,
)
from .rohf import (
    ROHFOptions,
    ROHFResult,
    compute_rohf_gradient,
    require_nonnegative_max_iter,
    run_rohf,
)
from .qvf_job import (
    load_job_container,
    run_container,
    write_pending_qvf,
)
from .roks import ROKSOptions, ROKSResult, run_roks
from .runner import run_job
from .scan import (
    ScanResult,
    ScanResult2D,
    relaxed_scan,
    relaxed_scan_2d,
)
from .scf_log import format_basis_summary, format_scf_trace, log_scf_trace
from .settings_dump import format_settings, print_settings
from .smearing import (
    EV_PER_HARTREE,
    HARTREE_PER_RYDBERG,
    KB_HARTREE_PER_K,
    SMEARING_PRESETS,
    SmearingOptions,
    SmearingResolution,
    SmearingResult,
    apply_smearing,
    aufbau_occupations_per_k,
    closed_shell_periodic_occupations,
    electronvolt_to_hartree_temperature,
    fermi_dirac_occupations_per_k,
    mermin_occupations_per_k,
    guess_smearing_temperature,
    hartree_to_kelvin_temperature,
    kelvin_to_hartree_temperature,
    resolve_smearing_temperature,
    rydberg_to_hartree_temperature,
    temperature_in_hartree,
)
from .solvation import (
    SOLVENT_PRESETS,
    CavityTessellation,
    CPCMResult,
    SolventModel,
    SolventResult,
    build_cavity,
    cpcm_gradient,
    cpcm_gradient_fd,
    resolve_solvent,
    run_cpcm_scf,
)
from .solvers import (
    CASPT2Options,
    CC3Iteration,
    CC3Options,
    CC3Result,
    CCSDTIteration,
    CCSDTOptions,
    CCSDTResult,
    CISDOptions,
    CISDResult,
    DMRGOptions,
    DMRGSolver,
    Hamiltonian,
    SelectedCIOptions,
    SelectedCISolver,
    SolverOptions,
    SolverResult,
    TranscorrelatedOptions,
    V2RDMOptions,
    V2RDMSolver,
    build_hamiltonian_ao,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
    canonical_orthogonalize,
    cc3,
    ccsdt,
    cisd,
    get_hf_orbital_provider,
    solve_dmrg,
    solve_selected_ci,
    solve_v2rdm,
)
from .structured_log import (
    StructuredLog,
    active_structured_log,
    run_fingerprint,
    structured_log,
)
from .structured_log import (
    emit as emit_structured,
)
from .symmetry_ao import (
    AtomPermutation,
    atom_permutation_under_op,
    build_ao_permutation_matrix,
)
from .symmetry_core import (
    euler_angles_from_rotation,
    real_spherical_to_complex_unitary,
    wigner_d_complex,
    wigner_d_real,
    wigner_small_d,
)
from .symmetry_fock_reduced import (
    compute_jk_gamma_reduced,
)
from .symmetry_integrals import (
    OrbitReducedLatticeMatrix,
    SymmetrisedOrbitIntegralResult,
    compute_kinetic_lattice_symmetrised_with_orbits,
    compute_kinetic_lattice_with_orbits,
    compute_nuclear_lattice_symmetrised_with_orbits,
    compute_nuclear_lattice_with_orbits,
    compute_overlap_lattice_symmetrised_with_orbits,
    compute_overlap_lattice_with_orbits,
    symmorphic_operations,
    verify_lattice_matrix_set_symmetry,
)
from .symmetry_integrals_reduced import (
    compression_summary,
    compute_kinetic_lattice_reduced,
    compute_nuclear_lattice_reduced,
    compute_overlap_lattice_reduced,
)
from .symmetry_lattice import (
    LatticeOrbit,
    LatticeOrbits,
    compress_lattice_matrix_set,
    identify_lattice_orbits,
    lattice_to_cartesian_rotation,
    reconstruct_lattice_matrix_set,
)
from .symmetry_lattice_c import (
    AtomPairOrbit,
    AtomPairOrbits,
    compress_lattice_matrix_set_c,
    identify_atom_pair_orbits,
    reconstruct_lattice_matrix_set_c,
)
from .symmetry_salc import (
    PointGroupCharacters,
    character_table,
    symmetry_project_matrix,
)
from .symmetry_scf import (
    build_ao_permutation_cache,
    symmetrize_density,
    symmetrize_fock,
    symmetrize_forces,
    symmetrize_matrix,
)
from .system_info import system_info, write_system_manifest
from .tddft import (
    NTOResult,
    TDDFTResult,
    TDDFTState,
    compute_nto,
    eri_ao_to_mo,
    oscillator_strength,
    run_tddft_casida,
    run_tddft_casida_uhf,
    run_tddft_tda,
    run_tddft_tda_periodic,
    run_tddft_tda_uhf,
)
from .thermo import (
    ThermoOptions,
    ThermoResult,
    compute_thermochemistry,
)
from .xsf import write_bxsf, write_xsf_structure, write_xsf_volume

# Attach the XYZ parser as a Molecule classmethod so `Molecule.from_xyz(...)`
# works from Python. The native Molecule class was defined in C++ via pybind11
# and can be augmented at runtime.
Molecule.from_xyz = staticmethod(from_xyz)  # type: ignore[attr-defined]
Crystal.from_cif = staticmethod(read_cif)  # type: ignore[attr-defined]

# ``__version__`` is sourced from the installed package metadata (see
# ``banner.VIBEQC_VERSION``). Hard-coding here would drift from
# pyproject.toml on release bumps.
__version__ = VIBEQC_VERSION


def define_functional(
    name: str,
    components: list,
    *,
    hf_exchange_fraction: float = 0.0,
) -> None:
    """Register a user-defined XC functional alias at runtime.

    Once registered, ``Functional(name)`` resolves the alias exactly
    like a built-in functional.

    Parameters
    ----------
    name
        Name for the functional (case-insensitive). Registering the same
        dynamic alias again replaces its recipe. An external-functional name
        cannot be replaced.
    components
        List of ``(libxc_name, weight)`` pairs.  ``libxc_name`` is a
        libxc functional name such as ``"GGA_X_PW91"``; the weight
        scales the component's contribution to the total XC energy.
    hf_exchange_fraction
        Global HF-exchange fraction for hand-mixed hybrids.  Set to 0.20
        for a PW1PW-style hybrid, 0.0 for pure DFT.

    Examples
    --------
    Define a PW1PW clone:

    >>> vq.define_functional(
    ...     "my-pw1pw",
    ...     [("GGA_X_PW91", 0.80), ("GGA_C_PW91", 1.00)],
    ...     hf_exchange_fraction=0.20,
    ... )
    >>> f = Functional("my-pw1pw")
    >>> f.hf_exchange_fraction
    0.2

    Notes
    -----
    Thread-safe. Define or replace an alias before any SCF calculation that
    references it; existing ``Functional`` objects keep their original recipe.
    """
    _define_functional_core(name, components, hf_exchange_fraction)


def define_external_functional(
    name: str,
    callback: object,
    *,
    hf_exchange_fraction: float = 0.0,
    capability_version: int = 1,
    required_grid_profile: str = "",
) -> None:
    """Register a differentiable full-grid XC provider.

    The callback receives one atom-major dictionary containing ``grid_profile``,
    grid coordinates and weights, both spin densities, Cartesian density
    gradients, and kinetic-energy densities. It returns the integrated scalar
    XC energy and its full-grid adjoints for those six density feature arrays.

    Capability protocol version 1 supports ``required_grid_profile``. Use an
    empty string to accept any grid, or ``"pyscf-level3"`` to require that
    complete atomic-grid protocol. A mismatch fails before callback execution
    in every molecular and periodic AO-grid route.
    """

    # Import citation bookkeeping before mutating the native registry. If the
    # Python output layer is unavailable, registration must fail without
    # leaving a provider that the caller was told did not register.
    from .output.citations.registry import _register_runtime_external_functional

    _define_external_functional_core(
        name,
        callback,
        hf_exchange_fraction,
        capability_version=capability_version,
        required_grid_profile=required_grid_profile,
    )
    # Registration succeeded, so citation assembly must not attribute this
    # application-supplied backend to libxc.  A custom name still produces a
    # missing functional-route warning until the application supplies its own
    # defining reference, which is preferable to inventing a citation.
    _register_runtime_external_functional(name)


__all__ = [
    "__version__",
    "RELEASE_CODENAMES",
    "VIBEQC_VERSION",
    "banner",
    "build_info",
    "codename_for_version",
    "enforce_runtime_pin_from_env",
    "library_versions",
    "print_banner",
    "InsufficientMemoryError",
    "MemoryEstimate",
    "BatchMemoryPlan",
    "available_memory_bytes",
    "check_memory",
    "estimate_memory",
    "estimate_neb_memory",
    "estimate_semiempirical_memory",
    "estimate_periodic_gpw_gapw",
    "estimate_periodic_xc_gradient",
    "format_memory_report",
    "plan_batch_memory",
    "Atom",
    "BandDiag",
    "BondPath",
    "BandStructure",
    "CriticalPoint",
    "CubeGrid",
    "make_uniform_grid",
    "write_cube_density",
    "write_cube_mo",
    "write_cube_mos",
    "write_bxsf",
    "write_xsf_structure",
    "write_xsf_volume",
    "DensityOfStates",
    "KPath",
    "ProjectedDensityOfStates",
    "ao_groups_per_atom",
    "ao_groups_per_atom_l",
    "ao_pairs_per_atom_pair",
    "band_structure",
    "band_structure_hcore",
    "density_of_states",
    "density_of_states_hcore",
    "density_of_states_projected",
    "density_of_states_projected_hcore",
    "kpath_from_segments",
    "BasisSet",
    "BlochKMesh",
    "CosxVariant",
    "CoulombMethod",
    "Crystal",
    "ADIIS",
    "D3BJParams",
    "DIIS",
    "DIISDepthPolicy",
    "DipoleIntegrals",
    "DipoleMoment",
    "DispersionResult",
    "EDIIS",
    "KDIIS",
    "EwaldOptions",
    "InitialGuess",
    "Fragment",
    "PeriodicFragment",
    "IrreducibleKMesh",
    "JKBuilder",
    "JKMatrices",
    "LatticeCell",
    "LatticeMatrixSet",
    "LatticeSumOptions",
    "MP2Options",
    "Molecule",
    "PairJKContribution",
    "PeriodicKSOptions",
    "PeriodicKSResult",
    "PeriodicRHFOptions",
    "PeriodicRHFResult",
    "PeriodicSCFOptions",
    "PeriodicSystem",
    "PeriodicXCDensityDomain",
    "CellParameters",
    "NearestNeighbour",
    "CRYSTAL_SYSTEMS",
    "LatticeDeclarationError",
    "PLANE_LATTICE_SYSTEMS",
    "cell_parameters",
    "check_crystal_system",
    "check_space_group",
    "lattice_from_vectors",
    "lattice_vectors",
    "nearest_neighbour_distance",
    "BULK_LATTICE_CONSTANTS",
    "SlabInfo",
    "build_molecule",
    "place_adsorbate",
    "slab",
    "slab_2d",
    "synthesize_slab_a3",
    "PeriodicDispersionResult",
    "compute_d3bj_periodic",
    "PeriodicXCContribution",
    "PeriodicUKSXCContribution",
    "RHFOptions",
    "RHFResult",
    "MP2Result",
    "RKSOptions",
    "RKSResult",
    "SCFAccelerator",
    "SCFIteration",
    "SCFMode",
    "SCFRestartOptions",
    "OpenTrustRegionOptions",
    "has_opentrustregion",
    "NewtonOptions",
    "SOSCFOptions",
    "TRAHOptions",
    "ShellInfo",
    "SpaceGroup",
    "ROHFOptions",
    "ROHFResult",
    "ROKSOptions",
    "ROKSResult",
    "SpinlockMode",
    "SymmetriseReport",
    "SymmetriseResult",
    "SymmetryOp",
    "UHFOptions",
    "UHFResult",
    "UKSOptions",
    "UKSResult",
    "UMP2Options",
    "UMP2Result",
    "CCSDIteration",
    "CCSDOptions",
    "CCSDResult",
    "CC3Iteration",
    "CC3Options",
    "CC3Result",
    "CCSDTIteration",
    "CCSDTOptions",
    "CCSDTResult",
    "CISDOptions",
    "CISDResult",
    "BruecknerCCDResult",
    "BruecknerIteration",
    "PUBLISHED_FROZEN_CORE_CONVENTION",
    "chemical_core_orbital_count",
    "published_frozen_core_orbital_count",
    "resolve_frozen_core",
    "resolve_frozen_core_count",
    "cc3",
    "ccsdt",
    "run_bccd",
    "run_ccsd",
    "run_uccsd",
    "run_rohf_ccsd",
    "run_rohf_mp2",
    "Functional",
    "Grid",
    "AtomicGridProfile",
    "GridOptions",
    "UHFXCKernelBuilder",
    "XCKernelBuilder",
    "CPHFConvergenceError",
    "CPHFOptions",
    "HessianFDOptions",
    "HessianResult",
    "ThermoOptions",
    "ThermoResult",
    "XCKind",
    "analyze_symmetry",
    "attach_symmetry",
    "bloch_sum",
    "build_grid",
    "build_fock_2e_real_space",
    "build_jk_2e_real_space",
    "build_jk_2e_real_space_explicit",
    "build_jk_gamma_molecular_limit",
    "build_jk_gamma_molecular_limit_explicit",
    "build_jk_pair_contributions",
    "build_xc_periodic",
    "build_xc_periodic_uks",
    "center_of_mass",
    "cisd",
    "compute_d3bj",
    "compute_2c_eri",
    "compute_2c_eri_gradient_contribution",
    "compute_2c_eri_gradient_weighted",
    "compute_3c_eri",
    "compute_3c_eri_gradient_contribution",
    "compute_3c_eri_gradient_weighted",
    "compute_coop_cohp",
    "COOPCOHPResult",
    "compute_rohf_gradient",
    "build_cosx_q",
    "build_cosx_schwarz",
    "compute_cosx_k",
    "compute_cosx_k_gradient_contribution",
    "compute_robust_cosx_k",
    "cosx_basis_cardinality_from_name",
    "cosx_grid_options_for_level",
    "cosx_grid_stages_for_level",
    "resolve_cosx_grid_level",
    "resolve_cosx_variant",
    "make_cosx_jk_builder",
    "make_df_jk_builder",
    "make_direct_jk_builder",
    "make_four_index_jk_builder",
    "make_periodic_gamma_jk_builder",
    "compute_df_j_gradient",
    "compute_df_jk_gradient",
    "compute_df_k_gradient",
    "compute_dipole",
    "auto_ecp_centers",
    "BasisRecord",
    "EcpRequirement",
    "basis_registry_lookup",
    "basis_replaces_core",
    "canonical_basis_name",
    "default_aux_basis",
    "ecp_requirements",
    "element_coverage",
    "basis_sidecar_replaces_core",
    "ecp_centers_from_dict",
    "compute_ecp_gradient_contribution",
    "compute_ecp_gradient_contribution_from_primitives",
    "compute_ecp_matrix",
    "compute_ecp_matrix_from_primitives",
    "ECPCenter",
    "ECPPrimitiveBlock",
    "ecp_core_electrons",
    "EcpHeader",
    "InlineECPRecord",
    "attach_inline_ecp_options_from_basis_sidecar",
    "inline_ecp_data_for",
    "is_ecp_paired_basis",
    "library_for",
    "molecular_options_replace_core",
    "parse_inline_ecp_sidecar",
    "parse_sidecar_path",
    "sidecar_path_for",
    "validate_ecp_required",
    "compute_eri",
    "compute_external_charge_density_gradient",
    "ExternalChargeGradient",
    "DensityFitting",
    "DensityFittingInfo",
    "AuxiliaryBasisCoverageError",
    "aux_basis_coverage_gaps",
    "check_aux_basis_coverage",
    "default_aux_basis_for",
    "make_checked_aux_basis",
    "d3_coordination_numbers",
    "d3_r2r4",
    "d3_rcov",
    "d3bj_params_for",
    "dftd3_available",
    "HubbardSite",
    "ao_group_indices",
    "compute_occupation_matrices",
    "compute_dudarev_energy",
    "compute_2c_eri_lattice",
    "compute_2c_eri_lattice_blocks",
    "compute_2c_eri_lattice_sr",
    "compute_3c_eri_lattice",
    "compute_3c_eri_lattice_blocks",
    "compute_3c_eri_lattice_sr",
    "compute_kinetic_lattice",
    "compute_kinetic_lattice_explicit",
    "compute_nuclear_erfc_lattice",
    "compute_nuclear_lattice",
    "compute_nuclear_lattice_ewald",
    "compute_nuclear_lattice_explicit",
    "compute_nuclear_with_charges",
    "compute_overlap_lattice",
    "compute_overlap_lattice_explicit",
    "compute_gradient",
    "compute_gradient_rks",
    "functional_gradient_terms_missing",
    "compute_gradient_uhf",
    "compute_gradient_uks",
    "compute_hessian_fd",
    "compute_hessian_rhf_analytic",
    "compute_hessian_rks_analytic",
    "KPoints",
    "KPointConvergence",
    "as_bloch_kmesh",
    "compute_hessian_uhf_analytic",
    "compute_hessian_uks_analytic",
    "compute_hessian_gpw",
    "compute_vibrational_frequencies",
    "compute_gradient_periodic_rhf_fd",
    "compute_gradient_periodic_rhf_gamma",
    "compute_gdf_gradient",
    "compute_gradient_periodic_rhf_multi_k",
    "compute_gradient_periodic_rks_gamma",
    "compute_gradient_periodic_rks_multi_k",
    "compute_gradient_periodic_uks_multi_k",
    "compute_thermochemistry",
    "cphf_solve_rhf",
    "dipole_polarizability_rhf",
    "ir_intensities",
    "compute_kinetic",
    "evaluate_ao",
    "evaluate_ao_with_gradient",
    "evaluate_ao_with_hessian",
    "evaluate_bloch_ao",
    "evaluate_bloch_orbital",
    "make_batched_polarised_xc_kernel_builder",
    "make_polarised_gga_xc_kernel_builder",
    "make_polarised_lda_xc_kernel_builder",
    "make_polarised_xc_kernel_builder",
    "make_unpolarised_xc_kernel_builder",
    "get_num_threads",
    "compute_nuclear",
    "compute_overlap",
    "compute_overlap_two_basis",
    "analyse_ibo",
    "build_iaos",
    "iao_charges",
    "iao_reference",
    "iao_unsupported_reason",
    "IAOReference",
    "IBOAnalysis",
    "sad_density",
    "diagonalize_bloch",
    "dipole_moment",
    "hirshfeld_charges",
    "HirshfeldResult",
    "idempotency_deviation",
    "loewdin_charges",
    "mayer_bond_orders",
    "mulliken_charges",
    "natural_orbitals",
    "NaturalOrbitals",
    "qtaim_analysis",
    "qtaim_result_to_qvf",
    "QTAIMResult",
    "direct_lattice_cells",
    "pair_complete_lattice_cells",
    "ewald_2d_point_charge_energy",
    "ewald_2d_point_charge_energy_with_background",
    "ewald_2d_point_charge_gradient_with_background",
    "ewald_2d_point_charge_potential",
    "ewald_2d_point_charge_potential_with_background",
    "ewald_nuclear_potential",
    "ewald_nuclear_repulsion",
    "ewald_point_charge_energy",
    "ewald_point_charge_potential",
    "auto_grid",
    "auto_ewald_alpha",
    "build_j_ewald_3d",
    "compute_j_ewald_3d_ft_gamma",
    "compute_j_ewald_3d_ft_lattice",
    "build_j_long_range",
    "makov_payne_coefficient_cubic",
    "PeriodicRHFEwaldResult",
    "run_rhf_periodic_gamma_ewald2d",
    "run_rhf_periodic_gamma_ewald3d",
    "PeriodicRHFGDFResult",
    "run_rhf_periodic_gamma_gdf",
    "run_rks_periodic_gamma_gdf",
    "PBCExxDiv",
    "PBCGDFResult",
    "PBCGDFUHFResult",
    "PBCGDFUKSResult",
    "PBCMethod",
    "run_pbc_gdf_rhf",
    "run_pbc_gdf_rks",
    "run_pbc_gdf_uhf",
    "run_pbc_gdf_uks",
    "PBCBipoleEnergyComponents",
    "PBCBipoleRHFResult",
    "run_pbc_bipole_rhf",
    "PBCBipoleUHFResult",
    "run_pbc_bipole_uhf",
    "PBCBipoleRKSResult",
    "run_pbc_bipole_rks",
    "PBCBipoleUKSResult",
    "run_pbc_bipole_uks",
    "RcutStrategy",
    "estimate_rcut_pyscf",
    "estimate_rcut_pyscf_per_shell",
    "make_lattice_opts",
    "PeriodicKRHFGDFResult",
    "PeriodicKRKSGDFResult",
    "PeriodicKUHFGDFResult",
    "PeriodicKROHFGDFResult",
    "run_krhf_periodic_gdf",
    "run_krks_periodic_gdf",
    "run_krohf_periodic_gdf",
    "run_kuhf_periodic_gdf",
    "run_kuks_periodic_gdf",
    "AICCM2026DevAExperimentalWarning",
    "AICCM2026DevBDiagnostics",
    "AICCM2026DevBBackend",
    "AICCM2026DevBExactExchangeAssembly",
    "AICCM2026DevBExperimentalWarning",
    "AICCM2026DevBFiniteTorusConvention",
    "AICCM2026DevBEwaldElectrostaticEnergyComponents",
    "AICCM2026DevBEwaldElectrostaticGradientComponents",
    "AICCM2026DevBGradientStatus",
    "aiccm2026dev_b_gradient_status",
    "compute_aiccm2026dev_b_ewald_electrostatic_energy_components",
    "compute_aiccm2026dev_b_ewald_electrostatic_gradient_components",
    "compute_aiccm2026dev_b_ewald_electron_nuclear_gradient",
    "compute_aiccm2026dev_b_ewald_nuclear_gradient",
    "compute_aiccm2026dev_b_fixed_density_kinetic_energy",
    "compute_aiccm2026dev_b_fixed_density_kinetic_gradient",
    "compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy",
    "compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient",
    "compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy",
    "compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient",
    "compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice",
    "compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice",
    "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient",
    "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian",
    "compute_aiccm2026dev_b_gradient",
    "compute_aiccm2026dev_b_scf_density_lattice",
    "compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components",
    "compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components",
    "run_aiccm2026dev_b_gradient",
    "AICCM2026DevBLocalizationMethod",
    "AICCM2026DevBLocalizationResult",
    "AICCM2026DevBUnrestrictedLocalizationResult",
    "AICCM2026DevBLocalizationWarning",
    "AICCM2026DevBLatticeExtension",
    "AICCM2026DevBResidueTransformExecution",
    "AICCM2026DevBLocalCorrelationSpace",
    "AICCM2026DevBPAOSpace",
    "AICCM2026DevBPNOSpace",
    "AICCM2026DevBPairOrbit",
    "AICCM2026DevBSymmetryDiagnostics",
    "AICCM2026DevBSymmetryMode",
    "AICCM2026DevBSymmetryOperation",
    "AICCM2026DevBSymmetryPlan",
    "AICCM2026DevBDLPNOCCSDResult",
    "AICCM2026DevBDLPNOMP2Result",
    "AICCM2026DevBMP2Result",
    "AICCM2026DevBUCCSDResult",
    "AICCM2026DevBUMP2Result",
    "AICCM2026DevBBondAnalysis",
    "AICCM2026DevBBondOrder",
    "AICCM2026DevBSCFProperties",
    "WignerSeitzRepresentative",
    "cyclic_gamma_mesh",
    "cyclic_lattice_extension",
    "build_aiccm2026dev_b_symmetry_plan",
    "build_pair_natural_orbitals",
    "enumerate_pair_orbits",
    "gamma_matrix_symmetry_residual",
    "inverse_bloch_transform",
    "localize_aiccm2026dev_b_occupied",
    "localize_aiccm2026dev_b_occupied_blocks",
    "localize_aiccm2026dev_b_unrestricted_occupied",
    "pair_wigner_seitz_representatives",
    "projected_atomic_orbitals",
    "rhf_idempotency_error",
    "uhf_idempotency_error",
    "run_aiccm2026dev_b_rhf",
    "run_aiccm2026dev_b_rks",
    "run_aiccm2026dev_b_uhf",
    "run_aiccm2026dev_b_uks",
    "run_aiccm2026dev_b_ccsd",
    "run_aiccm2026dev_b_ccsd_t",
    "run_aiccm2026dev_b_dlpno_ccsd",
    "run_aiccm2026dev_b_dlpno_ccsd_t",
    "run_aiccm2026dev_b_dlpno_mp2",
    "run_aiccm2026dev_b_dlpno_uccsd",
    "run_aiccm2026dev_b_dlpno_uccsd_t",
    "run_aiccm2026dev_b_dlpno_ump2",
    "run_aiccm2026dev_b_mp2",
    "run_aiccm2026dev_b_uccsd",
    "run_aiccm2026dev_b_uccsd_t",
    "run_aiccm2026dev_b_ump2",
    "aiccm2026dev_b_band_structure",
    "aiccm2026dev_b_mayer_bond_orders",
    "aiccm2026dev_b_one_electron_expectation",
    "derive_aiccm2026dev_b_scf_properties",
    "shell_pair_orbits",
    "shell_quartet_orbits",
    "symmetrize_gamma_ao_matrix",
    "wigner_seitz_representatives",
    "bloch_sum_2c_eri_blocks",
    "bloch_sum_3c_eri_blocks",
    "gdf_block_phases",
    "run_periodic_job",
    "PeriodicJKMethod",
    "pick_jk_method",
    "validate_jk_method",
    "describe_jk_method",
    "is_orthorhombic",
    "jk_via_direct",
    "PeriodicROHFEwaldResult",
    "run_rohf_periodic_gamma_ewald3d",
    "PeriodicROHFMultiKEwaldResult",
    "run_rohf_periodic_multi_k_ewald3d",
    "PeriodicROKSMultiKEwaldResult",
    "run_roks_periodic_multi_k_ewald3d",
    "PeriodicUHFEwaldResult",
    "run_uhf_periodic_gamma_ewald3d",
    "PeriodicUHFMultiKEwaldResult",
    "run_uhf_periodic_multi_k_ewald3d",
    "PeriodicRKSEwaldResult",
    "run_rks_periodic_gamma_ewald2d",
    "run_rks_periodic_gamma_ewald3d",
    "PeriodicRKSMultiKEwaldResult",
    "run_rks_periodic_multi_k_ewald3d",
    "PeriodicUKSEwaldResult",
    "run_uks_periodic_gamma_ewald2d",
    "run_uks_periodic_gamma_ewald3d",
    "PeriodicUKSMultiKEwaldResult",
    "run_uks_periodic_multi_k_ewald3d",
    "build_j_long_range_periodic",
    "evaluate_periodic_density_on_grid",
    "PrimitiveCellGrid",
    "make_primitive_cell_grid",
    "write_cube_mo_periodic",
    "write_xsf_density",
    "write_xsf_mo",
    "build_fock_2e_ewald3d_blocks",
    "build_periodic_fock_ewald3d_k",
    "ewald_3d_j_blocks",
    "PeriodicRHFMultiKEwaldResult",
    "run_rhf_periodic_multi_k_ewald3d",
    "run_rhf_periodic_gamma_scf",
    "run_rhf_periodic_scf",
    "run_rks_periodic_gamma_scf",
    "run_rks_periodic_scf",
    "build_periodic_becke_grid",
    "extended_partition_atoms",
    "OrbitReducedLatticeMatrix",
    "SymmetrisedOrbitIntegralResult",
    "compress_lattice_matrix_set_c",
    "compression_summary",
    "compute_kinetic_lattice_reduced",
    "compute_kinetic_lattice_symmetrised_with_orbits",
    "compute_kinetic_lattice_with_orbits",
    "compute_jk_gamma_reduced",
    "compute_nuclear_lattice_reduced",
    "compute_nuclear_lattice_symmetrised_with_orbits",
    "compute_nuclear_lattice_with_orbits",
    "compute_overlap_lattice_reduced",
    "compute_overlap_lattice_symmetrised_with_orbits",
    "compute_overlap_lattice_with_orbits",
    "identify_atom_pair_orbits",
    "reconstruct_lattice_matrix_set_c",
    "symmorphic_operations",
    "verify_lattice_matrix_set_symmetry",
    "apply_madelung_correction",
    "build_ao_permutation_cache",
    "symmetrize_density",
    "symmetrize_fock",
    "symmetrize_forces",
    "symmetrize_matrix",
    "PointGroupCharacters",
    "character_table",
    "symmetry_project_matrix",
    "apply_madelung_correction_per_k",
    "cell_electron_charge",
    "cell_net_charge",
    "cell_nuclear_charge",
    "cubic_cell_edge",
    "madelung_alpha",
    "madelung_correction_scalar",
    "DEFAULT_ERROR_THRESHOLD",
    "DEFAULT_NEGATIVE_THRESHOLD",
    "DEFAULT_WARN_THRESHOLD",
    "LinearDependenceError",
    "LinearDependenceOffender",
    "LinearDependenceReport",
    "check_linear_dependence",
    "check_overlap_matrix",
    "format_linear_dependence_report",
    "raise_if_severe",
    "scf_preflight_overlap_check",
    "BasisFilterReport",
    "DroppedPrimitive",
    "clear_filtered_basis_cache",
    "filter_basis_by_exponent",
    "format_basis_filter_report",
    "make_basis",
    "DisambiguationReport",
    "EIGSReport",
    "TruncationOptimizationReport",
    "disambiguate_critical_overlap",
    "eigs_preflight",
    "format_disambiguation_report",
    "format_eigs_report",
    "format_truncation_optimization_report",
    "optimize_truncation",
    "OrthogonalisationInfo",
    "canonical_orth",
    "orthogonalise_overlap",
    "pivoted_cholesky_orth",
    "symmetric_orth",
    "euler_angles_from_rotation",
    "real_spherical_to_complex_unitary",
    "wigner_d_complex",
    "wigner_d_real",
    "wigner_small_d",
    "AtomPermutation",
    "atom_permutation_under_op",
    "build_ao_permutation_matrix",
    "LatticeOrbit",
    "LatticeOrbits",
    "compress_lattice_matrix_set",
    "identify_lattice_orbits",
    "lattice_to_cartesian_rotation",
    "reconstruct_lattice_matrix_set",
    "AtomPairOrbit",
    "AtomPairOrbits",
    "compress_lattice_matrix_set_c",
    "identify_atom_pair_orbits",
    "reconstruct_lattice_matrix_set_c",
    "KB_HARTREE_PER_K",
    "EV_PER_HARTREE",
    "HARTREE_PER_RYDBERG",
    "SMEARING_PRESETS",
    "SmearingOptions",
    "SmearingResolution",
    "SmearingResult",
    "apply_smearing",
    "aufbau_occupations_per_k",
    "closed_shell_periodic_occupations",
    "electronvolt_to_hartree_temperature",
    "fermi_dirac_occupations_per_k",
    "mermin_occupations_per_k",
    "guess_smearing_temperature",
    "hartree_to_kelvin_temperature",
    "kelvin_to_hartree_temperature",
    "resolve_smearing_temperature",
    "rydberg_to_hartree_temperature",
    "temperature_in_hartree",
    "hartree_energy_on_grid",
    "solve_poisson_coulomb",
    "solve_poisson_erf_screened",
    "format_basis_summary",
    "format_scf_trace",
    "format_settings",
    "from_xyz",
    "read_cif",
    "hello",
    "irreducible_kpoints",
    "libecpint_version",
    "libint_version",
    "LevelShiftSchedule",
    "level_shift_at_iter",
    "libxc_version",
    "log_scf_trace",
    "monkhorst_pack",
    "nuclear_repulsion_per_cell",
    "nuclear_repulsion_slab_ewald_2d_gradient",
    "PerfScope",
    "PerfTracker",
    "active_tracker",
    "format_perf_report",
    "perf_log",
    "print_settings",
    "ProgressLogger",
    "StructuredLog",
    "active_structured_log",
    "active_crash_dump_stem",
    "classify_failure",
    "crash_dump_context",
    "dump_on_failure",
    "detect_spacegroup",
    "emit_structured",
    "load_dump",
    "run_fingerprint",
    "structured_log",
    "read_poscar",
    "real_space_density_from_kpoints",
    "resolve_progress",
    "run_job",
    "load_job_container",
    "run_container",
    "write_pending_qvf",
    "D4Result",
    "DoubleHybridResult",
    "D4_WEIGHTING_FACTOR",
    "EEQOptions",
    "EEQResult",
    "ReferenceSystem",
    "all_reference_elements",
    "all_reference_systems",
    "build_reference_molecule",
    "casimir_polder_c6",
    "cn_gaussian_weights",
    "compute_d4",
    "coupled_polarizability_imag_freq",
    "dftd4_available",
    "eeq_charges",
    "eeq_coordination_numbers",
    "imaginary_frequency_grid",
    "london_c6_single_pole",
    "molecular_c6",
    "reference_systems_for",
    "rhf_result_from_rks",
    "uncoupled_polarizability_imag_freq",
    "run_b2plyp",
    "run_double_hybrid",
    "run_dsd_pbep86",
    "run_revdsd_pbep86",
    "run_pwpb95",
    "compute_vv10",
    "compute_b97m_semilocal_exc",
    "compute_b97m_semilocal_vxc",
    "run_wb97x_d",
    "WB97XDResult",
    "compute_chg_dispersion",
    "chg_max_supported_Z",
    "run_mp2",
    "run_rhf_scf_with_jk",
    "run_rks_scf_with_jk",
    "run_uhf_scf_with_jk",
    "run_uks_scf_with_jk",
    "run_rhf",
    "run_rhf_periodic",
    "run_rhf_periodic_gamma",
    "run_rks",
    # v0.9.0 -- implicit solvation (CPCM/COSMO).
    "solvation",
    "SOLVENT_PRESETS",
    "CavityTessellation",
    "CPCMResult",
    "SolventModel",
    "SolventResult",
    "build_cavity",
    "cpcm_gradient",
    "cpcm_gradient_fd",
    "resolve_solvent",
    "run_cpcm_scf",
    "run_rks_periodic",
    "run_scs_mp2",
    "run_rohf",
    "run_roks",
    "run_scs_ump2",
    "run_sos_mp2",
    "run_sos_ump2",
    "run_uhf",
    "run_uks",
    "run_ump2",
    "set_num_threads",
    "spglib_version",
    "symmetrise",
    "system_info",
    "to_primitive",
    "write_molden",
    "write_trexio",
    "read_trexio",
    "read_trexio_fields",
    "write_trexio_fields",
    "read_qcschema",
    "write_qcschema",
    "molecule_from_qcschema",
    "molecule_to_qcschema",
    "run_qcschema",
    "TrexioData",
    "TrexioMOBlock",
    "TrexioSparse",
    "write_system_manifest",
    "write_orca_hess",
    "write_xyz_trajectory",
    "write_opt_trajectory",
    "normal_mode_trajectory",
    "write_poscar",
    # v0.9.0 -- gCP (Kruse-Grimme 2012)
    "GCPDataMissing",
    "GCPParams",
    "GCPResult",
    "available_gcp_basis_sets",
    "available_gcp_variants",
    "compute_gcp",
    "gcp_params_for",
    # v0.9.0 -- composite 3c methods
    "CompositeAvailability",
    "CompositeRecipe",
    "CompositeUnavailable",
    "ShortRangeCorrection",
    "list_composites",
    "resolve_composite",
    # v0.8.x -- native molecular geometry optimization (no ASE required)
    "MolecularOptimizeResult",
    "brent_minimize_1d",
    "optimize_molecule",
    "optimize_molecule_brent",
    # v0.8.x -- relaxed coordinate scans (molecular + periodic)
    "ScanResult",
    "ScanResult2D",
    "relaxed_scan",
    "relaxed_scan_2d",
    # v0.9.x -- Nudged Elastic Band (NEB). Increment 1 ships path-
    # construction primitives + dataclasses; Increment 2 adds the
    # improved-tangent driver (run_neb / NEBResult).
    "NEBImage",
    "NEBPath",
    "NEBResult",
    "NEBImageSCFError",
    "interpolate_idpp",
    "interpolate_linear",
    "run_neb",
    # geomopt -- uniform geometry optimisation framework.
    "CartesianCoordinates",
    "CellStrainCoordinates",
    "ConvergencePolicy",
    "ConvergenceReport",
    "CoordinateRepresentation",
    "DelocalizedInternalCoordinates",
    "EnergyGradientProvider",
    "FractionalCoordinates",
    "GeomOptResult",
    "HessianProvider",
    "HistoryManager",
    "HistoryRecord",
    "MolecularHessianFDProvider",
    "MolecularSCFProvider",
    "OptimizerState",
    "PeriodicSCFProvider",
    "RestartSerializer",
    "bfgs",
    "conjugate_gradient",
    "lbfgs",
    "resolve_optimizer",
    "run_geomopt",
    "run_periodic_geomopt",
    "steepest_descent",
    # Dimer method -- single-ended saddle-point search.
    "DimerResult",
    "run_dimer",
    # IRC -- intrinsic reaction coordinate (reaction-path following).
    "IRCResult",
    "run_irc",
    # Runtime functional-alias registration (Part B, PW1PW handover).
    "define_functional",
    "define_external_functional",
    # Optional Microsoft SKALA-1.1 full-grid XC backend.
    "SKALA_FUNCTIONAL",
    "SKALA_MODEL_REVISION",
    "SKALA_MODEL_SHA256",
    "ensure_skala_model",
    "skala_model_provenance",
]
