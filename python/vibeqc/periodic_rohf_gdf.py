"""Restricted-open-shell periodic HF (ROHF) on the native GDF route.

The spin-restricted open-shell sibling of
:func:`vibeqc.periodic_k_gdf.run_kuhf_periodic_gdf`. One set of spatial
orbitals per k-point carries integer ``2 / 1 / 0`` occupations, so the
determinant is a spin eigenfunction and ``<S^2> = S(S+1)`` exactly --
unlike UHF, which relaxes alpha and beta independently and contaminates.

Gamma is reached as a ``(1,1,1)`` mesh through the same code path (the
BIPOLE ROHF convention, ``run_rohf_periodic_multi_k_ewald3d``): there is
no separate Gamma driver to keep in sync.

Machinery reused verbatim from the closed/open-shell multi-k GDF
drivers, so the conventions cannot drift:

* the per-``(k_i,k_j)`` ``Lpq`` cderi cache (rsgdf / compcell / mdf) and
  its shared-q batched build;
* Hartree ``J`` from the BZ-summed total density ``S_j w_j D_t(k_j)``,
  with **no** conjugate on the diagonal cderi (the 2026-06-18 ``E_J``
  fix -- see ``handovers/HANDOVER_GDF_OUTSTANDING.md`` Sec. 9 item 3);
* the ``exxdiv='ewald'`` BvK-supercell Madelung K-shift, applied once
  per spin channel (:func:`vibeqc.madelung.apply_exxdiv_ewald_to_K`);
* the converged Ewald nuclear repulsion for ``dim == 3``.

What is genuinely new here is the SCF's orbital step: per k the two spin
Focks are combined by Roothaan's effective Fock
(:func:`vibeqc.rohf.roothaan_effective_fock`, Roothaan, Rev. Mod. Phys.
**32**, 179 (1960), Eq. 20), which is complex-Hermitian at a non-Gamma
k-point, and its aufbau diagonalisation supplies one MO set whose
occupations follow :func:`vibeqc.rohf._roothaan_occupations`.

Efficiency + parallelism over the k-points
-----------------------------------------

*Setup.* The ``rsgdf`` cderi cache is built by
:func:`vibeqc.periodic_k_gdf._build_rsgdf_lpq_cache_shared_q`, whose
per-momentum-transfer group loop is the existing MPI seam
(``handovers/HANDOVER_MPI.md`` step 4): whole q-groups are farmed across
ranks via :class:`vibeqc.mpi.KPointPartition` and reassembled with one
ordered gather. This driver reuses that path unchanged, so its dominant
setup cost is already distributed -- and byte-identical on one rank.

*Per iteration.* Exchange is the only ``n_k^2`` term, and a
restricted-open-shell SCF needs it for **both** spin channels. It is
contracted against the *occupied MO blocks* rather than the
``nbf x nbf`` densities
(:func:`vibeqc.periodic_k_gdf._build_k_from_lpq_factors`): the same
operator at ``n_occ / nbf`` of the flops, with no per-pair ``L.conj()``
copy, expressed as two GEMM calls per ``(k_i, k_j)`` pair so the work is
BLAS/OpenMP-threaded rather than looped in Python. Since 2026-08-02
every multi-k GDF driver uses that contraction, so this is the shared
kernel and not a ROHF peculiarity; what *is* specific here is that this
driver already holds the factors (they are its MO blocks) and so skips
the eigendecomposition the density-taking wrapper needs. A damped
iterate reuses the previous bare ``K`` by linearity instead of
rebuilding, so every exchange build stays at exactly ``n_occ`` columns.

Out of scope for this driver: ROKS (a functional), smearing and 2D slabs
raise; analytic gradients, COSX exchange and the ``ibz_native`` wedge are
not on the argument surface at all. On ``ibz_native`` specifically, note
that the closed/open-shell drivers gate it on whether the *state* is
invariant under the wedge's symmetry
(``_require_ibz_symmetric_state``, 2026-08-01) rather than on
multiplicity, so an integer-occupation ROHF state is not excluded in
principle -- wiring and validating it here is simply a separate
increment.
"""

from __future__ import annotations

from .guess import periodic_result_selection

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    InitialGuess,
    LatticeSumOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    SCFIteration,
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    ewald_nuclear_repulsion,
    nuclear_repulsion_per_cell,
)
from .guess import (
    _coerce_periodic_driver_guess,
    initial_densities_open_shell,
    periodic_fock_guess_k,
)
from .aux_basis import (
    build_lpq_bloch_compcell,
    build_lpq_bloch_mdf,
    default_aux_for,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from .kpoints import KPoints
from .linear_dependence import PeriodicLinearDependenceSummary
from .madelung import apply_exxdiv_ewald_to_K
from .periodic_k_gdf import (
    _build_k_from_lpq_factors,
    _build_k_from_densities,
    _signed_gram_factors,
    _build_rsgdf_lpq_cache_shared_q,
    _build_scf_range_separated_lpq_cache,
    _check_energy_sanity,
    _expand_ibz_kmesh_to_full_bz,
    _gdf_overlap_preflight,
    _kmesh_to_kpoints_weights,
    _madelung_for_kmesh,
    _mesh_tuple_for_system,
    _gdf_density_return_finalizer,
    _oneel_lattice_opts,
    _preflight_gdf_oneel_memory,
    _options_or_default,
    _preflight_gdf_lpq_memory,
    _reject_slab_dim,
    _warn_multik_dense_core_gdf_parity_hold,
)
from .periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _diag_in_orth_basis,
)
from .periodic_scf_accelerators import (
    _DIIS_LIKE,
    DynamicDamping,
    SCFAccelerator,
    _below_extrapolation_floor,
    _k_weighted_error_norm,
    _multik_diis_with_policy,
)
from .progress import ProgressLogger, resolve_progress
from .rohf import (
    _commutator_error,
    _densities_from_occupations,
    _roothaan_occupations,
    roothaan_effective_fock,
)

__all__ = [
    "PeriodicKROHFGDFResult",
    "run_krohf_periodic_gdf",
]


from .guess import periodic_guess_capabilities

ROHF_GDF_GUESSES = periodic_guess_capabilities("gdf", "ROHF", multi_k=True)


@dataclass
class PeriodicKROHFGDFResult:
    """Result of :func:`run_krohf_periodic_gdf`.

    Restricted-open-shell multi-k GDF. One MO set per k-point
    (``mo_coeffs`` / ``mo_energies`` from the effective Fock) with
    ``mo_occupations`` in ``{2, 1, 0}`` from a canonical refill. The
    separately returned alpha and beta AO densities are the accepted
    densities used for the energy and physical spin Focks. At finite
    iteration or with damping they can differ from that refill.
    ``fock`` is the Roothaan effective Fock per k, which is the operator
    the returned orbitals diagonalise. ``s_squared`` equals
    ``s_squared_ideal`` by construction -- a spin-restricted open shell
    carries no spin contamination, so any deviation would be a bug, not
    a diagnostic.
    """

    energy: float
    e_electronic: float
    e_nuclear: float
    n_iter: int
    converged: bool
    s_squared: float
    s_squared_ideal: float

    mo_energies: List[np.ndarray]
    mo_coeffs: List[np.ndarray]
    mo_occupations: List[np.ndarray]
    density_alpha: List[np.ndarray]
    density_beta: List[np.ndarray]
    fock: List[np.ndarray]
    fock_alpha: List[np.ndarray]
    fock_beta: List[np.ndarray]

    overlap: List[np.ndarray]
    hcore: List[np.ndarray]
    kpoints_cart: np.ndarray
    kpoint_weights: np.ndarray
    n_alpha: int = 0
    n_beta: int = 0
    scf_trace: List[SCFIteration] = field(default_factory=list)

    e_coulomb: float = 0.0
    e_hf_exchange: float = 0.0
    aux_basis_name: str = ""
    n_aux: int = 0
    # What canonical orthogonalisation and the density fit discarded;
    # rendered into the .out by periodic_runner.
    linear_dependence: Optional[PeriodicLinearDependenceSummary] = None
    method: str = "rohf"
    backend: str = "native-multi-k-gdf-rohf"

    density_alpha_lattice: Optional[object] = None
    density_beta_lattice: Optional[object] = None
    density_lattice_mesh: Optional[Tuple[int, int, int]] = None
    density_lattice_reserved_peak_bytes: int = 0

    @property
    def energy_per_cell_ha(self) -> float:
        return float(self.energy)

    @property
    def density(self) -> List[np.ndarray]:
        """Runner-facing total density, one AO block per k-point."""
        return [
            np.asarray(density_alpha) + np.asarray(density_beta)
            for density_alpha, density_beta in zip(
                self.density_alpha, self.density_beta
            )
        ]

    @property
    def occupations(self) -> List[np.ndarray]:
        """Runner-facing alias for the restricted 2/1/0 occupations."""
        return self.mo_occupations

    @property
    def runtime_backend(self) -> str:
        """Public-runner backend provenance, including any parity hold."""
        return self.backend

    restart_basis: object = None
    restart_lattice: object = None
    guess_selection: object = None

    restart_kpoints: object = None
    restart_weights: object = None
    rsgdf_ke_cutoff: float = 200.0


def _occupied_factors(
    c: np.ndarray, occ: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha / beta occupied MO blocks of one restricted-open MO set.

    ``D_alpha = W_a W_a^H`` and ``D_beta = W_b W_b^H`` for the returned
    blocks, exactly matching :func:`vibeqc.rohf._densities_from_occupations`
    -- alpha takes every occupied orbital, beta only the doubly occupied
    ones. These are the factors the ``n_k^2`` exchange build contracts
    against instead of the densities.
    """
    occ_arr = np.asarray(occ, dtype=float)
    c = np.asarray(c)
    return c[:, occ_arr > 0.0], c[:, occ_arr == 2.0]


def run_krohf_periodic_gdf(
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: Union[Sequence[int], KPoints, BlochKMesh] = (1, 1, 1),
    options: Optional[PeriodicRHFOptions] = None,
    *,
    initial_density_k: Optional[Sequence] = None,
    aux_basis: Optional[str] = None,
    aux_drop_eta: float = 0.0,
    linear_dep_threshold: float = 1e-7,
    gdf_linear_dep_threshold: float = 1e-9,
    compcell_eta: float = 1.0,
    apply_aft_correction: bool = True,
    aft_ft_convention: str = "libint",
    aft_precision: float = 1e-10,
    rcut_strategy: Optional[object] = "pyscf_auto",
    rcut_precision: float = 1e-8,
    gdf_method: str = "rsgdf",
    rsgdf_omega: float = 0.4,
    rsgdf_g_precision: float = 1e-10,
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: Optional[float] = None,
    fit_screen_threshold: float = 0.0,
    mdf_ke_cutoff: float = 40.0,
    return_lattice_density: bool = False,
    lattice_density_memory_bytes: int = 128 * 1024**2,
    check_energy_sanity: bool = True,
    progress: Union[bool, ProgressLogger, None] = None,
    verbose: Optional[int] = None,
    _lpq_cache_builder=None,
) -> PeriodicKROHFGDFResult:
    """Restricted-open-shell periodic HF via native multi-k GDF.

    One spatial MO set per k-point with integer ``2 / 1 / 0``
    occupations from ``system.multiplicity``; Gamma is the ``(1,1,1)``
    mesh of this same path. ``multiplicity=1`` reduces to RHF and
    reproduces :func:`vibeqc.run_krhf_periodic_gdf` (the collapse gate).

    Per iteration, at every k: ``F_s = Hcore + J - K_s`` with ``J`` from
    the BZ-summed total density and ``K_s`` from the per-``(k_i,k_j)``
    cderi cache carrying the ``exxdiv='ewald'`` supercell-Madelung shift,
    then one Roothaan effective Fock
    (:func:`vibeqc.rohf.roothaan_effective_fock`) whose aufbau
    diagonalisation gives the next MO set. The energy is the ordinary
    open-shell expression ``S_k w_k [Tr(D_a H) + Tr(D_b H) +
    1/2 Tr(D_a F_a^2e) + 1/2 Tr(D_b F_b^2e)] + E_nuc``, identical to the
    UHF driver's -- ROHF differs only in which determinant it is
    evaluated at.

    Fails closed (rather than returning an approximation) on: a
    ``dim == 2`` slab, ``smearing_temperature > 0``, a non-3D
    ``exxdiv='ewald'`` requirement, and any ``gdf_method`` outside
    ``{'rsgdf', 'compcell', 'mdf'}``. A functional (ROKS), analytic
    gradients, COSX exchange and the ``ibz_native`` wedge are not part of
    this driver's surface at all -- they are separate, separately
    validated increments.

    ``rsgdf_tail_ke_cutoff`` behaves exactly as in the closed- and
    open-shell drivers: it extends the fit's ``M`` / ``T`` G-sums over
    the exact complementary reciprocal shell, and a parity-sized tail
    lifts the dense-core ``+PARITY_HELD`` tag.

    Notes
    -----
    SCF acceleration is Pulay DIIS on the effective Fock (with the
    ``R_CDIIS`` / ``AD_CDIIS`` depth policies honoured). The
    energy-functional accelerators -- EDIIS, ADIIS and KDIIS -- are
    deliberately **not** dispatched here even when
    ``options.scf_accelerator`` selects one: they model the Fock as
    ``dE/dD`` for a single density (EDIIS/ADIIS) or build a density from
    the lowest ``n_occ`` columns (KDIIS), and Roothaan's effective Fock
    is neither. Silently feeding it to them would extrapolate on a wrong
    model; the driver logs the substitution instead.
    """
    from .periodic_mdf import _MdfScfSource

    private_mdf = isinstance(_lpq_cache_builder, _MdfScfSource)
    if private_mdf:
        _lpq_cache_builder.require_driver(
            system, gdf_method=gdf_method, k_exchange='gdf', ibz_native=False)
    if _lpq_cache_builder is not None and (int(system.dim) != 3 or gdf_method != 'rsgdf'):
        raise NotImplementedError('ROHF private cache injection requires the 3D rsgdf driver')
    _finish_density_return = _gdf_density_return_finalizer(
        system, basis, kmesh, return_lattice_density, 2,
        lattice_density_memory_bytes,
    )
    _reject_slab_dim(system, "run_krohf_periodic_gdf")

    from .periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from .periodic_v_ne import compute_nuclear_lattice_dispatch

    plog = resolve_progress(progress, verbose=verbose)
    opts = _options_or_default(options, is_ks=False)
    from .pbc_gdf import _refuse_ecp_options
    _refuse_ecp_options(opts, "run_krohf_periodic_gdf", system=system)
    lat_opts: LatticeSumOptions = opts.lattice_opts

    if str(getattr(opts, "functional", "") or ""):
        raise NotImplementedError(
            "run_krohf_periodic_gdf: a functional selects ROKS, which is "
            "not implemented on the GDF route (it is a separate increment "
            "of the same gate). Use jk_method='bipole' "
            "(run_roks_periodic_multi_k_ewald3d) for periodic ROKS, or "
            "drop the functional for ROHF."
        )

    guess = _coerce_periodic_driver_guess(
        getattr(opts, "initial_guess", None),
        driver="run_krohf_periodic_gdf",
        supported=ROHF_GDF_GUESSES, restart_supplied=initial_density_k is not None,
    )
    from .guess import select_initial_guess
    select_initial_guess(
        system.unit_cell_molecule(), opts.initial_guess,
        is_periodic=True, is_open_shell=True,
        atomic_spins=getattr(opts, "atomic_spins", None),
        restart_supplied=initial_density_k is not None,
    )


    if gdf_method not in ("compcell", "rsgdf", "mdf"):
        raise ValueError(
            f"run_krohf_periodic_gdf: gdf_method must be 'compcell', "
            f"'rsgdf', or 'mdf'; got {gdf_method!r}"
        )
    if float(fit_screen_threshold) < 0.0:
        raise ValueError(
            "run_krohf_periodic_gdf: fit_screen_threshold must be >= 0; "
            f"got {fit_screen_threshold}"
        )
    if float(fit_screen_threshold) > 0.0 and gdf_method != "rsgdf":
        raise NotImplementedError(
            "run_krohf_periodic_gdf: fit_screen_threshold is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    if rsgdf_tail_ke_cutoff is not None and gdf_method != "rsgdf":
        raise NotImplementedError(
            "run_krohf_periodic_gdf: rsgdf_tail_ke_cutoff is implemented "
            f"for gdf_method='rsgdf' only (got {gdf_method!r})."
        )
    if float(getattr(opts, "smearing_temperature", 0.0) or 0.0) > 0.0:
        # Fractional occupations are not a restricted-open-shell state:
        # the 2/1/0 filling is what makes the determinant a spin
        # eigenfunction. A smeared ROHF needs a fractional frontier-shell
        # model (the periodic counterpart of the molecular
        # ``fractional_open_shell``), which is not implemented -- refuse
        # rather than silently return a different method's answer.
        raise NotImplementedError(
            "run_krohf_periodic_gdf: smearing_temperature > 0 is not "
            "supported. Restricted-open-shell occupations are integer "
            "2/1/0 by construction; a smeared metallic open shell needs a "
            "fractional frontier-shell model that this driver does not "
            "implement. Use run_kuhf_periodic_gdf for a smeared "
            "spin-polarised metal."
        )


    n_elec = system.n_electrons()
    mult = int(system.multiplicity)
    if mult < 1:
        raise ValueError(
            f"run_krohf_periodic_gdf: multiplicity must be >= 1, got {mult}"
        )
    if (n_elec + mult - 1) % 2 != 0:
        raise ValueError(
            f"run_krohf_periodic_gdf: n_electrons={n_elec} and multiplicity="
            f"{mult} cannot be split into integer a/b occupations."
        )
    n_alpha = (n_elec + mult - 1) // 2
    n_beta = (n_elec - mult + 1) // 2

    # Symmetry-reduced (IBZ) input: adopt the expanded full-BZ mesh
    # before anything is derived from it. Weighting irreducible
    # representatives is not orbit summation for exchange (+1.389 Ha on
    # MgO (2,2,2)); see the matching note in run_krhf_periodic_gdf.
    kmesh_full = _expand_ibz_kmesh_to_full_bz(system, kmesh)
    if kmesh_full is not None:
        kmesh = kmesh_full
    kpoints_cart, weights = _kmesh_to_kpoints_weights(system, kmesh)
    n_k = int(kpoints_cart.shape[0])

    aux_name = aux_basis or default_aux_for(basis.name)
    plog.banner(f"run_krohf_periodic_gdf  ROHF  kmesh={n_k} k-points")
    bulk_sr = gdf_method == "rsgdf" and int(system.dim) == 3
    if bulk_sr and rsgdf_tail_ke_cutoff is not None:
        import warnings
        warnings.warn(
            "rsgdf_tail_ke_cutoff is obsolete for the SR/LR fit; "
            "use rsgdf_g_precision to control the raw integral error",
            DeprecationWarning, stacklevel=2,
        )
        rsgdf_tail_ke_cutoff = None
    dense_core_parity_held = not bulk_sr and _warn_multik_dense_core_gdf_parity_hold(
        system,
        gdf_method,
        basis,
        rsgdf_tail_ke_cutoff,
        rsgdf_ke_cutoff,
        "run_krohf_periodic_gdf",
    )
    plog.info(
        f"ROHF multi-k GDF / aux={aux_name}, n_alpha={n_alpha}, "
        f"n_beta={n_beta} (mult={mult}), {n_alpha - n_beta} open shell(s)"
    )

    # ---- One-electron integrals (Ewald-3D gauge for V_ne / e_nuc) -----
    oneel_lat_opts = _oneel_lattice_opts(
        system, basis, lat_opts,
        rcut_strategy=rcut_strategy, k_points_cart=kpoints_cart, plog=plog,
    )
    _preflight_gdf_oneel_memory(system, basis, oneel_lat_opts, n_kpoints=n_k)
    with plog.stage(
        "integrals_lattice",
        detail=f"S/T cutoff {oneel_lat_opts.cutoff_bohr:.2f}, "
        f"V cutoff {oneel_lat_opts.cutoff_bohr:.2f}",
    ):
        S_lat = compute_overlap_lattice(basis, system, oneel_lat_opts)
        T_lat = compute_kinetic_lattice(basis, system, oneel_lat_opts)
        gauge_lat_opts = _gauge_lat_opts_for_v_ne_and_e_nuc(oneel_lat_opts, system)
        V_lat = compute_nuclear_lattice_dispatch(basis, system, gauge_lat_opts)

    S_k: List[np.ndarray] = []
    Hcore_k: List[np.ndarray] = []
    X_k: List[np.ndarray] = []
    n_kept_k: List[int] = []
    _s_lo, _s_hi = float("inf"), float("-inf")
    for k_idx in range(n_k):
        k_arr = kpoints_cart[k_idx]
        Sk = np.asarray(bloch_sum(S_lat, k_arr))
        Tk = np.asarray(bloch_sum(T_lat, k_arr))
        Vk = np.asarray(bloch_sum(V_lat, k_arr))
        Sk = 0.5 * (Sk + Sk.conj().T)
        Hk = 0.5 * ((Tk + Vk) + (Tk + Vk).conj().T)
        _gdf_overlap_preflight(Sk, plog=plog, label=f"S(k={k_idx})", basis=basis)
        _ev = np.linalg.eigvalsh(Sk)
        _s_lo = min(_s_lo, float(_ev[0]))
        _s_hi = max(_s_hi, float(_ev[-1]))
        Xk, n_kept = _canonical_orthogonalizer_complex(
            Sk, linear_dep_threshold, normalize_diag_first=True
        )
        if n_alpha > n_kept:
            raise RuntimeError(
                f"run_krohf_periodic_gdf: orthogonalisation at k={k_idx} kept "
                f"{n_kept} directions; need >= {n_alpha}."
            )
        S_k.append(Sk)
        Hcore_k.append(Hk)
        X_k.append(Xk)
        n_kept_k.append(n_kept)

    if int(system.dim) == 3:
        # Converged Ewald nuclear energy -- the truncated direct sum is
        # unconverged at the 1e-5 Ha level on dense ionic cells and
        # carries a spurious geometry dependence (2026-07-29 finding).
        e_nuc = float(ewald_nuclear_repulsion(system))
    else:
        e_nuc = float(nuclear_repulsion_per_cell(system, gauge_lat_opts))

    # ---- Per-(k_i,k_j) Lpq cderi cache (spin-independent) -------------
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=float(aux_drop_eta))
    q_metric_cache = None
    if gdf_method == "rsgdf":
        aux_modrho = aux if bulk_sr else make_modrho_aux_basis(aux, mol)
        q_metric_cache = {}
    elif gdf_method == "mdf":

        def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray) -> np.ndarray:
            return build_lpq_bloch_mdf(
                system, basis, aux, ki, kj, molecule=mol, lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                mdf_ke_cutoff=float(mdf_ke_cutoff),
                rcut_strategy=rcut_strategy, rcut_precision=float(rcut_precision),
            )
    else:

        def _build_pair_lpq(ki: np.ndarray, kj: np.ndarray) -> np.ndarray:
            return build_lpq_bloch_compcell(
                system, basis, aux, kj - ki, molecule=mol, lat_opts=lat_opts,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                compcell_eta=float(compcell_eta),
                apply_aft_correction=bool(apply_aft_correction),
                aft_ft_convention=str(aft_ft_convention),
                aft_precision=float(aft_precision),
                rcut_strategy=rcut_strategy, rcut_precision=float(rcut_precision),
            )

    lpq_cache: Dict[Tuple[int, int], np.ndarray] = {}
    _preflight_gdf_lpq_memory(
        plog,
        n_basis=basis.nbasis,
        n_aux=aux.nbasis,
        n_kpoints=n_k,
        need_k_pairs=True,
        open_shell=True,
        route_label="ROHF",
        options=opts,
    )
    with plog.stage("gdf_cderi", detail=f"per-pair Lpq, {n_k} k ({gdf_method})"):
        if gdf_method == "rsgdf":
            if bulk_sr:
                lpq_cache = _build_scf_range_separated_lpq_cache(
                    system, basis, aux, kpoints_cart, True,
                    omega=rsgdf_omega, raw_integral_error=rsgdf_g_precision,
                    ke_cutoff=rsgdf_ke_cutoff, linear_dep_thr=gdf_linear_dep_threshold,
                    lat_opts=lat_opts, fit_screen_threshold=fit_screen_threshold,
                    options=opts, open_shell=True, progress=plog,
                    pair_cache_builder=_lpq_cache_builder,
                    bra_rows=None, oneel_lattices=(S_lat, T_lat, V_lat),
                    density_return_bytes=getattr(
                        getattr(_finish_density_return, '__self__', None),
                        'reserved_peak_bytes', 0,
                    ),
                )
            else:
                lpq_cache = _build_rsgdf_lpq_cache_shared_q(
                    system,
                    basis,
                    aux_modrho,
                    kpoints_cart,
                    True,
                    ke_cutoff=float(rsgdf_ke_cutoff),
                    tail_ke_cutoff=(
                        float(rsgdf_tail_ke_cutoff)
                        if rsgdf_tail_ke_cutoff is not None
                        else None
                    ),
                    lat_opts=lat_opts,
                    linear_dep_thr=float(gdf_linear_dep_threshold),
                    fit_screen_threshold=float(fit_screen_threshold),
                    progress=plog,
                    q_metric_cache=q_metric_cache,
                )
        else:
            for i in range(n_k):
                ki = kpoints_cart[i]
                for j in range(n_k):
                    lpq_cache[(i, j)] = _build_pair_lpq(ki, kpoints_cart[j])
    n_fit = int(lpq_cache[(0, 0)].shape[0])
    plog.info(f"Lpq cache: {len(lpq_cache)} pairs, {n_fit} fit vectors")

    nbf = int(basis.nbasis)
    madelung = _madelung_for_kmesh(system, _mesh_tuple_for_system(system, kmesh))

    def _build_j_from_lpq(D_k_in: Sequence[np.ndarray]) -> List[np.ndarray]:
        from .aux_basis import _build_coulomb_from_diagonal_factors

        return _build_coulomb_from_diagonal_factors(
            [lpq_cache[(i, i)] for i in range(n_k)], D_k_in, weights,
        )

    guess_fock = periodic_fock_guess_k(
        system, basis, kpoints_cart, guess, lattice_opts=oneel_lat_opts,
        overlap_lattice=S_lat, kinetic_lattice=T_lat,
    )
    # Fock-mode guesses use the same restricted-open occupation convention
    # as the SCF; density-mode guesses replace both densities and factors.
    C_k: List[np.ndarray] = []
    eps_k: List[np.ndarray] = []
    occ_k: List[np.ndarray] = []
    for i in range(n_k):
        c, eps = _diag_in_orth_basis(
            Hcore_k[i] if guess_fock is None else guess_fock[i], X_k[i]
        )
        c = np.asarray(c, dtype=complex)
        # At the Hcore guess the alpha Fock IS Hcore, so the open-shell
        # tie-break energies are the same eigenvalues.
        occ_k.append(_roothaan_occupations(eps, eps, n_alpha, n_beta))
        C_k.append(c)
        eps_k.append(eps)

    def _state_from_orbitals(
        C_list: Sequence[np.ndarray], occ_list: Sequence[np.ndarray]
    ):
        """Per-k ``(D_a, D_b, W_a, W_b)`` from one restricted-open MO set."""
        D_a, D_b, W_a, W_b = [], [], [], []
        for i in range(n_k):
            da, db = _densities_from_occupations(C_list[i], occ_list[i])
            wa, wb = _occupied_factors(C_list[i], occ_list[i])
            D_a.append(np.asarray(da, dtype=complex))
            D_b.append(np.asarray(db, dtype=complex))
            W_a.append(wa)
            W_b.append(wb)
        return D_a, D_b, W_a, W_b

    D_a_k, D_b_k, W_a_k, W_b_k = _state_from_orbitals(C_k, occ_k)
    if guess in (InitialGuess.SAD, InitialGuess.MINAO, InitialGuess.PATOM):
        da, db = initial_densities_open_shell(
            system.unit_cell_molecule(), basis, n_alpha, n_beta,
            InitialGuess.SAD if guess == InitialGuess.PATOM else guess,
            periodic_system=system, lattice_opts=oneel_lat_opts,
            overlap=S_k, weights=weights,
            atomic_spins=getattr(opts, "atomic_spins", None),
        )
        D_a_k = [np.asarray(da, dtype=complex).copy() for _ in range(n_k)]
        D_b_k = [np.asarray(db, dtype=complex).copy() for _ in range(n_k)]

        if guess == InitialGuess.PATOM:
            # The common open-shell PATOM contract is one in-field HF step
            # per spin. ROHF imposes common orbitals at its first orbital step.
            seed_j = _build_j_from_lpq([a + b for a, b in zip(D_a_k, D_b_k)])
            for densities, population in ((D_a_k, n_alpha), (D_b_k, n_beta)):
                exchange = _build_k_from_densities(
                    lpq_cache, densities, weights, range(n_k), nbasis=nbf
                )
                exchange = apply_exxdiv_ewald_to_K(exchange, S_k, densities, madelung)
                for i in range(n_k):
                    coeffs, _ = _diag_in_orth_basis(
                        Hcore_k[i] + seed_j[i] - exchange[i], X_k[i]
                    )
                    occupied = coeffs[:, :population]
                    densities[i] = occupied @ occupied.conj().T

    if guess == InitialGuess.READ:
        from .guess import normalize_spin_density_k_guess
        if initial_density_k is None:
            if n_k != 1:
                raise ValueError("ROHF/GDF READ requires complete per-k spin densities")
            from .guess_read import resolve_periodic_read_densities_open
            da, db = resolve_periodic_read_densities_open(
                basis, read_density_alpha=getattr(opts, "read_density_alpha", None),
                read_density_beta=getattr(opts, "read_density_beta", None),
                read_path=getattr(opts, "read_path", ""),
            )
            initial_density_k = ([da], [db])
        if len(initial_density_k) != 2:
            raise ValueError("ROHF/GDF READ requires alpha and beta block lists")
        D_a_k, D_b_k = normalize_spin_density_k_guess(
            *initial_density_k, S_k, weights, n_alpha, n_beta,
        )
    if guess in (InitialGuess.SAD, InitialGuess.MINAO, InitialGuess.PATOM, InitialGuess.READ):
        def density_factor(density):
            factors = _signed_gram_factors(density)
            if factors is None or any(sign < 0 for sign, _ in factors):
                raise ValueError("ROHF/GDF initial density must be positive semidefinite")
            return (
                np.hstack([factor for _, factor in factors])
                if factors else np.zeros((nbf, 0), dtype=complex)
            )

        W_a_k = [density_factor(density) for density in D_a_k]
        W_b_k = [density_factor(density) for density in D_b_k]
    plog.info(f"initial guess: {guess.name} (shared construction)")
    density_mode_start = guess in (InitialGuess.SAD, InitialGuess.MINAO, InitialGuess.PATOM, InitialGuess.READ)
    D_a_prev, D_b_prev = [D.copy() for D in D_a_k], [D.copy() for D in D_b_k]
    # Bare (pre-exxdiv) exchange at the previous iteration's *used*
    # densities, so a damped iterate reuses it by linearity instead of
    # rebuilding -- see the damping branch in the SCF loop.
    Ka_bare_prev: Optional[List[np.ndarray]] = None
    Kb_bare_prev: Optional[List[np.ndarray]] = None

    # ---- SCF setup ----------------------------------------------------
    damping = float(opts.damping)
    if not (0.0 <= damping < 1.0):
        raise ValueError(
            f"run_krohf_periodic_gdf: damping must be in [0, 1); got {damping}"
        )
    damper: Optional[DynamicDamping] = None
    if bool(getattr(opts, "dynamic_damping", False)):
        damper = DynamicDamping(
            initial_alpha=damping,
            alpha_min=float(getattr(opts, "dynamic_damping_min", 0.0)),
            alpha_max=float(getattr(opts, "dynamic_damping_max", 0.95)),
        )
    use_diis = bool(opts.use_diis)
    diis_start_iter = int(opts.diis_start_iter)
    diis = (
        _multik_diis_with_policy(opts, int(opts.diis_subspace_size))
        if use_diis
        else None
    )
    if diis is not None:
        requested = getattr(opts, "scf_accelerator", SCFAccelerator.DIIS)
        if requested not in _DIIS_LIKE:
            plog.info(
                "SCF accelerator: Pulay DIIS on the Roothaan effective Fock "
                f"(requested {requested!r} is an energy-functional or "
                "orbital-rotation accelerator whose model does not apply to "
                "an effective Fock; see the driver docstring)"
            )
    max_iter = int(opts.max_iter)

    spin = 0.5 * (n_alpha - n_beta)
    s_squared = float(spin * (spin + 1.0))
    scf_trace: List[SCFIteration] = []
    _lindep = PeriodicLinearDependenceSummary(
        n_basis=int(basis.nbasis),
        n_kept_per_k=list(n_kept_k),
        threshold=float(linear_dep_threshold),
        min_overlap_eigenvalue=_s_lo,
        max_overlap_eigenvalue=_s_hi,
        n_aux=int(aux.nbasis),
        n_fit_kept=int(n_fit),
        aux_threshold=float(gdf_linear_dep_threshold),
    )
    result = PeriodicKROHFGDFResult(
                 restart_kpoints=kpoints_cart.copy(), restart_weights=weights.copy(),
                 guess_selection=periodic_result_selection(system, opts.initial_guess, restarted=initial_density_k is not None),
                 restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        energy=0.0, e_electronic=0.0, e_nuclear=float(e_nuc), n_iter=0,
        converged=False, s_squared=s_squared, s_squared_ideal=s_squared,
        mo_energies=[] if density_mode_start else [e.copy() for e in eps_k],
        mo_coeffs=[] if density_mode_start else [c.copy() for c in C_k],
        mo_occupations=[] if density_mode_start else [o.copy() for o in occ_k],
        density_alpha=[D.copy() for D in D_a_k],
        density_beta=[D.copy() for D in D_b_k],
        fock=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        fock_alpha=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        fock_beta=[np.empty((0, 0), dtype=complex) for _ in range(n_k)],
        overlap=[S.copy() for S in S_k], hcore=[H.copy() for H in Hcore_k],
        kpoints_cart=kpoints_cart.copy(), kpoint_weights=weights.copy(),
        n_alpha=int(n_alpha), n_beta=int(n_beta), scf_trace=scf_trace,
        aux_basis_name=aux_name, n_aux=int(aux.nbasis),
        linear_dependence=_lindep,
        rsgdf_ke_cutoff=float(getattr(lpq_cache, 'lr_ke_cutoff', rsgdf_ke_cutoff)),
    )
    if private_mdf:
        result.backend += '+private-mdf-unvalidated'
    if dense_core_parity_held:
        result.backend = result.backend + "+PARITY_HELD"
    plog.banner("SCF (ROHF multi-k, native GDF)")

    E_prev = 0.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if damper is not None:
            damping = damper.alpha
        diis_active = use_diis and it >= diis_start_iter
        damped = not (it == 1 or damping == 0.0 or diis_active)
        if damped:
            Da_used = [damping * Dp + (1.0 - damping) * Dn
                       for Dp, Dn in zip(D_a_prev, D_a_k)]
            Db_used = [damping * Dp + (1.0 - damping) * Dn
                       for Dp, Dn in zip(D_b_prev, D_b_k)]
        else:
            Da_used = [D.copy() for D in D_a_k]
            Db_used = [D.copy() for D in D_b_k]
        D_total = [Da_used[i] + Db_used[i] for i in range(n_k)]

        J_k = _build_j_from_lpq(D_total)
        # Exchange from the OCCUPIED MO blocks of the current aufbau
        # iterate -- the n_occ/nbf-cost contraction. The damped iterate is
        # then reached by linearity of K in the density (K is a linear map,
        # so K(a.D_prev + (1-a).D_new) = a.K(D_prev) + (1-a).K(D_new)),
        # reusing the previous iteration's bare K. Chaining sqrt-scaled
        # factors instead would be equally exact but would grow the factor
        # rank by n_occ every iteration; the cached-K recursion keeps every
        # build at exactly n_occ columns.
        Ka_bare = _build_k_from_lpq_factors(
            lpq_cache, [[W] for W in W_a_k], weights, nbasis=nbf
        )
        Kb_bare = _build_k_from_lpq_factors(
            lpq_cache, [[W] for W in W_b_k], weights, nbasis=nbf
        )
        if damped and Ka_bare_prev is not None:
            Ka_bare = [damping * Kp + (1.0 - damping) * Kn
                       for Kp, Kn in zip(Ka_bare_prev, Ka_bare)]
            Kb_bare = [damping * Kp + (1.0 - damping) * Kn
                       for Kp, Kn in zip(Kb_bare_prev, Kb_bare)]
        Ka_bare_prev, Kb_bare_prev = Ka_bare, Kb_bare
        # exxdiv is the linear -xi.S.D.S shift, so applying it to the
        # already-combined K with the combined density is the same
        # operator either way.
        Ka_k = list(apply_exxdiv_ewald_to_K(Ka_bare, S_k, Da_used, madelung))
        Kb_k = list(apply_exxdiv_ewald_to_K(Kb_bare, S_k, Db_used, madelung))

        Fa_2e = [J_k[i] - Ka_k[i] for i in range(n_k)]
        Fb_2e = [J_k[i] - Kb_k[i] for i in range(n_k)]
        Fa_k = [
            0.5 * ((Fa_2e[i] + Hcore_k[i]) + (Fa_2e[i] + Hcore_k[i]).conj().T)
            for i in range(n_k)
        ]
        Fb_k = [
            0.5 * ((Fb_2e[i] + Hcore_k[i]) + (Fb_2e[i] + Hcore_k[i]).conj().T)
            for i in range(n_k)
        ]

        E_coulomb = 0.5 * sum(
            float(weights[i]) * float(np.real(np.trace(D_total[i] @ J_k[i])))
            for i in range(n_k)
        )
        E_hf_K = -0.5 * sum(
            float(weights[i]) * (
                float(np.real(np.trace(Da_used[i] @ Ka_k[i])))
                + float(np.real(np.trace(Db_used[i] @ Kb_k[i])))
            )
            for i in range(n_k)
        )
        E_elec = 0.0
        for i in range(n_k):
            w = float(weights[i])
            E_elec += w * (
                float(np.real(np.trace(Da_used[i] @ Hcore_k[i])))
                + float(np.real(np.trace(Db_used[i] @ Hcore_k[i])))
                + 0.5 * float(np.real(np.trace(Da_used[i] @ Fa_2e[i])))
                + 0.5 * float(np.real(np.trace(Db_used[i] @ Fb_2e[i])))
            )
        E_total = E_elec + float(e_nuc)

        # Roothaan effective Fock + its commutator with the TOTAL density
        # (the occupied-virtual block of F_eff vanishes at convergence).
        Feff_k: List[np.ndarray] = []
        err_k: List[np.ndarray] = []
        gnorm2 = 0.0
        for i in range(n_k):
            f_eff = roothaan_effective_fock(
                Fa_k[i], Fb_k[i], Da_used[i], Db_used[i], S_k[i]
            )
            err = _commutator_error(f_eff, D_total[i], S_k[i], X_k[i])
            Feff_k.append(f_eff)
            err_k.append(err)
            gnorm2 += float(weights[i]) * float(np.linalg.norm(err) ** 2)
        grad_norm = float(np.sqrt(gnorm2))

        dE = E_total - E_prev
        converged = (
            it > 1
            and abs(dE) < float(opts.conv_tol_energy)
            and grad_norm < float(opts.conv_tol_grad)
        )
        scf_trace.append(SCFIteration(
            iter=it, energy=float(E_total),
            delta_e=float(dE if it > 1 else 0.0), grad_norm=float(grad_norm),
            diis_subspace=(diis.subspace_size if diis is not None else 0),
        ))
        plog.iteration(
            it, energy=float(E_total), dE=float(dE if it > 1 else 0.0),
            grad=float(grad_norm),
            diis=(diis.subspace_size if diis is not None else 0),
        )

        terminating_state = converged or it == max_iter
        if terminating_state:
            # Canonicalize the accepted state's effective Fock, not the
            # previous iteration's or DIIS-extrapolated operator. Keep
            # Da_used/Db_used: refilling these orbitals is a new trial.
            C_k, eps_k, occ_k = [], [], []
            for i in range(n_k):
                c, eps = _diag_in_orth_basis(Feff_k[i], X_k[i])
                c = np.asarray(c, dtype=complex)
                eps_alpha = np.einsum(
                    "pi,pq,qi->i", c.conj(), Fa_k[i], c, optimize=True
                ).real
                C_k.append(c)
                eps_k.append(eps)
                occ_k.append(_roothaan_occupations(eps, eps_alpha, n_alpha, n_beta))

        result.energy = float(E_total)
        result.e_electronic = float(E_elec)
        result.e_coulomb = float(E_coulomb)
        result.e_hf_exchange = float(E_hf_K)
        result.n_iter = it
        result.mo_energies = [e.copy() for e in eps_k]
        result.mo_coeffs = [c.copy() for c in C_k]
        result.mo_occupations = [np.asarray(o, dtype=float) for o in occ_k]
        if density_mode_start and it == 1 and not terminating_state:
            # A non-idempotent SAD/MINAO density has no restricted-open
            # integer-occupation MO representation. Never expose old HCORE
            # orbitals as if they generated the density evaluated here.
            result.mo_energies = []
            result.mo_coeffs = []
            result.mo_occupations = []
        result.density_alpha = [D.copy() for D in Da_used]
        result.density_beta = [D.copy() for D in Db_used]
        result.fock = [F.copy() for F in Feff_k]
        result.fock_alpha = [F.copy() for F in Fa_k]
        result.fock_beta = [F.copy() for F in Fb_k]

        if converged:
            result.converged = True
            # Guard first: a converged non-physical energy RAISES, so it
            # must not be announced as a good result before the check.
            if check_energy_sanity:
                _check_energy_sanity(
                    result, system, plog, entry="run_krohf_periodic_gdf"
                )
            plog.converged(n_iter=it, energy=E_total, converged=True)
            return _finish_density_return(result)

        if terminating_state:
            break

        F_diag = Feff_k
        if diis_active and not _below_extrapolation_floor(
            _k_weighted_error_norm(err_k, weights),
            max(float(np.linalg.norm(F)) for F in Feff_k),
        ):
            F_diag = diis.extrapolate(Feff_k, err_k, list(weights))

        C_new: List[np.ndarray] = []
        eps_new: List[np.ndarray] = []
        occ_new: List[np.ndarray] = []
        for i in range(n_k):
            c, eps = _diag_in_orth_basis(F_diag[i], X_k[i])
            c = np.asarray(c, dtype=complex)
            # Open-shell orbitals are picked by their ALPHA energy among
            # the non-core candidates (rohf._roothaan_occupations): the
            # effective Fock orders the closed space reliably but not the
            # open one.
            eps_alpha = np.einsum(
                "pi,pq,qi->i", c.conj(), Fa_k[i], c, optimize=True
            ).real
            C_new.append(c)
            eps_new.append(eps)
            occ_new.append(
                _roothaan_occupations(eps, eps_alpha, n_alpha, n_beta)
            )

        D_a_prev, D_b_prev = Da_used, Db_used
        C_k, eps_k, occ_k = C_new, eps_new, occ_new
        D_a_k, D_b_k, W_a_k, W_b_k = _state_from_orbitals(C_k, occ_k)
        if damper is not None:
            damper.update(E_total)
        E_prev = E_total

    result.converged = False
    plog.converged(
        n_iter=result.n_iter, energy=result.energy, converged=False
    )
    if check_energy_sanity:
        _check_energy_sanity(
            result, system, plog, entry="run_krohf_periodic_gdf"
        )
    return _finish_density_return(result)
