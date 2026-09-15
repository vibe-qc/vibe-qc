"""KS-DFT on a Cyclic Cluster Model reference (RKS-CCM / UKS-CCM).

Kohn-Sham CCM with the **four-center** WSSC-weighted Coulomb (and exact exchange
for hybrids) plus a semi-local exchange-correlation potential. The Coulomb /
exact-exchange operators carry the cyclic boundary conditions through the CCM
four-center (``method="union12"`` or ``"aiccm2026dev-a"``).  The historical
libxc path evaluates the local functional on an ordinary molecular Becke grid
over the supercell.  Full-grid external providers instead use the literal CCM
numerical-integration construction: the reference-cluster atom grid, periodic
images of every cluster AO, and an atomic partition normalized over the real
and virtual cluster atoms (Janetzko, Köster, and Salahub, J. Chem. Phys. 128,
024102 (2008), doi:10.1063/1.2817582, Eqs. 27-32). Concretely the KS matrix is

    F = h^CCM + J^CCM + V_xc[r]  (- a K^CCM for a hybrid),

The ordinary libxc branch is assembled by the production C++
``run_rks_scf_with_jk`` driver fed the CCM four-center JK builder and a
molecular supercell XC grid. Full-grid providers use the same production C++
JK builder inside a small Python SCF injection loop so their periodic
reference-cluster potential is kept variational -- the DFT analogue of
:func:`vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`.

Validation: the isolated-cell limit reproduces vibe-qc's molecular ``run_rks`` /
``run_uks`` (the CCM Coulomb is the molecular Coulomb and the XC grid is the
molecular grid there). Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839
(2014), doi:10.1002/jcc.23550.
"""

from __future__ import annotations

from .scf import _ccm_initial_guess, _with_ccm_guess

from dataclasses import dataclass

import numpy as np

from .experimental import (
    _reject_double_hybrid,
    _reject_open_shell_cluster,
    _warn_experimental,
)
from .integrals import ccm_overlap
from .padded import ccm_hcore, ccm_nuclear_repulsion
from .scf import (
    _ccm_scalable_cxx_method,
    _make_ccm_jk_builder,
    _validate_conv_tol_grad,
)

__all__ = ["CCMKSResult", "run_ccm_rks", "run_ccm_uks"]


@dataclass
class CCMKSResult:
    """KS-CCM result on the finite cyclic cluster.

    ``energy`` is the total energy of the finite cyclic cluster/supercell, NOT
    normalized per reference cell, exactly as
    :class:`~vibeqc.periodic.ccm.scf.CCMSCFResult` documents for the HF
    drivers: compare with a primitive-cell method using
    ``energy / ccm.n_cells``. Measured on H2/STO-3G in a 6-bohr cube,
    nrep=(2,1,1): ``energy`` is twice the (1,1,1) value. The comment on this
    field previously said "per reference cell", which was wrong by a factor of
    ``N_c`` for every cluster above one cell.
    """

    converged: bool
    n_iter: int
    energy: float                  # TOTAL cyclic-cluster energy (Ha), not per cell
    energy_per_atom: float
    e_xc: float
    e_coulomb: float
    e_hf_exchange: float           # exact-exchange energy (hybrids; 0 for pure)
    functional: str
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    open_shell: bool
    # Per-spin open-shell data (None for RKS). ``mo_energies``/``mo_coeffs`` and
    # ``fock`` hold the alpha channel and ``density`` the *total* Pa+Pb; these
    # expose the beta channel and the per-spin densities that the open-shell
    # nuclear gradient and orbital-integrity checks need.
    density_alpha: np.ndarray | None = None
    density_beta: np.ndarray | None = None
    mo_energies_beta: np.ndarray | None = None
    mo_coeffs_beta: np.ndarray | None = None
    fock_beta: np.ndarray | None = None
    exchange_q0: str | None = None  # exchange-q=0 convention label (direct route)
    #: ``"active"`` iff a full-range exact-exchange arm is present, so the q=0
    #: convention materially selects the finite Hamiltonian; ``"inactive"`` for
    #: pure functionals and for screened-only (HSE-class) hybrids, whose erfc
    #: kernel has a finite zero mode and carries no seam (the aiccm2026dev-b
    #: binding contract's eta = 0). Mirrors ``CCMGDFResult.exchange_q0_applicability``.
    exchange_q0_applicability: str | None = None
    #: Identity of the route/operator that actually produced this energy
    #: (IID 344): e.g. ``"ccm-fourcenter-direct-union12-rks"`` or
    #: ``"ccm-neutral-direct-rks"``. Stamped by every public producer so a
    #: record consumer can tell WHICH backend a number came from without
    #: out-of-band knowledge; ``""`` only on results predating the field.
    backend: str = ""
    #: The RSGDF high-``|G|`` tail cutoff (Ha) actually applied when building
    #: this route's neutral cderi, or ``None`` for base-mesh-only (GitLab IID
    #: 307). Recorded because the tail is what makes the direct and multi-k
    #: GDF controls the *same* Hamiltonian: they disagreed by -4.99e-01
    #: Ha/cell on MgO/STO-3G at ``nrep=(1,1,1)`` while one silently tailed and
    #: the other could not. Resolved by
    #: :func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_tail_ke_cutoff`. A
    #: caller-supplied ``cderi`` carries its own tail, which this route cannot
    #: observe, so the field stays ``None`` there.
    rsgdf_tail_ke_cutoff: float | None = None
    guess_selection: object = None

    @property
    def parity_held(self) -> bool:
        """Whether the executing driver held this result for external parity.

        Derived from the ``+PARITY_HELD`` marker on :attr:`backend` -- the
        hold mechanism's canonical carrier
        (:func:`vibeqc.pbc_gdf._gdf_backend_with_parity_hold`) -- so the
        structured flag can never drift from the string (IID 344). The
        four-center and neutral-direct KS cluster routes never touch the
        hold class, so ``False`` there is an affirmative verdict.
        """
        return "+PARITY_HELD" in str(self.backend or "")


def _ccm_jk_builder(ccm, method, four_center="direct", schwarz_threshold=0.0):
    cxx_method = _ccm_scalable_cxx_method(method, four_center)
    return _make_ccm_jk_builder(ccm, cxx_method, schwarz_threshold)


def _xc_grid(ccm, grid_options):
    from vibeqc import GridOptions, build_grid
    return build_grid(ccm.supercell, grid_options or GridOptions())


def _four_center_external_xc_context(
    ccm,
    functional,
    spin,
    *,
    grid_options,
    becke_image_radius_bohr,
    xc_lattice_options,
    who,
):
    """Prepare the paper-defined cyclic atom grid for an external provider.

    The reference cell here is the *whole real CCM cluster*, not the primitive
    unit cell used by the neutral real-Gamma control.  Repeating its density
    matrix over the cluster lattice periodizes every cluster AO (Janetzko et
    al. 2008, Eq. 30), while :func:`build_periodic_becke_grid` owns points on
    the real atoms and normalizes their partition over real plus virtual image
    atoms (Eqs. 27-32).  ``PeriodicExternalXC`` also uses a difference-closed
    lattice domain, so every active image-pair derivative reaches the Gamma
    potential.

    Capability/profile validation and the pure-functional gate happen before
    the four-centre integral builder is constructed.  Libxc returns ``None``
    and continues through the unchanged molecular-supercell path below.
    """
    from vibeqc import Functional

    func = Functional(functional, spin)
    if not bool(getattr(func, "is_external", False)):
        return None
    if (
        abs(float(getattr(func, "hf_exchange_fraction", 0.0))) > 1.0e-15
        or bool(getattr(func, "is_range_separated", False))
        or bool(getattr(func, "is_double_hybrid", False))
    ):
        raise NotImplementedError(
            f"{who}: the literal four-centre WSSC external-XC route currently "
            "supports pure full-grid functionals only. External hybrids need "
            "a validated exact-exchange convention on the WSSC operator."
        )
    from vibeqc.pbc_bipole_common import reject_bipole_ecp_options

    try:
        reject_bipole_ecp_options(
            object(),
            driver=f"{who} external XC",
            basis=ccm.basis,
            system=ccm.unit_system,
        )
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{who}: ECP-bearing bases are not implemented with external "
            "XC; use an all-electron basis"
        ) from exc
    from vibeqc.periodic_external_xc import PeriodicExternalXC

    return PeriodicExternalXC.prepare(
        ccm.basis,
        ccm.cluster_system,
        func,
        grid_options=grid_options,
        image_radius_bohr=float(becke_image_radius_bohr),
        lattice_options=xc_lattice_options,
    )


def _external_rks_loop(
    ccm,
    S,
    h,
    e_nn,
    jk,
    xc_context,
    functional,
    *,
    max_iter,
    conv_tol,
    conv_tol_grad,
    lindep_tol,
):
    """Restricted SCF loop with an injected full-grid periodic XC builder."""
    from .scf import (
        _commutator_exit_ok,
        _diis_extrapolate,
        _orthonormaliser,
        _require_retained_occ,
    )

    n_elec = int(ccm.supercell.n_electrons())
    if n_elec % 2:
        raise ValueError("closed-shell driver needs an even-electron cluster.")
    n_occ = n_elec // 2
    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(
        X, n_occ, who="run_ccm_rks", lindep_tol=lindep_tol
    )

    def diagonalise(F):
        eps, transformed = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ transformed

    def density(C):
        occupied = C[:, :n_occ]
        return 2.0 * (occupied @ occupied.T)

    eps, C = diagonalise(h)
    D = density(C)
    fock_history = []
    error_history = []
    e_last = 0.0
    e_tot = e_xc = e_coul = 0.0
    converged = False
    F = np.asarray(h, dtype=float)
    diis_dim = 8
    for iteration in range(1, int(max_iter) + 1):
        J = np.asarray(jk.build_J(D), dtype=float)
        e_xc, V_xc = xc_context.build_gamma(D)
        F = np.asarray(h) + J + np.asarray(V_xc)
        F = 0.5 * (F + F.T)
        e_coul = 0.5 * float(np.sum(D * J))
        e_tot = (
            float(np.sum(D * h)) + e_coul + float(e_xc) + float(e_nn)
        )
        error = X.T @ (F @ D @ S - S @ D @ F) @ X
        if iteration > 1 and _commutator_exit_ok(
            e_tot - e_last, error, conv_tol, conv_tol_grad
        ):
            converged = True
            break
        if iteration == int(max_iter):
            break
        if len(fock_history) == diis_dim:
            fock_history.pop(0)
            error_history.pop(0)
        fock_history.append(F)
        error_history.append(error)
        trial_fock = (
            _diis_extrapolate(fock_history, error_history)
            if len(fock_history) >= 2
            else F
        )
        eps, C = diagonalise(trial_fock)
        D = density(C)
        e_last = e_tot
    else:  # pragma: no cover - max_iter is validated by the public wrapper
        iteration = 0

    # DIIS diagonalises an extrapolated trial Fock, whereas ``F`` above is the
    # accepted physical F[D] stored on the result.  Canonicalise that final
    # matrix so the reported eigenvalues/orbitals obey F C = S C eps rather
    # than describing the preceding DIIS trial.
    eps, C = diagonalise(F)
    if converged and int(ccm.unit_system.charge) == 0 and e_tot > 0.0:
        raise ValueError(
            "closed-shell neutral CCM KS SCF converged to a positive total "
            f"energy ({e_tot:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )
    return CCMKSResult(
        converged=converged,
        n_iter=iteration,
        energy=float(e_tot),
        energy_per_atom=float(e_tot) / ccm.n_atoms,
        e_xc=float(e_xc),
        e_coulomb=float(e_coul),
        e_hf_exchange=0.0,
        functional=functional,
        mo_energies=np.asarray(eps),
        mo_coeffs=np.asarray(C),
        density=np.asarray(D),
        fock=np.asarray(F),
        overlap=np.asarray(S),
        open_shell=False,
    )


def _external_uks_loop(
    ccm,
    S,
    h,
    e_nn,
    jk,
    xc_context,
    functional,
    *,
    max_iter,
    conv_tol,
    conv_tol_grad,
    lindep_tol,
):
    """Spin-polarized SCF loop with cyclic full-grid external XC."""
    from .scf import (
        _commutator_exit_ok,
        _diis_extrapolate,
        _orthonormaliser,
        _require_retained_occ,
    )

    n_elec = int(ccm.supercell.n_electrons())
    multiplicity = int(ccm.supercell.multiplicity)
    n_unpaired = multiplicity - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"cluster electron count {n_elec} and multiplicity "
            f"{multiplicity} are incompatible."
        )
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha
    X = _orthonormaliser(S, lindep_tol)
    _require_retained_occ(
        X, max(n_alpha, n_beta), who="run_ccm_uks", lindep_tol=lindep_tol
    )

    def diagonalise(F):
        eps, transformed = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ transformed

    def density(C, n_occ):
        occupied = C[:, :n_occ]
        return occupied @ occupied.T

    eps_alpha, C_alpha = diagonalise(h)
    eps_beta, C_beta = diagonalise(h)
    D_alpha = density(C_alpha, n_alpha)
    D_beta = density(C_beta, n_beta)
    fock_alpha_history = []
    fock_beta_history = []
    error_history = []
    e_last = 0.0
    e_tot = e_xc = e_coul = 0.0
    converged = False
    F_alpha = F_beta = np.asarray(h, dtype=float)
    diis_dim = 8
    for iteration in range(1, int(max_iter) + 1):
        D_total = D_alpha + D_beta
        J = np.asarray(jk.build_J(D_total), dtype=float)
        e_xc, V_alpha, V_beta = xc_context.build_gamma_uks(
            D_alpha, D_beta
        )
        F_alpha = np.asarray(h) + J + np.asarray(V_alpha)
        F_beta = np.asarray(h) + J + np.asarray(V_beta)
        F_alpha = 0.5 * (F_alpha + F_alpha.T)
        F_beta = 0.5 * (F_beta + F_beta.T)
        e_coul = 0.5 * float(np.sum(D_total * J))
        e_tot = (
            float(np.sum(D_total * h))
            + e_coul
            + float(e_xc)
            + float(e_nn)
        )
        error_alpha = X.T @ (
            F_alpha @ D_alpha @ S - S @ D_alpha @ F_alpha
        ) @ X
        error_beta = X.T @ (
            F_beta @ D_beta @ S - S @ D_beta @ F_beta
        ) @ X
        error = np.stack([error_alpha, error_beta])
        if iteration > 1 and _commutator_exit_ok(
            e_tot - e_last, error, conv_tol, conv_tol_grad
        ):
            converged = True
            break
        if iteration == int(max_iter):
            break
        if len(fock_alpha_history) == diis_dim:
            fock_alpha_history.pop(0)
            fock_beta_history.pop(0)
            error_history.pop(0)
        fock_alpha_history.append(F_alpha)
        fock_beta_history.append(F_beta)
        error_history.append(error)
        if len(fock_alpha_history) >= 2:
            trial_alpha = _diis_extrapolate(
                fock_alpha_history, error_history
            )
            trial_beta = _diis_extrapolate(fock_beta_history, error_history)
        else:
            trial_alpha, trial_beta = F_alpha, F_beta
        eps_alpha, C_alpha = diagonalise(trial_alpha)
        eps_beta, C_beta = diagonalise(trial_beta)
        D_alpha = density(C_alpha, n_alpha)
        D_beta = density(C_beta, n_beta)
        e_last = e_tot
    else:  # pragma: no cover - max_iter is validated by the public wrapper
        iteration = 0

    # As in the restricted loop, the last DIIS trial is not necessarily the
    # physical Fock accepted above. Canonicalise both stored spin operators so
    # each returned MO channel is an eigenbasis of its corresponding Fock.
    eps_alpha, C_alpha = diagonalise(F_alpha)
    eps_beta, C_beta = diagonalise(F_beta)
    return CCMKSResult(
        converged=converged,
        n_iter=iteration,
        energy=float(e_tot),
        energy_per_atom=float(e_tot) / ccm.n_atoms,
        e_xc=float(e_xc),
        e_coulomb=float(e_coul),
        e_hf_exchange=0.0,
        functional=functional,
        mo_energies=np.asarray(eps_alpha),
        mo_coeffs=np.asarray(C_alpha),
        density=np.asarray(D_alpha + D_beta),
        fock=np.asarray(F_alpha),
        overlap=np.asarray(S),
        open_shell=True,
        density_alpha=np.asarray(D_alpha),
        density_beta=np.asarray(D_beta),
        mo_energies_beta=np.asarray(eps_beta),
        mo_coeffs_beta=np.asarray(C_beta),
        fock_beta=np.asarray(F_beta),
    )


def run_ccm_rks(ccm, functional="pbe", *, initial_guess: object = "AUTO", method="union12", four_center="direct",
                schwarz_threshold=0.0, max_iter=128, conv_tol=1e-8,
                conv_tol_grad=1e-6, lindep_tol=1e-7, grid_options=None,
                becke_image_radius_bohr=10.0,
                xc_lattice_options=None) -> CCMKSResult:
    """Closed-shell RKS-CCM on ``ccm`` (a :class:`CCMSystem`).

    ``functional`` is any libxc name vibe-qc accepts (``"pbe"``, ``"pbe0"``,
    ``"b3lyp"``, ...), or a registered pure full-grid external functional.
    External hybrids remain gated. ``method`` selects the CCM four-center
    (``"union12"`` / ``"aiccm2026dev-a"``). ``four_center`` selects the J/K build:
    ``"direct"`` (default, integral-direct, O(nbf**2) memory -- scales to real 3-D
    cells) or ``"full"``/``"dense"`` (the dense O(nbf**4) effective-tensor
    comparison reference); see :func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`.

    ``lindep_tol`` is a **screening** threshold, forwarded to the C++ SCF's
    ``linear_dep_threshold``: overlap eigenvectors with eigenvalue <= the
    threshold are projected out of the canonical orthogonalisation subspace
    instead of refusing the run (the C++ driver raises only if the retained
    subspace cannot hold the occupied orbitals).

    For a full-grid external functional, ``grid_options`` controls the
    reference-cluster atom grid, ``becke_image_radius_bohr`` controls the
    real-plus-virtual-atom partition, and ``xc_lattice_options`` controls the
    translated cluster-AO reach. The latter defaults to a 25-bohr cutoff and
    is expanded internally to a difference-closed potential domain.

    Returns a :class:`CCMKSResult`.

    Validated: isolated-cell limit reproduces vibe-qc's molecular ``run_rks``.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rks"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_rks',
    )
    _warn_experimental()
    _reject_double_hybrid(functional, who="run_ccm_rks")
    external_xc = _four_center_external_xc_context(
        ccm,
        functional,
        1,
        grid_options=grid_options,
        becke_image_radius_bohr=becke_image_radius_bohr,
        xc_lattice_options=xc_lattice_options,
        who="run_ccm_rks",
    )

    S = ccm_overlap(ccm)
    h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)
    _reject_open_shell_cluster(ccm, who="run_ccm_rks", sibling="run_ccm_uks")
    n_elec = ccm.supercell.n_electrons()

    if external_xc is not None:
        if int(max_iter) < 1:
            raise ValueError("run_ccm_rks: max_iter must be >= 1")
        res = _external_rks_loop(
            ccm,
            np.asarray(S, float),
            np.asarray(h, float),
            float(e_nn),
            _ccm_jk_builder(ccm, method, four_center, schwarz_threshold),
            external_xc,
            functional,
            max_iter=max_iter,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            lindep_tol=lindep_tol,
        )
        res.backend = (
            "ccm-fourcenter-"
            f"{'direct' if four_center == 'direct' else 'dense'}-{method}-rks"
        )
        return _with_ccm_guess(res, guess_selection)

    from vibeqc import RKSOptions, run_rks_scf_with_jk

    opts = RKSOptions()
    opts.functional = functional
    opts.max_iter = max_iter
    opts.conv_tol_energy = conv_tol
    # conv_tol gates the energy criterion only; conv_tol_grad is the
    # independent DIIS-commutator residual bound (default 1e-6).
    opts.conv_tol_grad = conv_tol_grad
    opts.linear_dep_threshold = lindep_tol
    opts.initial_guess = guess_selection.requested
    res = run_rks_scf_with_jk(
        ccm.basis, n_elec, np.asarray(S, float), np.asarray(h, float),
        float(e_nn), _ccm_jk_builder(ccm, method, four_center, schwarz_threshold),
        _xc_grid(ccm, grid_options), opts, np.empty((0, 0)), molecule=ccm.supercell, guess_selection=guess_selection)

    e_tot = float(res.energy)
    if (bool(res.converged) and int(ccm.unit_system.charge) == 0
            and e_tot > 0.0):
        raise ValueError(
            "closed-shell neutral CCM KS SCF converged to a positive total "
            f"energy ({e_tot:.6f} Ha): unphysical for a neutral cell, and "
            "the signature of a vacuum-padded or gauge-inconsistent regime "
            "(IID 291). Refusing to report this as a successful result."
        )
    return _with_ccm_guess(CCMKSResult(
        converged=bool(res.converged), n_iter=int(res.n_iter), energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms, e_xc=float(res.e_xc),
        e_coulomb=float(res.e_coulomb), e_hf_exchange=float(res.e_hf_exchange),
        functional=functional, mo_energies=np.asarray(res.mo_energies),
        mo_coeffs=np.asarray(res.mo_coeffs), density=np.asarray(res.density),
        fock=np.asarray(res.fock), overlap=np.asarray(S), open_shell=False,
        backend=(
            "ccm-fourcenter-"
            f"{'direct' if four_center == 'direct' else 'dense'}-{method}-rks"
        )), guess_selection)


def run_ccm_uks(ccm, functional="pbe", *, initial_guess: object = "AUTO", method="union12", four_center="direct",
                schwarz_threshold=0.0, max_iter=128, conv_tol=1e-8,
                conv_tol_grad=1e-6, lindep_tol=1e-7, grid_options=None,
                becke_image_radius_bohr=10.0,
                xc_lattice_options=None) -> CCMKSResult:
    """Open-shell UKS-CCM on ``ccm`` (a :class:`CCMSystem`).

    Spin counts come from the cluster supercell's charge + multiplicity (as in
    :func:`vibeqc.periodic.ccm.uhf.run_ccm_uhf`). ``four_center`` selects the J/K
    build (``"direct"`` default, integral-direct / O(nbf**2); ``"full"``/``"dense"``
    the O(nbf**4) reference) -- see :func:`run_ccm_rks`. Returns a
    :class:`CCMKSResult` (``density`` is the total density Pa+Pb). Pure
    full-grid external functionals use the same cyclic reference-cluster grid
    and image controls as :func:`run_ccm_rks`; external hybrids remain gated.
    Validated: isolated-cell limit reproduces vibe-qc's molecular ``run_uks``;
    reduces to RKS for a closed shell.
    """
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_uks"
    )
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_uks',
    )
    _warn_experimental()
    _reject_double_hybrid(functional, who="run_ccm_uks")
    external_xc = _four_center_external_xc_context(
        ccm,
        functional,
        2,
        grid_options=grid_options,
        becke_image_radius_bohr=becke_image_radius_bohr,
        xc_lattice_options=xc_lattice_options,
        who="run_ccm_uks",
    )

    S = ccm_overlap(ccm)
    h, _, _ = ccm_hcore(ccm)
    e_nn = ccm_nuclear_repulsion(ccm)
    n_elec = ccm.supercell.n_electrons()
    mult = int(ccm.supercell.multiplicity)
    n_unpaired = mult - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"cluster electron count {n_elec} and multiplicity {mult} are incompatible.")
    n_alpha = (n_elec + n_unpaired) // 2
    n_beta = n_elec - n_alpha

    if external_xc is not None:
        if int(max_iter) < 1:
            raise ValueError("run_ccm_uks: max_iter must be >= 1")
        res = _external_uks_loop(
            ccm,
            np.asarray(S, float),
            np.asarray(h, float),
            float(e_nn),
            _ccm_jk_builder(ccm, method, four_center, schwarz_threshold),
            external_xc,
            functional,
            max_iter=max_iter,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            lindep_tol=lindep_tol,
        )
        res.backend = (
            "ccm-fourcenter-"
            f"{'direct' if four_center == 'direct' else 'dense'}-{method}-uks"
        )
        return _with_ccm_guess(res, guess_selection)

    from vibeqc import UKSOptions, run_uks_scf_with_jk

    opts = UKSOptions()
    opts.functional = functional
    opts.max_iter = max_iter
    opts.conv_tol_energy = conv_tol
    # conv_tol gates the energy criterion only; conv_tol_grad is the
    # independent DIIS-commutator residual bound (default 1e-6).
    opts.conv_tol_grad = conv_tol_grad
    opts.linear_dep_threshold = lindep_tol
    opts.initial_guess = guess_selection.requested
    res = run_uks_scf_with_jk(
        ccm.basis, n_alpha, n_beta, np.asarray(S, float), np.asarray(h, float),
        float(e_nn), _ccm_jk_builder(ccm, method, four_center, schwarz_threshold),
        _xc_grid(ccm, grid_options), opts, np.empty((0, 0)), np.empty((0, 0)), molecule=ccm.supercell, guess_selection=guess_selection)

    e_tot = float(res.energy)
    # UKSResult exposes per-spin MOs/densities; report a MOs + total density.
    Da = np.asarray(res.density_alpha)
    Db = np.asarray(res.density_beta)
    return _with_ccm_guess(CCMKSResult(
        converged=bool(res.converged), n_iter=int(res.n_iter), energy=e_tot,
        energy_per_atom=e_tot / ccm.n_atoms, e_xc=float(res.e_xc),
        e_coulomb=float(res.e_coulomb), e_hf_exchange=float(res.e_hf_exchange),
        functional=functional, mo_energies=np.asarray(res.mo_energies_alpha),
        mo_coeffs=np.asarray(res.mo_coeffs_alpha), density=Da + Db,
        fock=np.asarray(res.fock_alpha), overlap=np.asarray(S), open_shell=True,
        density_alpha=Da, density_beta=Db,
        mo_energies_beta=np.asarray(res.mo_energies_beta),
        mo_coeffs_beta=np.asarray(res.mo_coeffs_beta),
        fock_beta=np.asarray(res.fock_beta),
        backend=(
            "ccm-fourcenter-"
            f"{'direct' if four_center == 'direct' else 'dense'}-{method}-uks"
        )), guess_selection)
