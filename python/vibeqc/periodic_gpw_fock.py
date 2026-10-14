"""Evaluated Gamma GPW RHF states with the existing libint exchange source.

GPW collocation/Poisson/projection follows Lippert, Hutter and Parrinello,
Mol. Phys. 92, 477 (1997), doi:10.1080/00268979709482119, section 2,
equations (14)-(18). This adapter preserves the existing finite grid, image
sum and zero mode; it does not establish complete-basis or bulk parity.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from . import _vibeqc_core as core
from .bipole_domain import LATTICE_FIELDS, OptionSnapshot
from .fock_state import EnergyComponents, FockBuild, SourceProvenance
from .solvers._scf import FockDiagonalizer
from .solvers.operators import OperatorFrame, _hermitian, _snapshot
from .smearing import apply_smearing


def _digest(array):
    a = np.asarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


class _GpwRHFBuilder:
    """Guard live integral/grid captures, own the one-electron matrices.

    The large collocation table remains provider-owned and is hashed, not
    copied into a build. Dense FockBuild storage is not a peak-memory bound.
    No generic adapter is introduced into MANTA or symmetry kernels.
    """
    def __init__(self, basis, system, lattice_options, gpw, kinetic, nuclear,
                 frame, e_nuclear, *, nuclear_convention, smearing_alpha,
                 dispersion=0.0, dispersion_model=None):
        self.basis, self.system, self.lattice_options = basis, system, lattice_options
        self.gpw, self.frame = gpw, frame
        self.kinetic, self.nuclear = _snapshot(kinetic), _snapshot(nuclear)
        self.hcore = _snapshot(self.kinetic + self.nuclear)
        self.e_nuclear, self.dispersion = float(e_nuclear), float(dispersion)
        self.nuclear_convention, self.smearing_alpha = nuclear_convention, smearing_alpha
        self.dispersion_model = dispersion_model
        self.source = SourceProvenance.capture("gpw-grid+libint2", **self._details())
        self.generation = 0

    def _details(self):
        from .periodic_gapw_j import _ewald_v_ne_gamma_fingerprint
        grid = self.gpw.grid
        return dict(
            schema="vibeqc.gamma-gpw-rhf-source.v1",
            geometry_basis=_ewald_v_ne_gamma_fingerprint(self.basis, self.system).hex(),
            provider_geometry_basis=_ewald_v_ne_gamma_fingerprint(self.gpw.basis, self.system).hex(),
            ordering=self.frame.ordering, frame_id=self.frame.identity,
            grid_lattice=np.asarray(grid.lattice_bohr).tolist(), grid_shape=list(grid.shape),
            grid_cutoff=grid.cutoff_ha, collocation=_digest(self.gpw.chi()),
            mpi_aware=bool(self.gpw._mpi_aware),
            kinetic=_digest(self.kinetic), nuclear=_digest(self.nuclear),
            nuclear_convention=self.nuclear_convention, smearing_alpha=self.smearing_alpha,
            nuclear_repulsion=self.e_nuclear, dispersion=self.dispersion,
            dispersion_model=self.dispersion_model,
            lattice_options=OptionSnapshot.capture(self.lattice_options, LATTICE_FIELDS).values,
            hartree_kernel="fft-coulomb", hartree_zero_mode="omitted-jellium",
            exchange_kernel="coulomb", exchange_q0="molecular-limit", exchange_fraction=1.0,
            xc="none", symmetry_reduction="unreduced", mesh="full-gamma", weights=[1.0])

    def __call__(self, density):
        if SourceProvenance.capture(self.source.provider, **self._details()) != self.source:
            raise ValueError("GPW RHF numerical source changed; rebuild its provider and metric frame")
        d = _snapshot(density)
        _hermitian(d, "GPW RHF density")
        if np.iscomplexobj(d):
            raise NotImplementedError("Gamma GPW RHF requires a real total density")
        if d.shape != (self.frame.n, self.frame.n):
            raise ValueError("GPW RHF density and frame dimensions differ")
        j = self.gpw.build_J(d)
        k = np.asarray(core.build_jk_gamma_molecular_limit(
            self.basis, self.system, self.lattice_options, d).K)
        f = self.hcore + j - .5 * k
        f = .5 * (f + f.T)
        components = (
            ("kinetic", float(np.einsum("ij,ij->", d, self.kinetic))),
            ("nuclear_attraction", float(np.einsum("ij,ij->", d, self.nuclear))),
            ("coulomb", .5 * float(np.einsum("ij,ij->", d, j))),
            ("exchange", -.25 * float(np.einsum("ij,ij->", d, k))))
        state = FockBuild.from_dense(
            method="RHF", generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=sum(v for _, v in components),
                nuclear=self.e_nuclear, components=components,
                corrections=(("dispersion", self.dispersion),)),
            densities=(d,), focks=(f,), frames=(self.frame,), spins=("restricted",),
            kpoints=((0., 0., 0.),), weights=(1.,))
        self.generation += 1
        return state


def _density_orbitals(density, frame, fock):
    """Natural orbitals for a mixed/restart density, not another SCF step.

    Resolve degenerate occupations with the physical Fock so a converged
    integer-occupation state retains meaningful occupied/virtual energies.
    """
    sx = frame.overlap @ frame.X
    occ, u = np.linalg.eigh(sx.T @ density @ sx)
    occ, c = occ[::-1].copy(), frame.X @ u[:, ::-1]
    start = 0
    while start < len(occ):
        end = start + 1
        while end < len(occ) and abs(occ[end] - occ[start]) < 1e-13:
            end += 1
        block = c[:, start:end]
        _, rotation = np.linalg.eigh(block.T @ fock @ block)
        c[:, start:end] = block @ rotation
        start = end
    energies = np.diag(c.T @ fock @ c).copy()
    return c, energies, occ


def _run_gpw_rhf_state(*, basis, system, gpw, lattice_options, kinetic, nuclear,
                       overlap, X, e_nuclear, density, n_occ, patom, max_iter,
                       conv_tol_energy, conv_tol_density, damping, use_diis,
                       diis_subspace_size, diis_start_iter, use_davidson,
                       davidson_min_dim, smearing_opts, smearing_temperature,
                       nuclear_convention, smearing_alpha, dispersion,
                       dispersion_functional, plog, guess_selection):
    from .periodic_gapw_j import GpwEnergyBreakdown, GpwScfResult, _make_diis, _compute_d3_correction
    from .periodic_gapw_j import _ewald_v_ne_gamma_fingerprint
    frame = OperatorFrame(
        basis_id=_ewald_v_ne_gamma_fingerprint(basis, system).hex(), n=basis.nbasis,
        ordering="libint-standard-pure-m-ascending-cartesian-lx-ly-descending",
        image_gauge="gamma-periodic-ao", overlap=overlap, X=X)
    S, X = frame.overlap, frame.X
    e_disp = _compute_d3_correction(system, dispersion, dispersion_functional)
    if nuclear_convention == "smeared_erfc" and smearing_alpha is None:
        from .periodic_gapw_smearing import default_smearing_alpha_from_grid
        smearing_alpha = default_smearing_alpha_from_grid(gpw.grid)
    builder = _GpwRHFBuilder(
        basis, system, lattice_options, gpw, kinetic, nuclear, frame, e_nuclear,
        nuclear_convention=nuclear_convention, smearing_alpha=smearing_alpha,
        dispersion=e_disp, dispersion_model=(dispersion, dispersion_functional))
    diagonalise = FockDiagonalizer(
        X, S, SimpleNamespace(use_davidson=use_davidson,
            davidson_min_dim=davidson_min_dim, davidson=core.DavidsonOptions()),
        n_required=frame.n if smearing_temperature > 0 else n_occ,
        frame=frame, real_only=True)

    def occupied(operator):
        c, eps = diagonalise(operator)
        sm = apply_smearing([eps], weights=[1.0], n_electrons_per_cell=2.0*n_occ,
                            n_occ_each=n_occ, smearing=smearing_opts)
        occ = np.asarray(sm.occupations_per_k[0])
        d = (c * occ[None, :]) @ c.T
        return .5 * (d + d.T), c, occ, sm

    D = np.asarray(density).copy()
    if patom:
        seed = builder(D)
        # PATOM retains its full-HF integer-occupation purification.
        c, _ = diagonalise(seed.channels[0].operator)
        D = 2 * c[:, :n_occ] @ c[:, :n_occ].T
    diis = _make_diis(use_diis, diis_subspace_size)
    trace, history = [], []
    E_prev, converged, n_iter = 0.0, False, 0
    terminal = None
    C = occ = sm = None
    for it in range(1, max_iter + 1):
        physical = builder(D)
        channel = physical.channels[0]
        F, E = channel.operator.matrix, physical.energy.total
        operator = channel.operator
        if diis is not None and it >= diis_start_iter:
            err = X.T @ (F @ D @ S - S @ D @ F) @ X
            F = diis.extrapolate(F, err)
            operator = physical.solver_operator(0, F, modification="pulay-diis")
        D_new, C, occ, sm = occupied(operator)
        if damping > 0:
            D_new = (1.0 - damping) * D_new + damping * D
        dE, dD = E - E_prev, float(np.linalg.norm(D_new - D))
        n_iter = it
        trace.append(dict(iter=it, energy=E, delta_e=dE, grad_norm=dD, e_xc=0.0))
        history.append(dict(iteration=it, generation=physical.generation,
                            build_id=physical.identity, density_id=physical.density_id))
        plog.iteration(it, energy=E, dE=dE, grad=dD,
                       diis=diis.subspace_size if diis is not None else 0)
        D, E_prev = D_new, E
        candidate = it > 1 and abs(dE) < conv_tol_energy and dD < conv_tol_density
        if candidate or it == max_iter:
            terminal = builder(D)
            next_density, _, _, _ = occupied(terminal.channels[0].operator)
            closure = float(np.linalg.norm(next_density - D))
            converged = (candidate and abs(terminal.energy.total-E) < conv_tol_energy
                         and terminal.commutator_norm < conv_tol_density
                         and closure < conv_tol_density)
            if converged or it == max_iter:
                break
            terminal = None
    if terminal is None:
        terminal = builder(D)
        next_density, _, _, _ = occupied(terminal.channels[0].operator)
        closure = float(np.linalg.norm(next_density - D))
    channel = terminal.channels[0]
    F = channel.operator.matrix
    if C is None or not np.allclose((C * occ[None, :]) @ C.T, D, atol=1e-13, rtol=0):
        C, eps, occ = _density_orbitals(D, frame, F)
    else:
        eps = np.diag(C.T @ F @ C).copy()
    # Entropy belongs to the evaluated density, including a damped/capped state.
    fraction = np.clip(occ / 2.0, 0.0, 1.0)
    mask = (fraction > 0) & (fraction < 1)
    entropy = (-2.0 * float(np.sum(fraction[mask]*np.log(fraction[mask])
        + (1-fraction[mask])*np.log1p(-fraction[mask]))) if smearing_temperature > 0 else 0.0)
    parts = dict(terminal.energy.components)
    breakdown = GpwEnergyBreakdown(
        e_kinetic=parts["kinetic"], e_nuclear_attraction=parts["nuclear_attraction"],
        e_hartree=parts["coulomb"], e_hf_exchange=parts["exchange"],
        e_nuclear_repulsion=e_nuclear, e_total=terminal.energy.total,
        grid=gpw.grid, e_dispersion=e_disp)
    history.append(dict(iteration=n_iter, generation=terminal.generation,
                        build_id=terminal.identity, density_id=terminal.density_id, terminal=True))
    if trace:
        trace[-1] = dict(trace[-1], energy=terminal.energy.total,
                         delta_e=terminal.energy.total-trace[-1]["energy"], grad_norm=closure)
    return GpwScfResult(
        energy=terminal.energy.total, breakdown=breakdown, density=channel.density.copy(),
        mo_coeffs=C, mo_energies=eps, converged=converged, n_iter=n_iter, grid=gpw.grid,
        scf_trace=tuple(trace), e_dispersion=e_disp,
        smearing_temperature=smearing_temperature, smearing_entropy=entropy,
        fermi_level=float(sm.mu) if sm is not None else 0.0, occupations=tuple(occ.tolist()),
        fock=F.copy(), overlap=S.copy(), fock_build=terminal,
        next_density=_snapshot(next_density), density_fixed_point_residual=closure,
        evaluated_build_history=tuple(history),
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
        guess_selection=guess_selection)
