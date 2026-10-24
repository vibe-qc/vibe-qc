"""Evaluated Gamma GPW HF/KS states with the existing libint exchange source.

GPW collocation/Poisson/projection follows Lippert, Hutter and Parrinello,
Mol. Phys. 92, 477 (1997), doi:10.1080/00268979709482119, section 2,
equations (14)-(18). This adapter preserves the existing finite grid, image
sum and zero mode; it does not establish complete-basis or bulk parity.
"""
from __future__ import annotations

from dataclasses import dataclass
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
        # The self-adjoint exchange: -k/2 is the derivative of -tr(d k)/4
        # also when the images overlap (#534).
        k = np.asarray(core.build_jk_gamma_molecular_limit(
            self.basis, self.system, self.lattice_options, d,
            adjoint_symmetric=True).K)
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


class _GpwUHFBuilder:
    """One coupled physical state: J[D_alpha + D_beta], K[D_spin].

    Preserve the established libint per-spin convention K(2 D_spin)/2.
    Source capture shares the restricted provider guard; no integral payload
    or prepared-source terminal cache is added here.
    """
    method = "UHF"

    def __init__(self, hf_builder):
        self.hf_builder, self.frame = hf_builder, hf_builder.frame
        self.source = SourceProvenance.capture("gpw-grid+libint2",
            schema="vibeqc.gamma-gpw-uhf-source.v1",
            hf_source=hf_builder.source.as_dict(),
            density_convention="ordered-alpha-beta-per-spin",
            coulomb_density="alpha+beta", exchange_density="2*spin",
            exchange_matrix_factor=0.5, exchange_energy_factor=-0.5)
        self.generation = 0

    def __call__(self, density_alpha, density_beta):
        from .periodic_gapw_open_shell import _k_per_spin
        hf = self.hf_builder
        if SourceProvenance.capture(hf.source.provider, **hf._details()) != hf.source:
            raise ValueError("GPW UHF numerical source changed; rebuild its provider and metric frame")
        densities = tuple(_snapshot(d) for d in (density_alpha, density_beta))
        for d in densities:
            _hermitian(d, "GPW UHF spin density")
            if np.iscomplexobj(d):
                raise NotImplementedError("Gamma GPW UHF requires real spin densities")
            if d.shape != (self.frame.n, self.frame.n):
                raise ValueError("GPW UHF density and frame dimensions differ")
        total = densities[0] + densities[1]
        j = hf.gpw.build_J(total)
        exchange = tuple(_k_per_spin(hf.basis, hf.system, d, hf.lattice_options,
                                     adjoint_symmetric=True) for d in densities)
        focks = tuple(hf.hcore + j - k for k in exchange)
        components = (
            ("kinetic", float(np.einsum("ij,ij->", total, hf.kinetic))),
            ("nuclear_attraction", float(np.einsum("ij,ij->", total, hf.nuclear))),
            ("coulomb", .5 * float(np.einsum("ij,ij->", total, j))),
            ("exchange", -.5 * sum(float(np.einsum("ij,ij->", d, k))
                                  for d, k in zip(densities, exchange))))
        state = FockBuild.from_dense(
            method=self.method, generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=sum(v for _, v in components),
                nuclear=hf.e_nuclear, components=components),
            densities=densities, focks=tuple(.5*(f+f.T) for f in focks),
            frames=(self.frame, self.frame), spins=("alpha", "beta"),
            kpoints=((0., 0., 0.),)*2, weights=(1., 1.))
        self.generation += 1
        return state


class _GpwDudarev:
    """Shared restricted Hubbard source and the existing AO projector."""
    def _init_dftu(self, basis, sites):
        self.dftu_sites = tuple(sites or ())
        self.dftu_groups = {}
        if self.dftu_sites:
            from .dft_plus_u import ao_group_indices
            groups = ao_group_indices(basis)
            for site in self.dftu_sites:
                key = (site.atom_index, site.l)
                if key not in groups:
                    raise ValueError(f"GPW HubbardSite{key} has no AOs in the basis")
                self.dftu_groups[key] = tuple(groups[key])
        self.dftu_source = self._dftu_source()

    def _dftu_source(self):
        if not self.dftu_sites:
            return None
        return SourceProvenance.capture("gpw-dudarev",
            schema="vibeqc.gamma-gpw-dudarev-source.v1",
            frame_id=self.frame.identity, projector="S(D/2)S-active-AO-block",
            spin_energy_factor=2.0, potential="S-V-per-spin-S",
            sites=[dict(atom_index=site.atom_index, l=site.l,
                        u_eff_hartree=site.U_eff_hartree,
                        ao_indices=self.dftu_groups.get((site.atom_index, site.l), ()))
                   for site in self.dftu_sites])

    def _dftu_correction(self, density):
        # Dudarev et al., PRB 57, 1505 (1998), section II, eqs. (5)-(6),
        # doi:10.1103/PhysRevB.57.1505. Retain the existing nonorthogonal
        # AO projector: n_sigma = [S(D/2)S]_site and dE_U/dD = S V_sigma S.
        # E_U is explicit, separate from electronic XC and the half trace.
        from .dft_plus_u import (
            compute_occupation_matrices, compute_dudarev_energy, _v_ao_per_spin)
        s, p = self.frame.overlap, .5 * density
        n = compute_occupation_matrices(self.dftu_sites, p, s, self.dftu_groups)
        energy = 2.0 * compute_dudarev_energy(self.dftu_sites, n)
        potential = s @ _v_ao_per_spin(self.dftu_sites, p, s, self.dftu_groups) @ s
        return energy, potential


class _GpwRHFUBuilder(_GpwDudarev):
    """Add the explicit Dudarev correction to a physical GPW RHF state.

    The underlying full-HF builder also supplies the unchanged PATOM seed.
    Its half-trace energy never includes the Hubbard potential.
    """
    def __init__(self, hf_builder, dft_plus_u_sites):
        self.hf_builder, self.frame = hf_builder, hf_builder.frame
        self._init_dftu(hf_builder.basis, dft_plus_u_sites)
        self.source = SourceProvenance.capture("gpw-grid+libint2+dudarev", **self._details())
        self.generation = 0

    def _details(self):
        source = self._dftu_source()
        return dict(schema="vibeqc.gamma-gpw-rhf-dudarev-source.v1",
                    hf_source=self.hf_builder.source.as_dict(),
                    dft_plus_u=None if source is None else source.as_dict())

    def __call__(self, density):
        if SourceProvenance.capture(self.source.provider, **self._details()) != self.source:
            raise ValueError("GPW RHF+U numerical source changed; rebuild its provider and metric frame")
        bare = self.hf_builder(density)
        channel = bare.channels[0]
        e_u, v_u = self._dftu_correction(channel.density)
        f = channel.operator.matrix + v_u
        state = FockBuild.from_dense(
            method="RHF", generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=bare.energy.electronic,
                nuclear=bare.energy.nuclear, components=bare.energy.components,
                corrections=bare.energy.corrections + (("dft_plus_u", e_u),)),
            densities=(channel.density,), focks=(.5*(f+f.T),), frames=(self.frame,),
            spins=("restricted",), kpoints=((0., 0., 0.),), weights=(1.,))
        self.generation += 1
        return state


class _GpwUHFUBuilder(_GpwDudarev):
    """Explicit spin-summed Dudarev energy and two physical UHF potentials."""
    method = "UHF"
    kind = "fock"
    provider = "gpw-grid+libint2+dudarev"
    schema = "vibeqc.gamma-gpw-uhf-dudarev-source.v1"

    def __init__(self, uhf_builder, dft_plus_u_sites):
        self.uhf_builder, self.frame = uhf_builder, uhf_builder.frame
        self._init_dftu(uhf_builder.hf_builder.basis, dft_plus_u_sites)
        self.source = SourceProvenance.capture(self.provider, **self._details())
        self.generation = 0

    def _dftu_source(self):
        return SourceProvenance.capture("gpw-dudarev",
            schema="vibeqc.gamma-gpw-spin-dudarev-source.v1",
            frame_id=self.frame.identity, projector="S-D_spin-S-active-AO-block",
            spin_energy_factor=1.0, potential="S-V-per-spin-S",
            sites=[dict(atom_index=site.atom_index, l=site.l,
                        u_eff_hartree=site.U_eff_hartree,
                        ao_indices=self.dftu_groups.get((site.atom_index, site.l), ()))
                   for site in self.dftu_sites])

    def _details(self):
        return dict(schema=self.schema,
                    hf_source=self.uhf_builder.source.as_dict(),
                    dft_plus_u=self._dftu_source().as_dict())

    def _dftu_correction(self, density):
        # Dudarev et al., PRB 57, 1505 (1998), section II, eqs. (5)-(6),
        # doi:10.1103/PhysRevB.57.1505. Preserve the existing nonorthogonal
        # n_spin = [S D_spin S]_site projector and its S V_spin S derivative.
        from .dft_plus_u import (
            compute_occupation_matrices, compute_dudarev_energy, _v_ao_per_spin)
        s = self.frame.overlap
        n = compute_occupation_matrices(self.dftu_sites, density, s, self.dftu_groups)
        energy = compute_dudarev_energy(self.dftu_sites, n)
        potential = s @ _v_ao_per_spin(self.dftu_sites, density, s, self.dftu_groups) @ s
        return energy, potential

    def __call__(self, density_alpha, density_beta):
        if SourceProvenance.capture(self.source.provider, **self._details()) != self.source:
            raise ValueError(f"GPW {self.method}+U numerical source changed; rebuild its provider and metric frame")
        bare = self.uhf_builder(density_alpha, density_beta)
        corrections = tuple(self._dftu_correction(c.density) for c in bare.channels)
        focks = tuple(c.operator.matrix + v for c, (_, v) in zip(bare.channels, corrections))
        state = FockBuild.from_dense(
            method=self.method, kind=self.kind, generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=bare.energy.electronic,
                nuclear=bare.energy.nuclear, components=bare.energy.components,
                corrections=bare.energy.corrections + (("dft_plus_u", sum(e for e, _ in corrections)),)),
            densities=tuple(c.density for c in bare.channels),
            focks=tuple(.5*(f+f.T) for f in focks), frames=(self.frame, self.frame),
            spins=("alpha", "beta"), kpoints=((0., 0., 0.),)*2, weights=(1., 1.))
        self.generation += 1
        return state


class _GpwUKSUBuilder(_GpwUHFUBuilder):
    """Compose the same spin Dudarev correction with a joint UKS state.

    The parent builder owns XC and exchange. Hubbard energy remains an
    explicit additive correction, outside both the HF half trace and XC.
    """
    method = "UKS"
    kind = "ks"
    provider = "gpw-grid+libint2+libxc+dudarev"
    schema = "vibeqc.gamma-gpw-uks-dudarev-source.v1"


def _gpw_xc_source(hf_builder, functional, cache, *, polarised=False):
    from . import periodic_gapw_j as gj
    func = functional
    if cache is None or cache.chi is None:
        raise ValueError("GPW KS requires the prepared collocation cache")
    grid = hf_builder.gpw.grid
    return SourceProvenance.capture("libxc-gpw",
        name=func.name, kind=func.kind.name, components=func.components,
        spin=int(func.spin), hf_exchange_fraction=func.hf_exchange_fraction,
        cam_alpha=func.cam_alpha, cam_beta=func.cam_beta, rsh_omega=func.rsh_omega,
        collocation=_digest(cache.chi), reciprocal_vectors=_digest(cache.recip),
        meta_gga_policy=(dict(density_screen=gj._MGGA_DENSITY_SCREEN,
            vsigma_stiff_max=gj._MGGA_VSIGMA_STIFF_MAX,
            vtau_stiff_max=gj._MGGA_VTAU_STIFF_MAX,
            tau_floor="von-weizsaecker-with-constrained-chain-rule")
            if func.kind == core.XCKind.MGGA else None),
        grid_lattice=np.asarray(grid.lattice_bohr).tolist(), grid_shape=list(grid.shape),
        density_convention=("ordered-alpha-beta-per-spin-periodic-gamma-ao" if polarised
                            else "total-density-periodic-gamma-ao"),
        gradient="spectral-density", kinetic_energy_density="half-spectral-ao-gradient-product")


class _GpwRKSBuilder(_GpwDudarev):
    """Joint GPW/libxc energy and generalized KS potential at one density.

    E = tr(D H) + tr(D J)/2 - alpha tr(D K)/4 + E_xc[D].
    XC is an explicit energy, not half a trace of its potential. See Lehtola
    et al., SoftwareX 7 (2018), sections 1-2, equations 2/4/7,
    doi:10.1016/j.softx.2017.11.002. Grid derivatives and meta-GGA tau
    projection retain the existing GPW discretization.
    """
    def __init__(self, hf_builder, functional, collocation_cache, dft_plus_u_sites=()):
        from .periodic_screened_exchange import reject_unscreened_range_separated
        if int(functional.spin) != 1:
            raise ValueError("Gamma GPW RKS requires an unpolarized functional")
        if functional.is_external:
            raise NotImplementedError("Gamma GPW RKS evaluated states require a libxc functional")
        reject_unscreened_range_separated(functional, where="Gamma GPW RKS FockBuild")
        self.hf_builder, self.functional = hf_builder, functional
        self.cache = collocation_cache
        self.frame = hf_builder.frame
        self._init_dftu(hf_builder.basis, dft_plus_u_sites)
        self.xc_source = self._xc_source()
        details = dict(
            schema="vibeqc.gamma-gpw-rks-source.v1",
            hf_source=hf_builder.source.as_dict(), xc=self.xc_source.as_dict())
        if self.dftu_source is not None:
            details["dft_plus_u"] = self.dftu_source.as_dict()
        self.source = SourceProvenance.capture("gpw-grid+libint2+libxc", **details)
        self.generation = 0

    @property
    def alpha(self):
        return float(self.functional.hf_exchange_fraction)

    def _xc_source(self):
        return _gpw_xc_source(self.hf_builder, self.functional, self.cache)

    def __call__(self, density):
        from .periodic_gapw_j import (
            collocate_density_on_grid, _evaluate_xc_on_grid,
            _project_vxc_to_ao, _project_vtau_to_ao)
        hf = self.hf_builder
        if (SourceProvenance.capture(hf.source.provider, **hf._details()) != hf.source
                or self._xc_source() != self.xc_source
                or self._dftu_source() != self.dftu_source):
            raise ValueError("GPW RKS numerical source changed; rebuild its provider and metric frame")
        d = _snapshot(density)
        _hermitian(d, "GPW RKS density")
        if np.iscomplexobj(d):
            raise NotImplementedError("Gamma GPW RKS requires a real total density")
        if d.shape != (self.frame.n, self.frame.n):
            raise ValueError("GPW RKS density and frame dimensions differ")
        j = hf.gpw.build_J(d)
        k = (np.asarray(core.build_jk_gamma_molecular_limit(
            hf.basis, hf.system, hf.lattice_options, d,
            adjoint_symmetric=True).K) if self.alpha > 0 else None)
        grid = hf.gpw.grid
        rho = collocate_density_on_grid(hf.basis, d, grid, cache=self.cache)
        e_xc, v_rho, v_sigma, grad_rho, v_tau = _evaluate_xc_on_grid(
            rho, grid, self.functional, basis=hf.basis, density_matrix=d, cache=self.cache)
        v_xc = _project_vxc_to_ao(hf.basis, v_rho, grid,
            v_sigma_grid=v_sigma, grad_rho=grad_rho, cache=self.cache)
        if v_tau is not None:
            v_xc = v_xc + _project_vtau_to_ao(hf.basis, v_tau, grid, cache=self.cache)
        f = hf.hcore + j + v_xc
        if k is not None:
            f = f - .5 * self.alpha * k
        corrections = (("dispersion", hf.dispersion),)
        if self.dftu_sites:
            e_u, v_u = self._dftu_correction(d)
            f = f + v_u
            corrections += (("dft_plus_u", e_u),)
        f = .5 * (f + f.T)
        components = (
            ("kinetic", float(np.einsum("ij,ij->", d, hf.kinetic))),
            ("nuclear_attraction", float(np.einsum("ij,ij->", d, hf.nuclear))),
            ("coulomb", .5 * float(np.einsum("ij,ij->", d, j))),
            ("exchange", 0.0 if k is None else -.25 * self.alpha * float(np.einsum("ij,ij->", d, k))),
            ("xc", float(e_xc)))
        state = FockBuild.from_dense(
            method="RKS", kind="ks", generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=sum(v for _, v in components),
                nuclear=hf.e_nuclear, components=components,
                corrections=corrections),
            densities=(d,), focks=(f,), frames=(self.frame,), spins=("restricted",),
            kpoints=((0., 0., 0.),), weights=(1.,))
        self.generation += 1
        return state


class _GpwUKSBuilder:
    """Joint spin XC energy/potentials, shared J and optional per-spin K.

    Lehtola et al., SoftwareX 7 (2018), sections 1-2, equations 2/4/7,
    doi:10.1016/j.softx.2017.11.002. Preserve the existing polarised GPW
    gradient cross term and per-spin meta-GGA kinetic-energy projection.
    XC energy is explicit; no half trace of the XC potential enters E.
    """
    method = "UKS"

    def __init__(self, hf_builder, functional, collocation_cache):
        from .periodic_screened_exchange import reject_unscreened_range_separated
        if int(functional.spin) != 2:
            raise ValueError("Gamma GPW UKS requires a polarized functional")
        if functional.is_external:
            raise NotImplementedError("Gamma GPW UKS evaluated states require a libxc functional")
        reject_unscreened_range_separated(functional, where="Gamma GPW UKS FockBuild")
        self.hf_builder, self.frame = hf_builder, hf_builder.frame
        self.functional, self.cache = functional, collocation_cache
        self.xc_source = self._xc_source()
        self.source = SourceProvenance.capture("gpw-grid+libint2+libxc",
            schema="vibeqc.gamma-gpw-uks-source.v1",
            hf_source=hf_builder.source.as_dict(), xc=self.xc_source.as_dict(),
            coulomb_density="alpha+beta", exchange_density="2*spin",
            exchange_matrix_factor=0.5, exchange_energy_factor=-0.5)
        self.generation = 0

    def _xc_source(self):
        return _gpw_xc_source(self.hf_builder, self.functional, self.cache, polarised=True)

    def __call__(self, density_alpha, density_beta):
        from .periodic_gapw_j import collocate_density_on_grid, _project_vtau_to_ao
        from .periodic_gapw_open_shell import (
            _k_per_spin, _evaluate_xc_polarised_on_grid, _project_vxc_polarised_to_ao)
        hf = self.hf_builder
        if (SourceProvenance.capture(hf.source.provider, **hf._details()) != hf.source
                or self._xc_source() != self.xc_source):
            raise ValueError("GPW UKS numerical source changed; rebuild its provider and metric frame")
        densities = tuple(_snapshot(d) for d in (density_alpha, density_beta))
        for d in densities:
            _hermitian(d, "GPW UKS spin density")
            if np.iscomplexobj(d):
                raise NotImplementedError("Gamma GPW UKS requires real spin densities")
            if d.shape != (self.frame.n, self.frame.n):
                raise ValueError("GPW UKS density and frame dimensions differ")
        total = densities[0] + densities[1]
        j = hf.gpw.build_J(total)
        alpha = float(self.functional.hf_exchange_fraction)
        exchange = (tuple(_k_per_spin(hf.basis, hf.system, d, hf.lattice_options,
                                      adjoint_symmetric=True) for d in densities)
                    if alpha > 0 else (None, None))
        grid = hf.gpw.grid
        rho = tuple(collocate_density_on_grid(hf.basis, d, grid, cache=self.cache)
                    for d in densities)
        e_xc, vr_a, vr_b, vs_aa, vs_ab, vs_bb, grad_a, grad_b, vt_a, vt_b = (
            _evaluate_xc_polarised_on_grid(*rho, grid, self.functional, basis=hf.basis,
                density_matrix_alpha=densities[0], density_matrix_beta=densities[1], cache=self.cache))
        focks = []
        for spin, vr, vs, grad, other_grad, vt, k in (
                ("alpha", vr_a, vs_aa, grad_a, grad_b, vt_a, exchange[0]),
                ("beta", vr_b, vs_bb, grad_b, grad_a, vt_b, exchange[1])):
            vxc = _project_vxc_polarised_to_ao(hf.basis, spin, vr, grid,
                v_sigma_self=vs, v_sigma_cross=vs_ab, grad_rho_self=grad,
                grad_rho_other=other_grad, cache=self.cache)
            if vt is not None:
                vxc = vxc + _project_vtau_to_ao(hf.basis, vt, grid, cache=self.cache)
            f = hf.hcore + j + vxc
            if k is not None:
                f = f - alpha*k
            focks.append(.5*(f+f.T))
        components = (
            ("kinetic", float(np.einsum("ij,ij->", total, hf.kinetic))),
            ("nuclear_attraction", float(np.einsum("ij,ij->", total, hf.nuclear))),
            ("coulomb", .5 * float(np.einsum("ij,ij->", total, j))),
            ("exchange", -.5*alpha*sum(float(np.einsum("ij,ij->", d, k))
                                      for d, k in zip(densities, exchange) if k is not None)),
            ("xc", float(e_xc)))
        state = FockBuild.from_dense(
            method=self.method, kind="ks", generation=self.generation, source=self.source,
            energy=EnergyComponents(electronic=sum(v for _, v in components),
                nuclear=hf.e_nuclear, components=components),
            densities=densities, focks=focks, frames=(self.frame, self.frame),
            spins=("alpha", "beta"), kpoints=((0., 0., 0.),)*2, weights=(1., 1.))
        self.generation += 1
        return state



class _GpwROHFBuilder(_GpwUHFBuilder):
    """Physical spin build; the Roothaan caller owns orbital constraints."""
    method = "ROHF"


class _GpwROKSBuilder(_GpwUKSBuilder):
    """Joint physical XC/spin build; effective coupling stays method-owned."""
    method = "ROKS"


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


@dataclass(frozen=True)
class RestrictedOpenDensityOrbitals:
    """Density factors, explicitly separate from canonical orbital reporting.

    ``roothaan`` has one shared spatial basis and the actual accepted spin
    occupations. ``spin-natural`` factors an initial or damped mixed state;
    its spin bases may differ and do not assert a restricted determinant.
    Energies are physical spin-Fock expectations, not canonical eigenvalues.
    """
    representation: str
    coefficients: tuple
    occupations: tuple
    energies: tuple
    build_id: str
    density_id: str
    frame_id: str


def _restricted_open_density_orbitals(build, accepted):
    frame = build.channels[0].operator.frame
    if accepted is None:
        orbitals = tuple(_density_orbitals(c.density, frame, c.operator.matrix)
                         for c in build.channels)
        coefficients = tuple(_snapshot(c) for c, _, _ in orbitals)
        occupations = tuple(_snapshot(o) for _, _, o in orbitals)
        representation = "spin-natural"
    else:
        c, occ_alpha, occ_beta = accepted
        shared = _snapshot(c)
        coefficients = (shared, shared)
        occupations = (_snapshot(occ_alpha), _snapshot(occ_beta))
        representation = "roothaan"
    energies = []
    for channel, c, occ in zip(build.channels, coefficients, occupations):
        reconstructed = (c * occ) @ c.conj().T
        if not np.allclose(reconstructed, channel.density, atol=1e-9, rtol=1e-10):
            raise ValueError("Restricted-open orbital factors do not reproduce the evaluated density")
        energies.append(_snapshot(np.diag(c.conj().T @ channel.operator.matrix @ c).real))
    return RestrictedOpenDensityOrbitals(
        representation, coefficients, occupations, tuple(energies),
        build.identity, build.density_id, frame.identity)


def _run_gpw_restricted_state(*, basis, system, gpw, lattice_options, kinetic, nuclear,
                       overlap, X, e_nuclear, density, n_occ, patom, max_iter,
                       conv_tol_energy, conv_tol_density, damping, use_diis,
                       diis_subspace_size, diis_start_iter, use_davidson,
                       davidson_min_dim, smearing_opts, smearing_temperature,
                       nuclear_convention, smearing_alpha, dispersion,
                       dispersion_functional, plog, guess_selection,
                       functional=None, collocation_cache=None, dft_plus_u_sites=()):
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
    hf_builder = _GpwRHFBuilder(
        basis, system, lattice_options, gpw, kinetic, nuclear, frame, e_nuclear,
        nuclear_convention=nuclear_convention, smearing_alpha=smearing_alpha,
        dispersion=e_disp, dispersion_model=(dispersion, dispersion_functional))
    if functional is None:
        builder = _GpwRHFUBuilder(hf_builder, dft_plus_u_sites) if dft_plus_u_sites else hf_builder
    else:
        builder = _GpwRKSBuilder(hf_builder, functional, collocation_cache, dft_plus_u_sites)
    kind = "fock" if functional is None else "ks"
    diagonalise = FockDiagonalizer(
        X, S, SimpleNamespace(use_davidson=use_davidson,
            davidson_min_dim=davidson_min_dim, davidson=core.DavidsonOptions()),
        n_required=frame.n if smearing_temperature > 0 else n_occ,
        frame=frame, real_only=True, kind=kind)

    def occupied(operator):
        c, eps = diagonalise(operator)
        sm = apply_smearing([eps], weights=[1.0], n_electrons_per_cell=2.0*n_occ,
                            n_occ_each=n_occ, smearing=smearing_opts)
        occ = np.asarray(sm.occupations_per_k[0])
        d = (c * occ[None, :]) @ c.T
        return .5 * (d + d.T), c, occ, sm

    D = np.asarray(density).copy()
    if patom:
        seed = hf_builder(D)
        # PATOM retains its full-HF integer-occupation purification.
        seed_diagonalise = (diagonalise if functional is None else FockDiagonalizer(
            X, S, SimpleNamespace(use_davidson=False), n_required=n_occ,
            frame=frame, real_only=True))
        c, _ = seed_diagonalise(seed.channels[0].operator)
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
        trace.append(dict(iter=it, energy=E, delta_e=dE, grad_norm=dD,
                          e_xc=dict(physical.energy.components).get("xc", 0.0)))
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
    e_dft_plus_u = dict(terminal.energy.corrections).get("dft_plus_u", 0.0)
    breakdown = GpwEnergyBreakdown(
        e_kinetic=parts["kinetic"], e_nuclear_attraction=parts["nuclear_attraction"],
        e_hartree=parts["coulomb"], e_hf_exchange=parts["exchange"],
        e_nuclear_repulsion=e_nuclear, e_total=terminal.energy.total,
        grid=gpw.grid, e_dispersion=e_disp, e_xc=parts.get("xc", 0.0),
        e_dft_plus_u=e_dft_plus_u,
        functional=None if functional is None else functional.name)
    history.append(dict(iteration=n_iter, generation=terminal.generation,
                        build_id=terminal.identity, density_id=terminal.density_id, terminal=True))
    if trace:
        previous_energy = trace[-2]["energy"] if len(trace) > 1 else 0.0
        trace[-1] = dict(trace[-1], energy=terminal.energy.total,
                         delta_e=terminal.energy.total-previous_energy, grad_norm=closure,
                         e_xc=parts.get("xc", 0.0))
    return GpwScfResult(
        energy=terminal.energy.total, breakdown=breakdown, density=channel.density.copy(),
        mo_coeffs=C, mo_energies=eps, converged=converged, n_iter=n_iter, grid=gpw.grid,
        scf_trace=tuple(trace), e_dispersion=e_disp, e_dft_plus_u=e_dft_plus_u,
        smearing_temperature=smearing_temperature, smearing_entropy=entropy,
        fermi_level=float(sm.mu) if sm is not None else 0.0, occupations=tuple(occ.tolist()),
        fock=F.copy(), overlap=S.copy(), fock_build=terminal,
        next_density=_snapshot(next_density), density_fixed_point_residual=closure,
        evaluated_build_history=tuple(history),
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        restart_kpoints=np.zeros((1, 3)), restart_weights=np.ones(1),
        guess_selection=guess_selection)


# Retain the private RHF entry spelling for existing method callers.
_run_gpw_rhf_state = _run_gpw_restricted_state


def _run_gpw_uhf_state(*, basis, system, gpw, lattice_options, kinetic, nuclear,
                       overlap, X, e_nuclear, densities, occupations, patom,
                       max_iter, conv_tol_energy, conv_tol_density, damping,
                       use_diis, diis_subspace_size, diis_start_iter,
                       nuclear_convention, smearing_alpha, plog, dft_plus_u_sites=(),
                       functional=None, collocation_cache=None):
    """Shared evaluated UHF/UKS loop with method-owned physical builders."""
    from .periodic_gapw_j import GpwEnergyBreakdown, _make_diis, _ewald_v_ne_gamma_fingerprint
    from .periodic_gapw_open_shell import GpwUhfScfResult, GpwUksScfResult
    frame = OperatorFrame(
        basis_id=_ewald_v_ne_gamma_fingerprint(basis, system).hex(), n=basis.nbasis,
        ordering="libint-standard-pure-m-ascending-cartesian-lx-ly-descending",
        image_gauge="gamma-periodic-ao", overlap=overlap, X=X)
    S, X = frame.overlap, frame.X
    if nuclear_convention == "smeared_erfc" and smearing_alpha is None:
        from .periodic_gapw_smearing import default_smearing_alpha_from_grid
        smearing_alpha = default_smearing_alpha_from_grid(gpw.grid)
    def make_builder(metric_frame):
        return _GpwUHFBuilder(_GpwRHFBuilder(
            basis, system, lattice_options, gpw, kinetic, nuclear, metric_frame, e_nuclear,
            nuclear_convention=nuclear_convention, smearing_alpha=smearing_alpha))

    builder = make_builder(frame)
    if functional is not None:
        builder = _GpwUKSBuilder(builder.hf_builder, functional, collocation_cache)
        if dft_plus_u_sites:
            builder = _GpwUKSUBuilder(builder, dft_plus_u_sites)
    elif dft_plus_u_sites:
        builder = _GpwUHFUBuilder(builder, dft_plus_u_sites)
    kind = "fock" if functional is None else "ks"
    solvers = tuple(FockDiagonalizer(X, S, None, frame=frame, real_only=True,
                    kind=kind, spin=spin, n_required=n) for spin, n in zip(("alpha", "beta"), occupations))

    def occupied(operators):
        result = []
        for solve, operator, n in zip(solvers, operators, occupations):
            c, _ = solve(operator)
            result.append(c[:, :n] @ c[:, :n].T)
        return tuple(result)

    def distance(left, right):
        return float(np.sqrt(sum(np.linalg.norm(a-b)**2 for a, b in zip(left, right))))

    D = tuple(np.asarray(d).copy() for d in densities)
    if patom:
        from .guess import _canonical_orthogonalizer
        # Preserve patom_spin_density_step's retained-space threshold. Its
        # seed frame can differ from this route's full SCF inverse overlap.
        seed_x = _canonical_orthogonalizer(S, 1e-7)
        if max(occupations) > seed_x.shape[1]:
            raise ValueError("PATOM: insufficient independent orbitals for spin population")
        seed_frame = OperatorFrame(basis_id=frame.basis_id, n=frame.n,
            ordering=frame.ordering, image_gauge=frame.image_gauge, overlap=S, X=seed_x)
        seed = make_builder(seed_frame)(*D)
        seed_densities = []
        for channel, n in zip(seed.channels, occupations):
            solve = FockDiagonalizer(seed_frame.X, seed_frame.overlap, None,
                frame=seed_frame, real_only=True, spin=channel.operator.metadata.spin, n_required=n)
            c, _ = solve(channel.operator)
            seed_densities.append(c[:, :n] @ c[:, :n].T)
        D = tuple(seed_densities)
    diis = _make_diis(use_diis, diis_subspace_size)
    trace, history = [], []
    E_prev, converged, n_iter = 0.0, False, 0
    terminal = None
    for it in range(1, max_iter + 1):
        physical = builder(*D)
        operators = tuple(c.operator for c in physical.channels)
        E = physical.energy.total
        if diis is not None and it >= diis_start_iter:
            focks = tuple(o.matrix for o in operators)
            errors = tuple(X.T @ (f @ d @ S - S @ d @ f) @ X for f, d in zip(focks, D))
            mixed = diis.extrapolate_spin_coupled(*focks, *errors)
            operators = tuple(physical.solver_operator(i, f, modification="pulay-diis")
                              for i, f in enumerate(mixed))
        D_new = occupied(operators)
        if damping > 0:
            D_new = tuple((1-damping)*new + damping*old for new, old in zip(D_new, D))
        dE, dD = E-E_prev, distance(D_new, D)
        n_iter = it
        trace.append(dict(iter=it, energy=E, delta_e=dE, grad_norm=dD,
                          e_xc=dict(physical.energy.components).get("xc", 0.0),
                          e_dft_plus_u=dict(physical.energy.corrections).get("dft_plus_u", 0.0)))
        history.append(dict(iteration=it, generation=physical.generation,
                            build_id=physical.identity, density_id=physical.density_id))
        plog.iteration(it, energy=E, dE=dE, grad=dD)
        D, E_prev = D_new, E
        candidate = it > 1 and abs(dE) < conv_tol_energy and dD < conv_tol_density
        if candidate or it == max_iter:
            terminal = builder(*D)
            next_density = occupied(tuple(c.operator for c in terminal.channels))
            closure = distance(next_density, D)
            converged = (candidate and abs(terminal.energy.total-E) < conv_tol_energy
                         and terminal.commutator_norm < conv_tol_density
                         and closure < conv_tol_density)
            if converged or it == max_iter:
                break
            terminal = None
    if terminal is None:
        terminal = builder(*D)
        next_density = occupied(tuple(c.operator for c in terminal.channels))
        closure = distance(next_density, D)
    orbitals = tuple(_density_orbitals(c.density, frame, c.operator.matrix)
                     for c in terminal.channels)
    parts = dict(terminal.energy.components)
    e_dft_plus_u = dict(terminal.energy.corrections).get("dft_plus_u", 0.0)
    breakdown = GpwEnergyBreakdown(
        e_kinetic=parts["kinetic"], e_nuclear_attraction=parts["nuclear_attraction"],
        e_hartree=parts["coulomb"], e_hf_exchange=parts["exchange"],
        e_nuclear_repulsion=e_nuclear, e_total=terminal.energy.total, grid=gpw.grid,
        e_dft_plus_u=e_dft_plus_u, e_xc=parts.get("xc", 0.0),
        functional=None if functional is None else functional.name)
    history.append(dict(iteration=n_iter, generation=terminal.generation,
                        build_id=terminal.identity, density_id=terminal.density_id, terminal=True))
    if trace:
        previous_energy = trace[-2]["energy"] if len(trace) > 1 else 0.0
        trace[-1] = dict(trace[-1], energy=terminal.energy.total,
                         delta_e=terminal.energy.total-previous_energy, grad_norm=closure,
                         e_dft_plus_u=e_dft_plus_u, e_xc=parts.get("xc", 0.0))
    result_type = GpwUhfScfResult if functional is None else GpwUksScfResult
    return result_type(
        energy=terminal.energy.total, breakdown=breakdown,
        density_alpha=terminal.channels[0].density.copy(), density_beta=terminal.channels[1].density.copy(),
        mo_coeffs_alpha=orbitals[0][0], mo_coeffs_beta=orbitals[1][0],
        mo_energies_alpha=orbitals[0][1], mo_energies_beta=orbitals[1][1],
        occupations_alpha=tuple(orbitals[0][2]), occupations_beta=tuple(orbitals[1][2]),
        n_alpha=occupations[0], n_beta=occupations[1], converged=converged,
        e_dft_plus_u=e_dft_plus_u, dft_plus_u_sites=tuple(dft_plus_u_sites),
        n_iter=n_iter, grid=gpw.grid, scf_trace=tuple(trace),
        fock_alpha=terminal.channels[0].operator.matrix.copy(),
        fock_beta=terminal.channels[1].operator.matrix.copy(), overlap=S.copy(), fock_build=terminal,
        next_density=tuple(_snapshot(d) for d in next_density),
        density_fixed_point_residual=closure, evaluated_build_history=tuple(history))
