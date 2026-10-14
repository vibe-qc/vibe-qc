"""Evaluated Gamma Ewald KS states over the existing libint/libxc sources.

The method owns the total-density factors and the joint XC energy/potential
evaluation. No integral dispatch or symmetry admission is selected here.
"""
import hashlib
import inspect

import numpy as np

from ._vibeqc_core import (
    CoulombMethod, bloch_sum, build_jk_gamma_molecular_limit, build_xc_periodic,
    build_xc_periodic_uks,
)
from .bipole_domain import GRID_FIELDS, OptionSnapshot
from .fock_state import EnergyComponents, FockBuild, SourceProvenance
from .periodic_fock_source import _gamma_source_details
from .periodic_screened_exchange import build_exchange_gamma
from .solvers.operators import _snapshot


def _array_receipt(array):
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _gamma_ks_source_details(basis, system, lattice_options, options, functional, exchange, grid,
                             density_set, *, omega, grid_shape, origin,
                             spacing_bohr, ewald_options=None, hf_seed=False):
    """Extend the HF receipt without changing existing HF source identities."""
    lat = lattice_options
    details = _gamma_source_details(
        basis, system, lat, omega=omega, grid_shape=grid_shape,
        origin=origin, spacing_bohr=spacing_bohr)
    details["schema"] = "vibeqc.gamma-ewald-ks-source.v1"
    if lat.coulomb_method == CoulombMethod.SLAB_EWALD_2D:
        from .ewald_composed_slab import build_j_slab_ewald_2d_gamma_cache
        from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_gamma

        # The slab source has an open normal boundary and a Parry G=0
        # profile. Ewald-3D environment/grid controls do not select it.
        def quadrature_defaults(fn):
            parameters = inspect.signature(fn).parameters
            return {key: parameters[key].default for key in (
                "recip_tol", "n_gz", "z_pad", "z_spacing", "gz_profile_max", "profile_tol")}

        details.update(
            hartree_backend="slab-erfc+analytic-ft-profile", hartree_ke_cutoff=None,
            nuclear_backend="slab-erfc+analytic-ft-profile", nuclear_ke_cutoff=None,
            nuclear_grid=None, nuclear_boundary="open-normal-parry",
            hartree_zero_mode="parry-z-profile", grid_shape=None, origin=None,
            spacing_bohr=None,
            hartree_quadrature=quadrature_defaults(build_j_slab_ewald_2d_gamma_cache),
            nuclear_quadrature=quadrature_defaults(compute_v_ne_slab_ewald_2d_gamma),
            nuclear_ewald=OptionSnapshot.capture(ewald_options, (
                "alpha", "real_cutoff_bohr", "recip_cutoff_bohr_inv", "tolerance")).values)
    if hf_seed:
        details["purpose"] = "full-hf-patom-seed"
        return details
    details.update(
        exchange_kernel="coefficient-folded-full-and-erfc",
        exchange_fraction=None,
        exchange_assembly=dict(c_full=exchange.c_full, c_sr=exchange.c_sr,
                               omega_screen=exchange.omega_screen),
        xc=dict(requested=str(options.functional), name=functional.name,
                kind=functional.kind.name, components=functional.components,
                spin=int(functional.spin), hf_exchange_fraction=functional.hf_exchange_fraction,
                cam_alpha=functional.cam_alpha, cam_beta=functional.cam_beta,
                rsh_omega=functional.rsh_omega,
                grid_options=OptionSnapshot.capture(options.grid, GRID_FIELDS).values,
                grid_points=_array_receipt(grid.points), grid_weights=_array_receipt(grid.weights),
                density_convention=("gamma-torus" if options.use_periodic_becke else "home-cell-only"),
                image_radius_bohr=float(options.becke_image_radius_bohr),
                cells=[list(cell.r_cart) for cell in density_set.cells]))
    return details


class _GammaRKSBuilder:
    """One physical state: F = H + J - K_HF/2 + V_xc at the same D.

    E_xc is supplied by the joint libxc evaluation, never inferred from
    half a trace of D V_xc. Solver acceleration remains outside this builder.
    """
    def __init__(self, basis, system, lattice_options, kinetic, nuclear, frame,
                 e_nuclear, j_build, functional, exchange, grid, density_set,
                 set_xc_density, source_details):
        self.basis, self.system, self.lattice_options = basis, system, lattice_options
        self.kinetic, self.nuclear = _snapshot(kinetic), _snapshot(nuclear)
        self.hcore = _snapshot(.5 * ((kinetic + nuclear) + (kinetic + nuclear).T))
        self.frame, self.e_nuclear, self.j_build = frame, e_nuclear, j_build
        self.functional, self.exchange, self.grid = functional, exchange, grid
        self.density_set, self.set_xc_density = density_set, set_xc_density
        self.source_details = source_details
        self.source = SourceProvenance.capture("libint2+periodic-ewald-j+libxc", **source_details())
        self.generation = 0

    def __call__(self, density, *, hf_seed=False):
        if SourceProvenance.capture(self.source.provider, **self.source_details()) != self.source:
            raise ValueError("Gamma RKS numerical source changed; rebuild its provider and metric frame")
        d = _snapshot(density)
        if np.iscomplexobj(d):
            raise NotImplementedError("Gamma Ewald RKS requires a real total density")
        j = self.j_build(d)
        if hf_seed:
            k = np.asarray(build_jk_gamma_molecular_limit(
                self.basis, self.system, self.lattice_options, d, 0.0).K)
            v_xc, e_xc = 0.0, 0.0
            source = SourceProvenance.capture(
                "libint2+periodic-ewald-j", **self.source_details(hf_seed=True))
        else:
            k = build_exchange_gamma(self.basis, self.system, self.lattice_options, d, self.exchange)
            self.set_xc_density(self.density_set, d)
            xc = build_xc_periodic(self.basis, self.system, self.grid, self.functional,
                                   self.density_set, self.lattice_options)
            v_xc = np.real(bloch_sum(xc.V_xc, np.zeros(3)))
            v_xc = .5 * (v_xc + v_xc.T)
            e_xc, source = float(xc.e_xc), self.source
        f_hf = j if k is None else j - .5 * k
        f = self.hcore + f_hf + v_xc
        f = .5 * (f + f.T)
        electronic = (e_xc + float(np.einsum("ij,ij->", d, self.hcore))
                      + .5 * float(np.einsum("ij,ij->", d, f_hf)))
        # Both supported Ewald boundaries include the nuclear/Hartree zero
        # mode consistently; neither uses a separate Madelung correction.
        if self.lattice_options.coulomb_method not in (
                CoulombMethod.EWALD_3D, CoulombMethod.SLAB_EWALD_2D):
            raise ValueError("Gamma RKS builder requires a resolved Ewald boundary")
        state = FockBuild.from_dense(
            method="RHF" if hf_seed else "RKS", kind="fock" if hf_seed else "ks",
            generation=self.generation, source=source,
            energy=EnergyComponents(electronic=electronic, nuclear=self.e_nuclear,
                components=(("kinetic", float(np.einsum("ij,ij->", d, self.kinetic))),
                            ("nuclear_attraction", float(np.einsum("ij,ij->", d, self.nuclear))),
                            ("coulomb", .5 * float(np.einsum("ij,ij->", d, j))),
                            ("exchange", 0.0 if k is None else -.25 * float(np.einsum("ij,ij->", d, k))),
                            ("xc", e_xc)), corrections=(("madelung", 0.0),)),
            densities=(d,), focks=(f,), frames=(self.frame,), spins=("restricted",),
            kpoints=((0., 0., 0.),), weights=(1.,))
        self.generation += 1
        return state


class _GammaUKSBuilder:
    """A joint spin state: J[Da+Db], K[2 Ds]/2 and one polarized XC call.

    E_xc belongs to the density pair, not to either spin independently.
    Completed states retain immutable values rather than this provider.
    """
    def __init__(self, basis, system, lattice_options, kinetic, nuclear, frame,
                 e_nuclear, j_build, functional, exchange, grid, density_sets,
                 set_xc_density, source_details):
        self.basis, self.system, self.lattice_options = basis, system, lattice_options
        self.kinetic, self.nuclear = _snapshot(kinetic), _snapshot(nuclear)
        self.hcore = _snapshot(.5 * ((kinetic + nuclear) + (kinetic + nuclear).T))
        self.frame, self.e_nuclear, self.j_build = frame, e_nuclear, j_build
        self.functional, self.exchange, self.grid = functional, exchange, grid
        self.density_sets, self.set_xc_density = density_sets, set_xc_density
        self.source_details = source_details
        self.source = SourceProvenance.capture("libint2+periodic-ewald-j+libxc", **source_details())
        self.generation = 0

    def __call__(self, density_alpha, density_beta, *, hf_seed=False):
        if SourceProvenance.capture(self.source.provider, **self.source_details()) != self.source:
            raise ValueError("Gamma UKS numerical source changed; rebuild its provider and metric frame")
        da, db = _snapshot(density_alpha), _snapshot(density_beta)
        if np.iscomplexobj(da) or np.iscomplexobj(db):
            raise NotImplementedError("Gamma Ewald UKS requires real spin densities")
        if self.lattice_options.coulomb_method not in (
                CoulombMethod.EWALD_3D, CoulombMethod.SLAB_EWALD_2D):
            raise ValueError("Gamma UKS builder requires a resolved Ewald boundary")
        dt = da + db
        j = self.j_build(dt)
        if hf_seed:
            ka, kb = (.5 * np.asarray(build_jk_gamma_molecular_limit(
                self.basis, self.system, self.lattice_options, 2. * d, 0.).K) for d in (da, db))
            va, vb, e_xc = 0., 0., 0.
            source = SourceProvenance.capture(
                "libint2+periodic-ewald-j", **self.source_details(hf_seed=True))
        else:
            if self.exchange.needs_exchange:
                ka, kb = (.5 * build_exchange_gamma(
                    self.basis, self.system, self.lattice_options, 2. * d, self.exchange)
                    for d in (da, db))
            else:
                ka, kb = None, None
            for template, d in zip(self.density_sets, (da, db)):
                self.set_xc_density(template, d)
            xc = build_xc_periodic_uks(self.basis, self.system, self.grid, self.functional,
                                       *self.density_sets, self.lattice_options)
            va, vb = (np.real(bloch_sum(v, np.zeros(3))) for v in (xc.V_alpha, xc.V_beta))
            va, vb = .5 * (va + va.T), .5 * (vb + vb.T)
            e_xc, source = float(xc.e_xc), self.source
        hf_a, hf_b = (j, j) if ka is None else (j - ka, j - kb)
        fa, fb = self.hcore + hf_a + va, self.hcore + hf_b + vb
        fa, fb = .5 * (fa + fa.T), .5 * (fb + fb.T)
        electronic = (e_xc + float(np.einsum("ij,ij->", dt, self.hcore))
                      + .5 * float(np.einsum("ij,ij->", da, hf_a))
                      + .5 * float(np.einsum("ij,ij->", db, hf_b)))
        exchange = (0. if ka is None else -.5 * float(np.einsum("ij,ij->", da, ka))
                    - .5 * float(np.einsum("ij,ij->", db, kb)))
        state = FockBuild.from_dense(
            method="UHF" if hf_seed else "UKS", kind="fock" if hf_seed else "ks",
            generation=self.generation, source=source,
            energy=EnergyComponents(electronic=electronic, nuclear=self.e_nuclear,
                components=(("kinetic", float(np.einsum("ij,ij->", dt, self.kinetic))),
                            ("nuclear_attraction", float(np.einsum("ij,ij->", dt, self.nuclear))),
                            ("coulomb", .5 * float(np.einsum("ij,ij->", dt, j))),
                            ("exchange", exchange), ("xc", e_xc)),
                corrections=(("madelung", 0.),)),
            densities=(da, db), focks=(fa, fb), frames=(self.frame, self.frame),
            spins=("alpha", "beta"), kpoints=((0., 0., 0.),) * 2, weights=(1., 1.))
        self.generation += 1
        return state
