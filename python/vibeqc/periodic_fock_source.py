"""Resolved source identity shared by dense Gamma Ewald HF producers.

The v1 schema name retains its original UHF spelling for identity stability.
It describes the numerical integrals, independently of the HF spin convention.
This receipt does not admit additional symmetry or source approximations.
"""
import os
import numpy as np
from .periodic.exchange_convention import MOLECULAR_LIMIT


def _gamma_source_details(basis, system, lattice_options, *, omega, grid_shape,
                          origin, spacing_bohr):
    """Capture the resolved source, including AO ordering and shell images.

    Reuse the existing lattice option snapshot vocabulary without changing
    BIPOLE's domain policy. Gamma molecular-limit exchange remains distinct.
    """
    from .bipole_domain import LATTICE_FIELDS, OptionSnapshot
    from ._vibeqc_core import EwaldOptions, GridOptions
    from .bipole_domain import GRID_FIELDS
    hartree_backend = os.environ.get("VIBEQC_J_EWALD3D_BACKEND", "analytic_ft").lower()
    nuclear_backend = os.environ.get("VIBEQC_VNE_EWALD3D_BACKEND", "analytic_ft").lower()
    ewald = EwaldOptions()
    ewald.real_cutoff_bohr = lattice_options.nuclear_cutoff_bohr
    shells = [dict(atom_index=int(s.atom_index), l=int(s.l), pure=bool(s.pure),
                   exponents=list(s.exponents), coefficients=list(s.coefficients),
                   origin=list(s.origin)) for s in basis.shells()]
    return dict(
        schema="vibeqc.gamma-ewald-uhf-source.v1", shells=shells,
        ordering="libint-standard-pure-m-ascending-cartesian-lx-ly-descending",
        lattice=np.asarray(system.lattice).tolist(), dimensionality=int(system.dim),
        nuclei=[dict(Z=int(a.Z), position=list(a.xyz)) for a in system.unit_cell_molecule().atoms],
        lattice_options=OptionSnapshot.capture(lattice_options, LATTICE_FIELDS).values,
        hartree_backend=hartree_backend,
        hartree_ke_cutoff=(float(os.environ.get("VIBEQC_J_EWALD3D_KE", "200.0"))
                           if hartree_backend == "analytic_ft" else None),
        nuclear_backend=nuclear_backend,
        nuclear_ke_cutoff=(float(os.environ.get("VIBEQC_VNE_EWALD3D_KE", "200.0"))
                           if nuclear_backend != "grid" else None),
        nuclear_ewald=OptionSnapshot.capture(ewald, (
            "alpha", "real_cutoff_bohr", "recip_cutoff_bohr_inv", "tolerance")).values,
        nuclear_grid=(OptionSnapshot.capture(GridOptions(), GRID_FIELDS).values
                      if nuclear_backend == "grid" else None),
        nuclear_boundary="tin-foil" if system.dim == 3 else "declared-lattice-method",
        omega=float(omega), grid_shape=list(grid_shape),
        origin=None if origin is None else list(origin), spacing_bohr=float(spacing_bohr),
        hartree_kernel="coulomb", hartree_zero_mode="omitted-jellium",
        exchange_kernel="coulomb", exchange_q0=MOLECULAR_LIMIT,
        exchange_fraction=1.0, xc="none", symmetry_reduction="unreduced",
        mesh="full-gamma", weights=[1.0],
    )


def _slab_source_details(basis, system, lattice_options, *, ewald_options):
    """HF source with Parry's open-normal boundary and resolved quadrature.

    Grid-shaped dispatcher arguments do not select the analytic slab source.
    Keep the existing 3D receipt unchanged for its already migrated consumers.
    """
    import inspect
    from .bipole_domain import OptionSnapshot
    from .ewald_composed_slab import build_j_slab_ewald_2d_gamma_cache
    from .periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_gamma

    details = _gamma_source_details(
        basis, system, lattice_options, omega=ewald_options.alpha,
        grid_shape=(0, 0, 0), origin=None, spacing_bohr=0.0)

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
    return details
