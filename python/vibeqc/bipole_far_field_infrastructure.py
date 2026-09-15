"""Shared bipolar far-field infrastructure builder for BIPOLE SCF drivers.

Provides a low-level entry point for the dormant quartet-level bipolar
far-field prototype: per-pair spherical multipole moments, a geometric
classifier, and cached Coulomb (J) and Exchange (K) contraction data.
Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the source for the periodic
quartet expansion. The classifier and cached kernel implemented here
are prototypes for which that source gives no derivation.

Architecture
------------
The far-field pipeline has three phases:

**Phase 1 — Geometry setup** (once per geometry, before SCF loop):
  1. Compute global-origin Cartesian multipole moments via libint
     (``compute_multipole_moments_lattice``, C++).
  2. Shift moments to per-pair adjoined-Gaussian centres
     (``pair_center_moments``, C++ OpenMP).
  3. Convert Cartesian → spherical moments and build the spherical
     moment buffer (``build_spherical_moment_buffer``, C++ OpenMP).
  4. Precompute moment derivatives dM/dC for the gradient
     (``compute_moment_derivatives_for_buffer``, C++ OpenMP).
  5. Symmetry-reduce the buffer (``build_symmetry_reduced_moment_buffer``).
  6. Build the geometric penetration dispatch
     (``build_point_group_reduced_penetration_dispatch``, then
     unfolded via ``unfold_symmetry_reduced_dispatch``).
  7. Pre-compute interaction tensors (``build_quartet_tensor_cache``).
  8. Pre-compute the far-field Fock kernel
     (``build_far_field_fock_kernel_native``, C++).

**Phase 2 — Per-iteration Fock build** (inside SCF loop):
  a. Build exact ERIs for NEAR-field quartets only
     (``build_jk_2e_real_space_bipolar_dispatch`` with skip mask).
  b. Apply pre-computed Fock kernel (``apply_far_field_fock_kernel``),
     or use C++ direct contractor (``compute_bipolar_coulomb_far_field_cpp``).
  c. Combine near + far Fock blocks.

**Phase 3 — Gradient** (post-SCF, research preview):
  i. Compute dT/dR interaction-tensor gradient (C++ OpenMP,
     ``compute_bipolar_far_field_gradient_cpp``).
  ii. Add dM/dA moment-derivative terms (Python, using precomputed
      ``moment_grads`` in ``SphericalMomentBuffer``).

The public BIPOLE SCF routes default this feature off and reject an
explicit request to enable it under the current fail-closed contract.
This builder remains a dormant low-level implementation surface.

References
----------
Pisani, Dovesi, and Roetti, *Hartree-Fock Ab Initio Treatment of
Crystalline Systems* (1988), Ch. II.4c, Eqs. II.4.7-II.4.10,
doi:10.1007/978-3-642-93385-1.
Saunders et al., Mol. Phys. 77, 629 (1992), Sec. 5.3 for radial
derivatives and Secs. 6.7-6.10 for electrostatic penetration and the
spherical second-moment correction, doi:10.1080/00268979200102671.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def build_bipolar_far_field_infrastructure(
    system: Any,
    basis: Any,
    lattice_sum_options_2e: Any,
    *,
    use_multipole_far_field: bool = False,
    exchange_split_active: bool = False,
    multipole_l_max: int = 4,
    multipole_engine: str = "libint",  # "libint" or "polipo"
    compute_exchange_dispatch: bool = False,
    ewald_omega: float = 0.0,
    plog: Any = None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Build the per-geometry bipolar far-field data structures.

    Returns ``(spherical_moment_buffer, penetration_dispatch_j,
    penetration_dispatch_k, quartet_tensor_cache,
    far_field_fock_kernel, symmetry_fock_reconstruction_map)``.
    All are ``None`` if the feature is disabled or the build fails.

    Parameters
    ----------
    system : PeriodicSystem
    basis : BasisSet
    lattice_sum_options_2e : LatticeSumOptions
        Two-electron lattice-sum options; its ``cutoff_bohr``
        determines the cell set for moment computation.
    use_multipole_far_field : bool
        Internal low-level gate. Public SCF routes reject ``True`` while
        the quartet far field remains fail-closed.
    exchange_split_active : bool
        Must be True (the far-field is only Ewald-consistent under
        the corrected exchange split).
    multipole_l_max : int
        Maximum multipole order (default 4, hexadecapole).
    compute_exchange_dispatch : bool
        If True, also build the Exchange (K) far-field dispatch.
    plog : ProgressLogger or None
        Progress logger for status messages.

    Returns
    -------
    (spherical_moment_buffer, penetration_dispatch_j, penetration_dispatch_k)
        The Exchange dispatch is None when ``compute_exchange_dispatch``
        is False.
    """
    if not use_multipole_far_field or not exchange_split_active:
        return None, None, None, None, None, None

    try:
        from ._vibeqc_core import (
            LatticeSumOptions,
            compute_multipole_moments_lattice,
        )
        from .bipole_pair_moments import pair_center_moments
        from .bipole_spherical_moment_buffer import (
            build_spherical_moment_buffer,
        )
        from .bipole_dispatch import (
            build_penetration_dispatch_for_bipole_context,
            build_shell_quartet_penetration_dispatch,
            PenetrationDispatchParameters,
        )
    except ImportError as exc:
        if plog is not None:
            plog.warn(
                f"  Bipolar far-field infrastructure unavailable: {exc}"
            )
        return None, None, None, None, None, None

    if plog is not None:
        plog.info(
            "  Building per-pair spherical multipole moments "
            "(dormant PDR 1988 quartet prototype, L_max="
            f"{multipole_l_max})..."
        )

    lo_mom = LatticeSumOptions()
    lo_mom.cutoff_bohr = float(lattice_sum_options_2e.cutoff_bohr)

    try:
        # Compute moments at the highest order supported by the target.
        # For L_max ≤ 3: use emultipole3 (Cartesian, 20 components).
        # For L_max = 4: compute both emultipole3 (full Cartesian L≤3) and
        #   sphemultipole (spherical L=4, 25 components), then merge into a
        #   combined 35-component Cartesian set for the analytical shift to
        #   adjoined-Gaussian product centres. Pisani-Dovesi (1980), Sec. 4,
        #   supplies the diffuse s-Gaussian convention and the standard
        #   Gaussian product theorem supplies its centre. Pisani-Dovesi-
        #   Roetti (1988), Ch. II.4c, uses product-distribution centroids in
        #   the periodic quartet expansion; the shift itself is the standard
        #   binomial moment translation. The sphemultipole moments carry
        #   the traceless L=4 content; the missing trace contributions are
        #   generated automatically by the binomial shift formula via the
        #   D² terms coupling L=2 → L=4.
        if multipole_l_max >= 4 and multipole_engine == "polipo":
            # POLIPO: single native call, no sphemultipole merge needed.
            from .bipole_polipo import compute_moments_via_polipo
            cart_moments = compute_moments_via_polipo(
                basis, system, lo_mom, multipole_l_max)
            mom_L = multipole_l_max
            pair_mom = pair_center_moments(cart_moments, basis, L_target=mom_L)
            if plog is not None:
                plog.info(
                    f"    POLIPO: L={multipole_l_max} Cartesian moments "
                    f"({len(cart_moments.cells)} cells)"
                )

        if multipole_l_max >= 4 and multipole_engine != "polipo":
            # Compute full Cartesian L≤3 (20 components, exact traces)
            cart_L3 = compute_multipole_moments_lattice(
                basis, system, lo_mom, 3, (0.0, 0.0, 0.0)
            )
            # Compute spherical L≤4 (25 components, traceless)
            sph_L4 = compute_multipole_moments_lattice(
                basis, system, lo_mom, 4, (0.0, 0.0, 0.0)
            )
            # Merge: L≤3 from emultipole3 + L=4 from sphemultipole
            from ._sph_to_cart import spherical_to_cartesian_with_traces
            from types import SimpleNamespace

            merged_blocks = spherical_to_cartesian_with_traces(
                sph_L4, cart_L3, basis.nbasis
            )
            # Build a pseudo LatticeMultipoleSet for pair_center_moments
            # with 35 Cartesian components (L=0-4).
            MergedMoments = SimpleNamespace(
                nbf=basis.nbasis,
                L_max=4,
                spherical=False,
                cells=list(cart_L3.cells),
                origin=(0.0, 0.0, 0.0),
                blocks=merged_blocks,
            )
            # L=5,6 isotropic extension. This is an implementation-specific
            # prototype and is not derived in the cited quartet source.
            if multipole_l_max >= 5:
                # Build a real LatticeMultipoleSet for the C++ extender.
                from ._vibeqc_core import LatticeMultipoleSet as LMS, extend_lattice_moments
                M4_real = LMS()
                M4_real.nbf = basis.nbasis
                M4_real.L_max = 4
                M4_real.spherical = False
                M4_real.cells = cart_L3.cells
                M4_real.origin = (0.0, 0.0, 0.0)
                for comp_blocks in merged_blocks:
                    M4_real.blocks.append([np.asarray(b, dtype=float) for b in comp_blocks])
                M_ext = extend_lattice_moments(M4_real, basis, min(multipole_l_max, 6))
                # Convert back to SimpleNamespace blocks.
                merged_blocks_ext = []
                for c in range(len(M_ext.cells)):
                    comps = [np.asarray(M_ext.blocks[c][comp], dtype=float)
                             for comp in range(len(M_ext.blocks[c]))]
                    merged_blocks_ext.append(comps)
                MergedMoments = SimpleNamespace(
                    nbf=basis.nbasis,
                    L_max=M_ext.L_max,
                    spherical=False,
                    cells=list(M_ext.cells),
                    origin=(0.0, 0.0, 0.0),
                    blocks=merged_blocks_ext,
                )
                if plog is not None:
                    plog.info(
                        f"    Extended to L={M_ext.L_max} via isotropic "
                        f"adjoined-Gaussian formulas"
                    )
            pair_mom = pair_center_moments(MergedMoments, basis, L_target=MergedMoments.L_max)
            # Alias for downstream dispatch/symmetry code that references cart_moments
            cart_moments = cart_L3
            if plog is not None:
                plog.info(
                    f"    sphemultipole L=4 spherical moments computed "
                    f"({len(sph_L4.cells)} cells, 25 components); "
                    f"merged with emultipole3 Cartesian for exact shift"
                )
        if multipole_l_max < 4 and multipole_engine == "polipo":
            # POLIPO for L_max <= 3: simple Cartesian call.
            from .bipole_polipo import compute_moments_via_polipo
            cart_moments = compute_moments_via_polipo(
                basis, system, lo_mom, multipole_l_max)
            mom_L = multipole_l_max
            pair_mom = pair_center_moments(cart_moments, basis, L_target=mom_L)
            if plog is not None:
                plog.info(
                    f"    POLIPO: L={multipole_l_max} Cartesian moments "
                    f"({len(cart_moments.cells)} cells)"
                )

        if multipole_l_max < 4 and multipole_engine != "polipo":
            mom_L = 3 if multipole_l_max >= 3 else 2
            cart_moments = compute_multipole_moments_lattice(
                basis, system, lo_mom, mom_L, (0.0, 0.0, 0.0)
            )
            # Use C++ OpenMP shift when available (faster for large cells).
            try:
                from ._vibeqc_core import shift_multipole_moments_to_pair_centres
                from .bipole_pair_moments_native import pair_center_moments_native
                pair_mom = pair_center_moments_native(cart_moments, basis, L_target=mom_L)
            except (ImportError, TypeError):
                pair_mom = pair_center_moments(cart_moments, basis, L_target=mom_L)

        spherical_buffer = build_spherical_moment_buffer(
            pair_mom, basis, L_max=multipole_l_max,
        )
        if plog is not None:
            plog.info(
                f"    {len(spherical_buffer)} shell-pair entries across "
                f"{len(spherical_buffer.cells)} cells"
            )

        # ---- Symmetry-reduced buffer (cell-orbit compression) ---------
        try:
            from .bipole_symmetry_buffer import (
                build_symmetry_reduced_moment_buffer,
            )

            spherical_buffer = build_symmetry_reduced_moment_buffer(
                spherical_buffer, system,
            )
            if plog is not None:
                plog.info(
                    f"    Symmetry-reduced buffer: "
                    f"{len(spherical_buffer)} entries retained"
                )
        except Exception as exc:
            if plog is not None:
                plog.warn(
                    f"    Symmetry buffer reduction failed, using full buffer: {exc}"
                )

        # ---- Symmetry-reduced dispatch (point-group + permutation) ------
        # Try symmetry reduction first (Python, fast at ~0.02s).
        # Falls back to C++ native dispatch if symmetry is unavailable
        # or the reduction fails.
        penetration_dispatch_j = None
        _sym_recon_map = None
        _reduced_dispatch_j = None  # kept for symmetry-reconstructed kernel build
        try:
            from .bipole_symmetry_dispatch import (
                build_point_group_reduced_penetration_dispatch,
                build_symmetry_fock_reconstruction_map,
                unfold_symmetry_reduced_dispatch,
            )
            from .bipole_bravais_utils import cell_volume_bohr, cell_dimensionality
            from .bipole_dispatch import PenetrationDispatchParameters

            vol = cell_volume_bohr(system)
            dim = cell_dimensionality(system)
            sym_params = PenetrationDispatchParameters.for_cell_volume(
                vol,
                maximum_multipole_order=multipole_l_max,
                dimensionality=dim,
            )
            sym_result_j, _sym_result_k = (
                build_point_group_reduced_penetration_dispatch(
                    system,
                    basis,
                    list(cart_moments.cells),
                    sym_params,
                    compute_exchange=False,
                )
            )
            if plog is not None:
                plog.info(
                    f"    Symmetry-reduced dispatch: "
                    f"{sym_result_j.n_reduced} unique quartets "
                    f"(factor {sym_result_j.symmetry_factor:.1f}x "
                    f"vs {sym_result_j.n_full} full)"
                )
            n_sh = len(list(basis.shells()))
            # Keep reduced dispatch for symmetry-reconstructed kernel.
            _reduced_dispatch_j = sym_result_j.dispatch
            penetration_dispatch_j = unfold_symmetry_reduced_dispatch(
                _reduced_dispatch_j, n_sh,
            )
            # Build full symmetry reconstruction map from reduced dispatch.
            # Point-group expansion works correctly: the density dict already
            # has entries for all cells including symmetry-equivalent ones,
            # and the AO-basis permutation is handled via shell-index mapping.
            # No explicit density rotation is needed.
            shell_slices_list = list(spherical_buffer.shell_slices)
            _sym_recon_map = build_symmetry_fock_reconstruction_map(
                _reduced_dispatch_j,
                system,
                basis,
                list(cart_moments.cells),
                shell_slices_list,
            )
            if plog is not None:
                plog.info(
                    f"    Unfolded: {len(penetration_dispatch_j)} quartets; "
                    f"reconstruction map: {len(_sym_recon_map)} canonical entries, "
                    f"{_sym_recon_map.n_sym_operations} sym ops"
                )
        except ImportError:
            _sym_recon_map = None
            pass  # spglib not available, fall back to C++ dispatch

        # Fall back to C++ native dispatch if symmetry didn't produce one.
        if penetration_dispatch_j is None:
            _sym_recon_map = None
            penetration_dispatch_j = (
                build_penetration_dispatch_for_bipole_context(
                    basis,
                    system,
                    list(cart_moments.cells),
                    maximum_multipole_order=multipole_l_max,
                )
            )
            if plog is not None:
                plog.info(
                    f"    {len(penetration_dispatch_j)} far-field Coulomb "
                    f"quartets dispatched (C++ native)"
                )

        # Exchange (K) dispatch: also symmetry-reduced when available.
        penetration_dispatch_k = None
        if compute_exchange_dispatch:
            try:
                vol = cell_volume_bohr(system)
                sym_params_k = PenetrationDispatchParameters.for_cell_volume(
                    vol, dimensionality=dim,
                    maximum_multipole_order=multipole_l_max,
                )
                _j_disp_k, sym_result_k = (
                    build_symmetry_reduced_penetration_dispatch(
                        basis,
                        list(cart_moments.cells),
                        sym_params_k,
                        compute_exchange=True,
                    )
                )
                if sym_result_k is not None:
                    penetration_dispatch_k = (
                        unfold_symmetry_reduced_dispatch(
                            sym_result_k.dispatch, n_sh,
                        )
                    )
                    if plog is not None:
                        plog.info(
                            f"    {len(penetration_dispatch_k)} far-field "
                            f"Exchange quartets dispatched"
                        )
            except ImportError:
                from .bipole_dispatch import (
                    build_shell_quartet_penetration_dispatch,
                )

                vol = cell_volume_bohr(system)
                params = PenetrationDispatchParameters.for_cell_volume(
                    vol, dimensionality=dim,
                    maximum_multipole_order=multipole_l_max,
                )
                _j_disp, k_disp = (
                    build_shell_quartet_penetration_dispatch(
                        basis,
                        list(cart_moments.cells),
                        params,
                        compute_exchange=True,
                    )
                )
                penetration_dispatch_k = k_disp
            if plog is not None:
                plog.info(
                    f"    {len(penetration_dispatch_k)} far-field Exchange "
                    f"quartets dispatched"
                )

        # ---- Pre-compute interaction tensor cache ------------------------
        _tensor_cache = None
        try:
            from .bipole_quartet_tensor_cache import build_quartet_tensor_cache

            if plog is not None:
                plog.info(
                    "  Pre-computing quartet interaction tensors..."
                )
            _tensor_cache = build_quartet_tensor_cache(
                spherical_buffer,
                penetration_dispatch_j,
                ewald_omega=ewald_omega,
            )
            if plog is not None:
                plog.info(
                    f"    {len(_tensor_cache)} unique tensors cached"
                )
        except Exception as exc:
            if plog is not None:
                plog.warn(
                    f"    Tensor cache build failed, continuing without it: {exc}"
                )

        # ---- Pre-compute the dormant prototype Fock-kernel cache --------
        _fock_kernel = None
        try:
            from .bipole_fock_kernel_native import (
                build_far_field_fock_kernel_native,
            )

            if plog is not None:
                plog.info(
                    "  Building dormant prototype far-field Fock kernel..."
                )
            # Use reduced dispatch when symmetry reconstruction is available.
            _kernel_dispatch = (
                _reduced_dispatch_j
                if _sym_recon_map is not None and _reduced_dispatch_j is not None
                else penetration_dispatch_j
            )
            # Use Python kernel builder for reduced dispatch to preserve
            # dispatch-index tracking needed for reconstruction alignment.
            if _sym_recon_map is not None:
                from .bipole_far_field_kernel import build_far_field_fock_kernel as _py_kernel
                _fock_kernel = _py_kernel(
                    spherical_buffer,
                    _kernel_dispatch,
                    ewald_omega=ewald_omega,
                    nbf=basis.nbasis if hasattr(basis, 'nbasis') else 0,
                )
            else:
                _fock_kernel = build_far_field_fock_kernel_native(
                    spherical_buffer,
                    _kernel_dispatch,
                    ewald_omega=ewald_omega,
                    nbf=basis.nbasis if hasattr(basis, 'nbasis') else 0,
                )
            if plog is not None:
                plog.info(
                    f"    {len(_fock_kernel)} kernel entries"
                    + (" (symmetry-reduced)" if _sym_recon_map is not None else "")
                )
        except Exception as exc:
            if plog is not None:
                plog.warn(
                    f"    Fock kernel build failed, falling back to direct contractor: {exc}"
                )

        return spherical_buffer, penetration_dispatch_j, penetration_dispatch_k, _tensor_cache, _fock_kernel, _sym_recon_map

    except Exception as exc:
        if plog is not None:
            plog.warn(
                f"  Bipolar far-field infrastructure build failed: "
                f"{exc}. Falling back to exact-only Fock build."
            )
        return None, None, None, None, None, None
