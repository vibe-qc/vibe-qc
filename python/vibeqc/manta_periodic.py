"""Selected finite periodic operators with explicit supports and native receipts."""
from __future__ import annotations

import math

import numpy as np

from . import _vibeqc_core as _core
from .manta_scf import IntegralEngineOptions


def _selected_periodic_hartree(basis, system, densities, momenta, weights, cells,
                               omega, *, lattice_opts, ke_cutoff, reciprocal_batch,
                               cell_chunk_size, integral_engine):
    """Admit Python grid/snapshots, then build one immutable whole-mesh epoch."""
    from .aux_basis import _reciprocal_index_box_per_axis, rsgdf_dense_g_mesh
    from .ewald_composed import _physical_pair_cutoff
    from .output import record

    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError("Selected periodic Hartree requires MANTA, fallback='error' and channel_batch=1")
    if isinstance(reciprocal_batch, bool) or not isinstance(reciprocal_batch, int) or reciprocal_batch <= 0:
        raise ValueError("Selected periodic Hartree reciprocal batch must be a positive integer")
    if cell_chunk_size is not None and (isinstance(cell_chunk_size, bool) or cell_chunk_size != 1):
        raise ValueError("Selected periodic Hartree streams one cell at a time; cell_chunk_size must be None or 1")
    if system.dim != 3:
        raise ValueError("Selected periodic Hartree requires dim == 3")
    if not math.isfinite(float(omega)):
        raise ValueError("Selected periodic Hartree omega must be finite (the unsplit result is independent of it)")
    ke = float(ke_cutoff)
    if not math.isfinite(ke) or ke <= 0 or not math.isfinite(2*ke):
        raise ValueError("Selected periodic Hartree kinetic cutoff must be finite and positive")
    nk, nc, n = len(momenta), len(cells), int(basis.nbasis)
    if not nk or len(densities) != nk or len(weights) != nk:
        raise ValueError("Selected periodic Hartree needs equally sized, nonempty density, momentum and weight lists")
    for d in densities:
        if (not isinstance(d, np.ndarray) or d.dtype not in (np.dtype('float64'), np.dtype('complex128'))
                or d.shape != (n, n) or not (d.flags.c_contiguous or d.flags.f_contiguous)):
            raise ValueError("Selected periodic Hartree densities must be contiguous float64/complex128 AO matrices")
    a = np.asarray(system.lattice, dtype=np.float64)
    if a.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(np.linalg.det(a)) or np.linalg.det(a) == 0:
        raise ValueError("Selected periodic Hartree lattice must be finite and nonsingular")
    reach = math.sqrt(2*ke)
    if not np.isfinite(np.linalg.norm(a, axis=0)*reach).all():
        raise ValueError("Selected periodic Hartree reciprocal grid extent overflow")
    half_width = _reciprocal_index_box_per_axis(a, reach, 3)
    candidates = math.prod(2*x+1 for x in half_width)
    # Conservative inventory of the legacy mesh generator's index grids,
    # Cartesian vectors, masks/sort indices, Python input copies and exports.
    # The native plan separately inventories its caller arrays and snapshots.
    python_bytes = 256*candidates + 64*nk*n*n + 256*(nk+nc) + 4096
    python_terms = 256*candidates + 16*nk*n*n + 32*(nk+nc)
    budget = _core._manta.ResourceBudget()
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError("Selected periodic Hartree budget must be ResourceBudget")
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    if (python_bytes + budget.other_live_bytes > budget.memory_bytes or python_terms > budget.scalar_terms
            or 32*python_terms > budget.kernel_terms):
        raise ValueError("MANTA periodic Hartree Python grid/snapshot resource budget exceeded")
    budget.other_live_bytes += python_bytes
    budget.scalar_terms -= python_terms
    budget.kernel_terms -= 32*python_terms
    cutoff = _physical_pair_cutoff(lattice_opts)
    # The enclosing integer box is conservative for the sphere. Refuse an
    # oversized request before allocating the mesh or any density snapshot.
    _core._manta.plan_periodic_hartree(basis, candidates-1, nc, nk, reciprocal_batch, cutoff, budget)
    g_all = rsgdf_dense_g_mesh(system, ke)
    g = np.ascontiguousarray(g_all[np.any(g_all != 0., axis=1)])
    translations = np.array([list(c.r_cart) for c in cells], dtype=np.float64).reshape(nc, 3)
    k = np.ascontiguousarray(momenta, dtype=np.float64)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    d = np.stack(densities).astype(np.complex128, copy=False)
    result = _core._manta.compute_periodic_hartree(
        basis, a, g, translations, k, w, d, reciprocal_batch, cutoff, budget)
    receipt = result.receipt
    fields = ("provider", "provider_version", "normalization", "algorithm", "boundary", "source_identity",
              "density_identity", "panel_calls", "shell_calls", "skipped_shell_pairs", "primitive_pairs",
              "vector_primitive_products", "phase_products", "phase_accumulations", "contraction_terms",
              "modeled_peak_bytes", "energy", "density_hermiticity_residual", "cell_imaginary_residual",
              "response_hermiticity_residual", "cell_volume")
    evidence = {name: getattr(receipt, name) for name in fields}
    evidence.update(schema_version=1, citation_methods=["manta_periodic_hartree"],
                    reciprocal_vectors=len(g), cells=nc, momenta=nk, ke_cutoff=ke,
                    pair_cutoff_bohr=cutoff, python_grid_candidates=candidates,
                    python_reserved_bytes=python_bytes, python_scalar_terms=python_terms,
                    scalar_terms=python_terms+result.plan.scalar_terms,
                    kernel_terms=32*python_terms+result.plan.kernel_terms)
    record("manta_periodic_hartree",
           f"  Periodic Hartree: manta; {nk} k points; {len(g)} G vectors; G=0 omitted.\n",
           integral_provenance=evidence)
    return result


def compute_one_electron_lattice(basis, cells, *, operation="overlap",
                                 pair_cutoff_bohr=None, integral_engine):
    """Build an ordered real S/T LatticeMatrixSet from explicitly supplied cells.

    The caller owns the finite translation domain. No lattice enumeration,
    symmetry reduction, screening, normalization adjustment or matrix repair
    is inferred. ``None`` disables the physical shell-pair cutoff; zero keeps
    only coincident centers. This producer does not select a periodic SCF route.
    """
    from .output import record

    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError("Selected periodic S/T requires MANTA, fallback='error' and channel_batch=1")
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError("Selected periodic S/T budget must be ResourceBudget")
    nc, n = len(cells), int(basis.nbasis)
    # Account for Python exports, list metadata and the legacy matrix-set
    # conversion in addition to the native result, snapshots and workspace.
    python_bytes = 64*nc*n*n + 512*nc + 4096
    python_terms = 8*nc*n*n + 16*nc
    budget = _core._manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    if (python_bytes + budget.other_live_bytes > budget.memory_bytes or python_terms > budget.scalar_terms
            or 32*python_terms > budget.kernel_terms):
        raise ValueError("MANTA periodic S/T Python conversion resource budget exceeded")
    budget.other_live_bytes += python_bytes
    budget.scalar_terms -= python_terms
    budget.kernel_terms -= 32*python_terms
    _core._manta.plan_bloch_one_electron(basis, operation, nc, 0, pair_cutoff_bohr, budget)
    if any(not isinstance(c, _core.LatticeCell) for c in cells):
        raise TypeError("Selected periodic S/T cells must be LatticeCell objects")
    translations = np.array([list(c.r_cart) for c in cells], dtype=np.float64).reshape(nc, 3)
    result = _core._manta.compute_bloch_one_electron(
        basis, operation, translations, np.empty((0, 3)), pair_cutoff_bohr, budget)
    blocks = result.cell_values
    matrix_set = _core.make_lattice_matrix_set(n, cells, list(blocks))
    fields = ("provider", "provider_version", "normalization", "convention", "operation", "source_identity",
              "candidate_shell_pairs", "skipped_shell_pairs", "shell_calls", "primitive_pairs",
              "phase_products", "phase_accumulations", "integral_elements", "modeled_peak_bytes")
    evidence = {name: getattr(result.receipt, name) for name in fields}
    evidence.update(schema_version=1, citation_methods=["manta_bloch_one_electron"],
                    cells=nc, momenta=0, pair_cutoff_bohr=pair_cutoff_bohr,
                    python_reserved_bytes=python_bytes, python_scalar_terms=python_terms,
                    scalar_terms=python_terms+result.plan.scalar_terms,
                    kernel_terms=32*python_terms+result.plan.kernel_terms)
    record("manta_periodic_one_electron",
           f"  Periodic {operation}: manta; {nc} explicit cells.\n",
           integral_provenance=evidence)
    return matrix_set


def compute_nuclear_lattice(basis, system, cells, wave_vectors, *, alpha,
                            nuclear_cutoff_bohr, reciprocal_batch=64,
                            pair_cutoff_bohr=None, integral_engine):
    """Return the finite 3D Ewald nuclear operator on explicit cell/G supports.

    Nuclear images use the shared physical midpoint sphere. The caller owns
    the AO cells, reciprocal support, split and both cutoffs. No image-domain
    substitution, reciprocal screening or AO rescaling is applied.
    """
    from .output import record

    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError("Selected periodic nuclear operator requires MANTA, fallback='error' and channel_batch=1")
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError("Selected periodic nuclear budget must be ResourceBudget")
    if isinstance(reciprocal_batch, bool) or not isinstance(reciprocal_batch, int) or reciprocal_batch<=0:
        raise ValueError("Selected periodic nuclear reciprocal batch must be a positive integer")
    if (not isinstance(wave_vectors, np.ndarray) or wave_vectors.dtype != np.dtype('float64')
            or wave_vectors.ndim != 2 or wave_vectors.shape[1] != 3
            or not (wave_vectors.flags.c_contiguous or wave_vectors.flags.f_contiguous)):
        raise ValueError("Selected periodic nuclear wave_vectors must be contiguous float64 (count,3) arrays")
    nc, n, ng = len(cells), int(basis.nbasis), len(wave_vectors)
    python_bytes = 96*nc*n*n + 512*nc + 4096
    python_terms = 8*nc*n*n + 16*nc
    budget = _core._manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    if (python_bytes + budget.other_live_bytes > budget.memory_bytes or python_terms > budget.scalar_terms
            or 32*python_terms > budget.kernel_terms):
        raise ValueError("MANTA periodic nuclear Python conversion resource budget exceeded")
    budget.other_live_bytes += python_bytes
    budget.scalar_terms -= python_terms
    budget.kernel_terms -= 32*python_terms
    _core._manta.plan_periodic_nuclear(basis, system, ng, nc, 0, alpha, nuclear_cutoff_bohr,
                                     reciprocal_batch, pair_cutoff_bohr, budget)
    if any(not isinstance(c, _core.LatticeCell) for c in cells):
        raise TypeError("Selected periodic nuclear cells must be LatticeCell objects")
    translations = np.array([list(c.r_cart) for c in cells], dtype=np.float64).reshape(nc, 3)
    result = _core._manta.compute_periodic_nuclear(
        basis, system, wave_vectors, translations, np.empty((0, 3)), alpha, nuclear_cutoff_bohr,
        reciprocal_batch, pair_cutoff_bohr, budget)
    # The native producer checks the raw imaginary residual before this
    # conversion to the legacy real LatticeMatrixSet representation.
    blocks = result.cell_values.real
    matrix_set = _core.make_lattice_matrix_set(n, cells, list(blocks))
    fields = ("provider", "provider_version", "normalization", "algorithm", "boundary", "nuclear_support",
              "source_identity", "nuclear_shell_calls", "overlap_shell_calls", "fourier_shell_calls",
              "skipped_shell_pairs", "primitive_pairs", "primitive_charge_products", "nuclear_images_retained",
              "vector_primitive_products", "contraction_terms", "modeled_peak_bytes", "alpha", "nuclear_cutoff_bohr",
              "cell_volume", "nuclear_charge", "constant_coefficient", "cell_imaginary_residual")
    evidence = {name: getattr(result.receipt, name) for name in fields}
    evidence.update(schema_version=1, citation_methods=["manta_periodic_nuclear"],
                    cells=nc, momenta=0, reciprocal_vectors=ng, pair_cutoff_bohr=pair_cutoff_bohr,
                    python_reserved_bytes=python_bytes, python_scalar_terms=python_terms,
                    scalar_terms=python_terms+result.plan.scalar_terms,
                    kernel_terms=32*python_terms+result.plan.kernel_terms)
    record("manta_periodic_nuclear",
           f"  Periodic nuclear attraction: manta; {nc} cells; {ng} G vectors; G=0 omitted.\n",
           integral_provenance=evidence)
    return matrix_set


def build_periodic_exchange(basis, lattice, wave_vectors, transfer_offsets,
                            translations, momenta, weights, densities, *,
                            reciprocal_batch=64, pair_cutoff_bohr=None, integral_engine,
                            kernel="coulomb", omega=0.):
    """Build Coulomb/erf/erfc exchange for an immutable all-k/all-density epoch.

    Transfer domains are explicit; positive-range erfc admits its finite Q=0
    term when supplied. Coulomb and erf omit Q=0. No Madelung correction,
    mesh reconstruction or matrix repair is inferred. The result retains source,
    density, work and per-channel exchange-energy receipts.
    """
    from .output import record

    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError("Selected periodic exchange requires MANTA, fallback='error' and channel_batch=1")
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError("Selected periodic exchange budget must be ResourceBudget")
    if isinstance(reciprocal_batch, bool) or not isinstance(reciprocal_batch, int) or reciprocal_batch<=0:
        raise ValueError("Selected periodic exchange reciprocal batch must be a positive integer")
    if not isinstance(densities, np.ndarray) or densities.ndim != 4:
        raise ValueError("Selected periodic exchange densities require a (density,k,AO,AO) array")
    # Native input/export owners are already modeled. Reserve this adapter's
    # per-density identity/energy provenance objects before native allocation.
    python_bytes = 512*len(densities) + 4096
    budget = _core._manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    if python_bytes + budget.other_live_bytes > budget.memory_bytes:
        raise ValueError("MANTA periodic exchange Python provenance memory budget exceeded")
    budget.other_live_bytes += python_bytes
    result = _core._manta.compute_periodic_exchange(basis, lattice, wave_vectors, transfer_offsets,
        translations, momenta, weights, densities, reciprocal_batch, pair_cutoff_bohr, budget, kernel, omega)
    fields = ("provider", "provider_version", "normalization", "algorithm", "boundary", "convention",
              "source_identity", "density_identity", "density_identities", "energies", "transfer_vectors",
              "panel_calls", "shell_calls", "skipped_shell_pairs", "primitive_pairs", "vector_primitive_products",
              "phase_products", "phase_accumulations", "matrix_products", "contraction_terms", "modeled_peak_bytes",
              "cell_volume", "density_hermiticity_residual", "response_hermiticity_residual", "kernel", "omega", "zero_vectors")
    evidence = {name: getattr(result.receipt, name) for name in fields}
    evidence.update(schema_version=1, citation_methods=["manta_periodic_exchange"],
                    cells=result.plan.cells, momenta=result.plan.momenta, densities=result.plan.densities,
                    pair_cutoff_bohr=pair_cutoff_bohr, python_reserved_bytes=python_bytes,
                    scalar_terms=result.plan.scalar_terms, kernel_terms=result.plan.kernel_terms)
    record("manta_periodic_exchange",
           f"  Periodic exchange: manta; {result.plan.momenta} k points; "
           f"{result.plan.densities} density channels; {result.receipt.boundary}.\n",
           integral_provenance=evidence)
    return result


def build_periodic_fock(basis, lattice, hartree_vectors, exchange_vectors, transfer_offsets,
                        translations, momenta, weights, hcore, spin_densities, *,
                        reciprocal_batch=64, pair_cutoff_bohr=None, kernel="coulomb", omega=0.,
                        exchange_scale=1., integral_engine):
    """Freeze F_alpha/F_beta and matching electronic energy for one density epoch.

    Hcore is supplied explicitly. No XC, nuclear energy or exchange singularity
    correction is inferred. Use D_alpha=D_beta=D_total/2 for restricted input.
    """
    from .output import record

    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError("Selected periodic Fock requires MANTA, fallback='error' and channel_batch=1")
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError("Selected periodic Fock budget must be ResourceBudget")
    if isinstance(reciprocal_batch, bool) or not isinstance(reciprocal_batch, int) or reciprocal_batch<=0:
        raise ValueError("Selected periodic Fock reciprocal batch must be a positive integer")
    budget = _core._manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    python_bytes = 16384
    if python_bytes+budget.other_live_bytes>budget.memory_bytes:
        raise ValueError("MANTA periodic Fock Python provenance memory budget exceeded")
    budget.other_live_bytes += python_bytes
    result = _core._manta.compute_periodic_fock(basis, lattice, hartree_vectors, exchange_vectors,
        transfer_offsets, translations, momenta, weights, hcore, spin_densities, reciprocal_batch,
        pair_cutoff_bohr, budget, kernel, omega, exchange_scale)
    fields = ("provider", "algorithm", "convention", "source_identity", "density_identity", "one_electron_identity",
              "snapshot_identity", "exchange_scale", "one_electron_energy", "coulomb_energy", "exchange_energy",
              "electronic_energy", "hcore_hermiticity_residual", "fock_hermiticity_residual", "modeled_peak_bytes")
    evidence = {name: getattr(result.receipt, name) for name in fields}
    j, k = result.hartree_receipt, result.exchange_receipt
    shared = ("provider", "provider_version", "normalization", "algorithm", "boundary", "source_identity",
              "density_identity", "shell_calls", "primitive_pairs", "vector_primitive_products", "contraction_terms")
    evidence.update(schema_version=1, citation_methods=["manta_periodic_hartree", "manta_periodic_exchange"],
                    hartree={name: getattr(j, name) for name in shared},
                    exchange={name: getattr(k, name) for name in shared+("kernel", "omega", "zero_vectors", "density_identities")},
                    scalar_terms=result.plan.scalar_terms, kernel_terms=result.plan.kernel_terms,
                    python_reserved_bytes=python_bytes, pair_cutoff_bohr=pair_cutoff_bohr,
                    momenta=result.plan.momenta, spin_labels=["alpha", "beta"], one_electron_source="supplied-matrix")
    record("manta_periodic_fock", f"  Periodic spin Fock snapshot: manta; {result.plan.momenta} k points; {k.boundary}.\n",
           integral_provenance=evidence)
    return result


def _periodic_hf_configuration(options, integral_engine, method):
    option_type = getattr(_core._manta, f"Periodic{method}Options")
    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if (integral_engine.backend != "manta" or integral_engine.fallback != "error"
            or integral_engine.channel_batch != 1 or isinstance(integral_engine.channel_batch, bool)
            or not isinstance(integral_engine.channel_batch, int)):
        raise ValueError(f"Selected periodic {method} requires MANTA, fallback='error' and channel_batch=1")
    if not isinstance(integral_engine.budget, _core._manta.ResourceBudget):
        raise TypeError(f"Selected periodic {method} budget must be ResourceBudget")
    if options is None:
        options = option_type()
    if not isinstance(options, option_type):
        raise TypeError(f"Selected periodic {method} options must be manta.Periodic{method}Options")
    budget = _core._manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets", "scalar_terms", "kernel_terms",
                 "other_live_bytes", "backend_margin_bytes"):
        setattr(budget, name, getattr(integral_engine.budget, name))
    python_bytes = 49152 if method in ("GlobalHybridKS", "RangeSeparatedKS", "RIGlobalHybridKS", "RIRangeSeparatedKS") else 32768
    if python_bytes+budget.other_live_bytes>budget.memory_bytes:
        raise ValueError(f"MANTA periodic {method} Python provenance memory budget exceeded")
    budget.other_live_bytes += python_bytes
    return options, budget, python_bytes


def _periodic_rhf_configuration(options, integral_engine):
    return _periodic_hf_configuration(options, integral_engine, "RHF")


def _periodic_hf_provenance(result, mesh, options, python_bytes, method):
    """Copy the executed finite periodic source and convergence evidence."""
    fields = ("provider", "algorithm", "boundary", "source_identity", "overlap_identity", "kinetic_identity",
              "nuclear_identity", "nuclear_energy_identity", "final_density_identity", "final_fock_identity",
              "confirmed", "nuclear_real", "nuclear_reciprocal", "nuclear_self", "nuclear_background",
              "nuclear_energy", "one_electron_energy", "coulomb_energy", "exchange_energy", "energy",
              "confirmation_fock_error", "confirmation_energy_error", "nuclear_pair_images", "fock_calls",
              "shell_calls", "primitive_pairs", "modeled_peak_bytes", "exchange_q0", "exchange_correction_identity",
              "madelung", "uncorrected_exchange_energy", "exchange_correction_energy",
              "madelung_real_vectors", "madelung_reciprocal_vectors", "domain_selection", "domain_identity",
              "reciprocal_energy_cutoff", "domain_translations", "domain_hartree_vectors",
              "domain_exchange_vectors", "domain_conjugacy_vectors")
    evidence = {name: getattr(result.receipt, name) for name in fields}
    methods = [f"manta_periodic_{method}"]
    if result.receipt.exchange_q0 == "BvK-ewald":
        methods.append("manta_periodic_madelung")
    probe = result.plan.madelung
    evidence["madelung_plan"] = {name: getattr(probe, name) for name in (
        "alpha", "real_cutoff_bohr", "reciprocal_cutoff_bohr_inv", "real_candidates",
        "reciprocal_candidates", "modeled_peak_bytes", "kernel_terms")}
    s = result.solver_snapshot
    evidence.update(schema_version=1, citation_methods=methods, converged=result.converged,
                    status=s.status.name, iterations=s.evaluated_iteration,
                    commutator_frobenius_rms=s.commutator_frobenius_rms,
                    maximum_density_closure_frobenius=s.maximum_density_closure_frobenius,
                    maximum_electron_count_error=s.maximum_electron_count_error,
                    global_band_gap=s.global_band_gap, retained_rank=result.solver_memory.retained_rank,
                    mesh=list(mesh.mesh), is_shift=list(mesh.is_shift),
                    alpha=options.alpha, nuclear_cutoff_bohr=options.nuclear_cutoff_bohr,
                    pair_cutoff_bohr=options.pair_cutoff_bohr, python_reserved_bytes=python_bytes,
                    scalar_terms=result.plan.scalar_terms, kernel_terms=result.plan.kernel_terms,
                    domain_live_bytes=result.plan.domain_live_bytes,
                    domain_scalar_terms=result.plan.domain_scalar_terms,
                    domain_kernel_terms=result.plan.domain_kernel_terms,
                    scf_history_steps=result.solver_snapshot.evaluated_iteration,
                    scf_history_capacity=result.plan.history_capacity,
                    scf_history_reserved_bytes=result.plan.history_live_bytes)
    return evidence


def periodic_rhf_provenance(result, mesh, options, python_bytes):
    return _periodic_hf_provenance(result, mesh, options, python_bytes, "rhf")


def periodic_uhf_provenance(result, mesh, options, python_bytes):
    evidence = _periodic_hf_provenance(result, mesh, options, python_bytes, "uhf")
    populations = [result.receipt.n_alpha, result.receipt.n_beta]
    snapshot = result.solver_snapshot
    evidence.update(spin_labels=["alpha", "beta"], spin_electrons_per_cell=populations,
                    occupation_policy="fixed-spin-integer-uniform-full-mesh",
                    spin_band_gaps=[gap if occupied else None for gap, occupied in
                                    zip(snapshot.spin_band_gap, populations)])
    return evidence


def _periodic_exchange_accuracy(result):
    """Uniform full-Coulomb omission, with the declared fixed BvK correction."""
    r = result.receipt.exchange_accuracy
    evidence = {name: getattr(r, name) for name in (
        'qualified', 'bvk_correction', 'method_scale', 'fixed_madelung', 'maximum_error', 'energy_error_bound',
        'maximum_correction_error', 'correction_energy_error_bound', 'overlap_identity', 'certificate_identity',
        'shell_calls', 'primitive_pairs')}
    evidence['qualification'] = 'full-coulomb-fixed-madelung-canonical-occupied-space'
    evidence['excludes'] = ['madelung-numerics', 'integral-roundoff', 'erfc-exchange', 'fitting-metric', 'density-relaxation', 'k-mesh-error']
    evidence['targets'] = {name: getattr(r.targets, name) for name in (
        'matrix_error', 'energy_error', 'ao_completion_cube_radius', 'reciprocal_completion_cube_radius')}
    evidence['exchange'] = {name: getattr(r.exchange, name) for name in (
        'qualification', 'source_identity', 'pair_source_identity', 'density_envelope_identity', 'certificate_identity',
        'maximum_error', 'energy_error_bound', 'entry_energy_error_bound', 'l1_energy_error_bound', 'roundoff_included')}
    evidence['density_l1_limit'] = r.exchange.density.density_l1_limit
    evidence['images'] = None if r.images is None else {name: getattr(r.images.receipt, name) for name in (
        'source_identity', 'pair_source_identity', 'cube_radius', 'maximum_overlap_error', 'roundoff_included')}
    evidence['plan'] = {name: getattr(result.plan.exchange_accuracy, name) for name in (
        'modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
    return evidence


def _periodic_nuclear_accuracy(result):
    """Physical-source nuclear enclosure over the canonical density space."""
    r = result.receipt.nuclear_accuracy
    if r is None:
        return None
    evidence = {name: getattr(r, name) for name in (
        'qualified', 'entry_attraction_energy_error_bound', 'l1_attraction_energy_error_bound',
        'attraction_energy_error_bound', 'total_energy_error_bound', 'density_l1_limit',
        'density_envelope_identity', 'certificate_identity', 'shell_calls', 'primitive_pairs')}
    evidence['qualification'] = 'full-images-nuclear-matrix-and-canonical-density-energy'
    evidence['excludes'] = ['integral-roundoff', 'overlap-rank-stability', 'kinetic', 'two-electron',
                            'metric-solve', 'density-relaxation']
    evidence['targets'] = {name: getattr(r.targets, name) for name in (
        'matrix_error', 'attraction_energy_error', 'nuclear_energy_error', 'ao_completion_cube_radius',
        'real_completion_cube_radius', 'reciprocal_completion_cube_radius')}
    evidence['nuclear'] = {name: getattr(r.nuclear.receipt, name) for name in (
        'source_identity', 'nuclear_source_identity', 'pair_source_identity', 'pair_image_identity',
        'boundary', 'nuclear_support', 'maximum_potential_error', 'nuclear_real_error_bound',
        'nuclear_reciprocal_error_bound', 'nuclear_energy_error_bound', 'roundoff_included')}
    evidence['plan'] = {name: getattr(result.plan.nuclear_accuracy, name) for name in (
        'modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
    return evidence


def _periodic_one_electron_accuracy(result):
    """Separate image certificate; no nuclear, exchange or relaxed-state claim."""
    r = result.receipt.one_electron_accuracy
    if r is None:
        return None
    evidence = {name: getattr(r, name) for name in (
        'qualified', 'kinetic_energy_error_bound', 'density_l1_limit',
        'density_envelope_identity', 'certificate_identity', 'shell_calls', 'primitive_pairs')}
    if r.overlap_rank is not None:
        rank = r.overlap_rank
        evidence['overlap_image_rank'] = {name: getattr(rank, name) for name in (
            'qualified', 'maximum_perturbation_bound', 'minimum_stability_radius',
            'perturbation_bounds', 'projection_bounds', 'certificate_identity')}
        evidence['overlap_image_rank'].update(
            qualification='canonical-overlap-rank-under-omitted-image-perturbations',
            spectra=[_periodic_metric_spectrum(c) for c in rank.spectra],
            excludes=['integral-construction-arithmetic', 'occupied-space-rotation',
                      'density-envelope-change', 'density-relaxation', 'total-energy'])
    evidence['qualification'] = 'full-ao-image-matrix-and-canonical-density-kinetic-energy'
    evidence['excludes'] = ['integral-roundoff', 'nuclear', 'two-electron', 'metric-solve', 'density-relaxation']
    evidence['targets'] = {name: getattr(r.targets, name) for name in (
        'overlap_error', 'kinetic_error', 'kinetic_energy_error', 'completion_cube_radius', 'overlap_image_rank')}
    evidence['images'] = {name: getattr(r.images.receipt, name) for name in (
        'source_identity', 'pair_source_identity', 'cube_radius', 'cube_cells', 'shell_pairs_evaluated',
        'primitive_pairs_evaluated', 'retained_ordered_pairs', 'omitted_ordered_pairs_in_cube',
        'real_metric_lower', 'maximum_overlap_error', 'maximum_kinetic_error', 'roundoff_included')}
    evidence['plan'] = {name: getattr(result.plan.one_electron_accuracy, name) for name in (
        'modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
    return evidence


# #627: composition of the separate fixed-density certificates. Each
# certificate bounds one energy component at the canonical density against
# its own reference object; the components are disjoint, so their bounds may
# be summed into one fixed-density component bound. The sum is not a relaxed
# total-energy tolerance: density relaxation is delimited by an uncoupled
# second-order estimate below, never bounded here.
_LEDGER_CERTIFICATES = (
    ('one_electron_accuracy', 'kinetic', 'full-ao-image-overlap-and-kinetic-matrices',
     'kinetic_energy_error_bound', ('images', 'maximum_kinetic_error'), ('density_envelope_identity',), 'canonical-density'),
    ('nuclear_accuracy', 'nuclear-attraction-and-repulsion', 'full-image-nuclear-potential-and-point-ewald-energy',
     'total_energy_error_bound', ('nuclear', 'maximum_potential_error'), ('density_envelope_identity',), 'canonical-density'),
    ('exchange_accuracy', 'exact-exchange', 'full-coulomb-exchange-domains-with-fixed-madelung',
     'energy_error_bound', ('maximum_error',), ('exchange', 'density_envelope_identity'), 'canonical-occupied-subspaces'),
)
# Coupling-only certificates leave physical Hartree domains excluded. The
# source-linked full-Coulomb certificate can cover that component instead;
# overlapping coupling and full-Coulomb bounds must never be added.
_LEDGER_ALWAYS_EXCLUDED = ('density-relaxation', 'k-mesh-error', 'basis-incompleteness', 'integral-roundoff',
                           'overlap-rank-stability', 'madelung-numerics', 'xc', 'hartree-omitted-domains')
_LEDGER_COMPONENT_TAGS = {'kinetic': 'kinetic', 'nuclear': 'nuclear-attraction-and-repulsion', 'exchange': 'exact-exchange'}


def _normalised_exclusions(raw, claimed):
    """Drop a certificate's exclusion of a component another certificate claims."""
    out = set()
    for tag in raw:
        if tag in _LEDGER_COMPONENT_TAGS:
            if _LEDGER_COMPONENT_TAGS[tag] not in claimed:
                out.add(tag)
        elif tag == 'two-electron':
            if 'coulomb' not in claimed:
                out.add('hartree-omitted-domains')
            if 'exact-exchange' not in claimed:
                out.add('exact-exchange-domains')
        else:
            out.add(tag)
    return out


def _nested(mapping, path):
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def relaxation_delimiter(fock_entry_error_bounds, orbital_energies, overlaps, n_occupied, retained_rank):
    """Uncoupled closed-shell second-order estimate of the density-relaxation energy.

    ``fock_entry_error_bounds`` are per-component maximum-entry bounds on the
    AO Fock contributions (kinetic, nuclear potential, exchange). With
    ``n`` AO functions, ``||dF||_F <= n * sum(entries)``; in the S-orthonormal
    retained frame ``||dF_mo||_F <= ||dF||_F / s_min`` with ``s_min`` the
    smallest retained overlap eigenvalue over the mesh; the closed-shell
    second-order energy ``-2 sum_ia |dF_ia|^2/(e_a-e_i)`` is bounded in
    magnitude by ``2 ||dF_mo||_F^2 / gap`` with ``gap`` the smallest
    occupied-virtual gap over the mesh. This is an uncoupled estimate, not a
    certificate: the coupled orbital response differs by the orbital-Hessian
    correction, overlap perturbations are excluded (dS changes the metric),
    and it applies only without level crossing (``||dF_mo|| < gap/2``).
    """
    import numpy as np
    energies = [np.asarray(e, dtype=float).ravel() for e in orbital_energies]
    n_occupied = int(n_occupied); retained_rank = int(retained_rank)
    overlaps = [np.asarray(s) for s in overlaps]
    n_basis = int(overlaps[0].shape[-1]) if overlaps else 0
    entry = float(sum(float(v) for v in fock_entry_error_bounds.values()))
    out = dict(qualification='uncoupled-second-order-estimate', is_certificate=False,
               fock_entry_error_bounds={k: float(v) for k, v in fock_entry_error_bounds.items()},
               fock_entry_error_bound=entry, n_basis=n_basis, n_occupied=n_occupied, retained_rank=retained_rank,
               applies_only_if=['occupied-virtual gap > 0 at every k point',
                                '||dF_mo||_F < gap/2 (no level crossing; perturbative regime)',
                                'overlap perturbations excluded: dS changes the metric, not the Fock operator alone',
                                'coupled orbital response not included; needs the stability eigenvalue (#625)'])
    if n_occupied <= 0 or n_occupied >= retained_rank or not energies or not overlaps:
        out.update(applies=False, energy_estimate=None, minimum_gap=None, per_k_gaps=[],
                   reason='no occupied-virtual gap in the retained frame')
        return out
    gaps = [float(e[n_occupied]-e[n_occupied-1]) for e in energies]
    gap = min(gaps)
    s_min = min(float(np.sort(np.linalg.eigvalsh(s))[::-1][retained_rank-1]) for s in overlaps)
    frobenius = n_basis*entry
    mo_bound = frobenius/s_min if s_min > 0 else float('inf')
    applies = gap > 0 and np.isfinite(mo_bound) and mo_bound < gap/2
    estimate = 2.*mo_bound*mo_bound/gap if gap > 0 and np.isfinite(mo_bound) else None
    out.update(per_k_gaps=gaps, minimum_gap=gap, overlap_retained_minimum=s_min, fock_frobenius_bound=frobenius,
               fock_mo_frobenius_bound=mo_bound if np.isfinite(mo_bound) else None,
               energy_estimate=estimate, applies=bool(applies))
    return out


def _ledger_nonnegative_bound(value):
    """Admit a finite nonnegative stored scalar, without repairing a bad bound."""
    try:
        bound = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return bound if math.isfinite(bound) and bound >= 0. else None


def _ledger_add_up(total, bound):
    """Outward positive binary64 sum; see scientific contract Section 5.38.

    Each positive rounded addition is advanced toward +infinity. Zero terms
    and the initial stored bound need no arithmetic and remain exact. This
    encloses addition of stored certificates, not their construction errors.
    """
    if total == 0.:
        return bound
    if bound == 0.:
        return total
    upper = math.nextafter(total + bound, math.inf)
    return upper if math.isfinite(upper) else None


def _ledger_scaled_bound(bound, coefficient):
    """Outward absolute method scaling of a stored nonnegative component."""
    bound = _ledger_nonnegative_bound(bound)
    try:
        coefficient = float(coefficient)
    except (TypeError, ValueError, OverflowError):
        return None
    if bound is None or not math.isfinite(coefficient):
        return None
    if not bound or not coefficient:
        return 0.
    return _ledger_nonnegative_bound(math.nextafter(abs(coefficient) * bound, math.inf))


def compose_fixed_density_ledger(evidence, *, method='rhf', orbital_energies=None, overlaps=None,
                                 n_occupied=None, retained_rank=None):
    """Compose the separate fixed-density certificates into one component ledger (#627).

    Each requested certificate is mapped to the disjoint energy component it
    bounds and its reference object. The ledger refuses (``qualified`` False
    with reasons) when a requested certificate is unqualified, when two
    certificates claim one component, or when the one-electron and nuclear
    certificates were not evaluated over the same canonical density envelope.
    ``fixed_density_energy_error_bound`` is the sum over the qualified disjoint
    components at the canonical density; it is explicitly not a relaxed
    total-energy tolerance. Positive stored bounds are summed outward; invalid
    or overflowing bounds refuse qualification. This encloses only composition
    arithmetic, not the construction of the input certificates.
    The relaxation scale is delimited, not bounded,
    by :func:`relaxation_delimiter` (closed-shell results only).
    """
    components, reasons, claimed, raw_excludes = [], [], {}, set()
    envelopes = {}
    fock_entries = {}
    total = 0.

    def accumulate(bound, name):
        nonlocal total
        if bound is None:
            reasons.append(f'{name} energy bound must be finite and nonnegative')
        elif total is not None:
            total = _ledger_add_up(total, bound)
            if total is None:
                reasons.append('fixed-density component sum has no finite outward bound')

    for name, component, reference, energy_key, entry_path, envelope_path, family in _LEDGER_CERTIFICATES:
        cert = evidence.get(name)
        if cert is None:
            components.append(dict(certificate=name, component=component, requested=False))
            continue
        bound = _ledger_nonnegative_bound(cert.get(energy_key))
        identity = _nested(cert, envelope_path)
        entry = dict(certificate=name, component=component, requested=True, qualified=bool(cert.get('qualified')) and bound is not None,
                     reference_object=reference, density_family=family, density_envelope_identity=identity,
                     certificate_identity=cert.get('certificate_identity'), energy_error_bound=bound,
                     excludes=list(cert.get('excludes') or ()))
        raw_excludes.update(entry['excludes'])
        if component in claimed:
            reasons.append(f'{name} and {claimed[component]} both claim the {component} component')
        claimed.setdefault(component, name)
        if not entry['qualified']:
            reasons.append(f'{name} is not qualified')
        if family == 'canonical-density':
            envelopes.setdefault(identity, []).append(name)
        value = _nested(cert, entry_path)
        if value is not None:
            matrix_bound = _ledger_nonnegative_bound(value)
            if matrix_bound is None:
                entry['qualified'] = False
                reasons.append(f'{name} matrix bound must be finite and nonnegative')
            else:
                fock_entries[component] = matrix_bound
        accumulate(bound, name)
        components.append(entry)
    fit = evidence.get('fitting') or {}
    fitting = fit.get('accuracy') or {}
    full = fit.get('full_coulomb_accuracy') or {}
    full_requested = bool(full.get('targets_requested'))
    if full_requested:
        bound = _ledger_nonnegative_bound(full.get('energy_error_bound'))
        matrix_bound = _ledger_nonnegative_bound(full.get('j_error_bound'))
        identity = full.get('density_envelope_identity')
        gamma = (fit.get('metrics') or [{}])[0].get('source_identity')
        valid = bool(full.get('qualified')) and bound is not None and matrix_bound is not None
        if not valid:
            reasons.append('full_coulomb_accuracy is not qualified')
        if not identity or full.get('density_limit_origin') != 'canonical-overlap' or not full.get('component_density_envelope'):
            reasons.append('full_coulomb_accuracy requires an identified canonical density envelope')
            valid = False
        if not gamma or gamma != full.get('ri_source_identity'):
            reasons.append('full_coulomb_accuracy does not match the selected Gamma RI source')
            valid = False
        if full.get('qualification') != 'full-zero-average-coulomb-on-fixed-weighted-hermitian-density-family;stored-integrals':
            reasons.append('full_coulomb_accuracy has an unsupported reference qualification')
            valid = False
        entry = dict(certificate='full_coulomb_accuracy', component='coulomb', requested=True,
                     qualified=valid, reference_object='full-zero-average-coulomb;stored-integrals',
                     density_family='canonical-density', density_envelope_identity=identity,
                     certificate_identity=full.get('certificate_identity'),
                     source_identity=full.get('ri_source_identity'), reference_source_identity=full.get('reference_source_identity'),
                     energy_error_bound=bound, excludes=list(full.get('excludes') or ()),
                     supersedes=['fitting_accuracy'] if fitting.get('enabled') else [])
        if fitting.get('enabled') and not fitting.get('qualified'):
            reasons.append('superseded fitting_accuracy is not qualified')
        claimed['coulomb'] = 'full_coulomb_accuracy'
        envelopes.setdefault(identity, []).append('full_coulomb_accuracy')
        raw_excludes.update(entry['excludes'])
        accumulate(bound, 'full_coulomb_accuracy')
        if matrix_bound is not None:
            fock_entries['coulomb'] = matrix_bound
        components.append(entry)
    elif fitting.get('enabled'):

        entry = dict(certificate='fitting_accuracy', component='coulomb-fitting', requested=True,
                     qualified=bool(fitting.get('qualified')), reference_object='declared-finite-coupling-source',
                     density_family='fixed-density-family', density_envelope_identity=fitting.get('density_envelope_identity'),
                     certificate_identity=fitting.get('certificate_identity'),
                     energy_error_bound=_ledger_nonnegative_bound(fitting.get('energy_bound')), excludes=['fitting-basis-incompleteness', 'metric-solve'])
        if 'coulomb-fitting' in claimed:
            reasons.append('two certificates claim the coulomb-fitting component')
        claimed['coulomb-fitting'] = 'fitting_accuracy'
        if entry['energy_error_bound'] is None:
            entry['qualified'] = False
        if not entry['qualified']:
            reasons.append('fitting_accuracy is not qualified')
        accumulate(entry['energy_error_bound'], 'fitting_accuracy')
        raw_excludes.update(entry['excludes'])
        components.append(entry)
    else:
        components.append(dict(certificate='fitting_accuracy', component='coulomb-fitting', requested=False,
                               note='exact finite coupling or no coupling target; metric inverse certificates feed this component only'))
    exchange = evidence.get('exchange_fitting_accuracy') or {}
    if exchange.get('targets_requested'):
        # Section 5.47 already embeds the HF spin density and validates every
        # state. This is its raw stored-map energy term, not executed K/Fock
        # arithmetic or the separately owned BvK/source correction.
        name = 'exchange_fitting_accuracy'
        bound = _ledger_nonnegative_bound(exchange.get('energy_error_bound'))
        matrix_bound = _ledger_nonnegative_bound(exchange.get('maximum_error'))
        identity = exchange.get('canonical_density_envelope_identity')
        valid = bool(exchange.get('qualified')) and bound is not None and matrix_bound is not None
        if not valid:
            reasons.append(f'{name} is not qualified')
        coefficient = exchange.get('exchange_fraction')
        if method in ('rks', 'uks'):
            if (evidence.get('range_separated') or coefficient != evidence.get('exact_exchange_fraction')
                    or _ledger_scaled_bound(0., coefficient) is None or coefficient == 0.):
                reasons.append(f'{name} does not match the global-hybrid exchange coefficient')
                valid = False
            bound = _ledger_scaled_bound(bound, coefficient)
            matrix_bound = _ledger_scaled_bound(matrix_bound, coefficient)
            if bound is None or matrix_bound is None:
                reasons.append(f'{name} has no finite outward method-scaled bound')
                valid = False
        elif method not in ('rhf', 'uhf') or coefficient != 1.:
            reasons.append(f'{name} ledger requires a supported HF/KS exchange coefficient')
            valid = False
        if not fit.get('has_k') or not fit.get('source_identity') or exchange.get('ri_source_identity') != fit['source_identity']:
            reasons.append(f'{name} does not match the selected fitted K source')
            valid = False
        if not identity:
            reasons.append(f'{name} requires an identified canonical density envelope')
            valid = False
        if (exchange.get('kernel') != 'coulomb' or exchange.get('physical_omega') != 0.
                or exchange.get('qualification') != 'full-coulomb-exchange-on-unweighted-hermitian-spin-density-family;zero-mode-omitted;stored-integrals'):
            reasons.append(f'{name} has an unsupported reference qualification')
            valid = False
        if 'exact-exchange' in claimed:
            reasons.append(f'{name} and {claimed["exact-exchange"]} both claim the exact-exchange component')
            valid = False
        claimed.setdefault('exact-exchange', name)
        envelopes.setdefault(identity, []).append(name)
        excluded = ['integral-roundoff', 'rank-stability', 'density-relaxation',
                    'exchange-response-and-assembly-arithmetic', 'exchange-source-correction', 'restricted-density-embedding']
        raw_excludes.update(excluded)
        components.append(dict(certificate=name, component='exact-exchange', requested=True, qualified=valid,
            reference_object='full-coulomb-exchange;zero-mode-omitted;stored-integrals',
            density_family='canonical-density', density_envelope_identity=identity,
            certificate_identity=exchange.get('certificate_identity'), source_identity=exchange.get('ri_source_identity'),
            reference_source_identity=exchange.get('reference_source_identity'), energy_error_bound=bound, excludes=excluded,
            method_coefficient=coefficient if _ledger_scaled_bound(0., coefficient) is not None else None))
        accumulate(bound, name)
        if matrix_bound is not None and valid:
            fock_entries['exact-exchange'] = matrix_bound
    range_exchange = evidence.get('range_exchange_accuracy') or {}
    if range_exchange:
        # Section 5.50 owns the signed physical sum. Validate its children
        # against the actual method components, then add the aggregate once.
        name = 'range_exchange_accuracy'
        bound = _ledger_nonnegative_bound(range_exchange.get('energy_error_bound'))
        matrix_bound = _ledger_nonnegative_bound(range_exchange.get('maximum_error'))
        valid = bool(range_exchange.get('qualified')) and bound is not None and matrix_bound is not None
        if not valid:
            reasons.append(f'{name} is not qualified')
        if method not in ('rks', 'uks') or not evidence.get('range_separated'):
            reasons.append(f'{name} requires a physical range-separated KS method')
            valid = False
        if range_exchange.get('qualification') != 'signed-physical-range-exchange-on-canonical-spin-family;component-sum;no-madelung-or-scf-relaxation':
            reasons.append(f'{name} has an unsupported reference qualification')
            valid = False
        children = range_exchange.get('components') or []
        coefficients = range_exchange.get('coefficients') or []
        physical = evidence.get('exchange_components') or []
        required = (evidence.get('xc') or {}).get('required_exchange_components') or []
        if not children or not (len(children) == len(coefficients) == len(physical) == len(required)):
            reasons.append(f'{name} does not cover the complete physical exchange component list')
            valid = False
        energy_sum = 0.
        families = []
        for index, (child, coefficient, actual, expected) in enumerate(zip(children, coefficients, physical, required)):
            label = f'{name}[{index}]'
            kernel, omega = actual.get('kernel'), actual.get('omega')
            fit_child = actual.get('fitting') or {}
            qualification = ('full-coulomb-exchange-on-unweighted-hermitian-spin-density-family;zero-mode-omitted;stored-integrals'
                if kernel == 'coulomb' else f'physical-{kernel}-exchange-on-unweighted-hermitian-spin-density-family;screened-zero-accounted;stored-integrals;no-madelung')
            matching = (kernel in ('coulomb', 'erf', 'erfc') and _ledger_nonnegative_bound(omega) is not None
                and (omega == 0. if kernel == 'coulomb' else omega > 0.)
                and child.get('kernel') == kernel and child.get('physical_omega') == omega
                and child.get('qualification') == qualification
                and fit_child.get('exchange_kernel') == kernel and fit_child.get('physical_omega') == omega
                and fit_child.get('has_k') and bool(fit_child.get('source_identity'))
                and child.get('ri_source_identity') == fit_child.get('source_identity')
                and expected.get('kernel') == kernel and expected.get('omega') == omega
                and expected.get('coefficient') == coefficient == actual.get('coefficient') == child.get('exchange_fraction'))
            scaled = _ledger_scaled_bound(child.get('energy_error_bound'), coefficient)
            child_matrix = _ledger_nonnegative_bound(child.get('maximum_error'))
            if not matching or not child.get('qualified') or child.get('targets_requested') or scaled is None or child_matrix is None:
                reasons.append(f'{label} has incompatible source, physical coefficient or qualification')
                valid = False
            identity = child.get('canonical_density_envelope_identity')
            if not identity:
                reasons.append(f'{label} requires an identified canonical density envelope')
                valid = False
            envelopes.setdefault(identity, []).append(label)
            families.append(identity)
            if energy_sum is not None and scaled is not None:
                energy_sum = _ledger_add_up(energy_sum, scaled)
            else:
                energy_sum = None
        if energy_sum is None or bound is None or bound < energy_sum:
            reasons.append(f'{name} aggregate does not enclose its scaled child energy bounds')
            valid = False
        if 'exact-exchange' in claimed:
            reasons.append(f'{name} and {claimed["exact-exchange"]} both claim the exact-exchange component')
            valid = False
        claimed.setdefault('exact-exchange', name)
        excluded = ['integral-roundoff', 'rank-stability', 'density-relaxation',
                    'exchange-response-and-assembly-arithmetic', 'exchange-source-correction', 'restricted-density-embedding']
        raw_excludes.update(excluded)
        components.append(dict(certificate=name, component='exact-exchange', requested=True, qualified=valid,
            reference_object='signed-physical-range-exchange;screened-zero-accounted;stored-integrals',
            density_family='canonical-density', density_envelope_identity=families[0] if families else None,
            certificate_identity=range_exchange.get('certificate_identity'), energy_error_bound=bound, excludes=excluded,
            component_certificate_identities=[c.get('certificate_identity') for c in children]))
        accumulate(bound, name)
    if len(envelopes) > 1:
        reasons.append('certificates were evaluated over different canonical density envelopes: '
                       + '; '.join(f'{k}: {v}' for k, v in envelopes.items()))
    requested = [c for c in components if c.get('requested')]
    qualified = bool(requested) and not reasons
    excludes = set(_LEDGER_ALWAYS_EXCLUDED) | _normalised_exclusions(raw_excludes, claimed)
    if full_requested and qualified:
        excludes.discard('hartree-omitted-domains')
    ledger = dict(schema_version=1,
                  scope='fixed-canonical-density;disjoint-components;not-a-relaxed-energy-tolerance',
                  composition_arithmetic='outward-binary64-sum-of-stored-nonnegative-bounds',
                  components=components, claimed_components=sorted(claimed), qualified=qualified, reasons=reasons,
                  fixed_density_energy_error_bound=total if qualified else None,
                  excludes=sorted(excludes))
    if qualified and method == 'rhf' and orbital_energies is not None and overlaps is not None and fock_entries:
        ledger['relaxation_delimiter'] = relaxation_delimiter(fock_entries, orbital_energies, overlaps, n_occupied, retained_rank)
    else:
        ledger['relaxation_delimiter'] = dict(qualification='not-evaluated', is_certificate=False,
            reason='component ledger is not qualified' if not qualified else
                   ('closed-shell estimate only' if method != 'rhf' else 'no certificate matrix bounds or no spectrum'))
    return ledger


_REFERENCE_COMPONENT_BOUND_FIELDS = tuple(
    f'reference_{component}_{quantity}_bound'
    for component in ('approximation', 'real_domain', 'reciprocal_domain', 'pair_image')
    for quantity in ('j', 'energy')
)


def _periodic_metric_spectrum(spectrum):
    return dict({name: getattr(spectrum, name) for name in (
        'spectrum_certified', 'rank_certified', 'rank', 'residual_bound', 'orthogonality_bound',
        'eigenvalue_error_bound', 'cutoff_lower', 'cutoff_upper', 'rank_stability_radius')},
        scope='exact-stored-hermitian-metric;strict-perturbation-radius;no-integral-or-fock-roundoff')


def _periodic_metric_inverse(certificate):
    return dict({name: getattr(certificate, name) for name in (
        'qualified', 'retained_lower', 'inverse_norm_bound', 'divided_difference_bound',
        'residual_frobenius_bound', 'residual_term', 'orthogonality_term',
        'factor_product_bound', 'inverse_error_bound')},
        scope='stored-factor-product-versus-exact-stored-retained-metric-inverse;spectral-norm;no-integral-or-fock-roundoff')


def _periodic_metric_action_admission(bound):
    return dict({name.removeprefix('metric_action_'): getattr(bound, name) for name in (
        'metric_action_coupling', 'metric_action_requested', 'metric_action_qualified', 'metric_action_j_target', 'metric_action_energy_target',
        'metric_action_j_bound', 'metric_action_energy_bound', 'metric_action_columns', 'metric_action_adjoint_norm_bound',
        'metric_action_shell_calls', 'metric_action_primitive_products',
        'metric_action_assembly_requested', 'metric_action_assembly_qualified', 'metric_action_assembly_identity',
        'metric_action_stored_j_bound', 'metric_action_stored_energy_bound',
        'metric_action_assembly_j_bound', 'metric_action_assembly_energy_bound')},
        scope=('declared-stored-coupling;stored-factor-plus-conditional-metric-assembly-response;fixed-density-family'
               if bound.metric_action_assembly_requested else
               'declared-stored-coupling-representation;factor-product-versus-ideal-retained-inverse;fixed-density-family;no-integral-or-fock-roundoff'),
        excludes=['coupling-basis-data-and-vector-generation', 'coupling-construction',
                  'metric-domain-and-screening-perturbations', 'executed-fock-and-energy-arithmetic', 'total-relaxed-energy'])


def _periodic_metric_rank_admission(bound):
    return dict(requested_perturbation=bound.metric_rank_perturbation,
        qualified=bound.metric_rank_qualified, spectrum=_periodic_metric_spectrum(bound.metric_spectrum),
        scope='selected-gamma-coulomb-metric-only;additional-norm-margin-not-an-integral-error-estimate')


def _periodic_fitting_provenance(info, plan, options, density_envelope=None, fitting_bound_plan=None):
    fitting = {name: getattr(info, name) for name in (
        'has_j', 'has_k', 'source_identity', 'frame_identity', 'weight_identity', 'metric_shell_calls',
        'metric_primitive_products', 'reverse_metric_residual', 'reverse_projector_residual',
        'reverse_inverse_residual')}
    fitting['accuracy'] = {name: getattr(info.accuracy,name) for name in (
        'enabled','qualified','j_target','energy_target','density_l1_limit','selected_coupling_error',
        'j_bound','energy_bound','attempts','shell_calls','primitive_products',
        'reference_coupling_identity','selected_coupling_identity','certificate_identity',
        'component_density_envelope','density_envelope_identity')}
    if info.accuracy.enabled:
        if info.accuracy.metric_truncation_certified:
            fitting['accuracy']['metric_truncation'] = {name: getattr(info.accuracy,name) for name in (
                'metric_reference_cutoff','metric_reference_rank','metric_discarded_rank',
                'metric_j_bound','metric_energy_bound','coupling_j_bound','coupling_energy_bound',
                'metric_discarded_factor_identity','metric_reference_identity')}
            fitting['accuracy']['qualification'] = 'coupling-and-nested-stored-positive-metric-factor'
            fitting['accuracy']['excludes'] = ['finite-fitting-basis-and-domains','metric-eigensolver',
                'integral-roundoff','rank-stability','density-relaxation']
        fitting['accuracy']['requested_density_l1_limit'] = options.ri_accuracy.density_l1_limit
        fitting['accuracy']['density_limit_origin'] = 'canonical-overlap' if options.ri_accuracy.density_l1_limit == 0 else 'explicit'
        if density_envelope is not None and options.ri_accuracy.density_l1_limit == 0:
            fitting['accuracy']['density_envelope'] = dict(
                per_k_l1=density_envelope.per_k_l1.tolist(), retained_rank=density_envelope.memory.retained_rank,
                n_alpha=density_envelope.n_alpha, n_beta=density_envelope.n_beta,
                weighted_entry_bounds_identity=info.accuracy.density_envelope_identity,
                owned_bytes=density_envelope.memory.peak_owned_numerical_bytes,
                work_units=density_envelope.memory.maximum_work_units)
    if info.fitting_bound is not None:
        bound = info.fitting_bound
        full = {name: getattr(bound, name) for name in (
            'qualification', 'ri_source_identity', 'reference_source_identity', 'pair_source_identity',
            'certificate_identity', 'density_envelope_identity', 'probes', 'density_l1_limit', 'j_error_bound', 'energy_error_bound',
            'finite_difference_j_bound', 'reference_remainder_j_bound',
            'finite_difference_energy_bound', 'reference_remainder_energy_bound',
            'reference_batches', 'reference_source_passes', 'reference_shell_calls', 'reference_majorant_pairs',
            *_REFERENCE_COMPONENT_BOUND_FIELDS, 'roundoff_included',
            'component_density_envelope', 'targets_requested')}
        full.update(j_target=options.fitting_accuracy.j_error, energy_target=options.fitting_accuracy.energy_error,
                    probe_batch_size=options.fitting_accuracy.probe_batch_size,
                    density_limit_origin='canonical-overlap', includes=['selected-fit-rank-coupling-and-source-domains'],
                    excludes=['integral-contraction-arithmetic', 'rank-stability', 'density-relaxation'],
                    component_bounds_additive=False)
        full['qualified'] = bool(bound.targets_requested and
            bound.j_error_bound <= options.fitting_accuracy.j_error and
            bound.energy_error_bound <= options.fitting_accuracy.energy_error and
            (bound.metric_rank_perturbation is None or bound.metric_rank_qualified) and
            (not bound.metric_action_requested or bound.metric_action_qualified))
        full['density_envelope'] = dict(per_k_l1=density_envelope.per_k_l1.tolist(),
            retained_rank=density_envelope.memory.retained_rank, n_alpha=density_envelope.n_alpha,
            n_beta=density_envelope.n_beta, owned_bytes=density_envelope.memory.peak_owned_numerical_bytes,
            work_units=density_envelope.memory.maximum_work_units)
        ref = options.fitting_accuracy.reference
        reference_receipt = ref.receipt if ref is not None else info.fitting_reference
        reference_fields = (
            'identity', 'pair_source_identity', 'tree_identity', 'kernel_identity', 'electronic_source_identity',
            'pair_cube_radius', 'real_cube_radius', 'reciprocal_cube_radius', 'near_cube_radius',
            'pairs', 'nodes', 'leaf_capacity', 'order', 'branching', 'maximum_depth',
            'eta', 'opening', 'integral_tolerance', 'pair_cutoff_bohr',
            'pointwise_tolerance', 'tail_mass_tolerance', 'minimum_radius', 'maximum_radius')
        full['reference_preparation'] = {name: getattr(reference_receipt, name) for name in reference_fields}
        full['plan'] = {name: getattr(fitting_bound_plan, name) for name in (
            'maximum_probes', 'maximum_trials', 'modeled_peak_bytes', 'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms')}
        if info.fitting_trials:
            full['metric_selection'] = dict(
                maximum_refinements=options.fitting_accuracy.maximum_metric_refinements,
                cutoff_reduction=options.fitting_accuracy.metric_cutoff_reduction,
                minimum_cutoff=options.fitting_accuracy.minimum_metric_cutoff,
                qualification='first-passing-full-coulomb-map-before-scf',
                work_accounting=('cumulative-staged-search-reservation' if (options.fitting_accuracy.reference_selection is not None or options.fitting_accuracy.domain_selection is not None or options.fitting_accuracy.auxiliary_selection is not None)
                    else 'all-allowed-trials-and-refactorizations-reserved-including-discarded-trials'),
                trials=[dict(cutoff=t.cutoff, rank=t.rank, **{name: getattr(t.bound, name) for name in (
                    'ri_source_identity', 'reference_source_identity', 'certificate_identity',
                    'j_error_bound', 'energy_error_bound', 'finite_difference_j_bound', 'reference_remainder_j_bound',
                    'finite_difference_energy_bound', 'reference_remainder_energy_bound',
                    'reference_batches', 'reference_source_passes', 'reference_shell_calls', 'reference_majorant_pairs',
                    *_REFERENCE_COMPONENT_BOUND_FIELDS)})
                    for t in info.fitting_trials])
        if info.reference_trials:
            policy = options.fitting_accuracy.reference_selection
            full['reference_selection'] = dict(
                qualification='shared-bupo-reference-prepared-and-selected-before-scf',
                work_accounting='cumulative-admitted-stage-and-probe-bounds-including-discarded-trials',
                selective_refinement=policy.selective_refinement,
                maximum_refinements=policy.maximum_refinements, maximum_cube_radius=policy.maximum_cube_radius,
                maximum_pair_cutoff_bohr=policy.maximum_pair_cutoff_bohr,
                growth_factor=policy.growth_factor, reference_fraction=policy.reference_fraction,
                reservation={name: getattr(policy.budget, name) for name in (
                    'memory_bytes', 'backend_margin_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms')},
                charged={name: getattr(info.reference_search, name) for name in (
                    'modeled_peak_bytes', 'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms')},
                trials=[dict(ready=t.ready, refinement=t.refinement,
                    preparation={name: getattr(t.preparation, name) for name in reference_fields},
                    bound={name: getattr(t.bound, name) for name in (
                        'certificate_identity', 'ri_source_identity', 'reference_source_identity',
                        'j_error_bound', 'energy_error_bound', 'reference_remainder_j_bound',
                        'reference_remainder_energy_bound', 'reference_batches', 'reference_source_passes',
                        'reference_shell_calls', 'reference_majorant_pairs',
                        *_REFERENCE_COMPONENT_BOUND_FIELDS)})
                    for t in info.reference_trials])
        if info.domain_trials:
            policy = options.fitting_accuracy.domain_selection
            fields = ('pair_cutoff_bohr', 'auxiliary_cutoff_bohr', 'metric_cutoff',
                      'reciprocal_cube_radius', 'vectors', 'rank', 'domain_attempt', 'metric_refinement')
            limits = ('memory_bytes', 'backend_margin_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms')
            effective = _core._manta.periodic_ri_fitting_search_budget(options.fitting_accuracy)
            full['domain_selection'] = dict(
                qualification='first-passing-full-coulomb-j-map;fixed-bases-kernel-and-ao-frame',
                work_accounting='joint-cumulative-search;maximum-scf-source-capacity-reserved',
                controls={name: getattr(policy, name) for name in (
                    'initial_pair_cutoff_bohr', 'initial_auxiliary_cutoff_bohr', 'initial_reciprocal_cube_radius',
                    'maximum_refinements', 'maximum_reciprocal_cube_radius', 'growth_factor',
                    'maximum_pair_cutoff_bohr', 'maximum_auxiliary_cutoff_bohr')},
                requested_reservation={name: getattr(policy.budget, name) for name in limits},
                effective_reservation={name: getattr(effective, name) for name in limits},
                charged={name: getattr(info.reference_search, name) for name in (
                    'modeled_peak_bytes', 'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms')},
                selected={name: getattr(info.domain_trials[-1], name) for name in fields},
                trials=[dict(**{name: getattr(t, name) for name in fields}, bound={name: getattr(t.bound, name) for name in (
                    'ri_source_identity', 'reference_source_identity', 'certificate_identity', 'j_error_bound',
                    'energy_error_bound', 'finite_difference_j_bound', 'reference_remainder_j_bound',
                    'finite_difference_energy_bound', 'reference_remainder_energy_bound',
                    'reference_batches', 'reference_source_passes', 'reference_shell_calls', 'reference_majorant_pairs',
                    *_REFERENCE_COMPONENT_BOUND_FIELDS)}) for t in info.domain_trials])
        if info.auxiliary_trials:
            from .auxiliary_generation import product_auxiliary_receipt_provenance
            policy = options.fitting_accuracy.auxiliary_selection
            effective = _core._manta.periodic_ri_fitting_search_budget(options.fitting_accuracy)
            limits = ('memory_bytes', 'backend_margin_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms')
            fields = ('pair_cutoff_bohr', 'auxiliary_cutoff_bohr', 'metric_cutoff',
                      'reciprocal_cube_radius', 'vectors', 'rank', 'domain_attempt', 'metric_refinement')
            rows = [dict(candidate=t.candidate, functions=t.functions,
                generation_identity=t.generation_identity, auxiliary_basis_identity=t.auxiliary_basis_identity,
                **{name: getattr(t.source, name) for name in fields},
                bound={name: getattr(t.source.bound, name) for name in (
                    'ri_source_identity', 'reference_source_identity', 'certificate_identity', 'j_error_bound',
                    'energy_error_bound', 'finite_difference_j_bound', 'reference_remainder_j_bound',
                    'finite_difference_energy_bound', 'reference_remainder_energy_bound',
                    'reference_batches', 'reference_source_passes', 'reference_shell_calls', 'reference_majorant_pairs',
                    *_REFERENCE_COMPONENT_BOUND_FIELDS)}) for t in info.auxiliary_trials]
            full['auxiliary_selection'] = dict(
                qualification='first-passing-full-coulomb-j-map;fixed-reference-family-kernel-and-orbital-frame',
                order='seed-then-shared-generated-candidates;domains-then-metric-within-each-basis',
                work_accounting='joint-cumulative-search;maximum-scf-source-capacity-reserved',
                initial={name: getattr(policy.initial, name) for name in ('beta', 'refinement', 'maximum_angular_momentum')},
                maximum_refinements=policy.maximum_refinements,
                generation_limits={name: getattr(policy.generation_limits, name) for name in (
                    'memory_bytes', 'work_units', 'maximum_shells', 'maximum_functions')},
                generation_plan={name: getattr(policy.prepared.plan, name) for name in (
                    'candidates', 'modeled_peak_bytes', 'work_units', 'maximum_shells', 'maximum_functions')},
                requested_reservation={name: getattr(policy.budget, name) for name in limits},
                effective_reservation={name: getattr(effective, name) for name in limits},
                charged={name: getattr(info.reference_search, name) for name in (
                    'modeled_peak_bytes', 'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms')},
                selected=rows[-1], trials=rows,
                selected_generation=(product_auxiliary_receipt_provenance(info.selected_auxiliary_generation)
                    if info.selected_auxiliary_generation is not None else None))
            if info.domain_trials:
                full['domain_selection']['qualification'] = 'full-coulomb-j-domains-within-auxiliary-candidate-search'
        full['metric_action'] = _periodic_metric_action_admission(info.fitting_bound)
        full['metric_rank'] = _periodic_metric_rank_admission(info.fitting_bound)
        for row, trial in zip(full.get('metric_selection', {}).get('trials', []), info.fitting_trials):
            row['metric_action'] = _periodic_metric_action_admission(trial.bound)
            row['metric_rank'] = _periodic_metric_rank_admission(trial.bound)
        for key, trials in (('reference_selection', info.reference_trials),
                            ('domain_selection', info.domain_trials), ('auxiliary_selection', info.auxiliary_trials)):
            for row, trial in zip(full.get(key, {}).get('trials', []), trials):
                bound = trial.source.bound if key == 'auxiliary_selection' else trial.bound
                row['bound']['metric_action'] = _periodic_metric_action_admission(bound)
                row['bound']['metric_rank'] = _periodic_metric_rank_admission(bound)
        fitting['full_coulomb_accuracy'] = full
    fitting['metrics'] = [{name: getattr(metric, name) for name in (
        'rank', 'cutoff', 'negative_limit', 'minimum_eigenvalue', 'maximum_eigenvalue',
        'hermiticity_residual', 'zero_transfer', 'metric_source_identity', 'canonical_metric_identity',
        'factor_identity', 'projector_identity', 'source_identity', 'shell_calls', 'primitive_products')}
        for metric in info.metrics]
    for row, metric in zip(fitting['metrics'], info.metrics):
        row['spectrum'] = _periodic_metric_spectrum(metric.spectrum)
        row['assembly_rank'] = dict(
            requested=metric.assembly_rank_requested, qualified=metric.assembly_rank_qualified,
            phase_included=metric.assembly_phase_included, weight_included=metric.assembly_weight_included,
            prefactor_included=metric.assembly_prefactor_included, fourier_included=metric.assembly_fourier_included, integrals_included=metric.assembly_integrals_included, representation_included=metric.assembly_representation_included,
            perturbation_bound=metric.assembly_perturbation_bound, certificate_identity=metric.assembly_certificate_identity,
            qualification='lower-triangle-metric-rank-under-sr-integral-phase-lattice-weight-auxiliary-fourier-and-pure-transform-assembly-error',
            excludes=['coupling-evaluation', 'basis-data-generation',
                      'momentum-and-translation-generation', 'source-domains-and-screening', 'retained-space-rotation', 'inverse-response', 'total-energy'])
        row['assembly_response'] = {name: getattr(metric.assembly_response, name) for name in (
            'qualified', 'dimension', 'rank', 'perturbation_bound', 'frobenius_perturbation_bound',
            'retained_lower', 'gap_lower', 'divided_difference_bound', 'projector_error_bound',
            'inverse_error_bound', 'factor_inverse_error_bound')}
        row['assembly_response'].update(
            qualification='finite-retained-projector-and-inverse-response-to-sr-integral-phase-lattice-weight-auxiliary-fourier-and-pure-transform-metric-assembly',
            excludes=['coupling-basis-data-and-vector-generation', 'source-domains-and-screening',
                      'coupling-perturbations', 'fock-and-energy-propagation', 'total-relaxed-energy'])
        row['inverse_certificate'] = _periodic_metric_inverse(metric.inverse_certificate)
    fitting['controls'] = {name: getattr(options.fitting, name) for name in (
        'omega', 'pair_cutoff_bohr', 'auxiliary_cutoff_bohr', 'integral_screen_error', 'coupling_screen_error',
        'coupling_multipole_error', 'coupling_multipole_order', 'coupling_multipole_hierarchy', 'coupling_multipole_reuse', 'coupling_multipole_compress', 'coupling_multipole_global_actions', 'coupling_multipole_pair_tree', 'coupling_multipole_partition_bytes',
        'image_candidate_cap', 'compute_metric', 'metric_assembly_rank')}
    if info.accuracy.enabled:
        fitting['controls']['coupling_multipole_error_requested'] = options.fitting.coupling_multipole_error
        fitting['controls']['coupling_multipole_error'] = info.accuracy.selected_coupling_error
    fitting['controls']['coupling_multipole_extents'] = {name: getattr(options.fitting.coupling_multipole_extents, name)
        for name in ('pointwise_tolerance', 'tail_mass_tolerance', 'minimum_radius', 'maximum_radius')}
    fitting['projection'] = {name: getattr(options.ri, name) for name in (
        'absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block')}
    fitting['plan'] = {name: getattr(plan, name) for name in (
        'has_j', 'has_k', 'channels', 'n_auxiliary', 'retained_source_bytes', 'modeled_peak_bytes', 'metric_shell_calls',
        'metric_primitive_products', 'build_shell_calls', 'build_primitive_products')}
    if info.domain_trials:
        for name in ('pair_cutoff_bohr', 'auxiliary_cutoff_bohr'):
            fitting['controls'][name + '_requested'] = fitting['controls'][name]
            fitting['controls'][name] = getattr(info.domain_trials[-1], name)
    if info.selected_fitting_plan is not None:
        fitting['plan']['qualification'] = 'maximum-admitted-fitting-source-capacity'
        fitting['selected_plan'] = {name: getattr(info.selected_fitting_plan, name) for name in (
            'n_auxiliary', 'retained_source_bytes', 'modeled_peak_bytes', 'metric_shell_calls', 'metric_primitive_products',
            'build_shell_calls', 'build_primitive_products', 'scalar_terms', 'kernel_terms')}
    if info.fitting_trials:
        fitting['projection']['requested_absolute_cutoff'] = options.ri.absolute_cutoff
        fitting['projection']['requested_relative_cutoff'] = options.ri.relative_cutoff
        if (info.auxiliary_trials[-1].source.metric_refinement > 0 if info.auxiliary_trials else
                info.domain_trials[-1].metric_refinement > 0 if info.domain_trials else len(info.fitting_trials) > 1):
            fitting['projection']['absolute_cutoff'] = info.metrics[0].cutoff
            fitting['projection']['relative_cutoff'] = 0.
    return fitting


def _assembly_roundoff(receipt):
    return dict({name: getattr(receipt, name) for name in (
        'fitted_exchange_roundoff_included', 'fitted_exchange_fock_bound', 'fitted_exchange_energy_bound',
        'exchange_correction_roundoff_included', 'exchange_correction_fock_bound', 'exchange_correction_energy_bound',
        'evaluated', 'paired_coulomb_included', 'retained_coulomb_included', 'direct_exchange_roundoff_included',
        'direct_exchange_fock_bound', 'direct_exchange_energy_bound', 'fock_bound', 'energy_bound',
        'combined_fock_bound', 'combined_energy_bound')},
        scope='stored-physical-components-and-evaluated-fock',
        source_construction_included=False, relaxed_scf_included=False)


def periodic_ks_provenance(result, mesh, options, python_bytes):
    """Executed KS sources, including grid, exchange and evaluated epoch.

    Source ownership remains with the method. A solver may consume each
    fixed spin/k Fock and overlap without reconstructing density from the
    terminal orbitals or interpreting a geometry label as source symmetry.
    """
    fields = ('provider', 'algorithm', 'boundary', 'source_identity', 'overlap_identity',
              'kinetic_identity', 'nuclear_identity', 'nuclear_energy_identity',
              'final_density_identity', 'final_fock_identity', 'confirmed',
              'nuclear_real', 'nuclear_reciprocal', 'nuclear_self', 'nuclear_background',
              'nuclear_energy', 'one_electron_energy', 'coulomb_energy', 'xc_energy', 'energy',
              'confirmation_fock_error', 'confirmation_energy_error', 'maximum_grid_population_error',
              'nuclear_pair_images', 'fock_calls', 'shell_calls', 'primitive_pairs', 'primitive_points',
              'modeled_peak_bytes', 'domain_selection', 'domain_identity', 'reciprocal_energy_cutoff',
              'xc_image_cutoff_bohr', 'domain_translations', 'domain_hartree_vectors',
              'domain_xc_images', 'domain_conjugacy_vectors', 'grid_radial_points', 'grid_lebedev_order',
              'grid_partition_radius_bohr', 'grid_stratmann', 'domain_exchange_vectors',
              'exact_exchange_fraction', 'uncorrected_exchange_energy', 'exchange_correction_energy',
              'exchange_energy', 'exchange_q0', 'exchange_correction_identity', 'madelung',
              'madelung_real_vectors', 'madelung_reciprocal_vectors', 'range_separated',
              'long_range_exchange_fraction', 'short_range_exchange_fraction', 'exchange_q0_requested')
    receipt, plan, snapshot = result.receipt, result.plan, result.solver_snapshot
    evidence = {name: getattr(receipt, name) for name in fields}
    evidence['assembly_roundoff'] = _assembly_roundoff(receipt.assembly_roundoff)
    fitted = receipt.fitting_info is not None or receipt.algorithm == 'finite-3d-ri-range-separated-ks'
    evidence.update({name: getattr(receipt, name) for name in (
        'coulomb_algorithm', 'exchange_algorithm', 'coulomb_identity', 'exchange_identity')})
    evidence['hartree'] = None if receipt.coulomb_algorithm == 'ri' else {name: getattr(receipt.hartree, name) for name in (
        'provider', 'provider_version', 'normalization', 'algorithm', 'boundary', 'source_identity',
        'density_identity', 'energy', 'shell_calls', 'primitive_pairs', 'contraction_terms')}
    evidence['xc'] = {name: getattr(receipt.xc, name) for name in (
        'algorithm', 'functional', 'backend_version', 'component_ids', 'component_weights',
        'source_identity', 'density_identity', 'snapshot_identity', 'convention', 'support',
        'energy', 'electrons_grid', 'minimum_density', 'rounded_negative_points', 'functional_batches',
        'semilocal_component_only', 'required_exact_exchange_fraction')}
    range_separated = receipt.range_separated
    hybrid = range_separated or receipt.exact_exchange_fraction != 0.
    physical = options.ks if hybrid or fitted else options
    methods = ['manta_periodic_range_separated_ks' if range_separated else
               'manta_periodic_global_hybrid_ks' if hybrid else 'manta_periodic_ks']
    if fitted:
        methods = ['manta_periodic_ri_range_separated_ks' if range_separated else
                   'manta_periodic_ri_global_hybrid_ks' if hybrid else 'manta_periodic_ri_ks']
    def fitted_receipt(child):
        return dict({name: getattr(child, name) for name in (
            'has_j', 'has_k', 'frame_identity', 'weight_identity', 'source_identity', 'density_identity',
            'snapshot_identity', 'exchange_scale', 'one_electron_energy', 'coulomb_energy', 'exchange_energy',
            'electronic_energy', 'shell_calls', 'primitive_products', 'coupling_passes',
            'reference_coupling_identity', 'coupling_error_bound', 'approximation_j_bound', 'approximation_energy_bound',
            'weighted_input_roundoff_evaluated', 'weighted_input_roundoff_bound', 'coupling_forward_input_error_included', 'paired_input_roundoff_included',
            'forward_metric_roundoff_evaluated', 'forward_metric_coefficient_roundoff_bound', 'forward_metric_energy_roundoff_bound', 'paired_roundoff_evaluated', 'paired_j_roundoff_bound', 'paired_energy_roundoff_bound',
            'fitted_exchange_roundoff_evaluated', 'fitted_exchange_roundoff_bound',
            'retained_j_roundoff_evaluated', 'retained_j_roundoff_bound', 'retained_energy_roundoff_bound',
        'coupling_roundoff_evaluated', 'coupling_forward_roundoff_bound', 'coupling_adjoint_roundoff_bound', 'metric_roundoff_evaluated', 'metric_j_roundoff_evaluated', 'metric_j_roundoff_bound', 'metric_projection_roundoff_bound', 'metric_coefficient_roundoff_bound', 'metric_energy_roundoff_bound',
            'coupling_multipole_partition_builds',
            'coupling_multipole_partition_reuses',
            'coupling_multipole_partition_bypasses',
            'coupling_multipole_trials', 'coupling_multipole_images',
            'coupling_multipole_tree_nodes', 'coupling_multipole_far_blocks', 'coupling_multipole_translations',
            'coupling_multipole_moment_panels', 'coupling_multipole_envelope_evaluations',
            'coupling_multipole_reused_moment_panels', 'coupling_multipole_reused_extents',
            'coupling_multipole_global_pairs',
            'coupling_multipole_global_pair_reuses',
            'coupling_multipole_global_auxiliary',
            'coupling_multipole_global_projections', 'coupling_multipole_global_auxiliary_projections', 'coupling_multipole_pair_groups',
            'coupling_multipole_compressed_pairs', 'coupling_multipole_compressed_nodes', 'coupling_multipole_action_products', 'coupling_multipole_action_translations')}, component_only=True)
    if receipt.fitting_info is not None:
        evidence['fitting'] = _periodic_fitting_provenance(receipt.fitting_info, plan.fitting, options, receipt.accuracy_density, plan.fitting_accuracy)
        evidence['fitted_fock'] = fitted_receipt(receipt.fitted_fock)
    if fitted:
        evidence['fitting_density_embedding'] = ('equal-spin-halves-of-restricted-total'
                                                 if plan.spins == 1 else 'collinear-spin-densities')
    def exchange_receipt(child):
        return {name: getattr(child, name) for name in (
            'roundoff_evaluated', 'roundoff_scope', 'roundoff_bound', 'energy_roundoff_bounds',
            'provider', 'provider_version', 'normalization', 'algorithm', 'boundary', 'convention',
            'kernel', 'omega', 'zero_vectors', 'source_identity', 'density_identity', 'density_identities',
            'energies', 'transfer_vectors', 'panel_calls', 'shell_calls', 'skipped_shell_pairs',
            'primitive_pairs', 'vector_primitive_products', 'phase_products', 'phase_accumulations',
            'matrix_products', 'contraction_terms', 'modeled_peak_bytes', 'cell_volume',
            'density_hermiticity_residual', 'response_hermiticity_residual')}
    if range_separated:
        evidence['exchange_components'] = []
        for c, p in zip(receipt.exchange_components, plan.exchange_components):
            component = dict(kernel=c.kernel, coefficient=c.coefficient, omega=c.omega,
                energy=c.energy, energy_scale=c.coefficient/(2 if plan.spins == 1 else 1),
                exchange=exchange_receipt(c.exchange) if c.fitted_fock is None else None)
            if c.fitted_fock is not None:
                component.update(energy_scale=1., fitting=_periodic_fitting_provenance(c.fitting_info, p.fitting, options),
                    fitted_fock=fitted_receipt(c.fitted_fock))
                component['fitting'].update(exchange_kernel=c.fitting_info.exchange_kernel, physical_omega=c.fitting_info.physical_omega)
            evidence['exchange_components'].append(component)
        evidence['xc']['required_exchange_components'] = [dict(kernel=k, coefficient=a, omega=w)
            for k, a, w in receipt.xc.required_exchange_components]
        evidence['exchange_zero_mode_convention'] = 'canonical-coulomb-plus-erfc;finite-screened-q0'
    elif hybrid and receipt.exchange is not None:
        evidence['exchange'] = exchange_receipt(receipt.exchange)
        # Child K energies retain -Tr(D K[D])/2 per channel. Only the
        # method-level energies include the hybrid and restricted factors.
        evidence['exchange_energy_scale'] = receipt.exact_exchange_fraction/(2 if plan.spins == 1 else 1)
    if hybrid:
        evidence['madelung_plan'] = {name: getattr(plan.madelung, name) for name in (
            'alpha', 'real_cutoff_bohr', 'reciprocal_cutoff_bohr_inv', 'real_candidates',
            'reciprocal_candidates', 'modeled_peak_bytes', 'kernel_terms')}
        evidence.update(madelung_alpha_requested=options.madelung_alpha,
                        madelung_tolerance=options.madelung_tolerance,
                        exchange_correction_kernel_terms_per_call=plan.exchange_correction_kernel_terms_per_call,
                        provider_kernel_terms_per_call=plan.provider_kernel_terms_per_call)
        if receipt.exchange_q0 == 'BvK-ewald':
            methods.append('manta_periodic_madelung')
    numerics = ['manta_periodic_semilocal_xc']
    if receipt.assembly_roundoff.evaluated:
        numerics.append('manta_periodic_assembly_roundoff')
        if receipt.assembly_roundoff.retained_coulomb_included:
            numerics.append('manta_periodic_retained_roundoff')
        if receipt.assembly_roundoff.fitted_exchange_roundoff_included:
            numerics.append('manta_periodic_fitted_exchange_roundoff')
        if receipt.assembly_roundoff.exchange_correction_roundoff_included:
            numerics.append('manta_periodic_correction_roundoff')
        if receipt.assembly_roundoff.direct_exchange_roundoff_included:
            numerics.append('manta_periodic_exchange_roundoff')
    if receipt.range_exchange_accuracy is not None:
        bound = receipt.range_exchange_accuracy
        range_evidence = {key: getattr(bound, key) for key in (
            'qualification', 'certificate_identity', 'coefficients', 'maximum_error', 'energy_error_bound', 'qualified')}
        range_evidence['targets'] = _exchange_accuracy_targets(bound.targets)
        range_evidence['error_bounds'] = bound.error_bounds.tolist()
        range_evidence['components'] = [_exchange_fitting_evidence(c, p.exchange_fitting_accuracy,
            r.fitted_fock, scale, None) for c, p, r, scale in zip(bound.components,
            plan.exchange_components, receipt.exchange_components, bound.coefficients)]
        range_evidence['scope'] = 'sum of physical signed exchange components; fixed canonical spin family; separate correction and arithmetic'
        range_evidence['exclusions'] = ['source construction', 'BvK/source correction', 'response and assembly arithmetic',
                                        'rank perturbation', 'XC and other Fock components', 'SCF relaxation']
        evidence['range_exchange_accuracy'] = range_evidence
        numerics.append('manta_periodic_ri_exchange_fitting')
    if receipt.exchange_fitting_accuracy is not None:
        evidence['exchange_fitting_accuracy'] = _periodic_exchange_fitting_accuracy(result, options)
        numerics.append('manta_periodic_ri_exchange_fitting')
    if receipt.nuclear_accuracy is not None:
        evidence['nuclear_accuracy'] = _periodic_nuclear_accuracy(result)
        numerics.append('manta_periodic_nuclear_bounds')
    if receipt.exchange_accuracy is not None:
        evidence['exchange_accuracy'] = _periodic_exchange_accuracy(result)
        numerics.append('manta_periodic_exchange_bounds')
    if receipt.one_electron_accuracy is not None:
        evidence['one_electron_accuracy'] = _periodic_one_electron_accuracy(result)
        numerics.append('manta_periodic_one_electron_images')
    if fitted:
        numerics.extend(('manta_periodic_ri_fock', 'manta_hermitian_spectrum', 'manta_hermitian_inverse'))
        if options.fitting.metric_assembly_rank:
            numerics.extend(('manta_bloch_phase_enclosure', 'manta_gdf_weight_enclosure', 'manta_auxiliary_fourier_enclosure', 'manta_metric_integral_enclosure', 'manta_representation_enclosure'))
        if receipt.fitting_info is not None and receipt.fitting_info.fitting_bound is not None:
            numerics.append('manta_periodic_ri_fitting_accuracy')
            if receipt.fitting_info.fitting_bound.metric_action_requested:
                numerics.append('manta_hermitian_inverse_action')
        if options.fitting.coupling_multipole_error:
            numerics.append('manta_periodic_ri_multipole')
    if receipt.quadrature is not None:
        evidence['quadrature'] = {name: getattr(receipt.quadrature, name) for name in (
            'algorithm', 'source_identity', 'nonzero_weights', 'weight_sum', 'cell_volume')}
        evidence['quadrature'].update(grid_points=plan.quadrature.grid_points,
                                      threads=plan.quadrature.threads)
        numerics.append('manta_periodic_quadrature')
        if receipt.grid_stratmann:
            numerics.append('stratmann_grid')
    populations = [plan.n_alpha, plan.n_beta]
    evidence.update(schema_version=1, citation_methods=methods, citation_numerics=numerics,
                    converged=result.converged, status=snapshot.status.name,
                    iterations=snapshot.evaluated_iteration,
                    operator_kind='ks', operator_generation=snapshot.evaluated_iteration,
                    basis_frame_identity=receipt.overlap_identity, operator_layout='spin,k,ao,ao',
                    spin_labels=['restricted-total'] if plan.spins == 1 else ['alpha', 'beta'],
                    spin_electrons_per_cell=populations,
                    occupation_policy='fixed-spin-integer-uniform-full-mesh',
                    exchange_q0_applicability='active' if receipt.exchange_q0 != 'not-applicable' else 'inactive',
                    energy_convention='nuclear + one_electron + coulomb + xc + exchange; '
                                      'xc is semilocal; exchange = uncorrected_exchange + exchange_correction; '
                                      'method exchange energies already scaled',
                    accuracy_qualification='finite-source-only', symmetry_reduced=False,
                    commutator_frobenius_rms=snapshot.commutator_frobenius_rms,
                    maximum_density_closure_frobenius=snapshot.maximum_density_closure_frobenius,
                    maximum_electron_count_error=snapshot.maximum_electron_count_error,
                    global_band_gap=snapshot.global_band_gap, retained_rank=result.solver_memory.retained_rank,
                    mesh=list(mesh.mesh), is_shift=list(mesh.is_shift),
                    alpha=physical.alpha, nuclear_cutoff_bohr=physical.nuclear_cutoff_bohr,
                    pair_cutoff_bohr=physical.pair_cutoff_bohr, python_reserved_bytes=python_bytes,
                    reciprocal_batch=physical.reciprocal_batch, grid_batch=physical.grid_batch,
                    scalar_terms=plan.scalar_terms, kernel_terms=plan.kernel_terms,
                    domain_live_bytes=plan.domain_live_bytes, domain_scalar_terms=plan.domain_scalar_terms,
                    domain_kernel_terms=plan.domain_kernel_terms, quadrature_live_bytes=plan.quadrature_live_bytes,
                    scf_history_steps=snapshot.evaluated_iteration, scf_history_capacity=plan.history_capacity,
                    scf_history_reserved_bytes=plan.history_live_bytes)
    if fitted:
        evidence['component_ledger'] = compose_fixed_density_ledger(evidence, method='rks' if plan.spins == 1 else 'uks')
    if plan.spins == 2:
        evidence['spin_band_gaps'] = [gap if occupied else None for gap, occupied in
                                      zip(snapshot.spin_band_gap, populations)]
    return evidence


def run_periodic_uhf(basis, system, mesh, reciprocal_vectors, exchange_vectors,
                     transfer_offsets, translations, *, options=None, integral_engine):
    """Run fixed-spin neutral all-electron UHF on explicit finite 3D supports."""
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, "UHF")
    result = _core._manta.run_periodic_uhf(basis, system, mesh, reciprocal_vectors,
        exchange_vectors, transfer_offsets, translations, options, budget)
    return _record_periodic_uhf(result, mesh, options, python_bytes)


def _auxiliary_selection_export_bytes(accuracy):
    policy = accuracy.auxiliary_selection
    if policy is None:
        return 0
    domains = accuracy.domain_selection
    trials = (3+policy.maximum_refinements)*(1+(domains.maximum_refinements if domains else 0))*(1+accuracy.maximum_metric_refinements)
    metadata = policy.prepared.plan.modeled_peak_bytes if policy.prepared is not None else policy.generation_limits.memory_bytes
    return 67584*trials + metadata


def _periodic_ri_ks_configuration(options, integral_engine, mesh, hybrid):
    method = 'RIGlobalHybridKS' if hybrid else 'RIKS'
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, method)
    channels = 1 + (len(mesh)*(len(mesh)-1) if hybrid and options.exchange_algorithm == 'ri' else 0)
    extra = 32768 + 18432*channels
    if (options.fitting_accuracy.reference is not None or options.fitting_accuracy.reference_selection is not None):
        extra += 73728 + (18432*(1+options.fitting_accuracy.maximum_metric_refinements) if options.fitting_accuracy.maximum_metric_refinements else 0)
        if options.fitting_accuracy.reference_selection is not None:
            extra += 18432*(2+options.fitting_accuracy.reference_selection.maximum_refinements)
        if options.fitting_accuracy.domain_selection is not None:
            extra += 34816*(2+options.fitting_accuracy.domain_selection.maximum_refinements)*(1+options.fitting_accuracy.maximum_metric_refinements)
        extra += _auxiliary_selection_export_bytes(options.fitting_accuracy)
    range_policy = getattr(options, 'range_exchange_accuracy', None)
    if options.exchange_fitting_accuracy.matrix_error > 0 or (range_policy is not None and range_policy.matrix_error > 0):
        extra += 65536  # per-entry exports are included in native capacity
    if options.nuclear_accuracy.matrix_error > 0:
        extra += 49152
    if options.exchange_accuracy.matrix_error > 0:
        extra += 49152
    if options.one_electron_accuracy.overlap_error > 0:
        extra += 32768
        if options.one_electron_accuracy.overlap_image_rank:
            extra += 4096*len(mesh)  # per-k spectrum dictionaries and serialized copies
    if extra + budget.other_live_bytes > budget.memory_bytes:
        raise ValueError('MANTA periodic RI KS Python provenance memory budget exceeded')
    budget.other_live_bytes += extra
    return options, budget, python_bytes + extra


def _record_periodic_ri_ks(result, mesh, options, python_bytes, hybrid):
    from .output import record

    evidence = periodic_ks_provenance(result, mesh, options, python_bytes)
    profile = 'ri_global_hybrid_ks' if hybrid else 'ri_ks'
    method = 'RKS' if result.plan.spins == 1 else 'UKS'
    record('manta_periodic_'+profile, f'  Finite periodic RI {method}: manta; '
           f'J={result.receipt.coulomb_algorithm}; K={result.receipt.exchange_algorithm}; '
           f'converged={result.converged}; E={result.receipt.energy:.12f}.\n', integral_provenance=evidence)
    return result


def run_periodic_ri_ks(basis, auxiliary, system, mesh, functional, reciprocal_vectors, translations,
                       xc_images, points, quadrature_weights, *, options=None, integral_engine):
    """Complete finite pure LDA/GGA RKS/UKS with fixed RI J and explicit XC sources."""
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, False)
    result = _core._manta.run_periodic_ri_ks(basis, auxiliary, system, mesh, functional, reciprocal_vectors,
        translations, xc_images, points, quadrature_weights, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, False)


def run_periodic_ri_global_hybrid_ks(basis, auxiliary, system, mesh, functional, reciprocal_vectors,
                                     exchange_vectors, transfer_offsets, translations, xc_images, points,
                                     quadrature_weights, *, options=None, integral_engine):
    """Complete full-Coulomb global RKS/UKS with independent exact/RI J/K sources.

    RI K uses G+k_source-k_target and empty diagonal segments; exact K uses
    G+k_target-k_source. The method owns the exchange fraction and correction.
    Numerical fitting omega does not select a physical exchange range.
    """
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, True)
    result = _core._manta.run_periodic_ri_global_hybrid_ks(basis, auxiliary, system, mesh, functional,
        reciprocal_vectors, exchange_vectors, transfer_offsets, translations, xc_images, points,
        quadrature_weights, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, True)


def _periodic_ri_range_ks_configuration(options, integral_engine, mesh, functional):
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, 'RIRangeSeparatedKS')
    # Each resolved erf can add a Coulomb term; merging all such terms leaves
    # at most one more kernel than the functional's original component count.
    channels = (1 if options.coulomb_algorithm == 'ri' else 0)
    if options.exchange_algorithm == 'ri':
        channels += (len(functional.exchange_components)+1)*(1+len(mesh)*(len(mesh)-1))
    extra = 32768+18432*channels
    if (options.fitting_accuracy.reference is not None or options.fitting_accuracy.reference_selection is not None):
        extra += 73728 + (18432*(1+options.fitting_accuracy.maximum_metric_refinements) if options.fitting_accuracy.maximum_metric_refinements else 0)
        if options.fitting_accuracy.reference_selection is not None:
            extra += 18432*(2+options.fitting_accuracy.reference_selection.maximum_refinements)
        if options.fitting_accuracy.domain_selection is not None:
            extra += 34816*(2+options.fitting_accuracy.domain_selection.maximum_refinements)*(1+options.fitting_accuracy.maximum_metric_refinements)
        extra += _auxiliary_selection_export_bytes(options.fitting_accuracy)
    if options.nuclear_accuracy.matrix_error > 0:
        extra += 49152
    if options.exchange_accuracy.matrix_error > 0:
        extra += 49152
    if options.one_electron_accuracy.overlap_error > 0:
        extra += 32768
        if options.one_electron_accuracy.overlap_image_rank:
            extra += 4096*len(mesh)  # per-k spectrum dictionaries and serialized copies
    if extra+budget.other_live_bytes > budget.memory_bytes:
        raise ValueError('MANTA periodic RI range KS Python provenance memory budget exceeded')
    budget.other_live_bytes += extra
    return options, budget, python_bytes+extra


def _record_periodic_ri_range_ks(result, mesh, options, python_bytes):
    from .output import record

    evidence = periodic_ks_provenance(result, mesh, options, python_bytes)
    record('manta_periodic_ri_range_separated_ks', f'  Finite periodic RI range KS: manta; '
           f'J={result.receipt.coulomb_algorithm}; K={result.receipt.exchange_algorithm}; '
           f'converged={result.converged}; E={result.receipt.energy:.12f}.\n', integral_provenance=evidence)
    return result


def run_periodic_ri_range_separated_ks(basis, auxiliary, system, mesh, functional, reciprocal_vectors,
        exchange_vectors, transfer_offsets, translations, xc_images, points, quadrature_weights, *, options=None, integral_engine):
    """Complete finite range KS with separate bare-Coulomb J and physical K fits.

    The functional resolves into Coulomb plus distinct erfc components before
    fitting. Signed method coefficients scale positive-kernel responses; no
    screened metric is reused for J. RI K uses G+k_source-k_target, with empty
    diagonal segments. Exact K keeps G+k_target-k_source and primed diagonals.
    """
    options, budget, python_bytes = _periodic_ri_range_ks_configuration(options, integral_engine, mesh, functional)
    result = _core._manta.run_periodic_ri_range_separated_ks(basis, auxiliary, system, mesh, functional,
        reciprocal_vectors, exchange_vectors, transfer_offsets, translations, xc_images, points,
        quadrature_weights, options, budget)
    return _record_periodic_ri_range_ks(result, mesh, options, python_bytes)


def run_periodic_ri_ks_from_cutoffs(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, points, quadrature_weights, *, options=None, integral_engine):
    """Certified physical cutoffs with caller-supplied quadrature."""
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, False)
    result = _core._manta.run_periodic_ri_ks_from_cutoffs(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, points, quadrature_weights, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, False)


def run_periodic_ri_ks_with_quadrature(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, grid_options, *, options=None, integral_engine):
    """Jointly admitted atom-centred quadrature, sources and functional SCF."""
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, False)
    result = _core._manta.run_periodic_ri_ks_with_quadrature(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, grid_options, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, False)


def run_periodic_ri_global_hybrid_ks_from_cutoffs(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, points, quadrature_weights, *, options=None, integral_engine):
    """Certified physical cutoffs with caller-supplied quadrature."""
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, True)
    result = _core._manta.run_periodic_ri_global_hybrid_ks_from_cutoffs(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, points, quadrature_weights, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, True)


def run_periodic_ri_global_hybrid_ks_with_quadrature(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, grid_options, *, options=None, integral_engine):
    """Jointly admitted atom-centred quadrature, sources and functional SCF."""
    options, budget, python_bytes = _periodic_ri_ks_configuration(options, integral_engine, mesh, True)
    result = _core._manta.run_periodic_ri_global_hybrid_ks_with_quadrature(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, grid_options, options, budget)
    return _record_periodic_ri_ks(result, mesh, options, python_bytes, True)


def run_periodic_ri_range_separated_ks_from_cutoffs(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, points, quadrature_weights, *, options=None, integral_engine):
    """Certified physical cutoffs with caller-supplied quadrature."""
    options, budget, python_bytes = _periodic_ri_range_ks_configuration(options, integral_engine, mesh, functional)
    result = _core._manta.run_periodic_ri_range_separated_ks_from_cutoffs(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, points, quadrature_weights, options, budget)
    return _record_periodic_ri_range_ks(result, mesh, options, python_bytes)


def run_periodic_ri_range_separated_ks_with_quadrature(basis, auxiliary, system, mesh, functional, reciprocal_energy_cutoff,
        xc_image_cutoff_bohr, grid_options, *, options=None, integral_engine):
    """Jointly admitted atom-centred quadrature, sources and functional SCF."""
    options, budget, python_bytes = _periodic_ri_range_ks_configuration(options, integral_engine, mesh, functional)
    result = _core._manta.run_periodic_ri_range_separated_ks_with_quadrature(basis, auxiliary, system, mesh, functional,
        reciprocal_energy_cutoff, xc_image_cutoff_bohr, grid_options, options, budget)
    return _record_periodic_ri_range_ks(result, mesh, options, python_bytes)


def _exchange_accuracy_targets(options):
    return {name: getattr(options, name) for name in (
        'matrix_error', 'energy_error', 'reference_ao_cube_radius', 'reference_reciprocal_cube_radius',
        'ao_completion_cube_radius', 'maximum_probes', 'reciprocal_batch', 'fourier_decay_bins',
        'metric_assembly_matrix_error', 'metric_assembly_energy_error')}


def _periodic_exchange_fitting_accuracy(result, options):
    return _exchange_fitting_evidence(result.receipt.exchange_fitting_accuracy, result.plan.exchange_fitting_accuracy,
        result.receipt.fitted_fock, getattr(result.receipt, 'exact_exchange_fraction', 1.), options.exchange_fitting_accuracy)


def _exchange_fitting_evidence(r, plan, response, scale, targets):
    evidence = {name: getattr(r, name) for name in (
        'kernel', 'physical_omega', 'qualification', 'canonical_density_envelope_identity', 'ri_source_identity', 'reference_source_identity',
        'finite_reference_source_identity', 'pair_source_identity', 'certificate_identity',
        'probe_roundoff_included', 'source_construction_included', 'targets_requested',
        'qualified', 'density_l1_limit', 'maximum_error', 'energy_error_bound',
        'finite_difference_maximum', 'reference_remainder_maximum',
        'finite_difference_energy_bound', 'reference_remainder_energy_bound', 'probes')}
    evidence['targets'] = _exchange_accuracy_targets(targets) if targets is not None else None
    evidence['plan'] = {name: getattr(plan, name) for name in (
        'modeled_peak_bytes', 'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms')}
    evidence['error_bounds'] = r.error_bounds.tolist()
    evidence['metric_assembly'] = {name: getattr(r, 'metric_assembly_' + name) for name in (
        'requested', 'qualified', 'matrix_bound', 'energy_bound', 'identity', 'channels', 'error_bounds')}
    evidence['metric_assembly']['scope'] = 'conditional metric-source response; fixed stored coupling; raw physical K component; unweighted spin-density family'
    evidence['metric_assembly']['excludes'] = ['coupling evaluation and basis/vector data generation', 'coupling construction',
        'metric domain/screening perturbations', 'stored factor error', 'executed K/Fock/energy arithmetic',
        'BvK/source correction', 'total relaxed energy']
    evidence['response_roundoff_bound'] = response.fitted_exchange_roundoff_bound
    evidence['response_roundoff_evaluated'] = response.fitted_exchange_roundoff_evaluated
    evidence['exchange_fraction'] = scale
    def scale_up(bound):
        return math.nextafter(abs(scale)*bound, math.inf) if scale and bound else 0.
    evidence['scaled_component_matrix_bound'] = scale_up(r.maximum_error)
    evidence['scaled_component_energy_bound'] = scale_up(r.energy_error_bound)
    evidence['scope'] = ('raw full-Coulomb K component; fixed canonical density family; q0 omitted' if r.kernel == 'coulomb'
                         else 'raw physical range K component; fixed canonical density family; screened zero accounted')
    evidence['exclusions'] = ['source construction', 'BvK/source correction', 'other Fock components', 'SCF relaxation']
    return evidence


def _periodic_ri_hf_provenance(result, mesh, options, python_bytes, method):
    """Method energies and the frozen whole-mesh fit, separate from its epoch."""
    provenance = periodic_rhf_provenance if method == 'rhf' else periodic_uhf_provenance
    evidence = provenance(result, mesh, options, python_bytes)
    evidence['assembly_roundoff'] = _assembly_roundoff(result.receipt.assembly_roundoff)
    evidence['citation_methods'][0] = f'manta_periodic_ri_{method}'
    evidence['citation_numerics'] = ['manta_periodic_ri_fock', 'manta_hermitian_spectrum', 'manta_hermitian_inverse']
    if options.fitting.metric_assembly_rank:
        evidence['citation_numerics'].extend(('manta_bloch_phase_enclosure', 'manta_gdf_weight_enclosure', 'manta_auxiliary_fourier_enclosure', 'manta_metric_integral_enclosure', 'manta_representation_enclosure'))
    if result.receipt.assembly_roundoff.evaluated:
        evidence['citation_numerics'].append('manta_periodic_assembly_roundoff')
        if result.receipt.assembly_roundoff.retained_coulomb_included:
            evidence['citation_numerics'].append('manta_periodic_retained_roundoff')
        if result.receipt.assembly_roundoff.fitted_exchange_roundoff_included:
            evidence['citation_numerics'].append('manta_periodic_fitted_exchange_roundoff')
        if result.receipt.assembly_roundoff.exchange_correction_roundoff_included:
            evidence['citation_numerics'].append('manta_periodic_correction_roundoff')
        if result.receipt.assembly_roundoff.direct_exchange_roundoff_included:
            evidence['citation_numerics'].append('manta_periodic_exchange_roundoff')
    if result.receipt.nuclear_accuracy is not None:
        evidence['nuclear_accuracy'] = _periodic_nuclear_accuracy(result)
        evidence['citation_numerics'].append('manta_periodic_nuclear_bounds')
    if result.receipt.exchange_fitting_accuracy is not None:
        evidence['exchange_fitting_accuracy'] = _periodic_exchange_fitting_accuracy(result, options)
        evidence['citation_numerics'].append('manta_periodic_ri_exchange_fitting')
    if result.receipt.exchange_accuracy is not None:
        evidence['exchange_accuracy'] = _periodic_exchange_accuracy(result)
        evidence['citation_numerics'].append('manta_periodic_exchange_bounds')
    if result.receipt.one_electron_accuracy is not None:
        evidence['one_electron_accuracy'] = _periodic_one_electron_accuracy(result)
        evidence['citation_numerics'].append('manta_periodic_one_electron_images')
    if options.fitting.coupling_multipole_error:
        evidence['citation_numerics'].append('manta_periodic_ri_multipole')
    info = result.receipt.fitting_info
    _ledger_inputs = dict(method=method)
    if method == 'rhf':
        memory = result.solver_memory
        _ledger_inputs.update(orbital_energies=list(result.orbital_energies), overlaps=list(result.overlap),
                              n_occupied=memory.n_occupied, retained_rank=memory.retained_rank)
    if info.fitting_bound is not None:
        evidence['citation_numerics'].append('manta_periodic_ri_fitting_accuracy')
        if result.receipt.fitting_info.fitting_bound.metric_action_requested:
            evidence['citation_numerics'].append('manta_hermitian_inverse_action')
    fitting = _periodic_fitting_provenance(info, result.plan.fock, options, result.receipt.accuracy_density, result.plan.fitting_accuracy)
    evidence['component_ledger'] = compose_fixed_density_ledger(dict(evidence, fitting=fitting), **_ledger_inputs)
    evidence['coupling_approximation'] = {name: getattr(result.receipt.fitted_fock, name) for name in (
        'reference_coupling_identity', 'coupling_error_bound', 'approximation_j_bound', 'approximation_energy_bound',
            'weighted_input_roundoff_evaluated', 'weighted_input_roundoff_bound', 'coupling_forward_input_error_included', 'paired_input_roundoff_included',
            'forward_metric_roundoff_evaluated', 'forward_metric_coefficient_roundoff_bound', 'forward_metric_energy_roundoff_bound', 'paired_roundoff_evaluated', 'paired_j_roundoff_bound', 'paired_energy_roundoff_bound',
            'fitted_exchange_roundoff_evaluated', 'fitted_exchange_roundoff_bound',
            'retained_j_roundoff_evaluated', 'retained_j_roundoff_bound', 'retained_energy_roundoff_bound',
        'coupling_roundoff_evaluated', 'coupling_forward_roundoff_bound', 'coupling_adjoint_roundoff_bound', 'metric_roundoff_evaluated', 'metric_j_roundoff_evaluated', 'metric_j_roundoff_bound', 'metric_projection_roundoff_bound', 'metric_coefficient_roundoff_bound', 'metric_energy_roundoff_bound',
            'coupling_multipole_partition_builds',
            'coupling_multipole_partition_reuses',
            'coupling_multipole_partition_bypasses',
            'coupling_multipole_trials', 'coupling_multipole_images',
            'coupling_multipole_tree_nodes', 'coupling_multipole_far_blocks', 'coupling_multipole_translations',
            'coupling_multipole_moment_panels', 'coupling_multipole_envelope_evaluations',
            'coupling_multipole_reused_moment_panels', 'coupling_multipole_reused_extents',
            'coupling_multipole_global_pairs',
            'coupling_multipole_global_pair_reuses',
            'coupling_multipole_global_auxiliary',
            'coupling_multipole_global_projections', 'coupling_multipole_global_auxiliary_projections', 'coupling_multipole_pair_groups',
            'coupling_multipole_compressed_pairs', 'coupling_multipole_compressed_nodes', 'coupling_multipole_action_products', 'coupling_multipole_action_translations')}
    evidence['coupling_approximation']['qualification'] = 'fixed-density-same-finite-metric;excludes-roundoff-and-domain-error'
    evidence.update(fitting=fitting, operator_kind='hf',
        operator_layout='k,ao,ao' if method == 'rhf' else 'spin,k,ao,ao',
        operator_generation=result.solver_snapshot.evaluated_iteration,
        basis_frame_identity=result.receipt.overlap_identity,
        accuracy_qualification='finite-fitted-source-only', symmetry_reduced=False,
        energy_convention='nuclear + one_electron + coulomb + exchange; '
                          'exchange = uncorrected_exchange + exchange_correction')
    evidence.update({name: getattr(result.receipt, name) for name in (
        'coulomb_algorithm', 'exchange_algorithm', 'coulomb_identity', 'exchange_identity')})
    evidence['two_electron_plan'] = {name: getattr(result.plan, name) for name in (
        'mixed_driver_live_bytes', 'two_electron_peak_bytes', 'two_electron_kernel_terms_per_call')}
    for name in ('exact_hartree', 'exact_exchange'):
        receipt = getattr(result.receipt, name)
        if receipt is not None:
            evidence['citation_methods'].append('manta_periodic_hartree' if name == 'exact_hartree' else 'manta_periodic_exchange')
            # Fixed scalar/two-spin schema keeps the separate component's
            # evidence within its admitted serialization reservation.
            fields = ('provider', 'provider_version', 'normalization', 'algorithm', 'boundary',
                'source_identity', 'density_identity', 'panel_calls', 'shell_calls',
                'skipped_shell_pairs', 'primitive_pairs', 'vector_primitive_products',
                'phase_products', 'phase_accumulations', 'contraction_terms', 'modeled_peak_bytes',
                'density_hermiticity_residual', 'response_hermiticity_residual', 'cell_volume')
            plan_fields = ('n_basis', 'cells', 'momenta', 'reciprocal_batch', 'shell_calls',
                'primitive_pairs', 'vector_primitive_products', 'contraction_terms', 'scalar_terms',
                'kernel_terms', 'output_bytes', 'driver_live_bytes', 'modeled_peak_bytes')
            if name == 'exact_hartree':
                fields += ('energy', 'cell_imaginary_residual')
                plan_fields += ('vectors', 'batches', 'panel_calls')
            else:
                fields += ('roundoff_evaluated', 'roundoff_scope', 'roundoff_bound', 'energy_roundoff_bounds',
                           'convention', 'kernel', 'omega', 'zero_vectors', 'density_identities',
                           'energies', 'transfer_vectors', 'matrix_products')
                plan_fields += ('roundoff_requested', 'densities', 'vectors_per_pair', 'transfer_pairs',
                                'phase_products', 'phase_accumulations', 'matrix_products')
            evidence[name] = {field: getattr(receipt, field) for field in fields}
            direct_plan = getattr(result.plan, name)
            evidence[name]['plan'] = {field: getattr(direct_plan, field) for field in plan_fields}
    if method == 'rhf':
        evidence.update(spin_labels=['restricted-total'], spin_electrons_per_cell=[result.plan.electrons],
                        occupation_policy='closed-shell-integer-uniform-full-mesh',
                        fitting_density_embedding='equal-spin-halves-of-restricted-total')
    return evidence


def periodic_ri_uhf_provenance(result, mesh, options, python_bytes):
    return _periodic_ri_hf_provenance(result, mesh, options, python_bytes, 'uhf')


def periodic_ri_rhf_provenance(result, mesh, options, python_bytes):
    return _periodic_ri_hf_provenance(result, mesh, options, python_bytes, 'rhf')


def _periodic_ri_hf_configuration(options, integral_engine, mesh, method):
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, method)
    channels = 1 + (len(mesh)*(len(mesh)-1) if options.exchange_algorithm == 'ri' else 0)
    # Retained metric evidence and its output dictionary/serialized owners.
    # The native parent independently includes typed receipt copies.
    extra = 18432*channels
    if (options.fitting_accuracy.reference is not None or options.fitting_accuracy.reference_selection is not None):
        extra += 73728 + (18432*(1+options.fitting_accuracy.maximum_metric_refinements) if options.fitting_accuracy.maximum_metric_refinements else 0)
        if options.fitting_accuracy.reference_selection is not None:
            extra += 18432*(2+options.fitting_accuracy.reference_selection.maximum_refinements)
        if options.fitting_accuracy.domain_selection is not None:
            extra += 34816*(2+options.fitting_accuracy.domain_selection.maximum_refinements)*(1+options.fitting_accuracy.maximum_metric_refinements)
        extra += _auxiliary_selection_export_bytes(options.fitting_accuracy)
    if options.exchange_fitting_accuracy.matrix_error > 0:
        extra += 65536  # per-entry certificate/export storage is in the native plan
    if options.nuclear_accuracy.matrix_error > 0:
        extra += 49152
    if options.exchange_accuracy.matrix_error > 0:
        extra += 49152
    if options.one_electron_accuracy.overlap_error > 0:
        extra += 32768
        if options.one_electron_accuracy.overlap_image_rank:
            extra += 4096*len(mesh)  # per-k spectrum dictionaries and serialized copies
    if options.coulomb_algorithm != 'ri' or options.exchange_algorithm != 'ri':
        extra += 32768
    if extra + budget.other_live_bytes > budget.memory_bytes:
        raise ValueError(f'MANTA periodic {method} Python provenance memory budget exceeded')
    budget.other_live_bytes += extra
    python_bytes += extra
    return options, budget, python_bytes


def _periodic_ri_uhf_configuration(options, integral_engine, mesh):
    return _periodic_ri_hf_configuration(options, integral_engine, mesh, 'RIUHF')


def _periodic_ri_rhf_configuration(options, integral_engine, mesh):
    return _periodic_ri_hf_configuration(options, integral_engine, mesh, 'RIRHF')


def _record_periodic_ri_uhf(result, mesh, options, python_bytes):
    from .output import record

    evidence = periodic_ri_uhf_provenance(result, mesh, options, python_bytes)
    record('manta_periodic_ri_uhf', f'  Finite periodic RI UHF: manta; converged={result.converged}; '
           f'E={result.receipt.energy:.12f}; J={options.coulomb_algorithm}; '
           f'K={options.exchange_algorithm}; exchange {result.receipt.exchange_q0}.\n',
           integral_provenance=evidence)
    return result


def run_periodic_ri_uhf(basis, auxiliary, system, mesh, reciprocal_vectors, pair_vectors,
                        transfer_offsets, translations, *, options=None, integral_engine):
    """Selected neutral 3D UHF with immutable per-transfer Coulomb RI projectors.

    RI K uses G+k_source-k_target with empty diagonal segments; exact K
    uses G+k_target-k_source with explicit nonempty primed diagonals. The
    method owns the physical H/S, energy ledger and optional BvK correction.
    This explicit-source entry does not infer source symmetry or convergence
    of the finite image/reciprocal domains from SCF convergence.
    """
    options, budget, python_bytes = _periodic_ri_uhf_configuration(options, integral_engine, mesh)
    result = _core._manta.run_periodic_ri_uhf(basis, auxiliary, system, mesh, reciprocal_vectors,
        pair_vectors, transfer_offsets, translations, options, budget)
    return _record_periodic_ri_uhf(result, mesh, options, python_bytes)


def run_periodic_ri_uhf_from_cutoffs(basis, auxiliary, system, mesh, reciprocal_energy_cutoff,
                                     *, options, integral_engine):
    """Generate admitted physical-pair/reciprocal spheres and run fitted 3D UHF.

    HF and fitting pair cutoffs must agree. The reciprocal kinetic cutoff
    selects primed Gamma G and the requested RI/exact K transfer spheres.
    Auxiliary/nuclear image radii, fitting omega, metric truncation and
    method correction retain their explicit independent controls.
    """
    options, budget, python_bytes = _periodic_ri_uhf_configuration(options, integral_engine, mesh)
    result = _core._manta.run_periodic_ri_uhf_from_cutoffs(basis, auxiliary, system, mesh,
        reciprocal_energy_cutoff, options, budget)
    return _record_periodic_ri_uhf(result, mesh, options, python_bytes)


def _record_periodic_ri_rhf(result, mesh, options, python_bytes):
    from .output import record

    evidence = periodic_ri_rhf_provenance(result, mesh, options, python_bytes)
    record('manta_periodic_ri_rhf', f'  Finite periodic RI RHF: manta; converged={result.converged}; '
           f'E={result.receipt.energy:.12f}; J={options.coulomb_algorithm}; '
           f'K={options.exchange_algorithm}; exchange {result.receipt.exchange_q0}.\n',
           integral_provenance=evidence)
    return result


def run_periodic_ri_rhf(basis, auxiliary, system, mesh, reciprocal_vectors, pair_vectors,
                        transfer_offsets, translations, *, options=None, integral_engine):
    """Selected neutral 3D RHF with immutable per-transfer Coulomb RI projectors.

    Closed-shell normalization follows Helgaker, Jorgensen and Olsen,
    Molecular Electronic-Structure Theory (2000).
    RI K uses G+k_source-k_target with empty diagonal segments; exact K
    uses G+k_target-k_source with explicit nonempty primed diagonals. The
    method owns the physical H/S, energy ledger and optional BvK correction.
    This explicit-source entry does not infer source symmetry or convergence
    of the finite image/reciprocal domains from SCF convergence.
    """
    options, budget, python_bytes = _periodic_ri_rhf_configuration(options, integral_engine, mesh)
    result = _core._manta.run_periodic_ri_rhf(basis, auxiliary, system, mesh, reciprocal_vectors,
        pair_vectors, transfer_offsets, translations, options, budget)
    return _record_periodic_ri_rhf(result, mesh, options, python_bytes)


def run_periodic_ri_rhf_from_cutoffs(basis, auxiliary, system, mesh, reciprocal_energy_cutoff,
                                     *, options, integral_engine):
    """Generate admitted physical-pair/reciprocal spheres and run fitted 3D RHF.

    HF and fitting pair cutoffs must agree. The reciprocal kinetic cutoff
    selects primed Gamma G and the requested RI/exact K transfer spheres.
    Auxiliary/nuclear image radii, fitting omega, metric truncation and
    method correction retain their explicit independent controls.
    """
    options, budget, python_bytes = _periodic_ri_rhf_configuration(options, integral_engine, mesh)
    result = _core._manta.run_periodic_ri_rhf_from_cutoffs(basis, auxiliary, system, mesh,
        reciprocal_energy_cutoff, options, budget)
    return _record_periodic_ri_rhf(result, mesh, options, python_bytes)


def _record_periodic_uhf(result, mesh, options, python_bytes):
    from .output import record

    evidence = periodic_uhf_provenance(result, mesh, options, python_bytes)
    record("manta_periodic_uhf", f"  Finite periodic UHF: manta; converged={result.converged}; "
           f"E={result.receipt.energy:.12f}; exchange {result.receipt.exchange_q0}.\n",
           integral_provenance=evidence)
    return result


def _record_periodic_rhf(result, mesh, options, python_bytes):
    from .output import record

    evidence = periodic_rhf_provenance(result, mesh, options, python_bytes)
    record("manta_periodic_rhf", f"  Finite periodic RHF: manta; converged={result.converged}; "
           f"E={result.receipt.energy:.12f}; exchange {result.receipt.exchange_q0}.\n",
           integral_provenance=evidence)
    return result


def run_periodic_rhf(basis, system, mesh, reciprocal_vectors, exchange_vectors,
                     transfer_offsets, translations, *, options=None, integral_engine):
    """Run neutral all-electron RHF on an explicit finite 3D regular-mesh source."""
    options, budget, python_bytes = _periodic_rhf_configuration(options, integral_engine)
    result = _core._manta.run_periodic_rhf(basis, system, mesh, reciprocal_vectors,
        exchange_vectors, transfer_offsets, translations, options, budget)
    return _record_periodic_rhf(result, mesh, options, python_bytes)


def run_periodic_rhf_from_cutoffs(basis, system, mesh, reciprocal_energy_cutoff,
                                  *, options, integral_engine):
    """Generate finite physical-pair/reciprocal spheres and run native 3D RHF.

    ``options.pair_cutoff_bohr`` must be explicit. The reciprocal kinetic
    cutoff is in hartree and applies to G and every shifted exchange sphere.
    Nuclear images and the optional BvK correction retain their own controls.
    """
    options, budget, python_bytes = _periodic_rhf_configuration(options, integral_engine)
    result = _core._manta.run_periodic_rhf_from_cutoffs(basis, system, mesh,
        reciprocal_energy_cutoff, options, budget)
    return _record_periodic_rhf(result, mesh, options, python_bytes)


def run_periodic_uhf_from_cutoffs(basis, system, mesh, reciprocal_energy_cutoff,
                                  *, options, integral_engine):
    """Generate admitted physical-pair/reciprocal spheres and run fixed-spin 3D UHF."""
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, "UHF")
    result = _core._manta.run_periodic_uhf_from_cutoffs(basis, system, mesh,
        reciprocal_energy_cutoff, options, budget)
    return _record_periodic_uhf(result, mesh, options, python_bytes)
