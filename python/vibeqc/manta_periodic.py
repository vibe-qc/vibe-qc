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


def _periodic_fitting_provenance(info, plan, options):
    fitting = {name: getattr(info, name) for name in (
        'has_j', 'has_k', 'source_identity', 'frame_identity', 'weight_identity', 'metric_shell_calls',
        'metric_primitive_products', 'reverse_metric_residual', 'reverse_projector_residual',
        'reverse_inverse_residual')}
    fitting['metrics'] = [{name: getattr(metric, name) for name in (
        'rank', 'cutoff', 'negative_limit', 'minimum_eigenvalue', 'maximum_eigenvalue',
        'hermiticity_residual', 'zero_transfer', 'metric_source_identity', 'canonical_metric_identity',
        'factor_identity', 'projector_identity', 'source_identity', 'shell_calls', 'primitive_products')}
        for metric in info.metrics]
    fitting['controls'] = {name: getattr(options.fitting, name) for name in (
        'omega', 'pair_cutoff_bohr', 'auxiliary_cutoff_bohr', 'integral_screen_error', 'coupling_screen_error',
        'image_candidate_cap', 'compute_metric')}
    fitting['projection'] = {name: getattr(options.ri, name) for name in (
        'absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block')}
    fitting['plan'] = {name: getattr(plan, name) for name in (
        'has_j', 'has_k', 'channels', 'retained_source_bytes', 'modeled_peak_bytes', 'metric_shell_calls',
        'metric_primitive_products', 'build_shell_calls', 'build_primitive_products')}
    return fitting


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
            'reference_coupling_identity', 'coupling_error_bound', 'approximation_j_bound', 'approximation_energy_bound')}, component_only=True)
    if receipt.fitting_info is not None:
        evidence['fitting'] = _periodic_fitting_provenance(receipt.fitting_info, plan.fitting, options)
        evidence['fitted_fock'] = fitted_receipt(receipt.fitted_fock)
    if fitted:
        evidence['fitting_density_embedding'] = ('equal-spin-halves-of-restricted-total'
                                                 if plan.spins == 1 else 'collinear-spin-densities')
    def exchange_receipt(child):
        return {name: getattr(child, name) for name in (
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
    if fitted:
        numerics.append('manta_periodic_ri_fock')
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


def _periodic_ri_ks_configuration(options, integral_engine, mesh, hybrid):
    method = 'RIGlobalHybridKS' if hybrid else 'RIKS'
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, method)
    channels = 1 + (len(mesh)*(len(mesh)-1) if hybrid and options.exchange_algorithm == 'ri' else 0)
    extra = 32768 + 12288*channels
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
    extra = 32768+12288*channels
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


def _periodic_ri_hf_provenance(result, mesh, options, python_bytes, method):
    """Method energies and the frozen whole-mesh fit, separate from its epoch."""
    provenance = periodic_rhf_provenance if method == 'rhf' else periodic_uhf_provenance
    evidence = provenance(result, mesh, options, python_bytes)
    evidence['citation_methods'][0] = f'manta_periodic_ri_{method}'
    evidence['citation_numerics'] = ['manta_periodic_ri_fock']
    info = result.receipt.fitting_info
    fitting = _periodic_fitting_provenance(info, result.plan.fock, options)
    evidence['coupling_approximation'] = {name: getattr(result.receipt.fitted_fock, name) for name in (
        'reference_coupling_identity', 'coupling_error_bound', 'approximation_j_bound', 'approximation_energy_bound')}
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
                fields += ('convention', 'kernel', 'omega', 'zero_vectors', 'density_identities',
                           'energies', 'transfer_vectors', 'matrix_products')
                plan_fields += ('densities', 'vectors_per_pair', 'transfer_pairs',
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
    extra = 12288*channels
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
