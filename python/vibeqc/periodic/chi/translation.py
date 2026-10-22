"""Experimental chi sources with direct cyclic overlap and Fock blocks.

This opt-in preparation runs a fresh restricted SCF calculation or reuses
a sealed accepted reference under the existing chi admission rules. It folds
primitive k matrices directly to home-row operators and retains home-auxiliary factors. It does
not construct the dense real reference used by the historical post-HF
wrappers, change their defaults, or broaden chi's scientific qualification.

The finite-group equivalence follows Nejad et al., doi:10.1063/5.0290816,
Sec. II, Eqs. (2)-(6), (15)-(16). The source retains the declared finite
Coulomb and exchange-zero-mode conventions; no thermodynamic-limit
replacement or fragment LNO approximation is introduced.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from math import prod
from numbers import Integral

import numpy as np

from ..._vibeqc_core import BasisSet
from ...aux_basis import default_aux_for, make_aux_basis_set
from ...dlpno.source import GammaReference, SourceConventions, _real_torus_source
from ...dlpno.translation_factors import TranslationHomeFactors
from ...options_dump import _looks_like_options_struct, _walk_attributes
from ...periodic_fit_source import RSGDFFitSource
from . import posthf as p
from .reference import _admit_state, _digest, _options, require_reference
from .scf import _normalise_mesh, cyclic_gamma_mesh

__all__ = ["prepare_chi_translation_source"]


def _copy_option_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_copy_option_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_option_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy_option_value(item) for key, item in value.items()}
    if _looks_like_options_struct(value):
        copied = type(value)()
        for name, item in _walk_attributes(value):
            setattr(copied, name, _copy_option_value(item))
        return copied
    # Native enums are immutable value descriptors. Some do not implement
    # pickle/deepcopy, while native array and list payloads above do own data.
    if hasattr(value, "name") and hasattr(value, "value"):
        return value
    return deepcopy(value)


def _validate_frozen(system, mesh, n_frozen):
    electrons = int(system.n_electrons())
    if electrons <= 0 or electrons % 2:
        raise ValueError("chi translation sources require a positive even electron count")
    n_cells = prod(mesh)
    if (isinstance(n_frozen, (bool, np.bool_)) or not isinstance(n_frozen, Integral)
            or n_frozen < 0 or n_frozen % n_cells or n_frozen >= n_cells*(electrons//2)):
        raise ValueError("n_frozen must contain complete translated bands and leave active occupied orbitals")


def _translation_real_reference(system, basis, mesh, folded, transform_record, cache, auxiliary_name):
    """Finish an admitted Bloch fold with one compact home factor transform."""
    kfrac = np.asarray(cyclic_gamma_mesh(system, mesh).kpoints_frac)
    home, imaginary, symmetry = p._lpq_cache_to_real_home_auxiliary(cache, kfrac, mesh)
    if (not np.isfinite(imaginary) or not np.isfinite(symmetry)
            or max(imaginary, symmetry) > 1e-7):
        raise RuntimeError("chi translation home-auxiliary transform violated real Hermitian symmetry")
    super_system = p._build_supercell_system(system, mesh)
    molecule = super_system.unit_cell_molecule()
    super_basis = BasisSet(molecule, basis.name)
    if int(super_basis.nbasis) != home.shape[1] or home.shape[1:] != folded.overlap.shape:
        raise RuntimeError("chi translation supercell basis ordering does not match the finite-group transform")
    super_auxiliary = make_aux_basis_set(molecule, aux_name=auxiliary_name)
    if int(super_auxiliary.nbasis) != prod(mesh)*home.shape[0]:
        raise RuntimeError("chi translation supercell auxiliary ordering does not match the factor transform")
    factors = TranslationHomeFactors(home, mesh, auxiliary_basis=super_auxiliary)
    return p._BRealReference(super_system, molecule, super_basis, factors, folded, prod(mesh),
                            float(imaginary), float(symmetry),
                            float(transform_record["matrix_imaginary_residual"]))


def _source_from_real(real, *, convention, scf_fit, correlation_fit, fit_source,
                      options_identity, residuals, transform_record, n_frozen,
                      auxiliary_name, accepted_record=None):
    """Share immutable operators/factors and detach all mutable source state."""
    def snapshot(value):
        array = np.ascontiguousarray(value, dtype=float)
        return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)

    original = real.hf
    from ...dlpno.translation_projectors import BlochOccupiedProjectors, ProjectorReference, reference_orbitals
    orbitals = reference_orbitals(original)
    compact = type(orbitals) is BlochOccupiedProjectors
    reference_class = ProjectorReference if compact else GammaReference
    reference = reference_class(orbitals if compact else snapshot(orbitals), original.fock, original.overlap,
                                snapshot(original.mo_energies), float(original.energy),
                                ecp_total_ncore=int(getattr(original, "ecp_total_ncore", 0)))
    factors = real.df
    provenance = {
        "adapter": "chi-translation-blocks-v1", "scf_options_identity": options_identity,
        "scf_fit": scf_fit.identity, "correlation_fit": correlation_fit.identity,
        "correlation_fit_status": "reused" if accepted_record is not None or fit_source == "classic" else "built",
        "finite_torus_convention": asdict(convention),
        "orbital_residuals": dict(residuals), "bloch_transform": deepcopy(dict(transform_record)),
        "n_frozen": int(n_frozen), "factor_identity": factors.identity,
        "factor_imaginary_residual": float(real.cderi_imaginary_residual),
        "factor_symmetry_residual": float(real.cderi_symmetry_residual),
        "quantitative_status": "not-qualified",
    }
    if accepted_record is not None:
        provenance["accepted_reference"] = deepcopy(accepted_record)
    return _real_torus_source(
        real.super_system, real.basis, reference, factors, kind="chi-translation-blocks",
        conventions=SourceConventions(
            exchange_q0=convention.exchange_q0, kmesh=tuple(factors.mesh),
            bloch_normalisation="real-bvk-supercell",
            same_source=bool(fit_source == "classic" and scf_fit.identity == correlation_fit.identity),
        ),
        factor_tensor=factors.home_factors, factor_representation="chi-home-auxiliary",
        aux_basis_name=auxiliary_name, provenance=provenance,
    )


def prepare_chi_translation_source(
    system, basis, mesh=(1, 1, 1), options=None, *,
    aux_basis=None, lattice_cutoff_bohr=15., rsgdf_ke_cutoff=200.,
    fit_source="auto", rsgdf_tail_ke_cutoff=None,
    gdf_linear_dep_threshold=1e-9, n_frozen=0, reference=None, compact_occupied=False,
    progress=None, verbose=None,
):
    """Prepare a blockwise source for experimental translation MP2.

    ``n_frozen`` counts frozen spatial orbitals in the entire finite torus;
    it must comprise complete translated bands. The returned source orders
    frozen occupied orbitals before active occupied orbitals. Pass this same
    total count to ``run_translation_dlpno_mp2``.
    ``compact_occupied=True`` retains frozen/active home-row projectors
    instead of canonical coefficients, replaying the full-spectrum orbital
    audit without assembling the full real canonical coefficient array.
    For a block handle this flag must match its prepared representation.

    ``classic`` reuses attested SCF factors and may report ``same_source``.
    Historical ``auto`` runs its selected SCF and a separate classic
    correlation refit, which remains separately identified. ECP, explicit
    SR, corrected RIJCOSX, unrestricted, and explicit fitted-support sources
    retain their existing chi post-HF refusal. ``reference`` may be a sealed
    ``ChiAcceptedReference``. Block handles reuse all transforms and bind the
    prepared frozen count; dense handles fold their stored Bloch matrices
    directly and reuse compact factors. Every existing handle seal is checked.
    Disk cache transport is not added. Caller-owned options are copied before
    any SCF driver receives them, including on failure paths.
    """
    from ...dlpno.translation_bloch import bloch_translation_reference
    from ...dlpno.translation_projectors import BlochOccupiedProjectors, reference_orbitals

    if type(compact_occupied) is not bool:
        raise ValueError('compact_occupied must be a boolean')
    if int(system.dim) != 3 or int(system.multiplicity) != 1:
        raise NotImplementedError("chi translation sources require 3D restricted closed shell")
    mesh = _normalise_mesh(system, mesh)
    _validate_frozen(system, mesh, n_frozen)
    electrons = int(system.n_electrons())
    p._validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr, rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    auxiliary_name = aux_basis or default_aux_for(basis.name)
    if reference is not None:
        accepted = require_reference(reference, system, basis, mesh, options,
            matrix_representation=None, auxiliary_name=auxiliary_name,
            lattice_cutoff_bohr=float(lattice_cutoff_bohr), rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
            fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
            gdf_linear_dep_threshold=float(gdf_linear_dep_threshold))
        real = accepted.real_reference
        if accepted.matrix_representation == "cyclic-blocks":
            if n_frozen != accepted.n_frozen:
                raise ValueError("chi block reference frozen partition mismatch")
            if compact_occupied != (type(reference_orbitals(real.hf)) is BlochOccupiedProjectors):
                raise ValueError('chi block reference occupied representation mismatch')
            transform_record = accepted.translation_record
        else:
            folded, transform_record = bloch_translation_reference(
                system, basis, accepted.hf_result, mesh, n_frozen=int(n_frozen),
                compact_occupied=compact_occupied)
            real = p._BRealReference(real.super_system, real.molecule, real.basis, real.df,
                folded, real.n_cells, real.cderi_imaginary_residual, real.cderi_symmetry_residual,
                float(transform_record["matrix_imaginary_residual"]))
        return _source_from_real(real, convention=accepted.finite_torus_convention,
            scf_fit=accepted.scf_fit_source, correlation_fit=accepted.correlation_fit_source,
            fit_source=fit_source, options_identity=accepted.options_identity,
            residuals=accepted.orbital_residuals, transform_record=transform_record,
            n_frozen=n_frozen, auxiliary_name=auxiliary_name,
            accepted_record={"seal": accepted.seal, "state_identity": accepted.state_identity,
                "real_identity": accepted.real_identity,
                "matrix_representation": accepted.matrix_representation,
                "scf": "reused", "correlation_fit": "reused", "home_factors": "reused",
                "bloch_transform": "reused" if accepted.matrix_representation == "cyclic-blocks" else "built-from-sealed-bloch"})
    owned_options = _copy_option_value(_options(options))
    options_identity = _digest(owned_options)
    p._admit_selected_reference(fit_source, rsgdf_tail_ke_cutoff, owned_options, lattice_cutoff_bohr)
    hf = p.run_aiccm2026dev_b_rhf(
        system, basis, mesh, owned_options, backend="ri", aux_basis=auxiliary_name,
        gdf_method="rsgdf", rsgdf_ke_cutoff=rsgdf_ke_cutoff, fit_source=fit_source,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold, progress=progress, verbose=verbose,
    )
    residuals = _admit_state(hf, electrons//2)
    convention = p._reference_finite_torus_convention(hf)
    if tuple(convention.character_mesh_shape) != mesh:
        raise ValueError("chi reference convention mesh does not match the requested translation mesh")
    scf_fit = getattr(hf, "executed_fit_source", None)
    if not isinstance(scf_fit, RSGDFFitSource):
        raise ValueError("chi translation source lacks executed SCF source telemetry")
    assembly = getattr(hf, "exact_exchange_assembly", None)
    if (getattr(assembly, "c_full", None) != 1 or getattr(assembly, "c_sr", None) != 0):
        raise ValueError("chi translation correlation requires full RHF exchange")
    folded, transform_record = bloch_translation_reference(
        system, basis, hf, mesh, n_frozen=int(n_frozen), compact_occupied=compact_occupied,
    )
    cache, auxiliary = p._selected_lpq_cache(
        system, basis, np.asarray(cyclic_gamma_mesh(system, mesh).kpoints_cart), auxiliary_name,
        reference=hf, fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        lattice_cutoff_bohr=lattice_cutoff_bohr, rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    correlation_fit = scf_fit if fit_source == "classic" else getattr(hf, "_chi_correlation_fit_source", None)
    if (not isinstance(correlation_fit, RSGDFFitSource)
            or correlation_fit.source != "classic"
            or correlation_fit.auxiliary_frame != "canonical-modrho"
            or correlation_fit.factor_space != "ao-pair"):
        raise ValueError("unsupported chi translation correlation source/frame")
    real = _translation_real_reference(system, basis, mesh, folded, transform_record, cache, auxiliary_name)
    return _source_from_real(real, convention=convention, scf_fit=scf_fit,
        correlation_fit=correlation_fit, fit_source=fit_source,
        options_identity=options_identity, residuals=residuals, transform_record=transform_record,
        n_frozen=n_frozen, auxiliary_name=auxiliary_name)
