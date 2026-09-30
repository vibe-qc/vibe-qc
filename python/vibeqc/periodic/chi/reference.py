"""Process-local accepted references for experimental chi correlation.

This transports an already executed operator, not a change of fitted source.
Only the builder below can supply the admission record; raw SCF results, disk
caches, unrestricted and corrected RIJCOSX states are outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import os
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from ...banner import loaded_process_provenance
from ...periodic_fit_source import RSGDFFitSource, factor_digest
from .scf import _normalise_mesh, cyclic_gamma_mesh

__all__ = ["ChiAcceptedReference", "prepare_chi_reference"]


def _value(value):
    """Lossless, address-free values for admission keys (not log formatting)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if is_dataclass(value):
        return {f.name: _value(getattr(value, f.name)) for f in fields(value)}
    if hasattr(value, "name") and hasattr(value, "value"):
        return (type(value).__name__, value.name)
    if type(value).__name__.endswith("Options"):
        from ...options_dump import _walk_attributes
        return {k: _value(v) for k, v in _walk_attributes(value)}
    raise TypeError(f"unsupported reference control type: {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(_value(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _options(options):
    from ..._vibeqc_core import PeriodicRHFOptions
    if options is None:
        options = PeriodicRHFOptions()
        options.conv_tol_energy = 1e-11
        options.conv_tol_grad = 1e-9
    if type(options) is not PeriodicRHFOptions:
        raise TypeError("accepted chi references require PeriodicRHFOptions")
    for name, limit in (("conv_tol_energy", 1e-11), ("conv_tol_grad", 1e-9)):
        value = float(getattr(options, name))
        if not np.isfinite(value) or not 0 < value <= limit:
            raise ValueError(f"accepted chi reference requires {name} <= {limit:g}")
    return options


def orbital_residuals(hf, n_occ):
    """Measure each residual on its named density/operator, never infer it."""
    F, S, D, C = (np.asarray(getattr(hf, key)) for key in
                  ("fock", "overlap", "density", "mo_coeffs"))
    eps = np.asarray(hf.mo_energies)
    weights = np.asarray(hf.kpoint_weights)
    if not all(np.all(np.isfinite(x)) for x in (F, S, D, C, eps, weights)):
        raise ValueError("nonfinite chi reference state")
    orbital_density = 2 * C[:, :, :n_occ] @ C[:, :, :n_occ].conj().transpose(0, 2, 1)

    def commutator(fock, density):
        fds = fock @ density @ S
        error = fds - fds.conj().transpose(0, 2, 1)
        return float(np.sqrt(np.sum(weights * np.sum(abs(error)**2, axis=(1, 2)))))

    return MappingProxyType({
        "accepted_density_returned_fock_commutator": commutator(F, D),
        "orbital_density_returned_fock_commutator": commutator(F, orbital_density),
        "fc_minus_sc_epsilon_max": float(np.max(abs(F @ C - (S @ C) * eps[:, None, :]))),
        "metric_orthogonality_max": float(np.max(abs(C.conj().transpose(0, 2, 1) @ S @ C - np.eye(C.shape[-1])))),
        "orbital_density_minus_accepted_max": float(np.max(abs(orbital_density - D))),
        "iterated_density_commutator": commutator(np.asarray(getattr(hf, "iterated_fock", F)), D),
    })


def _admit_state(hf, n_occ):
    diagnostics = getattr(hf, "aiccm2026dev_b", None)
    if (getattr(diagnostics, "electronic_method", None) != "RHF"
            or getattr(diagnostics, "backend", None) != "ri"
            or getattr(hf, "returned_operator", "iterated") != "iterated"):
        raise ValueError("accepted chi correlation requires a matched iterated RI-RHF state; corrected RIJCOSX and unrestricted states are unsupported")
    if not hf.converged:
        raise ValueError("chi reference is not converged")
    delta = getattr(diagnostics, "final_scf_delta_e_ha", None)
    if delta is None or not np.isfinite(delta) or abs(delta) > 1e-11:
        raise ValueError("chi reference lacks achieved energy convergence <= 1e-11")
    if float(getattr(hf, "smearing_temperature", 0)) != 0:
        raise ValueError("chi reference requires integer occupations without smearing")
    residuals = orbital_residuals(hf, n_occ)
    if any(not np.isfinite(v) or v > 1e-9 for v in residuals.values()):
        raise ValueError(f"chi reference fails orbital stationarity: {dict(residuals)}")
    occupations = np.asarray(getattr(hf, "occupations", []))
    if occupations.size:
        expected = np.zeros_like(occupations)
        expected[:, :n_occ] = 2
        if not np.allclose(occupations, expected, atol=1e-12, rtol=0):
            raise ValueError("chi reference requires closed-shell integer occupations")
    return residuals


def _state_signature(hf):
    names = ("energy", "e_nuclear", "converged", "mo_coeffs", "mo_energies",
             "fock", "overlap", "hcore", "density", "kpoints_cart",
             "kpoint_weights", "occupations", "smearing_temperature",
             "aiccm2026dev_b", "finite_torus_convention", "exact_exchange_assembly",
             "executed_fit_source", "_chi_correlation_fit_source", "rsgdf_fit_request",
             "process_provenance", "backend", "runtime_backend", "rsgdf_tail_ke_cutoff",
             "rsgdf_ke_cutoff", "aiccm_resolved_rsgdf_tail_ke_cutoff",
             "aiccm_resolved_rsgdf_ke_cutoff", "returned_operator", "energy_operator")
    return _digest({key: getattr(hf, key, None) for key in names})


def _real_signature(real):
    from ...aux_basis import _basis_fingerprint, _range_separated_gdf_source_signature
    return _digest((_range_separated_gdf_source_signature(real.super_system, real.basis, real.df.aux_basis),
                    int(real.molecule.n_electrons()), real.hf.energy, real.hf.converged, real.hf.fock,
                    real.hf.overlap, real.hf.mo_coeffs, real.hf.mo_energies,
                    real.df.home_factors, real.df.mesh, real.n_cells,
                    _basis_fingerprint(real.basis), _basis_fingerprint(real.df.aux_basis)))


@dataclass(frozen=True)
class ChiAcceptedReference:
    """Validated, process-local RI-RHF state and separate correlation handle.

    Construct with :func:`prepare_chi_reference`. Factors are borrowed read-only;
    any later state, source, frame, support or factor mutation fails admission.
    All fitted/post-HF results remain not-qualified (D93).
    """
    hf_result: object
    owner_pid: int
    mesh: tuple
    geometry_basis_identity: str
    charge_multiplicity: tuple
    options_identity: str
    controls: tuple
    scf_fit_source: RSGDFFitSource
    correlation_fit_source: RSGDFFitSource
    finite_torus_convention: object
    exact_exchange_assembly: object
    orbital_residuals: Mapping
    process_provenance: Mapping
    build_stages: tuple
    factors: Mapping
    auxiliary: object
    real_reference: object
    state_identity: str
    real_identity: str
    seal: str = ""
    energy_operator: str = "iterated"
    returned_operator: str = "iterated"
    normalization: str = "energy per primitive cell; real torus divided by Nc once"
    quantitative_status: str = "not-qualified"

    def _metadata(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in
                {"hf_result", "factors", "auxiliary", "real_reference", "seal"}}

    def validate(self, system, basis, mesh, options, **controls):
        from ...aux_basis import _range_separated_gdf_source_signature, _basis_fingerprint
        if self.owner_pid != os.getpid():
            raise ValueError("chi reference belongs to a different process")
        if self.seal != _digest(self._metadata()):
            raise ValueError("chi reference metadata/source/frame/tail changed")
        if tuple(mesh) != self.mesh or (int(system.charge), int(system.multiplicity)) != self.charge_multiplicity:
            raise ValueError("chi reference geometry/charge/mesh mismatch")
        if self.geometry_basis_identity != _range_separated_gdf_source_signature(system, basis, self.auxiliary):
            raise ValueError("chi reference geometry/basis/auxiliary mismatch")
        if _basis_fingerprint(self.auxiliary) != self.correlation_fit_source.auxiliary_basis_sha256:
            raise ValueError("chi reference auxiliary basis changed")
        if options is not None and _digest(_options(options)) != self.options_identity:
            raise ValueError("chi reference SCF controls/support mismatch")
        if tuple(sorted(controls.items())) != self.controls:
            raise ValueError("chi reference fit source/auxiliary/frame/tail/support mismatch")
        if dict(self.process_provenance) != loaded_process_provenance():
            raise ValueError("chi reference process release/source/native core mismatch")
        if _state_signature(self.hf_result) != self.state_identity:
            raise ValueError("chi reference SCF state/source changed")
        if factor_digest(self.factors) != self.correlation_fit_source.factor_sha256:
            raise ValueError("chi reference correlation factors changed")
        if _real_signature(self.real_reference) != self.real_identity:
            raise ValueError("chi reference home-auxiliary state changed")
        _admit_state(self.hf_result, int(system.n_electrons()) // 2)
        return self

    def execution(self, *, local):
        return {"scf": {"status": "reused", "source": self.scf_fit_source.identity},
                "correlation_fit": {"status": "reused", "source": self.correlation_fit_source.identity},
                "home_auxiliary_reference": {"status": "reused" if local else "not-used",
                                             "source": self.correlation_fit_source.identity,
                                             "state": self.real_identity,
                                             "representation": "chi-home-auxiliary"},
                "orbital_projection": {"status": "not-used" if local else "built",
                                       "source": self.correlation_fit_source.identity,
                                       "state": self.state_identity},
                "process_provenance": dict(self.process_provenance)}


def require_reference(reference, system, basis, mesh, options, **controls):
    if not isinstance(reference, ChiAcceptedReference):
        raise TypeError("reference must be a ChiAcceptedReference from prepare_chi_reference")
    return reference.validate(system, basis, mesh, options, **controls)


def prepare_chi_reference(system, basis, mesh=(1, 1, 1), options=None, *,
                          aux_basis=None, lattice_cutoff_bohr=15., rsgdf_ke_cutoff=200.,
                          fit_source="auto", rsgdf_tail_ke_cutoff=None,
                          gdf_linear_dep_threshold=1e-9, progress=None, verbose=None):
    """Build and accept one restricted chi reference for multiple consumers.

    Uses the unchanged selected SCF and correlation builders. ``auto`` keeps
    SR SCF and classic correlation separate. ``classic`` borrows its attested
    canonical SCF factors. No SR-to-classic/frame conversion is attempted.
    Explicit options must request energy <= 1e-11 and gradient <= 1e-9.
    """
    from . import posthf as p
    from ...aux_basis import default_aux_for, _range_separated_gdf_source_signature
    options = _options(options)
    if int(system.dim) != 3 or int(system.multiplicity) != 1:
        raise NotImplementedError("accepted chi references require 3D restricted closed shell")
    mesh = _normalise_mesh(system, mesh)
    p._validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr, rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold)
    controls = dict(auxiliary_name=aux_basis or default_aux_for(basis.name),
                    lattice_cutoff_bohr=float(lattice_cutoff_bohr), rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
                    fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
                    gdf_linear_dep_threshold=float(gdf_linear_dep_threshold))
    p._admit_selected_reference(fit_source, rsgdf_tail_ke_cutoff, options, lattice_cutoff_bohr)
    hf = p.run_aiccm2026dev_b_rhf(system, basis, mesh, options, backend="ri",
        aux_basis=controls["auxiliary_name"], gdf_method="rsgdf", rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        fit_source=fit_source, rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold, progress=progress, verbose=verbose)
    residuals = _admit_state(hf, int(system.n_electrons()) // 2)
    convention = p._reference_finite_torus_convention(hf)
    scf_source = getattr(hf, "executed_fit_source", None)
    if not isinstance(scf_source, RSGDFFitSource):
        raise ValueError("chi reference lacks executed SCF source telemetry")
    if (hf.exact_exchange_assembly.c_full != 1 or hf.exact_exchange_assembly.c_sr != 0):
        raise ValueError("chi correlation requires full RHF exchange")
    cache, auxiliary = p._selected_lpq_cache(system, basis,
        np.asarray(cyclic_gamma_mesh(system, mesh).kpoints_cart), controls["auxiliary_name"],
        reference=hf, **{k:v for k,v in controls.items() if k != "auxiliary_name"})
    correlation_source = scf_source if fit_source == "classic" else hf._chi_correlation_fit_source
    if correlation_source.source != "classic" or correlation_source.auxiliary_frame != "canonical-modrho":
        raise ValueError("unsupported chi correlation source/frame")
    real = p._build_b_real_reference(system, basis, mesh, hf, cache, controls["auxiliary_name"])
    for value in cache.values():
        value.setflags(write=False)
    real.df.home_factors.setflags(write=False)
    result = ChiAcceptedReference(hf, os.getpid(), mesh,
        _range_separated_gdf_source_signature(system, basis, auxiliary),
        (int(system.charge), int(system.multiplicity)), _digest(options), tuple(sorted(controls.items())),
        scf_source, correlation_source, convention, hf.exact_exchange_assembly, residuals,
        MappingProxyType(dict(hf.process_provenance)),
        (("scf", "built", scf_source.identity),
         ("correlation_fit", "reused" if fit_source == "classic" else "built", correlation_source.identity),
         ("home_auxiliary_reference", "built", correlation_source.identity)),
        MappingProxyType(dict(cache)), auxiliary, real, _state_signature(hf), _real_signature(real))
    return replace(result, seal=_digest(result._metadata()))
