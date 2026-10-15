"""Neutral fitted-torus CCM reference as a shared DLPNO correlation source.

This is an experimental adapter, not a new coupled-cluster implementation
or a claim of Gamma-CCM/chi-CCM Hamiltonian identity. Bloch references pass
through the existing finite-character fold in ``bloch_reference``.
"""

from __future__ import annotations

import copy
import operator
from types import SimpleNamespace

import numpy as np

from ...dlpno.source import (
    CorrelationSource,
    CorrelationSourceError,
    LpqFactors,
    SourceConventions,
    _array_identity,
    _atom_record,
    _real_array,
    detect_reference_kind,
    refuse_unsupported_periodic_reference,
)


def ccm_neutral_source(
    ccm, scf_result, *, cderi=None, localize="pm", n_frozen=0, ke_cutoff=200.0,
) -> CorrelationSource:
    """Adapt a closed-shell neutral-control reference, retaining its AO metric.

    Carried factors take precedence over rebuilding. ``same_source`` is true
    only when these are the reference's retained factors (or an identical
    explicit array). Explicit factors or a legacy refit without such evidence
    remain usable on this experimental route, with ``same_source=False``.
    The identity hashes the actual factors, Fock and overlap, including for
    legacy references that have no fit-support receipt.

    Only the active occupied block is localised. Frozen orbitals are not
    mixed into it, and the reference supplied by the caller is not modified.
    ``localize="boys"`` retains its historical non-wrapping-orbital scope;
    Pipek-Mezey is the periodic-safe default.
    """
    from ...correlation_conventions import effective_electron_count
    from ...periodic_localise import localise_periodic_gamma
    from ...periodic_toroidal_mp2 import toroidal_pair_distance_fn
    from .bloch_reference import as_real_reference
    from .neutral import carried_cderi, ccm_neutral_cderi

    methods = {
        "none": None, "pm": "pipek-mezey",
        "pipek-mezey": "pipek-mezey", "boys": "boys",
    }
    if localize not in methods:
        raise CorrelationSourceError(
            f"unknown localize option {localize!r}; choose from {sorted(methods)}"
        )
    refuse_unsupported_periodic_reference(scf_result)
    reference = as_real_reference(ccm, scf_result)
    if not bool(getattr(reference, "converged", False)):
        raise CorrelationSourceError("coupled DLPNO-CCSD requires a converged CCM SCF reference")
    if detect_reference_kind(reference) != "rhf" or getattr(reference, "functional", None):
        raise CorrelationSourceError("coupled DLPNO-CCSD requires a closed-shell HF reference")
    ne = effective_electron_count(ccm.supercell, reference)
    if ne % 2 or int(ccm.supercell.multiplicity) != 1:
        raise CorrelationSourceError("coupled DLPNO-CCSD requires a closed-shell HF reference")
    n_occ = ne // 2
    try:
        nf = operator.index(n_frozen)
    except TypeError as exc:
        raise CorrelationSourceError("n_frozen must be an integer") from exc
    if isinstance(n_frozen, (bool, np.bool_)) or nf < 0 or nf >= n_occ:
        raise CorrelationSourceError(f"n_frozen={n_frozen!r} out of range for n_occ={n_occ}")

    S = _real_array(reference.overlap, name="CCM overlap")
    F = _real_array(reference.fock, name="CCM Fock")
    C = _real_array(reference.mo_coeffs, name="CCM orbitals").copy()
    nbf = int(ccm.nbf)
    if S.shape != (nbf, nbf) or F.shape != S.shape or C.ndim != 2 or C.shape[0] != nbf:
        raise CorrelationSourceError("CCM reference matrices do not match the supercell AO basis")
    if C.shape[1] < n_occ or not np.allclose(C.T @ S @ C, np.eye(C.shape[1]), rtol=0, atol=1e-6):
        raise CorrelationSourceError("CCM orbitals must be S-orthonormal and contain the occupied space")
    if not np.isfinite(reference.energy):
        raise CorrelationSourceError("CCM reference energy is not finite")

    carried = carried_cderi(reference)
    if cderi is not None:
        factor_origin = "explicit"
    else:
        factor_origin = "reference" if carried is not None else "rebuilt"
    if cderi is None:
        cderi = carried if carried is not None else ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff)
    L = _real_array(cderi, name="CCM neutral factors")
    factors = LpqFactors(L)
    if factors.n_orb != nbf:
        raise CorrelationSourceError("CCM factors do not match the supercell AO basis")
    same_source = carried is not None and np.array_equal(L, carried)
    conventions = SourceConventions(
        exchange_q0=str(getattr(reference, "exchange_q0", None) or "unspecified"),
        kmesh=tuple(int(n) for n in ccm.nrep),
        bloch_normalisation="real-bvk-supercell",
        same_source=bool(same_source),
    )
    record = {
        "kind": "ccm-neutral", "reference_kind": "rhf",
        "basis": ccm.basis_name, "n_basis": nbf,
        "lattice": np.asarray(ccm.cluster_lattice).tolist(),
        "atoms": [_atom_record(atom) for atom in ccm.supercell.atoms],
        "charge": int(ccm.supercell.charge), "multiplicity": int(ccm.supercell.multiplicity),
        "reference_energy": float(reference.energy),
        "reference_backend": str(getattr(reference, "backend", "") or ""),
        "fock": _array_identity(F), "overlap": _array_identity(S),
        "factors": _array_identity(L), "factor_origin": factor_origin,
        "n_frozen": nf, "localisation": methods[localize] or "none",
        "conventions": conventions.as_record(),
    }
    method = methods[localize]
    if method is not None:
        active = SimpleNamespace(mo_coeffs=C[:, nf:n_occ], overlap=S)
        C[:, nf:n_occ] = localise_periodic_gamma(
            active, ccm.basis, ccm.cluster_system, method=method, n_occ=n_occ - nf,
        ).C_loc
    adapted = copy.copy(reference)
    adapted.mo_coeffs, adapted.fock, adapted.overlap = C, F, S
    return CorrelationSource(
        kind="ccm-neutral", reference_kind="rhf", molecule=ccm.supercell,
        basis=ccm.basis, reference=adapted, factors=factors, overlap=S,
        pair_distance_fn=toroidal_pair_distance_fn(ccm.cluster_system, (1, 1, 1)),
        conventions=conventions, provenance=record,
    )
