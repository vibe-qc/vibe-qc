"""CI wavefunction payloads in a common spatial MO basis.

Determinants use the alpha-string/beta-string ordering shared by TREXIO
and vibe-qc's Slater-Condon engine. One-particle densities reuse that
engine's fermionic phases; no reference-program calculations are imported.
"""

from types import SimpleNamespace

import numpy as np


def ci_wavefunction(coefficients, ci, *, n_core=0, reference=None, root=0):
    """Build an exportable molecular result from CASCI/CI data and AO MOs.

    ``coefficients`` must already include any CASSCF orbital rotation.
    Frozen core orbitals precede the active orbitals; virtuals follow.
    The selected root's wavefunction and energy are exported, including for
    state-averaged orbital optimizations.
    """
    from ...solvers._rdm import _apply_aq_adp_spin

    C = np.asarray(coefficients)
    if C.ndim != 2 or np.iscomplexobj(C) or not np.all(np.isfinite(C)):
        raise ValueError("TREXIO CI export requires finite real spatial MO coefficients.")
    if isinstance(n_core, bool) or int(n_core) != n_core or n_core < 0:
        raise ValueError("TREXIO CI n_core must be a nonnegative integer.")
    n_core = int(n_core)
    determinants = getattr(ci, "determinants", None)
    if determinants is None:
        determinants = getattr(ci, "ci_labels", None)
    if not determinants:
        raise ValueError("TREXIO CI export requires determinant occupation lists.")
    determinants = [(tuple(a), tuple(b)) for a, b in determinants]
    n_active = int(getattr(ci, "n_active_orb", 0) or (C.shape[1] - n_core))
    if n_core + n_active > C.shape[1]:
        raise ValueError("TREXIO CI active space exceeds the supplied MO basis.")
    for pair in determinants:
        for occupied in pair:
            if (tuple(sorted(set(occupied))) != occupied
                    or any(i < 0 or i >= n_active for i in occupied)):
                raise ValueError("TREXIO CI determinants must contain sorted distinct active orbital indices.")
    counts = {(len(a), len(b)) for a, b in determinants}
    if len(counts) != 1 or len(set(determinants)) != len(determinants):
        raise ValueError("TREXIO CI determinants must be distinct and conserve each spin population.")
    nroots = int(getattr(ci, "nroots", 1))
    if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root < nroots:
        raise ValueError("TREXIO CI root is outside the available states.")
    all_coefficients = getattr(ci, "ci_coeffs_all", None)
    vector = (np.asarray(all_coefficients)[:, root] if all_coefficients is not None
              else np.asarray(ci.ci_coeffs) if root == 0 else None)
    if (vector is None or vector.shape != (len(determinants),) or np.iscomplexobj(vector)
            or not np.all(np.isfinite(vector))
            or not np.isclose(np.dot(vector, vector), 1., rtol=0, atol=1e-8)):
        raise ValueError("TREXIO CI coefficients must be a finite normalized real vector for the selected root.")
    index = {det: i for i, det in enumerate(determinants)}
    spin_rdms = [np.zeros((C.shape[1], C.shape[1])) for _ in range(2)]
    for spin, rdm in enumerate(spin_rdms):
        rdm[np.arange(n_core), np.arange(n_core)] = 1.
        for j, det in enumerate(determinants):
            for q in det[spin]:
                for p in range(n_active):
                    phase, occupied = _apply_aq_adp_spin(det[spin], q, p)
                    if not phase:
                        continue
                    bra = (occupied, det[1]) if spin == 0 else (det[0], occupied)
                    i = index.get(bra)
                    if i is not None:
                        rdm[n_core + p, n_core + q] += phase * vector[i] * vector[j]
    norb = C.shape[1]
    words = (norb + 63) // 64
    bits = np.zeros((len(determinants), 2 * words), dtype=np.uint64)
    for row, det in enumerate(determinants):
        for spin in range(2):
            for orbital in list(range(n_core)) + [n_core + i for i in det[spin]]:
                bits[row, spin * words + orbital // 64] |= np.uint64(1) << np.uint64(orbital % 64)
    energies = getattr(ci, "e_totals", None)
    energies = [] if energies is None else list(energies)
    if energies and len(energies) != nroots:
        raise ValueError("TREXIO CI root energies disagree with the number of states.")
    energy = energies[root] if energies else getattr(ci, "e_total", getattr(ci, "energy", None))
    if energy is None:
        raise ValueError("TREXIO CI export requires the selected root energy.")
    fields = {
        "determinant_list": bits.view(np.int64),
        "determinant_coefficient": np.asarray(vector, dtype=float),
        "rdm_1e": spin_rdms[0] + spin_rdms[1],
        "rdm_1e_up": spin_rdms[0], "rdm_1e_dn": spin_rdms[1],
    }
    return SimpleNamespace(
        mo_coeffs=C, mo_energies=None, occupations=np.diag(fields["rdm_1e"]),
        density=C @ fields["rdm_1e"] @ C.T, energy=float(energy),
        trexio_spatial_orbitals=True, trexio_fields=fields, trexio_ecp_source=reference,
        trexio_state=root, trexio_state_num=nroots,
        trexio_classes=["Core"] * n_core + ["Active"] * n_active + ["Virtual"] * (norb - n_core - n_active),
        trexio_spin_counts=tuple(n_core + n for n in next(iter(counts))),
    )
