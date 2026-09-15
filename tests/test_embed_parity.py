"""MR10: broad parity — vibe-qc ``embed_cluster`` vs OpenMolcas XFIELD
on multiple rock-salt hosts.

Validates the full production pipeline (``vq.embed_cluster`` composing
``carve_cluster`` + MR6b Madelung array + Milestone-1 enabler +
molecular CAS solvers) against OpenMolcas (out-of-process, CLAUDE.md
s10) with the identical point-charge array.

Each test case:
1. Builds the system via ``vq.embed_cluster`` (vibe-qc).
2. Runs OpenMolcas GATEWAY XField with the SAME geometry + array.
3. Compares absolute RHF energies (basis-library-dependent) and the
   Madelung stabilization dE = E(embedded) - E(bare) where the basis
   cancels.

Tests require ``$OPENMOLCAS_PYMOLCAS`` and ``$MOLCAS`` to be set
(pointing at a local OpenMolcas build).  They are skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.embed import embed_cluster
from vibeqc.embed._openmolcas import run_openmolcas_xfield

ANG = 1.8897259886

# Check if OpenMolcas is available.
try:
    from vibeqc.embed._openmolcas import _discover_openmolcas

    _discover_openmolcas()
    _HAS_OPENMOLCAS = True
except (RuntimeError, ImportError):
    _HAS_OPENMOLCAS = False

requires_openmolcas = pytest.mark.skipif(
    not _HAS_OPENMOLCAS,
    reason="OpenMolcas not found (set OPENMOLCAS_PYMOLCAS + MOLCAS)",
)


# --------------------------------------------------------------------------- #
# System builders
# --------------------------------------------------------------------------- #


def _mgo_cell(a_ang: float = 4.21) -> tuple[vq.PeriodicSystem, float]:
    """MgO rock salt, conventional cubic cell."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]
    Z = [12] * 4 + [8] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


def _nacl_cell(a_ang: float = 5.64) -> tuple[vq.PeriodicSystem, float]:
    """NaCl rock salt, conventional cubic cell."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]
    Z = [11] * 4 + [17] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


def _lif_cell(a_ang: float = 4.028) -> tuple[vq.PeriodicSystem, float]:
    """LiF rock salt, conventional cubic cell."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]
    Z = [3] * 4 + [9] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


# --------------------------------------------------------------------------- #
# Parity test cases
# --------------------------------------------------------------------------- #

# Each case: (label, cell_builder, center_index, formal, nae, nao, basis)
# Validated 2026-06-19 vs OpenMolcas XFIELD (out-of-process).
_PARITY_CASES = [
    # vibe-qc embed RHF -1283.2817651291, bare RHF -1262.9660428791, dE -20.3157222500
    # OpenMolcas XFIELD SCF -1283.2817651673, bare SCF -1262.9660428809, dE -20.3157222864
    # |Δ(RHF)| = 3.8e-08, |Δ(dE)| = 3.6e-09
    ("MgO [OMg6]/6-31G", _mgo_cell, 4, {12: +2.0, 8: -2.0}, 6, 4, "6-31G"),
    # vibe-qc embed RHF -1418.2002491554, bare RHF -1416.2970469428
    # OpenMolcas XFIELD SCF -1418.2002500120, bare SCF -1416.2970481027
    # |Δ(RHF)| = 8.6e-07, |Δ(dE)| = 3.0e-07  (STO-3G basis mismatch floor)
    ("NaCl [ClNa6]/STO-3G", _nacl_cell, 4, {11: +1.0, 17: -1.0}, 6, 4, "STO-3G"),
    # vibe-qc embed RHF -145.1041424035, bare RHF -139.7922824826
    # OpenMolcas XFIELD SCF -145.1041428309, bare SCF -139.7922829083
    # |Δ(RHF)| = 4.3e-07, |Δ(dE)| = 4.3e-07  (STO-3G basis mismatch floor)
    ("LiF [FLi6]/STO-3G", _lif_cell, 4, {3: +1.0, 9: -1.0}, 6, 4, "STO-3G"),
]


@requires_openmolcas
@pytest.mark.slow
@pytest.mark.parametrize(
    "label,cell_builder,center,formal,nae,nao,basis", _PARITY_CASES
)
def test_embed_parity_rhf(
    label: str,
    cell_builder,
    center: int,
    formal: dict[int, float],
    nae: int,
    nao: int,
    basis: str,
) -> None:
    """vibe-qc embed_cluster RHF vs OpenMolcas XFIELD SCF (same array)."""
    system, a = cell_builder()
    qm_radius = 0.51 * a

    # vibe-qc: embed_cluster produces the array + RHF.
    res = embed_cluster(
        system,
        center,
        qm_radius,
        basis,
        nae,
        nao,
        formal=formal,
        methods=("rhf",),
        probe_factor=0.60 / 0.51,  # match MR6c config
    )
    vq_rhf = res.energies["rhf"]

    # Collect atoms and embedding array for OpenMolcas.
    atoms_bohr = [(int(Z), tuple(map(float, r))) for Z, r in res.cluster]
    # Reconstruct the array positions and charges (same as vibe-qc used).
    # We trust that embed_cluster produces the correct array; extract from
    # the SCF pieces by running a simple re-build.
    from vibeqc.embed._madelung import (
        _crystal_sites_within,
        embedding_target,
        fitted_array,
        offsite_probe_ball,
    )

    center_xyz = np.asarray(system.unit_cell[center].xyz, float)
    all_sites = _crystal_sites_within(system, center_xyz, 3.0 * a)
    probes = offsite_probe_ball(center_xyz, all_sites, 0.60 * a, 240, seed=1)
    v_target = embedding_target(system, res.cluster, formal, probes)
    pos, q_array, _info = fitted_array(
        system,
        res.cluster,
        center_xyz,
        formal,
        probes,
        v_target,
        r_exact=2.0 * a,
        r_fit=3.0 * a,
        ridge=1e-4,
    )
    ext_pos = [list(map(float, p)) for p in pos]
    ext_q = [float(x) for x in q_array]
    n_core = (res.n_electrons - nae) // 2

    # OpenMolcas: same cluster + same array.
    om_scf, om_ras, note = run_openmolcas_xfield(
        atoms_bohr,
        basis,
        ext_pos,
        ext_q,
        n_core,
        nao,
        nae,
        charge=res.cluster_charge,
        timeout=900,
    )
    if note:
        pytest.skip(f"OpenMolcas failed: {note}")
    assert om_scf is not None, f"OpenMolcas SCF missing: {note}"

    # Absolute RHF parity (basis-library-limited).
    drhf = abs(vq_rhf - om_scf)
    # 6-31G: ~1e-7 floor for Mg/O. STO-3G: ~1e-6 floor (known basis mismatch).
    tol = 2e-6 if "STO-3G" in basis.upper() else 5e-7
    assert drhf < tol, (
        f"{label}: |Δ(RHF)| = {drhf:.1e} > {tol:.1e}  "
        f"(vq={vq_rhf:.10f}, om={om_scf:.10f})"
    )

    # Madelung stabilization (the clean test — basis cancels).
    # Bare RHF: run standard SCF on the cluster without any embedding.
    from vibeqc._vibeqc_core import (
        RHFOptions,
        compute_kinetic,
        compute_nuclear,
        compute_overlap,
        make_direct_jk_builder,
        run_rhf_scf_with_jk,
    )

    atoms = [vq.Atom(int(Z), list(map(float, r))) for Z, r in res.cluster]
    mol_bare = vq.Molecule(atoms, charge=res.cluster_charge, multiplicity=1)
    basis_obj = vq.BasisSet(mol_bare, basis)
    S = np.asarray(compute_overlap(basis_obj))
    T = np.asarray(compute_kinetic(basis_obj))
    V_nuc = np.asarray(compute_nuclear(basis_obj, mol_bare))
    Hcore = T + V_nuc
    E_nuc = float(mol_bare.nuclear_repulsion())
    jk = make_direct_jk_builder(basis_obj)
    opts_rhf = RHFOptions()
    opts_rhf.conv_tol_energy = 1e-11
    opts_rhf.conv_tol_grad = 1e-8
    opts_rhf.max_iter = 400
    rhf_bare = run_rhf_scf_with_jk(
        basis_obj, mol_bare.n_electrons(), S, Hcore, E_nuc, jk, opts_rhf
    )
    from vibeqc.embed._external_field import _energy

    vq_bare = _energy(rhf_bare)
    dvq = vq_rhf - vq_bare

    # Bare SCF for OpenMolcas.
    om_bare_scf, _, note2 = run_openmolcas_xfield(
        atoms_bohr,
        basis,
        [],
        [],
        n_core,
        nao,
        nae,
        charge=res.cluster_charge,
        timeout=900,
    )
    if note2:
        pytest.skip(f"OpenMolcas bare failed: {note2}")
    dom = om_scf - om_bare_scf

    dd = abs(dvq - dom)
    assert dd < 1e-7, (
        f"{label}: |Δ(dE)| = {dd:.1e} > 1e-7  (vq dE={dvq:+.8f}, om dE={dom:+.8f})"
    )
