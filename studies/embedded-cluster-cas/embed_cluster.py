"""MR8 (studies prototype): the `embed_cluster` driver -- carve a finite cluster
from a periodic crystal, embed it in a Madelung point-charge array (MR6b), and
run the molecular CAS solvers on it, all in one call.

This composes the MR6 bricks into the single ergonomic entry point the MR8
production driver will expose:

    cluster_carve (Milestone 2)            -- carve the QM region
  + fitted_array / evjen (MR6a + MR6b)     -- the Madelung embedding
  + compute_nuclear_with_charges + custom  -- the Milestone-1 enabler
    Hcore + run_rhf_scf_with_jk
  + casci / casscf / caspt2                -- the molecular CAS solvers

The API here is the proposed public surface for `vq.embed_cluster(...)`. It
STAYS in studies/ (no core code); productionizing it into
`python/vibeqc/embed/` is the review-gated follow-on (CLAUDE.md s9), where the
deferred citation-database Evjen/Derenzo entries + routes also land.

STATUS: VALIDATED (2026-06-19).  The driver reproduces the MR6c MgO [OMg6]
vibe-qc energies (RHF -1283.2817651291, CASSCF -1283.2957753171) to 5e-11 Ha
through the single ``embed_cluster(...)`` call and exercises a second rock-salt
host (NaCl [ClNa6]/STO-3G) with all checks passing.  The carve + Madelung
array + CAS pipeline is complete in prototype (studies spike, no core code).

Run:
    .venv/bin/python studies/embedded-cluster-cas/embed_cluster.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vibeqc as vq  # noqa: E402
from cluster_carve import _mgo_cell, cluster_carve  # noqa: E402
from madelung_embedding import (  # noqa: E402
    _crystal_sites_within,
    _rms,
    embedding_target,
    evjen_embedding_array,
    finite_array_potential,
    fitted_array,
    offsite_probe_ball,
)
from spike_external_field import (  # noqa: E402
    _attr,
    _energy,
    embedded_mo_hamiltonian,
    embedded_pieces,
)
from vibeqc._vibeqc_core import RHFOptions, run_rhf_scf_with_jk  # noqa: E402
from vibeqc.solvers import casci, caspt2, casscf  # noqa: E402

ANG = 1.8897259886


@dataclass
class EmbeddedClusterResult:
    """Result of an embedded-cluster CAS run (energies in Hartree)."""

    energies: dict  # {"rhf":..., "casci":..., "casscf":..., "caspt2":...}
    cluster: list  # [(int Z, ndarray xyz_bohr), ...]
    cluster_charge: int  # ionic formal net charge of the QM region
    n_electrons: int
    active_space: tuple  # (n_active_elec, n_active_orb)
    basis: str
    array_kind: str  # "fitted" | "evjen"
    n_charges: int  # size of the embedding point-charge array
    array_net: float  # net charge of the array (should be -cluster_charge)
    potential_rms: float  # off-site match vs the Ewald Madelung target (MR6b)

    def report(self):
        comp = {}
        for Z, _ in self.cluster:
            comp[Z] = comp.get(Z, 0) + 1
        formula = " ".join(f"Z{Z}x{n}" for Z, n in sorted(comp.items()))
        lines = [
            f"  cluster: {len(self.cluster)} atoms ({formula}), formal charge "
            f"{self.cluster_charge:+d}, {self.n_electrons} e, basis {self.basis}",
            f"  embedding: {self.array_kind} array, {self.n_charges} charges, "
            f"net {self.array_net:+.4f} (QM+array neutral), off-site potential "
            f"vs Ewald target rms {self.potential_rms:.2e} Ha/e",
            f"  active space: CAS{self.active_space}",
        ]
        for m in ("rhf", "casci", "casscf", "caspt2"):
            if m in self.energies:
                lines.append(f"  {m:7s} = {self.energies[m]:.8f} Ha")
        return "\n".join(lines)


def _center_xyz(system, center):
    """Cartesian (bohr) center for the array, from an atom index or a point."""
    if np.ndim(center) == 0:
        return np.asarray(system.unit_cell[int(center)].xyz, float)
    return np.asarray(center, float)


def embed_cluster(
    system,
    center,
    qm_radius,
    basis,
    nae,
    nao,
    formal,
    *,
    array="fitted",
    cluster_charge=None,
    methods=("rhf", "casci", "casscf"),
    char_len=None,
    r_exact_factor=2.0,
    r_fit_factor=3.0,
    probe_factor=1.2,
    evjen_half_side=6,
    ridge=1e-4,
    n_probes=240,
    rhf_conv=1e-11,
    casscf_conv=1e-8,
):
    """Carve a finite cluster from ``system``, embed it in a Madelung array, and
    run the requested CAS methods on it.

    Parameters
    ----------
    system, center, qm_radius : the carve (see ``cluster_carve``); ``center`` is
        an atom index or a cartesian point (bohr), ``qm_radius`` in bohr.
    basis : basis-set name handed to the QM cluster.
    nae, nao : active electrons / active orbitals (CAS(nae,nao)).
    formal : dict Z -> formal ionic charge, used both for the cluster charge and
        for the Madelung array (the embedding represents the ionic lattice).
    array : "fitted" (Derenzo-Klintenberg-Weber, MR6b) or "evjen" (Evjen, MR6b).
    cluster_charge : ionic formal net charge of the QM region; default
        ``sum(formal[Z] for QM atoms)``.
    methods : which of rhf/casci/casscf/caspt2 to run.
    char_len : characteristic lattice length for the array zone radii; default
        the shortest lattice-vector norm (the cubic constant for rock salt).

    Returns an :class:`EmbeddedClusterResult`.
    """
    a = (
        float(np.min(np.linalg.norm(np.asarray(system.lattice, float), axis=0)))
        if char_len is None
        else float(char_len)
    )
    cxyz = _center_xyz(system, center)

    # 1. carve the QM region
    qm_mol, qm = cluster_carve(system, center, qm_radius)
    if cluster_charge is None:
        cluster_charge = int(round(sum(formal[int(Z)] for Z, _ in qm)))
    atoms = [vq.Atom(int(Z), list(map(float, r))) for Z, r in qm]
    mol = vq.Molecule(atoms, charge=int(cluster_charge), multiplicity=1)

    # 2. build the Madelung embedding array (MR6b) + its provenance
    all_sites = _crystal_sites_within(system, cxyz, r_fit_factor * a)
    probes = offsite_probe_ball(
        cxyz, all_sites, probe_factor * qm_radius, n_probes, seed=1
    )
    v_target = embedding_target(system, qm, formal, probes)
    if array == "fitted":
        pos, q, _info = fitted_array(
            system,
            qm,
            cxyz,
            formal,
            probes,
            v_target,
            r_exact=r_exact_factor * a,
            r_fit=r_fit_factor * a,
            ridge=ridge,
        )
    elif array == "evjen":
        pos, q = evjen_embedding_array(system, qm, cxyz, formal, evjen_half_side, a)
    else:
        raise ValueError(f"array must be 'fitted' or 'evjen', got {array!r}")
    pos = np.asarray(pos, float)
    q = np.asarray(q, float)
    # independent test probes -> the embedding-quality figure of merit (MR6b)
    test = offsite_probe_ball(
        cxyz, all_sites, probe_factor * qm_radius, n_probes, seed=99
    )
    pot_rms = _rms(
        finite_array_potential(pos, q, test)
        - embedding_target(system, qm, formal, test)
    )

    # 3. embedded RHF + the requested CAS methods (the Milestone-1 enabler)
    basis_obj = vq.BasisSet(mol, basis)
    ext_pos = [list(map(float, p)) for p in pos]
    ext_q = [float(x) for x in q]
    S, Hcore, E_nuc, jk = embedded_pieces(mol, basis_obj, ext_pos, ext_q)
    opts = RHFOptions()
    opts.conv_tol_energy = rhf_conv
    opts.conv_tol_grad = max(rhf_conv * 1e3, 1e-8)
    opts.max_iter = 400
    rhf = run_rhf_scf_with_jk(
        basis_obj, mol.n_electrons(), S, Hcore, float(E_nuc), jk, opts
    )
    C = np.asarray(_attr(rhf, "mo_coeffs", "C", "orbitals"))
    H = embedded_mo_hamiltonian(mol, basis_obj, C, Hcore, E_nuc)
    n_core = (mol.n_electrons() - nae) // 2

    energies = {}
    if "rhf" in methods:
        energies["rhf"] = _energy(rhf)
    ci = None
    if "casci" in methods or "caspt2" in methods:
        ci = casci(
            H.h1e, H.h2e, nae, nao, n_core=n_core, nuclear_repulsion=H.nuclear_repulsion
        )
        if "casci" in methods:
            energies["casci"] = _energy(ci)
    if "casscf" in methods:
        sc = casscf(
            H.h1e,
            H.h2e,
            nae,
            nao,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            conv_tol_grad=casscf_conv,
        )
        energies["casscf"] = _energy(sc)
    if "caspt2" in methods:
        n_virt = H.norb - n_core - nao
        energies["caspt2"] = _energy(
            caspt2(ci, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt)
        )

    return EmbeddedClusterResult(
        energies=energies,
        cluster=qm,
        cluster_charge=int(cluster_charge),
        n_electrons=mol.n_electrons(),
        active_space=(nae, nao),
        basis=basis,
        array_kind=array,
        n_charges=len(q),
        array_net=float(q.sum()),
        potential_rms=pot_rms,
    )


def _nacl_cell(a_ang=5.64):
    """NaCl rock salt, conventional cubic cell (4 Na + 4 Cl)."""
    a = a_ang * ANG
    lat = np.diag([a, a, a])
    frac = [
        (0, 0, 0),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),  # Na
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 0, 0.5),
        (0.5, 0.5, 0.5),
    ]  # Cl
    Z = [11] * 4 + [17] * 4
    atoms = [vq.Atom(z, list(np.asarray(f) @ lat.T)) for z, f in zip(Z, frac)]
    return vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1), a


def main():
    print("=" * 72)
    print("MR8 (prototype): embed_cluster driver -- carve + Madelung array + CAS")
    print("=" * 72)

    # --- MgO [OMg6]/6-31G: reproduce the MR6c result through the driver ---
    # MR6c used probe_radius = 0.60*a (hardcoded); the driver defaults to
    # probe_factor * qm_radius = 1.2 * 0.51*a = 0.612*a.  Reproduce the exact
    # MR6c array by matching its probe ball: 0.60*a = factor * 0.51*a.
    sys_mgo, a_mgo = _mgo_cell()
    probe_f_mgo = 0.60 / 0.51
    res = embed_cluster(
        sys_mgo,
        4,
        0.51 * a_mgo,
        "6-31G",
        6,
        4,
        formal={12: +2.0, 8: -2.0},
        probe_factor=probe_f_mgo,
    )
    print("\n[1] MgO [OMg6] (reproduce MR6c through one call):")
    print(res.report())
    # MR6c reference vibe-qc energies (parity_mr6c.py / HANDOVER MR6c).
    ref = {"rhf": -1283.2817651291, "casscf": -1283.2957753171}
    drhf = abs(res.energies["rhf"] - ref["rhf"])
    dcas = abs(res.energies["casscf"] - ref["casscf"])
    print(
        f"  vs MR6c: |Δ(RHF)| = {drhf:.1e}, |Δ(CASSCF)| = {dcas:.1e}  "
        f"{'PASS' if drhf < 1e-6 and dcas < 1e-5 else 'CHECK'}"
    )

    # --- NaCl [ClNa6]/STO-3G: generality on a second rock-salt host ---
    sys_nacl, a_nacl = _nacl_cell()
    res2 = embed_cluster(
        sys_nacl, 4, 0.51 * a_nacl, "STO-3G", 6, 4, formal={11: +1.0, 17: -1.0}
    )
    print("\n[2] NaCl [ClNa6] (generality: different ions + active space):")
    print(res2.report())
    ok2 = (
        abs(res2.array_net + res2.cluster_charge) < 1e-6
        and res2.potential_rms < 1e-3
        and res2.energies["casscf"] < res2.energies["rhf"]
    )
    print(
        f"  array neutralizes QM ({res2.array_net:+.2f} = -{res2.cluster_charge}), "
        f"potential matched, CASSCF below RHF: {'PASS' if ok2 else 'CHECK'}"
    )

    ok = drhf < 1e-6 and dcas < 1e-5 and ok2
    print(
        "\n"
        + (
            "MR8 prototype: PASS -- one embed_cluster(...) call carves, "
            "embeds, and runs\n  the CAS solvers; reproduces MR6c on MgO "
            "and generalizes to NaCl."
            if ok
            else "SOME CHECK FAILED"
        )
    )


if __name__ == "__main__":
    main()
