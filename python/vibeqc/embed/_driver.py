"""``vq.embed_cluster`` — carve a finite cluster from a periodic crystal,
embed it in a Madelung point-charge array (MR6), optionally add an AIMP
frontier shell (MR7), and run the molecular CAS solvers on it, all in
one call.

This is the production driver for the MR8 embedded-cluster CASSCF/CASPT2
pipeline.  It composes:

    ``carve_cluster`` (B1.5)                 — carve the QM region
  + ``fitted_array`` / ``evjen_array`` (MR6) — the Madelung embedding
  + ``embedded_pieces`` (Milestone 1)        — external-field Hamiltonian
  + AIMP ECP layer (MR7, optional)           — frontier-ion core potential
  + ``casci`` / ``casscf`` / ``caspt2``      — the molecular CAS solvers

Productionised from ``studies/embedded-cluster-cas/embed_cluster.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import ECPCenter, RHFOptions, run_rhf_scf_with_jk
from vibeqc.solvers import casci, caspt2, casscf

from ._aimp import AIMPEntry
from ._aimp_ecp import _ELEMENT_Z, write_aimp_ecp_library
from ._carve import carve_cluster
from ._external_field import _attr, _energy, embedded_mo_hamiltonian, embedded_pieces
from ._madelung import (
    _crystal_sites_within,
    _rms,
    embedding_target,
    evjen_array,
    finite_array_potential,
    fitted_array,
    offsite_probe_ball,
)


@dataclass
class EmbeddedClusterResult:
    """Result of an embedded-cluster CAS run (energies in Hartree)."""

    energies: dict  # {"rhf": ..., "casci": ..., "casscf": ..., "caspt2": ...}
    cluster: list  # [(int Z, ndarray xyz_bohr), ...]
    cluster_charge: int  # ionic formal net charge of the QM region
    n_electrons: int
    active_space: tuple  # (n_active_elec, n_active_orb)
    basis: str
    array_kind: str  # "fitted" | "evjen"
    n_charges: int  # size of the embedding point-charge array
    array_net: float  # net charge of the array (should be −cluster_charge)
    potential_rms: float  # off-site match vs the Ewald Madelung target (MR6b)
    aimp_n_centres: int = 0  # number of AIMP ECP centres (MR7, 0 = none)

    def report(self) -> str:
        comp: dict[int, int] = {}
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


def _center_xyz(system: vq.PeriodicSystem, center: int | np.ndarray) -> np.ndarray:
    """Cartesian (bohr) centre for the array, from an atom index or a point."""
    if np.ndim(center) == 0:
        return np.asarray(system.unit_cell[int(center)].xyz, float)
    return np.asarray(center, float)


def embed_cluster(
    system: vq.PeriodicSystem,
    center: int | np.ndarray,
    qm_radius: float,
    basis: str,
    nae: int,
    nao: int,
    formal: dict[int, float],
    *,
    array: str = "fitted",
    cluster_charge: int | None = None,
    methods: tuple[str, ...] = ("rhf", "casci", "casscf"),
    char_len: float | None = None,
    r_exact_factor: float = 2.0,
    r_fit_factor: float = 3.0,
    probe_factor: float = 1.2,
    evjen_half_side: int = 6,
    ridge: float = 1e-4,
    n_probes: int = 240,
    rhf_conv: float = 1e-11,
    casscf_conv: float = 1e-8,
    aimp_entries: list | None = None,
    aimp_radius: float | None = None,
) -> EmbeddedClusterResult:
    """Carve a finite cluster from *system*, embed it in a Madelung
    array, and run the requested CAS methods on it.

    Parameters
    ----------
    system : vq.PeriodicSystem
        The periodic crystal.
    center : int or (3,) array-like
        Carve centre (atom index or cartesian point, bohr).
    qm_radius : float
        Carve radius (bohr).
    basis : str
        Basis-set name handed to the QM cluster.
    nae, nao : int
        Active electrons / active orbitals (CAS(nae,nao)).
    formal : dict[int, float]
        *Z* → formal ionic charge; used both for the cluster charge
        and for the Madelung array (the embedding represents the ionic
        lattice).
    array : ``"fitted"`` or ``"evjen"``, optional
        Which MR6b Madelung array builder to use.
    cluster_charge : int, optional
        Ionic formal net charge of the QM region; default
        ``sum(formal[Z] for QM atoms)``.
    methods : tuple of str, optional
        Which of ``rhf`` / ``casci`` / ``casscf`` / ``caspt2`` to run.
    char_len : float, optional
        Characteristic lattice length for the array zone radii;
        default the shortest lattice-vector norm (the cubic constant
        for rock salt).
    r_exact_factor : float, optional
        Inner (exact formal) zone radius as a multiple of *char_len*.
    r_fit_factor : float, optional
        Fit-shell outer radius as a multiple of *char_len*.
    probe_factor : float, optional
        Off-site probe ball radius = *probe_factor* × *qm_radius*.
    evjen_half_side : int, optional
        Evjen block half-side in units of a/2.
    ridge : float, optional
        Tikhonov ridge for the Derenzo fitted array.
    n_probes : int, optional
        Number of off-site probe points.
    rhf_conv : float, optional
        RHF SCF convergence tolerance (energy).
    casscf_conv : float, optional
        CASSCF gradient convergence tolerance.

    Returns
    -------
    EmbeddedClusterResult
    """
    a = (
        float(np.min(np.linalg.norm(np.asarray(system.lattice, float), axis=0)))
        if char_len is None
        else float(char_len)
    )
    cxyz = _center_xyz(system, center)

    # 1. Carve the QM region.
    _qm_mol, qm = carve_cluster(system, center, qm_radius)
    if cluster_charge is None:
        cluster_charge = int(round(sum(formal[int(Z)] for Z, _ in qm)))
    atoms = [vq.Atom(int(Z), list(map(float, r))) for Z, r in qm]
    mol = vq.Molecule(atoms, charge=int(cluster_charge), multiplicity=1)

    # 2. Build the Madelung embedding array (MR6b) + its provenance.
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
        pos, q = evjen_array(system, qm, cxyz, formal, evjen_half_side, a)
    else:
        raise ValueError(f"array must be 'fitted' or 'evjen', got {array!r}")
    pos = np.asarray(pos, float)
    q = np.asarray(q, float)
    # Independent test probes → the embedding-quality figure of merit.
    test = offsite_probe_ball(
        cxyz, all_sites, probe_factor * qm_radius, n_probes, seed=99
    )
    pot_rms = _rms(
        finite_array_potential(pos, q, test)
        - embedding_target(system, qm, formal, test)
    )

    # 3. Optional AIMP frontier shell (MR7): select bare (0s.0s) entries
    #    for each element, build an ECP library, and create ECP centres
    #    for frontier ions between the QM cluster and point-charge array.
    aimp_ecp_centres: list = []
    aimp_lib_name = ""
    aimp_share_dir = ""
    if aimp_entries is not None:
        # Default AIMP shell: one lattice constant beyond the QM cluster
        # (captures the nearest-neighbour shell outside the QM region).
        a_radius = aimp_radius if aimp_radius is not None else qm_radius + a
        # Select bare entries (one per element).
        bare_entries: dict[str, AIMPEntry] = {}
        for e in aimp_entries:
            if e.m1_n_terms > 0 and not e.has_valence():
                if e.element not in bare_entries:
                    bare_entries[e.element] = e
        if bare_entries:
            aimp_lib_name, aimp_share_dir = write_aimp_ecp_library(
                list(bare_entries.values()), _ELEMENT_Z
            )
            # Find frontier ions: within aimp_radius, outside QM cluster.
            qm_pos_set = {tuple(np.round(r, 4)) for _, r in qm}
            for Z, r in all_sites:
                if tuple(np.round(r, 4)) in qm_pos_set:
                    continue
                d = float(np.linalg.norm(np.asarray(r) - cxyz))
                if d <= a_radius + 1e-9:
                    sym = (
                        vq.Atom(int(Z), [0.0, 0.0, 0.0]).symbol
                        if hasattr(vq.Atom, "symbol")
                        else ""
                    )  # noqa: E501
                    # Look up element symbol from Z
                    for sym_name, z_val in _ELEMENT_Z.items():
                        if z_val == int(Z):
                            if sym_name in bare_entries:
                                ec = ECPCenter()
                                ec.Z = int(Z)
                                ec.xyz = [float(x) for x in r]
                                aimp_ecp_centres.append(ec)
                            break

    # 4. Embedded RHF + the requested CAS methods (Milestone 1 enabler).
    basis_obj = vq.BasisSet(mol, basis)
    ext_pos = [list(map(float, p)) for p in pos]
    ext_q = [float(x) for x in q]
    ecp_kwargs = {}
    if aimp_ecp_centres:
        ecp_kwargs = dict(
            ecp_centers=aimp_ecp_centres,
            ecp_library=aimp_lib_name,
            ecp_share_dir=aimp_share_dir,
        )
    S, Hcore, E_nuc, jk = embedded_pieces(mol, basis_obj, ext_pos, ext_q, **ecp_kwargs)
    opts = RHFOptions()
    opts.conv_tol_energy = rhf_conv
    opts.conv_tol_grad = max(rhf_conv * 1e3, 1e-8)
    opts.max_iter = 400
    rhf = run_rhf_scf_with_jk(
        basis_obj, mol.n_electrons(), S, Hcore, float(E_nuc), jk, opts, molecule=mol
    )
    C = np.asarray(_attr(rhf, "mo_coeffs", "C", "orbitals"))
    H = embedded_mo_hamiltonian(mol, basis_obj, C, Hcore, E_nuc)
    n_core = (mol.n_electrons() - nae) // 2

    energies: dict[str, float] = {}
    if "rhf" in methods:
        energies["rhf"] = _energy(rhf)
    ci = None
    if "casci" in methods or "caspt2" in methods:
        ci = casci(
            H.h1e,
            H.h2e,
            nae,
            nao,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
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
        aimp_n_centres=len(aimp_ecp_centres),
    )
