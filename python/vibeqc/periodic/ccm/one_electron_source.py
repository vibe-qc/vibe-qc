"""One-electron sources of the neutral real-Gamma torus (#519, RG-8b).

The real-Gamma direct drivers assemble ``S``, ``h = T + V_ne`` and the nuclear
repulsion of the torus by folding unit-cell lattice matrices onto it
(:func:`~vibeqc.periodic.ccm.direct.ccm_direct_oneelectron`, the classic
libint lattice sums with vibe-qc's Ewald-gauge nuclear attraction). A
MANTA-sourced torus needs MANTA's one-electron side as well, or it mixes two
finite sources. :class:`MANTAOneElectronSource` supplies it: the overlap,
kinetic and Ewald nuclear-attraction cell blocks of MANTA's own producers
(``compute_one_electron_lattice``, ``compute_nuclear_lattice``) on the
translation domain and Gamma vector set of MANTA's fitted RI HF for the same
BvK mesh, folded onto the torus exactly as the classic blocks are, and the
nuclear repulsion as MANTA's RI HF evaluates it.

A one-electron source answers ``torus(ccm) -> (S, h, e_nn, record)`` with
supercell matrices, the total nuclear repulsion of the torus and a JSON-able
record that joins the neutral-source identity (RG-2) under ``one_electron``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["MANTA_ONE_ELECTRON_ENGINE", "MANTAOneElectronSource",
           "manta_rhf_domains", "manta_nuclear_repulsion_per_cell"]

#: Engine identity of MANTA's one-electron producers.
MANTA_ONE_ELECTRON_ENGINE = "manta-bloch-one-electron+periodic-nuclear"


def manta_rhf_domains(orbital, auxiliary, system, nrep, cutoff, options, budget):
    """The domains MANTA's fitted RI HF builds for this BvK mesh.

    The UHF factory for UHF options (an odd-electron cell), else the RHF one;
    both return the same geometric schema (translations, Gamma vectors,
    exchange segments and their offsets). The factory admits the upper bound
    of a whole RI HF run under ``budget``, which scales with
    ``options.scf.maximum_iterations``; pass the options of the MANTA run the
    torus is compared with, or size the budget for them.
    """
    from vibeqc import manta

    mesh = manta.RegularKMesh(tuple(int(n) for n in nrep), (0, 0, 0))
    if isinstance(options, manta.PeriodicRIUHFOptions):
        domains = manta.build_periodic_ri_uhf_domains(
            orbital, auxiliary, system, mesh, float(cutoff), options, budget)
    else:
        domains = manta.build_periodic_ri_rhf_domains(
            orbital, auxiliary, system, mesh, float(cutoff), options, budget)
    return mesh, domains


def manta_nuclear_repulsion_per_cell(system, gamma_vectors, *, alpha,
                                     nuclear_cutoff_bohr):
    """Nuclear repulsion per cell as MANTA's RI HF evaluates it.

    Ewald (1921) zero-average point-charge energy with a tin-foil surface and
    a uniform compensating background: the real-space images within
    ``nuclear_cutoff_bohr`` of each nucleus (self image excluded), the
    reciprocal sum over the explicit ``gamma_vectors`` (MANTA's primed Gamma
    set, ``G = 0`` excluded), the self term and the background term, at the
    split ``alpha`` (``manta_periodic_hf_source.hpp``, ``nuclear_energy``).
    Returns ``(energy, components)``.
    """
    lattice = np.asarray(system.lattice, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    atoms = list(system.unit_cell_molecule().atoms)
    z = np.array([float(a.Z) for a in atoms])
    xyz = np.array([np.asarray(a.xyz, dtype=float) for a in atoms])
    alpha = float(alpha)
    cutoff = float(nuclear_cutoff_bohr)
    # Lattice images covering a sphere of radius cutoff + the cell span.
    span = float(np.max(np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1))) if len(atoms) > 1 else 0.0
    inv = np.linalg.inv(lattice)
    reach = [int(math.ceil((cutoff + span) * float(np.linalg.norm(inv[i])))) + 1
             for i in range(3)]
    dims = int(system.dim)
    ranges = [range(-reach[i], reach[i] + 1) if i < dims else range(0, 1)
              for i in range(3)]
    shifts = np.array([(i, j, k) for i in ranges[0] for j in ranges[1]
                       for k in ranges[2]], dtype=float) @ lattice.T
    real = math.fsum(
        0.5 * z[a] * z[b] * math.erfc(alpha * d) / d
        for a in range(len(atoms)) for b in range(len(atoms))
        for d in np.linalg.norm(xyz[b] + shifts - xyz[a], axis=1)
        if 0.0 < d <= cutoff)
    g = np.asarray(gamma_vectors, dtype=float).reshape(-1, 3)
    norm = np.einsum("gi,gi->g", g, g)
    structure = np.exp(1j * (g @ xyz.T)) @ z
    reciprocal = math.fsum(
        2.0 * math.pi / volume * np.exp(-norm / (4.0 * alpha * alpha)) / norm
        * np.abs(structure) ** 2)
    self_term = -alpha / math.sqrt(math.pi) * float(np.sum(z * z))
    background = -0.5 * math.pi / volume / (alpha * alpha) * float(np.sum(z)) ** 2
    components = {"real": real, "reciprocal": reciprocal, "self": self_term,
                  "background": background}
    return real + reciprocal + self_term + background, components


class MANTAOneElectronSource:
    """MANTA's one-electron side of its fitted RI HF, on the torus.

    ``options`` and ``reciprocal_energy_cutoff`` are those of the matching
    :class:`~vibeqc.periodic.ccm.fit_source.MANTAFitSource` (the same finite
    source of ``run_periodic_ri_rhf_from_cutoffs`` / the UHF sibling): pair
    cutoff, nuclear split ``alpha``, nuclear image cutoff and reciprocal
    batch. The overlap, kinetic and nuclear-attraction cell blocks come from
    MANTA's producers on that method's translation domain and Gamma vectors
    and fold onto the torus with the classic one-electron fold; the nuclear
    repulsion is :func:`manta_nuclear_repulsion_per_cell` times ``N_c``.

    Measured (``tests/test_ccm_manta_one_electron.py``): the torus ``S`` and
    ``h`` equal MANTA's RI HF overlap and core Hamiltonian moved onto the
    torus to 2e-16, the nuclear repulsion its receipt to rounding, and the
    real-Gamma HF on MANTA fits and this source equals MANTA's RI HF energy.
    """

    engine = MANTA_ONE_ELECTRON_ENGINE

    def __init__(self, options, reciprocal_energy_cutoff, *,
                 auxiliary_basis=None, budget=None):
        cutoff = float(reciprocal_energy_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "MANTAOneElectronSource: reciprocal_energy_cutoff must be "
                f"finite and positive, got {reciprocal_energy_cutoff!r}")
        for name in ("pair_cutoff_bohr", "alpha", "nuclear_cutoff_bohr",
                     "reciprocal_batch"):
            if not hasattr(options, name):
                raise TypeError(
                    "MANTAOneElectronSource: options must be a MANTA periodic "
                    f"RI HF options object (missing {name!r}); got "
                    f"{type(options).__name__}")
        self.options = options
        self.reciprocal_energy_cutoff = cutoff
        self.auxiliary_basis = auxiliary_basis
        self.budget = budget

    def torus(self, ccm):
        from vibeqc import manta
        from vibeqc._vibeqc_core import BasisSet, LatticeCell
        from vibeqc.aux_basis import default_aux_for, make_aux_basis_set
        from vibeqc.manta_periodic import (
            compute_nuclear_lattice,
            compute_one_electron_lattice,
        )
        from vibeqc.manta_scf import IntegralEngineOptions
        from .direct import _fold_lattice_matrix_to_torus

        unit = ccm.unit_system
        if int(unit.dim) != 3:
            raise NotImplementedError(
                "MANTAOneElectronSource: MANTA's periodic HF sources are 3-D")
        mol = unit.unit_cell_molecule()
        orbital = BasisSet(mol, ccm.basis_name)
        aux_name = str(self.auxiliary_basis or default_aux_for(ccm.basis_name))
        auxiliary = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0)
        budget = self.budget if self.budget is not None else manta.ResourceBudget()
        o = self.options
        _, domains = manta_rhf_domains(
            orbital, auxiliary, unit, ccm.nrep, self.reciprocal_energy_cutoff,
            o, budget)
        lattice = np.asarray(unit.lattice, dtype=float)
        translations = np.asarray(domains.translations, dtype=float)
        index = np.round(np.linalg.solve(lattice, translations.T).T).astype(int)
        if not np.allclose(index @ lattice.T, translations, atol=1e-9):
            raise ValueError(
                "MANTAOneElectronSource: the MANTA translation domain is not "
                "on the lattice")
        cells = [LatticeCell(unit, np.asarray(i, dtype=np.int32)) for i in index]
        engine = IntegralEngineOptions(backend="manta", fallback="error",
                                       budget=budget)
        pair = float(o.pair_cutoff_bohr)
        gamma = np.ascontiguousarray(domains.reciprocal_vectors, dtype=float)
        overlap = compute_one_electron_lattice(
            orbital, cells, operation="overlap", pair_cutoff_bohr=pair,
            integral_engine=engine)
        kinetic = compute_one_electron_lattice(
            orbital, cells, operation="kinetic", pair_cutoff_bohr=pair,
            integral_engine=engine)
        nuclear = compute_nuclear_lattice(
            orbital, unit, cells, gamma, alpha=float(o.alpha),
            nuclear_cutoff_bohr=float(o.nuclear_cutoff_bohr),
            reciprocal_batch=int(o.reciprocal_batch), pair_cutoff_bohr=pair,
            integral_engine=engine)
        n_mu = int(orbital.nbasis)
        S = _fold_lattice_matrix_to_torus(overlap, ccm.nrep, n_mu)
        h = (_fold_lattice_matrix_to_torus(kinetic, ccm.nrep, n_mu)
             + _fold_lattice_matrix_to_torus(nuclear, ccm.nrep, n_mu))
        S, h = 0.5 * (S + S.T), 0.5 * (h + h.T)
        e_cell, parts = manta_nuclear_repulsion_per_cell(
            unit, gamma, alpha=float(o.alpha),
            nuclear_cutoff_bohr=float(o.nuclear_cutoff_bohr))
        record = {
            "provider": "manta",
            "engine": self.engine,
            "reciprocal_energy_cutoff_ha": float(self.reciprocal_energy_cutoff),
            "pair_cutoff_bohr": pair,
            "alpha": float(o.alpha),
            "nuclear_cutoff_bohr": float(o.nuclear_cutoff_bohr),
            "translations": int(len(cells)),
            "gamma_vectors": int(len(gamma)),
            "nuclear_repulsion_per_cell": {k: float(v) for k, v in parts.items()},
        }
        return S, h, float(e_cell) * float(ccm.n_cells), record
