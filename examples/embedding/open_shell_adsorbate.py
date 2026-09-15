#!/usr/bin/env python3
"""Example: open-shell (spin-polarised) embedded surface via ASE.

:class:`~vibeqc.periodic_embedding.ase_calc.EmbeddedSurfaceCalculator`
accepts ``charge`` and ``multiplicity`` and threads them into the
:class:`PeriodicSystem` it builds from the ASE ``Atoms``.  An
odd-electron cell (or a deliberately non-singlet one) needs

  * ``multiplicity > 1``, otherwise the unit-cell molecule fails its
    parity check (``n_electrons and multiplicity are inconsistent``)
    *before* any embedding work runs, and
  * a spin-polarised density route, via ``scf_method="uhf"``
    (forwarded to :func:`run_embedded_surface`).

Here: a 3-atom H chain in a slab cell (3 electrons -> doublet), with the
bottom two atoms as substrate and the top atom as region I.  We drive it
through ASE to get the embedded energy (eV) and the numerical force
(eV/A) on the mobile region-I atom, exactly what an ASE optimiser would
call.

Closed-shell counterpart + the Layer-B adsorption energy via Lloyd's
formula: ``simple_adsorbate.py``.

NB: the periodic-embedding subpackage is experimental and not exported
from the top-level ``vibeqc`` namespace; it stays internal until
thick-slab GDF parity (see ``handovers/HANDOVER_GF_EMBEDDING.md`` SS7).

Usage:
    .venv/bin/python examples/embedding/open_shell_adsorbate.py
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator


def build_h3_slab() -> Atoms:
    """A 3-atom H chain in a slab cell (long c-axis -> dim=2).

    Three H atoms -> 3 electrons (odd) -> an open-shell doublet.  The
    chain sits on the in-plane centre so the x/y forces vanish by
    symmetry and only the surface-normal (z) force is nonzero.
    """
    a_lat, c_vac, spacing = 4.0, 18.0, 1.5  # Angstrom
    z0 = (c_vac - 2 * spacing) / 2.0
    atoms = Atoms(
        "H3",
        positions=[(a_lat / 2, a_lat / 2, z0 + i * spacing) for i in range(3)],
        cell=[a_lat, a_lat, c_vac],
        pbc=[True, True, True],
    )
    atoms.set_tags([0, 0, 1])  # bottom 2 = substrate, top = region I
    return atoms


def main():
    print("=" * 60)
    print(" Open-shell embedded surface: H3 doublet via ASE")
    print("=" * 60)

    atoms = build_h3_slab()

    # multiplicity=2 (doublet) + scf_method="uhf" is the spin-polarised
    # route.  Without them this cell raises the parity error before any
    # embedding work runs.
    calc = EmbeddedSurfaceCalculator(
        basis_name="sto-3g",
        surface_k_mesh=(1, 1),
        contour_n_nodes=24,
        e_bottom=-2.2,
        e_fermi=-1.0,
        lat_cutoff_bohr=15.0,
        charge=0,
        multiplicity=2,  # 3 electrons -> doublet
        scf_method="uhf",  # spin-polarised density
        force_atom_indices=[2],  # only the region-I atom is mobile
    )
    atoms.calc = calc

    print("\nSystem: 3-atom H chain (slab cell), tags", atoms.get_tags().tolist())
    print(f"charge = {calc.charge}, multiplicity = {calc.multiplicity}")
    print("scf_method = uhf (spin-polarised)")

    energy = atoms.get_potential_energy()  # eV
    print(f"\n  Embedded energy: {energy:.6f} eV")

    forces = atoms.get_forces()  # eV / Angstrom
    print("  Forces (eV/A):")
    for i, f in enumerate(forces):
        tag = "region I" if atoms.get_tags()[i] else "substrate"
        print(f"    atom {i} ({tag:9s}): {f[0]:+.4f} {f[1]:+.4f} {f[2]:+.4f}")

    # Region-I occupation from the underlying embedded-surface result.
    res = calc.last_result
    if res is not None:
        print(f"\n  Region-I occupation: {res.occupation:.6f} e-")

    # The on-axis chain forces the in-plane components to (near) zero.
    fz = forces[2, 2]
    print(f"\n  Surface-normal force on the mobile atom: {fz:+.4f} eV/A")
    assert np.allclose(forces[:2], 0.0)  # masked substrate stays zero

    print("\n" + "=" * 60)
    print(" Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
