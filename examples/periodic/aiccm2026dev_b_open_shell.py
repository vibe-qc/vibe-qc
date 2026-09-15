"""Experimental unrestricted χ-CCM calculation on a Li doublet."""

from __future__ import annotations

import numpy as np

import vibeqc as vq


system = vq.PeriodicSystem(
    3,
    np.eye(3) * 15.0,
    [vq.Atom(3, [0.0, 0.0, 0.0])],
    multiplicity=2,
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# One cell is intentionally tiny: this example checks the complete route, not
# the infinite-crystal limit. Increase the extension before scientific use.
extension = (1, 1, 1)
uhf = vq.run_aiccm2026dev_b_uhf(
    system,
    basis,
    lattice_extension=extension,
    backend="ri",
    progress=False,
)
uks = vq.run_aiccm2026dev_b_uks(
    system,
    basis,
    "lda",
    lattice_extension=extension,
    backend="ri",
    progress=False,
)
ump2 = vq.run_aiccm2026dev_b_ump2(
    system,
    basis,
    lattice_extension=extension,
    progress=False,
)
uccsdt = vq.run_aiccm2026dev_b_uccsd_t(
    system,
    basis,
    lattice_extension=extension,
    progress=False,
)
properties = vq.derive_aiccm2026dev_b_scf_properties(uhf, system, basis)

print(f"UHF       {uhf.energy: .12f} Ha/cell")
print(f"UKS/LDA   {uks.energy: .12f} Ha/cell")
print(f"UMP2      {ump2.energy: .12f} Ha/cell")
print(f"UCCSD(T)  {uccsdt.energy: .12f} Ha/cell")
print(f"<S^2>     {properties.s_squared: .12f}")
print("spin populations", properties.mulliken_spin_populations)
