"""NiO bulk — UKS+LDA+U / STO-3G — DFT+U recipe for the canonical
transition-metal-oxide validation system.

Rock-salt NiO is the textbook DFT+U benchmark: plain LDA/GGA
gives a metallic / sub-eV gap (wrong), Hubbard ``U`` on Ni-3d
opens it toward the experimental ~4.3 eV. This script wires up
the open-shell multi-k +U surface (Increment 4d-bipole UKS) for
NiO and sweeps ``U_eff ∈ {0, 4, 7} eV`` on Ni's d-channel,
printing the ``e_dft_plus_u`` contribution + the α-spin
HOMO/LUMO gap at Γ as a quick spectroscopic proxy.

**Runtime warning.** NiO BIPOLE integral builds are heavy even
at STO-3G: a 2x2x2 mesh runs ~15+ minutes per SCF on a laptop;
the Γ-only mesh used by default below is faster but still not
free. Set ``kpoints=(2, 2, 2)`` for a real benchmark; keep the
default for a tractable surface check.

**Caveats — this is a recipe, not a publication benchmark.**

* STO-3G is far too small for transition-metal d-orbitals — a
  real run uses cc-pVDZ-DK, def2-TZVP, or pob-TZVP.
* Ferromagnetic ordering is unphysical for NiO; the real ground
  state is AFM-II, which needs a *doubled* primitive cell so
  Ni alternates spin direction.
* LDA is the cheapest XC; PBE / HSE06 are better choices.

The full multi-k +U *surface* is exercised by the test suite
(``tests/test_dft_plus_u.py::test_run_periodic_job_multi_k_*``,
``test_run_periodic_job_uks_dft_plus_u_runs``, ...) — this script
documents the user-facing pattern on a physically meaningful
system.

Run (long — ~minutes per SCF):
    .venv/bin/python examples/periodic/input-bipole-nio-uks-plus-u.py
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.periodic_runner import run_periodic_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

ANG2BOHR = 1.0 / 0.529177210903

# Rock-salt NiO conventional edge a = 4.164 Å → FCC primitive cell.
a = 4.164 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
atoms = [
    vq.Atom(28, [0.0, 0.0, 0.0]),                 # Ni
    vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),      # O at body-centre
]
# Ferromagnetic ordering with Ni's d⁸ giving 2 unpaired electrons
# (high-spin Ni²⁺: t₂g⁶ eg² → S=1, Ms=1, 2S+1=3). A real AFM-II
# benchmark would need a doubled cell.
sysp = vq.PeriodicSystem(3, lattice, atoms, charge=0, multiplicity=3)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

# Sweep U on Ni's d-channel (atom_index=0, l=2).
U_sweep_ev = [0.0, 4.0, 7.0]
results = []

for U_ev in U_sweep_ev:
    out_stem = HERE / f"{STEM}_U{int(U_ev):d}"
    print(f"\n=== NiO UKS+PBE  U={U_ev:.1f} eV  ============================")
    dft_plus_u = (
        [vq.HubbardSite(atom_index=0, l=2, U_ev=U_ev)] if U_ev > 0.0 else None
    )
    result = run_periodic_job(
        sysp,
        basis,
        method="UKS",
        functional="lda",
        jk_method="bipole",
        kpoints=(1, 1, 1),  # Γ-only — multi-k via (N,N,N), heavier.
        output=str(out_stem),
        max_iter=80,
        conv_tol_energy=1e-6,
        initial_guess="SAD",
        dft_plus_u=dft_plus_u,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
    )

    # α-spin Γ-point gap as a quick spectroscopic proxy.
    eps_a_gamma = np.asarray(result.mo_energies_alpha[0])
    eps_b_gamma = np.asarray(result.mo_energies_beta[0])
    n_a = int(result.n_alpha) if hasattr(result, "n_alpha") else None
    n_b = int(result.n_beta) if hasattr(result, "n_beta") else None

    def _gap(eps, n_occ):
        if n_occ is None or n_occ <= 0 or n_occ >= len(eps):
            return float("nan")
        return float(eps[n_occ] - eps[n_occ - 1])

    gap_a = _gap(eps_a_gamma, n_a)
    gap_b = _gap(eps_b_gamma, n_b)
    HA2EV = 27.211386245988
    e_du = float(getattr(result, "e_dft_plus_u", 0.0))

    print(
        f"  converged={result.converged}  n_iter={result.n_iter}  "
        f"E={result.energy:.6f} Ha"
    )
    print(
        f"  e_dft_plus_u = {e_du:.6f} Ha ({e_du * HA2EV:.3f} eV)"
    )
    print(
        f"  α HOMO/LUMO gap at Γ = {gap_a:.4f} Ha "
        f"({gap_a * HA2EV:.3f} eV)"
    )
    print(
        f"  β HOMO/LUMO gap at Γ = {gap_b:.4f} Ha "
        f"({gap_b * HA2EV:.3f} eV)"
    )
    results.append((U_ev, e_du, gap_a, gap_b))


print("\n=== Summary ============================================")
print("   U(eV)   e_dft+u(Ha)   α-gap(eV)   β-gap(eV)")
for U_ev, e_du, gap_a, gap_b in results:
    print(
        f"   {U_ev:>5.1f}   {e_du:>10.5f}   {gap_a * 27.2114:>8.3f}   "
        f"{gap_b * 27.2114:>8.3f}"
    )
