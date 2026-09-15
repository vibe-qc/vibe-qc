# DFT+U (Hubbard correction)

See also the [tutorial](../tutorial/dft_plus_u.md) for a worked
transition-metal-oxide walkthrough.

vibe-qc implements the **Dudarev rotationally-invariant** flavour of
DFT+U on Mulliken-style AO projectors. The same surface works for
molecular and periodic calculations, closed and open shell,
energies and analytic gradients.

The Hubbard correction adds an on-site penalty to fractional
occupation of a chosen (atom, l-channel), opening the gap in
strongly-correlated transition-metal oxides like NiO or
self-interaction-corrected `d⁵` ions, and stabilising integer
filling against the spurious LDA/GGA delocalisation. The math is

$$
E_U = \tfrac{1}{2} \sum_{A,l} U_\text{eff}^{A,l} \,
      \Big(\operatorname{tr}\, n^{A,l} - \operatorname{tr}\, (n^{A,l})^2\Big),
\qquad
U_\text{eff}^{A,l} = U^{A,l} - J^{A,l},
$$

with the AO-projected occupation matrix

$$
n^{A,l}_{mm'} = (S P S)_{mm'} \quad\text{for } m, m' \in (A, l).
$$

The corresponding Fock contribution is

$$
V_U^{A,l}_{mm'} = U_\text{eff}^{A,l} \big(\tfrac{1}{2}\delta_{mm'} - n^{A,l}_{mm'}\big),
$$

scattered back into the AO basis and then **sandwiched with S** on both
sides, i.e. `F_AO += S V_U_AO S`, so that the SCF is variational with
respect to the total energy in a non-orthogonal AO basis. (Orthonormal-
projector codes, plane-wave PAW, show `V_U` un-sandwiched. For
Gaussian AO bases the sandwich is load-bearing; without it the analytic
gradient mismatches finite-difference by `O(10 mHa/bohr)` and the SCF
converges to a non-variational stationary point.)

Cite Dudarev *et al.* PRB **57**, 1505 (1998) for the rotationally-
invariant formulation, and Cococcioni & de Gironcoli PRB **71**, 035105
(2005) for the linear-response prescription for `U_eff`, both fire
automatically into the run's `.bibtex` / `.references` when
`dft_plus_u=` is set.

## The minimal API

A Hubbard site is a `(atom_index, l, U_eff)` triple. The user
constructs it with `U_ev=…` (eV) and optionally `J_ev=…`:

```python
import vibeqc as vq

site = vq.HubbardSite(atom_index=0, l=2, U_ev=4.0)
# Or with explicit J:
site = vq.HubbardSite(atom_index=0, l=2, U_ev=7.5, J_ev=0.9)
```

Then pass `dft_plus_u=[…]` through any of vibe-qc's drivers:

| Driver | Open-shell? | Multi-k? | Surface |
|---|---|---|---|
| `vq.run_rhf` / `vq.run_rks` | no  | n/a   | molecular closed-shell |
| `vq.run_uhf` / `vq.run_uks` | yes | n/a   | molecular open-shell |
| `vq.run_job(...)` | both | n/a | high-level molecular dispatcher; `optimize=True` works too |
| `vq.run_rhf_periodic_gamma` / `vq.run_rks_periodic_gamma` | no | Γ-only | the simple periodic path |
| `vq.run_rks_periodic` | no | yes  | multi-k periodic RKS (NiO bandgap / TMO target) |
| `vq.run_pbc_bipole_uhf` / `vq.run_pbc_bipole_uks` | yes | yes | open-shell periodic via BIPOLE |
| `vq.run_periodic_job(...)` | both | yes | high-level periodic dispatcher |

The +U energy is surfaced as `result.e_dft_plus_u` on every driver's
result type.

## A worked molecular example

H₂O / STO-3G with a Hubbard penalty on the O 2p channel:

```python
import vibeqc as vq

molecule = vq.Molecule.from_xyz_string("""
3
H2O
O   0.000000  0.000000  0.117790
H   0.000000  0.755453 -0.471161
H   0.000000 -0.755453 -0.471161
""")
basis = vq.BasisSet(molecule, "sto-3g")

result = vq.run_rks(
    molecule, basis,
    functional="pbe",
    dft_plus_u=[vq.HubbardSite(atom_index=0, l=1, U_ev=4.0)],
    output="h2o-pbe-plus-u",
)
print(f"E_total      = {result.energy:.6f} Ha")
print(f"e_dft_plus_u = {result.e_dft_plus_u:.6f} Ha")
```

`atom_index=0` selects the O atom (first in the geometry) and `l=1`
selects all `p`-AOs on that atom (Mulliken projector).

## A worked periodic example, NiO multi-k UKS+U

Rock-salt NiO is the textbook DFT+U benchmark: plain PBE gives a
sub-eV gap, `U ≈ 7` eV on Ni-3d opens it toward the ~4.3 eV
experimental value. The runnable script lives at
[`examples/periodic/input-bipole-nio-uks-plus-u.py`](../../examples/periodic/input-bipole-nio-uks-plus-u.py)
- here's the gist:

```python
import vibeqc as vq
from vibeqc.periodic_runner import run_periodic_job

# Rock-salt NiO primitive (a = 4.164 Å) — Ni + O / 2 atoms / FCC.
# (Real benchmark would use a doubled cell for AFM-II ordering.)
sysp = vq.PeriodicSystem(3, lattice, atoms, charge=0, multiplicity=3)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

result = run_periodic_job(
    sysp, basis,
    method="UKS", functional="pbe", jk_method="bipole",
    kpoints=(2, 2, 2),
    dft_plus_u=[vq.HubbardSite(atom_index=0, l=2, U_ev=7.0)],  # Ni-3d
    output="nio-uks-pbe-plus-u",
)
```

`atom_index=0` is Ni (first in the geometry), `l=2` is the d-channel.
The driver auto-routes through the open-shell BIPOLE multi-k path; +U
fires per spin per k internally.

## What `dft_plus_u=` does inside

* **Per (atom, l) channel**, the AOs in that channel form a `(2l+1)`-
  dimensional block, for `l=2` that's 5 AOs per Ni atom.
* Each SCF iteration computes `n^{A,l} = (S P S)_{(A,l)}` (per spin σ in
  the open-shell case). For multi-k periodic systems this is averaged
  over the k-mesh as `n = Σ_k w_k Re[(S(k) P(k) S(k))_{(A,l)}]`.
* The Dudarev energy `E_σ = (U_eff/2)(tr n − tr n²)` is added to the
  electronic energy.
* The corresponding `V_AO` is sandwiched with S and added to the AO
  Fock (or to each F(k) for multi-k via a real-space-convolution path
  that keeps DIIS / FDS / diag all consistent).

## What's wired today

* Molecular RHF / RKS / UHF / UKS, energy + analytic gradient. The
  gradient passes finite-difference parity at the `µHa/bohr` level
  on H₂O / N₂ / CH₃ test systems.
* `vq.run_job(..., optimize=True, dft_plus_u=[...])`, geometry
  optimisation with +U works end-to-end for molecules.
* Γ-only periodic RHF / RKS, energy.
* Multi-k periodic RKS, energy.
* Multi-k periodic UHF / UKS via the BIPOLE drivers, energy.

## What's queued

* **Γ-only periodic +U gradient** is wired via
  `compute_gradient_periodic_rhf_gamma(..., dft_plus_u=[...])` and the
  RKS sibling. Matches the molecular +U gradient in the molecular
  limit at the few-µHa/bohr level.
* **Multi-k BIPOLE +U gradient** for **open-shell** UHF / UKS is
  wired via the new `kmesh=` + `dft_plus_u=` kwargs on
  `compute_bipole_gradient_uhf` / `compute_bipole_gradient_uks`,
  and threaded through `bipole_optimize.relax_atoms` →
  `run_periodic_job(optimize=True, method="UHF"|"UKS",
  jk_method="bipole", dft_plus_u=[...])`. The Bloch-folded
  Pulay term `M(g) = Σ_k w_k Re[exp(-i k·g) V_AO S(k) P(k)]` is
  exact in the molecular limit; the BIPOLE baseline gradient has
  its own ~mHa/bohr Γ-W tile noise that is pre-existing and
  independent of +U.
* **Closed-shell BIPOLE +U**, both `run_pbc_bipole_rhf` /
  `_rks` and the gradient siblings now take `dft_plus_u=`. The
  `run_periodic_job(optimize=True, method="RHF"|"RKS",
  jk_method="bipole", dft_plus_u=[...])` path runs end-to-end.

## Common pitfalls

* **U is per (atom, l), not per orbital.** Passing `l=2` on a Ni
  atom in a basis whose `d`-channel actually has *two* shells (e.g.
  a TZ basis with 4d outer-core polarisation in addition to 3d) puts
  *all* of them in the same Mulliken block. If you want a single
  3d-shell projector you need a basis that doesn't expose the second.
* **`U_ev` is in eV, `U_eff_hartree` (HartreeAU) is what the kernel
  uses internally.** The `HubbardSite` dataclass converts for you.
* **Citation discipline (CLAUDE.md § 8).** When a job uses `+U`,
  Dudarev 1998 + Cococcioni 2005 fire automatically into the
  end-of-run citation block. Don't re-cite them by hand in tutorials.

## See also

* [Functionals](functionals.md), pick the underlying XC.
* [Geometry optimization](geometry_optimization.md), `optimize=True`.
* [Multi-k SCF](multi_k_scf.md), kmesh + Bloch sums.
* `python/vibeqc/dft_plus_u.py`, Python AO-projection helpers.
* `cpp/include/vibeqc/dft_plus_u.hpp`, C++ kernel API.
* The DFT+U implementation is complete and shipped on main.
