---
myst:
  html_meta:
    "description": "Reference for Green's-function ionisation potentials and electron affinities in vibe-qc: run_job(method='ovgf') and result.ovgf (diagonal second-order GF2 quasiparticle energies and pole strengths), the renormalised third-order HOMO value, the open-shell spin-orbital kernel, vibeqc.propagator, msindo_ovgf, size guards and what is and is not the literal OVGF."
    "og:title": "vibe-qc user guide - Green's-function ionisation potentials (GF2 and OVGF)"
---

# Green's-function ionisation potentials and electron affinities (GF2 and OVGF)

Koopmans' theorem reads an ionisation potential off an orbital energy and is
wrong by one to two electronvolts. The one-particle electron propagator corrects
every orbital energy of one reference calculation at once through a
frequency-dependent self-energy and attaches to each ionisation a pole strength
that says whether the one-hole picture still holds. `run_job(method="ovgf")`
runs a Hartree-Fock reference (RHF for singlets, UHF otherwise) and then solves the
quasiparticle equation with the **diagonal second-order self-energy (GF2)** for the
outer-valence orbitals. It is production for that scope, including open-shell
references through a spin-orbital kernel. The result list `result.ovgf` holds the GF2
quasiparticle energies and pole strengths. For small closed-shell systems the `.out`
additionally prints one **renormalised** HOMO value that includes the full diagonal
third-order self-energy. What is not implemented is the literal outer-valence Green's
function (OVGF) of von Niessen, Schirmer and Cederbaum with its A, B and C screening
recipes; the name `ovgf` records the lineage
({vibeqc-cite}`cederbaum_ovgf_1975`, {vibeqc-cite}`vonniessen_ovgf_1984`), as the
`vibeqc.propagator` module docstring states. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels). The
tutorial [Ionisation potentials from the electron propagator](../tutorial/ovgf_ionization.md)
works water through in detail, with a Delta-SCF comparison and the third-order
theory; this page is the reference.

## Quick start

Water at RHF/6-31G\*:

```python
from vibeqc import Atom, Molecule, run_job

ang = 1.8897259886
EV = 27.211386245988
mol = Molecule([
    Atom(8, [0.0, 0.0, 0.1173 * ang]),
    Atom(1, [0.0, 0.7572 * ang, -0.4692 * ang]),
    Atom(1, [0.0, -0.7572 * ang, -0.4692 * ang]),
], charge=0, multiplicity=1)

result = run_job(mol, basis="6-31g*", method="ovgf", output="h2o_ovgf")
print(result)                                            # RHFResult+OVGF(4 quasiparticle orbitals)
n_occ = mol.n_electrons() // 2
for q in result.ovgf:                                    # one QuasiparticleResult per orbital
    tag = "HOMO" if q.orbital == n_occ - 1 else "LUMO" if q.orbital == n_occ else ""
    print(f"MO {q.orbital} {tag:>4}  Koopmans {-q.eps_scf * EV:8.3f} eV  "
          f"GF2 {-q.eps_qp * EV:8.3f} eV  Z = {q.pole_strength:.3f}  "
          f"({q.iterations} Dyson iterations, converged={q.converged})")
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the job takes under two
seconds. The list holds the three highest occupied orbitals and the LUMO. The
orbital-energy columns of the `.out` are energies in eV, negative for bound
orbitals; an ionisation potential is the negative of the corrected energy.

```text
MO 2       Koopmans   19.221 eV  GF2   17.977 eV  Z = 0.930  (3 Dyson iterations, converged=True)
MO 3       Koopmans   15.536 eV  GF2   13.448 eV  Z = 0.914  (3 Dyson iterations, converged=True)
MO 4 HOMO  Koopmans   13.539 eV  GF2   11.004 eV  Z = 0.907  (4 Dyson iterations, converged=True)
MO 5 LUMO  Koopmans   -5.839 eV  GF2   -5.292 eV  Z = 0.980  (3 Dyson iterations, converged=True)
```

and the `.out` prints, under `OVGF / GF2 quasiparticle energies (diagonal second-order self-energy)`,

```text
  MO  label  Koopmans(eV)  OVGF (eV)  pole Z
   4   HOMO      -13.5394   -11.0040  0.9067
   5   LUMO        5.8389     5.2917  0.9805
  OVGF ionization potential (HOMO) = 11.0040 eV
  OVGF electron affinity   (LUMO) = -5.2917 eV   (small-basis EA -- qualitative only)
  GF2(renorm) ionization potential (HOMO) = 11.6628 eV   (renormalized; pole Z=0.925)
```

(the table also lists MOs 2 and 3). Against the experimental vertical IP of 12.62
eV: Koopmans 13.54, GF2 11.00, renormalised 11.66, and 10.85 for a Delta-SCF cation
(from the tutorial). The GF2 value undershoots because second order treats the
two-hole-one-particle states as non-interacting and overestimates relaxation; the
renormalised value pulls it back. The negative "electron affinity" for water is
correct in sign but is the energy of a 6-31G\* virtual orbital without diffuse
functions and should not be quoted.

An open-shell molecule takes the UHF spin-orbital kernel automatically. The OH radical
at RHF/6-31G\*, with a 0.97 Å bond:

```python
from vibeqc import Atom, Molecule, run_job

ang = 1.8897259886
EV = 27.211386245988
oh = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 0.97 * ang])], 0, 2)
r = run_job(oh, basis="6-31g*", method="ovgf", output="oh_ovgf")
for spin, q in r.ovgf:                                   # (spin, QuasiparticleResult)
    print(f"{spin:5s} MO {q.orbital}  Koopmans {q.eps_scf * EV:9.4f}  "
          f"GF2 {q.eps_qp * EV:9.4f} eV  Z = {q.pole_strength:.3f}")
```

```text
alpha MO 4  Koopmans  -14.9808  GF2  -12.7864 eV  Z = 0.922
alpha MO 5  Koopmans    5.9828  GF2    5.5918 eV  Z = 0.984
beta  MO 3  Koopmans  -13.7108  GF2  -11.6509 eV  Z = 0.925
beta  MO 4  Koopmans    3.6091  GF2    0.2722 eV  Z = 0.939
```

The `.out` reports `OVGF first ionization potential = 11.6509 eV`: the smallest
ionisation potential over the alpha and beta channels. For each spin it corrects the
spin-HOMO and the spin-LUMO. The job takes about 3 s. The MSINDO analogue is
`msindo_ovgf([8, 1, 1], xyz)`, described [below](#msindo-and-other-references).

## Theory

The one-particle Green's function has poles at the exact ionisation potentials and
electron affinities, and the residues are the pole strengths. For a Hartree-Fock
reference the zeroth-order function has poles at the orbital energies, which is
Koopmans' theorem, and the exact function follows from the Dyson equation with a
self-energy $\Sigma(\omega)$ that carries relaxation and correlation. In the diagonal
approximation each ionisation $p$ solves the quasiparticle equation

$$
\omega_p=\varepsilon_p+\Sigma_{pp}(\omega_p),\qquad
Z_p=\Big(1-\frac{\partial\Sigma_{pp}}{\partial\omega}\Big|_{\omega_p}\Big)^{-1},
$$

by Newton iteration from $\omega=\varepsilon_p$ (default: `tol=1e-8`, at most 50
iterations), and $Z_p\in(0,1]$ is the weight of the one-hole configuration. With the
non-iterative choice `iterate=False` one gets $\varepsilon_p+\Sigma_{pp}(\varepsilon_p)$
instead, which differed from the converged value by 0.3 eV for the water HOMO.

The second-order self-energy for a closed-shell reference, with $i,j$ occupied and
$a,b$ virtual spatial orbitals and chemist's notation, is

$$
\Sigma^{(2)}_{pp}(\omega)=\sum_{iaj}\frac{\big[2(pi|aj)-(pj|ai)\big](pi|aj)}{\omega+\varepsilon_a-\varepsilon_i-\varepsilon_j}
+\sum_{aib}\frac{\big[2(pa|ib)-(pb|ia)\big](pa|ib)}{\omega+\varepsilon_i-\varepsilon_a-\varepsilon_b},
$$

the two-hole-one-particle ($2h1p$) channel that lowers ionisation potentials and the
two-particle-one-hole ($2p1h$) channel that matters for electron attachment. For a
UHF reference the same quantity is written over spin orbitals with antisymmetrised
integrals $\langle pq\|rs\rangle$,
$\Sigma_{pp}(\omega)=\tfrac12\sum_{iab}|\langle pi\|ab\rangle|^2/(\omega+\varepsilon_i-\varepsilon_a-\varepsilon_b)+\tfrac12\sum_{aij}|\langle pa\|ij\rangle|^2/(\omega+\varepsilon_a-\varepsilon_i-\varepsilon_j)$,
the electron-propagator analogue of UMP2. Both need the full MO two-electron tensor,
which is why the routes are size-guarded. The approximations are the diagonal
truncation (the ionised state keeps the shape of one canonical orbital), the
second-order truncation of the interaction and the Hartree-Fock reference. The
second-order truncation makes the relaxation part too large, so GF2 ionisation
potentials sit below experiment.

The **renormalised** value adds the diagonal third-order self-energy,
$\Sigma^{(3)}_c$ per channel $c\in\{2h1p,2p1h\}$, with its static ($W$) and dynamic
($U$) terms, and resums the ratio geometrically per channel,

$$
\Sigma_c(\omega)\approx\frac{\Sigma^{(2)}_c(\omega)}{1-\Sigma^{(3)}_c(\omega)/\Sigma^{(2)}_c(\omega)},
$$

falling back to the bare $\Sigma^{(2)}_c+\Sigma^{(3)}_c$ when $|\Sigma^{(3)}_c/\Sigma^{(2)}_c|$
exceeds `screen_cap` (0.5) or $\Sigma^{(2)}_c\approx0$. This is the outer-valence
Green's-function idea of Cederbaum ({vibeqc-cite}`cederbaum_ovgf_1975`) and of von
Niessen, Schirmer and Cederbaum ({vibeqc-cite}`vonniessen_ovgf_1984`), applied to the
full diagonal third-order self-energy, and it is deliberately not their A/B/C
screening partition, which cannot be validated here against an external code. Semiempirical
NDDO references work the same way (Danovich, {vibeqc-cite}`danovich_ovgf_1997`).

Where it helps: outer-valence ionisations of closed-shell molecules with $Z>0.85$, all
corrected from one reference calculation at MP2-like cost, to a few tenths of an
electronvolt in a good basis. Where it fails, and says so through a small $Z$:
inner-valence and core ionisations and satellite structure (the water $2a_1$ has
$Z=0.30$ at second order), strongly correlated references, and electron affinities in
a basis without diffuse functions.

## Options

There are no OVGF-specific `run_job` keywords: `method="ovgf"` selects the route,
`basis` sets the basis, and the reference is chosen by the multiplicity.

| What | Behaviour |
|---|---|
| Reference | RHF for `multiplicity == 1`, UHF otherwise. ROHF and Kohn-Sham references are not used by this route. |
| Orbitals corrected, closed shell | The three highest occupied orbitals and the LUMO (the whole outer valence for small molecules). |
| Orbitals corrected, open shell | The spin-HOMO and spin-LUMO of each spin channel. |
| Solver | Newton iteration on the Dyson equation, always iterated. |
| `result` | An object that forwards every attribute of the SCF result and adds `.ovgf` and `.method`. |
| `result.ovgf` | Closed shell: a list of `QuasiparticleResult`. Open shell: a list of `(spin, QuasiparticleResult)`, `spin` being `"alpha"` or `"beta"`. |

`QuasiparticleResult` (frozen dataclass, energies in Hartree): `orbital` (0-based spatial
MO index), `eps_scf` (Koopmans), `eps_qp`, `sigma` (the self-energy at the solution),
`pole_strength`, `iterations`, `converged`, and the property `correction`
(`eps_qp - eps_scf`).

The kernels are importable from `vibeqc.propagator` for your own orbital windows and
references (see the tutorial for a complete example):

| Function | Purpose |
|---|---|
| `quasiparticle_energies(eps, n_occ, eri_mo, orbitals, *, iterate=True, max_iter=50, tol=1e-8)` | Closed-shell GF2 for a list of MO indices; `quasiparticle_energy` does one. |
| `diagonal_self_energy(eps, n_occ, eri_mo, p, omega)` | $\Sigma^{(2)}_{pp}(\omega)$ in Hartree. |
| `unrestricted_quasiparticle_energies(eri_ao, Ca, Cb, eps_a, eps_b, n_a, n_b, orbitals, *, iterate=True, max_iter=50, tol=1e-8, max_spinorbitals=80)` | Spin-orbital GF2; `orbitals` are `(spin, index)` pairs. |
| `renormalized_quasiparticle_energies(eri_ao, Ca, Cb, eps_a, eps_b, n_a, n_b, orbitals, *, iterate=True, max_iter=50, tol=1e-7, screen_cap=0.5, max_spinorbitals=60)` | Third-order renormalised GF2; a closed shell is `Ca == Cb`, `n_a == n_b`. |

The kernels are reference-agnostic: any set of orbital energies and MO integrals
(Hartree-Fock, Kohn-Sham, semiempirical) can be fed in, though the runner itself
uses Hartree-Fock.

### MSINDO and other references

`vibeqc.semiempirical.methods.msindo.msindo_ovgf(atomic_numbers, coords_angstrom, *, charge=0, multiplicity=None, orbitals=None, renormalize=False, iterate=True, max_iter=200, conv_tol=1e-10, output=None)`
runs the MSINDO SCF and the same self-energy over the rigorous INDO integral tensor. It
returns `MsindoOVGFResult` for a closed shell or `MsindoUOVGFResult` for an open shell
(odd valence-electron count, or `multiplicity > 1`; s and p elements only). Verified
for water:

```python
from vibeqc.semiempirical.methods.msindo import msindo_ovgf

xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]     # Angstrom
bare = msindo_ovgf([8, 1, 1], xyz)
print(f"{bare.ip_homo_ev:.3f} eV  EA {bare.ea_lumo_ev:.3f} eV  Z {[round(float(z), 3) for z in bare.pole_strength]}")
print(f"{msindo_ovgf([8, 1, 1], xyz, renormalize=True).ip_homo_ev:.3f} eV (renormalised)")
```

```text
13.129 eV  EA -7.083 eV  Z [0.98, 0.988]
12.874 eV (renormalised)
```

The MSINDO propagator is a Python reference implementation; see [MSINDO](msindo.md)
for scope and the comparison with the reference program's factorised-integral
approximation. Its `output=` writes the Danovich, Cederbaum and von Niessen citations.
Kohn-Sham (DFT) orbitals can be passed to the `vibeqc.propagator` kernels by hand, but
there is no `run_job` route for a DFT-referenced propagator.

## Output

- **`.out`**: a header `OVGF / GF2 quasiparticle energies (diagonal second-order
  self-energy)` (open shell: `(open-shell, spin-resolved second-order self-energy)`) with
  a table of MO index, label, Koopmans energy, corrected energy and pole strength, the
  summary lines shown above, and for a small closed-shell system the `GF2(renorm)` line.
- **Citations** (`.bibtex`, `.references`): Cederbaum 1975 and von Niessen 1984 with the
  route, next to the basis and SCF entries; Danovich 1997 joins for `msindo_ovgf`.
- **Python**: `result.ovgf`, as above. A job with `optimize=True` or `hessian=True`
  relaxes or differentiates the Hartree-Fock reference, not the propagator surface, and
  tags the surface accordingly.

## Limits and gates

- **Size**: the closed-shell route builds the $n_\text{bf}^4$ MO tensor and refuses
  more than 100 basis functions with `NotImplementedError`. The open-shell route builds a
  $(2n_\text{bf})^4$ spin-orbital tensor and refuses more than 40 basis functions
  (`2 n_bf <= 80`); verified with the OH radical in cc-pVTZ (`2.n_bf=88`), which raises
  after the SCF has converged. The renormalised third-order HOMO line is printed only when
  $2n_\text{bf}\le60$ (at most 30 functions), and `renormalized_quasiparticle_energies`
  raises `NotImplementedError` beyond `max_spinorbitals` (60). Use a smaller basis or
  the `vibeqc.propagator` API on a truncated window. The size guards fire after the SCF.
- **Electron affinities** in a basis without diffuse functions are qualitative: the
  virtual orbitals are not variational and a bound anion needs diffuse functions. The
  `.out` flags them; a positive $-\varepsilon^\text{qp}$ does not establish a stable anion.
- **Approximation**: diagonal self-energy only; no off-diagonal (ADC(3)) treatment.
  Pole strengths below about 0.8 mark a breakdown of the orbital picture; do not quote
  such a line as an ionisation potential.
- **ECPs**: a heavy-element reference with an effective core potential runs (HI at
  RHF/def2-SVP, valence orbitals, first ionisation 10.09 eV, $Z$ = 0.95) and the
  self-energy is built over the valence orbitals the SCF filled. The feature matrix
  and the methods page still say that OVGF refuses an ECP reference; that statement is
  out of date for this build, and there is no scalar-relativistic spin-orbit treatment
  behind the number.
- **Solvent**: `solvent=` with `method="ovgf"` is refused (no validated post-SCF
  composition in solvent).
- **Frozen core**: none; all occupied orbitals enter the sums.
- **Unconverged SCF**: no propagator block is produced.

## See also

- [Ionisation potentials from the electron propagator](../tutorial/ovgf_ionization.md):
  water with a Delta-SCF comparison, all valence orbitals and the theory of the
  third-order terms.
- [MSINDO](msindo.md): `msindo_ovgf` and semiempirical ionisation potentials.
- [MP2 and double hybrids](mp2_and_double_hybrids.md) and [CCSD](ccsd.md): the
  correlated energies that use the same MO integral seam.
- [TDDFT, TDA and CIS excited states](tddft.md): neutral excitations from the same
  response-theory family, and [Properties](properties.md).
- [Citations](citations.md), [Methods: theory and status](../methods.md) and the
  [feature matrix](../features.md).
