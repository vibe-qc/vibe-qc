---
myst:
  html_meta:
    "description": "Hands-on vibe-qc tutorial for crystal orbital overlap and Hamilton populations (COOP and COHP) and periodic Mayer bond orders: run a dimerised hydrogen chain through run_periodic_job on the GDF route with coop_cohp=True, read the dos.coop, dos.cohp and bond_orders sections of the QVF archive with the vibeqc coop and vibeqc mayer commands, stretch the bond, use the low-level compute_coop_cohp API, and derive the definitions, sign conventions and the k-space Mayer index."
    "og:title": "vibe-qc tutorial - COOP and COHP bonding analysis"
---

# COOP and COHP: bonding analysis of a periodic system

A density of states counts states per energy; a projected DOS says which
atoms they live on; neither says whether a state holds two atoms
together or pushes them apart. The crystal orbital overlap population
(COOP, {vibeqc-cite}`hughbanks_hoffmann_coop_1983`) and the crystal
orbital Hamilton population (COHP, {vibeqc-cite}`dronskowski_cohp_1993`)
answer that question. Both resolve the DOS by *atom pair* and weight
every Bloch state by the sign of its overlap (COOP) or Hamiltonian (COHP)
matrix element between the two atoms: bonding states come out with one
sign, antibonding states with the other, and the integral up to the
Fermi level, ICOOP or ICOHP, is a single number that ranks bonds in a
crystal the way a bond order ranks them in a molecule. The periodic
Mayer bond order ({vibeqc-cite}`mayer_bond_order_1983`) is that
molecular index carried over to a k mesh.

vibe-qc computes all three after a periodic SCF and writes them into the
QVF archive as `dos.coop`, `dos.cohp` and `bond_orders` sections. On the
GDF routes the analysis is production: it projects the accepted SCF
eigenpairs, the COHP weights use the analysed spin Fock matrix (the
operator that defines the bands), a request for any other operator is
refused rather than mislabelled, and the integrated values contract the
accepted density so that fractional occupations are kept. On the BIPOLE,
AICCM and external-XC routes the request fails closed before the SCF,
with a message naming the missing converged-Hamiltonian projection. See
[Methods: theory and status](../methods.md#how-to-read-the-status-labels)
for the label vocabulary. This page runs the analysis on a dimerised
hydrogen chain, reads the archive three ways, stretches the bond, and
shows the low-level API that the runner wraps.

## Setup

The system is a chain of H$_2$ molecules along $x$: a 1.4 bohr bond in a
6 bohr cell, so that the intra-cell H-H distance is 1.4 bohr and the
inter-cell one 4.6 bohr. The GDF route needs a three-dimensional cell
(a strictly one-dimensional `dim=1` system is refused because its bare
nuclear lattice sums have no consistent Coulomb gauge), so the chain sits
in 10 bohr of vacuum in $y$ and $z$. Eight k points along the chain are
plenty for an insulator.

```python
import json
import zipfile

import numpy as np
import vibeqc as vq

a, vac = 6.0, 10.0
system = vq.PeriodicSystem(3, [[a, 0.0, 0.0], [0.0, vac, 0.0], [0.0, 0.0, vac]],
                           [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])])
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(system, basis, method="RHF", jk_method="gdf", kpoints=[8, 1, 1],
                             coop_cohp=True, output_qvf=True, output="h2_chain")
print(f"E = {result.energy:.6f} Ha per cell, converged = {result.converged}")

def read_section(kind):
    """Metadata and the binary arrays of one QVF section."""
    with zipfile.ZipFile("h2_chain.qvf") as qvf:
        manifest = json.loads(qvf.read("manifest.json"))
        sec = next(s for s in manifest["sections"] if s["kind"] == kind)
        arrays = {name: np.frombuffer(qvf.read(m["path"]), dtype=m.get("dtype", "float64"))
                  .reshape(m["shape"])
                  for name, m in sec["members"].items() if not m["path"].endswith(".json")}
    return {k: v for k, v in sec.items() if k != "members"}, arrays

meta, coop = read_section("dos.coop")
_, cohp = read_section("dos.cohp")
print("pair:", meta["pairs"][0], " sigma =", meta["sigma_ev"], "eV")

e = coop["energies"]                                   # eV, relative to the Fermi level
proj = coop["projections"][0]                          # COOP(E) of the first (only) pair
i_b, i_a = np.argmax(proj), np.argmin(proj)
print(f"ICOOP  = {coop['integrated'][0]:+.4f} electrons")
print(f"-ICOHP = {cohp['integrated'][0]:+.3f} eV")
print(f"COOP peaks: bonding {proj[i_b]:+.3f} at {e[i_b]:.2f} eV, antibonding {proj[i_a]:+.3f} at {e[i_a]:.2f} eV")
print("occupied COOP >= 0:", bool((proj[e <= 0] >= -1e-9).all()),
      " empty COOP <= 0:", bool((proj[e > 0] <= 1e-9).all()))

with zipfile.ZipFile("h2_chain.qvf") as qvf:
    manifest = json.loads(qvf.read("manifest.json"))
    sec = next(s for s in manifest["sections"] if s["kind"] == "bond_orders")
    bonds = json.loads(qvf.read(sec["members"]["bond_orders"]["path"]))
print("Mayer:", [(f"{p['symbol_i']}{p['i']}-{p['symbol_j']}{p['j']}", round(p["order"], 4))
                 for p in bonds["pairs"]])
```

On a vibe-qc 0.17.10 development build (Apple M5 Max) the run takes
about 40 seconds on an idle machine, nearly all of it the GDF lattice
sums in the vacuum box; the analysis itself is a fraction of a second.

```
E = -1.116339 Ha per cell, converged = True
pair: {'i': 0, 'j': 1, 'symbol_i': 'H', 'symbol_j': 'H', 'distance_ang': 0.7408480952641999}  sigma = 0.05 eV
ICOOP  = +0.3972 electrons
-ICOHP = +9.602 eV
COOP peaks: bonding +0.817 at -16.32 eV, antibonding -4.400 at 19.98 eV
occupied COOP >= 0: True  empty COOP <= 0: True
Mayer: [('H0-H1', 1.0002)]
```

The same archive can be read without Python:

```sh
vibeqc coop h2_chain.qvf            # pair table with ICOOP and -ICOHP
vibeqc coop --json h2_chain.qvf     # machine-readable
vibeqc mayer h2_chain.qvf           # periodic Mayer bond orders above 0.05
vibe-view open h2_chain.qvf         # the curves in the viewer; terminal mode charts dos.coop and dos.cohp
```

```
# COOP  (E_F = 0.495 eV,  σ = 0.050 eV)
#         pair  distance/Å         ICOOP
# ------------------------------------------
       H0-H1        0.74       +0.3972

# -COHP  (E_F = 0.495 eV,  σ = 0.050 eV)
#         pair  distance/Å        -ICOHP
# ------------------------------------------
       H0-H1        0.74       +9.6020
# Mayer bond orders (method=mayer, threshold=0.05)
#         pair  distance/Å       order
# --------------------------------------
       H0-H1        0.74      1.0002
```

## What just happened

After the SCF converged, the runner took the accepted eigenpairs
$\varepsilon_{n\mathbf k}$, $\mathbf C(\mathbf k)$, the overlap
$\mathbf S(\mathbf k)$, the Fock matrix $\mathbf F(\mathbf k)$ and the
density $\mathbf P(\mathbf k)$ at each of the eight k points, formed the
list of atom pairs in the home cell within 8 bohr of each other (here
exactly one, H0-H1 at 1.4 bohr), and for every band and k point computed
the pair-resolved overlap population and Hamilton population defined in
the theory section. Broadening each with a 0.05 eV Gaussian and summing
with the k weights gives the two curves stored in `dos.coop` and
`dos.cohp`, on a common energy axis in eV measured from the Fermi level.
The `.system` manifest records the provenance of the operator:
`property_hamiltonian = "accepted-scf"` and `property_kpoints = 8`, and
the `.references` file gains the four COOP/COHP papers and Mayer's.

The output is the textbook picture of H$_2$. The one occupied band, the
$\sigma_g$ combination, has a positive COOP (bonding overlap population)
and lies 16 eV below the Fermi level; the empty $\sigma_u^*$ band, 20 eV
above it, has a negative COOP, larger in magnitude because the
antibonding combination has the larger absolute overlap population
($-S/(1-S)$ against $+S/(1+S)$ per electron for a two-centre bond). The
sign checks in the script confirm that every occupied state is bonding
and every empty one antibonding. Integrating the COOP up to the Fermi
level gives the overlap population of the bond, $\mathrm{ICOOP} = 0.397$
electrons, the Mulliken overlap population of an H$_2$ molecule in a
minimal basis. The Hamilton population carries the energy weight: the
runner reports $-\mathrm{ICOHP} = 9.60$ eV, the bonding energy the pair
contributes to the band-structure energy in the sense of Dronskowski and
Blochl, positive because the Lobster sign convention is used (the
returned curve is $-\mathrm{COHP}$, so bonding is positive in both
panels; see the theory). The periodic Mayer bond order, 1.0002, says the
pair shares one electron pair, as it should for a two-electron bond that
has no partner within the cutoff.

What the runner did *not* do matters as much. It did not enumerate the
inter-cell H1-H0$'$ pair at 4.6 bohr: pairs are formed from raw home-cell
coordinates, so a bond that crosses the cell boundary, or a bond from an
atom to its own periodic image, does not appear (the
[user guide](../user_guide/coop_cohp.md) states this limitation; choose
the cell so that the bonds you care about lie inside it). It did not
interpolate the Fock matrix to a finer mesh: on the GDF route the
spectrum is built on the accepted SCF k mesh, and a different
`dos_kmesh` is refused (with a warning and no property payload) rather
than served from an interpolated operator. And it did not use any
operator other than the SCF Fock matrix for the COHP; the low-level API
below shows the refusal.

## Variation: stretch the bond

The chemistry test of any bond descriptor is what it does when the bond
weakens. Repeat the run with the intra-cell distance at 1.8 and 2.2 bohr
(the inter-cell gap shrinks from 4.6 to 3.8 bohr, so the chain is still
clearly dimerised).

```python
for bond in (1.4, 1.8, 2.2):
    sys_b = vq.PeriodicSystem(3, [[6.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                              [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [bond, 0.0, 0.0])])
    res_b = vq.run_periodic_job(sys_b, vq.BasisSet(sys_b.unit_cell_molecule(), "sto-3g"),
                                method="RHF", jk_method="gdf", kpoints=[8, 1, 1],
                                coop_cohp=True, output_qvf=True,
                                output=f"h2_chain_{bond:.1f}", verbose=0)
    with zipfile.ZipFile(f"h2_chain_{bond:.1f}.qvf") as qvf:
        manifest = json.loads(qvf.read("manifest.json"))
        def integrated(kind):
            m = next(s for s in manifest["sections"] if s["kind"] == kind)["members"]["integrated"]
            return float(np.frombuffer(qvf.read(m["path"]), dtype=m.get("dtype", "float64"))[0])
        m = next(s for s in manifest["sections"] if s["kind"] == "bond_orders")["members"]["bond_orders"]
        mayer = json.loads(qvf.read(m["path"]))["pairs"][0]["order"]
        print(f"d = {bond:.1f} bohr  E = {res_b.energy:.6f} Ha  ICOOP = {integrated('dos.coop'):.4f}"
              f"  -ICOHP = {integrated('dos.cohp'):.3f} eV  Mayer = {mayer:.4f}")
```

```
d = 1.4 bohr  E = -1.116339 Ha  ICOOP = 0.3972  -ICOHP = 9.602 eV  Mayer = 1.0002
d = 1.8 bohr  E = -1.076705 Ha  ICOOP = 0.3441  -ICOHP = 8.728 eV  Mayer = 1.0004
d = 2.2 bohr  E = -1.011717 Ha  ICOOP = 0.2886  -ICOHP = 7.828 eV  Mayer = 1.0010
```

The two integrated populations fall monotonically as the bond stretches,
by 27 percent (ICOOP) and 18 percent ($-$ICOHP) between 1.4 and 2.2 bohr,
tracking the loss of overlap and of bonding energy. The Mayer bond order
does not move: it counts the electron pair shared between the two atoms,
and a stretched H$_2$ still shares exactly one pair at the RHF level (it
would only drop once the RHF determinant is allowed to break symmetry or
correlation is included). This is the practical division of labour: the
Mayer index classifies a bond (single, double, none), the integrated
populations grade its strength, and only the energy-resolved curves say
which states are responsible.

## Variation: the low-level API on an explicit operator

`compute_coop_cohp` and `periodic_mayer_bond_orders` take real-space
lattice blocks (`LatticeMatrixSet` objects) and a k mesh directly. The
runner's GDF path does not hand its Fock blocks back, so the low-level
route is for workflows that already own an operator. The shipped example
`examples/periodic/coop_cohp_h2_chain.py` runs the same GDF SCF in a 3-D
vacuum box as this tutorial and then, in its last block, applies the
low-level API to the one-electron Hamiltonian $\mathbf T + \mathbf V$ on a
strictly one-dimensional chain for exactly that reason; that block is a
Huckel-like teaching operator, not the SCF Hamiltonian, and its absolute
energies and ICOHP are not comparable with the runner's.

```python
import numpy as np
import vibeqc as vq
from vibeqc.coop_cohp import compute_coop_cohp, periodic_mayer_bond_orders

EV = 27.211386245988
a, vac = 6.0, 30.0
chain = vq.PeriodicSystem(1, [[a, 0.0, 0.0], [0.0, vac, 0.0], [0.0, 0.0, vac]],
                          [vq.Atom(1, [0.0, vac / 2, vac / 2]), vq.Atom(1, [1.4, vac / 2, vac / 2])])
basis_1d = vq.BasisSet(chain.unit_cell_molecule(), "sto-3g")

opts = vq.LatticeSumOptions()
S = vq.compute_overlap_lattice(basis_1d, chain, opts)
T = vq.compute_kinetic_lattice(basis_1d, chain, opts)
V = vq.compute_nuclear_lattice(basis_1d, chain, opts)
kmesh = vq.monkhorst_pack(chain, [32, 1, 1])

cc = compute_coop_cohp([T, V], S, chain, basis_1d, kmesh, include_cohp=True,
                       sigma=0.01, n_grid=401, n_electrons_per_cell=2)
print("pairs:", [(p["symbol_i"], p["i"], p["symbol_j"], p["j"]) for p in cc.pairs])
print(f"E_F = {cc.fermi_energy:.4f} Ha, grid {cc.energies[0]:.3f} .. {cc.energies[-1]:.3f} Ha, "
      f"coop shape {cc.coop.shape}")
print(f"ICOOP = {cc.integrated_coop[0]:.4f} e, -ICOHP = {cc.integrated_cohp[0]:.4f} Ha "
      f"= {cc.integrated_cohp[0] * EV:.2f} eV")
bo = periodic_mayer_bond_orders([T, V], S, chain, basis_1d, kmesh, n_electrons_per_cell=2)
print(f"Mayer(H0-H1) = {bo[0, 1]:.4f}")

try:                                   # a different operator is refused, not relabelled
    compute_coop_cohp([T, V], S, chain, basis_1d, kmesh, H_terms=[T], n_electrons_per_cell=2)
except ValueError as exc:
    print("refused:", exc)
```

```
pairs: [('H', 0, 'H', 1)]
E_F = -2.6542 Ha, grid -2.716 .. -1.784 Ha, coop shape (1, 401)
ICOOP = 0.1341 e, -ICOHP = 0.3828 Ha = 10.42 eV
Mayer(H0-H1) = 1.0004
refused: COHP requires the analyzed spin Hamiltonian; H_terms describes a different operator. Use include_cohp=True for standard COHP or cohp_weights_k for explicit operator projections.
```

Three differences from the runner path are instructive. The energies
are in Hartree and absolute (the low-level result is not shifted to the
Fermi level; `cc.fermi_energy` is the highest occupied eigenvalue for
the electron count you pass). The Mayer index is again 1.00, because the
one-electron operator also fills the bonding band with one pair. But the
ICOOP is 0.134 instead of 0.397: the `H_core` bands are a different
operator with different Bloch coefficients, which is the whole point of
insisting on the SCF Fock matrix for a quantitative COHP. `include_cohp=True`
projects the operator that generated the eigenpairs; the `H_terms`
argument is accepted only when its Bloch sum equals that operator, and
the `cohp_weights_k` kernel exists for explicitly labelled projections
of anything else. For an unrestricted operator pass the beta blocks as
`F_terms_beta=` and the arrays gain a leading spin axis.

## Theory

Expand the Bloch orbitals of band $n$ at wave vector $\mathbf k$ in the
atom-centred basis, $\psi_{n\mathbf k} = \sum_\mu C_{\mu n}(\mathbf k)\,
\chi_{\mu\mathbf k}$, with $\mathbf S(\mathbf k)$ and $\mathbf H(\mathbf k)$
the Bloch-summed overlap and Hamiltonian. The Mulliken population of
that state, $\sum_{\mu\nu} C^*_{\mu n} S_{\mu\nu} C_{\nu n} = 1$, can be
split into on-site and pair terms, and the pair term between the
functions on atom $A$ and those on atom $B$ is the **overlap population**
of the state. Weighting it with a broadened delta function and summing
over states gives the COOP,

$$
\mathrm{COOP}_{AB}(E) = \sum_{\mathbf k} w_{\mathbf k} \sum_n
\Big[\operatorname{Re}\sum_{\mu\in A}\sum_{\nu\in B} C^*_{\mu n}(\mathbf k)\,S_{\mu\nu}(\mathbf k)\,C_{\nu n}(\mathbf k)\Big]
\,\delta_\sigma(E - \varepsilon_{n\mathbf k}),
$$

and replacing the overlap matrix element by the Hamiltonian matrix
element gives the COHP,

$$
\mathrm{COHP}_{AB}(E) = \sum_{\mathbf k} w_{\mathbf k} \sum_n
\Big[\operatorname{Re}\sum_{\mu\in A}\sum_{\nu\in B} C^*_{\mu n}(\mathbf k)\,H_{\mu\nu}(\mathbf k)\,C_{\nu n}(\mathbf k)\Big]
\,\delta_\sigma(E - \varepsilon_{n\mathbf k}).
$$

The restricted spin degeneracy multiplies both by two. The **sign
conventions** follow from the two-centre case: a bonding combination
$\chi_A + \chi_B$ has a positive overlap population and, because the
off-diagonal Hamiltonian element $H_{AB}$ is negative for a bonding
interaction, a *negative* Hamilton population; antibonding states
reverse both. COOP is therefore positive for bonding and COHP negative,
and because plots are easier to read when bonding points up, the Lobster
convention ({vibeqc-cite}`maintz_lobster3_2016`) is to plot and report
$-\mathrm{COHP}$; vibe-qc's `cohp` array and its `integrated_cohp`
already carry the minus sign, so both curves and both integrals are
positive for bonding. Always say which convention a number uses, because
some programs print raw COHP.

The **integrals** up to the Fermi level,
$\mathrm{ICOOP}_{AB} = \int^{E_F}\mathrm{COOP}_{AB}(E)\,dE$ and likewise
$\mathrm{ICOHP}_{AB}$, sum over the occupied states. ICOOP is the
Mulliken overlap population of the pair, a dimensionless electron count
that correlates with bond order but is basis-dependent (it shrinks toward
zero as the basis becomes more diffuse or more complete, the usual
Mulliken pathology). ICOHP has the dimension of an energy and is the
pair's share of the band-structure energy $\sum_{n\mathbf k}
f_{n\mathbf k}\varepsilon_{n\mathbf k}$, which Dronskowski and Blochl
showed by writing that energy as $\sum_{AB}\mathrm{ICOHP}_{AB}$ plus
on-site terms. It is more robust to the basis than ICOOP (Deringer,
Tchougreeff and Dronskowski project even plane-wave states onto local
orbitals to obtain it, {vibeqc-cite}`deringer_cohp_2011`), but it has a
gauge: adding a constant $c$ to the Hamiltonian, $\mathbf H \to \mathbf H
+ c\,\mathbf S$, shifts $\mathrm{COHP}_{AB}$ by $c\,\mathrm{COOP}_{AB}$.
In a periodic calculation the average electrostatic potential is not
absolutely defined, so $-\mathrm{ICOHP}$ values are comparable within
one code, one route and one basis, as the stretch series above, and not
across programs or against a molecular bond energy. On the GDF route the
runner integrates with the accepted density matrix rather than by
refilling integer occupations, so smeared or fractional occupations are
honoured, and the Gaussian width $\sigma$ (0.05 eV in the archive,
`sigma` in Hartree in the low-level call) is a presentation choice that
does not change the integrals.

The **periodic Mayer bond order** generalises Mayer's molecular index
({vibeqc-cite}`mayer_bond_order_1983`), $B_{AB} = \sum_{\mu\in A}
\sum_{\nu\in B} (\mathbf{PS})_{\mu\nu}(\mathbf{PS})_{\nu\mu}$, by
averaging over the mesh:

$$
B_{AB} = \sum_{\mathbf k} w_{\mathbf k} \sum_{\mu\in A}\sum_{\nu\in B}
\big[\mathbf P(\mathbf k)\mathbf S(\mathbf k)\big]_{\mu\nu}\,
\big[\mathbf P(\mathbf k)\mathbf S(\mathbf k)\big]_{\nu\mu},
$$

with $\mathbf P(\mathbf k) = 2\,\mathbf C_\text{occ}(\mathbf k)\mathbf
C_\text{occ}^\dagger(\mathbf k)$ for a restricted state and the
spin-summed form ($2\sum_\sigma$ over the spin densities) for an
unrestricted one; the k average is exact for the lattice-summed
products by the inverse Bloch relation. Like its molecular parent it
counts shared electron pairs, is exactly 1 for an isolated two-electron
bond, and does not measure bond energy, as the stretch series shows.

Where the analysis helps: assigning bands to bonds and antibonds,
locating the antibonding states that a Peierls distortion or a dopant
would empty, ranking chemically similar contacts within one structure,
and following a bond through a series of geometries or compositions.
Where it fails: absolute comparisons across bases, codes or the
potential gauge; systems whose bonds cross the cell boundary or whose
atoms bond to their own images, which the home-cell pair list does not
enumerate; and any diffuse basis, in which Mulliken-type projections
become meaningless. Converge the k mesh and the SCF thresholds before the
broadening, and keep the QVF with its provenance so that the operator,
mesh and convention travel with the numbers.

## Resources

- [COOP and COHP user guide](../user_guide/coop_cohp.md): result shapes,
  pair selection, the plotters `coop_figure` and `cohp_figure`, and the
  convergence checklist.
- [Bond analysis](../user_guide/bond_analysis.md) for the molecular
  Mayer and Wiberg indices; [Projected DOS](pdos.md) and
  [Band structure](band_structure.md) for the spectra the populations
  resolve.
- [Peierls distortion](peierls.md): the dimerised chain of this page as
  a physical phenomenon.
- [Smearing](../user_guide/smearing.md) for metals, where the Fermi
  level and the occupations must be converged together.
- `examples/periodic/coop_cohp_h2_chain.py`: this tutorial's SCF in a
  vacuum box, plus the low-level example on the one-dimensional chain.
- [QVF file format](qvf_file_format.md) and
  [vibe-view terminal mode](../user_guide/vibe_view_terminal_mode.md),
  which charts `dos.coop` and `dos.cohp`.
- [Feature matrix](../features.md) and
  [Methods: theory and status](../methods.md#electronic-structure-of-solids-bands-dos-coopcohp-bond-orders-qtaim).

Cost: the analysis diagonalises nothing new on the GDF route (it reuses
the accepted eigenpairs) and scales as $N_{\mathbf k}\,N_\text{pair}\,
N_\text{bf}^2$ for the pair weights plus $N_\text{pair}\,N_E$ for the
curves; it reserves its spectral storage under a 128 MiB cap before
allocating. The 40 seconds of the example are the GDF SCF in a vacuum
box, not the analysis.

## References

- **COOP.** T. Hughbanks and R. Hoffmann, "Chains of trans-edge-sharing
  molybdenum octahedra: metal-metal bonding in extended systems,"
  *J. Am. Chem. Soc.* **105**, 3528 (1983),
  {vibeqc-cite}`hughbanks_hoffmann_coop_1983`. The crystal orbital
  overlap population.
- **COHP.** R. Dronskowski and P. E. Blochl, "Crystal orbital Hamilton
  populations (COHP): energy-resolved visualization of chemical bonding
  in solids based on density-functional calculations," *J. Phys. Chem.*
  **97**, 8617 (1993), {vibeqc-cite}`dronskowski_cohp_1993`. The
  Hamilton population and the band-energy partition.
- **Projected COHP.** V. L. Deringer, A. L. Tchougreeff and
  R. Dronskowski, "Crystal orbital Hamilton population (COHP) analysis
  as projected from plane-wave basis sets," *J. Phys. Chem. A* **115**,
  5461 (2011), {vibeqc-cite}`deringer_cohp_2011`.
- **Lobster and the sign convention.** S. Maintz, V. L. Deringer,
  A. L. Tchougreeff and R. Dronskowski, "LOBSTER: A tool to extract
  chemical bonding from plane-wave based DFT," *J. Comput. Chem.* **37**,
  1030 (2016), {vibeqc-cite}`maintz_lobster3_2016`.
- **Mayer bond order.** I. Mayer, "Charge, bond order and valence in the
  ab initio SCF theory," *Chem. Phys. Lett.* **97**, 270 (1983),
  {vibeqc-cite}`mayer_bond_order_1983`.

## Next

- [Projected DOS](pdos.md) resolves the same bands by atom and angular
  momentum instead of by pair.
- [Peierls distortion](peierls.md) follows the dimerisation of a
  hydrogen chain and is where the antibonding states above the Fermi
  level start to matter.
- [Periodic HF](periodic_hf.md) and [Periodic methods compared](periodic_methods_compared.md)
  explain the GDF route this page ran on and the ones that refuse the
  analysis.
