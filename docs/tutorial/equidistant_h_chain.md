---
myst:
  html_meta:
    "description": "The equidistant hydrogen chain as a cyclic cluster: the 1D-metal sibling of the dimerized H2-pair chain, folded onto a finite H6 ring. Why the equal-spacing chain is the hard case for reciprocal-space sampling and the clean case for the cyclic cluster model, the crystal-orbital degeneracy pattern, and the published CCM-HF/STO-3G benchmark from Peintinger's thesis (now reproduced by vibe-qc's Γ-CCM)."
    "og:title": "vibe-qc tutorial - the equidistant H chain on a cyclic ring"
---

# The equidistant H chain: a 1D metal on a cyclic ring

This is the **equal-spacing** companion to the
[dimerized H-chain tutorial](h8_chain_ccm_ancestor.md). That one strings
hydrogen into H₂ *pairs* (a short bond, then a long gap) and gets a gapped,
band-insulating Peierls chain. Here every H-H distance is the same, and that
one change turns the chain into a **one-dimensional metal**, the
Peierls-unstable parent the pairs chain dimerizes away from. The metal is the
harder system, and it is exactly the case the
[cyclic cluster model](cyclic_cluster_model.md) handles most gracefully.

```{admonition} Two H-chains, kept separate on purpose
:class: note
The [pairs chain](h8_chain_ccm_ancestor.md) (dimerized, gapped) and this
equidistant chain (metallic) are different physics and stay separate
tutorials. The pairs chain demonstrates Born-von-Kármán equivalence across
supercell and k-mesh representations on vibe-qc's shipped periodic SCF. This
one demonstrates why the **finite cyclic ring** is the elegant route when the
infinite chain is metallic, using the published CCM-HF benchmark from the
project author's thesis.
```

## Equal spacing makes a metal

Put one hydrogen atom per cell, spaced a uniform $a$ apart, one electron each.
The single $1s$ band is then **half filled**, and a half-filled band with no
gap at the Fermi level is a metal. Peierls' theorem says such a 1D chain is
unstable: it lowers its energy by dimerizing into H₂ pairs, which opens a gap
at the zone boundary and turns the metal into the insulator of the
[pairs tutorial](h8_chain_ccm_ancestor.md). The equidistant chain is the
undistorted, metallic reference that instability starts from.

Metals are the awkward case for reciprocal-space sampling. The Brillouin-zone
integrand is discontinuous at the Fermi surface, so an integer-occupation
calculation on a coarse k-mesh will not settle, and the cure is
[Fermi-Dirac smearing](fermi_dirac_smearing.md) on a fine mesh. vibe-qc is
explicit about this: ask its Γ-only periodic route for smearing and it
refuses rather than silently running an integer-Aufbau metal, pointing you at
the multi-k driver instead.

## The cyclic cluster makes a ring

The cyclic cluster model takes the other road. Wrap $N$ equidistant atoms into
a **ring** by imposing the cyclic Born-von-Kármán boundary conditions (the
[derivation here](kpoints_brillouin_bloch.md)), and the infinite metallic
chain becomes a finite system: a particle on a ring. A ring of $N$ atoms holds
exactly $N$ crystal orbitals, the Bloch states $e^{ikx}$ at the discrete
wavevectors

$$
k = \frac{n}{N}\cdot\frac{2\pi}{a}, \qquad n = 0,\ \pm1,\ \pm2,\ \dots
$$

folded onto the single Γ point of the cluster. For the $N=6$ ring the
wavevectors are $k = 0\ (\Gamma),\ \pm\tfrac16,\ \pm\tfrac13,\ \tfrac12$ (in
units of $2\pi/a$). Qualitatively the energies follow the tight-binding band
$E(k) \approx \alpha + 2\beta\cos(ka)$, so they order from the fully bonding
$\Gamma$ state ($\cos 0 = +1$) up to the fully antibonding zone-boundary state
($\cos\pi = -1$), and the $+k$ and $-k$ partners are **degenerate** because
$\cos(ka) = \cos(-ka)$. That degeneracy pattern, a single bonding level, then
degenerate pairs, then a single antibonding level, is the same one Hückel
theory gives for the $\pi$ system of a ring such as benzene.

The decisive point: at $N = 6$ the six electrons fill the lowest three
orbitals, $\Gamma$ plus the $k=\pm\tfrac16$ pair, and stop at a **real gap**
below the empty $k=\pm\tfrac13$ pair. The finite ring is closed-shell and
gapped even though the infinite chain it samples is metallic. The cyclic
cluster has converted the awkward metallic integral into a clean molecular
diagonalization, no smearing, no k-mesh convergence, one Γ-point solve.

## The CCM-HF benchmark

The orbital energies of the equidistant H₆ ring, computed at Hartree-Fock /
STO-3G with the ab-initio cyclic cluster model, are tabulated in the project
author's dissertation (M. F. Peintinger, *Elektronenstruktur von
Molekülkristallen*, Univ. Bonn, 2013, §8.6.4, Table 8.6; published as
Peintinger and Bredow, *J. Comput. Chem.* **35**, 839, 2014):

| Crystal orbital | Occupation | Energy (Ha) | $k$ ($2\pi/a$) |
|---|---|---|---|
| 1 | 2.00 | $-0.74295030$ | $0$ ($\Gamma$) |
| 2 | 2.00 | $-0.45266305$ | $+\tfrac16$ |
| 3 | 2.00 | $-0.45266305$ | $-\tfrac16$ |
| 4 | 0.00 | $+0.44076948$ | $+\tfrac13$ |
| 5 | 0.00 | $+0.44076948$ | $-\tfrac13$ |
| 6 | 0.00 | $+1.30010143$ | $+\tfrac12$ |

The degeneracies are exact ($k$ and $-k$ agree to every printed digit), the
filled set ends at the $k=\pm\tfrac16$ pair, and the HOMO-LUMO gap is a clean
$0.893$ Ha. The crystal-orbital coefficients (Table 8.7) make the band
endpoints concrete: the lowest orbital is **uniform**, every atom $+0.272185$,
the fully in-phase $\Gamma$ combination, while the highest is strictly
**alternating**, $\pm0.856484$ around the ring, the fully out-of-phase
zone-boundary combination. The middle four spread phase smoothly between those
limits.

```{admonition} What this benchmark is, and is not
:class: important
These numbers are the **ab-initio CCM-HF** result computed in the thesis
(with the AICCM program of that work), not a vibe-qc run. vibe-qc now
reproduces them with [Γ-CCM / `aiccm2026dev-a`](../aiccm2026dev_a.md),
the union-and-weight/Wigner--Seitz integral-weighting CCM that ships as an
experimental feature.
Run the [Γ-CCM tutorial](aiccm2026dev_a.md) to reproduce Table 8.6. The
[ab-initio CCM theory section](cyclic_cluster_model.md#how-the-ab-initio-ccm-works)
works through the Wigner-Seitz weighting that produces it.
```

## Why the ring is the right tool here

The equidistant chain draws the contrast between the two periodic pictures
sharply. The supercell-and-k-mesh route has to integrate a discontinuous
metallic band and needs smearing to do it; the cyclic-cluster route folds the
same physics onto a finite ring that simply has a gap. For a perfect crystal
the symmetry of the k-mesh still makes reciprocal-space sampling cheaper, but
the moment the symmetry is broken, by a defect, a surface, or an adsorbate,
the real-space ring keeps its clean molecular character while the k-mesh does
not. That is the argument the cyclic cluster model is built on, and the
equidistant H₆ ring is its smallest honest illustration.

## See also

- [The dimerized H-chain (CCM ancestor)](h8_chain_ccm_ancestor.md): the
  gapped, Peierls sibling, and the Born-von-Kármán supercell/k-mesh
  equivalence on vibe-qc's shipped periodic SCF.
- [The cyclic cluster model](cyclic_cluster_model.md): what a cyclic cluster
  is, the ab-initio derivation, and the two experimental lines that ship today
  (Γ-CCM and χ-CCM).
- [k-points, the Brillouin zone, and Bloch phase](kpoints_brillouin_bloch.md):
  where the cyclic boundary conditions and the discrete wavevectors come from.
- [Peierls dimerisation of a 1D H-chain](peierls.md): how the metal on this
  page distorts into the insulator of the pairs tutorial.
- [Fermi-Dirac smearing](fermi_dirac_smearing.md): the reciprocal-space route
  to metals, the alternative the ring sidesteps.
