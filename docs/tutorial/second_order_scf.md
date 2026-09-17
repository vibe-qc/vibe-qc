---
myst:
  html_meta:
    "description": "Second-order SCF in vibe-qc for difficult convergence: full-Hessian Newton, SOSCF, and trust-region augmented Hessian (TRAH). How to turn each on with its threshold option, a worked H2O comparison against DIIS, and how to choose among them."
    "og:title": "vibe-qc tutorial - second-order SCF (Newton, SOSCF, TRAH)"
---

# Second-order SCF: Newton, SOSCF, and TRAH

DIIS, the default accelerator, is fast and robust for most closed-shell
molecules, but it can stall on **difficult SCF problems**: transition-metal
complexes with near-degenerate d orbitals, stretched bonds near
dissociation, diffuse anions. When the density oscillates and the energy
will not settle, the cure is a **second-order** method that uses the
curvature of the energy (the orbital Hessian) to take a Newton step toward
the nearest stationary point.

vibe-qc ships three, all as fields on the standard options object and all
**off by default**:

| Option | Method | Character |
|---|---|---|
| `newton_threshold` | Full-Hessian Newton | exact curvature; quadratic near convergence |
| `soscf_threshold` | SOSCF (approximate second order) | cheaper Hessian-vector steps |
| `trah_threshold` | Trust-region augmented Hessian | robust far from convergence |

## Turning it on

Each is a *threshold*: it stays off (`0.0`) until the DIIS error drops below
the value you set, then the SCF switches from DIIS to the second-order step.
A large threshold (`1.0`) switches almost immediately; a small one lets DIIS
get close first, then polishes:

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(8, [0, 0, 0]),
                   vq.Atom(1, [0,  1.43, -0.98]),
                   vq.Atom(1, [0, -1.43, -0.98])])
basis = vq.BasisSet(mol, "6-31g*")

opts = vq.RHFOptions()
opts.trah_threshold = 1.0            # switch to TRAH once the DIIS error < 1.0
result = vq.run_rhf(mol, basis, opts)
print(result.energy, result.n_iter)
```

## What it buys you

On an easy closed-shell case the second-order step already trims iterations;
on a hard one it is the difference between converging and not. H2O / 6-31G*,
every method reaching the same energy:

| Method | Iterations | E (Ha) |
|---|---|---|
| DIIS (default) | 17 | -76.00667789 |
| Newton | 8 | -76.00667789 |
| TRAH | 8 | -76.00667789 |
| SOSCF | 28 | -76.00667789 |

Newton and TRAH roughly halve the iteration count here; SOSCF, the cheap
approximate variant, takes more steps on this easy system but earns its keep
on hard ones where its low per-step cost matters. The point is that all four
land on the *same* energy: a second-order method changes *how* the SCF gets
there, not *where*.

## Choosing among them

- **TRAH** is the safe choice for a hard case: the trust region keeps the
  step sane even far from convergence, so it rarely diverges.
- **Newton** uses the exact orbital Hessian and converges quadratically once
  close, at the cost of building that Hessian.
- **SOSCF** trades exactness for cheaper steps; good when the Hessian is
  expensive and you only need to get unstuck.

They are mutually exclusive: if you set more than one threshold, Newton
takes precedence, then TRAH, then SOSCF. All four of RHF, UHF, RKS, and UKS
support them, through the matching `UHFOptions` / `RKSOptions` /
`UKSOptions` fields.

## See also

- [Molecular SCF with OpenTrustRegion](opentrustregion.md) and
  [READ restarts and orbital stability](opentrustregion_stability.md), for the
  separately selected optional orbital optimizer and its worked examples.
- [Stiff molecular SCF: EDIIS+DIIS](ediis_diis_hybrid.md), the first thing to
  reach for before second order.
- [Initial guesses](initial_guess_walkthrough.md), since a better starting
  density often fixes convergence without any of this.
- [scf_convergence](../user_guide/scf_convergence.md) and the
  [keyword index](../user_guide/keyword_index.md) for every convergence knob
  and its default.
