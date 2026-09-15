"""Periodic H₂⁺ stretching NEB with DFT+U on the H 1s channel.

Run:

    .venv/bin/python input-periodic-h2plus-neb-plus-u.py

Produces:

    output-periodic-h2plus-neb-plus-u.out      — timing + energies summary
    output-periodic-h2plus-neb-no-u.qvf        — baseline NEB animation
    output-periodic-h2plus-neb-with-u.qvf      — +U NEB animation

End-to-end demo of ``run_neb(PeriodicSystem, ..., method="UHF",
dft_plus_u=[HubbardSite(...)])`` — the periodic UHF + DFT+U surface
that landed on top of Increment 4d-bipole UHF / UKS. The driver
routes every per-image BIPOLE SCF through ``run_pbc_bipole_uhf(...,
dft_plus_u=[...])``, so the per-spin Dudarev V_U Fock contribution
lands in the reference SCF + the 6N FD-displaced SCFs that build
the per-image gradient.

System: H₂⁺ doublet in a 12-bohr cubic box, Γ-only Bloch mesh,
STO-3G basis. The reactant stretches the H–H bond from 1.4 to 1.8
bohr; the NEB walks the bond-stretch coordinate and lands a low
transition state in between. With ``U=4 eV`` on H[0]'s 1s channel
the +U Fock perturbation raises the TS energy by ~9.5 mHa
(~0.27 eV) — the canonical Dudarev signature on a
fractionally-occupied AO projection (the singly-occupied 1s is
spread across both H 1s AOs by the bonding MO, so the projection
onto H[0]'s 1s is ~0.5 and ``tr(n − n²) ≈ 0.25``).

Why H₂⁺ + STO-3G + 12-bohr box?

* H₂⁺ has one electron, so the open-shell UHF SCF converges
  reliably (no broken-symmetry knot to untangle) and a U on the
  single occupied 1s channel produces a clean, easy-to-interpret
  E_TS shift.
* STO-3G keeps each per-image BIPOLE SCF cheap (~3 SCF iters
  from a SAD guess; ~1–2 with warm-start), so the full
  FD-gradient × NEB sweep finishes in a couple of minutes on a
  laptop.
* The 12-bohr cubic box keeps periodic AO image overlap small
  enough that the +U projector evaluates on essentially the
  molecular density. Tighter boxes (~4 bohr) produce a vastly
  exaggerated +U shift via heavy image overlap and should be
  avoided for +U on diffuse channels until the periodic +U
  projector is hardened against small-cell pathologies.

This is a *recipe*, not a publication benchmark — see
``input-bipole-nio-uks-plus-u.py`` for the canonical
transition-metal-oxide DFT+U recipe (NiO at U on Ni 3d) and
``docs/user_guide/neb.md`` § DFT+U for the full coverage matrix.

Algorithm: improved-tangent NEB (Henkelman+Jónsson 2000) with
default quick-min outer loop. Periodic gradients are central
finite differences in Cartesian space (Increment 4); analytic
periodic gradient is the periodic-NEB chat's roadmap.

Citations (auto-bundled inside each ``.qvf`` archive at
``citations/references.bib``):

* Henkelman & Jónsson 2000 — improved-tangent NEB
* Dudarev et al. 1998 — rotationally-invariant DFT+U (only
  in the +U run's archive — the ``dft_plus_u`` flag on
  ``NEBResult`` fires the ``routes.methods.dft_plus_u`` route)
* Cococcioni & de Gironcoli 2005 — linear-response U (same)

References for the BIPOLE periodic SCF + STO-3G basis +
libxc / libint linkages land via the standard
``write_qvf`` citation surface for every run.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vibeqc import (
    Atom,
    HubbardSite,
    LatticeSumOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    run_neb,
)

HERE = Path(__file__).parent
TEXT_OUT = HERE / "output-periodic-h2plus-neb-plus-u.out"
QVF_NOU = HERE / "output-periodic-h2plus-neb-no-u"
QVF_WITHU = HERE / "output-periodic-h2plus-neb-with-u"


BOX_BOHR = 12.0


def _h2_plus_in_box(z2: float) -> PeriodicSystem:
    """H₂⁺ along z in a cubic cell. charge=1, multiplicity=2."""
    L = np.diag([BOX_BOHR, BOX_BOHR, BOX_BOHR])
    return PeriodicSystem(
        3, L,
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])],
        charge=1,
        multiplicity=2,
    )


def _opts() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    lat = LatticeSumOptions()
    lat.cutoff_bohr = BOX_BOHR / 2.0
    opts.lattice_opts = lat
    opts.max_iter = 100
    return opts


def run_once(*, dft_plus_u, label: str) -> tuple:
    """Run the H₂⁺ stretch NEB once; return (wall_time, NEBResult)."""
    reactant = _h2_plus_in_box(1.4)
    product = _h2_plus_in_box(1.8)

    t0 = time.time()
    result = run_neb(
        reactant, product,
        basis="sto-3g",
        n_images=2,
        method="UHF",
        uhf_options=_opts(),
        interpolation="linear",
        max_iter=5,
        conv_tol_force=5e-2,
        n_jobs=1,
        kpoints=(1, 1, 1),
        fd_step_bohr=5e-3,
        warm_start=True,
        dft_plus_u=dft_plus_u,
    )
    return time.time() - t0, result


def main() -> None:
    lines: list[str] = []

    def out(line: str = "") -> None:
        print(line)
        lines.append(line)

    out("Periodic H₂⁺ stretch NEB — DFT+U sweep")
    out("=" * 70)

    out("\nBaseline (no +U):")
    t_base, r_base = run_once(dft_plus_u=None, label="baseline")
    out(
        f"  converged={r_base.converged}  n_iter={r_base.n_iter}  "
        f"max_force={r_base.max_force:.4e} Ha/bohr  wall={t_base:.2f}s"
    )
    out(f"  energies: {[float(e) for e in r_base.energies]}")
    e_ts_base = float(r_base.energies[r_base.transition_state_index])
    out(f"  TS image: {r_base.transition_state_index}  E_TS = {e_ts_base:.6f} Ha")

    out("\nWith U=4 eV on H[0]'s 1s channel:")
    sites = [HubbardSite(atom_index=0, l=0, U_ev=4.0)]
    t_u, r_u = run_once(dft_plus_u=sites, label="with-u")
    out(
        f"  converged={r_u.converged}  n_iter={r_u.n_iter}  "
        f"max_force={r_u.max_force:.4e} Ha/bohr  wall={t_u:.2f}s"
    )
    out(f"  energies: {[float(e) for e in r_u.energies]}")
    e_ts_u = float(r_u.energies[r_u.transition_state_index])
    out(f"  TS image: {r_u.transition_state_index}  E_TS = {e_ts_u:.6f} Ha")

    delta_ha = e_ts_u - e_ts_base
    out("\nDFT+U effect on the transition state:")
    out(f"  ΔE_TS = E_TS(U=4) − E_TS(U=0) = {delta_ha * 1000:+.3f} mHa")
    out(f"        = {delta_ha * 27.211386:+.4f} eV")
    out(
        "  (Dudarev (U_eff/2) tr(n−n²) on a half-projected open-shell "
        "1s gives ~9 mHa; positive sign = V_U raises the singly-"
        "occupied orbital energy.)"
    )

    # Emit QVFs (lattice + per-image atom positions) for vibe-view animation.
    # Each archive bundles its own bibliography at
    # citations/references.bib; the +U run additionally surfaces
    # Dudarev 1998 + Cococcioni-Gironcoli 2005 via the
    # routes.methods.dft_plus_u route.
    for qvf_stem in (QVF_NOU, QVF_WITHU):
        qvf_stem.with_suffix(".qvf").unlink(missing_ok=True)
    r_base.write_qvf(QVF_NOU)
    r_u.write_qvf(QVF_WITHU)
    out(f"\nWrote {QVF_NOU.with_suffix('.qvf').name}")
    out(f"Wrote {QVF_WITHU.with_suffix('.qvf').name}")
    out(
        "Open either .qvf in vibe-view to animate the band; the "
        "auto-assembled bibliography lives at "
        "citations/references.bib inside the archive (the +U run "
        "additionally cites Dudarev 1998 + Cococcioni-Gironcoli 2005)."
    )

    TEXT_OUT.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {TEXT_OUT.name}")


if __name__ == "__main__":
    main()
