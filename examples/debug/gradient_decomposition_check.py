"""Gradient-piece decomposition check — periodic RHF analytic vs FD,
piece-by-piece.

Companion to ``scf_translation_invariance_check.py`` for the gradient
side. Where the SCF check isolates the J / V_ne gauge bug, this
script isolates which piece of the analytic Γ-only RHF gradient
driver disagrees with the finite-difference oracle. The five pieces:

  (1) nuclear-rep                 nuclear_repulsion_gradient_per_cell
  (2) overlap-Lagrangian (-W∂S)   overlap_lattice_gradient_contribution
  (3) kinetic Pulay (D∂T)         kinetic_lattice_gradient_contribution
  (4) V-attraction (D∂V_ne)       nuclear_lattice_gradient_contribution
  (5) 2-e Pulay (J + α_HF·K)      eri_lattice_gradient_contribution

Known bad piece (per ``periodic_gradient.py`` G1a docstring): piece
(5) with α_HF > 0 — the K-style derivative-buffer routing in the C++
``eri_lattice_gradient_contribution`` shows ~5e-3 Ha/bohr disagreement
with FD on small periodic cells. Pieces (1)-(4) and the J part of (5)
match FD to N3-precision (~1e-13). Pure-DFT periodic gradients (α_HF
= 0) work; HF / hybrid DFT periodic gradients are approximate until
the K bug closes.

**Empirical note (v0.7.x).** On the default 1D H chain config
(2 H per cell at a = 2 Å, STO-3G, cutoff 10 bohr) this script does
*not* currently reproduce the docstring's ~5e-3 K-piece drift —
analytic matches FD to ~3e-10. Either the bug was closed since the
docstring was written, or it requires a different chain
configuration (1 H per cell? different a? different basis?). When
running this script as a debugging tool, verify the verdict against
the upstream G1a docstring first; if they disagree, the docstring is
likely stale.

Default test systems
--------------------

  - **H₂ in a 20-Å cubic box, Γ-only** — molecular limit. cutoff <
    box means cell list reduces to a single cell; *every* piece
    must match FD to ≤ 1e-7 Ha/bohr. If anything disagrees here,
    the bug is in the molecular-limit routing of that primitive
    itself, not the periodic generalisation.

  - **1D H chain at a = 2 Å, STO-3G, Γ-only** — known K-piece-bug
    repro per the G1a docstring (~5e-3 Ha/bohr drift on small
    periodic cells). The SCF is always HF here, and α_HF in the
    analytic driver is pinned to 1.0 to match it. The K piece's
    value is *isolated* without an apples-to-oranges FD comparison
    by computing the analytic gradient twice — once at α_HF=1 and
    once at α_HF=0 — the difference is the K-piece contribution
    alone (the rest of the gradient is identical between the two
    calls because the SCF density is fixed).

  - **LiH conventional rocksalt, 8 atoms** — opt-in via
    ``--include-lih``. Matches the system used in
    ``examples/debug/lih_energy_decomposition.py``. Wall ~10 min
    for the FD oracle (48 SCFs).

Caveat about α_HF
-----------------

``compute_gradient_periodic_rhf_fd`` always uses HF SCF (it calls
``run_rhf_periodic`` internally — there's no pure-DFT FD oracle in
the same module). So this script pins ``α_HF=1.0`` everywhere the
analytic gradient is compared to FD. To probe the K piece in
isolation, the script *also* calls the analytic driver at α_HF=0
on the same SCF density and reports the K-only contribution as a
derived diagnostic — that doesn't need an FD comparison at all
(the SCF density is fixed; the diff between α=0 and α=1 analytic
results IS the K piece by construction).

The pure-DFT analog would build on
``vq.compute_gradient_periodic_rks_gamma`` (G1b driver) plus an
RKS FD oracle — sibling task; this script is RHF-only.

Wall budget: ~1 min on a laptop for the default two systems.

Run
---

    .venv/bin/python examples/debug/gradient_decomposition_check.py
    .venv/bin/python examples/debug/gradient_decomposition_check.py \\
        --include-lih

Output
------

One ``<label>.npz`` per system under
``examples/debug/output/gradient-decomposition/`` containing the
analytic driver result, FD oracle result, and each piece (with the
2-e piece further split into J-only, K-only, and J+K) — ready for
offline plotting / further triangulation.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import (
    compute_overlap_lattice,
    eri_lattice_gradient_contribution,
    kinetic_lattice_gradient_contribution,
    nuclear_lattice_gradient_contribution,
    nuclear_repulsion_gradient_per_cell,
    overlap_lattice_gradient_contribution,
)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "gradient-decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANGSTROM_TO_BOHR = 1.8897261339213


# ── system builders ─────────────────────────────────────────────────────────


def h2_in_box(box_ang: float = 20.0,
              bond_bohr: float = 1.4) -> vq.PeriodicSystem:
    """H₂ centred in an L-bohr cubic box. cutoff < L → molecular
    limit (every primitive should reduce to its molecular counterpart
    bit-for-bit)."""
    L = box_ang * ANGSTROM_TO_BOHR
    centre = np.array([L / 2.0, L / 2.0, L / 2.0])
    half = np.array([0.0, 0.0, bond_bohr / 2.0])
    return vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [
            vq.Atom(1, (centre - half).tolist()),
            vq.Atom(1, (centre + half).tolist()),
        ],
    )


def h_chain_1d(a_ang: float = 2.0,
               vacuum_ang: float = 16.0) -> vq.PeriodicSystem:
    """1D H chain along x with vacuum padding in y, z. Per the G1a
    docstring this is the canonical K-piece-bug repro: small a brings
    cross-cell ERIs into play; HF-style 2-e gradient piece disagrees
    with FD by ~5e-3 Ha/bohr on STO-3G."""
    a = a_ang * ANGSTROM_TO_BOHR
    L_perp = vacuum_ang * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(
        3,
        np.diag([a, L_perp, L_perp]),
        [
            vq.Atom(1, [0.0, L_perp / 2.0, L_perp / 2.0]),
            vq.Atom(1, [a / 2.0, L_perp / 2.0, L_perp / 2.0]),
        ],
    )


def lih_rocksalt(a_ang: float = 4.084) -> vq.PeriodicSystem:
    """LiH conventional rocksalt, 4 Li + 4 H. Same setup as
    ``examples/debug/lih_energy_decomposition.py``. 48 SCFs for the
    FD oracle — opt-in via ``--include-lih``."""
    a = a_ang * ANGSTROM_TO_BOHR
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                       (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                       (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    return vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)


# ── per-piece breakdown ─────────────────────────────────────────────────────


@dataclass
class GradPieces:
    nuc_rep: np.ndarray   # (n_atoms, 3)
    overlap: np.ndarray
    kinetic: np.ndarray
    v_ne: np.ndarray
    eri: np.ndarray       # 2-e Pulay (J + α_HF·K)

    def total(self) -> np.ndarray:
        return (self.nuc_rep + self.overlap + self.kinetic
                + self.v_ne + self.eri)


def _gamma_density_set(template, D: np.ndarray):
    """Replicate D across every cell block of a LatticeMatrixSet
    (Γ-only convention: D(g) = D for every g — mirrors
    ``periodic_gradient._gamma_density_lattice_set`` so per-piece sum
    matches the integrated driver bit-for-bit)."""
    n_cells = len(template.cells)
    D_arr = np.asarray(D, dtype=np.float64)
    for c in range(n_cells):
        template.set_block(c, D_arr)
    return template


def decompose_analytic_gradient(
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    result,
    lattice_opts: vq.LatticeSumOptions,
    *,
    alpha_hf: float,
) -> GradPieces:
    """Call each of the 5 lattice gradient primitives in isolation
    and return per-piece (n_atoms, 3) arrays. Mirrors the assembly
    inside ``compute_gradient_periodic_rhf_gamma`` so the per-piece
    sum equals the driver's output to machine precision."""
    D_gamma = np.asarray(result.density, dtype=np.float64)
    C = np.asarray(result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(result.mo_energies, dtype=np.float64)
    n_elec = system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "decompose_analytic_gradient: open-shell not supported "
            "(closed-shell RHF only). Multi-k UKS gradient lives at "
            "G1d, separate driver."
        )
    nocc = n_elec // 2
    W_gamma = 2.0 * (C[:, :nocc] * eps[:nocc][None, :]) @ C[:, :nocc].T

    D_set = compute_overlap_lattice(basis, system, lattice_opts)
    W_set = compute_overlap_lattice(basis, system, lattice_opts)
    _gamma_density_set(D_set, D_gamma)
    _gamma_density_set(W_set, W_gamma)

    return GradPieces(
        nuc_rep=np.asarray(
            nuclear_repulsion_gradient_per_cell(system, lattice_opts)),
        overlap=np.asarray(
            overlap_lattice_gradient_contribution(
                basis, system, W_set, lattice_opts)),
        kinetic=np.asarray(
            kinetic_lattice_gradient_contribution(
                basis, system, D_set, lattice_opts)),
        v_ne=np.asarray(
            nuclear_lattice_gradient_contribution(
                basis, system, D_set, lattice_opts)),
        eri=np.asarray(
            eri_lattice_gradient_contribution(
                basis, system, D_set, lattice_opts, float(alpha_hf))),
    )


# ── per-system runner ───────────────────────────────────────────────────────


def _make_opts(*, conv_tol: float = 1e-12,
               cutoff_bohr: float = 25.0,
               nuclear_cutoff_bohr: float = 25.0
               ) -> vq.PeriodicSCFOptions:
    """Default DIRECT_TRUNCATED options matching tests/test_periodic_
    gradient_g1a.py — Γ-only, cutoff sized for the test box."""
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = conv_tol
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = nuclear_cutoff_bohr
    return opts


def run_one(label: str,
            system: vq.PeriodicSystem,
            basis_name: str,
            *,
            cutoff_bohr: float = 25.0,
            nuclear_cutoff_bohr: float = 25.0,
            fd_step: float = 1e-3) -> dict:
    """Per-system decomposition. Pinned to α_HF=1 (HF) because the FD
    oracle is HF-only; the K piece is isolated by re-calling the eri
    primitive at α_HF=0 on the same density (no extra SCF needed)."""
    print()
    print("─" * 72)
    print(f"  {label}   (basis={basis_name}, RHF / α_HF=1)")
    print("─" * 72)

    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = _make_opts(cutoff_bohr=cutoff_bohr,
                      nuclear_cutoff_bohr=nuclear_cutoff_bohr)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    # ─── reference SCF ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    scf = vq.run_rhf_periodic(system, basis, kmesh, opts)
    t_scf = time.perf_counter() - t0
    if not scf.converged:
        print(f"  ✗ SCF failed in {scf.n_iter} iters — abort case")
        return {"label": label, "ok": False}
    print(f"  SCF: E = {float(scf.energy):+.8f} Ha   "
          f"({scf.n_iter} iters, {t_scf:.2f} s)")

    # ─── analytic per-piece breakdown @ α_HF=1 ──────────────────────────────
    t0 = time.perf_counter()
    pieces = decompose_analytic_gradient(
        system, basis, scf, opts.lattice_opts, alpha_hf=1.0,
    )
    grad_piece_sum = pieces.total()
    t_pieces = time.perf_counter() - t0
    print(f"  per-piece breakdown (α_HF=1): {t_pieces:.3f} s")

    # ─── J-only re-call to isolate the K contribution ───────────────────────
    # piece_eri at α_HF=0 is the J-only 2-e gradient. The diff between
    # piece_eri at α_HF=1 and α_HF=0 is the K-only contribution. No
    # extra SCF — same density, just a second call to the eri primitive.
    pieces_J_only = decompose_analytic_gradient(
        system, basis, scf, opts.lattice_opts, alpha_hf=0.0,
    )
    K_only = pieces.eri - pieces_J_only.eri
    K_max = float(np.linalg.norm(K_only, axis=1).max())

    # ─── integrated analytic driver (cross-check) ───────────────────────────
    grad_driver = np.asarray(vq.compute_gradient_periodic_rhf_gamma(
        system, basis, scf, lattice_opts=opts.lattice_opts, alpha_hf=1.0,
    ))
    consistency = float(np.abs(grad_piece_sum - grad_driver).max())
    if consistency > 1e-12:
        print(f"  ⚠ piece-sum vs driver mismatch {consistency:.2e} — "
              "decomposition is not faithful, fix this first")
    else:
        print(f"  ✓ piece-sum ≡ driver to {consistency:.2e}")

    # ─── FD oracle ──────────────────────────────────────────────────────────
    n_atoms = len(system.unit_cell)
    print(f"  FD oracle: 6 × {n_atoms} = {6 * n_atoms} SCFs at step "
          f"{fd_step:.0e} bohr")
    t0 = time.perf_counter()
    grad_fd = vq.compute_gradient_periodic_rhf_fd(
        system, basis_name, kmesh, opts, step_bohr=fd_step,
    )
    t_fd = time.perf_counter() - t0
    print(f"  FD oracle: {t_fd:.1f} s")

    # ─── per-piece atom-norm report ─────────────────────────────────────────
    print()
    print("  per-piece atom-norms (max |F_atom| over atoms, Ha/bohr):")
    rows = [
        ("(1) nuclear-rep         ", pieces.nuc_rep),
        ("(2) overlap-Lagrangian  ", pieces.overlap),
        ("(3) kinetic Pulay       ", pieces.kinetic),
        ("(4) V-attraction        ", pieces.v_ne),
        ("(5a) 2-e J only         ", pieces_J_only.eri),
        ("(5b) 2-e K only (α=1−α=0)", K_only),
        ("(5)  2-e total (J + K)  ", pieces.eri),
    ]
    for name, arr in rows:
        max_norm = float(np.linalg.norm(arr, axis=1).max())
        print(f"    {name}  max ‖F_atom‖ = {max_norm:.6e}")

    # ─── analytic vs FD verdict ─────────────────────────────────────────────
    diff = grad_piece_sum - grad_fd
    max_abs = float(np.abs(diff).max())
    print()
    print(f"  ‖analytic − FD‖_∞ = {max_abs:.3e} Ha/bohr")

    # piece-by-piece localisation hint: if analytic disagrees with FD,
    # compare the mismatch magnitude to per-piece magnitudes to point
    # the finger.
    if max_abs > 1e-6:
        oneE_mag = float(np.linalg.norm(
            pieces.kinetic + pieces.v_ne, axis=1).max())
        if K_max > 0 and max_abs > 0.1 * K_max:
            hint = (f"(K-piece magnitude {K_max:.2e}, mismatch "
                    f"{max_abs:.2e} → piece 5b K likely)")
        elif oneE_mag > 0 and max_abs > 0.1 * oneE_mag:
            hint = (f"(1-e piece magnitude {oneE_mag:.2e}, mismatch "
                    f"{max_abs:.2e} → pieces 3-4)")
        else:
            hint = "(small absolute force — check FD step or geometry)"
    else:
        hint = ""

    if max_abs < 1e-6:
        verdict = "✓ analytic ≡ FD to ≤1e-6 Ha/bohr — every piece clean"
    elif max_abs < 1e-3:
        verdict = (f"~ partial agreement ({max_abs:.2e}) — small drift "
                   f"{hint}")
    else:
        verdict = (f"✗ ANALYTIC ≠ FD ({max_abs:.2e}) — gradient bug "
                   f"{hint}")
    print(f"  verdict: {verdict}")

    # ─── persist artefact ───────────────────────────────────────────────────
    out_path = OUT_DIR / f"{label}.npz"
    np.savez(
        out_path,
        atoms_z=np.array([a.Z for a in system.unit_cell], dtype=np.int32),
        atoms_xyz=np.array([list(a.xyz) for a in system.unit_cell],
                           dtype=np.float64),
        grad_driver=grad_driver,
        grad_piece_sum=grad_piece_sum,
        grad_fd=grad_fd,
        piece_nuc_rep=pieces.nuc_rep,
        piece_overlap=pieces.overlap,
        piece_kinetic=pieces.kinetic,
        piece_v_ne=pieces.v_ne,
        piece_eri_J_only=pieces_J_only.eri,
        piece_eri_K_only=K_only,
        piece_eri_total=pieces.eri,
        scf_energy_ha=np.float64(scf.energy),
        scf_n_iter=np.int32(scf.n_iter),
    )
    print(f"  artefact: {out_path.name}")

    return {
        "label": label,
        "ok": True,
        "max_abs_diff": max_abs,
        "consistency": consistency,
        "K_max": K_max,
        "out_path": str(out_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-lih", action="store_true",
                   help="Also run LiH conventional rocksalt "
                        "(8 atoms → 48 SCFs for FD, ~10 min wall).")
    p.add_argument("--fd-step", type=float, default=1e-3,
                   help="Half-step (bohr) for the FD central difference. "
                        "Default 1e-3 — large enough to clear SCF "
                        "convergence noise (~1e-8 Ha → ~1e-5 Ha/bohr at "
                        "this step), small enough to stay linear.")
    p.add_argument("--skip-h2", action="store_true",
                   help="Skip H₂ molecular-limit case.")
    p.add_argument("--skip-chain", action="store_true",
                   help="Skip 1D H chain (the K-bug repro).")
    args = p.parse_args()

    print("=" * 72)
    print(" Gradient-piece decomposition check")
    print("=" * 72)
    print(" periodic RHF Γ-only — analytic driver vs FD oracle, "
          "per-piece localisation.")

    runs = []

    if not args.skip_h2:
        # H₂ in 20-Å box: molecular limit, every piece must match FD.
        runs.append(run_one(
            "h2_box20_angstrom", h2_in_box(box_ang=20.0),
            "sto-3g",
            cutoff_bohr=25.0, nuclear_cutoff_bohr=25.0,
            fd_step=args.fd_step,
        ))

    if not args.skip_chain:
        # 1D H chain at 2 Å. Per G1a docstring this is the canonical
        # K-piece bug repro on small periodic cells.
        # cutoff_bohr ≈ 10 bohr ≈ 5 cells along the chain.
        runs.append(run_one(
            "h_chain_1d_a2", h_chain_1d(a_ang=2.0), "sto-3g",
            cutoff_bohr=10.0, nuclear_cutoff_bohr=15.0,
            fd_step=args.fd_step,
        ))

    if args.include_lih:
        runs.append(run_one(
            "lih_rocksalt_a4.084", lih_rocksalt(a_ang=4.084), "sto-3g",
            cutoff_bohr=15.0, nuclear_cutoff_bohr=22.5,
            fd_step=args.fd_step,
        ))

    # ─── summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"  {'system':28s}  {'‖a−FD‖∞':>12s}  {'K mag':>10s}  verdict")
    for r in runs:
        if not r["ok"]:
            print(f"  {r['label']:28s}  {'SCF failed':>12s}  "
                  f"{'-':>10s}  ✗")
            continue
        d = r["max_abs_diff"]
        K = r["K_max"]
        flag = "✓" if d < 1e-6 else ("~" if d < 1e-3 else "✗")
        meaning = ("clean" if d < 1e-6 else
                   "small drift" if d < 1e-3 else
                   "BUG — see per-piece breakdown")
        print(f"  {r['label']:28s}  {d:12.3e}  {K:10.3e}  "
              f"{flag} {meaning}")

    print()
    print(f"  artefacts under: {OUT_DIR}")


if __name__ == "__main__":
    main()
