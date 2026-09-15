"""Natural-orbital occupations — tutorial 20 figure.

Two-row, four-panel figure showing how natural orbitals diagnose the
single-determinant character of an SCF result.

  Top row — occupation-number bar charts:
    (a) H₂O / 6-31G* RHF — exactly integer {2, 0}, idempotency Δ = 0.
    (b) O₂ triplet / 6-31G* UHF, ``kind="uhf-total"`` — D_α + D_β,
        again integer {2, 0} for a single-determinant UHF state.
    (c) O₂ triplet / 6-31G* UHF, ``kind="uhf-spin"`` — D_α − D_β,
        the two largest eigenvalues are ≈ +1 (the singly-occupied π*
        spin-NOs that carry the open-shell character).

  Bottom row:
    (d) 2D contour slice of one of the singly-occupied O₂ spin-NOs in
        the perpendicular plane, showing the textbook π* nodal pattern
        with lobes above/below the molecular axis.

Writes ``docs/_static/plots/natural-orbital-occupations.png``.

Run:
    .venv/bin/python examples/plots/natural-orbital-occupations.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "natural-orbital-occupations.png"


# --- Geometries (bohr) ------------------------------------------------------

H2O = [
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
]

# O2 triplet ground state: bond length 1.208 Å along z, centered on origin.
R_O2 = 1.208 / 0.529177210903
O2 = [
    vq.Atom(8, [0.0, 0.0, +0.5 * R_O2]),
    vq.Atom(8, [0.0, 0.0, -0.5 * R_O2]),
]


def grid_xz(half_size: float = 4.0, n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Z, points) where points is (n*n, 3) in Cartesian bohr,
    sampled in the y = 0 plane (perpendicular to the molecular axis is
    *not* this plane for O₂ along z; this slice contains the molecular
    axis so π* lobes appear above/below it)."""
    u = np.linspace(-half_size, half_size, n)
    v = np.linspace(-half_size, half_size, n)
    U, V = np.meshgrid(u, v)
    pts = np.column_stack([U.ravel(), np.zeros(U.size), V.ravel()])
    return U, V, pts


def plot_occupations(ax, occupations: np.ndarray, *, kind: str, title: str,
                     n_show: int = 12, ymin: float | None = None,
                     ymax: float | None = None) -> None:
    """Bar chart of the leading-magnitude occupation numbers."""
    n = np.asarray(occupations, dtype=float)
    order = np.argsort(-np.abs(n))[:n_show]
    n_show_arr = n[order]
    idx = np.arange(n_show_arr.size)

    colours = ["#1f77b4" if v >= 0 else "#d62728" for v in n_show_arr]
    ax.bar(idx, n_show_arr, color=colours, edgecolor="black", linewidth=0.6)

    # Reference lines for the relevant max/min for this NO kind.
    if kind in ("rhf", "uhf-total"):
        ax.axhline(2.0, color="#7f7f7f", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=0.8)
    elif kind == "uhf-spin":
        ax.axhline(+1.0, color="#7f7f7f", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=0.8)
        ax.axhline(-1.0, color="#7f7f7f", linestyle="--", linewidth=0.8)
    ax.set_xlabel("NO index (sorted by |n|)")
    ax.set_ylabel("occupation $n_i$")
    ax.set_title(title, fontsize=10)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx], fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def plot_no_slice(ax, U, V, phi: np.ndarray, title: str,
                  atoms_in_plane: list[tuple[str, float, float]]) -> None:
    """2D filled contour of a (signed) NO with atoms overlaid."""
    amp = float(np.percentile(np.abs(phi), 99.5))
    levels = np.linspace(-amp, amp, 21)
    ax.contourf(U, V, phi, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(U, V, phi, levels=[0.0], colors="black",
               linewidths=0.4, linestyles="--")
    for (label, x, y) in atoms_in_plane:
        ax.plot(x, y, "o", ms=11, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (bohr)")
    ax.set_ylabel("z (bohr)")
    ax.set_title(title, fontsize=10)


def main() -> None:
    t_total = time.perf_counter()

    # --- (a) H₂O closed-shell NOs --------------------------------------------
    t0 = time.perf_counter()
    h2o = vq.Molecule(H2O)
    h2o_basis = vq.BasisSet(h2o, "6-31g*")
    h2o_res = vq.run_rhf(h2o, h2o_basis)
    h2o_no = vq.natural_orbitals(h2o_res, h2o_basis)
    h2o_idem = vq.idempotency_deviation(h2o_no)
    print(f"  H₂O / 6-31G* RHF  E = {h2o_res.energy:.6f} Ha  "
          f"({time.perf_counter()-t0:.2f} s)")
    print(f"    NO occupations (top 8): "
          f"{np.array2string(h2o_no.occupations[:8], precision=6)}")
    print(f"    idempotency Δ = {h2o_idem:.3e}")

    # --- (b)+(c) O₂ triplet UHF ----------------------------------------------
    t0 = time.perf_counter()
    # multiplicity=3 (triplet: two unpaired α electrons in π*)
    o2 = vq.Molecule(O2, multiplicity=3)
    o2_basis = vq.BasisSet(o2, "6-31g*")
    o2_res = vq.run_uhf(o2, o2_basis)
    o2_no_total = vq.natural_orbitals(o2_res, o2_basis, kind="uhf-total")
    o2_no_spin = vq.natural_orbitals(o2_res, o2_basis, kind="uhf-spin")
    print(f"  O₂ triplet / 6-31G* UHF  E = {o2_res.energy:.6f} Ha  "
          f"({time.perf_counter()-t0:.2f} s)")
    print(f"    total NO occupations (top 10): "
          f"{np.array2string(o2_no_total.occupations[:10], precision=4)}")
    print(f"    spin  NO occupations (top 6):  "
          f"{np.array2string(o2_no_spin.occupations[:6], precision=4)}")
    print(f"    idempotency Δ (uhf-total) = "
          f"{vq.idempotency_deviation(o2_no_total):.3e}")
    n_unp_count = int(np.sum(np.abs(o2_no_spin.occupations) > 0.95))
    print(f"    Yamaguchi spin-deviation  = "
          f"{vq.idempotency_deviation(o2_no_spin):.3f}  "
          f"(spin contamination across the AO basis, NOT N_unpaired)")
    print(f"    spin NOs with |n| > 0.95: {n_unp_count}  "
          f"→ this is the textbook 'two unpaired π* electrons'")

    # --- (d) Render the leading O₂ spin NO in the (x, z) plane ---------------
    U, V, pts = grid_xz(half_size=3.5, n=200)
    chi = vq.evaluate_ao(o2_basis, pts)            # (n_pts, n_bf)
    leading_spin_no = o2_no_spin.coefficients[:, 0]
    phi = (chi @ leading_spin_no).reshape(U.shape)

    # --- Compose figure ------------------------------------------------------
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.0, 7.4), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.10],
                          hspace=0.50, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_occupations(ax_a, h2o_no.occupations, kind="rhf",
                     title=f"(a) H$_2$O RHF\n5 occupied; $\\Delta = {h2o_idem:.1e}$",
                     ymin=-0.1, ymax=2.3)

    n_unpaired_count = int(np.sum(np.abs(o2_no_spin.occupations) > 0.95))
    ax_b = fig.add_subplot(gs[0, 1])
    plot_occupations(ax_b, o2_no_total.occupations, kind="uhf-total",
                     title=r"(b) O$_2$ triplet UHF, $D_\alpha + D_\beta$"
                           "\ntwo NOs at $n = 1$ (the unpaired electrons)",
                     ymin=-0.1, ymax=2.3)

    ax_c = fig.add_subplot(gs[0, 2])
    plot_occupations(ax_c, o2_no_spin.occupations, kind="uhf-spin",
                     title=r"(c) O$_2$ triplet UHF, $D_\alpha - D_\beta$"
                           "\n%d spin-NOs with $|n| \\approx 1$" % n_unpaired_count,
                     ymin=-1.2, ymax=1.2)

    ax_d = fig.add_subplot(gs[1, :])
    atoms_in_plane = [("O", 0.0, +0.5 * R_O2), ("O", 0.0, -0.5 * R_O2)]
    plot_no_slice(ax_d, U, V, phi,
                  title=r"(d) Leading O₂ spin natural orbital "
                        r"(π*$_x$, $n \approx +1$) — slice at $y = 0$",
                  atoms_in_plane=atoms_in_plane)
    ax_d.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Natural orbitals diagnose single-determinant vs open-shell character",
        fontsize=12, y=0.995,
    )
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")
    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}  "
          f"(total {time.perf_counter()-t_total:.1f} s)")


if __name__ == "__main__":
    main()
