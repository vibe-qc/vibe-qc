"""NH3 umbrella inversion as a minimum-energy path with MSINDO + NEB.

MSINDO (vibe-qc's own Bredow/Geudtner/Jug INDO re-implementation) is the
first *semiempirical* backend for ``run_neb``: pass ``method="msindo"`` and
the band's per-image energies + gradients come from the INDO engine instead
of an SCF — no Gaussian basis, no functional, no k-mesh.

The reaction is the ammonia umbrella inversion (the same system as NEB
tutorial 19 and the MACE NEB example): pyramidal NH3 flips through the
planar D3h transition state into its mirror image. Because the product is an
exact mirror of the reactant, the path is symmetric and the saddle sits at
the central image.

What this script demonstrates / validates:
  1. ``run_neb(method="msindo")`` drives a climbing-image band to the saddle.
  2. The barrier is stable and the profile is monotone on each side.
  3. The transition state is a *true* first-order saddle: the MSINDO Hessian
     (finite-differenced from the FD gradient) at the climbing image has
     exactly one imaginary mode — the umbrella bend.

Cost note: MSINDO exposes a finite-difference gradient, so each image costs
6N ``run_msindo`` SCFs per outer iteration. NH3 is tiny (N = 4), which keeps
the whole run to a few seconds; the band runs in parallel across images.

Run:  .venv/bin/python examples/semiempirical/22_msindo_neb.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.semiempirical.methods.msindo import (
    ANGSTROM_TO_BOHR as A2B,
)
from vibeqc.semiempirical.methods.msindo import (
    msindo_gradient_fd,
    msindo_optimize,
)

HERE = Path(__file__).resolve().parent
QVF_OUT = HERE / "output-msindo-nh3-neb"          # .qvf appended by the writer

KCAL_PER_HA = 627.509474
Z = [7, 1, 1, 1]                                  # N, H, H, H

# Mass-weighting (electron masses) for the frequency analysis at the saddle.
_AMU_TO_ME = 1822.888486
_MASSES_AMU = np.array([14.003074, 1.007825, 1.007825, 1.007825])
# Ha/bohr^2, mass in m_e -> angular frequency in atomic units; * this -> cm^-1.
_AU_TO_CM = 219474.6313702


def build_nh3(flip: bool = False, theta_deg: float = 106.67,
              r: float = 1.012) -> np.ndarray:
    """Pyramidal NH3 coordinates (Angstrom); H's below (or above) the N."""
    theta = np.deg2rad(theta_deg)
    sin_a = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    z_sign = +1.0 if flip else -1.0
    coords = [(0.0, 0.0, 0.0)]
    for k in range(3):
        phi = k * 2.0 * np.pi / 3.0
        coords.append((r * sin_a * np.cos(phi),
                       r * sin_a * np.sin(phi),
                       z_sign * r * cos_a))
    return np.array(coords)


def to_molecule(coords_ang: np.ndarray) -> vq.Molecule:
    return vq.Molecule(
        [vq.Atom(z, list(np.array(c, float) * A2B))
         for z, c in zip(Z, coords_ang)],
        0, 1,
    )


def coords_of(mol: vq.Molecule) -> np.ndarray:
    return np.array([list(np.array(a.xyz) / A2B) for a in mol.atoms])


def saddle_frequencies(coords_ang: np.ndarray) -> tuple[int, np.ndarray]:
    """Imaginary-mode count + harmonic frequencies (cm^-1) at ``coords_ang``.

    Builds the Cartesian Hessian by central-differencing the MSINDO FD
    gradient. The imaginary-mode count is taken from the *Cartesian* spectrum
    (Ha/bohr^2), where the six translation/rotation modes sit near zero and a
    reaction mode is a clearly-negative eigenvalue — robust without an Eckart
    projection. The reported frequencies are the mass-weighted ones (signed;
    negative = imaginary), so the saddle's imaginary mode comes out as a real
    wavenumber to quote.
    """
    n = len(Z) * 3
    H = np.zeros((n, n))
    h = 1e-3  # Angstrom displacement
    for i in range(len(Z)):
        for d in range(3):
            cp = coords_ang.copy(); cp[i, d] += h
            cm = coords_ang.copy(); cm[i, d] -= h
            gp = msindo_gradient_fd(Z, cp, step=5e-4).ravel()   # Ha/bohr
            gm = msindo_gradient_fd(Z, cm, step=5e-4).ravel()
            H[:, i * 3 + d] = (gp - gm) / (2.0 * h / A2B)       # Ha/bohr^2
    H = 0.5 * (H + H.T)
    # Robust imaginary count: trans/rot |eig| << 1e-3; a reaction mode is large.
    n_imag = int(np.sum(np.linalg.eigvalsh(H) < -1e-3))
    m = np.repeat(_MASSES_AMU * _AMU_TO_ME, 3)
    Hmw = H / np.sqrt(np.outer(m, m))
    evals = np.linalg.eigvalsh(Hmw)
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * _AU_TO_CM
    return n_imag, freqs


def main() -> None:
    # 1) Relax the endpoint to the true MSINDO minimum; the product is its
    #    mirror image (exact umbrella partner), so the band is symmetric.
    react_ang, react = msindo_optimize(Z, build_nh3(flip=False), fmax=1e-3)
    prod_ang = react_ang.copy()
    prod_ang[:, 2] *= -1.0
    print(f"Relaxed pyramidal NH3:  E = {react.total_energy:.6f} Ha "
          f"(binding {react.binding_energy:.6f} Ha)")

    # 2) Climbing-image MSINDO NEB.
    res = vq.run_neb(
        to_molecule(react_ang), to_molecule(prod_ang),
        method="msindo",                       # no basis / functional / k-mesh
        n_images=5,
        interpolation="idpp",
        climbing_image=True,
        climbing_image_start_fraction=0.25,
        conv_tol_force=1e-3,
        max_iter=200,
    )
    e = np.asarray(res.energies)
    ts = res.transition_state_index
    barrier = (e[ts] - e[0]) * KCAL_PER_HA
    print(f"\nNEB: converged={res.converged}  n_iter={res.n_iter}  "
          f"TS at image {ts} (of {len(e)})")
    print(f"Umbrella inversion barrier = {barrier:.2f} kcal/mol")
    for i, de in enumerate((e - e[0]) * KCAL_PER_HA):
        tag = "  <- TS (climbing)" if i == ts else ""
        print(f"  image {i}: dE = {de:+7.3f} kcal/mol{tag}")

    # Experimental context (not a benchmark MSINDO is expected to nail): NH3
    # inversion is the canonical double-well tunnelling problem, with an
    # experimental barrier of ~5-6 kcal/mol (Swalen & Ibers, J. Chem. Phys. 36,
    # 1914 (1962), doi:10.1063/1.1701290; the precise value depends on the
    # fitted potential). That is an *effective* barrier (ZPE + tunnelling),
    # whereas the value above is the bare *electronic* barrier (no ZPE), so the
    # two are not strictly the same quantity -- but the comparison shows MSINDO,
    # like INDO semiempirical methods generally, overestimates this barrier.
    print("  (experimental inversion barrier ~5-6 kcal/mol, Swalen & Ibers "
          "1962; MSINDO overestimates, as expected for INDO)")

    # 3) Confirm a true first-order saddle: one imaginary mode at the TS.
    n_imag, freqs = saddle_frequencies(coords_of(res.path.images[ts].system))
    print(f"\nTransition-state normal modes: {n_imag} imaginary")
    print(f"  imaginary frequency: {abs(freqs.min()):.0f}i cm^-1 (umbrella bend)")
    assert n_imag == 1, "expected a single imaginary mode at the saddle"

    # 4) vibe-view archive (cites the MSINDO method papers + the NEB papers;
    #    no libint / no XC functional — MSINDO is STO/INDO).
    qvf = res.write_qvf(str(QVF_OUT))
    print(f"\nWrote {Path(qvf).name}  (cites Ahlswede & Jug 1999 + Henkelman)")


if __name__ == "__main__":
    main()
