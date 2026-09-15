"""NH3 umbrella inversion via Nudged Elastic Band on a MACE potential.

Requires the ``[mace]`` extra (PyTorch + e3nn; Python <= 3.13):
    pip install 'vibe-qc[mace]'
Run:
    .venv/bin/python 03_neb_mace.py

The same reaction as tutorial 19 (NH3 flips through a planar D3h
transition state), but the per-image energies + forces come from a
pre-trained MACE machine-learned interatomic potential instead of an
SCF. MACE returns analytic forces, so there is no finite-difference
gradient and no Gaussian basis — ``run_neb(method="mace")`` just needs
the two endpoints and the climbing-image flag.

Energy-scale note: MACE energies are on a model-specific reference
scale (see vibeqc.mlip.mace) — meaningful for the *relative* barrier,
not as an absolute total. The default model here is MACE-MPA-0 (MIT,
materials, ungated) and covers H/N. For an organic-family study, the
documented organic model is MACE-OFF23, whose ASL license requires an
explicit academic, non-commercial acknowledgment.
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.mlip import MLIPOptions

HERE = Path(__file__).parent
QVF_OUT = HERE / "output-nh3-neb-mace"          # .qvf appended by the writer

ANG_TO_BOHR = 1.8897259886
KCAL_PER_HARTREE = 627.509474


def build_nh3(flip: bool = False) -> vq.Molecule:
    """Pyramidal NH3 (vibe-qc Molecule, bohr); H's below or above N."""
    theta = np.deg2rad(106.67)
    r_nh = 1.012 * ANG_TO_BOHR
    sin_a = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    z_sign = +1 if flip else -1
    atoms = [vq.Atom(7, [0.0, 0.0, 0.0])]
    for k in range(3):
        phi = k * 2.0 * np.pi / 3.0
        atoms.append(vq.Atom(1, [
            r_nh * sin_a * np.cos(phi),
            r_nh * sin_a * np.sin(phi),
            z_sign * r_nh * cos_a,
        ]))
    return vq.Molecule(atoms, 0, 1)


def main() -> None:
    result = vq.run_neb(
        build_nh3(flip=False), build_nh3(flip=True),
        method="mace",                                  # no basis / functional
        mlip_options=MLIPOptions(model="medium-mpa-0"),  # MIT MACE-MPA-0
        n_images=5,
        interpolation="linear",
        climbing_image=True,
        climbing_image_start_fraction=0.1,
        conv_tol_force=5e-3,
        max_iter=200,
    )

    e = np.asarray(result.energies)
    barrier = (e[result.transition_state_index] - e[0]) * KCAL_PER_HARTREE
    print(f"converged={result.converged}  n_iter={result.n_iter}")
    print(f"TS at image {result.transition_state_index}; "
          f"barrier = {barrier:.2f} kcal/mol (MACE reference scale)")
    for i, de in enumerate((e - e[0]) * KCAL_PER_HARTREE):
        print(f"  image {i}: dE = {de:+.3f} kcal/mol")

    qvf = result.write_qvf(str(QVF_OUT))
    print(f"Wrote {Path(qvf).name}  (cites the MACE method + model papers)")


if __name__ == "__main__":
    main()
