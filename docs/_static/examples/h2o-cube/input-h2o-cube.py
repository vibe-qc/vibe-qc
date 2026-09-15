"""H2O — write electron density and frontier MOs as Gaussian cube files.

Run:
    .venv/bin/python input-h2o-cube.py

Produces:
    output-h2o-cube.out       — SCF trace and output manifest siblings
    output-h2o-cube.qvf       — vibe-view archive with density + MOs
    output-h2o-density.cube   — total ρ(r), suitable for VMD/Avogadro
    output-h2o-mo-homo.cube   — HOMO orbital
    output-h2o-mo-lumo.cube   — LUMO orbital
    output-h2o-mos.cube       — multi-volume cube (HOMO-1 … LUMO+1)

Visualize in VMD: ``vmd -e``-style or just *Open* the file. Avogadro,
PyMOL, ChimeraX read the same files.
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent


def main() -> None:
    mol = vq.Molecule.from_xyz(str(HERE.parent / "h2o.xyz"))
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_job(
        mol,
        basis="6-31g*",
        method="rhf",
        output=HERE / "output-h2o-cube",
        output_qvf=True,
    )

    grid = vq.make_uniform_grid(mol, spacing=0.15, padding=5.0)
    print(f"grid: {grid.shape} = {grid.n_points:,} voxels")

    # Density
    p = vq.write_cube_density(
        HERE / "output-h2o-density.cube",
        res.density, basis, mol, grid=grid,
    )
    print(f"wrote {p.name}")

    # Frontier MOs
    homo = mol.n_electrons() // 2 - 1
    lumo = homo + 1
    vq.write_cube_mo(
        HERE / "output-h2o-mo-homo.cube",
        res.mo_coeffs, homo, basis, mol, grid=grid,
        title=f"H2O HOMO (MO {homo})",
    )
    vq.write_cube_mo(
        HERE / "output-h2o-mo-lumo.cube",
        res.mo_coeffs, lumo, basis, mol, grid=grid,
        title=f"H2O LUMO (MO {lumo})",
    )
    vq.write_cube_mos(
        HERE / "output-h2o-mos.cube",
        res.mo_coeffs, [homo - 1, homo, lumo, lumo + 1],
        basis, mol, grid=grid,
        title="H2O frontier MOs",
    )
    print("wrote MO cubes (HOMO, LUMO, multi-MO stack)")


if __name__ == "__main__":
    main()
