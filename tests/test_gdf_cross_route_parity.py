"""One bulk GDF Hamiltonian across every Gamma route.

Bulk (dim = 3) GDF has one Coulomb and exchange convention: the fitted
Hartree term plus the exxdiv='ewald' exchange, of which Gamma is the
one-point mesh. These tests pin that every entry point evaluates it:

* the exact-Gamma drivers ``run_pbc_gdf_rhf`` / ``run_pbc_gdf_uhf`` and the
  k-mesh drivers at ``kmesh=(1, 1, 1)`` agree, RHF and UHF(M=1) alike
  (#54, #87);
* ``run_periodic_job`` at the default Gamma mesh gives the same energy with
  or without a level shift, Fock mixing or smearing, and on a charged cell
  (#400; before, each of these switched to a legacy Ewald-J driver whose
  exchange has no divergence correction: +75.65 mHa on the H2 cube below);
* the exact-Gamma drivers refuse options they do not implement instead of
  dropping them, and ``run_pbc_gdf_rhf`` defaults to the SR/LR fit like
  every other GDF entry point;
* the compcell/MDF builders sum the auxiliary images of their Gaussian
  three-centre tensor and land on PySCF on a cell where those images
  matter, and compcell no longer depends on its compensating exponent
  (#172).

The H2 fixture is a 6-bohr cube, where AO and auxiliary images overlap;
a vacuum box would be blind to the image errors pinned here.

External references are PySCF 2.14.0 run out of process on the identical
cells (``cell.unit='B'``, ``precision=1e-10``, ``auxbasis='def2-svp-jkfit'``,
the vibe-qc STO-3G stand-in ``def2-svp-jk``, ``exxdiv='ewald'``,
``conv_tol=1e-10``); PySCF is never imported here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.pbc_gdf import run_pbc_gdf_rhf, run_pbc_gdf_uhf
from vibeqc.periodic_k_gdf import (
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
)

#: PySCF 2.14.0 KRHF(GDF) at Gamma, settings in the module docstring.
PYSCF_H2_CUBE6 = -1.238556290205344
PYSCF_HEH_CUBE7 = -3.0738676583702174
PYSCF_MGO_STO3G = -271.0496908186412
PYSCF_CDIA_STO3G = -74.00247971119255

ROUTE_TOL = 1e-9


def _h2_cube6():
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])], 0, 1
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _heh_cube7():
    system = vq.PeriodicSystem(
        3, np.eye(3) * 7.0, [vq.Atom(2, [0, 0, 0]), vq.Atom(1, [1.46, 0, 0])], 1, 1
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _fcc(a_bohr, atoms, basis):
    h = a_bohr / 2
    lattice = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(z, r) for z, r in atoms], 0, 1)
    return system, vq.BasisSet(system.unit_cell_molecule(), basis)


def _job(system, basis, tmp_path, tag, **kwargs):
    return vq.run_periodic_job(
        system,
        basis,
        jk_method="gdf",
        output=str(tmp_path / tag),
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        **kwargs,
    )


# --- the Gamma routes are one Hamiltonian ------------------------------------


def test_gamma_routes_agree_on_overlapping_images():
    system, basis = _h2_cube6()
    energies = {
        "run_pbc_gdf_rhf": run_pbc_gdf_rhf(system, basis, progress=False).energy,
        "run_pbc_gdf_uhf": run_pbc_gdf_uhf(system, basis, progress=False).energy,
        "krhf(1,1,1)": run_krhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).energy,
        "kuhf(1,1,1)": run_kuhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).energy,
    }
    reference = energies["run_pbc_gdf_rhf"]
    for route, energy in energies.items():
        assert energy == pytest.approx(reference, abs=ROUTE_TOL), route
    # measured 4.9e-12 from PySCF
    assert reference == pytest.approx(PYSCF_H2_CUBE6, abs=1e-9)


@pytest.mark.parametrize(
    "knob",
    [{}, {"fock_mixing": 0.3}, {"level_shift": 0.2}, {"smearing_temperature": 1e-3}],
    ids=["plain", "fock_mixing", "level_shift", "smearing"],
)
def test_default_gamma_rhf_energy_does_not_depend_on_convergence_aids(tmp_path, knob):
    """#400: every aid used to switch to the legacy driver (-1.1629043 Ha)."""
    system, basis = _h2_cube6()
    result = _job(system, basis, tmp_path, "h2", method="RHF", **knob)
    assert result.converged
    assert result.energy == pytest.approx(PYSCF_H2_CUBE6, abs=1e-9)


@pytest.mark.parametrize(
    "knob",
    [{}, {"level_shift": 0.2}, {"smearing_temperature": 1e-3}],
    ids=["plain", "level_shift", "smearing"],
)
def test_default_gamma_hybrid_energy_does_not_depend_on_convergence_aids(tmp_path, knob):
    """The hybrid twin: the legacy switch moved PBE0 by a_x x 75.65 mHa."""
    system, basis = _h2_cube6()
    reference = run_krks_periodic_gdf(
        system, basis, (1, 1, 1), functional="pbe0", progress=False
    ).energy
    result = _job(system, basis, tmp_path, "h2pbe0", method="RKS",
                  functional="pbe0", **knob)
    assert result.converged
    assert result.energy == pytest.approx(reference, abs=ROUTE_TOL)


def test_default_gamma_charged_cell_matches_pyscf(tmp_path):
    """A charged cell now runs the k-mesh engine at (1,1,1) (measured 3.4e-12
    from PySCF); the legacy fallback it used to take sat 18.4 mHa higher."""
    system, basis = _heh_cube7()
    default = _job(system, basis, tmp_path, "heh", method="RHF")
    explicit = _job(system, basis, tmp_path, "heh111", method="RHF", kpoints=(1, 1, 1))
    assert default.energy == pytest.approx(explicit.energy, abs=ROUTE_TOL)
    assert default.energy == pytest.approx(PYSCF_HEH_CUBE7, abs=1e-9)


# --- the exact-Gamma drivers refuse what they do not implement ---------------


@pytest.mark.parametrize("field,value", [
    ("smearing_temperature", 1e-3), ("fock_mixing", 0.3), ("level_shift", 0.2),
])
def test_run_pbc_gdf_rhf_refuses_unimplemented_controls(field, value):
    system, basis = _h2_cube6()
    options = vq.PeriodicRHFOptions()
    setattr(options, field, value)
    with pytest.raises(NotImplementedError, match="run_krhf_periodic_gdf"):
        run_pbc_gdf_rhf(system, basis, options, progress=False)


def test_run_pbc_gdf_uhf_refuses_a_level_shift():
    system, basis = _h2_cube6()
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.2
    with pytest.raises(NotImplementedError, match="run_kuhf_periodic_gdf"):
        run_pbc_gdf_uhf(system, basis, options, progress=False)


def test_run_pbc_gdf_rhf_defaults_to_the_sr_lr_fit():
    """Its former compcell default returned +2.9e7 Ha on MgO/STO-3G."""
    system, basis = _h2_cube6()
    result = run_pbc_gdf_rhf(system, basis, progress=False)
    assert result.backend.startswith("pbc-gdf-rsgdf")


# --- compcell / MDF: the auxiliary images (#172) ------------------------------


def _fused(system, basis):
    from vibeqc.aux_basis import (
        default_aux_for,
        fuse_transform_matrix,
        make_aux_basis_set,
        make_compensating_basis,
        make_fused_basis,
        make_modrho_aux_basis,
    )

    molecule = system.unit_cell_molecule()
    aux = make_aux_basis_set(molecule, aux_name=default_aux_for(basis.name))
    modrho = make_modrho_aux_basis(aux, molecule)
    chg = make_compensating_basis(modrho, molecule, eta=1.0)
    return make_fused_basis(modrho, chg, molecule), fuse_transform_matrix(modrho, chg)


def _brute_force_images(system, basis, fused, transform, lat_opts, n_max):
    """sum_{T != 0} (phi_T | mu_0 nu_R), every fused shell at every image."""
    from vibeqc._vibeqc_core import ShellInfo, compute_3c_eri_lattice

    molecule = system.unit_cell_molecule()
    n_electrons = sum(int(atom.Z) for atom in molecule.atoms)
    lattice = np.asarray(system.lattice, dtype=float)
    total = 0.0
    for n in itertools.product(range(-n_max, n_max + 1), repeat=3):
        if n == (0, 0, 0):
            continue
        image = lattice @ np.array(n, dtype=float)
        shifted_molecule = vq.Molecule(
            [vq.Atom(int(a.Z), list(np.asarray(a.xyz) + image)) for a in molecule.atoms],
            0, 1 + n_electrons % 2,
        )
        shells = [
            ShellInfo(int(s.atom_index), int(s.l), bool(s.pure), list(s.exponents),
                      list(s.coefficients), list(np.asarray(s.origin) + image))
            for s in fused.shells()
        ]
        tensor = np.asarray(compute_3c_eri_lattice(
            basis, vq.BasisSet(shifted_molecule, shells, "image", True), system, lat_opts))
        total = total + np.einsum("Pq,qmn->Pmn", transform, tensor)
    return total


def test_auxiliary_image_sum_matches_translated_images():
    """The images the compcell/MDF tensor used to omit, summed exactly.

    Largest element 4.1e-3 on this cube (it was the whole 33 mHa MDF and
    162 mHa compcell error); beyond |n| = 2 the images fall below 1e-11.
    """
    from vibeqc.aux_basis import aux_image_3c

    system, basis = _h2_cube6()
    fused, transform = _fused(system, basis)
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 30.0
    exact = _brute_force_images(system, basis, fused, transform, lat_opts, n_max=2)
    summed = np.einsum("Pq,qmn->Pmn", transform,
                       aux_image_3c(system, basis, fused, lat_opts, precision=1e-8))
    assert np.max(np.abs(exact)) > 4e-3
    assert np.max(np.abs(summed - exact)) < 1e-10
    # The (k_bra, k_ket) form reduces to the Gamma tensor at the zero pair.
    # compute_3c_eri_lattice symmetrises (mu, nu) per call and the block
    # kernel does not; neither the home cell nor one image is symmetric on
    # its own, their sum is, so compare the whole tensor.
    from vibeqc._vibeqc_core import compute_3c_eri_lattice, compute_3c_eri_lattice_blocks

    _, _, blocks = compute_3c_eri_lattice_blocks(basis, fused, system, lat_opts)
    (pair,) = aux_image_3c(system, basis, fused, lat_opts, precision=1e-8,
                           k_pairs=[(np.zeros(3), np.zeros(3))])
    whole_pair = np.einsum("Pq,qmn->Pmn", transform, np.asarray(blocks).sum(axis=0) + pair)
    whole_gamma = np.einsum("Pq,qmn->Pmn", transform, np.asarray(
        compute_3c_eri_lattice(basis, fused, system, lat_opts))) + summed
    assert np.max(np.abs(whole_pair - whole_gamma)) < 1e-10


#: PySCF 2.14.0 KRHF with mix_density_fit at Gamma, settings as above.
PYSCF_H2_CUBE6_MDF = -1.2385310881509575


@pytest.mark.parametrize("gdf_method, reference", [
    ("compcell", PYSCF_H2_CUBE6), ("mdf", PYSCF_H2_CUBE6_MDF),
])
def test_compcell_and_mdf_match_pyscf_on_overlapping_images(gdf_method, reference):
    """Before the image sum: MDF 33 mHa and compcell 162 mHa from PySCF.

    compcell fits the same periodic auxiliary functions as PySCF's GDF;
    MDF is compared with PySCF's MDF. Measured 2026-09-29: compcell
    -8.3e-8, MDF +1.9e-8 Ha.
    """
    system, basis = _h2_cube6()
    gamma = run_pbc_gdf_rhf(system, basis, gdf_method=gdf_method, progress=False)
    k111 = run_krhf_periodic_gdf(system, basis, (1, 1, 1), gdf_method=gdf_method,
                                 progress=False)
    assert gamma.energy == pytest.approx(reference, abs=1e-6)
    assert k111.energy == pytest.approx(gamma.energy, abs=ROUTE_TOL)


def test_compcell_energy_does_not_depend_on_compcell_eta():
    """compcell_eta is a conditioning knob; the G = 0 term made it physical.

    Without the G = 0 term of the compensated tensor the 16-bohr cube sat
    -0.6 mHa from the SR/LR fit at eta = 1 and crossed it near eta = 1 only
    by accident of the eta scan.
    """
    system = vq.PeriodicSystem(
        3, np.eye(3) * 10.0, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])], 0, 1
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    rsgdf = run_pbc_gdf_rhf(system, basis, progress=False).energy
    for eta in (0.5, 1.0, 2.0):
        compcell = run_pbc_gdf_rhf(system, basis, gdf_method="compcell",
                                   compcell_eta=eta, progress=False).energy
        assert compcell == pytest.approx(rsgdf, abs=1e-7), eta


def test_compcell_gradient_differentiates_the_auxiliary_images(monkeypatch):
    """Analytic Gamma compcell gradient against FD on the overlapping cube.

    The image derivative needs its own atom mapping (the native kernel
    assigns shells to atoms by position); dropping it broke translation
    invariance by 4e-3 Ha/bohr here.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4

    def run(dz, gradient=False):
        atoms = [vq.Atom(1, [3.0, 3.0, 2.3 + dz]), vq.Atom(1, [3.3, 3.0, 3.7])]
        system = vq.PeriodicSystem(3, np.eye(3) * 6.0, atoms, 0, 1)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        options = vq.PeriodicRHFOptions()
        options.lattice_opts.cutoff_bohr = 12.0
        options.lattice_opts.nuclear_cutoff_bohr = 12.0
        options.conv_tol_energy = 1e-11
        return run_pbc_gdf_rhf(
            system, basis, options, gdf_method="compcell", aux_basis="def2-svp-jk",
            compcell_eta=1.0, apply_aft_correction=True, exxdiv="ewald",
            compute_gradient=gradient, progress=False,
        )

    result = run(0.0, gradient=True)
    gradient = np.asarray(result.gradient)
    fd = (run(+delta).energy - run(-delta).energy) / (2 * delta)
    assert np.max(np.abs(gradient.sum(axis=0))) < 1e-10
    assert gradient[0, 2] == pytest.approx(fd, abs=1e-6)


# --- one crystal, any lattice image of an atom (#438) ------------------------


def _h2_cube6_shifted(shift, cutoff=12.0):
    """The #438 fixture with atom 0 moved by the lattice vector ``shift``."""
    atoms = [vq.Atom(1, list(np.array([3.0, 3.0, 2.3]) + shift)),
             vq.Atom(1, [3.3, 3.0, 3.7])]
    system = vq.PeriodicSystem(3, np.eye(3) * 6.0, atoms, 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.lattice_opts.cutoff_bohr = cutoff
    options.lattice_opts.nuclear_cutoff_bohr = cutoff
    options.conv_tol_energy = 1e-11
    return system, basis, options


@pytest.mark.parametrize("cutoff", [12.0, 24.0])
@pytest.mark.parametrize("gdf_method", ["compcell", "mdf", "rsgdf"])
def test_gamma_energy_is_invariant_under_a_lattice_translation(monkeypatch, gdf_method, cutoff):
    """Moving atom 0 by -6 bohr along z describes the same crystal.

    Two defects made it a different energy. The compcell AFT correction and
    the MDF plane-wave projection Fourier transformed the AO pairs of a
    ball of cells around the origin, while the Gaussian three-centre tensor
    they correct keeps every pair within the cutoff of each other. And the
    one-electron nuclear attraction took its erfc images around the origin
    rather than around each AO pair (every route). Filed: compcell E moved
    by 2.3e-4 Ha and the gradient by 1.5e-4 Ha/bohr, MDF E by 2.1e-4 Ha,
    rsgdf E by 1.4e-5 Ha. Measured after, at cutoffs 12, 18 and 24 bohr:
    1.8e-13 Ha and 1.8e-13 Ha/bohr (compcell), 1.4e-13 Ha (MDF), 2e-16 Ha and
    8e-14 Ha/bohr (rsgdf).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    gradient = gdf_method != "mdf"  # no public MDF gradient
    results = []
    for shift in ([0.0, 0.0, 0.0], [0.0, 0.0, -6.0]):
        system, basis, options = _h2_cube6_shifted(np.array(shift), cutoff)
        kwargs = dict(compcell_eta=1.0) if gdf_method == "compcell" else {}
        results.append(run_pbc_gdf_rhf(
            system, basis, options, gdf_method=gdf_method, aux_basis="def2-svp-jk",
            exxdiv="ewald", compute_gradient=gradient, progress=False, **kwargs))
    base, moved = results
    assert moved.energy == pytest.approx(base.energy, abs=1e-9)
    assert moved.e_coulomb == pytest.approx(base.e_coulomb, abs=1e-9)
    assert moved.e_hf_exchange == pytest.approx(base.e_hf_exchange, abs=1e-9)
    if gradient:
        np.testing.assert_allclose(moved.gradient, base.gradient, atol=1e-7, rtol=0)


@pytest.mark.parametrize("gdf_method", ["compcell", "mdf", "rsgdf"])
def test_kmesh_energy_is_invariant_under_a_lattice_translation(monkeypatch, gdf_method):
    """#438 on a (2,1,1) mesh: a translation along x multiplies atom 0's
    Bloch functions by -1 at the second k point, one along z by nothing.
    Measured: at most 2.6e-13 Ha on all three fits (main: compcell 6.8e-4,
    MDF 3.5e-4 Ha before the pair-FT fix)."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    energies = []
    for shift in ([0.0, 0.0, 0.0], [0.0, 0.0, -6.0], [-6.0, 0.0, 0.0]):
        system, basis, options = _h2_cube6_shifted(np.array(shift))
        energies.append(run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), options, gdf_method=gdf_method,
            aux_basis="def2-svp-jk", progress=False).energy)
    assert energies[1] == pytest.approx(energies[0], abs=1e-9)
    assert energies[2] == pytest.approx(energies[0], abs=1e-9)


def test_bulk_one_electron_domain_is_pair_complete_and_screened():
    """The bulk GDF one-electron domain keeps the AO pairs within a cutoff
    of each other (``pair_complete_1e``), sized by the #196 overlap screen
    measured on that pair-complete set, so it does not depend on the
    labelling (#438): both labellings of the fixture give 20.40 bohr."""
    from vibeqc.periodic_k_gdf import _oneel_lattice_opts

    cutoffs = []
    for shift in ([0.0, 0.0, 0.0], [0.0, 0.0, -6.0]):
        system, basis, options = _h2_cube6_shifted(np.array(shift))
        resolved = _oneel_lattice_opts(system, basis, options.lattice_opts,
                                       rcut_strategy="pyscf_auto",
                                       k_points_cart=np.zeros((1, 3)))
        assert resolved.pair_complete_1e
        cutoffs.append(resolved.cutoff_bohr)
    assert cutoffs[0] == pytest.approx(cutoffs[1], abs=1e-9)
    assert 12.0 < cutoffs[0] < 25.0




def test_gamma_supercell_matches_the_folded_k_mesh_with_a_short_nuclear_cutoff():
    """#457/#438: two H2 pairs at Gamma and one pair on a (2,1,1) mesh are the
    same crystal. A 6-bohr nuclear cutoff puts the face nuclei of the
    two-pair cell (|L| = 7.8 bohr) outside a ball around the origin, where
    the one-electron nuclear images used to be taken: main split the two by
    -4.40e-3 Ha per pair. With the images taken around each AO pair:
    -3.5e-13."""
    def chain(pairs):
        atoms = []
        for i in range(pairs):
            atoms += [vq.Atom(1, [i * 3.9, 0.0, 0.0]), vq.Atom(1, [i * 3.9 + 1.4, 0.0, 0.0])]
        system = vq.PeriodicSystem(3, np.diag([pairs * 3.9, 10.0, 10.0]), atoms, 0, 1)
        return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def options():
        o = vq.PeriodicRHFOptions()
        o.lattice_opts.nuclear_cutoff_bohr = 6.0
        o.conv_tol_energy = 1e-11
        return o

    system, basis = chain(1)
    folded = run_krhf_periodic_gdf(system, basis, (2, 1, 1), options(), progress=False)
    system, basis = chain(2)
    supercell = run_krhf_periodic_gdf(system, basis, (1, 1, 1), options(), progress=False)
    assert supercell.energy / 2 == pytest.approx(folded.energy, abs=1e-9)

#: rsgdf at Gamma on the four-pair chain below, Ha per H2 pair; equal to the
#: (4,1,1) k mesh of the one-pair cell to 1e-12 (#458).
RSGDF_H8_CHAIN_PER_PAIR = -1.0995145426166342


@pytest.mark.parametrize("gdf_method, tol", [("compcell", 1e-7), ("mdf", 5e-5)])
def test_chain_cell_with_eight_atoms_matches_rsgdf(gdf_method, tol):
    """#458: four H2 pairs in one cell (R_S 1.4, R_pair 3.9 bohr).

    The compcell AFT and the MDF plane-wave pair FTs took the AO pairs of
    a ball of cells around the origin. With a 13-bohr intra-cell offset
    that set is far from the Gaussian tensor's own, and the fits converged
    to -40.38 (compcell) and -20.66 (mdf) Ha per pair. Measured after:
    -1.099514536 and -1.099499926 against rsgdf -1.099514543.
    """
    n = 4
    atoms = []
    for i in range(n):
        atoms += [vq.Atom(1, [i * 3.9, 0.0, 0.0]), vq.Atom(1, [i * 3.9 + 1.4, 0.0, 0.0])]
    system = vq.PeriodicSystem(3, np.diag([n * 3.9, 15.0, 15.0]), atoms, 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-9
    options.max_iter = 80
    result = run_krhf_periodic_gdf(system, basis, (1, 1, 1), options,
                                   gdf_method=gdf_method, progress=False)
    assert result.converged
    assert result.energy / n == pytest.approx(RSGDF_H8_CHAIN_PER_PAIR, abs=tol)


# --- the classic fit's AO-pair support (#466) ----------------------------------


def _classic_chain_energy_per_pair(n_pairs, kx, **kwargs):
    """H-chain (R_S 1.4, R_pair 3.9 bohr) on the classic fit, 6-bohr pairs.

    The short lattice cutoff puts the pairs that cross the 7.8-bohr cell's
    face (2.5 bohr apart, |g| = 7.8) outside a 6-bohr ball of cells around
    the origin, which is what a 15-bohr cutoff does to an H8 cell.
    """
    atoms = [vq.Atom(1, [i * 3.9 + d, 0.0, 0.0])
             for i in range(n_pairs) for d in (0.0, 1.4)]
    system = vq.PeriodicSystem(3, np.diag([n_pairs * 3.9, 10.0, 10.0]), atoms, 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-10
    options.max_iter = 80
    options.lattice_opts.cutoff_bohr = 6.0
    result = run_krhf_periodic_gdf(system, basis, (kx, 1, 1), options,
                                   fit_source="classic", progress=False, **kwargs)
    assert result.converged
    policy = dict(result.executed_fit_source.numerical_support)["classic_support_policy"]
    return result.energy / n_pairs, policy


def test_classic_fit_on_a_supercell_matches_its_folded_k_mesh():
    """#466: H4 at Gamma and H2 on (2,1,1) describe one torus.

    The classic fit took its AO pairs from cells within the cutoff of the
    origin; an H8 chain cell at Gamma sat 9.8 mHa per pair from its folded
    (4,1,1) mesh. The default now keeps every pair within the cutoff of
    each other, the native three-centre term set.
    """
    supercell, policy = _classic_chain_energy_per_pair(2, 1)
    primitive, _ = _classic_chain_energy_per_pair(1, 2)
    assert policy == "physical_pairs"
    assert supercell == pytest.approx(primitive, abs=1e-9)


def test_classic_fit_historical_replay_is_explicit():
    """The former support stays available by name and is recorded as such."""
    supercell, policy = _classic_chain_energy_per_pair(
        2, 1, classic_support_policy="historical_index_ball")
    primitive, _ = _classic_chain_energy_per_pair(
        1, 2, classic_support_policy="historical_index_ball")
    assert policy == "historical_index_ball"
    assert abs(supercell - primitive) > 1e-4


def test_classic_support_policy_refusals():
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])]
    system = vq.PeriodicSystem(3, np.diag([3.9, 10.0, 10.0]), atoms, 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="classic_support_policy applies"):
        run_krhf_periodic_gdf(system, basis, (2, 1, 1), vq.PeriodicRHFOptions(),
                              classic_support_policy="physical_pairs", progress=False)
    with pytest.raises(ValueError, match="classic_support_policy must be"):
        run_krhf_periodic_gdf(system, basis, (2, 1, 1), vq.PeriodicRHFOptions(),
                              fit_source="classic", classic_support_policy="ball",
                              progress=False)
    with pytest.raises(NotImplementedError, match="unscreened"):
        run_krhf_periodic_gdf(system, basis, (2, 1, 1), vq.PeriodicRHFOptions(),
                              fit_source="classic", fit_screen_threshold=1e-10,
                              progress=False)
    # #404: the analytic gradient mirrors the historical index ball only.
    with pytest.raises(NotImplementedError, match="historical_index_ball"):
        run_krhf_periodic_gdf(system, basis, (2, 1, 1), vq.PeriodicRHFOptions(),
                              fit_source="classic", compute_gradient=True,
                              progress=False)


def _classic_h2_cube_energy(dz=0.0, **kwargs):
    atoms = [vq.Atom(1, [3.0, 3.0, 2.2]), vq.Atom(1, [3.0, 3.0, 3.8 + dz])]
    system = vq.PeriodicSystem(3, np.diag([6.0, 6.0, 6.0]), atoms, 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    result = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), options, fit_source="classic",
        classic_support_policy="historical_index_ball", progress=False, **kwargs)
    assert result.converged
    return result


def test_classic_fit_gradient_matches_finite_difference():
    """#404: the explicit classic fit (canonical auxiliary frame, historical
    index ball, the source the neutral Gamma-CCM producers pin) has an
    analytic multi-k gradient equal to the central difference of its energy."""
    h = 2e-4
    grad = np.asarray(_classic_h2_cube_energy(compute_gradient=True).gradient)
    fd = (_classic_h2_cube_energy(+h).energy
          - _classic_h2_cube_energy(-h).energy) / (2 * h)
    assert grad.shape == (2, 3)
    assert grad[1, 2] == pytest.approx(fd, abs=1e-6)


# --- dense-core cells against PySCF (slow) ------------------------------------


@pytest.mark.slow
def test_mgo_gamma_routes_agree_and_match_pyscf():
    """#87: filed with the k(1,1,1) path 0.46 Ha below the Gamma path.

    Measured on main 2026-09-27: run_pbc_gdf_rhf -271.04968793762896,
    KRHF(1,1,1) -271.04968793762635, KUHF(1,1,1) -271.04968793762646 Ha,
    PySCF -271.0496908186412 (+2.9 uHa).
    """
    system, basis = _fcc(7.958, [(12, [0, 0, 0]), (8, [3.979] * 3)], "sto-3g")
    gamma = run_pbc_gdf_rhf(system, basis, progress=False).energy
    k111 = run_krhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).energy
    uhf = run_kuhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).energy
    assert k111 == pytest.approx(gamma, abs=ROUTE_TOL)
    assert uhf == pytest.approx(gamma, abs=ROUTE_TOL)
    assert gamma == pytest.approx(PYSCF_MGO_STO3G, abs=1e-5)


@pytest.mark.slow
def test_diamond_gamma_keeps_the_t2_valence_degeneracy():
    """#163: the filed GDF spectrum split diamond's Gamma T2 valence top by
    165 mHa. Measured on main 2026-09-27: 0.311262 Ha three-fold, total
    -74.00247767843784 Ha against PySCF -74.00247971119255 (+2.0 uHa)."""
    a = 6.740653086540123
    system, basis = _fcc(a, [(6, [0, 0, 0]), (6, [a / 4] * 3)], "sto-3g")
    result = run_pbc_gdf_rhf(system, basis, progress=False)
    n_occ = system.n_electrons() // 2
    top = np.sort(np.asarray(result.mo_energies))[n_occ - 3:n_occ]
    assert np.ptp(top) < 1e-6
    assert result.energy == pytest.approx(PYSCF_CDIA_STO3G, abs=1e-5)
