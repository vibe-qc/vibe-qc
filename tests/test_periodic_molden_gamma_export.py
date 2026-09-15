"""Periodic Molden export: the Γ block, real, and only when Γ exists.

Molden is a molecular format. It has no lattice-vector block, no k-point,
no Bloch phase, and no complex coefficients, so the only thing a periodic
run can put in one is the Γ-point orbital set, truncated to the home cell.
Three defects made that export unusable on the BIPOLE routes:

* **Unrestricted results never wrote at all.** ``PBCBipoleUHFResult`` /
  ``PBCBipoleUKSResult`` carry one ``(nbf, nmo)`` block per k-point under
  ``mo_coeffs_alpha`` / ``mo_coeffs_beta`` and have no ``mo_coeffs``, so the
  runner's Γ proxy (which keyed on ``mo_coeffs``) passed the whole per-k
  list through. ``write_molden`` then unpacked a 3-D array and raised
  ``ValueError: too many values to unpack``, leaving the run with a
  "molden write failed" warning and no file (PF028).

* **Restricted results wrote a corrupt file.** Bloch MO coefficients are
  complex128 even at Γ, and numpy formats a complex scalar as ``a+bj``, so
  every coefficient landed as ``9.93E-01+2.83E-20j`` -- text no Molden
  reader can parse, emitted silently under a "written" log line. Two such
  files shipped in ``docs/_static/examples/mgo-route-gdf-bipole/``.

* **Multi-k requests aborted the job.** The sidecar capability gate
  required the mesh to *be* Γ rather than to *contain* Γ, so
  ``write_molden_file=True`` on a Γ-centred Monkhorst-Pack mesh raised
  ``NotImplementedError`` before the SCF started (PF031/PF032/PF034).

These regressions pin the fix and, just as importantly, the refusals: an
orbital at k != 0 is irreducibly complex, and a mesh without Γ has no
exportable block. Both must fail loudly rather than produce a file whose
numbers a viewer would render as something they are not.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.output.formats.molden import (
    _MOLDEN_IMAG_RTOL,
    _real_mo_coefficients,
    write_molden,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
from vibeqc.periodic_runner import (
    _gamma_orbital_proxy,
    _kmesh_contains_gamma,
    _result_with_ecp_ncore,
)


def _bipole_options() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = 2
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.conv_tol_energy = 1e-30
    opts.conv_tol_grad = 1e-30
    return opts


def _ne_cubic():
    """Closed-shell Ne/STO-3G in a 7-bohr box: 1s + 2s + 2p, five AOs.

    A p shell is the point: L >= 1 is where Molden and libint disagree on
    the within-shell ordering, so a single-s fixture would not exercise the
    AO permutation the writer applies to the realified coefficients.
    """
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(10, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _f_cubic_doublet():
    """Open-shell F/STO-3G doublet: the PF028 shape, five AOs per spin."""
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(9, [0.0, 0.0, 0.0])], multiplicity=2
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _coefficient_lines(text: str) -> list[str]:
    """Molden [MO] coefficient rows: ``<index> <value>``."""
    lines: list[str] = []
    in_mo = False
    for line in text.splitlines():
        if line.startswith("[MO]"):
            in_mo = True
            continue
        if not in_mo or line.startswith("["):
            continue
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit():
            lines.append(line)
    return lines


def _spin_labels(text: str) -> list[str]:
    return [
        line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith(" Spin=")
    ]


# ---------------------------------------------------------------------
# The coefficient realification itself
# ---------------------------------------------------------------------


def test_real_coefficients_pass_through_unchanged():
    """A real array keeps its values and its signs exactly."""
    coeffs = np.array([[0.6, -0.8], [-0.8, -0.6]])
    np.testing.assert_array_equal(
        _real_mo_coefficients(coeffs, "alpha"), coeffs
    )


def test_global_phase_is_divided_out():
    """A column that is real up to one global phase comes back real.

    This is the Γ case: H(k=0) and S(k=0) are real symmetric, so the
    complex eigensolver returns each eigenvector real times an arbitrary
    e^{i*theta}. Dividing that out is exact, not an approximation.
    """
    real_column = np.array([0.6, 0.8])
    rotated = (real_column * np.exp(1j * 0.7)).reshape(2, 1)

    out = _real_mo_coefficients(rotated, "alpha")

    assert out.dtype == np.float64
    # Recovered up to the residual global sign, which is unobservable.
    np.testing.assert_allclose(np.abs(out[:, 0]), real_column, atol=1e-12)
    assert np.allclose(out[:, 0], real_column, atol=1e-12) or np.allclose(
        out[:, 0], -real_column, atol=1e-12
    )


def test_already_real_complex_column_keeps_its_sign():
    """A complex-typed but real-valued column is not sign-flipped.

    Periodic drivers hand back complex128 containers holding real Γ data.
    Rotating a column whose largest coefficient is negative would flip
    every sign for no physical reason, so realification leaves it alone.
    """
    column = np.array([-0.8 + 0.0j, 0.6 + 0.0j]).reshape(2, 1)
    out = _real_mo_coefficients(column, "alpha")
    np.testing.assert_allclose(out[:, 0], [-0.8, 0.6], atol=1e-15)


def test_genuinely_complex_orbital_is_refused():
    """A k != 0 Bloch orbital cannot be written, and says so.

    No phase choice makes [1, i] real. Molden has no representation for
    it, so the writer must raise rather than emit numbers a viewer would
    silently misread as real coefficients.
    """
    column = np.array([1.0 + 0.0j, 0.0 + 1.0j]).reshape(2, 1)

    with pytest.raises(ValueError, match="not real after removing"):
        _real_mo_coefficients(column, "beta")


def test_refusal_names_the_spin_and_orbital():
    """The diagnostic locates the offending orbital, not just the file."""
    coeffs = np.array(
        [[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 1.0j]]
    )
    with pytest.raises(ValueError, match=r"beta orbital 2"):
        _real_mo_coefficients(coeffs, "beta")


def test_roundoff_imaginary_residual_is_accepted():
    """Eigensolver roundoff at Γ must not trip the refusal."""
    column = np.array([1.0 + 1e-16j, 0.5 + 0.0j]).reshape(2, 1)
    out = _real_mo_coefficients(column, "alpha")
    np.testing.assert_allclose(out[:, 0], [1.0, 0.5], atol=1e-12)


def test_tolerance_separates_degeneracy_roundoff_from_complex_orbitals():
    """The threshold sits between the only two regimes that occur.

    Degeneracy-amplified roundoff at Γ reaches ~1e-5 (measured: the
    exactly degenerate Ne 2p pair below). A genuine Bloch orbital at
    k != 0 is O(1). The threshold must accept the first and reject the
    second with room on both sides.
    """
    assert 1e-4 <= _MOLDEN_IMAG_RTOL <= 1e-2

    degeneracy_roundoff = np.array(
        [1.0 + 7.5e-6j, 0.5 + 0.0j], dtype=complex
    ).reshape(2, 1)
    _real_mo_coefficients(degeneracy_roundoff, "alpha")  # accepted

    genuinely_complex = np.array([1.0 + 0.0j, 0.0 + 0.1j]).reshape(2, 1)
    with pytest.raises(ValueError, match="not real after removing"):
        _real_mo_coefficients(genuinely_complex, "alpha")


def test_small_degenerate_unitary_rotation_is_reorthonormalized():
    """A below-threshold imaginary rotation must not leak into Molden.

    The old gate looked only at ``max(abs(Im C))``.  A unitary rotation of
    two exactly degenerate real orbitals can sit below that threshold while
    dropping its imaginary part still changes ``C.T S C`` by O(theta**2).
    That made the release gate depend on the eigensolver/BLAS accumulation
    order.  The overlap metric, not the complex-container residue, decides
    whether the real projection is already a valid orbital basis.
    """
    theta = 5.0e-5
    c = np.cos(theta)
    s = np.sin(theta)
    coeffs = np.array([[c, 1j * s], [1j * s, c]], dtype=complex)
    assert np.max(np.abs(coeffs.imag)) < _MOLDEN_IMAG_RTOL

    out = _real_mo_coefficients(
        coeffs,
        "alpha",
        energies=np.zeros(2),
        overlap=np.eye(2),
    )

    np.testing.assert_allclose(out.T @ out, np.eye(2), atol=1.0e-13)


def test_genuinely_complex_degenerate_subspace_is_refused():
    """Degeneracy must not manufacture real orbitals away from Gamma."""
    coeffs = np.array(
        [
            [1.0, 0.0],
            [1.0j, 0.0],
            [0.0, 1.0],
            [0.0, 1.0j],
        ],
        dtype=complex,
    ) / np.sqrt(2.0)

    with pytest.raises(ValueError, match="not real after removing"):
        _real_mo_coefficients(
            coeffs,
            "alpha",
            energies=np.zeros(2),
            overlap=np.eye(4),
        )


def test_degenerate_gamma_orbitals_export_cleanly():
    """An exactly degenerate Γ block is writable, not a refusal.

    Ne/STO-3G in a cubic box has an exactly degenerate 2p pair. Inside a
    degenerate block the eigensolver returns an arbitrary unitary mixture
    of the partners, so roundoff is amplified far above machine epsilon
    and -- unlike convergence error -- does not shrink as the SCF
    converges. Taking the real part is still exact: the result stays an
    eigenvector at the same eigenvalue. Refusing here would make the
    export useless on essentially every crystal, since degeneracy is the
    norm rather than the exception.
    """
    system, basis = _ne_cubic()
    result = run_pbc_bipole_rhf(
        system,
        basis,
        monkhorst_pack(system, [2, 1, 1], use_symmetry=False),
        _bipole_options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    energies = np.real(np.asarray(result.mo_energies[0]))
    gaps = np.abs(np.diff(np.sort(energies)))
    assert np.min(gaps) < 1e-10, "fixture lost its exact degeneracy"

    proxy = _gamma_orbital_proxy(result)
    coeffs = _real_mo_coefficients(
        np.asarray(proxy.mo_coeffs),
        "alpha",
        np.asarray(proxy.mo_energies),
        np.asarray(proxy.overlap),
    )
    assert coeffs.dtype == np.float64
    assert np.isfinite(coeffs).all()


def test_per_k_list_shape_is_rejected_with_a_useful_message():
    """A whole per-k stack must not be mistaken for one orbital set."""
    stack = np.zeros((2, 3, 3), dtype=complex)
    with pytest.raises(ValueError, match="select the Gamma block"):
        _real_mo_coefficients(stack, "alpha")


# ---------------------------------------------------------------------
# Γ membership decides whether an export exists at all
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mesh", [[1, 1, 1], [2, 1, 1], [2, 2, 2], [3, 3, 3]])
def test_gamma_centred_meshes_contain_gamma(mesh):
    """Every Γ-centred Monkhorst-Pack mesh carries an exportable Γ block."""
    system, _ = _ne_cubic()
    assert _kmesh_contains_gamma(monkhorst_pack(system, mesh))


def test_shifted_mesh_has_no_gamma_block():
    """A shifted mesh has no k = 0, so no orbital set can be exported."""
    system, _ = _ne_cubic()

    class _ShiftedMesh:
        # Half-shifted 2x2x2 in units where the Γ-centred mesh would
        # include the origin; no entry is Γ.
        kpoints = np.full((8, 3), 0.25)

    assert not _kmesh_contains_gamma(_ShiftedMesh())


# ---------------------------------------------------------------------
# End to end on real BIPOLE results
# ---------------------------------------------------------------------


def test_bipole_uhf_gamma_export_writes_both_spin_blocks(tmp_path):
    """PF028: an open-shell Γ BIPOLE result produces a usable Molden file.

    Before the fix ``mo_coeffs_alpha`` reached the writer as a per-k list
    and ``write_molden`` raised ``too many values to unpack``.
    """
    system, basis = _f_cubic_doublet()
    molecule = system.unit_cell_molecule()
    result = run_pbc_bipole_uhf(
        system,
        basis,
        monkhorst_pack(system, [1, 1, 1]),
        _bipole_options(),
        ewald_precision=1e-6,
        progress=False,
    )
    # The shape that used to break the writer.
    assert isinstance(result.mo_coeffs_alpha, list)

    path = tmp_path / "uhf.molden"
    write_molden(path, molecule, basis, _gamma_orbital_proxy(result), title="t")

    text = path.read_text()
    assert _spin_labels(text) == ["Alpha"] * len(
        result.mo_energies_alpha[0]
    ) + ["Beta"] * len(result.mo_energies_beta[0])
    rows = _coefficient_lines(text)
    assert rows, "no [MO] coefficients written"
    for row in rows:
        float(row.split()[1])  # parses as a real number, not "a+bj"


def test_bipole_multik_export_writes_the_gamma_block(tmp_path):
    """PF031/032/034: a Γ-centred multi-k run exports its Γ orbitals.

    The written coefficients must be the Γ block of the converged multi-k
    result -- not the first k-point, and not a complex-formatted anything.
    """
    system, basis = _ne_cubic()
    molecule = system.unit_cell_molecule()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _bipole_options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    kpts = np.asarray(result.kpoints_cart, dtype=float)
    gamma_idx = int(np.argmin(np.linalg.norm(kpts, axis=1)))
    assert np.linalg.norm(kpts[gamma_idx]) < 1e-10

    proxy = _gamma_orbital_proxy(result)
    np.testing.assert_allclose(
        np.abs(np.asarray(proxy.mo_coeffs)),
        np.abs(np.asarray(result.mo_coeffs[gamma_idx])),
        atol=1e-12,
    )

    path = tmp_path / "multik.molden"
    write_molden(path, molecule, basis, proxy, title="t")
    text = path.read_text()
    assert not [line for line in text.splitlines() if line.rstrip().endswith("j")]
    assert _coefficient_lines(text)


def test_exported_coefficients_match_the_gamma_eigenvectors(tmp_path):
    """The file's numbers are the Γ eigenvectors, not a re-derivation.

    Read the coefficients back and compare against the driver's own Γ
    block (up to the AO permutation's magnitude-preserving reorder and the
    unobservable global sign).
    """
    system, basis = _ne_cubic()
    molecule = system.unit_cell_molecule()
    result = run_pbc_bipole_rhf(
        system,
        basis,
        monkhorst_pack(system, [1, 1, 1]),
        _bipole_options(),
        ewald_precision=1e-6,
        progress=False,
    )
    path = tmp_path / "gamma.molden"
    write_molden(
        path, molecule, basis, _gamma_orbital_proxy(result), title="t"
    )

    written = np.array(
        [float(row.split()[1]) for row in _coefficient_lines(path.read_text())]
    )
    expected = np.asarray(result.mo_coeffs[0])
    assert written.size == expected.size
    np.testing.assert_allclose(
        np.sort(np.abs(written)),
        np.sort(np.abs(expected).ravel()),
        atol=1e-9,
    )


def test_result_occupations_are_preferred_over_an_aufbau_guess(tmp_path):
    """Fractional occupations survive the export.

    Under finite-T smearing the exported k-block has fractionally occupied
    frontier orbitals; an aufbau guess from the cell's electron count would
    misreport them. Molecular results carry no occupations and keep the
    aufbau fallback.
    """
    system, basis = _ne_cubic()
    molecule = system.unit_cell_molecule()

    n_ao = 5  # Ne/STO-3G

    class _SmearedGammaResult:
        mo_coeffs_alpha = np.eye(n_ao)
        mo_coeffs_beta = np.eye(n_ao)
        mo_energies_alpha = np.linspace(-1.0, 1.0, n_ao)
        mo_energies_beta = np.linspace(-1.0, 1.0, n_ao)
        occupations_alpha = np.array([1.0, 1.0, 1.0, 0.37, 0.0])
        occupations_beta = np.array([1.0, 1.0, 1.0, 0.63, 0.0])

    path = tmp_path / "smeared.molden"
    write_molden(path, molecule, basis, _SmearedGammaResult(), title="t")

    occupations = [
        float(line.split("=", 1)[1])
        for line in path.read_text().splitlines()
        if line.startswith(" Occup=")
    ]
    # The fractional frontier occupations survive; an aufbau guess from the
    # cell's 10 electrons would have written 1.0 and 0.0 here.
    assert occupations[3] == pytest.approx(0.37)
    assert occupations[n_ao + 3] == pytest.approx(0.63)


def test_periodic_ecp_fallback_uses_variational_electron_count(tmp_path):
    """Runner provenance keeps ECP-removed virtual MOs unoccupied."""
    system, basis = _ne_cubic()
    molecule = system.unit_cell_molecule()
    n_ao = basis.nbasis

    class _GammaResultWithoutOccupations:
        mo_coeffs = np.eye(n_ao)
        mo_energies = np.linspace(-1.0, 1.0, n_ao)

    result = _result_with_ecp_ncore(
        _GammaResultWithoutOccupations(),
        2,
    )
    path = tmp_path / "effective-electrons.molden"
    write_molden(path, molecule, basis, result, title="t")

    occupations = [
        float(line.split("=", 1)[1])
        for line in path.read_text().splitlines()
        if line.startswith(" Occup=")
    ]
    assert occupations == pytest.approx([2.0, 2.0, 2.0, 2.0, 0.0])


def test_multik_gdf_exports_a_real_gamma_block(tmp_path):
    """Multi-k GDF exports too, once degenerate blocks are real-ified.

    GDF was briefly excluded from the multi-k allowlist because its Gamma
    block left a degenerate frontier orbital complex after global-phase
    removal. The cause was degeneracy, not the route: one global phase per
    column cannot undo a rotation *between* columns. With degenerate blocks
    re-expressed on a real basis of their own span, GDF exports the same as
    BIPOLE.
    """
    system, basis = _ne_cubic()
    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        aux_basis="def2-svp-jk",
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0,
        max_iter=60,
        conv_tol_energy=1e-9,
        output=str(tmp_path / "gdf"),
        citations=False,
        write_molden_file=True,
        write_xyz_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        progress=False,
    )
    assert result.converged
    path = tmp_path / "gdf.molden"
    assert path.exists(), "multi-k GDF should now export a Gamma-block Molden"
    text = path.read_text()
    assert not [ln for ln in text.splitlines() if ln.rstrip().endswith("j")]


def test_shifted_mesh_refuses_an_explicit_molden_request():
    """No Γ in the mesh means no exportable orbital set, and it says so."""
    system, basis = _ne_cubic()
    shifted = vq.monkhorst_pack(system, [2, 2, 2], [1, 1, 1], False)
    assert not _kmesh_contains_gamma(shifted)

    with pytest.raises(NotImplementedError, match="write_molden_file"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            kpoints=shifted,
            write_molden_file=True,
            progress=False,
        )


@pytest.mark.parametrize("jk", ["bipole", "gdf"])
def test_degenerate_blocks_realify_to_s_orthonormal_orbitals(jk, tmp_path):
    """Realified degenerate orbitals stay an orthonormal eigenbasis.

    Re-expressing a degenerate block on a real basis of its own span is
    only legitimate if what comes back is still a set of S-orthonormal
    eigenvectors at that eigenvalue. Any rotation within the eigenspace is
    as valid as any other, so the test pins the invariant that must hold --
    C^T S C = I in the Gamma overlap metric -- not a particular rotation.
    """
    from vibeqc.output.formats.molden import _real_mo_energies

    system, basis = _ne_cubic()
    kwargs = dict(
        system=system,
        basis=basis,
        method="RHF",
        jk_method=jk,
        kpoints=(2, 1, 1),
        max_iter=60,
        conv_tol_energy=1e-9,
        output=str(tmp_path / f"{jk}-degenerate"),
        progress=False,
        citations=False,
        write_molden_file=None,
        write_population_file=None,
        write_xyz_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
    )
    if jk == "bipole":
        kwargs.update(bipole_cutoff_bohr=12.0, bipole_nuclear_cutoff_bohr=12.0)
    else:
        kwargs.update(
            aux_basis="def2-svp-jk", gdf_method="rsgdf", rsgdf_ke_cutoff=200.0
        )
    result = vq.run_periodic_job(**kwargs)

    proxy = _gamma_orbital_proxy(result)
    energies = _real_mo_energies(proxy.mo_energies)
    # The fixture must actually contain a degenerate block, or the test
    # would pass without exercising the path it exists for.
    assert np.min(np.abs(np.diff(np.sort(energies)))) < 1e-10

    overlap = np.real(np.asarray(proxy.overlap))
    coeffs = _real_mo_coefficients(
        proxy.mo_coeffs, "alpha", energies, proxy.overlap
    )
    assert coeffs.dtype == np.float64
    np.testing.assert_allclose(
        coeffs.T @ overlap @ coeffs,
        np.eye(coeffs.shape[1]),
        atol=1e-10,
    )
