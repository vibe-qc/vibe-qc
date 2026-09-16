"""Phase 14c tests: ECP-aware molecular SCF drivers.

Contracts exercised:

1. **Inertness with empty ECP list** — for each of RHF, UHF, RKS,
   UKS, calling the driver with ``options.ecp_centers = []`` must
   produce the all-electron SCF bit-for-bit (same energy, same
   iteration count, same density).

2. **Default-options inertness** — an ``RHFOptions()`` (or sibling)
   that doesn't touch ``ecp_centers`` matches an explicitly-empty
   list bit-for-bit.

3. **ECP-using SCF converges** — Zn²⁺ valence (18 electrons after
   the 10-electron ecp10mdf core) at 6-31G converges to a finite,
   negative energy.

4. **Default ecp_library = ecp10mdf** — leaving ``ecp_library`` empty
   resolves to ``ecp10mdf`` and produces the same energy as setting
   it explicitly.

5. **Library override** — switching ``ecp_library`` changes the SCF
   energy (different physics).

6. **Symmetry preserved** — the ECP SCF density and Fock are
   symmetric to ~10⁻¹².

7. **All four molecular drivers** carry the ``ecp_centers`` /
   ``ecp_library`` fields and exercise the same code path.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    return mol, basis


def _zn_ion():
    """Zn²⁺ for closed-shell ECP testing.

    Zn (Z=30, neutral) has 30 electrons; Zn²⁺ has 28. Molecule.charge
    is the physical ionic charge (+2), so n_electrons() = 28 — the full
    physical count. The SCF subtracts the ecp10mdf core (10 electrons,
    1s²2s²2p⁶) itself, leaving 18 valence (= 3d¹⁰).
    """
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 2, 1)
    basis = vq.BasisSet(mol, "6-31g")
    return mol, basis


def _out_block(text, header):
    assert header in text, f"the .out carries no {header!r} block"
    return text.split(header, 1)[1]


def _out_atomic_charges(text, element):
    """Return the (Mulliken, Loewdin, Hirshfeld) row for ``element`` from the
    .out "Atomic charges" table.

    Parsed rather than substring-matched. The table prints six decimals
    (``scf_log.py``, ``_fixed_unsigned_zero(..., 6)``), so an ``in text``
    literal pins the charge to +/-5e-7 -- twenty times tighter than the
    numeric assertions elsewhere in this file, and tighter than the ~3e-6
    that a legitimate libecpint radial-quadrature change moves it (#269).
    """
    for line in _out_block(text, "Atomic charges").splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[1] == element:
            return tuple(float(x) for x in fields[2:])
    raise AssertionError(f"no {element} row in the .out atomic-charges table")


def _out_dipole_debye(text):
    """``|mu|`` in Debye from the .out dipole table.

    Same reasoning as :func:`_out_atomic_charges`: the printed 1.9950 is
    2.2e-6 from rendering as 1.9949, a tighter implicit pin than the charge
    literal that #269 was filed about.
    """
    for line in _out_block(text, "Dipole moment").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "|mu|":
            return float(fields[2])
    raise AssertionError("no |mu| row in the .out dipole table")


def _h2s_lanl2dz():
    """MF076 geometry: neutral singlet H2S with an S/LANL2DZ ECP."""
    mol = vq.Molecule(
        [
            vq.Atom(16, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.815, 1.425]),
            vq.Atom(1, [0.0, -1.815, 1.425]),
        ],
        0,
        1,
    )
    return mol


# ---------------------------------------------------------------------------
# 1. Empty-list / default-options inertness — RHF
# ---------------------------------------------------------------------------

def test_rhf_empty_ecp_list_matches_baseline():
    mol, basis = _h2()
    r0 = vq.run_rhf(mol, basis, vq.RHFOptions())

    opts = vq.RHFOptions()
    opts.ecp_centers = []
    r1 = vq.run_rhf(mol, basis, opts)
    assert r0.energy == pytest.approx(r1.energy, abs=1e-14)
    assert r0.n_iter == r1.n_iter


def test_rhf_default_options_match_empty_ecp_list():
    mol, basis = _h2()
    r_default = vq.run_rhf(mol, basis, vq.RHFOptions())
    opts = vq.RHFOptions()
    opts.ecp_centers = []
    r_explicit = vq.run_rhf(mol, basis, opts)
    assert r_default.energy == pytest.approx(r_explicit.energy, abs=1e-14)


def test_run_job_auto_attaches_lanl2dz_ecp_to_rhf_options(tmp_path):
    """MF076: named ECP bases must not run as all-electron calculations.

    GitLab #143: the archived mb076/mf076/ml076e rows landed -89.637 Ha
    (neither valence-only nor all-electron) because the ECP was never
    attached. The attach itself was fixed by 0247a07e3; this test also
    pins the *consequences* -- the valence electron count actually
    treated by the libecpint route and the valence-only energy scale --
    so a regression to the all-electron treatment cannot pass silently.
    """
    opts = vq.RHFOptions()
    opts.max_iter = 100

    result = vq.run_job(
        _h2s_lanl2dz(),
        basis="lanl2dz",
        method="rhf",
        rhf_options=opts,
        output=tmp_path / "h2s_lanl2dz",
    )

    assert result.converged
    # The sidecar is attached inline (its own primitives), not through a
    # libecpint XML library: one block on sulfur, ten core electrons gone.
    assert list(opts.ecp_centers) == [] and opts.ecp_library == ""
    assert len(opts.ecp_primitive_blocks) == 1
    assert list(opts.ecp_primitive_centers[0]) == list(_h2s_lanl2dz().atoms[0].xyz)
    assert list(opts.ecp_effective_charges) == [6.0, 1.0, 1.0]
    assert opts.ecp_total_ncore == 10
    assert result.ecp_operator_applied and result.ecp_total_ncore == 10

    # Valence electron count actually treated: the molecule carries 18
    # physical electrons; the LANL2DZ ECP replaces the 10-electron sulfur
    # core, so the converged RHF density must integrate to 8 (S 3s²3p⁴ +
    # 2 x H 1s¹).  With AO-basis MO coefficients that is tr(D . S), not
    # tr(D).  An all-electron misapplication would give 18.
    S = vq.compute_overlap(vq.BasisSet(_h2s_lanl2dz(), "lanl2dz"))
    n_elec_treated = float(
        np.trace(np.asarray(result.density, dtype=float) @ np.asarray(S, dtype=float))
    )
    assert n_elec_treated == pytest.approx(8.0, abs=1e-8)

    # Energy scale: the matched ORCA leg is -11.009573292528 Ha at the
    # archived geometry; this fixture's rounded geometry gives
    # -11.009452171730.  The ECP-less all-electron run measured -89.70 Ha
    # -- 78.6 Ha away -- so this pin discriminates the defect class by
    # two orders of magnitude more than any plausible valence-level
    # geometry or convergence drift.
    assert result.energy == pytest.approx(-11.009452171730, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Empty-list inertness on the other three molecular drivers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("driver_name", ["uhf", "rks", "uks"])
def test_other_drivers_empty_ecp_list_matches_baseline(driver_name):
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1 if driver_name in ("rks",) else 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")

    if driver_name == "uhf":
        baseline = vq.run_uhf(mol, basis, vq.UHFOptions())
        opts = vq.UHFOptions()
        opts.ecp_centers = []
        with_empty = vq.run_uhf(mol, basis, opts)
    elif driver_name == "rks":
        opts0 = vq.RKSOptions()
        opts0.functional = "LDA"
        baseline = vq.run_rks(mol, basis, opts0)
        opts = vq.RKSOptions()
        opts.functional = "LDA"
        opts.ecp_centers = []
        with_empty = vq.run_rks(mol, basis, opts)
    else:  # uks
        opts0 = vq.UKSOptions()
        opts0.functional = "LDA"
        baseline = vq.run_uks(mol, basis, opts0)
        opts = vq.UKSOptions()
        opts.functional = "LDA"
        opts.ecp_centers = []
        with_empty = vq.run_uks(mol, basis, opts)
    assert baseline.energy == pytest.approx(with_empty.energy, abs=1e-13)


# ---------------------------------------------------------------------------
# 3. ECP-using SCF on Zn²⁺
# ---------------------------------------------------------------------------

def test_rhf_with_ecp_converges():
    mol, basis = _zn_ion()
    opts = vq.RHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts.ecp_library = "ecp10mdf"
    opts.max_iter = 200
    opts.damping = 0.5

    r = vq.run_rhf(mol, basis, opts)
    assert r.converged
    assert np.isfinite(r.energy)
    assert r.energy < 0.0   # bound electrons
    # Density and Fock are symmetric.
    assert np.abs(r.density - r.density.T).max() < 1e-10
    assert np.abs(r.fock - r.fock.T).max() < 1e-10


# ---------------------------------------------------------------------------
# 4. Default ecp_library = ecp10mdf
# ---------------------------------------------------------------------------

def test_default_ecp_library_is_ecp10mdf():
    mol, basis = _zn_ion()
    opts_default = vq.RHFOptions()
    opts_default.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts_default.max_iter = 200
    # ecp_library left empty → default 'ecp10mdf'

    opts_explicit = vq.RHFOptions()
    opts_explicit.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts_explicit.ecp_library = "ecp10mdf"
    opts_explicit.max_iter = 200

    r_def = vq.run_rhf(mol, basis, opts_default)
    r_exp = vq.run_rhf(mol, basis, opts_explicit)
    assert r_def.converged and r_exp.converged
    assert r_def.energy == pytest.approx(r_exp.energy, abs=1e-12)


# ---------------------------------------------------------------------------
# 5. Library override changes physics
# ---------------------------------------------------------------------------

def test_ecp_library_override_changes_energy():
    mol, basis = _zn_ion()
    opts_mdf = vq.RHFOptions()
    opts_mdf.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts_mdf.ecp_library = "ecp10mdf"
    opts_mdf.max_iter = 200

    opts_lan = vq.RHFOptions()
    opts_lan.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    opts_lan.ecp_library = "lanl2dz"
    opts_lan.max_iter = 200

    r_mdf = vq.run_rhf(mol, basis, opts_mdf)
    r_lan = vq.run_rhf(mol, basis, opts_lan)
    if not (r_mdf.converged and r_lan.converged):
        pytest.skip("Both libraries must converge for the comparison")
    # Different cores (ecp10mdf vs lanl2dz) → different energies.
    assert abs(r_mdf.energy - r_lan.energy) > 1e-3


# ---------------------------------------------------------------------------
# 6. Field exposure on all four option structs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    vq.RHFOptions, vq.UHFOptions, vq.RKSOptions, vq.UKSOptions,
])
def test_ecp_fields_on_all_molecular_options(cls):
    o = cls()
    assert hasattr(o, "ecp_centers")
    assert hasattr(o, "ecp_library")
    assert list(o.ecp_centers) == []
    assert o.ecp_library == ""
    o.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])]
    o.ecp_library = "ecp10mdf"
    assert len(o.ecp_centers) == 1
    assert o.ecp_library == "ecp10mdf"


def test_run_job_hessian_on_ecp_system_reports_real_frequencies(tmp_path):
    """#576: the molecular runner's ``hessian=True`` path passed no
    ``gradient_options`` to ``compute_hessian_fd``, and the Hessian left the
    ECP centres behind while displacing the nucleus, so on an ECP system the
    frequency block either raised (SCF divergence at the first displacement
    of the ECP atom) or silently reported bare-Z force constants. Post-fix
    H2S/LANL2DZ gives its three real modes (bend + two stretches) and the
    thermochemistry block that depends on them."""
    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    result = vq.run_job(
        _h2s_lanl2dz(), basis="lanl2dz", method="rhf", rhf_options=opts,
        hessian=True, output=tmp_path / "h2s_lanl2dz_hess", verbose=0,
    )
    assert result.converged
    assert len(opts.ecp_primitive_blocks) == 1  # the ECP was attached and is still there
    assert list(opts.ecp_primitive_centers[0]) == list(_h2s_lanl2dz().atoms[0].xyz)
    out = (tmp_path / "h2s_lanl2dz_hess.out").read_text()
    assert "## Vibrational Frequencies" in out
    assert "Imaginary modes: 0" in out
    assert "## Thermochemistry (RRHO ideal gas)" in out
    # Measured on the fixed tree: 1029.3, 3756.6, 3814.1 cm^-1 (HF/LANL2DZ,
    # unscaled). Pin the count and the physical range rather than the digits.
    freqs = []
    in_table = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Mode"):
            in_table = True
            continue
        if in_table:
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                freqs.append(float(parts[1]))
            elif freqs:
                break
    assert len(freqs) == 3, out
    assert 900.0 < freqs[0] < 1200.0 and all(3500.0 < f < 4000.0 for f in freqs[1:]), freqs


@pytest.mark.parametrize("backend", ["ase", "native"])
def test_run_job_optimize_and_hessian_on_ecp_system(tmp_path, backend):
    """#643: the runner attaches the ECP for the start geometry BEFORE the
    optimizer starts, the optimizer follows the centres per step, and the
    runner moves them onto the optimised geometry for the final single point
    and the Hessian. Pre-fix run_job(optimize=True) on this fixture raised at
    the first force evaluation ('libecpint reports 4 atoms but the molecule
    has 3'). Measured on the fixed tree: E = -11.03125 Ha at the HF/LANL2DZ
    minimum (start geometry -11.00945), three real modes."""
    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    result = vq.run_job(
        _h2s_lanl2dz(), basis="lanl2dz", method="rhf", rhf_options=opts,
        optimize=True, hessian=True, optimizer_backend=backend,
        output=tmp_path / f"h2s_opt_{backend}", verbose=0,
    )
    assert result.converged
    assert result.energy == pytest.approx(-11.03125, abs=5e-4)
    assert result.energy < -11.0094 - 0.01
    out = (tmp_path / f"h2s_opt_{backend}.out").read_text()
    assert "Optimized geometry" in out
    assert "## Vibrational Frequencies" in out
    assert "Imaginary modes: 0" in out
    # The options describe the optimised geometry afterwards: S moved off
    # the origin along z and the centre went with it.
    centre = list(opts.ecp_primitive_centers[0])
    assert abs(centre[2]) > 0.1, centre


# ---------------------------------------------------------------------------
# GitLab #642: electrostatic properties use Z_eff = Z - n_core on ECP atoms
# ---------------------------------------------------------------------------


def _h2s_lanl2dz_scf():
    """H2S/LANL2DZ RHF with the ECP attached the way the drivers attach it
    (XML-library route: ``ecp_centers`` + ``ecp_library``)."""
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar

    mol = _h2s_lanl2dz()
    basis = vq.BasisSet(mol, "lanl2dz")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    attach_inline_ecp_options_from_basis_sidecar(opts, mol, basis)
    assert list(opts.ecp_primitive_blocks) and opts.ecp_total_ncore == 10
    return mol, basis, opts, vq.run_rhf(mol, basis, opts)


def test_effective_nuclear_charges_xml_route_is_z_minus_ncore():
    """S/LANL2DZ replaces the 10-electron [Ne] core: Z_eff = [6, 1, 1], bare Z
    on the hydrogens, bare Z everywhere for options without an ECP."""
    import numpy as np
    from vibeqc.ecp_metadata import effective_nuclear_charges

    mol, basis, opts, _ = _h2s_lanl2dz_scf()
    np.testing.assert_allclose(effective_nuclear_charges(mol, opts), [6.0, 1.0, 1.0])
    np.testing.assert_allclose(effective_nuclear_charges(mol, None), [16.0, 1.0, 1.0])
    np.testing.assert_allclose(
        effective_nuclear_charges(mol, vq.RHFOptions()), [16.0, 1.0, 1.0]
    )


def test_ecp_dipole_is_origin_independent_and_uses_effective_charges():
    """GitLab #642 closure criterion (XML route). With bare Z the nuclear term
    carries a spurious n_core (R_S - O): the H2S/LANL2DZ dipole came out as
    0.148 D with the wrong sign and moved by exactly -10 au per bohr of
    origin shift. With Z_eff it is 1.99496 D and origin-free (Dolg & Cao,
    Chem. Rev. 112, 403 (2012), Sec. 5: cores are point charges Q = Z -
    n_core in every term of the valence-only Hamiltonian).
    """
    import numpy as np
    from vibeqc.ecp_metadata import effective_nuclear_charges
    from vibeqc.properties import center_of_mass, dipole_moment

    mol, basis, opts, res = _h2s_lanl2dz_scf()
    z_eff = effective_nuclear_charges(mol, opts)
    com = center_of_mass(mol)
    d0 = dipole_moment(res, basis, mol, nuclear_charges=z_eff)
    d1 = dipole_moment(
        res, basis, mol, origin=com + np.array([1.0, 0.0, 0.0]), nuclear_charges=z_eff
    )
    shift = np.array([d1.x - d0.x, d1.y - d0.y, d1.z - d0.z])
    assert np.linalg.norm(shift) < 1e-8  # neutral molecule: origin-free
    assert d0.total_debye == pytest.approx(1.99496, abs=1e-5)
    assert d0.z > 0.0  # S^- H^+ polarity, the physical sign

    # The pre-fix behaviour, kept as the documented counterexample: bare Z
    # shifts the dipole by -n_core per bohr of origin displacement.
    b0 = dipole_moment(res, basis, mol)
    b1 = dipole_moment(res, basis, mol, origin=com + np.array([1.0, 0.0, 0.0]))
    assert (b1.x - b0.x) == pytest.approx(-10.0, abs=1e-8)
    assert b0.total_debye == pytest.approx(0.148, abs=2e-3)


def test_ecp_population_charges_sum_to_the_molecular_charge():
    """Mulliken / Loewdin / Hirshfeld reference the valence-only nuclear
    charge (#642). Bare Z left every partition +n_core too positive on the
    ECP atom and the sums at +10 for neutral H2S/LANL2DZ."""
    import numpy as np
    from vibeqc.ecp_metadata import effective_nuclear_charges
    from vibeqc.properties import hirshfeld_charges, loewdin_charges, mulliken_charges

    mol, basis, opts, res = _h2s_lanl2dz_scf()
    z_eff = effective_nuclear_charges(mol, opts)
    q_mul = mulliken_charges(res, basis, mol, nuclear_charges=z_eff)
    q_low = loewdin_charges(res, basis, mol, nuclear_charges=z_eff)
    q_hir = hirshfeld_charges(res, basis, mol, nuclear_charges=z_eff).charges
    for q in (q_mul, q_low):
        assert abs(float(q.sum())) < 1e-10
        assert q[0] < 0.0 < q[1] == pytest.approx(q[2], abs=1e-10)
    assert abs(float(q_hir.sum())) < 1e-3  # grid quadrature
    assert q_mul[0] == pytest.approx(-0.158809, abs=1e-5)
    # bare Z: the pre-fix numbers, +n_core on the ECP atom
    assert float(mulliken_charges(res, basis, mol).sum()) == pytest.approx(10.0, abs=1e-10)
    assert (mulliken_charges(res, basis, mol) - q_mul)[0] == pytest.approx(10.0, abs=1e-12)


def test_run_job_ecp_properties_reach_out_and_population_file(tmp_path):
    """The runner passes the charges the SCF ran with (its route kwarg was
    None, so the ECP was auto-attached inside the dispatcher) to every
    property surface: the .out atomic-charges and dipole blocks and the
    population sidecar carry the Z_eff numbers (#642)."""
    import json

    out = tmp_path / "h2s642"
    r = vq.run_job(
        _h2s_lanl2dz(),
        basis="lanl2dz",
        method="rhf",
        output=out,
        write_population_file=True,
    )
    assert r.converged
    text = (tmp_path / "h2s642.out").read_text()
    # These assert that the .out carries the Z_eff numbers rather than the
    # bare-Z ones -- bare Z gives a Mulliken S charge of +9.84 and a different
    # dipole -- not that the last printed digit is stable. Both were exact
    # substring pins and both were tighter than the ~3e-6 a legitimate
    # libecpint quadrature change moves them (#269).
    assert _out_dipole_debye(text) == pytest.approx(1.99496, abs=1e-3)
    q_s_out = _out_atomic_charges(text, "S")[0]
    assert q_s_out == pytest.approx(-0.158809, abs=1e-5)
    assert q_s_out < 0.0
    pop = json.loads((tmp_path / "h2s642.population.json").read_text())
    assert pop["dipole"]["total_debye"] == pytest.approx(1.99496, abs=1e-5)
    mul = pop.get("mulliken") or pop.get("mulliken_charges")
    if isinstance(mul, dict):
        mul = mul.get("charges") or mul.get("values") or list(mul.values())[0]
    if mul is not None:
        vals = [row if isinstance(row, (int, float)) else (row.get("charge") if isinstance(row, dict) else row[-1]) for row in mul]
        assert abs(sum(float(v) for v in vals)) < 1e-6
