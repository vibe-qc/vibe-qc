"""Pin the published frozen-core convention across correlated routes (#140).

The pre-#140 defaults were deliberately preserved in the history below: MP2
and DLPNO correlated every electron while canonical CCSD froze the chemical
core.  The maintainer ruling unifies the public MP2/CCSD, restricted/
unrestricted, canonical/DLPNO routes on the published ORCA chemical-core
table.  Every route must expose the same explicit all-electron escape hatch
and disclose the resolved orbital count in its own output.
"""

from __future__ import annotations

import numpy as np
import ast
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import vibeqc.runner as runner_module
from vibeqc import (
    ECPPrimitiveBlock,
    BasisSet,
    MP2Options,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UMP2Options,
    auto_ecp_centers,
    basis_sidecar_replaces_core,
    chemical_core_orbital_count,
    molecular_options_replace_core,
    run_double_hybrid,
    run_job,
    run_mp2,
    run_rhf,
    run_rohf,
    run_roks,
    run_uhf,
    run_ump2,
)
from vibeqc._vibeqc_core import Atom, ECPCenter
from vibeqc.rohf import ROHFOptions
from vibeqc.roks import ROKSOptions

ANGSTROM_TO_BOHR = 1.8897261254578281

# Empty since 2026-09-07 (#740). Every molecular route now consumes an ECP
# reference. The CAS routes never had a separate integral path -- they are
# handed the same reference-built Hamiltonian as the determinant family -- so
# only their active-space partition and OVGF's occupied count had to move to
# the valence count.
_ECP_UNSUPPORTED_MOLECULAR_ROUTES: tuple[str, ...] = ()

# Previously refused, now expected to run on an ECP reference.
_ECP_NEWLY_SUPPORTED_ROUTES = ("casci", "nevpt2", "caspt2", "mrci")


def _h2o():
    """Closed shell; O has one chemical-core (1s) orbital."""
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.117 * ANGSTROM_TO_BOHR]),
            Atom(1, [0.0, 0.757 * ANGSTROM_TO_BOHR, -0.467 * ANGSTROM_TO_BOHR]),
            Atom(1, [0.0, -0.757 * ANGSTROM_TO_BOHR, -0.467 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )


def _oh():
    """Open shell (doublet); routes to the U- variants."""
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.9697 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )


def _h2s_lanl2dz():
    """Neutral singlet H2S; LANL2DZ replaces ten sulfur core electrons."""
    return Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.815, 1.425]),
            Atom(1, [0.0, -1.815, 1.425]),
        ],
        charge=0,
        multiplicity=1,
    )


def _si_lanl2dz():
    """Triplet silicon; LANL2DZ replaces ten core electrons."""
    return Molecule([Atom(14, [0.0, 0.0, 0.0])], multiplicity=3)


def test_ecp_preflight_route_inventory_is_explicit_and_complete():
    """Deleting a production route cannot silently delete its guard test."""
    assert set(_ECP_UNSUPPORTED_MOLECULAR_ROUTES) == set(
        runner_module._MOLECULAR_ECP_UNSUPPORTED_METHODS
    )
    assert not runner_module._MOLECULAR_ECP_OPERATOR_UNSUPPORTED_METHODS
    assert not runner_module._MOLECULAR_ECP_REPLACED_CORE_UNSUPPORTED_METHODS
    assert {"rohf", "roks"}.isdisjoint(
        runner_module._FROZEN_CORE_ROUTE_METHODS
    )


@pytest.mark.parametrize("method", _ECP_NEWLY_SUPPORTED_ROUTES)
def test_cas_family_runs_on_an_ecp_reference_at_the_valence_count(
    method, tmp_path
):
    """#740: refused before, because the CAS partition counted physical
    electrons. H2S/LANL2DZ has 18 physical and 8 valence; CAS(4e,4o) leaves 2
    doubly-occupied core orbitals at the valence count and an impossible 7 at
    the physical one."""
    result = run_job(
        _h2s_lanl2dz(), basis="lanl2dz", method=method, active_space=(4, 4),
        output=tmp_path / f"ecp-{method}", progress=False,
    )
    assert f"{method}(4e,4o)" in result.method
    # Valence H2S under the LANL2DZ ECP; all-electron sits near -399 Ha.
    assert -11.5 < float(result.energy) < -10.0


def test_ovgf_partitions_the_valence_count_on_an_ecp_reference(tmp_path):
    """#740: OVGF took its occupied count from Molecule.n_electrons(), which
    on an ECP reference indexes orbitals the SCF never filled.

    H2S in LANL2DZ has 18 physical electrons and 10 replaced, so 8 valence
    electrons and four occupied orbitals: the HOMO is orbital 3.  Under the
    physical count it would be orbital 8, which on this reference is a
    *virtual* orbital -- the old partition therefore reported an electron
    affinity where an ionization potential belongs, and the previous bound
    here (``0 <= q.orbital < 20``) was far too loose to notice.  Anchored
    against the experimental first IP, 10.4682 eV (NIST), the way
    ``tests/test_runner_ovgf.py`` anchors the all-electron case.
    """
    EV = 27.211386245988
    mol = _h2s_lanl2dz()
    result = run_job(
        mol, basis="lanl2dz", method="ovgf",
        output=tmp_path / "ecp-ovgf", progress=False,
    )
    assert result.ovgf, "no quasiparticle orbitals returned"

    n_physical = sum(a.Z for a in mol.atoms)
    assert n_physical == 18
    assert int(result.ecp_total_ncore) == 10
    n_occ = (n_physical - int(result.ecp_total_ncore)) // 2
    assert n_occ == 4

    for q in result.ovgf:
        assert 0 <= q.orbital < 20
        assert 0.0 < float(q.pole_strength) <= 1.0
    assert any(float(q.eps_qp) != float(q.eps_scf) for q in result.ovgf)

    # The bug-specific check: the HOMO sits at the *valence* boundary.
    homo = next((q for q in result.ovgf if q.orbital == n_occ - 1), None)
    assert homo is not None, (
        f"no quasiparticle at the valence HOMO index {n_occ - 1}; orbitals "
        f"reported: {sorted(q.orbital for q in result.ovgf)}"
    )
    assert homo.converged

    koopmans_ip = -float(homo.eps_scf) * EV
    ovgf_ip = -float(homo.eps_qp) * EV
    # A physical outer-valence IP, not a virtual orbital's negative one.
    assert koopmans_ip == pytest.approx(10.5, abs=1.0)
    assert ovgf_ip < koopmans_ip          # the self-energy corrects downward
    assert 8.0 < ovgf_ip                  # and does not run away
    assert float(homo.pole_strength) > 0.80

    # LANL2DZ is a double-zeta valence basis, so its virtual space is too
    # small for OVGF to land on experiment: measured 10.5646 eV Koopmans ->
    # 9.7958 eV OVGF against 10.4682 eV experimental, i.e. it over-corrects
    # by ~0.67 eV. That is a basis limitation, not a partition error, so it
    # is recorded rather than asserted as agreement.


def test_manual_ecp_options_on_all_electron_basis_are_consumed(tmp_path):
    """A manual ECP without a basis sidecar reaches the CAS Hamiltonian."""
    # A zero-core model potential: it removes no electrons, so any energy
    # change is the operator itself reaching the reference Hamiltonian.
    block = ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.0, 1.5]
    block.coefficients = [0.0, -1.5]
    block.ams = [1, 0]
    block.ns = [2, 2]
    options = RHFOptions()
    options.ecp_primitive_blocks = [block]
    options.ecp_primitive_centers = [list(_h2o().atoms[0].xyz)]
    options.ecp_effective_charges = [8.0, 1.0, 1.0]
    options.ecp_total_ncore = 0

    plain = run_job(
        _h2o(), basis="sto-3g", method="casci", active_space=(2, 2),
        output=tmp_path / "plain", progress=False,
    )
    manual = run_job(
        _h2o(), basis="sto-3g", method="casci", active_space=(2, 2),
        rhf_options=options, output=tmp_path / "manual-ecp", progress=False,
    )
    # Consumed, not silently dropped (#740). run_job builds the CAS reference
    # internally, so an ECP supplied only through rhf_options reaches it via
    # get_hf_orbital_provider(scf_options=...) and nowhere else; before that
    # threading existed these two energies were byte-identical.
    assert abs(float(manual.energy) - float(plain.energy)) > 1e-6


def test_manual_uhf_ecp_options_are_consumed(tmp_path):
    """Open-shell dispatch threads the selected UHF ECP into the CAS run."""
    block = ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.0, 1.5]
    block.coefficients = [0.0, -1.5]
    block.ams = [1, 0]
    block.ns = [2, 2]
    options = UHFOptions()
    options.ecp_primitive_blocks = [block]
    options.ecp_primitive_centers = [[0.0, 0.0, 0.0]]
    options.ecp_effective_charges = [14.0]
    options.ecp_total_ncore = 0

    plain = run_job(
        _si_lanl2dz(), basis="sto-3g", method="casci", active_space=(2, 2),
        output=tmp_path / "plain-uhf", progress=False,
    )
    manual = run_job(
        _si_lanl2dz(), basis="sto-3g", method="casci", active_space=(2, 2),
        uhf_options=options, output=tmp_path / "manual-uhf-ecp", progress=False,
    )
    # Open-shell dispatch must thread the selected UHF options the same way.
    assert abs(float(manual.energy) - float(plain.energy)) > 1e-6


def test_shared_basis_sidecar_core_detector_matches_molecular_elements():
    """The shared guard distinguishes matching and irrelevant sidecars."""
    assert basis_sidecar_replaces_core(_h2s_lanl2dz(), "lanl2dz")
    assert not basis_sidecar_replaces_core(_h2o(), "lanl2dz")
    assert not basis_sidecar_replaces_core(_h2s_lanl2dz(), "def2-svp")


@pytest.mark.parametrize(
    ("route", "kwargs"),
    [
        pytest.param(run_rohf, {}, id="rohf"),
        pytest.param(run_roks, {"functional": "pbe"}, id="roks"),
    ],
)
def test_direct_restricted_open_shell_routes_attach_ecp_sidecars(route, kwargs):
    """Pure-Python ROHF/ROKS solve the sidecar's ECP Hamiltonian (2026-09)."""
    molecule = _h2s_lanl2dz()
    basis = BasisSet(molecule, "lanl2dz")
    result = route(molecule, basis, **kwargs)
    assert result.converged
    assert result.ecp_operator_applied and result.ecp_total_ncore == 10
    assert result.n_alpha + result.n_beta == 8
    assert list(result.ecp_effective_charges) == pytest.approx([6.0, 1.0, 1.0])
    assert result.ecp_xml_library == ""


@pytest.mark.parametrize(
    ("route", "options_type", "kwargs"),
    [
        pytest.param(run_rohf, ROHFOptions, {}, id="rohf"),
        pytest.param(
            run_roks,
            ROKSOptions,
            {"functional": "pbe"},
            id="roks",
        ),
    ],
)
def test_direct_restricted_open_shell_routes_honour_manual_xml_ecp(
    route,
    options_type,
    kwargs,
):
    """A manual XML centre on an all-electron basis is applied, not dropped."""
    molecule = _h2s_lanl2dz()
    basis = BasisSet(molecule, "def2-svp")
    options = options_type()
    options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "lanl2dz"
    result = route(molecule, basis, options, **kwargs)
    assert result.ecp_operator_applied and result.ecp_total_ncore == 10
    assert result.ecp_xml_library == "lanl2dz"
    assert result.n_alpha + result.n_beta == 8


@pytest.mark.parametrize("form", ["count", "charges"])
@pytest.mark.parametrize(
    ("route", "options_type", "kwargs"),
    [
        pytest.param(run_rohf, ROHFOptions, {}, id="rohf"),
        pytest.param(
            run_roks,
            ROKSOptions,
            {"functional": "pbe"},
            id="roks",
        ),
    ],
)
def test_direct_restricted_open_shell_routes_refuse_orphan_ecp_metadata(
    route,
    options_type,
    kwargs,
    form,
):
    """A core count or charge vector without an operator cannot run bare-Z."""
    molecule = _h2s_lanl2dz()
    basis = BasisSet(molecule, "def2-svp")
    options = options_type()
    if form == "count":
        options.ecp_total_ncore = 10
    else:
        options.ecp_effective_charges = [6.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="no ECP operator"):
        route(molecule, basis, options, **kwargs)


@pytest.mark.parametrize(
    ("route", "options_type", "kwargs"),
    [
        pytest.param(run_rohf, ROHFOptions, {}, id="rohf"),
        pytest.param(
            run_roks,
            ROKSOptions,
            {"functional": "pbe"},
            id="roks",
        ),
    ],
)
def _zero_core_model_options(options_type, molecule):
    """A one-primitive s-channel model potential on sulfur that removes no
    electrons: the operator changes the Hamiltonian, the count does not."""
    from vibeqc import ECPPrimitiveBlock

    block = ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.0, 1.25]
    block.coefficients = [0.0, -2.0]
    block.ams = [1, 0]
    block.ns = [2, 2]
    options = options_type()
    options.ecp_primitive_blocks = [block]
    options.ecp_primitive_centers = [list(molecule.atoms[0].xyz)]
    options.ecp_effective_charges = [float(a.Z) for a in molecule.atoms]
    options.ecp_total_ncore = 0
    return options


@pytest.mark.parametrize(
    ("route", "options_type", "kwargs"),
    [
        pytest.param(run_rohf, ROHFOptions, {}, id="rohf"),
        pytest.param(
            run_roks,
            ROKSOptions,
            {"functional": "pbe"},
            id="roks",
        ),
    ],
)
def test_direct_restricted_open_shell_routes_apply_zero_core_model(
    route,
    options_type,
    kwargs,
):
    """A zero-core model potential is applied, and recorded as applied."""
    molecule = _h2s_lanl2dz()
    basis = BasisSet(molecule, "def2-svp")
    plain = route(molecule, basis, options_type(), **kwargs)
    modelled = route(
        molecule, basis, _zero_core_model_options(options_type, molecule), **kwargs
    )
    assert not plain.ecp_operator_applied
    assert modelled.ecp_operator_applied and modelled.ecp_total_ncore == 0
    assert modelled.n_alpha + modelled.n_beta == 18
    assert abs(modelled.energy - plain.energy) > 1e-6


@pytest.mark.parametrize(
    ("method", "option_keyword", "options_type"),
    [
        pytest.param("rohf", "rohf_options", ROHFOptions, id="rohf"),
        pytest.param("roks", "roks_options", ROKSOptions, id="roks"),
    ],
)
def test_runner_restricted_open_shell_applies_zero_core_model(
    method,
    option_keyword,
    options_type,
    tmp_path,
):
    """run_job hands the ROHF/ROKS drivers the same operator it would hand
    the native drivers."""
    molecule = _h2s_lanl2dz()
    options = _zero_core_model_options(options_type, molecule)
    result = run_job(
        molecule,
        basis="def2-svp",
        method=method,
        output=tmp_path / f"{method}-zero-core",
        progress=False,
        **{option_keyword: options},
    )
    assert result.ecp_operator_applied and result.ecp_total_ncore == 0


def test_direct_double_hybrid_runs_on_ecp_reference(tmp_path):
    """Standalone double hybrids consume the reference's valence count.
    LANL2DZ has no fitting aux, so the MP2 half runs conventional; the
    hybrid SCF runs conventional too when no aux is registered."""
    result = run_double_hybrid(
        _h2s_lanl2dz(),
        "lanl2dz",
        "b2plyp",
        density_fit=False,
        density_fit_mp2=True,
        aux_basis_mp2="def2-svp-rifit",
        output=tmp_path / "ecp-double-hybrid",
    )
    assert result.mp2.e_correlation < 0.0
    assert result.rks.ecp_operator_applied and result.rks.ecp_total_ncore == 10
    assert (tmp_path / "ecp-double-hybrid.out").is_file()


def test_direct_double_hybrid_honours_manual_xml_ecp(tmp_path):
    """Explicit SCF ECP options reach the double-hybrid SCF and its MP2."""
    options = RKSOptions()
    options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "lanl2dz"
    result = run_double_hybrid(
        _h2s_lanl2dz(),
        "def2-svp",
        "b2plyp",
        rks_options=options,
        density_fit=False,
        density_fit_mp2=True,
        aux_basis_mp2="def2-svp-rifit",
        output=tmp_path / "manual-ecp-double-hybrid",
    )
    assert result.rks.ecp_operator_applied and result.rks.ecp_total_ncore == 10
    assert result.mp2.e_correlation < 0.0


def test_ecp_preflight_distinguishes_operator_from_replaced_core_count():
    """MP2/CC reject replaced cores, not safe zero-core model potentials."""
    h2 = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    assert not runner_module._basis_sidecar_replaces_molecular_core(
        h2, "lanl2dz"
    )
    zero_core_model = SimpleNamespace(
        ecp_primitive_blocks=[object()],
        ecp_total_ncore=0,
    )
    assert not molecular_options_replace_core(zero_core_model)
    assert runner_module.molecular_options_request_ecp_operator(
        zero_core_model
    )
    assert not runner_module._options_activate_molecular_ecp(zero_core_model)
    runner_module._refuse_molecular_ecp_correlated_route(
        "mp2",
        "rhf",
        _h2s_lanl2dz(),
        "def2-svp",
        rhf_options=zero_core_model,
    )


def test_cisd_hamiltonian_carries_the_reference_ecp_operator():
    """The determinant Hamiltonian is built from the SCF reference, so a
    reference whose Hcore carries a (zero-core) ECP operator produces a
    different one-electron term -- the solver does not rebuild bare-Z T+V.
    Tested at the solver layer because run_job builds the CI reference SCF
    internally and does not thread a caller's manual ECP options into it."""
    from vibeqc import ECPPrimitiveBlock, RHFOptions, run_rhf
    from vibeqc.solvers import build_hamiltonian_mo

    molecule = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])], 0, 1)
    basis = BasisSet(molecule, "sto-3g")

    plain = run_rhf(molecule, basis, RHFOptions())

    block = ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.0, 1.5]
    block.coefficients = [0.0, -1.5]
    block.ams = [1, 0]
    block.ns = [2, 2]
    opts = RHFOptions()
    opts.ecp_primitive_blocks = [block]
    opts.ecp_primitive_centers = [[0.0, 0.0, 0.0]]
    opts.ecp_effective_charges = [3.0, 1.0]
    opts.ecp_total_ncore = 0
    modelled = run_rhf(molecule, basis, opts)
    assert modelled.ecp_operator_applied and modelled.ecp_total_ncore == 0
    assert abs(modelled.energy - plain.energy) > 1e-6

    C = np.asarray(modelled.mo_coeffs)
    h_plain = build_hamiltonian_mo(molecule, basis, C, reference=plain)
    h_ecp = build_hamiltonian_mo(molecule, basis, C, reference=modelled)
    assert h_plain.nelec == h_ecp.nelec == molecule.n_electrons()
    assert not np.allclose(np.asarray(h_plain.h1e), np.asarray(h_ecp.h1e))


def test_negative_ecp_provenance_fails_closed():
    from vibeqc.correlation_conventions import effective_electron_count

    with pytest.raises(ValueError, match="negative ecp_total_ncore"):
        effective_electron_count(_h2o(), SimpleNamespace(ecp_total_ncore=-1))


def test_effective_electron_count_uses_scf_ecp_provenance():
    """Artifact consumers see variational, not physical, electrons."""
    from vibeqc.correlation_conventions import effective_electron_count

    molecule = _h2o()
    assert molecule.n_electrons() == 10
    assert effective_electron_count(
        molecule, SimpleNamespace(ecp_total_ncore=2)
    ) == 8
    assert effective_electron_count(molecule, SimpleNamespace()) == 10


@pytest.mark.parametrize(
    ("ncore", "message"),
    [
        (-1, "negative ecp_total_ncore"),
        (11, "exceeds the molecule's physical electron count"),
        ("invalid", "invalid SCF ecp_total_ncore provenance"),
    ],
)
def test_effective_electron_count_rejects_invalid_provenance(ncore, message):
    from vibeqc.correlation_conventions import effective_electron_count

    with pytest.raises(ValueError, match=message):
        effective_electron_count(
            _h2o(), SimpleNamespace(ecp_total_ncore=ncore)
        )


@pytest.mark.parametrize(
    ("ncore", "expected"),
    [(0, False), (10, True), ("invalid", True)],
)
def test_ecp_detection_honours_result_provenance(ncore, expected):
    """Automatic sidecars remain visible after their option copy is gone."""
    detected = runner_module._detect_uses_ecp(
        None,
        None,
        None,
        None,
        result=SimpleNamespace(ecp_total_ncore=ncore),
    )
    assert detected is expected


def test_trexio_dispatch_uses_effective_ecp_result_without_options(
    tmp_path,
    monkeypatch,
):
    """Artifact refusals see auto-ECP provenance even without caller options."""
    from vibeqc.output.writer import OutputWriter

    original_dispatch = OutputWriter.dispatch_role
    observed_uses_ecp = []

    class TrexioDispatchObserved(RuntimeError):
        pass

    def capture_trexio_dispatch(self, role, *args, **kwargs):
        if kwargs.get("only_format") == "trexio":
            observed_uses_ecp.append(kwargs.get("uses_ecp"))
            raise TrexioDispatchObserved
        return original_dispatch(self, role, *args, **kwargs)

    monkeypatch.setattr(OutputWriter, "dispatch_role", capture_trexio_dispatch)
    with pytest.raises(TrexioDispatchObserved):
        run_job(
            _h2s_lanl2dz(),
            basis="lanl2dz",
            method="rhf",
            output=tmp_path / "implicit-ecp-trexio",
            trexio=True,
            verbose=False,
            progress=False,
        )

    assert observed_uses_ecp == [True]


def test_native_mp2_and_ccsd_partition_the_valence_count():
    """The native kernels use n_electrons - ecp_total_ncore; the element
    table default (-1) is refused natively on an ECP reference because it
    cannot subtract the removed cores, and the Python wrappers resolve it."""
    molecule = _h2s_lanl2dz()
    basis = BasisSet(molecule, "lanl2dz")
    rhf_options = RHFOptions()
    rhf_options.max_iter = 100
    reference = run_rhf(molecule, basis, rhf_options)
    assert reference.converged
    assert reference.ecp_operator_applied
    assert reference.ecp_total_ncore == 10

    options = MP2Options()
    options.n_frozen_core = None
    options.density_fit = False
    with pytest.raises(
        (RuntimeError, ValueError),
        match="cannot be applied natively to an ECP reference",
    ):
        run_mp2(molecule, basis, reference, options)
    options.n_frozen_core = 0
    assert run_mp2(molecule, basis, reference, options).e_correlation < 0.0

    from vibeqc import _vibeqc_core as core

    options = core.CCSDOptions()
    options.density_fit = False
    options.compute_triples = False
    options.n_frozen_core = 0
    assert core.run_ccsd(molecule, basis, reference, options).e_ccsd_correlation < 0.0
    options.n_frozen_core = 4
    with pytest.raises(RuntimeError, match="n_frozen_core >= n_occ"):
        core.run_ccsd(molecule, basis, reference, options)
    with pytest.raises(
        (RuntimeError, ValueError),
        match="cannot be applied natively to an ECP reference",
    ):
        core.run_ccsd_from_mos(
            molecule,
            basis,
            reference.mo_coeffs,
            reference.fock,
            float(reference.energy),
            10,
            core.CCSDOptions(),
        )


def test_native_unrestricted_post_hf_partitions_the_valence_count():
    """The UHF result carries the same authoritative ECP provenance."""
    from vibeqc import _vibeqc_core as core

    molecule = _si_lanl2dz()
    basis = BasisSet(molecule, "lanl2dz")
    uhf_options = UHFOptions()
    uhf_options.ecp_centers, uhf_options.ecp_library = auto_ecp_centers(
        molecule, "lanl2dz"
    )
    uhf_options.max_iter = 200
    uhf_options.stability_check = False
    reference = run_uhf(molecule, basis, uhf_options)
    assert reference.converged
    assert reference.ecp_operator_applied
    assert reference.ecp_total_ncore == 10

    options = UMP2Options()
    options.n_frozen_core = None
    options.density_fit = False
    with pytest.raises(
        (RuntimeError, ValueError),
        match="cannot be applied natively to an ECP reference",
    ):
        run_ump2(molecule, basis, reference, options)
    options.n_frozen_core = 0
    assert run_ump2(molecule, basis, reference, options).e_correlation <= 0.0

    options = core.CCSDOptions()
    options.density_fit = False
    options.compute_triples = False
    options.n_frozen_core = 0
    assert core.run_uccsd(molecule, basis, reference, options).e_ccsd_correlation <= 0.0


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
@pytest.mark.parametrize("mismatch", ["reduced-with-zero", "bare-with-positive"])
def test_native_scf_rejects_inconsistent_primitive_ecp_provenance(
    route_name,
    options_name,
    molecule_factory,
    mismatch,
):
    """Manual Z_eff and replaced-core provenance form one native contract."""
    from vibeqc import _vibeqc_core as core
    from vibeqc import inline_ecp_data_for

    molecule = molecule_factory()
    basis = BasisSet(molecule, "lanl2dz")
    blocks, centers, effective_charges, total_ncore = inline_ecp_data_for(
        molecule, "lanl2dz"
    )
    assert blocks and total_ncore > 0

    options = getattr(core, options_name)()
    options.ecp_primitive_blocks = blocks
    options.ecp_primitive_centers = centers
    if mismatch == "reduced-with-zero":
        options.ecp_effective_charges = effective_charges
        options.ecp_total_ncore = 0
    else:
        options.ecp_effective_charges = [
            float(atom.Z) for atom in molecule.atoms
        ]
        options.ecp_total_ncore = int(total_ncore)

    # Upstream moved this check into the initial guess and reworded it
    # (2026-09-07); accept either phrasing so the test pins the guard
    # rather than the sentence.
    with pytest.raises(
        ValueError,
        match="inconsistent with total_ncore|core count disagrees",
    ):
        getattr(core, route_name)(molecule, basis, options)


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
@pytest.mark.parametrize(
    ("orphan_field", "message"),
    [
        pytest.param("ecp_library", "requires non-empty ecp_centers", id="library"),
        pytest.param(
            "ecp_primitive_centers",
            "requires non-empty ecp_primitive_blocks",
            id="primitive-centers",
        ),
        pytest.param(
            "ecp_effective_charges",
            "requires non-empty ecp_primitive_blocks",
            id="effective-charges",
        ),
        pytest.param(
            "ecp_total_ncore",
            "requires non-empty ecp_primitive_blocks",
            id="core-count",
        ),
        pytest.param(
            "negative_ecp_total_ncore",
            "ecp_total_ncore must be non-negative",
            id="negative-core-count",
        ),
    ],
)
def test_native_scf_rejects_orphan_ecp_metadata_before_dispatch(
    route_name,
    options_name,
    molecule_factory,
    orphan_field,
    message,
):
    """Partial manual ECP options cannot silently select all-electron SCF."""
    from vibeqc import _vibeqc_core as core

    molecule = molecule_factory()
    basis = BasisSet(molecule, "lanl2dz")
    options = getattr(core, options_name)()
    values = {
        "ecp_library": "lanl2dz",
        "ecp_primitive_centers": [[0.0, 0.0, 0.0]],
        "ecp_effective_charges": [float(atom.Z) for atom in molecule.atoms],
        "ecp_total_ncore": 10,
    }
    if orphan_field == "negative_ecp_total_ncore":
        options.ecp_total_ncore = -1
    else:
        setattr(options, orphan_field, values[orphan_field])

    with pytest.raises(ValueError, match=message):
        getattr(core, route_name)(molecule, basis, options)


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
def test_native_scf_rejects_mixed_xml_and_primitive_ecp_inputs(
    route_name,
    options_name,
    molecule_factory,
):
    """Two independently configured ECP operators are ambiguous."""
    from vibeqc import _vibeqc_core as core
    from vibeqc import inline_ecp_data_for

    molecule = molecule_factory()
    basis = BasisSet(molecule, "lanl2dz")
    blocks, centers, effective_charges, total_ncore = inline_ecp_data_for(
        molecule, "lanl2dz"
    )
    heavy_atom = max(molecule.atoms, key=lambda atom: int(atom.Z))

    options = getattr(core, options_name)()
    options.ecp_centers = [
        ECPCenter(Z=int(heavy_atom.Z), xyz=list(heavy_atom.xyz))
    ]
    options.ecp_library = "lanl2dz"
    options.ecp_primitive_blocks = blocks
    options.ecp_primitive_centers = centers
    options.ecp_effective_charges = effective_charges
    options.ecp_total_ncore = int(total_ncore)

    with pytest.raises(ValueError, match="mutually exclusive"):
        getattr(core, route_name)(molecule, basis, options)


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
def test_native_scf_marks_zero_core_model_operator_as_applied(
    route_name,
    options_name,
    molecule_factory,
):
    """Operator provenance remains true when no electrons are replaced."""
    from vibeqc import _vibeqc_core as core

    # A genuine zero-core model potential: an ALL-ELECTRON basis plus an
    # operator that replaces nothing, so the electron count and the basis
    # agree. The earlier fixture paired LANL2DZ (valence-only, ten core
    # electrons gone) with ecp_total_ncore=0 and the bare nuclear charges,
    # i.e. eighteen electrons in a basis describing eight. It only survived
    # because max_iter=0 meant no SCF ran; upstream's guess engine now
    # refuses it up front, correctly.
    molecule = molecule_factory()
    basis = BasisSet(molecule, "sto-3g")
    block = ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [1.0, 1.5]
    block.coefficients = [0.0, -1.5]
    block.ams = [1, 0]
    block.ns = [2, 2]
    blocks = [block]
    centers = [list(molecule.atoms[0].xyz)]
    options = getattr(core, options_name)()
    options.max_iter = 0
    options.ecp_primitive_blocks = blocks
    options.ecp_primitive_centers = centers
    options.ecp_effective_charges = [
        float(atom.Z) for atom in molecule.atoms
    ]
    options.ecp_total_ncore = 0

    result = getattr(core, route_name)(molecule, basis, options)
    assert result.ecp_operator_applied
    assert result.ecp_provenance_verified
    assert list(result.ecp_xml_centers) == []
    assert result.ecp_xml_library == ""
    result_blocks = list(result.ecp_primitive_blocks)
    assert len(result_blocks) == len(blocks)
    for actual, expected in zip(result_blocks, blocks):
        assert actual.n_primitive == expected.n_primitive
        assert list(actual.exponents) == pytest.approx(list(expected.exponents))
        assert list(actual.coefficients) == pytest.approx(
            list(expected.coefficients)
        )
        assert list(actual.ams) == list(expected.ams)
        assert list(actual.ns) == list(expected.ns)
    result_centers = list(result.ecp_primitive_centers)
    assert len(result_centers) == len(centers)
    for actual, expected in zip(result_centers, centers):
        assert list(actual) == pytest.approx(list(expected))
    assert list(result.ecp_effective_charges) == pytest.approx(
        [float(atom.Z) for atom in molecule.atoms]
    )
    assert result.ecp_total_ncore == 0
    if route_name == "run_rks":
        adapted = core.rhf_result_from_rks(result)
        assert adapted.ecp_operator_applied
        assert adapted.ecp_provenance_verified
        assert list(adapted.ecp_xml_centers) == []
        assert adapted.ecp_xml_library == ""
        assert len(adapted.ecp_primitive_blocks) == len(blocks)
        adapted_centers = list(adapted.ecp_primitive_centers)
        assert len(adapted_centers) == len(centers)
        for actual, expected in zip(adapted_centers, centers):
            assert list(actual) == pytest.approx(list(expected))
        assert list(adapted.ecp_effective_charges) == pytest.approx(
            [float(atom.Z) for atom in molecule.atoms]
        )
        assert adapted.ecp_total_ncore == 0


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
def test_native_xml_scf_result_carries_exact_ecp_provenance(
    route_name,
    options_name,
    molecule_factory,
):
    """Every molecular SCF result records its operative XML Hamiltonian."""
    from vibeqc import _vibeqc_core as core

    molecule = molecule_factory()
    heavy_atom = max(molecule.atoms, key=lambda atom: int(atom.Z))
    basis = BasisSet(molecule, "lanl2dz")
    center = ECPCenter(Z=int(heavy_atom.Z), xyz=list(heavy_atom.xyz))
    options = getattr(core, options_name)()
    options.max_iter = 0
    options.ecp_centers = [center]
    options.ecp_library = "lanl2dz"

    result = getattr(core, route_name)(molecule, basis, options)
    assert result.ecp_operator_applied
    assert result.ecp_provenance_verified
    [result_center] = list(result.ecp_xml_centers)
    assert int(result_center.Z) == int(center.Z)
    assert list(result_center.xyz) == pytest.approx(list(center.xyz))
    assert result.ecp_xml_library == "lanl2dz"
    assert list(result.ecp_primitive_blocks) == []
    assert list(result.ecp_primitive_centers) == []
    assert list(result.ecp_effective_charges) == []
    assert result.ecp_total_ncore == 10
    if route_name == "run_rks":
        adapted = core.rhf_result_from_rks(result)
        assert adapted.ecp_operator_applied
        assert adapted.ecp_provenance_verified
        [adapted_center] = list(adapted.ecp_xml_centers)
        assert int(adapted_center.Z) == int(center.Z)
        assert list(adapted_center.xyz) == pytest.approx(list(center.xyz))
        assert adapted.ecp_xml_library == "lanl2dz"
        assert list(adapted.ecp_primitive_blocks) == []
        assert list(adapted.ecp_primitive_centers) == []
        assert list(adapted.ecp_effective_charges) == []
        assert adapted.ecp_total_ncore == 10


@pytest.mark.parametrize(
    ("route_name", "options_name"),
    [
        pytest.param("run_rhf", "RHFOptions", id="rhf"),
        pytest.param("run_rks", "RKSOptions", id="rks"),
        pytest.param("run_uhf", "UHFOptions", id="uhf"),
        pytest.param("run_uks", "UKSOptions", id="uks"),
    ],
)
def test_native_all_electron_scf_result_is_verified(route_name, options_name):
    """High-level SCF certifies a known all-electron Hamiltonian."""
    from vibeqc import _vibeqc_core as core

    molecule = _h2o()
    basis = BasisSet(molecule, "sto-3g")
    options = getattr(core, options_name)()
    options.max_iter = 0

    result = getattr(core, route_name)(molecule, basis, options)
    assert result.ecp_provenance_verified
    assert not result.ecp_operator_applied
    assert list(result.ecp_xml_centers) == []
    assert result.ecp_xml_library == ""
    assert list(result.ecp_primitive_blocks) == []
    assert list(result.ecp_primitive_centers) == []
    assert list(result.ecp_effective_charges) == []
    assert result.ecp_total_ncore == 0


@pytest.mark.parametrize(
    ("route_name", "options_name", "molecule_factory"),
    [
        pytest.param("run_rhf", "RHFOptions", _h2s_lanl2dz, id="rhf"),
        pytest.param("run_rks", "RKSOptions", _h2s_lanl2dz, id="rks"),
        pytest.param("run_uhf", "UHFOptions", _si_lanl2dz, id="uhf"),
        pytest.param("run_uks", "UKSOptions", _si_lanl2dz, id="uks"),
    ],
)
def test_native_scf_rejects_unmatched_manual_xml_ecp_center(
    route_name,
    options_name,
    molecule_factory,
):
    """A valid center cannot mask a stale/off-geometry XML center."""
    from vibeqc import _vibeqc_core as core

    molecule = molecule_factory()
    heavy_atom = max(molecule.atoms, key=lambda atom: int(atom.Z))
    basis = BasisSet(molecule, "lanl2dz")
    options = getattr(core, options_name)()
    options.ecp_centers = [
        ECPCenter(Z=int(heavy_atom.Z), xyz=list(heavy_atom.xyz)),
        ECPCenter(
            Z=int(heavy_atom.Z),
            xyz=[
                float(heavy_atom.xyz[0]) + 0.5,
                float(heavy_atom.xyz[1]),
                float(heavy_atom.xyz[2]),
            ],
        ),
    ]
    options.ecp_library = "lanl2dz"

    with pytest.raises(
        ValueError,
        match="must match exactly one molecule atom|does not match an atom",
    ):
        getattr(core, route_name)(molecule, basis, options)


def test_cpcm_rejects_unmatched_manual_xml_ecp_center():
    """CPCM validates every center rather than trusting an aggregate match."""
    from vibeqc import SolventModel, run_cpcm_scf

    molecule = _h2s_lanl2dz()
    sulfur = max(molecule.atoms, key=lambda atom: int(atom.Z))
    basis = BasisSet(molecule, "def2-svp")
    options = RHFOptions()
    options.ecp_centers = [
        ECPCenter(Z=int(sulfur.Z), xyz=list(sulfur.xyz)),
        ECPCenter(
            Z=int(sulfur.Z),
            xyz=[
                float(sulfur.xyz[0]) + 0.5,
                float(sulfur.xyz[1]),
                float(sulfur.xyz[2]),
            ],
        ),
    ]
    options.ecp_library = "lanl2dz"

    with pytest.raises(
        ValueError,
        match="must match exactly one molecule atom|does not match an atom",
    ):
        run_cpcm_scf(
            molecule,
            basis,
            method="rhf",
            options=options,
            solvent=SolventModel(epsilon=2.0),
        )


def test_explicit_mo_uccsd_requires_and_checks_ecp_provenance():
    """Raw arrays cannot silently stand in for missing SCF provenance."""
    from vibeqc.cc import run_uccsd_from_mos

    args = (_h2o(), None, None, None, None, None, 0.0)
    with pytest.raises(TypeError, match="ecp_total_ncore"):
        run_uccsd_from_mos(*args)
    with pytest.raises(ValueError, match="explicit n_frozen_core"):
        run_uccsd_from_mos(*args, ecp_total_ncore=10)


# Historical defaults on main @ 672e7974d were MP2=0, DLPNO-MP2=0, CCSD=1,
# DLPNO-CCSD=0 for these oxygen-containing witnesses.  Every moved 0 -> 1 row
# below is the intended #140 convention change, not a silent energy rebaseline.
FROZEN_CORE_CONVENTION = [
    ("mp2", 1, "Frozen core orbitals = 1"),
    ("dlpno-mp2", 1, "Frozen core orbitals = 1"),
    ("ccsd", 1, "Frozen core orbitals = 1"),
    ("dlpno-ccsd", 1, "Frozen core orbitals = 1"),
]

_DONE_EVENT = {
    ("mp2", "closed"): ("mp2_done", "n_frozen_core"),
    ("mp2", "open"): ("mp2_done", "n_frozen_core"),
    ("dlpno-mp2", "closed"): ("dlpno_mp2_done", "n_frozen"),
    ("dlpno-mp2", "open"): ("dlpno_ump2_done", "n_frozen"),
    ("ccsd", "closed"): ("ccsd_converged", "n_frozen_core"),
    ("ccsd", "open"): ("ccsd_converged", "n_frozen_core"),
    ("dlpno-ccsd", "closed"): ("dlpno_ccsd_done", "n_frozen"),
    ("dlpno-ccsd", "open"): ("dlpno_uccsd_done", "n_frozen"),
}

_PUBLISHED_DEFAULT = "orca-6.1-table-2.69-count-only-default"


@pytest.mark.parametrize("method,n_frozen,phrase", FROZEN_CORE_CONVENTION)
@pytest.mark.parametrize("system", ["closed", "open"])
def test_route_frozen_core_is_disclosed_and_pinned(
    method, n_frozen, phrase, system, tmp_path, monkeypatch
):
    """Every correlated route states its frozen-core treatment in its .out.

    Disclosure is the half of #140 that can be pinned without choosing a
    convention: whatever a route does, the run must say so, so that a number
    is interpretable from its own artifact rather than from the source.
    """
    monkeypatch.chdir(tmp_path)
    mol = _h2o() if system == "closed" else _oh()
    run_job(
        mol,
        basis="def2-svp",
        method=method,
        output="fc",
        structured_log=True,
    )
    out = (tmp_path / "fc.out").read_text()
    references = (tmp_path / "fc.references").read_text()

    assert phrase in out, (
        f"{method} ({system} shell) does not disclose its frozen-core "
        f"treatment in the .out; expected the phrase {phrase!r}"
    )
    assert "published count-only default" in out
    assert "ORCA Manual, Release 6.1.1: Frozen Core Options" in references

    manifest = tomllib.loads((tmp_path / "fc.system").read_text())
    assert manifest["run"]["frozen_core_orbitals"] == n_frozen
    assert manifest["run"]["frozen_core_convention"] == _PUBLISHED_DEFAULT

    records = [
        json.loads(line)
        for line in (tmp_path / "fc.scf.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_name, count_field = _DONE_EVENT[(method, system)]
    done = next(record for record in records if record["event"] == event_name)
    assert done[count_field] == n_frozen
    assert done["frozen_core_convention"] == _PUBLISHED_DEFAULT

    if method == "dlpno-mp2":
        assert "Threshold convention" in out
        assert "NormalPNO" in out
        if system == "closed":
            assert "tcut_pairs 0.0001" in out
            assert "tcut_pno 3.33e-07" in out
            assert "tcut_mkn 0.001" in out
        else:
            assert "tcut_pno 3.33e-07" in out
            assert "tcut_pairs 0.0001" not in out
            assert "Unsupported cutoffs" in out and "tcut_mkn" in out
            assert "Inactive cutoffs" in out and "tcut_pairs" in out


def test_ccsd_and_dlpno_ccsd_share_published_frozen_core_default(
    tmp_path, monkeypatch
):
    """The original CCSD/DLPNO-CCSD #140 divergence stays closed."""
    monkeypatch.chdir(tmp_path)
    run_job(_h2o(), basis="def2-svp", method="ccsd", output="canon")
    run_job(_h2o(), basis="def2-svp", method="dlpno-ccsd", output="local")

    canonical = (tmp_path / "canon.out").read_text()
    local = (tmp_path / "local.out").read_text()

    assert "Frozen core orbitals = 1" in canonical
    assert "Frozen core orbitals = 1" in local


def test_native_rmp2_default_matches_explicit_published_count_across_modes():
    """The native sentinel reaches the RMP2 slice, not just output metadata."""

    molecule = _h2o()
    basis = BasisSet(molecule, "def2-svp")
    rhf = run_rhf(molecule, basis, RHFOptions())
    assert rhf.converged

    default_options = MP2Options()
    default_options.memory_mode = "incore"
    default = run_mp2(molecule, basis, rhf, default_options)

    explicit_options = MP2Options()
    explicit_options.n_frozen_core = chemical_core_orbital_count(molecule)
    explicit_options.memory_mode = "direct"
    explicit = run_mp2(molecule, basis, rhf, explicit_options)

    all_electron_options = MP2Options()
    all_electron_options.n_frozen_core = 0
    all_electron_options.memory_mode = "direct"
    all_electron = run_mp2(
        molecule,
        basis,
        rhf,
        all_electron_options,
    )

    assert default.n_frozen_core == explicit.n_frozen_core == 1
    assert default.e_os == pytest.approx(explicit.e_os, abs=1e-11)
    assert default.e_ss == pytest.approx(explicit.e_ss, abs=1e-11)
    assert default.e_correlation == pytest.approx(
        explicit.e_correlation,
        abs=1e-11,
    )
    assert all_electron.n_frozen_core == 0
    assert abs(all_electron.e_correlation - default.e_correlation) > 1e-7


def test_native_ump2_default_matches_explicit_published_count_across_modes():
    """The native sentinel reaches every open-shell UMP2 channel."""

    molecule = _oh()
    basis = BasisSet(molecule, "def2-svp")
    uhf_options = UHFOptions()
    uhf_options.stability_check = False
    uhf = run_uhf(molecule, basis, uhf_options)
    assert uhf.converged

    default_options = UMP2Options()
    default_options.memory_mode = "incore"
    default = run_ump2(molecule, basis, uhf, default_options)

    explicit_options = UMP2Options()
    explicit_options.n_frozen_core = chemical_core_orbital_count(molecule)
    explicit_options.memory_mode = "direct"
    explicit = run_ump2(molecule, basis, uhf, explicit_options)

    all_electron_options = UMP2Options()
    all_electron_options.n_frozen_core = 0
    all_electron_options.memory_mode = "direct"
    all_electron = run_ump2(
        molecule,
        basis,
        uhf,
        all_electron_options,
    )

    assert default.n_frozen_core == explicit.n_frozen_core == 1
    assert default.e_aa == pytest.approx(explicit.e_aa, abs=1e-11)
    assert default.e_bb == pytest.approx(explicit.e_bb, abs=1e-11)
    assert default.e_ab == pytest.approx(explicit.e_ab, abs=1e-11)
    assert default.e_correlation == pytest.approx(
        explicit.e_correlation,
        abs=1e-11,
    )
    assert all_electron.n_frozen_core == 0
    assert abs(all_electron.e_correlation - default.e_correlation) > 1e-7


def test_native_ccsd_default_matches_explicit_published_count():
    """The low-level closed-shell CC boundary resolves its None sentinel."""

    from vibeqc import _vibeqc_core as core

    molecule = _h2o()
    basis = BasisSet(molecule, "sto-3g")
    rhf = run_rhf(molecule, basis, RHFOptions())
    assert rhf.converged

    default_options = core.CCSDOptions()
    assert default_options.n_frozen_core is None
    default_options.density_fit = False
    default_options.compute_triples = False
    default = core.run_ccsd(molecule, basis, rhf, default_options)

    explicit_options = core.CCSDOptions()
    explicit_options.n_frozen_core = chemical_core_orbital_count(molecule)
    explicit_options.density_fit = False
    explicit_options.compute_triples = False
    explicit = core.run_ccsd(molecule, basis, rhf, explicit_options)

    all_electron_options = core.CCSDOptions()
    all_electron_options.n_frozen_core = 0
    all_electron_options.density_fit = False
    all_electron_options.compute_triples = False
    all_electron = core.run_ccsd(
        molecule, basis, rhf, all_electron_options
    )

    assert default.converged and explicit.converged and all_electron.converged
    assert default.e_ccsd_correlation == pytest.approx(
        explicit.e_ccsd_correlation, abs=1e-11
    )
    assert default.t1_amplitudes.shape == explicit.t1_amplitudes.shape
    assert all_electron.t1_amplitudes.shape[0] == default.t1_amplitudes.shape[0] + 1


def test_native_uccsd_default_matches_explicit_published_count():
    """The low-level open-shell CC boundary resolves the same sentinel."""

    from vibeqc import _vibeqc_core as core

    molecule = _oh()
    basis = BasisSet(molecule, "sto-3g")
    uhf_options = UHFOptions()
    uhf_options.stability_check = False
    uhf = run_uhf(molecule, basis, uhf_options)
    assert uhf.converged

    default_options = core.CCSDOptions()
    assert default_options.n_frozen_core is None
    default_options.density_fit = False
    default_options.compute_triples = False
    default = core.run_uccsd(molecule, basis, uhf, default_options)

    explicit_options = core.CCSDOptions()
    explicit_options.n_frozen_core = chemical_core_orbital_count(molecule)
    explicit_options.density_fit = False
    explicit_options.compute_triples = False
    explicit = core.run_uccsd(molecule, basis, uhf, explicit_options)

    all_electron_options = core.CCSDOptions()
    all_electron_options.n_frozen_core = 0
    all_electron_options.density_fit = False
    all_electron_options.compute_triples = False
    all_electron = core.run_uccsd(
        molecule, basis, uhf, all_electron_options
    )

    from vibeqc import estimate_memory

    default_space = estimate_memory(
        molecule,
        basis,
        method="uccsd",
        options=default_options,
    )
    explicit_space = estimate_memory(
        molecule,
        basis,
        method="uccsd",
        options=explicit_options,
    )
    all_electron_space = estimate_memory(
        molecule,
        basis,
        method="uccsd",
        options=all_electron_options,
    )

    assert default.converged and explicit.converged and all_electron.converged
    assert default.e_ccsd_correlation == pytest.approx(
        explicit.e_ccsd_correlation, abs=1e-11
    )
    # The unrestricted native kernel deliberately does not retain its
    # spin-orbital T1 matrix in CCSDResult.  Its norms and energy are the
    # reliable realised-result observables; the empty compatibility matrix
    # is not an active-space dimension.  Equality with the explicit count and
    # separation from the all-electron result together pin sentinel handling.
    assert default.t1_norm == pytest.approx(explicit.t1_norm, abs=1e-12)
    assert default.t2_norm == pytest.approx(explicit.t2_norm, abs=1e-12)
    assert default_space.dims["n_occ"] == explicit_space.dims["n_occ"]
    assert (
        all_electron_space.dims["n_occ"]
        == default_space.dims["n_occ"] + 2
    )
    assert (
        abs(all_electron.e_ccsd_correlation - default.e_ccsd_correlation)
        > 1e-7
    )


def test_uccsd_published_default_rejects_core_larger_than_beta_space():
    """A fully spin-polarised atom fails before a negative beta count forms."""

    from vibeqc.cc import CCSDOptions, run_uccsd

    molecule = Molecule([Atom(5, [0.0, 0.0, 0.0])], multiplicity=6)
    basis = BasisSet(molecule, "def2-svp")
    uhf_options = UHFOptions()
    uhf_options.stability_check = False
    uhf = run_uhf(molecule, basis, uhf_options)
    assert uhf.converged

    options = CCSDOptions(density_fit=False, compute_triples=False)
    with pytest.raises(RuntimeError, match="exceeds the beta-occupied"):
        run_uccsd(molecule, basis, uhf, options)


@pytest.mark.parametrize(
    "method,selector",
    [
        ("mp2", 0),
        ("dlpno-mp2", False),
        ("ccsd", False),
        ("dlpno-ccsd", False),
    ],
)
def test_public_all_electron_escape_hatch_is_uniform(
    method, selector, tmp_path, monkeypatch
):
    """Each closed-shell dispatcher honors the same top-level escape hatch."""

    monkeypatch.chdir(tmp_path)
    run_job(
        _h2o(),
        basis="def2-svp",
        method=method,
        frozen_core=selector,
        output="ae",
        structured_log=True,
    )

    out = (tmp_path / "ae.out").read_text()
    references = (tmp_path / "ae.references").read_text()
    assert "Frozen core orbitals = 0 (explicit all-electron)" in out
    assert "ORCA Manual, Release 6.1.1: Frozen Core Options" not in references
    manifest = tomllib.loads((tmp_path / "ae.system").read_text())
    assert manifest["run"]["frozen_core_orbitals"] == 0
    assert manifest["run"]["frozen_core_convention"] == "all-electron-explicit"


def test_public_frozen_core_rejects_a_second_route_selector(tmp_path):
    """Top-level and route-option selectors never silently pick a winner."""

    from vibeqc._vibeqc_core import MP2Options

    options = MP2Options()
    options.n_frozen_core = 0
    with pytest.raises(ValueError, match="contradicts"):
        run_job(
            _h2o(),
            basis="sto-3g",
            method="mp2",
            frozen_core="published",
            mp2_options=options,
            output=tmp_path / "conflict",
        )


@pytest.mark.parametrize("n_frozen", [0, 1])
def test_bare_native_cc_options_preserve_explicit_frozen_count(n_frozen):
    """Passing a low-level native options object makes its value explicit."""

    from vibeqc._vibeqc_core import CCSDOptions as NativeCCSDOptions
    from vibeqc.cc import _copy_ccsd_options, _resolve_ccsd_frozen_core

    native = NativeCCSDOptions()
    native.n_frozen_core = n_frozen
    copied = _copy_ccsd_options(native)
    resolved = _resolve_ccsd_frozen_core(copied, _h2o())

    assert resolved.n_frozen_core == n_frozen
    assert resolved._n_frozen_core_explicit is True


def test_unknown_dlpno_preset_fails_before_creating_artifacts(tmp_path):
    """A misspelled named set is rejected during input validation."""

    stem = tmp_path / "bad-preset"
    with pytest.raises(ValueError, match="unknown DLPNO threshold preset"):
        run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            dlpno_thresholds="NearlyTightPNO",
            output=stem,
        )
    assert not stem.with_suffix(".out").exists()


def test_named_dlpno_preset_conflict_is_explicit(tmp_path):
    """A named selector cannot obscure threshold-bearing route options."""

    from vibeqc.dlpno.mp2 import DLPNOMP2Options

    with pytest.raises(ValueError, match="cannot be combined"):
        run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            dlpno_thresholds="tight",
            dlpno_options=DLPNOMP2Options(),
            output=tmp_path / "threshold-conflict",
        )


def test_open_dlpno_mp2_reports_only_effective_named_cutoffs(
    tmp_path, monkeypatch
):
    """Requested, applied, unsupported, and inactive remain distinguishable."""

    monkeypatch.chdir(tmp_path)
    run_job(
        _oh(),
        basis="def2-svp",
        method="dlpno-mp2",
        dlpno_thresholds="tight",
        output="tight-u",
        structured_log=True,
    )

    out = (tmp_path / "tight-u.out").read_text()
    assert "Threshold convention" in out and "TightPNO (supported subset)" in out
    assert "tcut_pno 1e-07" in out
    assert "tcut_pairs 1e-05" not in out
    assert "tcut_mkn 0.0001" not in out
    assert "Inactive cutoffs" in out and "tcut_pairs" in out
    assert "Unsupported cutoffs" in out and "tcut_mkn" in out

    manifest = tomllib.loads((tmp_path / "tight-u.system").read_text())
    run = manifest["run"]
    assert run["dlpno_threshold_preset"] == "TightPNO"
    assert run["dlpno_tcut_pno"] == pytest.approx(1e-7)
    assert run["dlpno_tcut_pairs"] == ""
    assert run["dlpno_tcut_mkn"] == ""
    assert run["dlpno_threshold_inactive"] == "tcut_pairs"
    assert run["dlpno_threshold_unsupported"] == "tcut_mkn"

    records = [
        json.loads(line)
        for line in (tmp_path / "tight-u.scf.jsonl").read_text().splitlines()
        if line.strip()
    ]
    done = next(r for r in records if r["event"] == "dlpno_ump2_done")
    assert done["threshold_preset"] == "TightPNO"
    assert done["threshold_requested"] == {
        "tcut_pairs": pytest.approx(1e-5),
        "tcut_pno": pytest.approx(1e-7),
        "tcut_mkn": pytest.approx(1e-4),
    }
    assert done["threshold_applied"] == {"tcut_pno": pytest.approx(1e-7)}
    assert done["threshold_inactive"] == ["tcut_pairs"]
    assert done["threshold_unsupported"] == ["tcut_mkn"]


@pytest.mark.parametrize(
    "atomic_number,expected_orbitals",
    [
        (0, 0),
        (1, 0),
        (4, 0),
        (5, 1),
        (12, 1),
        (13, 5),
        (30, 5),
        (31, 9),
        (38, 9),
        (39, 14),
        (48, 14),
        (49, 18),
        (70, 18),
        (71, 23),
        (80, 23),
        (81, 34),
        (103, 34),
        (104, 50),
        (112, 50),
    ],
)
def test_published_chemical_core_table(atomic_number, expected_orbitals):
    """ORCA table counts plus the zero-electron ghost-centre boundary."""

    inventory = type("AtomicInventory", (), {})()
    inventory.atoms = [Atom(atomic_number, [0.0, 0.0, 0.0])]
    assert chemical_core_orbital_count(inventory) == expected_orbitals


@pytest.mark.parametrize("atomic_number", [-1, 113])
def test_published_chemical_core_table_rejects_out_of_range(atomic_number):
    from vibeqc._vibeqc_core import (
        _published_frozen_core_orbitals_for_atomic_number,
    )

    with pytest.raises(ValueError, match="atomic numbers 0..112"):
        _published_frozen_core_orbitals_for_atomic_number(atomic_number)


def test_checked_in_mp2_benchmark_decks_preserve_all_electron_protocol():
    """The old S22/open-shell evidence cannot inherit the new default."""

    repo = Path(__file__).resolve().parents[1]
    roots = (
        repo
        / "examples/molecular/mp2_benchmarks/s22/inputs/vibeqc",
        repo
        / "examples/molecular/mp2_benchmarks/open_shell/inputs/vibeqc",
    )
    decks = sorted(path for root in roots for path in root.rglob("*.py"))
    assert len(decks) == 325

    native_variants = {"mp2", "rimp2", "ump2", "riump2"}
    wrapper_variants = {"scsmp2", "sosmp2", "scsump2", "sosump2"}

    def assigned_name(statement):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return statement.targets[0].id
        return None

    def call_leaf(call):
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def attribute_chain(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ()
        return tuple([node.id, *reversed(parts)])

    def result_value(tree, key):
        result_assign = next(
            statement
            for statement in tree.body
            if assigned_name(statement) == "result"
            and isinstance(statement.value, ast.Dict)
        )
        for dict_key, value in zip(
            result_assign.value.keys,
            result_assign.value.values,
        ):
            if isinstance(dict_key, ast.Constant) and dict_key.value == key:
                return value
        raise AssertionError(f"result dictionary has no {key!r} field")

    for path in decks:
        variant = path.stem.rsplit("__", 1)[-1]
        tree = ast.parse(path.read_text(), filename=str(path))
        module_doc = ast.get_docstring(tree, clean=False) or ""
        assert "Core:     all-electron (explicit historical" in module_doc, path
        # Generated evidence decks are deliberately straight-line programs;
        # only direct module statements can establish their protocol.
        assert not any(
            isinstance(statement, (ast.If, ast.For, ast.While, ast.Try))
            for statement in tree.body
        ), path
        if variant in native_variants:
            is_open = variant.startswith("u") or variant.startswith("riu")
            option_name = "ump2_opts" if is_open else "mp2_opts"
            option_type = "UMP2Options" if is_open else "MP2Options"
            runner_name = "run_ump2" if is_open else "run_mp2"
            constructor = next(
                statement
                for statement in tree.body
                if assigned_name(statement) == option_name
                and call_leaf(statement.value) == option_type
            )
            zero = next(
                statement
                for statement in tree.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and attribute_chain(statement.targets[0])
                == (option_name, "n_frozen_core")
                and ast.literal_eval(statement.value) == 0
            )
            run = next(
                statement
                for statement in tree.body
                if assigned_name(statement) == "mp2"
                and call_leaf(statement.value) == runner_name
                and any(
                    isinstance(arg, ast.Name) and arg.id == option_name
                    for arg in statement.value.args
                )
            )
            assert constructor.lineno < zero.lineno < run.lineno, path
            expected_result_chain = ("mp2", "n_frozen_core")
        elif variant in wrapper_variants:
            runner_name = {
                "scsmp2": "run_scs_mp2",
                "sosmp2": "run_sos_mp2",
                "scsump2": "run_scs_ump2",
                "sosump2": "run_sos_ump2",
            }[variant]
            run = next(
                statement
                for statement in tree.body
                if assigned_name(statement) == "mp2"
                and call_leaf(statement.value) == runner_name
            )
            frozen = next(
                keyword
                for keyword in run.value.keywords
                if keyword.arg == "frozen_core"
            )
            assert ast.literal_eval(frozen.value) is False, path
            expected_result_chain = ("mp2", "n_frozen_core")
        else:
            assert variant == "b2plyp", path
            next(
                statement
                for statement in tree.body
                if assigned_name(statement) == "db"
                and call_leaf(statement.value) == "run_b2plyp"
            )
            expected_result_chain = ("db", "mp2", "n_frozen_core")
        assert attribute_chain(result_value(tree, "n_frozen_core")) == (
            expected_result_chain
        ), path


def test_cross_code_regression_runners_pin_all_electron_mp2():
    """ORCA NoFrozenCore must remain paired with an explicit vibe-qc zero."""

    repo = Path(__file__).resolve().parents[1]
    parity_vq = (
        repo / "examples/regression/parity_matrix_orca/vibeqc_compare.py"
    ).read_text()
    parity_orca = (
        repo / "examples/regression/parity_matrix_orca/orca_input.py"
    ).read_text()
    core_vq = (
        repo / "examples/regression/core/runner_vibeqc.py"
    ).read_text()
    core_orca = (
        repo / "examples/regression/core/runner_orca.py"
    ).read_text()
    core_pyscf = (
        repo / "examples/regression/core/runner_pyscf.py"
    ).read_text()
    ase_compare = (
        repo
        / "examples/ase_compare/cross_code_regression"
        / "compare_h2o_mp2_basis_scan.py"
    ).read_text()

    def call_leaf(call):
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def assigned_zero_and_used(
        source, function_name, option_name, consumer_name
    ):
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )

        def is_zero_assignment(statement):
            return (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Attribute)
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id == option_name
                and statement.targets[0].attr == "n_frozen_core"
                and isinstance(statement.value, ast.Constant)
                and statement.value.value == 0
            )

        def overwrites_selector(statement):
            return (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == option_name
                    and target.attr == "n_frozen_core"
                    for target in statement.targets
                )
            )

        def reaches_consumer(statement):
            if consumer_name == "return":
                return (
                    isinstance(statement, ast.Return)
                    and isinstance(statement.value, ast.Name)
                    and statement.value.id == option_name
                )
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                return False
            value = getattr(statement, "value", None)
            return any(
                isinstance(node, ast.Call)
                and call_leaf(node) == consumer_name
                and any(
                    isinstance(arg, ast.Name) and arg.id == option_name
                    for arg in node.args
                )
                for node in ast.walk(value)
            )

        def scan(statements, selector_is_zero=False):
            state = selector_is_zero
            for statement in statements:
                if isinstance(statement, ast.If):
                    if isinstance(statement.test, ast.Constant):
                        branch = statement.body if statement.test.value else statement.orelse
                        found, state = scan(branch, state)
                        if found:
                            return True, state
                        continue
                    found_body, state_body = scan(statement.body, state)
                    found_else, state_else = scan(statement.orelse, state)
                    if found_body or found_else:
                        return True, state_body and state_else
                    state = state_body and state_else
                    continue
                if isinstance(statement, ast.Try):
                    found, state_body = scan(statement.body, state)
                    if found:
                        return True, state_body
                    handler_states = []
                    for handler in statement.handlers:
                        found, handler_state = scan(handler.body, state)
                        if found:
                            return True, handler_state
                        handler_states.append(handler_state)
                    state = state_body and all(handler_states or [True])
                    continue
                if isinstance(statement, (ast.For, ast.While)):
                    found, _ = scan(statement.body, state)
                    if found:
                        return True, state
                    continue
                if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if is_zero_assignment(statement):
                    state = True
                elif overwrites_selector(statement):
                    state = False
                if state and reaches_consumer(statement):
                    return True, state
            return False, state

        return scan(function.body)[0]

    def function_has_runtime_string(source, function_name, fragment):
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )

        def literal_strings(node):
            if node is None:
                return set()
            return {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            }

        def scan(statements, bindings):
            state = {name: set(values) for name, values in bindings.items()}
            for statement in statements:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name):
                        values = literal_strings(statement.value)
                        for name in {
                            node.id
                            for node in ast.walk(statement.value)
                            if isinstance(node, ast.Name)
                        }:
                            values.update(state.get(name, set()))
                        state[target.id] = values
                        continue
                if (
                    isinstance(statement, ast.AugAssign)
                    and isinstance(statement.target, ast.Name)
                ):
                    state.setdefault(statement.target.id, set()).update(
                        literal_strings(statement.value)
                    )
                    continue
                if isinstance(statement, ast.If):
                    if isinstance(statement.test, ast.Constant):
                        branch = statement.body if statement.test.value else statement.orelse
                        found, state = scan(branch, state)
                        if found:
                            return True, state
                        continue
                    found_body, body_state = scan(statement.body, state)
                    found_else, else_state = scan(statement.orelse, state)
                    if found_body or found_else:
                        return True, state
                    for name in set(body_state) | set(else_state):
                        state[name] = body_state.get(name, set()) | else_state.get(
                            name, set()
                        )
                    continue
                if isinstance(statement, ast.Return):
                    values = literal_strings(statement.value)
                    for name in {
                        node.id
                        for node in ast.walk(statement.value)
                        if isinstance(node, ast.Name)
                    }:
                        values.update(state.get(name, set()))
                    if any(fragment in value for value in values):
                        return True, state
            return False, state

        return scan(function.body, {})[0]

    assert assigned_zero_and_used(
        parity_vq, "_vibeqc_decompose_mp2", "mp2_opts", "run_mp2"
    )
    assert assigned_zero_and_used(
        parity_vq, "_vibeqc_decompose_mp2", "ump2_opts", "run_ump2"
    )
    assert function_has_runtime_string(
        parity_orca, "_orca_mp2_simple_input", "NoFrozenCore"
    )
    assert function_has_runtime_string(
        parity_orca, "build_orca_input", "FrozenCore FC_NONE"
    )
    assert assigned_zero_and_used(
        core_vq, "_make_mp2_options", "opts", "return"
    )
    assert function_has_runtime_string(
        core_orca, "_orca_simpleinput", "NoFrozenCore"
    )
    assert function_has_runtime_string(
        core_orca, "_orca_blocks", "FrozenCore FC_NONE"
    )

    outer_pyscf = ast.parse(core_pyscf)
    embedded = next(
        node.value.value
        for node in outer_pyscf.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_PYSCF_EXTERNAL_SCRIPT"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    embedded_tree = ast.parse(embedded)
    for function_name, expected_calls in (("run_periodic", 2), ("run_molecule", 2)):
        function = next(
            node
            for node in embedded_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        constructors = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"MP2", "UMP2", "KMP2"}
        ]
        assert len(constructors) == expected_calls
        assert all(
            any(
                keyword.arg == "frozen"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 0
                for keyword in constructor.keywords
            )
            for constructor in constructors
        )
    assert assigned_zero_and_used(
        ase_compare, "run_vibeqc", "mp2_options", "run_mp2"
    )

    ase_tree = ast.parse(ase_compare)
    pyscf_function = next(
        node
        for node in ase_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_pyscf"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "MP2"
        and any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 0
            for keyword in node.keywords
        )
        for node in ast.walk(pyscf_function)
    )
    orca_function = next(
        node
        for node in ase_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_orca"
    )
    assert any(
        isinstance(node, ast.keyword)
        and node.arg == "orcasimpleinput"
        and "NoFrozenCore" in ast.unparse(node.value)
        for node in ast.walk(orca_function)
    )


def test_double_hybrid_pyscf_reconstructions_pin_all_electron_mp2():
    """External composite-model anchors must not inherit an MP2 default."""

    repo = Path(__file__).resolve().parents[1]
    helpers = {
        "test_b2plyp.py": "_pyscf_b2plyp",
        "test_dsd_pbep86.py": "_pyscf_dsd_pbep86",
        "test_revdsd_pbep86.py": "_pyscf_revdsd_pbep86",
    }
    for filename, function_name in helpers.items():
        tree = ast.parse((repo / "tests" / filename).read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        constructors = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "MP2"
        ]
        assert len(constructors) == 1
        assert any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 0
            for keyword in constructors[0].keywords
        )


def test_retained_dlpno_ccsd_t_example_pins_legacy_recipe():
    """The checked-in accuracy table stays tied to its measured protocol."""

    repo = Path(__file__).resolve().parents[1]
    source = (
        repo / "examples/molecular/benchmark-dlpno-ccsd-t.py"
    ).read_text()
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def call_leaf(call):
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    heading = main.body[0]
    assert (
        isinstance(heading, ast.Expr)
        and call_leaf(heading.value) == "print"
        and "legacy pinned recipe" in ast.unparse(heading.value)
    )

    loop = next(
        statement for statement in main.body if isinstance(statement, ast.For)
    )
    constructor = next(
        statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "cc_opts"
        and call_leaf(statement.value) == "CCSDOptions"
    )
    zero_assignment = next(
        statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Attribute)
        and isinstance(statement.targets[0].value, ast.Name)
        and statement.targets[0].value.id == "cc_opts"
        and statement.targets[0].attr == "n_frozen_core"
        and isinstance(statement.value, ast.Constant)
        and statement.value.value == 0
    )
    canonical = next(
        statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "canon"
        and call_leaf(statement.value) == "run_ccsd"
        and any(
            isinstance(arg, ast.Name) and arg.id == "cc_opts"
            for arg in statement.value.args
        )
    )
    assert constructor.lineno < zero_assignment.lineno < canonical.lineno

    local = next(
        statement
        for statement in loop.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "dlpno"
        and call_leaf(statement.value) == "run_local_dlpno_ccsd"
    )
    local_call = next(
        arg
        for arg in local.value.args
        if call_leaf(arg) == "LocalCCSDOptions"
    )
    keywords = {keyword.arg: keyword.value for keyword in local_call.keywords}
    assert ast.literal_eval(keywords["n_frozen"]) == 0
    assert ast.literal_eval(keywords["tcut_pno"]) == pytest.approx(1e-7)
    assert ast.literal_eval(keywords["tcut_mkn"]) == 0.0
    assert ast.literal_eval(keywords["tcut_pairs"]) == pytest.approx(1e-4)
    assert ast.literal_eval(keywords["residual_domain"]) == "pair"
    assert ast.literal_eval(keywords["triples_mode"]) == "t1"


def test_molecular_examples_bind_comparisons_and_exactness_to_one_recipe():
    """Runnable examples cannot mix current defaults with old references."""

    repo = Path(__file__).resolve().parents[1]

    def _tree(name):
        return ast.parse((repo / "examples/molecular" / name).read_text())

    def _call_leaf(call):
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def _direct_assignment(tree, name, call_name=None):
        return next(
            statement
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
            and (call_name is None or _call_leaf(statement.value) == call_name)
        )

    def _run_job_keywords(tree):
        assignment = _direct_assignment(tree, "result", "run_job")
        return {
            keyword.arg: keyword.value for keyword in assignment.value.keywords
        }

    def _published_count_assignment(tree):
        assignment = _direct_assignment(
            tree, "n_frozen", "chemical_core_orbital_count"
        )
        call = assignment.value
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "mol"
        return assignment

    for name in ("input-h2o-ccsd-t.py", "input-ch3-uccsd-t.py"):
        keywords = _run_job_keywords(_tree(name))
        assert isinstance(keywords["frozen_core"], ast.Constant)
        assert keywords["frozen_core"].value is False

    current = _run_job_keywords(_tree("input-h2o-dlpno-ccsd-t.py"))
    assert ast.literal_eval(current["frozen_core"]) == "published"
    assert ast.literal_eval(current["dlpno_thresholds"]) == "normal"

    ccsd_tree = _tree("input-h2o-dlpno-ccsd.py")
    count_assignment = _published_count_assignment(ccsd_tree)
    exact_cc_assignment = _direct_assignment(
        ccsd_tree, "exact", "run_local_dlpno_ccsd"
    )
    exact_cc = next(
        arg
        for arg in exact_cc_assignment.value.args
        if _call_leaf(arg) == "LocalCCSDOptions"
    )
    exact_cc_keywords = {
        keyword.arg: keyword.value for keyword in exact_cc.keywords
    }
    for field in ("tcut_pairs", "tcut_pno", "tcut_mkn", "coupling_radius"):
        assert ast.literal_eval(exact_cc_keywords[field]) == 0.0
    assert ast.literal_eval(exact_cc_keywords["residual_domain"]) == "full"
    assert (
        isinstance(exact_cc_keywords["n_frozen"], ast.Name)
        and exact_cc_keywords["n_frozen"].id == "n_frozen"
    )
    assert count_assignment.lineno < exact_cc_assignment.lineno

    mp2_tree = _tree("input-h2o-dlpno-mp2.py")
    count_assignment = _published_count_assignment(mp2_tree)
    canonical_options = _direct_assignment(mp2_tree, "mp2_opts", "MP2Options")
    count_selector = next(
        statement
        for statement in mp2_tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Attribute)
        and isinstance(statement.targets[0].value, ast.Name)
        and statement.targets[0].value.id == "mp2_opts"
        and statement.targets[0].attr == "n_frozen_core"
        and isinstance(statement.value, ast.Name)
        and statement.value.id == "n_frozen"
    )
    canonical_run = _direct_assignment(mp2_tree, "canonical", "run_mp2")
    assert any(
        isinstance(arg, ast.Name) and arg.id == "mp2_opts"
        for arg in canonical_run.value.args
    )
    assert (
        count_assignment.lineno
        < canonical_options.lineno
        < count_selector.lineno
        < canonical_run.lineno
    )

    sweep = next(
        statement
        for statement in mp2_tree.body
        if isinstance(statement, ast.For)
    )
    assert (
        isinstance(sweep.target, ast.Tuple)
        and [item.id for item in sweep.target.elts if isinstance(item, ast.Name)]
        == ["label", "opts"]
    )
    local_run = next(
        node
        for statement in sweep.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and _call_leaf(node) == "run_dlpno_mp2"
        and any(
            isinstance(arg, ast.Name) and arg.id == "opts"
            for arg in node.args
        )
    )
    exact_mp2 = next(
        node
        for node in ast.walk(sweep.iter)
        if isinstance(node, ast.Call)
        and _call_leaf(node) == "DLPNOMP2Options"
        and {
            keyword.arg
            for keyword in node.keywords
            if isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 0.0
        }
        >= {
            "tcut_pairs",
            "tcut_pairs_weak",
            "tcut_pno",
            "tcut_pno_weak",
            "tcut_mkn",
        }
    )
    exact_mp2_keywords = {
        keyword.arg: keyword.value for keyword in exact_mp2.keywords
    }
    assert (
        isinstance(exact_mp2_keywords["n_frozen"], ast.Name)
        and exact_mp2_keywords["n_frozen"].id == "n_frozen"
    )
    assert exact_mp2.lineno < local_run.lineno


def test_correlated_pyscf_reference_calls_pin_all_electron():
    """Exact-reference constructors cannot silently inherit PySCF defaults."""

    repo = Path(__file__).resolve().parents[1]

    def assert_frozen_zero(call):
        frozen = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "frozen"
        )
        assert isinstance(frozen, ast.Constant) and frozen.value == 0

    canonical_tree = ast.parse(
        (repo / "tests/test_ccsd_canonical_noDF.py").read_text()
    )
    for function_name, constructor_name in (
        ("_pyscf_conventional_rccsd_t", "CCSD"),
        ("_pyscf_conventional_uccsd_t", "UCCSD"),
    ):
        function = next(
            statement
            for statement in canonical_tree.body
            if isinstance(statement, ast.FunctionDef)
            and statement.name == function_name
        )
        assignment = next(
            statement
            for statement in function.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "mycc"
        )
        call = assignment.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute)
        assert call.func.attr == constructor_name
        assert_frozen_zero(call)

    diagnostic_tree = ast.parse(
        (repo / "examples/regression/scripts/cc_diag.py").read_text()
    )
    diagnostic_assignment = next(
        statement
        for statement in diagnostic_tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "pcc"
    )
    assert isinstance(diagnostic_assignment.value, ast.Call)
    assert isinstance(diagnostic_assignment.value.func, ast.Attribute)
    assert diagnostic_assignment.value.func.attr == "CCSD"
    assert_frozen_zero(diagnostic_assignment.value)

    rohf_tree = ast.parse((repo / "tests/test_rohf_ccsd.py").read_text())
    rohf_helper = next(
        statement
        for statement in rohf_tree.body
        if isinstance(statement, ast.FunctionDef)
        and statement.name == "_pyscf_rohf_ccsd_t"
    )
    script_assignment = next(
        statement
        for statement in rohf_helper.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "script"
        and isinstance(statement.value, ast.JoinedStr)
    )
    replacements = {
        "atom_str": "H 0 0 0",
        "basis_name": "sto-3g",
        "bool(triples)": "False",
    }
    script_parts = []
    for part in script_assignment.value.values:
        if isinstance(part, ast.Constant):
            script_parts.append(part.value)
        else:
            assert isinstance(part, ast.FormattedValue)
            expression = ast.unparse(part.value)
            script_parts.append(replacements[expression])
    embedded_tree = ast.parse("".join(script_parts))
    embedded_assignment = next(
        statement
        for statement in embedded_tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "mycc"
    )
    assert isinstance(embedded_assignment.value, ast.Call)
    assert isinstance(embedded_assignment.value.func, ast.Attribute)
    assert embedded_assignment.value.func.attr == "RCCSD"
    assert_frozen_zero(embedded_assignment.value)
