"""Pins the ``method=`` gate of
:func:`vibeqc.periodic_runner.run_periodic_job`, and in particular the
K-prefixed spellings.

Background (BUG 125, 2026-08-05): a user report claimed ``method='KRHF'``
had worked at ``v0.15.21`` and was removed by ``v0.15.112``. It was not:
``git show v0.15.21:python/vibeqc/periodic_runner.py`` gates on
``("RHF", "RKS", "UHF", "UKS")``, so ``'KRHF'`` was rejected there too, and
the accepted set has only grown since (ROHF/ROKS). ``KRHF`` has never been
a ``run_periodic_job`` method.

The confusion is understandable and worth catching in the error text rather
than in a bug report: PySCF spells its multi-k driver ``KRHF``, and vibe-qc's
own multi-k GDF result type is ``PeriodicKRHFGDFResult``. But k-point
sampling is not a method here — it is the ``kpoints`` argument, and every
supported method is multi-k whenever ``kpoints`` is set.

Pins:

* every supported method passes the gate (they fail later, on the missing
  basis, not on the method name);
* each K-prefixed spelling raises ``NotImplementedError`` naming the
  un-prefixed replacement *and* ``kpoints``;
* a genuinely unknown method still gets the plain unsupported-list message,
  with no ``kpoints`` advice attached.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core

SUPPORTED = ("RHF", "ROHF", "ROKS", "RKS", "UHF", "UKS")


def _he_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _call(method: str):
    """Trip the method gate without running any physics.

    The system-type check runs first, so a real cell is required; the gate
    then runs before anything expensive. ``basis=None`` is enough — a
    supported method gets past the method gate and dies on the basis check.
    """
    return vq.run_periodic_job(_he_system(), None, method=method)


@pytest.mark.parametrize("method", SUPPORTED)
def test_supported_methods_pass_the_method_gate(method):
    # Not NotImplementedError: these must fall through to the basis check.
    with pytest.raises(Exception) as exc:
        _call(method)
    assert "not supported" not in str(exc.value)


@pytest.mark.parametrize("method", SUPPORTED)
def test_k_prefixed_spelling_names_the_replacement(method):
    kmethod = "K" + method
    with pytest.raises(NotImplementedError) as exc:
        _call(kmethod)
    msg = str(exc.value)
    # The un-prefixed method must be named as the replacement, and the user
    # must be told where k-point sampling actually lives.
    assert repr(method) in msg
    assert "kpoints" in msg


def test_k_prefix_advice_is_case_insensitive():
    with pytest.raises(NotImplementedError) as exc:
        _call("krhf")
    assert "'RHF'" in str(exc.value)
    assert "kpoints" in str(exc.value)


def test_unknown_method_keeps_the_plain_unsupported_message():
    with pytest.raises(NotImplementedError) as exc:
        _call("CCSD")
    msg = str(exc.value)
    assert "not supported" in msg
    # No k-point advice for a method that has nothing to do with k-points.
    assert "kpoints" not in msg


def test_k_prefix_advice_does_not_fire_on_unrelated_k_names():
    """``KS`` is not ``K`` + a supported method; it must not get the advice."""
    with pytest.raises(NotImplementedError) as exc:
        _call("KS")
    assert "kpoints" not in str(exc.value)


def _quiet_output_options(tmp_path, stem):
    """Disable unrelated sidecars for early runner-preflight probes."""
    return {
        "output": tmp_path / stem,
        "write_molden_file": False,
        "write_density": False,
        "write_xyz_file": False,
        "write_poscar_file": False,
        "write_xsf_structure_file": False,
        "write_cif_file": False,
        "write_population_file": False,
        "output_qvf": False,
        "citations": False,
        "progress": False,
    }


@pytest.mark.parametrize("dim", [1, 2])
def test_periodic_sap_rejects_lower_dimension_before_scf(
    monkeypatch,
    tmp_path,
    dim,
):
    """The 3-D SAP lattice contract fails before a GDF driver is entered."""
    import vibeqc.periodic_runner as runner_module

    def unexpected_driver(*_args, **_kwargs):
        pytest.fail("lower-dimensional SAP reached the SCF driver")

    monkeypatch.setattr(
        runner_module,
        "run_krhf_periodic_gdf",
        unexpected_driver,
    )
    lattice = np.diag([4.0, 12.0, 12.0])
    atoms = [core.Atom(1, [0.0, 0.0, 0.0]), core.Atom(1, [0.0, 0.0, 1.4])]
    system = core.PeriodicSystem(dim, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="initial_guess=SAP.*is not implemented by this route"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            initial_guess="SAP",
            **_quiet_output_options(tmp_path, f"sap-{dim}d"),
        )


class _AICCMDriverReached(Exception):
    """Sentinel proving that a public route passed all preflight gates."""


_HCORE_ONLY_AICCM_ROUTES = (
    (
        "real-gamma",
        "vibeqc.periodic.ccm.real_gamma_runner",
        "run_real_gamma_scf",
    ),
    (
        "four-center",
        "vibeqc.periodic.ccm.four_center_runner",
        "run_four_center_scf",
    ),
)


@pytest.mark.parametrize(
    "initial_guess",
    [
        core.InitialGuess.SAD,
        core.InitialGuess.SAP,
        core.InitialGuess.PATOM,
        core.InitialGuess.HUECKEL,
        core.InitialGuess.MINAO,
        core.InitialGuess.READ,
        core.InitialGuess.FRAGMO,
    ],
    ids=lambda guess: guess.name.lower(),
)
@pytest.mark.parametrize(
    ("variant", "module_name", "driver_name"),
    _HCORE_ONLY_AICCM_ROUTES,
    ids=("real-gamma", "four-center"),
)
def test_aiccm_supercell_gamma_routes_reject_every_non_hcore_guess(
    monkeypatch,
    tmp_path,
    initial_guess,
    variant,
    module_name,
    driver_name,
):
    """A route that hard-codes HCORE must reject every other selector."""

    def unexpected_driver(*_args, **_kwargs):
        pytest.fail(f"{variant} accepted {initial_guess.name} before SCF")

    monkeypatch.setattr(
        importlib.import_module(module_name),
        driver_name,
        unexpected_driver,
    )
    system = _he_system(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(
        NotImplementedError,
        match=r"only initial_guess='HCORE'.*fails before SCF",
    ) as exc:
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant=variant,
            aiccm_lattice_extension=(1, 1, 1),
            initial_guess=initial_guess,
            fragments=[[0]] if initial_guess == core.InitialGuess.FRAGMO else None,
            **_quiet_output_options(
                tmp_path,
                f"{variant}-{initial_guess.name.lower()}",
            ),
        )
    message = str(exc.value)
    assert f"variant='{variant}'" in message
    assert initial_guess.name in message


@pytest.mark.parametrize(
    ("variant", "module_name", "driver_name"),
    _HCORE_ONLY_AICCM_ROUTES,
    ids=("real-gamma", "four-center"),
)
@pytest.mark.parametrize("guess_kwargs", [{}, {"initial_guess": "AUTO"}, {"initial_guess": "core"}])
def test_aiccm_supercell_gamma_routes_allow_hcore_to_reach_driver(
    monkeypatch,
    tmp_path,
    variant,
    module_name,
    driver_name,
    guess_kwargs,
):
    """HCORE is truthful for both adapters and therefore reaches SCF."""

    def reached_driver(*_args, **_kwargs):
        raise _AICCMDriverReached

    monkeypatch.setattr(
        importlib.import_module(module_name),
        driver_name,
        reached_driver,
    )
    system = _he_system(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(_AICCMDriverReached):
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant=variant,
            aiccm_lattice_extension=(1, 1, 1),
            **guess_kwargs,
            **_quiet_output_options(tmp_path, f"{variant}-hcore"),
        )


@pytest.mark.parametrize("variant", ["neutral-bloch", "chi"])
def test_aiccm_option_routes_forward_sap_instead_of_being_globally_blocked(
    monkeypatch,
    tmp_path,
    variant,
):
    """The Bloch and B routes receive options and own their guess support."""
    import vibeqc.periodic_runner as runner_module

    def reached_driver(*args, **kwargs):
        options = args[3] if variant == "chi" else kwargs["options"]
        assert options.initial_guess == core.InitialGuess.SAP
        raise _AICCMDriverReached

    if variant == "chi":
        monkeypatch.setattr(
            runner_module,
            "run_aiccm2026dev_b_rhf",
            reached_driver,
        )
    else:
        monkeypatch.setattr(
            importlib.import_module(
                "vibeqc.periodic.ccm.neutral_bloch_runner"
            ),
            "run_neutral_bloch_scf",
            reached_driver,
        )

    system = _he_system(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(_AICCMDriverReached):
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant=variant,
            aiccm_lattice_extension=(1, 1, 1),
            initial_guess="SAP",
            **_quiet_output_options(tmp_path, f"{variant}-sap"),
        )


@pytest.mark.parametrize("fragments", [None, []], ids=["missing", "empty"])
def test_rohf_gdf_fragmo_requires_fragments_before_driver(
    monkeypatch,
    tmp_path,
    fragments,
):
    """FRAGMO requires a valid partition before constructing any density."""
    import vibeqc.periodic_runner as runner_module

    def unexpected_driver(*_args, **_kwargs):
        pytest.fail("ROHF/GDF accepted a non-Hcore guess before SCF")

    monkeypatch.setattr(
        runner_module,
        "run_krohf_periodic_gdf",
        unexpected_driver,
    )
    system = _he_system(12.0)
    system.unit_cell = [core.Atom(1, [6.0, 6.0, 6.0])]
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(
        ValueError,
        match=r"FRAGMO requires fragments",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="gdf",
            initial_guess="FRAGMO",
            fragments=fragments,
            **_quiet_output_options(
                tmp_path,
                "rohf-gdf-fragmo",
            ),
        )


def test_periodic_sap_rejects_atomic_spins_before_scf(monkeypatch, tmp_path):
    """An ATOMSPIN seed cannot be combined with the unrelated SAP guess."""
    import vibeqc.periodic_runner as runner_module

    def unexpected_driver(*_args, **_kwargs):
        pytest.fail("atomic_spins + SAP reached the SCF driver")

    monkeypatch.setattr(runner_module, "run_pbc_gdf_uhf", unexpected_driver)
    system = _he_system(12.0)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(
        ValueError,
        match="atomic_spins requires SAD without a restart density",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="UHF",
            jk_method="gdf",
            atomic_spins=[1],
            initial_guess="SAP",
            **_quiet_output_options(tmp_path, "sap-atomspin"),
        )


@pytest.mark.parametrize(
    ("method", "jk_method", "kpoints", "message"),
    [
        (
            "RHF",
            "gdf",
            None,
            "restart_from is implemented only for GPW and GAPW",
        ),
        (
            "RKS",
            "gpw",
            (2, 1, 1),
            "restart_from currently supplies one Gamma density",
        ),
        (
            "ROHF",
            "gpw",
            None,
            "restart_from currently supplies one Gamma density",
        ),
        (
            "UHF",
            "gapw",
            None,
            "restart_from currently supplies one Gamma density",
        ),
    ],
    ids=["gdf", "gpw-multik", "gpw-rohf", "gapw-uhf"],
)
def test_restart_from_rejects_routes_that_cannot_consume_it(
    tmp_path,
    method,
    jk_method,
    kpoints,
    message,
):
    """Do not label an ignored restart as the effective initial seed."""
    system = _he_system(12.0)
    if method in ("ROHF", "UHF"):
        system.unit_cell = [core.Atom(1, [6.0, 6.0, 6.0])]
        system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    restart_path = tmp_path / "restart.npz"
    restart_path.touch()

    with pytest.raises(
        NotImplementedError,
        match=message,
    ):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "RKS" else None,
            jk_method=jk_method,
            kpoints=kpoints,
            gapw_molecular_limit=(jk_method == "gapw"),
            initial_guess="SAP",
            restart_from=restart_path,
            **_quiet_output_options(
                tmp_path,
                f"sap-restart-{jk_method}-{method.lower()}",
            ),
        )


def test_periodic_public_default_is_auto():
    import inspect

    assert inspect.signature(vq.run_periodic_job).parameters["initial_guess"].default == "AUTO"


def test_auto_capability_resolution_preserves_explicit_refusal():
    from vibeqc.guess import resolve_initial_guess

    mol = _he_system(12.0).unit_cell_molecule()
    assert resolve_initial_guess(mol, "AUTO", is_periodic=True) == core.InitialGuess.SAD
    assert resolve_initial_guess(
        mol, "AUTO", is_periodic=True, supported=("HCORE",)
    ) == core.InitialGuess.HCORE
    with pytest.raises(NotImplementedError, match="SAD"):
        resolve_initial_guess(mol, "SAD", is_periodic=True, supported=("HCORE",))


@pytest.mark.parametrize("variant", ["real-gamma", "four-center"])
def test_aiccm_auto_hcore_rejects_atomic_spin_seed(tmp_path, variant):
    system = vq.PeriodicSystem(3, np.eye(3) * 12.0, [vq.Atom(3, [0, 0, 0])], 0, 2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="atomic_spins requires SAD.*AUTO resolves to HCORE"):
        vq.run_periodic_job(
            system, basis, method="aiccm", variant=variant,
            atomic_spins=[1],
            aiccm_lattice_extension=(1, 1, 1),
            **_quiet_output_options(tmp_path, variant),
        )


def test_rohf_gdf_valid_fragmo_reaches_source_preparation(monkeypatch, tmp_path):
    """Valid FRAGMO passes route admission; stop before fragment SCF."""
    import vibeqc.guess_fragmo as fragmo_module
    import vibeqc.periodic_runner as runner_module

    class SourcePreparationReached(Exception):
        pass

    def source_boundary(opts, system, basis, kmesh, fragments):
        assert opts.initial_guess == core.InitialGuess.READ
        assert len(fragments) == 1
        assert tuple(fragments[0].atoms) == (0,)
        assert fragments[0].multiplicity == 2
        raise SourcePreparationReached

    def unexpected_driver(*args, **kwargs):
        pytest.fail("Numerical driver must not run in source-admission test")

    monkeypatch.setattr(fragmo_module, "resolve_periodic_fragmo_source", source_boundary)
    monkeypatch.setattr(runner_module, "run_krohf_periodic_gdf", unexpected_driver)
    system = _he_system(12.0)
    system.unit_cell = [core.Atom(1, [6.0, 6.0, 6.0])]
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(SourcePreparationReached):
        vq.run_periodic_job(
            system, basis, method="ROHF", jk_method="gdf",
            initial_guess="FRAGMO",
            fragments=[fragmo_module.Fragment(atoms=[0], multiplicity=2)],
            **_quiet_output_options(tmp_path, "rohf-gdf-valid-fragmo"),
        )
