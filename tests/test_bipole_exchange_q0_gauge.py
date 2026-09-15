"""The corrected-exchange ``q + G = 0`` gauge: one definition, and disclosed.

Issue #82 filed a ``+6.43e-1 Ha`` MgO/STO-3G (2,2,2) disagreement against a
sealed CRYSTAL23 SHRINK-2 HF reference as an "exchange finite-size gauge".
Measured (2026-08-28, CRYSTAL23 1.0.1 run natively, same deck, SHRINK varied):

    SHRINK 2  -271.85639651657      SHRINK 4  -271.21780959957
    SHRINK 3  -271.22261953049      SHRINK 8  -271.21814374982

so **99.19 % of the filed offset is the reference's own SHRINK-2 finite-size
error** -- CRYSTAL applies no Gygi-Baldereschi / probe-charge-Madelung
correction, vibe-qc does, and vibe-qc's (2,2,2) total reproduces PySCF KRHF
``exxdiv='ewald'`` on the same cell to 0.37 mHa. The exact-exchange-off
control (PBE, same cell, same mesh, same code) moves only +46.9 mHa between
SHRINK 2 and 8, and in the opposite direction.

**None of that was new.** ``docs/periodic_jk_routes.md`` has carried the
finding since 2026-06-16 -- maintainer-confirmed, and retiring the coarse
CRYSTAL value as a BIPOLE target -- and
``test_pbc_bipole_multik_ewald_split.py::test_mgo_shrink22_converged_matches_crystal_kconverged``
pins it. #82 was filed against the retired seal anyway, two months later, and
worked by four lanes. That is the argument for these tests: the convention was
written down and tested, and it still was not *findable* from a run.

The vibe-qc-side defect that made a 14-day investigation possible is that the
gauge was **invisible from a run**: ``e_exchange`` folded the ``K_LR`` and
``xi_M . S D S`` pieces together, so a Ha-scale convention term appeared in
the total with nothing in the ``.out`` or on the result naming it. These tests
pin the disclosure and the single definition of the gauge constant.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    monkhorst_pack,
)
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from vibeqc.output.formats.scf_log import format_scf_trace
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

MESH = (2, 1, 1)


def _he_cubic():
    """The cheapest real multi-k BIPOLE cell in the suite (1 BF, 2 electrons)."""
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _rhf_opts():
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    return opts


def _run_rhf(exxdiv: str = "ewald"):
    system, basis = _he_cubic()
    kmesh = monkhorst_pack(system, list(MESH), use_symmetry=False)
    return system, run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _rhf_opts(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        exchange_exxdiv=exxdiv,
        progress=False,
    )


# --------------------------------------------------------------------------
# 1. One definition of the gauge constant.
# --------------------------------------------------------------------------

# #478 landed this failure mode one level up: two hand-rolled copies of the
# Ewald real-space cutoff desynced the analytic gradient from the energy
# (max|dg| = 1.27e-3 against a 1e-3 gate) the moment one was edited. The q+G=0
# gauge constant appears in BOTH an energy and its gradient, so a second copy
# is a latent desync, not a style question -- at the parent of this commit
# there were eleven, across five modules.
#
# The guard deliberately does NOT carry a module list. A list drifts: a new
# module that started deriving the constant would simply never be looked at,
# and the guard would go on passing (LEARNINGS L123, on contract items that
# nothing checks). The whole package is scanned instead, keyed on the names
# the constant is bound to.
_GAUGE_HOME = "bipole_fock_ewald.py"  # where the one definition lives

# The names the gauge constant is bound to at every call site.
_GAUGE_NAMES = frozenset({"c_g0", "exchange_g0"})
_HELPER = "exchange_q0_gauge_constant"


def _gauge_assignments_not_from_helper(path: Path) -> list[str]:
    """Assignments to a gauge name whose RHS is not the shared helper.

    Parsed with ``ast`` rather than matched with a regex on purpose: a
    textual guard for this is trivially defeated by splitting the
    expression across two statements (bind ``xi_M`` on one line, subtract
    on the next), and a guard that can be fooled is worse than no guard --
    it certifies the result. Binding the *name* is the thing no rewrite of
    the arithmetic can avoid.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        if not (names & _GAUGE_NAMES):
            continue
        value = node.value
        if value is None:
            # A bare dataclass annotation (``exchange_g0: float``) declares
            # the field; it does not derive the constant.
            continue
        ok = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == _HELPER
        )
        # A literal (``exchange_g0 = 0.0``, the exact-exchange-off default)
        # derives nothing. Everything else -- arithmetic, or a call to
        # anything but the helper -- is a second definition.
        if not ok and not isinstance(value, ast.Constant):
            bad.append(f"{path.name}:{node.lineno}")
    return bad


def test_gauge_constant_has_exactly_one_definition():
    """No module may re-derive ``xi_M - pi/(alpha^2 V n_k)`` on its own."""
    src_root = Path(vq.__file__).resolve().parent
    modules = sorted(
        p
        for p in src_root.rglob("*.py")
        if p.name != _GAUGE_HOME
    )
    assert modules, f"no vibeqc sources found under {src_root}"
    offenders: list[str] = []
    for path in modules:
        offenders.extend(_gauge_assignments_not_from_helper(path))
    assert offenders == [], (
        "these sites bind the q+G=0 gauge constant from something other than "
        f"vibeqc.bipole_fock_ewald.{_HELPER}: {offenders}. A gauge constant "
        "that appears in an energy AND its analytic gradient must have one "
        "definition -- #478 landed exactly this failure one level up, where "
        "two copies of the Ewald real-space cutoff desynced the pair the "
        "moment one was edited."
    )


def test_gauge_constant_rejects_a_wedge_sized_n_k():
    """``n_k`` is the FULL mesh size; the helper fails closed on nonsense."""
    from vibeqc.bipole_fock_ewald import exchange_q0_gauge_constant

    with pytest.raises(ValueError):
        exchange_q0_gauge_constant(0.5, 0.5, 100.0, 0)
    with pytest.raises(ValueError):
        exchange_q0_gauge_constant(0.5, 0.0, 100.0, 8)
    with pytest.raises(ValueError):
        exchange_q0_gauge_constant(0.5, 0.5, 0.0, 8)


def test_gauge_constant_matches_the_closed_form():
    from vibeqc.bipole_fock_ewald import exchange_q0_gauge_constant

    xi, alpha, volume, n_k = 0.288147805790, 0.5586807818, 125.887583575, 8
    assert exchange_q0_gauge_constant(xi, alpha, volume, n_k) == pytest.approx(
        xi - np.pi / (alpha * alpha * volume * n_k), rel=0.0, abs=1e-15
    )


# --------------------------------------------------------------------------
# 2. The disclosed finite-size energy, against its closed form.
# --------------------------------------------------------------------------


def test_finite_size_gauge_matches_the_closed_form_rhf():
    """``E_fs = -1/4 a_HF xi_M S_k w_k Tr[D S D S] = -N_e/2 . xi_M`` (RHF).

    ``D = 2 C_occ C_occ^H`` with ``C^H S C = I`` gives
    ``Tr[D S D S] = 2 Tr[D S] = 2 N_e`` at every SCF iteration, so the
    closed form is exact and needs no converged density.
    """
    system, result = _run_rhf("ewald")
    comp = result.energy_components[-1]
    xi_m = probe_charge_madelung_supercell(system, MESH)
    expected = -0.5 * float(system.n_electrons()) * xi_m
    disclosed = getattr(comp, "e_exchange_finite_size", None)
    assert disclosed is not None, (
        "the run applied an exchange-divergence correction worth "
        f"{expected:.6f} Ha and reported no part of it"
    )
    assert disclosed == pytest.approx(expected, rel=1e-10, abs=1e-12)
    # Sanity: the gauge is a real, non-negligible share of the exchange.
    assert abs(disclosed) > 1e-3


def test_finite_size_gauge_is_exactly_zero_with_exxdiv_none():
    """Negative control: the SAME route with the feature off (LEARNINGS L125).

    ``exchange_exxdiv='none'`` zeroes ``xi_M``, so the disclosed term must be
    **exactly** 0.0 -- not merely small. A tolerance band here would pass for
    a value that is only nearly zero, which is the failure the control exists
    to catch. The corrected split itself still runs (``K_LR`` and the
    ``-pi/(alpha^2 V n_k)`` transfer are untouched), so this is the same code
    path with one convention flipped, not a neighbouring route.
    """
    _system, result = _run_rhf("none")
    comp = result.energy_components[-1]
    assert getattr(comp, "e_exchange_finite_size", None) == 0.0
    assert comp.e_exchange is not None  # the split still ran


def test_exxdiv_none_and_ewald_differ_by_exactly_the_disclosed_amount():
    """The disclosure is the *whole* difference the convention makes.

    Both runs are one SCF iteration from the same SAD guess, so the density
    entering the energy expression is identical and the totals differ by the
    gauge term alone.
    """
    _s, res_ewald = _run_rhf("ewald")
    _s2, res_none = _run_rhf("none")
    delta = res_ewald.energy_components[-1].e_total - (
        res_none.energy_components[-1].e_total
    )
    disclosed = getattr(
        res_ewald.energy_components[-1], "e_exchange_finite_size", None
    )
    assert disclosed is not None, (
        f"flipping exchange_exxdiv moves the reported total by {delta:.10f} "
        "Ha and the report names no term accounting for it"
    )
    assert delta == pytest.approx(disclosed, rel=1e-9, abs=1e-11)


# --------------------------------------------------------------------------
# 3. It reaches the user-facing report (the #82 prevention).
# --------------------------------------------------------------------------

_ROW = "of which q=0 finite-size"


def test_energy_components_block_discloses_the_gauge():
    """Behavioural: the rendered report names the gauge and its value.

    Asserted on the rendered text, not on the attribute -- at the parent
    commit this fails because the row is absent from a real report, which is
    the defect, rather than raising ``AttributeError`` on a symbol the fix
    introduces (LEARNINGS L124).
    """
    system, result = _run_rhf("ewald")
    report = format_scf_trace(result, include_properties=False)
    assert _ROW in report, (
        "the energy-components block does not disclose the exchange "
        "finite-size gauge; a cross-code comparison cannot see a term that "
        "is Ha-scale on MgO (#82).\n" + report
    )
    xi_m = probe_charge_madelung_supercell(system, MESH)
    expected = -0.5 * float(system.n_electrons()) * xi_m
    row = next(line for line in report.splitlines() if _ROW in line)
    printed = float(re.search(r"(-?\d+\.\d+)", row).group(1))
    assert printed == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_pure_functional_report_has_no_finite_size_row():
    """A pure functional has no exact exchange, hence no gauge and no row.

    The second half of the negative control: the row must not appear where
    the quantity does not exist, so its presence always means something.
    """
    system, basis = _he_cubic()
    kmesh = monkhorst_pack(system, list(MESH), use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.functional = "lda"
    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    comp = result.energy_components[-1]
    assert getattr(comp, "e_exchange_finite_size", None) is None
    assert _ROW not in format_scf_trace(result, include_properties=False)
