"""HF/DFT cross-code parity — vibe-qc vs ORCA 6.1.1.

The ORCA axis of the parity matrix. ``tests/test_parity_hf_dft.py``
certifies vibe-qc's per-intermediate energy decomposition against
PySCF; this file does the same against ORCA — a second, independent
reference.

ORCA runs out-of-process on compute-reference via the ``vq`` queue (CLAUDE.md
§ 10 — vibe-qc never imports ORCA), which is slow and serialised, so
the ORCA results are **cached** as committed JSON under
``examples/regression/parity_matrix_orca/cache/``. This test reads the
cache and runs only the (fast) vibe-qc side live — no ORCA, no vq, no
network in CI. Cells without a cached ORCA result are skipped; refresh
or extend the cache with::

    python -m examples.regression.parity_matrix_orca.run_parity --run

**Combined two-electron bucket.** ORCA's standard output does not
split Coulomb from exchange (see
``examples/regression/parity_matrix_orca/parse_orca.py``), so the
Fock-build comparison is the single ``e_coulomb_plus_exchange`` bucket
rather than separate E_J / E_K — five energy buckets plus the MO
eigenvalues. A failing bucket still localises the discrepancy:
e_nuc / e_1e -> integrals; e_coulomb_plus_exchange -> Fock build;
e_xc -> XC quadrature; MO eigenvalues -> diagonalisation / convergence.
"""
from __future__ import annotations

import pytest

from examples.regression.parity_matrix_orca.cases import PARITY_CELLS, Cell
from examples.regression.parity_matrix_orca.parse_orca import (
    parse_orca_output,  # noqa: F401 - imported to assert it stays importable
)
from examples.regression.parity_matrix_orca.run_parity import load_cache

# The parser's self-check budget — the cached ORCA record must still
# satisfy ORCA's own E_total = E_nuc + E_1e + E_2e identity.
_ORCA_SELF_CHECK_TOL = 1e-8


def test_oh_def2_svp_uks_pbe_uses_hard_open_shell_recipe() -> None:
    """Guard the small open-shell row that otherwise stalls at max_iter."""
    from examples.regression.parity_matrix_orca.vibeqc_compare import (
        _HARD_OPEN_SHELL,
    )

    assert ("OH", "def2-svp") in _HARD_OPEN_SHELL


def _cache_or_skip(cell: Cell) -> dict:
    cached = load_cache(cell)
    if cached is None:
        pytest.skip(
            f"no cached ORCA result for {cell.cell_id} — generate with "
            f"`python -m examples.regression.parity_matrix_orca.run_parity "
            f"--run`"
        )
    return cached


@pytest.mark.parametrize("cell", PARITY_CELLS, ids=[c.cell_id for c in PARITY_CELLS])
def test_parity_vs_orca(cell: Cell) -> None:
    """Each decomposed energy piece must match the cached ORCA reference.

    Skipped when the cell has no cached ORCA result, so the suite stays
    green on a fresh checkout and only tightens as the cache fills in.
    """
    cached = _cache_or_skip(cell)

    # The cached record is keyed to a cell — guard against a misfiled or
    # hand-edited cache JSON.
    rc = cached["cell"]
    assert (rc["system"], rc["basis"], rc["method"], rc["df"]) == (
        cell.system, cell.basis, cell.method, cell.df
    ), f"cache file for {cell.cell_id} describes a different cell: {rc}"

    # The ORCA parser already ran its self-check at generation time;
    # re-assert it here so a tampered cache fails loudly rather than
    # feeding bad reference numbers into the comparison.
    assert abs(cached["e_total_residual"]) < _ORCA_SELF_CHECK_TOL, (
        f"{cell.cell_id}: cached ORCA result fails its own self-check "
        f"(E_nuc+E_1e+E_2e off E_total by {cached['e_total_residual']:.2e})"
    )

    # vibe-qc side imports vibe-qc — kept lazy so an all-skip run (fresh
    # checkout, empty cache) needs nothing installed.
    from examples.regression.parity_matrix_orca.vibeqc_compare import (
        compare, vibeqc_decompose,
    )

    vq = vibeqc_decompose(cell)
    assert vq["converged"], f"vibe-qc {cell.method} did not converge"
    # The decomposition sums to result.energy to SCF-stationarity
    # precision. Direct + DF cells hit ~1e-10 on this; RIJCOSX cells
    # stack DF-J + seminumerical COSX K on top of the XC quadrature,
    # and the per-shell radial cutoff that landed at `50e9b6a` (XC
    # modernization stage 3) introduced a ~1e-8 numerical difference
    # between the SCF's COSX K(D) call and the post-SCF decomposition's
    # COSX K(D) re-build at the same D — worst observed 5.5e-8 on
    # H2CO/def2-svp/RKS-PBE0/RIJCOSX. The cosx floor is therefore set
    # at 1e-7 (2x headroom over the worst observed). The non-cosx
    # cells stay at the tight 1e-9 — a real loss of self-consistency
    # there would be orders of magnitude worse.
    _self_consistency_floor = 1e-7 if cell.cosx else 1e-9
    assert abs(vq["e_total_residual"]) < _self_consistency_floor, (
        f"{cell.cell_id}: vibe-qc decomposition is not self-consistent "
        f"(residual {vq['e_total_residual']:.2e}, "
        f"floor {_self_consistency_floor:.0e})"
    )

    verdict = compare(vq, cached)

    for p in verdict["pieces"]:
        assert p["ok"], (
            f"{cell.cell_id}: {p['piece']} disagrees with ORCA by "
            f"{p['delta']:.3e} Ha (tol {p['tol']:.1e}) — "
            f"vibe-qc {p['vibeqc']:.10f}, ORCA {p['orca']:.10f}"
        )
    mo = verdict["mo"]
    assert mo["ok"], (
        f"{cell.cell_id}: MO eigenvalues disagree with ORCA by "
        f"{mo['max_delta']:.3e} Ha (tol {mo['tol']:.1e}, "
        f"n={mo['n_compared']} compared)"
    )
