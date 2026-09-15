"""OH·/def2-TZVP UHF ``run_job`` convergence regression.

2026-07 release-paper rerun M02 blocker: with the per-spin DIIS
histories the OH radical UHF/def2-TZVP single point stalled just above
the 1e-6 gradient threshold (residual pinned at ~1e-6..1e-4 for 90+
iterations with the energy already flat to 1e-13) and failed closed
after the default max_iter = 100. Root cause: F_α and F_β are coupled
through the shared Coulomb term J(D_α + D_β), so two independent
per-spin Pulay extrapolations can chase each other's field
indefinitely. Fixed by the spin-coupled DIIS
(``DIIS::extrapolate_spin_coupled``, cpp/src/diis.cpp): one B matrix
over the stacked (e_α; e_β) error, one coefficient set applied to both
Focks. Post-fix the same job converges in ~12 iterations.

Coverage:

1. ``run_job(OH, basis="def2-tzvp", method="uhf")`` with all-default
   options converges, at the PySCF-validated energy and ⟨S²⟩.
2. Fail-closed stays intact: a genuinely non-converged UHF run_job
   raises and does not emit properties / QVF / molden artefacts from
   the non-converged density.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc import Atom, Molecule, UHFOptions, run_job

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _oh_radical() -> Molecule:
    # The M02 release-paper geometry: r(O-H) = 0.97 A.
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )


# Published target: PySCF 2.13 scf.UHF (conv_tol = 1e-11) on the same
# geometry / basis gives E = -75.4222223086842 Ha, <S^2> = 0.75639866.
E_REF_PYSCF = -75.4222223086842
S2_REF_PYSCF = 0.75639866


def test_run_job_oh_uhf_def2tzvp_converges_with_defaults(tmp_path: Path):
    """The exact M02 reproducer: default UHF options, max_iter = 100."""
    stem = tmp_path / "oh-uhf-def2tzvp"
    result = run_job(
        _oh_radical(),
        basis="def2-tzvp",
        method="uhf",
        output=stem,
    )
    assert result.converged
    # Pre-fix this ran the full 100 iterations and failed; post-fix the
    # spin-coupled DIIS needs ~12. 50 leaves generous slack without
    # letting a re-stalled tail sneak back in.
    assert result.n_iter < 50, (
        f"UHF took {result.n_iter} iterations — DIIS tail stall regressed?"
    )
    assert result.energy == pytest.approx(E_REF_PYSCF, abs=5e-9)
    assert result.s_squared == pytest.approx(S2_REF_PYSCF, abs=1e-6)
    assert result.s_squared_deviation == pytest.approx(
        S2_REF_PYSCF - 0.75,
        abs=1e-6,
    )
    text = stem.with_suffix(".out").read_text("utf-8")
    assert "Atomic charges" in text
    assert "Dipole moment" in text


def test_run_job_uhf_nonconverged_fails_closed_no_artifacts(tmp_path: Path):
    """A non-converged UHF run_job must raise and must not emit
    properties / QVF / molden outputs computed from the non-converged
    density (the runner fail-closed contract the M02 job relied on)."""
    stem = tmp_path / "oh-uhf-nonconv"
    opts = UHFOptions()
    opts.max_iter = 3  # guaranteed non-convergence
    with pytest.raises(RuntimeError, match="did not converge"):
        run_job(
            _oh_radical(),
            basis="def2-tzvp",
            method="uhf",
            uhf_options=opts,
            output=stem,
            output_qvf=True,
            write_molden_file=True,
            write_population_file=True,
        )
    for suffix in (".qvf", ".molden", ".population.txt", ".bibtex",
                   ".references"):
        artifact = stem.with_suffix(suffix)
        assert not artifact.exists(), (
            f"{artifact.name} was written from a non-converged density"
        )
    text = stem.with_suffix(".out").read_text("utf-8")
    assert "NOT converged" in text
    assert "FATAL: UHF SCF did not converge" in text
    for heading in ("Atomic charges", "Bond orders", "Dipole moment"):
        assert heading not in text, (
            f"{heading} was computed from a non-converged density"
        )
