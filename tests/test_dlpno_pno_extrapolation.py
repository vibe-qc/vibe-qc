"""Complete-PNO-space extrapolation, Altun, Neese and Bistoni (2020).

The historical attempt in this code base corrected a single truncated DLPNO
calculation with an estimate of the discarded PNO tail. On the retained
all-electron ``residual_domain="pair", n_frozen=0, tcut_mkn=0`` witness, that
failed because the truncation error was not sign-definite, so an additive
estimate carrying one sign could not fix both over- and under-correlating
systems. The published scheme extrapolates from *two* thresholds instead,
which makes no sign assumption at all.

The algebra is pinned against the paper's own published fit constants, so a
regression here means the implementation drifted from the paper rather than
from some locally invented reference.
"""

from __future__ import annotations

import pytest
from vibeqc.dlpno.pno_extrapolation import (
    PNO_EXTRAPOLATION_BETA,
    RECOMMENDED_F,
    extrapolate_pno_limit,
    extrapolation_factor,
    threshold_exponent,
)

# Altun et al. (2020), Figure 1: benzene dimer, TightPNO/aug-cc-pVDZ-DK.
# E^X = E + A * X**-beta fitted with R^2 = 1.000.
PAPER_LIMIT = -1.78656
PAPER_A = 84.92
PAPER_BETA = 5.55


def _paper_energy(x):
    """`E^X` from the paper's fitted Eq. (1) constants."""
    return PAPER_LIMIT + PAPER_A * x ** (-PAPER_BETA)


def test_beta_matches_the_published_fit():
    assert PNO_EXTRAPOLATION_BETA == PAPER_BETA
    assert RECOMMENDED_F == 1.5


def test_threshold_exponent_maps_tcut_pno_to_x():
    assert threshold_exponent(1e-6) == pytest.approx(6.0)
    assert threshold_exponent(1e-5) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="must be positive"):
        threshold_exponent(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        threshold_exponent(-1e-6)


@pytest.mark.parametrize("x,y", [(5, 6), (6, 7), (7, 8)])
def test_analytic_factor_inverts_the_convergence_law_exactly(x, y):
    """Eq. (4) with the Eq. (3) factor must recover the limit of Eq. (1).

    Generate `E^X` and `E^Y` from the paper's own fitted convergence law, then
    extrapolate. Because Eq. (2) is the exact elimination of `A` between two
    instances of Eq. (1), the limit must come back to machine precision. This
    is what pins the algebra: a sign slip or a swapped X/Y would show up here
    immediately.
    """
    e_x, e_y = _paper_energy(x), _paper_energy(y)
    f = extrapolation_factor(x, y)
    assert extrapolate_pno_limit(e_x, e_y, f=f) == pytest.approx(
        PAPER_LIMIT, abs=1e-12
    )


def test_extrapolation_beats_both_of_its_inputs():
    """The point of the scheme: the extrapolate is closer than either input."""
    e_x, e_y = _paper_energy(5), _paper_energy(6)
    extrapolated = extrapolate_pno_limit(e_x, e_y, f=extrapolation_factor(5, 6))
    assert abs(extrapolated - PAPER_LIMIT) < abs(e_y - PAPER_LIMIT)
    assert abs(extrapolated - PAPER_LIMIT) < abs(e_x - PAPER_LIMIT)
    # The raw truncated values really are far off, so the win is not trivial.
    assert abs(e_x - PAPER_LIMIT) > 1e-2
    assert abs(e_y - PAPER_LIMIT) > 1e-3


@pytest.mark.parametrize("x,y,expected", [(5, 6, 1.5712), (6, 7, 1.7393)])
def test_analytic_factor_is_not_the_recommended_flat_value(x, y, expected):
    """`F = 1.5` and the analytic `F` target different things, and differ.

    The analytic Eq. (3) factor extrapolates to the complete-PNO limit *of the
    DLPNO curve itself*. The paper's flat 1.5 is fitted to minimise error
    against **canonical CCSD(T)**, which the complete-PNO limit still differs
    from through every other DLPNO approximation (domains, TCutPairs,
    TCutMKN). So 1.5 is not an approximation to the analytic value and should
    not be "corrected" towards it; they answer different questions. Pinned so
    nobody reconciles them.
    """
    assert extrapolation_factor(x, y) == pytest.approx(expected, abs=1e-4)
    assert abs(extrapolation_factor(x, y) - RECOMMENDED_F) > 0.05
    # Both stay inside the 1.5 +/- 0.2 band the paper reports for 5/6.
    if (x, y) == (5, 6):
        assert abs(extrapolation_factor(x, y) - RECOMMENDED_F) < 0.2


def test_default_factor_is_the_paper_recommendation():
    e_x, e_y = -1.0, -1.1
    assert extrapolate_pno_limit(e_x, e_y) == pytest.approx(
        e_x + RECOMMENDED_F * (e_y - e_x)
    )


def test_extrapolation_is_sign_agnostic():
    """The property the additive tail estimate could not have.

    Whether the truncated energies approach the limit from above or below, the
    same formula applies: it fits the approach rather than adding a correction
    of an assumed sign. Mirroring the sequence reverses the direction of
    convergence, and the extrapolate follows.
    """
    x, y = 5, 6
    f = extrapolation_factor(x, y)

    # The paper's fit has A > 0, so truncated energies sit *above* the limit
    # (less negative): the under-correlating case, H2O-like.
    under_x, under_y = _paper_energy(x), _paper_energy(y)
    assert under_x > PAPER_LIMIT and under_y > PAPER_LIMIT
    assert extrapolate_pno_limit(under_x, under_y, f=f) == pytest.approx(
        PAPER_LIMIT, abs=1e-12
    )

    # The same law with A < 0 puts them *below* the limit (more negative):
    # the over-correlating case, N2-like. Identical formula, no sign input.
    over_x = PAPER_LIMIT - PAPER_A * x ** (-PAPER_BETA)
    over_y = PAPER_LIMIT - PAPER_A * y ** (-PAPER_BETA)
    assert over_x < PAPER_LIMIT and over_y < PAPER_LIMIT
    assert extrapolate_pno_limit(over_x, over_y, f=f) == pytest.approx(
        PAPER_LIMIT, abs=1e-12
    )


def test_ordering_of_thresholds_is_enforced():
    """`Y` must be the tighter threshold; a swap is an error, not a guess."""
    with pytest.raises(ValueError, match="larger exponent"):
        extrapolation_factor(6, 5)
    with pytest.raises(ValueError, match="larger exponent"):
        extrapolation_factor(6, 6)
    with pytest.raises(ValueError, match="must be positive"):
        extrapolation_factor(0, 6)


# --- Retained negative control for the legacy pair-residual recipe ----------


@pytest.mark.slow
def test_legacy_pair_residual_pno_convergence_is_not_monotone():
    """The former pair-residual recipe violates Eq. (1)'s precondition.

    Altun's extrapolation assumes the correlation energy approaches the
    complete-PNO limit smoothly and monotonically in ``X = -log10(TCutPNO)``;
    their Figure 1 fit has ``R^2 = 1.000``. Measured 2026-08-03, the explicit
    historical ``residual_domain="pair", n_frozen=0, tcut_mkn=0`` recipe did
    not do that on def2-SVP. Error against the full-PNO-space energy of the
    same recipe, in uHa:

        X:        5        6        7        8        9
        H2O:  -3662     -867     +158     +195     +120
        N2:   -4657     +198     -964     +536     +187

    The sign flips repeatedly in exactly the ``X = 5..8`` window where a 5/6 or
    6/7 extrapolation would be taken, so the scheme extrapolates noise rather
    than a trend, and in practice lands further from the limit than its own
    tighter input (H2O 6/7: raw ``E^7`` is ``+158`` uHa, the extrapolate
    ``+916``).

    This is a retained negative control for that legacy pair-domain policy,
    not a claim about the current atom-based ``extended`` default and not
    evidence that extrapolation is validated there. The operative default has
    its own monotonic threshold-ladder regression.
    """
    import numpy as np
    from vibeqc import BasisSet, RHFOptions, run_rhf
    from vibeqc._vibeqc_core import Atom, Molecule
    from vibeqc.density_fitting import DensityFitting
    from vibeqc.dlpno.ccsd_local_solver import (
        LocalCCSDOptions,
        run_local_dlpno_ccsd,
    )

    bohr = 1.8897259886
    atoms = [
        (7, [0.0, 0.0, 0.0]),
        (7, [0.0, 0.0, 1.0977 * bohr]),
    ]
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    scf = RHFOptions()
    scf.max_iter = 200
    rhf = run_rhf(mol, basis, scf)
    assert rhf.converged
    df = DensityFitting(
        basis, BasisSet(mol, "def2-svp-rifit"), aux_basis_name="def2-svp-rifit"
    )

    def e_corr(tcut):
        result = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            LocalCCSDOptions(
                n_frozen=0,
                tcut_pno=tcut,
                tcut_pairs=0.0,
                tcut_mkn=0.0,
                residual_domain="pair",
                compute_triples=False,
            ),
        )
        assert result.converged
        return result.e_corr

    limit = e_corr(0.0)
    errors = [e_corr(10.0 ** (-x)) - limit for x in (5, 6, 7, 8)]

    # Not monotone: the sequence is neither increasing nor decreasing.
    assert errors != sorted(errors)
    assert errors != sorted(errors, reverse=True)
    # And it genuinely changes sign rather than merely wobbling in magnitude.
    signs = {e > 0 for e in errors}
    assert signs == {True, False}
