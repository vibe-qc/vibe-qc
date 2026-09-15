"""Scalable C++ CCM four-center J/K builder (``run_ccm_rhf_scalable``).

The scalable driver uses the C++ ``build_jk_ccm_weighted`` (method
``bra_home_full``) instead of the dense Python ``ccm_eri`` tensor. It must
reproduce the validated padded route (:func:`run_ccm_rhf`) — which itself
reproduces the gold H₄ value −0.542875 Ha/atom (Peintinger & Bredow 2014,
Tab. 2). This pins the two bugs the route had to clear:

  * the K-channel must contract the *one* bra-ket-symmetrised effective tensor
    (not a separately-weighted K tensor), and
  * the symmetrisation must not alias (``V = 0.5*(V + V.transpose())`` aliases
    in Eigen and silently yields an asymmetric tensor → non-convergent SCF).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, PeriodicSystem, RHFOptions, run_rhf
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _cell(lattice, atoms):
    mult = 1 if sum(a.Z for a in atoms) % 2 == 0 else 2
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=mult)


def _h4_unit():
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    return _cell(np.diag([4.0 * BOHR, 40.0, 40.0]), [Atom(1, p) for p in pos])


def test_scalable_isolated_equals_molecular():
    """Isolated (huge-cell) cluster: scalable CCM == molecular RHF."""
    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    res = run_ccm_rhf_scalable(ccm)
    mol = ccm.supercell
    ref = run_rhf(mol, BasisSet(mol, "sto-3g"), RHFOptions())
    assert res.converged
    assert res.energy == pytest.approx(ref.energy, abs=1e-7)


def test_scf_result_backend_identity_and_hold_flag():
    """IID 344: every CCM SCF result identifies its executing route/operator
    (``backend``) and reports the parity-hold state as a structured boolean
    (``parity_held``). The four-center cluster routes can never hit the
    multi-k GDF dense-core hold, so they must POSITIVELY report held=False --
    the aic7 demotion was total precisely because good rows and held rows
    were indistinguishable on the record surface."""
    from vibeqc.periodic.ccm.scf import _CCM_ERI_METHODS

    ccm = CCMSystem(_cell(np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]), (1, 1, 1), "sto-3g")
    sc = run_ccm_rhf_scalable(ccm)
    assert sc.backend == "ccm-fourcenter-direct-union12-rhf"
    assert sc.parity_held is False
    pad = run_ccm_rhf(ccm)
    assert pad.backend == "ccm-fourcenter-dense-union12-rhf"
    assert pad.parity_held is False
    # A caller-injected effective tensor overrides ``method``, so the result
    # must not claim a weighting the run did not apply.
    inj = run_ccm_rhf(ccm, eri=_CCM_ERI_METHODS["union12"](ccm))
    assert inj.backend == "ccm-fourcenter-dense-injected-rhf"
    assert inj.parity_held is False


def test_scalable_matches_padded_h4():
    """H₄ chain: the C++ scalable builder reproduces the padded route to µHa
    and converges (DIIS, no damping crutch). The padded route is the validated
    −0.542875 reference, so this transitively checks the gold value."""
    unit = _h4_unit()
    ccm = CCMSystem(unit, (4, 1, 1), "sto-3g")
    sc = run_ccm_rhf_scalable(ccm)
    pad = run_ccm_rhf(ccm)
    assert sc.converged
    assert sc.energy_per_atom == pytest.approx(pad.energy_per_atom, abs=5e-6)
    # Fock from the builder must be Hermitian (the aliasing bug made it not).
    F = np.asarray(sc.fock, float)
    assert np.linalg.norm(F - F.T) < 1e-8


def test_scalable_matches_padded_2d():
    """Multi-dimensional parity: the C++ lattice-sum builder reproduces the dense
    four-center on a genuine **2-D** lattice (not just 1-D), to machine ε.

    This is the resolution of the c-diamond over-binding question: ``scalable``
    over-binds vs the neutral RI *because the bare four-center over-binds 3-D*
    (the Madelung self-image), NOT because the C++ kernel is wrong — it
    reproduces the dense bare four-center exactly in every dimension the dense
    path can still be built (1-D above, 2-D here; 3-D is un-storable densely).
    A minimal 2-D He square lattice keeps the dense padded ERI small (~0.75 GiB).
    """
    unit = PeriodicSystem(
        2, np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 12.0]]),
        [Atom(2, [0, 0, 0])], charge=0, multiplicity=1)
    ccm = CCMSystem(unit, (2, 2, 1), "sto-3g")
    sc = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a")  # default: integral-direct
    full = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a", four_center="full")
    pad = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    assert sc.converged and pad.converged and full.converged
    # full (dense effective tensor) reproduces the padded reference,
    assert full.energy == pytest.approx(pad.energy, abs=1e-9)
    # and the integral-direct default reproduces full in 2-D (the Phase-3b gate).
    assert sc.energy == pytest.approx(full.energy, abs=1e-9)
    assert sc.energy == pytest.approx(pad.energy, abs=1e-9)


@pytest.mark.parametrize("method", ["union12", "aiccm2026dev-a"])
def test_scalable_direct_matches_full(method):
    """Phase 3b: the integral-direct J/K (``four_center="direct"``, the default)
    reproduces the dense effective-tensor path (``four_center="full"``) -- and the
    Python padded reference (:func:`run_ccm_rhf`) -- on the 1-D H₄ chain, for both
    four-center weights. ``"full"`` is the preserved O(nbf**4) comparison
    reference (it builds the explicit symmetrised effective tensor); ``"direct"``
    folds each weighted quartet straight into J/K at O(nbf**2) memory and is what
    lets real 3-D production-basis cells run. The two agree to ~1e-12 (a
    summation reorder), so this pins direct against the dense reference."""
    ccm = CCMSystem(_h4_unit(), (4, 1, 1), "sto-3g")
    pad = run_ccm_rhf(ccm, method=method)
    full = run_ccm_rhf_scalable(ccm, method=method, four_center="full")
    direct = run_ccm_rhf_scalable(ccm, method=method, four_center="direct")
    assert pad.converged and full.converged and direct.converged
    assert full.energy_per_atom == pytest.approx(pad.energy_per_atom, abs=5e-6)
    assert direct.energy_per_atom == pytest.approx(full.energy_per_atom, abs=1e-9)
    Fd = np.asarray(direct.fock, float)
    assert np.linalg.norm(Fd - np.asarray(full.fock, float)) < 1e-8
    assert np.linalg.norm(Fd - Fd.T) < 1e-8


@pytest.mark.slow
def test_scalable_h4_converges_to_gold():
    """H₄ chain energy/atom approaches the gold −0.542875 with cluster size,
    matching the padded route bit-for-bit at each size."""
    unit = _h4_unit()
    for nrep in [(6, 1, 1), (8, 1, 1)]:
        ccm = CCMSystem(unit, nrep, "sto-3g")
        sc = run_ccm_rhf_scalable(ccm)
        pad = run_ccm_rhf(ccm)
        assert sc.converged
        assert sc.energy_per_atom == pytest.approx(pad.energy_per_atom, abs=5e-6)
        assert sc.energy_per_atom == pytest.approx(-0.542875, abs=1e-4)


# c-diamond fcc primitive (2 C/cell), a = 3.567 Å.
_CDIA = """
import json, resource, numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable
BOHR = 1.0 / 0.529177210903
a = 3.567 * BOHR
lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
unit = PeriodicSystem(3, lat, [Atom(6, [0, 0, 0]), Atom(6, [a / 4, a / 4, a / 4])],
                      charge=0, multiplicity=1)
ccm = CCMSystem(unit, (2, 2, 2), "sto-3g")          # nbf = 80
res = run_ccm_rhf_scalable(ccm, method="aiccm2026dev-a",
                           four_center={fc!r}, max_iter=1)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({{"energy": float(res.energy), "maxrss": int(peak)}}))
"""


def _scalable_cdiamond_rss(four_center):
    """Run one JK build of c-diamond (2,2,2)/STO-3G (nbf=80) in a fresh subprocess
    (``max_iter=1``, ``OMP_NUM_THREADS=2``) and return ``(energy, peak_rss_gib)``.
    A subprocess isolates the peak RSS to this one calculation."""
    import json
    import os
    import platform
    import subprocess
    import sys

    env = dict(os.environ, OMP_NUM_THREADS="2")
    out = subprocess.run(
        [sys.executable, "-c", _CDIA.format(fc=four_center)],
        capture_output=True, text=True, env=env, timeout=900,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    # ru_maxrss is bytes on Darwin, KiB on Linux.
    div = 1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2
    return info["energy"], info["maxrss"] / div


@pytest.mark.slow
def test_scalable_direct_memory_regression():
    """Phase 3b memory regression on c-diamond (2,2,2)/STO-3G (nbf=80) — the
    canonical cell that OOMs the dense Python four-center (55 800 TiB padded ERI).

    The integral-direct path (``four_center="direct"``, the default) holds its
    JK-build working set to O(nbf**2); the ``"full"`` path materialises the
    O(nbf**4) effective tensor ((n_threads+2)·nbf**4·8 B). Measured here at
    ``OMP_NUM_THREADS=2``: direct ≈ 0.23 GiB vs full ≈ 1.55 GiB. This pins that
    (a) direct stays lean, (b) the ``"full"`` path really does allocate the
    tensor — so the contrast is meaningful, not a too-small cell — and (c) the two
    give the *same* energy on a genuine 3-D cell. The authoritative
    production-basis (pob-tzvp-rev2, ~300 GiB full vs lean direct) validation is a
    vq job, not this in-repo test."""
    e_direct, rss_direct = _scalable_cdiamond_rss("direct")
    e_full, rss_full = _scalable_cdiamond_rss("full")
    # direct reproduces the full effective tensor byte-for-byte on a 3-D cell
    assert e_direct == pytest.approx(e_full, abs=1e-9)
    # the full path really allocates the O(nbf**4) tensor; direct does not
    assert rss_full > 1.0, f"full peak RSS {rss_full:.2f} GiB — tensor not built?"
    assert rss_direct < 0.6, f"direct peak RSS {rss_direct:.2f} GiB — nbf**4 regression?"
    assert rss_direct < 0.5 * rss_full


@pytest.mark.parametrize("method", ["union12", "aiccm2026dev-a"])
def test_scalable_direct_schwarz_screening_opt_in(method):
    """Opt-in Cauchy-Schwarz screening on the integral-direct kernels:

    * ``schwarz_threshold=0.0`` (the default) is **off** -- the direct kernel is
      exact, reproducing the ``"full"`` effective tensor (the integral-direct
      guarantee is preserved by default).
    * a tight threshold (``1e-12``) reproduces the unscreened energy to ~1e-9 while
      skipping shell-quartets below the rigorous ``|w|·Q_bra·Q_ket·D_max`` bound (a
      throughput lever; ~1.1× on c-diamond (2,2,2)/STO-3G, more on diffuse bases).
    """
    unit = PeriodicSystem(
        2, np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 12.0]]),
        [Atom(2, [0, 0, 0])], charge=0, multiplicity=1)
    ccm = CCMSystem(unit, (2, 2, 1), "sto-3g")
    off = run_ccm_rhf_scalable(ccm, method=method, four_center="direct")  # default: off
    full = run_ccm_rhf_scalable(ccm, method=method, four_center="full")
    screened = run_ccm_rhf_scalable(
        ccm, method=method, four_center="direct", schwarz_threshold=1e-12)
    assert off.converged and full.converged and screened.converged
    # default-off direct is exact vs the full tensor
    assert off.energy == pytest.approx(full.energy, abs=1e-9)
    # a tight screen reproduces the unscreened result
    assert screened.energy == pytest.approx(off.energy, abs=1e-9)
