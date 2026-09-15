"""Composite 3c methods — end-to-end integration tests.

These tests exercise the full keyword-shortcut dispatch:
``vq.run_job(method="<composite>", ...)`` runs SCF + dispersion + gCP
+ SRB and the per-block contribution is reported in the ``.out``
file. Tests validate against approximate literature reference
energies — exact bit-match is not the goal (grid + SCF tolerances
vary), but landing in the published window is.

Coverage status (post audit 2026-05-31, v0.10.x):

* **RUNNABLE** end-to-end (fully turnkey): HF-3c, PBEh-3c, B97-3c,
  B3LYP-3c, r²SCAN-3c, HSE-3c, wB97X-3c. The gCP and SRB components are
  validated
  against the grimme-lab/gcp Fortran reference (tight component pins in
  test_composites.py):
  - HF-3c    = SCF + D3-BJ + gCP(MINIX, undamped) + SRB(hf3c_base).
  - PBEh-3c  = SCF + D3-BJ + gCP(def2-mSVP, DAMPED).
  - B97-3c   = SCF + D3-BJ + SRB(b973c_mod)   [NO gCP — SRB-only].
  - B3LYP-3c = SCF + D3-BJ + gCP(def2-mSVP, DAMPED) [community recipe].
  - r²SCAN-3c= SCF + D4 + gCP(def2-mTZVPP, DAMPED)  [gCP is NOT zero].
  - HSE-3c   = SCF + D3-BJ + gCP(def2-mSVP, DAMPED, hse-3c constants).
    Its own D3-BJ damping and its own gCP fit, both from ESI Table S1;
    it shares only the parent basis with PBEh-3c.
  - wB97X-3c = SCF + D4 + gCP(vDZP, DAMPED) + inline sidecar ECPs.
"""

from __future__ import annotations

import dataclasses
import math
import re

import pytest

import vibeqc as vq


def _composite_total_block(out: str) -> dict:
    """Parse the per-component lines of the '.out' Composite total block
    into a dict of floats (Hartree)."""
    blk = out[out.index("Composite total"):]
    vals = {}
    for key in ("E_SCF", "E_disp", "E_gCP", "E_SRB", "E_total"):
        m = re.search(rf"{key}\s+([-\d.]+)\s+Ha", blk)
        vals[key] = float(m.group(1)) if m else None
    return vals


# Standard H2O test geometry, in bohr.
def _h2o() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.43, 0.0, 1.10]),
        vq.Atom(1, [-1.43, 0.0, 1.10]),
    ], 0, 1)


def _h2(R: float = 1.4) -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R, 0.0, 0.0]),
    ], 0, 1)


# ---------------------------------------------------------------------------
# RUNNABLE composites — full turnkey path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not vq.dftd4_available(),
                    reason="dftd4 not installed; r²SCAN-3c needs it")
def test_r2scan_3c_h2_runs_end_to_end(tmp_path):
    """Headline v0.9.0 composite: r²SCAN-3c on H2 should run cleanly
    through SCF + D4 + gCP=0. Energy ~-1.17 Ha at R=1.4 bohr."""
    r = vq.run_job(_h2(), method="r2scan-3c",
                   output=str(tmp_path / "r2scan_h2"), progress=False)
    assert r.converged
    assert -1.20 < r.energy < -1.15


@pytest.mark.skipif(not vq.dftd4_available(),
                    reason="dftd4 not installed; r²SCAN-3c needs it")
def test_r2scan_3c_h2o_against_literature(tmp_path):
    """r²SCAN-3c / H2O total energy should land in the published
    window. Grimme et al. 2021 SI reports ~-76.42 Ha for raw SCF at
    this basis; D4 dispersion contributes ~-mHa."""
    r = vq.run_job(_h2o(), method="r2scan-3c",
                   output=str(tmp_path / "r2scan_h2o"), progress=False)
    assert r.converged
    assert -76.50 < r.energy < -76.30


# ---------------------------------------------------------------------------
# PENDING_GCP_DATA composites — SCF + D3(BJ) + SRB run; gCP skipped
# ---------------------------------------------------------------------------

def test_pbeh_3c_runs_with_d3bj_modified_alpha(tmp_path):
    """PBEh-3c: modified PBE0 (αHF=0.42) + def2-mSVP + D3(BJ) +
    composite-specific damping (s8=0, a1=0.486, a2=4.5) + damped
    def2-mSVP gCP. The test asserts the SCF + dispersion path lands
    in the expected literature window."""
    r = vq.run_job(_h2o(), method="pbeh-3c",
                   output=str(tmp_path / "pbeh3c_h2o"), progress=False)
    assert r.converged
    # Grimme 2015 SI reports ~-76.32 Ha for PBEh-3c on H2O (raw SCF;
    # D3 + gCP are smaller perturbations).
    assert -76.40 < r.energy < -76.30
    # D3-BJ disp should be present and small for H2O
    assert hasattr(r, "e_dispersion")
    assert abs(r.e_dispersion) < 0.01     # < 10 mHa


def test_b97_3c_runs_with_srb(tmp_path):
    """B97-3c: B97-GGA + def2-mTZVP + D3(BJ) + 3c-SRB. SRB is wired
    into the dispatcher's additive-corrections block; NO gCP by
    design (SRB-only recipe). Asserts the SCF energy lands close
    to the Brandenburg 2018 SI ~-76.39 Ha reference."""
    r = vq.run_job(_h2o(), method="b97-3c",
                   output=str(tmp_path / "b97_3c_h2o"), progress=False)
    assert r.converged
    assert -76.45 < r.energy < -76.35


def test_b3lyp_3c_runs_end_to_end(tmp_path):
    """B3LYP-3c: standard B3LYP-VWN5 + def2-mSVP + D3(BJ) damping +
    damped def2-mSVP gCP. Community-extension recipe; lands at
    ~-76.36 Ha on H2O at this basis."""
    r = vq.run_job(_h2o(), method="b3lyp-3c",
                   output=str(tmp_path / "b3lyp_3c_h2o"), progress=False)
    assert r.converged
    assert -76.40 < r.energy < -76.30


def test_hf_3c_runs_with_srb(tmp_path):
    """HF-3c: pure HF + MINIX + D3(BJ) Sure-Grimme 2013 damping +
    3c-SRB + undamped MINIX gCP. Run on H2 instead of H2O because
    MINIX is a minimal basis (~7 BF on H2O, less rich on heavier
    elements)."""
    r = vq.run_job(_h2(R=1.4), method="hf-3c",
                   output=str(tmp_path / "hf_3c_h2"), progress=False)
    assert r.converged
    # HF/H2/MINIX ≈ -1.118 Ha; SRB pulls it more bound on bonded pairs.
    assert -1.20 < r.energy < -1.10


# ---------------------------------------------------------------------------
# Composite-specific D3-BJ damping is wired correctly
# ---------------------------------------------------------------------------

def test_composite_d3bj_damping_uses_recipe_not_functional_default():
    """PBEh-3c's D3-BJ damping is re-fit and differs from the standard
    PBE0-D3BJ damping. The dispatcher must use the recipe's damping,
    not fall back to a per-functional lookup that would (a) fail (no
    'pbeh-3c' in the standard table) or (b) silently use PBE0's
    damping, producing wrong numbers."""
    recipe = vq.resolve_composite("pbeh-3c")
    assert recipe is not None
    assert recipe.d3bj_damping is not None
    # PBEh-3c's damping is (s6=1, s8=0, a1=0.4860, a2=4.5000) per
    # Grimme 2015 — distinctively different from PBE0-D3BJ which has
    # s8 > 0.
    assert recipe.d3bj_damping.s8 == 0.0
    assert abs(recipe.d3bj_damping.a1 - 0.4860) < 1e-12
    assert abs(recipe.d3bj_damping.a2 - 4.5000) < 1e-12


def test_hf_3c_d3bj_damping_distinct_from_hf_default():
    """HF-3c carries its own Sure-Grimme 2013 damping
    (s6=1, s8=0.8777, a1=0.4171, a2=2.9149) — distinct from the
    standard plain-HF-D3BJ damping."""
    recipe = vq.resolve_composite("hf-3c")
    assert recipe is not None
    assert recipe.d3bj_damping is not None
    assert abs(recipe.d3bj_damping.s8 - 0.8777) < 1e-12
    assert abs(recipe.d3bj_damping.a2 - 2.9149) < 1e-12


def test_hse_3c_d3bj_damping_matches_published_esi():
    """HSE-3c carries its OWN D3(BJ) damping, re-fit on S66x8 -- it does
    not inherit PBEh-3c's.

    Published targets: Table S1 ("Empirical parameters of the HSE-3c
    method"), E_disp row, of the ESI to Brandenburg, Caldeweyher &
    Grimme, Phys. Chem. Chem. Phys. 18, 15519 (2016),
    doi:10.1039/c6cp01697a:

        s6 = 1.00000 (constrained)   s8 = 0.00000 (constrained)
        a1 = 0.44110                 a2 = 4.51820

    Regression guard: through v0.15.x the recipe shipped PBEh-3c's
    (a1=0.4860, a2=4.5000), so run_job(method="hse-3c") computed a
    dispersion correction that was not HSE-3c's.
    """
    recipe = vq.resolve_composite("hse-3c")
    assert recipe is not None
    assert recipe.d3bj_damping is not None
    assert recipe.d3bj_damping.s6 == 1.0
    assert recipe.d3bj_damping.s8 == 0.0
    assert abs(recipe.d3bj_damping.a1 - 0.4411) < 1e-12
    assert abs(recipe.d3bj_damping.a2 - 4.5182) < 1e-12

    # The two composites share a basis and a short-range potential, which
    # is exactly why the wrong constants looked plausible. Pin them apart.
    pbeh = vq.resolve_composite("pbeh-3c")
    assert pbeh is not None and pbeh.d3bj_damping is not None
    assert recipe.d3bj_damping.a1 != pbeh.d3bj_damping.a1
    assert recipe.d3bj_damping.a2 != pbeh.d3bj_damping.a2


def _esi_benzene() -> vq.Molecule:
    """Gas-phase benzene at the geometry of the '!bz gas' CRYSTAL14 MOLECULE
    input in section B of the HSE-3c ESI (doi:10.1039/c6cp01697a).

    The ESI gives point group 40 (D6h) and two symmetry-unique atoms, in
    Angstrom; both lie at polar angle 30 deg, so the D6h orbit of each is
    r * (cos(30 + 60i), sin(30 + 60i), 0). Expanded here because vibe-qc's
    Molecule takes explicit Cartesians in bohr.

        C  1.201622903718  0.693757306926  0.0   ->  r = 1.38751 A
        H  2.138625514345  1.234736016402  0.0   ->  r = 2.46947 A
    """
    a2b = 1.8897261254578281
    cx, cy = 1.201622903718, 0.693757306926
    hx, hy = 2.138625514345, 1.234736016402
    r_c, th_c = math.hypot(cx, cy), math.atan2(cy, cx)
    r_h, th_h = math.hypot(hx, hy), math.atan2(hy, hx)
    atoms = []
    for z, r, th in ((6, r_c, th_c), (1, r_h, th_h)):
        for i in range(6):
            a = th + i * math.pi / 3.0
            atoms.append(
                vq.Atom(z, [r * math.cos(a) * a2b, r * math.sin(a) * a2b, 0.0])
            )
    return vq.Molecule(atoms, 0, 1)


# Published target: E_gCP = 0.0148400663 a.u. for gas-phase benzene under
# HSE-3c (Brandenburg, Caldeweyher & Grimme, Phys. Chem. Chem. Phys. 18,
# 15519 (2016), ESI section B, Table S2, "gCP" row, "gas [a.u.]" column).
E_GCP_BENZENE_REF = 0.0148400663


def test_hse_3c_gcp_matches_published_benzene():
    """HSE-3c carries its OWN gCP fit constants, and they are the paper's,
    not Grimme's gcp program's.

    Published targets: ESI Table S1, E_gCP row:

        sigma = 1.00000 (constrained)   eta  = 1.40858
        alpha = 0.29083                 beta = 1.95260

    These are arbitrated against the paper's own numerical example rather
    than merely transcribed, because the sources conflict: gcp.f90's
    case ('hse3c') sets eta=1.32378, alpha=0.28314, beta=1.94527, and has
    done so in every release of both the classic and the reworked program.
    Evaluating the ESI's own benzene geometry against the bundled def2-mSVP
    e_mis / n_virt tables reproduces the ESI's published E_gCP only with
    Table S1's constants (to ~2e-10 Ha); gcp.f90's miss it by 1.1e-4 Ha and
    PBEh-3c's by 4.4e-4 Ha. See grimme-lab/gcp issue #17 for the upstream bug.

    Regression guard: through v0.15.x hse-3c silently used PBEh-3c's gCP
    constants, because vibe-qc keyed gCP parameters by basis and the two
    composites share def2-mSVP.
    """
    recipe = vq.resolve_composite("hse-3c")
    assert recipe is not None
    assert recipe.gcp_basis == "def2-msvp"
    assert recipe.gcp_variant == "hse-3c"

    p = vq.gcp_params_for("def2-msvp", variant="hse-3c")
    assert p is not None
    assert abs(p.sigma - 1.00000) < 1e-12
    assert abs(p.eta - 1.40858) < 1e-12
    assert abs(p.alpha - 0.29083) < 1e-12
    assert abs(p.beta - 1.95260) < 1e-12

    # The whole point of the keying: same basis, different fit constants.
    base = vq.gcp_params_for("def2-msvp")
    assert base is not None
    assert base.eta != p.eta
    # ... while the per-element tables stay single-sourced.
    assert base.e_mis == p.e_mis
    assert base.n_virt == p.n_virt

    # The arbitration itself. HSE-3c applies the damped gCP (dmp_scal=4,
    # dmp_exp=6), which the undamped value (~0.106 a.u.) would badly miss.
    assert recipe.gcp_damping is not None
    damping = (recipe.gcp_damping.dmp_scal, recipe.gcp_damping.dmp_exp)
    e = vq.compute_gcp(
        _esi_benzene(), "def2-msvp", variant="hse-3c", damping=damping
    ).energy
    assert abs(e - E_GCP_BENZENE_REF) < 1e-9, (
        f"HSE-3c gCP on the ESI benzene geometry is {e:.10f} a.u.; the "
        f"paper publishes {E_GCP_BENZENE_REF:.10f} a.u."
    )

    # And the sets vibe-qc rejected really do fail to reproduce it.
    for eta, alpha, beta in ((1.32378, 0.28314, 1.94527),   # gcp.f90 hse3c
                             (1.32492, 0.27649, 1.95600)):  # gcp.f90 pbeh3c
        wrong = dataclasses.replace(p, eta=eta, alpha=alpha, beta=beta)
        e_wrong = vq.compute_gcp(
            _esi_benzene(), params=wrong, damping=damping
        ).energy
        assert abs(e_wrong - E_GCP_BENZENE_REF) > 1e-5


def test_gcp_unknown_variant_raises_rather_than_falling_back():
    """A typo'd variant must not silently resolve to the basis-level
    parameter set -- that is exactly how hse-3c came to use PBEh-3c's
    constants. ValueError, not GCPDataMissing: the runner degrades
    gracefully on GCPDataMissing (E_gCP = 0), which would hide this.
    """
    with pytest.raises(ValueError):
        vq.gcp_params_for("def2-msvp", variant="hse3c")   # missing hyphen
    with pytest.raises(ValueError):
        vq.compute_gcp(_h2o(), "def2-msvp", variant="pbeh-3c")  # not registered

    assert vq.available_gcp_variants("def2-msvp") == ["hse-3c"]
    assert vq.available_gcp_variants("minix") == []

    # variant= is a registry selector; it is meaningless with explicit params.
    with pytest.raises(ValueError):
        vq.compute_gcp(
            _h2o(), params=vq.gcp_params_for("def2-msvp"), variant="hse-3c"
        )


def test_pbeh_3c_and_b3lyp_3c_keep_the_basis_level_gcp_constants():
    """Only hse-3c overrides. pbeh-3c is the fit the def2-mSVP file's
    [parameters] block records, and b3lyp-3c deliberately reuses it
    (undamped) for want of a 3c paper of its own.
    """
    for name in ("pbeh-3c", "b3lyp-3c"):
        recipe = vq.resolve_composite(name)
        assert recipe is not None
        assert recipe.gcp_basis == "def2-msvp"
        assert recipe.gcp_variant is None


# ---------------------------------------------------------------------------
# Bases bundled at the expected coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis,expected_min_nbf", [
    ("def2-msvp",   18),    # H2O / def2-mSVP ~ 19 BF
    ("def2-mtzvp",  28),    # H2O / def2-mTZVP ~ 30 BF
    ("def2-mtzvpp", 32),    # H2O / def2-mTZVPP ~ 34 BF
    ("vdzp",        20),    # H2O / vDZP ~ 23 BF
    ("minix",        5),    # H2O / minix ~ 7 BF
])
def test_each_3c_basis_loads_on_h2o(basis, expected_min_nbf):
    """Every newly-bundled 3c parent basis must load via BasisSet on
    H2O and report a sensible basis-function count. This is the
    smoke-check that ECP separation, libint dispatch, and the .g94
    file headers are all clean."""
    bs = vq.BasisSet(_h2o(), basis)
    assert bs.nbasis >= expected_min_nbf, (
        f"{basis} on H2O: only {bs.nbasis} basis functions "
        f"(expected at least {expected_min_nbf})"
    )


# ---------------------------------------------------------------------------
# End-to-end composite TOTAL assembly (audit F1.1/F1.2/F1.4/F1.5)
# ---------------------------------------------------------------------------

def test_hf3c_composite_total_assembly(tmp_path):
    """hf-3c on H2O exercises the whole stack: RHF (not FCI — F1.1) +
    D3-BJ (must NOT be dropped — F1.2) + undamped MINIX gCP (F1.5) +
    canonical SRB (F1.4). The composite total must equal the sum of its
    parts, and the gCP/SRB components must match the mctc-gcp reference
    (+0.029207 Ha gCP, -0.035456 Ha SRB on this geometry)."""
    r = vq.run_job(_h2o(), method="hf-3c",
                   output=str(tmp_path / "hf3c"), progress=False)
    assert r.converged
    out = (tmp_path / "hf3c.out").read_text()
    assert "fci" not in out.lower()                       # F1.1
    v = _composite_total_block(out)
    assert abs(v["E_disp"]) > 1e-9                         # F1.2: D3-BJ present
    assert v["E_gCP"] == pytest.approx(0.0292067, abs=1e-4)   # F1.5 (minix)
    assert v["E_SRB"] == pytest.approx(-0.0354562, abs=1e-4)  # F1.4 (hf3c SRB)
    assert v["E_total"] == pytest.approx(
        v["E_SCF"] + v["E_disp"] + v["E_gCP"] + v["E_SRB"], abs=1e-6)


@pytest.mark.skipif(not vq.dftd4_available(),
                    reason="dftd4 not installed; r²SCAN-3c needs it")
def test_r2scan3c_composite_total_assembly(tmp_path):
    """r²SCAN-3c on H2O: SCF + D4 + DAMPED gCP. The gCP component is
    NOT zero (audit F1.5) — it is +0.001787 Ha (+1.12 kcal/mol),
    matching mctc-gcp. Total = SCF + D4 + gCP."""
    r = vq.run_job(_h2o(), method="r2scan-3c",
                   output=str(tmp_path / "r2"), progress=False)
    assert r.converged
    v = _composite_total_block((tmp_path / "r2.out").read_text())
    assert v["E_gCP"] == pytest.approx(0.001787149, abs=1e-5)
    assert v["E_SRB"] == pytest.approx(0.0, abs=1e-12)     # no SRB
    assert v["E_total"] == pytest.approx(
        v["E_SCF"] + v["E_disp"] + v["E_gCP"] + v["E_SRB"], abs=1e-6)


def test_b973c_total_has_srb_no_gcp(tmp_path):
    """B97-3c total = SCF + D3-BJ + SRB, with NO gCP (audit F1.5)."""
    r = vq.run_job(_h2o(), method="b97-3c",
                   output=str(tmp_path / "b97"), progress=False)
    assert r.converged
    v = _composite_total_block((tmp_path / "b97.out").read_text())
    assert v["E_gCP"] == pytest.approx(0.0, abs=1e-12)          # no gCP
    assert v["E_SRB"] == pytest.approx(-0.005713243, abs=1e-4)  # SRB present
    assert v["E_total"] == pytest.approx(
        v["E_SCF"] + v["E_disp"] + v["E_gCP"] + v["E_SRB"], abs=1e-6)
