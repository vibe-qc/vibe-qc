"""Density-artifact capability contract for ``run_periodic_job`` (#679).

Before this contract, ``write_density=True`` could discard a *completed*
periodic calculation: the density writer dropped the grid as an OPTIONAL
artefact with a warning while the output plan had declared ``{stem}.xsf``
GUARANTEED, so finalization raised ``IncompleteOutputError`` after the SCF had
converged and every other sibling file had been written.

Four distinct mechanisms sat behind that one symptom.  Each is pinned here.

1. **Spin-channel presence must be tested by content, not by attribute.**
   ``CCMRealGammaResult`` and ``CCMFourCentreResult`` are closed-shell
   dataclasses that *declare* ``density_alpha`` / ``density_beta`` as ``None``.
   Testing ``hasattr`` alone routed them into the open-shell branch, which then
   built ``densities = [None, None]`` and discarded the perfectly good
   ``LatticeMatrixSet`` in ``result.density``.

2. **A Gamma-only Bloch mesh carries its density in a one-entry per-k
   container.**  At ``kpoints=(1, 1, 1)`` the multi-k GDF driver returns a list
   holding one block at exactly k = 0, where the Bloch matrix *is* the
   real-space unit-cell matrix.  The Gamma path demanded a rank-2 array, so the
   *same cell* wrote its density with ``kpoints=None`` and was discarded with
   ``kpoints=(1, 1, 1)`` -- a knob that does not change the physics changed the
   artefact outcome.

3. **A real-space supercell-Gamma result has no Bloch mesh.**  The BvK triple
   (``kpoints_cart``, ``kpoint_weights``, ``bvk_mesh``) is metadata *about a
   k-mesh*; the repetition count the runner resolves from
   ``aiccm_lattice_extension`` describes a supercell.  Counting that lone value
   made the triple look half-supplied and refused the artefact.

4. **A route that genuinely cannot produce a density must refuse up front.**
   The plain Gaussian routes return per-k Bloch blocks on a true multi-k mesh
   and the artifact path refuses -- by design -- to invent a real-space lattice
   density from them.  Failing in the first second beats failing after a
   five-hour SCF; this is the discipline ``write_molden_file`` already follows
   through ``_resolve_sidecar_request``.

The electron-count certification (``D:S`` against the expected electron count)
is what makes admitting a block in (2) safe: an admitted density that did not
represent the right charge would still be refused, so these paths cannot pass a
wrong density silently.  Test 5 pins that the certification still bites.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq

A_BOHR = 6.0
D_BOHR = 1.4


def _h2_cell(a: float = A_BOHR):
    """H2 in a cubic box, the cell the issue's 30-second repro uses."""
    return vq.PeriodicSystem(
        3,
        np.diag([a] * 3),
        [
            vq.Atom(1, [a / 2, a / 2, a / 2 - D_BOHR / 2]),
            vq.Atom(1, [a / 2, a / 2, a / 2 + D_BOHR / 2]),
        ],
        0,
        1,
    )


def _basis(cell):
    return vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")


def _run(tmp_path, stem, **kwargs):
    cell = _h2_cell(kwargs.pop("a", A_BOHR))
    out = tmp_path / stem
    vq.run_periodic_job(
        cell,
        _basis(cell),
        output=out,
        write_density=True,
        **kwargs,
    )
    return out


@pytest.mark.slow
def test_gamma_only_kmesh_writes_density_like_the_kpoints_none_route(tmp_path):
    """Mechanism 2: ``kpoints=(1,1,1)`` must not differ from ``kpoints=None``.

    This is the issue's headline reproduction.  Both spellings are the same
    Gamma-only Hamiltonian on the same cell; before the fix the first was
    discarded at finalization and the second succeeded.
    """
    explicit = _run(
        tmp_path, "k111", method="RHF", jk_method="gdf", kpoints=(1, 1, 1)
    )
    implicit = _run(tmp_path, "knone", method="RHF", jk_method="gdf")

    for stem in (explicit, implicit):
        xsf = stem.with_suffix(".xsf")
        assert xsf.exists(), f"{stem.name}: guaranteed density artefact missing"
        assert xsf.stat().st_size > 0

    # The two routes evaluate the same converged density, so the grids agree.
    assert (
        explicit.with_suffix(".xsf").read_text().splitlines()[:4]
        == implicit.with_suffix(".xsf").read_text().splitlines()[:4]
    )


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["real-gamma", "four-center"])
def test_closed_shell_supercell_gamma_variants_write_density(tmp_path, variant):
    """Mechanisms 1 and 3, on the two real-space supercell-Gamma variants.

    ``a=12`` bohr: wide enough that the four-centre construction's returned
    lattice density certifies against the electron count.  Mechanism 1 (the
    ``None`` spin channels) and mechanism 3 (the absent Bloch mesh) are what
    these two routes hit, and both are independent of the box width.
    """
    stem = _run(
        tmp_path,
        f"v_{variant}",
        a=12.0,
        method="aiccm",
        variant=variant,
        initial_guess="HCORE",
    )
    xsf = stem.with_suffix(".xsf")
    assert xsf.exists(), f"{variant}: guaranteed density artefact missing"
    assert xsf.stat().st_size > 0


@pytest.mark.parametrize("jk_method", ["rijcosx", "gpw"])
def test_true_multik_per_k_routes_refuse_write_density_before_scf(
    tmp_path, jk_method
):
    """Mechanism 4: refuse in the first second, not after the SCF.

    These routes return per-k Bloch blocks on a true multi-k mesh and have no
    real-space lattice density to evaluate.  Before the fix each ran its SCF to
    convergence and *then* raised ``IncompleteOutputError``, throwing the
    completed calculation away.
    """
    cell = _h2_cell()
    with pytest.raises(NotImplementedError) as excinfo:
        vq.run_periodic_job(
            cell,
            _basis(cell),
            method="RHF",
            jk_method=jk_method,
            kpoints=(2, 2, 2),
            write_density=True,
            output=tmp_path / f"multik_{jk_method}",
        )
    message = str(excinfo.value)
    assert "write_density" in message
    # The refusal must say what to do instead, not merely that it refused.
    assert "Gamma-only mesh" in message or "lattice density" in message
    # Nothing may be left behind: the refusal precedes the calculation.
    assert not (tmp_path / f"multik_{jk_method}.out").exists()


def test_lattice_density_routes_are_not_gated_at_true_multik(tmp_path):
    """The gate must not catch routes that *do* return a lattice density.

    BIPOLE and the AICCM variants carry a real-space ``LatticeMatrixSet`` on
    any mesh, so a true multi-k mesh is no obstacle for them.  Pinning this
    keeps the gate from growing into a blanket multi-k refusal.
    """
    from vibeqc.periodic_jk_method import PeriodicJKMethod

    cell = _h2_cell()
    # A capability refusal would raise NotImplementedError naming write_density
    # before any compute; a dry run reaches the manifest instead.
    vq.run_periodic_job(
        cell,
        _basis(cell),
        method="RHF",
        jk_method="bipole",
        kpoints=(2, 2, 2),
        write_density=True,
        dry_run=True,
        output=tmp_path / "bipole_multik",
    )
    assert PeriodicJKMethod.BIPOLE is not None


@pytest.mark.slow
def test_uncertifiable_density_is_refused_rather_than_written(tmp_path):
    """Mechanism 2's safety net: D:S certification still bites.

    The four-centre construction in a tight 6-bohr box returns a lattice
    density whose ``sum_g D(g):S(g)`` is 2.229 against 2 electrons -- an error
    that decays with box width (2.0186 at 8 bohr, 2.0010 at 10, certified at
    12).  Whatever else changes, a density that does not represent the right
    charge must never reach an artefact.
    """
    cell = _h2_cell(A_BOHR)
    with pytest.raises(Exception) as excinfo:
        vq.run_periodic_job(
            cell,
            _basis(cell),
            method="aiccm",
            variant="four-center",
            initial_guess="HCORE",
            write_density=True,
            output=tmp_path / "tight_box",
        )
    # It must not be a silent success, and the .xsf must not exist.
    assert not (tmp_path / "tight_box.xsf").exists()
    assert excinfo.value is not None
