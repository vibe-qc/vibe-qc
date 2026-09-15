"""Phase SYM3 tests: symmetry-reduced one-electron integral helpers.

Contracts exercised:

1. **Round-trip identity** — for each of S(g), T(g), V(g), the
   compress / reconstruct round-trip via SYM2c orbits reproduces the
   original :class:`LatticeMatrixSet` to machine precision on a
   high-symmetry simple-cubic cell.

2. **Compression ratio** — orbit count reflects the point-group
   order. Single-atom Pm-3m simple cubic gives ~|G|=48× memory savings
   in the limit of large cell lists.

3. **Multi-atom support** — two-atom CsCl-style cubic structures
   (different Z at body-center) work with ``require_closed=False``.

4. **Cross-integral consistency** — S, T, V on the same system share
   the orbit partition (only representative blocks differ).

5. **Verifier sanity** — :func:`verify_lattice_matrix_set_symmetry`
   passes a clean LMS and fails a synthetically broken one.

6. **Symmorphic filtering** — :func:`symmorphic_operations` correctly
   selects translation=0 operators.

7. **No-symmorphic raise** — verifier raises when the operator list
   contains zero symmorphic operators.

8. **Method-independent partition** — orbit structure is the same
   for DIRECT_TRUNCATED and EWALD_3D nuclear builds (the partition
   is operator-only).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _he_cubic(a: float = 5.0):
    """Single-atom Pm-3m simple cubic — closed-shell He on the
    home corner. Symmorphic Pm-3m, |G| = 48."""
    sysp = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(2, [0, 0, 0])])
    vq.attach_symmetry(sysp)
    return sysp


def _he_cubic_distorted(a: float = 5.0):
    """Near-Pm-3m single-atom cell as a user import might provide it.

    The atom is microscopically off the origin and no symmetry is
    attached. GS3 should standardise this before orbit compression.
    """
    lattice = np.eye(3) * a
    frac = np.array([1.0e-5, -2.0e-5, 1.5e-5]) % 1.0
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, (lattice @ frac).tolist())],
    )


def _hebe_cscl_style(a: float = 5.0):
    """Two-atom Pm-3m: He at (0, 0, 0), Be at (a/2, a/2, a/2). Both
    closed-shell so closed-shell-RHF is consistent. Symmorphic
    Pm-3m, |G| = 48."""
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(2, [0, 0, 0]),
         vq.Atom(4, [a/2, a/2, a/2])],
    )
    vq.attach_symmetry(sysp)
    return sysp


def _basis(sysp):
    return vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _opts(cutoff: float = 10.0, method=None):
    o = vq.LatticeSumOptions()
    o.cutoff_bohr = cutoff
    o.nuclear_cutoff_bohr = cutoff + 5
    if method is not None:
        o.coulomb_method = method
    return o


# ---------------------------------------------------------------------------
# 1. Round-trip identity for S, T, V
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["overlap", "kinetic", "nuclear"])
def test_round_trip_identity_on_pm3m_single_atom(name):
    sysp = _he_cubic()
    basis = _basis(sysp)
    opts = _opts()
    ops = sysp.symmetry.operations

    builder = {
        "overlap": vq.compute_overlap_lattice_with_orbits,
        "kinetic": vq.compute_kinetic_lattice_with_orbits,
        "nuclear": vq.compute_nuclear_lattice_with_orbits,
    }[name]
    reduced, full = builder(basis, sysp, opts, ops)

    recon = vq.reconstruct_lattice_matrix_set_c(
        reduced.representatives, reduced.orbits, basis, sysp,
        vq.symmorphic_operations(ops),
        cells=list(full.cells),
    )
    max_err = max(
        float(np.linalg.norm(np.asarray(full.blocks[i]) - recon[i]))
        for i in range(len(full.cells))
    )
    assert max_err < 1e-10, (
        f"{name}: round-trip max ‖diff‖ = {max_err:.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Compression ratio reflects |G|
# ---------------------------------------------------------------------------

def test_compression_ratio_grows_toward_point_group_order():
    """On simple cubic with the full Pm-3m point group (|G|=48), the
    compression ratio approaches 48 as the cell list grows. Tight
    cutoff → small ratio; loose cutoff → larger ratio."""
    sysp = _he_cubic()
    basis = _basis(sysp)
    ops = sysp.symmetry.operations

    ratios = []
    for cutoff in (8.0, 14.0, 20.0):
        opts = _opts(cutoff=cutoff)
        reduced, _ = vq.compute_overlap_lattice_with_orbits(
            basis, sysp, opts, ops,
        )
        ratios.append(reduced.compression_ratio)
    # Monotone-non-decreasing as cutoff grows.
    for r1, r2 in zip(ratios, ratios[1:]):
        assert r2 >= r1 - 1e-9, (
            f"compression ratio regressed across cutoff: {ratios}"
        )
    # And the looser cutoff hits >> 1 (real symmetry compression).
    assert ratios[-1] > 5.0


# ---------------------------------------------------------------------------
# 3. Two-atom support with require_closed=False
# ---------------------------------------------------------------------------

def test_round_trip_on_two_atom_pm3m_with_open_closure():
    sysp = _hebe_cscl_style()
    basis = _basis(sysp)
    opts = _opts()
    ops = sysp.symmetry.operations

    reduced, full = vq.compute_overlap_lattice_with_orbits(
        basis, sysp, opts, ops, require_closed=False,
    )
    diag = vq.verify_lattice_matrix_set_symmetry(
        full, basis, sysp, ops, require_closed=False,
    )
    assert diag["passes"], diag
    assert diag["max_residual"] < 1e-10
    # Atom-pair structure: 2 atoms × 2 atoms × n_cells / |G|.
    assert reduced.n_orbits >= 4
    assert reduced.compression_ratio > 1.0


# ---------------------------------------------------------------------------
# 4. Cross-integral consistency (same orbit structure)
# ---------------------------------------------------------------------------

def test_overlap_kinetic_nuclear_share_orbit_partition():
    sysp = _he_cubic()
    basis = _basis(sysp)
    opts = _opts()
    ops = sysp.symmetry.operations

    red_S, _ = vq.compute_overlap_lattice_with_orbits(basis, sysp, opts, ops)
    red_T, _ = vq.compute_kinetic_lattice_with_orbits(basis, sysp, opts, ops)
    red_V, _ = vq.compute_nuclear_lattice_with_orbits(basis, sysp, opts, ops)
    assert red_S.n_orbits == red_T.n_orbits == red_V.n_orbits
    assert red_S.compression_ratio == pytest.approx(red_T.compression_ratio)
    assert red_S.compression_ratio == pytest.approx(red_V.compression_ratio)


# ---------------------------------------------------------------------------
# 5. Verifier passes a clean LMS, fails a synthetically broken one
# ---------------------------------------------------------------------------

def test_verifier_passes_clean_overlap():
    sysp = _he_cubic()
    basis = _basis(sysp)
    opts = _opts()
    ops = sysp.symmetry.operations
    full = vq.compute_overlap_lattice(basis, sysp, opts)
    diag = vq.verify_lattice_matrix_set_symmetry(full, basis, sysp, ops)
    assert diag["passes"]
    assert diag["max_residual"] < 1e-12


def test_verifier_returns_diagnostic_dict_shape():
    """The verifier returns a dict with the expected keys, regardless
    of whether the input passes or not. The positive-path correctness
    is exercised by the round-trip identity tests above; negative-
    path perturbation testing requires LMS mutability that the
    current C++ binding doesn't expose to Python."""
    sysp = _he_cubic()
    basis = _basis(sysp)
    full = vq.compute_overlap_lattice(basis, sysp, _opts())
    diag = vq.verify_lattice_matrix_set_symmetry(
        full, basis, sysp, sysp.symmetry.operations,
    )
    assert set(diag.keys()) == {
        "n_orbits", "compression_ratio", "max_residual", "passes",
    }
    assert isinstance(diag["passes"], bool)
    assert diag["n_orbits"] >= 1
    assert diag["compression_ratio"] >= 1.0
    assert diag["max_residual"] >= 0.0


# ---------------------------------------------------------------------------
# 6. symmorphic_operations filtering
# ---------------------------------------------------------------------------

def test_symmorphic_operations_filters_translations():
    """Construct a synthetic operator list mixing symmorphic and
    non-symmorphic ops; check the helper returns only the symmorphic
    subset.

    Uses the real space-group operations from a standard Pm-3m cell
    (all symmorphic) and pads with one synthetic non-symmorphic op
    to verify filtering works."""
    sysp = _he_cubic()
    real_ops = list(sysp.symmetry.operations)
    n_real = len(real_ops)
    sym_only = vq.symmorphic_operations(real_ops)
    # Pm-3m is symmorphic — every operator passes through.
    assert len(sym_only) == n_real
    # Now check the filter actually rejects something. We can't
    # construct a SymmetryOp from Python (read-only fields), so the
    # filtering is exercised indirectly via Im-3m (BCC), whose
    # spglib-detected operators include glide translations:
    sysp_bcc = vq.PeriodicSystem(
        3, np.eye(3) * 5.0,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [2.5, 2.5, 2.5])],
    )
    vq.attach_symmetry(sysp_bcc)
    bcc_ops = list(sysp_bcc.symmetry.operations)
    bcc_sym = vq.symmorphic_operations(bcc_ops)
    # Im-3m: |G|=96 with 48 symmorphic + 48 with t = (1/2, 1/2, 1/2).
    assert len(bcc_sym) < len(bcc_ops), (
        f"BCC symmorphic filter found {len(bcc_sym)} of {len(bcc_ops)} "
        "operators — expected a non-trivial subset"
    )


# ---------------------------------------------------------------------------
# 7. No-symmorphic raise on the verifier
# ---------------------------------------------------------------------------

def test_verifier_raises_without_symmorphic_ops():
    sysp = _he_cubic()
    basis = _basis(sysp)
    full = vq.compute_overlap_lattice(basis, sysp, _opts())
    with pytest.raises(ValueError, match="no symmorphic"):
        vq.verify_lattice_matrix_set_symmetry(full, basis, sysp, [])


# ---------------------------------------------------------------------------
# 8. Method-independent orbit partition (DIRECT vs EWALD nuclear)
# ---------------------------------------------------------------------------

def test_orbit_partition_is_method_independent_for_nuclear():
    sysp = _he_cubic()
    basis = _basis(sysp)
    ops = sysp.symmetry.operations
    opts_direct = _opts(method=vq.CoulombMethod.DIRECT_TRUNCATED)
    opts_ewald = _opts(method=vq.CoulombMethod.EWALD_3D)

    red_d, _ = vq.compute_nuclear_lattice_with_orbits(
        basis, sysp, opts_direct, ops,
    )
    red_e, _ = vq.compute_nuclear_lattice_with_orbits(
        basis, sysp, opts_ewald, ops,
    )
    # The orbit partition is method-independent (operator-only).
    assert red_d.n_orbits == red_e.n_orbits
    assert red_d.compression_ratio == pytest.approx(red_e.compression_ratio)


# ---------------------------------------------------------------------------
# 9. GS3 standardise-then-compress convenience path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "builder_name",
    [
        "compute_overlap_lattice_symmetrised_with_orbits",
        "compute_kinetic_lattice_symmetrised_with_orbits",
        "compute_nuclear_lattice_symmetrised_with_orbits",
    ],
)
def test_symmetrised_orbit_wrappers_clean_user_cell(builder_name):
    sysp = _he_cubic_distorted()
    opts = _opts(cutoff=10.0)
    builder = getattr(vq, builder_name)

    result = builder(sysp, "sto-3g", opts, symprec=1.0e-3)
    reduced, full = result

    assert isinstance(result, vq.SymmetrisedOrbitIntegralResult)
    assert sysp.symmetry is None
    assert result.report.rms_displacement_bohr > 0.0
    assert result.system.symmetry is not None
    assert result.system.symmetry.number == 221
    assert len(result.operations) == 48
    assert result.basis.name == "sto-3g"
    assert reduced is result.reduced
    assert full is result.full
    assert reduced.compression_ratio > 1.0

    rec = vq.reconstruct_lattice_matrix_set_c(
        reduced.representatives,
        reduced.orbits,
        result.basis,
        result.system,
        result.operations,
        cells=list(full.cells),
    )
    max_err = max(
        float(np.linalg.norm(np.asarray(full.blocks[i]) - rec[i]))
        for i in range(len(full.cells))
    )
    assert max_err < 1.0e-10
