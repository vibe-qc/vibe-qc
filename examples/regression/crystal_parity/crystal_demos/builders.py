"""Geometry + basis builders for every CRYSTAL eval-version demo system.

One ``build_<system>()`` per ``.d12`` file in this directory. The
geometry is constructed **by hand** from the CRYSTAL ``.d12``
parameters (space group + Wyckoff position + lattice constants),
mirroring CRYSTAL's auto-expansion of the asymmetric unit. The basis
is constructed via ``vq.BasisSet(...)`` from the in-tree libint data
where the demo uses STO-3G or an existing BSE-fetchable name; for
demos that use **inline atom-specific bases** (Pople 6-21G with
custom exponents, atom-specific 8511G/8411G, etc.) the basis returned
is ``None`` and the caller must raise ``NotImplementedError`` until
an inline-basis parser lands (planned for the basissetdev branch).

Per CLAUDE.md §10: this module does not import CRYSTAL. The
geometries are independently constructed Python — the ``.d12`` files
in this directory are reference inputs only.

Notation:
  * `_ANG = 1 Å in bohr`
  * For FCC SG 225/227 the primitive vectors are ``(a/2)·(0,1,1),
    (1,0,1),(1,1,0)``; ``a`` is the **conventional cubic edge**.
  * For HCP SG 194 the primitive vectors are ``a·(1,0,0),
    a·(−1/2,√3/2,0), c·(0,0,1)``.
  * Two-atom diamond/Si: positions ``+(a/8,a/8,a/8)`` and
    ``+(3a/8,3a/8,3a/8)`` — origin-2 expansion of Wyckoff 8a.
  * For 2D ``SLAB`` and 1D ``POLYMER`` cells we pad the
    non-periodic axes with vacuum (50 bohr) inside the ``[3,3]``
    lattice that ``PeriodicSystem`` requires.
"""
from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc import attach_symmetry

_ANG = 1.0 / 0.529177210903


# ---------------------------------------------------------------------
# Bravais helpers
# ---------------------------------------------------------------------
def _fcc_primitive(a_bohr: float) -> np.ndarray:
    """FCC primitive lattice vectors as a (3,3) array (rows = a1,a2,a3).
    Conventional cubic edge ``a`` in bohr."""
    return (a_bohr / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])


def _hcp_primitive(a_bohr: float, c_bohr: float) -> np.ndarray:
    """HCP primitive lattice. a in xy-plane (γ=120°), c along z."""
    return np.array([
        [a_bohr, 0.0, 0.0],
        [-0.5 * a_bohr, 0.5 * np.sqrt(3.0) * a_bohr, 0.0],
        [0.0, 0.0, c_bohr],
    ])


# ---------------------------------------------------------------------
# STO-3G 3D closed-shell systems — RUNNABLE
# ---------------------------------------------------------------------
def build_mgo_sto3g():
    """``mgo_sto3g.d12``: MgO bulk SG 225 (FCC rocksalt), a=4.21 Å, STO-3G.

    2 atoms / 1 FU per primitive cell. SHRINK 8 8.
    CRYSTAL14 reference: -271.21814372 Ha/FU (10 cycles, sealed).
    """
    a = 4.21 * _ANG
    lattice = _fcc_primitive(a)
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),                # Mg @ Wyckoff 4a
        vq.Atom(8,  [a / 2.0, a / 2.0, a / 2.0]),    # O  @ Wyckoff 4b
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def build_diamond_sto3g():
    """``diamond_sto3g.d12``: C diamond SG 227, a=3.57 Å, STO-3G.

    2 atoms per primitive cell (Wyckoff 8a origin-2 expanded). SHRINK 8 8.
    """
    a = 3.57 * _ANG
    lattice = _fcc_primitive(a)
    s = a / 8.0
    atoms = [
        vq.Atom(6, [s, s, s]),
        vq.Atom(6, [3 * s, 3 * s, 3 * s]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def build_sibulk_sto3g():
    """``sibulk_sto3g.d12``: Si bulk SG 227, a=5.42 Å, STO-3G.

    2 atoms per primitive cell. SHRINK 8 8, FMIXING 30.
    """
    a = 5.42 * _ANG
    lattice = _fcc_primitive(a)
    s = a / 8.0
    atoms = [
        vq.Atom(14, [s, s, s]),
        vq.Atom(14, [3 * s, 3 * s, 3 * s]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------
# STO-3G 3D open-shell — geometry ready, UHF parity pending
# ---------------------------------------------------------------------
def build_nio_sto3g():
    """``nio_sto3g.d12``: NiO bulk SG 225, a=4.164 Å, STO-3G, UHF.

    Geometry: 2 atoms / 1 FU. CRYSTAL spec: SHRINK 8 8, SPINLOCK 2 15,
    LEVSHIFT 3 1, FMIXING 30. **UHF** — needs the multi-k UHF driver.
    """
    a = 4.164 * _ANG
    lattice = _fcc_primitive(a)
    atoms = [
        vq.Atom(28, [0.0, 0.0, 0.0]),
        vq.Atom(8,  [a / 2.0, a / 2.0, a / 2.0]),
    ]
    # multiplicity from SPINLOCK 2: Ms = 2/2 = 1 → 2S+1 = 3
    system = vq.PeriodicSystem(3, lattice, atoms, 0, 3)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------
# STO-3G metallic / hex — geometry ready, smearing pending
# ---------------------------------------------------------------------
def build_be_sto3g():
    """``be_sto3g.d12``: Be bulk SG 194 (HCP), a=2.29 Å, c=3.59 Å, STO-3G.

    2 atoms per primitive cell at (1/3,2/3,1/4) and (2/3,1/3,3/4)
    (P6_3/mmc Wyckoff 2c). SHRINK 12 24, FMIXING 30, MAXCYCLE 130.
    Metallic — needs Fermi-Dirac smearing for k-mesh integration.
    """
    a = 2.29 * _ANG
    c = 3.59 * _ANG
    lattice = _hcp_primitive(a, c)
    # Wyckoff 2c — two atoms via P6_3 screw along c
    a1, a2, a3 = lattice[0], lattice[1], lattice[2]
    p1 = (1.0 / 3.0) * a1 + (2.0 / 3.0) * a2 + 0.25 * a3
    p2 = (2.0 / 3.0) * a1 + (1.0 / 3.0) * a2 + 0.75 * a3
    atoms = [vq.Atom(4, p1.tolist()), vq.Atom(4, p2.tolist())]
    system = vq.PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------
# Low-dim (SLAB / POLYMER) — geometry ready, driver pending
# ---------------------------------------------------------------------
def build_graphite_sto3g():
    """``graphite_sto3g.d12``: 2D graphite layer, SLAB p6/mmm, a=2.47 Å, STO-3G.

    CRYSTAL .d12 lists one atom at fractional (-1/3,1/3,0). The CRYSTAL
    SLAB ``p6/mmm`` layer group expands this to a 2-atom graphene
    primitive cell (the second atom is generated by the inversion +
    3-fold rotation). We expand explicitly: C atoms at
    ``(0, a/√3, 0)`` and ``(a/2, a/(2√3), 0)`` in Cartesian (one of
    the standard graphene unit-cell conventions). 50 bohr vacuum on z.
    SHRINK 8 16. **2D** — direct-space cutoff needs to be 2D-aware.
    """
    a = 2.47 * _ANG
    vacuum_z = 50.0
    lattice = np.array([
        [a,        0.0,                0.0],
        [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
        [0.0,      0.0,                vacuum_z],
    ])
    # graphene 2-atom basis (A and B sublattices)
    p_A = (-1.0 / 3.0) * lattice[0] + (1.0 / 3.0) * lattice[1]
    p_B = (1.0 / 3.0) * lattice[0] + (-1.0 / 3.0) * lattice[1]
    atoms = [
        vq.Atom(6, p_A.tolist()),
        vq.Atom(6, p_B.tolist()),
    ]
    system = vq.PeriodicSystem(2, lattice, atoms)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def build_sn_polym_sto3g():
    """``sn_polym_sto3g.d12``: 1D (SN)x polymer, a=4.431 bohr, STO-3G.

    1D POLYMER along x. S at (0, -0.844969, 0), N at (0.1416, 0.667077,
    -0.000930) Cartesian (in bohr per CRYSTAL convention). SHRINK 13 13.
    **1D** — needs 1D direct-space cutoff handling.
    """
    a = 4.431  # already in bohr per CRYSTAL POLYMER convention
    vacuum = 50.0
    lattice = np.array([
        [a,   0.0,    0.0],
        [0.0, vacuum, 0.0],
        [0.0, 0.0,    vacuum],
    ])
    atoms = [
        vq.Atom(16, [0.0,    -0.844969,  0.0]),
        vq.Atom(7,  [0.1416,  0.667077, -0.000930]),
    ]
    # 23 electrons per minimal SN unit → doublet (open-shell metallic).
    # Real (SN)x has a 4-atom S2N2 primitive at low T; CRYSTAL's .d12
    # uses the 2-atom minimal cell with default RHF treatment that
    # falls back to fractional-occupation handling. For the vibe-qc
    # geometry here we mark as multiplicity=2 (one unpaired electron)
    # so PeriodicSystem accepts it; the actual SCF needs metallic
    # smearing.
    system = vq.PeriodicSystem(1, lattice, atoms, 0, 2)
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def build_mgo001_sto3g_geometry():
    """``mgo001.d12`` is **extended basis** (8411G/8511G). The
    geometry is the same single-layer MgO(001) slab: a=4.21 Å,
    one Mg+O pair, p4mm-like layer group. We provide the geometry
    here for completeness; the runnable parity test requires the
    inline-basis parser to land first.
    """
    a = 4.21 * _ANG
    vacuum_z = 50.0
    # 2D square lattice (in-plane). One-layer slab built from
    # rocksalt: Mg at (0,0,0), O at (a/2, a/2, 0) in the surface plane.
    lattice = np.array([
        [a,   0.0, 0.0],
        [0.0, a,   0.0],
        [0.0, 0.0, vacuum_z],
    ])
    atoms = [
        vq.Atom(12, [0.0,     0.0,     0.0]),
        vq.Atom(8,  [a / 2.0, a / 2.0, 0.0]),
    ]
    system = vq.PeriodicSystem(2, lattice, atoms)
    attach_symmetry(system)
    return system


# ---------------------------------------------------------------------
# Extended-basis 3D systems — geometry ready, inline-basis parser pending
# ---------------------------------------------------------------------
def build_mgo_bulk_extended_geometry():
    """``mgo_bulk.d12``: MgO bulk SG 225, a=4.21 Å, 8411G/8511G basis.

    Geometry identical to :func:`build_mgo_sto3g`; only basis differs.
    """
    sys_, _ = build_mgo_sto3g()
    return sys_


def build_diamond_extended_geometry():
    """``diamond.d12``: C diamond SG 227, a=3.57 Å, 6-21G + d-polarization.

    Geometry identical to :func:`build_diamond_sto3g`; only basis differs.
    """
    sys_, _ = build_diamond_sto3g()
    return sys_


def build_sibulk_extended_geometry():
    """``sibulk.d12``: Si bulk SG 227, a=5.42 Å, 6-21G modified.

    Geometry identical to :func:`build_sibulk_sto3g`; only basis differs.
    """
    sys_, _ = build_sibulk_sto3g()
    return sys_


def build_bebulk_extended_geometry():
    """``bebulk.d12``: Be bulk SG 194 (HCP), a=2.29 Å, c=3.59 Å, 4-shell s-only.

    Geometry identical to :func:`build_be_sto3g`; only basis differs.
    """
    sys_, _ = build_be_sto3g()
    return sys_


def build_nio_extended_geometry():
    """``nio.d12``: NiO bulk SG 225, a=4.164 Å, extended basis + polarization,
    UHF SPINLOCK 2 15. Geometry identical to :func:`build_nio_sto3g`."""
    sys_, _ = build_nio_sto3g()
    return sys_


def build_urea_321G_geometry():
    """``urea_bulk_321G.d12``: Urea bulk SG 113 (P-42_1m), 3-21G basis,
    a=5.565 Å, c=4.684 Å, 5-atom asymmetric unit (C/O/N/H/H).

    SG 113 is a tetragonal non-centrosymmetric group; the 5 atoms in
    the .d12 are the asymmetric unit and CRYSTAL auto-expands to 16
    atoms per primitive cell via its symmetry operations. **vibe-qc
    does not auto-expand from a space-group label** — to do this
    properly we need either:
      (a) explicit hand-expansion of all 16 atoms (tractable but error-
          prone), or
      (b) spglib-driven symmetry expansion from the asymmetric unit +
          SG number (the cleaner solution; defer).
    Geometry not constructed here.
    """
    raise NotImplementedError(
        "urea_bulk_321G: SG 113 asymmetric-unit expansion not yet "
        "implemented. Requires spglib-driven expansion from "
        "asymmetric unit + SG number, OR hand-listing all 16 atoms. "
        "Also needs the inline-basis parser for 3-21G."
    )


def build_mgo001co_extended_geometry():
    """``mgo001co.d12``: CO molecule on MgO(001) 1-layer slab, extended basis.

    Adds a CO molecule (with separate molecular-oxygen basis at Z=108
    ghost) above the MgO(001) layer. Geometry not yet constructed —
    requires ghost-atom support in vibe-qc to mirror CRYSTAL's
    `Z=108` molecular-oxygen-basis pattern.
    """
    raise NotImplementedError(
        "mgo001co: requires (a) inline-basis parser for the 8411G/8511G "
        "atom-specific basis and the molecular-oxygen Z=108 ghost-atom "
        "convention; (b) verified MgO(001) slab geometry."
    )
