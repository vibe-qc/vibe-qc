"""AICCM-2026 benchmark test set — system registry.

A curated set of periodic systems for the ab-initio Cyclic Cluster Model (AICCM)
paper, spanning dimensionality (1-D chains, 2-D sheets/slabs, 3-D bulk), bonding
(covalent, ionic, molecular, oxide, metal), shell (closed / open), and
**Bravais lattice** — all seven crystal systems (triclinic → cubic) and the
cubic centerings (cP / cI / cF), to demonstrate the AICCM's lattice-generality
(the symmetric four-center + spglib symmetry are lattice-general: the reference
cell is the Wigner–Seitz cell). Geometries use **primitive cells** (smallest, so
the cyclic cluster stays tractable) with the lattice constants of the matching
CRYSTAL23 ``.d12`` in ``~/gitlab/qc-input-library/crystal/`` (so the comparison
is apples-to-apples).

The Bravais-coverage crystals (corundum, wurtzite, black-P, AlCl₃, fluorite,
anatase, CsCl, boric-acid, bcc-Na, …) are built by :func:`_crystal` from
grounded space-group specs in ``crystal_specs.json`` (extracted from the library
``.d12`` and expanded by ASE's ``spacegroup.crystal``). Adding any further
library crystal is a one-line registry entry + its spec — see ``references.md``.

Each entry records the metadata the driver and the paper need: dimensionality,
bonding class, atoms/cell, the recommended cluster size for the four-center route
(``nrep_4c``) and the k-mesh for the periodic reference (``kmesh``), the target
machine, whether the bare-1/r four-center is expected to be accurate, and the
CRYSTAL23 reference path.

**Construction note:** ``aiccm2026dev-a`` is the union-and-weight/Wigner-Seitz
Γ-CCM construction. The density-fitted GDF/RI and BIPOLE routes are external
periodic controls, not alternate spellings of Γ-CCM. The historical
``four_center_quantitative`` flag is triage metadata for the observed
four-center/control gap; it does not establish its cause or authorize route
substitution. See the README.

Geometry provenance: extracted/verified against the CRYSTAL23 ``.d12`` sources
(see ``references.md``).  Reference: Peintinger & Bredow, J. Comput. Chem. 35,
839 (2014).
"""

from __future__ import annotations

import site_settings

import json
import os

import numpy as np

import vibeqc as vq

ANG = 1.0 / 0.529177210903   # Angstrom -> bohr


# ---------------------------------------------------------------- builders ---
def _chain(a_ang, atoms_frac, vac=40.0):
    """1-D periodic chain along x (vacuum in y,z). atoms_frac = [(Z, frac_x), ...]."""
    a = a_ang * ANG
    lat = np.array([[a, 0.0, 0.0], [0.0, vac, 0.0], [0.0, 0.0, vac]])
    atoms = []
    for spec in atoms_frac:
        Z, fx = spec[0], spec[1]
        y = spec[2] * ANG if len(spec) > 2 else 0.0
        z = spec[3] * ANG if len(spec) > 3 else 0.0
        atoms.append(vq.Atom(Z, [fx * a, y, z]))
    return vq.PeriodicSystem(1, lat, atoms)


def _rocksalt(a_ang, Z_cat, Z_an):
    """2-atom primitive FCC rocksalt."""
    a = a_ang * ANG
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    an = (lat @ np.array([0.5, 0.5, 0.5])).tolist()
    return vq.PeriodicSystem(3, lat, [vq.Atom(Z_cat, [0, 0, 0]), vq.Atom(Z_an, an)])


def _zincblende(a_ang, Z1, Z2):
    """2-atom primitive FCC zincblende / diamond (Z1==Z2)."""
    a = a_ang * ANG
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(Z1, [0, 0, 0]),
                                      vq.Atom(Z2, [a / 4, a / 4, a / 4])])


def _hex2d(a_ang, atoms_frac, vac=50.0):
    """2-D hexagonal sheet (vacuum in z). atoms_frac = [(Z, f1, f2), ...]."""
    a = a_ang * ANG
    lattice_rows = np.array(
        [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, vac]]
    )
    lat = lattice_rows.T
    atoms = [
        vq.Atom(Z, (lat @ np.array([f1, f2, 0.0])).tolist())
        for Z, f1, f2 in atoms_frac
    ]
    return vq.PeriodicSystem(2, lat, atoms)


def _mgo_slab(a_ang=4.21):
    a = a_ang * ANG
    lattice_rows = np.array(
        [[a / 2, a / 2, 0], [-a / 2, a / 2, 0], [0, 0, 50.0]]
    )
    lat = lattice_rows.T
    oxygen = lat @ np.array([0.5, 0.5, 0.0])
    return vq.PeriodicSystem(
        2,
        lat,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, oxygen.tolist())],
    )


def _frac3d(a, b, c, gamma_deg, frac):
    """General 3-D cell from a,b,c (Ang),gamma and fractional atoms [(Z,f1,f2,f3)]."""
    g = np.radians(gamma_deg)
    lattice_rows = np.array([[a, 0, 0],
                             [b * np.cos(g), b * np.sin(g), 0],
                             [0, 0, c]]) * ANG
    lat = lattice_rows.T
    atoms = [
        vq.Atom(Z, (lat @ np.array([f1, f2, f3])).tolist())
        for Z, f1, f2, f3 in frac
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _rutile(a_ang, c_ang, u):
    """Rutile TiO2 (P4_2/mnm), 6 atoms."""
    a, c = a_ang * ANG, c_ang * ANG
    lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
    fr = [(22, 0, 0, 0), (22, 0.5, 0.5, 0.5),
          (8, u, u, 0), (8, -u, -u, 0), (8, 0.5 + u, 0.5 - u, 0.5), (8, 0.5 - u, 0.5 + u, 0.5)]
    atoms = [vq.Atom(Z, ((f1 % 1) * lat[0] + (f2 % 1) * lat[1] + (f3 % 1) * lat[2]).tolist())
             for Z, f1, f2, f3 in fr]
    return vq.PeriodicSystem(3, lat, atoms)


def _ice_ih():
    a, c = 4.4970 * ANG, 7.3220 * ANG
    lattice_rows = np.array(
        [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    )
    lat = lattice_rows.T
    frac = [(8, 1/3, 2/3, 0.0625), (8, 2/3, 1/3, 0.5625),
            (1, 1/3, 2/3, 0.1953), (1, 0.4497, 0.5503, 0.0083),
            (1, 2/3, 1/3, 0.6953), (1, 0.5503, 0.4497, 0.5083)]
    atoms = [
        vq.Atom(Z, (lat @ np.array([f1, f2, f3])).tolist())
        for Z, f1, f2, f3 in frac
    ]
    return vq.PeriodicSystem(3, lat, atoms)


_CRYSTAL_SPECS = None


def _crystal(name):
    """Build a PeriodicSystem from a grounded space-group spec.

    ``crystal_specs.json`` (committed alongside this file, extracted from the
    CRYSTAL ``.d12`` sources in ``qc-input-library`` — see ``references.md``)
    holds ``{sg, cellpar, asym}`` per crystal; ASE's ``spacegroup.crystal``
    expands the asymmetric unit to the **primitive cell**. This is lattice-general
    (handles every Bravais lattice) and portable (no library access at runtime).
    Multiplicity is set parity-correct so odd-electron primitive cells build.
    """
    global _CRYSTAL_SPECS
    if _CRYSTAL_SPECS is None:
        with open(os.path.join(os.path.dirname(__file__), "crystal_specs.json")) as fh:
            _CRYSTAL_SPECS = json.load(fh)
    from ase.spacegroup import crystal as ase_crystal

    s = _CRYSTAL_SPECS[name]
    at = ase_crystal(symbols=[a[0] for a in s["asym"]],
                     basis=[a[1:] for a in s["asym"]],
                     spacegroup=int(s["sg"]), cellpar=list(s["cellpar"]),
                     primitive_cell=True)
    n_elec = int(sum(int(z) for z in at.numbers))
    mult = 1 if n_elec % 2 == 0 else 2
    lat = np.asarray(at.cell.array, dtype=float).T * ANG
    atoms = [vq.Atom(int(z), (np.asarray(p, dtype=float) * ANG).tolist())
             for z, p in zip(at.numbers, at.positions)]
    return vq.PeriodicSystem(3, lat, atoms, 0, mult)


# Routes that build the Coulomb operator from the periodic reciprocal mesh (the
# neutral cderi / multi-k GDF). For dim < 3 that mesh spans only the periodic axes
# and pins every non-periodic axis at G_perp = 0, so the kernel 4*pi/|G+q|^2/V it
# feeds is a transverse-uniform sheet term proportional to 1/V, not 1/r: the
# electron repulsion vanishes as the vacuum padding grows, and the total diverges
# with the nuclear lattice-sum cutoff. vibeqc fails closed on this since 2026-07-10
# (aux_basis.rsgdf_dense_g_mesh). The four-center routes (bare 1/r minimum image,
# including the scalable WSSC SCF behind properties/localize/pao) are unaffected and
# remain the low-D reference; the gauge-correct low-D Hamiltonian is the
# mixed-boundary wire kernel (run_ccm_rhf_wire, dim=1, light elements).
#
# Consumed by run_case.py (emits an "unavailable" row) and make_jobs.py (skips the
# job for dim<3 rather than burning a cluster slot to record a NotImplementedError).
#
# Membership is decided by which Coulomb builder the route's handler reaches, not by
# the route's name. Three routes look like they belong here and do not:
#   aiccm-rijcosx      -- ccm_ri_tensors (WSSC RI-J on a molecular aux basis) + COSX
#                         on a supercell grid. No reciprocal mesh.
#   aiccm-mp2/-ccsd    -- run_ccm_mp2/run_ccm_ccsd build the effective four-center
#                         WSSC ERI (eri=None), the same tensor as the reference SCF.
# Their dim<3 numbers are four-center numbers and stay valid. The DLPNO variants do
# belong here: they sit on ccm_neutral_cderi_fold + run_ccm_rhf_ri_neutral.
NEEDS_3D_COULOMB = {
    "aiccm-hf-direct",                              # run_ccm_rhf_direct
    "aiccm-ri", "aiccm-ks-ri",                      # run_ccm_{rhf,rks}_gdf
    "aiccm-ri-cosx", "aiccm-ks-ri-cosx",            # ditto, k_exchange="cosx"
    "aiccm-dlpno-mp2", "aiccm-dlpno-ccsd",          # ccm_neutral_cderi_fold
    "aiccm-uccsd",                                  # ccm_neutral_cderi
    "gdf", "bipole",                                # periodic reference drivers
}


# ------------------------------------------------------------- the registry ---
# Each value: (builder, metadata dict). nrep_4c = cluster for the 4-center AICCM
# run; kmesh = Monkhorst-Pack mesh for the periodic (gdf/bipole) reference.
SYSTEMS = {
    # ---- 1-D (the AICCM's validated home) -------------------------------------
    "h-chain": dict(
        build=lambda: _chain(7.9376582, [(1, 0.0), (1, 0.0933333)]),
        dim=1, klass="molecular", atoms=2, basis="sto-3g",
        nrep_4c=(8, 1, 1), kmesh=(16, 1, 1), tier="A", machine=site_settings.system_host('h-chain', default="local"),
        four_center_quantitative=True,
        crystal_ref="crystal/h2-chain-1d/pbe_pob-tzvp-rev2/INPUT.d12"),
    # Uniform (equal-spacing) 1-D H chain — the near-metallic counterpart to the
    # dimerised h-chain above. 2 H/cell at fractional 0, 1/2 so a = 2 R_HH; R_HH =
    # 1.3229 Å (2.5 bohr) is stretched enough to keep S^CCM well-conditioned while
    # the near-uniform band is the interesting (correlated) case.
    "uniform-h-chain": dict(
        build=lambda: _chain(2.6458, [(1, 0.0), (1, 0.5)]),
        dim=1, klass="metallic-1d", atoms=2, basis="sto-3g",
        nrep_4c=(8, 1, 1), kmesh=(16, 1, 1), tier="A", machine=site_settings.system_host('uniform-h-chain', default="local"),
        four_center_quantitative=True,
        crystal_ref=None),
    "lih-chain": dict(
        # The library .d12 uses a=2.0 Å (Li-H=1.0 Å) — unphysically compressed,
        # so the cyclic-cluster overlap S^CCM goes non-positive-definite (Fig-6
        # guard) and the SCF lands on an ill-conditioned state. Use the physical
        # LiH spacing a=3.2 Å (Li-H=1.6 Å), which is well-conditioned
        # (min eig S^CCM ≈ 0.23) and sensible (RHF ≈ -3.97 Ha/atom). For the
        # CRYSTAL comparison re-run the .d12 at a=3.2 Å. Ionic ⇒ the bare-1/r
        # four-center and neutral GDF control differ substantially. Do not
        # substitute one for the other or assign the gap without a clean audit.
        build=lambda: _chain(3.2, [(3, 0.0), (1, 0.5)]),
        dim=1, klass="ionic-1d", atoms=2, basis="sto-3g",
        nrep_4c=(6, 1, 1), kmesh=(16, 1, 1), tier="A", machine=site_settings.system_host('lih-chain', default="local"),
        four_center_quantitative=False,
        crystal_ref="crystal/lih-chain-1d/pbe_pob-tzvp-rev2/INPUT.d12"),
    "polyethylene": dict(
        build=lambda: _chain(2.55, [(6, 0.0, 0.0, 0.0), (1, 0.0, 0.785, 0.785),
                                    (1, 0.0, -0.785, -0.785), (6, 0.5, 0.0, 0.0),
                                    (1, 0.5, 0.785, 0.785), (1, 0.5, -0.785, -0.785)]),
        dim=1, klass="molecular", atoms=6, basis="sto-3g",
        nrep_4c=(4, 1, 1), kmesh=(8, 1, 1), tier="A", machine=site_settings.system_host('polyethylene', default="local"),
        four_center_quantitative=True,
        crystal_ref="crystal/polyethylene-1d/pbe_pob-tzvp-rev2/INPUT.d12"),

    # ---- covalent 3-D (the AICCM works) --------------------------------------
    "c-diamond": dict(
        build=lambda: _zincblende(3.5670, 6, 6),
        dim=3, klass="covalent", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="A", machine=site_settings.system_host('c-diamond'),
        four_center_quantitative=True,
        crystal_ref="crystal/c-diamond/pbe_pob-tzvp-rev2/INPUT.d12"),
    "si-diamond": dict(
        build=lambda: _zincblende(5.4310, 14, 14),
        dim=3, klass="covalent", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('si-diamond'),
        four_center_quantitative=True,
        crystal_ref="crystal/si-diamond/pbe_pob-tzvp-rev2/INPUT.d12"),
    "sic-zb": dict(
        build=lambda: _zincblende(4.3580, 14, 6),
        dim=3, klass="covalent", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('sic-zb'),
        four_center_quantitative=True,
        crystal_ref="crystal/sic-zincblende/pbe_pob-tzvp-rev2/INPUT.d12"),
    "bn-zb": dict(
        build=lambda: _zincblende(3.6150, 5, 7),
        dim=3, klass="covalent", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="A", machine=site_settings.system_host('bn-zb'),
        four_center_quantitative=True,
        crystal_ref="crystal/bn-zincblende/pbe_pob-tzvp-rev2/INPUT.d12"),

    # ---- 2-D ------------------------------------------------------------------
    "graphene": dict(
        build=lambda: _hex2d(2.46, [(6, -1/3, 1/3), (6, 1/3, -1/3)]),
        dim=2, klass="covalent", atoms=2, basis="sto-3g",
        nrep_4c=(3, 3, 1), kmesh=(8, 8, 1), tier="A", machine=site_settings.system_host('graphene'),
        four_center_quantitative=True,
        crystal_ref="crystal/graphene-2d/pbe_pob-tzvp-rev2/INPUT.d12"),
    "mgo-slab": dict(
        build=lambda: _mgo_slab(4.21),
        dim=2, klass="ionic", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 1), kmesh=(6, 6, 1), tier="B", machine=site_settings.system_host('mgo-slab'),
        four_center_quantitative=False,
        crystal_ref="crystal/mgo-001-slab/rks-pbe_tut/INPUT.d12"),

    # ---- molecular crystals (the AICCM works; weak Madelung) ------------------
    "ice-ih": dict(
        build=_ice_ih,
        dim=3, klass="molecular", atoms=6, basis="sto-3g",
        nrep_4c=(2, 2, 1), kmesh=(4, 4, 2), tier="B", machine=site_settings.system_host('ice-ih'),
        four_center_quantitative=True,
        crystal_ref="crystal/ice-ih/pbe_pob-tzvp-rev2/INPUT.d12"),
    "co2-dryice": dict(
        build=lambda: _frac3d(5.6240, 5.6240, 5.6240, 90.0,
                              [(6, 0, 0, 0), (8, 0.118, 0.118, 0.118),
                               (8, -0.118, -0.118, -0.118)]),
        dim=3, klass="molecular", atoms=3, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('co2-dryice'),
        four_center_quantitative=True,
        crystal_ref="crystal/co2-dryice/rks-pbe_sto-3g/INPUT.d12"),

    # ---- ionic 3-D (large Γ-CCM/control gaps; causality remains unassigned) ----
    "lih-rocksalt": dict(
        build=lambda: _rocksalt(4.0840, 3, 1),
        dim=3, klass="ionic", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="A", machine=site_settings.system_host('lih-rocksalt', default="local"),
        four_center_quantitative=False,
        crystal_ref="crystal/lih-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12"),
    "lif-rocksalt": dict(
        build=lambda: _rocksalt(4.0260, 3, 9),
        dim=3, klass="ionic", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="A", machine=site_settings.system_host('lif-rocksalt'),
        four_center_quantitative=False,
        crystal_ref="crystal/lif-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12"),
    "nacl-rocksalt": dict(
        build=lambda: _rocksalt(5.6400, 11, 17),
        dim=3, klass="ionic", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('nacl-rocksalt'),
        four_center_quantitative=False,
        crystal_ref="crystal/nacl-rocksalt/pbe_pob-tzvp-rev2/INPUT.d12"),
    "mgo": dict(
        build=lambda: _rocksalt(4.22389871, 12, 8),
        dim=3, klass="ionic", atoms=2, basis="sto-3g",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('mgo'),
        four_center_quantitative=False,
        crystal_ref="crystal/mgo-rocksalt/r2scan_seg/INPUT.d12"),

    # ---- oxides (larger cells; RI/gdf route) ---------------------------------
    "tio2-rutile": dict(
        build=lambda: _rutile(4.5940, 2.9590, 0.305),
        dim=3, klass="oxide", atoms=6, basis="sto-3g",
        nrep_4c=(1, 1, 2), kmesh=(4, 4, 6), tier="C", machine=site_settings.system_host('tio2-rutile'),
        four_center_quantitative=False,
        crystal_ref="crystal/tio2-rutile/pbe_pob-tzvp-rev2/INPUT.d12"),
    "sio2-quartz": dict(
        build=lambda: _frac3d(4.9160, 4.9160, 5.4050, 120.0,
                              [(14, 0.4699, 0.0, 1/3), (14, 0.0, 0.4699, 2/3),
                               (14, -0.4699, -0.4699, 0.0),
                               (8, 0.2670, 0.4118, 0.2144), (8, -0.4118, -0.1448, 0.5477),
                               (8, 0.1448, -0.2670, -0.1190), (8, 0.4118, 0.2670, -0.2144),
                               (8, -0.2670, 0.1448, 0.1190), (8, -0.1448, -0.4118, 0.4523)]),
        dim=3, klass="oxide", atoms=9, basis="sto-3g",
        nrep_4c=(1, 1, 1), kmesh=(4, 4, 4), tier="C", machine=site_settings.system_host('sio2-quartz'),
        four_center_quantitative=False,
        crystal_ref="crystal/sio2-alpha-quartz/pbe_pob-tzvp-rev2/INPUT.d12"),

    # ---- open-shell AFM (the open-shell CCM; heavier, RI/gdf) -----------------
    "nio-afm": dict(
        build=lambda: _rocksalt(4.162, 28, 8),
        dim=3, klass="ionic-afm", atoms=2, basis="sto-3g", open_shell=True,
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="C", machine=site_settings.system_host('nio-afm'),
        four_center_quantitative=False,
        crystal_ref="crystal/nio-rocksalt/uks-r2scan_pob-tzvp-rev2/INPUT.d12"),

    # ---- Bravais-lattice coverage (expanded) ---------------------------------
    # Grounded geometries from the qc-input-library .d12 sources, extracted to
    # crystal_specs.json and expanded by ASE's spacegroup.crystal (primitive
    # cell). These fill the remaining crystal systems / centerings so the test
    # set spans all 14 Bravais lattices and demonstrates the AICCM's
    # lattice-generality (the symmetric four-center + spglib symmetry are
    # lattice-general — the reference cell is the WSC).
    "aln-wurtzite": dict(   # hexagonal P6_3mc — wurtzite (3-D hexagonal compound)
        build=lambda: _crystal("aln-wurtzite"),
        dim=3, klass="covalent", atoms=4, basis="sto-3g", bravais="hP",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('aln-wurtzite'),
        four_center_quantitative=True,
        crystal_ref="crystal/aln-wurtzite/b3lyp_pob-tzvp-rev2/INPUT.d12"),
    "black-phosphorus": dict(   # orthorhombic Cmce (base-centered, oS)
        build=lambda: _crystal("black-phosphorus"),
        dim=3, klass="covalent", atoms=4, basis="sto-3g", bravais="oS",
        nrep_4c=(2, 2, 1), kmesh=(4, 4, 2), tier="B", machine=site_settings.system_host('black-phosphorus'),
        four_center_quantitative=True,
        crystal_ref="crystal/black-phosphorus/rhf_sto-3g/INPUT.d12"),
    "caf2-fluorite": dict(   # cubic Fm-3m fluorite (AB2; distinct from rocksalt, cF)
        build=lambda: _crystal("caf2-fluorite"),
        dim=3, klass="ionic", atoms=3, basis="sto-3g", bravais="cF",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="B", machine=site_settings.system_host('caf2-fluorite'),
        four_center_quantitative=False,
        crystal_ref="crystal/caf2-fluorite/b3lyp_pob-tzvp-rev2/INPUT.d12"),
    "tio2-anatase": dict(   # tetragonal I4_1/amd — body-centered tetragonal (tI)
        build=lambda: _crystal("tio2-anatase"),
        dim=3, klass="oxide", atoms=6, basis="sto-3g", bravais="tI",
        nrep_4c=(1, 1, 1), kmesh=(4, 4, 2), tier="C", machine=site_settings.system_host('tio2-anatase'),
        four_center_quantitative=False,
        crystal_ref="literature: Horn, Schwerdtfeger & Meagher, "
                    "Z. Kristallogr. 136, 273 (1972) (a=3.7842, c=9.5146 Å)"),
    "al2o3-corundum": dict(   # rhombohedral R-3c — corundum (hR) [the example]
        build=lambda: _crystal("al2o3-corundum"),
        dim=3, klass="oxide", atoms=10, basis="sto-3g", bravais="hR",
        nrep_4c=(1, 1, 1), kmesh=(2, 2, 2), tier="C", machine=site_settings.system_host('al2o3-corundum'),
        four_center_quantitative=False,
        crystal_ref="crystal/al2o3-corundum/b3lyp_pob-tzvp-rev2/INPUT.d12"),
    "alcl3": dict(   # monoclinic C2/m (base-centered, mS) — layered AlCl3
        build=lambda: _crystal("alcl3"),
        dim=3, klass="molecular", atoms=8, basis="sto-3g", bravais="mS",
        nrep_4c=(1, 1, 1), kmesh=(2, 2, 2), tier="C", machine=site_settings.system_host('alcl3'),
        four_center_quantitative=True,
        crystal_ref="crystal/alcl3/pw1pw_pob-tzvp-rev2/INPUT.d12"),
    "cscl": dict(   # cubic Pm-3m — CsCl-type simple cubic (cP). Cs ⇒ pob-tzvp-rev2.
        build=lambda: _crystal("cscl"),
        dim=3, klass="ionic", atoms=2, basis="pob-tzvp-rev2", bravais="cP",
        nrep_4c=(2, 2, 2), kmesh=(4, 4, 4), tier="C", machine=site_settings.system_host('cscl'),
        four_center_quantitative=False,
        crystal_ref="crystal/cscl/r2scan_pob-tzvp-rev2/INPUT.d12"),
    "boric-acid": dict(   # triclinic P-1 (aP) — H3BO3 layers. Large ⇒ symmetry/geom demo.
        build=lambda: _crystal("boric-acid"),
        dim=3, klass="molecular", atoms=28, basis="sto-3g", bravais="aP",
        nrep_4c=(1, 1, 1), kmesh=(2, 2, 2), tier="C", machine=site_settings.system_host('boric-acid'),
        four_center_quantitative=True,
        crystal_ref="crystal/boric-acid/rhf_sto-3g/INPUT.d12"),
    "na-bcc": dict(   # cubic Im-3m — body-centered cubic (cI). Metal ⇒ lattice/symmetry demo.
        build=lambda: _crystal("na-bcc"),
        dim=3, klass="metal", atoms=1, basis="sto-3g", bravais="cI",
        nrep_4c=(2, 2, 2), kmesh=(8, 8, 8), tier="C", machine=site_settings.system_host('na-bcc'),
        four_center_quantitative=False, backup="sc2o3-bixbyite",
        crystal_ref="crystal/na-bcc/r2scan_pob-tzvp-rev2/INPUT.d12"),

    # ---- backups for convergence-risky systems -------------------------------
    # The CCM/AICCM (like HF and the 2014 model) targets *gapped* systems; metals
    # have no gap, so na-bcc's SCF may oscillate. sc2o3-bixbyite is the robust
    # **cI insulator** fallback (Sc2O3, Ia-3) — a converged body-centered-cubic
    # reference for the paper if the metal won't settle. Other lattices already
    # carry insulating siblings (rocksalts, covalents, zincblendes) as backups.
    "sc2o3-bixbyite": dict(   # body-centered cubic INSULATOR (cI) — na-bcc backup
        build=lambda: _crystal("sc2o3-bixbyite"),
        dim=3, klass="oxide", atoms=40, basis="sto-3g", bravais="cI",
        nrep_4c=(1, 1, 1), kmesh=(2, 2, 2), tier="C", machine=site_settings.system_host('sc2o3-bixbyite'),
        four_center_quantitative=False, backup_for="na-bcc",
        crystal_ref="crystal/sc2o3/pw1pw_pob-tzvp-rev2/INPUT.d12"),
}


def build(name):
    """Return the PeriodicSystem for a registry entry."""
    return SYSTEMS[name]["build"]()


def basis_for(name):
    s = SYSTEMS[name]
    return vq.BasisSet(build(name).unit_cell_molecule(), s["basis"])


def _smoke():
    """Build every geometry + report size (no SCF)."""
    print(f"{'system':16}{'dim':>4}{'class':>12}{'atoms':>6}{'tier':>5}  basis     crystal_ref")
    for name, s in SYSTEMS.items():
        try:
            n = len(list(build(name).unit_cell_molecule().atoms))
            print(f"{name:16}{s['dim']:>4}{s['klass']:>12}{n:>6}{s['tier']:>5}  "
                  f"{s['basis']:9} {s['crystal_ref']}")
        except Exception as e:
            print(f"{name:16}  BUILD FAILED: {repr(e)[:60]}")


def _check():
    """Pre-flight: cluster-overlap conditioning at each benchmark nrep (no SCF).

    This campaign intentionally admits only positive, full-rank benchmark
    overlaps before expensive runs are queued. General CCM SCF routes may
    canonically screen below ``lindep_tol`` when occupied rank remains; this
    stronger full-rank check is a benchmark-comparability policy. Exit 1 if any
    system violates it."""
    import numpy as np
    from vibeqc.periodic.ccm import CCMSystem, ccm_overlap
    print(f"{'system':14}{'nrep':>9}{'nbf':>6}{'minEig S^CCM':>15}  status")
    bad = 0
    for name, s in SYSTEMS.items():
        try:
            ccm = CCMSystem(build(name), tuple(s["nrep_4c"]), s["basis"])
            emin = float(np.linalg.eigvalsh(ccm_overlap(ccm))[0])
            ok = emin > 1e-3
            status = "OK" if ok else ("ILL-COND" if emin > 1e-7 else "NON-PD")
            if emin <= 1e-7:
                bad += 1
            print(f"{name:14}{str(tuple(s['nrep_4c'])):>9}{ccm.nbf:>6}{emin:>15.2e}  {status}")
        except Exception as e:
            bad += 1
            print(f"{name:14}{str(tuple(s['nrep_4c'])):>9}  ERR: {repr(e)[:45]}")
    return bad


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        sys.exit(1 if _check() else 0)
    _smoke()
