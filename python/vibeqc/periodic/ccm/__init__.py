"""Ab Initio Cyclic Cluster Model (AICCM).

A Γ-point Cyclic Cluster Model: a finite supercell of the crystal with
cyclic (Born-von Kármán) boundary conditions, where translational
symmetry is imposed by **Wigner-Seitz-supercell integral weighting**
rather than Bloch summation. The weighted integrals make the cluster
formally a molecular wavefunction, so vibe-qc's molecular method kernels
(HF/DFT and the post-HF correlation methods) apply to it.

This module is the **Γ-CCM** ``method="aiccm2026dev-a"`` line: the
union-and-weight/Wigner-Seitz integral-weighting approach within the
variational finite-BvK-torus CCM family. Its sibling ``aiccm2026dev-b``
(χ-CCM) uses the finite-translation-group character construction. The two are
distinct approaches compared at a declared common exchange-q=0 convention;
their distinction is not a choice of Coulomb kernels, and equality for a
specified operator and route is evidence to establish. ``Γ-CCM`` is the
descriptive name for prose; the code selector / method keyword stays exactly
``"aiccm2026dev-a"`` and selects the symmetric Wigner--Seitz four-center
construction. :func:`run_ccm_scf` therefore defaults to ``"four-center"``;
``"gamma"`` is an alias for that Γ-CCM construction, while
``"neutral-bloch"`` / ``"gdf"`` / ``"aiccm-ri"`` and ``"real-gamma"`` select
the neutral fitted-torus Bloch/real-Γ representation controls instead.

Those two controls are exactly Fourier-related after their common
block-circulant Hamiltonian is specified. This is the general same-H
representation theorem, not evidence that the Γ-CCM and χ-CCM constructions
are identical.

Reference
---------
M. F. Peintinger & T. Bredow, "The Cyclic Cluster Model at Hartree-Fock
Level", *J. Comput. Chem.* **35**, 839 (2014), doi:10.1002/jcc.23550.

Four-center weighting
---------------------
Two weightings are selectable on every driver via ``method=``:

* ``"union12"`` (default) -- the historical eq-18 product weight
  ``w_muν.1/2(w_mur+w_νr).w_rs``; validated in 1-D **in a minimal basis only**,
  breaks 8-fold ERI symmetry in >=2-D.

  The basis qualifier is not a caution, it is a measured limit (issue #242).
  The eq-18 weighted four-center is not a congruence of the padded ERI Gram
  matrix -- the bra selector and the weighted image-gathering map applied to
  the ket are different operators -- so it carries no positive-semidefinite
  guarantee, and it does carry a negative subspace. On the published 1-D
  alternating H chain the weighted supermatrix has 77 negative eigenvalues
  (minimum -0.236) while the CCM *overlap* is positive definite, and the SCF
  descends into that subspace: the traceable DZVP variants return -7.11 to
  -8.48 Ha/atom against a published -0.552592, with 3-21g at -1.97 and 6-31g
  at -4.92. STO-3G reproduces the
  published value to 3.2e-06 Ha/atom because a minimal basis cannot reach the
  offending sector, which is also why the historical validations did not see
  this. The negative subspace is a property of the printed equation, not of
  this reimplementation: deleting the entire eq-18 weighting leaves the
  minimum eigenvalue unchanged to within 1 %, and it is size-converged with
  the count of negative directions growing linearly in the cluster, so
  enlarging the cluster does not remove it either. Where the form is negative
  the nominal Coulomb energy has no lower bound, so there is no minimum for
  the SCF to find. Treat any ``union12`` energy on a basis with more than one
  function per centre as unbounded, not merely inaccurate.
* ``"aiccm2026dev-a"`` -- the symmetric Born-von Kármán-torus four-center
  (:func:`vibeqc.periodic.ccm.padded.ccm_eri_symmetric`,
  ``AICCM_ALGORITHM.md`` Sec.13): symmetric bridge ``1/4(w_mur+w_νr+w_mus+w_νs)`` with
  r, s folded independently to the home bra's WSC -- *exactly* 8-fold
  permutationally symmetric for any lattice.

The bare-1/r union-and-weight four-center and the neutral fitted-torus
:func:`~vibeqc.periodic.ccm.neutral.ccm_eri_neutral` are distinct finite
Hamiltonian constructions. Their observed 3-D gap must not be assigned to one
term without a clean term-by-term audit: their overlap, one-electron, nuclear,
and two-electron layers can all differ. In the molecular limit the leading
two-electron difference contains a rank-1 background ``ξ.S⊗S`` (``ξ`` = the
cell Madelung constant), but a finite-torus remainder survives. At a fixed
reference that rank-1 term has no occupied-virtual numerator element, while it
still shifts occupied-virtual denominators. The neutral cderi/GDF paths are
useful neutral fitted-torus controls and references for the post-HF research
stack; they are not a substitute for, or an RI implementation of, the
union-and-weight Γ-CCM construction. Quantitative absolute-energy claims must
therefore be route-specific and externally validated. See
:mod:`vibeqc.periodic.ccm.neutral`.

Status
------
**Experimental.** The whole Γ-CCM line is research-grade: every public SCF
driver emits :class:`~vibeqc.periodic.ccm.experimental.AICCM2026DevAExperimentalWarning`
(exported as ``vibeqc.AICCM2026DevAExperimentalWarning``; correlation, gradient,
and property drivers inherit it through the SCF they run). Filter via
``warnings.filterwarnings("ignore", category=vibeqc.AICCM2026DevAExperimentalWarning)``.

Implemented and validated (see ``handovers/HANDOVER_AICCM.md`` for the full map):
M1 Wigner-Seitz geometry + two-center weighted ``S^CCM``/``T^CCM`` +
overlap-spectrum guard (all lattices, arbitrary orbitals); M2a three-center
``V^CCM``; M2b four-center ``J^CCM``/``K^CCM`` + DIIS HF-CCM SCF. The full
molecular method stack runs on the CCM-weighted integrals:

* HF -- closed-shell :func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf` (~1e-5 vs
  Peintinger PhD Tab. 8.3) and open-shell
  :func:`~vibeqc.periodic.ccm.uhf.run_ccm_uhf`; the scalable C++ four-center
  (:func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`) reaches genuine 3-D.
* Correlation -- MP2 :func:`~vibeqc.periodic.ccm.mp2.run_ccm_mp2`, open-shell
  UMP2 :func:`~vibeqc.periodic.ccm.ump2.run_ccm_ump2`, and CCSD(T)
  :func:`~vibeqc.periodic.ccm.ccsd.run_ccm_ccsd`. All exact in the molecular
  limit (CCSD(T) matches molecular ``run_ccsd`` to the DF gap; (T) to ~1e-8).

The post-HF / padded-ERI builders are small / 1-D (feasibly 2-D) validation
paths -- ``run_ccm_*`` are importable but kept out of ``__all__`` pending the
3-D-scalable post-HF four-center. The neutral fitted-torus Bloch and real-Γ
controls are available for same-H representation audits. They must not be used
as a naming shortcut from ``aiccm2026dev-a`` to periodic GDF or as Γ-CCM /
χ-CCM construction evidence.
"""

from __future__ import annotations

from .integrals import ccm_kinetic, ccm_overlap, fold_lattice_matrix_set
from .convergence import (
    InteractionRangePoint,
    InteractionRangeScan,
    ccm_interaction_range_scan,
)
from .dlpno import (
    CCMDLPNOMP2Result,
    ccm_dlpno_mp2,
    ccm_pao,
    pao_occupied_orthogonality,
)
from .dlpno_ccsd import CCMDLPNOCCSDResult, ccm_dlpno_ccsd
from .dlpno_ccsd_coupled import (
    CCMDLPNOCoupledCCSDResult,
    ccm_dlpno_ccsd_coupled,
)
from .dlpno_ump2 import CCMDLPNOUMP2Result, ccm_dlpno_ump2
from .dlpno_uccsd import CCMDLPNOUCCSDResult, ccm_dlpno_uccsd
from .properties import (
    CCMBondAnalysis,
    CCMBondOrder,
    CCMGap,
    CCMPopulation,
    ccm_band_structure,
    ccm_dipole,
    ccm_homo_lumo_gap,
    ccm_mayer_bond_orders,
    ccm_lowdin_charges,
    ccm_mulliken_charges,
    ccm_numerical_gradient,
)
from .uccsd import CCMUCCSDResult, run_ccm_uccsd
from .localize import (
    WannierAliasingReport,
    localise_ccm,
    localization_aliasing,
    localization_density_residual,
)
from .symmetry import (
    analyze_ccm_symmetry,
    ccm_symmetry_basis_rotations,
    ccm_symmetry_invariance_residuals,
    ccm_symmetry_unique_atom_pairs,
)
from .neutral import (
    ccm_eri_neutral,
    ccm_neutral_background_constant,
    ccm_neutral_cderi,
    ccm_neutral_cderi_fold,
)
from .direct import (
    ccm_direct_oneelectron,
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_rhf_direct_rijcosx,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from .experimental import AICCM2026DevAExperimentalWarning
from .route import CCM_ROUTES, resolve_ccm_route, run_ccm_scf
from .lowd_four_center import (
    ccm_eri_wire,
    ccm_wire_cderi,
    ccm_wire_e_nn,
    ccm_wire_v_ne,
    wire_quadrature,
)
from .lowd_scf import run_ccm_rhf_wire
from .qvf import write_ccm_periodic_qvf
from .padded import ccm_hcore, ccm_nuclear
from .gradient_analytic import (
    run_ccm_ccsd_gradient,
    run_ccm_mp2_gradient,
    run_ccm_rhf_gradient,
    run_ccm_rks_gradient,
    run_ccm_uccsd_gradient,
    run_ccm_uhf_gradient,
    run_ccm_uks_gradient,
    run_ccm_ump2_gradient,
)
from .system import CCMSystem
from .wigner_seitz import (
    first_shell_vectors,
    interplanar_spacings,
    kspacing_for_interaction_range,
    min_image_multiplicity,
    minimum_image,
    nrep_for_interaction_range,
    wsc_inscribed_radius,
)

__all__ = [
    "AICCM2026DevAExperimentalWarning",
    "CCMSystem",
    "ccm_overlap",
    "ccm_kinetic",
    "ccm_nuclear",
    "ccm_hcore",
    "interplanar_spacings",
    "wsc_inscribed_radius",
    "nrep_for_interaction_range",
    "kspacing_for_interaction_range",
    "ccm_interaction_range_scan",
    "InteractionRangeScan",
    "InteractionRangePoint",
    "ccm_eri_neutral",
    "ccm_eri_wire",
    "ccm_wire_cderi",
    "ccm_wire_e_nn",
    "ccm_wire_v_ne",
    "wire_quadrature",
    "run_ccm_rhf_wire",
    "write_ccm_periodic_qvf",
    "ccm_neutral_cderi",
    "ccm_neutral_background_constant",
    "run_ccm_rhf_direct",
    "run_ccm_uhf_direct",
    "run_ccm_rhf_direct_rijcosx",
    "run_ccm_rks_direct",
    "run_ccm_uks_direct",
    "ccm_direct_oneelectron",
    "ccm_exchange_q0_madelung",
    "run_ccm_scf",
    "resolve_ccm_route",
    "CCM_ROUTES",
    "localise_ccm",
    "localization_aliasing",
    "localization_density_residual",
    "WannierAliasingReport",
    "analyze_ccm_symmetry",
    "ccm_symmetry_basis_rotations",
    "ccm_symmetry_invariance_residuals",
    "ccm_symmetry_unique_atom_pairs",
    "ccm_pao",
    "pao_occupied_orthogonality",
    "ccm_dlpno_mp2",
    "CCMDLPNOMP2Result",
    "ccm_dlpno_ccsd",
    "CCMDLPNOCCSDResult",
    "ccm_dlpno_ccsd_coupled",
    "CCMDLPNOCoupledCCSDResult",
    "ccm_dlpno_ump2",
    "CCMDLPNOUMP2Result",
    "ccm_dlpno_uccsd",
    "CCMDLPNOUCCSDResult",
    "ccm_band_structure",
    "ccm_homo_lumo_gap",
    "ccm_mayer_bond_orders",
    "ccm_mulliken_charges",
    "ccm_lowdin_charges",
    "ccm_dipole",
    "ccm_numerical_gradient",
    "run_ccm_rhf_gradient",
    "run_ccm_uhf_gradient",
    "run_ccm_rks_gradient",
    "run_ccm_uks_gradient",
    "run_ccm_mp2_gradient",
    "run_ccm_ump2_gradient",
    "run_ccm_ccsd_gradient",
    "run_ccm_uccsd_gradient",
    "CCMBondAnalysis",
    "CCMBondOrder",
    "CCMGap",
    "CCMPopulation",
    "run_ccm_uccsd",
    "CCMUCCSDResult",
    "fold_lattice_matrix_set",
    "minimum_image",
    "min_image_multiplicity",
    "first_shell_vectors",
]
