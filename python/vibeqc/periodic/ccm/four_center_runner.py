"""Runner adapter for the union-and-weight four-centre Γ-CCM (EXPERIMENTAL).

The third sibling of :mod:`real_gamma_runner` and :mod:`neutral_bloch_runner`,
for ``run_periodic_job(method="aiccm", variant="four-center")``. This is the
line the family is named after: the 2014 union-and-weight / Wigner-Seitz
construction (Peintinger & Bredow), whose finite Hamiltonian applies the WSSC
four-centre weights to a bare 1/r lattice sum on the torus. Ruling R1 keeps it
distinct from the two neutral producers: a difference between this variant and
``real-gamma`` / ``neutral-bloch`` is a CONSTRUCTION difference, never a
convergence or representation artefact, and nothing here may be compared with
them as if it were a parity check (D74, D89).

Two facts decide everything this module does, and both were measured rather
than read:

* **The library drivers return the TOTAL cyclic-cluster energy**, not a
  per-cell one, for all three of HF, KS and UHF. On H2/STO-3G in a 6-bohr cube
  at ``nrep=(2,1,1)`` each is exactly twice its ``(1,1,1)`` value. The
  ``CCMSCFResult`` docstring says so; the ``CCMKSResult`` and ``CCMUHFResult``
  field comments claimed "per reference cell" and were wrong by a factor of
  ``N_c`` (corrected in the same commit as this module). This adapter divides.
* **The orbitals span the supercell.** Like the real-Γ producer, the SCF
  solves one supercell-Γ eigenproblem, so ``mo_coeffs`` is ``(N_c*n_mu)`` wide.
  The density is folded back to unit-cell lattice blocks here; the orbitals are
  retained verbatim and the runner's unit-cell artefact gate (GitLab #654,
  #655) keeps them out of any artefact that pairs orbitals with the unit
  cell's basis.

The front door never exposes ``union12``. The library drivers still default to
it, and its Coulomb supermatrix carries a negative subspace on any basis with
more than one function per centre (GitLab #242), so this adapter always passes
``method="aiccm2026dev-a"``, the symmetric Born-von Karman-torus four-centre of
record. Changing the LIBRARY default is a separate, maintainer-approved change
(D-6) that moves numbers for direct callers.

HF runs on the scalable builder (D-6b): ``run_ccm_rhf_scalable`` applies the
weights inside the shell-quartet loop with O(nbf^2) peak memory, where the
dense ``run_ccm_rhf`` materialises the O(nbf^4) effective ERI tensor and runs
out of memory on LiH rocksalt (2,2,2)/STO-3G. The dense builder stays reachable
as the library entry, and remains the validation reference the two agree
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..._vibeqc_core import BasisSet, LatticeSumOptions, PeriodicSystem

__all__ = [
    "CCMFourCentreConvention",
    "CCMFourCentreResult",
    "FOUR_CENTRE_METHODS",
    "FOUR_CENTRE_WEIGHTING",
    "run_four_center_scf",
]

#: The SCF references this arm implements.
FOUR_CENTRE_METHODS = ("RHF", "RKS", "UHF", "UKS")

#: The four-centre weighting the front door always executes. Never
#: ``"union12"``: see the module docstring and GitLab #242.
FOUR_CENTRE_WEIGHTING = "aiccm2026dev-a"

#: The four-centre SCF loops implement no Fock mixing, damping or level shift,
#: exactly like the direct-torus loops. Recorded as executed values (D86)
#: rather than inferred by a reader.
_EXECUTED_FOCK_MIXING = 0.0
_EXECUTED_DAMPING = 0.0
_EXECUTED_LEVEL_SHIFT = 0.0


@dataclass(frozen=True)
class CCMFourCentreConvention:
    """Machine-readable convention record of a four-centre Γ-CCM run."""

    route: str                        # "four-center"
    #: What was BUILT, not how it was represented. The neutral producers
    #: record "none (neutral fitted-torus representation control)" here
    #: because they are representations of a different Hamiltonian; this line
    #: is the construction itself, so the two strings must never converge.
    ccm_construction: str
    #: The four-centre weighting executed (always the symmetric BvK-torus one
    #: through the front door; the library default union12 is refused there).
    weighting: str
    #: How the weighted quartets were contracted: "direct" is integral-direct.
    four_center_contraction: str
    #: Whether the scalable C++ lattice-sum JK builder ran (HF) or the Python
    #: padded route did.
    builder: str
    lattice_vector_convention: str    # "columns"
    nrep: tuple
    n_cells: int
    executed_fock_mixing: float
    executed_damping: float
    executed_level_shift: float
    #: The bare 1/r minimum-image lattice sum puts every channel in one gauge,
    #: so this construction has no exchange-q=0 seam to declare. Recorded as
    #: an explicit "not-applicable" rather than left absent, so a reader
    #: comparing the three AICCM variants sees why the field differs.
    exchange_q0: str = "not-applicable (bare 1/r minimum image)"
    exchange_q0_applicability: str = "inactive"
    experimental: bool = True


@dataclass
class CCMFourCentreResult:
    """Per-unit-cell runner-contract view of a four-centre CCM result."""

    converged: bool
    n_iter: int
    energy: float                     # per unit cell (Ha)
    backend: str                      # "ccm-four-center"
    system: PeriodicSystem
    mo_energies: np.ndarray           # supercell-Γ eigenvalues
    mo_coeffs: np.ndarray             # supercell-Γ MOs (N_c*n_mu AO rows)
    overlap: np.ndarray               # supercell-Γ overlap
    fock: Optional[np.ndarray]
    hcore: Optional[np.ndarray]
    density: object                   # unit-cell LatticeMatrixSet (folded)
    effective_n_electrons: int        # per supercell
    four_center: CCMFourCentreConvention
    ccm_result: object                # the underlying CCM API result
    #: The :class:`~vibeqc.periodic.ccm.system.CCMSystem` this SCF ran on.
    #: Kept so a correlation driver can be handed the SAME cluster the
    #: reference converged on (M4b, #778) instead of rebuilding one and
    #: hoping the two agree. Not a copy -- the identical object.
    ccm_system: object = None
    #: The post-HF result when ``run_periodic_job(correlation=...)`` asked for
    #: one, else ``None`` (M4b, #778). A ``CCMMP2Result`` / ``CCMUMP2Result``
    #: from the BARE four-centre drivers, which are construction-matched to
    #: this SCF (maintainer ruling 2026-09-08). Its own energy fields are
    #: TOTAL cyclic-cluster values, like every driver in this package; the
    #: per-unit-cell numbers are :attr:`e_correlation` and
    #: :attr:`e_total_correlated` below, which this adapter divides.
    correlation: object = None
    #: ``correlation.e_correlation / N_c`` (Ha per unit cell), or None.
    e_correlation: Optional[float] = None
    #: SCF + correlation, per unit cell (Ha), or None.
    e_total_correlated: Optional[float] = None
    fock_mixing: float = _EXECUTED_FOCK_MIXING
    functional: Optional[str] = None
    e_xc: Optional[float] = None      # per unit cell (KS)
    e_hf_exchange: Optional[float] = None
    # Open-shell fields (None closed-shell)
    mo_energies_beta: Optional[np.ndarray] = None
    mo_coeffs_beta: Optional[np.ndarray] = None
    fock_beta: Optional[np.ndarray] = None
    density_alpha: Optional[object] = None
    density_beta: Optional[object] = None
    guess_selection: object = None


def _fold_density_to_lattice_set(ccm, D_sc, lat_opts=None):
    """Supercell-Γ density -> unit-cell ``LatticeMatrixSet`` (circulant fold).

    The same projection the real-Γ adapter uses, and exact for the converged
    translation-invariant state.
    """
    from ..._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set
    from .direct import _supercell_density_to_lattice_blocks

    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    n_mu = int(ubasis.nbasis)
    base = lat_opts if lat_opts is not None else LatticeSumOptions()
    cells = direct_lattice_cells(unit, float(base.cutoff_bohr))
    blocks = _supercell_density_to_lattice_blocks(
        np.asarray(D_sc, dtype=float), ccm.nrep, n_mu, cells)
    return make_lattice_matrix_set(n_mu, cells, blocks)


def run_four_center_scf(
    system: PeriodicSystem,
    basis_name: str,
    method: str,
    mesh: Sequence[int],
    *,
    initial_guess: object = "AUTO",
    functional: Optional[str] = None,
    four_center: str = "direct",
    schwarz_threshold: float = 0.0,
    max_iter: int = 128,
    conv_tol: float = 1e-8,
    lat_opts: Optional[LatticeSumOptions] = None,
    grid_options: Optional[object] = None,
    becke_image_radius_bohr: float = 10.0,
) -> CCMFourCentreResult:
    """Run the union-and-weight four-centre Γ-CCM SCF on the BvK torus.

    ``method`` is one of :data:`FOUR_CENTRE_METHODS` (the runner's
    ``method_upper``); ``mesh`` is the BvK ``nrep``. The weighting is always
    :data:`FOUR_CENTRE_WEIGHTING`; ``union12`` is not reachable from here.
    For KS methods, ``grid_options`` and ``becke_image_radius_bohr`` control
    the periodic reference-cluster grid, while ``lat_opts`` controls both the
    translated-AO XC domain and the folded unit-cell density returned to the
    unified runner. When a direct caller leaves ``lat_opts`` unset for a
    full-grid external functional, both paths use the provider contract's
    25-bohr default instead of resolving different implicit cutoffs.
    """
    from .dft import run_ccm_rks, run_ccm_uks
    from .scf import run_ccm_rhf_scalable
    from .system import CCMSystem
    from .uhf import run_ccm_uhf

    from ...guess import InitialGuess, select_initial_guess
    from ...kpoints import _integer_counts

    nrep = tuple(_integer_counts(mesh, name="four-center BvK mesh"))
    selection = select_initial_guess(
        system.unit_cell_molecule(), initial_guess, is_periodic=True,
        is_open_shell=str(method).strip().upper() in ("UHF", "UKS"),
        supported=(InitialGuess.HCORE,), driver="run_four_center_scf",
    )

    method_upper = str(method).strip().upper()
    if len(nrep) != 3 or any(n < 1 for n in nrep):
        raise ValueError(
            "run_four_center_scf: mesh must be three positive integers "
            f"(the BvK nrep); got {mesh!r}"
        )
    if method_upper not in FOUR_CENTRE_METHODS:
        raise NotImplementedError(
            "variant='four-center' implements "
            f"{', '.join(FOUR_CENTRE_METHODS)}; got method={method_upper!r}. "
            "Post-HF on this line is library-only "
            "(vibeqc.periodic.ccm.mp2 / .ccsd)."
        )
    if method_upper in ("RKS", "UKS") and functional is None:
        raise ValueError(
            f"run_four_center_scf: method={method_upper!r} requires "
            "functional=..."
        )

    resolved_lat_opts = lat_opts
    if resolved_lat_opts is None and method_upper in ("RKS", "UKS"):
        from vibeqc import Functional

        spin = 2 if method_upper == "UKS" else 1
        if bool(getattr(Functional(functional, spin), "is_external", False)):
            # Keep this direct adapter consistent with
            # PeriodicExternalXC.prepare's external-provider default. The
            # same object is passed into the SCF and density-folding paths.
            resolved_lat_opts = LatticeSumOptions()
            resolved_lat_opts.cutoff_bohr = 25.0

    ccm = CCMSystem(system, nrep, basis_name)
    if method_upper == "RHF":
        res = run_ccm_rhf_scalable(
            ccm, method=FOUR_CENTRE_WEIGHTING, four_center=four_center,
            schwarz_threshold=float(schwarz_threshold),
            max_iter=int(max_iter), conv_tol=float(conv_tol),
            initial_guess=selection.requested,
        )
        builder = "scalable-cxx-lattice-sum-jk"
    elif method_upper == "UHF":
        # No scalable open-shell builder exists yet: the dense padded route is
        # the only UHF entry, so this arm inherits its cluster-size ceiling.
        res = run_ccm_uhf(
            ccm, method=FOUR_CENTRE_WEIGHTING, max_iter=int(max_iter),
            conv_tol=float(conv_tol),
            initial_guess=selection.requested,
        )
        builder = "dense-python-padded"
    elif method_upper == "RKS":
        res = run_ccm_rks(
            ccm, functional, method=FOUR_CENTRE_WEIGHTING,
            four_center=four_center,
            schwarz_threshold=float(schwarz_threshold),
            max_iter=int(max_iter), conv_tol=float(conv_tol),
            grid_options=grid_options,
            becke_image_radius_bohr=float(becke_image_radius_bohr),
            xc_lattice_options=resolved_lat_opts,
            initial_guess=selection.requested,
        )
        builder = "scalable-cxx-lattice-sum-jk"
    else:  # UKS
        res = run_ccm_uks(
            ccm, functional, method=FOUR_CENTRE_WEIGHTING,
            four_center=four_center,
            schwarz_threshold=float(schwarz_threshold),
            max_iter=int(max_iter), conv_tol=float(conv_tol),
            grid_options=grid_options,
            becke_image_radius_bohr=float(becke_image_radius_bohr),
            xc_lattice_options=resolved_lat_opts,
            initial_guess=selection.requested,
        )
        builder = "scalable-cxx-lattice-sum-jk"

    n_c = int(ccm.n_cells)
    convention = CCMFourCentreConvention(
        route="four-center",
        ccm_construction=(
            "union-and-weight Wigner-Seitz four-centre (Gamma-CCM, "
            "Peintinger-Bredow 2014 lineage)"
        ),
        weighting=FOUR_CENTRE_WEIGHTING,
        four_center_contraction=str(four_center),
        builder=builder,
        lattice_vector_convention="columns",
        nrep=nrep,
        n_cells=n_c,
        executed_fock_mixing=_EXECUTED_FOCK_MIXING,
        executed_damping=_EXECUTED_DAMPING,
        executed_level_shift=_EXECUTED_LEVEL_SHIFT,
    )

    open_shell = (
        getattr(res, "density_alpha", None) is not None
        and getattr(res, "density_beta", None) is not None
    )
    D_total = getattr(res, "density", None)
    if D_total is None:
        D_total = np.asarray(res.density_alpha) + np.asarray(res.density_beta)
    density_set = _fold_density_to_lattice_set(
        ccm, D_total, resolved_lat_opts
    )
    density_alpha_set = density_beta_set = None
    if open_shell:
        density_alpha_set = _fold_density_to_lattice_set(
            ccm, res.density_alpha, resolved_lat_opts)
        density_beta_set = _fold_density_to_lattice_set(
            ccm, res.density_beta, resolved_lat_opts)

    mo_e = getattr(res, "mo_energies", None)
    if mo_e is None:
        mo_e = res.mo_energies_alpha
    mo_c = getattr(res, "mo_coeffs", None)
    if mo_c is None:
        mo_c = res.mo_coeffs_alpha

    overlap = np.asarray(res.overlap)
    fock_value = getattr(res, "fock", None)
    hcore_value = getattr(res, "hcore", None)
    fock_beta_value = getattr(res, "fock_beta", None)
    expected_shape = tuple(overlap.shape)
    for name, value in (
        ("fock", fock_value),
        ("hcore", hcore_value),
        ("fock_beta", fock_beta_value),
    ):
        if value is not None and tuple(np.asarray(value).shape) != expected_shape:
            raise RuntimeError(
                "run_four_center_scf: four-centre result operator "
                f"{name} has shape {np.asarray(value).shape}, expected the "
                f"supercell-Gamma AO shape {expected_shape}"
            )

    return CCMFourCentreResult(
        converged=bool(res.converged),
        n_iter=int(res.n_iter),
        # Measured, not assumed: every four-centre driver returns the TOTAL
        # cyclic-cluster energy (see the module docstring).
        energy=float(res.energy) / n_c,
        backend="ccm-four-center",
        system=system,
        ccm_system=ccm,
        mo_energies=np.asarray(mo_e),
        mo_coeffs=np.asarray(mo_c),
        overlap=overlap,
        fock=(np.asarray(fock_value) if fock_value is not None else None),
        hcore=(np.asarray(hcore_value) if hcore_value is not None else None),
        density=density_set,
        effective_n_electrons=int(ccm.supercell.n_electrons()),
        four_center=convention,
        ccm_result=res,
        guess_selection=getattr(res, "guess_selection", None),
        functional=(functional if method_upper in ("RKS", "UKS") else None),
        e_xc=(float(res.e_xc) / n_c
              if getattr(res, "e_xc", None) is not None else None),
        e_hf_exchange=(
            float(res.e_hf_exchange) / n_c
            if getattr(res, "e_hf_exchange", None) is not None else None),
        mo_energies_beta=(
            np.asarray(res.mo_energies_beta)
            if getattr(res, "mo_energies_beta", None) is not None else None),
        mo_coeffs_beta=(
            np.asarray(res.mo_coeffs_beta)
            if getattr(res, "mo_coeffs_beta", None) is not None else None),
        fock_beta=(
            np.asarray(fock_beta_value)
            if fock_beta_value is not None else None),
        density_alpha=density_alpha_set,
        density_beta=density_beta_set,
    )
