"""DFTB0 and SCC-DFTB models for molecules.

Stage 1: DFTB0 (non-SCC) energy + finite-difference gradients.
Stage 2: Analytic gradients for DFTB0.
Stage 3: SCC-DFTB with charge self-consistency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vibeqc._vibeqc_core import Molecule as Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx

from .model import SemiempiricalModel
from .parameters import default_parameters
from .routes import SemiempiricalRoutePlan

# Attribute access on the submodule object -- NOT a dotted
# ``from vibeqc._vibeqc_core.semiempirical import ...``. The dotted form
# needs ``sys.modules['vibeqc._vibeqc_core.semiempirical']`` present, which
# is lost if any earlier-collected test drops vibeqc from sys.modules (the
# single-phase-init C extension does not re-register submodules on reimport).
SemiempiricalParameters = _se_cxx.SemiempiricalParameters
_run_dftb0_cxx = _se_cxx.run_dftb0


def _dftb_route_plan(
    method: str,
    mol: Molecule,
    *,
    unrestricted: bool | None = None,
) -> SemiempiricalRoutePlan:
    return SemiempiricalRoutePlan.from_request(
        method,
        boundary="molecule",
        charge=int(mol.charge),
        multiplicity=int(mol.multiplicity),
        unrestricted=unrestricted,
    )


def _restricted_dftb_route_plan(
    method: str,
    mol: Molecule,
    *,
    direct_name: str,
    unrestricted_name: str,
) -> SemiempiricalRoutePlan:
    plan = _dftb_route_plan(method, mol)
    if plan.spin == "unrestricted":
        raise NotImplementedError(
            f"{direct_name} is restricted; use {unrestricted_name} or "
            "run_semiempirical(...) for an open-shell molecule."
        )
    return plan


def _maybe_emit_dftb_citations(output, *, method_key: str) -> None:
    """Best-effort citation emission for the DFTB runners. Skipped
    when ``output`` is None; non-fatal on any failure."""
    if output is None:
        return
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(output, method=method_key, basis="sto-3g")
    except Exception:
        pass


# Radii (bohr) at which the pair repulsive is probed to identify the
# ``A/R^12`` placeholder branch of ``eval_repulsive``
# (cpp/include/vibeqc/semiempirical/repulsive_spline.hpp). Three points are
# used, not two, so that a Born-Mayer ``A*exp(-B*R)`` cannot impersonate the
# power law by matching a single ratio. The values bracket the bonding range
# where the placeholder's irrelevance was quantified in issue #306 (C-C at
# 2.90 bohr = 1.69580e-4 Ha against genuine DFTB repulsives of order 1e-1 Ha).
_R12_PROBE_RADII_BOHR = (2.0, 3.0, 4.0)
_R12_PROBE_RTOL = 1.0e-9


def _is_r12_placeholder(
    params: SemiempiricalParameters,
    z1: int,
    z2: int,
) -> bool:
    """True when ``(z1, z2)``'s repulsive is evaluated by the ``A/R^12``
    placeholder branch (issue #306).

    The predicate is a *functional* probe of ``repulsive_energy`` rather than
    a table-membership test, for two reasons. First, it cannot drift from
    what the evaluator actually does. Second, table membership does not
    answer the question: ``dftb0_default`` stores every built-in pair as
    ``{A, B = 0}`` with an empty spline, so the tabulated tier and the
    combining-rule fallback take the identical ``A/R^12`` branch and differ
    only in the value of ``A``. Measured on the shipped set, ``E * R^12`` is
    constant to full double precision for C-C (60), H-H (5), O-O (40), H-O
    (15), N-N (80) exactly as it is for Si-Si (9) and Mg-O (11.1475).

    A spline-fitted pair and a Born-Mayer ``A*exp(-B*R)`` pair both fail the
    power-law test, which is the discrimination this predicate exists for.
    """
    values = [params.repulsive_energy(z1, z2, r) for r in _R12_PROBE_RADII_BOHR]
    if not all(v > 0.0 for v in values):
        # A spline past its cutoff, or an all-zero custom pair: not the
        # placeholder branch.
        return False
    scaled = [v * r**12 for v, r in zip(values, _R12_PROBE_RADII_BOHR, strict=True)]
    reference = scaled[0]
    return all(abs(s - reference) <= _R12_PROBE_RTOL * reference for s in scaled[1:])


def placeholder_repulsive_pairs(
    params: SemiempiricalParameters,
    atomic_numbers,
) -> list[tuple[int, int]]:
    """Distinct element pairs in ``atomic_numbers`` whose repulsive potential
    is the ``A/R^12`` placeholder (issue #306). Unique, sorted, and
    independent of input order.

    This covers both tiers: pairs absent from the table (combining-rule ``A``)
    and pairs present in it (hand-rounded ``A``). Both are the same
    placeholder functional form -- see :func:`_is_r12_placeholder`.
    """
    present = sorted({int(z) for z in atomic_numbers})
    placeholders: list[tuple[int, int]] = []
    for i, z1 in enumerate(present):
        for z2 in present[i:]:
            if _is_r12_placeholder(params, z1, z2):
                placeholders.append((z1, z2))
    return placeholders


def warn_placeholder_repulsives(
    params: SemiempiricalParameters,
    atomic_numbers,
    *,
    route: str,
) -> None:
    """Emit a loud warning when a run consumes a repulsive pair carried by the
    ``A/R^12`` placeholder, so a placeholder number is never promoted as
    chemistry (issue #306). Fixed-geometry differences stay exactly valid.
    Duck-typed parameter stand-ins without a ``repulsive_energy`` probe
    are stored verbatim by the models and cannot be inspected; the warning
    is skipped for them."""
    if getattr(params, "repulsive_energy", None) is None:
        return
    pairs = placeholder_repulsive_pairs(params, atomic_numbers)
    if not pairs:
        return
    import warnings

    from vibeqc.semiempirical import DFTB0RepulsivePlaceholderWarning

    labels = ", ".join(f"{z1}-{z2}" for z1, z2 in pairs)
    warnings.warn(
        f"{route}: the DFTB parameter set carries no fitted repulsive for "
        f"the pair(s) {labels}; their potential is the A/R^12 placeholder "
        "(a tabulated A where one exists, a combining-rule A otherwise), "
        "which is numerically irrelevant at equilibrium separations, so "
        "absolute energies, equilibrium geometries, and EOS fits for this "
        "system are placeholder physics. Fixed-geometry differences (k-mesh "
        "or supercell convergence, Madelung on/off, Gamma vs multi-k) remain "
        "exactly valid. See docs/user_guide/semiempirical.md.",
        category=DFTB0RepulsivePlaceholderWarning,
        stacklevel=2,
    )


def run_dftb0(
    mol: Molecule,
    params: Optional[SemiempiricalParameters] = None,
    *,
    output=None,
):
    """Run a one-shot DFTB0 energy calculation.

    ``output`` -- when set, writes ``{output}.bibtex`` and
    ``{output}.references`` carrying the Porezag-Frauenheim 1995
    DFTB0 citation alongside the STO-NG Slater-orbital
    Gaussian-expansion paper.
    """
    _restricted_dftb_route_plan(
        "dftb0",
        mol,
        direct_name="run_dftb0(...)",
        unrestricted_name="UDFTB0Model",
    )
    if params is None:
        params = default_parameters()
    warn_placeholder_repulsives(
        params, [atom.Z for atom in mol.atoms], route="run_dftb0"
    )
    result = _run_dftb0_cxx(mol, params)
    _maybe_emit_dftb_citations(output, method_key="dftb0")
    return result


def run_scc_dftb(
    mol: Molecule,
    params: Optional[SemiempiricalParameters] = None,
    *,
    output=None,
):
    """Run an SCC-DFTB energy calculation with charge self-consistency.

    ``output`` -- see :func:`run_dftb0`. The SCC route additionally
    carries the Elstner 1998 SCC-DFTB defining paper.
    """
    _restricted_dftb_route_plan(
        "scc_dftb",
        mol,
        direct_name="run_scc_dftb(...)",
        unrestricted_name="USCCDFTBModel",
    )
    if params is None:
        params = default_parameters()
    warn_placeholder_repulsives(
        params, [atom.Z for atom in mol.atoms], route="run_scc_dftb"
    )
    result = _se_cxx.run_scc_dftb(mol, params)
    _maybe_emit_dftb_citations(output, method_key="scc_dftb")
    return result


class DFTB0Model(SemiempiricalModel):
    """Non-self-consistent tight-binding model (DFTB0)."""

    def __init__(self, mol: Molecule, params: Optional[SemiempiricalParameters] = None):
        super().__init__(mol)
        self._route_plan = _restricted_dftb_route_plan(
            "dftb0",
            mol,
            direct_name="DFTB0Model",
            unrestricted_name="UDFTB0Model",
        )
        self._params = default_parameters() if params is None else params
        warn_placeholder_repulsives(
            self._params,
            [atom.Z for atom in mol.atoms],
            route="DFTB0Model",
        )
        self._last_result = None

    @property
    def params(self) -> SemiempiricalParameters:
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        """Exact identity of the parameters consumed by the last run."""
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        """Canonical parameter hash bound to the last native result."""
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    def energy(self) -> float:
        self._last_result = _run_dftb0_cxx(self._mol, self._params)
        return float(self._last_result.energy)

    def gradient(self) -> np.ndarray:
        result = self._last_result
        if result is None:
            result = _run_dftb0_cxx(self._mol, self._params)
            self._last_result = result
        return np.asarray(
            _se_cxx.compute_dftb0_gradient(self._mol, result, self._params)
        )

    def _energy_at(self, mol: Molecule) -> float:
        return float(_run_dftb0_cxx(mol, self._params).energy)


class SCCDFTBModel(SemiempiricalModel):
    """Self-consistent-charge tight-binding model (SCC-DFTB).

    Adds charge self-consistency via Mulliken analysis to DFTB0.
    Provides improved charge distributions and more physical energetics.

    Parameters
    ----------
    mol : Molecule
        Closed-shell molecular geometry (bohr).
    params : SemiempiricalParameters or None
        Parameter set (must include Hubbard U). Defaults to H/C/N/O table.
    charge_mixing : float
        Fraction of new charges to mix per iteration (0-1, default 0.2).
    max_iter : int
        Maximum SCC iterations (default 100).
    conv_tol_charge : float
        Convergence threshold for max charge change (default 1e-6).
    electronic_temperature : float
        Fermi-Dirac occupation temperature in Hartree (default 0).
    initial_charges : array-like or None
        Optional Mulliken charge-fluctuation guess, one value per atom.
    """

    def __init__(
        self,
        mol: Molecule,
        params: Optional[SemiempiricalParameters] = None,
        *,
        charge_mixing: float = 0.2,
        max_iter: int = 100,
        conv_tol_charge: float = 1e-6,
        electronic_temperature: float = 0.0,
        initial_charges: Optional[np.ndarray] = None,
        use_diis: bool = False,
    ):
        super().__init__(mol)
        self._route_plan = _restricted_dftb_route_plan(
            "scc_dftb",
            mol,
            direct_name="SCCDFTBModel",
            unrestricted_name="USCCDFTBModel",
        )
        self._params = default_parameters() if params is None else params
        warn_placeholder_repulsives(
            self._params,
            [atom.Z for atom in mol.atoms],
            route="SCCDFTBModel",
        )
        self._charge_mixing = charge_mixing
        self._max_iter = max_iter
        self._conv_tol_charge = conv_tol_charge
        self._electronic_temperature = electronic_temperature
        self._initial_charges = (
            np.asarray(initial_charges, dtype=np.float64)
            if initial_charges is not None
            else None
        )
        self._use_diis = use_diis
        self._last_result = None

    @property
    def params(self) -> SemiempiricalParameters:
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        """Exact identity of the parameters consumed by the last run."""
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        """Canonical parameter hash bound to the last native result."""
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    @property
    def converged(self) -> bool:
        """Whether the most recent SCC evaluation converged."""
        return bool(self._last_result is not None and self._last_result.converged)

    @property
    def n_iter(self) -> int:
        """SCC iterations used by the most recent evaluation."""
        return (
            int(self._last_result.n_iter) if self._last_result is not None else 0
        )

    def energy(self) -> float:
        self._last_result = self._run_at(self._mol)
        return float(self._last_result.energy)

    def gradient(self) -> np.ndarray:
        """Analytic SCC-DFTB nuclear gradient dE/dR (Hartree/bohr).

        The SCC energy is variational in the density, so the analytic
        fixed-charge gradient (band + Pulay + Mulliken-overlap + gamma
        + repulsive terms) is exact at the converged SCC solution;
        ``compute_scc_dftb_gradient_response`` is a kept-for-API alias
        of the same formula.
        """
        result = self._last_result
        if result is None:
            result = self._run_at(self._mol)
            self._last_result = result
        return np.asarray(
            _se_cxx.compute_scc_dftb_gradient_response(self._mol, result, self._params)
        )

    def _run_at(self, mol: Molecule):
        opts = _se_cxx.SCCOptions()
        opts.charge_mixing = self._charge_mixing
        opts.max_iter = self._max_iter
        opts.conv_tol_charge = self._conv_tol_charge
        opts.electronic_temperature = self._electronic_temperature
        opts.use_diis = self._use_diis
        if self._initial_charges is not None:
            opts.initial_charges = self._initial_charges
        return _se_cxx.run_scc_dftb(mol, self._params, opts)

    def _energy_at(self, mol: Molecule) -> float:
        return float(self._run_at(mol).energy)


class UDFTB0Model(SemiempiricalModel):
    """Unrestricted non-self-consistent tight-binding model.

    Supports open-shell molecules (doublets, triplets, etc.).

    Parameters
    ----------
    mol : Molecule
        Molecular geometry (bohr). Multiplicity > 1 for open-shell.
    params : SemiempiricalParameters or None
        Parameter set. Defaults to the built-in table.
    """

    def __init__(self, mol: Molecule, params=None):
        super().__init__(mol)
        self._route_plan = _dftb_route_plan(
            "dftb0",
            mol,
            unrestricted=True,
        )
        self._params = default_parameters() if params is None else params
        warn_placeholder_repulsives(
            self._params,
            [atom.Z for atom in mol.atoms],
            route="UDFTB0Model",
        )
        self._last_result = None

    @property
    def params(self):
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        """Exact identity of the parameters consumed by the last run."""
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        """Canonical parameter hash bound to the last native result."""
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    def energy(self) -> float:
        self._last_result = _se_cxx.run_udftb0(self._mol, self._params)
        return float(self._last_result.energy)

    def gradient(self) -> np.ndarray:
        """Analytic nuclear gradient dE/dR (Hartree/bohr)."""
        result = self._last_result
        if result is None:
            result = _se_cxx.run_udftb0(self._mol, self._params)
            self._last_result = result
        return np.asarray(
            _se_cxx.compute_udftb0_gradient(self._mol, result, self._params)
        )

    def _energy_at(self, mol: Molecule) -> float:
        return float(_se_cxx.run_udftb0(mol, self._params).energy)


class USCCDFTBModel(SemiempiricalModel):
    """Unrestricted self-consistent-charge tight-binding model.

    Supports open-shell molecules with charge self-consistency.

    Parameters
    ----------
    mol : Molecule
        Molecular geometry (bohr). Multiplicity > 1 for open-shell.
    params : SemiempiricalParameters or None
    charge_mixing : float
        Fraction of new charges per iteration (default 0.2).
    max_iter : int
        Maximum SCC iterations (default 100).
    initial_charges : array-like or None
        Optional Mulliken charge-fluctuation guess, one value per atom.
    """

    def __init__(
        self,
        mol: Molecule,
        params=None,
        *,
        charge_mixing=0.2,
        max_iter=100,
        conv_tol_charge=1e-6,
        initial_charges=None,
    ):
        super().__init__(mol)
        self._route_plan = _dftb_route_plan(
            "scc_dftb",
            mol,
            unrestricted=True,
        )
        self._params = default_parameters() if params is None else params
        warn_placeholder_repulsives(
            self._params,
            [atom.Z for atom in mol.atoms],
            route="USCCDFTBModel",
        )
        self._charge_mixing = charge_mixing
        self._max_iter = max_iter
        self._conv_tol_charge = conv_tol_charge
        self._initial_charges = (
            None
            if initial_charges is None
            else np.asarray(initial_charges, dtype=float)
        )
        self._last_result = None

    @property
    def params(self):
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        """Exact identity of the parameters consumed by the last run."""
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        """Canonical parameter hash bound to the last native result."""
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    @property
    def converged(self) -> bool:
        """Whether the most recent unrestricted SCC evaluation converged."""
        return bool(self._last_result is not None and self._last_result.converged)

    @property
    def n_iter(self) -> int:
        """SCC iterations used by the most recent evaluation."""
        return (
            int(self._last_result.n_iter) if self._last_result is not None else 0
        )

    def energy(self) -> float:
        self._last_result = self._run_at(self._mol)
        return float(self._last_result.energy)

    def gradient(self) -> np.ndarray:
        result = self._last_result
        if result is None:
            result = self._run_at(self._mol)
            self._last_result = result
        return np.asarray(
            _se_cxx.compute_uscc_dftb_gradient(self._mol, result, self._params)
        )

    def _run_at(self, mol: Molecule):
        opts = _se_cxx.SCCOptions()
        opts.charge_mixing = self._charge_mixing
        opts.max_iter = self._max_iter
        opts.conv_tol_charge = self._conv_tol_charge
        if self._initial_charges is not None:
            opts.initial_charges = self._initial_charges
        return _se_cxx.run_uscc_dftb(mol, self._params, opts)

    def _energy_at(self, mol: Molecule) -> float:
        return float(self._run_at(mol).energy)
