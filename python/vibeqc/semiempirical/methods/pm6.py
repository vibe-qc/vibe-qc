"""PM6/UPM6 molecular models.

Provides :class:`PM6Model` (closed-shell) and :class:`UPM6Model`
(open-shell) as public API wrappers around the C++ PM6 NDDO backend.
These models follow the same pattern as :class:`DFTB0Model` and
:class:`GFN2Model`, implementing the :class:`SemiempiricalModel` contract.

.. note::

   PM6 is a molecular development NDDO path. H-only and H-heavy s/p
   interactions are source-correct, but the full heavy-heavy two-center tensor
   remains open. vibe-qc reports a PM6-like total energy rather than a MOPAC
   heat of formation, and fallback core-core / gamma terms apply wherever the
   bundled diatomic data is incomplete.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vibeqc import Molecule
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.model import SemiempiricalModel
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan


class PM6Model(SemiempiricalModel):
    """PM6 semiempirical model (closed-shell).

    Parameters
    ----------
    mol : Molecule
    params : PM6ParameterSet or None
        PM6 parameter set.  If ``None``, auto-selects the published Stewart
        2007 H/C/N/O/F set for those elements and the bundled MOPAC-derived
        parameter cache for wider element coverage.
    max_iter : int (default 100)
    conv_tol : float (default 1e-7)
    solver : str (default ``"dense"``)
        Eigensolver for the SCF step.  Currently a reserved keyword;
        the PM6 C++ backend always uses dense diagonalisation.
    """

    def __init__(
        self,
        mol: Molecule,
        params: Optional[_nddo.PM6ParameterSet] = None,
        *,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        self._route_plan = SemiempiricalRoutePlan.from_request(
            "pm6",
            boundary="molecule",
            charge=int(mol.charge),
            multiplicity=int(mol.multiplicity),
        )
        if self._route_plan.spin == "unrestricted":
            raise ValueError(
                "PM6Model is restricted; use UPM6Model or "
                "run_semiempirical(...) for an open-shell molecule."
            )
        if params is None:
            from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

            params = load_pm6_params_auto([atom.Z for atom in mol.atoms])
        if params.method_name() != "pm6":
            raise ValueError(
                "PM6Model requires PM6 parameters, not "
                f"{params.method_name()}"
            )
        self._params = params
        self._max_iter = max_iter
        self._conv_tol = conv_tol
        self._n_iter = 0
        self._converged = False
        self._last_result = None

    @property
    def params(self) -> _nddo.PM6ParameterSet:
        return self._params

    @property
    def n_iter(self) -> int:
        """SCF iterations taken by the most recent energy evaluation."""
        return self._n_iter

    @property
    def converged(self) -> bool:
        """Whether the most recent SCF evaluation converged."""
        return self._converged

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
        """Total energy (Hartree) at the current geometry."""
        result = _nddo.run_pm6(
            self._mol, self._params, self._max_iter, self._conv_tol
        )
        self._n_iter = int(result.n_iter)
        self._converged = bool(result.converged)
        self._last_result = result
        return float(result.energy)

    def _energy_at(self, mol: Molecule) -> float:
        result = _nddo.run_pm6(mol, self._params, self._max_iter, self._conv_tol)
        return float(result.energy)

    def gradient(self) -> np.ndarray:
        """Finite-difference gradient bound to the retained energy snapshot."""
        if self._last_result is None:
            self.energy()
        return np.asarray(
            _nddo.compute_pm6_gradient_fd_from_result(
                self._mol,
                self._params,
                self._last_result,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
            ),
            dtype=float,
        )


class UPM6Model(SemiempiricalModel):
    """Unrestricted PM6 model (open-shell).

    Parameters
    ----------
    mol : Molecule
        Must have multiplicity > 1 (i.e., at least one unpaired electron).
    params : PM6ParameterSet or None
    max_iter : int (default 100)
    conv_tol : float (default 1e-7)
    solver : str (default ``"dense"``)
        Eigensolver for the SCF step.  Currently a reserved keyword;
        the UPM6 C++ backend always uses dense diagonalisation.
    """

    def __init__(
        self,
        mol: Molecule,
        params: Optional[_nddo.PM6ParameterSet] = None,
        *,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        if mol.multiplicity <= 1:
            raise ValueError(
                "UPM6Model requires an open-shell molecule (multiplicity > 1). "
                "Use PM6Model for closed-shell calculations."
            )
        self._route_plan = SemiempiricalRoutePlan.from_request(
            "upm6",
            boundary="molecule",
            charge=int(mol.charge),
            multiplicity=int(mol.multiplicity),
        )
        if params is None:
            from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

            params = load_pm6_params_auto([atom.Z for atom in mol.atoms])
        if params.method_name() != "pm6":
            raise ValueError(
                "UPM6Model requires PM6 parameters, not "
                f"{params.method_name()}"
            )
        self._params = params
        self._max_iter = max_iter
        self._conv_tol = conv_tol
        self._n_iter = 0
        self._converged = False
        self._n_alpha = 0
        self._n_beta = 0
        self._last_result = None

    @property
    def params(self) -> _nddo.PM6ParameterSet:
        return self._params

    @property
    def n_iter(self) -> int:
        return self._n_iter

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def n_alpha(self) -> int:
        """Number of alpha electrons."""
        return self._n_alpha

    @property
    def n_beta(self) -> int:
        """Number of beta electrons."""
        return self._n_beta

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
        result = _nddo.run_upm6(
            self._mol, self._params, self._max_iter, self._conv_tol
        )
        self._n_iter = int(result.n_iter)
        self._converged = bool(result.converged)
        self._n_alpha = int(result.n_alpha)
        self._n_beta = int(result.n_beta)
        self._last_result = result
        return float(result.energy)

    def _energy_at(self, mol: Molecule) -> float:
        result = _nddo.run_upm6(mol, self._params, self._max_iter, self._conv_tol)
        return float(result.energy)

    def gradient(self) -> np.ndarray:
        """Finite-difference gradient bound to the retained energy snapshot."""
        if self._last_result is None:
            self.energy()
        return np.asarray(
            _nddo.compute_upm6_gradient_fd_from_result(
                self._mol,
                self._params,
                self._last_result,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
            ),
            dtype=float,
        )
