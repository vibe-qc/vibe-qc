"""OMx molecular models (OM1, OM2, OM3).

Provides :class:`OMxModel` as a public API wrapper around the C++ OMx
NDDO backend with Loewdin orthogonalization.  Supports all three OMx
variants via the ``variant`` parameter.

.. note::

   OM2/OM3 are validated for molecular development and prescreening on the
   published H/C/N/O/F surface, within documented integral stand-ins and
   benchmark residuals; external implementation parity remains open.
   OM1 remains experimental until the Kolb-Thiel core-valence ECP is ported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vibeqc import Molecule
from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
from vibeqc.semiempirical.model import SemiempiricalModel
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan


class OMxModel(SemiempiricalModel):
    """OM1/OM2/OM3 semiempirical model with Loewdin orthogonalization.

    Parameters
    ----------
    mol : Molecule
    variant : str
        One of ``"om1"``, ``"om2"``, or ``"om3"``.
    params : OMxParameterSet or None
        Loads parameters for the given variant from Dral 2016 Tables 1-3
        if ``None``.
    max_iter : int (default 100)
    conv_tol : float (default 1e-7)
    warmup_iters : int (default 5)
        Number of Loewdin warm‑up iterations before the main SCF.
    density_mixing : float (default 0.3)
    solver : str (default ``"dense"``)
        Eigensolver for the SCF step.  Currently a reserved keyword;
        the OMx C++ backend always uses dense diagonalisation.
    """

    _VALID_VARIANTS = ("om1", "om2", "om3")

    def __init__(
        self,
        mol: Molecule,
        variant: str = "om2",
        params: Optional[_nddo.OMxParameterSet] = None,
        *,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
        warmup_iters: int = 5,
        density_mixing: float = 0.3,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        if variant not in self._VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self._VALID_VARIANTS}, got {variant!r}"
            )
        self._route_plan = SemiempiricalRoutePlan.from_request(
            variant,
            boundary="molecule",
            charge=int(mol.charge),
            multiplicity=int(mol.multiplicity),
        )
        if variant == "om1":
            import warnings

            from vibeqc.semiempirical import NDDOExperimentalWarning

            warnings.warn(
                "OM1's analytic core-valence ECP (Kolb & Thiel 1993) is not "
                "implemented; heavy-atom bonds can be too short and close "
                "contacts can variationally collapse. Prefer OM2/OM3 for "
                "molecular prescreening.",
                category=NDDOExperimentalWarning,
                stacklevel=2,
            )
        self._variant = variant
        if params is None:
            from vibeqc.semiempirical.methods.omx_params import load_omx_params

            params = load_omx_params(variant)
        if params.method_name() != variant:
            raise ValueError(
                f"OMxModel variant {variant!r} requires matching parameters, "
                f"not {params.method_name()!r}"
            )
        self._params = params
        self._max_iter = max_iter
        self._conv_tol = conv_tol
        self._warmup_iters = warmup_iters
        self._density_mixing = density_mixing
        self._n_iter = 0
        self._converged = False
        self._last_result = None

    @property
    def params(self) -> _nddo.OMxParameterSet:
        return self._params

    @property
    def variant(self) -> str:
        """The OMx variant in use (``"om1"``, ``"om2"``, or ``"om3"``)."""
        return self._variant

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
        result = self._run_at(self._mol)
        self._n_iter = int(result.n_iter)
        self._converged = bool(result.converged)
        self._last_result = result
        return float(result.energy)

    def _run_at(self, mol: Molecule):
        run_omx = (
            _nddo.run_uomx_v2
            if self._route_plan.spin == "unrestricted"
            else _nddo.run_omx_v2
        )
        result = run_omx(
            mol,
            self._params,
            self._max_iter,
            self._conv_tol,
        )
        return result

    def _energy_at(self, mol: Molecule) -> float:
        return float(self._run_at(mol).energy)

    def gradient(self) -> np.ndarray:
        """Finite-difference gradient bound to the retained energy snapshot."""
        if self._last_result is None:
            self.energy()
        compute = (
            _nddo.compute_uomx_v2_gradient_fd_from_result
            if self._route_plan.spin == "unrestricted"
            else _nddo.compute_omx_v2_gradient_fd_from_result
        )
        return np.asarray(
            compute(
                self._mol,
                self._params,
                self._last_result,
                max_iter=self._max_iter,
                conv_tol=self._conv_tol,
            ),
            dtype=float,
        )
