"""CCSD and CCSD(T) correlation-energy drivers (closed- and open-shell).

Density-fitted integrals by default; ``CCSDOptions(density_fit=False)``
selects the canonical conventional route with exact four-index MO
integrals (small molecules -- the AO ERI tensor is held in memory).

Usage (standalone)::

    from vibeqc import Molecule, BasisSet, run_rhf, RHFOptions
    from vibeqc import run_ccsd, CCSDOptions

    mol = Molecule(...)
    basis = BasisSet(mol, "cc-pVDZ")
    hf = run_rhf(mol, basis, RHFOptions())

    opts = CCSDOptions()
    opts.aux_basis = "cc-pvdz-ri"
    opts.compute_triples = True
    result = run_ccsd(mol, basis, hf, opts)

    print(f"CCSD(T)  = {result.e_total:.12f} Ha")
    print(f"CCSD corr = {result.e_ccsd_correlation:.12f} Ha")
    print(f"(T) corr  = {result.e_t:.12f} Ha")

Or through ``run_job`` with ``method="ccsd(t)"``::

    from vibeqc import run_job
    run_job(mol, basis="cc-pVDZ", method="ccsd(t)")
"""

from __future__ import annotations

import math
from typing import Optional

from ._vibeqc_core import CCSDOptions as _CCSDOptions
from ._vibeqc_core import CCSDResult as _CCSDResult
from ._vibeqc_core import run_ccsd as _run_ccsd
from ._vibeqc_core import run_ccsd_from_mos as _run_ccsd_from_mos
from ._vibeqc_core import run_uccsd as _run_uccsd
from ._vibeqc_core import run_uccsd_from_mos as _run_uccsd_from_mos
from .correlation_conventions import (
    effective_electron_count,
    published_frozen_core_orbital_count,
    resolve_frozen_core_count,
)
from .density_fitting import default_aux_basis_for
from .density_fitting import (
    guard_aux_coverage_for_options as _guard_aux_coverage,
)

# FNO option attributes carried on the Python CCSDOptions wrapper only (the
# C++ kernel never sees them: FNO truncation is orchestrated in Python, then
# the truncated semicanonical orbitals are handed to run_ccsd_from_mos).
_FNO_ATTRIBUTES = (
    "fno",
    "fno_occ_threshold",
    "fno_keep_fraction",
    "fno_delta_mp2",
)
_FNO_DEFAULTS = {
    "fno": False,
    "fno_occ_threshold": 1e-5,
    "fno_keep_fraction": None,
    "fno_delta_mp2": True,
}
_BRUECKNER_ATTRIBUTES = (
    "brueckner_max_iter",
    "brueckner_tol",
    "brueckner_damping",
)
_BRUECKNER_DEFAULTS = {
    "brueckner_max_iter": 20,
    "brueckner_tol": 1e-6,
    "brueckner_damping": 0.5,
}

_OPTION_ATTRIBUTES = (
    "density_fit",
    "aux_basis",
    "max_iter",
    "conv_tol_energy",
    "conv_tol_residual",
    "diis_subspace_size",
    "n_frozen_core",
    "compute_triples",
    "triples_memory_mode",
    "requested_memory_bytes",
    "triples_tile_size",
    "triples_max_threads",
    "triples_scratch_directory",
    "triples_variant",
    "cc_variant",
)

_TRIPLES_NONE = frozenset({"none", "off", "false", "0", "no", "ccsd"})
_TRIPLES_STANDARD_T = frozenset(
    {
        "(t)",
        "t",
        "true",
        "1",
        "yes",
        "standard",
        "standard(t)",
        "raghavachari",
        "raghavachari(t)",
        "ccsd(t)",
    }
)
# The bracket correction CCSD[T] and its original name CCSD+T(CCSD)
# (Urban, Noga, Cole, Bartlett 1985) are the SAME number: Raghavachari's
# (T) renamed the correction and added the fifth-order singles term.
_TRIPLES_BRACKET_T = frozenset(
    {
        "ccsd[t]",
        "[t]",
        "bracket",
        "ccsd+t(ccsd)",
        "+t(ccsd)",
        "t(ccsd)",
    }
)
_TRIPLES_LAMBDA_T = frozenset(
    {
        "a-ccsd(t)",
        "accsd(t)",
        "asymmetric",
        "lambda-ccsd(t)",
        "l-ccsd(t)",
        "ccsd(t)-lambda",
    }
)


def _normalise_triples_selector(selector: object) -> str:
    if isinstance(selector, bool):
        return "(t)" if selector else "none"
    if not isinstance(selector, str):
        raise ValueError(
            "triples= expects 'none', '(t)', '[t]', or one of the named "
            f"roadmap triples variants; got {selector!r}."
        )
    key = selector.strip().lower().replace(" ", "").replace("_", "-")
    if key in _TRIPLES_NONE:
        return "none"
    if key in _TRIPLES_STANDARD_T:
        return "(t)"
    if key in _TRIPLES_BRACKET_T:
        return "[t]"
    if key in _TRIPLES_LAMBDA_T:
        return "a-ccsd(t)"
    raise ValueError(
        "triples= expects 'none', '(t)', '[t]', or one of the named roadmap "
        f"triples variants; got {selector!r}."
    )


def resolve_triples_variant(selector: object) -> str:
    """Normalise a CCSD triples selector.

    ``"(t)"`` is the standard Raghavachari correction; ``"[t]"`` is the
    fourth-order bracket correction CCSD[T], identically Urban's
    CCSD+T(CCSD). ``"a-ccsd(t)"`` is the closed-shell asymmetric/Lambda
    triples correction.
    """
    return _normalise_triples_selector(selector)


def resolve_triples_selector(selector: object) -> bool:
    """Return whether a CCSD triples selector requests a triples correction
    (``"(t)"``, ``"[t]"``, or ``"a-ccsd(t)"``)."""
    return _normalise_triples_selector(selector) != "none"


class CCSDOptions(_CCSDOptions):
    """Python-friendly CCSD option struct with aux-basis autodetection."""

    def __init__(
        self,
        *,
        density_fit: bool = True,
        aux_basis: Optional[str] = None,
        max_iter: int = 100,
        conv_tol_energy: float = 1e-8,
        conv_tol_residual: float = 1e-7,
        diis_subspace_size: int = 6,
        n_frozen_core: Optional[int] = None,
        compute_triples: bool = True,
        triples: Optional[object] = None,
        triples_memory_mode: str = "auto",
        requested_memory_bytes: int = 0,
        triples_tile_size: int = 0,
        triples_max_threads: int = 0,
        triples_scratch_directory: Optional[str] = None,
        cc_variant: str = "ccsd",
        fno: bool = False,
        fno_occ_threshold: float = 1e-5,
        fno_keep_fraction: Optional[float] = None,
        fno_delta_mp2: bool = True,
        brueckner_max_iter: int = 20,
        brueckner_tol: float = 1e-6,
        brueckner_damping: float = 0.5,
    ):
        super().__init__()
        _triples_variant = "(t)"
        if triples is not None:
            _tv = resolve_triples_variant(triples)
            compute_triples = _tv != "none"
            if _tv != "none":
                _triples_variant = _tv
        self.density_fit = density_fit
        if aux_basis is not None:
            self.aux_basis = aux_basis
        self.max_iter = max_iter
        self.conv_tol_energy = conv_tol_energy
        self.conv_tol_residual = conv_tol_residual
        self.diis_subspace_size = diis_subspace_size
        self.n_frozen_core = 0 if n_frozen_core is None else int(n_frozen_core)
        self._n_frozen_core_explicit = n_frozen_core is not None
        self.compute_triples = compute_triples
        self.triples_memory_mode = triples_memory_mode
        self.requested_memory_bytes = int(requested_memory_bytes)
        self.triples_tile_size = int(triples_tile_size)
        self.triples_max_threads = int(triples_max_threads)
        if triples_scratch_directory is not None:
            self.triples_scratch_directory = str(triples_scratch_directory)
        # Which triples correction lands in e_t: "(t)" standard
        # Raghavachari, "[t]" bracket CCSD[T] (= CCSD+T(CCSD)),
        # or "a-ccsd(t)" asymmetric/Lambda triples.
        self.triples_variant = _triples_variant
        # Approximate coupled-cluster / coupled-pair / QCI variant of the
        # amplitude equations ("ccsd", "cc2", "ccd", "lccd",
        # "lccsd" = "cepa(0)", "cepa(1)", "cepa(2)", "cepa(3)",
        # "qcisd"). Closed-shell kernel only; the C++ side
        # rejects compute_triples except with CCSD/QCISD amplitudes.
        self.cc_variant = cc_variant
        # Frozen natural orbitals (Python-only; orchestrated by run_fno_ccsd).
        # fno=True truncates the virtual space to the dominant MP2 natural
        # orbitals before CCSD(T).  Selection is by NO occupation cutoff
        # (fno_occ_threshold) unless fno_keep_fraction is set, which keeps that
        # fraction of the virtual orbitals instead.  fno_delta_mp2 adds the
        # (E_MP2[full] - E_MP2[trunc]) correction (DePrince-Sherrill 2013).
        self.fno = fno
        self.fno_occ_threshold = fno_occ_threshold
        self.fno_keep_fraction = fno_keep_fraction
        self.fno_delta_mp2 = fno_delta_mp2
        self.brueckner_max_iter = int(brueckner_max_iter)
        self.brueckner_tol = float(brueckner_tol)
        self.brueckner_damping = float(brueckner_damping)

    # Frozen-core provenance: the .out / structured-log label
    # "(explicit)" vs "(chemical-core default)" (v0.15.21) is driven by
    # ``_n_frozen_core_explicit``.  The constructor kwarg path has always
    # set it; this write-through property makes the equally common
    # attribute-assignment path (``opts = CCSDOptions();
    # opts.n_frozen_core = 0``) mark explicitness too — pre-fix, an
    # explicit all-electron request printed as "(chemical-core default)".
    # run_job's internal chemical-core default assignments reset the flag
    # right after writing (see runner.py), keeping the default label
    # truthful.
    @property
    def n_frozen_core(self) -> int:
        return _CCSDOptions.n_frozen_core.__get__(self, type(self))

    @n_frozen_core.setter
    def n_frozen_core(self, value) -> None:
        # Write through to the native pybind member so the C++ kernels
        # see the value, then record user intent.
        _CCSDOptions.n_frozen_core.__set__(self, int(value))
        self._n_frozen_core_explicit = True

    @property
    def triples(self) -> str:
        """Human-readable triples selector."""
        if not bool(self.compute_triples):
            return "none"
        return str(self.triples_variant or "(t)")

    @triples.setter
    def triples(self, selector: object) -> None:
        tv = resolve_triples_variant(selector)
        self.compute_triples = tv != "none"
        if tv != "none":
            self.triples_variant = tv

    def resolve_aux_basis(self, orbital_basis_name: str) -> None:
        """If ``aux_basis`` is empty, auto-detect from the orbital basis name."""
        if not self.aux_basis:
            self.aux_basis = default_aux_basis_for(orbital_basis_name, kind="ri")


def _copy_ccsd_options(options: _CCSDOptions) -> CCSDOptions:
    """Copy native/Python CCSD options into the Python wrapper type."""
    if not isinstance(options, _CCSDOptions):
        raise TypeError(
            "CCSD options must be a vibeqc.CCSDOptions or native "
            f"_vibeqc_core.CCSDOptions instance, got {type(options).__name__}."
        )
    py_opts = CCSDOptions()
    requested_frozen = getattr(options, "n_frozen_core")
    for attr in _OPTION_ATTRIBUTES:
        if attr == "n_frozen_core":
            continue
        setattr(py_opts, attr, getattr(options, attr))
    if requested_frozen is not None:
        py_opts.n_frozen_core = requested_frozen
    # FNO attributes are Python-only; a bare C++ _CCSDOptions lacks them, so
    # fall back to the wrapper defaults.
    for attr in _FNO_ATTRIBUTES:
        setattr(py_opts, attr, getattr(options, attr, _FNO_DEFAULTS[attr]))
    for attr in _BRUECKNER_ATTRIBUTES:
        setattr(
            py_opts, attr,
            getattr(options, attr, _BRUECKNER_DEFAULTS[attr]),
        )
    # A bare native options object has no Python intent marker. Its None
    # sentinel means the published default; a numerical value (including
    # zero) is explicit. The Python wrapper carries its own marker and keeps
    # the usual implicit/default distinction.
    py_opts._n_frozen_core_explicit = getattr(
        options,
        "_n_frozen_core_explicit",
        requested_frozen is not None,
    )
    return py_opts


def _resolve_ccsd_frozen_core(
    options: CCSDOptions, molecule, reference=None
) -> CCSDOptions:
    """Apply the published default without overwriting an explicit zero.

    ``reference`` (the SCF result the CC kernel consumes) makes the published
    count ECP-aware: cores an ECP already removed are not frozen again.
    """

    if not getattr(options, "_n_frozen_core_explicit", False):
        options.n_frozen_core = resolve_frozen_core_count(
            molecule, None, reference=reference
        )
        # The assignment above passes through the intent-tracking property;
        # restore provenance because this was the library default, not a user
        # supplied integer.
        options._n_frozen_core_explicit = False
    return options


class _CCSDTraceStep:
    __slots__ = ("iter", "energy", "delta_e", "r1_norm", "r2_norm", "diis_subspace")

    def __init__(self, *, iter, energy, delta_e, r1_norm, r2_norm, diis_subspace):
        self.iter = int(iter)
        self.energy = float(energy)
        self.delta_e = float(delta_e)
        self.r1_norm = float(r1_norm)
        self.r2_norm = float(r2_norm)
        self.diis_subspace = int(diis_subspace)


class LambdaCCSDTResult:
    """CCSDResult-compatible view of a closed-shell A-CCSD(T) calculation."""

    __slots__ = (
        "e_hf",
        "e_ccsd_correlation",
        "e_ccsd",
        "e_t",
        "e_t4",
        "e_t5_st",
        "e_ccsd_t",
        "e_total",
        "n_iter",
        "converged",
        "t1_norm",
        "t2_norm",
        "t1_amplitudes",
        "t2_amplitudes",
        "lambda_t1_amplitudes",
        "lambda_t2_amplitudes",
        "lambda_residual_norm",
        "lambda_condition_number",
        "lambda_t1_norm",
        "lambda_t2_norm",
        "cc_trace",
    )

    def __init__(self, ccsd, lam, *, e_t):
        import numpy as np

        self.e_hf = float(ccsd.e_hf)
        self.e_ccsd_correlation = float(ccsd.e_corr)
        self.e_ccsd = float(ccsd.e_total)
        self.e_t = float(e_t)
        self.e_t4 = 0.0
        self.e_t5_st = 0.0
        self.e_ccsd_t = self.e_ccsd + self.e_t
        self.e_total = self.e_ccsd_t
        self.n_iter = int(ccsd.n_iter)
        self.converged = bool(ccsd.converged and lam.converged)
        self.t1_norm = float(np.linalg.norm(ccsd.t1))
        self.t2_norm = float(np.linalg.norm(ccsd.t2))
        self.t1_amplitudes = ccsd.t1
        self.t2_amplitudes = ccsd.t2
        self.lambda_t1_amplitudes = lam.l1
        self.lambda_t2_amplitudes = lam.l2
        self.lambda_residual_norm = float(lam.residual_norm)
        self.lambda_condition_number = float(lam.condition_number)
        self.lambda_t1_norm = float(np.linalg.norm(lam.l1))
        self.lambda_t2_norm = float(np.linalg.norm(lam.l2))
        self.cc_trace = [
            _CCSDTraceStep(
                iter=row.get("iter", idx + 1),
                energy=self.e_hf + float(row.get("e_corr", 0.0)),
                delta_e=row.get("delta_e", 0.0),
                r1_norm=row.get("r1_norm", 0.0),
                r2_norm=row.get("r2_norm", 0.0),
                diis_subspace=row.get("diis_subspace", 0),
            )
            for idx, row in enumerate(ccsd.trace)
        ]

    def __repr__(self):
        return (
            f"LambdaCCSDTResult(e_ccsd_t={self.e_ccsd_t:.10f}, "
            f"e_t={self.e_t:+.3e}, converged={self.converged})"
        )


def _exact_closed_shell_cc_blocks(basis, C_occ, C_vir):
    import numpy as np

    from ._vibeqc_core import compute_eri

    eri = np.asarray(compute_eri(basis), dtype=float)
    return {
        "ovov": np.einsum(
            "pqrs,pi,qa,rj,sb->iajb", eri, C_occ, C_vir, C_occ, C_vir,
            optimize=True,
        ),
        "ovvv": np.einsum(
            "pqrs,pi,qa,rb,sc->iabc", eri, C_occ, C_vir, C_vir, C_vir,
            optimize=True,
        ),
        "ooov": np.einsum(
            "pqrs,pi,qj,rk,sa->ijka", eri, C_occ, C_occ, C_occ, C_vir,
            optimize=True,
        ),
        "oooo": np.einsum(
            "pqrs,pi,qj,rk,sl->ijkl", eri, C_occ, C_occ, C_occ, C_occ,
            optimize=True,
        ),
        "vvvv": np.einsum(
            "pqrs,pa,qb,rc,sd->abcd", eri, C_vir, C_vir, C_vir, C_vir,
            optimize=True,
        ),
        "oovv": np.einsum(
            "pqrs,pi,qj,ra,sb->ijab", eri, C_occ, C_occ, C_vir, C_vir,
            optimize=True,
        ),
    }


def _run_accsd_t_from_mos(molecule, basis, C, F, e_hf, options, ecp_total_ncore=0):
    import numpy as np

    from . import BasisSet
    from .density_fitting import DensityFitting
    from .dlpno._ccsd_cs import (
        CSCCSDLambdaResult,
        _blocks,
        cs_lambda_triples_correction,
        run_cs_ccsd_blocks,
        run_cs_ccsd_lambda_iterative,
    )

    if str(getattr(options, "cc_variant", "ccsd") or "ccsd").lower() != "ccsd":
        raise ValueError("A-CCSD(T) is defined for CCSD amplitudes only")

    memory_mode = str(
        getattr(options, "triples_memory_mode", "auto") or "auto"
    ).strip().lower()
    memory_mode = {"in-core": "fast", "incore": "fast"}.get(
        memory_mode,
        memory_mode,
    )
    if memory_mode not in {"auto", "fast"}:
        raise NotImplementedError(
            "A-CCSD(T) still uses the dense Lambda-triples implementation; "
            "triples_memory_mode must be 'auto' or 'fast'. Use standard "
            "CCSD(T) for blocked/direct/disk triples."
        )
    if int(getattr(options, "requested_memory_bytes", 0) or 0) > 0:
        raise NotImplementedError(
            "A-CCSD(T) cannot enforce requested_memory_bytes in its dense "
            "Lambda-triples implementation. Use run_job(memory_budget_bytes=...) "
            "for phase-aware admission, or standard CCSD(T) for a native "
            "bounded triples kernel."
        )
    if (
        int(getattr(options, "triples_tile_size", 0) or 0) > 0
        or int(getattr(options, "triples_max_threads", 0) or 0) > 0
        or bool(getattr(options, "triples_scratch_directory", "") or "")
    ):
        raise NotImplementedError(
            "A-CCSD(T) does not implement triples tile, worker, or scratch "
            "controls; use standard CCSD(T) for requested-memory triples."
        )

    from types import SimpleNamespace

    n_occ_total = (
        effective_electron_count(
            molecule, SimpleNamespace(ecp_total_ncore=int(ecp_total_ncore))
        )
        // 2
    )
    frozen = int(options.n_frozen_core)
    if frozen < 0 or frozen >= n_occ_total:
        raise ValueError(
            f"run_ccsd: invalid n_frozen_core={frozen} for n_occ={n_occ_total}"
        )

    C = np.asarray(C, dtype=float)
    F = np.asarray(F, dtype=float)
    C_occ = np.asfortranarray(C[:, frozen:n_occ_total])
    C_vir = np.asfortranarray(C[:, n_occ_total:])
    no = C_occ.shape[1]
    if no < 1 or C_vir.shape[1] < 1:
        raise RuntimeError("run_ccsd: A-CCSD(T) requires occupied and virtual orbitals")

    C_corr = np.asfortranarray(np.hstack([C_occ, C_vir]))
    f_mo = C_corr.T @ F @ C_corr

    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="_run_accsd_t_from_mos"
        )
        aux = BasisSet(molecule, options.aux_basis, require_all_atoms=False)
        df = DensityFitting(
            basis, aux, aux_basis_name=options.aux_basis, molecule=molecule
        )
        B_ov = np.asarray(df.mo_transform(C_occ, C_vir), dtype=float)
        B_vv = np.asarray(df.mo_transform(C_vir, C_vir), dtype=float)
        B_oo = np.asarray(df.mo_transform(C_occ, C_occ), dtype=float)
        V = _blocks(B_ov, B_vv, B_oo)
    else:
        V = _exact_closed_shell_cc_blocks(basis, C_occ, C_vir)

    ccsd = run_cs_ccsd_blocks(
        f_mo,
        V,
        no,
        e_hf=float(e_hf),
        max_iter=int(options.max_iter),
        conv_tol=float(options.conv_tol_energy),
        conv_tol_residual=float(options.conv_tol_residual),
        diis_size=int(options.diis_subspace_size),
    )

    f_oo = f_mo[:no, :no]
    f_vv = f_mo[no:, no:]
    f_ov = f_mo[:no, no:]
    if not ccsd.converged:
        lam = CSCCSDLambdaResult(
            l1=np.zeros_like(ccsd.t1),
            l2=np.zeros_like(ccsd.t2),
            residual_norm=float("inf"),
            condition_number=float("nan"),
            n_amplitudes=int(ccsd.t1.size + ccsd.t2.size),
            n_iter=0,
            converged=False,
        )
        return LambdaCCSDTResult(ccsd, lam, e_t=0.0)
    lam = run_cs_ccsd_lambda_iterative(
        ccsd.t1,
        ccsd.t2,
        f_oo,
        f_vv,
        f_ov,
        V,
        max_iter=int(options.max_iter),
        residual_tol=float(options.conv_tol_residual),
        diis_size=int(options.diis_subspace_size),
    )
    if not lam.converged:
        return LambdaCCSDTResult(ccsd, lam, e_t=0.0)
    eps_o = np.diag(f_oo)
    eps_v = np.diag(f_vv)
    e_t = cs_lambda_triples_correction(
        ccsd.t1,
        ccsd.t2,
        lam.l1,
        lam.l2,
        V["ovvv"],
        V["ooov"],
        V["ovov"],
        eps_o,
        eps_v,
        f_ov,
    )
    return LambdaCCSDTResult(ccsd, lam, e_t=e_t)


def run_accsd_t(molecule, basis, rhf_result, options=None):
    """Run closed-shell asymmetric/Lambda CCSD(T)."""
    if options is None:
        options = CCSDOptions(triples="a-ccsd(t)")
    elif isinstance(options, _CCSDOptions) and not isinstance(options, CCSDOptions):
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, rhf_result)

    if not getattr(rhf_result, "converged", False):
        raise RuntimeError("run_accsd_t: RHF reference is not converged")
    if getattr(molecule, "multiplicity", 1) != 1:
        raise ValueError(
            "run_accsd_t: A-CCSD(T) requires a closed-shell (singlet) reference; "
            f"got multiplicity={getattr(molecule, 'multiplicity', 1)}"
        )

    options = _copy_ccsd_options(options)
    options.compute_triples = True
    options.triples_variant = "a-ccsd(t)"
    return _run_accsd_t_from_mos(
        molecule,
        basis,
        rhf_result.mo_coeffs,
        rhf_result.fock,
        float(rhf_result.energy),
        options,
        ecp_total_ncore=int(getattr(rhf_result, "ecp_total_ncore", 0) or 0),
    )


def run_ccsd(molecule, basis, rhf_result, options=None):
    """Run closed-shell CCSD (and optionally CCSD(T)).

    Density-fitted by default; ``density_fit=False`` selects the canonical
    conventional route with exact four-index MO integrals (small molecules
    only -- the AO ERI tensor is held in memory).

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    rhf_result : RHFResult
        Converged RHF reference.
    options : CCSDOptions, optional
        CCSD options.  If not provided, defaults are used with
        ``density_fit=True`` and ``compute_triples=True``.

    Returns
    -------
    CCSDResult
    """
    if options is None:
        options = CCSDOptions()
    elif isinstance(options, _CCSDOptions) and not isinstance(options, CCSDOptions):
        # Upcast bare C++ options to Python wrapper for aux resolution.
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, rhf_result)

    if not getattr(rhf_result, "converged", False):
        raise RuntimeError("run_ccsd: RHF reference is not converged")

    if getattr(molecule, "multiplicity", 1) != 1:
        raise ValueError(
            "run_ccsd: CCSD requires a closed-shell (singlet) reference; "
            f"got multiplicity={getattr(molecule, 'multiplicity', 1)}"
        )

    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="run_ccsd"
        )

    # Frozen-natural-orbital path: truncate the virtual space first.
    if getattr(options, "fno", False):
        return run_fno_ccsd(molecule, basis, rhf_result, options)

    if (
        bool(getattr(options, "compute_triples", False))
        and str(getattr(options, "triples_variant", "(t)") or "(t)") == "a-ccsd(t)"
    ):
        return run_accsd_t(molecule, basis, rhf_result, options)

    return _run_ccsd(molecule, basis, rhf_result, options)


class BruecknerIteration:
    """One outer orbital-optimization row for BCCD/BCCD(T)."""

    __slots__ = (
        "iter",
        "e_ccsd",
        "t1_norm",
        "next_t1_norm",
        "step_norm",
        "damping",
        "sign",
    )

    def __init__(
        self,
        *,
        iter,
        e_ccsd,
        t1_norm,
        next_t1_norm,
        step_norm,
        damping,
        sign,
    ):
        self.iter = int(iter)
        self.e_ccsd = float(e_ccsd)
        self.t1_norm = float(t1_norm)
        self.next_t1_norm = float(next_t1_norm)
        self.step_norm = float(step_norm)
        self.damping = float(damping)
        self.sign = float(sign)


class BruecknerCCDResult:
    """CCSDResult-compatible view of a BCCD/BCCD(T) calculation."""

    __slots__ = (
        "e_hf",
        "e_ccsd_correlation",
        "e_ccsd",
        "e_t",
        "e_t4",
        "e_t5_st",
        "e_ccsd_t",
        "e_total",
        "n_iter",
        "converged",
        "t1_norm",
        "t2_norm",
        "t1_amplitudes",
        "cc_trace",
        "triples_memory_mode_used",
        "triples_tile_size_used",
        "triples_threads_used",
        "triples_workspace_bytes",
        "triples_disk_bytes",
        "brueckner_iterations",
        "brueckner_converged",
        "brueckner_t1_norm",
        "brueckner_trace",
        "brueckner_mo_coeffs",
        "brueckner_reference_energy",
    )

    def __init__(
        self,
        base,
        *,
        brueckner_iterations,
        brueckner_t1_norm,
        brueckner_trace,
        brueckner_mo_coeffs,
        brueckner_reference_energy,
    ):
        self.e_hf = base.e_hf
        self.e_ccsd_correlation = base.e_ccsd_correlation
        self.e_ccsd = base.e_ccsd
        self.e_t = base.e_t
        self.e_t4 = getattr(base, "e_t4", 0.0)
        self.e_t5_st = getattr(base, "e_t5_st", 0.0)
        self.e_ccsd_t = base.e_ccsd_t
        self.e_total = base.e_total
        self.n_iter = base.n_iter
        self.converged = base.converged
        # For BCCD, report the residual CCSD singles norm on the Brueckner
        # orbitals.  The final CCD solve has T1 pinned to zero by definition.
        self.t1_norm = float(brueckner_t1_norm)
        self.t2_norm = base.t2_norm
        self.t1_amplitudes = getattr(base, "t1_amplitudes", None)
        self.cc_trace = base.cc_trace
        self.triples_memory_mode_used = getattr(
            base, "triples_memory_mode_used", ""
        )
        self.triples_tile_size_used = int(
            getattr(base, "triples_tile_size_used", 0) or 0
        )
        self.triples_threads_used = int(
            getattr(base, "triples_threads_used", 0) or 0
        )
        self.triples_workspace_bytes = int(
            getattr(base, "triples_workspace_bytes", 0) or 0
        )
        self.triples_disk_bytes = int(
            getattr(base, "triples_disk_bytes", 0) or 0
        )
        self.brueckner_iterations = int(brueckner_iterations)
        self.brueckner_converged = True
        self.brueckner_t1_norm = float(brueckner_t1_norm)
        self.brueckner_trace = list(brueckner_trace)
        self.brueckner_mo_coeffs = brueckner_mo_coeffs
        self.brueckner_reference_energy = float(brueckner_reference_energy)

    def __repr__(self):
        return (
            f"BruecknerCCDResult(e_total={self.e_total:.10f}, "
            f"brueckner_t1={self.brueckner_t1_norm:.3e}, "
            f"converged={self.converged})"
        )


def _build_closed_shell_reference(molecule, basis, C, H, eri, n_occ):
    import numpy as np

    from ._vibeqc_core import build_fock_g

    C_occ = np.asarray(C[:, :n_occ], dtype=float)
    density = 2.0 * (C_occ @ C_occ.T)
    F = np.asarray(H + build_fock_g(eri, density), dtype=float)
    e_elec = 0.5 * float(np.einsum("ij,ij->", density, H + F, optimize=True))
    return F, e_elec + float(molecule.nuclear_repulsion())


def _semicanonicalize_brueckner_blocks(C, F, n_frozen, n_occ):
    import numpy as np

    C = np.asarray(C, dtype=float)
    F = np.asarray(F, dtype=float)
    n_mo = C.shape[1]
    U = np.eye(n_mo)
    f_mo = C.T @ F @ C
    if n_occ - n_frozen > 1:
        _, Uoo = np.linalg.eigh(f_mo[n_frozen:n_occ, n_frozen:n_occ])
        U[n_frozen:n_occ, n_frozen:n_occ] = Uoo
    if n_mo - n_occ > 1:
        _, Uvv = np.linalg.eigh(f_mo[n_occ:, n_occ:])
        U[n_occ:, n_occ:] = Uvv
    return C @ U


def _brueckner_rotation_matrix(t1, n_mo, n_frozen, n_occ, damping, sign):
    import numpy as np
    from scipy.linalg import expm

    no = n_occ - n_frozen
    nv = n_mo - n_occ
    if t1.shape != (no, nv):
        raise RuntimeError(
            "run_bccd: unexpected T1 amplitude shape "
            f"{t1.shape}, expected {(no, nv)}"
        )
    K = np.zeros((n_mo, n_mo), dtype=float)
    step = float(sign) * float(damping) * np.asarray(t1, dtype=float)
    # Cap a single orbital-rotation step to keep expm in the local Thouless
    # regime on hard starts; the line search still chooses the accepted size.
    step_norm = float(np.linalg.norm(step))
    if step_norm > 0.25:
        step *= 0.25 / step_norm
        step_norm = 0.25
    occ = slice(n_frozen, n_occ)
    vir = slice(n_occ, n_mo)
    K[vir, occ] = step.T
    K[occ, vir] = -step
    return expm(K), step_norm


def run_bccd(molecule, basis, rhf_result, options=None):
    """Run closed-shell Brueckner CCD/BCCD(T).

    The public route first optimizes Brueckner orbitals by iterating CCSD
    singles amplitudes to zero, then runs the doubles-only BCCD equations
    (and optionally the standard perturbative triples correction) on those
    orbitals.
    """
    import numpy as np

    from ._vibeqc_core import compute_eri, compute_kinetic, compute_nuclear

    if options is None:
        options = CCSDOptions(compute_triples=False)
    elif isinstance(options, _CCSDOptions) and not isinstance(options, CCSDOptions):
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, rhf_result)

    if not getattr(rhf_result, "converged", False):
        raise RuntimeError("run_bccd: RHF reference is not converged")
    if getattr(molecule, "multiplicity", 1) != 1:
        raise ValueError(
            "run_bccd: BCCD requires a closed-shell (singlet) reference; "
            f"got multiplicity={getattr(molecule, 'multiplicity', 1)}"
        )
    if str(getattr(options, "triples_variant", "(t)") or "(t)") == "[t]":
        raise NotImplementedError(
            "run_bccd: bracket triples '[t]' are defined for CCSD; use "
            "triples='(t)' or method='bccd(t)' for BCCD(T)."
        )
    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="run_bccd"
        )

    n_occ = effective_electron_count(molecule, rhf_result) // 2
    n_frozen = int(options.n_frozen_core)
    if n_frozen < 0 or n_frozen >= n_occ:
        raise ValueError(
            f"run_bccd: invalid n_frozen_core={n_frozen} for n_occ={n_occ}"
        )
    max_outer = int(getattr(options, "brueckner_max_iter", 20))
    if max_outer < 1:
        raise ValueError("run_bccd: brueckner_max_iter must be >= 1")
    tol = float(getattr(options, "brueckner_tol", 1e-6))
    damping0 = float(getattr(options, "brueckner_damping", 0.5))
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("run_bccd: brueckner_tol must be finite and positive")
    if not math.isfinite(damping0) or damping0 <= 0.0:
        raise ValueError(
            "run_bccd: brueckner_damping must be finite and positive"
        )

    H = np.asarray(compute_kinetic(basis) + compute_nuclear(basis, molecule))
    eri = np.asarray(compute_eri(basis))
    C = np.asarray(rhf_result.mo_coeffs, dtype=float)
    n_mo = C.shape[1]

    probe_opts = _copy_ccsd_options(options)
    probe_opts.compute_triples = False
    probe_opts.triples_variant = "(t)"
    probe_opts.cc_variant = "ccsd"
    probe_opts.fno = False

    final_opts = _copy_ccsd_options(options)
    final_opts.cc_variant = "bccd"
    final_opts.fno = False

    def probe(C_in):
        F, e_ref = _build_closed_shell_reference(
            molecule, basis, C_in, H, eri, n_occ
        )
        C_semi = _semicanonicalize_brueckner_blocks(C_in, F, n_frozen, n_occ)
        cc = _run_ccsd_from_mos(
            molecule,
            basis,
            np.asfortranarray(C_semi),
            np.asfortranarray(F),
            float(e_ref),
            int(getattr(rhf_result, "ecp_total_ncore", 0) or 0),
            probe_opts,
        )
        if not bool(cc.converged):
            raise RuntimeError(
                "run_bccd: inner CCSD singles probe did not converge "
                f"after {cc.n_iter} iterations"
            )
        t1 = np.asarray(cc.t1_amplitudes, dtype=float)
        return C_semi, F, float(e_ref), cc, float(np.linalg.norm(t1)), t1

    trace = []
    last_norm = None
    last_ref_energy = float(rhf_result.energy)
    for outer in range(1, max_outer + 1):
        C, F, e_ref, cc_probe, t1_norm, t1 = probe(C)
        last_norm = t1_norm
        last_ref_energy = e_ref
        if t1_norm <= tol:
            break

        best = None
        for sign in (1.0, -1.0):
            for damping in (damping0, 0.5 * damping0, 0.25 * damping0):
                U, step_norm = _brueckner_rotation_matrix(
                    t1, n_mo, n_frozen, n_occ, damping, sign
                )
                C_trial = C @ U
                try:
                    C_next, F_next, e_next, cc_next, norm_next, _ = probe(C_trial)
                except RuntimeError:
                    continue
                candidate = (
                    norm_next,
                    C_next,
                    F_next,
                    e_next,
                    cc_next,
                    step_norm,
                    damping,
                    sign,
                )
                if best is None or norm_next < best[0]:
                    best = candidate
                if norm_next < t1_norm:
                    break
            if best is not None and best[0] < t1_norm:
                break

        if best is None:
            raise RuntimeError("run_bccd: no stable Brueckner rotation step found")

        trace.append(
            BruecknerIteration(
                iter=outer,
                e_ccsd=cc_probe.e_ccsd,
                t1_norm=t1_norm,
                next_t1_norm=best[0],
                step_norm=best[5],
                damping=best[6],
                sign=best[7],
            )
        )
        C, F, last_ref_energy = best[1], best[2], best[3]
        last_norm = best[0]
        if last_norm <= tol:
            break
    else:
        raise RuntimeError(
            "run_bccd: Brueckner orbital optimization did not converge "
            f"after {max_outer} iterations; final |T1|={last_norm:.6e}"
        )

    base = _run_ccsd_from_mos(
        molecule,
        basis,
        np.asfortranarray(C),
        np.asfortranarray(F),
        float(last_ref_energy),
        int(getattr(rhf_result, "ecp_total_ncore", 0) or 0),
        final_opts,
    )
    return BruecknerCCDResult(
        base,
        brueckner_iterations=len(trace),
        brueckner_t1_norm=float(last_norm),
        brueckner_trace=trace,
        brueckner_mo_coeffs=np.asarray(C),
        brueckner_reference_energy=float(last_ref_energy),
    )


def run_uccsd(molecule, basis, uhf_result, options=None):
    """Run open-shell UCCSD (and optionally UCCSD(T)) on a UHF reference.

    The spin-orbital unrestricted sibling of :func:`run_ccsd`. Accepts a
    converged UHF reference (any multiplicity); the same ``CCSDOptions`` are
    used as for the closed-shell kernel, including ``density_fit=False``
    for the canonical conventional route (exact four-index integrals).

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    uhf_result : UHFResult
        Converged UHF reference.
    options : CCSDOptions, optional
        CCSD options.  If not provided, defaults are used with
        ``density_fit=True`` and ``compute_triples=True``.

    Returns
    -------
    CCSDResult
    """
    if options is None:
        options = CCSDOptions()
    elif isinstance(options, _CCSDOptions) and not isinstance(options, CCSDOptions):
        # Upcast bare C++ options to Python wrapper for aux resolution.
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, uhf_result)

    if not getattr(uhf_result, "converged", False):
        raise RuntimeError("run_uccsd: UHF reference is not converged")

    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="run_uccsd"
        )

    return _run_uccsd(molecule, basis, uhf_result, options)


def run_uccsd_from_mos(
    molecule,
    basis,
    mo_coeffs_alpha,
    mo_coeffs_beta,
    fock_alpha,
    fock_beta,
    e_hf,
    options=None,
    *,
    ecp_total_ncore,
):
    """Run open-shell UCCSD(T) from explicit MO-coefficient and Fock arrays.

    Low-level entry point for references that do not produce a UHFResult
    (ROHF, semicanonical orbitals from other sources).  The spin-orbital
    kernel is reference-agnostic -- it only needs per-spin AO->MO
    coefficients and the corresponding Fock matrices.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    mo_coeffs_alpha, mo_coeffs_beta : ndarray (n_ao, n_orb)
        AO-to-MO coefficient matrices.  For an ROHF reference these are
        identical (a single set of spatial orbitals).
    fock_alpha, fock_beta : ndarray (n_ao, n_ao)
        Per-spin AO Fock matrices at convergence.
    e_hf : float
        SCF reference energy (Hartree).  Attached as ``result.e_hf``.
    options : CCSDOptions, optional
    ecp_total_ncore : int, keyword-only
        Authoritative number of physical core electrons removed by the
        reference Hamiltonian. Pass zero for an all-electron reference. A
        positive value is rejected because this kernel still partitions the
        arrays from ``Molecule.n_electrons()``. The argument is mandatory
        because explicit arrays carry no ECP provenance of their own.
    """
    import numpy as np

    if options is None:
        options = CCSDOptions()
    else:
        options = _copy_ccsd_options(options)
    if int(ecp_total_ncore) > 0 and not getattr(
        options, "_n_frozen_core_explicit", False
    ):
        raise ValueError(
            "run_uccsd_from_mos: an ECP reference (ecp_total_ncore="
            f"{int(ecp_total_ncore)}) needs an explicit n_frozen_core; the "
            "published element count cannot subtract the removed cores "
            "from bare arrays. Resolve it with "
            "vibeqc.correlation_conventions.resolve_frozen_core_count("
            "molecule, None, reference=<uhf result>) and pass the count."
        )
    options = _resolve_ccsd_frozen_core(options, molecule)

    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="run_uccsd_from_mos"
        )

    return _run_uccsd_from_mos(
        molecule,
        basis,
        np.asfortranarray(mo_coeffs_alpha),
        np.asfortranarray(mo_coeffs_beta),
        np.asfortranarray(fock_alpha),
        np.asfortranarray(fock_beta),
        float(e_hf),
        int(ecp_total_ncore),
        options,
    )


def run_rohf_ccsd(molecule, basis, rohf_result, options=None):
    """Run ROHF-reference UCCSD (and optionally UCCSD(T)).

    The spin-orbital CCSD kernel is reference-agnostic -- it accepts any
    set of per-spin MO coefficients and Fock matrices.  ROHF supplies a
    single set of spatial orbitals (``mo_coeffs``) with unrestricted-
    view aliases (``mo_coeffs_alpha`` / ``mo_coeffs_beta``) and
    per-spin AO Fock matrices (``fock_alpha`` / ``fock_beta``).  The
    occupied/virtual partition differs per spin via multiplicity, so
    the singly-occupied orbitals enter the alpha virtual space for the
    beta amplitude equations and vice versa -- exactly the
    reference-agnostic behaviour the kernel was designed for.

    The (T) correction uses the ROHF orbital energies (identical for
    both spins -- semicanonical), which satisfy the canonical-orbital
    assumption (f_ov = 0).

    Parameters
    ----------
    molecule : Molecule
        Must be open-shell (multiplicity > 1); closed-shell molecules
        should use :func:`run_ccsd` directly.
    basis : BasisSet
    rohf_result : ROHFResult
        Converged ROHF reference from :func:`vibeqc.run_rohf`.
    options : CCSDOptions, optional

    Returns
    -------
    CCSDResult
    """
    if not getattr(rohf_result, "converged", False):
        raise RuntimeError("run_rohf_ccsd: ROHF reference is not converged")

    if options is None:
        options = CCSDOptions()
    else:
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, rohf_result)

    if options.density_fit:
        options.resolve_aux_basis(basis.name)
        # #480: the native DF driver builds its own auxiliary
        # BasisSet from options.aux_basis, so the DensityFitting
        # guard never sees it. Pre-flight the coverage check.
        _guard_aux_coverage(
            options, molecule, basis, route="run_rohf_ccsd"
        )

    return run_uccsd_from_mos(
        molecule,
        basis,
        rohf_result.mo_coeffs_alpha,
        rohf_result.mo_coeffs_beta,
        rohf_result.fock_alpha,
        rohf_result.fock_beta,
        float(rohf_result.energy),
        options,
        ecp_total_ncore=int(
            getattr(rohf_result, "ecp_total_ncore", 0) or 0
        ),
    )


def run_rohf_mp2(
    molecule,
    basis,
    rohf_result,
    *,
    n_frozen_core=None,
    aux_basis=None,
    os_scale=1.0,
    ss_scale=1.0,
):
    """Semicanonical ROHF-MP2 (RI) on a converged ROHF reference.

    Builds the per-spin MO Fock matrices and density-fitted MO B-tensors and
    hands them to the reference-agnostic spin-orbital MP2 kernel
    :func:`vibeqc.dlpno._ccsd_ref.run_ref_ump2`. The ROHF orbitals are
    semicanonicalised inside the kernel (occ-occ and vir-vir Fock blocks
    diagonalised per spin), so the off-diagonal occ-vir Fock (non-zero for
    ROHF) contributes the MP2 singles term -- see :func:`run_ref_ump2` for the
    energy expression and citation (Knowles, Andrews, Amos, Handy & Pople,
    *Chem. Phys. Lett.* **186**, 130 (1991)).

    O(n⁵-n⁶) dense over the full MO space -- the same "reference, not
    production" tier as :func:`run_rohf_ccsd`; a C++ ROHF-MP2 is a later
    milestone (handovers/HANDOVER_ROHF.md M6b-MP2).

    Parameters
    ----------
    molecule, basis : Molecule, BasisSet
        Open-shell molecule (multiplicity > 1) and its orbital basis.
    rohf_result : ROHFResult
        Converged ROHF reference (``mo_coeffs`` common spatial orbitals,
        ``fock_alpha`` / ``fock_beta`` AO Fock matrices, ``energy``).
    n_frozen_core : int, optional
        Frozen core orbitals (default: one chemical-core count per atom).
    aux_basis : str, optional
        RI auxiliary basis (default: ``default_aux_basis_for(basis.name,
        kind="ri")``).
    os_scale, ss_scale : float
        Opposite-/same-spin doubles scaling (1.0 = plain MP2; the levers
        SCS-MP2 / SOS-MP2 and double hybrids tune). The singles term is not
        spin-scaled.

    Returns
    -------
    RefMP2Result
        With ``e_corr`` (scaled), ``e_singles``, ``e_doubles``, ``e_os``,
        ``e_ss``, ``e_hf``, ``e_total``.
    """
    import numpy as np

    from . import BasisSet
    from .density_fitting import DensityFitting
    from .dlpno._ccsd_ref import RefMP2Result, run_ref_ump2

    if not getattr(rohf_result, "converged", False):
        raise RuntimeError("run_rohf_mp2: ROHF reference is not converged")
    if molecule.multiplicity <= 1:
        raise ValueError(
            "run_rohf_mp2: closed-shell molecule; use RHF-MP2 (run_job "
            "method='mp2') instead."
        )

    ne = effective_electron_count(molecule, rohf_result)
    two_s = molecule.multiplicity - 1
    na = (ne + two_s) // 2
    nb = (ne - two_s) // 2

    nf = resolve_frozen_core_count(molecule, n_frozen_core, reference=rohf_result)
    if nf < 0 or nf > nb:
        raise ValueError(
            f"run_rohf_mp2: n_frozen_core={nf} out of range for n_beta={nb}"
        )

    aux_name = aux_basis or default_aux_basis_for(basis.name, kind="ri")
    aux_obj = BasisSet(molecule, aux_name, require_all_atoms=False)
    df = DensityFitting(
        basis, aux_obj, aux_basis_name=aux_name, molecule=molecule
    )

    C = np.asarray(rohf_result.mo_coeffs)[:, nf:]   # drop frozen core
    Fa = np.asarray(rohf_result.fock_alpha)
    Fb = np.asarray(rohf_result.fock_beta)
    fa_mo = C.T @ Fa @ C
    fb_mo = C.T @ Fb @ C
    B_mo = np.ascontiguousarray(np.asarray(df.mo_transform(C, C)))

    res = run_ref_ump2(
        fa_mo, fb_mo, B_mo, B_mo, na - nf, nb - nf, e_hf=float(rohf_result.energy)
    )

    # Apply spin-component scaling to the doubles (singles unscaled).
    e_corr = res.e_singles + os_scale * res.e_os + ss_scale * res.e_ss
    return RefMP2Result(
        e_corr=e_corr,
        e_singles=res.e_singles,
        e_doubles=res.e_doubles,
        e_os=res.e_os,
        e_ss=res.e_ss,
        e_hf=res.e_hf,
        e_total=res.e_hf + e_corr,
    )


class FNOCCSDResult:
    """CCSDResult-compatible view of an FNO-CCSD(T) run.

    Exposes the same energy and iteration fields as the C++ ``CCSDResult`` so
    it drops into ``run_job`` and the standard result handling unchanged, with
    the FNO delta-MP2 correction folded into the correlation energy and a few
    extra diagnostics (``n_virtual_kept`` / ``n_virtual_total`` /
    ``delta_mp2`` / ``n_frozen``).
    """

    __slots__ = (
        "e_hf", "e_ccsd_correlation", "e_ccsd", "e_t", "e_t4", "e_t5_st",
        "e_ccsd_t", "e_total", "n_iter", "converged", "t1_norm", "t2_norm",
        "cc_trace", "lambda_residual_norm", "lambda_t1_norm", "lambda_t2_norm",
        "delta_mp2", "n_virtual_kept", "n_virtual_total", "fno_occ_threshold",
        "n_frozen", "triples_memory_mode_used", "triples_tile_size_used",
        "triples_threads_used", "triples_workspace_bytes", "triples_disk_bytes",
    )

    def __init__(self, base, *, delta_mp2, n_vir_kept, n_vir_total,
                 occ_threshold, n_frozen):
        self.delta_mp2 = float(delta_mp2)
        self.n_virtual_kept = int(n_vir_kept)
        self.n_virtual_total = int(n_vir_total)
        self.fno_occ_threshold = float(occ_threshold)
        self.n_frozen = int(n_frozen)
        # delta-MP2 corrects the doubles (MP2-recoverable) part; it is folded
        # into the CCSD correlation, leaving the truncated-space (T) as is.
        self.e_hf = base.e_hf
        self.e_ccsd_correlation = base.e_ccsd_correlation + self.delta_mp2
        self.e_ccsd = self.e_hf + self.e_ccsd_correlation
        self.e_t = base.e_t
        self.e_t4 = getattr(base, "e_t4", 0.0)
        self.e_t5_st = getattr(base, "e_t5_st", 0.0)
        self.e_ccsd_t = self.e_ccsd + self.e_t
        self.e_total = self.e_ccsd_t
        self.n_iter = base.n_iter
        self.converged = base.converged
        self.t1_norm = base.t1_norm
        self.t2_norm = base.t2_norm
        self.cc_trace = base.cc_trace
        self.lambda_residual_norm = getattr(base, "lambda_residual_norm", float("nan"))
        self.lambda_t1_norm = getattr(base, "lambda_t1_norm", float("nan"))
        self.lambda_t2_norm = getattr(base, "lambda_t2_norm", float("nan"))
        self.triples_memory_mode_used = getattr(
            base, "triples_memory_mode_used", ""
        )
        self.triples_tile_size_used = int(
            getattr(base, "triples_tile_size_used", 0) or 0
        )
        self.triples_threads_used = int(
            getattr(base, "triples_threads_used", 0) or 0
        )
        self.triples_workspace_bytes = int(
            getattr(base, "triples_workspace_bytes", 0) or 0
        )
        self.triples_disk_bytes = int(
            getattr(base, "triples_disk_bytes", 0) or 0
        )

    def __repr__(self):
        return (
            f"FNOCCSDResult(e_ccsd_t={self.e_ccsd_t:.10f}, "
            f"kept {self.n_virtual_kept}/{self.n_virtual_total} virt, "
            f"delta_mp2={self.delta_mp2:+.3e}, converged={self.converged})"
        )


def run_fno_ccsd(molecule, basis, rhf_result, options=None):
    """Run closed-shell frozen-natural-orbital DF-CCSD(T).

    Truncates the virtual space to the dominant MP2 natural orbitals,
    semicanonicalizes the retained block, runs CCSD(T) in the smaller space,
    and (by default) adds the delta-MP2 correction
    ``E_MP2[full] - E_MP2[trunc]``.  The procedure follows DePrince and
    Sherrill, J. Chem. Theory Comput. 9, 2687 (2013), doi:10.1021/ct400250u.

    Selection of the retained virtuals:

    * ``options.fno_keep_fraction`` (if set): keep that fraction of the
      virtual orbitals, ordered by natural-orbital occupation.
    * otherwise ``options.fno_occ_threshold`` (default 1e-5): keep every
      natural orbital whose occupation is at or above the threshold.

    The decisive correctness check is invariance: with no truncation
    (threshold 0 / keep_fraction 1.0) this reproduces canonical CCSD(T) to
    machine precision, because CCSD is invariant to virtual rotation and
    semicanonicalization restores the canonical-orbital (T).

    Returns an :class:`FNOCCSDResult` (CCSDResult-compatible).
    """
    import numpy as np

    from . import BasisSet
    from .density_fitting import DensityFitting

    if options is None:
        options = CCSDOptions(fno=True)
    elif isinstance(options, _CCSDOptions) and not isinstance(options, CCSDOptions):
        options = _copy_ccsd_options(options)
    options = _resolve_ccsd_frozen_core(options, molecule, rhf_result)

    if not getattr(rhf_result, "converged", False):
        raise RuntimeError("run_fno_ccsd: RHF reference is not converged")
    if getattr(molecule, "multiplicity", 1) != 1:
        raise ValueError(
            "run_fno_ccsd: FNO-CCSD requires a closed-shell (singlet) reference"
        )
    if not options.density_fit:
        raise ValueError("run_fno_ccsd: FNO requires density_fit=True")
    options.resolve_aux_basis(basis.name)
    # #480: the native DF driver builds its own auxiliary
    # BasisSet from options.aux_basis, so the DensityFitting
    # guard never sees it. Pre-flight the coverage check.
    _guard_aux_coverage(
        options, molecule, basis, route="run_fno_ccsd"
    )

    C = np.asarray(rhf_result.mo_coeffs)
    F = np.asarray(rhf_result.fock)
    n_occ = effective_electron_count(molecule, rhf_result) // 2
    frozen = int(options.n_frozen_core)
    if frozen < 0 or frozen >= n_occ:
        raise ValueError(
            f"run_fno_ccsd: invalid n_frozen_core={frozen} for n_occ={n_occ}"
        )
    n_mo = C.shape[1]
    nv = n_mo - n_occ
    if nv < 1:
        raise RuntimeError("run_fno_ccsd: no virtual orbitals")

    C_occ_act = C[:, frozen:n_occ]
    C_vir = C[:, n_occ:]
    # Orbital energies as diag(C^T F C); the correlated window excludes the
    # frozen core (frozen orbitals enter only through the AO Fock downstream).
    fmo_diag = np.einsum("pi,pq,qi->i", C, F, C, optimize=True)
    eps_o = fmo_diag[frozen:n_occ]
    eps_v = fmo_diag[n_occ:]

    aux = BasisSet(molecule, options.aux_basis, require_all_atoms=False)
    df = DensityFitting(
        basis, aux, aux_basis_name=options.aux_basis, molecule=molecule
    )
    B_ov = df.mo_transform(C_occ_act, C_vir)            # (n_aux, no, nv)

    # MP2 amplitudes t_ij^ab and the (ia|jb) block, both indexed (i,j,a,b).
    g = np.einsum("Pia,Pjb->ijab", B_ov, B_ov, optimize=True)
    d = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
         - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    t = g / d
    tt = 2.0 * t - t.transpose(0, 1, 3, 2)
    e_mp2_full = float(np.einsum("ijab,ijab->", g, tt))

    # MP2 virtual one-particle density  D_ab = sum_ijc t_ij^ac (2 t_ij^bc -
    # t_ij^cb); diagonalize for the natural-orbital occupations + vectors.
    D = np.einsum("ijac,ijbc->ab", t, tt, optimize=True)
    D = 0.5 * (D + D.T)
    occ_no, U = np.linalg.eigh(D)
    order = np.argsort(occ_no)[::-1]                    # descending occupation
    occ_no = occ_no[order]
    U = U[:, order]

    if options.fno_keep_fraction is not None:
        kf = float(options.fno_keep_fraction)
        if not 0.0 < kf <= 1.0:
            raise ValueError(
                "run_fno_ccsd: fno_keep_fraction must be in (0, 1]"
            )
        k = max(1, int(round(kf * nv)))
    else:
        thr = float(options.fno_occ_threshold)
        k = int(np.count_nonzero(occ_no >= thr))
        k = max(1, min(k, nv))

    Uk = U[:, :k]
    # Semicanonicalize the retained virtual block (diagonalize f_vv): MANDATORY
    # so the canonical-orbital (T) denominators stay valid (f_ov stays zero
    # because the occupied block is untouched and the RHF reference is
    # canonical).
    f_vv = C_vir.T @ F @ C_vir
    eps_semi, W = np.linalg.eigh(Uk.T @ f_vv @ Uk)
    C_vir_fno = C_vir @ (Uk @ W)

    # delta-MP2: recover the MP2 correlation lost to truncation.
    delta_mp2 = 0.0
    if options.fno_delta_mp2:
        B_ov_t = df.mo_transform(C_occ_act, C_vir_fno)
        g_t = np.einsum("Pia,Pjb->ijab", B_ov_t, B_ov_t, optimize=True)
        d_t = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
               - eps_semi[None, None, :, None] - eps_semi[None, None, None, :])
        t_t = g_t / d_t
        e_mp2_trunc = float(
            np.einsum("ijab,ijab->", g_t,
                      2.0 * t_t - t_t.transpose(0, 1, 3, 2))
        )
        delta_mp2 = e_mp2_full - e_mp2_trunc

    # The MP2 natural-orbital setup is a dense Python phase, not retained
    # native CC workspace. Release both the full-space and optional truncated
    # delta-MP2 arrays (and the eager DF substrate) before entering the final
    # CC kernel so its requested-memory cap applies to the arrays that are
    # actually live there.
    del B_ov, g, d, t, tt, D
    if options.fno_delta_mp2:
        del B_ov_t, g_t, d_t, t_t
    del df, aux
    del C_occ_act, C_vir, fmo_diag, eps_o, eps_v
    del occ_no, U, Uk, f_vv, W, eps_semi

    # CCSD(T) in the truncated semicanonical space via the explicit-MO entry.
    C_fno = np.hstack([C[:, :n_occ], C_vir_fno])        # all occ + kept virt
    core_opts = _copy_ccsd_options(options)
    core_opts.fno = False                               # avoid re-entry
    if (
        bool(getattr(core_opts, "compute_triples", False))
        and str(getattr(core_opts, "triples_variant", "(t)") or "(t)")
        == "a-ccsd(t)"
    ):
        base = _run_accsd_t_from_mos(
            molecule, basis,
            np.asfortranarray(C_fno), np.asfortranarray(F),
            float(rhf_result.energy), core_opts,
            ecp_total_ncore=int(getattr(rhf_result, "ecp_total_ncore", 0) or 0),
        )
    else:
        base = _run_ccsd_from_mos(
            molecule, basis,
            np.asfortranarray(C_fno), np.asfortranarray(F),
            float(rhf_result.energy),
            int(getattr(rhf_result, "ecp_total_ncore", 0) or 0),
            core_opts,
        )
    return FNOCCSDResult(
        base, delta_mp2=delta_mp2, n_vir_kept=k, n_vir_total=nv,
        occ_threshold=float(options.fno_occ_threshold), n_frozen=frozen,
    )


def chemical_core_orbital_count(molecule, reference=None) -> int:
    """Return the published element-based frozen-core orbital count.

    This compatibility name delegates to ORCA 6.1 Table 2.69 via
    :func:`published_frozen_core_orbital_count`. vibe-qc uses its numerical
    count but does not implement ORCA's orbital-character reorder for unusual
    canonical orderings. ``reference`` (an SCF result) makes the count
    ECP-aware: cores an ECP already removed are not frozen again.
    """

    return published_frozen_core_orbital_count(molecule, reference)
