"""Geometric counterpoise (gCP) correction -- Kruse & Grimme 2012.

The gCP correction is a per-atom-pair, density-free additive correction
that approximates basis-set superposition error (BSSE) in the Boys-
Bernardi counterpoise sense without running the multiple ghost-atom
calculations the rigorous CP requires. It is the "missing third leg" of
the Grimme-school **composite 3c methods** alongside dispersion (D3 / D4)
and the per-functional modified short-range correction.

The functional form (Kruse & Grimme, *J. Chem. Phys.* **136**, 154101
(2012), eq. 4 + 5)::

    E_gCP = s . S_a S_{b!=a}  e^(-a . R_ab^b)  .  e_a^MIS  /  √(N_b^virt . S_ab)

where

* ``s, a, b`` are the three basis-set-specific fit parameters (Table 2
  of the paper);
* ``e_a^MIS`` is the per-atom "missing basis-set incompleteness" energy
  -- the difference between an isolated-atom HF energy in the target
  small basis and in a much larger reference basis (Table 3 + SI of the
  paper);
* ``N_b^virt`` is the number of virtual orbitals of atom *b* in the
  target basis (basis-dependent count);
* ``S_ab`` is the 1s-1s Slater-type-orbital overlap between atoms *a*
  and *b* with effective per-element exponents ``ζ_a``.

The correction is **structure-only**: no density, no orbitals, no SCF --
just geometry + the per-basis parameter set. This is what makes it
cheap to add to the 3c composites at zero SCF cost.

vibe-qc carries the published ``(s, a, b)`` fit parameters for the
basis sets the 3c composites use, plus the ``e_a^MIS``, ``N_b^virt``,
and Slater ``ζ_a`` tables for the H / C / N / O / F first-row organic
chemistry slice -- the most-published reference data, cross-verified
against the Kruse-Grimme 2012 paper and the canonical Grimme-group
``mctc-gcp`` reference implementation. Coverage beyond first-row is
not yet bundled; ``compute_gcp`` raises :class:`GCPDataMissing` with
a contribute-via-PR pointer when called on an unsupported element /
basis combination.

See :mod:`vibeqc.composites` for the keyword shortcuts (``hf-3c``,
``pbeh-3c``, ``b97-3c``, etc.) that wire gCP into a single-call
turnkey 3c run via :func:`vibeqc.run_job`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

try:
    # Python 3.11+: tomllib is stdlib. vibe-qc requires Python >= 3.11
    # per pyproject.toml so the import is unconditional in practice.
    import tomllib as _tomllib
except ImportError:  # pragma: no cover
    import tomli as _tomllib   # type: ignore[no-redef]

from ._vibeqc_core import Molecule
from ._d3_r0ab import r0ab as _r0ab
from ._gcp_slater import (
    PER_ELEMENT_SLATER_SHELL,
    effective_zeta,
    slater_overlap,
)


__all__ = [
    "GCPParams",
    "GCPResult",
    "GCPDataMissing",
    "compute_gcp",
    "gcp_params_for",
    "available_basis_sets",
]


class GCPDataMissing(ValueError):
    """Raised by :func:`compute_gcp` when an element or basis set falls
    outside the bundled gCP parameter coverage. Lists the missing
    (element, basis) pairs and points at the contribution pathway.
    """


@dataclass(frozen=True)
class GCPParams:
    """Geometric-counterpoise correction parameters for one basis set.

    Holds the four Kruse-Grimme fit constants (s, η, a, b) plus the
    per-element (e_mis, n_virt) tables. Per-element keys are atomic
    numbers (1 = H, 6 = C, ...). The per-element Slater-overlap exponent
    is *not* stored per-element -- it is computed from the universal
    Clementi-Raimondi single-zeta table in :mod:`vibeqc._gcp_slater`
    scaled by η, matching the canonical mctc-gcp Fortran reference.

    Attributes
    ----------
    basis_name : str
        Lowercase canonical name of the target basis set (e.g.
        ``"def2-svp"``, ``"minix"``, ``"def2-mtzvp"``).
    sigma, eta, alpha, beta : float
        The four basis-set-specific fit constants from the per-recipe
        ``case`` block in mctc-gcp ``src/gcp.f90``::

            E_gCP = s . S_a S_{b!=a}
                        exp(-a . R_ab^b) . e_a^MIS /
                        sqrt(N_b^virt . S_ab(η))

        where ``S_ab(η)`` is the shell-aware Slater overlap with
        per-element effective exponents ζ_X = η . (universal
        Clementi-Raimondi single-zeta). Dimensions: s dimensionless,
        η dimensionless (a multiplicative Slater-exponent scaling),
        a has units of bohr^(-b), b dimensionless.
    e_mis : dict[int, float]
        Per-element "missing basis-set incompleteness" energy
        (Hartree). Positive in the stored convention; the gCP
        formula applies the negative sign internally.
    n_virt : dict[int, float]
        Per-element virtual-orbital count in the target basis. The
        denominator under the square root in the gCP formula; floating
        point so half-integer "fractional" definitions remain
        expressible.
    etaspec : float
        Extra heavier-element (Z >= 11) Slater-exponent scaling factor.
        Default ``1.0``; r^2SCAN-3c sets this to ``1.15`` per the
        Grimme et al. 2021 recipe. Mirrors mctc-gcp's ``etaspec``
        parameter.
    citation : str
        One-line bibliographic pointer to the original publication
        introducing this parameter set. Surfaced in the SCF log under
        "Parameter provenance".
    variant : str
        Name of the per-method variant this instance was resolved from
        (``""`` for the basis-level default). The four fit constants are
        strictly per ``(method, basis)`` pair, not per basis: mctc-gcp's
        ``gcp.f90`` keys them on the method (``case ('hse3c')``), and
        several composites share one parent basis. See the ``[variants.*]``
        blocks in ``data_library/gcp/*.toml``.
    """

    basis_name: str
    sigma: float
    alpha: float
    beta: float
    eta: float = 1.0
    e_mis: dict[int, float] = field(default_factory=dict)
    n_virt: dict[int, float] = field(default_factory=dict)
    etaspec: float = 1.0
    citation: str = ""
    variant: str = ""

    def covers(self, atomic_numbers) -> tuple[list[int], list[int]]:
        """Return ``(supported, missing)`` element lists for the given
        atomic-number sequence. Used by :func:`compute_gcp` to surface
        coverage gaps before evaluating.

        An element is "supported" iff it has rows registered in **both**
        ``e_mis`` and ``n_virt`` (key existence, *not* nonzero values --
        the canonical mctc-gcp tables encode "this element-pair has
        zero gCP contribution" as ``e_mis = 0.0`` while still having
        the row registered; we honour that convention).
        """
        supported, missing = [], []
        for z in atomic_numbers:
            zi = int(z)
            if zi in self.e_mis and zi in self.n_virt:
                supported.append(zi)
            else:
                missing.append(zi)
        return supported, missing


@dataclass
class GCPResult:
    """Return value of :func:`compute_gcp`.

    Attributes
    ----------
    energy : float
        gCP correction energy (Hartree). Always non-negative -- gCP
        opposes the BSSE over-binding of incomplete basis sets, so a
        positive correction is added to the SCF total.
    gradient : np.ndarray, optional
        ``(n_atoms, 3)`` nuclear gradient dE_gCP/dR_A (Hartree/bohr).
        Only present when ``compute_gcp`` was called with
        ``with_gradient=True``; otherwise ``None``.
    basis_name : str
        Echo of the target basis set the correction was evaluated for.
    params : GCPParams
        The parameter set that was used. Recorded so callers can dump
        provenance into a log without a second lookup.
    """

    energy: float
    basis_name: str
    params: GCPParams
    gradient: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"GCPResult(basis={self.basis_name!r}, "
            f"energy={self.energy:+.8e}"
            + (", gradient=<...>)" if self.gradient is not None else ")")
        )


# ---------------------------------------------------------------------------
# Slater-type-orbital overlap is now implemented in :mod:`vibeqc._gcp_slater`
# (shell-aware analytic form ported from the canonical mctc-gcp Fortran
# reference at https://github.com/grimme-lab/gcp). The simpler 1s-1s
# geometric-mean form used in the v0.9.0-prep first-cut has been
# retired -- it didn't match the parameter fit conventions Grimme et al.
# used, so the bundled (s, η, a, b) constants weren't usable with it.
# Callers should use :func:`vibeqc._gcp_slater.slater_overlap` directly
# if they need the overlap surface from outside this module.
#
# A 1s-1s convenience kept under the original name for downstream
# tests / debugging -- defers to the canonical implementation with
# both shells pinned to 1.
def _slater_overlap_1s_1s(zeta_a: float, zeta_b: float, R: float) -> float:
    """Compatibility wrapper for the original 1s-1s overlap surface.
    Delegates to :func:`vibeqc._gcp_slater.slater_overlap` with both
    shells = 1 (Roothaan 1951 <1s|1s> analytic form). Symmetric in
    (ζ_a, ζ_b), positive on (0, inf), reduces to 1 at R = 0.
    """
    if R < 1e-10:
        return 1.0
    return slater_overlap(R, 1, 1, zeta_a, zeta_b)


# ---------------------------------------------------------------------------
# Parameter registry. Loaded from the TOML files in
# ``python/vibeqc/data_library/gcp/`` -- one file per basis, with
# per-publication citation + DOI + license metadata in the file's
# ``[metadata]`` block. See ``data_library/README.md`` for the full
# storage standard, ``data_library/_schema.toml`` for the per-kind
# schema, and ``data_library/gcp/_template.toml`` for a worked
# template.
#
# The registry is built lazily at module-import time so the file I/O
# only runs once per Python process; the dict comprehension below is
# the entire loader. Adding a new basis is a single TOML file in
# ``data_library/gcp/`` -- no code change.
# ---------------------------------------------------------------------------

_GCP_DATA_DIR = Path(__file__).resolve().parent / "data_library" / "gcp"

_ELEMENT_SYMBOLS_TO_Z = {
    "h": 1,  "he": 2,
    "li": 3, "be": 4, "b": 5,  "c": 6,  "n": 7,  "o": 8,  "f": 9,  "ne": 10,
    "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15, "s": 16, "cl": 17, "ar": 18,
    "k": 19, "ca": 20, "sc": 21, "ti": 22, "v": 23, "cr": 24, "mn": 25, "fe": 26,
    "co": 27, "ni": 28, "cu": 29, "zn": 30, "ga": 31, "ge": 32, "as": 33,
    "se": 34, "br": 35, "kr": 36, "rb": 37, "sr": 38, "y": 39, "zr": 40,
    "nb": 41, "mo": 42, "tc": 43, "ru": 44, "rh": 45, "pd": 46, "ag": 47,
    "cd": 48, "in": 49, "sn": 50, "sb": 51, "te": 52, "i": 53, "xe": 54,
    "cs": 55, "ba": 56, "la": 57, "ce": 58, "pr": 59, "nd": 60, "pm": 61,
    "sm": 62, "eu": 63, "gd": 64, "tb": 65, "dy": 66, "ho": 67, "er": 68,
    "tm": 69, "yb": 70, "lu": 71, "hf": 72, "ta": 73, "w": 74, "re": 75,
    "os": 76, "ir": 77, "pt": 78, "au": 79, "hg": 80, "tl": 81, "pb": 82,
    "bi": 83, "po": 84, "at": 85, "rn": 86,
}


def _parse_gcp_toml(path: Path) -> tuple[GCPParams, dict[str, GCPParams]]:
    """Parse one ``data_library/gcp/<name>.toml`` file into a basis-level
    :class:`GCPParams` plus its per-method ``[variants.*]`` overrides.

    The schema is documented in ``data_library/README.md`` and
    ``data_library/_schema.toml``. The loader validates the
    ``metadata.kind`` discriminator and propagates the citation
    string into the returned ``GCPParams.citation`` field.

    An optional ``[variants.<method>]`` block overrides any of the four
    fit constants (and ``etaspec``) for one composite method while reusing
    the file's ``[elements.*]`` tables verbatim. This mirrors mctc-gcp,
    where ``(sigma, eta, alpha, beta)`` are selected by *method*
    (``case ('hse3c')``) while ``e_mis`` / ``n_virt`` are selected by
    *basis* (``HFmsvp`` / ``BASmsvp``). Several composites share one parent
    basis -- pbeh-3c, b3lyp-3c and hse-3c all sit on def2-mSVP -- so a
    basis-only key cannot express them.

    Files with ``metadata.status = "pending"`` (no [elements.*]
    bodies populated yet) load cleanly into a ``GCPParams`` with
    empty per-element dicts; the eventual ``compute_gcp(..., basis)``
    call raises :class:`GCPDataMissing` with the right contribute-
    via-PR pointer.
    """
    with path.open("rb") as fh:
        doc = _tomllib.load(fh)

    meta = doc.get("metadata", {})
    if meta.get("kind") != "gcp_parameters":
        raise ValueError(
            f"{path.name}: metadata.kind is {meta.get('kind')!r}, "
            f"expected 'gcp_parameters' (see data_library/_schema.toml)."
        )
    basis_name = str(meta["name"]).lower()
    if path.stem.lower() != basis_name:
        raise ValueError(
            f"{path.name}: metadata.name={basis_name!r} does not match "
            f"file stem {path.stem!r}."
        )

    params = doc.get("parameters", {})
    sigma = float(params.get("sigma", 0.0))
    alpha = float(params.get("alpha", 0.0))
    beta  = float(params.get("beta", 0.0))
    # ``eta`` is optional for back-compat with v0.9.0-prep TOMLs that
    # only carried (sigma, alpha, beta); a missing eta defaults to 1.0
    # which makes the gCP attenuator use un-scaled Clementi-Raimondi
    # Slater exponents. Bundled TOMLs from extract_gcp_data.py carry
    # the canonical η.
    eta   = float(params.get("eta", 1.0))
    # ``etaspec`` is the extra heavier-element scaling factor; only
    # set for r^2SCAN-3c / def2-mTZVPP (where it = 1.15 per Grimme
    # 2021); default 1.0 elsewhere.
    etaspec = float(params.get("etaspec", 1.0))

    e_mis: dict[int, float] = {}
    n_virt: dict[int, float] = {}
    for sym, block in doc.get("elements", {}).items():
        z = _ELEMENT_SYMBOLS_TO_Z.get(sym.lower())
        if z is None:
            raise ValueError(
                f"{path.name}: unrecognised element symbol {sym!r} in "
                f"[elements.*]."
            )
        e_mis[z]  = float(block["e_mis"])
        n_virt[z] = float(block["n_virt"])

    def _cite(block: dict) -> str:
        cit = str(block.get("citation", ""))
        d = str(block.get("doi", ""))
        if d:
            cit = f"{cit} (DOI: {d})" if cit else f"DOI: {d}"
        return cit

    base = GCPParams(
        basis_name=basis_name,
        sigma=sigma,
        eta=eta,
        alpha=alpha,
        beta=beta,
        e_mis=e_mis,
        n_virt=n_virt,
        etaspec=etaspec,
        citation=_cite(meta),
    )

    variants: dict[str, GCPParams] = {}
    for vname, block in doc.get("variants", {}).items():
        key = vname.lower()
        variants[key] = replace(
            base,
            sigma=float(block.get("sigma", base.sigma)),
            eta=float(block.get("eta", base.eta)),
            alpha=float(block.get("alpha", base.alpha)),
            beta=float(block.get("beta", base.beta)),
            etaspec=float(block.get("etaspec", base.etaspec)),
            citation=_cite(block) or base.citation,
            variant=key,
        )

    return base, variants


def _load_gcp_registry() -> tuple[
    dict[str, GCPParams], dict[tuple[str, str], GCPParams]
]:
    """Discover and parse every ``data_library/gcp/*.toml`` file.

    Returns ``(by_basis, by_method)`` where ``by_method`` is keyed on
    ``(basis_name, variant_name)`` for the per-method ``[variants.*]``
    overrides.

    Files prefixed with ``_`` are treated as templates / schema and
    skipped. The result is cached at module-import time; new files
    require a process restart to register.
    """
    out: dict[str, GCPParams] = {}
    variants: dict[tuple[str, str], GCPParams] = {}
    if not _GCP_DATA_DIR.is_dir():
        # Build-time / source-tree mismatch -- surface clearly rather
        # than letting compute_gcp fail much later with an unhelpful
        # "no parameters registered" error.
        raise RuntimeError(
            f"vibeqc.gcp: data_library/gcp directory not found at "
            f"{_GCP_DATA_DIR}. Reinstall vibe-qc (`pip install -e .`) "
            f"to repopulate the parameter directory."
        )
    for path in sorted(_GCP_DATA_DIR.glob("*.toml")):
        if path.name.startswith("_"):
            continue  # _template.toml, _schema.toml, etc.
        params, vmap = _parse_gcp_toml(path)
        out[params.basis_name] = params
        for vname, vparams in vmap.items():
            variants[(params.basis_name, vname)] = vparams
    return out, variants


_BASIS_PARAMS, _VARIANT_PARAMS = _load_gcp_registry()


def available_basis_sets() -> list[str]:
    """Return the lowercase basis-set names known to the gCP registry,
    sorted alphabetically. Useful for surface introspection (e.g. the
    ``vq.list_methods()`` companion of the composites surface).
    """
    return sorted(_BASIS_PARAMS.keys())


def available_variants(basis_name: str) -> list[str]:
    """Return the per-method gCP variant names registered for a basis,
    sorted alphabetically. Empty when the basis carries only the
    basis-level default parameter set.
    """
    key = basis_name.lower().strip()
    return sorted(v for (b, v) in _VARIANT_PARAMS if b == key)


def gcp_params_for(
    basis_name: str, *, variant: Optional[str] = None
) -> Optional[GCPParams]:
    """Look up the gCP parameter set for a target basis. Case-insensitive,
    canonicalises ``"def2-SVP"`` / ``"DEF2-SVP"`` / ``"def2-svp"`` to
    the same registry entry. Returns ``None`` when the basis is not
    registered -- callers decide whether that's an error.

    ``variant`` selects a per-method override of the four fit constants
    (e.g. ``"hse-3c"`` on ``"def2-msvp"``). An unregistered ``variant``
    raises :class:`ValueError` rather than silently falling back to the
    basis-level default: a typo'd method name must not quietly reintroduce
    another composite's constants, which is precisely the class of bug this
    keying fixes. ``ValueError`` (not :class:`GCPDataMissing`) because a bad
    variant is a caller bug, not a data-coverage gap -- callers that degrade
    gracefully on missing element data must still fail loudly here.
    """
    key = basis_name.lower().strip()
    base = _BASIS_PARAMS.get(key)
    if base is None or variant is None:
        return base
    vkey = variant.lower().strip()
    try:
        return _VARIANT_PARAMS[(key, vkey)]
    except KeyError:
        known = ", ".join(available_variants(key)) or "(none)"
        raise ValueError(
            f"gcp_params_for(basis_name={basis_name!r}, "
            f"variant={variant!r}): no such gCP variant registered for "
            f"this basis. Registered variants: [{known}]. Add a "
            f"[variants.{vkey}] block to "
            f"data_library/gcp/{key}.toml, or drop the variant= "
            f"argument to use the basis-level parameter set."
        ) from None


def _ensure_full_data(params: GCPParams, atomic_numbers) -> None:
    """Raise GCPDataMissing if any atom's element is not covered by the
    parameter set's per-element tables. Run before evaluating to fail
    fast with a clear message rather than silently producing a wrong
    correction.
    """
    if params.beta == 0.0 and params.alpha == 0.0 and params.sigma == 0.0:
        # Recipe-side intentional "no gCP" marker (r^2SCAN-3c sets all
        # three to zero -- gCP is absorbed into the def2-mTZVPP basis
        # contraction by construction). Treat as a no-op.
        return
    _, missing = params.covers(atomic_numbers)
    if missing:
        unique_missing = sorted(set(missing))
        raise GCPDataMissing(
            f"compute_gcp(basis={params.basis_name!r}): per-element "
            f"gCP parameters not yet bundled for Z={unique_missing}.\n"
            f"  Bundled coverage for this basis: "
            f"Z={sorted(params.e_mis.keys())} (e_mis), "
            f"Z={sorted(params.n_virt.keys())} (n_virt).\n"
            f"  Reference parameter source: {params.citation or '(not recorded)'}.\n"
            f"  Contribute the missing per-element data via PR -- see "
            f"docs/user_guide/composites.md Sec. 'Extending the gCP "
            f"coverage', or supply your own table via "
            f"vibeqc.GCPParams(...) and call compute_gcp(..., "
            f"params=<your_params>)."
        )
    if not params.e_mis or not params.n_virt:
        raise GCPDataMissing(
            f"compute_gcp(basis={params.basis_name!r}): the (s, η, a, b) "
            f"fit constants for this basis are registered, but the "
            f"per-element e_mis / n_virt tables have not yet "
            f"been bundled in this first-cut release.\n"
            f"  Reference parameter source: {params.citation or '(not recorded)'}.\n"
            f"  Workaround: pass an explicit ``params=GCPParams(...)`` "
            f"with the per-element tables filled in. See "
            f"docs/user_guide/composites.md Sec. 'Extending the gCP "
            f"coverage'."
        )


def compute_gcp(
    mol: Molecule,
    basis_name: Optional[str] = None,
    *,
    variant: Optional[str] = None,
    params: Optional[GCPParams] = None,
    with_gradient: bool = False,
    damping: Optional[tuple] = None,
) -> GCPResult:
    """Geometric counterpoise correction (Kruse & Grimme 2012).

    Parameters
    ----------
    mol
        :class:`vibeqc.Molecule` (atomic positions in bohr; the gCP
        evaluator works in bohr natively).
    basis_name
        Lowercase canonical name of the target basis set (e.g.
        ``"def2-svp"``, ``"minis"``, ``"def2-mtzvpp"``). Looked up
        in the bundled parameter registry. Mutually exclusive with
        ``params``.
    variant
        Optional per-method gCP variant name selecting a method-specific
        override of the four fit constants for that basis (e.g.
        ``"hse-3c"`` on ``"def2-msvp"``, whose constants differ from the
        pbeh-3c ones the same basis file carries as its default). Only
        meaningful together with ``basis_name``. An unregistered name
        raises :class:`GCPDataMissing`.
    params
        Explicit :class:`GCPParams` instance -- bypasses the registry
        lookup. Use this to plug in published per-element data for a
        basis that's not yet bundled, or to evaluate gCP with a custom
        parameter set.
    with_gradient
        If True, the returned :class:`GCPResult` carries an
        ``(n_atoms, 3)`` array of dE_gCP/dR_A in Hartree/bohr. The
        gCP energy is a closed-form sum over atom pairs; the gradient
        is the analytic derivative of the same expression with respect
        to nuclear positions. Not yet available together with
        ``damping`` (raises :class:`NotImplementedError`).
    damping
        Optional ``(dmp_scal, dmp_exp)`` tuple enabling the short-range
        "damped gCP" used by the pbeh-3c / hse-3c / r2scan-3c composites
        (``gcp.f90`` ``damp=.true.`` path; Grimme group default
        ``(4.0, 6.0)``). Each pair contribution is scaled by
        ``1 - 1/(1 + dmp_scal.(R/r0ab)^dmp_exp)``, which suppresses the
        bonded-pair gCP so only genuine long-range (intermolecular) BSSE
        survives. ``None`` (default) reproduces the plain Kruse-Grimme
        gCP -- unchanged for every non-damped caller (def2-svp, minix,
        b3lyp-3c, standalone use). Without damping an isolated molecule's
        gCP is grossly over-counted (audit F1.5: r2scan-3c/H2O gives
        +28.8 undamped vs the correct +1.12 kcal/mol damped).

    Returns
    -------
    GCPResult

    Raises
    ------
    ValueError
        If neither ``basis_name`` nor ``params`` is given, or if both
        are.
    GCPDataMissing
        If the basis-set / element combination falls outside the
        bundled parameter coverage. The message lists the missing
        elements and points at the contribute-via-PR pathway.
    """
    if (basis_name is None) == (params is None):
        raise ValueError(
            "compute_gcp: pass exactly one of ``basis_name`` (registry "
            "lookup) or ``params`` (explicit parameter set)."
        )
    if params is not None and variant is not None:
        raise ValueError(
            "compute_gcp: ``variant`` selects a registry override and is "
            "meaningless together with an explicit ``params``. Pass the "
            "already-resolved GCPParams, or basis_name + variant."
        )
    if params is None:
        params = gcp_params_for(basis_name, variant=variant)  # type: ignore[arg-type]
        if params is None:
            known = ", ".join(available_basis_sets())
            raise GCPDataMissing(
                f"compute_gcp(basis_name={basis_name!r}): no gCP "
                f"parameter set registered for this basis. Bundled "
                f"registry: [{known}]. Pass an explicit "
                f"``params=GCPParams(...)`` to evaluate with a custom "
                f"parameter set."
            )

    if damping is not None and with_gradient:
        raise NotImplementedError(
            "compute_gcp: analytic gradient with short-range damping is "
            "not implemented yet. The composite energy totals use the "
            "damped gCP; geometry optimisation of a damped composite "
            "(pbeh-3c / hse-3c / r2scan-3c) needs the damped gradient -- "
            "tracked as a follow-up."
        )

    atoms = list(mol.atoms)
    n = len(atoms)
    if n < 2:
        # Single-atom or empty system -- no pairs, gCP is zero.
        return GCPResult(
            energy=0.0,
            basis_name=params.basis_name,
            params=params,
            gradient=(np.zeros((n, 3)) if with_gradient else None),
        )

    atomic_numbers = [int(a.Z) for a in atoms]
    _ensure_full_data(params, atomic_numbers)

    # r^2SCAN-3c case: the recipe sets (s, a, b) all to zero to signal
    # "no gCP" -- the basis contraction is tuned to absorb the BSSE.
    if params.sigma == 0.0 and params.alpha == 0.0 and params.beta == 0.0:
        return GCPResult(
            energy=0.0,
            basis_name=params.basis_name,
            params=params,
            gradient=(np.zeros((n, 3)) if with_gradient else None),
        )

    positions = np.array([list(a.xyz) for a in atoms], dtype=float)

    # Pre-compute the per-element effective Slater exponents and shell
    # quantum numbers; these are O(n_atoms) and used in the O(n^2) pair
    # loop below, so the precomputation is mandatory for performance on
    # larger systems.
    zeta = [
        effective_zeta(z, eta=params.eta, etaspec=params.etaspec)
        for z in atomic_numbers
    ]
    n_shell = [PER_ELEMENT_SLATER_SHELL[z] for z in atomic_numbers]

    energy = 0.0
    gradient = np.zeros((n, 3)) if with_gradient else None
    eps = 1e-12

    # Double-sum over ordered pairs (a, b) with a != b. The Kruse-Grimme
    # form is asymmetric in (a, b) -- e_a^MIS appears with a's identity
    # in the numerator while N_b^virt and S_ab appear with b's.
    for a in range(n):
        za = atomic_numbers[a]
        e_mis_a = params.e_mis[za]
        if e_mis_a == 0.0:
            # Atom a contributes nothing -- short-circuit the inner loop.
            continue
        zeta_a = zeta[a]
        shell_a = n_shell[a]
        for b in range(n):
            if b == a:
                continue
            zb = atomic_numbers[b]
            n_virt_b = params.n_virt[zb]
            if n_virt_b == 0.0:
                continue   # avoid 1/0 -- and a zero-virt atom adds nothing
            zeta_b = zeta[b]
            shell_b = n_shell[b]
            dvec = positions[a] - positions[b]
            R = float(np.linalg.norm(dvec))
            if R < eps:
                # Atoms on top of each other -- geometry pathology; skip
                # the pair rather than dividing by zero. The SCF would
                # also crash on this geometry; this just keeps gCP from
                # being the first complaint.
                continue
            S_ab = slater_overlap(R, shell_a, shell_b, zeta_a, zeta_b)
            S_safe = max(abs(S_ab), eps)
            denom = np.sqrt(n_virt_b * S_safe)
            attenuation = np.exp(-params.alpha * R ** params.beta)
            pair_contrib = (
                params.sigma * attenuation * e_mis_a / denom
            )
            if damping is not None:
                # Damped gCP (gcp.f90 damp=.true. path): scale the pair
                # contribution by 1 - 1/(1 + dmp_scal.(R/r0ab)^dmp_exp).
                # This is ~0 for short bonds (R << r0ab) and ~1 at long
                # range, so it strips the spurious bonded-pair gCP and
                # keeps only genuine intermolecular BSSE. Used by
                # pbeh-3c / hse-3c / r2scan-3c (audit F1.5).
                dmp_scal, dmp_exp = damping
                rs = R / _r0ab(za, zb)
                pair_contrib *= 1.0 - 1.0 / (1.0 + dmp_scal * rs ** dmp_exp)
            energy += pair_contrib

            if with_gradient:
                # d/dR_a of one (a, b) term:
                #   pair = s . e_a . g(R) / √(N_b . S(R))
                # with g(R) = e^(-a R^b), and R = |R_a - R_b|.
                # The (s . e_a / √N_b) prefactor is R-independent. The
                # radial derivative collects the two R-dependent pieces:
                #
                #   d(pair)/dR = (s . e_a / √N_b) .
                #                [g'(R) . S^(-1/2)
                #                 - 1/2 . g(R) . S^(-3/2) . S'(R)]
                #
                # dS/dR comes from a central finite difference on the
                # shell-aware Slater overlap -- the analytic derivative
                # branches by (n_a, n_b) pair so a single central FD is
                # both cleaner and accurate to ~1e-9 at h = 1e-4 R.
                dR = max(1e-5, 1e-4 * R)
                S_plus = slater_overlap(R + dR, shell_a, shell_b,
                                        zeta_a, zeta_b)
                S_minus = slater_overlap(max(R - dR, 1e-6), shell_a,
                                         shell_b, zeta_a, zeta_b)
                dS_dR = (S_plus - S_minus) / (2.0 * dR)
                dg_dR = (-params.alpha * params.beta
                         * R ** (params.beta - 1.0)
                         * attenuation)
                d_pair_dR = (
                    params.sigma * e_mis_a / np.sqrt(n_virt_b) * (
                        dg_dR / np.sqrt(S_safe)
                        - 0.5 * attenuation * S_safe ** (-1.5) * dS_dR
                    )
                )
                # Project radial derivative onto Cartesian: dR/dR_a =
                # (R_a - R_b)/R = dvec/R; dR/dR_b is the negative.
                d_pair_d_R_a = d_pair_dR * dvec / R
                gradient[a] += d_pair_d_R_a
                gradient[b] -= d_pair_d_R_a

    return GCPResult(
        energy=float(energy),
        basis_name=params.basis_name,
        params=params,
        gradient=gradient,
    )
