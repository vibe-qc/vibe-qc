"""Wording for the vibrational-frequency block's potential-energy surface.

``run_job`` resolves a post-SCF correlated request down to the mean-field
reference its SCF actually runs -- ``_select_method("mp2", ...)`` returns
``"rhf"``, ``"ccsd"`` returns ``"rhf"``/``"uhf"``/``"rohf"``, and so on.
``compute_hessian_fd`` then differentiates *that* reference's analytic
gradient, because it accepts RHF / UHF / ROHF / RKS / UKS and nothing
else.  So the frequencies, the normal modes and the RRHO thermochemistry
built on them describe the **reference** surface, never the correlated
one the job's headline energy came from.

Until this module existed the ``.out`` said none of that: a
``run_job(method="mp2", hessian=True)`` frequency block was byte-for-byte
the ``method="rhf"`` block, and the thermochemistry rows were labelled
``E(elec)`` while carrying the RHF electronic energy in a job whose
reported energy was MP2.  A reader could not tell which surface the
numbers described.

Everything user-facing about that distinction lives here, so the wording
has one home and the runner only decides *when* to print it:

* :func:`hessian_surface_label` -- the reference's short name.
* :func:`hessian_surface_lines` -- the ``Surface:`` line, plus the
  explicit note when the surface is not the requested method's.
* :func:`thermochemistry_energy_labels` -- the row labels for the
  ``E + ZPE`` / ``H`` / ``G`` lines, naming the surface the electronic
  energy came from.
* :func:`hessian_unsupported_surface_lines` -- the SKIPPED block for a
  surface that has no finite-difference Hessian at all.
* :func:`hessian_surface_manifest_fields` -- the same facts as flat
  scalars for the ``.system`` manifest's ``[hessian]`` section.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "FD_HESSIAN_SURFACES",
    "hessian_surface_label",
    "hessian_surface_lines",
    "hessian_surface_manifest_fields",
    "hessian_unsupported_surface_lines",
    "thermochemistry_energy_labels",
]


# The mean-field references ``vibeqc.hessian.compute_hessian_fd`` accepts.
# Kept as the single source of truth so the runner's "can this surface be
# differentiated at all?" gate and this module's wording cannot drift.
FD_HESSIAN_SURFACES: frozenset[str] = frozenset(
    {"rhf", "uhf", "rohf", "rks", "uks"}
)

# ``method="auto"`` is a request to *choose* a method, not a request for a
# particular surface, so resolving it to RHF is not a substitution the
# user needs warning about.  Same for a bare mean-field request.
_NOT_A_REQUESTED_SURFACE: frozenset[str] = frozenset({"auto", ""})

# Width of the label column in the thermochemistry block, so the ``=``
# signs line up with the "Zero-point energy" / "Thermal corr. to U" rows
# above them.  The longest label this module emits is
# ``"H = E(ROHF) + H_corr"`` at 20 characters, so every surface fits.
_THERMO_LABEL_WIDTH = 25


def hessian_surface_label(
    resolved_method: str,
    *,
    functional: Optional[str] = None,
    basis: Optional[str] = None,
) -> str:
    """Return the short name of the surface a Hessian was built on.

    ``resolved_method`` is ``run_job``'s resolved method -- the mean-field
    reference the SCF ran, not the method the caller asked for.  The
    functional is appended for KS surfaces and the basis when given, so
    the label names the Hamiltonian rather than just its family::

        hessian_surface_label("rhf", basis="sto-3g")       -> "RHF/sto-3g"
        hessian_surface_label("uks", functional="b3lyp")   -> "UKS(B3LYP)"
    """
    label = str(resolved_method or "").upper()
    if functional and str(resolved_method).lower() in ("rks", "uks", "roks"):
        label = f"{label}({str(functional).upper()})"
    if basis:
        label = f"{label}/{basis}"
    return label


def _is_substituted(requested_method: str, resolved_method: str) -> bool:
    """True when the surface is not the surface the caller asked for.

    Compares the *requested* method with the *resolved* one rather than
    consulting a list of correlated methods: every post-SCF route that
    keeps a mean-field reference differs from it by construction, and a
    new one needs no edit here.  ``"auto"`` is excluded -- it asks
    vibe-qc to pick, so whatever it picks is what was requested.
    """
    req = str(requested_method or "").strip().lower()
    res = str(resolved_method or "").strip().lower()
    if req in _NOT_A_REQUESTED_SURFACE:
        return False
    return req != res


def hessian_surface_lines(
    *,
    requested_method: str,
    resolved_method: str,
    functional: Optional[str] = None,
    basis: Optional[str] = None,
    indent: str = "  ",
) -> str:
    """Return the surface provenance lines for the frequency block.

    Always names the surface.  When the surface is not the requested
    method's -- the correlated case -- it also says so outright, because
    the numbers underneath are otherwise indistinguishable from a job
    that really did run on the named method.
    """
    surface = hessian_surface_label(
        resolved_method, functional=functional, basis=basis
    )
    reference = hessian_surface_label(resolved_method)
    if not _is_substituted(requested_method, resolved_method):
        return f"{indent}Surface: {surface}\n"

    requested = str(requested_method).upper()
    return (
        f"{indent}Surface: {surface}  (NOT {requested})\n"
        f"{indent}This job's energy is {requested}, but the Hessian "
        f"finite-differences\n"
        f"{indent}the analytic {reference} gradient. The frequencies, "
        f"normal modes and\n"
        f"{indent}thermochemistry below describe the {reference} "
        f"reference surface;\n"
        f"{indent}no {requested} second derivatives were computed.\n"
    )


def thermochemistry_energy_labels(
    resolved_method: str,
) -> tuple[str, str, str]:
    """Return the ``(ZPE, enthalpy, Gibbs)`` row labels, padded to width.

    The electronic energy these rows add the thermal corrections to is
    the one the Hessian's surface produced, so the label names it:
    ``E(RHF) + ZPE`` rather than ``E(elec) + ZPE``.  In a correlated job
    that is the difference between a row a reader can check and a row
    they will read as the correlated enthalpy.
    """
    e = f"E({hessian_surface_label(resolved_method)})"
    return (
        f"{e} + ZPE".ljust(_THERMO_LABEL_WIDTH),
        f"H = {e} + H_corr".ljust(_THERMO_LABEL_WIDTH),
        f"G = {e} + G_corr".ljust(_THERMO_LABEL_WIDTH),
    )


def hessian_unsupported_surface_lines(
    *,
    requested_method: str,
    resolved_method: str,
    indent: str = "  ",
) -> str:
    """Return the SKIPPED block for a surface with no FD Hessian.

    ``compute_hessian_fd`` differentiates a mean-field analytic gradient
    and nothing else, so a method that does *not* collapse to one
    (CASSCF, CISD, CASPT2, FCI, ...) has no Hessian route through
    ``run_job``.  Saying that plainly beats letting the raw
    ``ValueError`` land in the block as ``FAILED: ValueError: FD
    Hessian: unknown method 'CASSCF'`` after the whole job has run.
    """
    requested = str(requested_method).upper()
    surface = hessian_surface_label(resolved_method)
    families = " / ".join(
        sorted(s.upper() for s in FD_HESSIAN_SURFACES)
    )
    lines = (
        f"{indent}SKIPPED -- no Hessian is available on the {surface} "
        f"surface.\n"
        f"{indent}run_job's finite-difference Hessian differentiates a "
        f"mean-field\n"
        f"{indent}analytic gradient ({families}), and\n"
        f"{indent}method='{requested.lower()}' does not resolve to one. "
        f"No frequencies and\n"
        f"{indent}no thermochemistry are reported.\n"
    )
    if str(resolved_method).strip().lower() == "casscf":
        lines += (
            f"{indent}A CASSCF Hessian exists as a Python API:\n"
            f"{indent}  vibeqc.hessian_casscf.compute_hessian_casscf"
            f"(mol, basis, active_space)\n"
            f"{indent}It is deliberately not wired into run_job. It "
            f"returns a\n"
            f"{indent}CASSCFHessianResult whose modes are unprojected "
            f"(no translation /\n"
            f"{indent}rotation projection) and which carries no dipole "
            f"derivatives, so the\n"
            f"{indent}frequency table, the RRHO thermochemistry and the "
            f"QVF vibrations\n"
            f"{indent}section cannot consume it as they stand.\n"
        )
    return lines


def hessian_surface_manifest_fields(
    *,
    requested_method: str,
    resolved_method: str,
    functional: Optional[str] = None,
    basis: Optional[str] = None,
    available: bool = True,
) -> dict[str, object]:
    """Return the ``[hessian]`` manifest section as flat scalars.

    ``surface_is_requested_method`` is the machine-readable form of the
    printed note: a consumer that reads frequencies out of a ``.system``
    manifest can gate on it without parsing prose.
    """
    return {
        "requested_method": str(requested_method or ""),
        "surface": hessian_surface_label(
            resolved_method, functional=functional, basis=basis
        ),
        "surface_method": str(resolved_method or ""),
        "surface_is_requested_method": not _is_substituted(
            requested_method, resolved_method
        ),
        "available": bool(available),
    }
