"""Build ``GradientOptions`` that mirror a molecular SCF options struct.

The analytic gradient must differentiate the Hamiltonian the SCF actually
solved. Two groups of settings decide what that Hamiltonian is, and both
live on the SCF options struct (``RHFOptions`` / ``UHFOptions`` /
``RKSOptions`` / ``UKSOptions``):

* the **JK backend** -- ``density_fit`` / ``aux_basis`` / ``cosx`` and the
  COSX grid selection;
* the **ECP setup**, which reaches the SCF by one of two routes: libecpint
  XML-library centres (``ecp_centers`` + ``ecp_library``) or inline
  primitive blocks (``ecp_primitive_blocks`` + ``ecp_primitive_centers`` +
  ``ecp_effective_charges`` + ``ecp_total_ncore``).

A default-constructed ``GradientOptions`` selects the direct four-index
all-electron path for both, so any caller that runs the SCF with one of
the settings above and the gradient with defaults differentiates a
different Hamiltonian than the energy (2026-05-18 audit P1/P2; #574;
#576). :func:`gradient_options_from_scf` is the one place that mirrors
the fields; ``vibeqc.ase``, ``vibeqc.hessian`` and every other
gradient-consuming driver go through it.

This module deliberately imports nothing optional (in particular not
``ase``), so ``vibeqc.hessian`` and ``vibeqc.irc`` can use it without
pulling the ASE calculator in.
"""

from __future__ import annotations

from ._vibeqc_core import GradientOptions


def gradient_options_from_scf(scf_options) -> GradientOptions:
    """Build a ``GradientOptions`` that mirrors the JK backend and ECP
    setup selected on ``scf_options``.

    Without this the gradient defaults to the four-index direct ERI path
    and the bare all-electron Z values, while the SCF may have used
    density fitting + COSX and/or ECPs, producing a force that
    differentiates a different Hamiltonian than the SCF energy
    (2026-05-18 audit P1/P2, #574, #576).

    Every field is read with ``getattr`` defaults, so any options object
    works -- including ``ROHFOptions`` (a Python dataclass without ECP
    fields) and ``None``-tolerant duck types used in tests.
    """
    grad_opts = GradientOptions()
    grad_opts.density_fit = bool(getattr(scf_options, "density_fit", False))
    grad_opts.aux_basis = str(getattr(scf_options, "aux_basis", "") or "")
    grad_opts.cosx = bool(getattr(scf_options, "cosx", False))
    cosx_grid = getattr(scf_options, "cosx_grid", None)
    if cosx_grid is not None:
        grad_opts.cosx_grid = cosx_grid
    # Match the COSX grid the SCF actually used so the force differentiates
    # the SAME exchange Hamiltonian as the energy. The SCF's grid choice is
    # conv-tolerance-aware (cosx_use_gridx in cosx.hpp): an AUTO grid level
    # (-1) uses a GridX tier at normal tolerance but falls back to the legacy
    # grid when conv_tol_grad is tighter than the GridX commutator floor
    # (1e-6). Mirror that decision here; GradientOptions has no conv_tol_grad
    # of its own, so we resolve it from the SCF options. (Explicit 0 / 1..4
    # pass through unchanged.)
    grad_opts.cosx_variant = getattr(
        scf_options, "cosx_variant", grad_opts.cosx_variant)
    if hasattr(grad_opts, "thresh_cosx"):
        grad_opts.thresh_cosx = float(
            getattr(scf_options, "thresh_cosx", grad_opts.thresh_cosx)
        )
    _gl = int(getattr(scf_options, "cosx_grid_level", grad_opts.cosx_grid_level))
    _conv = float(getattr(scf_options, "conv_tol_grad", 1e-6) or 1e-6)
    if _gl <= -1 and _conv < 1e-6:
        grad_opts.cosx_grid_level = 0     # auto + tight SCF -> legacy grid
    else:
        grad_opts.cosx_grid_level = _gl   # GridX (auto/explicit) or legacy
    copy_ecp_fields(grad_opts, scf_options)
    return grad_opts


def copy_ecp_fields(gradient_options, scf_options) -> None:
    """Overwrite the ECP fields of ``gradient_options`` with those of
    ``scf_options`` (both routes), in place.

    ECPs reach the SCF by two mutually exclusive routes and the gradient must
    follow whichever one ran, or it differentiates a different Hamiltonian
    than the energy (#574). Bases whose per-element cores have no libecpint
    XML library (vDZP, def2-mSVP, CRYSTAL/pob) use inline primitives and leave
    ``ecp_centers`` empty. The route that did *not* run is cleared, so a
    ``GradientOptions`` prepared for one geometry can be re-pointed at
    another (``vibeqc.hessian`` does this for every displaced geometry,
    #576).
    """
    ecp_blocks = getattr(scf_options, "ecp_primitive_blocks", None)
    ecp_centers = getattr(scf_options, "ecp_centers", None)
    if ecp_blocks and ecp_centers:
        raise ValueError(
            "copy_ecp_fields: SCF options mix the mutually exclusive XML "
            "and inline-primitive ECP routes"
        )
    if ecp_blocks:
        gradient_options.ecp_primitive_blocks = list(ecp_blocks)
        gradient_options.ecp_primitive_centers = [
            list(c) for c in getattr(scf_options, "ecp_primitive_centers", [])
        ]
        gradient_options.ecp_effective_charges = [
            float(q)
            for q in getattr(scf_options, "ecp_effective_charges", [])
        ]
        gradient_options.ecp_total_ncore = int(
            getattr(scf_options, "ecp_total_ncore", 0) or 0
        )
        gradient_options.ecp_centers = []
        gradient_options.ecp_library = ""
    elif ecp_centers:
        gradient_options.ecp_centers = list(ecp_centers)
        gradient_options.ecp_library = str(
            getattr(scf_options, "ecp_library", "") or ""
        )
        gradient_options.ecp_primitive_blocks = []
        gradient_options.ecp_primitive_centers = []
        gradient_options.ecp_effective_charges = []
        gradient_options.ecp_total_ncore = 0
    else:
        gradient_options.ecp_centers = []
        gradient_options.ecp_library = ""
        gradient_options.ecp_primitive_blocks = []
        gradient_options.ecp_primitive_centers = []
        gradient_options.ecp_effective_charges = []
        gradient_options.ecp_total_ncore = 0


__all__ = ["copy_ecp_fields", "gradient_options_from_scf"]
