"""The solvation model a job ran, as the lines the ``.out`` prints.

A solvated energy is only reproducible if the file says which model produced
it: the dielectric and screening family, the cavity construction and how many
segments it ended with, the segment-charge kernel, and whether the
reaction-field cycles converged. The kernel and the cavity move the energy by
percent (the point-charge kernel is 17 to 27 percent off the sharp-cavity limit
at the default switching width, #681), so a block that printed only the energy
would hide the choice that matters most.
"""

from __future__ import annotations

from typing import Any

__all__ = ["solvation_model_lines"]

_BOHR2_TO_ANG2 = 0.529177210903**2


def solvation_model_lines(solvent_result: Any, *, indent: str = "  ") -> str:
    """The model description for a ``SolventResult``, one line per fact.

    Every line is read off the result; a field the result does not carry (an
    MSINDO result has no Gaussian-basis cavity, an older result no screening
    model) is skipped rather than guessed.
    """
    sr = solvent_result
    if sr is None:
        return ""
    lines: list[str] = []

    def add(label: str, text: str) -> None:
        lines.append(f"{indent}{label:<23}= {text}\n")

    name = getattr(sr, "solvent_name", None)
    eps = getattr(sr, "epsilon", None)
    if name is not None and eps is not None:
        eps_txt = "infinity (conductor)" if eps == float("inf") else f"{float(eps):.3f}"
        add("Solvent", f"{name}, epsilon = {eps_txt}")

    screening = getattr(sr, "screening", None)
    if screening is not None:
        variant = str(screening.variant).upper()
        if screening.is_conductor:
            add("Screening", f"{variant}, conductor limit, f = 1")
        else:
            add(
                "Screening",
                f"{variant}, f(eps) = (eps - 1)/(eps + {screening.x:g}) = {screening.f:.5f}",
            )

    model = getattr(sr, "solvent_model", None)
    cavity = getattr(sr, "cavity", None)
    if cavity is not None and hasattr(cavity, "points"):
        n_seg = int(cavity.points.shape[0])
        area = float(cavity.weights.sum())
        from ...solvation.cavity import CavityTessellation

        if getattr(model, "cavity", None) == "fine":
            add(
                "Cavity",
                f"COSMO FINE cavity (Klamt and Diedenhofen 2018), "
                f"{getattr(model, 'fine_frame', 'molecular')} frame",
            )
        elif isinstance(cavity, CavityTessellation):
            n_sphere = getattr(cavity, "n_points_per_sphere", None)
            sigma = getattr(cavity, "switching_sigma_bohr", None)
            scale = getattr(model, "radii_scale", None)
            parts = []
            if n_sphere is not None:
                parts.append(f"{int(n_sphere)} points per sphere")
            if scale is not None:
                parts.append(f"Bondi radii x {float(scale):g}")
            if sigma is not None:
                parts.append(f"switching width {float(sigma):g} bohr")
            add("Cavity", "switched Lebedev spheres" + (f" ({', '.join(parts)})" if parts else ""))
        add(
            "Cavity segments",
            f"{n_seg}, area {area:.2f} bohr^2 ({area * _BOHR2_TO_ANG2:.2f} Ang^2)",
        )
        rep = getattr(cavity, "charge_representation", None)
        if rep == "gaussian":
            add("Segment charges", "Gaussian (York and Karplus 1999)")
        elif rep == "point":
            add("Segment charges", "point charges (Scalmani and Frisch 2010 diagonal)")

    n_macro = getattr(sr, "n_macro_iter", None)
    if n_macro is not None:
        converged = getattr(sr, "converged", None)
        state = "" if converged is None else (" (converged)" if converged else " (NOT converged)")
        add("Reaction-field cycles", f"{int(n_macro)}{state}")
    return "".join(lines)
