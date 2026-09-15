"""D4 reference polarizability dataset -- Phase D4b-4 part 3.

Each element in the D4 model is represented by a small set of
*reference systems* -- the element in different coordination
environments (catalogued in :mod:`vibeqc.dispersion_d4_reference_systems`).
For each reference system, the model needs:

* **Reference polarizability** ``a_i(iw)`` -- the dynamic dipole
  polarizability of the target atom on a fixed imaginary-frequency
  grid, used to compute the reference C6 coefficients via
  Casimir-Polder integration.
* **Reference charge** ``q_ref,i`` -- the EEQ partial charge of the
  target atom in the reference molecule, used by the charge-scaling
  factor ζ(q) (:func:`vibeqc.dispersion_d4_model.zeta_charge_scaling`).
* **Reference coordination number** ``cn_ref,i`` -- the nominal CN
  of the target atom (from the catalogue; the same as
  ``ReferenceSystem.nominal_cn``).

This module provides the storage format (:class:`D4ReferenceDataset`)
and the generation machinery -- run the D4b-2 pipeline
(:func:`vibeqc.molecular_c6`) over the D4b-3 catalogue.

Dataset convention -- per-atom Eq.-6 extraction (2026-06-26)
-----------------------------------------------------------
The generator stores the **per-atom** reference polarizability
``a_ref,A(iw)`` extracted from each reference host's whole-molecule
``a(iw)`` via Eq. 6 of Caldeweyher et al., *J. Chem. Phys.* **150**,
154122 (2019), doi:10.1063/1.5090222. For a reference host containing
``m`` equivalent target atoms of element A and any number of partner
atoms of other elements, subtract each partner's charge-scaled
canonical atomic reference and divide by ``m``:

    a_ref,A(iw) = [ a(host)(iw)
                    - Σ_{j: Z_j ≠ A}  ζ(q_j; q_ref,P_j) . a_canon,P_j(iw)
                  ] / m

Atoms of element A in the host are part of the ``m``-fold target and
are *not* subtracted. ``a_canon,P(iw)`` is element P's *canonical*
atomic reference, and ``ζ`` is the same D4 charge-scaling factor
(:func:`vibeqc.dispersion_d4_model.zeta_charge_scaling`) used at run
time -- here it maps the canonical reference from its own charge
state ``q_ref,P`` to the partner's actual in-host charge ``q_j``. The
ζ factor is essential: dropping it over-subtracts (e.g. α_C from CH₄
comes out ~5.7 a.u. without ζ, ~7.8 a.u. with it, against the dftd4
target 8.06).

The canonical atomic references are built first, in dependency order:

* **H** -- from the homonuclear H₂ reference: ``a_canon,H = a(H₂)/2``,
  ``q_ref,H = 0`` (neutral by symmetry).
* **Noble gases (He, Ne)** -- the free-atom reference directly.
* **B, C, N, O, F** -- from the pure hydride host (BH₃, CH₄, NH₃,
  H₂O, HF): all partners are H, subtracted via ``a_canon,H``.

Every other reference system (the heteroatomic / multi-heavy hosts
CO, CO₂, BF₃, HCN, N₂, F₂, C₂H₂, C₂H₄) is then extracted with the
same formula, subtracting the already-built canonical references of
its partner elements.

Accuracy. The molecular ``a(iw)`` is the **coupled CPKS / TD-DFT
(adiabatic) response** at PBE38
(:func:`vibeqc.dispersion_d4_refdata.coupled_polarizability_imag_freq_dft`),
*not* TD-HF. PBE38 (37.5 % exact exchange) is the level dftd4's
reference polarizabilities were computed at, so its correlation lifts
the O/N polarizabilities -- under-bound by ~15 % at the HF level -- up
to the reference. With the corrected r4r2 / C8 table
(:func:`vibeqc.dispersion_d4_model.r4r2_val`) the native pairwise C6
agree with dftd4 to ~5 % (CH₄) / ~8 % (H₂O) at aug-cc-pVDZ and the
CH₄-dimer D4 energy to <0.05 kcal/mol -- the handover Sec. 5 bars are
met and the native backend is **un-gated** (no
:class:`~vibeqc.dispersion_d4.D4NativeExperimentalWarning`). The
``dftd4`` backend remains the default for full periodic-table coverage
(this catalogue covers H-Ne).

Re-derivation stays first-principles per the 2026-05 maintainer
decision (``handovers/HANDOVER_D4_NATIVE.md`` Sec. 1) -- dftd4's LGPL-3.0
reference C6 table is *not* transcribed; it is the validation target
only. (The small per-element r4r2 / EEQ / BJ-damping parameter tables
are published scientific constants, transcribed with citation per
CLAUDE.md s1 -- distinct from the large reference C6 dataset.)

The imaginary-frequency grid and Casimir-Polder integrator are
vibe-qc's own (:func:`dispersion_d4_refdata.imaginary_frequency_grid`
+ :func:`dispersion_d4_refdata.casimir_polder_c6`), which use a
Gauss-Legendre quadrature with rational change-of-variable -- more
accurate for the same point count than the dftd4 trapezoidal grid.

Because vibe-qc's re-derived reference C6 table does **not** aim for
bit-exactness with the dftd4 LGPL dataset, the grid choice is a
physical-accuracy decision, not a compatibility constraint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "D4ReferenceDataset",
    "generate_reference_dataset",
    "N_GRID_POINTS",
]

# Number of imaginary-frequency grid points (dftd4 uses 23; we match
# for easy comparison but the actual grid differs -- see module docstring).
N_GRID_POINTS = 23


@dataclass
class _PerElement:
    """Reference data for one element."""

    # Atomic number.
    z: int

    # Reference coordination numbers, length n_ref.
    cns: list[float]

    # Reference EEQ charges (q_ref,i) for each reference system.
    qs: list[float]

    # Per-reference-system isotropic polarizability a_i(iw) on the
    # imaginary-frequency grid, shape (n_ref, N_GRID_POINTS).
    alpha_iw: np.ndarray

    # Pre-computed reference C6 coefficients between this element's
    # reference systems, shape (n_ref, n_ref).
    c6_self: np.ndarray

    @property
    def n_ref(self) -> int:
        return len(self.cns)


@dataclass
class D4ReferenceDataset:
    """The complete D4 reference dataset -- per-element reference
    polarizabilities, charges, coordination numbers, and pre-computed
    intra-element C6 coefficients.

    This is the data structure that :func:`compute_d4_c6_pair`
    consumes.  It is normally constructed once (either loaded from a
    JSON file or generated from scratch) and then treated as read-only.

    Attributes
    ----------
    elements
        Dict mapping atomic number -> :class:`_PerElement`.
    c6_cross
        Dict mapping ``(z_a, z_b)`` -> ``np.ndarray`` of shape
        ``(n_ref_a, n_ref_b)`` with the pre-computed reference C6
        between reference system *i* of element A and reference system
        *j* of element B.  Symmetric: only ``z_a <= z_b`` is stored.
    """

    elements: dict[int, _PerElement] = field(default_factory=dict)
    c6_cross: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    # ---- query API ----

    def get_cns(self, z: int) -> list[float]:
        """Reference coordination numbers for element ``z``."""
        return self.elements[z].cns

    def get_qs(self, z: int) -> list[float]:
        """Reference EEQ charges for element ``z``."""
        return self.elements[z].qs

    def get_c6_ref(self, za: int, ia: int, zb: int, jb: int) -> float:
        """Reference C6 between reference system ``ia`` of element
        ``za`` and reference system ``jb`` of element ``zb``."""
        if za == zb:
            return float(self.elements[za].c6_self[ia, jb])
        key = (za, zb) if za <= zb else (zb, za)
        mat = self.c6_cross[key]
        if za <= zb:
            return float(mat[ia, jb])
        return float(mat[jb, ia])

    @property
    def supported_z(self) -> list[int]:
        """Atomic numbers for which reference data is available."""
        return sorted(self.elements.keys())

    # ---- serialization ----

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict."""
        elem_dict = {}
        for z, e in self.elements.items():
            elem_dict[str(z)] = {
                "cns": e.cns,
                "qs": e.qs,
                "alpha_iw": e.alpha_iw.tolist(),
                "c6_self": e.c6_self.tolist(),
            }
        cross_dict = {}
        for (za, zb), mat in self.c6_cross.items():
            cross_dict[f"{za},{zb}"] = mat.tolist()
        return {"elements": elem_dict, "c6_cross": cross_dict}

    @classmethod
    def from_dict(cls, d: dict) -> "D4ReferenceDataset":
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        elements = {}
        for z_str, ed in d["elements"].items():
            z = int(z_str)
            elements[z] = _PerElement(
                z=z,
                cns=ed["cns"],
                qs=ed["qs"],
                alpha_iw=np.array(ed["alpha_iw"], dtype=float),
                c6_self=np.array(ed["c6_self"], dtype=float),
            )
        c6_cross = {}
        for key_str, mat in d.get("c6_cross", {}).items():
            za_str, zb_str = key_str.split(",")
            c6_cross[(int(za_str), int(zb_str))] = np.array(mat, dtype=float)
        return cls(elements=elements, c6_cross=c6_cross)

    def save_json(self, path: str | Path) -> None:
        """Write the dataset to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "D4ReferenceDataset":
        """Load a dataset from a JSON file written by
        :meth:`save_json`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---- Reference-data generation ----


def generate_reference_dataset(
    basis_name: str = "aug-cc-pvdz",
    *,
    functional: str = "pbe38",
    n_grid: int = N_GRID_POINTS,
    omega_scale: float = 0.5,
    verbose: bool = True,
) -> D4ReferenceDataset:
    """Run the full reference-data generation pipeline.

    The pipeline performs the per-atom Eq.-6 extraction (Caldeweyher
    et al. 2019; see module docstring): each reference host's
    whole-molecule ``a(iw)`` is partitioned to the target atom by
    subtracting the charge-scaled canonical atomic references of its
    partner atoms. The result is a per-atom reference dataset.

    The molecular ``a(iw)`` is the **coupled CPKS / TD-DFT (adiabatic)**
    response of an RKS reference at ``functional``
    (:func:`vibeqc.dispersion_d4_refdata.coupled_polarizability_imag_freq_dft`),
    *not* TD-HF. The correlation it carries is what lifts the O/N
    polarizabilities up to the level dftd4's PBE38 reference was built
    at, so the resulting reference C6 meet the dftd4 parity bars
    (handover Sec. 5: ~10 % pairwise C6, sub-0.1 kcal/mol dimer). The
    default ``functional="pbe38"`` (37.5 % exact exchange) matches
    dftd4's own reference protocol; ``basis_name="aug-cc-pvdz"`` is the
    validated production setting. Re-derivation stays first-principles
    (CLAUDE.md s1) -- dftd4's reference C6 table is never transcribed.

    For every reference system in the D4b-3 catalogue
    (:mod:`vibeqc.dispersion_d4_reference_systems`):

    1. Build the host molecule.
    2. Run RKS -> coupled CPKS a(iw) on the imaginary-frequency grid ->
       molecular isotropic ``a(iw)``.
    3. Compute EEQ charges (every atom; the target's is ``q_ref``).

    Then build the canonical atomic references (H from H₂, noble gases
    from their free atom, B/C/N/O/F from their pure hydride), extract
    the per-atom ``a_ref(iw)`` for every system via the Eq.-6 partner
    subtraction, and pre-compute the intra-element and cross-element
    reference C6 coefficients via Casimir-Polder integration.

    Parameters
    ----------
    basis_name
        Orbital basis for the RKS reference and CPKS response. The
        production default ``"aug-cc-pvdz"`` (diffuse functions matter
        for quantitative polarizabilities) is the validated setting;
        a smaller basis (e.g. ``"def2-svp"``) is faster but does not
        meet the parity bars.
    functional
        XC functional for the RKS reference and its TD-DFT response
        kernel. Default ``"pbe38"`` (37.5 % exact exchange, dftd4's own
        reference protocol; auto-registered on first use). Must be a
        pure functional or a global hybrid.
    n_grid
        Number of imaginary-frequency grid points.
    omega_scale
        Grid scale parameter -- see
        :func:`dispersion_d4_refdata.imaginary_frequency_grid`.

    Returns
    -------
    D4ReferenceDataset
        The complete dataset, ready to be saved or consumed directly.
    """
    # Deferred imports to avoid circularity at module level.
    from ._vibeqc_core import RKSOptions, eeq_charges, run_rks
    from .dispersion_d4_refdata import (
        casimir_polder_c6,
        coupled_polarizability_imag_freq_dft,
        imaginary_frequency_grid,
    )
    from .dispersion_d4_reference_systems import (
        all_reference_systems,
        build_molecule,
    )

    _ensure_functional_registered(functional)

    omegas, weights = imaginary_frequency_grid(
        n_points=n_grid,
        omega_scale=omega_scale,
    )

    rks_opts = RKSOptions()
    rks_opts.functional = functional
    rks_opts.conv_tol_energy = 1.0e-10

    # ---- Phase 1: collect per-reference-system molecular a(iw) and
    # the EEQ charge of EVERY atom (partners are needed for the Eq.-6
    # extraction; the target's is q_ref). ----
    systems: list[dict] = []
    for ref in all_reference_systems():
        if verbose:
            print(f"  {ref.label:20s} ...", end=" ", flush=True)

        mol = build_molecule(ref)
        basis = _resolve_basis(basis_name, mol)

        # RKS reference + coupled CPKS (TD-DFT adiabatic) a(iw).
        rks = run_rks(mol, basis, rks_opts)
        if not rks.converged:
            raise RuntimeError(
                f"RKS did not converge for {ref.label} (n_iter={rks.n_iter})."
            )
        alpha = coupled_polarizability_imag_freq_dft(
            rks, basis, mol, omegas, functional)

        # EEQ charge of every atom in the host.
        charges = np.asarray(eeq_charges(mol).charges, dtype=float)
        zs = [atom.Z for atom in mol.atoms]

        if verbose:
            print(
                f"a(0)={alpha[0]:.3f}  q_ref={charges[ref.target_index]:+.4f}"
            )

        systems.append(
            {
                "ref": ref,
                "zs": zs,
                "alpha": alpha.copy(),
                "charges": charges,
            }
        )

    # ---- Phase 1b: per-atom Eq.-6 extraction ----
    # Build canonical atomic references, then partition every host's
    # molecular a(iw) onto its target atom (see module docstring).
    canon = _build_canonical_references(systems)
    by_element: dict[int, list[tuple[float, np.ndarray, float]]] = {}
    for s in systems:
        ref = s["ref"]
        alpha_atom = _extract_atom_alpha(s, canon)
        z = ref.element
        by_element.setdefault(z, []).append(
            (
                float(ref.nominal_cn),
                alpha_atom,
                float(s["charges"][ref.target_index]),
            )
        )
        if verbose:
            print(
                f"    extracted {ref.label:18s} a_ref(0)={alpha_atom[0]:7.3f}"
            )

    # ---- Phase 2: build PerElement objects ----
    elements: dict[int, _PerElement] = {}
    all_alpha: dict[int, np.ndarray] = {}  # z -> (n_ref, n_grid)

    for z, entries in by_element.items():
        entries.sort(key=lambda x: x[0])  # sort by CN
        cns = [e[0] for e in entries]
        qs = [e[2] for e in entries]
        alpha_iw = np.array([e[1] for e in entries], dtype=float)
        n_ref = len(cns)

        # Pre-compute intra-element reference C6.
        c6_self = np.zeros((n_ref, n_ref))
        for i in range(n_ref):
            for j in range(n_ref):
                c6_self[i, j] = casimir_polder_c6(
                    alpha_iw[i],
                    alpha_iw[j],
                    weights,
                )

        elements[z] = _PerElement(
            z=z,
            cns=cns,
            qs=qs,
            alpha_iw=alpha_iw,
            c6_self=c6_self,
        )
        all_alpha[z] = alpha_iw

    # ---- Phase 3: pre-compute cross-element reference C6 ----
    c6_cross: dict[tuple[int, int], np.ndarray] = {}
    sorted_z = sorted(all_alpha.keys())
    for iza, za in enumerate(sorted_z):
        for zb in sorted_z[iza:]:  # upper triangle only
            alpha_a = all_alpha[za]
            alpha_b = all_alpha[zb]
            n_a, n_b = alpha_a.shape[0], alpha_b.shape[0]
            mat = np.zeros((n_a, n_b))
            for i in range(n_a):
                for j in range(n_b):
                    mat[i, j] = casimir_polder_c6(
                        alpha_a[i],
                        alpha_b[j],
                        weights,
                    )
            c6_cross[(za, zb)] = mat
            if verbose:
                print(f"  C6 cross Z=({za},{zb}) ... done ({n_a}x{n_b} entries)")

    return D4ReferenceDataset(elements=elements, c6_cross=c6_cross)


def _resolve_basis(basis_name: str, mol: "Molecule") -> "BasisSet":
    """Resolve ``basis_name`` to a :class:`BasisSet` for ``mol``."""
    from ._vibeqc_core import BasisSet

    if isinstance(basis_name, BasisSet):
        return basis_name
    return BasisSet(mol, basis_name)


def _ensure_functional_registered(functional: str) -> None:
    """Make sure ``functional`` resolves; register the reference
    functional ``pbe38`` on first use if it is not already known.

    PBE38 (0.375 exact exchange + 0.625 PBE exchange + PBE correlation)
    is the level dftd4's reference polarizabilities were computed at, so
    it is the default reference functional here. It is not a libxc
    built-in, so it is registered as a hand-mixed hybrid via
    :func:`vibeqc.define_functional` the first time it is requested.
    """
    from ._vibeqc_core import Functional

    try:
        Functional(functional, 1)
        return
    except Exception:
        pass
    if functional.lower() == "pbe38":
        from . import define_functional

        define_functional(
            "pbe38",
            [("GGA_X_PBE", 0.625), ("GGA_C_PBE", 1.0)],
            hf_exchange_fraction=0.375,
        )
        return
    raise ValueError(
        f"generate_reference_dataset: unknown functional {functional!r}; "
        f"register it with vibeqc.define_functional before generating."
    )


# ---- Per-atom Eq.-6 extraction helpers ----


def _num_target_atoms(system: dict) -> int:
    """Count atoms of the target element in a reference host."""
    z = system["ref"].element
    return sum(1 for zz in system["zs"] if zz == z)


def _zeta_partner(z_p: int, q_partner: float, q_ref: float) -> float:
    """ζ charge-scaling factor that maps element ``z_p``'s canonical
    atomic reference (built at charge ``q_ref``) to a partner atom's
    actual in-host charge ``q_partner``.

    Same factor as the run-time D4 weight
    (:func:`vibeqc.dispersion_d4_model.zeta_charge_scaling`); the
    charge arguments are shifted by the effective nuclear charge, per
    the D4 convention.
    """
    from .dispersion_d4_model import (
        D4_CHARGE_HEIGHT,
        D4_CHARGE_STEEPNESS,
        effective_nuclear_charge,
        zeta_charge_scaling,
        zeta_hardness,
    )

    gi = zeta_hardness(z_p) * D4_CHARGE_STEEPNESS
    z_eff = effective_nuclear_charge(z_p)
    return zeta_charge_scaling(
        D4_CHARGE_HEIGHT, gi, q_ref + z_eff, q_partner + z_eff
    )


def _build_canonical_references(
    systems: list[dict],
) -> dict[int, tuple[np.ndarray, float]]:
    """Canonical atomic reference ``a_canon(iw)`` and reference charge
    ``q_ref`` for every element present in ``systems``.

    The canonical reference of element A is the per-atom polarizability
    extracted from a designated host (see module docstring):

    * **H** -- the homonuclear H₂ host: ``a(H₂)/2``.
    * **Noble gases** -- the single-atom host directly.
    * **B/C/N/O/F** -- the *pure hydride* host (every partner is H),
      extracted by subtracting the canonical H reference per partner.

    Built in dependency order (H and noble gases first, then the
    hydrides) so each heavy-element extraction sees the H reference it
    needs.
    """
    canon: dict[int, tuple[np.ndarray, float]] = {}

    def find_host(predicate):
        for s in systems:
            if predicate(s):
                return s
        return None

    elements = sorted({s["ref"].element for s in systems})

    # Pass 1: H (homonuclear host) and noble-gas free atoms -- no heavy
    # partners, so no canonical references required to build them.
    for z in elements:
        hosts = [s for s in systems if s["ref"].element == z]
        if z == 1:
            # The pure-H host (H₂): every atom is hydrogen.
            host = find_host(
                lambda s: s["ref"].element == 1 and all(zz == 1 for zz in s["zs"])
            )
            if host is None:
                raise RuntimeError(
                    "generate_reference_dataset: no homonuclear H reference "
                    "(H₂) in the catalogue; cannot build the canonical H "
                    "polarizability."
                )
            m = _num_target_atoms(host)
            q_ref = float(host["charges"][host["ref"].target_index])
            canon[1] = (host["alpha"] / m, q_ref)
        elif len(hosts) == 1 and _host_is_free_atom(hosts[0]):
            s = hosts[0]
            canon[z] = (s["alpha"].copy(), float(s["charges"][s["ref"].target_index]))

    # Pass 2: heavy elements (B, C, N, O, F) -- from the pure-hydride
    # host (all partners are H), subtracting the canonical H reference.
    for z in elements:
        if z in canon:
            continue
        # A clean canonical hydride has exactly ONE heavy (target) atom
        # and every partner is H -- CH₄, NH₃, H₂O, BH₃, HF. The
        # single-heavy requirement excludes multi-heavy hosts that are
        # also all-C-or-H (e.g. C₂H₄, C₂H₂), which would otherwise be
        # picked by catalogue order and mis-define the reference.
        host = find_host(
            lambda s: s["ref"].element == z
            and _num_target_atoms(s) == 1
            and all(zz == z or zz == 1 for zz in s["zs"])
            and any(zz == 1 for zz in s["zs"])
        )
        if host is None:
            raise RuntimeError(
                f"generate_reference_dataset: no single-heavy pure-hydride "
                f"reference host for element Z={z}; cannot build its canonical "
                f"atomic polarizability. Add a hydride reference system or "
                f"extend _build_canonical_references."
            )
        canon[z] = (
            _extract_atom_alpha(host, canon),
            float(host["charges"][host["ref"].target_index]),
        )

    return canon


def _host_is_free_atom(system: dict) -> bool:
    """True when the host is a single free atom."""
    return len(system["zs"]) == 1


def _extract_atom_alpha(
    system: dict,
    canon: dict[int, tuple[np.ndarray, float]],
) -> np.ndarray:
    """Per-atom reference ``a_ref,A(iw)`` for one reference host, via
    the Eq.-6 partner subtraction (Caldeweyher 2019; module docstring).

        a_ref,A = [ a(host) - Σ_{j: Z_j≠A} ζ(q_j; q_ref,P) . a_canon,P ] / m

    Partner elements must already be present in ``canon``.
    """
    ref = system["ref"]
    z_target = ref.element
    alpha = system["alpha"].copy()
    for j, z_j in enumerate(system["zs"]):
        if z_j == z_target:
            continue  # same-element atoms are part of the m-fold target
        if z_j not in canon:
            raise RuntimeError(
                f"_extract_atom_alpha: partner element Z={z_j} in host "
                f"{ref.label!r} has no canonical reference yet."
            )
        a_canon, q_ref_p = canon[z_j]
        zeta = _zeta_partner(z_j, float(system["charges"][j]), q_ref_p)
        alpha = alpha - zeta * a_canon
    return alpha / _num_target_atoms(system)
