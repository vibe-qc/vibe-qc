"""Write side-by-side Γ-CCM and χ-CCM diamond property bundles.

This is a full input-style script for a chemically analyzable covalent solid.
It runs closed-shell HF and KS-DFT on the same primitive diamond cell through
the two distinct CCM constructions, then writes:

* JSON summaries with energies, finite-net gaps, Mulliken charges, C-C Mayer
  bond orders, localization diagnostics, and optional local-PNO information;
* QVF archives with the primitive structure, bond connectivity, and band
  structure for both streams;
* QVF archives carrying finite-supercell localized occupied orbitals;
* PNG band plots when matplotlib is installed.

The output does not define a cross-approach numerical comparison. The default
STO-3G / 2x2x2 extension is a route smoke calculation, not a published
benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import vibeqc as vq
from vibeqc.output.formats.qvf import qvf_wf_data, write_qvf
from vibeqc.output.plan import OutputPlan
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_band_structure,
    ccm_homo_lumo_gap,
    ccm_mayer_bond_orders,
    ccm_mulliken_charges,
    localise_ccm,
    localization_density_residual,
)
from vibeqc.periodic.ccm.dft import run_ccm_rks
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable
from vibeqc.periodic_runner import (
    _aiccm_b_qvf_vendor_sections,
    _json_safe_qvf_value,
    _qvf_extensions_with,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
HARTREE_TO_EV = 27.211386245988


def diamond_primitive() -> vq.PeriodicSystem:
    """Return primitive diamond C in bohr.

    The periodic core and CCM utilities use direct-lattice vectors as columns.
    This conventional fcc primitive matrix is symmetric as written.
    """

    a = 3.567 * ANGSTROM_TO_BOHR
    lattice = 0.5 * a * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(6, [0.0, 0.0, 0.0]),
            vq.Atom(6, [0.25 * a, 0.25 * a, 0.25 * a]),
        ],
        charge=0,
        multiplicity=1,
    )


def qvf_plan(
    stem: Path,
    *,
    method: str,
    basis: str,
    functional: str | None = None,
) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=stem,
        method=method,
        basis=basis,
        functional=functional,
        job_kind="periodic_scf",
        output_qvf=True,
        write_molden_file=False,
        write_xyz=True,
        write_cif=True,
        citations=False,
    )


def qvf_bonds(bonds: Any) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for bond in bonds.bonds:
        if tuple(bond.translation) == (0, 0, 0) and bond.atom_i != bond.atom_j:
            out.append((int(bond.atom_i), int(bond.atom_j), float(bond.order)))
    if out:
        return out
    return [
        (int(bond.atom_i), int(bond.atom_j), float(bond.order))
        for bond in bonds.bonds[:8]
        if bond.atom_i != bond.atom_j
    ]


def _aiccm_b_qvf_context(result: Any) -> dict[str, Any]:
    """Return the runner-owned χ-CCM convention payload for a manual QVF."""

    sections = _aiccm_b_qvf_vendor_sections(result)
    if not sections:
        raise ValueError(
            "χ-CCM QVF output requires finite-torus convention diagnostics"
        )
    return {
        "extensions": _qvf_extensions_with(None, "x_vibeqc"),
        "vendor_json_sections": sections,
    }


def write_periodic_qvf(
    stem: Path,
    *,
    system: vq.PeriodicSystem,
    basis_name: str,
    method: str,
    functional: str | None,
    bands: vq.BandStructure,
    bonds: Any,
    aiccm_b_result: Any | None = None,
) -> Path:
    if "aiccm2026dev-b" in method.casefold() and aiccm_b_result is None:
        raise ValueError(
            "χ-CCM QVF output requires the source result so construction "
            "identity cannot be omitted"
        )
    vendor_context = (
        {} if aiccm_b_result is None else _aiccm_b_qvf_context(aiccm_b_result)
    )
    return write_qvf(
        stem,
        qvf_plan(stem, method=method, basis=basis_name, functional=functional),
        system=system,
        band_structure=bands,
        bonds_data=qvf_bonds(bonds),
        **vendor_context,
    )


def write_localized_qvf(
    stem: Path,
    *,
    molecule: vq.Molecule,
    basis: vq.BasisSet,
    coefficients: np.ndarray,
    method: str,
    basis_name: str,
    aiccm_b_result: Any | None = None,
) -> Path:
    if "aiccm2026dev-b" in method.casefold() and aiccm_b_result is None:
        raise ValueError(
            "χ-CCM QVF output requires the source result so construction "
            "identity cannot be omitted"
        )
    n_mo = int(coefficients.shape[1])
    fake_result = SimpleNamespace(
        mo_coeffs=np.asarray(coefficients, dtype=float),
        mo_energies=np.zeros(n_mo, dtype=float),
    )
    wf = qvf_wf_data(
        fake_result,
        basis,
        molecule,
        orbital_kind="localized",
        k_point=[0.0, 0.0, 0.0],
    )
    plan = OutputPlan.from_run_job_kwargs(
        output=stem,
        method=method,
        basis=basis_name,
        functional=None,
        job_kind="post_scf",
        output_qvf=True,
        write_molden_file=False,
        write_xyz=True,
        citations=False,
    )
    vendor_context = (
        {} if aiccm_b_result is None else _aiccm_b_qvf_context(aiccm_b_result)
    )
    return write_qvf(
        stem,
        plan,
        molecule=molecule,
        result=fake_result,
        basis=basis,
        wf_data=wf,
        **vendor_context,
    )


def bond_rows(bonds: Any) -> list[dict[str, Any]]:
    return [
        {
            "atom_i": int(bond.atom_i),
            "atom_j": int(bond.atom_j),
            "symbols": f"{bond.symbol_i}-{bond.symbol_j}",
            "translation": [int(x) for x in bond.translation],
            "distance_angstrom": float(bond.distance_bohr / ANGSTROM_TO_BOHR),
            "mayer_order": float(bond.order),
        }
        for bond in bonds.bonds
    ]


def convention_record(convention: Any) -> dict[str, Any]:
    record = _json_safe_qvf_value(convention)
    if not isinstance(record, dict):
        raise TypeError("finite-torus convention must serialize to a JSON object")
    return record


def band_summary(bands: vq.BandStructure) -> dict[str, Any]:
    shifted_ev = bands.shifted_energies * HARTREE_TO_EV
    summary: dict[str, Any] = {
        "n_points": int(bands.kpath.n_points),
        "n_bands": int(bands.n_bands),
        "fermi_hartree": None if bands.e_fermi is None else float(bands.e_fermi),
        "energy_window_ev": [
            float(np.min(shifted_ev)),
            float(np.max(shifted_ev)),
        ],
    }
    convention = bands.metadata.get("finite_torus_convention")
    if convention is not None:
        summary["finite_torus_convention"] = convention_record(convention)
    return summary


def save_band_plot(path: Path, bands: vq.BandStructure, title: str) -> None:
    try:
        from vibeqc.plot import band_structure_figure
    except ImportError:
        return
    fig = band_structure_figure(bands, title=title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except ImportError:
        pass


def summarize_a_result(
    *,
    label: str,
    result: Any,
    ccm: CCMSystem,
    bands: vq.BandStructure,
    bonds: Any,
    localization: Any,
) -> dict[str, Any]:
    gap = ccm_homo_lumo_gap(result, ccm)
    mulliken = ccm_mulliken_charges(result, ccm)
    return {
        "label": label,
        "energy_per_cell_hartree": float(result.energy_per_atom * ccm.n_basis_atoms),
        "converged": bool(result.converged),
        "gap_ev": float(gap.gap * HARTREE_TO_EV),
        "mulliken_charges_per_cell": [
            float(value) for value in mulliken.charges_per_cell
        ],
        "charge_spread": float(mulliken.translational_spread),
        "bond_order_spread": float(bonds.translational_spread),
        "bonds": bond_rows(bonds),
        "bands": band_summary(bands),
        "localization": {
            "method": "pipek-mezey",
            "density_residual": float(
                localization_density_residual(result, localization)
            ),
            "n_occ": int(localization.n_occ),
            "spreads_bohr2": [float(x) for x in np.asarray(localization.spreads)],
            "centers_bohr": np.asarray(localization.centers).tolist(),
        },
    }


def summarize_b_result(
    *,
    label: str,
    result: Any,
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    bands: vq.BandStructure,
    bonds: Any,
    localization: Any,
) -> dict[str, Any]:
    props = vq.derive_aiccm2026dev_b_scf_properties(result, system, basis)
    return {
        "label": label,
        "energy_per_cell_hartree": float(result.energy),
        "converged": bool(result.converged),
        "backend": str(result.aiccm2026dev_b.backend),
        "finite_torus_convention": convention_record(
            props.finite_torus_convention
        ),
        "gap_ev": float(props.band_gap_hartree * HARTREE_TO_EV),
        "mulliken_charges_per_cell": [
            float(value) for value in props.mulliken_charges
        ],
        "density_idempotency_error": float(props.density_idempotency_error),
        "bond_order_spread": float(bonds.translational_spread),
        "bond_order_convention": convention_record(
            bonds.finite_torus_convention
        ),
        "bonds": bond_rows(bonds),
        "bands": band_summary(bands),
        "localization": {
            "method": "wannier",
            "density_invariance_error": float(localization.density_invariance_error),
            "unitary_error": float(localization.unitary_error),
            "translation_covariance_error": float(
                localization.translation_covariance_error
            ),
            "spreads_bohr2": [float(x) for x in np.asarray(localization.spreads_bohr2)],
            "centers_bohr": np.asarray(localization.centers_bohr).tolist(),
            "aliasing_detected": [
                bool(value) for value in np.asarray(localization.aliasing_detected)
            ],
        },
    }


def run_a_stream(
    *,
    system: vq.PeriodicSystem,
    basis_name: str,
    extension: tuple[int, int, int],
    functional: str,
    kpath: vq.KPath,
    outdir: Path,
    max_iter: int,
) -> dict[str, Any]:
    ccm = CCMSystem(system, extension, basis_name)
    rhf = run_ccm_rhf_scalable(
        ccm,
        method="aiccm2026dev-a",
        max_iter=max_iter,
        conv_tol=1.0e-8,
    )
    rks = run_ccm_rks(
        ccm,
        functional=functional,
        method="aiccm2026dev-a",
        max_iter=max_iter,
        conv_tol=1.0e-7,
    )
    rhf_bonds = ccm_mayer_bond_orders(rhf, ccm, threshold=0.02, max_bonds=32)
    rks_bonds = ccm_mayer_bond_orders(rks, ccm, threshold=0.02, max_bonds=32)
    rhf_bands = ccm_band_structure(rhf, ccm, kpath)
    rks_bands = ccm_band_structure(rks, ccm, kpath)
    localized = localise_ccm(rhf, ccm, method="pipek-mezey")
    write_periodic_qvf(
        outdir / "diamond-aiccm2026dev-a-rhf",
        system=system,
        basis_name=basis_name,
        method="RHF/aiccm2026dev-a",
        functional=None,
        bands=rhf_bands,
        bonds=rhf_bonds,
    )
    write_periodic_qvf(
        outdir / "diamond-aiccm2026dev-a-rks",
        system=system,
        basis_name=basis_name,
        method=f"RKS/{functional}/aiccm2026dev-a",
        functional=functional,
        bands=rks_bands,
        bonds=rks_bonds,
    )
    write_localized_qvf(
        outdir / "diamond-aiccm2026dev-a-rhf-localized",
        molecule=ccm.supercell,
        basis=ccm.basis,
        coefficients=np.asarray(localized.C_loc, dtype=float),
        method="RHF/aiccm2026dev-a/localized",
        basis_name=basis_name,
    )
    save_band_plot(
        outdir / "diamond-aiccm2026dev-a-rhf-bands.png",
        rhf_bands,
        "Diamond AICCM-A RHF",
    )
    save_band_plot(
        outdir / "diamond-aiccm2026dev-a-rks-bands.png",
        rks_bands,
        f"Diamond AICCM-A RKS/{functional}",
    )
    return {
        "rhf": summarize_a_result(
            label="aiccm2026dev-a RHF",
            result=rhf,
            ccm=ccm,
            bands=rhf_bands,
            bonds=rhf_bonds,
            localization=localized,
        ),
        "rks": summarize_a_result(
            label=f"aiccm2026dev-a RKS/{functional}",
            result=rks,
            ccm=ccm,
            bands=rks_bands,
            bonds=rks_bonds,
            localization=localized,
        ),
    }


def run_b_stream(
    *,
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    basis_name: str,
    extension: tuple[int, int, int],
    functional: str,
    backend: str,
    kpath: vq.KPath,
    outdir: Path,
    max_iter: int,
) -> dict[str, Any]:
    rhf_options = vq.PeriodicRHFOptions()
    rhf_options.max_iter = max_iter
    rhf_options.conv_tol_energy = 1.0e-8
    rks_options = vq.PeriodicKSOptions()
    rks_options.max_iter = max_iter
    rks_options.conv_tol_energy = 1.0e-7
    rhf = vq.run_aiccm2026dev_b_rhf(
        system,
        basis,
        extension,
        rhf_options,
        backend=backend,
        progress=False,
    )
    rks = vq.run_aiccm2026dev_b_rks(
        system,
        basis,
        functional,
        extension,
        rks_options,
        backend=backend,
        progress=False,
    )
    rhf_bonds = vq.aiccm2026dev_b_mayer_bond_orders(
        rhf,
        system,
        basis,
        threshold=0.02,
        max_bonds=32,
    )
    rks_bonds = vq.aiccm2026dev_b_mayer_bond_orders(
        rks,
        system,
        basis,
        threshold=0.02,
        max_bonds=32,
    )
    rhf_bands = vq.aiccm2026dev_b_band_structure(rhf, system, kpath)
    rks_bands = vq.aiccm2026dev_b_band_structure(rks, system, kpath)
    localized = vq.localize_aiccm2026dev_b_occupied(
        rhf,
        system,
        basis,
        method="wannier",
    )
    write_periodic_qvf(
        outdir / "diamond-aiccm2026dev-b-rhf",
        system=system,
        basis_name=basis_name,
        method=f"RHF/aiccm2026dev-b/{backend}",
        functional=None,
        bands=rhf_bands,
        bonds=rhf_bonds,
        aiccm_b_result=rhf,
    )
    write_periodic_qvf(
        outdir / "diamond-aiccm2026dev-b-rks",
        system=system,
        basis_name=basis_name,
        method=f"RKS/{functional}/aiccm2026dev-b/{backend}",
        functional=functional,
        bands=rks_bands,
        bonds=rks_bonds,
        aiccm_b_result=rks,
    )
    from vibeqc.periodic.chi.posthf import _build_supercell_system

    super_system = _build_supercell_system(system, extension)
    super_molecule = super_system.unit_cell_molecule()
    super_basis = vq.BasisSet(super_molecule, basis_name)
    write_localized_qvf(
        outdir / "diamond-aiccm2026dev-b-rhf-localized",
        molecule=super_molecule,
        basis=super_basis,
        coefficients=np.asarray(localized.coefficients, dtype=float),
        method=f"RHF/aiccm2026dev-b/{backend}/localized",
        basis_name=basis_name,
        aiccm_b_result=rhf,
    )
    save_band_plot(
        outdir / "diamond-aiccm2026dev-b-rhf-bands.png",
        rhf_bands,
        "Diamond χ-CCM RHF",
    )
    save_band_plot(
        outdir / "diamond-aiccm2026dev-b-rks-bands.png",
        rks_bands,
        f"Diamond χ-CCM RKS/{functional}",
    )
    return {
        "rhf": summarize_b_result(
            label=f"aiccm2026dev-b RHF/{backend}",
            result=rhf,
            system=system,
            basis=basis,
            bands=rhf_bands,
            bonds=rhf_bonds,
            localization=localized,
        ),
        "rks": summarize_b_result(
            label=f"aiccm2026dev-b RKS/{functional}/{backend}",
            result=rks,
            system=system,
            basis=basis,
            bands=rks_bands,
            bonds=rks_bonds,
            localization=localized,
        ),
    }


def maybe_run_b_pno(
    *,
    system: vq.PeriodicSystem,
    basis: vq.BasisSet,
    extension: tuple[int, int, int],
    outdir: Path,
) -> dict[str, Any] | None:
    try:
        from vibeqc.dlpno.mp2 import DLPNOMP2Options
    except ImportError:
        return None
    # This optional smoke predates the molecular #140/#448 convention sweep.
    # Keep its old all-electron/custom-threshold protocol exact and serialize
    # every coordinate beside the energy instead of silently mixing old and
    # new option-class defaults.
    dlpno_options = DLPNOMP2Options(
        localise="wannier",
        n_frozen=0,
        tcut_pno=1.0e-8,
        tcut_pno_weak=1.0e-7,
        tcut_mkn=1.0e-3,
        tcut_pairs=0.0,
        tcut_pairs_weak=1.0e-4,
    )
    result = vq.run_aiccm2026dev_b_dlpno_mp2(
        system,
        basis,
        extension,
        dlpno_options=dlpno_options,
        progress=False,
    )
    payload = {
        "e_corr_per_cell_hartree": float(result.e_corr_per_cell),
        "e_total_per_cell_hartree": float(result.e_total_per_cell),
        "finite_torus_convention": convention_record(
            result.finite_torus_convention
        ),
        "n_pairs": int(result.n_pairs),
        "n_pairs_screened": int(result.n_pairs_screened),
        "localization": str(result.localization),
        "dlpno_recipe": {
            "convention": "pre-#140/#448-explicit-all-electron",
            "n_frozen": int(dlpno_options.n_frozen),
            "tcut_pairs": float(dlpno_options.tcut_pairs),
            "tcut_pairs_weak": float(dlpno_options.tcut_pairs_weak),
            "tcut_pno": float(dlpno_options.tcut_pno),
            "tcut_pno_weak": float(dlpno_options.tcut_pno_weak),
            "tcut_mkn": float(dlpno_options.tcut_mkn),
        },
        "local_correlation_space": None,
    }
    if result.local_correlation_space is not None:
        local = result.local_correlation_space
        payload["local_correlation_space"] = {
            "pao_rank": int(local.pao_rank),
            "canonical_virtual_rank": int(local.canonical_virtual_rank),
            "n_translation_unique_pairs": int(local.n_translation_unique_pairs),
            "translation_reduction_factor": float(local.translation_reduction_factor),
            "pao_orthonormality_error": float(local.pao_orthonormality_error),
        }
    (outdir / "diamond-aiccm2026dev-b-dlpno-mp2.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("both", "a", "b"), default="both")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--functional", default="pbe")
    parser.add_argument(
        "--backend",
        choices=("four_center", "ri", "rijcosx"),
        default="four_center",
        help="χ-CCM electron-repulsion backend.",
    )
    parser.add_argument("--extension", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument("--band-points", type=int, default=18)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--output-dir", default="output/aiccm-diamond-compare")
    parser.add_argument(
        "--with-b-pno-mp2",
        action="store_true",
        help="Also run the B local-PNO MP2 pilot and write a PNO summary.",
    )
    args = parser.parse_args()

    system = diamond_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), args.basis)
    extension = tuple(int(value) for value in args.extension)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    kpath = vq.KPoints.band_path(
        system,
        scheme="manual",
        segments=[
            ([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.5], "X"),
            ([0.5, 0.0, 0.5], "X", [0.5, 0.25, 0.75], "W"),
            ([0.5, 0.25, 0.75], "W", [0.375, 0.375, 0.75], "K"),
            ([0.375, 0.375, 0.75], "K", [0.0, 0.0, 0.0], "G"),
            ([0.0, 0.0, 0.0], "G", [0.5, 0.5, 0.5], "L"),
        ],
        points_per_segment=int(args.band_points),
    ).to_kpath()

    summary: dict[str, Any] = {
        "system": "diamond primitive C2",
        "basis": args.basis,
        "functional": args.functional,
        "extension": list(extension),
        "band_path": "G-X-W-K-G-L",
        "gamma_ccm_chi_ccm_approach_comparison_status": "not-defined",
        "notes": [
            "Gamma-CCM is union-and-weight; chi-CCM is finite-character.",
            "No matched cross-approach operator contract is defined.",
            "STO-3G is for fast route smoke calculations only.",
            "Band paths are finite-torus Fock-block interpolations.",
            "QVF localized archives use finite-supercell occupied orbitals.",
        ],
    }
    if args.route in ("both", "a"):
        summary["aiccm2026dev_a"] = run_a_stream(
            system=system,
            basis_name=args.basis,
            extension=extension,
            functional=args.functional,
            kpath=kpath,
            outdir=outdir,
            max_iter=int(args.max_iter),
        )
    if args.route in ("both", "b"):
        summary["aiccm2026dev_b"] = run_b_stream(
            system=system,
            basis=basis,
            basis_name=args.basis,
            extension=extension,
            functional=args.functional,
            backend=args.backend,
            kpath=kpath,
            outdir=outdir,
            max_iter=int(args.max_iter),
        )
        if args.with_b_pno_mp2:
            summary["aiccm2026dev_b_pno_mp2"] = maybe_run_b_pno(
                system=system,
                basis=basis,
                extension=extension,
                outdir=outdir,
            )

    summary_path = outdir / "diamond-aiccm2026dev-a-vs-b-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print("QVF and plot outputs are in", outdir)


if __name__ == "__main__":
    main()
