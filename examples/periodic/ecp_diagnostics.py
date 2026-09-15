"""Quick pre-SCF ECP diagnostics for periodic systems.

Prints a one-line sanity summary before the main SCF — catch
silent all-electron fallbacks, wrong valence counts, or missing
ECP data without waiting for a converged (or diverging) SCF.

Usage:
    # From the command line:
    python examples/periodic/ecp_diagnostics.py

    # Or import and use programmatically:
    from examples.periodic.ecp_diagnostics import print_ecp_diagnostics
    print_ecp_diagnostics(system, basis, opts)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the example directory directly.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import vibeqc as vq

ANG = 1.8897261246257702


def print_ecp_diagnostics(system: vq.PeriodicSystem, basis: vq.BasisSet, opts) -> None:
    """Print a one-line ECP sanity summary.

    Call this just before the SCF to confirm:
      - ECP data was attached (not silently fallen back to all-electron)
      - valence electron count matches expectations
      - effective nuclear charges make physical sense
    """
    n_elec_full = system.n_electrons()
    has_ecp = bool(getattr(opts, "ecp_primitive_blocks", None))
    ncore = int(getattr(opts, "ecp_total_ncore", 0))
    n_elec_valence = n_elec_full - ncore

    atoms = list(system.unit_cell)
    z_list = [int(a.Z) for a in atoms]
    n_atoms = len(atoms)

    print("\n  ── ECP diagnostics ──")
    print(f"  basis:           {basis.name}")
    print(f"  unit cell:       {n_atoms} atoms  Z = {z_list}")
    print(f"  n_elec full:     {n_elec_full}")

    if has_ecp:
        from vibeqc.periodic_runner import _resolve_ecp_data

        # Re-resolve to double-check (the opts fields are authoritative).
        ecp_blocks, ecp_centers, eff_z, total_ncore = _resolve_ecp_data(system, basis)
        print(f"  ECP attached:    YES  ({len(ecp_blocks)} ECP centres)")
        print(f"  ncore replaced:  {total_ncore}")
        print(
            f"  n_elec valence:  {n_elec_valence}  "
            f"({'even' if n_elec_valence % 2 == 0 else 'ODD — check!'})"
        )
        print(f"  effective Z:     {[float(z) for z in eff_z]}")
        for i, a in enumerate(atoms):
            bare_z = int(a.Z)
            eff = float(eff_z[i]) if i < len(eff_z) else bare_z
            if eff != bare_z:
                print(
                    f"    atom {i}:  Z={bare_z}  →  Z_eff={eff}  "
                    f"(ncore removed: {bare_z - int(eff)})"
                )
            else:
                print(f"    atom {i}:  Z={bare_z}  all-electron (no ECP)")
    else:
        is_pob = basis.name.lower().startswith("pob")
        if is_pob:
            print("  ECP attached:    NO — pob basis WITHOUT ECP blocks?")
            print("  ⚠  This may be an all-electron fallback — check warnings.")
        else:
            print(f"  ECP attached:    NO — all-electron (basis {basis.name!r})")
    print()


def _build_ag_fcc() -> vq.PeriodicSystem:
    lat = np.eye(3) * 4.062 * ANG
    a = 4.062 * ANG
    atoms = [
        vq.Atom(47, [0.0, 0.0, 0.0]),
        vq.Atom(47, [0.0, a / 2, a / 2]),
        vq.Atom(47, [a / 2, 0.0, a / 2]),
        vq.Atom(47, [a / 2, a / 2, 0.0]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _build_au_fcc() -> vq.PeriodicSystem:
    lat = np.eye(3) * 4.063 * ANG
    a = 4.063 * ANG
    atoms = [
        vq.Atom(79, [0.0, 0.0, 0.0]),
        vq.Atom(79, [0.0, a / 2, a / 2]),
        vq.Atom(79, [a / 2, 0.0, a / 2]),
        vq.Atom(79, [a / 2, a / 2, 0.0]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def _build_w_bcc() -> vq.PeriodicSystem:
    lat = np.eye(3) * 3.191 * ANG
    a = 3.191 * ANG
    atoms = [
        vq.Atom(74, [0.0, 0.0, 0.0]),
        vq.Atom(74, [a / 2, a / 2, a / 2]),
    ]
    return vq.PeriodicSystem(3, lat, atoms)


def main() -> None:
    vq.print_banner()

    # Ensure pob-TZVP-REV2 .g94 files are in the libint data path.
    from vibeqc.basis_crystal import fetch_bredow_basis_sets

    path_map = fetch_bredow_basis_sets(names=["pob-tzvp-rev2"], verbose=True)
    if not path_map:
        print("WARNING: could not fetch pob-TZVP-REV2; libint BasisSet load will fail.")

    for label, build_fn in [
        ("Ag fcc", _build_ag_fcc),
        ("Au fcc", _build_au_fcc),
        ("W bcc", _build_w_bcc),
    ]:
        print(f"  [{label}]")
        system = build_fn()
        basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
        from vibeqc.periodic_runner import _resolve_ecp_data

        ecp_blocks, ecp_centers, eff_z, total_ncore = _resolve_ecp_data(system, basis)
        opts = vq.PeriodicKSOptions()
        opts.functional = "pbe"
        if ecp_blocks:
            opts.ecp_primitive_blocks = ecp_blocks
            opts.ecp_home_centers = ecp_centers
            opts.ecp_effective_charges = eff_z
            opts.ecp_total_ncore = total_ncore
        print_ecp_diagnostics(system, basis, opts)


if __name__ == "__main__":
    main()
