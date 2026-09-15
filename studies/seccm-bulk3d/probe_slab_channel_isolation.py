"""Channel isolation on a crystallographically valid B1 MgO(100) slab.

Each (100) layer is a neutral Mg/O checkerboard, adjacent layers swap the
cation and anion sites, and the corresponding 3-D extension has six unlike
nearest neighbours at ``a / 2``. This replaces the historical IID 141
single-species-plane fixture. Results from that retired synthetic fixture
remain synthetic observations and must not be reinterpreted as MgO surface
claims.

The probe isolates the WS-folded AES channel and the GAM3 parameter
contribution. Its corrected geometry does not by itself establish a
surface-chemistry result.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.semiempirical.methods.gfn2_params import (
    _compute_pairwise_repulsive,
    _read_cached_toml,
)
from vibeqc.semiempirical.seccm._adapter_common import (
    flatten_topology_records,
    topology_length_unit_scale,
)
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def _params_with(gam3_scale: float) -> _se_cxx.xtb.GFN2ParameterSet:
    data = _read_cached_toml()
    params = _se_cxx.xtb.GFN2ParameterSet()
    rep_params: dict[int, tuple[float, float]] = {}
    for elem_blob in data.get("element", []):
        Z = int(elem_blob["Z"])
        ed = _se_cxx.xtb.GFN2ElementData()
        ed.Z = Z
        ed.gam = float(elem_blob.get("gam", 0.5))
        ed.gam3 = float(elem_blob.get("gam3", 0.0)) * gam3_scale
        ed.alpha = float(elem_blob.get("alpha", 1.0))
        ed.dpol = float(elem_blob.get("dpol", 0.0))
        ed.qpol = float(elem_blob.get("qpol", 0.0))
        ed.mp_rad = float(elem_blob.get("mp_rad", 0.0))
        ed.mp_vcn = float(elem_blob.get("mp_vcn", 0.0))
        for sh in elem_blob.get("shells", []):
            ed.add_shell(
                int(sh.get("l", 0)),
                float(sh.get("en", 0.0)),
                float(sh.get("zeta", 1.0)),
                float(sh.get("k_en", 1.0)),
                float(sh.get("kcn", 0.0)),
                float(sh.get("poly", 0.0)),
                int(sh.get("n", 0)),
            )
        params.add_element(ed)
        rep_params[Z] = (
            float(elem_blob.get("repa", 0.0)),
            float(elem_blob.get("repb", 0.0)),
        )
    _compute_pairwise_repulsive(params, rep_params)
    return params


def _rocksalt_100_slab(layers: int):
    """Build neutral checkerboard planes in a 2x2 B1 MgO(100) cell."""
    if layers < 1:
        raise ValueError("layers must be positive")
    a = 4.212
    t1 = np.array([a / 2.0, a / 2.0, 0.0])
    t2 = np.array([-a / 2.0, a / 2.0, 0.0])
    unlike_offset = np.array([a / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(layers):
        z_offset = np.array([0.0, 0.0, k * a / 2.0])
        for i in range(2):
            for j in range(2):
                home = i * t1 + j * t2 + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend((12, 8))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend((12, 8))
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    translations = [2.0 * t1, 2.0 * t2]
    primitives = [t1, t2]
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in primitives],
        replicas=(2, 2, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return molecule, topology


def _run_native(molecule, topology, params, include_aes, max_iter=1200,
                temperature=0.0):
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = flatten_topology_records(molecule, topology, route_name="GFN2-SECCM")
    group = topology.finite_group
    native = _se_cxx._run_gfn2_seccm_from_records(
        molecule,
        params,
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        max_iter=int(max_iter),
        conv_tol_charge=1.0e-6,
        charge_mixing=0.1,
        electronic_temperature=temperature,
        madelung=False,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma=True,
        include_aes=include_aes,
    )
    return native


def _report(label, native):
    if bool(native.converged) and bool(native.physical_basin):
        q = np.asarray(native.charges)
        energy_per_atom = (
            float(native.energy) * int(native.group_order) / q.size
        )
        print(f"{label}: CONVERGED iter={native.n_iter} "
              f"E/atom={energy_per_atom:12.6f} "
              f"q_rms={np.sqrt((q**2).mean()):6.3f}")
    else:
        trace = np.asarray(native.scc_max_change_trace)
        tail = trace[-24:] if trace.size else trace
        print(f"{label}: not converged (n_iter={native.n_iter}, "
              f"physical={bool(native.physical_basin)})")
        print(f"        max_change tail (last 24 of {trace.size}): "
              f"{np.array2string(tail, precision=3)}")


def main() -> None:
    molecule, topology = _rocksalt_100_slab(4)
    params_full = _params_with(1.0)
    params_nogam3 = _params_with(0.0)

    print("== 4-layer B1 MgO(100) channel fixture, ewald_gamma=True ==")
    _report("aes on,  gam3 on", _run_native(molecule, topology, params_full, True))
    _report("aes off, gam3 on", _run_native(molecule, topology, params_full, False))
    _report("aes on,  gam3 off", _run_native(molecule, topology, params_nogam3, True))
    _report("aes off, gam3 off", _run_native(molecule, topology, params_nogam3, False))

    print("== frontier-occupation sweep (T = 0.005 Ha) ==")
    for label, params, aes in (
        ("aes on,  gam3 on", params_full, True),
        ("aes off, gam3 on", params_full, False),
        ("aes on,  gam3 off", params_nogam3, True),
        ("aes off, gam3 off", params_nogam3, False),
    ):
        _report(label, _run_native(molecule, topology, params, aes,
                                   max_iter=3600, temperature=0.005))

    # Public-API cross-check for the aes-off path (2-layer sanity first).
    molecule2, topology2 = _rocksalt_100_slab(2)
    r = run_gfn2_seccm(molecule2, topology2, ewald_gamma=True, include_aes=False)
    n_atoms = len(molecule2.atoms)
    print(f"2-layer sanity (aes off, public API): converged={r.converged} "
          f"E/atom={r.energy * r.group_order / n_atoms:12.6f}")


if __name__ == "__main__":
    main()
