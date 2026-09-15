"""Translate the applied libecpint potential to TREXIO's ECP group.

TREXIO specification, ecp group: the stored radial power is n, whereas
libecpint/XML/Gaussian input stores n+2 (the radial volume factor).
"""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np


def ecp_fields(molecule, source, *, required=False):
    from ...ecp_metadata import effective_nuclear_charges_from

    atoms = list(molecule.atoms)
    coords = np.asarray([a.xyz for a in atoms], dtype=float)
    blocks = list(getattr(source, "ecp_primitive_blocks", None) or [])
    centers = list(getattr(source, "ecp_primitive_centers", None)
                   or getattr(source, "ecp_home_centers", None) or [])
    xml_centers = list(getattr(source, "ecp_xml_centers", None)
                       or getattr(source, "ecp_centers", None) or [])
    if not blocks and xml_centers:
        from ... import _VIBEQC_ECP_SHARE_DIR
        from .molden import _element

        library = (getattr(source, "ecp_xml_library", "")
                   or getattr(source, "ecp_library", "") or "ecp10mdf")
        # Read the same library selected by the SCF, never the basis sidecar.
        root = ET.parse(Path(_VIBEQC_ECP_SHARE_DIR) / "xml" / f"{library}.xml").getroot()
        for center in xml_centers:
            element = root.find(_element(center.Z).lower())
            if element is None:
                raise ValueError(f"TREXIO: applied ECP library has no Z={center.Z} record.")
            rows = [(int(shell.attrib["lval"]), p.attrib)
                    for shell in element.findall("Shell") for p in shell.findall("nxc")]
            blocks.append(SimpleNamespace(
                n_primitive=len(rows), ams=[l for l, _ in rows],
                ns=[int(p["n"]) for _, p in rows],
                exponents=[float(p["x"]) for _, p in rows],
                coefficients=[float(p["c"]) for _, p in rows],
            ))
            centers.append(center.xyz)
    if not blocks:
        if required or bool(getattr(source, "ecp_operator_applied", False)):
            raise ValueError("TREXIO: effective core potential parameters are missing from the result; "
                             "pass the applied ECP options as ecp_source.")
        return {}, np.asarray([a.Z for a in atoms], dtype=float)
    if len(blocks) != len(centers):
        raise ValueError("TREXIO: ECP blocks and centers have different lengths.")
    charges = effective_nuclear_charges_from(molecule, source)
    cores = np.asarray([a.Z for a in atoms]) - charges
    if not np.allclose(cores, np.rint(cores), rtol=0, atol=1e-10) or np.any(cores < 0):
        raise ValueError("TREXIO: ECP core counts must be nonnegative integers.")
    fields = {"ecp_z_core": np.rint(cores).astype(int),
              "ecp_max_ang_mom_plus_1": np.zeros(len(atoms), dtype=int)}
    rows, assigned = [], set()
    for center, block in zip(centers, blocks):
        matches = np.flatnonzero(np.all(np.isclose(coords, center, rtol=0, atol=1e-10), axis=1))
        if len(matches) != 1 or int(matches[0]) in assigned:
            raise ValueError("TREXIO: each ECP must match exactly one distinct nucleus.")
        atom = int(matches[0])
        assigned.add(atom)
        arrays = [block.ams, block.ns, block.exponents, block.coefficients]
        if block.n_primitive <= 0 or any(len(a) != block.n_primitive for a in arrays):
            raise ValueError("TREXIO: inconsistent ECP primitive dimensions.")
        fields["ecp_max_ang_mom_plus_1"][atom] = max(block.ams)
        for l, n, exponent, coefficient in zip(*arrays):
            if l < 0 or n < 0 or exponent <= 0:
                raise ValueError("TREXIO: invalid ECP angular momentum, power or exponent.")
            rows.append((atom, l, n - 2, exponent, coefficient))
    if any(cores[i] and i not in assigned for i in range(len(atoms))):
        raise ValueError("TREXIO: a replaced core has no ECP primitives.")
    fields["ecp_num"] = len(rows)
    for i, name in enumerate(("nucleus_index", "ang_mom", "power", "exponent", "coefficient")):
        fields["ecp_" + name] = np.asarray([r[i] for r in rows], dtype=int if i < 3 else float)
    return fields, charges


def ecp_options(fields, coords, options=None):
    """Restore inline ECP options, independent of any external ECP library."""
    from ..._vibeqc_core import ECPPrimitiveBlock, RHFOptions

    options = RHFOptions() if options is None else options
    if "ecp_num" not in fields:
        if np.any(fields.get("ecp_z_core", 0)) or any(
                key in fields for key in ("ecp_exponent", "ecp_coefficient", "ecp_ang_mom")):
            raise ValueError("TREXIO: incomplete ECP group: missing ecp_num.")
        return options
    required = {"ecp_nucleus_index", "ecp_ang_mom", "ecp_power", "ecp_exponent",
                "ecp_coefficient", "ecp_z_core", "ecp_max_ang_mom_plus_1", "nucleus_charge"}
    if not required <= fields.keys():
        raise ValueError(f"TREXIO: incomplete ECP group: {sorted(required - fields.keys())}")
    atom_index = np.asarray(fields["ecp_nucleus_index"])
    if (np.any(np.asarray(fields["ecp_z_core"]) < 0)
            or np.any(np.asarray(fields["ecp_exponent"]) <= 0)
            or np.any(np.asarray(fields["ecp_ang_mom"]) < 0)
            or any(not np.all(np.isfinite(fields[name])) for name in required)):
        raise ValueError("TREXIO: invalid ECP parameters.")
    if any(core and atom not in atom_index for atom, core in enumerate(fields["ecp_z_core"])):
        raise ValueError("TREXIO: a replaced core has no ECP primitives.")
    blocks, centers = [], []
    for atom in sorted(set(atom_index.tolist())):
        if not 0 <= atom < len(coords):
            raise ValueError("TREXIO: ECP nucleus index out of range.")
        sel = atom_index == atom
        block = ECPPrimitiveBlock()
        block.n_primitive = int(np.count_nonzero(sel))
        block.ams = np.asarray(fields["ecp_ang_mom"])[sel].tolist()
        if max(block.ams) != fields["ecp_max_ang_mom_plus_1"][atom]:
            raise ValueError("TREXIO: ECP local channel is not the largest angular momentum.")
        block.ns = (np.asarray(fields["ecp_power"])[sel] + 2).tolist()
        if min(block.ns) < 0:
            raise ValueError("TREXIO: libecpint cannot evaluate ECP powers below -2.")
        block.exponents = np.asarray(fields["ecp_exponent"])[sel].tolist()
        block.coefficients = np.asarray(fields["ecp_coefficient"])[sel].tolist()
        blocks.append(block)
        centers.append(np.asarray(coords[atom]).tolist())
    if hasattr(options, "ecp_centers"):
        options.ecp_centers = []
    options.ecp_primitive_blocks = blocks
    if hasattr(options, "ecp_home_centers"):
        options.ecp_home_centers = centers
    else:
        options.ecp_primitive_centers = centers
    options.ecp_effective_charges = np.asarray(fields["nucleus_charge"]).tolist()
    options.ecp_total_ncore = int(np.sum(fields["ecp_z_core"]))
    return options
