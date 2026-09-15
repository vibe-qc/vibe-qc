"""Optional external naming backends.

Provides fallback name resolution via:
- RDKit (local, if installed): ``Chem.MolToIUPACName(mol)``
- CACTUS REST API (NCI/CADD): formula/SMILES → IUPAC name
- PubChem REST API: formula → CID → IUPAC name

These are optional — all imports and network calls are wrapped in
try/except so that the built-in naming engine is always available.

References
----------
- RDKit: https://www.rdkit.org/docs/IUPAC.html
- CACTUS (NCI): https://cactus.nci.nih.gov/
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)
# ── In-memory result cache (256-entry LRU) ────────────────────────────
_CACHE: dict[str, str] = {}

def _cached_lookup(key: str, fetcher) -> Optional[str]:
    """Cache-aware lookup: skip network if result already known."""
    if key in _CACHE:
        return _CACHE[key]
    result = fetcher(key)
    if result:
        _CACHE[key] = result
    return result


def _pubchem_smiles_uncached(smiles: str) -> Optional[str]:
    """Get IUPAC name from PubChem using SMILES as input.

    The SMILES → CID → IUPAC path is more accurate than formula lookup
    because SMILES uniquely identifies the compound.
    """
    try:
        import urllib.request
        import urllib.parse
        encoded = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/IUPACName/json"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        name = props.get("IUPACName")
        if name and name != "NaN":
            return name
    except Exception as exc:
        logger.debug("PubChem SMILES naming failed: %s", exc)
    return None


def _cactus_smiles_uncached(smiles: str) -> Optional[str]:
    """Get IUPAC name from NCI CACTUS using SMILES."""
    try:
        import urllib.request
        import urllib.parse
        encoded = urllib.parse.quote(smiles)
        url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/iupac_name"
        req = urllib.request.Request(url)
        req.add_header("Accept", "text/plain")
        with urllib.request.urlopen(req, timeout=15) as resp:
            name = resp.read().decode("utf-8").strip()
            if name and name != "NaN":
                return name
    except Exception as exc:
        logger.debug("CACTUS SMILES naming failed: %s", exc)
    return None


def name_from_pubchem_smiles(smiles: str) -> Optional[str]:
    return _cached_lookup(smiles, _pubchem_smiles_uncached)


def name_from_cactus_smiles(smiles: str) -> Optional[str]:
    return _cached_lookup(smiles, _cactus_smiles_uncached)


def name_from_smiles_external(smiles: str) -> Optional[tuple[str, str]]:
    """Get IUPAC name from a SMILES string using REST APIs.

    Returns (name, source) or None.
    """
    name = name_from_cactus_smiles(smiles)
    if name:
        return (name, "cactus")
    name = name_from_pubchem_smiles(smiles)
    if name:
        return (name, "pubchem")
    return None



def _try_import_rdkit():
    """Import RDKit, return None if unavailable."""
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import rdFreeSASA  # noqa: F401

        return True
    except ImportError:
        return False


def name_from_rdkit(atoms: list[tuple[int, float, float, float]]) -> Optional[str]:
    """Try to get an IUPAC name from RDKit.

    Parameters
    ----------
    atoms :
        List of (Z, x_ang, y_ang, z_ang) tuples.

    Returns
    -------
    IUPAC name string or None if RDKit is not installed / fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcFormula

        # Build an RDKit molecule from atomic numbers and coordinates
        conf = Chem.Conformer(len(atoms))
        for i, (z, x, y, z_coord) in enumerate(atoms):
            conf.SetAtomPosition(i, (x, y, z_coord))

        mol = Chem.RWMol()
        atom_ids = []
        for i, (z, *_rest) in enumerate(atoms):
            a = Chem.Atom(z)
            aid = mol.AddAtom(a)
            atom_ids.append(aid)
            conf.SetAtomPosition(i, (x, y, z_coord))

        mol.AddConformer(conf)

        # Try to sanitize and add hydrogens (required for IUPAC naming)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        name = Chem.MolToIUPACName(mol)
        if name:
            return name
    except Exception as exc:  # noqa: BLE001
        logger.debug("RDKit naming failed: %s", exc)

    return None


def name_from_pubchem(formula: str) -> Optional[str]:
    """Try to get an IUPAC name from PubChem via REST API.

    Parameters
    ----------
    formula :
        Hill-system formula (e.g. "C2H6O").

    Returns
    -------
    IUPAC name string or None if the lookup fails.
    """
    try:
        import urllib.request

        # First, get the CID from the formula
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/CID/json"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        cids = data.get("PropertyList", {}).get("CID", [])
        if not cids:
            return None

        # Get the canonical IUPAC name for the first match
        cid = cids[0]
        url2 = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            f"cid/{cid}/property/IUPACName,IsCanonicallyNamed/json"
        )
        req2 = urllib.request.Request(url2)
        req2.add_header("Accept", "application/json")
        with urllib.request.urlopen(req2, timeout=10) as resp:
            data2 = json.loads(resp.read())

        props = data2.get("PropertyTable", {}).get("Properties", [{}])[0]
        iupac_name = props.get("IUPACName")
        if iupac_name and iupac_name != "NaN":
            return iupac_name
    except Exception as exc:  # noqa: BLE001
        logger.debug("PubChem naming failed: %s", exc)

    return None


def name_from_cactus(formula: str, smiles: Optional[str] = None) -> Optional[str]:
    """Try to get an IUPAC name from NCI CACTUS REST API.

    Parameters
    ----------
    formula :
        Hill-system formula (fallback).
    smiles :
        SMILES string (preferred over formula for accuracy).

    Returns
    -------
    IUPAC name string or None if the lookup fails.
    """
    try:
        import urllib.request

        # CACTUS accepts both formula and SMILES
        if smiles:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"
        else:
            url = f"https://cactus.nci.nih.gov/chemical/structure/{formula}/iupac_name"

        req = urllib.request.Request(url)
        req.add_header("Accept", "text/plain")
        with urllib.request.urlopen(req, timeout=15) as resp:
            name = resp.read().decode("utf-8").strip()
            if name and name != "NaN":
                return name
    except Exception as exc:  # noqa: BLE001
        logger.debug("CACTUS naming failed: %s", exc)

    return None


def name_from_formula_external(formula: str) -> Optional[tuple[str, str]]:
    """Try multiple external backends for formula-to-name resolution.

    Returns (name, source) or None if all backends fail.

    Parameters
    ----------
    formula :
        Hill-system formula.

    Returns
    -------
    Tuple of (name, source_tag) or None.
    """
    # Try CACTUS first (fastest REST API)
    name = name_from_cactus(formula)
    if name:
        return (name, "cactus")

    # Try PubChem
    name = name_from_pubchem(formula)
    if name:
        return (name, "pubchem")

    return None


def name_from_atoms_external(
    atoms: list[tuple[int, float, float, float]],
) -> Optional[tuple[str, str]]:
    """Try external backends for atom-coordinate-to-name resolution.

    Tries RDKit first (fastest), then CACTUS/PubChem via formula conversion.

    Parameters
    ----------
    atoms :
        List of (Z, x_ang, y_ang, z_ang) tuples.

    Returns
    -------
    Tuple of (name, source_tag) or None if all backends fail.
    """
    # Try RDKit first
    name = name_from_rdkit(atoms)
    if name:
        return (name, "rdkit")

    # Fallback: convert to formula and use REST APIs
    from collections import Counter

    from .molecular_graph import ATOMIC_NUMBER_TO_SYMBOL

    counts = Counter(a[0] for a in atoms)
    parts: list[str] = []
    if 6 in counts:
        c_count = counts[6]
        h_count = counts.get(1, 0)
        if c_count > 0:
            parts.append(f"C{c_count}" if c_count > 1 else "C")
            if h_count > 0:
                parts.append(f"H{h_count}" if h_count > 1 else "H")
                del counts[1]
    for sym, count in sorted(counts.items()):
        if sym not in (6, 1):  # Already handled
            s = ATOMIC_NUMBER_TO_SYMBOL.get(sym, f"X{sym}")
            parts.append(f"{s}{count}" if count > 1 else s)

    formula = "".join(parts)
    return name_from_formula_external(formula)
