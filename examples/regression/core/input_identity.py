"""Resolved physical-input identities for periodic comparisons (#648).

A shared basis or functional name, mesh and energy do not prove that two
codes solved the same Hamiltonian. Each runner records what its own code
actually built, reduced to convention-free forms that can be compared:

* geometry: lattice metric (Gram matrix, bohr^2), handedness, and atomic
  numbers with fractional positions. These are invariant to the Cartesian
  orientation each code chooses.
* basis: per atom, every contracted shell as (l, pure, exponents,
  contraction). The contraction is over normalized primitives, scaled to a
  unit-norm function, with the largest coefficient positive.
* ecp: core electrons removed and scalar semilocal Gaussian terms per atom,
  distinguishing the local potential from angular projectors.
* xc: exact-exchange fraction, range-separation omega, and the energy per
  volume and its rho/sigma derivatives on a fixed sample of
  (rho, sigma). Evaluation compares definitions, not aliases: B3LYP with
  VWN5 and with VWN-RPA differ at low density whatever they are called.

A field a code cannot resolve from its submitted input carries an
``unverified`` reason and never qualifies. Resolved domain, grid and electrostatic definitions are mandatory for the
compared-field qualification. Remaining conventions are listed in
``NOT_COMPARED``; even matched fields are not full physical equivalence.
"""
from __future__ import annotations

import math
import re

import numpy as np

FIELDS = ("geometry", "basis", "ecp", "xc", "domain", "integration", "electrostatics")

NOT_COMPARED = (
    "auxiliary (fitting) basis",
    "integral screening and lattice-sum cutoffs",
    "SCF convergence thresholds and occupation smearing",
    "spin-orbit ECP operators",
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Unpolarized sample: densities over five decades, each with a zero, a
# moderate and a large reduced gradient. Sent to every code, so a code
# evaluates its own resolved functional at identical points.
_XC_RHO = np.geomspace(1e-4, 10.0, 11)
XC_SAMPLE = {
    "rho": [float(r) for r in _XC_RHO for _ in range(3)],
    "sigma": [float(s * r ** (8.0 / 3.0)) for r in _XC_RHO for s in (0.0, 0.3, 4.0)],
}

GEOMETRY_TOL = {"metric_rel": 1e-6, "frac_abs": 1e-6}
BASIS_TOL = {"exponent_rel": 1e-10, "coefficient_abs": 1e-8}
XC_TOL = {"rel": 1e-9, "abs": 1e-12}
ECP_CONVENTION = "scalar semilocal; channel -1 local, l>=0 projector; c*r^power*exp(-alpha*r^2)"


def unverified(reason: str) -> dict:
    return {"unverified": reason}


# ----------------------------------------------------------------- geometry

def geometry_identity(lattice_rows_bohr, atoms) -> dict:
    """``atoms``: (Z, x, y, z) in bohr, in the code's own Cartesian frame."""
    lattice = np.asarray(lattice_rows_bohr, dtype=float)
    cart = np.asarray([a[1:] for a in atoms], dtype=float).reshape(-1, 3)
    frac = np.linalg.solve(lattice.T, cart.T).T
    frac -= np.floor(frac)
    return {
        "metric_bohr2": (lattice @ lattice.T).tolist(),
        "handedness": int(np.sign(np.linalg.det(lattice))),
        "atoms": [[int(a[0]), *map(float, f)] for a, f in zip(atoms, frac)],
    }


def _metric_from_parameters(a, b, c, alpha, beta, gamma):
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    return [[a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c]]


def crystal_scalar_ecp_input(block: str, atomic_number: int) -> dict:
    """Read a CRYSTAL23 scalar INPUT ECP (manual pp84–85, Eqs3.16–3.19).

    ZNUC is effective nuclear charge. M,M0,...,M4 count local and projected
    terms. ALFKL,CGKL,NKL are exponent, coefficient and actual radial power;
    unlike integral-library n values, NKL is NOT shifted by minus two.
    This is submitted operator evidence, not proof of what a remote run used.
    """
    rows = [line.split() for line in block.splitlines() if line.strip()]
    if not rows or rows[0] != ["INPUT"]:
        raise ValueError("only explicit scalar INPUT ECP is supported; library/SOC unresolved")
    if type(atomic_number) is not int or not 1 <= atomic_number <= 118:
        raise ValueError("invalid ECP atomic number")
    if len(rows) < 2 or len(rows[1]) != 7:
        raise ValueError("ECP requires ZNUC and six local/projector term counts")
    effective = float(rows[1][0])
    counts = [int(x) for x in rows[1][1:]]
    if (not math.isfinite(effective) or not effective.is_integer()
            or not 0 <= effective <= atomic_number or any(n < 0 for n in counts)
            or len(rows) != 2 + sum(counts)):
        raise ValueError("invalid ECP charge/counts or truncated/extra operator terms")
    terms, offset = [], 2
    for channel, count in zip(range(-1, 5), counts):
        for row in rows[offset:offset + count]:
            if len(row) != 3:
                raise ValueError("ECP primitive requires exponent, coefficient, radial power")
            exponent, coefficient = (float(v.replace("D", "E")) for v in row[:2])
            terms.append([channel, int(row[2]), exponent, coefficient])
        offset += count
    result = ecp_identity({"0": atomic_number - int(effective)}, {"0": terms},
                          "CRYSTAL23 manual pp84–85 INPUT Eqs3.16–3.19")
    result["evidence"] = "submitted"
    return result


def crystal_explicit_basis_ecp(lines: list[str], start: int, atoms: list) -> dict:
    """Extract explicit user-basis ECP requests; refuse ambiguous atom conventions."""
    by_number = {}
    try:
        while True:
            header = lines[start].split()
            if len(header) != 2:
                raise ValueError("expected atom/shell-count header")
            number, shells = map(int, header)
            start += 1
            if number == 99 and shells == 0:
                break
            if number in by_number or shells < 0 or not (1 <= number <= 118 or 201 <= number <= 298):
                raise ValueError("ambiguous/unsupported conventional atomic number")
            z = number - 200 if number > 200 else number
            operator = ecp_identity({"0": 0}, {"0": []})
            if number > 200:
                if lines[start] != "INPUT":
                    raise ValueError("library and spin-orbit ECP operators unresolved")
                counts = [int(v) for v in lines[start + 1].split()[1:]]
                end = start + 2 + sum(counts)
                operator = crystal_scalar_ecp_input("\n".join(lines[start:end]), z)
                start = end
            by_number[number] = operator
            for _ in range(shells):
                shell = lines[start].split()
                if len(shell) != 5 or int(shell[0]) != 0 or int(shell[2]) < 1:
                    raise ValueError("only explicit general-basis shell records supported")
                start += 1 + int(shell[2])
        cores, terms = {}, {}
        for index, atom in enumerate(atoms):
            number = atom[0]
            operator = by_number[number]
            cores[str(index)] = operator["core_electrons"]["0"]
            terms[str(index)] = operator["scalar_terms"]["0"]
        result = ecp_identity(cores, terms, "CRYSTAL explicit submitted INPUT basis deck")
        result["evidence"] = "submitted"
        return result
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return unverified(f"CRYSTAL explicit ECP request unresolved: {exc}")


_CONVENTION_KEYS = {
    "domain": {"dimension", "periodic_axes", "normalization", "charge_background"},
    "integration": {"points_sha256", "weights_sha256", "coordinates", "domain", "partition"},
    "electrostatics": {"coulomb_kernel", "exchange_kernel", "g0_coulomb", "g0_exchange", "boundary"},
}


def _qualify_convention(name: str, x: dict, y: dict) -> dict:
    """Only complete observed definitions qualify; equal requested aliases do not."""
    for record in (x, y):
        if record.get("evidence") != "resolved" or not record.get("source"):
            return {"status": "unverified", "reasons": [f"{name}: resolved evidence/source missing"]}
        definition = record.get("definition")
        if not isinstance(definition, dict) or set(definition) != _CONVENTION_KEYS[name]:
            return {"status": "unverified", "reasons": [f"{name}: incomplete convention definition"]}
        if name == "domain":
            axes = definition["periodic_axes"]
            valid = (type(definition["dimension"]) is int
                     and isinstance(axes, list) and len(axes) == 3
                     and all(type(a) is bool for a in axes)
                     and sum(axes) == definition["dimension"]
                     and definition["normalization"] in {"per_primitive_cell", "per_molecule"}
                     and definition["charge_background"] in {"none", "uniform_neutralizing"})
        else:
            valid = all(isinstance(v, str) and bool(v.strip()) for v in definition.values())
            if name == "integration":
                valid = valid and all(re.fullmatch(r"[0-9a-f]{64}", definition[k])
                                      for k in ("points_sha256", "weights_sha256"))
                valid = valid and definition["coordinates"] in {"cartesian_bohr", "fractional"}
        if not valid:
            return {"status": "unverified", "reasons": [f"{name}: invalid convention definition"]}
    if x["definition"] != y["definition"]:
        return {"status": "mismatch", "reasons": [f"resolved {name} definitions differ"]}
    return {"status": "matched", "source": [x["source"], y["source"]]}


def crystal_deck_identity(deck: str) -> dict:
    """What a P1 CRYSTAL deck from ``runner_crystal.build_d12`` specifies.

    The deck fixes the cell by a, b, c, alpha, beta, gamma, which define a
    right-handed cell in CRYSTAL's standard orientation; the basis and DFT
    functional are internal-library keywords whose definitions are not in
    the submitted input.
    """
    lines = [line.strip() for line in deck.splitlines()]
    start = lines.index("CRYSTAL")
    if lines[start + 2] != "1":
        raise ValueError("only P1 CRYSTAL decks are supported")
    a, b, c, alpha, beta, gamma = map(float, lines[start + 3].split())
    natoms = int(lines[start + 4])
    atoms = []
    conventional_atoms = []
    for line in lines[start + 5:start + 5 + natoms]:
        z, *frac = line.split()
        conventional_atoms.append([int(z), *map(float, frac)])
        atoms.append([int(z) % 100, *(float(f) % 1.0 for f in frac)])
    scale = ANGSTROM_TO_BOHR ** 2
    metric = [[x * scale for x in row]
              for row in _metric_from_parameters(a, b, c, alpha, beta, gamma)]
    basis_kw = lines[lines.index("BASISSET") + 1] if "BASISSET" in lines else None
    if "DFT" in lines:
        xc = crystal_documented_xc(lines[lines.index("DFT") + 1])
    else:
        xc = {"kind": "hf", "hf_fraction": 1.0, "omega": 0.0}
    library = (f"internal-library BASISSET {basis_kw}: primitives and any ECP "
               "are not in the submitted input" if basis_kw is not None else
               "explicit basis contractions are not resolved by the ECP input extractor")
    return {
        "geometry": {"metric_bohr2": metric, "handedness": 1, "atoms": atoms},
        "basis": unverified(library),
        "ecp": (unverified(library) if basis_kw is not None else
                crystal_explicit_basis_ecp(lines, start + 6 + natoms, conventional_atoms)),
        "xc": xc,
    }


_FLOAT = re.compile(r"[-+]?\d+\.\d+E[-+]\d+")
_SHELL_HEADER = re.compile(r"^\s*(\d+)(?:-\s*(\d+))?\s+(S|SP|P|D|F|G)\s*$")
_ATOM_HEADER = re.compile(r"^\s*(\d+)\s+([A-Z]{1,2})\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*$")
_SHELL_L = {"S": (0,), "SP": (0, 1), "P": (1,), "D": (2,), "F": (3,), "G": (4,)}
_COEF_COLUMN = {0: 0, 1: 1}  # S COEF, P COEF; every l >= 2 uses D/F/G COEF


def crystal_output_identity(text: str) -> dict:
    """What a CRYSTAL run reports it used, from its printed output.

    Geometry comes from DIRECT LATTICE VECTORS CARTESIAN COMPONENTS and
    CARTESIAN COORDINATES - PRIMITIVE CELL (12 digits). The basis comes from
    LOCAL ATOMIC FUNCTIONS BASIS SET, printed to four significant digits; it
    is recorded per element with that precision. No pseudopotential section
    means an all-electron run.
    """
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines)
                if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)" in line)
    lattice = [[float(v) for v in _FLOAT.findall(lines[head + 2 + r])] for r in range(3)]
    start = next(i for i, line in enumerate(lines)
                 if "CARTESIAN COORDINATES - PRIMITIVE CELL" in line) + 4
    atoms = []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) < 6 or not fields[0].isdigit():
            break
        atoms.append((int(fields[1]) % 100, *(float(v) * ANGSTROM_TO_BOHR for v in fields[3:6])))
    rows = np.asarray(lattice) * ANGSTROM_TO_BOHR

    number_to_z = {i + 1: a[0] for i, a in enumerate(atoms)}
    by_element: dict[str, list] = {}
    start = next(i for i, line in enumerate(lines)
                 if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line) + 4
    atom, shells, current = None, [], None
    for line in lines[start:]:
        if (m := _ATOM_HEADER.match(line)):
            atom = number_to_z[int(m.group(1))]
            continue
        if (m := _SHELL_HEADER.match(line)):
            first, last = int(m.group(1)), int(m.group(2) or m.group(1))
            current = {"atom": atom, "type": m.group(3), "n_ao": last - first + 1, "rows": []}
            shells.append(current)
            continue
        values = _FLOAT.findall(line)
        if len(values) == 4 and current is not None:
            current["rows"].append([float(v) for v in values])
            continue
        if line.strip():
            break
    for shell in shells:
        exps = [r[0] for r in shell["rows"]]
        for l in _SHELL_L[shell["type"]]:
            column = 1 + _COEF_COLUMN.get(l, 2)
            pure = shell["type"] not in ("S", "SP", "P") and shell["n_ao"] == 2 * l + 1
            by_element.setdefault(str(shell["atom"]), []).append(
                [l, pure, exps, canonical_contraction(l, exps, [r[column] for r in shell["rows"]])])
    for entries in by_element.values():
        entries.sort(key=lambda s: (s[0], [-e for e in s[2]], s[3]))
    pseudo = any("PSEUDOPOTENTIAL" in line and "OVP" not in line for line in lines)
    return {
        "geometry": geometry_identity(rows, atoms),
        "basis": {"shells_by_element": by_element,
                  "precision": "CRYSTAL printout, four significant digits"},
        "ecp": (unverified("pseudopotential printed; operator comparison not implemented")
                if pseudo else ecp_identity({str(i): 0 for i in range(len(atoms))},
                                           {str(i): [] for i in range(len(atoms))},
                                           "CRYSTAL all-electron output")),
    }


# -------------------------------------------------------------------- basis

def canonical_contraction(l: int, exponents, coefficients) -> list[float]:
    """Unit-norm contraction over normalized primitives, largest term positive."""
    alpha = np.asarray(exponents, dtype=float)
    d = np.asarray(coefficients, dtype=float)
    overlap = (2.0 * np.sqrt(np.outer(alpha, alpha))
               / np.add.outer(alpha, alpha)) ** (l + 1.5)
    d = d / math.sqrt(float(d @ overlap @ d))
    return (d * np.sign(d[np.argmax(np.abs(d))])).tolist()


def basis_identity(shells) -> dict:
    """``shells``: (atom, l, pure, exponents, coefficient columns), where each
    column holds one contraction over normalized primitives."""
    by_atom: dict[str, list] = {}
    for atom, l, pure, exponents, columns in shells:
        for column in columns:
            by_atom.setdefault(str(int(atom)), []).append([
                int(l), bool(pure), [float(e) for e in exponents],
                canonical_contraction(l, exponents, column)])
    for entries in by_atom.values():
        entries.sort(key=lambda s: (s[0], [-e for e in s[2]], s[3]))
    return {"shells_by_atom": by_atom}


# ----------------------------------------------------------------------- xc

def xc_identity_from_functional(functional, source: str | None = None) -> dict:
    """Evaluate a vibe-qc ``Functional`` on ``XC_SAMPLE`` (energy per volume)."""
    import vibeqc as vq

    rho = np.asarray(XC_SAMPLE["rho"]).reshape(-1, 1)
    if functional.kind == vq.XCKind.LDA:
        exc, vrho, vsigma = functional.eval_unpolarised(rho, np.zeros((0, 1)))
    elif functional.kind == vq.XCKind.GGA:
        sigma = np.asarray(XC_SAMPLE["sigma"]).reshape(-1, 1)
        exc, vrho, vsigma = functional.eval_unpolarised(rho, sigma)
    else:
        return unverified(f"{functional.kind} functional: tau is not sampled")
    record = xc_identity(kind=functional.kind.name.lower(),
                         hf_fraction=functional.hf_exchange_fraction,
                         omega=functional.rsh_omega, exc_per_volume=np.ravel(exc),
                         vrho=np.ravel(vrho), vsigma=np.ravel(vsigma))
    return {**record, "source": source} if source else record


# Standalone exchange-correlation keywords of a CRYSTAL DFT block, as libxc
# mixtures. CRYSTAL23 manual pp. 93, 134-135 (CRYSTAL14 p. 84): SVWN is
# Slater + VWN, and CRYSTAL's VWN is VWN5. B3LYP is EXCHANGE BECKE, CORRELAT
# LYP, HYBRID 20, NONLOCAL 0.9 0.81 on the VWN5 form, so
# E_xc = 0.8 E_x^LDA + 0.72 dE_x^B88 + 0.2 E_x^HF + 0.19 E_c^VWN5 + 0.81 E_c^LYP.
# A bare PBE is only an EXCHANGE/CORRELAT sub-keyword; full PBE is PBEXC.
CRYSTAL_DOCUMENTED_XC = {
    "SVWN": ([("LDA_X", 1.0), ("LDA_C_VWN", 1.0)], 0.0),
    "BLYP": ([("GGA_X_B88", 1.0), ("GGA_C_LYP", 1.0)], 0.0),
    "PBEXC": ([("GGA_X_PBE", 1.0), ("GGA_C_PBE", 1.0)], 0.0),
    "B3LYP": ([("LDA_X", 0.08), ("GGA_X_B88", 0.72),
               ("LDA_C_VWN", 0.19), ("GGA_C_LYP", 0.81)], 0.20),
}
CRYSTAL_XC_SOURCE = ("CRYSTAL23 manual pp. 134-135 documented composition, "
                     "evaluated with libxc; not observed in a CRYSTAL output")


def crystal_documented_xc(keyword: str) -> dict:
    import vibeqc as vq

    if keyword not in CRYSTAL_DOCUMENTED_XC:
        return unverified(f"DFT keyword {keyword}: not a documented standalone "
                          "exchange-correlation keyword of a CRYSTAL DFT block")
    components, hf_fraction = CRYSTAL_DOCUMENTED_XC[keyword]
    name = f"vibeqc-regression-crystal-{keyword.lower()}"
    vq.define_functional(name, components, hf_exchange_fraction=hf_fraction)
    return xc_identity_from_functional(vq.Functional(name), CRYSTAL_XC_SOURCE)


def from_raw(raw: dict) -> dict:
    """Reduce an external worker's raw record (lattice rows and Cartesian
    atoms in bohr, shells over normalized primitives) to the compared form."""
    return {
        "geometry": geometry_identity(raw["lattice_rows_bohr"], raw["atoms"]),
        "basis": basis_identity(raw["shells"]),
        "ecp": (ecp_identity(raw["ecp"]["core_electrons"], raw["ecp"]["scalar_terms"],
                             raw["ecp"].get("source", ""))
                if "scalar_terms" in raw["ecp"] else raw["ecp"]),
        "xc": raw["xc"],
        **{name: raw[name] for name in _CONVENTION_KEYS if name in raw},
    }


def xc_identity(*, kind: str, hf_fraction: float, omega: float,
                rho=None, exc_per_volume=None, vrho=None, vsigma=None) -> dict:
    record = {"kind": kind, "hf_fraction": float(hf_fraction), "omega": float(omega)}
    if kind != "hf":
        record.update(
            exc_per_volume=[float(x) for x in exc_per_volume],
            vrho=[float(x) for x in vrho],
            vsigma=None if vsigma is None or len(vsigma) == 0 else [float(x) for x in vsigma])
    return record


# --------------------------------------------------------------- comparison

def _close(a, b, rel, abs_=0.0) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= abs_ + rel * np.abs(b)))


def _compare_geometry(x, y) -> list[str]:
    problems = []
    scale = float(np.max(np.abs(y["metric_bohr2"])))
    if not _close(x["metric_bohr2"], y["metric_bohr2"], GEOMETRY_TOL["metric_rel"],
                  GEOMETRY_TOL["metric_rel"] * scale):
        problems.append("lattice metric differs")
    if x["handedness"] != y["handedness"]:
        problems.append("cell handedness differs")
    if [a[0] for a in x["atoms"]] != [a[0] for a in y["atoms"]]:
        problems.append("atomic numbers or order differ")
    else:
        d = np.asarray([a[1:] for a in x["atoms"]]) - np.asarray([a[1:] for a in y["atoms"]])
        if np.max(np.abs(d - np.round(d)), initial=0.0) > GEOMETRY_TOL["frac_abs"]:
            problems.append("fractional positions differ")
    return problems


def _compare_basis(x, y) -> list[str]:
    xs, ys = x["shells_by_atom"], y["shells_by_atom"]
    if set(xs) != set(ys):
        return ["basis covers different atoms"]
    problems = []
    for atom in sorted(xs, key=int):
        a, b = xs[atom], ys[atom]
        if [(s[0], s[1], len(s[2])) for s in a] != [(s[0], s[1], len(s[2])) for s in b]:
            problems.append(f"atom {atom}: shell structure differs")
        elif not all(_close(s[2], t[2], BASIS_TOL["exponent_rel"]) for s, t in zip(a, b)):
            problems.append(f"atom {atom}: primitive exponents differ")
        elif not all(_close(s[3], t[3], 0.0, BASIS_TOL["coefficient_abs"]) for s, t in zip(a, b)):
            problems.append(f"atom {atom}: contraction coefficients differ")
    return problems


def ecp_identity(core_electrons: dict, scalar_terms: dict, source: str = "") -> dict:
    """Canonical scalar ECP terms, with the actual radial power (not n+2).

    Shaw and Hill, JCP 147, 074108 (2017), Sec. I Eqs. (1)-(3),
    DOI:10.1063/1.4986887: local U_L plus projected U_l and effective charge.
    No radial normalization or fitting is applied. Only identical exponent,
    power and channel terms are combined; exact zero terms are removed.
    """
    if set(core_electrons) != set(scalar_terms):
        raise ValueError("ECP counts and operator terms must cover the same atoms")
    result = {}
    for atom, terms in scalar_terms.items():
        core = core_electrons[atom]
        if type(core) is not int or core < 0:
            raise ValueError("ECP core counts must be nonnegative integers")
        grouped = {}
        for channel, power, exponent, coefficient in terms:
            if (type(channel) is not int or channel < -1 or type(power) is not int
                    or power < -2 or not math.isfinite(exponent) or exponent <= 0
                    or not math.isfinite(coefficient)):
                raise ValueError("invalid scalar ECP primitive")
            grouped.setdefault((channel, power, float(exponent)), []).append(float(coefficient))
        result[atom] = [[*key, value] for key, values in sorted(grouped.items())
                        if (value := math.fsum(sorted(values))) != 0.0]
    return {"core_electrons": dict(core_electrons), "scalar_terms": result,
            "convention": ECP_CONVENTION, "source": source}


def _qualify_ecp(x, y) -> dict:
    # Retain definite count mismatches even for legacy records.
    xc, yc = x.get("core_electrons", x), y.get("core_electrons", y)
    if (all(type(v) is int for r in (xc, yc) for v in r.values()) and xc != yc):
        return {"status": "mismatch", "reasons": ["ECP core electrons differ"]}
    # Older count-only records cannot establish a zero-core model potential.
    if any("core_electrons" not in r or "scalar_terms" not in r for r in (x, y)):
        return {"status": "unverified", "reasons": ["ECP operator terms not recorded"]}
    if any(r.get("convention") != ECP_CONVENTION for r in (x, y)):
        return {"status": "unverified", "reasons": ["ECP operator convention not established"]}
    if x["core_electrons"] != y["core_electrons"]:
        return {"status": "mismatch", "reasons": ["ECP core electrons differ"]}
    xs, ys = x["scalar_terms"], y["scalar_terms"]
    if set(xs) != set(ys) or set(xs) != set(x["core_electrons"]):
        return {"status": "unverified", "reasons": ["ECP operator atom coverage is incomplete"]}
    problems = []
    for atom in sorted(xs, key=int):
        a, b = xs[atom], ys[atom]
        if [(t[0], t[1]) for t in a] != [(t[0], t[1]) for t in b]:
            problems.append(f"atom {atom}: ECP channels, radial powers or term counts differ")
        elif not _close([t[2:] for t in a], [t[2:] for t in b], 1e-12, 1e-12):
            problems.append(f"atom {atom}: ECP exponents or coefficients differ")
    return ({"status": "mismatch", "reasons": problems} if problems else
            {"status": "matched", "source": [r["source"] for r in (x, y) if r.get("source")]})


def _compare_xc(x, y) -> list[str]:
    if x["kind"] == "hf" or y["kind"] == "hf":
        return [] if x["kind"] == y["kind"] else ["Hartree-Fock against a density functional"]
    problems = []
    for key in ("hf_fraction", "omega"):
        if abs(x[key] - y[key]) > 1e-12:
            problems.append(f"{key} differs")
    for key in ("exc_per_volume", "vrho", "vsigma"):
        if (x[key] is None) != (y[key] is None) or (
                x[key] is not None and not _close(x[key], y[key], XC_TOL["rel"], XC_TOL["abs"])):
            problems.append(f"{key} differs on the sample")
    return problems


_COMPARE = {"geometry": _compare_geometry, "basis": _compare_basis, "xc": _compare_xc}

# Four printed significant digits round by at most 5e-4 relative.
PRINTED_TOL = {"exponent_rel": 6e-4, "coefficient_abs": 2e-3}


def _shells_by_element(record: dict):
    basis = record["basis"]
    if "shells_by_element" in basis:
        return basis["shells_by_element"], None
    geometry = record.get("geometry") or {}
    if "atoms" not in geometry:
        return None, "the element of each atom is not established"
    grouped: dict[str, list] = {}
    for index, atom in enumerate(geometry["atoms"]):
        shells = basis["shells_by_atom"].get(str(index), [])
        z = str(atom[0])
        if z in grouped and grouped[z] != shells:
            return None, f"atoms of element {z} carry different bases"
        grouped[z] = shells
    return grouped, None


def _compare_basis_by_element(x: dict, y: dict) -> list[str]:
    (xs, x_issue), (ys, y_issue) = _shells_by_element(x), _shells_by_element(y)
    if x_issue or y_issue:
        return [x_issue or y_issue]
    if set(xs) != set(ys):
        return ["basis covers different elements"]
    problems = []
    for z in sorted(xs, key=int):
        a, b = xs[z], ys[z]
        if [(s[0], s[1] if s[0] >= 2 else None, len(s[2])) for s in a] != \
                [(s[0], s[1] if s[0] >= 2 else None, len(s[2])) for s in b]:
            problems.append(f"element {z}: shell structure differs")
        elif not all(_close(s[2], t[2], PRINTED_TOL["exponent_rel"]) for s, t in zip(a, b)):
            problems.append(f"element {z}: primitive exponents differ at printed precision")
        elif not all(_close(s[3], t[3], 0.0, PRINTED_TOL["coefficient_abs"]) for s, t in zip(a, b)):
            problems.append(f"element {z}: contraction coefficients differ at printed precision")
    return problems


def qualify_pair(candidate: dict | None, reference: dict | None) -> dict:
    """Field-by-field qualification of one code's inputs against the reference."""
    fields = {}
    for name in FIELDS:
        x = (candidate or {}).get(name) or unverified("no identity recorded")
        y = (reference or {}).get(name) or unverified("no identity recorded")
        reasons = [r["unverified"] for r in (x, y) if "unverified" in r]
        if reasons:
            fields[name] = {"status": "unverified", "reasons": reasons}
        elif name in _CONVENTION_KEYS:
            fields[name] = _qualify_convention(name, x, y)
        elif name == "ecp":
            fields[name] = _qualify_ecp(x, y)
            if fields[name]["status"] == "matched" and any(
                    r.get("evidence") == "submitted" for r in (x, y)):
                fields[name] = {"status": "unverified", "reasons": [
                    "matching submitted ECP operators lack resolved execution evidence"]}
        elif name == "basis" and ("shells_by_element" in x or "shells_by_element" in y):
            problems = _compare_basis_by_element(candidate, reference)
            precision = x.get("precision") or y.get("precision")
            fields[name] = ({"status": "mismatch", "reasons": problems} if problems
                            else {"status": "matched", "precision": precision})
        else:
            problems = _COMPARE[name](x, y)
            if problems:
                fields[name] = {"status": "mismatch", "reasons": problems}
            else:
                fields[name] = {"status": "matched"}
                sources = [r["source"] for r in (x, y) if isinstance(r, dict) and r.get("source")]
                if sources:
                    fields[name]["source"] = sources
    statuses = {f["status"] for f in fields.values()}
    status = ("mismatch" if "mismatch" in statuses
              else "unverified" if "unverified" in statuses else "qualified")
    return {"status": status, "fields": fields, "not_compared": list(NOT_COMPARED),
            "full_physical_equivalence": False,
            "scope": "only the named resolved fields; remaining conventions are not equivalent"}


def qualify_case(identities: dict, reference_code: str) -> dict:
    """Qualification of every code against ``reference_code``."""
    reference = identities.get(reference_code)
    return {
        "reference_code": reference_code,
        "requested_vs_resolved": "every compared field is resolved from what "
                                 "each code built or was submitted; names are not trusted",
        "pairs": {code: qualify_pair(record, reference)
                  for code, record in sorted(identities.items()) if code != reference_code},
    }
