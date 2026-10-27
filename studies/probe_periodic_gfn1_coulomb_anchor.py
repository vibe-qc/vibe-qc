"""External anchor for the shared periodic shell-gamma kernel (#245).

Compares ``build_periodic_shell_gamma`` (``cpp/src/semiempirical/core/
periodic_gamma.cpp``) with the periodic GFN1-xTB Coulomb matrix of xtb 6.7
(``src/coulomb/klopmanohno.F90``, ``getCoulombMatrixPBC3D``) on the same
cell, the same shell hardnesses and the same real-space truncation.  xtb's
periodic GFN1 kernel is the Ewald-split Klopman-Ohno construction with the
harmonic hardness average (``KlopmanOhnoAverage.InverseHardnessMean``) and
``gExp = 2``, i.e. the shared kernel with ``ShellGammaForm.KlopmanOhno``.

What xtb sums, per shell pair (i on atom a, j on atom b), is

    J_ij = sum_img w_img [ Phi_recip(vec_img) - pi / (V alpha^2) ]
         + sum_img w_img sum_T [ KO_ij(|vec_img + T|) - erf(alpha |vec_img + T|) / |vec_img + T| ]
         + delta_ab onsite_ij

over the Wigner-Seitz images ``img`` of b around a (weights ``w`` share
boundary ties) and the real-space translations ``T`` with ``|T| <= r_cut``.
The shared kernel sums

    Gamma_ij = Phi_Ewald(R_b - R_a) + sum_records w [ KO_ij(r) - 1 / r ] + delta_ab onsite_ij

with a converged Ewald potential.  Feeding it the record set
``{(a, b, vec_img + T, w_img)}`` makes the Klopman-Ohno remainders identical
by construction; the two then differ only by xtb's real-space erfc sum,
which runs over that same weighted set instead of the whole lattice and so
under-weights lattice images of pairs with tied Wigner-Seitz images.  That
difference is an elementary function of the record set (``erfc_deficit``
below) and is subtracted before comparing.

Inputs (all produced out of process; xtb is never imported):

* an xtb GFN1 run directory: ``coord`` (TURBOMOLE format with ``$periodic 3``
  and ``$lattice bohr``), ``xtb.out`` and the binary ``xtbrestart``, which
  carries the converged shell charges (Fortran sequential: a 48-byte record
  of six int64 ``iver, idum, n_atoms, n_shells, n_electrons, n_open``, then
  one record of ``n_shells`` doubles, each record framed by 4-byte markers);
* optionally ``coulomb.dump``, written by an xtb built with a small local
  patch: ``setupBoundaryConditions`` (``src/type/coulomb.f90``) reads the
  Ewald tolerance from ``XTB_EWALD_TOLERANCE`` when set, and
  ``getCoulombMatrix`` (``src/coulomb/klopmanohno.F90``) writes, when
  ``XTB_DUMP_COULOMB`` names a file, ``natom``, ``nshell``, ``alpha``,
  ``volume``, ``tolerance``, ``rcutoff``, ``gcutoff``, the real and
  reciprocal translation lists, the wrapped ``mol%xyz``, the shell index
  table, the per-shell hardnesses, the Wigner-Seitz neighbour lists (image
  atom, weight, image coordinates) and ``jmat``.  With the dump the
  comparison is element-wise; without it only the contracted energy
  ``0.5 qsh^T J qsh = (printed "-> electrostatic") - E_3rd`` is available,
  where ``E_3rd = sum_A Gamma_A q_A^3 / 3`` with ``Gamma_A`` = 0.1 times the
  ``thirdOrderAtom`` table (the parameter-file convention xtb applies).

The GFN1 hardness tables are parsed from xtb's ``src/xtb/gfn1.f90`` when
``XTB_SOURCE_DIR`` is set; otherwise the Mg/O values verified against the
dump on 2026-09-24 are used.

Usage::

    python studies/probe_periodic_gfn1_coulomb_anchor.py RUN_DIR [--tol 1e-8]

Numbers of record (2026-09-24, 64-atom conventional MgO cell, a = 4.212 A,
xtb 6.7.0 with the patch above, vibe-qc main e659504): at xtb's Ewald
tolerance 1e-14 the kernel reproduces xtb's Coulomb matrix element-wise to
2.5e-13 Ha and the second-order energy to 9e-14 Eh after the erfc-deficit
correction (raw: 7.9e-8 per element, 2.3e-6 Eh); at xtb's default tolerance
1e-8 the raw discrepancy is 5.9e-3 per element and 1.1e-2 Eh, all of it the
deficit (corrected: 3.1e-7 per element, 2.6e-10 Eh).  xtb's own periodic
GFN1 electrostatic energy moves by 6.1e-2 Eh between the two tolerances on
this cell (4.0211 vs 3.9602 Eh, total energy 0.39 Eh), because the
Klopman-Ohno remainder is real-space truncated at ``r_cut`` (the former-tracker issue 211
thermodynamic-limit statement) and because of the tie under-weighting.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import struct
import sys

import numpy as np
from scipy.special import erfc

from vibeqc._vibeqc_core import semiempirical as _se

# GFN1-xTB Coulomb parameters for Mg and O (xtb src/xtb/gfn1.f90:
# chemicalHardness, shellHardness(l), thirdOrderAtom), verified against the
# per-shell hardnesses xtb writes to the dump (max difference 0.0).
_GFN1_FALLBACK = {
    8: dict(hardness=0.583349, shell=(0.0, 0.0374608), third=-0.005102),
    12: dict(hardness=0.354151, shell=(0.0, 0.0674819), third=1.1),
}
_SYMBOL_TO_Z = {"mg": 12, "o": 8, "c": 6, "h": 1, "li": 3, "f": 9, "na": 11, "cl": 17}

N_SEARCH_ITER = 30
TOL_SAME_DIST2 = 1.0e-10
WS_TOL2 = 0.01 ** 2
MIN_NEIGH_DIST2 = 0.01 ** 2
PBC_EPS = 1.0e-14


# ---------------------------------------------------------------------------
# xtb parameter tables
# ---------------------------------------------------------------------------

def _fortran_array(name, text, kind=float):
    match = re.search(name + r"\([^)]*\)\s*=\s*(?:reshape\()?\[(.*?)\]", text, re.S)
    body = match.group(1).replace("&", "").replace("_wp", "").replace("\n", " ")
    return np.array([kind(v.strip()) for v in body.split(",") if v.strip()])


def gfn1_shell_hardness_and_third(Z: int):
    """Per-shell hardnesses U_l = U_Z (1 + kappa_l) and Gamma_A (raw table)."""
    source = os.environ.get("XTB_SOURCE_DIR")
    if source:
        text = open(os.path.join(source, "src", "xtb", "gfn1.f90")).read()
        hardness = _fortran_array("chemicalHardness", text)
        third = _fortran_array("thirdOrderAtom", text)
        shell_h = _fortran_array("shellHardness", text).reshape(-1, 3)
        nshell = _fortran_array("nShell", text, int)
        angshell = _fortran_array("angShell", text, int).reshape(-1, 3)
        shells = [
            hardness[Z - 1] * (1.0 + shell_h[Z - 1, angshell[Z - 1, ish]])
            for ish in range(nshell[Z - 1])
        ]
        return shells, float(third[Z - 1])
    row = _GFN1_FALLBACK[Z]
    return [row["hardness"] * (1.0 + k) for k in row["shell"]], row["third"]


# ---------------------------------------------------------------------------
# xtb run artefacts
# ---------------------------------------------------------------------------

def read_coord(path):
    coords, Z, lattice, mode = [], [], [], None
    for line in open(path):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "$coord":
            mode = "c"
            continue
        if tokens[0] == "$lattice":
            mode = "l"
            continue
        if tokens[0].startswith("$"):
            mode = None
            continue
        if mode == "c":
            coords.append([float(x) for x in tokens[:3]])
            Z.append(_SYMBOL_TO_Z[tokens[3].lower()])
        elif mode == "l":
            lattice.append([float(x) for x in tokens[:3]])
    # columns are the lattice vectors, the xtb convention
    return np.array(coords), Z, np.array(lattice).T


def read_xtbrestart(path):
    raw = open(path, "rb").read()
    pos = 0

    def record():
        nonlocal pos
        (n,) = struct.unpack("<i", raw[pos:pos + 4])
        pos += 4
        data = raw[pos:pos + n]
        pos += n
        (n2,) = struct.unpack("<i", raw[pos:pos + 4])
        pos += 4
        if n != n2:
            raise ValueError("xtbrestart record markers disagree")
        return data

    iver, _idum, nat, nsh, nel, nopen = struct.unpack("<6q", record())
    qsh = np.frombuffer(record(), dtype="<f8")
    if len(qsh) != nsh:
        raise ValueError("xtbrestart shell-charge record has the wrong length")
    return dict(iver=iver, nat=nat, nsh=nsh, nel=nel, nopen=nopen, qsh=qsh)


def read_dump(path):
    lines = iter(open(path).read().split("\n"))
    out = {}
    out["natom"] = int(next(lines).split()[1])
    out["nshell"] = int(next(lines).split()[1])
    for key in ("alpha", "volume", "tolerance", "rcutoff", "gcutoff"):
        out[key] = float(next(lines).split()[1])
    n = int(next(lines).split()[1])
    out["rtrans"] = np.array([[float(x) for x in next(lines).split()] for _ in range(n)])
    n = int(next(lines).split()[1])
    out["gtrans"] = np.array([[float(x) for x in next(lines).split()] for _ in range(n)])
    assert next(lines).strip() == "xyz"
    out["xyz"] = np.array([[float(x) for x in next(lines).split()] for _ in range(out["natom"])])
    assert next(lines).strip() == "itbl"
    for _ in range(out["natom"]):
        next(lines)
    assert next(lines).strip() == "hardness"
    hardness = {}
    while True:
        line = next(lines)
        if line.strip() == "wsc":
            break
        a, s, h = line.split()
        hardness[(int(a) - 1, int(s) - 1)] = float(h)
    out["hardness"] = hardness
    wsc = {}
    for _ in range(out["natom"]):
        iat, count = (int(x) for x in next(lines).split())
        entries = []
        for _ in range(count):
            t = next(lines).split()
            entries.append((int(t[0]) - 1, float(t[1]), np.array([float(x) for x in t[2:5]])))
        wsc[iat - 1] = entries
    out["wsc"] = wsc
    assert next(lines).strip() == "jmat"
    ns = out["nshell"]
    J = np.zeros((ns, ns))
    for ish in range(ns):
        for jsh in range(ns):
            J[jsh, ish] = float(next(lines))
    out["J"] = J
    return out


# ---------------------------------------------------------------------------
# Python replica of xtb's construction (validated against the dump: same
# alpha, r_cut, translation count and Wigner-Seitz lists on MgO 2x2x2)
# ---------------------------------------------------------------------------

def wrap_like_xtb(coords, lattice):
    """xtb ``wrap_back``: fractional coordinates into [0, 1) with a 1e-14 eps."""
    frac = np.linalg.solve(lattice, coords.T).T
    wrapped = frac.copy()
    for value in np.nditer(wrapped, op_flags=["readwrite"]):
        v = float(value)
        out = v
        if v < -PBC_EPS:
            out = v + math.ceil(-v)
        if v > 1.0 + PBC_EPS:
            out = v - math.floor(v)
        if abs(v - 1.0) < PBC_EPS:
            out = v - 1.0
        value[...] = out
    return (lattice @ wrapped.T).T


def _r_term(r, alpha):
    return erfc(alpha * r) / r


def _g_term(g, alpha, volume):
    return 4.0 * math.pi * (math.exp(-0.25 * g * g / (alpha * alpha)) / (volume * g * g))


def optimal_alpha(lattice, tol):
    volume = abs(np.linalg.det(lattice))
    reciprocal = 2.0 * math.pi * np.linalg.inv(lattice).T
    min_g = math.sqrt(min(np.sum(reciprocal ** 2, axis=0)))
    min_r = math.sqrt(min(np.sum(lattice ** 2, axis=0)))

    def diff(alpha):
        return ((_g_term(4.0 * min_g, alpha, volume) - _g_term(5.0 * min_g, alpha, volume))
                - (_r_term(2.0 * min_r, alpha) - _r_term(3.0 * min_r, alpha)))

    alpha = 1.0e-8
    value = diff(alpha)
    while value < -tol:
        alpha *= 2.0
        value = diff(alpha)
    left = 0.5 * alpha
    while value < tol:
        alpha *= 2.0
        value = diff(alpha)
    right = alpha
    alpha = 0.5 * (left + right)
    value = diff(alpha)
    iteration = 0
    while abs(value) > tol and iteration <= N_SEARCH_ITER:
        if value < 0:
            left = alpha
        else:
            right = alpha
        alpha = 0.5 * (left + right)
        value = diff(alpha)
        iteration += 1
    return alpha


def _max_x(func, tol):
    x = 1.0e-8
    y = func(x)
    while y > tol:
        x *= 2.0
        y = func(x)
    x_left, y_left = 0.5 * x, func(0.5 * x)
    x_right, y_right = x, y
    iteration = 1
    while y_left - y_right > tol and iteration <= N_SEARCH_ITER:
        x = 0.5 * (x_left + x_right)
        y = func(x)
        if y >= tol:
            x_left, y_left = x, y
        else:
            x_right, y_right = x, y
        iteration += 1
    return x


def real_space_cutoff(alpha, tol):
    return _max_x(lambda r: _r_term(r, alpha), tol)


def lattice_points(lattice, cutoff):
    normals = [np.cross(lattice[:, 1], lattice[:, 2]), np.cross(lattice[:, 2], lattice[:, 0]),
               np.cross(lattice[:, 0], lattice[:, 1])]
    tmax = []
    for k, normal in enumerate(normals):
        normal = normal / np.linalg.norm(normal)
        tmax.append(math.ceil(abs(cutoff / float(normal @ lattice[:, k]))))
    points = []
    for n in itertools.product(*[range(-t, t + 1) for t in tmax]):
        T = lattice @ np.asarray(n, float)
        d2 = float(T @ T)
        if d2 <= cutoff ** 2 + TOL_SAME_DIST2:
            points.append((d2, T))
    points.sort(key=lambda p: p[0])
    return np.array([p[1] for p in points])


def wigner_seitz(coords, r_trans, cutoff):
    """``TWignerSeitzCell%generate`` with ``symmetric = .false.``.

    Returns, per list owner ``a``, the entries ``(b, weight, image_xyz)`` with
    ``b >= a``: the closest images of b around a within ``cutoff`` (ties
    within ``WS_TOL2`` of the minimum squared distance share the weight)."""
    n = len(coords)
    cutoff2 = cutoff ** 2
    lists = {a: [] for a in range(n)}
    for i in range(n):
        for j in range(i + 1):
            candidates = []
            for k, T in enumerate(r_trans):
                d2 = float(np.sum((coords[i] + T - coords[j]) ** 2))
                if d2 > cutoff2:
                    continue
                if d2 < MIN_NEIGH_DIST2:
                    if k == 0 and i == j:
                        continue
                    raise RuntimeError("very close distance found")
                candidates.append((d2, k))
            if not candidates:
                continue
            candidates.sort(key=lambda c: c[0])
            keep = [c for c in candidates if c[0] <= candidates[0][0] + WS_TOL2]
            weight = 1.0 / len(keep)
            for _d2, k in keep:
                lists[j].append((i, weight, coords[i] + r_trans[k]))
    return lists


def xtb_construction(coords, lattice, tol):
    alpha = optimal_alpha(lattice, tol)
    r_cut = real_space_cutoff(alpha, tol)
    r_trans = lattice_points(lattice, r_cut)
    wsc = wigner_seitz(coords, r_trans, r_cut)
    return alpha, r_cut, r_trans, wsc


def records_from_construction(coords, wsc, r_trans):
    """Directed image records reproducing xtb's real-space sum; the r = 0
    self term is left to the Ewald kernel's ``exclude_self``."""
    rec_a, rec_b, shifts, weights = [], [], [], []
    for a, entries in wsc.items():
        for b, w, image_xyz in entries:
            vec = image_xyz - coords[a]
            for T in r_trans:
                d = vec + T
                if float(d @ d) < 1.0e-18:
                    continue
                shift = d - (coords[b] - coords[a])
                rec_a.append(a)
                rec_b.append(b)
                shifts.append(shift)
                weights.append(w)
                if a != b:
                    rec_a.append(b)
                    rec_b.append(a)
                    shifts.append(-shift)
                    weights.append(w)
    return rec_a, rec_b, np.asarray(shifts).reshape(-1, 3), weights


def erfc_deficit(coords, lattice, wsc, r_trans, alpha, site_atom):
    """Full-lattice erfc sum minus xtb's weighted partial sum, per shell pair.

    xtb sums ``erfc(alpha r) / r`` over the same Wigner-Seitz-weighted set as
    its Klopman-Ohno remainder; lattice images reachable from fewer than all
    tied images of a pair are counted with weight below one.  The shared
    kernel's Ewald potential carries the whole lattice, so this deficit is
    what separates the two constructions once the remainders coincide."""
    n = len(coords)
    r_cut = max(np.linalg.norm(T) for T in r_trans)
    span = 2.0 * max(np.linalg.norm(lattice, axis=0))
    big = lattice_points(lattice, r_cut + span + 5.0)
    deficit_atom = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            points = (coords[b] - coords[a]) + big
            r = np.linalg.norm(points, axis=1)
            mask = r > 1.0e-9
            full = float(np.sum(erfc(alpha * r[mask]) / r[mask]))
            partial = 0.0
            for bb, w, image_xyz in wsc[a]:
                if bb != b:
                    continue
                d = (image_xyz - coords[a]) + r_trans
                rr = np.linalg.norm(d, axis=1)
                keep = rr > 1.0e-9
                partial += w * float(np.sum(erfc(alpha * rr[keep]) / rr[keep]))
            deficit_atom[a, b] = deficit_atom[b, a] = full - partial
    sites = np.asarray(site_atom)
    return deficit_atom[np.ix_(sites, sites)]


def vibeqc_gamma(coords, lattice, site_atom, hardness, rec_a, rec_b, shifts, weights, alpha):
    spec = _se.ShellGammaSpec()
    spec.form = _se.ShellGammaForm.KlopmanOhno
    spec.ko_average = _se.KlopmanOhnoAverage.InverseHardnessMean
    kernel = _se.EwaldCoulombKernel(np.ascontiguousarray(lattice.T), alpha)
    return _se._periodic_shell_gamma_from_records(
        list(site_atom), list(hardness), np.ascontiguousarray(coords), spec, kernel,
        list(rec_a), list(rec_b), shifts, list(weights),
    ), kernel.alpha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="xtb GFN1 run directory (coord, xtb.out, xtbrestart[, coulomb.dump])")
    parser.add_argument("--tol", type=float, default=1.0e-8, help="xtb Ewald tolerance of the run (default 1e-8)")
    args = parser.parse_args(argv)

    coords_raw, Z, lattice = read_coord(os.path.join(args.run_dir, "coord"))
    periodic = lattice.size == 9
    coords = wrap_like_xtb(coords_raw, lattice) if periodic else coords_raw
    restart = read_xtbrestart(os.path.join(args.run_dir, "xtbrestart"))
    text = open(os.path.join(args.run_dir, "xtb.out")).read()
    printed = float(re.search(r"-> electrostatic\s+([-\d.]+)", text).group(1))

    site_atom, hardness, third = [], [], []
    for a, z in enumerate(Z):
        shells, gamma3 = gfn1_shell_hardness_and_third(z)
        third.append(0.1 * gamma3)
        for U in shells:
            site_atom.append(a)
            hardness.append(U)
    qsh = restart["qsh"]
    if len(qsh) != len(hardness):
        raise SystemExit("shell count of xtbrestart disagrees with the GFN1 shell tables")
    q_atom = np.zeros(len(Z))
    for i, a in enumerate(site_atom):
        q_atom[a] += qsh[i]
    e_third = float(np.sum(np.asarray(third) * q_atom ** 3) / 3.0)
    target = printed - e_third
    print(f"xtb: printed electrostatic {printed:.12f}, E_3rd {e_third:.12f}, 0.5 qsh J qsh {target:.12f}")

    if not periodic:
        # Molecular limit: the same tables, average and on-site rule through
        # build_molecular_shell_gamma; xtb's printed value is reproduced to
        # its last digit (MgO diatomic at 3.5 bohr: 0.080549134921 Eh).
        spec = _se.ShellGammaSpec()
        spec.form = _se.ShellGammaForm.KlopmanOhno
        spec.ko_average = _se.KlopmanOhnoAverage.InverseHardnessMean
        gamma = _se._molecular_shell_gamma(list(site_atom), list(hardness), np.ascontiguousarray(coords), spec)
        e2 = 0.5 * qsh @ gamma @ qsh
        print(f"vibe-qc molecular: E2 {e2:.12f} (vs xtb {e2 - target:+.3e})")
        return 0

    dump_path = os.path.join(args.run_dir, "coulomb.dump")
    dump = read_dump(dump_path) if os.path.exists(dump_path) else None
    tol = dump["tolerance"] if dump else args.tol
    alpha, r_cut, r_trans, wsc = xtb_construction(coords, lattice, tol)
    print(f"replica: alpha {alpha:.10f}, r_cut {r_cut:.6f} bohr, {len(r_trans)} translations, "
          f"{sum(len(v) for v in wsc.values())} Wigner-Seitz entries")
    if dump is not None:
        print(f"dump:    alpha {dump['alpha']:.10f}, r_cut {dump['rcutoff']:.6f} bohr, "
              f"{len(dump['rtrans'])} translations, {sum(len(v) for v in dump['wsc'].values())} entries; "
              f"max |wrapped coords - dump xyz| {np.abs(coords - dump['xyz']).max():.1e}")
        mism = 0
        for a in range(len(Z)):
            mine = sorted((b, round(w, 9), tuple(np.round(x, 6))) for b, w, x in wsc[a])
            theirs = sorted((b, round(w, 9), tuple(np.round(x, 6))) for b, w, x in dump["wsc"][a])
            mism += mine != theirs
        print(f"         Wigner-Seitz lists differing from the replica: {mism}")
        dh = max(abs(dump["hardness"][(a, s)] - U) for (a, s), U in
                 zip(((site_atom[i], sum(1 for j in range(i) if site_atom[j] == site_atom[i])) for i in range(len(hardness))), hardness))
        print(f"         max |hardness table - dump| {dh:.1e}")

    rec_a, rec_b, shifts, weights = records_from_construction(coords, wsc, r_trans)
    deficit = erfc_deficit(coords, lattice, wsc, r_trans, alpha, site_atom)
    for alpha_vq in (0.0, alpha, 2.0 * alpha):
        gamma, used = vibeqc_gamma(coords, lattice, site_atom, hardness, rec_a, rec_b, shifts, weights, alpha_vq)
        e_raw = 0.5 * qsh @ gamma @ qsh
        e_corr = 0.5 * qsh @ (gamma - deficit) @ qsh
        line = (f"vibe-qc alpha {used:.6f}: E2 raw {e_raw:.12f} (vs xtb {e_raw - target:+.3e}), "
                f"corrected {e_corr:.12f} (vs xtb {e_corr - target:+.3e})")
        if dump is not None:
            line += (f"; max|J - Jxtb| raw {np.abs(gamma - dump['J']).max():.3e}, "
                     f"corrected {np.abs(gamma - deficit - dump['J']).max():.3e}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
