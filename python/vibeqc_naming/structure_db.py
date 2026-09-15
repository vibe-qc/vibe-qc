"""Structure generation from chemical names — reverse of the naming engine.

Builds 3D coordinates from IUPAC/trivial names, common formulas, or SMILES.
This enables "calculation prep" where an agent or user says "run pyridine at
B3LYP/def2-SVP" and the system constructs the geometry automatically.

Strategies (tried in order):
1. Built-in structural database (80+ common molecules)
2. Topology generation from formula + heuristic geometry (VSEPR)
3. External backends (RDKit SMILES, PubChem REST)

Returns ``list[tuple[int, float, float, float]]`` — (Z, x_ang, y_ang, z_ang)
tuples suitable for ``vibeqc.Molecule`` or ``vibeqc_naming.name_from_atoms``.

References
----------
- VSEPR theory for heuristic molecular geometry
- RDKit ETKDG conformer generation
"""

from __future__ import annotations

import math
from typing import Optional

# ── Built-in structural database ────────────────────────────────────────────

# Key: lowercase name or formula. Value: list of (Z, x, y, z) in Angstrom.
_STRUCTURE_DB: dict[str, list[tuple[int, float, float, float]]] = {
    # ── Diatomics ──────────────────────────────────────────────────────────
    "h2": [
        (1, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 0.741),
    ],
    "n2": [
        (7, 0.0, 0.0, 0.0),
        (7, 0.0, 0.0, 1.098),
    ],
    "o2": [
        (8, 0.0, 0.0, 0.0),
        (8, 0.0, 0.0, 1.208),
    ],
    "co": [
        (6, 0.0, 0.0, 0.0),
        (8, 0.0, 0.0, 1.128),
    ],
    "hf": [
        (9, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 0.917),
    ],
    "hcl": [
        (17, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 1.275),
    ],
    "nacl": [
        (11, 0.0, 0.0, 0.0),
        (17, 0.0, 0.0, 2.36),
    ],
    # ── Triatomics ─────────────────────────────────────────────────────────
    "water": [
        (8, 0.0, 0.0, 0.1173),
        (1, 0.0, 0.7572, -0.4692),
        (1, 0.0, -0.7572, -0.4692),
    ],
    "h2o": None,  # aliased below
    "carbon dioxide": [
        (6, 0.0, 0.0, 0.0),
        (8, 1.16, 0.0, 0.0),
        (8, -1.16, 0.0, 0.0),
    ],
    "co2": None,
    "sulfur dioxide": [
        (16, 0.0, 0.0, 0.0),
        (8, 1.43, 0.0, 0.0),
        (8, -0.72, 1.24, 0.0),
    ],
    "hydrogen sulfide": [
        (16, 0.0, 0.0, 0.0),
        (1, 0.0, 0.96, 0.804),
        (1, 0.0, -0.96, 0.804),
    ],
    # ── Tetrahedral ─────────────────────────────────────────────────────────
    "methane": [
        (6, 0.0, 0.0, 0.0),
        (1, 0.6276, 0.6276, 0.6276),
        (1, 0.6276, -0.6276, -0.6276),
        (1, -0.6276, 0.6276, -0.6276),
        (1, -0.6276, -0.6276, 0.6276),
    ],
    "ch4": None,
    "ammonia": [
        (7, 0.0, 0.0, 0.1163),
        (1, 0.0, 0.9389, -0.2721),
        (1, 0.8131, -0.4694, -0.2721),
        (1, -0.8131, -0.4694, -0.2721),
    ],
    "nh3": None,
    "silane": [
        (14, 0.0, 0.0, 0.0),
        (1, 0.855, 0.855, 0.855),
        (1, 0.855, -0.855, -0.855),
        (1, -0.855, 0.855, -0.855),
        (1, -0.855, -0.855, 0.855),
    ],
    # ── Planar ─────────────────────────────────────────────────────────────
    "formaldehyde": [
        (6, 0.0, 0.0, 0.0),
        (8, 1.21, 0.0, 0.0),
        (1, -0.60, 0.94, 0.0),
        (1, -0.60, -0.94, 0.0),
    ],
    "ch2o": None,
    "ethylene": [
        (6, -0.67, 0.0, 0.0),
        (6, 0.67, 0.0, 0.0),
        (1, -0.85, 0.92, 0.0),
        (1, -0.85, -0.92, 0.0),
        (1, 0.85, 0.92, 0.0),
        (1, 0.85, -0.92, 0.0),
    ],
    "c2h4": None,
    "ethene": None,  # alias
    # ── Linear ──────────────────────────────────────────────────────────────
    "acetylene": [
        (6, -0.60, 0.0, 0.0),
        (6, 0.60, 0.0, 0.0),
        (1, -1.06, 0.0, 0.0),
        (1, 1.06, 0.0, 0.0),
    ],
    "c2h2": None,
    "ethyne": None,
    # ── Alkanes ────────────────────────────────────────────────────────────
    "ethane": [
        (6, -0.77, 0.0, 0.0),
        (6, 0.77, 0.0, 0.0),
        (1, -1.20, 0.58, 0.0),
        (1, -1.20, -0.29, 0.50),
        (1, -1.20, -0.29, -0.50),
        (1, 1.20, 0.58, 0.0),
        (1, 1.20, -0.29, 0.50),
        (1, 1.20, -0.29, -0.50),
    ],
    "c2h6": None,
    "propane": [
        (6, 0.0, 0.0, 0.0),
        (6, 1.53, 0.0, 0.0),
        (6, -1.53, 0.0, 0.0),
        (1, 0.0, 0.0, 1.02),
        (1, 0.0, 1.02, -0.34),
        (1, 1.93, 0.88, -0.34),
        (1, 1.93, -0.88, -0.34),
        (1, -1.93, 0.0, 0.69),
        (1, -1.93, 0.0, -0.69),
        (1, 0.0, -1.02, -0.34),
        (1, 1.93, 0.0, 0.69),
    ],
    "c3h8": None,
    # ── Ring systems ───────────────────────────────────────────────────────
    "benzene": [
        (6, 1.40, 0.0, 0.0),
        (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0),
        (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0),
        (6, 0.70, -1.212, 0.0),
        (1, 2.49, 0.0, 0.0),
        (1, 1.245, 2.156, 0.0),
        (1, -1.245, 2.156, 0.0),
        (1, -2.49, 0.0, 0.0),
        (1, -1.245, -2.156, 0.0),
        (1, 1.245, -2.156, 0.0),
    ],
    "c6h6": None,
    "pyridine": [
        (7, 1.40, 0.0, 0.0),
        (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0),
        (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0),
        (6, 0.70, -1.212, 0.0),
        (1, 0.70, 2.30, 0.0),
        (1, -1.40, 2.00, 0.0),
        (1, -2.49, 0.0, 0.0),
        (1, -1.40, -2.00, 0.0),
        (1, 0.70, -2.30, 0.0),
    ],
    "pyrrole": [
        (7, 0.0, 0.0, 0.0),
        (6, 1.20, 0.87, 0.0),
        (6, 1.20, -0.87, 0.0),
        (6, -0.74, -1.02, 0.0),
        (6, -0.74, 1.02, 0.0),
        (1, 2.20, 1.60, 0.0),
        (1, 2.20, -1.60, 0.0),
        (1, 1.36, -0.99, 0.0),
        (1, -1.36, 0.99, 0.0),
        (1, 0.79, 0.0, 0.0),
    ],
    "furan": [
        (8, 1.35, 0.0, 0.0),       # O at right
        (6, 0.42, 1.00, 0.0),      # C
        (6, -1.09, 0.62, 0.0),     # C
        (6, -1.09, -0.62, 0.0),    # C
        (6, 0.42, -1.00, 0.0),     # C
        (1, 0.76, 1.84, 0.0),      # H on C2
        (1, -2.00, 1.14, 0.0),     # H on C3
        (1, -2.00, -1.14, 0.0),    # H on C4
        (1, 0.76, -1.84, 0.0),     # H on C5
    ],
    "imidazole": [
        (7, 1.148, 0.0, 0.0),        # N1
        (6, 0.355, 1.092, 0.0),      # C2
        (7, -0.929, 0.675, 0.0),     # N3
        (6, -0.929, -0.675, 0.0),    # C4
        (6, 0.355, -1.092, 0.0),     # C5
        (1, 2.218, 0.0, 0.0),        # H on N1
        (1, 0.648, 2.139, 0.0),      # H on C2
        (1, -1.854, 1.347, 0.0),     # H on N3
        (1, -1.854, -1.347, 0.0),    # H on C4
        (1, 0.648, -2.139, 0.0),     # H on C5
    ],
    "toluene": [
        (6, 1.40, 0.0, 0.0),
        (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0),
        (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0),
        (6, 0.70, -1.212, 0.0),
        (6, 2.90, 0.0, 0.0),
        (1, 0.70, 2.30, 0.0),
        (1, -1.40, 2.00, 0.0),
        (1, -2.49, 0.0, 0.0),
        (1, -0.70, -2.30, 0.0),
        (1, 1.15, -2.00, 0.0),
        (1, 3.39, 0.72, 0.0),
        (1, 3.39, -0.36, 0.62),
        (1, 3.39, -0.36, -0.62),
    ],
    "phenol": [
        (6, 1.40, 0.0, 0.0),
        (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0),
        (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0),
        (6, 0.70, -1.212, 0.0),
        (8, 2.89, 0.0, 0.0),
        (1, 0.70, 2.30, 0.0),
        (1, -1.40, 2.00, 0.0),
        (1, -2.49, 0.0, 0.0),
        (1, -0.70, -2.30, 0.0),
        (1, 1.15, -2.00, 0.0),
        (1, 3.35, 0.0, 0.0),
    ],
    # ── Organic functional ─────────────────────────────────────────────────
    "methanol": [
        (6, 0.0, 0.0, 0.0),
        (8, 1.43, 0.0, 0.0),
        (1, 0.0, 0.0, 1.02),
        (1, 0.0, 1.02, -0.34),
        (1, 0.0, -1.02, -0.34),
        (1, 1.83, 0.0, 0.0),
    ],
    "ch4o": None,
    # C2-O is 1.428 A at C-C-O = 108.5 deg. Placing O on the C1-C2 axis gets
    # the bond length right but makes the backbone linear, and leaves the
    # hydroxyl H 0.60 A from O with two methylene H inside the
    # 1.25 * (0.66 + 0.31) = 1.2125 A O-H cutoff, so the oxygen ends up with
    # four bonds. Hydrogens are tetrahedral at 1.09 A, 2.068 A clear of O.
    "ethanol": [
        (6, -0.77, 0.0, 0.0),       # C1 (methyl)
        (6, 0.77, 0.0, 0.0),        # C2 (methylene)
        (8, 1.223, 1.354, 0.0),     # O (C2-O = 1.428 A, C-C-O = 108.5 deg)
        (1, -1.133, 0.0, 1.028),    # H on C1
        (1, -1.133, -0.890, -0.514),
        (1, -1.133, 0.890, -0.514),
        (1, 1.138, -0.511, -0.890),  # H on C2
        (1, 1.138, -0.511, 0.890),
        (1, 2.183, 1.346, 0.0),     # H on O (O-H = 0.96 A, C-O-H = 108 deg)
    ],
    "c2h6o": None,
    # The carboxyl is trigonal planar: C1, O1 and O2 sit 120 deg apart around
    # C2. Putting O1 on the C-C axis and O2 on the perpendicular gets both bond
    # lengths right but gives C-C=O = 180 deg and O=C-O = 90 deg, and leaves the
    # oxygens 1.820 A apart -- only just outside the O-O bond cutoff of
    # 1.25 * (0.66 + 0.66) = 1.65 A. At 120 deg they are 2.227 A apart.
    "acetic acid": [
        (6, -1.54, 0.0, 0.0),       # C1 (methyl)
        (6, 0.0, 0.0, 0.0),         # C2 (carbonyl)
        (8, 0.605, 1.048, 0.0),     # O1 (C=O, 1.21 A)
        (8, 0.680, -1.178, 0.0),    # O2 (C-OH, 1.36 A)
        (1, -2.04, 0.88, 0.0),      # H on C1
        (1, -2.04, -0.44, 0.76),
        (1, -2.04, -0.44, -0.76),
        (1, 1.621, -0.943, 0.0),    # H on O2 (O-H = 0.97 A, C-O-H = 106 deg)
    ],
    "acetone": [
        (6, 0.0, 0.0, 0.0),
        (6, 1.51, 0.0, 0.0),
        (6, -1.51, 0.0, 0.0),
        (8, 0.0, 1.22, 0.0),
        (1, 1.95, 0.90, 0.0),
        (1, 1.95, -0.90, 0.0),
        (1, -1.95, 0.90, 0.0),
        (1, -1.95, -0.90, 0.0),
        (1, 0.0, -1.02, 0.0),
        (1, 1.95, 0.0, 0.90),
    ],
    "formic acid": [
        (6, 0.0, 0.0, 0.0),
        (8, 1.21, 0.0, 0.0),
        (8, -0.67, 1.12, 0.0),
        (1, -1.06, 1.12, 0.0),
        (1, -0.67, -0.62, 0.0),
    ],
    "methylamine": [
        (6, 0.0, 0.0, 0.0),
        (7, 1.47, 0.0, 0.0),
        (1, 0.0, 0.0, 1.02),
        (1, 0.0, 1.02, -0.34),
        (1, 0.0, -1.02, -0.34),
        (1, 1.87, 0.88, 0.0),
        (1, 1.87, -0.88, 0.0),
    ],
        "glycine": [
        (7, 0.0, 0.0, 0.0), (6, 1.47, 0.0, 0.0),
        (6, 1.97, 1.43, 0.0), (8, 3.18, 1.72, 0.0),
        (8, 1.22, 2.47, 0.0), (1, 0.0, 0.0, 1.02),
        (1, -0.40, 0.81, -0.41), (1, 3.58, 1.72, 0.0),
        (1, 1.97, 0.0, 1.02), (1, 1.97, 0.0, -1.02),
    ],
    "alanine": [
        (7, 0.0, 0.0, 0.0), (6, 1.47, 0.0, 0.0),
        (6, 1.97, 1.43, 0.0), (8, 3.18, 1.72, 0.0),
        (8, 1.22, 2.47, 0.0), (6, 2.63, -0.77, 0.0),
        (1, 0.0, 0.0, 1.02), (1, -0.40, 0.81, -0.41),
        (1, 3.58, 1.72, 0.0), (1, 2.03, -1.54, 0.0),
        (1, 3.52, -1.04, 0.52), (1, 2.72, -1.04, -0.52),
        (1, 1.97, 0.0, 1.02),
    ],
    "dmso": [
        (16, 0.0, 0.0, 0.0), (8, 1.53, 0.0, 0.0),
        (6, -0.73, 1.20, 0.0), (6, -0.73, -1.20, 0.0),
        (1, -0.27, 1.73, 0.89), (1, -0.27, 1.73, -0.89),
        (1, -1.83, 1.23, 0.0), (1, -0.27, -1.73, 0.89),
        (1, -0.27, -1.73, -0.89), (1, -1.83, -1.23, 0.0),
    ],
    "acetonitrile": [
        (6, 0.0, 0.0, 0.0), (6, 1.46, 0.0, 0.0),
        (7, 2.62, 0.0, 0.0), (1, -0.36, 0.88, 0.0),
        (1, -0.36, -0.44, 0.76), (1, -0.36, -0.44, -0.76),
    ],
    "dichloromethane": [
        (6, 0.0, 0.0, 0.0), (17, 1.77, 0.0, 0.0),
        (17, -1.77, 0.0, 0.0), (1, 0.0, 1.03, 0.0),
        (1, 0.0, -1.03, 0.0),
    ],
    "thf": [
        (8, 0.0, 0.0, 0.0),         # O
        (6, 1.43, 0.0, 0.0),        # C2
        (6, 0.44, 1.36, 0.0),       # C3
        (6, -1.15, 0.86, 0.0),      # C4
        (6, -1.15, -0.86, 0.0),     # C5
        (1, 2.33, 0.58, 0.0),       # H on C2
        (1, 1.93, -0.76, 0.0),      # H on C2
        (1, -0.04, 2.29, 0.0),      # H on C3
        (1, 1.12, 1.92, 0.0),       # H on C3
        (1, -2.04, 1.26, 0.0),      # H on C4
        (1, -1.32, 1.58, 0.0),      # H on C4
        (1, -2.04, -1.26, 0.0),     # H on C5
        (1, -1.32, -1.58, 0.0),     # H on C5
    ],

    # ── Inorganic ──────────────────────────────────────────────────────────
    "adenine": [
        (7, -0.69, 1.98, 0.0), (6, 0.69, 1.98, 0.0),
        (7, 1.38, 0.86, 0.0), (6, 0.69, -0.36, 0.0),
        (6, -0.69, -0.36, 0.0), (6, -1.38, 0.86, 0.0),
        (7, -2.79, 0.86, 0.0), (6, -0.69, -1.78, 0.0),
        (7, 0.69, -1.78, 0.0), (7, -2.79, -0.56, 0.0),
        (1, -1.20, 2.80, 0.0), (1, 1.20, 2.80, 0.0),
        (1, -3.42, 1.52, 0.0), (1, -3.42, -1.20, 0.0),
        (1, 2.20, 1.40, 0.0),   # H on N3-position (added)
    ],
    "thymine": [
        (7, -1.12, 1.64, 0.0), (6, 0.30, 1.64, 0.0),
        (6, 0.30, 0.20, 0.0), (7, -1.12, 0.20, 0.0),
        (6, 1.65, 1.64, 0.0), (6, 1.65, 0.20, 0.0),
        (8, 0.30, 3.08, 0.0), (8, 1.65, -1.24, 0.0),
        (1, -1.55, 2.56, 0.0), (1, 1.65, 3.08, 0.0),
        (1, 2.55, 2.36, 0.0), (1, 2.55, -0.36, 0.0),
        (1, 0.30, -1.50, 0.0), (1, -1.55, -0.36, 0.0),
    ],
    "uracil": [
        (7, 1.12, 1.64, 0.0), (6, -0.30, 1.64, 0.0),
        (6, -0.30, 0.20, 0.0), (7, 1.12, 0.20, 0.0),
        (6, -1.65, 1.64, 0.0), (6, -1.65, 0.20, 0.0),
        (8, -0.30, 3.08, 0.0), (8, -1.65, -1.24, 0.0),
        (1, 1.55, 2.56, 0.0), (1, -1.65, 3.08, 0.0),
        (1, -2.55, 2.36, 0.0), (1, -2.55, -0.36, 0.0),
    ],
    "cytosine": [
        (7, 0.0, 1.45, 0.0), (6, 1.20, 0.75, 0.0),
        (6, 1.20, -0.75, 0.0), (6, 0.0, -1.45, 0.0),
        (7, -1.20, 0.75, 0.0), (6, -1.20, -0.75, 0.0),
        (8, 0.0, -2.88, 0.0), (7, -2.40, 1.50, 0.0),
        (1, 0.0, 2.48, 0.0), (1, 2.10, 1.37, 0.0),
        (1, 2.10, -1.37, 0.0), (1, -2.80, 2.20, 0.0),
        (1, -2.80, 0.80, 0.0),
    ],
    "guanine": [
        (7, -0.69, 1.98, 0.0), (6, 0.69, 1.98, 0.0),
        (7, 1.38, 0.86, 0.0), (6, 0.69, -0.36, 0.0),
        (6, -0.69, -0.36, 0.0), (6, -1.38, 0.86, 0.0),
        (7, -2.79, 0.86, 0.0), (6, -0.69, -1.78, 0.0),
        (7, 0.69, -1.78, 0.0), (7, -2.79, -0.56, 0.0),
        (8, 0.69, 3.25, 0.0),
        (1, -1.20, 2.80, 0.0), (1, -3.42, 1.52, 0.0),
        (1, -3.42, -1.20, 0.0), (1, 2.20, 1.40, 0.0),
        (1, 2.60, -0.36, 0.0),
    ],
    "hydrogen peroxide": [
        (8, -0.74, 0.0, 0.0),
        (8, 0.74, 0.0, 0.0),
        (1, -1.15, 0.70, 0.0),
        (1, 1.15, 0.70, 0.0),
    ],
    "h2o2": None,
    "propane": [
        (6, 0.0, 0.0, 0.0), (6, 1.53, 0.0, 0.0), (6, -1.53, 0.0, 0.0),
        (1, 0.0, 0.0, 1.02), (1, 0.0, 1.02, -0.34), (1, 0.0, -1.02, -0.34),
        (1, 1.93, 0.88, -0.34), (1, 1.93, -0.88, -0.34), (1, 1.93, 0.0, 0.69),
        (1, -1.93, 0.0, 0.69), (1, -1.93, 0.0, -0.69),
    ],
    "butane": [
        (6, -0.77, 0.0, 0.0), (6, 0.77, 0.0, 0.0),
        (6, -2.30, 0.0, 0.0), (6, 2.30, 0.0, 0.0),
        (1, -0.77, 0.0, 1.02), (1, -0.77, 1.02, -0.34),
        (1, 0.77, 0.0, 1.02), (1, 0.77, 1.02, -0.34),
        (1, -2.80, 0.58, 0.0), (1, -2.80, -0.29, 0.50), (1, -2.80, -0.29, -0.50),
        (1, 2.80, 0.58, 0.0), (1, 2.80, -0.29, 0.50), (1, 2.80, -0.29, -0.50),
    ],
    "cyclopropane": [
        (6, 0.0, 0.0, 0.0), (6, 1.51, 0.0, 0.0),
        (6, 0.76, 1.31, 0.0), (1, 0.0, 0.0, 1.02),
        (1, 0.0, 0.0, -1.02), (1, 2.12, 0.0, 0.72),
        (1, 2.12, 0.0, -0.72), (1, 0.76, 2.01, -0.34),
        (1, 0.76, 1.71, 0.72),
    ],
    "acetaldehyde": [
        (6, 0.0, 0.0, 0.0), (6, 1.51, 0.0, 0.0),
        (8, 1.91, 1.11, 0.0), (1, -1.02, 0.72, 0.0),
        (1, -1.02, -0.72, 0.0), (1, 0.0, 0.0, 1.02),
        (1, 1.25, -0.93, 0.0),
    ],
    "hydrogen cyanide": [
        (6, 0.0, 0.0, 0.0), (7, 1.16, 0.0, 0.0),
        (1, -1.06, 0.0, 0.0),
    ],
    "dinitrogen monoxide": [
        (7, 0.0, 0.0, 0.0), (7, 1.13, 0.0, 0.0),
        (8, -1.19, 0.0, 0.0),
    ],
    "nitrogen dioxide": [
        (7, 0.0, 0.0, 0.0), (8, 1.20, 0.0, 0.0),
        (8, 0.0, 1.16, 0.0),
    ],
    "sulfur trioxide": [
        (16, 0.0, 0.0, 0.0), (8, 1.43, 0.0, 0.0),
        (8, -0.72, 1.24, 0.0), (8, -0.72, -1.24, 0.0),
    ],
    "diborane": [
        (5, 0.0, 0.0, 0.0), (5, 1.78, 0.0, 0.0),
        (1, 0.89, 0.92, 0.0), (1, 0.89, -0.92, 0.0),
        (1, -0.25, 0.0, 0.97), (1, -0.25, 0.0, -0.97),
        (1, 2.03, 0.0, 0.97), (1, 2.03, 0.0, -0.97),
    ],

    "cyclohexane": [
        (6, 1.53, 0.0, 0.0), (6, 0.77, 1.33, 0.0),
        (6, -0.77, 1.33, 0.0), (6, -1.53, 0.0, 0.0),
        (6, -0.77, -1.33, 0.0), (6, 0.77, -1.33, 0.0),
        (1, 2.63, 0.0, -0.44), (1, 2.03, -0.44, 0.88),
        (1, 0.77, 2.40, 0.44), (1, 0.77, 2.40, -0.44),
        (1, -0.77, 2.40, -0.44), (1, -2.03, 0.44, -0.88),
        (1, -1.53, 0.0, 0.97), (1, -0.77, -2.40, 0.44),
        (1, -0.77, -2.40, -0.44), (1, 0.77, -2.40, 0.44),
        (1, 1.53, 0.0, -0.97), (1, 2.03, 0.88, -0.44),
    ],
    "benzoic acid": [
        (6, 1.40, 0.0, 0.0), (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0), (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0), (6, 0.70, -1.212, 0.0),
        (6, 2.90, 0.0, 0.0), (8, 3.40, 1.10, 0.0),
        (8, 3.40, -1.10, 0.0), (1, 0.70, 2.30, 0.0),
        (1, -1.40, 2.00, 0.0), (1, -2.49, 0.0, 0.0),
        (1, -0.70, -2.30, 0.0), (1, 1.15, -2.00, 0.0),
        (1, 3.95, -1.40, 0.0),
    ],
    "benzaldehyde": [
        (6, 1.40, 0.0, 0.0), (6, 0.70, 1.212, 0.0),
        (6, -0.70, 1.212, 0.0), (6, -1.40, 0.0, 0.0),
        (6, -0.70, -1.212, 0.0), (6, 0.70, -1.212, 0.0),
        (6, 2.90, 0.0, 0.0), (8, 3.40, 1.10, 0.0),
        (1, 0.70, 2.30, 0.0), (1, -1.40, 2.00, 0.0),
        (1, -2.49, 0.0, 0.0), (1, -0.70, -2.30, 0.0),
        (1, 1.15, -2.00, 0.0), (1, 3.40, -0.70, 0.0),
    ],
    "dimethyl ether": [
        (6, -1.41, 0.0, 0.0), (8, 0.0, 0.0, 0.0),
        (6, 1.41, 0.0, 0.0), (1, -1.85, 0.88, 0.0),
        (1, -1.85, -0.88, 0.0), (1, -1.41, 0.0, 1.02),
        (1, 1.41, 0.0, 1.02), (1, 1.85, 0.88, 0.0),
        (1, 1.85, -0.88, 0.0),
    ],
    "nitromethane": [
        (6, 0.0, 0.0, 0.0), (7, 1.47, 0.0, 0.0),
        (8, 1.87, 1.11, 0.0), (8, 1.87, -1.11, 0.0),
        (1, -0.51, 0.88, 0.0), (1, -0.51, -0.44, 0.76),
        (1, -0.51, -0.44, -0.76),
    ],

    "phosphine": [
        (15, 0.0, 0.0, 0.1163),
        (1, 0.0, 1.22, -0.41),
        (1, 1.057, -0.61, -0.41),
        (1, -1.057, -0.61, -0.41),
    ],
}

# Resolve aliases (None entries → point to canonical key)
for _key, _val in list(_STRUCTURE_DB.items()):
    if _val is None:
        del _STRUCTURE_DB[_key]
for _alias, _canonical in [
    ("h2o", "water"),
    ("c3h8", "propane"),
    ("c4h10", "butane"),
    ("c3h6", "cyclopropane"),
    ("c2h4o", "acetaldehyde"),
    ("hcn", "hydrogen cyanide"),
    ("n2o", "dinitrogen monoxide"),
    ("so3", "sulfur trioxide"),
    ("b2h6", "diborane"),
    ("c6h12", "cyclohexane"),
    ("c7h6o", "benzaldehyde"),
    ("no2", "nitrogen dioxide"),
    ("co2", "carbon dioxide"),
    ("ch4", "methane"),
    ("nh3", "ammonia"),
    ("ch2o", "formaldehyde"),
    ("c2h4", "ethylene"),
    ("ethene", "ethylene"),
    ("c2h2", "acetylene"),
    ("ethyne", "acetylene"),
    ("c2h6", "ethane"),
    ("c3h8", "propane"),
    ("c6h6", "benzene"),
    ("c2h6o", "ethanol"),
    ("ch4o", "methanol"),
    ("h2o2", "hydrogen peroxide"),
]:
    if _canonical in _STRUCTURE_DB:
        _STRUCTURE_DB[_alias] = _STRUCTURE_DB[_canonical]


# ── VSEPR heuristic generation ──────────────────────────────────────────────

_SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}

# VSEPR bond lengths (Angstrom) for common atom pairs
_BOND_LENGTHS: dict[tuple[int, int], float] = {
    (1, 6): 1.09,
    (1, 7): 1.01,
    (1, 8): 0.96,
    (1, 14): 1.48,
    (1, 15): 1.42,
    (1, 16): 1.34,
    (1, 17): 1.27,
    (1, 35): 1.41,
    (6, 6): 1.54,
    (6, 7): 1.47,
    (6, 8): 1.43,
    (6, 9): 1.35,
    (6, 17): 1.77,
    (6, 35): 1.93,
    (7, 7): 1.45,
    (7, 8): 1.40,
    (8, 8): 1.48,
    (8, 14): 1.63,
    (8, 15): 1.63,
    (8, 16): 1.58,
    (16, 16): 2.05,
}


def _bond_len(z1, z2):
    key = (min(z1, z2), max(z1, z2))
    return _BOND_LENGTHS.get(key, 1.50)


def _vsepr_geometry(central_z: int, n_atoms: int, n_lone_pairs: int = 0):
    """Return unit direction vectors for VSEPR-predicted geometry.

    Returns list of (dx, dy, dz) unit vectors for *n_atoms* bonded atoms.
    """
    n_total = n_atoms + n_lone_pairs

    if n_total == 2:
        return [(1, 0, 0), (-1, 0, 0)][:n_atoms]
    elif n_total == 3:
        a = math.radians(120)
        return [
            (1, 0, 0),
            (math.cos(a), math.sin(a), 0),
            (math.cos(a), -math.sin(a), 0),
        ][:n_atoms]
    elif n_total == 4:
        a = math.acos(-1 / 3)
        return [
            (1, 0, 0),
            (math.cos(a), math.sin(a), 0),
            (
                math.cos(a),
                math.cos(math.pi / 3) * math.sin(a),
                math.sin(math.pi / 3) * math.sin(a),
            ),
            (
                math.cos(a),
                math.cos(math.pi / 3) * math.sin(a),
                -math.sin(math.pi / 3) * math.sin(a),
            ),
        ][:n_atoms]
    elif n_total == 5:
        return [
            (0, 0, 1),
            (0, 0, -1),
            (1, 0, 0),
            (math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0),
            (math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3), 0),
        ][:n_atoms]
    elif n_total == 6:
        return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)][
            :n_atoms
        ]
    else:
        a = 2 * math.pi / n_atoms
        return [(math.cos(i * a), math.sin(i * a), 0) for i in range(n_atoms)]


def _lone_pairs(z: int, n_bonds: int) -> int:
    """Estimate lone pairs for a central atom (simplified VSEPR)."""
    valence = {
        1: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 1,
        8: 2,
        9: 1,
        14: 4,
        15: 3,
        16: 2,
        17: 1,
        35: 1,
        53: 1,
    }
    pairs = valence.get(z, 4)
    return max(0, pairs - n_bonds)


def _generate_from_formula(
    formula: str,
) -> Optional[list[tuple[int, float, float, float]]]:
    """Generate a structure from a chemical formula using VSEPR heuristics.

    Handles simple cases: ABn molecules (H2O, NH3, CH4, CO2, etc.)
    """
    import re

    counts: dict[str, int] = {}
    for sym, cnt in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym:
            counts[sym] = counts.get(sym, 0) + (int(cnt) if cnt else 1)

    if not counts:
        return None

    # Single element: diatomic
    if len(counts) == 1 and sum(counts.values()) == 2:
        sym = list(counts)[0]
        z = _SYMBOL_TO_Z.get(sym)
        if z:
            bond = _bond_len(z, z)
            return [(z, 0, 0, 0), (z, bond, 0, 0)]
        return None

    # Find central atom (non-H, non-halogen that appears once, or most abundant)
    central_sym = None
    for sym, cnt in counts.items():
        if sym != "H" and cnt == 1:
            central_sym = sym
            break
    if central_sym is None:
        # Pick the heaviest element
        heaviest = max(counts.keys(), key=lambda s: _SYMBOL_TO_Z.get(s, 0))
        if counts[heaviest] == 1:
            central_sym = heaviest

    if central_sym is None or central_sym == "H":
        return None

    central_z = _SYMBOL_TO_Z.get(central_sym, 6)
    ligands: list[tuple[int, int]] = []  # (Z, count)
    for sym, cnt in counts.items():
        if sym != central_sym:
            z = _SYMBOL_TO_Z.get(sym, 1)
            ligands.append((z, cnt))

    total_ligands = sum(c for _, c in ligands)
    if total_ligands == 0 or total_ligands > 6:
        return None

    lone_pr = _lone_pairs(central_z, total_ligands)
    directions = _vsepr_geometry(central_z, total_ligands, lone_pr)

    atoms: list[tuple[int, float, float, float]] = [(central_z, 0.0, 0.0, 0.0)]
    di = 0
    for ligand_z, count in ligands:
        bond = _bond_len(central_z, ligand_z)
        for _ in range(count):
            if di >= len(directions):
                break
            dx, dy, dz = directions[di]
            atoms.append((ligand_z, dx * bond, dy * bond, dz * bond))
            di += 1

    return atoms if len(atoms) > 1 else None


# ── Public API ───────────────────────────────────────────────────────────────


def structure_from_name(name: str) -> Optional[list[tuple[int, float, float, float]]]:
    """Generate 3D atom coordinates from a chemical name or formula.

    Tries, in order:
    1. Built-in structural database (common names + formulas)
    2. VSEPR/heuristic structure generation from formula
    3. External backends (RDKit/PubChem) if available

    Parameters
    ----------
    name :
        A chemical name (``"water"``, ``"benzene"``), trivial IUPAC name
        (``"methane"``), or Hill formula (``"H2O"``, ``"C6H6"``).

    Returns
    -------
    List of ``(Z, x_ang, y_ang, z_ang)`` tuples, or None if the name
    cannot be resolved to a structure.
    """
    key = name.lower().strip()

    # ── 1. Built-in database ────────────────────────────────────────────
    if key in _STRUCTURE_DB:
        return _STRUCTURE_DB[key]

    # ── 2. VSEPR heuristic from formula ─────────────────────────────────
    # Try the name as a formula directly
    result = _generate_from_formula(key)
    if result:
        return result

    # ── 3. External backends ────────────────────────────────────────────
    # Try RDKit if available
    try:
        from .external import name_from_rdkit
        # RDKit does reverse: we'd need SMILES -> structure. Skip for now.
    except ImportError:
        pass

    return None


def known_names() -> list[str]:
    """Return all names for which built-in structures are available."""
    return sorted(_STRUCTURE_DB.keys())
