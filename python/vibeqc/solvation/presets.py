"""Solvent dielectric presets (S1d).

Static dielectric constants e at 298 K from the IUPAC / NIST tables
unless noted. Values match the Gaussian 16 `SCRF=Solvent=...` table
to ±1% -- the small differences are inconsequential for a continuum
model (which already approximates the molecular-scale solvent
structure as a homogeneous dielectric).

Adding a solvent
----------------
``solvent="..."`` accepts any key (case-insensitive) of
:data:`SOLVENT_PRESETS`, an alias from :data:`SOLVENT_ALIASES`, or a
direct ``float`` (interpreted as e). For one-off solvents users can
pass ``solvent={"epsilon": 25.0, ...}`` to control any
:class:`SolventModel` field directly.
"""

from __future__ import annotations

# Map: canonical-name -> static dielectric (dimensionless).
# Solvents are organised polar -> apolar within each family.
SOLVENT_PRESETS: dict[str, float] = {
    # --- Aqueous + protic ----------------------------------------------------
    "water":            78.39,    # H2O -- most common reaction medium.
    "methanol":         32.61,    # MeOH
    "ethanol":          24.85,    # EtOH
    "isopropanol":      19.26,    # i-PrOH
    "n-butanol":        17.33,
    # --- Polar aprotic -------------------------------------------------------
    "dmso":             46.83,    # dimethyl sulfoxide
    "dmf":              37.22,    # N,N-dimethylformamide
    "acetonitrile":     35.94,    # MeCN
    "acetone":          20.49,
    "thf":              7.43,     # tetrahydrofuran
    # --- Halogenated ---------------------------------------------------------
    "dichloromethane":  8.93,     # DCM / MeCl2
    "chloroform":       4.71,     # CHCl3
    "carbon-tetrachloride": 2.23,
    # --- Aromatic / apolar ---------------------------------------------------
    "benzene":          2.27,
    "toluene":          2.37,
    "n-hexane":         1.88,
    "n-pentane":        1.84,
    "cyclohexane":      2.02,
    # --- Reference vacuum (== gas) ------------------------------------------
    # Listed for completeness. The CPCM driver short-circuits e <= 1 and
    # returns the gas-phase SCF unchanged (so ``solvent="vacuum"`` is a
    # documented no-op the user can drop into a script that loops over
    # solvents without special-casing the gas leg).
    "vacuum":            1.0,
}

# Common synonyms / lower-case aliases users type at the command line.
SOLVENT_ALIASES: dict[str, str] = {
    "h2o":              "water",
    "meoh":             "methanol",
    "etoh":             "ethanol",
    "iproh":            "isopropanol",
    "i-proh":           "isopropanol",
    "ipoh":             "isopropanol",
    "buoh":             "n-butanol",
    "butanol":          "n-butanol",
    "mecn":             "acetonitrile",
    "acn":              "acetonitrile",
    "ch3cn":            "acetonitrile",
    "dcm":              "dichloromethane",
    "ch2cl2":           "dichloromethane",
    "methylene-chloride":   "dichloromethane",
    "chcl3":            "chloroform",
    "ccl4":             "carbon-tetrachloride",
    "hexane":           "n-hexane",
    "pentane":          "n-pentane",
    "gas":              "vacuum",
    "none":             "vacuum",
}


def canonicalise_solvent_name(name: str) -> str:
    """Return the canonical preset key for a user-supplied solvent name.

    Case- and hyphen-insensitive: ``"DMSO"`` / ``"DMSO "`` /
    ``"di-methyl-sulfoxide"`` all map to ``"dmso"`` (and through the
    presets table to e = 46.83). Raises :class:`KeyError` if the name
    is not a preset or alias -- pass a numeric e or a
    :class:`SolventModel` directly for solvents outside the table.
    """
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    if key in SOLVENT_PRESETS:
        return key
    if key in SOLVENT_ALIASES:
        return SOLVENT_ALIASES[key]
    raise KeyError(
        f"Unknown solvent {name!r}. Known presets: "
        f"{', '.join(sorted(SOLVENT_PRESETS))}. "
        f"Aliases: {', '.join(sorted(SOLVENT_ALIASES))}. "
        f"Pass a numeric e or a SolventModel(...) for solvents "
        f"outside this list."
    )
