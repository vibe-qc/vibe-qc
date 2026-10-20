"""Water / cc-pVDZ: the same RHF and PBE energies on libint and on MANTA (experimental).

Selects the integral engine and the Coulomb and exchange algorithms
independently through ``run_job(integral_engine=, coulomb=, exchange=)``
and prints each energy against the default libint route with the executed
provenance. See docs/user_guide/integral_engines.md.

Measured with the vibe-qc 0.17.17 development build (main, 2026-10-14),
HCORE start, auxiliary basis cc-pVDZ-JKFIT; differences to the exact
build in microhartree:

    RHF  libint exact           -76.0268153023      reference
    RHF  MANTA exact            -76.0268153023        0.0
    RHF  MANTA RI-J, exact K    -76.0268445288      -29.2
    RHF  MANTA RI-J, RI-K       -76.0267944326      +20.9
    RHF  MANTA BUPO-J, exact K  -76.0268153023        0.0  (no far-field pairs)
    PBE  libint exact           -76.3333672257      reference
    PBE  MANTA RI-J             -76.3333947179      -27.5

MANTA is a development engine: it is tens of times slower than libint here
and its routes refuse many workflows. Use the default route for production.

Run from an empty scratch directory (vibe-qc writes its outputs into the
current directory); the whole script takes a few minutes:

    python /path/to/examples/molecular/input-h2o-integral-engines.py
"""

import vibeqc as vq
from vibeqc import manta

AUX = "cc-pvdz-jkfit"


def budget() -> "manta.ResourceBudget":
    """Explicit resource limits; a job that would exceed one refuses up front."""
    b = manta.ResourceBudget()
    b.memory_bytes = 2**30
    b.shell_calls = 10**9
    b.primitive_quartets = 10**13
    b.scalar_terms = 10**17
    b.kernel_terms = 10**18
    return b


def engine() -> "manta.IntegralEngineOptions":
    return manta.IntegralEngineOptions(backend="manta", budget=budget())


CASES = {
    "rhf-libint": dict(method="rhf"),
    "rhf-manta-exact": dict(method="rhf", integral_engine=engine()),
    "rhf-manta-rij": dict(method="rhf", integral_engine=engine(),
                          coulomb=manta.CoulombBuildOptions("ri", AUX)),
    "rhf-manta-rijk": dict(method="rhf", integral_engine=engine(),
                           coulomb=manta.CoulombBuildOptions("ri", AUX),
                           exchange=manta.ExchangeBuildOptions("ri")),
    "rhf-manta-bupo": dict(method="rhf", integral_engine=engine(),
                           coulomb=manta.CoulombBuildOptions("bupo")),
    "pbe-libint": dict(method="rks", functional="pbe"),
    "pbe-manta-rij": dict(method="rks", functional="pbe", integral_engine=engine(),
                          coulomb=manta.CoulombBuildOptions("ri", AUX)),
}


def main() -> None:
    water = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.4304, -1.1060]),
        vq.Atom(1, [0.0, -1.4304, -1.1060]),
    ])
    reference = {}
    for name, kw in CASES.items():
        res = vq.run_job(water, basis="cc-pvdz", initial_guess="hcore",
                         output=name, **kw)
        family = name.split("-")[0]
        reference.setdefault(family, res.energy)
        prov = manta.molecular_engine_provenance(res)
        route = ("default libint route" if prov is None else
                 f"{prov['one_electron_provider']}: J={prov['coulomb_algorithm']}, "
                 f"K={prov['exchange_algorithm'] if prov['exchange_active'] else 'none'}, "
                 f"confirmed={prov['full_build_confirmed']}")
        print(f"{name:16s} {res.energy:.10f}  "
              f"{(res.energy - reference[family]) * 1e6:+8.1f} uHa  {route}")


if __name__ == "__main__":
    main()
