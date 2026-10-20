"""Static protocol support. Readiness is filled by real numerical witnesses."""


def method_specs():
    result = {}
    for method in ("ibo", "boys", "pipek-mezey"):
        result[method] = {
            "ready": False,
            "experimental": False,
            "molecular": True,
            "periodic": False,
            "coefficient_types": ["real"],
            "spins": ["restricted"],
            "occupations": [2],
            "kpoints": ["none"],
            "supplied_subspace": True,
            "fresh_rhf": True,
            "ao_convention": "qvf-gto-v1",
            "all_electron_only": True,
            "max_atomic_number": 86,
            "population_model": "iao-mini",
            "centroids": "position_expectation_bohr",
            "unsupported_reason": "Periodic, complex, unrestricted, spinor and fractional-occupation inputs are unsupported.",
        }
    result["aiccm-wannier"] = {
        "ready": False,
        "experimental": True,
        "molecular": False,
        "periodic": True,
        "coefficient_types": ["real", "complex_split_last_axis"],
        "spins": ["restricted"],
        "occupations": [2],
        "kpoints": ["gamma", "complete_unshifted_cyclic_mesh"],
        "dimensions": [1, 2, 3],
        "supplied_subspace": True,
        "fresh_rhf": False,
        "ao_convention": "qvf-gto-v1",
        "all_electron_only": True,
        "required_periodic_data": [
            "lattice_bohr",
            "dimension",
            "pbc",
            "mesh",
            "kpoints_fractional",
            "weights",
            "overlap",
            "representation",
            "bloch_convention",
            "orbital_energies_hartree",
        ],
        "population_model": "lowdin_finite_torus",
        "centroids": "projected_circular_ao_centres_bohr",
        "unsupported_reason": "Requires explicit finite-BvK provenance, time-reversal-compatible overlap, uniform full mesh and experimental opt-in; no general/shifted/irreducible meshes, metals or spinors.",
    }
    return result
