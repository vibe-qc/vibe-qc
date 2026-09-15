"""C++ SCF host for the GPW Γ-only route (v0.12 R3).

Pins the pybind11 JKBuilder trampoline plus the
:mod:`vibeqc.periodic_gapw_cpp_host` wrapper that drives the GPW
Hartree-J kernel from the C++ ``run_rhf_scf_with_jk`` SCF loop. The
shape of these tests:

* He / H2 STO-3G converge to the same total energy via the C++ host
  and the Python ``run_periodic_rhf_gpw``, to machine precision —
  same gauge, same kernels, just a different SCF driver.
* On a 4-electron-and-larger system (H4 chain), the C++ host keeps
  iteration counts no worse than the Python driver while preserving
  energy parity. The two paths now share the same DIIS machinery, so
  iteration ties are expected on this pinned case.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_cpp_host import (
    PyGpwJKBuilder,
    run_rhf_scf_gpw_cpp,
)
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

warnings.simplefilter("ignore", vq.GAPWExperimentalWarning)


# ---------- helpers ----------------------------------------------------


def _he_system(L: float = 10.0):
    sys_p = core.PeriodicSystem()
    sys_p.dim = 3
    sys_p.lattice = np.eye(3) * L
    sys_p.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    return sys_p, basis


def _h2_system(L: float = 12.0):
    atoms = [(1, [L / 2 - 0.7, L / 2, L / 2]),
             (1, [L / 2 + 0.7, L / 2, L / 2])]
    sys_p = core.PeriodicSystem()
    sys_p.dim = 3
    sys_p.lattice = np.eye(3) * L
    sys_p.unit_cell = [core.Atom(z, pos) for z, pos in atoms]
    mol = vq.Molecule(
        [vq.Atom(z, pos) for z, pos in atoms],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    return sys_p, basis


def _h4_system(L: float = 14.0):
    atoms = [(1, [L / 2 - 1.5, L / 2, L / 2]),
             (1, [L / 2 - 0.5, L / 2, L / 2]),
             (1, [L / 2 + 0.5, L / 2, L / 2]),
             (1, [L / 2 + 1.5, L / 2, L / 2])]
    sys_p = core.PeriodicSystem()
    sys_p.dim = 3
    sys_p.lattice = np.eye(3) * L
    sys_p.unit_cell = [core.Atom(z, pos) for z, pos in atoms]
    mol = vq.Molecule(
        [vq.Atom(z, pos) for z, pos in atoms],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    return sys_p, basis


# ---------- 1. JKBuilder trampoline exposed ---------------------------


def test_jk_builder_subclassable_from_python():
    """Sanity check the pybind11 trampoline: a Python subclass of
    ``_vibeqc_core.JKBuilder`` can be constructed and its ``build_J``
    / ``build_K`` overrides are reached when called from Python."""
    n = 3

    class _StubJK(core.JKBuilder):
        def __init__(self):
            super().__init__()
            self.j_calls = 0
            self.k_calls = 0

        def build_J(self, D):
            self.j_calls += 1
            return np.zeros_like(np.asarray(D))

        def build_K(self, D):
            self.k_calls += 1
            return np.zeros_like(np.asarray(D))

    jk = _StubJK()
    D = np.eye(n)
    J = jk.build_J(D)
    K = jk.build_K(D)
    assert J.shape == (n, n)
    assert K.shape == (n, n)
    assert jk.j_calls == 1
    assert jk.k_calls == 1


# ---------- 2. He STO-3G parity --------------------------------------


def test_run_rhf_scf_gpw_cpp_he_matches_python():
    """He STO-3G converges via the C++ host to the same total energy
    as ``run_periodic_rhf_gpw`` — same gauge, same kernels."""
    sys_p, basis = _he_system(L=10.0)

    cpp = run_rhf_scf_gpw_cpp(
        sys_p, basis, cutoff_ha=200.0, max_iter=50, quiet=True,
    )
    py = run_periodic_rhf_gpw(
        sys_p, basis, cutoff_ha=200.0, max_iter=50, quiet=True,
    )

    assert cpp.converged
    assert py.converged
    # Machine precision parity — the Fock builds are identical at
    # the converged density.
    assert cpp.energy == pytest.approx(py.energy, abs=1e-10)


# ---------- 3. H2 STO-3G parity --------------------------------------


def test_run_rhf_scf_gpw_cpp_h2_matches_python():
    """H2 STO-3G converges via the C++ host to the same total energy
    as ``run_periodic_rhf_gpw``."""
    sys_p, basis = _h2_system(L=12.0)

    cpp = run_rhf_scf_gpw_cpp(
        sys_p, basis, cutoff_ha=200.0, max_iter=50, quiet=True,
    )
    py = run_periodic_rhf_gpw(
        sys_p, basis, cutoff_ha=200.0, max_iter=50, quiet=True,
    )

    assert cpp.converged
    assert py.converged
    assert cpp.energy == pytest.approx(py.energy, abs=1e-10)


# ---------- 4. C++ SCF host parity on a DIIS-sensitive case ----------


def test_cpp_scf_matches_python_on_h4_chain():
    """On an H4 chain (8 electrons, 4 AOs) where DIIS history matters,
    the C++ host matches the Python GPW driver in energy and does not
    require more SCF iterations. The shared DIIS kernel makes equal
    iteration counts a valid outcome."""
    sys_p, basis = _h4_system(L=14.0)

    cpp = run_rhf_scf_gpw_cpp(
        sys_p, basis, cutoff_ha=200.0, max_iter=100, quiet=True,
    )
    py = run_periodic_rhf_gpw(
        sys_p, basis, cutoff_ha=200.0, max_iter=100, quiet=True,
    )

    assert cpp.converged
    assert py.converged
    # Parity at the converged density:
    assert cpp.energy == pytest.approx(py.energy, abs=1e-9)
    # The C++ host used to win strictly because it had a stronger DIIS
    # path. After the shared DIIS refactor, a tie is expected; keep a
    # guard that the trampoline path is not slower on this pinned case.
    assert cpp.n_iter <= py.n_iter, (
        f"expected C++ SCF to require no more iters than Python "
        f"(got cpp={cpp.n_iter}, py={py.n_iter})"
    )


# ---------- 5. PyGpwJKBuilder shape sanity ---------------------------


def test_pygpw_jk_builder_shapes():
    """``PyGpwJKBuilder.build_J`` / ``build_K`` return ``(n_bf, n_bf)``
    real matrices on a small periodic system."""
    sys_p, basis = _h2_system(L=12.0)
    from vibeqc.periodic_gapw_grid import make_grid as _make_grid

    grid = _make_grid(np.asarray(sys_p.lattice, dtype=float),
                       cutoff_ha=200.0)
    jk = PyGpwJKBuilder(basis, sys_p, grid)

    D = np.eye(basis.nbasis)
    J = jk.build_J(D)
    K = jk.build_K(D)
    n_bf = basis.nbasis
    assert J.shape == (n_bf, n_bf)
    assert K.shape == (n_bf, n_bf)
    # J and K real-valued and symmetric (to floating-point noise).
    assert np.allclose(J, J.T, atol=1e-10)
    assert np.allclose(K, K.T, atol=1e-10)


@pytest.mark.parametrize("kind", ["SAP", "HUECKEL", "MINAO"])
def test_native_host_reaches_periodic_builder_and_preserves_provenance(kind, monkeypatch):
    """The native host consumes the periodic construction in its Gamma metric."""
    import vibeqc.guess as guess
    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma

    system, basis = _h2_system()
    expected = getattr(core.InitialGuess, kind)
    reached = []
    first_j_density = []
    construct = guess.initial_density_closed_shell
    build_j = PyGpwJKBuilder.build_J

    def record_builder(*args, **kwargs):
        density = construct(*args, **kwargs)
        reached.append((args[3], kwargs["periodic_system"], density.copy()))
        return density

    def record_j(self, density):
        if not first_j_density:
            first_j_density.append(np.asarray(density).copy())
        return build_j(self, density)

    monkeypatch.setattr(guess, "initial_density_closed_shell", record_builder)
    monkeypatch.setattr(PyGpwJKBuilder, "build_J", record_j)
    result = run_rhf_scf_gpw_cpp(
        system, basis, initial_guess=kind, cutoff_ha=50.0, quiet=True,
    )
    assert result.converged
    assert len(reached) == 1
    assert reached[0][0] == expected
    assert reached[0][1] is system
    density = reached[0][2]
    overlap = _overlap_lattice_gamma(basis, system)
    assert np.trace(density @ overlap).real == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(density, density.conj().T, atol=1e-12)
    np.testing.assert_allclose(first_j_density[0], density, atol=1e-12)
    assert result.guess_selection.requested == expected
    assert result.guess_selection.effective == expected
    assert result.guess_selection.transport == expected
    reference = run_rhf_scf_gpw_cpp(
        system, basis, initial_guess="HCORE", cutoff_ha=50.0, quiet=True,
    )
    assert reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-10)


def test_native_host_refuses_complex_gamma_restart():
    system, basis = _h2_system()
    density = np.eye(basis.nbasis, dtype=complex)
    density[0, 1], density[1, 0] = 0.1j, -0.1j
    with pytest.raises(ValueError, match="requires a real initial density"):
        run_rhf_scf_gpw_cpp(
            system, basis, initial_guess="READ", initial_density=density,
            cutoff_ha=50.0, quiet=True,
        )
