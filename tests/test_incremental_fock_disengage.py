"""IIDs 129, 418, and 468: incremental ΔD Fock accuracy and certification.

Historical root cause: the DirectJKBuilder incremental path built G₂ₑ[ΔD]
with the same absolute Schwarz cutoff as the full build. Quartets whose
ΔD-weighted bound fell below that cutoff were dropped, and their contribution
was not recovered until the periodic full rebuild, so the cached G_prev
accumulated a bias across iterations. On the
adenine-thymine WC monomer B / cc-pVDZ case (156 BF, the AUTO mode
resolves to DIRECT) that bias stalls the SCF at a ~1e-6 gradient floor
(~1e-7 Ha energy drift, documented in benchmarks/orca_vs_vibeqc_speed.md)
and the run failed closed after 200 iterations at v0.15.134, while
v0.15.108 converged the same case in 16 iterations (BUG 87 flipped
``incremental_fock`` to default-on without handling the drift floor).

IIDs 129 and 418 added the correctness backstop: on the first stall signal
(the BUG 64 stall detector), or once the gradient is eligible for convergence,
the SCF drivers disengage the incremental cache via
``JKBuilder::set_incremental(false)`` so the fine phase runs a nonincremental,
tight-screened full-density rebuild — the same map as
``incremental_fock = False``,
the pre-BUG-87 default — and only a persistent post-disengagement stall
falls through to the orbital-rotation restart.

IID 468 removes the source of the coarse-phase floor. Each small nonzero ΔD
is amplitude-normalized before the density-weighted screen and the linear
Fock contribution is scaled back afterwards. The omitted increment therefore
shrinks with ΔD instead of retaining one absolute error allowance per update.
Periodic full rebuilds and final full-map certification remain independent
safeguards.

Pre-IID-129, ``run_rhf`` on this deck returns ``converged = False`` inside
100 iterations; post-fix it converges in ~40 at the same energy as the
full-rebuild path (asserted as the parity control). IID 418 covers the
complementary failure: the orbital gradient has already passed, so the
stall detector cannot end the incremental coarse phase while its screened
energy continues to dither.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq

# S22-07 adenine-thymine WC monomer B, Jurečka et al. PCCP 8, 1985 (2006)
# geometry as staged for the Liakos DLPNO wave (Angstrom; converted to
# bohr here). See the rp217cc065 bundle member deck in the article
# calculations archive for the verbatim source.
_ANG2BOHR = 1.8897259886
_ATOMS_ANG = [
    ('N', (-3.9211729, -0.0009646, -1.5163659)),
    ('C', (-4.6136833, 0.0169051, -0.3336520)),
    ('C', (-3.9917387, 0.0219348, 0.8663338)),
    ('C', (-2.5361367, 0.0074651, 0.8766724)),
    ('N', (-1.9256484, -0.0110593, -0.3638948)),
    ('C', (-2.5395897, -0.0149474, -1.5962357)),
    ('C', (-4.7106131, 0.0413373, 2.1738637)),
    ('O', (-1.8674730, 0.0112093, 1.9120833)),
    ('O', (-1.9416783, -0.0291878, -2.6573783)),
    ('H', (-4.4017172, -0.0036078, -2.4004924)),
    ('H', (-0.8838255, -0.0216168, -0.3784269)),
    ('H', (-5.6909220, 0.0269347, -0.4227183)),
    ('H', (-4.4439282, -0.8302573, 2.7695655)),
    ('H', (-4.4267056, 0.9186178, 2.7530256)),
    ('H', (-5.7883971, 0.0505530, 2.0247280)),
]
_ATOMIC_NUMBERS = {'N': 7, 'C': 6, 'O': 8, 'H': 1}

# The direct-path fixed point, measured on this machine for both the
# incremental and the full-rebuild path (they agree to 1e-9; the
# incremental path with the fix converges to the same value).
_E_REF = -451.5445940339

# S22-19 benzene-HCN dimer, Jurecka et al., PCCP 8, 1985 (2006).
# This is the geometry used by the IID 418 reproduction (Angstrom).
_S22_19_ATOMS_ANG = [
    ('C', (-0.7097741, -0.9904230, 1.2077018)),
    ('C', (-1.4065340, -0.9653529, 0.0000000)),
    ('C', (-0.7097741, -0.9904230, -1.2077018)),
    ('C', (0.6839651, -1.0405105, -1.2078652)),
    ('C', (1.3809779, -1.0655522, 0.0000000)),
    ('C', (0.6839651, -1.0405105, 1.2078652)),
    ('H', (-1.2499482, -0.9686280, 2.1440507)),
    ('H', (-2.4869197, -0.9237060, 0.0000000)),
    ('H', (-1.2499482, -0.9686280, -2.1440507)),
    ('H', (1.2242882, -1.0580753, -2.1442563)),
    ('H', (2.4615886, -1.1029818, 0.0000000)),
    ('H', (1.2242882, -1.0580753, 2.1442563)),
    ('N', (-0.0034118, 3.5353926, 0.0000000)),
    ('C', (0.0751963, 2.3707040, 0.0000000)),
    ('H', (0.1476295, 1.3052847, 0.0000000)),
]
_S22_19_E_REF = -323.60742975889


@pytest.fixture(scope="module")
def at_monob():
    return vq.Molecule(
        [
            vq.Atom(_ATOMIC_NUMBERS[symbol], [c * _ANG2BOHR for c in xyz])
            for symbol, xyz in _ATOMS_ANG
        ],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture(scope="module")
def s22_19():
    return vq.Molecule(
        [
            vq.Atom(_ATOMIC_NUMBERS[symbol], [c * _ANG2BOHR for c in xyz])
            for symbol, xyz in _S22_19_ATOMS_ANG
        ],
        charge=0,
        multiplicity=1,
    )


def _tight_rhf_opts(max_iter, incremental):
    opts = vq.RHFOptions()
    opts.max_iter = max_iter
    opts.conv_tol_energy = 1.0e-10
    opts.incremental_fock = incremental
    return opts


def test_liakos_at_monob_ccpvdz_direct_incremental_converges(at_monob):
    """The IID 129 case: AUTO resolves 156 BF to DIRECT, incremental
    default-on, conv_tol_energy = 1e-10. Pre-fix this stalls on the ΔD
    drift floor and returns converged = False; post-fix the stall
    disengages the cache and the full-density fine phase converges."""
    result = vq.run_rhf(at_monob, "cc-pvdz", _tight_rhf_opts(100, True))
    assert result.converged
    assert result.n_iter < 100
    assert abs(result.energy - _E_REF) < 1e-8


def test_liakos_at_monob_ccpvdz_direct_full_rebuild_parity(at_monob):
    """Positive control: the full-rebuild (incremental off) path must
    converge quickly to the same fixed point, so the disengaged fine
    phase above lands on the same answer."""
    result = vq.run_rhf(at_monob, "cc-pvdz", _tight_rhf_opts(100, False))
    assert result.converged
    assert result.n_iter < 60
    assert abs(result.energy - _E_REF) < 1e-8


@pytest.mark.parametrize("flavor", ["rhf", "rks", "uhf", "uks"])
def test_gradient_pass_ends_incremental_coarse_phase_before_acceptance(flavor):
    """A screened incremental energy plateau must enter a full-build phase.

    The mock alternates between adding ``1e-8 S`` and no perturbation.  This
    changes the reported energy while leaving the generalized eigenvectors
    and Pulay commutator unchanged.  Its second incremental row already has
    the eventual full-build energy, so the first full-build row has dE = 0:
    only the explicit cross-map-row gate prevents premature acceptance at
    iteration 3.  Before IID 418 the active builder keeps alternating until
    ``max_iter``; the fixed loop accepts the second comparable full-build row.
    """
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    basis = vq.BasisSet(mol, "sto-3g")
    overlap = np.asarray(vq.compute_overlap(basis), dtype=float)
    hcore = np.asarray(vq.compute_kinetic(basis), dtype=float) + np.asarray(
        vq.compute_nuclear(basis, mol), dtype=float
    )

    class _DitheringIncrementalBuilder(vq.JKBuilder):
        def __init__(self):
            super().__init__()
            self.active = True
            self.build_calls = 0
            self.disable_calls = 0
            self.reset_calls = 0

        def build_J(self, density):
            del density
            return self._coarse_shift()

        def build_K(self, density):
            return np.zeros_like(np.asarray(density))

        def build_g_rhf(self, density, alpha_hf=1.0):
            del density, alpha_hf
            return self._coarse_shift()

        def _coarse_shift(self):
            self.build_calls += 1
            if not self.active:
                return np.zeros_like(overlap)
            shift = 1.0e-8 if self.build_calls % 2 else 0.0
            return shift * overlap

        def reset_state(self):
            self.reset_calls += 1

        def set_incremental(self, enabled):
            if self.active and not enabled:
                self.disable_calls += 1
            self.active = bool(enabled)

        def incremental_active(self):
            return self.active

    builder = _DitheringIncrementalBuilder()
    options_cls = {
        "rhf": vq.RHFOptions,
        "rks": vq.RKSOptions,
        "uhf": vq.UHFOptions,
        "uks": vq.UKSOptions,
    }[flavor]
    opts = options_cls()
    opts.max_iter = 8
    opts.conv_tol_energy = 1.0e-10
    opts.conv_tol_grad = 1.0e-8
    opts.use_diis = False
    opts.damping = 0.0
    opts.dynamic_damping = False
    opts.auto_level_shift_on_oscillation = False
    opts.restart_opts.enabled = False
    # Keep this mock focused on the incremental-to-nonincremental transition;
    # the independent loose-to-tight Schwarz transition also resets builders.
    opts.schwarz_threshold_loose = opts.schwarz_threshold
    if flavor in {"rks", "uks"}:
        opts.functional = "LDA"
        grid = vq.build_grid(mol, vq.GridOptions())

    if flavor == "rhf":
        result = vq.run_rhf_scf_with_jk(
            basis, 2, overlap, hcore, 1.0 / 1.4, builder, opts
        )
    elif flavor == "rks":
        result = vq.run_rks_scf_with_jk(
            basis, 2, overlap, hcore, 1.0 / 1.4, builder, grid, opts
        )
    elif flavor == "uhf":
        result = vq.run_uhf_scf_with_jk(
            basis, 1, 1, overlap, hcore, 1.0 / 1.4, builder, opts
        )
    else:
        result = vq.run_uks_scf_with_jk(
            basis, 1, 1, overlap, hcore, 1.0 / 1.4, builder, grid, opts
        )

    assert result.converged, (
        f"n_iter={result.n_iter}, disable_calls={builder.disable_calls}, "
        f"last_dE={result.scf_trace[-1].delta_e:.12e}, "
        f"last_grad={result.scf_trace[-1].grad_norm:.12e}"
    )
    assert result.n_iter == 4
    assert builder.disable_calls == 1
    assert builder.reset_calls == 1
    assert abs(result.scf_trace[2].delta_e) < opts.conv_tol_energy
    assert result.scf_trace[2].grad_norm < opts.conv_tol_grad
    assert abs(result.scf_trace[-1].delta_e) < opts.conv_tol_energy
    assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad


def test_gradient_pass_forces_configured_tight_schwarz_map():
    """A custom tighten threshold cannot admit the loose-screened map."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    basis = vq.BasisSet(mol, "sto-3g")
    overlap = np.asarray(vq.compute_overlap(basis), dtype=float)
    hcore = np.asarray(vq.compute_kinetic(basis), dtype=float) + np.asarray(
        vq.compute_nuclear(basis, mol), dtype=float
    )

    class _LooseScreenBuilder(vq.JKBuilder):
        def __init__(self):
            super().__init__()
            self.loose = True
            self.build_calls = 0
            self.reset_calls = 0
            self.thresholds = []

        def build_J(self, density):
            return np.zeros_like(np.asarray(density))

        def build_K(self, density):
            return np.zeros_like(np.asarray(density))

        def build_g_rhf(self, density, alpha_hf=1.0):
            del density, alpha_hf
            self.build_calls += 1
            if not self.loose:
                return np.zeros_like(overlap)
            shift = 1.0e-8 if self.build_calls % 2 else 0.0
            return shift * overlap

        def set_schwarz_threshold(self, threshold):
            self.thresholds.append(float(threshold))
            self.loose = False

        def uses_runtime_schwarz_threshold(self):
            return True

        def reset_state(self):
            self.reset_calls += 1

        def incremental_active(self):
            return False

    builder = _LooseScreenBuilder()
    opts = vq.RHFOptions()
    opts.max_iter = 8
    opts.conv_tol_energy = 1.0e-10
    opts.conv_tol_grad = 1.0e-8
    opts.use_diis = False
    opts.damping = 0.0
    opts.dynamic_damping = False
    opts.auto_level_shift_on_oscillation = False
    opts.restart_opts.enabled = False
    opts.incremental_fock = False
    opts.schwarz_threshold = 1.0e-10
    opts.schwarz_threshold_loose = 1.0e-7
    # Deliberately below the achieved gradient.  The ordinary scheduled
    # tighten cannot fire, so convergence eligibility itself must force it.
    opts.schwarz_threshold_tighten_at = 0.0

    result = vq.run_rhf_scf_with_jk(
        basis, 2, overlap, hcore, 1.0 / 1.4, builder, opts
    )

    assert result.converged
    assert result.n_iter == 4
    assert builder.thresholds == [opts.schwarz_threshold]
    assert builder.reset_calls == 1
    assert abs(result.scf_trace[2].delta_e) < opts.conv_tol_energy
    assert result.scf_trace[2].grad_norm < opts.conv_tol_grad
    assert abs(result.scf_trace[-1].delta_e) < opts.conv_tol_energy
    assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad


def test_s22_19_ccpvdz_certifies_on_the_full_build_map(s22_19):
    """The IID 418 witness must not wait for a lucky cached-energy hit."""
    opts = _tight_rhf_opts(20, True)
    result = vq.run_rhf(s22_19, "cc-pvdz", opts)

    assert result.converged, (
        f"n_iter={result.n_iter}, "
        f"last_dE={result.scf_trace[-1].delta_e:.12e}, "
        f"last_grad={result.scf_trace[-1].grad_norm:.12e}"
    )
    assert result.n_iter < opts.max_iter
    assert abs(result.scf_trace[-1].delta_e) < opts.conv_tol_energy
    assert result.scf_trace[-1].grad_norm < opts.conv_tol_grad
    assert abs(result.energy - result.scf_trace[-1].energy) < opts.conv_tol_energy
    assert abs(result.energy - _S22_19_E_REF) < 1.0e-9
