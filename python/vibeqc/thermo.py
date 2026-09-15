"""Phase 17a-3 -- molecular thermochemistry post-processor.

Harmonic-oscillator + rigid-rotor + ideal-gas thermal corrections at
user-specified temperature and pressure, computed from the harmonic
frequencies returned by :func:`vibeqc.compute_hessian_fd`. Decomposes
into translational, rotational, vibrational, and electronic
contributions; reports zero-point energy and the standard partition-
function-derived ``U`` (internal energy), ``H`` (enthalpy), and
``G`` (Gibbs free energy) corrections.

API
---
::

    from vibeqc import (
        compute_hessian_fd, compute_thermochemistry, ThermoOptions,
    )

    hess = compute_hessian_fd(mol, "sto-3g", method="RHF")
    thermo = compute_thermochemistry(
        mol, hess, options=ThermoOptions(
            temperature=298.15,         # K
            pressure=101325.0,          # Pa (1 atm)
            symmetry_number=2,          # s for H2O point group C2v
        )
    )
    print(f"ZPE: {thermo.zpe:.6f} Hartree")
    print(f"G(T) correction: {thermo.g_thermal:.6f} Hartree")
    print(f"S(T): {thermo.s_total:.6f} Hartree/K")

Conventions
-----------
Energies in Hartree, entropies in Hartree/K -- matches PySCF's
``pyscf.hessian.thermo`` output verbatim. (PySCF reports "per mol"
in atomic units which works out the same: dividing by Avogadro's
number gives per-molecule quantities, which is what these formulas
return naturally. The numbers are identical.)

To convert to other unit systems:

    H[kcal/mol] = H[Hartree] x 627.509474
    H[kJ/mol]   = H[Hartree] x 2625.4996
    S[J/(K.mol)] = S[Hartree/K] x 4.359744 x 10^-18 x 6.02214 x 10^23
                 = S[Hartree/K] x 2625499.6 / 1000
                 ≈ S[Hartree/K] x 2625.5

Caveats
-------
* **Symmetry number** must be set explicitly. The default s = 1
  underestimates the rotational partition function (and overestimates
  rotational entropy) by ``ln(s)`` for any molecule with a non-trivial
  point group. Common values: H2 / H2O s = 2, NH3 s = 3, CH4 s = 12,
  C6H6 (benzene) s = 12.
* **Imaginary modes** (saddle points) are excluded from the
  vibrational partition function with a warning in the result
  (``n_imaginary_modes_excluded``). Treating them as ZPE-bearing or
  thermal contributors is unphysical (an imaginary frequency means
  the geometry is downhill in that direction).
* **Electronic degeneracy** defaults to the molecule's multiplicity
  (2S+1). This ignores excited states; for systems with low-lying
  electronic excitations (e.g. transition metals at high T) the
  correction is too crude.
* The harmonic approximation diverges from reality for low-frequency
  modes (< ~50 cm⁻¹) where the oscillator stops looking quadratic.
  Truhlar's quasi-harmonic correction is **not** applied here -- that
  would be Phase 17a-3a if anyone needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._vibeqc_core import Molecule
from .hessian import HessianResult
from .properties import _ATOMIC_MASSES

# ----------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ----------------------------------------------------------------------

_KB_J_PER_K: float = 1.380649e-23           # Boltzmann (J/K, exact since SI 2019)
_PLANCK_J_S: float = 6.62607015e-34         # h (J.s, exact since SI 2019)
_AVOGADRO: float = 6.02214076e23            # /mol (exact since SI 2019)
_HARTREE_TO_J: float = 4.3597447222071e-18  # 1 Hartree in Joules
_BOHR_TO_M: float = 5.29177210903e-11       # 1 bohr in meters
_AMU_TO_KG: float = 1.66053906660e-27       # 1 amu in kilograms
_AMU_TO_ELECTRON_MASS: float = 1822.888486209  # m_u / m_e

# Convenience: k_B in Hartree per Kelvin (per molecule)
_KB_HARTREE_PER_K: float = _KB_J_PER_K / _HARTREE_TO_J  # ≈ 3.166811e-6

# Wavenumber -> Hartree (matches hessian.py)
_CM_INV_TO_HARTREE: float = 1.0 / 219474.6313632

# Atomic-unit-of-inertia (m_e x bohr^2) in SI (kg x m^2)
_INERTIA_AU_TO_SI: float = (_AMU_TO_KG / _AMU_TO_ELECTRON_MASS) * _BOHR_TO_M ** 2


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------

@dataclass
class ThermoOptions:
    """Knobs for :func:`compute_thermochemistry`.

    temperature
        Kelvin. Default 298.15 (room temperature in standard
        thermochemistry tables).

    pressure
        Pascals. Default 101325 (1 atm). Only enters through the
        translational-partition-function molar volume.

    symmetry_number
        Rotational symmetry number s -- the number of indistinguishable
        rotational orientations of the rigid molecule. **Must be set
        manually** for non-trivial point groups; default 1 is right
        for asymmetric tops only. Typical values:
        H₂ / H₂O / O₂ / N₂ -> 2, NH₃ -> 3, CH₄ -> 12, benzene -> 12,
        SF₆ -> 24, C60 -> 60.

    electronic_degeneracy
        Spin/orbital degeneracy of the electronic ground state. If
        None (default), reads ``multiplicity = 2S+1`` from the
        molecule. Override for systems with non-spin-only electronic
        degeneracy (e.g. ^2Π states of NO).
    """
    temperature: float = 298.15
    pressure: float = 101325.0
    symmetry_number: int = 1
    electronic_degeneracy: Optional[int] = None


@dataclass
class ThermoResult:
    """Output of :func:`compute_thermochemistry`.

    All energies in **Hartree**, all entropies in **Hartree/K**, all
    per-molecule (multiply by Avogadro's number for per-mole). Same
    convention as :func:`pyscf.hessian.thermo.thermo`.

    The thermal corrections (``u_thermal``, ``h_thermal``, ``g_thermal``)
    are the additions to the electronic energy; the total free energy
    in solution / vacuum is ``E_electronic + g_thermal``.
    """
    # Conditions
    temperature: float          # Kelvin
    pressure: float             # Pascal

    # Per-component energies (Hartree)
    zpe: float                  # (1/2) S ℏw_i
    e_trans: float              # 3/2 . k_B . T
    e_rot: float                # 3/2 . k_B . T (nonlinear) or k_B . T (linear)
    e_vib: float                # S ℏw / (e^x - 1)  -- thermal part only, excludes ZPE

    # Cumulative thermal corrections (Hartree)
    u_thermal: float            # ZPE + e_trans + e_rot + e_vib
    h_thermal: float            # u_thermal + k_B T  (ideal gas)
    g_thermal: float            # h_thermal - T . s_total

    # Per-component entropies (Hartree/K)
    s_trans: float
    s_rot: float
    s_vib: float
    s_elec: float
    s_total: float

    # Per-component heat capacities at constant volume (Hartree/K)
    cv_trans: float
    cv_rot: float
    cv_vib: float
    cv_total: float

    # Diagnostics
    n_imaginary_modes_excluded: int
    """Number of modes with frequency < -1 cm⁻¹ that were dropped from
    the harmonic-oscillator partition function. > 0 means the
    geometry isn't a true minimum -- thermo numbers should be
    interpreted accordingly (the approximation is questionable at
    a saddle point)."""
    rotor_type: str
    """One of ``'atom'``, ``'linear'``, ``'nonlinear'``. Atom: no
    rotation. Linear: 2 rotational DOFs. Nonlinear: 3."""


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _principal_moments_of_inertia(positions_bohr: np.ndarray,
                                  masses_amu: np.ndarray) -> np.ndarray:
    """Sorted (ascending) principal moments of inertia in atomic units
    (m_e x bohr^2). For linear molecules, the smallest eigenvalue is
    ≈ 0 (axis along the molecular line)."""
    masses_e = masses_amu * _AMU_TO_ELECTRON_MASS
    com = (masses_e[:, None] * positions_bohr).sum(axis=0) / masses_e.sum()
    R = positions_bohr - com
    I = np.zeros((3, 3), dtype=np.float64)
    for i, mi in enumerate(masses_e):
        x, y, z = R[i]
        r2 = x * x + y * y + z * z
        I[0, 0] += mi * (r2 - x * x)
        I[1, 1] += mi * (r2 - y * y)
        I[2, 2] += mi * (r2 - z * z)
        I[0, 1] -= mi * x * y
        I[0, 2] -= mi * x * z
        I[1, 2] -= mi * y * z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return np.sort(np.linalg.eigvalsh(I))


def _resolve_masses(mol: Molecule,
                    hessian_result: HessianResult) -> np.ndarray:
    """Use the masses_amu the Hessian was built with if available,
    else look up standard atomic masses from the table."""
    if hessian_result.masses_amu is not None:
        return np.asarray(hessian_result.masses_amu, dtype=np.float64)
    masses = []
    for atom in mol.atoms:
        z = int(atom.Z)
        if z < len(_ATOMIC_MASSES) and _ATOMIC_MASSES[z] > 0:
            masses.append(_ATOMIC_MASSES[z])
        else:
            raise ValueError(
                f"compute_thermochemistry: no built-in mass for Z={z}; "
                "rebuild the Hessian with HessianFDOptions(atomic_masses_amu=...) "
                "or pre-fill masses on the result."
            )
    return np.asarray(masses, dtype=np.float64)


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_thermochemistry(
    mol: Molecule,
    hessian_result: HessianResult,
    *,
    options: Optional[ThermoOptions] = None,
) -> ThermoResult:
    """Compute ZPE, U(T), H(T,P), S(T,P), G(T,P) by combining the
    harmonic-oscillator vibrational partition function with the
    rigid-rotor and ideal-gas formulas.

    See module docstring for API examples and convention notes.

    Parameters
    ----------
    mol :
        Same molecule used to build ``hessian_result``. Provides
        atom positions for the inertia tensor and (when
        ``options.electronic_degeneracy is None``) the multiplicity
        for the electronic-degeneracy entropy.
    hessian_result :
        :class:`HessianResult` from :func:`compute_hessian_fd`.
        Must have ``frequencies_cm1``, ``is_linear``, and
        ``masses_amu`` populated (always true for results built
        through ``compute_hessian_fd``).
    options :
        :class:`ThermoOptions` for ``T``, ``P``, ``s``, and
        electronic-degeneracy override.
    """
    if options is None:
        options = ThermoOptions()

    T = float(options.temperature)
    P = float(options.pressure)
    sigma = int(options.symmetry_number)
    if T <= 0:
        raise ValueError("ThermoOptions: temperature must be > 0 K")
    if P <= 0:
        raise ValueError("ThermoOptions: pressure must be > 0 Pa")
    if sigma < 1:
        raise ValueError("ThermoOptions: symmetry_number must be >= 1")

    if options.electronic_degeneracy is None:
        g_elec = int(mol.multiplicity)
    else:
        g_elec = int(options.electronic_degeneracy)
    if g_elec < 1:
        raise ValueError("ThermoOptions: electronic_degeneracy must be >= 1")

    kT = _KB_HARTREE_PER_K * T  # Hartree per molecule

    # ---- Geometry / mass setup -----------------------------------------
    positions_bohr = np.asarray(
        [[a.xyz[0], a.xyz[1], a.xyz[2]] for a in mol.atoms],
        dtype=np.float64,
    )
    n_atoms = positions_bohr.shape[0]
    if n_atoms == 0:
        raise ValueError("compute_thermochemistry: empty molecule")
    masses_amu = _resolve_masses(mol, hessian_result)

    # ---- Vibrational ---------------------------------------------------
    freqs_cm1 = np.asarray(hessian_result.frequencies_cm1, dtype=np.float64)
    # Drop trans / rot zeros and imaginary modes from the partition
    # function. Threshold matches ir_intensities (1 cm⁻¹).
    vib_mask = freqs_cm1 > 1.0
    n_imag = int(np.sum(freqs_cm1 < -1.0))

    if n_atoms == 1:
        # Atomic gas: no rotation, no vibration.
        zpe = 0.0
        e_vib_thermal = 0.0
        s_vib = 0.0
        cv_vib = 0.0
    else:
        omega_au = freqs_cm1[vib_mask] * _CM_INV_TO_HARTREE  # Hartree
        zpe = 0.5 * float(np.sum(omega_au))
        # x = ℏw / kT  (dimensionless)
        x = omega_au / kT
        # Guard against overflow in exp(x) for very high modes (cold
        # vibrational populations); 1/(exp(x) - 1) -> 0 there anyway.
        with np.errstate(over='ignore'):
            inv_exp_minus = 1.0 / (np.exp(x) - 1.0)  # = e^(-x) / (1 - e^(-x))
        e_vib_thermal = float(np.sum(omega_au * inv_exp_minus))
        # S_vib = k_B S [x . n(x) - ln(1 - e^(-x))]   with n(x) = 1/(e^x - 1)
        #       = k_B S [x / (e^x - 1) - ln(1 - e^(-x))]
        s_vib = _KB_HARTREE_PER_K * float(np.sum(
            x * inv_exp_minus - np.log1p(-np.exp(-x))
        ))
        # C_v,vib = k_B S x^2 . e^x / (e^x - 1)^2 = k_B S x^2 . e^(-x) / (1 - e^(-x))^2
        with np.errstate(over='ignore'):
            cv_vib = _KB_HARTREE_PER_K * float(np.sum(
                x * x * np.exp(x) * inv_exp_minus ** 2
            ))

    # A partial (frozen-atom) Hessian describes an *anchored* system -- the
    # frozen atoms pin it in place, so there is no gas-phase translation or
    # rotation. Those partition functions are omitted and all 3M modes are
    # treated as vibrational (the standard adsorbate-on-frozen-slab model).
    is_vibrational_only = (
        getattr(hessian_result, "unfrozen_indices", None) is not None
    )

    # ---- Translational (Sackur-Tetrode) -------------------------------
    # All in SI then converted to Hartree.
    if is_vibrational_only:
        e_trans = s_trans = cv_trans = 0.0
    else:
        M_kg = float(np.sum(masses_amu)) * _AMU_TO_KG
        # Translational partition function per molecule:
        # q_trans = (2pi m k_B T / h^2)^(3/2) x V_per_molecule
        # V_per_molecule = k_B T / P  (ideal gas)
        q_trans = (
            (2.0 * np.pi * M_kg * _KB_J_PER_K * T / _PLANCK_J_S ** 2) ** 1.5
            * _KB_J_PER_K * T / P
        )
        s_trans = _KB_HARTREE_PER_K * (np.log(q_trans) + 2.5)
        e_trans = 1.5 * kT
        cv_trans = 1.5 * _KB_HARTREE_PER_K

    # ---- Rotational ---------------------------------------------------
    if is_vibrational_only:
        rotor_type = "frozen"   # anchored: no rigid-rotor partition
        e_rot = 0.0
        s_rot = 0.0
        cv_rot = 0.0
    elif n_atoms == 1:
        rotor_type = "atom"
        e_rot = 0.0
        s_rot = 0.0
        cv_rot = 0.0
    elif hessian_result.is_linear:
        rotor_type = "linear"
        I_eigs_au = _principal_moments_of_inertia(positions_bohr, masses_amu)
        # Two equal nonzero principal moments + one ≈ 0; take the larger ones.
        I_perp_au = 0.5 * (I_eigs_au[1] + I_eigs_au[2])
        I_perp_si = I_perp_au * _INERTIA_AU_TO_SI
        # Rotational constant B = h / (8pi^2 I)  (Hz)
        B_hz = _PLANCK_J_S / (8.0 * np.pi ** 2 * I_perp_si)
        q_rot = _KB_J_PER_K * T / (sigma * _PLANCK_J_S * B_hz)
        s_rot = _KB_HARTREE_PER_K * (np.log(q_rot) + 1.0)
        e_rot = kT
        cv_rot = _KB_HARTREE_PER_K
    else:
        rotor_type = "nonlinear"
        I_eigs_au = _principal_moments_of_inertia(positions_bohr, masses_amu)
        I_si = I_eigs_au * _INERTIA_AU_TO_SI
        B_hz = _PLANCK_J_S / (8.0 * np.pi ** 2 * I_si)
        # q_rot = (k_B T / h)^(3/2) x √pi / (s x √(B_a B_b B_c))
        q_rot = (
            (_KB_J_PER_K * T / _PLANCK_J_S) ** 1.5
            * np.sqrt(np.pi)
            / (sigma * np.sqrt(np.prod(B_hz)))
        )
        s_rot = _KB_HARTREE_PER_K * (np.log(q_rot) + 1.5)
        e_rot = 1.5 * kT
        cv_rot = 1.5 * _KB_HARTREE_PER_K

    # ---- Electronic ---------------------------------------------------
    s_elec = _KB_HARTREE_PER_K * np.log(g_elec)

    # ---- Totals -------------------------------------------------------
    s_total = s_trans + s_rot + s_vib + s_elec
    u_thermal = zpe + e_trans + e_rot + e_vib_thermal
    # Ideal gas H = U + PV = U + N k_B T -> per molecule + k_B T. An
    # anchored (partial-Hessian) system has no gas-phase PV term.
    h_thermal = u_thermal + (0.0 if is_vibrational_only else kT)
    g_thermal = h_thermal - T * s_total
    cv_total = cv_trans + cv_rot + cv_vib

    return ThermoResult(
        temperature=T, pressure=P,
        zpe=zpe,
        e_trans=e_trans, e_rot=e_rot, e_vib=e_vib_thermal,
        u_thermal=u_thermal, h_thermal=h_thermal, g_thermal=g_thermal,
        s_trans=s_trans, s_rot=s_rot, s_vib=s_vib, s_elec=s_elec,
        s_total=s_total,
        cv_trans=cv_trans, cv_rot=cv_rot, cv_vib=cv_vib, cv_total=cv_total,
        n_imaginary_modes_excluded=n_imag,
        rotor_type=rotor_type,
    )


__all__ = [
    "ThermoOptions",
    "ThermoResult",
    "compute_thermochemistry",
]
