"""FRAGMO initial guess -- assemble an SCF guess from converged fragment densities.

FRAGMO (``InitialGuess.FRAGMO``) seeds a molecular SCF from the *superposition
of converged fragment wavefunctions* (roadmap D2h, the fragment-MO assembly
guess). The user partitions the molecule into fragments; each fragment is
converged with its own SCF, and the fragments' converged density matrices are
assembled **block-diagonally** -- one diagonal block per fragment, inter-fragment
blocks left at zero -- into the guess density for the full system:

    D_guess = ⊕_F D_F ,

with each fragment block ``D_F`` placed into the supersystem AO rows/columns of
that fragment's atoms. Assembling the *density* (rather than carrying the MO
coefficients) needs no occupation bookkeeping and rides the same
``opts.read_density`` injection path the READ guess uses, so one implementation
covers RHF / UHF / RKS / UKS. For an open-shell supersystem each fragment
contributes its a and b blocks separately (a closed-shell fragment contributes
1/2 D_F to each spin).

For weakly-interacting assemblies -- hydrogen-bonded dimers, van-der-Waals
complexes, host-guest systems -- the fragments already carry almost all of the
electronic structure, so the assembled guess starts much closer to the
supersystem solution than SAD and converges in fewer SCF iterations. That is
the regime FRAGMO is for; for a strongly-coupled / covalently-bonded partition
the inter-fragment blocks matter and SAD/SAP are usually the better choice.

Periodic FRAGMO uses explicitly owned atom images through PeriodicFragment;
finite fragment orbitals are embedded with their Bloch translation phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import Atom, BasisSet, InitialGuess, Molecule


@dataclass
class Fragment:
    """One fragment of a FRAGMO partition.

    Parameters
    ----------
    atoms:
        0-based indices into the parent ``Molecule.atoms`` that belong to this
        fragment. Every parent atom must appear in exactly one fragment.
    charge:
        Net charge of the fragment (default 0). The fragment charges must sum
        to the parent molecule's charge so the electron count is conserved.
    multiplicity:
        Spin multiplicity 2S+1 of the fragment (default 1, singlet). A fragment
        with an odd electron count, or ``multiplicity > 1``, is converged with
        the unrestricted driver; its a/b densities feed the corresponding
        supersystem spin blocks.
    """

    atoms: Sequence[int]
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self) -> None:
        self.atoms = tuple(int(a) for a in self.atoms)
        self.charge = int(self.charge)
        self.multiplicity = int(self.multiplicity)
        if not self.atoms:
            raise ValueError("FRAGMO Fragment must contain at least one atom.")
        if self.multiplicity < 1:
            raise ValueError(
                "FRAGMO Fragment multiplicity must be >= 1, got "
                f"{self.multiplicity}."
            )


# ---- basis bookkeeping ----------------------------------------------------


def _shell_bf_size(shell) -> int:
    l = int(shell.l)
    return (2 * l + 1) if shell.pure else (l + 1) * (l + 2) // 2


def _atom_bf_indices(basis: BasisSet) -> dict:
    """Map ``atom_index -> list of AO indices`` owned by that atom, in AO order.

    Groups AO indices by each shell's ``atom_index`` rather than assuming one
    atom's shells are contiguous, so the assembly is correct for any shell
    ordering.
    """
    out: dict = {}
    offset = 0
    for sh in basis.shells():
        size = _shell_bf_size(sh)
        out.setdefault(int(sh.atom_index), []).extend(range(offset, offset + size))
        offset += size
    return out


def _guard_custom_basis(basis: BasisSet) -> None:
    name = basis.name
    if not name or name.startswith("<"):
        raise NotImplementedError(
            "FRAGMO currently requires a named basis set for its atomic "
            f"SCF references; the supplied basis is custom ({name!r}). "
            "Use a named basis set, or SAD/SAP for a custom basis."
        )


# ---- fragment construction + SCF ------------------------------------------


def _normalize_fragments(fragments) -> List[Fragment]:
    if not fragments:
        raise ValueError(
            "FRAGMO requires fragments=... -- a list of Fragment specs (or "
            "atom-index sequences). Example: "
            "run_rhf(mol, basis, opts, fragments=[[0, 1, 2], [3, 4, 5]])."
        )
    out: List[Fragment] = []
    for f in fragments:
        out.append(f if isinstance(f, Fragment) else Fragment(atoms=f))
    return out


def _validate_partition(fragments: Sequence[Fragment], molecule: Molecule) -> None:
    n_atoms = len(molecule.atoms)
    seen: dict = {}
    for fi, frag in enumerate(fragments):
        for a in frag.atoms:
            if a < 0 or a >= n_atoms:
                raise ValueError(
                    f"FRAGMO fragment {fi} references atom index {a}, outside "
                    f"the molecule's {n_atoms} atoms."
                )
            if a in seen:
                raise ValueError(
                    f"FRAGMO atom index {a} appears in both fragment {seen[a]} "
                    f"and fragment {fi}; fragments must be a disjoint partition."
                )
            seen[a] = fi
    missing = [a for a in range(n_atoms) if a not in seen]
    if missing:
        raise ValueError(
            "FRAGMO fragments must cover every atom exactly once; atoms "
            f"{missing} are not assigned to any fragment."
        )
    frag_charge = sum(f.charge for f in fragments)
    if frag_charge != molecule.charge:
        raise ValueError(
            f"FRAGMO fragment charges sum to {frag_charge} but the molecule "
            f"charge is {molecule.charge}; assign fragment charges so they add "
            "up (the total electron count must be conserved)."
        )


def _build_fragment(
    molecule: Molecule, basis: BasisSet, frag: Fragment
) -> Tuple[Molecule, BasisSet, List[int]]:
    """Return ``(frag_molecule, frag_basis, order)`` where ``order`` lists the
    parent atom indices in the fragment's own atom order."""
    order = list(frag.atoms)
    parent_atoms = molecule.atoms
    frag_atoms = [Atom(parent_atoms[i].Z, list(parent_atoms[i].xyz)) for i in order]
    frag_mol = Molecule(
        frag_atoms, charge=frag.charge, multiplicity=frag.multiplicity
    )
    # Preserve the parent's actual AO contractions and centers. Rebuilding
    # by name would silently discard per-atom shell changes and would make
    # the scattered density refer to a different basis.
    shells = []
    for local_atom, parent_atom in enumerate(order):
        for shell in basis.shells():
            if shell.atom_index == parent_atom:
                shell.atom_index = local_atom
                shells.append(shell)
    frag_basis = BasisSet(frag_mol, shells, basis.name)
    return frag_mol, frag_basis, order


def _parent_dft(options) -> Tuple[Optional[str], object]:
    """Return ``(functional, grid)`` for a KS supersystem, else ``(None, None)``.
    RKS/UKS options carry ``functional``; RHF/UHF options do not."""
    functional = getattr(options, "functional", None)
    grid = getattr(options, "grid", None)
    return functional, grid


def _frag_options(options, functional: Optional[str], grid, *, unrestricted: bool):
    """Build a fresh options object for a fragment SCF: same method/functional
    as the supersystem, a SAD guess (FRAGMO never recurses), and the parent's
    convergence tolerances so each fragment density is well converged."""
    from ._vibeqc_core import RHFOptions, RKSOptions, UHFOptions, UKSOptions

    is_dft = bool(functional)
    if unrestricted:
        o = UKSOptions() if is_dft else UHFOptions()
    else:
        o = RKSOptions() if is_dft else RHFOptions()
    if is_dft:
        o.functional = functional
        if grid is not None:
            o.grid = grid
    o.initial_guess = InitialGuess.SAD
    # Inherit the supersystem's convergence settings where present.
    for attr in ("conv_tol_energy", "conv_tol_grad", "max_iter",
                 "linear_dep_threshold"):
        val = getattr(options, attr, None)
        if val is not None:
            setattr(o, attr, val)
    return o


def _inherit_fragment_ecp(parent, target, fragment, order):
    """Restrict the actual supersystem operator to this fragment's atoms."""
    from .guess import guess_ecp_context

    context = guess_ecp_context(parent)
    atoms = list(fragment.atoms)
    if not context.active:
        return
    def belongs(xyz):
        return any(np.linalg.norm(np.asarray(a.xyz) - np.asarray(xyz)) < 1e-6 for a in atoms)
    if context.primitive_blocks:
        pairs = [(b, c) for b, c in zip(context.primitive_blocks, context.primitive_centers)
                 if belongs(c)]
        if pairs:
            target.ecp_primitive_blocks = [b for b, _ in pairs]
            target.ecp_primitive_centers = [c for _, c in pairs]
            target.ecp_effective_charges = [context.effective_charges[i] for i in order]
            target.ecp_total_ncore = round(sum(a.Z for a in atoms) - sum(target.ecp_effective_charges))
    else:
        centers = [c for c in context.xml_centers if belongs(c.xyz)]
        if centers:
            target.ecp_centers = centers
            target.ecp_library = context.xml_library


def _run_fragment_scf(
    frag_mol: Molecule, frag_basis: BasisSet, options, functional, grid, order
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converge one fragment. Returns ``(D_total, D_alpha, D_beta)`` in the
    fragment AO basis."""
    from . import run_rhf, run_rks, run_uhf, run_uks

    n_elec = frag_mol.n_electrons()
    mult = frag_mol.multiplicity
    closed = (mult == 1 and n_elec % 2 == 0)
    is_dft = bool(functional)

    if closed:
        o = _frag_options(options, functional, grid, unrestricted=False)
        _inherit_fragment_ecp(options, o, frag_mol, order)
        run = run_rks if is_dft else run_rhf
        r = run(frag_mol, frag_basis, o)
        if not r.converged:
            raise RuntimeError(
                "FRAGMO: a fragment SCF did not converge "
                f"({frag_mol.n_electrons()} e⁻, mult={mult}). Tighten/loosen "
                "the supersystem's SCF settings, or recheck the fragment "
                "charge/multiplicity."
            )
        D = np.asarray(r.density, dtype=float)
        return D, 0.5 * D, 0.5 * D

    o = _frag_options(options, functional, grid, unrestricted=True)
    _inherit_fragment_ecp(options, o, frag_mol, order)
    run = run_uks if is_dft else run_uhf
    r = run(frag_mol, frag_basis, o)
    if not r.converged:
        raise RuntimeError(
            "FRAGMO: an open-shell fragment SCF did not converge "
            f"({n_elec} e⁻, mult={mult}). Recheck the fragment "
            "charge/multiplicity."
        )
    Da = np.asarray(r.density_alpha, dtype=float)
    Db = np.asarray(r.density_beta, dtype=float)
    return Da + Db, Da, Db


# ---- block-diagonal assembly ----------------------------------------------


def _scatter_block(
    target: np.ndarray,
    parent_idx_map: dict,
    frag_basis: BasisSet,
    order: Sequence[int],
    block: np.ndarray,
) -> None:
    """Place a fragment AO ``block`` into ``target`` at the supersystem rows/cols
    of the fragment's atoms (preserving intra-fragment cross terms)."""
    frag_idx_map = _atom_bf_indices(frag_basis)
    dst: List[int] = []
    for k, parent_atom in enumerate(order):
        p_aos = parent_idx_map[parent_atom]
        f_aos = frag_idx_map.get(k, [])
        if len(p_aos) != len(f_aos):
            raise RuntimeError(
                f"FRAGMO internal error: fragment atom {k} (parent atom "
                f"{parent_atom}) has {len(f_aos)} AOs but the parent assigns "
                f"{len(p_aos)} -- basis mismatch."
            )
        dst.extend(p_aos)
    idx = np.asarray(dst, dtype=int)
    if block.shape != (idx.size, idx.size):
        raise RuntimeError(
            f"FRAGMO internal error: fragment block is {block.shape}, expected "
            f"{(idx.size, idx.size)}."
        )
    target[np.ix_(idx, idx)] = block


# ---- public resolvers (called by the run_* wrappers) ----------------------


def resolve_fragmo_density_closed(
    options, molecule: Molecule, basis: BasisSet, fragments
) -> np.ndarray:
    """Closed-shell supersystem: return the assembled block-diagonal total
    density (nbf x nbf) to inject as ``opts.read_density``."""
    from .guess import guess_ecp_context, validate_guess_ecp
    validate_guess_ecp(InitialGuess.FRAGMO, guess_ecp_context(options), molecule)
    frags = _normalize_fragments(fragments)
    _validate_partition(frags, molecule)
    _guard_custom_basis(basis)
    functional, grid = _parent_dft(options)
    parent_idx_map = _atom_bf_indices(basis)
    nbf = basis.nbasis
    total = np.zeros((nbf, nbf), dtype=float)
    for frag in frags:
        frag_mol, frag_basis, order = _build_fragment(molecule, basis, frag)
        d_total, _, _ = _run_fragment_scf(
            frag_mol, frag_basis, options, functional, grid, order
        )
        _scatter_block(total, parent_idx_map, frag_basis, order, d_total)
    return 0.5 * (total + total.T)


def resolve_fragmo_densities_open(
    options, molecule: Molecule, basis: BasisSet, fragments
) -> Tuple[np.ndarray, np.ndarray]:
    """Open-shell supersystem: return ``(D_alpha, D_beta)`` assembled
    block-diagonally to inject as ``opts.read_density_{alpha,beta}``."""
    from .guess import guess_ecp_context, validate_guess_ecp
    validate_guess_ecp(InitialGuess.FRAGMO, guess_ecp_context(options), molecule)
    frags = _normalize_fragments(fragments)
    _validate_partition(frags, molecule)
    _guard_custom_basis(basis)
    functional, grid = _parent_dft(options)
    parent_idx_map = _atom_bf_indices(basis)
    nbf = basis.nbasis
    d_alpha = np.zeros((nbf, nbf), dtype=float)
    d_beta = np.zeros((nbf, nbf), dtype=float)
    for frag in frags:
        frag_mol, frag_basis, order = _build_fragment(molecule, basis, frag)
        _, da, db = _run_fragment_scf(
            frag_mol, frag_basis, options, functional, grid, order
        )
        _scatter_block(d_alpha, parent_idx_map, frag_basis, order, da)
        _scatter_block(d_beta, parent_idx_map, frag_basis, order, db)
    return 0.5 * (d_alpha + d_alpha.T), 0.5 * (d_beta + d_beta.T)


__all__ = [
    "Fragment",
    "resolve_fragmo_density_closed",
    "resolve_fragmo_densities_open",
]


@dataclass
class PeriodicFragment(Fragment):
    """Finite fragment with one owned image of each listed unit-cell atom.

    ``images`` contains integer lattice translations in ``atoms`` order.
    Omission means the home cell. Every cell atom must have exactly one
    owner across the partition. A common translation of a fragment has no
    effect on its Bloch density. ``spin_orientation=-1`` reverses its spin.
    """
    images: Optional[Sequence[Sequence[int]]] = None
    spin_orientation: int = 1

    def __post_init__(self):
        super().__post_init__()
        images = np.zeros((len(self.atoms), 3)) if self.images is None else np.asarray(self.images)
        if (images.shape != (len(self.atoms), 3) or not np.all(np.isfinite(images))
                or not np.all(images == np.rint(images))):
            raise ValueError("FRAGMO images must be one integer lattice triplet per atom")
        self.images = tuple(tuple(int(v) for v in row) for row in images)
        if self.spin_orientation not in (-1, 1):
            raise ValueError("FRAGMO spin_orientation must be +1 or -1")


def resolve_periodic_fragmo_source(options, system, basis, kmesh, fragments):
    """Converge finite fragments and embed their translated AOs in Bloch space.

    This is a localized fragment superposition, without a periodic fragment
    SCF or inter-fragment polarization. Translation phases preserve bonded
    density blocks across unit-cell boundaries. The consuming driver applies
    the target overlap and weighted spin populations, as for a READ source.
    """
    from types import SimpleNamespace
    from .guess import guess_ecp_context
    specs = _normalize_fragments(fragments)
    mol = system.unit_cell_molecule()
    _validate_partition(specs, mol)
    _guard_custom_basis(basis)
    image_by_atom = np.zeros((len(mol.atoms), 3), dtype=int)
    for spec in specs:
        image_by_atom[list(spec.atoms)] = getattr(spec, "images", np.zeros((len(spec.atoms), 3)))
    if np.any(image_by_atom[:, int(system.dim):] != 0):
        raise ValueError("FRAGMO images may translate only periodic lattice axes")
    translations = image_by_atom @ np.asarray(system.lattice).T
    atoms = [Atom(a.Z, np.asarray(a.xyz) + shift) for a, shift in zip(mol.atoms, translations)]
    unwrapped = Molecule(atoms, mol.charge, mol.multiplicity)
    shells = []
    for shell in basis.shells():
        shell.origin = np.asarray(shell.origin) + translations[shell.atom_index]
        shells.append(shell)
    unwrapped_basis = BasisSet(unwrapped, shells, basis.name)
    # Translate each ECP along with its owning atom. Keep its physical Z,
    # primitive operator and effective charge unchanged.
    context = guess_ecp_context(options)
    parent = SimpleNamespace(**{key: getattr(options, key) for key in
        ("functional", "grid", "conv_tol_energy", "conv_tol_grad", "max_iter", "linear_dep_threshold")
        if hasattr(options, key)})
    def moved(center):
        matches = [i for i, a in enumerate(mol.atoms) if np.linalg.norm(np.asarray(a.xyz) - center) < 1e-6]
        if len(matches) != 1:
            raise ValueError("FRAGMO ECP center must have exactly one atom owner")
        return np.asarray(center) + translations[matches[0]]
    if context.primitive_blocks:
        parent.ecp_primitive_blocks = context.primitive_blocks
        parent.ecp_primitive_centers = [moved(c) for c in context.primitive_centers]
        parent.ecp_effective_charges = context.effective_charges
        parent.ecp_total_ncore = context.total_ncore
    elif context.xml_centers:
        from ._vibeqc_core import ECPCenter
        parent.ecp_centers = [ECPCenter(c.Z, moved(c.xyz)) for c in context.xml_centers]
        parent.ecp_library = context.xml_library
    functional, grid = _parent_dft(parent)
    ao_map = _atom_bf_indices(unwrapped_basis)
    da = np.zeros((basis.nbasis, basis.nbasis))
    db = np.zeros_like(da)
    for spec in specs:
        fragment, fragment_basis, order = _build_fragment(unwrapped, unwrapped_basis, spec)
        _, a, b = _run_fragment_scf(fragment, fragment_basis, parent, functional, grid, order)
        if getattr(spec, "spin_orientation", 1) == -1:
            a, b = b, a
        _scatter_block(da, ao_map, fragment_basis, order, a)
        _scatter_block(db, ao_map, fragment_basis, order, b)
    ao_translations = np.zeros((basis.nbasis, 3))
    for atom, indices in _atom_bf_indices(basis).items():
        ao_translations[indices] = translations[atom]
    alpha, beta = [], []
    for k in kmesh.kpoints:
        phase = np.exp(-1j * (ao_translations @ np.asarray(k)))
        gauge = phase[:, None] * phase.conj()[None, :]
        alpha.append(gauge * da)
        beta.append(gauge * db)
    return SimpleNamespace(
        density_alpha=alpha, density_beta=beta, density=[a + b for a, b in zip(alpha, beta)],
        restart_basis=basis, restart_lattice=np.asarray(system.lattice).copy(),
        restart_kpoints=np.asarray(kmesh.kpoints).copy(), restart_weights=np.asarray(kmesh.weights).copy())


__all__ += ["PeriodicFragment", "resolve_periodic_fragmo_source"]
