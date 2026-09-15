"""``CCMSystem`` -- the cluster geometry + basis container for the AICCM.

A :class:`CCMSystem` wraps a periodic unit cell plus a cluster size
``nrep = (N1, N2, N3)`` into everything the Cyclic Cluster Model needs:

* the **reference supercell** (the "real" atoms ``u_j`` of the cluster),
  built by replicating the unit cell ``N1*N2*N3`` times;
* the **cluster lattice** ``L_c = {N1 a1, N2 a2, N3 a3}`` whose Wigner-
  Seitz cell (centered on each atom) defines the interaction region;
* the **supercell AO basis** and the AO->atom map;
* a tuned lattice-sum cutoff and a cache of the WSSC two-center weights.

It holds *geometry and integrals only* -- no SCF state. Downstream CCM
integral builders (:mod:`vibeqc.periodic.ccm.integrals`) consume it, and
the resulting weighted ``S``/``Hcore``/ERIs feed the ordinary vibe-qc
method kernels. Keeping the model in this geometry layer is what makes
the CCM method-general (HF, DFT, MP2, CC, CAS all reuse it unchanged).

Reference: Peintinger & Bredow, *J. Comput. Chem.* **35**, 839 (2014),
doi:10.1002/jcc.23550 (Sec. Theory, eqs 1-7).
"""

from __future__ import annotations

import operator
from itertools import product
from numbers import Real
from typing import Optional, Sequence

import numpy as np

# vibe-qc is fully initialised by the time this submodule is imported
# (importing vibeqc.periodic.ccm requires the parent package first), so
# these absolute names resolve cleanly and without an import cycle.
from vibeqc import (  # type: ignore  noqa: E402
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
)
from vibeqc._vibeqc_core import Atom, Molecule  # noqa: E402

from .wigner_seitz import (
    _minkowski_reduce,
    first_shell_vectors,
    kspacing_for_interaction_range,
    minimum_image,
    nrep_for_interaction_range,
    wsc_inscribed_radius,
)

__all__ = ["CCMSystem"]

# CODATA bohr radius: 1 bohr = 0.529177210903 Å (matches the ``BOHR`` constant
# used throughout the CCM examples/docs). Backs the Å convenience on the
# interaction-range constructor.
_ANG_TO_BOHR = 1.0 / 0.529177210903


def _validate_nrep(nrep: Sequence[int]) -> tuple[int, int, int]:
    """Exact-identity validation of an explicit cluster mesh.

    Issue #390: an explicit ``nrep`` used to be coerced with ``int(x)``
    before validation, so ``(1.9, 1, 1)`` became ``(1, 1, 1)``, ``True``
    became ``1``, and ``array([2.7, 1, 1])`` became ``(2, 1, 1)`` -- the
    user silently ran a different cyclic Hamiltonian and normalization
    than the one requested. The mesh is now accepted only as exactly
    three non-boolean integer components of value at least one, with
    Python and NumPy integers both accepted (via ``operator.index``, the
    protocol that also underlies tuple indexing); every other shape --
    floats (including integer-valued ones), strings, booleans,
    non-finite values, and wrong lengths -- raises a bounded ValueError
    before any supercell, basis, or integral work.
    """
    if isinstance(nrep, (str, bytes)):
        raise ValueError(
            f"nrep must be a sequence of 3 integers, got {nrep!r}"
        )
    try:
        components = list(nrep)
    except TypeError as exc:
        raise ValueError(
            f"nrep must be a sequence of 3 integers, got {nrep!r}"
        ) from exc
    if len(components) != 3:
        raise ValueError(f"nrep must have 3 components, got {nrep!r}")
    checked: list[int] = []
    for x in components:
        if isinstance(x, bool):
            raise ValueError(
                f"nrep components must be integers, got {x!r} (bool)"
            )
        try:
            checked.append(operator.index(x))
        except TypeError as exc:
            raise ValueError(
                f"nrep components must be integers, got {x!r}"
            ) from exc
    if min(checked) < 1:
        raise ValueError(f"nrep components must be >= 1, got {tuple(checked)}")
    return (checked[0], checked[1], checked[2])


def _ao_atom_map(basis) -> np.ndarray:
    """Map each AO index to the index of the atom it is centered on.

    AO ordering follows libint's shell ordering (the order shells appear
    in ``basis.shells()``); each shell contributes ``2l+1`` spherical or
    ``(l+1)(l+2)/2`` Cartesian functions.
    """
    ao_atom = []
    for sh in basis.shells():
        l = int(sh.l)
        n_ao = (2 * l + 1) if sh.pure else (l + 1) * (l + 2) // 2
        ao_atom.extend([int(sh.atom_index)] * n_ao)
    arr = np.asarray(ao_atom, dtype=int)
    if arr.size != basis.nbasis:
        raise RuntimeError(
            f"AO->atom map size {arr.size} != basis.nbasis {basis.nbasis}"
        )
    return arr


class CCMSystem:
    """Cyclic-cluster geometry, basis, and WSSC weights for one cluster.

    The cluster size is specified one of two ways (exactly one):

    * an **interaction range** ``interaction_range`` (bohr) / ``interaction_range_ang``
      (Å) -- the recommended, physically-meaningful knob. The minimal cluster
      whose supercell Wigner-Seitz cell encloses a sphere of that radius around
      every atom is derived automatically (:func:`nrep_for_interaction_range`).
      This is the geometric **real-space dual of a character-mesh density**: a
      larger radius corresponds to a denser mesh (``Δk = pi/R_c``). It does not
      assert equality between different CCM constructions.
    * an explicit mesh ``nrep = (N1, N2, N3)`` -- the advanced override, the size
      along ``a1, a2, a3`` given directly.

    Parameters
    ----------
    unit_system : PeriodicSystem
        The periodic unit cell (primitive or conventional).
    nrep : (int, int, int), optional
        Explicit cluster size along ``a1, a2, a3`` (advanced override). Mutually
        exclusive with ``interaction_range`` / ``interaction_range_ang``.
    basis : str
        Basis-set name (e.g. ``"sto-3g"``, ``"pob-TZVP-rev2"``).
    interaction_range : float, optional
        Real-space WSC interaction radius ``R_c`` (bohr). ``nrep`` is derived so
        the cluster WS cell encloses a sphere of this radius around every atom.
    interaction_range_ang : float, optional
        Same, in ångström (converted to bohr internally).
    weight_tol_bohr : float
        Boundary-tie tolerance for the WSSC weighting (see
        :func:`wigner_seitz_weights`).
    lattice_sum_options : LatticeSumOptions, optional
        Override the auto-tuned lattice-sum options. The default sets a
        cutoff large enough to enclose the first neighbour shell of the
        cluster lattice (so every pair's minimum image is built).

    Attributes
    ----------
    nrep : (int, int, int)
        The cluster mesh actually used (given or derived).
    interaction_range : float or None
        The requested ``R_c`` (bohr) if the radius path was used, else ``None``.
    wsc_inscribed_radius : float
        The *achieved* WS inscribed-sphere radius ``r_in = l₁/2`` of the cluster
        lattice (bohr) -- always ``>= interaction_range`` when a radius was given.
    kspacing_equiv : float
        The equivalent uniform k-spacing ``Δk = pi/r_in`` (bohr⁻¹) a Bloch
        calculation would use for the same real-space reach.
    """

    def __init__(
        self,
        unit_system,
        nrep: Optional[Sequence[int]] = None,
        basis: Optional[str] = None,
        *,
        interaction_range: Optional[float] = None,
        interaction_range_ang: Optional[float] = None,
        weight_tol_bohr: float = 1e-6,
        lattice_sum_options: Optional["LatticeSumOptions"] = None,
    ):
        if basis is None:
            raise ValueError("a basis-set name is required (e.g. basis='sto-3g')")
        self.unit_system = unit_system
        self.basis_name = str(basis)
        self.weight_tol_bohr = float(weight_tol_bohr)

        # Lattice-vector convention. vibe-qc's PeriodicSystem stores lattice
        # vectors as the *columns* of the matrix (cpp/include/vibeqc/periodic.hpp:
        # "Columns = Cartesian lattice vectors"; the real-space lattice sum sets
        # r_cart = lattice @ index). For geometry we keep the vectors as rows
        # (``unit_vectors[j] = a_j``) -- the natural form for numpy -- and only
        # transpose back to the column form when handing matrices to the engine.
        self.unit_vectors = np.asarray(
            unit_system.lattice, dtype=float
        ).T  # (3,3) rows = a_j

        # Resolve the cluster size: explicit nrep XOR a real-space range.
        if interaction_range is not None and interaction_range_ang is not None:
            raise ValueError(
                "give interaction_range (bohr) or interaction_range_ang (Å), not both"
            )
        radius_bohr = None
        if interaction_range is not None:
            radius_bohr = float(interaction_range)
        elif interaction_range_ang is not None:
            radius_bohr = float(interaction_range_ang) * _ANG_TO_BOHR
        if nrep is None and radius_bohr is None:
            raise ValueError(
                "specify the cluster size: pass interaction_range=R_c (bohr) "
                "[recommended] or an explicit nrep=(N1,N2,N3)"
            )
        if nrep is not None and radius_bohr is not None:
            raise ValueError(
                "give an interaction range OR an explicit nrep, not both; nrep "
                "is the advanced override -- pass it alone to bypass the radius "
                "derivation"
            )
        self.interaction_range = radius_bohr  # requested R_c (bohr) or None
        if nrep is None:
            nrep = nrep_for_interaction_range(
                self.unit_vectors, radius_bohr, dim=int(unit_system.dim)
            )
        # Exact-identity validation (issue #390): no int(x) coercion. The
        # derived path returns Python ints already; routing it through the
        # same gate keeps one validation contract for every caller.
        self.nrep = _validate_nrep(nrep)

        # Cluster vectors A_j = N_j * a_j (rows).
        self.cluster_vectors = (
            np.asarray(self.nrep, dtype=float)[:, None] * self.unit_vectors
        )
        # Achieved WS inscribed radius r_in = l₁/2 of the cluster lattice, and
        # the equivalent k-spacing Δk = pi/r_in (the radius<->k-mesh duality).
        self.wsc_inscribed_radius = wsc_inscribed_radius(self.cluster_vectors)
        self.kspacing_equiv = kspacing_for_interaction_range(self.wsc_inscribed_radius)

        # Reference supercell molecule (the "real" cluster atoms u_j).
        self.supercell = self._build_supercell(unit_system)

        # Engine lattice for the real-space integral builders: columns = cluster
        # vectors, i.e. cluster_vectors transposed.
        self.cluster_lattice = self.cluster_vectors.T
        self.cluster_system = PeriodicSystem(
            int(unit_system.dim),
            self.cluster_lattice,
            list(self.supercell.atoms),
            charge=int(self.supercell.charge),
            multiplicity=int(self.supercell.multiplicity),
        )

        self.basis = BasisSet(self.supercell, self.basis_name)
        self.ao_atom = _ao_atom_map(self.basis)

        # Backstop guard: every supercell atom must contribute at least one
        # basis function. A basis that silently omits an element -- e.g. the
        # bundled cc-pVDZ.g94 covers only H-Ar, so Cs (Z=55) gets *zero* AOs --
        # otherwise crashes far downstream in build_padded_cluster with an
        # opaque ``KeyError(0)`` (pad atom 0 has no AO range). ``BasisSet``
        # itself refuses that case first with its own RuntimeError naming the
        # bare Z (#234), so this only fires for a basis that constructs yet
        # leaves an atom without AOs. Name the element(s): the real fix is a
        # basis/ECP that covers them.
        _sc_atoms = list(self.supercell.atoms)
        _atoms_with_ao = {int(a) for a in self.ao_atom}
        _missing = [A for A in range(len(_sc_atoms)) if A not in _atoms_with_ao]
        if _missing:
            from vibeqc.basis_crystal import _ELEMENT_SYMBOLS as _SYMS

            _bare_Z = sorted({int(_sc_atoms[A].Z) for A in _missing})
            _names = ", ".join(
                f"{_SYMS[z]} (Z={z})" if 0 < z < len(_SYMS) else f"Z={z}"
                for z in _bare_Z
            )
            raise ValueError(
                f"basis {self.basis_name!r} provides no basis functions for "
                f"element(s) {_names}: {len(_missing)} of {len(_sc_atoms)} "
                f"supercell atoms would carry zero AOs. The bundled .g94 likely "
                f"does not cover this element (e.g. cc-pVDZ covers only "
                f"H-Ar); supply a basis/ECP that includes it before building a "
                f"CCMSystem."
            )

        self.atom_positions = np.array(
            [list(at.xyz) for at in self.supercell.atoms], dtype=float
        )

        # Decompose each supercell atom into (unit-cell index (i,j,k), basis
        # atom beta). build_supercell_molecule fills atoms in the order
        # ``for i: for j: for k: for beta`` so the linear index factorises as
        # ((i*N1+j)*N2+k)*n_beta + beta. This decomposition is what lets the
        # WSSC weights be computed once per translational class (see
        # cell_weight_matrices) instead of once per atom pair.
        self.n_basis_atoms = len(unit_system.unit_cell)
        self.unit_atom_pos = np.array(
            [list(at.xyz) for at in unit_system.unit_cell], dtype=float
        )
        self.atom_beta, self.atom_cell = self._decompose_atoms()

        self.lattice_options = lattice_sum_options or self._default_lattice_options()

        # Caches.
        self._cell_weights = None  # dict {g: (n_atoms, n_atoms) weight matrix}

    def _build_supercell(self, unit_system):
        """Replicate the unit cell into the reference supercell molecule.

        Atom order is ``for i: for j: for k: for beta`` (matching
        :func:`vibeqc.periodic_megacell_mp2.build_supercell_molecule`), which
        :meth:`_decompose_atoms` relies on. Unlike the megacell builder, the
        multiplicity is set parity-correct (closed shell when the electron
        count is even, doublet otherwise) so CCM works for odd-electron and
        open-shell clusters too, not just closed-shell ones.

        **Explicit high spin (2026-07-16):** a unit cell carrying an explicit
        multiplicity with ``n_unpaired >= 2`` (e.g. a triplet) is replicated
        **ferromagnetically**: ``n_unpaired_supercell = n_unpaired_unit * N_c``
        (at ``(1,1,1)`` this is exactly the explicit multiplicity). This is the
        same per-unit-cell spin convention the multi-k open-shell drivers use
        (see :func:`~vibeqc.periodic.ccm.ri.run_ccm_uks_gdf`'s spin-bookkeeping
        note); antiferromagnetic or other spin arrangements need an explicit
        supercell-level spin state and are not expressible via ``nrep``
        replication. Before this rule, an explicit ``multiplicity >= 3`` was
        **silently discarded** in favour of the parity default — a
        wrong-number trap (the cluster ran closed-shell). ``n_unpaired <= 1``
        cells keep the parity behaviour exactly (a doublet cell on an
        even-cell cluster still pairs up — the historical, pinned convention).
        """
        n1, n2, n3 = self.nrep
        a = self.unit_vectors  # rows = unit-cell lattice vectors a_j
        cell_atoms = list(unit_system.unit_cell)
        out = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    shift = i * a[0] + j * a[1] + k * a[2]
                    for at in cell_atoms:
                        out.append(
                            Atom(
                                int(at.Z),
                                (np.asarray(at.xyz, dtype=float) + shift).tolist(),
                            )
                        )
        charge = int(unit_system.charge) * (n1 * n2 * n3)
        n_elec = sum(int(at.Z) for at in out) - charge
        n_unpaired_unit = int(unit_system.multiplicity) - 1
        if n_unpaired_unit >= 2:
            # Explicit high spin: validate the UNIT cell's own spin state
            # before replicating. Without this, an impossible request (e.g.
            # 3 electrons with multiplicity 3) would raise at odd N_c but
            # silently become a parity-consistent different state at even
            # N_c (Li mult-3 at (2,1,1) -> a quintet) -- the request must
            # fail loudly at construction, independent of cluster size.
            n_elec_unit = n_elec // (n1 * n2 * n3)
            if ((n_elec_unit - n_unpaired_unit) % 2 != 0
                    or n_unpaired_unit > n_elec_unit):
                raise ValueError(
                    f"CCMSystem: unit-cell electron count {n_elec_unit} and "
                    f"multiplicity {unit_system.multiplicity} are "
                    "incompatible (n_alpha/n_beta non-integer)."
                )
            # Ferromagnetic replication (see docstring).
            multiplicity = n_unpaired_unit * (n1 * n2 * n3) + 1
        else:
            multiplicity = 1 if (n_elec % 2 == 0) else 2
        return Molecule(out, charge, multiplicity)

    def _decompose_atoms(self):
        nb = self.n_basis_atoms
        n1, n2, n3 = self.nrep
        lin = np.arange(self.atom_positions.shape[0])
        beta = lin % nb
        t = lin // nb
        k = t % n3
        t //= n3
        j = t % n2
        i = t // n2
        return beta.astype(int), np.stack([i, j, k], axis=1).astype(int)

    # -- geometry-derived sizes -------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return self.atom_positions.shape[0]

    @property
    def n_cells(self) -> int:
        """Number of unit cells in the cluster (``N1*N2*N3``)."""
        return self.nrep[0] * self.nrep[1] * self.nrep[2]

    @property
    def nbf(self) -> int:
        return int(self.basis.nbasis)

    # -- constructors -----------------------------------------------------------
    @classmethod
    def from_periodic(
        cls,
        system,
        nrep: Optional[Sequence[int]] = None,
        basis: Optional[str] = None,
        **kwargs,
    ):
        """Build from a :class:`PeriodicSystem` unit cell.

        ``nrep`` may be omitted in favour of ``interaction_range=`` (bohr) /
        ``interaction_range_ang=`` (Å) in ``kwargs``.
        """
        return cls(system, nrep, basis, **kwargs)

    @classmethod
    def from_interaction_range(
        cls,
        unit_system,
        interaction_range: float,
        basis: str,
        *,
        units: str = "bohr",
        **kwargs,
    ):
        """Build a cluster sized by a real-space interaction radius.

        The ergonomic radius constructor: the minimal cluster whose supercell
        Wigner-Seitz cell encloses a sphere of ``interaction_range`` around every
        atom is derived (:func:`nrep_for_interaction_range`). This is the
        real-space dual of choosing a k-mesh density (``Δk = pi/R_c``).

        Parameters
        ----------
        unit_system : PeriodicSystem
            The unit cell.
        interaction_range : float
            WSC interaction radius ``R_c`` in ``units``.
        basis : str
            Basis-set name.
        units : {"bohr", "angstrom", "ang", "A"}
            Units of ``interaction_range`` (default bohr).
        """
        u = str(units).lower()
        if u in ("bohr", "au", "a.u."):
            return cls(
                unit_system,
                None,
                basis,
                interaction_range=float(interaction_range),
                **kwargs,
            )
        if u in ("angstrom", "ang", "a", "å"):
            return cls(
                unit_system,
                None,
                basis,
                interaction_range_ang=float(interaction_range),
                **kwargs,
            )
        raise ValueError(f"unknown units {units!r}; use 'bohr' or 'angstrom'")

    @classmethod
    def from_ase(
        cls,
        atoms,
        nrep: Optional[Sequence[int]] = None,
        basis: Optional[str] = None,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs,
    ):
        """Build from an ASE ``Atoms`` unit cell (fully 3D-periodic).

        ``nrep`` may be omitted in favour of ``interaction_range=`` (bohr) /
        ``interaction_range_ang=`` (Å) in ``kwargs``.

        ASE stores lattice vectors as the **rows** of ``atoms.cell``; vibe-qc's
        ``PeriodicSystem`` stores them as **columns**, so the cell is transposed
        here. (NB: ``vibeqc.ase_periodic.atoms_to_periodic_system`` omits this
        transpose, which transposes non-orthogonal cells -- flagged in
        ``handovers/HANDOVER_AICCM.md`` for the periodic-SCF owners; we build the system
        directly so CCM is correct for every lattice regardless.)
        """
        from ase.units import Bohr

        lattice_cols = np.asarray(atoms.cell.array, dtype=float).T / Bohr
        vq_atoms = [
            Atom(int(z), (np.asarray(p, dtype=float) / Bohr).tolist())
            for z, p in zip(atoms.numbers, atoms.positions)
        ]
        system = PeriodicSystem(
            3, lattice_cols, vq_atoms, charge=charge, multiplicity=multiplicity
        )
        return cls(system, nrep, basis, **kwargs)

    # -- lattice-sum setup ------------------------------------------------------
    def _default_lattice_options(self) -> "LatticeSumOptions":
        """Lattice-sum options with a cutoff enclosing the first shell.

        The minimum image of any pair within the supercell is reached by a
        cluster-lattice translation no longer than the first-shell corner
        ``max|±a1±a2±a3|``; a small margin guarantees every such cell is
        enumerated by the integral builder.
        """
        opts = LatticeSumOptions()
        corner = float(
            np.linalg.norm(first_shell_vectors(self.cluster_vectors), axis=1).max()
        )
        opts.cutoff_bohr = max(float(opts.cutoff_bohr), 1.05 * corner)
        return opts

    # -- WSSC weights -----------------------------------------------------------
    def cell_weight_matrices(self) -> dict:
        """WSSC two-center weights, grouped by minimum-image cluster cell.

        Returns ``{g: W}`` where ``g`` is an integer cluster-lattice cell
        index and ``W`` is an ``(n_atoms, n_atoms)`` matrix of weights such
        that ``M^CCM = sum_g W[A,B] * M(g)[A,B]`` for any two-center lattice
        operator ``M`` (eqs 5-6). Only cells that are a minimum image of
        some pair appear.

        Efficient by construction: the minimum image is computed once per
        translational class ``(beta_A, beta_B, delta_cell)`` -- there are
        ``O(n_beta * n_atoms)`` of these (linear in the cluster size), not
        ``O(n_atoms**2)`` -- and the per-class weights are then broadcast to
        the atom grid by a single vectorised gather. Result is cached.
        """
        if self._cell_weights is not None:
            return self._cell_weights

        N = np.asarray(self.nrep, dtype=int)
        nb = self.n_basis_atoms
        a_unit = self.unit_vectors  # rows = unit-cell lattice vectors a_j
        rpos = self.unit_atom_pos

        # Translational classes: delta_cell components in [-(N_i-1), N_i-1].
        off = N - 1
        dim = 2 * N - 1
        dcs = np.array(
            list(product(*[range(-(N[i] - 1), N[i]) for i in range(3)])), dtype=int
        )  # (P, 3)
        P = dcs.shape[0]

        # Unique class displacements d = (r_betaA - r_betaB) + delta_cell @ a_unit,
        # stacked in (betaA, betaB, p) order.
        shift_cart = dcs @ a_unit  # (P, 3)
        disps = (
            rpos[:, None, None, :]
            - rpos[None, :, None, :]
            + shift_cart[None, None, :, :]
        ).reshape(-1, 3)

        reduced = _minkowski_reduce(self.cluster_vectors)
        cells, weights = minimum_image(
            disps,
            self.cluster_vectors,
            tol_bohr=self.weight_tol_bohr,
            reduced_basis=reduced,
        )

        # Accumulate per-cell class tables T[g] of shape (nb, nb, P). The loop
        # is over the O(n_beta**2 * P) classes (linear in cluster size), each
        # contributing to 1 cell (interior) or n cells (boundary tie).
        tables: dict = {}
        lin = 0
        for bA in range(nb):
            for bB in range(nb):
                for p in range(P):
                    for c, w in zip(cells[lin], weights[lin]):
                        g = (int(c[0]), int(c[1]), int(c[2]))
                        Tg = tables.get(g)
                        if Tg is None:
                            Tg = np.zeros((nb, nb, P))
                            tables[g] = Tg
                        Tg[bA, bB, p] += w
                    lin += 1

        # Gather each class table to the (n_atoms, n_atoms) atom grid. The
        # delta_cell of atom pair (A, B) indexes the class axis.
        dcell = self.atom_cell[:, None, :] - self.atom_cell[None, :, :]  # (na, na, 3)
        pidx = ((dcell[..., 0] + off[0]) * dim[1] + (dcell[..., 1] + off[1])) * dim[
            2
        ] + (dcell[..., 2] + off[2])  # (na, na)
        beta = self.atom_beta
        Wg = {g: Tg[beta[:, None], beta[None, :], pidx] for g, Tg in tables.items()}
        self._cell_weights = Wg
        return Wg

    # -- diagnostics ------------------------------------------------------------
    def check_overlap_spectrum(self, threshold: float = 1e-5):
        """Classify the finite-cluster CCM overlap spectrum (Fig. 6).

        Small clusters or diffuse bases can drive the lowest eigenvalue of
        ``S^CCM`` toward zero or materially negative, making the unscreened
        finite Γ-point approximation unphysical (Peintinger & Bredow 2014,
        Fig. 6 and Sec. "Critical eigenvalues of the overlap matrix").

        This is a strict explicit diagnostic, not the SCF refusal gate. The
        production drivers canonically screen directions at or below their
        ``lindep_tol`` and fail closed only when the retained space cannot hold
        the occupied manifold. This helper distinguishes materially indefinite
        spectra from numerical non-positivity and positive near-singularity so
        its advice describes the condition that was actually measured.

        Returns
        -------
        (min_eigenvalue, eigenvalues) : (float, ndarray)

        Raises
        ------
        ValueError
            If ``threshold`` is not a finite non-negative real, or if the
            overlap is materially indefinite, numerically non-positive, or
            positive but below ``threshold``.
        """
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real)
        ):
            raise ValueError(
                "CCM overlap threshold must be a finite non-negative real."
            )
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "CCM overlap threshold must be a finite non-negative real."
            )

        from vibeqc.linear_dependence import DEFAULT_NEGATIVE_THRESHOLD

        from .integrals import ccm_overlap

        S = np.asarray(ccm_overlap(self), dtype=float)
        if not np.all(np.isfinite(S)):
            raise ValueError(
                "CCM overlap matrix contains non-finite entries; cannot "
                "classify its spectrum."
            )
        evals = np.linalg.eigvalsh(0.5 * S + 0.5 * S.T)
        if not np.all(np.isfinite(evals)):
            raise ValueError(
                "CCM overlap eigenspectrum contains non-finite values; "
                "cannot classify it."
            )
        emin = float(evals[0])

        if emin < DEFAULT_NEGATIVE_THRESHOLD:
            raise ValueError(
                f"CCM overlap matrix indefinite: min eigenvalue {emin:.3e} "
                f"< the material-negative floor "
                f"{DEFAULT_NEGATIVE_THRESHOLD:.1e}. The finite-cluster "
                "Γ-point overlap has lost positive definiteness, so the "
                "unscreened approximation is unphysical (Peintinger & "
                "Bredow 2014, Fig. 6 and the critical-eigenvalues "
                "discussion). Use canonical screening to remove the "
                "non-positive or below-threshold subspace only when enough "
                "directions remain for the occupied orbitals, or enlarge the "
                "cluster."
            )

        if emin <= 0.0:
            raise ValueError(
                f"CCM overlap matrix numerically non-positive: min eigenvalue "
                f"{emin:.3e} lies between the material-negative floor "
                f"{DEFAULT_NEGATIVE_THRESHOLD:.1e} and zero. Treat this as a "
                "singular noise-scale metric: use a positive canonical-"
                "screening threshold and require enough retained directions "
                "for the occupied orbitals."
            )

        if emin < threshold:
            raise ValueError(
                f"CCM overlap matrix near-singular but positive: min "
                f"eigenvalue {emin:.3e} < {threshold:.1e}. Enlarge the "
                "cluster, reduce basis diffuseness, or canonically screen the "
                "below-threshold subspace while retaining enough directions "
                "for the occupied orbitals (Peintinger & Bredow 2014, Fig. 6)."
            )
        return emin, evals

    def __repr__(self) -> str:
        rc = (
            f", R_c={self.interaction_range:.3g}b"
            if self.interaction_range is not None
            else ""
        )
        return (
            f"CCMSystem(nrep={self.nrep}{rc}, r_in={self.wsc_inscribed_radius:.3g}b, "
            f"n_atoms={self.n_atoms}, nbf={self.nbf}, basis={self.basis_name!r}"
        )


#: Absolute floor for a vacuum direction (bohr): an axis whose largest empty
#: inter-atom slab exceeds this length carries no meaningful periodic images
#: for the 3-D neutral construction. Calibrated against the two known boundary
#: cases (IID 291): the externally validated H₂ anchor cell (20 bohr transverse
#: vacuum, matches PySCF KRHF) stays below it, while the Paper-1 SI chain
#: (40 bohr transverse vacuum, diverging neutral energies) is above it.
_VACUUM_GAP_BOHR_FLOOR = 25.0

#: A vacuum direction must also exceed this multiple of the cell's shortest
#: lattice vector, so a uniformly dilute 3-D cell (e.g. a 80-bohr cubic box
#: with an isolated molecule -- a genuine 3-D molecular crystal) is never
#: misclassified: there every axis is equally "vacuous", and none is a
#: needle-cell vacuum direction.
_VACUUM_GAP_SHORTEST_VECTOR_RATIO = 4.0


def ccm_vacuum_directions(ccm: CCMSystem) -> list[int]:
    """Measured physical dimensionality audit of a CCM unit cell (IID 291).

    The 3-D neutral fitted-torus construction assumes genuine periodicity in
    all three directions: its BvK exchange-``q=0`` seam corrects for periodic
    images that a vacuum-padded direction does not have. The old gate tested
    the *declared* dimensionality, so a 1-D chain padded with transverse
    vacuum and declared 3-D passed it and then produced a diverging,
    unphysical neutral energy ladder.

    This classifier measures, per lattice direction ``i`` (the columns of
    ``unit.lattice``), the largest empty slab between the atoms' fractional
    coordinates along that axis (wrap-around included) in bohr. Direction ``i``
    is a **vacuum direction** when that slab exceeds both

    * :data:`_VACUUM_GAP_BOHR_FLOOR` (25 bohr -- no physical periodic images
      across that much empty space), and
    * :data:`_VACUUM_GAP_SHORTEST_VECTOR_RATIO` times the cell's shortest
      lattice vector (a uniformly dilute 3-D box keeps every axis active).

    Returns the 0-based vacuum axis indices, sorted, or ``[]`` when the cell
    is measured as genuinely 3-D. Purely geometric: no SCF state involved.
    """
    unit = ccm.unit_system
    lattice = np.asarray(unit.lattice, dtype=float)
    vectors = np.linalg.norm(lattice, axis=0)  # columns = lattice vectors
    shortest = float(np.min(vectors))
    threshold = max(_VACUUM_GAP_BOHR_FLOOR,
                    _VACUUM_GAP_SHORTEST_VECTOR_RATIO * shortest)
    vacuum: list[int] = []
    for axis in range(3):
        positions = np.asarray(
            [atom.xyz for atom in unit.unit_cell_molecule().atoms], dtype=float
        )
        frac = np.linalg.solve(lattice, positions.T)[axis]  # along this axis
        frac = np.sort(np.mod(frac, 1.0))
        gaps = np.diff(np.concatenate([frac, [frac[0] + 1.0]]))
        largest_gap_bohr = float(np.max(gaps)) * vectors[axis]
        if largest_gap_bohr > threshold:
            vacuum.append(axis)
    return vacuum
