"""Double-hybrid construction wording, in one place.

A double hybrid is two calculations: a hybrid-DFT SCF step and a scaled
MP2 correction on its converged orbitals. Which two depends on the spin
reference, and until #365 the output said ``RKS reference`` for every
run -- including open-shell ones, where the SCF is a spin-pure ROKS and
the correction is a semicanonical RI-ROHF-MP2 doubles-only term. A user
reading that ``.out``, or citing from that ``.bibtex``, was told about a
calculation vibe-qc did not perform.

The construction is therefore described once, here, and the three
consumers read from the same object: the ``.out`` energy block and its
header line, the ``.system`` ``[run]`` fields, and the citation routes
that fire for the components the run actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


__all__ = [
    "DoubleHybridConstruction",
    "double_hybrid_construction",
]


@dataclass(frozen=True)
class DoubleHybridConstruction:
    """What a double-hybrid run actually did, in printable form.

    ``scf_row_label`` and ``correlation_row_label`` are the left-hand
    labels of the ``Energy components`` block; ``summary_line`` is the
    one-line description printed under the dispatcher header.
    """

    open_shell: bool
    scf_method: str
    scf_row_label: str
    correlation_method: str
    correlation_row_label: str
    summary_line: str
    citation_methods: tuple[str, ...]

    def manifest_fields(self) -> dict[str, object]:
        """The same facts as flat ``[run]`` scalars.

        A consumer reading a ``.system`` manifest can tell a closed-shell
        from an open-shell double hybrid without parsing the ``.out``
        prose -- the two were previously indistinguishable there.
        """
        return {
            "scf_reference": self.scf_method,
            "correlation_method": self.correlation_method,
            "open_shell": bool(self.open_shell),
        }


def double_hybrid_construction(
    *,
    open_shell: bool,
    density_fit_mp2: bool = True,
    c_os: Optional[float] = None,
    c_ss: Optional[float] = None,
) -> DoubleHybridConstruction:
    """Describe the two steps a double-hybrid dispatch ran.

    ``open_shell`` is the branch the dispatcher took (multiplicity > 1),
    not a request: the open-shell route is always ROKS + semicanonical
    RI-ROHF-MP2, and its correction is doubles-only, so the singles term
    is named as excluded rather than silently absent.

    ``density_fit_mp2`` only qualifies the closed-shell correction. The
    open-shell ROHF-MP2 kernel is RI-only (there is no canonical
    four-index ROHF-MP2 route), which is why its label says RI
    unconditionally.
    """
    scale = ""
    if c_os is not None and c_ss is not None:
        scale = f" (c_os={float(c_os):.4g}, c_ss={float(c_ss):.4g})"

    if open_shell:
        return DoubleHybridConstruction(
            open_shell=True,
            scf_method="ROKS",
            scf_row_label="ROKS reference",
            correlation_method="rohf-mp2",
            correlation_row_label="RI-ROHF-MP2 doubles",
            summary_line=(
                "spin-pure ROKS SCF + semicanonical RI-ROHF-MP2, "
                f"doubles only{scale}"
            ),
            # The functional's own route fires from ``method``; these are
            # the two components the open-shell branch adds. Both already
            # exist in database.toml.
            citation_methods=("roks", "rohf-mp2"),
        )

    correction = "RI-MP2" if density_fit_mp2 else "canonical MP2"
    return DoubleHybridConstruction(
        open_shell=False,
        scf_method="RKS",
        scf_row_label="RKS reference",
        correlation_method="mp2",
        correlation_row_label="MP2 correlation",
        summary_line=f"hybrid RKS SCF + scaled {correction}{scale}",
        citation_methods=(),
    )
