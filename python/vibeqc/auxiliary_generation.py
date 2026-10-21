"""Bounded shared auxiliary candidates; accuracy belongs to the consuming method.

Atom-local primitive-product exponent ranges and even tempering follow the
motivation of Stoychev, Auer and Neese, JCTC 13, 554 (2017), Sec. 2, Eqs. 5-7,
DOI 10.1021/acs.jctc.6b01041. This deliberately differs from AutoAux: it keeps
primitive ranges, every Cartesian product-degree channel and a supplied seed,
without effective-exponent, ECP/valence or angular-pruning heuristics.
"""
from ._vibeqc_core import (
    ProductAuxiliaryOptions, ProductAuxiliaryLimits, ProductAuxiliaryPlan,
    ProductAuxiliaryBasis, ProductAuxiliaryReceipt, ProductAuxiliarySpan,
    ProductAuxiliarySequence, ProductAuxiliarySequencePlan,
    plan_product_auxiliary_sequence, prepare_product_auxiliary_sequence,
    plan_product_auxiliary_basis, prepare_product_auxiliary_basis,
)

__all__ = ["ProductAuxiliaryOptions", "ProductAuxiliaryLimits", "ProductAuxiliaryPlan",
           "ProductAuxiliaryBasis", "ProductAuxiliaryReceipt", "ProductAuxiliarySpan",
           "plan_product_auxiliary_basis", "prepare_product_auxiliary_basis"]



def product_auxiliary_provenance(source):
    """Serialize owned metadata; this contains no accuracy certificate."""
    if not isinstance(source, ProductAuxiliaryBasis):
        raise TypeError("source must be ProductAuxiliaryBasis")
    result = product_auxiliary_receipt_provenance(source.receipt)
    result["plan"] = {name: getattr(source.plan, name) for name in (
        "centers", "channels", "original_shells", "original_functions",
        "maximum_generated_shells", "maximum_generated_functions", "maximum_shells",
        "maximum_functions", "modeled_peak_bytes", "work_units")}
    return result


def product_auxiliary_receipt_provenance(receipt):
    """Serialize a copied native generation receipt independently of its owner."""
    if not isinstance(receipt, ProductAuxiliaryReceipt):
        raise TypeError("receipt must be ProductAuxiliaryReceipt")
    fields = ("algorithm", "qualification", "citation_method", "orbital_identity",
              "seed_identity", "basis_identity", "identity", "added_shells",
              "added_functions", "duplicate_seed_shells")
    result = {name: getattr(receipt, name) for name in fields}
    result["options"] = {name: getattr(receipt.options, name) for name in (
        "beta", "refinement", "maximum_angular_momentum")}
    result["spans"] = [{name: getattr(span, name) for name in (
        "atom", "coarse_intervals", "angular_momentum", "minimum_exponent",
        "maximum_product_exponent", "upper_grid_exponent")} for span in receipt.spans]
    return result


__all__.append("product_auxiliary_provenance")

__all__ += ["ProductAuxiliarySequence", "ProductAuxiliarySequencePlan",
            "plan_product_auxiliary_sequence", "prepare_product_auxiliary_sequence"]

__all__.append("product_auxiliary_receipt_provenance")
