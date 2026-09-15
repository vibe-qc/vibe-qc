// Internal shared symmetry interface. Mathematical validation only.
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vibeqc/symmetry_shared.hpp"
#include "vibeqc/basis.hpp"
#include <memory>
#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif
namespace shared_symmetry_binding_detail {
template<class T> const T* payload(const py::array& a) {
    if (!a.dtype().is(py::dtype::of<T>()) || !(a.flags() & py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(a.data())%alignof(T))
        throw std::invalid_argument("shared symmetry requires exact aligned C-contiguous arrays");
    return static_cast<const T*>(a.data());
}
}
void bind_symmetry_shared(py::module_& m) {
    using namespace vibeqc;
    using I=std::int64_t;
    using U=std::uint64_t;
    using Z=std::complex<double>;
    using shared_symmetry_binding_detail::payload;
    py::class_<SymmetryBudget>(m,"_SymmetryBudget")
        .def(py::init<>()).def_readwrite("maximum_bytes",&SymmetryBudget::maximum_bytes)
        .def_readwrite("maximum_work",&SymmetryBudget::maximum_work);
    py::class_<SymmetryPlan>(m,"_SymmetryPlan")
        .def_readonly("bytes",&SymmetryPlan::bytes).def_readonly("work",&SymmetryPlan::work);
    py::class_<SymmetryGroup>(m,"_SymmetryGroup")
        .def_property_readonly("order",&SymmetryGroup::order)
        .def_property_readonly("identity",&SymmetryGroup::identity)
        .def_property_readonly("memory",&SymmetryGroup::memory)
        .def("product",&SymmetryGroup::product).def("inverse",&SymmetryGroup::inverse)
        .def("antiunitary",&SymmetryGroup::antiunitary).def("cocycle",&SymmetryGroup::cocycle);
    m.def("_symmetry_molecule_count",[](const Molecule& molecule,const SymmetryBudget& budget) {
        const U n=molecule.atoms().size();
        if (budget.maximum_bytes<4096 || n>(budget.maximum_bytes-4096)/1024)
            throw std::length_error("molecular symmetry atom snapshot byte budget");
        if (!budget.maximum_work || (n && n>budget.maximum_work/256/n))
            throw std::length_error("molecular symmetry atom search work budget");
        return n;
    });
    m.def("_symmetry_basis_snapshot_bytes",[](const BasisSet& basis,const SymmetryBudget& budget) {
        // Count borrowed native metadata before BasisSet.shells() creates
        // Python shell/radial snapshots. Conservative logical role census.
        if (budget.maximum_bytes<4096 || !budget.maximum_work)
            throw std::length_error("molecular symmetry basis snapshot budget");
        U bytes=4096, work=0;
        auto charge=[&](U count,U width) {
            if (count>budget.maximum_bytes/width || bytes>budget.maximum_bytes-count*width)
                throw std::length_error("molecular symmetry basis snapshot byte budget");
            bytes+=count*width;
        };
        for (const auto& shell:basis.libint()) {
            for (const auto& contraction:shell.contr) {
                // shells() copies the exponents once per contraction, too.
                charge(1,1024);
                charge(shell.alpha.size(),128);
                charge(contraction.coeff.size(),128);
                for (const U count : {U(1), U(shell.alpha.size()), U(contraction.coeff.size())}) {
                    if (count > budget.maximum_work-work)
                        throw std::length_error("molecular symmetry basis snapshot work budget");
                    work += count;
                }
            }
        }
        return bytes;
    });
    m.def("_plan_symmetry_group",&plan_symmetry_group);
    m.def("_make_symmetry_group",[](py::array table,py::array anti,U identity,
                                  py::object rotation,py::object cocycle,const SymmetryBudget& budget) {
        if (table.ndim()!=2 || table.shape(0)!=table.shape(1) || anti.ndim()!=1 || anti.shape(0)!=table.shape(0))
            throw std::invalid_argument("shared symmetry group array shape mismatch");
        const U n=table.shape(0);
        const bool lattice=!rotation.is_none();
        plan_symmetry_group(n,lattice,budget); // before payload scans or copies
        if (lattice!=!cocycle.is_none()) throw std::invalid_argument("rotations and cocycle must occur together");
        if (!lattice) return make_symmetry_group(n,identity,payload<I>(table),payload<std::uint8_t>(anti),nullptr,nullptr,budget);
        // py::cast<array> may invoke __array__ and allocate an arbitrarily
        // large converted descriptor before its shape can be checked.
        if (!py::isinstance<py::array>(rotation) || !py::isinstance<py::array>(cocycle))
            throw py::type_error("shared symmetry lattice descriptors must be NumPy arrays");
        const auto r=py::reinterpret_borrow<py::array>(rotation);
        const auto c=py::reinterpret_borrow<py::array>(cocycle);
        if (r.ndim()!=3 || r.shape(0)!=static_cast<py::ssize_t>(n) || r.shape(1)!=3 || r.shape(2)!=3 ||
            c.ndim()!=3 || c.shape(0)!=static_cast<py::ssize_t>(n) || c.shape(1)!=static_cast<py::ssize_t>(n) || c.shape(2)!=3)
            throw std::invalid_argument("shared symmetry lattice array shape mismatch");
        return make_symmetry_group(n,identity,payload<I>(table),payload<std::uint8_t>(anti),payload<I>(r),payload<I>(c),budget);
    },py::arg("products").noconvert(),py::arg("antiunitary").noconvert(),py::arg("identity"),
       py::arg("rotations"),py::arg("cocycle"),py::arg("budget"));
    m.def("_plan_symmetry_blocks",&plan_symmetry_blocks);
    m.def("_apply_symmetry_blocks",[](py::array offsets,py::array dest,py::array matrices,
        py::array coefficients,bool anti,bool adjoint,const SymmetryBudget& budget) {
        if (offsets.ndim()!=1 || dest.ndim()!=1 || offsets.shape(0)!=dest.shape(0)+1 ||
            matrices.ndim()!=1 || coefficients.ndim()!=2)
            throw std::invalid_argument("shared symmetry block array shape mismatch");
        const U n=coefficients.shape(0), columns=coefficients.shape(1), blocks=dest.shape(0), elements=matrices.size();
        plan_symmetry_blocks(n,blocks,elements,columns,budget);
        auto result=apply_symmetry_blocks(n,blocks,elements,columns,payload<I>(offsets),payload<I>(dest),
            payload<Z>(matrices),payload<Z>(coefficients),anti,adjoint,budget);
        // Transfer the native allocation to NumPy without a second panel.
        auto owner=std::make_unique<std::vector<Z>>(std::move(result));
        auto* raw=owner.get();
        py::capsule capsule(raw,[](void* p){delete static_cast<std::vector<Z>*>(p);});
        owner.release();
        return py::array(py::dtype::of<Z>(),{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(columns)},
            {static_cast<py::ssize_t>(columns*sizeof(Z)),static_cast<py::ssize_t>(sizeof(Z))},raw->data(),capsule);
    },py::arg("offsets").noconvert(),py::arg("destinations").noconvert(),py::arg("matrices").noconvert(),
       py::arg("coefficients").noconvert(),py::arg("antiunitary"),py::arg("adjoint"),py::arg("budget"));
}
