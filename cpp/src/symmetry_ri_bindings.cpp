#include "vibeqc/symmetry_ri.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py=pybind11;
namespace {
using U=std::uint64_t;
using Z=std::complex<double>;
template<class T> const T* payload(const py::array& a) {
    if (!a.dtype().is(py::dtype::of<T>()) || !(a.flags()&py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(a.data())%alignof(T))
        throw std::invalid_argument("RI blocks require exact aligned C-contiguous arrays");
    return static_cast<const T*>(a.data());
}
py::object owned(std::vector<Z>&& values,U n,bool requested) {
    if (!requested) return py::none();
    auto owner=std::make_unique<std::vector<Z>>(std::move(values));
    auto* raw=owner.get();
    py::capsule capsule(raw,[](void* p){delete static_cast<std::vector<Z>*>(p);}); owner.release();
    return py::array(py::dtype::of<Z>(),{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)},
        {static_cast<py::ssize_t>(n*sizeof(Z)),static_cast<py::ssize_t>(sizeof(Z))},raw->data(),capsule);
}
}
void bind_symmetry_ri(py::module_& m) {
    using namespace vibeqc;
    m.def("_plan_symmetry_ri_blocks",&plan_symmetry_ri_blocks,py::arg("n"),py::arg("rank"),
        py::arg("groups"),py::arg("blocks"),py::arg("values"),py::arg("budget"));
    py::class_<SymmetryRIBlocks>(m,"_SymmetryRIBlocks")
        .def(py::init([](U n,U rank,py::array orbital,py::array auxiliary,py::array keys,
                        py::array factors,const SymmetryBudget& budget) {
            if (orbital.ndim()!=1 || orbital.size()<2 || auxiliary.ndim()!=1 || auxiliary.size()!=orbital.size() ||
                keys.ndim()!=2 || keys.shape(1)!=3 || factors.ndim()!=1)
                throw std::invalid_argument("RI block descriptor shape mismatch");
            const U groups=orbital.size()-1,blocks=keys.shape(0),values=factors.size();
            plan_symmetry_ri_blocks(n,rank,groups,blocks,values,budget);
            return std::make_unique<SymmetryRIBlocks>(n,rank,groups,blocks,values,
                payload<std::int64_t>(orbital),payload<std::int64_t>(auxiliary),
                payload<std::int64_t>(keys),payload<Z>(factors),budget);
        }),py::arg("n"),py::arg("rank"),py::arg("orbital_offsets").noconvert(),
           py::arg("auxiliary_offsets").noconvert(),py::arg("keys").noconvert(),
           py::arg("factors").noconvert(),py::arg("budget"))
        .def("contract",[](const SymmetryRIBlocks& plan,py::array density,bool j,bool k,const SymmetryBudget& budget) {
            // Refuse the resource request before even inspecting density data.
            const auto memory=plan.memory();
            if (memory.bytes>budget.maximum_bytes || memory.work>budget.maximum_work)
                throw std::length_error("RI block execution budget exceeded");
            if (density.ndim()!=2 || density.shape(0)!=static_cast<py::ssize_t>(plan.dimension()) ||
                density.shape(1)!=density.shape(0)) throw std::invalid_argument("RI block density shape mismatch");
            auto result=plan.contract(payload<Z>(density),j,k,budget);
            py::dict counts;
            counts["scalar_products"]=result.scalar_products;
            counts["block_contractions"]=result.block_contractions;
            return py::make_tuple(owned(std::move(result.j),plan.dimension(),j),
                owned(std::move(result.k),plan.dimension(),k),counts);
        },py::arg("density").noconvert(),py::arg("make_j"),py::arg("make_k"),py::arg("budget"));
}
