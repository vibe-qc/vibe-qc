#include "vibeqc/symmetry_quartets.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py=pybind11;
namespace {
using U=std::uint64_t;
using I=std::int64_t;
using Z=std::complex<double>;
template<class T> const T* payload(const py::array& a) {
    if (!a.dtype().is(py::dtype::of<T>()) || !(a.flags()&py::array::c_style) ||
        reinterpret_cast<std::uintptr_t>(a.data())%alignof(T))
        throw std::invalid_argument("quartet execution requires exact aligned C-contiguous arrays");
    return static_cast<const T*>(a.data());
}
py::array owned(std::vector<Z>&& values,U n,U nd) {
    auto owner=std::make_unique<std::vector<Z>>(std::move(values));
    auto* raw=owner.get();
    py::capsule capsule(raw,[](void* p){delete static_cast<std::vector<Z>*>(p);});
    owner.release();
    if (nd!=1) return py::array(py::dtype::of<Z>(),
        {static_cast<py::ssize_t>(nd),static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)},
        {static_cast<py::ssize_t>(n*n*sizeof(Z)),static_cast<py::ssize_t>(n*sizeof(Z)),static_cast<py::ssize_t>(sizeof(Z))},raw->data(),capsule);
    return py::array(py::dtype::of<Z>(),{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(n)},
        {static_cast<py::ssize_t>(n*sizeof(Z)),static_cast<py::ssize_t>(sizeof(Z))},raw->data(),capsule);
}
}

void bind_symmetry_quartets(py::module_& m) {
    using namespace vibeqc;
    m.def("_plan_symmetry_quartet_accumulator",&plan_symmetry_quartet_accumulator,
        py::arg("n"),py::arg("shells"),py::arg("operations"),py::arg("matrix_elements"),
        py::arg("maximum_shell"),py::arg("budget"),py::arg("density_count")=1,
        py::arg("j_count")=std::numeric_limits<U>::max(),py::arg("k_count")=std::numeric_limits<U>::max());
    py::class_<SymmetryQuartetAccumulator>(m,"_SymmetryQuartetAccumulator")
        .def(py::init([](py::array offsets,py::array matrices,py::array density,U maximum,const SymmetryBudget& budget,py::object targets) {
            if (offsets.ndim()!=1 || offsets.shape(0)<2 || matrices.ndim()!=2 ||
                (density.ndim()!=2 && density.ndim()!=3)) throw std::invalid_argument("quartet accumulator descriptor shape mismatch");
            const U nd=density.ndim()==3?density.shape(0):1;
            const U n=density.shape(density.ndim()-1), ns=offsets.shape(0)-1, ng=matrices.shape(0), nm=matrices.shape(1);
            if (density.shape(density.ndim()-2)!=static_cast<py::ssize_t>(n))
                throw std::invalid_argument("quartet accumulator descriptor shape mismatch");
            // The minimal one-target-per-density reservation bounds the
            // selector scan before inspecting even its payload.
            plan_symmetry_quartet_accumulator(n,ns,ng,nm,maximum,budget,nd,nd,0);
            U nj=nd, nk=nd;
            const std::uint8_t* selection=nullptr;
            if (!targets.is_none()) {
                if (!py::isinstance<py::array>(targets)) throw py::type_error("quartet targets must be a NumPy array");
                const auto flags=py::reinterpret_borrow<py::array>(targets);
                if (flags.ndim()!=1 || static_cast<U>(flags.shape(0))!=nd)
                    throw std::invalid_argument("quartet target descriptor shape mismatch");
                selection=payload<std::uint8_t>(flags);
                nj=nk=0;
                for (U channel=0;channel<nd;++channel) {
                    if (selection[channel]<1 || selection[channel]>3)
                        throw std::invalid_argument("quartet target flags must request J, K or both");
                    nj+=(selection[channel]&1)!=0; nk+=(selection[channel]&2)!=0;
                }
            }
            plan_symmetry_quartet_accumulator(n,ns,ng,nm,maximum,budget,nd,nj,nk);
            return std::make_unique<SymmetryQuartetAccumulator>(n,ns,ng,nm,maximum,
                payload<I>(offsets),payload<double>(matrices),payload<Z>(density),budget,nd,nj,nk,selection);
        }),py::arg("offsets").noconvert(),py::arg("inverse_matrices").noconvert(),
           py::arg("density").noconvert(),py::arg("maximum_shell"),py::arg("budget"),py::arg("targets")=py::none())
        .def("accumulate",[](SymmetryQuartetAccumulator& a,py::object representative_object,py::object tile_object,py::object members_object) {
            try {
                // Inspect objects without implicit __array__ conversion, so
                // descriptor/type failures also invalidate partial targets.
                if (!py::isinstance<py::array>(representative_object) || !py::isinstance<py::array>(tile_object) ||
                    !py::isinstance<py::array>(members_object)) throw py::type_error("quartet inputs must be NumPy arrays");
                const auto representative=py::reinterpret_borrow<py::array>(representative_object);
                const auto tile=py::reinterpret_borrow<py::array>(tile_object);
                const auto members=py::reinterpret_borrow<py::array>(members_object);
                if (representative.ndim()!=1 || representative.shape(0)!=4 || tile.ndim()!=4 ||
                    members.ndim()!=2 || members.shape(1)!=6) throw std::invalid_argument("quartet tile/member descriptor shape mismatch");
                const auto* r=payload<I>(representative);
                const std::array<I,4> key{{r[0],r[1],r[2],r[3]}};
                const std::array<U,4> shape{{static_cast<U>(tile.shape(0)),static_cast<U>(tile.shape(1)),
                    static_cast<U>(tile.shape(2)),static_cast<U>(tile.shape(3))}};
                a.accumulate(key,shape,payload<double>(tile),members.shape(0),payload<I>(members));
            } catch (...) {
                a.invalidate();
                throw;
            }
        },py::arg("representative").noconvert(),py::arg("tile").noconvert(),py::arg("members").noconvert())
        .def("finish",[](SymmetryQuartetAccumulator& a) {
            auto result=a.finish();
            const U n=result.dimension;
            py::dict counts;
            counts["representatives"]=result.counts.representatives;
            counts["scalar_products"]=result.counts.scalar_products;
            counts["block_contractions"]=result.counts.block_contractions;
            counts["transport_scalar_products"]=result.counts.transport_scalar_products;
            return py::make_tuple(owned(std::move(result.j),n,result.j_count),
                owned(std::move(result.k),n,result.k_count),counts);
        });
}
