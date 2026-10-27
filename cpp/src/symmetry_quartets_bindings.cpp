#include "vibeqc/symmetry_quartets.hpp"
#include "vibeqc/symmetry_quartets_source.hpp"
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
py::tuple export_targets(vibeqc::SymmetryQuartetTargets result) {
    const U n=result.dimension;
    py::dict counts;
    counts["representatives"]=result.counts.representatives;
    counts["scalar_products"]=result.counts.scalar_products;
    counts["block_contractions"]=result.counts.block_contractions;
    counts["transport_scalar_products"]=result.counts.transport_scalar_products;
    return py::make_tuple(owned(std::move(result.j),n,result.j_count),
        owned(std::move(result.k),n,result.k_count),counts);
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
        .def("accumulate_orbit",[](SymmetryQuartetAccumulator& a,py::object representative_object,
                py::object tile_object,py::object mapped_object,py::object identity_object,py::object mode_object) {
            try {
                if (!py::isinstance<py::array>(representative_object) || !py::isinstance<py::array>(tile_object) ||
                    !py::isinstance<py::array>(mapped_object)) throw py::type_error("quartet inputs must be NumPy arrays");
                const auto representative=py::reinterpret_borrow<py::array>(representative_object);
                const auto tile=py::reinterpret_borrow<py::array>(tile_object);
                const auto mapped=py::reinterpret_borrow<py::array>(mapped_object);
                if (representative.ndim()!=1 || representative.shape(0)!=4 || tile.ndim()!=4 ||
                    mapped.ndim()!=2 || mapped.shape(1)!=4)
                    throw std::invalid_argument("quartet orbit descriptor shape mismatch");
                if (!py::isinstance<py::int_>(identity_object) || py::isinstance<py::bool_>(identity_object) ||
                    !py::isinstance<py::int_>(mode_object) || py::isinstance<py::bool_>(mode_object))
                    throw py::type_error("quartet identity and mode must be integers");
                const U identity=identity_object.cast<U>(),mode=mode_object.cast<U>();
                if (mode>2) throw std::invalid_argument("invalid quartet orbit mode");
                const auto* r=payload<I>(representative);
                const std::array<I,4> key{{r[0],r[1],r[2],r[3]}};
                const std::array<U,4> shape{{static_cast<U>(tile.shape(0)),static_cast<U>(tile.shape(1)),
                    static_cast<U>(tile.shape(2)),static_cast<U>(tile.shape(3))}};
                a.accumulate_orbit(key,shape,payload<double>(tile),mapped.shape(0),payload<I>(mapped),
                    identity,static_cast<std::uint8_t>(mode));
            } catch (...) {
                a.invalidate();
                throw;
            }
        },py::arg("representative").noconvert(),py::arg("tile").noconvert(),py::arg("mapped_shells").noconvert(),
           py::arg("identity").noconvert(),py::arg("mode").noconvert())
        .def("finish",[](SymmetryQuartetAccumulator& a) {
            return export_targets(a.finish());
        });
    m.def("_plan_symmetry_quartet_source",&plan_symmetry_quartet_source,
        py::arg("provider"),py::arg("budget"));
    py::class_<SymmetryQuartetSourceAccumulator>(m,"_SymmetryQuartetSourceAccumulator")
        .def(py::init([](SymmetryQuartetAccumulator& a,py::object builder,py::object source,
                py::object digest,py::object count,const SymmetryBudget& budget) {
            try {
                if(!py::isinstance<py::int_>(count) || py::isinstance<py::bool_>(count))
                    throw py::type_error("quartet representative count must be an integer");
                // Native source IDs and canonical SHA-256 digests have fixed
                // size. Reject descriptors before copying Python strings.
                if(!py::isinstance<py::str>(source) || !py::isinstance<py::str>(digest))
                    throw py::type_error("quartet source identities must be strings");
                if(PyUnicode_GetLength(source.ptr())!=64 || PyUnicode_GetLength(digest.ptr())!=64)
                    throw std::invalid_argument("invalid quartet source identity or representative digest length");
                return std::make_unique<SymmetryQuartetSourceAccumulator>(a,
                    builder.cast<std::shared_ptr<manta::OrderedJKBuilder>>(),source.cast<std::string>(),
                    digest.cast<std::string>(),count.cast<U>(),budget);
            } catch(...) { a.invalidate(); throw; }
        }),py::arg("accumulator"),py::arg("provider"),py::arg("source_identity"),
           py::arg("representative_digest"),py::arg("representatives"),py::arg("budget"),py::keep_alive<1,2>())
        .def("bind_support",[](SymmetryQuartetSourceAccumulator& a,py::object support,
                py::object destinations,py::object identity,py::object mode) {
            try {
                if(!py::isinstance<py::array>(destinations))
                    throw py::type_error("quartet support maps must be a NumPy array");
                const auto maps=py::reinterpret_borrow<py::array>(destinations);
                if(maps.ndim()!=2) throw std::invalid_argument("quartet support map shape mismatch");
                if(!py::isinstance<py::int_>(identity) || py::isinstance<py::bool_>(identity) ||
                    !py::isinstance<py::int_>(mode) || py::isinstance<py::bool_>(mode))
                    throw py::type_error("quartet identity and mode must be integers");
                const U identity_value=identity.cast<U>(),mode_value=mode.cast<U>();
                if(mode_value>2) throw std::invalid_argument("invalid quartet orbit mode");
                a.bind_support(support.cast<std::shared_ptr<manta::MolecularSourceSupport>>(),
                    maps.shape(0),maps.shape(1),payload<I>(maps),identity_value,static_cast<std::uint8_t>(mode_value));
            } catch(...) { a.invalidate(); throw; }
        },py::arg("support").noconvert(),py::arg("destinations").noconvert(),
            py::arg("identity").noconvert(),py::arg("mode").noconvert())
        .def("consume_key",[](SymmetryQuartetSourceAccumulator& a,py::object representative) {
            try {
                if(!py::isinstance<py::array>(representative))
                    throw py::type_error("quartet representative must be a NumPy array");
                const auto key=py::reinterpret_borrow<py::array>(representative);
                if(key.ndim()!=1 || key.shape(0)!=4) throw std::invalid_argument("quartet key shape mismatch");
                const auto* r=payload<I>(key);
                a.consume_key({{r[0],r[1],r[2],r[3]}});
            } catch(...) { a.invalidate(); throw; }
        },py::arg("representative").noconvert())
        .def("consume_orbit",[](SymmetryQuartetSourceAccumulator& a,py::object representative,
                py::object mapped,py::object identity,py::object mode) {
            try {
                if(!py::isinstance<py::array>(representative) || !py::isinstance<py::array>(mapped))
                    throw py::type_error("quartet inputs must be NumPy arrays");
                const auto r=py::reinterpret_borrow<py::array>(representative);
                const auto m=py::reinterpret_borrow<py::array>(mapped);
                if(r.ndim()!=1 || r.shape(0)!=4 || m.ndim()!=2 || m.shape(1)!=4)
                    throw std::invalid_argument("quartet source descriptor shape mismatch");
                if(!py::isinstance<py::int_>(identity) || py::isinstance<py::bool_>(identity) ||
                   !py::isinstance<py::int_>(mode) || py::isinstance<py::bool_>(mode))
                    throw py::type_error("quartet identity and mode must be integers");
                const U operation=identity.cast<U>(),code=mode.cast<U>();
                if(code>2) throw std::invalid_argument("invalid quartet orbit mode");
                const auto* key=payload<I>(r);
                a.consume_orbit({{key[0],key[1],key[2],key[3]}},m.shape(0),payload<I>(m),operation,
                    static_cast<std::uint8_t>(code));
            } catch(...) { a.invalidate(); throw; }
        },py::arg("representative").noconvert(),py::arg("mapped_shells").noconvert(),
           py::arg("identity").noconvert(),py::arg("mode").noconvert())
        .def("finish",[](SymmetryQuartetSourceAccumulator& a) {
            auto targets=export_targets(a.finish());
            return py::make_tuple(targets[0],targets[1],targets[2],a.work());
        });
}
