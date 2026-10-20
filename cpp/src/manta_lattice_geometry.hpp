#pragma once

// Shared finite Ewald domain validation and identities. No numerical repair.
#include "vibeqc/manta_lattice_tail.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <algorithm>
#include <cstring>

namespace vibeqc::manta::lattice_detail {
struct Hash {
    vibeqc::detail::Sha256 value;
    void word(std::uint64_t x) {
        std::array<std::uint8_t,8> bytes{};
        for(int k=0;k<8;++k) bytes[k]=static_cast<std::uint8_t>(x>>(56-8*k));
        value.update(bytes.data(),bytes.size());
    }
    void number(double x) {
        if(x==0.) x=0.;
        std::uint64_t bits; std::memcpy(&bits,&x,sizeof(bits)); word(bits);
    }
    void text(const std::string& s) {
        word(s.size()); value.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size());
    }
};
inline std::vector<LatticeIndex> domain(const std::vector<LatticeIndex>& input,bool origin) {
    auto result=input;
    for(const auto& n:result) for(auto x:n)
        if(x < -2147483647LL || x > 2147483647LL)
            throw std::invalid_argument("MANTA lattice-tail indices exceed the signed 31-bit range");
    std::sort(result.begin(),result.end());
    if(std::adjacent_find(result.begin(),result.end())!=result.end())
        throw std::invalid_argument("MANTA lattice-tail domains must contain unique indices");
    if(std::binary_search(result.begin(),result.end(),LatticeIndex{})!=origin)
        throw std::invalid_argument("MANTA lattice-tail real/near domains require origin; reciprocal domain excludes zero");
    for(const auto& n:result)
        if(!std::binary_search(result.begin(),result.end(),LatticeIndex{-n[0],-n[1],-n[2]}))
            throw std::invalid_argument("MANTA lattice-tail domains must be inversion closed");
    return result;
}
inline std::string domain_identity(const char* kind,const std::vector<LatticeIndex>& indices) {
    Hash h; h.text("manta.lattice-tail.domain.v1"); h.text(kind); h.word(indices.size());
    for(const auto& n:indices) for(auto x:n) h.word(static_cast<std::uint64_t>(x));
    return h.value.finish_hex();
}
struct Geometry {
    std::vector<LatticeIndex> real,reciprocal,near;
    double volume;
    std::string real_identity,reciprocal_identity,near_identity,kernel_identity,expansion_identity;
    Geometry(const Eigen::Matrix3d& cell,const md::Point& displacement,double eta,
        const std::vector<LatticeIndex>& rs,const std::vector<LatticeIndex>& gs,
        const std::vector<LatticeIndex>& ns,int order) {
        volume=std::abs(cell.determinant());
        if(!cell.allFinite() || !std::isfinite(volume) || volume<=0 || !cell.inverse().allFinite())
            throw std::invalid_argument("MANTA lattice-tail requires a finite nonsingular 3D cell");
        if(!std::isfinite(eta) || eta<=0) throw std::invalid_argument("MANTA lattice-tail eta must be finite and positive");
        for(auto x:displacement) admission::require_finite(x);
        real=domain(rs,true); reciprocal=domain(gs,false); near=domain(ns,true);
        if(!std::includes(real.begin(),real.end(),near.begin(),near.end()))
            throw std::invalid_argument("MANTA lattice-tail near domain must be contained in real domain");
        real_identity=domain_identity("real",real);
        reciprocal_identity=domain_identity("reciprocal",reciprocal);
        near_identity=domain_identity("near",near);
        Hash kernel; kernel.text("manta.lattice-tail.v1;finite-ewald-minus-near;columns;raw-cartesian;total-degree;binary64");
        kernel.text("3d-zero-average-conducting"); kernel.number(eta);
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) kernel.number(cell(i,j));
        kernel.text(real_identity); kernel.text(reciprocal_identity); kernel.text(near_identity);
        kernel_identity=kernel.value.finish_hex();
        Hash source; source.text(kernel_identity); source.word(order);
        for(auto x:displacement) source.number(x);
        expansion_identity=source.value.finish_hex();
    }
};
} // namespace vibeqc::manta::lattice_detail
