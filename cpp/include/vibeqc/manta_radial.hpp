#pragma once

#include "vibeqc/manta.hpp"
#include "vibeqc/manta_md.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vibeqc::manta {
inline std::string md_request_error(const KernelRequest& r,int centers) {
    if(r.centers!=centers || r.derivative!=0 || r.scalar!="real")
        return "MANTA MD requires the requested center count, real scalars and derivative order zero";
    if(r.kernel!="coulomb" && r.kernel!="erf" && r.kernel!="erfc")
        return "MANTA MD kernel must be coulomb, erf or erfc";
    if(!std::isfinite(r.omega) || r.omega<0 || (r.kernel=="coulomb" && r.omega!=0))
        return "MANTA MD range must be finite and nonnegative, and zero for full Coulomb";
    for(const auto& image:r.images) for(auto n:image) if(n)
        return "MANTA molecular MD does not admit periodic image requests";
    return {};
}
inline md::RadialKernel radial_kernel(const KernelRequest& r) {
    if(r.kernel=="coulomb") return md::RadialKernel::Coulomb;
    if(r.kernel=="erf") return md::RadialKernel::Erf;
    if(r.kernel=="erfc") return md::RadialKernel::Erfc;
    throw std::invalid_argument("MANTA radial kernel");
}
inline std::string range_identity_suffix(const KernelRequest& r) {
    if(r.kernel=="coulomb") return {};
    const double omega=r.omega==0?0.:r.omega;
    std::uint64_t bits; std::memcpy(&bits,&omega,sizeof(bits));
    return ";omega-binary64="+std::to_string(bits);
}
// Upper arithmetic-work models for the bounded Boys series and positive
// erfc interval quadrature (at most 256 panels, each with 16+32 nodes).
// These are admission units, not measured processor instruction counts.
inline Count boys_work(int order) { return 4096+64*static_cast<Count>(order); }
inline Count radial_work(const KernelRequest& r,int order) {
    return r.kernel=="erfc" ? Count{256}*48*(32+4*(order+1)) : boys_work(order);
}
inline Count radial_workspace(const KernelRequest& r) {
    return r.kernel=="erfc" ? Count{128}<<10 : 0;
}
} // namespace vibeqc::manta
