#pragma once
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_lattice_tail.hpp"
namespace vibeqc::manta::lattice_cube {
using namespace admission;
inline Count cube_cells(Count radius) {
    if(radius>=static_cast<Count>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("MANTA BUPO preparation cube exceeds the supported integer extent");
    const Count side=add(mul(2,radius),1);return mul(square(side),side);
}
inline std::vector<LatticeIndex> cube(Count radius,bool nonzero=false) {
    std::vector<LatticeIndex> out;out.reserve(cube_cells(radius)-(nonzero?1:0));const auto r=static_cast<int>(radius);
    for(int i=-r;i<=r;++i)for(int j=-r;j<=r;++j)for(int k=-r;k<=r;++k)
        if(!nonzero || i || j || k)out.push_back({i,j,k});
    return out;
}
}
