#pragma once
#include "vibeqc/manta.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/grid.hpp"
namespace vibeqc::manta {
struct PeriodicQuadratureOptions {
    int radial_points=50,lebedev_order=29;
    double partition_radius_bohr=8.;
    bool stratmann=false;
};
struct PeriodicQuadraturePlan {
    Count atoms=0,angular_points=0,grid_points=0,threads=0;
    Count atlas_candidates=0,neighbor_candidates=0,candidate_visits=0;
    Count output_bytes=0,workspace_bytes=0,modeled_peak_bytes=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicQuadratureReceipt {
    std::string algorithm="periodic-atomic-lebedev-canonical-cell.v1",source_identity;
    Count nonzero_weights=0;
    double weight_sum=0.,cell_volume=0.;
};
struct PeriodicQuadratureResult {
    PeriodicQuadraturePlan plan;
    PeriodicQuadratureReceipt receipt;
    Grid grid;
};
// Reuses the shared Treutler-Ahlrichs/Lebedev and point-centered periodic
// Becke/Stratmann partition. Canonical nuclei and folded points define the
// finite source explicitly. No accuracy inference follows from grid counts.
PeriodicQuadraturePlan plan_periodic_quadrature(const PeriodicSystem&,
    const PeriodicQuadratureOptions&,const ResourceBudget&);
PeriodicQuadratureResult build_periodic_quadrature(const PeriodicSystem&,
    const PeriodicQuadratureOptions&,const ResourceBudget&);
} // namespace vibeqc::manta
