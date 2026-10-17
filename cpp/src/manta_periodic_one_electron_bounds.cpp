#include "vibeqc/manta_periodic_one_electron_bounds.hpp"
#include "manta_pair_image_internal.hpp"
#include "vibeqc/manta_admission.hpp"

namespace vibeqc::manta {
struct PeriodicOneElectronImageDriver {
    static PeriodicOneElectronImagePlan plan(const PeriodicPairSource& s,const ResourceBudget& budget) {
        using namespace admission;
        const auto base=pair_image_detail::plan(s.basis_,s.entries_,s.images_,s.plan_.momenta,budget);
        PeriodicOneElectronImagePlan p;p.n_basis=s.basis_.nbasis();p.momenta=s.plan_.momenta;
        p.cube_radius=base.radius;p.cube_cells=base.cells;
        p.shell_calls=base.shell_calls;p.primitive_pairs=base.primitive_products;
        p.scalar_terms=add(4096,mul(2,base.scalar_terms));p.kernel_terms=mul(128,p.scalar_terms);
        // The source stays alive alongside primitive gradient majorants,
        // both error matrices, owned exports and checked lattice geometry.
        p.modeled_peak_bytes=add(base.modeled_peak_bytes,add(s.plan_.retained_bytes,
            add(16384,mul(128,square(p.n_basis)))));
        cap(p.scalar_terms,budget.scalar_terms,"MANTA one-electron image scalar budget exceeded");
        cap(p.kernel_terms,budget.kernel_terms,"MANTA one-electron image kernel budget exceeded");
        cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA one-electron image memory budget exceeded");
        return p;
    }
    static PeriodicOneElectronImageResult run(const PeriodicPairSource& s,const ResourceBudget& budget) {
        const auto p=plan(s,budget);
        const auto base=pair_image_detail::plan(s.basis_,s.entries_,s.images_,s.plan_.momenta,budget);
        auto out=pair_image_detail::one_electron(s.basis_,s.lattice_,s.entries_,s.receipt_.source_identity,base);
        out.plan=p;return out;
    }
};
PeriodicOneElectronImagePlan plan_periodic_one_electron_images(const PeriodicPairSource& s,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::plan(s,b);
}
PeriodicOneElectronImageResult bound_periodic_one_electron_images(const PeriodicPairSource& s,const ResourceBudget& b) {
    return PeriodicOneElectronImageDriver::run(s,b);
}
} // namespace vibeqc::manta
