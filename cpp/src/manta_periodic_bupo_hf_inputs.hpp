#pragma once
#include "vibeqc/manta_periodic_bupo_hf.hpp"

namespace vibeqc::manta::bupo_hf_detail {
// Shared physical-input checks, also used before staged source construction.
// The caller admits the atom/shell traversal before calling this function.
void validate_inputs(const BasisSet&,const PeriodicSystem&,const PeriodicBupoHFOptions&);
}
