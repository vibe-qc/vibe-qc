#pragma once
#include "vibeqc/manta_bupo_errors.hpp"
// Internal reuse of the molecular Gaussian ledger for a translated bare
// Coulomb image. Only native source-owning consumers supply these envelopes.
namespace vibeqc::manta::bupo_detail {
BubbleInteraction interaction(const md::Point&,double,const BubbleEnvelope&,
    const md::Point&,double,const BubbleEnvelope&,int,const ResourceBudget&);
}
