#pragma once

// Opt-in SCF component timing (GPU programme stage G0-A; see
// docs/design/gpu_g0_g1m_plan.md, "Instrumentation and evidence").
//
// A molecular driver whose options set ``component_timing = true`` opens a
// ScfProfileSession for the duration of the call. While a session is active
// on the calling thread, ScfPhaseScope objects placed at phase boundaries
// record wall-clock intervals. Without an active session every hook is a
// single thread-local pointer test, so default runs are unchanged.
//
// Accounting contract:
//   * ``inclusive_s`` of a phase is the wall time between its start and stop.
//   * ``exclusive_s`` subtracts the inclusive time of phases nested inside
//     it. Paths join nested names with '/', so a fused driver call keeps one
//     timer and any separate builder kernels appear only as its children.
//   * The exclusive times of all phases plus ``unclassified_s`` equal the
//     session ``wall_s`` (driver entry to return), up to clock resolution.
//   * Iteration buckets run from one loop-top to the next (the last one ends
//     at loop exit), so a diagonalization is charged to the iteration that
//     performed it. This differs from SCFIteration::wall_s, which ends at the
//     trace record.
//   * Scopes opened on OpenMP worker threads are ignored: only the thread
//     that opened the session records phases.
//
// Memory samples are tracked container sizes at named construction
// boundaries, in bytes. They do not include allocator, BLAS or solver
// scratch and are not process RSS.

#include <string>
#include <vector>

namespace vibeqc {

struct ScfPhaseTiming {
    std::string path;          // '/'-joined nesting path
    double inclusive_s = 0.0;
    double exclusive_s = 0.0;
    long long calls = 0;
};

struct ScfIterationTiming {
    int segment = 0;           // SCF segment (COSX grid stage), 0-based
    int iter = 0;              // driver iteration number within the segment
    double wall_s = 0.0;       // loop-top to next loop-top or loop exit
    std::vector<ScfPhaseTiming> phases;  // per path, within this iteration
};

struct ScfMemorySample {
    std::string label;
    double bytes = 0.0;
};

struct ScfComponentTiming {
    bool enabled = false;
    double wall_s = 0.0;           // enclosing session clock
    double unclassified_s = 0.0;   // wall_s minus all exclusive phase time
    int n_segments = 0;
    std::vector<ScfPhaseTiming> phases;
    std::vector<ScfIterationTiming> iterations;
    std::vector<ScfMemorySample> memory;
};

// RAII session. ``enabled`` is the caller's options flag; ``scf_segment``
// marks an SCF-loop entry point (run_*_scf_with_jk), which opens a new
// iteration segment. A session constructed while another is active never
// owns it: an enabled nested session only advances the segment, and a
// disabled nested session (for example an atomic guess SCF) suppresses
// iteration records while its phases still nest under the active phase.
class ScfProfileSession {
public:
    ScfProfileSession(bool enabled, bool scf_segment);
    ~ScfProfileSession();
    ScfProfileSession(const ScfProfileSession&) = delete;
    ScfProfileSession& operator=(const ScfProfileSession&) = delete;

    bool owns() const noexcept { return owner_; }
    // Owner only: close the session and return its record. Non-owners and
    // disabled sessions return a default (enabled = false) record.
    ScfComponentTiming finish();

private:
    bool owner_ = false;
    bool suppressing_ = false;
    bool finished_ = false;
};

class ScfPhaseScope {
public:
    explicit ScfPhaseScope(const char* name);
    ~ScfPhaseScope();
    ScfPhaseScope(const ScfPhaseScope&) = delete;
    ScfPhaseScope& operator=(const ScfPhaseScope&) = delete;
    void stop();

private:
    bool open_ = false;
};

bool scf_profile_active() noexcept;
void scf_profile_begin_iteration(int iter);
void scf_profile_end_iteration();
void scf_profile_memory(const char* label, double bytes);

}  // namespace vibeqc
