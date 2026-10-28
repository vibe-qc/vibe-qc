#include "vibeqc/scf_profile.hpp"

#include <chrono>
#include <memory>
#include <unordered_map>
#include <utility>

namespace vibeqc {
namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Insertion-ordered phase table keyed by path.
struct PhaseTable {
    std::vector<ScfPhaseTiming> rows;
    std::unordered_map<std::string, std::size_t> index;

    void add(const std::string& path, double inclusive, double exclusive) {
        auto it = index.find(path);
        if (it == index.end()) {
            index.emplace(path, rows.size());
            rows.push_back({path, inclusive, exclusive, 1});
            return;
        }
        auto& row = rows[it->second];
        row.inclusive_s += inclusive;
        row.exclusive_s += exclusive;
        ++row.calls;
    }
};

struct Frame {
    std::string path;
    Clock::time_point start;
    double child_s = 0.0;
};

struct Profiler {
    Clock::time_point t0 = Clock::now();
    std::vector<Frame> stack;
    PhaseTable total;
    std::vector<ScfMemorySample> memory;
    std::vector<ScfIterationTiming> iterations;
    PhaseTable iteration_phases;
    bool iteration_open = false;
    int iteration_number = 0;
    Clock::time_point iteration_start;
    int segment = -1;
    int suppress = 0;

    void close_iteration(Clock::time_point now) {
        if (!iteration_open) return;
        ScfIterationTiming rec;
        rec.segment = segment < 0 ? 0 : segment;
        rec.iter = iteration_number;
        rec.wall_s = seconds(iteration_start, now);
        rec.phases = std::move(iteration_phases.rows);
        iterations.push_back(std::move(rec));
        iteration_phases = PhaseTable{};
        iteration_open = false;
    }
};

thread_local Profiler* t_active = nullptr;
thread_local std::unique_ptr<Profiler> t_owned;

}  // namespace

ScfProfileSession::ScfProfileSession(bool enabled, bool scf_segment) {
    if (t_active == nullptr) {
        if (!enabled) return;
        t_owned = std::make_unique<Profiler>();
        t_active = t_owned.get();
        owner_ = true;
        if (scf_segment) t_active->segment = 0;
        return;
    }
    if (!enabled) {
        ++t_active->suppress;
        suppressing_ = true;
        return;
    }
    if (scf_segment && t_active->suppress == 0) {
        t_active->close_iteration(Clock::now());
        ++t_active->segment;
    }
}

ScfProfileSession::~ScfProfileSession() {
    if (suppressing_ && t_active != nullptr) --t_active->suppress;
    if (owner_ && !finished_) {
        t_active = nullptr;
        t_owned.reset();
    }
}

ScfComponentTiming ScfProfileSession::finish() {
    ScfComponentTiming out;
    if (!owner_ || finished_ || t_active == nullptr) return out;
    finished_ = true;
    Profiler& p = *t_active;
    const auto now = Clock::now();
    p.close_iteration(now);
    out.enabled = true;
    out.wall_s = seconds(p.t0, now);
    double exclusive_sum = 0.0;
    for (const auto& row : p.total.rows) exclusive_sum += row.exclusive_s;
    out.unclassified_s = out.wall_s - exclusive_sum;
    out.n_segments = p.segment < 0 ? 0 : p.segment + 1;
    out.phases = std::move(p.total.rows);
    out.iterations = std::move(p.iterations);
    out.memory = std::move(p.memory);
    t_active = nullptr;
    t_owned.reset();
    return out;
}

ScfPhaseScope::ScfPhaseScope(const char* name) {
    Profiler* p = t_active;
    if (p == nullptr) return;
    std::string path = p->stack.empty()
        ? std::string(name)
        : p->stack.back().path + "/" + name;
    p->stack.push_back({std::move(path), Clock::now(), 0.0});
    open_ = true;
}

ScfPhaseScope::~ScfPhaseScope() { stop(); }

void ScfPhaseScope::stop() {
    if (!open_) return;
    open_ = false;
    Profiler* p = t_active;
    if (p == nullptr || p->stack.empty()) return;
    const auto now = Clock::now();
    Frame frame = std::move(p->stack.back());
    p->stack.pop_back();
    const double inclusive = seconds(frame.start, now);
    const double exclusive = inclusive - frame.child_s;
    if (!p->stack.empty()) p->stack.back().child_s += inclusive;
    p->total.add(frame.path, inclusive, exclusive);
    if (p->iteration_open) {
        p->iteration_phases.add(frame.path, inclusive, exclusive);
    }
}

bool scf_profile_active() noexcept { return t_active != nullptr; }

void scf_profile_begin_iteration(int iter) {
    Profiler* p = t_active;
    if (p == nullptr || p->suppress > 0) return;
    const auto now = Clock::now();
    p->close_iteration(now);
    p->iteration_open = true;
    p->iteration_number = iter;
    p->iteration_start = now;
}

void scf_profile_end_iteration() {
    Profiler* p = t_active;
    if (p == nullptr || p->suppress > 0) return;
    p->close_iteration(Clock::now());
}

void scf_profile_memory(const char* label, double bytes) {
    Profiler* p = t_active;
    if (p == nullptr) return;
    p->memory.push_back({label, bytes});
}

}  // namespace vibeqc
