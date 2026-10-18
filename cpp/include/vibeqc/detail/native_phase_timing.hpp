#pragma once
// Optional native phase attribution. Timings never enter numerical decisions.
// Like MANTA SectionClock, the disabled path performs no clock reads.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vibeqc::detail {
using PhaseClock = std::chrono::steady_clock;
struct NativePhase {
    std::string name, measure;
    double seconds = 0;
    std::uint64_t intervals = 0;
};
struct NativePhaseRecord {
    std::string kernel;
    double wall_seconds = 0;
    bool completed = false;
    std::vector<std::pair<std::string, std::uint64_t>> dimensions;
    std::vector<NativePhase> phases;
};
struct NativePhaseBatch {
    std::vector<NativePhaseRecord> records;
    std::uint64_t dropped_calls = 0;
};
struct NativePhaseFrame {
    std::uint64_t token;
    bool enabled;
    NativePhaseBatch batch;
    std::size_t active_calls = 0;
};
inline thread_local std::vector<std::unique_ptr<NativePhaseFrame>> native_phase_frames;
inline std::atomic<std::uint64_t> native_phase_next_token{1};

inline std::uint64_t begin_native_phase_profile(bool enabled) {
    const auto token = native_phase_next_token.fetch_add(1);
    native_phase_frames.push_back(std::make_unique<NativePhaseFrame>(
        NativePhaseFrame{token, enabled, {}}));
    return token;
}
inline NativePhaseBatch take_native_phase_profile(std::uint64_t token, bool finish) {
    if (native_phase_frames.empty() || native_phase_frames.back()->token != token)
        throw std::invalid_argument("native phase profile requires its owning thread and innermost token");
    if (finish && native_phase_frames.back()->active_calls)
        throw std::invalid_argument("cannot finish a profile with active native calls");
    auto batch = std::move(native_phase_frames.back()->batch);
    native_phase_frames.back()->batch = {};
    if (finish) native_phase_frames.pop_back();
    return batch;
}

// A worker owns its counters. No lock, allocation or clock read when disabled.
template<std::size_t N> class NativePhaseSections {
    bool enabled_;
    PhaseClock::time_point previous_;
public:
    std::array<double, N> seconds{};
    std::array<std::uint64_t, N> intervals{};
    explicit NativePhaseSections(bool enabled) : enabled_(enabled),
        previous_(enabled ? PhaseClock::now() : PhaseClock::time_point{}) {}
    void finish(std::size_t section) {
        if (!enabled_) return;
        const auto now = PhaseClock::now();
        seconds[section] += std::chrono::duration<double>(now - previous_).count();
        ++intervals[section];
        previous_ = now;
    }
};

template<std::size_t N> class NativePhaseCall {
    NativePhaseFrame* frame_ = nullptr;
    NativePhaseRecord record_;
    std::array<const char*, N> names_;
    std::array<double, N> worker_seconds_{};
    std::array<std::uint64_t, N> worker_intervals_{};
    std::mutex mutex_;
    PhaseClock::time_point started_;
    int exceptions_ = 0;
public:
    NativePhaseSections<N> elapsed;
    bool enabled() const { return frame_ != nullptr; }
    NativePhaseCall(const char* kernel, std::array<const char*, N> names,
        std::initializer_list<std::pair<const char*, std::uint64_t>> dimensions)
        : frame_(native_phase_frames.empty() || !native_phase_frames.back()->enabled
                 ? nullptr : native_phase_frames.back().get()), names_(names),
          started_(enabled() ? PhaseClock::now() : PhaseClock::time_point{}),
          exceptions_(std::uncaught_exceptions()), elapsed(enabled()) {
        if (enabled()) {
            record_.kernel = kernel;
            for (const auto& item : dimensions) record_.dimensions.emplace_back(item.first, item.second);
            ++frame_->active_calls;
        }
    }
    void add_worker(const NativePhaseSections<N>& worker) {
        if (!enabled()) return;
        const std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < N; ++i) {
            worker_seconds_[i] += worker.seconds[i];
            worker_intervals_[i] += worker.intervals[i];
        }
    }
    ~NativePhaseCall() noexcept {
        if (!enabled()) return;
        --frame_->active_calls;
        // Optional telemetry must not replace a calculation exception. Storage
        // is bounded between snapshots, with explicit loss rather than growth.
        try {
            if (frame_->batch.records.size() >= 256) {
                ++frame_->batch.dropped_calls;
                return;
            }
            record_.wall_seconds = std::chrono::duration<double>(PhaseClock::now() - started_).count();
            record_.completed = std::uncaught_exceptions() == exceptions_;
            for (std::size_t i = 0; i < N; ++i) {
                if (elapsed.intervals[i]) record_.phases.push_back(
                    {names_[i], "elapsed_wall", elapsed.seconds[i], elapsed.intervals[i]});
                if (worker_intervals_[i]) record_.phases.push_back(
                    {names_[i], "summed_worker_wall", worker_seconds_[i], worker_intervals_[i]});
            }
            frame_->batch.records.push_back(std::move(record_));
        } catch (...) { ++frame_->batch.dropped_calls; }
    }
};
} // namespace vibeqc::detail
