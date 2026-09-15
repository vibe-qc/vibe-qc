// Global C++ diagnostics channel — routed into the Python ``vibeqc.output``
// layer via a registered callback.  Zero-cost when diagnostics are off.
//
// Design
// ------
// * ``g_diag_level`` — a single global verbosity threshold (0=QUIET …
//   3=DEBUG), kept in sync with Python's ``channel.Level`` by the bridge
//   module ``vibeqc.output._cpp_diagnostics``.
//
// * ``g_diag_sink`` — a function pointer.  Null → no output; set →
//   diagnostics are forwarded to the Python callback (which routes them
//   into the active ``OutputChannel``).  The C++ side never opens a file
//   or writes to a stream directly.
//
// * ``diagnostic(tag, level, msg)`` — inline fast path.  One relaxed
//   atomic load + one branch; the message is never even evaluated when
//   the level is too low or no sink is installed.
//
// * ``VIBEQC_DIAG(tag, lvl, fmt, ...)`` — convenience macro for printf-
//   style formatted messages.  Uses ``std::snprintf`` into a stack
//   buffer (512 bytes); the formatting cost is only paid when the
//   message would actually be emitted.
//
// Thread safety
// -------------
// The level and sink pointer are atomic (lock-free on all platforms).
// The sink must be installed once, before any parallel region, and not
// changed during a computation.  The callback itself acquires the GIL
// when called from a GIL-released thread.
//
// ``snprintf`` is deliberately chosen over ``fprintf`` so this header
// passes ``tests/test_cpp_emits_no_user_output.py`` without an
// allowlist entry — the regex accepts ``fprintf`` / bare ``printf``
// only, not ``snprintf``.

#pragma once

#include <atomic>
#include <cstdio>
#include <string>

namespace vibeqc {

// ---- verbosity bands (must match Python ``vibeqc.output.Level``) -----------

enum class DiagLevel : int {
    QUIET    = 0,   // essential results only
    STANDARD = 1,   // normal .out content (the default)
    VERBOSE  = 2,   // extra detail
    DEBUG    = 3,   // developer diagnostics
};

// ---- sink type ------------------------------------------------------------

/// Callback signature: (tag, level_int, message).
/// ``level_int`` is the raw ``int`` of the ``DiagLevel`` so the Python
/// bridge can pass it straight to ``channel.write(…, Level(level_int))``.
using DiagSink = void (*)(const char* tag, int level, const char* msg);

// ---- global state ---------------------------------------------------------

/// Verbosity threshold.  A diagnostic with ``level > g_diag_level`` is
/// suppressed.  Default ``STANDARD`` so ``VERBOSE`` / ``DEBUG`` calls are
/// silent until the level is raised.
inline std::atomic<int> g_diag_level{
    static_cast<int>(DiagLevel::STANDARD)};

/// Sink function pointer.  ``nullptr`` → no output (default).  Set by
/// the Python bridge at initialisation.
inline std::atomic<DiagSink> g_diag_sink{nullptr};

// ---- public API (hot-path inline) -----------------------------------------

/// Emit a pre-formatted diagnostic message.
///
/// One relaxed atomic load + one branch; the call through the function
/// pointer happens only when the level check passes AND a sink is
/// installed.
inline void diagnostic(const char* tag, DiagLevel level, const char* msg) {
    if (static_cast<int>(level) > g_diag_level.load(std::memory_order_relaxed))
        return;
    auto sink = g_diag_sink.load(std::memory_order_relaxed);
    if (sink)
        sink(tag, static_cast<int>(level), msg);
}

/// Emit a diagnostic message from a ``std::string`` or ``std::ostringstream``.
inline void diagnostic(const char* tag, DiagLevel level,
                       const std::string& msg) {
    diagnostic(tag, level, msg.c_str());
}

}  // namespace vibeqc

// ---- convenience macro ----------------------------------------------------

/// ``VIBEQC_DIAG("subsystem", DiagLevel::DEBUG, "x = %.3f", x)``
///
/// The formatted message is built into a 512-byte stack buffer; the
/// formatting cost is only paid when the global level and sink checks
/// pass.  In the common case (level too low or no sink) this compiles to
/// a single relaxed atomic load + branch — no allocation, no formatting.
#define VIBEQC_DIAG(tag, lvl, fmt, ...)                                      \
    do {                                                                     \
        if (static_cast<int>(lvl) <=                                         \
            ::vibeqc::g_diag_level.load(std::memory_order_relaxed)) {        \
            auto _sink =                                                     \
                ::vibeqc::g_diag_sink.load(std::memory_order_relaxed);       \
            if (_sink) {                                                     \
                char _vibeqc_diag_buf[512];                                  \
                std::snprintf(_vibeqc_diag_buf, sizeof(_vibeqc_diag_buf),    \
                              fmt, __VA_ARGS__);                             \
                _sink(tag, static_cast<int>(lvl), _vibeqc_diag_buf);         \
            }                                                                \
        }                                                                    \
    } while (0)
