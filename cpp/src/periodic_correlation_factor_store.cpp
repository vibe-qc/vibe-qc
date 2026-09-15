#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "vibeqc/periodic_correlation_factor_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include "vibeqc/detail/sha256.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace vibeqc {
namespace {

using Complex = std::complex<double>;
using Bytes32 = std::array<std::uint8_t, 32>;
using Header = std::array<std::uint8_t, 1024>;
using Record = std::array<std::uint8_t, 256>;
using Footer = std::array<std::uint8_t, 256>;
constexpr std::uint64_t kH = kPeriodicCorrelationPrivateFactorStoreHeaderBytes;
constexpr std::uint64_t kR = kPeriodicCorrelationPrivateFactorStoreRecordBytes;
constexpr std::uint64_t kF = kPeriodicCorrelationPrivateFactorStoreFooterBytes;
constexpr std::uint64_t kPage = kPeriodicCorrelationPrivateFactorStoreCodecPageBytes;
constexpr std::size_t kRecordDigestOffset = 204;
constexpr std::size_t kFooterDigestOffset = 144;
constexpr double kPi = 3.14159265358979323846;
std::atomic<std::uint64_t> live_descriptors{0};

std::uint64_t plus(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::length_error("private factor store byte count overflow");
    return a + b;
}
std::uint64_t times(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
        throw std::length_error("private factor store extent overflow");
    return a * b;
}
void require_finite(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("private factor store rejects non-finite values");
}
void put64(std::uint8_t* out, std::uint64_t x) {
    for (unsigned i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>(x >> (56 - 8 * i));
}
void put32(std::uint8_t* out, std::uint32_t x) {
    for (unsigned i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(x >> (24 - 8 * i));
}
std::uint64_t get64(const std::uint8_t* in) {
    std::uint64_t x = 0;
    for (unsigned i = 0; i < 8; ++i) x = (x << 8) | in[i];
    return x;
}
void put_real(std::uint8_t* out, double x) {
    require_finite(x);
    if (x == 0.0) x = 0.0;
    std::uint64_t bits;
    std::memcpy(&bits, &x, 8);
    put64(out, bits);
}
double get_real(const std::uint8_t* in) {
    const auto bits = get64(in);
    double x;
    std::memcpy(&x, &bits, 8);
    require_finite(x);
    return x;
}
Bytes32 binary_digest(const std::string& hex) {
    if (hex.size() != 64) throw std::invalid_argument("private factor store requires SHA-256 identities");
    auto nibble = [](char c) -> std::uint8_t {
        if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
        throw std::invalid_argument("private factor store requires lowercase SHA-256 identities");
    };
    Bytes32 bytes{};
    for (std::size_t i = 0; i < 32; ++i) bytes[i] = 16 * nibble(hex[2 * i]) + nibble(hex[2 * i + 1]);
    return bytes;
}
std::string hexadecimal(const std::uint8_t* bytes) {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string text(64, '0');
    for (std::size_t i = 0; i < 32; ++i) {
        text[2 * i] = alphabet[bytes[i] >> 4];
        text[2 * i + 1] = alphabet[bytes[i] & 15];
    }
    return text;
}
struct Hash {
    detail::Sha256 impl;
    void raw(const void* p, std::size_t n) { impl.update(static_cast<const std::uint8_t*>(p), n); }
    void u32(std::uint32_t x) { std::array<std::uint8_t, 4> b{}; put32(b.data(), x); raw(b.data(), b.size()); }
    void u64(std::uint64_t x) { std::array<std::uint8_t, 8> b{}; put64(b.data(), x); raw(b.data(), b.size()); }
    void text(const std::string& x) { u64(x.size()); raw(x.data(), x.size()); }
    void real(double x) { std::array<std::uint8_t, 8> b{}; put_real(b.data(), x); raw(b.data(), b.size()); }
    std::string finish() { return impl.finish_hex(); }
};
Hash domain(const char* text) { Hash h; h.text(text); h.u32(1); return h; }

struct Encode {
    std::uint8_t* bytes;
    std::size_t cursor = 0;
    void u32(std::uint32_t x) { put32(bytes + cursor, x); cursor += 4; }
    void u64(std::uint64_t x) { put64(bytes + cursor, x); cursor += 8; }
    void real(double x) { put_real(bytes + cursor, x); cursor += 8; }
    void digest(const std::string& x) {
        const auto b = binary_digest(x);
        std::copy(b.begin(), b.end(), bytes + cursor);
        cursor += b.size();
    }
};
void encode_descriptor(Encode& e, const PeriodicCorrelationFactorTileDescriptor& d) {
    e.u64(d.sequence_index); e.u64(d.q_index); e.u64(d.k_bra_index); e.u64(d.k_ket_index);
    for (const int x : d.k_ket_reciprocal_wrap) e.u32(static_cast<std::uint32_t>(x));
    e.u64(d.ao_pair_begin); e.u64(d.ao_pair_count);
    e.u64(d.auxiliary_begin); e.u64(d.auxiliary_count); e.u64(d.element_count);
}
void hash_descriptor(Hash& h, const PeriodicCorrelationFactorTileDescriptor& d) {
    h.u64(d.sequence_index); h.u64(d.q_index); h.u64(d.k_bra_index); h.u64(d.k_ket_index);
    for (const int x : d.k_ket_reciprocal_wrap) h.u32(static_cast<std::uint32_t>(x));
    h.u64(d.ao_pair_begin); h.u64(d.ao_pair_count);
    h.u64(d.auxiliary_begin); h.u64(d.auxiliary_count); h.u64(d.element_count);
}

[[noreturn]] void io_error(const char* operation, int code = errno) {
    throw std::runtime_error(std::string("private factor store ") + operation
                             + " failed (error " + std::to_string(code) + ")");
}
struct Fd {
    int value = -1;
    Fd() = default;
    explicit Fd(int fd) : value(fd) { if (fd >= 0) ++live_descriptors; }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
    Fd& operator=(Fd&& other) noexcept {
        if (this != &other) { reset(); value = std::exchange(other.value, -1); }
        return *this;
    }
    ~Fd() { reset(); }
    int release_number() noexcept {
        const int old = std::exchange(value, -1);
        if (old >= 0) --live_descriptors;
        return old;
    }
    void reset() noexcept {
        const int old = release_number();
#if defined(__linux__) || defined(__APPLE__)
        // Never retry close: a retry can close a reused descriptor on the
        // supported platforms. Destructors deliberately cannot throw.
        if (old >= 0) static_cast<void>(::close(old));
#else
        static_cast<void>(old);
#endif
    }
    void close_checked() {
        const int old = release_number();
#if defined(__linux__) || defined(__APPLE__)
        if (old >= 0 && ::close(old) != 0) io_error("close");
#else
        static_cast<void>(old);
#endif
    }
};

#if defined(__linux__) || defined(__APPLE__)
static_assert(sizeof(off_t) >= 8, "private factor store requires 64-bit POSIX offsets");
off_t offset(std::uint64_t n) {
    if (n > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
        throw std::length_error("private factor store exceeds POSIX offset range");
    return static_cast<off_t>(n);
}
void write_all(int fd, const std::uint8_t* bytes, std::size_t n, std::uint64_t at) {
    while (n) {
        const auto part = std::min(n, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto written = ::pwrite(fd, bytes, part, offset(at));
        if (written < 0) { if (errno == EINTR) continue; io_error("pwrite"); }
        if (written == 0) throw std::runtime_error("private factor store pwrite made no progress");
        bytes += written; n -= static_cast<std::size_t>(written); at = plus(at, written);
    }
}
void read_all(int fd, std::uint8_t* bytes, std::size_t n, std::uint64_t at) {
    while (n) {
        const auto part = std::min(n, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto count = ::pread(fd, bytes, part, offset(at));
        if (count < 0) { if (errno == EINTR) continue; io_error("pread"); }
        if (count == 0) throw std::runtime_error("private factor store is truncated");
        bytes += count; n -= static_cast<std::size_t>(count); at = plus(at, count);
    }
}
void exact_size(int fd, std::uint64_t expected) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) io_error("fstat");
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || static_cast<std::uint64_t>(st.st_size) != expected)
        throw std::runtime_error("private factor store has incorrect file extent or type");
}
void sync_file(int fd) {
    while (::fsync(fd) != 0) { if (errno != EINTR) io_error("fsync"); }
}
void reserve_storage(int fd, std::uint64_t bytes) {
#if defined(__linux__)
    while (::fallocate(fd, 0, 0, offset(bytes)) != 0) {
        if (errno != EINTR) io_error("mandatory storage reservation");
    }
#else
    fstore_t request{};
    request.fst_flags = F_ALLOCATEALL;
    request.fst_posmode = F_PEOFPOSMODE;
    request.fst_length = offset(bytes);
    if (::fcntl(fd, F_PREALLOCATE, &request) != 0) io_error("mandatory storage reservation");
    if (request.fst_bytesalloc < offset(bytes))
        throw std::runtime_error("private factor store reservation was incomplete");
    while (::ftruncate(fd, offset(bytes)) != 0) {
        if (errno != EINTR) io_error("ftruncate after reservation");
    }
#endif
    exact_size(fd, bytes);
}
std::pair<Fd, Fd> private_file(const std::string& directory, std::uint64_t bytes,
                              std::uint64_t required_scratch) {
    if (directory.empty() || directory.size() > 4096 || directory.find('\0') != std::string::npos)
        throw std::invalid_argument("private factor store requires a bounded scratch directory path");
    Fd dir(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (dir.value < 0) io_error("open private directory");
    struct stat directory_stat{};
    if (::fstat(dir.value, &directory_stat) != 0) io_error("fstat directory");
    if (!S_ISDIR(directory_stat.st_mode) || directory_stat.st_uid != ::geteuid()
        || (directory_stat.st_mode & 0077) != 0)
        throw std::invalid_argument("factor scratch directory must be caller-owned and private");
    struct statvfs free_space{};
    if (::fstatvfs(dir.value, &free_space) != 0) io_error("fstatvfs");
    if (times(free_space.f_bavail, free_space.f_frsize) < required_scratch)
        throw std::runtime_error("private factor store has insufficient available scratch capacity");
    std::array<char, 64> name{};
    constexpr char prefix[] = ".vibeqc-factor-";
    constexpr char alphabet[] = "0123456789abcdef";
    Fd writer;
    for (unsigned attempt = 0; attempt < 16 && writer.value < 0; ++attempt) {
        std::array<std::uint8_t, 16> random{};
        if (::getentropy(random.data(), random.size()) != 0) io_error("temporary name entropy");
        std::copy(prefix, prefix + sizeof(prefix) - 1, name.begin());
        for (std::size_t i = 0; i < random.size(); ++i) {
            name[sizeof(prefix) - 1 + 2 * i] = alphabet[random[i] >> 4];
            name[sizeof(prefix) + 2 * i] = alphabet[random[i] & 15];
        }
        writer = Fd(::openat(dir.value, name.data(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
        if (writer.value < 0 && errno != EEXIST) io_error("exclusive temporary creation");
    }
    if (writer.value < 0) throw std::runtime_error("private factor store exhausted exclusive-name attempts");
    struct stat created{};
    bool created_identity_known = false;
    bool linked = true;
    const auto capture_created_identity = [&]() {
        while (::fstat(writer.value, &created) != 0) {
            if (errno != EINTR) io_error("fstat exclusive temporary");
        }
        created_identity_known = true;
    };
    const auto unlink_owned = [&]() {
        // Even an initial descriptor-stat failure enters verified cleanup.
        // Persistent stat failure must not turn into an unverified unlink.
        if (!created_identity_known) capture_created_identity();
        struct stat named{};
        for (;;) {
            if (::fstatat(dir.value, name.data(), &named, AT_SYMLINK_NOFOLLOW) != 0) {
                if (errno == EINTR) continue;
                if (errno == ENOENT) { linked = false; return; }
                io_error("fstatat temporary");
            }
            if (!S_ISREG(named.st_mode) || named.st_dev != created.st_dev || named.st_ino != created.st_ino)
                throw std::runtime_error("private factor store temporary identity changed");
            if (::unlinkat(dir.value, name.data(), 0) == 0) { linked = false; return; }
            // Recheck the directory entry after interruption, never blindly
            // retry an unlink whose first call may already have succeeded.
            if (errno != EINTR) io_error("unlink owned temporary");
        }
    };
    try {
        capture_created_identity();
        Fd reader(::openat(dir.value, name.data(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (reader.value < 0) io_error("open read-only temporary");
        struct stat read_stat{};
        if (::fstat(reader.value, &read_stat) != 0) io_error("fstat read-only temporary");
        if (!S_ISREG(created.st_mode) || created.st_nlink != 1 || created.st_size != 0
            || read_stat.st_dev != created.st_dev || read_stat.st_ino != created.st_ino)
            throw std::runtime_error("private factor store descriptors do not name one exclusive regular inode");
        unlink_owned();
        reserve_storage(writer.value, bytes);
        return {std::move(writer), std::move(reader)};
    } catch (...) {
        if (linked) {
            try { unlink_owned(); }
            catch (...) {
                // The file is still empty: storage is reserved only after
                // unlink succeeds. Preserve an unverified replacement, and
                // report the possible owned-name remainder explicitly.
                std::throw_with_nested(std::runtime_error(
                    "private factor store could not verify temporary cleanup; "
                    "an empty owned temporary name may remain"));
            }
        }
        throw;
    }
}
#else
std::uint64_t offset(std::uint64_t) { throw std::runtime_error("private factor store requires Linux or macOS"); }
void write_all(int, const std::uint8_t*, std::size_t, std::uint64_t) { offset(0); }
void read_all(int, std::uint8_t*, std::size_t, std::uint64_t) { offset(0); }
void exact_size(int, std::uint64_t) { offset(0); }
void sync_file(int) { offset(0); }
std::pair<Fd, Fd> private_file(const std::string&, std::uint64_t, std::uint64_t) { offset(0); return {}; }
#endif

bool same_codec(const PeriodicCorrelationFactorBuildCodecInventory& x,
                const PeriodicCorrelationFactorBuildCodecInventory& y) {
#define SAME(field) if (x.field != y.field) return false
    SAME(complete); SAME(fixed_header_bytes); SAME(fixed_footer_bytes); SAME(fixed_manifest_bytes);
    SAME(integrity_bytes_per_tile); SAME(rank_bytes_per_q); SAME(digest_bytes_per_tile);
    SAME(journal_header_bytes); SAME(journal_record_bytes_per_tile); SAME(checkpoint_record_bytes);
    SAME(checkpoint_records_per_generation); SAME(existing_generation_count);
#undef SAME
    return true;
}

}  // namespace

namespace detail {
struct PrivateFactorStoreImpl {
    explicit PrivateFactorStoreImpl(PeriodicCorrelationFactorStreamSchedule s) : schedule(std::move(s)) {}
    PeriodicCorrelationFactorStreamSchedule schedule;
    PeriodicCorrelationPrivateFactorStoreCaps caps;
    PeriodicCorrelationPrivateFactorStoreState state = PeriodicCorrelationPrivateFactorStoreState::Open;
    Fd writer, reader;
    std::array<std::uint8_t, kPage> page{};
    Header header{};
    Footer footer{};
    Bytes32 header_digest{}, ao_digest{}, auxiliary_digest{}, census_digest{}, plan_digest{};
    Bytes32 previous_source{}, previous_whitener{}, completion_digest{}, storage_digest{};
    Eigen::Matrix3d direct_lattice = Eigen::Matrix3d::Zero();
    double image_cutoff = 0, negative_tolerance = 0;
    std::uint64_t extent = 0, accepted = 0, accepted_elements = 0, accepted_bytes = 0;
    std::uint64_t maximum_tile_bytes = 0, previous_reciprocal_count = 0;
    Hash stream, records;
    std::atomic<bool> visiting{false};

    std::string ao() const { return hexadecimal(ao_digest.data()); }
    std::string auxiliary() const { return hexadecimal(auxiliary_digest.data()); }
    std::string census() const { return hexadecimal(census_digest.data()); }
    std::string plan() const { return hexadecimal(plan_digest.data()); }
    void open() const {
        if (state != PeriodicCorrelationPrivateFactorStoreState::Open || writer.value < 0)
            throw std::logic_error("private factor writer is not open");
    }
    Hash record_hash(Record record) const {
        std::fill(record.begin() + kRecordDigestOffset, record.begin() + kRecordDigestOffset + 32, 0);
        auto h = domain("vibeqc.periodic.correlation.private-factor-store.record");
        h.raw(header_digest.data(), header_digest.size());
        h.raw(record.data(), record.size());
        return h;
    }
    Hash records_hash() const {
        auto h = domain("vibeqc.periodic.correlation.private-factor-store.records");
        h.raw(header_digest.data(), header_digest.size());
        h.u64(schedule.shape().tile_count);
        return h;
    }
    void initialize_header() {
        std::memcpy(header.data(), "VQPFST01", 8);
        Encode e{header.data(), 8};
        e.u32(1); e.u32(kH); e.u32(kR); e.u32(kF); e.u32(1); e.u32(1);
        const auto& s = schedule.shape();
        for (const auto x : {s.n_basis, s.n_auxiliary, s.n_kpoints, s.n_ao_pairs,
                            s.ao_pair_block, s.auxiliary_block, s.tile_count,
                            s.logical_element_count, s.logical_bytes, extent}) e.u64(x);
        for (const auto x : schedule.mesh()) e.u32(static_cast<std::uint32_t>(x));
        for (const auto x : schedule.is_shift()) e.u32(static_cast<std::uint32_t>(x));
        e.real(image_cutoff); e.real(negative_tolerance);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) e.real(direct_lattice(i, j));
        e.digest(schedule.state_identity_sha256()); e.digest(schedule.calculation_identity());
        e.digest(schedule.allocation_identity()); e.digest(schedule.schedule_identity_sha256());
        e.digest(census()); e.digest(plan()); e.digest(ao()); e.digest(auxiliary());
        e.digest(periodic_correlation_private_factor_store_codec_identity_sha256());
        auto image_policy = domain(kPeriodicCorrelationThreeCenterImagePolicy);
        auto lattice_policy = domain(kPeriodicCorrelationThreeCenterLatticePolicy);
        e.digest(image_policy.finish()); e.digest(lattice_policy.finish());
        if (e.cursor != 576) throw std::logic_error("private factor store header layout changed");
        auto h = domain("vibeqc.periodic.correlation.private-factor-store.header");
        h.raw(header.data(), header.size() - 32);
        header_digest = binary_digest(h.finish());
        std::copy(header_digest.begin(), header_digest.end(), header.end() - 32);
        records = records_hash();
        stream = domain("vibeqc.periodic.correlation.three-center.stream");
        stream.text(schedule.schedule_identity_sha256()); stream.text(census()); stream.text(plan());
        stream.real(image_cutoff); stream.real(negative_tolerance);
        stream.u64(s.n_kpoints); stream.u64(s.tile_count);
        stream.u64(s.logical_element_count); stream.u64(s.logical_bytes);
    }
    Record encode_record(const PeriodicCorrelationThreeCenterTile& tile) const {
        Record record{};
        Encode e{record.data()};
        encode_descriptor(e, tile.descriptor());
        e.u64(tile.image_candidate_count()); e.u64(tile.retained_pair_image_count());
        e.u64(tile.reciprocal_vector_count());
        e.digest(tile.source_identity_sha256()); e.digest(tile.whitener_payload_identity_sha256());
        e.digest(tile.payload_identity_sha256());
        if (e.cursor != kRecordDigestOffset) throw std::logic_error("private factor store record layout changed");
        const auto hash = binary_digest(record_hash(record).finish());
        std::copy(hash.begin(), hash.end(), record.begin() + kRecordDigestOffset);
        return record;
    }
    void validate_record(const Record& record, std::uint64_t sequence) const {
        std::array<std::uint8_t, 84> expected{};
        Encode e{expected.data()};
        encode_descriptor(e, schedule.descriptor(sequence));
        if (!std::equal(expected.begin(), expected.end(), record.begin())
            || !std::all_of(record.begin() + 236, record.end(), [](std::uint8_t x) { return x == 0; })
            || get64(record.data() + 92) > get64(record.data() + 84)
            || get64(record.data() + 100) == 0)
            throw std::runtime_error("private factor store record descriptor or counts are corrupt");
        const auto hash = binary_digest(record_hash(record).finish());
        if (!std::equal(hash.begin(), hash.end(), record.begin() + kRecordDigestOffset))
            throw std::runtime_error("private factor store record checksum mismatch");
    }
    Hash tile_hash_prefix(const Record& record, const PeriodicCorrelationFactorTileDescriptor& d) const {
        auto h = domain("vibeqc.periodic.correlation.three-center.payload");
        h.text(hexadecimal(record.data() + 108)); h.text(hexadecimal(record.data() + 140));
        h.text(ao()); h.text(auxiliary());
        h.text(kPeriodicCorrelationThreeCenterImagePolicy); h.text(kPeriodicCorrelationThreeCenterLatticePolicy);
        h.real(image_cutoff);
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) h.real(direct_lattice(i, j));
        const auto& ket = schedule.state().kpoint_cartesian(d.k_ket_index);
        for (int i = 0; i < 3; ++i) h.real(ket[i]);
        hash_descriptor(h, d);
        h.u64(get64(record.data() + 84)); h.u64(get64(record.data() + 92));
        h.u64(get64(record.data() + 100)); h.u64(times(d.element_count, 16));
        return h;
    }
    void metadata_extent(bool finalized) {
        exact_size(reader.value, extent);
        Header observed{};
        read_all(reader.value, observed.data(), observed.size(), 0);
        if (observed != header) throw std::runtime_error("private factor store header is corrupt");
        if (finalized) {
            Footer observed_footer{};
            read_all(reader.value, observed_footer.data(), observed_footer.size(), extent - kF);
            if (observed_footer != footer) throw std::runtime_error("private factor store footer is corrupt");
        }
    }
    void poison() noexcept {
        state = PeriodicCorrelationPrivateFactorStoreState::Failed;
        writer.reset(); reader.reset();
    }
};
}  // namespace detail

std::string periodic_correlation_private_factor_store_codec_identity_sha256() {
    auto h = domain("vibeqc.periodic.correlation.private-factor-store.codec");
    h.text("posix-ephemeral-unlinked-exclusive-0600;single-rank;canonical-sequential-exactly-once");
    h.text("principal-auxiliary-ao-rows;finite-image-reference;big-endian-complex-binary64");
    h.text("header1024;record256;footer256;page65536;no-bitmap;no-restart");
    return h.finish();
}
PeriodicCorrelationFactorBuildCodecInventory periodic_correlation_private_factor_store_codec_inventory() {
    PeriodicCorrelationFactorBuildCodecInventory c;
    c.complete = true; c.fixed_header_bytes = kH; c.fixed_footer_bytes = kF;
    c.integrity_bytes_per_tile = kR;
    return c;
}
std::uint64_t periodic_correlation_private_factor_store_fixed_control_bytes() noexcept {
    // Covers the fixed object, four cloned 64-byte schedule strings, fixed
    // validation/hash stack buffers and temporary scalar identity strings.
    return sizeof(detail::PrivateFactorStoreImpl) - kPage + 4 * 65 + 8192
        + sizeof(PeriodicCorrelationReciprocalMetricSourceManifest);
}
std::uint64_t periodic_correlation_private_factor_store_receiver_bytes(
    const PeriodicCorrelationFactorStreamSchedule& s) {
    return plus(kPage, s.shape().maximum_tile_bytes);
}
std::uint64_t periodic_correlation_private_factor_store_file_bytes(
    const PeriodicCorrelationFactorStreamSchedule& s) {
    return plus(plus(kH, kF), plus(times(s.shape().tile_count, kR), s.shape().logical_bytes));
}
std::uint64_t periodic_correlation_private_factor_store_record_offset(
    const PeriodicCorrelationFactorStreamSchedule& s, std::uint64_t sequence) {
    const auto d = s.descriptor(sequence);
    const auto& shape = s.shape();
    auto prefix = times(plus(times(d.q_index, shape.n_kpoints), d.k_bra_index),
                        times(shape.n_auxiliary, shape.n_ao_pairs));
    prefix = plus(prefix, times(d.ao_pair_begin, shape.n_auxiliary));
    prefix = plus(prefix, times(d.auxiliary_begin, d.ao_pair_count));
    return plus(kH, plus(times(sequence, kR), times(prefix, 16)));
}

PeriodicCorrelationPrivateFactorWriter::PeriodicCorrelationPrivateFactorWriter(
    std::unique_ptr<detail::PrivateFactorStoreImpl> p) noexcept : impl_(std::move(p)) {}
PeriodicCorrelationPrivateFactorWriter::PeriodicCorrelationPrivateFactorWriter(
    PeriodicCorrelationPrivateFactorWriter&&) noexcept = default;
PeriodicCorrelationPrivateFactorWriter& PeriodicCorrelationPrivateFactorWriter::operator=(
    PeriodicCorrelationPrivateFactorWriter&&) noexcept = default;
PeriodicCorrelationPrivateFactorWriter::~PeriodicCorrelationPrivateFactorWriter() = default;
PeriodicCorrelationPrivateFactorStoreState PeriodicCorrelationPrivateFactorWriter::state() const noexcept {
    return impl_ ? impl_->state : (finalized_ ? PeriodicCorrelationPrivateFactorStoreState::Finalized
                                            : PeriodicCorrelationPrivateFactorStoreState::Aborted);
}
std::uint64_t PeriodicCorrelationPrivateFactorWriter::accepted_tile_count() const noexcept {
    return impl_ ? impl_->accepted : 0;
}
std::uint64_t PeriodicCorrelationPrivateFactorWriter::file_bytes() const {
    if (!impl_) throw std::logic_error("moved-from private factor writer");
    return impl_->extent;
}
void PeriodicCorrelationPrivateFactorWriter::abort() noexcept {
    if (impl_ && impl_->state == PeriodicCorrelationPrivateFactorStoreState::Open) {
        impl_->state = PeriodicCorrelationPrivateFactorStoreState::Aborted;
        impl_->writer.reset(); impl_->reader.reset();
    }
}
void PeriodicCorrelationPrivateFactorWriter::accept(const PeriodicCorrelationThreeCenterTile& tile) {
    if (!impl_) throw std::logic_error("moved-from private factor writer");
    auto& p = *impl_;
    p.open();
    try {
        if (tile.contract_version() != 1 || tile.descriptor().sequence_index != p.accepted
            || tile.census_identity_sha256() != p.census() || tile.plan_identity_sha256() != p.plan()
            || tile.ao_basis_identity_sha256() != p.ao() || tile.auxiliary_basis_identity_sha256() != p.auxiliary()
            || tile.image_cutoff_bohr() != p.image_cutoff || !tile.finite_image_reference()
            || tile.ao_image_source_certified())
            throw std::invalid_argument("private factor store received wrong sequence or tile provenance");
        const auto d = p.schedule.descriptor(p.accepted);
        if (tile.matrix_row_major().size() != d.element_count
            || tile.output_bytes() != times(d.element_count, 16)
            || tile.output_bytes() > p.caps.maximum_tile_bytes)
            throw std::invalid_argument("private factor store tile extent differs from its schedule");
        const auto record = p.encode_record(tile);
        p.validate_record(record, p.accepted);
        const bool new_q = p.accepted % p.schedule.shape().tiles_per_q == 0;
        if (!new_q && (!std::equal(p.previous_source.begin(), p.previous_source.end(), record.begin() + 108)
            || !std::equal(p.previous_whitener.begin(), p.previous_whitener.end(), record.begin() + 140)
            || p.previous_reciprocal_count != tile.reciprocal_vector_count()))
            throw std::invalid_argument("private factor store source or whitener changed within q");
        auto payload = p.tile_hash_prefix(record, d);
        for (const auto z : tile.matrix_row_major()) { payload.real(z.real()); payload.real(z.imag()); }
        if (payload.finish() != tile.payload_identity_sha256())
            throw std::invalid_argument("private factor store numerical tile payload checksum mismatch");
        auto at = periodic_correlation_private_factor_store_record_offset(p.schedule, p.accepted);
        write_all(p.writer.value, record.data(), record.size(), at);
        at = plus(at, kR);
        const auto& values = tile.matrix_row_major();
        for (std::size_t begin = 0; begin < values.size();) {
            const auto count = std::min<std::size_t>(values.size() - begin, kPage / 16);
            for (std::size_t i = 0; i < count; ++i) {
                put_real(p.page.data() + 16 * i, values[begin + i].real());
                put_real(p.page.data() + 16 * i + 8, values[begin + i].imag());
            }
            write_all(p.writer.value, p.page.data(), count * 16, at);
            at = plus(at, count * 16); begin += count;
        }
        if (new_q) {
            p.stream.u64(d.q_index); p.stream.text(tile.source_identity_sha256());
            p.stream.text(tile.whitener_payload_identity_sha256());
            std::copy_n(record.begin() + 108, 32, p.previous_source.begin());
            std::copy_n(record.begin() + 140, 32, p.previous_whitener.begin());
            p.previous_reciprocal_count = tile.reciprocal_vector_count();
        }
        p.stream.u64(d.sequence_index); p.stream.text(tile.payload_identity_sha256());
        p.records.raw(record.data() + kRecordDigestOffset, 32);
        p.accepted = plus(p.accepted, 1);
        p.accepted_elements = plus(p.accepted_elements, d.element_count);
        p.accepted_bytes = plus(p.accepted_bytes, tile.output_bytes());
        p.maximum_tile_bytes = std::max(p.maximum_tile_bytes, tile.output_bytes());
    } catch (...) { p.poison(); throw; }
}

PeriodicCorrelationPrivateFactorReader PeriodicCorrelationPrivateFactorWriter::finish(
    const PeriodicCorrelationThreeCenterStreamReceipt& completion) {
    if (!impl_) throw std::logic_error("moved-from private factor writer");
    auto& p = *impl_;
    p.open();
    try {
        const auto& s = p.schedule.shape();
        if (p.accepted != s.tile_count || p.accepted_elements != s.logical_element_count
            || p.accepted_bytes != s.logical_bytes || completion.contract_version != 1
            || !completion.finite_image_reference || completion.ao_image_source_certified
            || completion.completed_q_count != s.n_kpoints || completion.completed_tile_count != p.accepted
            || completion.completed_element_count != p.accepted_elements
            || completion.completed_logical_bytes != p.accepted_bytes
            || completion.maximum_tile_bytes != p.maximum_tile_bytes
            || completion.schedule_identity_sha256 != p.schedule.schedule_identity_sha256()
            || completion.census_identity_sha256 != p.census() || completion.plan_identity_sha256 != p.plan())
            throw std::invalid_argument("private factor store requires matching complete producer receipt");
        if (p.stream.finish() != completion.payload_identity_sha256)
            throw std::invalid_argument("private factor store producer completion checksum mismatch");
        p.metadata_extent(false);
        auto observed_records = p.records_hash();
        for (std::uint64_t i = 0; i < s.tile_count; ++i) {
            Record record{};
            read_all(p.reader.value, record.data(), record.size(),
                     periodic_correlation_private_factor_store_record_offset(p.schedule, i));
            p.validate_record(record, i);
            observed_records.raw(record.data() + kRecordDigestOffset, 32);
        }
        const auto records_identity = p.records.finish();
        if (observed_records.finish() != records_identity)
            throw std::runtime_error("private factor store written record stream checksum mismatch");
        p.completion_digest = binary_digest(completion.payload_identity_sha256);
        std::memcpy(p.footer.data(), "VQPFCM01", 8);
        Encode e{p.footer.data(), 8};
        e.u32(1); e.u32(1); e.u64(p.extent); e.u64(p.accepted);
        e.u64(p.accepted_elements); e.u64(p.accepted_bytes);
        e.digest(hexadecimal(p.header_digest.data())); e.digest(records_identity);
        e.digest(completion.payload_identity_sha256);
        if (e.cursor != kFooterDigestOffset) throw std::logic_error("private factor store footer layout changed");
        auto storage = domain("vibeqc.periodic.correlation.private-factor-store.finalized");
        storage.raw(p.footer.data(), p.footer.size());
        p.storage_digest = binary_digest(storage.finish());
        std::copy(p.storage_digest.begin(), p.storage_digest.end(), p.footer.begin() + kFooterDigestOffset);
        write_all(p.writer.value, p.footer.data(), p.footer.size(), p.extent - kF);
        sync_file(p.writer.value);
        p.metadata_extent(true);
        p.writer.close_checked();
        p.state = PeriodicCorrelationPrivateFactorStoreState::Finalized;
        finalized_ = true;
        return PeriodicCorrelationPrivateFactorReader(std::move(impl_));
    } catch (...) { p.poison(); throw; }
}

PeriodicCorrelationPrivateFactorReader::PeriodicCorrelationPrivateFactorReader(
    std::unique_ptr<detail::PrivateFactorStoreImpl> p) noexcept : impl_(std::move(p)) {}
PeriodicCorrelationPrivateFactorReader::PeriodicCorrelationPrivateFactorReader(
    PeriodicCorrelationPrivateFactorReader&&) noexcept = default;
PeriodicCorrelationPrivateFactorReader& PeriodicCorrelationPrivateFactorReader::operator=(
    PeriodicCorrelationPrivateFactorReader&&) noexcept = default;
PeriodicCorrelationPrivateFactorReader::~PeriodicCorrelationPrivateFactorReader() = default;
std::uint64_t PeriodicCorrelationPrivateFactorReader::file_bytes() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return impl_->extent;
}
std::uint64_t PeriodicCorrelationPrivateFactorReader::tile_count() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return impl_->schedule.shape().tile_count;
}
std::string PeriodicCorrelationPrivateFactorReader::storage_identity_sha256() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return hexadecimal(impl_->storage_digest.data());
}
std::string PeriodicCorrelationPrivateFactorReader::producer_completion_identity_sha256() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return hexadecimal(impl_->completion_digest.data());
}
const std::shared_ptr<const PeriodicRestrictedMeanFieldState>&
PeriodicCorrelationPrivateFactorReader::state_handle() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return impl_->schedule.state_handle();
}
#define STORE_GETTER(name, expression) \
std::string PeriodicCorrelationPrivateFactorReader::name() const { \
    if (!impl_) throw std::logic_error("moved-from private factor reader"); \
    return expression; \
}
STORE_GETTER(state_identity_sha256, impl_->schedule.state_identity_sha256())
STORE_GETTER(calculation_identity, impl_->schedule.calculation_identity())
STORE_GETTER(allocation_identity, impl_->schedule.allocation_identity())
STORE_GETTER(schedule_identity_sha256, impl_->schedule.schedule_identity_sha256())
STORE_GETTER(census_identity_sha256, impl_->census())
STORE_GETTER(plan_identity_sha256, impl_->plan())
STORE_GETTER(ao_basis_identity_sha256, impl_->ao())
STORE_GETTER(auxiliary_basis_identity_sha256, impl_->auxiliary())
#undef STORE_GETTER
double PeriodicCorrelationPrivateFactorReader::image_cutoff_bohr() const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return impl_->image_cutoff;
}
PeriodicCorrelationFactorTileDescriptor PeriodicCorrelationPrivateFactorReader::descriptor(std::uint64_t sequence) const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    return impl_->schedule.descriptor(sequence);
}
void PeriodicCorrelationPrivateFactorReader::visit_tile(
    std::uint64_t sequence, std::uint64_t cap,
    void (*receiver)(const PeriodicCorrelationPrivateFactorTileView&, void*), void* context) const {
    if (!impl_) throw std::logic_error("moved-from private factor reader");
    auto& p = *impl_;
    const auto d = p.schedule.descriptor(sequence);
    const auto bytes = times(d.element_count, 16);
    if (receiver == nullptr || cap == 0 || bytes > cap || bytes > p.caps.maximum_tile_bytes)
        throw std::invalid_argument("private factor reader requires a receiver and sufficient explicit tile cap");
    if (p.visiting.exchange(true)) throw std::logic_error("private factor reader forbids recursive or concurrent visits");
    struct Guard { std::atomic<bool>& value; ~Guard() { value.store(false); } } guard{p.visiting};
    if (p.state != PeriodicCorrelationPrivateFactorStoreState::Finalized)
        throw std::logic_error("private factor reader is not finalized or has been poisoned");
    Record record{};
    std::vector<Complex> values;
    try {
        p.metadata_extent(true);
        const auto at = periodic_correlation_private_factor_store_record_offset(p.schedule, sequence);
        read_all(p.reader.value, record.data(), record.size(), at);
        p.validate_record(record, sequence);
        Record first{};
        const auto first_sequence = d.q_index * p.schedule.shape().tiles_per_q;
        read_all(p.reader.value, first.data(), first.size(),
                 periodic_correlation_private_factor_store_record_offset(p.schedule, first_sequence));
        p.validate_record(first, first_sequence);
        if (!std::equal(record.begin() + 100, record.begin() + 172, first.begin() + 100))
            throw std::runtime_error("private factor store source or whitener differs within q");
        auto payload = p.tile_hash_prefix(record, d);
        values.resize(static_cast<std::size_t>(d.element_count));
        auto position = plus(at, kR);
        for (std::size_t begin = 0; begin < values.size();) {
            const auto count = std::min<std::size_t>(values.size() - begin, kPage / 16);
            read_all(p.reader.value, p.page.data(), count * 16, position);
            for (std::size_t i = 0; i < count; ++i) {
                const double real = get_real(p.page.data() + 16 * i);
                const double imag = get_real(p.page.data() + 16 * i + 8);
                payload.real(real); payload.real(imag); values[begin + i] = {real, imag};
            }
            position = plus(position, count * 16); begin += count;
        }
        if (payload.finish() != hexadecimal(record.data() + 172))
            throw std::runtime_error("private factor store numerical payload checksum mismatch");
    } catch (...) { p.poison(); throw; }
    const PeriodicCorrelationPrivateFactorTileView view{
        d, values.data(), d.element_count, bytes, get64(record.data() + 84),
        get64(record.data() + 92), get64(record.data() + 100), hexadecimal(record.data() + 108),
        hexadecimal(record.data() + 140), hexadecimal(record.data() + 172)};
    receiver(view, context);
}

PeriodicCorrelationPrivateFactorWriter make_periodic_correlation_private_factor_writer(
    const PeriodicCorrelationAdmittedReference& reference,
    const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationFactorBuildCensus& census,
    double image_cutoff, double negative_tolerance, const std::string& directory,
    const PeriodicCorrelationPrivateFactorStoreCaps& caps) {
    const auto& s = schedule.shape();
    const auto& c = census.config();
    const auto extent = periodic_correlation_private_factor_store_file_bytes(schedule);
    static_cast<void>(offset(extent));
    if (caps.maximum_file_bytes == 0 || caps.maximum_tile_bytes == 0 || caps.maximum_tile_count == 0
        || extent > caps.maximum_file_bytes || s.maximum_tile_bytes > caps.maximum_tile_bytes
        || s.tile_count > caps.maximum_tile_count
        || s.maximum_tile_element_count > std::vector<Complex>().max_size()
        || plus(4096, times(s.tile_count, 32)) > std::numeric_limits<std::uint64_t>::max() / 8)
        throw std::length_error("private factor store exceeds explicit file/tile/metadata caps");
    if (!std::isfinite(image_cutoff) || image_cutoff < 0 || !std::isfinite(negative_tolerance)
        || negative_tolerance <= 0 || negative_tolerance > c.metric_absolute_eigenvalue_threshold)
        throw std::invalid_argument("private factor store requires valid finite-source cutoffs");
    const auto compensation = times(times(s.n_auxiliary, s.ao_pair_block), 16);
    if (c.producer_mode != PeriodicCorrelationFactorProducerMode::FullCoulombAllReciprocalReference
        || c.short_range_policy != PeriodicCorrelationShortRangePolicy::DisabledAllReciprocal
        || c.backing_mode != PeriodicCorrelationFactorBackingMode::Disk
        || c.publisher_mode != PeriodicCorrelationFactorPublisherMode::CanonicalSequentialExactlyOnce
        || c.publisher_buffer_count != 0 || c.transpose_before_publish
        || c.codec_identity_sha256 != periodic_correlation_private_factor_store_codec_identity_sha256()
        || !same_codec(c.codec, periodic_correlation_private_factor_store_codec_inventory())
        || c.backend.exact_extra_control_bytes < periodic_correlation_private_factor_store_fixed_control_bytes()
        || c.backend.exact_extra_retained_bytes
            < plus(compensation, periodic_correlation_private_factor_store_receiver_bytes(schedule)))
        throw std::invalid_argument("private factor store requires its compiled codec and retained-workspace admission");
    std::string plan_identity;
    std::uint64_t required_scratch = 0;
    {
        const auto plan = plan_periodic_correlation_factor_build(reference, schedule, census);
        if (plan.admission != PeriodicCorrelationFactorBuildAdmissionCode::Admitted)
            throw std::runtime_error("private factor store resource plan is not admitted");
        const auto& p = plan.components;
        if (p.encoded_generation_bytes != extent || p.disk_generation_bytes != extent
            || p.publisher_bitmap_bytes != 0 || p.publisher_digest_table_bytes != 0
            || p.publisher_buffer_bytes != 0 || p.journal_bytes != 0 || p.checkpoint_bytes != 0)
            throw std::logic_error("private factor store layout differs from admitted serialized inventory");
        plan_identity = plan.plan_identity_sha256;
        required_scratch = plan.required_scratch_bytes;
    }
    if (reference.state_handle()->periodic_dimension() != 3)
        throw std::invalid_argument("private factor store currently requires the three-dimensional finite source");
    auto p = std::make_unique<detail::PrivateFactorStoreImpl>(
        make_periodic_correlation_factor_stream_schedule(reference));
    if (p->schedule.schedule_identity_sha256() != schedule.schedule_identity_sha256())
        throw std::invalid_argument("private factor store schedule provenance mismatch");
    p->caps = caps; p->extent = extent; p->image_cutoff = image_cutoff; p->negative_tolerance = negative_tolerance;
    p->ao_digest = binary_digest(c.ao_basis_identity_sha256);
    p->auxiliary_digest = binary_digest(c.auxiliary_basis_identity_sha256);
    p->census_digest = binary_digest(census.census_identity_sha256());
    p->plan_digest = binary_digest(plan_identity);
    const auto& reciprocal = reference.state_handle()->reciprocal_lattice();
    p->direct_lattice = (2 * kPi) * reciprocal.inverse().transpose();
    if (!p->direct_lattice.allFinite()) throw std::invalid_argument("private factor store has invalid direct lattice");
    p->initialize_header();
    auto handles = private_file(directory, extent, required_scratch);
    p->writer = std::move(handles.first); p->reader = std::move(handles.second);
    write_all(p->writer.value, p->header.data(), p->header.size(), 0);
    return PeriodicCorrelationPrivateFactorWriter(std::move(p));
}

void private_factor_store_corrupt_diagnostic(
    PeriodicCorrelationPrivateFactorWriter& writer, std::uint64_t at, std::uint8_t mask) {
    if (!writer.impl_) throw std::logic_error("moved-from private factor writer");
    auto& p = *writer.impl_; p.open();
    if (p.extent > 1048576 || p.schedule.shape().tile_count > 128 || at >= p.extent)
        throw std::length_error("private factor corruption diagnostic exceeds tiny store cap");
    std::uint8_t byte;
    read_all(p.reader.value, &byte, 1, at); byte ^= mask;
    write_all(p.writer.value, &byte, 1, at);
}
void private_factor_store_truncate_diagnostic(
    PeriodicCorrelationPrivateFactorWriter& writer, std::uint64_t size) {
    if (!writer.impl_) throw std::logic_error("moved-from private factor writer");
    auto& p = *writer.impl_; p.open();
    if (p.extent > 1048576 || p.schedule.shape().tile_count > 128 || size >= p.extent)
        throw std::length_error("private factor truncation diagnostic exceeds tiny store cap");
#if defined(__linux__) || defined(__APPLE__)
    if (::ftruncate(p.writer.value, offset(size)) != 0) io_error("diagnostic truncation");
#else
    static_cast<void>(offset(size));
#endif
}
std::uint64_t private_factor_store_live_owned_descriptors_diagnostic() noexcept {
    return live_descriptors.load();
}
std::vector<std::uint8_t> private_factor_store_bytes_diagnostic(
    const PeriodicCorrelationPrivateFactorReader& reader, std::uint64_t at, std::uint64_t count) {
    if (!reader.impl_ || reader.impl_->state != PeriodicCorrelationPrivateFactorStoreState::Finalized)
        throw std::logic_error("private factor bytes diagnostic requires a finalized reader");
    const auto& p = *reader.impl_;
    if (p.extent > 1048576 || p.schedule.shape().tile_count > 128 || count > 8192
        || at > p.extent || count > p.extent - at)
        throw std::length_error("private factor bytes diagnostic exceeds tiny store cap");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(count));
    read_all(p.reader.value, bytes.data(), bytes.size(), at);
    return bytes;
}

}  // namespace vibeqc
