#include "vibeqc/crystal.hpp"

#include <spglib.h>

#include <array>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vibeqc {

namespace {

// spglib takes ``double lattice[3][3]`` row-major, with ``lattice[i][j]`` the
// i-th Cartesian component of the j-th lattice vector. That happens to match
// Eigen::Matrix3d::operator()(i, j) with columns-as-basis-vectors, so the
// copy is an element-wise assignment regardless of Eigen's storage order.
void copy_lattice_to_c(const Eigen::Matrix3d& m, double out[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = m(i, j);
        }
    }
}

void copy_lattice_from_c(const double in[3][3], Eigen::Matrix3d& m) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m(i, j) = in[i][j];
        }
    }
}

// Pack atomic positions into a contiguous [N][3] buffer spglib can consume.
std::vector<std::array<double, 3>> pack_positions(const Eigen::Matrix3Xd& frac) {
    const int n = static_cast<int>(frac.cols());
    std::vector<std::array<double, 3>> pos(n);
    for (int a = 0; a < n; ++a) {
        pos[a] = {frac(0, a), frac(1, a), frac(2, a)};
    }
    return pos;
}

[[noreturn]] void throw_spg_error(const char* context) {
    const SpglibError code = spg_get_error_code();
    const char* msg = spg_get_error_message(code);
    std::ostringstream oss;
    oss << context << ": spglib error (" << static_cast<int>(code) << ")";
    if (msg && msg[0]) oss << " — " << msg;
    throw std::runtime_error(oss.str());
}

}  // namespace

SpaceGroup analyze(const Crystal& crystal, double symprec) {
    if (crystal.species.empty()) {
        throw std::runtime_error("analyze: empty crystal");
    }
    if (static_cast<int>(crystal.fractional_coords.cols()) !=
        crystal.n_atoms()) {
        throw std::runtime_error(
            "analyze: fractional_coords columns mismatch species size");
    }

    double lat[3][3];
    copy_lattice_to_c(crystal.lattice, lat);
    auto pos = pack_positions(crystal.fractional_coords);

    SpglibDataset* ds = spg_get_dataset(
        lat,
        reinterpret_cast<double(*)[3]>(pos.data()),
        crystal.species.data(),
        crystal.n_atoms(),
        symprec);
    if (!ds) throw_spg_error("analyze");

    SpaceGroup sg;
    sg.number = ds->spacegroup_number;
    sg.international_symbol = ds->international_symbol;
    sg.hall_number = ds->hall_number;
    sg.point_group = ds->pointgroup_symbol;

    sg.operations.resize(ds->n_operations);
    for (int k = 0; k < ds->n_operations; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                sg.operations[k].rotation(i, j) = ds->rotations[k][i][j];
            }
            sg.operations[k].translation(i) = ds->translations[k][i];
        }
    }

    sg.equivalent_atoms.assign(
        ds->equivalent_atoms, ds->equivalent_atoms + ds->n_atoms);

    spg_free_dataset(ds);
    return sg;
}

Crystal to_primitive(const Crystal& crystal, double symprec) {
    if (crystal.species.empty()) {
        throw std::runtime_error("to_primitive: empty crystal");
    }

    // spg_standardize_cell rewrites its buffers in place. Allocate worst-case:
    // the standardized cell can expand up to 4× the input (to conventional),
    // but we ask for to_primitive=1 which never expands — still, give 4× so
    // the same buffers would work if we later switch to conventional.
    const int n_in = crystal.n_atoms();
    const int n_buf = 4 * n_in;

    double lat[3][3];
    copy_lattice_to_c(crystal.lattice, lat);

    std::vector<std::array<double, 3>> pos(n_buf);
    auto pos_in = pack_positions(crystal.fractional_coords);
    for (int a = 0; a < n_in; ++a) pos[a] = pos_in[a];

    std::vector<int> types(n_buf, 0);
    for (int a = 0; a < n_in; ++a) types[a] = crystal.species[a];

    const int n_out = spg_standardize_cell(
        lat,
        reinterpret_cast<double(*)[3]>(pos.data()),
        types.data(),
        n_in,
        /*to_primitive=*/1,
        /*no_idealize=*/0,
        symprec);
    if (n_out <= 0) throw_spg_error("to_primitive");

    Crystal out;
    copy_lattice_from_c(lat, out.lattice);
    out.fractional_coords.resize(3, n_out);
    out.species.resize(n_out);
    for (int a = 0; a < n_out; ++a) {
        out.fractional_coords(0, a) = pos[a][0];
        out.fractional_coords(1, a) = pos[a][1];
        out.fractional_coords(2, a) = pos[a][2];
        out.species[a] = types[a];
    }
    return out;
}

IrreducibleKMesh irreducible_kpoints(const Crystal& crystal,
                                     const std::array<int, 3>& mesh,
                                     const std::array<int, 3>& is_shift,
                                     double symprec) {
    if (crystal.species.empty()) {
        throw std::runtime_error("irreducible_kpoints: empty crystal");
    }
    for (int i = 0; i < 3; ++i) {
        if (mesh[i] < 1) {
            throw std::runtime_error(
                "irreducible_kpoints: mesh dimensions must be ≥ 1");
        }
        // ``is_shift`` is a half-step flag, not a displacement. spglib takes
        // it at face value and the fractional coordinate below is
        // (g + is_shift/2) / mesh, so anything outside {0, 1} silently names
        // a different grid -- and for |is_shift| > 1 one that leaves the
        // first Brillouin zone entirely (is_shift = 7 on a 2-division axis
        // gives k_frac = 1.75). This is a public export (vq.irreducible_
        // kpoints) reachable without the Python layer's own screen in
        // kpoints.py, so the documented contract is enforced here (#691).
        if (is_shift[i] != 0 && is_shift[i] != 1) {
            throw std::runtime_error(
                "irreducible_kpoints: is_shift[" + std::to_string(i) +
                "] = " + std::to_string(is_shift[i]) +
                " must be 0 or 1; it is a half-step flag, not a "
                "displacement");
        }
    }

    double lat[3][3];
    copy_lattice_to_c(crystal.lattice, lat);
    auto pos = pack_positions(crystal.fractional_coords);

    const int n_total = mesh[0] * mesh[1] * mesh[2];
    std::vector<std::array<int, 3>> grid(n_total);
    std::vector<int> mapping(n_total);

    const int mesh_c[3] = {mesh[0], mesh[1], mesh[2]};
    const int shift_c[3] = {is_shift[0], is_shift[1], is_shift[2]};

    const int n_ir = spg_get_ir_reciprocal_mesh(
        reinterpret_cast<int(*)[3]>(grid.data()),
        mapping.data(),
        mesh_c, shift_c,
        /*is_time_reversal=*/1,
        lat,
        reinterpret_cast<double(*)[3]>(pos.data()),
        crystal.species.data(),
        crystal.n_atoms(),
        symprec);
    if (n_ir <= 0) throw_spg_error("irreducible_kpoints");

    // Collect unique grid indices and compute weights (number of full-mesh
    // points mapping to each representative, normalized to sum to 1).
    std::vector<int> rep_indices;
    rep_indices.reserve(n_ir);
    std::vector<int> rep_for_map(n_total, -1);
    std::vector<int> counts;
    for (int i = 0; i < n_total; ++i) {
        const int m = mapping[i];
        if (rep_for_map[m] < 0) {
            rep_for_map[m] = static_cast<int>(rep_indices.size());
            rep_indices.push_back(m);
            counts.push_back(0);
        }
        counts[rep_for_map[m]]++;
    }

    IrreducibleKMesh out;
    out.fractional_kpoints.reserve(rep_indices.size());
    out.weights.reserve(rep_indices.size());
    for (std::size_t i = 0; i < rep_indices.size(); ++i) {
        const auto& g = grid[rep_indices[i]];
        // spglib does NOT bake the shift into its grid addresses: every
        // ``g`` it returns is a plain integer in [0, mesh[d]), and the
        // caller adds is_shift/2 itself. Measured against spglib 2.7.0 on
        // a cubic cell with mesh (2,2,2): is_shift = (0,0,0) yields the
        // addresses (0,0,0), (1,0,0), (1,1,0), (1,1,1), while
        // is_shift = (1,1,1) yields the single address (0,0,0) -- offset
        // by nothing, not by half a step. So
        //
        //     k_frac[d] = (g[d] + is_shift[d]/2) / mesh[d]
        //
        // is the whole conversion, which is what this loop does.
        //
        // Equivalently, and exactly, that is the doubled modular address
        // a_d = 2 g_d + s_d over the modulus 2 mesh[d] -- the
        // representation vibeqc/kmesh_address.hpp is built on, where the
        // same grid is addressed in integers with no division at all.
        Eigen::Vector3d k_frac;
        for (int d = 0; d < 3; ++d) {
            k_frac[d] = (static_cast<double>(g[d]) +
                         0.5 * static_cast<double>(is_shift[d])) /
                        static_cast<double>(mesh[d]);
        }
        out.fractional_kpoints.push_back(k_frac);
        out.weights.push_back(static_cast<double>(counts[i]) /
                              static_cast<double>(n_total));
    }
    out.ir_mapping.resize(n_total);
    for (int i = 0; i < n_total; ++i) {
        out.ir_mapping[i] = rep_for_map[mapping[i]];
    }
    return out;
}

std::string spglib_version() {
    const char* v = spg_get_version();
    return v ? std::string(v) : std::string();
}

}  // namespace vibeqc
