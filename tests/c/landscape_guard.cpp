/* The landscape guard's minor is the minor of the landscape symbols. */
#include "rgsaddle/session.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(std::is_same<rgsaddle_hessian_fn, int (*)(void*, std::int64_t, const double*, double*)>::value,
              "rgsaddle_hessian_fn keeps the original signature");
static_assert(std::is_same<rgsaddle_hessian_at_fn,
                           int (*)(void*, std::int64_t, const double*, const double*, double*)>::value,
              "rgsaddle_hessian_at_fn signature");

static_assert(sizeof(rgsaddle_landscape_config_t) == 144, "landscape config size");
static_assert(alignof(rgsaddle_landscape_config_t) == 8, "landscape config alignment");
static_assert(offsetof(rgsaddle_landscape_config_t, version) == 0, "version offset");
static_assert(offsetof(rgsaddle_landscape_config_t, flags) == 8, "flags offset");
static_assert(offsetof(rgsaddle_landscape_config_t, duplicate_distance) == 16, "duplicate_distance offset");
static_assert(offsetof(rgsaddle_landscape_config_t, push) == 24, "push offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_nodes) == 32, "max_nodes offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_evaluations) == 40, "max_evaluations offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_steps) == 48, "max_steps offset");
static_assert(offsetof(rgsaddle_landscape_config_t, force_tol) == 56, "force_tol offset");
static_assert(offsetof(rgsaddle_landscape_config_t, relax_force_tol) == 64, "relax_force_tol offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_move) == 72, "max_move offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_atom_move) == 80, "max_atom_move offset");
static_assert(offsetof(rgsaddle_landscape_config_t, initial_step) == 88, "initial_step offset");
static_assert(offsetof(rgsaddle_landscape_config_t, dr) == 96, "dr offset");
static_assert(offsetof(rgsaddle_landscape_config_t, rotation_tol) == 104, "rotation_tol offset");
static_assert(offsetof(rgsaddle_landscape_config_t, residual_tol) == 112, "residual_tol offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_rotations) == 120, "max_rotations offset");
static_assert(offsetof(rgsaddle_landscape_config_t, max_block_iterations) == 128, "max_block_iterations offset");
static_assert(offsetof(rgsaddle_landscape_config_t, method) == 132, "method offset");
static_assert(offsetof(rgsaddle_landscape_config_t, cell) == 136, "cell offset");

static_assert(sizeof(rgsaddle_landscape_report_t) == 48, "landscape report size");
static_assert(alignof(rgsaddle_landscape_report_t) == 8, "landscape report alignment");
static_assert(offsetof(rgsaddle_landscape_report_t, version) == 0, "report version offset");
static_assert(offsetof(rgsaddle_landscape_report_t, flags) == 8, "report flags offset");
static_assert(offsetof(rgsaddle_landscape_report_t, n_nodes) == 16, "report n_nodes offset");
static_assert(offsetof(rgsaddle_landscape_report_t, n_edges) == 24, "report n_edges offset");
static_assert(offsetof(rgsaddle_landscape_report_t, bound) == 32, "report bound offset");
static_assert(offsetof(rgsaddle_landscape_report_t, evaluations) == 40, "report evaluations offset");

/* Introduction minor of rgsaddle_landscape_create, _config_default, _run,
 * _node, _modes, _edge, and _free. */
static constexpr std::uint32_t landscape_symbol_minor = 68;

int main(int argc, char** argv) {
    const double origin[3] = {0.0, 0.0, 0.0};
    if (argc > 1 && std::strcmp(argv[1], "--expect-refused") == 0) {
        if (rgsaddle::abi_has_landscape()) {
            std::fprintf(stderr, "abi_has_landscape accepted an older minor\n");
            return 10;
        }
        try {
            rgsaddle::Landscape session(rgsaddle::landscape_config(), 1, origin, nullptr, 0);
            std::fprintf(stderr, "Landscape constructor accepted an older minor\n");
            return 11;
        } catch (std::runtime_error const& err) {
            if (std::string(err.what()).find("1.68") == std::string::npos) {
                std::fprintf(stderr, "unexpected guard error: %s\n", err.what());
                return 12;
            }
            std::printf("RGSADDLE_LANDSCAPE_GUARD_REFUSED_OK\n");
            return 0;
        }
    }

    const auto create = &rgsaddle_landscape_create;
    const auto config_default = &rgsaddle_landscape_config_default;
    const auto run = &rgsaddle_landscape_run;
    const auto node = &rgsaddle_landscape_node;
    const auto modes = &rgsaddle_landscape_modes;
    const auto edge = &rgsaddle_landscape_edge;
    const auto release = &rgsaddle_landscape_free;
    const std::uintptr_t linked = reinterpret_cast<std::uintptr_t>(create) +
                                  reinterpret_cast<std::uintptr_t>(config_default) +
                                  reinterpret_cast<std::uintptr_t>(run) +
                                  reinterpret_cast<std::uintptr_t>(node) +
                                  reinterpret_cast<std::uintptr_t>(modes) +
                                  reinterpret_cast<std::uintptr_t>(edge) +
                                  reinterpret_cast<std::uintptr_t>(release);
    if (linked == 0) {
        std::fprintf(stderr, "a landscape symbol is missing\n");
        return 1;
    }
    if (rgsaddle::landscape_abi_minor != landscape_symbol_minor) {
        std::fprintf(stderr, "guard minor %u, landscape symbols %u\n", rgsaddle::landscape_abi_minor,
                     landscape_symbol_minor);
        return 2;
    }
    if (!rgsaddle::abi_has_landscape()) {
        std::fprintf(stderr, "the loaded library is below the landscape minor\n");
        return 3;
    }
    const rgsaddle_version_t below{RGSADDLE_ABI_MAJOR, landscape_symbol_minor - 1};
    const rgsaddle_version_t older{RGSADDLE_ABI_MAJOR, 15};
    const rgsaddle_version_t wrong_major{RGSADDLE_ABI_MAJOR + 1, landscape_symbol_minor};
    const rgsaddle_version_t at{RGSADDLE_ABI_MAJOR, landscape_symbol_minor};
    const rgsaddle_version_t newer{RGSADDLE_ABI_MAJOR, landscape_symbol_minor + 1};
    if (rgsaddle::abi_has_landscape_for(below) || rgsaddle::abi_has_landscape_for(older) ||
        rgsaddle::abi_has_landscape_for(wrong_major) || !rgsaddle::abi_has_landscape_for(at) ||
        !rgsaddle::abi_has_landscape_for(newer)) {
        std::fprintf(stderr, "the guard predicate disagrees with the symbol minor\n");
        return 4;
    }

    try {
        rgsaddle::Landscape session(rgsaddle::landscape_config(), 1, origin, nullptr, 0);
        if (session.n_atoms() != 1 || session.native() == nullptr) {
            std::fprintf(stderr, "landscape session was not opened\n");
            return 5;
        }
    } catch (std::runtime_error const& err) {
        std::fprintf(stderr, "landscape constructor: %s\n", err.what());
        return 6;
    }
    std::printf("RGSADDLE_LANDSCAPE_GUARD_OK\n");
    return 0;
}
