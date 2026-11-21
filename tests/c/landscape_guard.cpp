/* The landscape guard's minor is the minor of the landscape symbols. */
#include "rgsaddle/session.hpp"

#include <cstdint>
#include <cstdio>

/* Introduction minor of rgsaddle_landscape_create, _config_default, _run,
 * _node, _modes, _edge, and _free. */
static constexpr std::uint32_t landscape_symbol_minor = 49;

int main() {
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
    rgsaddle_version_t below{};
    below.major = RGSADDLE_ABI_MAJOR;
    below.minor = landscape_symbol_minor - 1;
    rgsaddle_version_t at{};
    at.major = RGSADDLE_ABI_MAJOR;
    at.minor = landscape_symbol_minor;
    const auto accepts = [](rgsaddle_version_t const& stamp) {
        return stamp.major == RGSADDLE_ABI_MAJOR && stamp.minor >= rgsaddle::landscape_abi_minor;
    };
    if (accepts(below) || !accepts(at)) {
        std::fprintf(stderr, "the guard predicate disagrees with the symbol minor\n");
        return 4;
    }

    const double origin[3] = {0.0, 0.0, 0.0};
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
