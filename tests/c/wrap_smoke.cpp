/* C++ wrap over the rgsaddle C ABI: create a band, fix an atom, step,
 * read positions, reset, free. Built and run by tests/cxx_abi.rs. */
#include "rgsaddle/session.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static rgsaddle_status_t surface(void* user, rgsaddle_surface_request_t* req) {
    (void)user;
    if (req->version.major != RGSADDLE_ABI_MAJOR) {
        return RGSADDLE_NULL_SESSION;
    }
    const long dof = 3 * static_cast<long>(req->n_atoms);
    for (long i = 0; i < static_cast<long>(req->n_images); ++i) {
        const double* p = req->positions + i * dof;
        double* g = req->gradients + i * dof;
        const double x = p[0], y = p[1], z = p[2];
        req->energies[i] =
            (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
        g[0] = 4.0 * x * (x * x - 1.0);
        g[1] = 4.0 * y;
        g[2] = 4.0 * z;
    }
    return RGSADDLE_OK;
}

/* Atom 0 in the double well and atom 1 on a spring to (0.5, 0.5, 0.5),
 * placed off it, so a free atom 1 moves. */
static rgsaddle_status_t pair_surface(void* user, rgsaddle_surface_request_t* req) {
    (void)user;
    if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != 2) {
        return RGSADDLE_NULL_SESSION;
    }
    for (long i = 0; i < static_cast<long>(req->n_images); ++i) {
        const double* p = req->positions + i * 6;
        double* g = req->gradients + i * 6;
        const double x = p[0], y = p[1], z = p[2];
        double e = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
        g[0] = 4.0 * x * (x * x - 1.0);
        g[1] = 4.0 * y;
        g[2] = 4.0 * z;
        for (int c = 0; c < 3; ++c) {
            const double d = p[3 + c] - 0.5;
            e += 0.5 * d * d;
            g[3 + c] = d;
        }
        req->energies[i] = e;
    }
    return RGSADDLE_OK;
}

int main() {
    if (!rgsaddle::abi_compatible()) {
        std::fprintf(stderr, "abi incompatible\n");
        return 1;
    }

    static_assert(rgsaddle::growing_string_abi_minor == 44,
                  "growing string C API arrived at ABI minor 44");
    const rgsaddle_version_t before_growing_string = {RGSADDLE_ABI_MAJOR, 43};
    const rgsaddle_version_t with_growing_string = {RGSADDLE_ABI_MAJOR, 44};
    const rgsaddle_version_t other_major = {RGSADDLE_ABI_MAJOR + 1, 44};
    if (rgsaddle::abi_stamp_at_least(before_growing_string, rgsaddle::growing_string_abi_minor) ||
        !rgsaddle::abi_stamp_at_least(with_growing_string, rgsaddle::growing_string_abi_minor) ||
        rgsaddle::abi_stamp_at_least(other_major, rgsaddle::growing_string_abi_minor) ||
        !rgsaddle::abi_at_least(rgsaddle::growing_string_abi_minor) ||
        rgsaddle::abi_at_least(RGSADDLE_ABI_MINOR + 1)) {
        std::fprintf(stderr, "growing string ABI minor gate\n");
        return 1;
    }

    if (rgsaddle_band_set_highs(nullptr, 0) != RGSADDLE_NULL_BAND ||
        rgsaddle_minmode_set_highs(nullptr, 0) != RGSADDLE_NULL_SESSION) {
        std::fprintf(stderr, "null feasible-step control\n");
        return 1;
    }
    if (rgsaddle_minmode_set_torque_window(nullptr, 0) != RGSADDLE_NULL_SESSION) {
        std::fprintf(stderr, "null torque window control\n");
        return 1;
    }
    if (rgsaddle_minmode_set_rotation_stop_length(nullptr, 0.0) != RGSADDLE_NULL_SESSION) {
        std::fprintf(stderr, "null rotation stop control\n");
        return 1;
    }
    const std::uint8_t free_mask[1] = {0};
    if (rgsaddle_band_set_fixed_atoms(nullptr, free_mask, 1) != RGSADDLE_NULL_BAND) {
        std::fprintf(stderr, "null fixed-atom mask control\n");
        return 1;
    }

    const long n_images = 9, n_atoms = 1;
    std::vector<double> pos(static_cast<std::size_t>(n_images * 3));
    for (long i = 0; i < n_images; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n_images - 1);
        pos[static_cast<std::size_t>(i * 3 + 0)] = -1.0 + 2.0 * t;
        pos[static_cast<std::size_t>(i * 3 + 1)] =
            0.3 * std::sin(3.14159265358979 * t);
        pos[static_cast<std::size_t>(i * 3 + 2)] = 0.0;
    }

    auto cfg = rgsaddle::band_config();
    cfg.method = RGSADDLE_METHOD_FIRE;
    cfg.ci_trigger_factor = 0.5;

    rgsaddle::Band band(cfg, n_images, n_atoms, pos.data());
    band.set_highs(true);
    band.set_highs(false);
    rgsaddle::Report last;
    for (int i = 0; i < 200; ++i) {
        last = band.step(surface, nullptr);
        if (last.converged()) {
            break;
        }
    }
    if (!last.converged()) {
        std::fprintf(stderr, "band did not converge\n");
        return 1;
    }
    if (!(last.curvature < 0.0)) {
        std::fprintf(stderr, "refined band curvature %g\n", last.curvature);
        return 1;
    }
    auto out = band.positions();
    if (out.size() != static_cast<std::size_t>(n_images * 3)) {
        std::fprintf(stderr, "position size\n");
        return 1;
    }
    auto certificate = band.saddle_index();
    if (!certificate.certified || certificate.index != 1 || certificate.curvatures.size() != 3 ||
        std::fabs(certificate.curvatures[0] + 4.0) > 0.05 || certificate.evaluations <= 0 ||
        certificate.total_evaluations != certificate.evaluations) {
        std::fprintf(stderr, "band saddle index %d, %zu curvatures\n", certificate.index,
                     certificate.curvatures.size());
        return 1;
    }
    auto point = rgsaddle::saddle_index(n_atoms, &out[static_cast<std::size_t>(last.ci_index * 3)],
                                        surface, nullptr);
    if (point.index != 1 || point.curvatures.size() != 3) {
        std::fprintf(stderr, "point saddle index %d\n", point.index);
        return 1;
    }
    auto escape = rgsaddle::saddle_check();
    escape.escape_attempts = 1;
    band.set_saddle_check(&escape);
    band.set_saddle_check(nullptr);
    if (band.saddle_index(0).certified) {
        std::fprintf(stderr, "a cleared check kept its certificate\n");
        return 1;
    }
    band.reset();

    cfg.flags |= RGSADDLE_BAND_NO_MIN_MODE;
    rgsaddle::Band plain(cfg, n_images, n_atoms, pos.data());
    for (int i = 0; i < 400; ++i) {
        auto report = plain.step(surface, nullptr);
        if (report.curvature != 0.0) {
            std::fprintf(stderr, "plain band reported a curvature\n");
            return 1;
        }
        if (report.converged()) {
            break;
        }
    }

    rgsaddle::Band str_band(cfg, n_images, n_atoms, pos.data());
    str_band.set_string(true, RGSADDLE_REPARAM_NOT_A_KNOT);
    rgsaddle::Report str_rep;
    for (int i = 0; i < 500; ++i) {
        str_rep = str_band.step(surface, nullptr);
        if (str_rep.converged()) {
            break;
        }
    }
    auto str_pos = str_band.positions();
    if (!str_rep.converged() || str_rep.ci_index < 1 ||
        std::fabs(str_pos[static_cast<std::size_t>(str_rep.ci_index * 3)]) > 0.01) {
        std::fprintf(stderr, "C++ string band did not converge to the saddle\n");
        return 1;
    }

    auto gcfg = rgsaddle::growing_string_config();
    gcfg.n_images = 7;
    gcfg.frontier_tol = 0.5;
    gcfg.band.force_tol = 0.001;
    gcfg.band.max_move = 0.1;
    const double greact[3] = {-1.0, 0.2, 0.0};
    const double gprod[3] = {1.0, 0.2, 0.0};
    rgsaddle::GrowingString gstr(gcfg, n_atoms, greact, gprod);
    if (gstr.joined() || gstr.n_images() != 4) {
        std::fprintf(stderr, "C++ growing string initial state invalid\n");
        return 1;
    }
    rgsaddle::Report grep;
    std::int64_t grown = 0;
    for (int i = 0; i < 500; ++i) {
        grep = gstr.step(surface, nullptr);
        if (grep.rotations != 0) {
            std::fprintf(stderr, "C++ growing string step reported rotations\n");
            return 1;
        }
        grown += gstr.nodes_added();
        if (grep.converged()) {
            break;
        }
    }
    if (grown != 3) {
        std::fprintf(stderr, "C++ growing string grew %lld nodes, expected 3\n",
                     static_cast<long long>(grown));
        return 1;
    }
    auto gpos = gstr.positions();
    auto gids = gstr.ids();
    if (!grep.converged() || !gstr.joined() || gstr.n_images() != 7 ||
        gids.size() != 7 || gids[0] != 0 || gids[6] != 1 || grep.ci_index < 1 ||
        std::fabs(gpos[static_cast<std::size_t>(grep.ci_index * 3)]) > 0.02) {
        std::fprintf(stderr, "C++ growing string did not converge to the saddle\n");
        return 1;
    }

    std::vector<double> pair(static_cast<std::size_t>(n_images * 6));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_images); ++i) {
        for (std::size_t c = 0; c < 3; ++c) {
            pair[i * 6 + c] = pos[i * 3 + c];
            pair[i * 6 + 3 + c] = 0.2;
        }
    }
    auto held_cfg = rgsaddle::band_config();
    held_cfg.method = RGSADDLE_METHOD_FIRE;
    held_cfg.ci_trigger_factor = 0.5;
    rgsaddle::Band held(held_cfg, n_images, 2, pair.data());
    auto refused = [&held](std::vector<std::uint8_t> const& mask) {
        try {
            held.set_fixed_atoms(mask);
        } catch (std::runtime_error const&) {
            return true;
        }
        return false;
    };
    if (!refused({0}) || !refused({1, 1}) || !refused({0, 2})) {
        std::fprintf(stderr, "a malformed fixed-atom mask was accepted\n");
        return 1;
    }
    const std::vector<std::uint8_t> mask{0, 1};
    held.set_fixed_atoms(mask.data());
    rgsaddle::Report held_last;
    for (int i = 0; i < 200; ++i) {
        held_last = held.step(pair_surface, nullptr);
        if (held_last.converged()) {
            break;
        }
    }
    if (!held_last.converged() || !(held_last.curvature < 0.0)) {
        std::fprintf(stderr, "band with a fixed atom: converged %d, curvature %g\n",
                     held_last.converged() ? 1 : 0, held_last.curvature);
        return 1;
    }
    auto settled = held.positions();
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_images); ++i) {
        if (std::memcmp(&settled[i * 6 + 3], &pair[i * 6 + 3], 3 * sizeof(double)) != 0) {
            std::fprintf(stderr, "the fixed atom moved in image %zu\n", i);
            return 1;
        }
    }
    held.set_fixed_atoms(std::vector<std::uint8_t>{0, 0});
    held.step(pair_surface, nullptr);
    auto released = held.positions();
    if (std::memcmp(&released[6 + 3], &pair[6 + 3], 3 * sizeof(double)) == 0) {
        std::fprintf(stderr, "a cleared mask kept atom 1 fixed\n");
        return 1;
    }

    auto ic = rgsaddle::indexk_config();
    ic.force_tol = 1e-8;
    ic.max_move = 0.1;
    const double origin[3] = {0.4, 0.2, -0.1};
    const double guess[3] = {1.0, 0.2, 0.0};
    rgsaddle::IndexK search(ic, 1, origin, guess);
    auto stepped = search.step(surface, nullptr, nullptr);
    auto moved = search.position();
    if (moved.size() != 3 || stepped.max_force < 0.0) {
        std::fprintf(stderr, "index-k wrapper\n");
        return 1;
    }
    ic.index = 0;
    bool rejected = false;
    try {
        rgsaddle::IndexK bad(ic, 1, origin, guess);
        (void)bad;
    } catch (std::runtime_error const&) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "index 0 was accepted\n");
        return 1;
    }

    std::printf("RGSADDLE_CXX_WRAP_OK force=%.6g iter=%lld\n", last.max_force,
                static_cast<long long>(last.iteration));
    return 0;
}
