/* C++ wrap over the rgsaddle C ABI: create a band, step, read
 * positions, reset, free. Built and run by tests/c_abi.rs. */
#include "rgsaddle/session.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int surface(void* user, rgsaddle_surface_request_t* req) {
    (void)user;
    if (req->version.major != RGSADDLE_ABI_MAJOR) {
        return -1;
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
    return 0;
}

int main() {
    if (!rgsaddle::abi_compatible()) {
        std::fprintf(stderr, "abi incompatible\n");
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
    auto out = band.positions();
    if (out.size() != static_cast<std::size_t>(n_images * 3)) {
        std::fprintf(stderr, "position size\n");
        return 1;
    }
    band.reset();
    std::printf("RGSADDLE_CXX_WRAP_OK force=%.6g iter=%lld\n", last.max_force,
                static_cast<long long>(last.iteration));
    return 0;
}
