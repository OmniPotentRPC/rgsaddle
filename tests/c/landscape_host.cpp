/* A limit is an error, and the Hessian callback receives the geometry. */
#include "rgsaddle/session.hpp"

#include <cmath>
#include <cstdio>
#include <string>

struct Host {
    int hessian_calls;
    int saw_start;
};

static rgsaddle_status_t saddle_surface(void *user, rgsaddle_surface_request_t *req) {
    auto *host = static_cast<Host *>(user);
    if (req == nullptr || req->n_images != 1 || req->positions == nullptr || req->energies == nullptr ||
        req->gradients == nullptr) {
        return RGSADDLE_INVALID_PARAMETER;
    }
    const double *x = req->positions;
    (void)host;
    req->energies[0] = -0.5 * x[0] * x[0] + 0.5 * x[1] * x[1] + 0.5 * x[2] * x[2];
    req->gradients[0] = -x[0];
    req->gradients[1] = x[1];
    req->gradients[2] = x[2];
    return RGSADDLE_OK;
}

static int saddle_hessian(void *user, int64_t n, const double *position, const double *vector,
                          double *action) {
    auto *host = static_cast<Host *>(user);
    if (host == nullptr || position == nullptr || vector == nullptr || action == nullptr || n != 3) {
        return -1;
    }
    host->hessian_calls += 1;
    const double start[3] = {0.2, -0.4, 0.6};
    int at_start = 1;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(position[i] - start[i]) > 0.0) {
            at_start = 0;
        }
    }
    if (at_start) {
        host->saw_start = 1;
    }
    action[0] = -vector[0];
    action[1] = vector[1];
    action[2] = vector[2];
    return 0;
}

static int vector_hessian(void *user, int64_t n, const double *vector, double *action) {
    auto *host = static_cast<Host *>(user);
    if (host == nullptr || vector == nullptr || action == nullptr || n != 3) {
        return -1;
    }
    host->hessian_calls += 1;
    action[0] = -vector[0];
    action[1] = vector[1];
    action[2] = vector[2];
    return 0;
}

static int original_hessian_keeps_its_signature() {
    Host host{};
    auto config = rgsaddle::indexk_config();
    config.index = 1;
    config.force_tol = 1.0e-6;
    const double start[3] = {0.2, -0.4, 0.6};
    rgsaddle::IndexK session(config, 1, start, nullptr);
    session.step(saddle_surface, vector_hessian, &host);
    if (host.hessian_calls < 1) {
        std::fprintf(stderr, "the vector-only hessian was not called\n");
        return 5;
    }
    return 0;
}

static int hessian_sees_the_geometry() {
    Host host{};
    auto config = rgsaddle::indexk_config();
    config.index = 1;
    config.force_tol = 1.0e-6;
    const double start[3] = {0.2, -0.4, 0.6};
    rgsaddle::IndexK session(config, 1, start, nullptr);
    session.step_at(saddle_surface, saddle_hessian, &host);
    if (host.hessian_calls < 1) {
        std::fprintf(stderr, "hessian was not called\n");
        return 1;
    }
    if (host.saw_start != 1) {
        std::fprintf(stderr, "hessian was not evaluated at the geometry\n");
        return 2;
    }
    return 0;
}

static int bound_is_an_error() {
    auto config = rgsaddle::landscape_config();
    config.max_evaluations = 1;
    config.force_tol = 1.0e-3;
    const double origin[3] = {0.0, 0.0, 0.0};
    Host host{};
    rgsaddle::Landscape session(config, 1, origin, nullptr, 0);
    try {
        session.run(saddle_surface, saddle_hessian, &host);
    } catch (std::runtime_error const &err) {
        if (std::string(err.what()).find("LANDSCAPE_BOUND") == std::string::npos) {
            std::fprintf(stderr, "limit raised the wrong error: %s\n", err.what());
            return 3;
        }
        return 0;
    }
    std::fprintf(stderr, "a landscape limit returned a report\n");
    return 4;
}

int main() {
    if (const int rc = hessian_sees_the_geometry()) {
        return rc;
    }
    if (const int rc = original_hessian_keeps_its_signature()) {
        return rc;
    }
    if (const int rc = bound_is_an_error()) {
        return rc;
    }
    std::printf("RGSADDLE_LANDSCAPE_HOST_OK\n");
    return 0;
}
