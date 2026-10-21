#pragma once

/**
 * \file rgsaddle/session.hpp
 * \brief C++ API for the rgsaddle hourglass.
 *
 * Sessions live in Rust. This header wraps the C ABI in include/rgsaddle.h.
 * It does not reimplement NEB, dimer, IRC, or Sella in C++.
 * Hosts that already dlopen dest librgsaddle (gpr_optim DestRgsaddleClient)
 * can switch to this header when they link the library directly.
 */

#include "../rgsaddle.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgsaddle {

inline const char* status_name(int status) noexcept {
    return rgsaddle_status_name(static_cast<rgsaddle_status_t>(status));
}

inline void check(int status) {
    if (status != RGSADDLE_OK) {
        throw std::runtime_error(status_name(status));
    }
}

inline int abi_version() noexcept { return rgsaddle_abi_version(); }

inline rgsaddle_version_t abi_stamp() {
    rgsaddle_version_t stamp{};
    check(rgsaddle_abi_stamp(&stamp));
    return stamp;
}

inline bool abi_compatible() noexcept {
    rgsaddle_version_t stamp{};
    if (rgsaddle_abi_stamp(&stamp) != RGSADDLE_OK) {
        return false;
    }
    return stamp.major == RGSADDLE_ABI_MAJOR && stamp.minor >= 15;
}

struct Report {
    int status = RGSADDLE_STATUS_RUNNING;
    double max_force = 0.0;
    std::int64_t ci_index = -1;
    std::int64_t iteration = 0;
    double curvature = 0.0;
    std::int64_t rotations = 0;

    static Report from_c(rgsaddle_report_t const& r) {
        Report out;
        out.status = r.status;
        out.max_force = r.max_force;
        out.ci_index = r.ci_index;
        out.iteration = r.iteration;
        out.curvature = r.curvature;
        out.rotations = r.rotations;
        return out;
    }

    [[nodiscard]] bool converged() const noexcept {
        return status == RGSADDLE_STATUS_CONVERGED;
    }
};

inline rgsaddle_band_config_t band_config() {
    rgsaddle_band_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.tangent = RGSADDLE_TANGENT_IMPROVED;
    c.spring = RGSADDLE_SPRING_UNIFORM;
    c.projection = RGSADDLE_PROJECTION_NEB;
    c.method = RGSADDLE_METHOD_LBFGS;
    c.spring_k = 5.0;
    c.force_tol = 1e-3;
    c.max_move = 0.1;
    return c;
}

inline rgsaddle_minmode_config_t minmode_config() {
    rgsaddle_minmode_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.kind = RGSADDLE_MINMODE_DIMER;
    c.method = RGSADDLE_METHOD_LBFGS;
    c.dr = 1e-3;
    c.rotation_tol = 1e-3;
    c.max_rotations = 20;
    c.force_tol = 1e-3;
    c.max_move = 0.1;
    return c;
}

inline rgsaddle_irc_config_t irc_config() {
    rgsaddle_irc_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.dx = 0.01;
    c.force_tol = 1e-3;
    c.force_gate = RGSADDLE_FORCE_LINF;
    c.max_move = 0.1;
    c.max_inner = 20;
    c.kind = RGSADDLE_IRC_GS2;
    c.direction = RGSADDLE_IRC_FORWARD;
    return c;
}

class Band {
public:
    Band(rgsaddle_band_config_t const& config, std::int64_t n_images,
         std::int64_t n_atoms, double const* positions)
        : n_images_(n_images), n_atoms_(n_atoms) {
        if (!abi_compatible()) {
            throw std::runtime_error("rgsaddle ABI 1.15 or later is required");
        }
        handle_ = rgsaddle_band_create(&config, n_images, n_atoms, positions);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_band_create failed");
        }
    }

    Band(Band const&) = delete;
    Band& operator=(Band const&) = delete;

    Band(Band&& other) noexcept
        : handle_(other.handle_), n_images_(other.n_images_),
          n_atoms_(other.n_atoms_) {
        other.handle_ = nullptr;
    }

    Band& operator=(Band&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_images_ = other.n_images_;
            n_atoms_ = other.n_atoms_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Band() { reset_handle(); }

    Report step(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_band_step(handle_, surface, user, &out));
        return Report::from_c(out);
    }

    void reset() { check(rgsaddle_band_reset(handle_)); }
    void restart() { check(rgsaddle_band_restart(handle_)); }
    void set_force_gate(rgsaddle_force_gate_t gate) {
        check(rgsaddle_band_set_force_gate(handle_, gate));
    }

    void set_highs(bool enabled) {
        check(rgsaddle_band_set_highs(handle_, enabled ? 1 : 0));
    }

    void set_positions(double const* positions) {
        check(rgsaddle_band_set_positions(handle_, positions));
    }

    void positions(double* out) const {
        check(rgsaddle_band_positions(handle_, out));
    }

    [[nodiscard]] std::vector<double> positions() const {
        std::vector<double> out(static_cast<std::size_t>(n_images_ * 3 * n_atoms_));
        positions(out.data());
        return out;
    }

    [[nodiscard]] std::int64_t n_images() const noexcept { return n_images_; }
    [[nodiscard]] std::int64_t n_atoms() const noexcept { return n_atoms_; }
    [[nodiscard]] RgsaddleBand* native() const noexcept { return handle_; }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_band_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleBand* handle_ = nullptr;
    std::int64_t n_images_ = 0;
    std::int64_t n_atoms_ = 0;
};

class MinMode {
public:
    MinMode(rgsaddle_minmode_config_t const& config, std::int64_t n_atoms,
            double const* position, double const* mode)
        : n_atoms_(n_atoms) {
        if (!abi_compatible()) {
            throw std::runtime_error("rgsaddle ABI 1.15 or later is required");
        }
        handle_ = rgsaddle_minmode_create(&config, n_atoms, position, mode);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_minmode_create failed");
        }
    }

    MinMode(MinMode const&) = delete;
    MinMode& operator=(MinMode const&) = delete;

    MinMode(MinMode&& other) noexcept
        : handle_(other.handle_), n_atoms_(other.n_atoms_) {
        other.handle_ = nullptr;
    }

    MinMode& operator=(MinMode&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_atoms_ = other.n_atoms_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~MinMode() { reset_handle(); }

    Report step(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_minmode_step(handle_, surface, user, &out));
        return Report::from_c(out);
    }

    void reset() { check(rgsaddle_minmode_reset(handle_)); }
    void set_force_gate(rgsaddle_force_gate_t gate) {
        check(rgsaddle_minmode_set_force_gate(handle_, gate));
    }

    void set_highs(bool enabled) {
        check(rgsaddle_minmode_set_highs(handle_, enabled ? 1 : 0));
    }
    void set_position(double const* position, double const* gradient = nullptr) {
        check(rgsaddle_minmode_set_position(handle_, position, gradient));
    }
    void set_mode(double const* mode) {
        check(rgsaddle_minmode_set_mode(handle_, mode));
    }
    Report estimate(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_minmode_estimate(handle_, surface, user, &out));
        return Report::from_c(out);
    }
    void set_kappa(rgsaddle_kappa_config_t const* config) {
        check(rgsaddle_minmode_set_kappa(handle_, config));
    }

    void position(double* out) const {
        check(rgsaddle_minmode_position(handle_, out));
    }

    void mode(double* out) const { check(rgsaddle_minmode_mode(handle_, out)); }

    [[nodiscard]] RgsaddleMinMode* native() const noexcept { return handle_; }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_minmode_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleMinMode* handle_ = nullptr;
    std::int64_t n_atoms_ = 0;
};

class Irc {
public:
    Irc(rgsaddle_irc_config_t const& config, std::int64_t n_atoms,
        double const* saddle, double const* masses, double const* mode)
        : n_atoms_(n_atoms) {
        if (!abi_compatible()) {
            throw std::runtime_error("rgsaddle ABI 1.15 or later is required");
        }
        handle_ = rgsaddle_irc_create(&config, n_atoms, saddle, masses, mode);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_irc_create failed");
        }
    }

    Irc(Irc const&) = delete;
    Irc& operator=(Irc const&) = delete;

    Irc(Irc&& other) noexcept : handle_(other.handle_), n_atoms_(other.n_atoms_) {
        other.handle_ = nullptr;
    }

    Irc& operator=(Irc&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_atoms_ = other.n_atoms_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Irc() { reset_handle(); }

    Report step(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_irc_step(handle_, surface, user, &out));
        return Report::from_c(out);
    }

    void reset() { check(rgsaddle_irc_reset(handle_)); }

    void set_direction(int32_t direction) {
        check(rgsaddle_irc_set_direction(handle_, direction));
    }

    void position(double* out) const {
        check(rgsaddle_irc_position(handle_, out));
    }

    [[nodiscard]] RgsaddleIrc* native() const noexcept { return handle_; }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_irc_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleIrc* handle_ = nullptr;
    std::int64_t n_atoms_ = 0;
};

inline rgsaddle_ocineb_config_t ocineb_config() {
    rgsaddle_ocineb_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.trigger_factor = 0.31;
    c.trigger_force = 0.0;
    c.angle_tol = 0.85;
    c.stability_count = 5;
    c.max_mmf_steps = 1000;
    c.restore_unhelpful = 0;
    return c;
}

struct OcinebReport {
    int status = RGSADDLE_STATUS_RUNNING;
    int phase = RGSADDLE_OCINEB_BAND;
    bool fell_back = false;
    int evaluations = 0;
    double max_force = 0.0;
    double curvature = 0.0;
    double alignment = 0.0;
    double threshold = 0.0;
    std::int64_t ci_index = -1;
    std::int64_t iteration = 0;
    std::int64_t rotations = 0;

    static OcinebReport from_c(rgsaddle_ocineb_report_t const& r) {
        OcinebReport out;
        out.status = r.status;
        out.phase = r.phase;
        out.fell_back = r.fell_back != 0;
        out.evaluations = r.evaluations;
        out.max_force = r.max_force;
        out.curvature = r.curvature;
        out.alignment = r.alignment;
        out.threshold = r.threshold;
        out.ci_index = r.ci_index;
        out.iteration = r.iteration;
        out.rotations = r.rotations;
        return out;
    }

    [[nodiscard]] bool converged() const noexcept {
        return status == RGSADDLE_STATUS_CONVERGED;
    }
};

class Ocineb {
public:
    Ocineb(rgsaddle_band_config_t const& band, rgsaddle_minmode_config_t const& minmode,
           rgsaddle_ocineb_config_t const& config, std::int64_t n_images, std::int64_t n_atoms,
           double const* positions)
        : n_images_(n_images), n_atoms_(n_atoms) {
        if (!abi_compatible()) {
            throw std::runtime_error("rgsaddle ABI 1.15 or later is required");
        }
        handle_ = rgsaddle_ocineb_create(&band, &minmode, &config, n_images, n_atoms, positions);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_ocineb_create failed");
        }
    }

    Ocineb(Ocineb const&) = delete;
    Ocineb& operator=(Ocineb const&) = delete;

    Ocineb(Ocineb&& other) noexcept
        : handle_(other.handle_), n_images_(other.n_images_), n_atoms_(other.n_atoms_) {
        other.handle_ = nullptr;
    }

    Ocineb& operator=(Ocineb&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_images_ = other.n_images_;
            n_atoms_ = other.n_atoms_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Ocineb() { reset_handle(); }

    OcinebReport step(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_ocineb_report_t out{};
        check(rgsaddle_ocineb_step(handle_, surface, user, &out));
        return OcinebReport::from_c(out);
    }

    void reset() { check(rgsaddle_ocineb_reset(handle_)); }

    [[nodiscard]] std::vector<double> positions() const {
        std::vector<double> out(static_cast<std::size_t>(n_images_ * 3 * n_atoms_));
        check(rgsaddle_ocineb_positions(handle_, out.data()));
        return out;
    }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_ocineb_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleOcineb* handle_ = nullptr;
    std::int64_t n_images_ = 0;
    std::int64_t n_atoms_ = 0;
};

}  // namespace rgsaddle
