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
#include <utility>
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

inline bool abi_stamp_at_least(rgsaddle_version_t const& stamp, std::uint32_t minor) noexcept {
    return stamp.major == RGSADDLE_ABI_MAJOR && stamp.minor >= minor;
}

inline bool abi_at_least(std::uint32_t minor) noexcept {
    rgsaddle_version_t stamp{};
    if (rgsaddle_abi_stamp(&stamp) != RGSADDLE_OK) {
        return false;
    }
    return abi_stamp_at_least(stamp, minor);
}

inline bool abi_compatible() noexcept { return abi_at_least(15); }

// First ABI minor that carries the rgsaddle_growing_string_* symbols.
inline constexpr std::uint32_t growing_string_abi_minor = 44;

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

// The certificate of a converged saddle, ABI minor 40. Without one,
// certified is false and the counts are the session's.
struct SaddleIndex {
    bool certified = false;
    int index = 0;
    bool converged = false;
    std::int64_t evaluations = 0;
    std::int64_t total_evaluations = 0;
    std::int64_t escapes = 0;
    std::vector<double> curvatures;

    static SaddleIndex from_c(rgsaddle_saddle_index_t const& r, std::vector<double> curvatures) {
        SaddleIndex out;
        out.certified = r.certified != 0;
        out.index = r.index;
        out.converged = r.converged != 0;
        out.evaluations = r.evaluations;
        out.total_evaluations = r.total_evaluations;
        out.escapes = r.escapes;
        const auto resolved = static_cast<std::size_t>(r.n_curvatures);
        if (curvatures.size() > resolved) {
            curvatures.resize(resolved);
        }
        out.curvatures = std::move(curvatures);
        return out;
    }
};

inline rgsaddle_saddle_check_t saddle_check() {
    rgsaddle_saddle_check_t c{};
    check(rgsaddle_saddle_check_default(&c));
    return c;
}

// Certify one point; a null config takes saddle_check().
inline SaddleIndex saddle_index(std::int64_t n_atoms, double const* position,
                                rgsaddle_surface_fn surface, void* user,
                                rgsaddle_saddle_check_t const* config = nullptr,
                                double const* gradient = nullptr, double const* mode = nullptr,
                                std::uint8_t const* fixed = nullptr, std::size_t capacity = 8) {
    rgsaddle_saddle_index_t out{};
    std::vector<double> curvatures(capacity);
    check(rgsaddle_saddle_index(config, n_atoms, position, gradient, mode, fixed, surface, user,
                                &out, curvatures.data(), static_cast<std::int64_t>(capacity)));
    return SaddleIndex::from_c(out, std::move(curvatures));
}

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
    // Looser than the Rust stock 1e-4, so the torque window is on: after
    // one dimer rotation, or two Lanczos vectors, a rotational force
    // under 1 is finished.
    c.rotation_tol = 1e-3;
    c.max_rotations = 20;
    c.force_tol = 1e-3;
    c.max_move = 0.1;
    // Left at 0 by the zero-initialized struct. The angle test is off.
    // The Rust MinModeConfig default is five degrees; a C config stores
    // the value written here.
    return c;
}

inline rgsaddle_indexk_config_t indexk_config() {
    rgsaddle_indexk_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.index = 1;
    c.method = RGSADDLE_MINMODE_DIMER;
    c.dr = 1e-3;
    c.rotation_tol = 1e-4;
    c.max_rotations = 20;
    c.force_tol = 1e-3;
    c.max_move = 0.2;
    c.residual_tol = 1e-4;
    c.max_block_iterations = 30;
    return c;
}

inline rgsaddle_irc_config_t irc_config() {
    rgsaddle_irc_config_t c{};
    c.version.major = RGSADDLE_ABI_MAJOR;
    c.version.minor = RGSADDLE_ABI_MINOR;
    c.dx = 0.01;
    c.force_tol = 1e-3;
    c.force_gate = RGSADDLE_FORCE_LINF;
    c.max_inner = 20;
    c.kind = RGSADDLE_IRC_SELLA_SPHERE;
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

    // One byte per atom of the band, 1 fixed and 0 free, shared by every
    // image; all zeros clears the mask. ABI minor 39.
    void set_fixed_atoms(std::uint8_t const* mask) {
        check(rgsaddle_band_set_fixed_atoms(handle_, mask, n_atoms_));
    }

    void set_fixed_atoms(std::vector<std::uint8_t> const& mask) {
        check(rgsaddle_band_set_fixed_atoms(handle_, mask.data(),
                                            static_cast<std::int64_t>(mask.size())));
    }

    // nullptr turns the check off. ABI minor 40.
    void set_saddle_check(rgsaddle_saddle_check_t const* config) {
        check(rgsaddle_band_set_saddle_check(handle_, config));
    }

    // Simplified string method, ABI minor 44.
    void set_string(bool enabled, std::int32_t scheme = RGSADDLE_REPARAM_NOT_A_KNOT) {
        check(rgsaddle_band_set_string(handle_, enabled ? 1 : 0, scheme));
    }

    [[nodiscard]] SaddleIndex saddle_index(std::size_t capacity = 8) const {
        rgsaddle_saddle_index_t out{};
        std::vector<double> curvatures(capacity);
        check(rgsaddle_band_saddle_index(handle_, &out, curvatures.data(),
                                         static_cast<std::int64_t>(capacity)));
        return SaddleIndex::from_c(out, std::move(curvatures));
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
    // -1 infers the window from rotation_tol, 0 turns it off, 1 on.
    void set_torque_window(int window) {
        check(rgsaddle_minmode_set_torque_window(handle_, window));
    }
    // 0 turns the curvature-aware rotation stop off.
    void set_rotation_stop_length(double length) {
        check(rgsaddle_minmode_set_rotation_stop_length(handle_, length));
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

    // nullptr turns the check off. ABI minor 40.
    void set_saddle_check(rgsaddle_saddle_check_t const* config) {
        check(rgsaddle_minmode_set_saddle_check(handle_, config));
    }

    [[nodiscard]] SaddleIndex saddle_index(std::size_t capacity = 8) const {
        rgsaddle_saddle_index_t out{};
        std::vector<double> curvatures(capacity);
        check(rgsaddle_minmode_saddle_index(handle_, &out, curvatures.data(),
                                            static_cast<std::int64_t>(capacity)));
        return SaddleIndex::from_c(out, std::move(curvatures));
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

    // nullptr turns the check off. ABI minor 40.
    void set_saddle_check(rgsaddle_saddle_check_t const* config) {
        check(rgsaddle_ocineb_set_saddle_check(handle_, config));
    }

    [[nodiscard]] SaddleIndex saddle_index(std::size_t capacity = 8) const {
        rgsaddle_saddle_index_t out{};
        std::vector<double> curvatures(capacity);
        check(rgsaddle_ocineb_saddle_index(handle_, &out, curvatures.data(),
                                           static_cast<std::int64_t>(capacity)));
        return SaddleIndex::from_c(out, std::move(curvatures));
    }

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

class IndexK {
public:
    IndexK(rgsaddle_indexk_config_t const& config, std::int64_t n_atoms,
           double const* position, double const* modes)
        : n_atoms_(n_atoms), k_(config.index) {
        if (!abi_compatible()) {
            throw std::runtime_error("rgsaddle ABI 1.15 or later is required");
        }
        handle_ = rgsaddle_indexk_create(&config, n_atoms, position, modes);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_indexk_create failed");
        }
    }

    IndexK(IndexK const&) = delete;
    IndexK& operator=(IndexK const&) = delete;

    IndexK(IndexK&& other) noexcept
        : handle_(other.handle_), n_atoms_(other.n_atoms_), k_(other.k_) {
        other.handle_ = nullptr;
    }

    IndexK& operator=(IndexK&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_atoms_ = other.n_atoms_;
            k_ = other.k_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~IndexK() { reset_handle(); }

    Report step(rgsaddle_surface_fn surface, rgsaddle_hessian_fn hessian, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_indexk_step(handle_, surface, hessian, user, &out));
        return Report::from_c(out);
    }

    void set_position(double const* position, double const* gradient = nullptr) {
        check(rgsaddle_indexk_set_position(handle_, position, gradient));
    }

    void position(double* out) const { check(rgsaddle_indexk_position(handle_, out)); }

    void modes(double* modes, double* curvatures) const {
        check(rgsaddle_indexk_modes(handle_, modes, curvatures));
    }

    void translated(double* out) const { check(rgsaddle_indexk_translated(handle_, out)); }

    [[nodiscard]] std::vector<double> position() const {
        std::vector<double> out(static_cast<std::size_t>(3 * n_atoms_));
        position(out.data());
        return out;
    }

    [[nodiscard]] RgsaddleIndexK* native() const noexcept { return handle_; }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_indexk_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleIndexK* handle_ = nullptr;
    std::int64_t n_atoms_ = 0;
    std::int32_t k_ = 1;
};

inline rgsaddle_growing_string_config_t growing_string_config() {
    rgsaddle_growing_string_config_t c{};
    check(rgsaddle_growing_string_config_default(&c));
    return c;
}

class GrowingString {
public:
    GrowingString(rgsaddle_growing_string_config_t const& config,
                  std::int64_t n_atoms, double const* reactant,
                  double const* product)
        : n_atoms_(n_atoms) {
        if (!abi_at_least(growing_string_abi_minor)) {
            throw std::runtime_error("rgsaddle ABI 1.44 or later is required");
        }
        handle_ = rgsaddle_growing_string_create(&config, n_atoms, reactant, product);
        if (handle_ == nullptr) {
            throw std::runtime_error("rgsaddle_growing_string_create failed");
        }
    }

    GrowingString(GrowingString const&) = delete;
    GrowingString& operator=(GrowingString const&) = delete;

    GrowingString(GrowingString&& other) noexcept
        : handle_(other.handle_), n_atoms_(other.n_atoms_) {
        other.handle_ = nullptr;
    }

    GrowingString& operator=(GrowingString&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            n_atoms_ = other.n_atoms_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~GrowingString() { reset_handle(); }

    Report step(rgsaddle_surface_fn surface, void* user) {
        rgsaddle_report_t out{};
        check(rgsaddle_growing_string_step(handle_, surface, user, &out));
        return Report::from_c(out);
    }

    [[nodiscard]] std::int64_t n_images() const noexcept {
        return rgsaddle_growing_string_n_images(handle_);
    }

    [[nodiscard]] std::int64_t nodes_added() const noexcept {
        return rgsaddle_growing_string_nodes_added(handle_);
    }

    [[nodiscard]] bool joined() const noexcept {
        return rgsaddle_growing_string_phase(handle_) == RGSADDLE_GROWING_STRING_JOINED;
    }

    void positions(double* out) const {
        check(rgsaddle_growing_string_positions(handle_, out));
    }

    [[nodiscard]] std::vector<double> positions() const {
        const auto rows = n_images();
        std::vector<double> out(static_cast<std::size_t>(rows * 3 * n_atoms_));
        positions(out.data());
        return out;
    }

    [[nodiscard]] std::vector<std::uint64_t> ids() const {
        const auto rows = n_images();
        std::vector<std::uint64_t> out(static_cast<std::size_t>(rows));
        check(rgsaddle_growing_string_ids(handle_, out.data()));
        return out;
    }

    void set_fixed_atoms(std::uint8_t const* mask) {
        check(rgsaddle_growing_string_set_fixed_atoms(handle_, mask, n_atoms_));
    }

    [[nodiscard]] SaddleIndex saddle_index(std::size_t capacity = 8) const {
        rgsaddle_saddle_index_t out{};
        std::vector<double> curvatures(capacity);
        check(rgsaddle_growing_string_saddle_index(
            handle_, &out, curvatures.data(), static_cast<std::int64_t>(capacity)));
        return SaddleIndex::from_c(out, std::move(curvatures));
    }

    [[nodiscard]] RgsaddleGrowingString* native() const noexcept { return handle_; }

private:
    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_growing_string_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleGrowingString* handle_ = nullptr;
    std::int64_t n_atoms_ = 0;
};

}  // namespace rgsaddle
