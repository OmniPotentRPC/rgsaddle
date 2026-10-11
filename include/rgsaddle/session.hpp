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

inline void check_text(int status) {
    if (status != RGSADDLE_OK) {
        const char* text = rgsaddle_last_error();
        if (text != nullptr && text[0] != '\0') {
            throw std::runtime_error(text);
        }
        throw std::runtime_error(status_name(status));
    }
}

enum class Preset : int32_t {
    CiNeb = RGSADDLE_PRESET_CI_NEB,
    EonZoom = RGSADDLE_PRESET_EON_ZOOM,
    OrcaZoomNebTs = RGSADDLE_PRESET_ORCA_ZOOM_NEB_TS,
    AutoNeb = RGSADDLE_PRESET_AUTONEB,
    Dimer = RGSADDLE_PRESET_DIMER,
    Lanczos = RGSADDLE_PRESET_LANCZOS,
    Art = RGSADDLE_PRESET_ART,
    Plain = RGSADDLE_PRESET_PLAIN
};

enum class ZoomEnds : int32_t {
    Fixed = 0,
    FreePerp = 1
};

struct StencilReport {
    int phase = RGSADDLE_STENCIL_PHASE_BAND;
    int status = RGSADDLE_STATUS_RUNNING;
    double max_force = 0.0;
    std::int64_t ci_index = -1;
    std::int64_t iteration = 0;
    std::int64_t surface_rows = 0;
    std::int64_t total_evaluations = 0;
    std::int64_t zoom_lo = -1;
    std::int64_t zoom_hi = -1;
    double curvature = 0.0;

    [[nodiscard]] bool converged() const noexcept {
        return status == RGSADDLE_STATUS_CONVERGED;
    }
};

class Stencil;

class StencilBuilder {
public:
    StencilBuilder(std::vector<double> positions, std::size_t rows, std::size_t dof, Preset preset)
        : positions_(std::move(positions)), rows_(rows), dof_(dof),
          config_(rgsaddle_stencil_config_new(static_cast<int32_t>(preset))) {
        if (config_ == nullptr) {
            throw std::runtime_error(rgsaddle_last_error());
        }
    }

    StencilBuilder(StencilBuilder const&) = delete;
    StencilBuilder& operator=(StencilBuilder const&) = delete;

    StencilBuilder(StencilBuilder&& other) noexcept
        : positions_(std::move(other.positions_)), rows_(other.rows_), dof_(other.dof_),
          config_(other.config_) {
        other.config_ = nullptr;
    }

    StencilBuilder& operator=(StencilBuilder&&) = delete;

    ~StencilBuilder() {
        if (config_ != nullptr) {
            rgsaddle_stencil_config_free(config_);
        }
    }

    StencilBuilder& ci_neb() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_CI_NEB));
        return *this;
    }

    StencilBuilder& plain() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_PLAIN));
        return *this;
    }

    StencilBuilder& autoneb() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_AUTONEB));
        return *this;
    }

    StencilBuilder& zoom() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_ZOOM));
        return *this;
    }

    StencilBuilder& dimer() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_DIMER));
        return *this;
    }

    StencilBuilder& lanczos() {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_METHOD,
                                                   RGSADDLE_STENCIL_METHOD_LANCZOS));
        return *this;
    }

    StencilBuilder& energy_weighted(double k_min, double k_max) {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_SPRING_MODEL,
                                                   RGSADDLE_STENCIL_SPRING_ENERGY_WEIGHTED));
        check_text(rgsaddle_stencil_config_set_f64(config_, RGSADDLE_STENCIL_KEY_K_MIN, k_min));
        check_text(rgsaddle_stencil_config_set_f64(config_, RGSADDLE_STENCIL_KEY_K_MAX, k_max));
        return *this;
    }

    StencilBuilder& ocineb(double trust) {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_REFINE, 1));
        check_text(rgsaddle_stencil_config_set_f64(config_, RGSADDLE_STENCIL_KEY_OCINEB_TRUST, trust));
        return *this;
    }

    StencilBuilder& ef(double activate) {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_REFINE, 2));
        check_text(rgsaddle_stencil_config_set_f64(config_, RGSADDLE_STENCIL_KEY_EF_ACTIVATE, activate));
        return *this;
    }

    StencilBuilder& ends(ZoomEnds ends) {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_ZOOM_ENDS,
                                                   static_cast<int64_t>(ends)));
        return *this;
    }

    StencilBuilder& threads(std::int64_t count) {
        check_text(rgsaddle_stencil_config_set_i64(config_, RGSADDLE_STENCIL_KEY_THREADS, count));
        return *this;
    }

    StencilBuilder& preset(Preset preset) {
        RgsaddleStencilConfig* next = rgsaddle_stencil_config_new(static_cast<int32_t>(preset));
        if (next == nullptr) {
            throw std::runtime_error(rgsaddle_last_error());
        }
        rgsaddle_stencil_config_free(config_);
        config_ = next;
        return *this;
    }

    [[nodiscard]] Stencil build();

private:
    friend class Stencil;
    std::vector<double> positions_;
    std::size_t rows_ = 0;
    std::size_t dof_ = 0;
    RgsaddleStencilConfig* config_ = nullptr;
};

class Stencil {
public:
    static StencilBuilder band(std::vector<double> positions, std::size_t images, std::size_t dof) {
        return StencilBuilder(std::move(positions), images, dof, Preset::CiNeb);
    }

    static StencilBuilder point(std::vector<double> x0) {
        std::size_t dof = x0.size();
        return StencilBuilder(std::move(x0), 1, dof, Preset::Dimer);
    }

    Stencil(Stencil const&) = delete;
    Stencil& operator=(Stencil const&) = delete;

    Stencil(Stencil&& other) noexcept
        : handle_(other.handle_), rows_(other.rows_), dof_(other.dof_) {
        other.handle_ = nullptr;
    }

    Stencil& operator=(Stencil&& other) noexcept {
        if (this != &other) {
            reset_handle();
            handle_ = other.handle_;
            rows_ = other.rows_;
            dof_ = other.dof_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Stencil() { reset_handle(); }

    template <class Surface>
    StencilReport step(Surface& surface) {
        return call_steps(surface, 0);
    }

    template <class Surface>
    StencilReport run(Surface& surface, std::size_t max_steps) {
        return call_steps(surface, max_steps);
    }

    [[nodiscard]] std::vector<double> positions() const {
        std::vector<double> out(rows_ * dof_);
        check_text(rgsaddle_stencil_positions(handle_, out.data()));
        return out;
    }

    [[nodiscard]] RgsaddleStencil* native() const noexcept { return handle_; }

private:
    friend class StencilBuilder;

    Stencil(RgsaddleStencil* handle, std::size_t rows, std::size_t dof)
        : handle_(handle), rows_(rows), dof_(dof) {}

    template <class Surface>
    static int trampoline(void* user, std::size_t rows, std::size_t dof, uint64_t const* ids,
                          double const* x, double* energy, double* gradient) {
        try {
            auto* surface = static_cast<Surface*>(user);
            return (*surface)(ids, rows, dof, x, energy, gradient);
        } catch (std::exception const&) {
            return 2;
        } catch (...) {
            return 3;
        }
    }

    template <class Surface>
    StencilReport call_steps(Surface& surface, std::size_t max_steps) {
        rgsaddle_stencil_report_t out{};
        out.struct_size = sizeof(out);
        int status = 0;
        if (max_steps == 0) {
            status = rgsaddle_stencil_step(handle_, &Stencil::trampoline<Surface>, &surface, &out);
        } else {
            status = rgsaddle_stencil_run(handle_, &Stencil::trampoline<Surface>, &surface,
                                          max_steps, &out);
        }
        check_text(status);
        refresh_shape();
        StencilReport report;
        report.phase = out.phase;
        report.status = out.status;
        report.max_force = out.max_force;
        report.ci_index = out.ci_index;
        report.iteration = out.iteration;
        report.surface_rows = out.surface_rows;
        report.total_evaluations = out.total_evaluations;
        report.zoom_lo = out.zoom_lo;
        report.zoom_hi = out.zoom_hi;
        report.curvature = out.curvature;
        return report;
    }

    void refresh_shape() {
        if (handle_ == nullptr) {
            return;
        }
        std::int64_t shape[2] = {};
        if (rgsaddle_stencil_query(handle_, RGSADDLE_STENCIL_QUERY_SHAPE, shape, 2) == RGSADDLE_OK &&
            shape[0] > 0 && shape[1] > 0) {
            rows_ = static_cast<std::size_t>(shape[0]);
            dof_ = static_cast<std::size_t>(shape[1]);
        }
    }

    void reset_handle() {
        if (handle_ != nullptr) {
            rgsaddle_stencil_free(handle_);
            handle_ = nullptr;
        }
    }

    RgsaddleStencil* handle_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t dof_ = 0;
};

inline Stencil StencilBuilder::build() {
    RgsaddleStencil* handle = rgsaddle_stencil_new(config_, rows_, dof_, positions_.data());
    if (handle == nullptr) {
        throw std::runtime_error(rgsaddle_last_error());
    }
    std::size_t rows = rows_;
    std::size_t dof = dof_;
    Stencil stencil(handle, rows, dof);
    stencil.refresh_shape();
    return stencil;
}

}  // namespace rgsaddle
