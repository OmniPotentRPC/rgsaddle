#pragma once

/**
 * \file rgsaddle/stencil.hpp
 * \brief RAII point stencil over the C ABI.
 *
 * The handle is move-only. A compare-exchange claim rejects a second
 * thread. Exceptions from the callback are stored and rethrown after
 * the C call returns, so nothing unwinds through the C boundary.
 * Band presets fail with the feat/band-methods-zoom seam.
 */

#include "../rgsaddle.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

namespace rgsaddle {

class Error : public std::runtime_error {
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

inline std::string last_error_string() {
  const char *text = rgsaddle_last_error();
  if (text == nullptr) {
    return {};
  }
  return std::string(text);
}

inline void check_stencil(int status) {
  if (status != RGSADDLE_OK) {
    std::string text = last_error_string();
    if (text.empty()) {
      const char *name = rgsaddle_status_name(static_cast<rgsaddle_status_t>(status));
      text = name == nullptr ? "stencil call failed" : name;
    }
    throw Error(text);
  }
}

struct Bridge {
  void (*run)(void *fn, std::size_t rows, std::size_t dof, const std::uint64_t *ids, const double *x,
              double *e, double *g) = nullptr;
  void *fn = nullptr;
  std::exception_ptr error;
};

extern "C" inline int rgsaddle_stencil_bridge_entry(void *user, std::size_t rows, std::size_t dof,
                                                    const std::uint64_t *ids, const double *x,
                                                    double *e, double *g) {
  auto *bridge = static_cast<Bridge *>(user);
  if (bridge == nullptr || bridge->run == nullptr) {
    return RGSADDLE_NULL_SURFACE;
  }
  try {
    bridge->run(bridge->fn, rows, dof, ids, x, e, g);
    return RGSADDLE_OK;
  } catch (...) {
    bridge->error = std::current_exception();
    return RGSADDLE_SURFACE_FAILED;
  }
}

struct StencilReport {
  int status = 0;
  double max_force = 0.0;
  double curvature = 0.0;
  std::uint64_t iteration = 0;
  std::uint64_t evaluations = 0;

  static StencilReport from_c(const rgsaddle_stencil_report_t &report) {
    StencilReport out;
    out.status = report.status;
    out.max_force = report.max_force;
    out.curvature = report.curvature;
    out.iteration = report.iteration;
    out.evaluations = report.evaluations;
    return out;
  }

  bool converged() const noexcept { return status == 1; }
};

class Stencil {
public:
  Stencil() = default;
  Stencil(const Stencil &) = delete;
  Stencil &operator=(const Stencil &) = delete;

  Stencil(Stencil &&other) noexcept : core_(std::move(other.core_)) {}

  Stencil &operator=(Stencil &&other) noexcept {
    if (this != &other) {
      core_ = std::move(other.core_);
    }
    return *this;
  }

  ~Stencil() = default;

  explicit operator bool() const noexcept { return static_cast<bool>(core_); }

  template <class Fn>
  StencilReport step(Fn &&fn) {
    return invoke(1, std::forward<Fn>(fn), false);
  }

  template <class Fn>
  StencilReport run(Fn &&fn, std::size_t max_steps) {
    return invoke(max_steps, std::forward<Fn>(fn), true);
  }

  std::vector<double> positions() const {
    Claim claim(busy());
    if (!core_ || core_->handle == nullptr) {
      throw Error("stencil is null");
    }
    std::vector<double> out(core_->dof, 0.0);
    check_stencil(rgsaddle_stencil_positions(core_->handle, out.data(), out.size()));
    return out;
  }

  std::vector<double> mode() const {
    Claim claim(busy());
    if (!core_ || core_->handle == nullptr) {
      throw Error("stencil is null");
    }
    std::vector<double> out(core_->dof, 0.0);
    int status = rgsaddle_stencil_query(core_->handle, RGSADDLE_STENCIL_QUERY_MODE, out.data(),
                                        out.size());
    if (status != RGSADDLE_OK) {
      return {};
    }
    return out;
  }

  double curvature() const {
    Claim claim(busy());
    if (!core_ || core_->handle == nullptr) {
      throw Error("stencil is null");
    }
    double value = 0.0;
    check_stencil(
        rgsaddle_stencil_query(core_->handle, RGSADDLE_STENCIL_QUERY_CURVATURE, &value, 1));
    return value;
  }

private:
  struct Core {
    rgsaddle_stencil_t *handle = nullptr;
    std::size_t dof = 0;
    mutable std::atomic<bool> busy{false};

    ~Core() {
      if (handle != nullptr) {
        rgsaddle_stencil_free(handle);
      }
    }
  };

  struct Claim {
    std::atomic<bool> *flag = nullptr;

    explicit Claim(std::atomic<bool> &busy) : flag(&busy) {
      bool expected = false;
      if (!flag->compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
        flag = nullptr;
        throw Error("Stencil is in use by another thread");
      }
    }

    Claim(const Claim &) = delete;
    Claim &operator=(const Claim &) = delete;

    ~Claim() {
      if (flag != nullptr) {
        flag->store(false, std::memory_order_release);
      }
    }
  };

  template <class Fn>
  StencilReport invoke(std::size_t max_steps, Fn &&fn, bool run_to_end) {
    Claim claim(busy());
    if (!core_ || core_->handle == nullptr) {
      throw Error("stencil is null");
    }
    std::decay_t<Fn> held(std::forward<Fn>(fn));
    Bridge bridge;
    bridge.run = [](void *fn, std::size_t rows, std::size_t dof, const std::uint64_t *ids,
                    const double *x, double *e, double *g) {
      (*static_cast<std::decay_t<Fn> *>(fn))(rows, dof, ids, x, e, g);
    };
    bridge.fn = &held;
    rgsaddle_stencil_report_t report{};
    report.struct_size = sizeof(report);
    int status =
        run_to_end ? rgsaddle_stencil_run(core_->handle, &rgsaddle_stencil_bridge_entry, &bridge,
                                          max_steps, &report)
                   : rgsaddle_stencil_step(core_->handle, &rgsaddle_stencil_bridge_entry, &bridge,
                                           &report);
    if (bridge.error) {
      std::rethrow_exception(bridge.error);
    }
    check_stencil(status);
    return StencilReport::from_c(report);
  }

  std::atomic<bool> &busy() const {
    if (!core_) {
      throw Error("stencil is null");
    }
    return core_->busy;
  }

  std::unique_ptr<Core> core_;

  friend class PointBuilder;
};

class PointBuilder {
public:
  PointBuilder(const double *x, std::size_t dof, int preset)
      : x_(x, x + dof), config_(rgsaddle_stencil_config_new(preset)) {
    if (config_ == nullptr) {
      throw Error(last_error_string().empty() ? "unknown stencil preset" : last_error_string());
    }
  }

  PointBuilder(const PointBuilder &) = delete;
  PointBuilder &operator=(const PointBuilder &) = delete;

  PointBuilder(PointBuilder &&other) noexcept
      : x_(std::move(other.x_)), config_(other.config_) {
    other.config_ = nullptr;
  }

  PointBuilder &operator=(PointBuilder &&other) noexcept {
    if (this != &other) {
      reset();
      x_ = std::move(other.x_);
      config_ = other.config_;
      other.config_ = nullptr;
    }
    return *this;
  }

  ~PointBuilder() { reset(); }

  PointBuilder &set_f64(int key, double value) {
    check_stencil(rgsaddle_stencil_config_set_f64(config_, key, value));
    return *this;
  }

  PointBuilder &set_i64(int key, std::int64_t value) {
    check_stencil(rgsaddle_stencil_config_set_i64(config_, key, value));
    return *this;
  }

  PointBuilder &mode(const double *values, std::size_t n) {
    check_stencil(rgsaddle_stencil_config_set_array(config_, RGSADDLE_STENCIL_KEY_MODE, values, n));
    return *this;
  }

  Stencil build() {
    Stencil out;
    auto core = std::make_unique<Stencil::Core>();
    core->dof = x_.size();
    core->handle = rgsaddle_stencil_new(config_, 1, x_.size(), x_.data());
    if (core->handle == nullptr) {
      throw Error(last_error_string().empty() ? "stencil build failed" : last_error_string());
    }
    out.core_ = std::move(core);
    return out;
  }

private:
  void reset() {
    if (config_ != nullptr) {
      rgsaddle_stencil_config_free(config_);
      config_ = nullptr;
    }
  }

  std::vector<double> x_;
  rgsaddle_stencil_config_t *config_ = nullptr;
};

inline Stencil point(const double *x, std::size_t dof, int preset = RGSADDLE_STENCIL_DIMER) {
  return PointBuilder(x, dof, preset).build();
}

}  // namespace rgsaddle
