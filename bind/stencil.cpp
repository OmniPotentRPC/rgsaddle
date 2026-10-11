// MIT License
// Copyright 2026 Rohit Goswami

/**
 * @file stencil.cpp
 * @brief Nanobind Stencil over the C ABI.
 *
 * The GIL is released around the Rust step and taken again inside the
 * callback. threads > 1 with a Python callable is refused: the callback
 * is not free-threaded. The calling form is detected once per callable
 * with inspect.signature.
 */

#include <rgsaddle.h>

#include <nanobind/nanobind.h>
#include <nanobind/eval.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

thread_local std::string callback_error;

struct PyStencil;

int detect_form(nb::handle fn) {
  nb::dict scope;
  nb::exec(
      "def form(fn):\n"
      "    import inspect\n"
      "    names = []\n"
      "    for param in inspect.signature(fn).parameters.values():\n"
      "        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,\n"
      "                          inspect.Parameter.POSITIONAL_OR_KEYWORD):\n"
      "            names.append(param.name)\n"
      "    if names and names[0] in ('ids', 'id'):\n"
      "        return 0\n"
      "    if len(names) <= 1:\n"
      "        return 1\n"
      "    return 2\n",
      scope, scope);
  return nb::cast<int>(scope["form"](fn));
}

struct PyBuilder {
  std::vector<double> positions;
  std::size_t rows = 0;
  std::size_t dof = 0;
  RgsaddleStencilConfig *config = nullptr;
  std::int64_t threads = 1;

  PyBuilder() = default;
  PyBuilder(const PyBuilder &) = delete;
  PyBuilder &operator=(const PyBuilder &) = delete;
  PyBuilder(PyBuilder &&other) noexcept
      : positions(std::move(other.positions)), rows(other.rows), dof(other.dof), config(other.config),
        threads(other.threads) {
    other.config = nullptr;
  }
  PyBuilder &operator=(PyBuilder &&) = delete;
  ~PyBuilder() {
    if (config != nullptr) {
      rgsaddle_stencil_config_free(config);
    }
  }

  static PyBuilder make(std::vector<double> positions, std::size_t rows, std::size_t dof, int preset) {
    PyBuilder builder;
    builder.positions = std::move(positions);
    builder.rows = rows;
    builder.dof = dof;
    builder.config = rgsaddle_stencil_config_new(preset);
    if (builder.config == nullptr) {
      throw std::runtime_error(rgsaddle_last_error());
    }
    return builder;
  }

  void set_i64(int key, std::int64_t value) {
    int status = rgsaddle_stencil_config_set_i64(config, key, value);
    if (status != RGSADDLE_OK) {
      throw std::runtime_error(rgsaddle_last_error());
    }
    if (key == RGSADDLE_STENCIL_KEY_THREADS) {
      threads = value;
    }
  }

  void set_f64(int key, double value) {
    int status = rgsaddle_stencil_config_set_f64(config, key, value);
    if (status != RGSADDLE_OK) {
      throw std::runtime_error(rgsaddle_last_error());
    }
  }
};

struct PyStencil {
  RgsaddleStencil *handle = nullptr;
  std::size_t rows = 0;
  std::size_t dof = 0;
  std::int64_t threads = 1;
  nb::object callback;
  int form = -1;
  uintptr_t callback_id = 0;

  PyStencil() = default;
  PyStencil(const PyStencil &) = delete;
  PyStencil &operator=(const PyStencil &) = delete;
  PyStencil(PyStencil &&other) noexcept
      : handle(other.handle), rows(other.rows), dof(other.dof), threads(other.threads),
        callback(std::move(other.callback)), form(other.form), callback_id(other.callback_id) {
    other.handle = nullptr;
  }
  PyStencil &operator=(PyStencil &&) = delete;
  ~PyStencil() {
    if (handle != nullptr) {
      rgsaddle_stencil_free(handle);
    }
  }

  void refresh() {
    std::int64_t shape[2] = {};
    if (rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_SHAPE, shape, 2) == RGSADDLE_OK && shape[0] > 0 &&
        shape[1] > 0) {
      rows = static_cast<std::size_t>(shape[0]);
      dof = static_cast<std::size_t>(shape[1]);
    }
  }

  void remember(nb::handle fn) {
    uintptr_t id = reinterpret_cast<uintptr_t>(fn.ptr());
    if (form >= 0 && id == callback_id) {
      return;
    }
    form = detect_form(fn);
    callback = nb::borrow(fn);
    callback_id = id;
  }

  int call_python(std::size_t nrows, std::size_t ndof, uint64_t const *ids, double const *x, double *energy,
                  double *gradient) {
    std::vector<std::int64_t> id_copy(nrows);
    std::vector<double> x_copy(nrows * ndof);
    for (std::size_t row = 0; row < nrows; ++row) {
      id_copy[row] = static_cast<std::int64_t>(ids[row]);
    }
    if (nrows * ndof != 0) {
      std::memcpy(x_copy.data(), x, nrows * ndof * sizeof(double));
    }
    auto *id_raw = new std::int64_t[nrows == 0 ? 1 : nrows];
    auto *x_raw = new double[x_copy.empty() ? 1 : x_copy.size()];
    if (nrows != 0) {
      std::memcpy(id_raw, id_copy.data(), nrows * sizeof(std::int64_t));
    }
    if (!x_copy.empty()) {
      std::memcpy(x_raw, x_copy.data(), x_copy.size() * sizeof(double));
    }
    nb::capsule id_owner(id_raw, [](void *ptr) noexcept { delete[] static_cast<std::int64_t *>(ptr); });
    nb::capsule x_owner(x_raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
    const std::size_t id_shape[1] = {nrows};
    const std::size_t x_shape[2] = {nrows, ndof};
    nb::ndarray<nb::numpy, std::int64_t> id_arr(id_raw, 1, id_shape, id_owner);
    nb::ndarray<nb::numpy, double> x_arr(x_raw, 2, x_shape, x_owner);
    nb::object result;
    try {
      if (form == 0) {
        result = callback(id_arr, x_arr);
      } else if (form == 1) {
        result = callback(x_arr);
      } else {
        result = callback(x_arr, id_arr);
      }
    } catch (std::exception const &err) {
      throw std::runtime_error(std::string("stencil callback: ") + err.what());
    }
    if (!nb::isinstance<nb::tuple>(result)) {
      throw std::runtime_error("stencil callback must return (energies, gradients)");
    }
    nb::tuple pair = nb::cast<nb::tuple>(result);
    if (pair.size() != 2) {
      throw std::runtime_error("stencil callback must return (energies, gradients)");
    }
    nb::ndarray<double, nb::ndim<1>, nb::c_contig> energies;
    try {
      energies = nb::cast<nb::ndarray<double, nb::ndim<1>, nb::c_contig>>(pair[0]);
    } catch (std::exception const &err) {
      throw std::runtime_error(std::string("stencil energies: ") + err.what());
    }
    if (static_cast<std::size_t>(energies.shape(0)) != nrows) {
      throw std::runtime_error("stencil energies have the wrong length");
    }
    std::memcpy(energy, energies.data(), nrows * sizeof(double));
    nb::object grad_obj = pair[1];
    bool copied = false;
    try {
      auto grad = nb::cast<nb::ndarray<double, nb::ndim<2>, nb::c_contig>>(grad_obj, false);
      if (static_cast<std::size_t>(grad.shape(0)) != nrows || static_cast<std::size_t>(grad.shape(1)) != ndof) {
        throw std::runtime_error("stencil gradients have the wrong shape");
      }
      std::memcpy(gradient, grad.data(), nrows * ndof * sizeof(double));
      copied = true;
    } catch (nb::cast_error const &) {
      copied = false;
    }
    if (!copied) {
      auto grad = nb::cast<nb::ndarray<double, nb::ndim<1>, nb::c_contig>>(grad_obj);
      if (static_cast<std::size_t>(grad.shape(0)) != nrows * ndof) {
        throw std::runtime_error("stencil gradients have the wrong length");
      }
      std::memcpy(gradient, grad.data(), nrows * ndof * sizeof(double));
    }
    return 0;
  }
};

int trampoline(void *user, std::size_t rows, std::size_t dof, uint64_t const *ids, double const *x, double *energy,
               double *gradient) {
  auto *self = static_cast<PyStencil *>(user);
  try {
    nb::gil_scoped_acquire gil;
    callback_error.clear();
    return self->call_python(rows, dof, ids, x, energy, gradient);
  } catch (nb::python_error &err) {
    callback_error = err.what();
    return 2;
  } catch (std::exception const &err) {
    callback_error = err.what();
    return 2;
  } catch (...) {
    callback_error = "stencil callback failed";
    return 3;
  }
}

struct PyReport {
  int phase = 0;
  int status = 0;
  double max_force = 0.0;
  std::int64_t ci_index = -1;
  std::int64_t iteration = 0;
  std::int64_t surface_rows = 0;
  std::int64_t total_evaluations = 0;
  std::int64_t zoom_lo = -1;
  std::int64_t zoom_hi = -1;
  double curvature = 0.0;
};

PyReport take_report(rgsaddle_stencil_report_t const &out) {
  PyReport report;
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

PyReport drive(PyStencil &stencil, nb::handle fn, std::size_t max_steps) {
  if (stencil.threads > 1) {
    throw nb::value_error("threads > 1 with a Python callable is refused");
  }
  stencil.remember(fn);
  rgsaddle_stencil_report_t out{};
  out.struct_size = sizeof(out);
  int status = 0;
  {
    nb::gil_scoped_release release;
    if (max_steps == 0) {
      status = rgsaddle_stencil_step(stencil.handle, trampoline, &stencil, &out);
    } else {
      status = rgsaddle_stencil_run(stencil.handle, trampoline, &stencil, max_steps, &out);
    }
  }
  if (status != RGSADDLE_OK) {
    if (!callback_error.empty()) {
      throw std::runtime_error(callback_error);
    }
    const char *text = rgsaddle_last_error();
    throw std::runtime_error(text == nullptr ? "stencil step failed" : text);
  }
  stencil.refresh();
  return take_report(out);
}

PyBuilder &method_key(PyBuilder &builder, int method) {
  builder.set_i64(RGSADDLE_STENCIL_KEY_METHOD, method);
  return builder;
}

} // namespace

void rgsaddle_bind_stencil(nb::module_ &m) {
  nb::class_<PyReport>(m, "StencilReport")
      .def_ro("phase", &PyReport::phase)
      .def_ro("status", &PyReport::status)
      .def_ro("max_force", &PyReport::max_force)
      .def_ro("ci_index", &PyReport::ci_index)
      .def_ro("iteration", &PyReport::iteration)
      .def_ro("surface_rows", &PyReport::surface_rows)
      .def_ro("total_evaluations", &PyReport::total_evaluations)
      .def_ro("zoom_lo", &PyReport::zoom_lo)
      .def_ro("zoom_hi", &PyReport::zoom_hi)
      .def_ro("curvature", &PyReport::curvature)
      .def("zoom_window", [](PyReport const &report) -> nb::object {
        if (report.zoom_lo < 0 || report.zoom_hi < 0) {
          return nb::none();
        }
        return nb::make_tuple(report.zoom_lo, report.zoom_hi);
      });

  nb::class_<PyBuilder>(m, "StencilBuilder")
      .def("ci_neb", [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_CI_NEB); },
           nb::rv_policy::reference)
      .def("plain", [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_PLAIN); },
           nb::rv_policy::reference)
      .def("zoom", [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_ZOOM); },
           nb::rv_policy::reference)
      .def("autoneb",
           [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_AUTONEB); },
           nb::rv_policy::reference)
      .def("dimer", [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_DIMER); },
           nb::rv_policy::reference)
      .def("lanczos",
           [](PyBuilder &builder) -> PyBuilder & { return method_key(builder, RGSADDLE_STENCIL_METHOD_LANCZOS); },
           nb::rv_policy::reference)
      .def(
          "energy_weighted",
          [](PyBuilder &builder, double k_min, double k_max) -> PyBuilder & {
            builder.set_i64(RGSADDLE_STENCIL_KEY_SPRING_MODEL, RGSADDLE_STENCIL_SPRING_ENERGY_WEIGHTED);
            builder.set_f64(RGSADDLE_STENCIL_KEY_K_MIN, k_min);
            builder.set_f64(RGSADDLE_STENCIL_KEY_K_MAX, k_max);
            return builder;
          },
          nb::arg("k_min"), nb::arg("k_max"), nb::rv_policy::reference)
      .def(
          "threads",
          [](PyBuilder &builder, std::int64_t count) -> PyBuilder & {
            builder.set_i64(RGSADDLE_STENCIL_KEY_THREADS, count);
            return builder;
          },
          nb::arg("count"), nb::rv_policy::reference)
      .def(
          "preset",
          [](PyBuilder &builder, int preset) -> PyBuilder & {
            std::int64_t kept = builder.threads;
            RgsaddleStencilConfig *next = rgsaddle_stencil_config_new(preset);
            if (next == nullptr) {
              throw std::runtime_error(rgsaddle_last_error());
            }
            rgsaddle_stencil_config_free(builder.config);
            builder.config = next;
            builder.set_i64(RGSADDLE_STENCIL_KEY_THREADS, kept);
            return builder;
          },
          nb::arg("preset"), nb::rv_policy::reference)
      .def(
          "ends",
          [](PyBuilder &builder, std::int64_t ends) -> PyBuilder & {
            builder.set_i64(RGSADDLE_STENCIL_KEY_ZOOM_ENDS, ends);
            return builder;
          },
          nb::arg("ends"), nb::rv_policy::reference)
      .def(
          "ci_tol",
          [](PyBuilder &builder, double tol) -> PyBuilder & {
            builder.set_f64(RGSADDLE_STENCIL_KEY_CI_TOL, tol);
            return builder;
          },
          nb::arg("tol"), nb::rv_policy::reference)
      .def(
          "zoom_stability",
          [](PyBuilder &builder, std::int64_t count) -> PyBuilder & {
            builder.set_i64(RGSADDLE_STENCIL_KEY_ZOOM_STABILITY, count);
            return builder;
          },
          nb::arg("count"), nb::rv_policy::reference)
      .def(
          "zoom_force_abs",
          [](PyBuilder &builder, double force) -> PyBuilder & {
            builder.set_i64(RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND, 1);
            builder.set_f64(RGSADDLE_STENCIL_KEY_ZOOM_FORCE, force);
            return builder;
          },
          nb::arg("force"), nb::rv_policy::reference)
      .def("set_i64", &PyBuilder::set_i64, nb::arg("key"), nb::arg("value"))
      .def("set_f64", &PyBuilder::set_f64, nb::arg("key"), nb::arg("value"))
      .def("build", [](PyBuilder &builder) {
        RgsaddleStencil *handle =
            rgsaddle_stencil_new(builder.config, builder.rows, builder.dof, builder.positions.data());
        if (handle == nullptr) {
          throw std::runtime_error(rgsaddle_last_error());
        }
        PyStencil stencil;
        stencil.handle = handle;
        stencil.rows = builder.rows;
        stencil.dof = builder.dof;
        stencil.threads = builder.threads;
        stencil.refresh();
        return stencil;
      });

  nb::class_<PyStencil>(m, "Stencil")
      .def_static(
          "band",
          [](nb::ndarray<double, nb::ndim<2>, nb::c_contig> positions) {
            const std::size_t rows = static_cast<std::size_t>(positions.shape(0));
            const std::size_t dof = static_cast<std::size_t>(positions.shape(1));
            std::vector<double> values(rows * dof);
            if (!values.empty()) {
              std::memcpy(values.data(), positions.data(), values.size() * sizeof(double));
            }
            return PyBuilder::make(std::move(values), rows, dof, RGSADDLE_PRESET_CI_NEB);
          },
          nb::arg("positions").noconvert())
      .def_static(
          "point",
          [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> x0) {
            const std::size_t dof = static_cast<std::size_t>(x0.shape(0));
            std::vector<double> values(dof);
            if (dof != 0) {
              std::memcpy(values.data(), x0.data(), dof * sizeof(double));
            }
            return PyBuilder::make(std::move(values), 1, dof, RGSADDLE_PRESET_DIMER);
          },
          nb::arg("x0").noconvert())
      .def(
          "step", [](PyStencil &stencil, nb::handle fn) { return drive(stencil, fn, 0); }, nb::arg("surface"))
      .def(
          "run",
          [](PyStencil &stencil, nb::handle fn, std::size_t max_steps) { return drive(stencil, fn, max_steps); },
          nb::arg("surface"), nb::arg("max_steps"))
      .def("positions", [](PyStencil &stencil) {
        stencil.refresh();
        const std::size_t n = stencil.rows * stencil.dof;
        auto *raw = new double[n == 0 ? 1 : n];
        int status = rgsaddle_stencil_positions(stencil.handle, raw);
        if (status != RGSADDLE_OK) {
          delete[] raw;
          throw std::runtime_error(rgsaddle_last_error());
        }
        nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
        const std::size_t shape[2] = {stencil.rows, stencil.dof};
        return nb::ndarray<nb::numpy, double>(raw, 2, shape, owner);
      });
}
