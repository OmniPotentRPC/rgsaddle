// MIT License
// Copyright 2026 Rohit Goswami

/**
 * @file ext.cpp
 * @brief Nanobind frontend for the rgsaddle Python package.
 *
 * F1 FREE_THREADED via nanobind_add_module. F2 the only mutable slot
 * is the thread-local last-error string. F3 ClaimSlot and
 * SurfaceAdapter take rgbind::Claim. F4 the eval loop drops the GIL
 * before workers run. F5 Python callbacks go through rgbind::trampoline
 * and rows are grouped by rgbind::serialise_by_owner. F6 float64
 * C-contiguous arrays, noconvert on the way in, capsule copies on the
 * way out. F7 raise_status maps rgsaddle_status_t. F8 the schedule
 * does not depend on the worker count. F9 the link exports PyInit__core.
 *
 * This module does not wrap band, minimum-mode, IRC, Sella, AutoNEB
 * or xts sessions. Those stay on the C ABI until the stencil owns them.
 */

#include <rgsaddle.h>

#include <rgsaddle/bind.h>

#include <rgbind/rgbind.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rgbind {

template <> struct status_traits<rgsaddle_status_t> {
  static int code(rgsaddle_status_t value) {
    switch (value) {
    case RGSADDLE_OK:
      return static_cast<int>(Status::ok);
    case RGSADDLE_SHAPE:
      return static_cast<int>(Status::build);
    case RGSADDLE_INVALID_PARAMETER:
      return static_cast<int>(Status::key);
    case RGSADDLE_SURFACE_FAILED:
      return static_cast<int>(Status::callback);
    case RGSADDLE_NON_FINITE:
      return static_cast<int>(Status::numerical);
    case RGSADDLE_NULL_SESSION:
    case RGSADDLE_NULL_BAND:
    case RGSADDLE_NULL_SURFACE:
    case RGSADDLE_NULL_REPORT:
    case RGSADDLE_SOLVER:
    case RGSADDLE_ABI_MISMATCH:
    case RGSADDLE_ALLOC:
    case RGSADDLE_NO_EVALUATION:
      return static_cast<int>(Status::other);
    }
    return static_cast<int>(Status::other);
  }
};

} // namespace rgbind

namespace nb = nanobind;

namespace {

thread_local std::string g_last_error;

struct HoldGate {
  std::atomic<bool> release_flag{false};
  std::atomic<bool> held_flag{false};

  void release() { release_flag.store(true, std::memory_order_release); }

  bool held() const { return held_flag.load(std::memory_order_acquire); }
};

struct ClaimSlot {
  rgbind::Busy busy;

  // F4: native hold, GIL released before the claim. F3: atomic claim.
  int work(int hold_ms) {
    nb::gil_scoped_release rel;
    rgbind::Claim claim(busy.flag(), "SurfaceAdapter");
    if (hold_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    }
    return 1;
  }

  void hold_until(HoldGate &gate) {
    nb::gil_scoped_release rel;
    rgbind::Claim claim(busy.flag(), "SurfaceAdapter");
    gate.held_flag.store(true, std::memory_order_release);
    while (!gate.release_flag.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};

using IdArray = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OwnerArray = nb::ndarray<const std::uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using FlagArray = nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

[[noreturn]] void shape_error(const char *name, const std::string &expected, const std::string &got) {
  const std::string msg = std::string(name) + ": expected shape " + expected + ", got " + got;
  throw nb::value_error(msg.c_str());
}

std::string vec1(std::size_t n) {
  return "(" + std::to_string(n) + ",)";
}

std::string vec2(std::size_t rows, std::size_t cols) {
  return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

struct GroupWork {
  std::vector<std::int64_t> row_ids;
};

struct EvalCall {
  nb::callable *callback = nullptr;
  const GroupWork *job = nullptr;
  const std::vector<double> *positions = nullptr;
  std::size_t dof = 0;
  const std::map<std::int64_t, std::size_t> *dest = nullptr;
  double *energies = nullptr;
  double *gradients = nullptr;

  void operator()() const {
    const std::size_t m = job->row_ids.size();
    std::vector<std::int64_t> ids(m);
    std::vector<double> rows(m * dof == 0 ? 1 : m * dof);
    for (std::size_t k = 0; k < m; ++k) {
      ids[k] = job->row_ids[k];
      const std::size_t src = dest->at(ids[k]);
      if (dof != 0) {
        std::memcpy(rows.data() + k * dof, positions->data() + src * dof, dof * sizeof(double));
      }
    }
    auto *id_raw = new std::int64_t[m == 0 ? 1 : m];
    auto *x_raw = new double[m * dof == 0 ? 1 : m * dof];
    if (m != 0) {
      std::memcpy(id_raw, ids.data(), m * sizeof(std::int64_t));
      if (dof != 0) {
        std::memcpy(x_raw, rows.data(), m * dof * sizeof(double));
      }
    }
    nb::capsule id_owner(id_raw, [](void *ptr) noexcept { delete[] static_cast<std::int64_t *>(ptr); });
    nb::capsule x_owner(x_raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
    const std::size_t id_shape[1] = {m};
    const std::size_t x_shape[2] = {m, dof};
    nb::ndarray<nb::numpy, std::int64_t> id_arr(id_raw, 1, id_shape, id_owner);
    nb::ndarray<nb::numpy, double> x_arr(x_raw, 2, x_shape, x_owner);
    nb::object result = (*callback)(id_arr, x_arr);
    if (!nb::isinstance<nb::tuple>(result)) {
      throw nb::value_error("eval callback must return (energies, gradients)");
    }
    nb::tuple tup = nb::cast<nb::tuple>(result);
    if (tup.size() != 2) {
      throw nb::value_error("eval callback must return (energies, gradients)");
    }
    auto e = nb::cast<nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(tup[0]);
    auto g = nb::cast<nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>>(tup[1]);
    if (e.shape(0) != m) {
      shape_error("energies", vec1(m), vec1(static_cast<std::size_t>(e.shape(0))));
    }
    if (g.shape(0) != m || g.shape(1) != dof) {
      shape_error("gradients", vec2(m, dof),
                  vec2(static_cast<std::size_t>(g.shape(0)), static_cast<std::size_t>(g.shape(1))));
    }
    for (std::size_t k = 0; k < m; ++k) {
      const std::size_t dst = dest->at(job->row_ids[k]);
      energies[dst] = e(k);
      if (dof != 0 && g.data() != nullptr) {
        std::memcpy(gradients + dst * dof, g.data() + k * dof, dof * sizeof(double));
      }
    }
  }
};

void run_jobs(nb::callable &callback, const std::vector<GroupWork> &jobs, const std::vector<double> &positions,
              std::size_t dof, const std::map<std::int64_t, std::size_t> &dest, double *energies, double *gradients,
              int threads, rgbind::CallContext &ctx) {
  if (jobs.empty()) {
    return;
  }
  const int nworkers = std::min(threads, static_cast<int>(jobs.size()));
  auto one = [&](const GroupWork &job, bool signals) {
    if (ctx.stop.load(std::memory_order_acquire)) {
      return;
    }
    EvalCall call;
    call.callback = &callback;
    call.job = &job;
    call.positions = &positions;
    call.dof = dof;
    call.dest = &dest;
    call.energies = energies;
    call.gradients = gradients;
    rgbind::trampoline<EvalCall> slot(call, &ctx, signals);
    const int rc = rgbind::trampoline<EvalCall>::entry(&slot);
    if (rc != static_cast<int>(rgbind::TrampolineStatus::ok) && !ctx.has_error()) {
      ctx.capture(std::make_exception_ptr(std::runtime_error("surface eval failed")));
    }
  };

  // threads == 1 stays on the calling thread so PyErr_CheckSignals runs
  // between callbacks (F5). More than one worker detaches first (F4).
  if (nworkers <= 1) {
    for (const GroupWork &job : jobs) {
      one(job, true);
      if (ctx.has_error()) {
        break;
      }
    }
    return;
  }

  std::vector<std::vector<std::size_t>> buckets(static_cast<std::size_t>(nworkers));
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    buckets[i % static_cast<std::size_t>(nworkers)].push_back(i);
  }
  {
    nb::gil_scoped_release rel;
    std::vector<std::thread> pool;
    pool.reserve(buckets.size());
    for (std::size_t worker = 0; worker < buckets.size(); ++worker) {
      if (buckets[worker].empty()) {
        continue;
      }
      pool.emplace_back([&, worker]() {
        for (std::size_t index : buckets[worker]) {
          one(jobs[index], false);
        }
      });
    }
    for (std::thread &worker : pool) {
      worker.join();
    }
  }
}

nb::tuple drive_eval(nb::callable callback, IdArray ids, Matrix x, OwnerArray owners, FlagArray reentrant,
                     int threads, ClaimSlot &slot) {
  const std::size_t n = static_cast<std::size_t>(ids.shape(0));
  const std::size_t x_rows = static_cast<std::size_t>(x.shape(0));
  const std::size_t dof = static_cast<std::size_t>(x.shape(1));
  if (x_rows != n) {
    shape_error("x", vec2(n, dof), vec2(x_rows, dof));
  }
  if (static_cast<std::size_t>(owners.shape(0)) != n) {
    shape_error("owners", vec1(n), vec1(static_cast<std::size_t>(owners.shape(0))));
  }
  if (static_cast<std::size_t>(reentrant.shape(0)) != n) {
    shape_error("reentrant", vec1(n), vec1(static_cast<std::size_t>(reentrant.shape(0))));
  }
  if (threads < 1) {
    throw nb::value_error("threads must be >= 1");
  }

  std::vector<rgbind::OwnerRow> rows;
  rows.reserve(n);
  std::map<std::int64_t, std::size_t> dest;
  std::vector<double> positions(n * dof == 0 ? 1 : n * dof);
  for (std::size_t i = 0; i < n; ++i) {
    rgbind::OwnerRow row;
    row.row_id = ids(i);
    row.owner = owners(i);
    row.reentrant = reentrant(i) != 0;
    if (!dest.emplace(row.row_id, i).second) {
      throw nb::value_error("rgbind::serialise_by_owner: duplicate row_id");
    }
    rows.push_back(row);
    if (dof != 0 && x.data() != nullptr) {
      std::memcpy(positions.data() + i * dof, x.data() + i * dof, dof * sizeof(double));
    }
  }

  rgbind::Schedule schedule;
  try {
    schedule = rgbind::serialise_by_owner(rows, threads);
  } catch (const std::invalid_argument &err) {
    throw nb::value_error(err.what());
  }

  std::vector<GroupWork> jobs;
  jobs.reserve(schedule.groups.size());
  for (const rgbind::OwnerGroup &group : schedule.groups) {
    GroupWork job;
    job.row_ids = group.row_ids;
    jobs.push_back(std::move(job));
  }

  auto energies = std::vector<double>(n == 0 ? 1 : n, 0.0);
  auto gradients = std::vector<double>(n * dof == 0 ? 1 : n * dof, 0.0);

  // F3: one adapter call at a time. The claim is an atomic flag, not a mutex.
  rgbind::Claim claim(slot.busy.flag(), "SurfaceAdapter");
  rgbind::CallContext ctx;
  run_jobs(callback, jobs, positions, dof, dest, energies.data(), gradients.data(), threads, ctx);
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (PyErr_CheckSignals() < 0) {
    throw nb::python_error();
  }

  auto *e_raw = new double[n == 0 ? 1 : n];
  auto *g_raw = new double[n * dof == 0 ? 1 : n * dof];
  if (n != 0) {
    std::memcpy(e_raw, energies.data(), n * sizeof(double));
  }
  if (n * dof != 0) {
    std::memcpy(g_raw, gradients.data(), n * dof * sizeof(double));
  }
  nb::capsule e_owner(e_raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  nb::capsule g_owner(g_raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  const std::size_t e_shape[1] = {n};
  const std::size_t g_shape[2] = {n, dof};
  nb::ndarray<nb::numpy, double> e_arr(e_raw, 1, e_shape, e_owner);
  nb::ndarray<nb::numpy, double> g_arr(g_raw, 2, g_shape, g_owner);
  return nb::make_tuple(e_arr, g_arr);
}

std::string schedule_canonical(IdArray ids, OwnerArray owners, FlagArray reentrant, int threads) {
  const std::size_t n = static_cast<std::size_t>(ids.shape(0));
  if (static_cast<std::size_t>(owners.shape(0)) != n) {
    shape_error("owners", vec1(n), vec1(static_cast<std::size_t>(owners.shape(0))));
  }
  if (static_cast<std::size_t>(reentrant.shape(0)) != n) {
    shape_error("reentrant", vec1(n), vec1(static_cast<std::size_t>(reentrant.shape(0))));
  }
  std::vector<rgbind::OwnerRow> rows;
  rows.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    rgbind::OwnerRow row;
    row.row_id = ids(i);
    row.owner = owners(i);
    row.reentrant = reentrant(i) != 0;
    rows.push_back(row);
  }
  try {
    return rgbind::serialise_by_owner(rows, threads).canonical();
  } catch (const std::invalid_argument &err) {
    throw nb::value_error(err.what());
  }
}

bool raise_status(int code) {
  std::string last = rgsaddle_last_error();
  if (code != RGSADDLE_OK && last.empty()) {
    const char *name = rgsaddle_status_name(static_cast<rgsaddle_status_t>(code));
    if (name != nullptr && name[0] != '\0') {
      rgsaddle_set_last_error(name);
      last = name;
    }
  }
  const rgbind::MappedStatus mapped =
      rgbind::map_status(static_cast<rgsaddle_status_t>(code), last, "SurfaceAdapter");
  return rgbind::apply_status(mapped, nullptr);
}

} // namespace

extern "C" const char *rgsaddle_last_error(void) { return g_last_error.c_str(); }

extern "C" void rgsaddle_set_last_error(const char *text) {
  g_last_error.assign(text == nullptr ? "" : text);
}

NB_MODULE(_core, m) {
  m.doc() = "rgsaddle nanobind surface: status mapping and the F5 eval loop.";
  m.attr("__version__") = RGSADDLE_PY_VERSION;
  m.def("version", []() { return std::string(RGSADDLE_PY_VERSION); }, "Package version string.");
  rgbind::install_policy(m);

  nb::class_<HoldGate>(m, "HoldGate")
      .def(nb::init<>())
      .def("release", &HoldGate::release, "Let the holding thread drop the claim.")
      .def("held", &HoldGate::held, "True once the holding thread owns the claim.");

  nb::class_<ClaimSlot>(m, "ClaimSlot")
      .def(nb::init<>())
      .def("work", &ClaimSlot::work, nb::arg("hold_ms") = 0,
           "Claim SurfaceAdapter, hold, and release (F3, F4).")
      .def("hold_until", &ClaimSlot::hold_until, nb::arg("gate"),
           "Claim and wait, with the GIL released, until gate.release().");

  m.def("last_error", []() { return std::string(rgsaddle_last_error()); },
        "Thread-local last-error text (F2, F7).");
  m.def("set_last_error", [](const std::string &text) { rgsaddle_set_last_error(text.c_str()); },
        nb::arg("text"), "Replace the thread-local last-error text.");
  m.def("raise_status", &raise_status, nb::arg("code"),
        "Map an rgsaddle_status_t value (F7). True on OK. False is not used: "
        "convergence is not a status code.");
  m.def("schedule_canonical", &schedule_canonical, nb::arg("ids").noconvert(), nb::arg("owners").noconvert(),
        nb::arg("reentrant").noconvert(), nb::arg("threads"),
        "Canonical F5 schedule. The worker count is not part of the text (F8).");
  m.def("echo_f64", &rgbind::echo_f64, nb::arg("values").noconvert(),
        "Byte-copy a float64 array. -0.0, NaN and subnormals pass through (F6).");
  m.def("drive_eval", &drive_eval, nb::arg("callback"), nb::arg("ids").noconvert(), nb::arg("x").noconvert(),
        nb::arg("owners").noconvert(), nb::arg("reentrant").noconvert(), nb::arg("threads"), nb::arg("slot"),
        "Fake stencil eval loop. Groups rows with the F5 scheduler and calls "
        "callback(ids, x) through the trampoline.");
}
