// MIT License
// Copyright 2026 Rohit Goswami

/**
 * @file ext.cpp
 * @brief Nanobind frontend for the rgsaddle Python package.
 *
 * Free-threaded builds use nanobind_add_module. The only mutable slot
 * is the thread-local last-error string, named with the rgsaddle_py_
 * prefix and kept inside this translation unit. ClaimSlot takes
 * rgbind::Claim for the whole SurfaceAdapter call. The test eval
 * loop drops the GIL before workers run. Python callbacks in that
 * loop go through rgbind::trampoline and rows are grouped by
 * rgbind::serialise_by_owner. Float64 C-contiguous arrays use
 * noconvert on the way in and capsule copies on the way out.
 * raise_status maps rgsaddle_status_t. The schedule text does not
 * depend on the worker count. Stencil is the point-search handle.
 * connect runs the double-ended search and releases the GIL around
 * rgsaddle_connect. The link exports PyInit__core.
 *
 * SurfaceAdapter does not own threads. The caller passes C ABI
 * request rows and the adapter evaluates them on that thread. The
 * test eval loop, the claim probes, and the float echo live in the
 * _testing submodule.
 */

#include <rgsaddle.h>
#include <rgsaddle/stencil.hpp>

#include <rgbind/rgbind.hpp>

#include "status.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

thread_local std::string rgsaddle_py_last_error_text;

const char *rgsaddle_py_last_error() { return rgsaddle_py_last_error_text.c_str(); }

void rgsaddle_py_set_last_error(const char *text) {
  rgsaddle_py_last_error_text.assign(text == nullptr ? "" : text);
}

struct HoldGate {
  std::atomic<bool> release_flag{false};
  std::atomic<bool> held_flag{false};

  void release() { release_flag.store(true, std::memory_order_release); }

  bool held() const { return held_flag.load(std::memory_order_acquire); }
};

struct ClaimSlot {
  rgbind::Busy busy;
  // The published pointer is the only shared write of the Claim object.
  // enter records this thread. exit from any other thread is refused.
  // The destructor releases a claim that is still held when Python frees the object.
  std::atomic<rgbind::Claim *> held{nullptr};
  std::atomic<std::thread::id> owner{};

  ClaimSlot() = default;
  ClaimSlot(const ClaimSlot &) = delete;
  ClaimSlot &operator=(const ClaimSlot &) = delete;
  ClaimSlot(ClaimSlot &&) = delete;
  ClaimSlot &operator=(ClaimSlot &&) = delete;

  void enter() {
    if (held.load(std::memory_order_acquire) != nullptr) {
      throw rgbind::InUse("SurfaceAdapter");
    }
    auto *claim = new rgbind::Claim(busy.flag(), "SurfaceAdapter");
    rgbind::Claim *expected = nullptr;
    if (!held.compare_exchange_strong(expected, claim, std::memory_order_acq_rel, std::memory_order_acquire)) {
      delete claim;
      throw rgbind::InUse("SurfaceAdapter");
    }
    owner.store(std::this_thread::get_id(), std::memory_order_release);
  }

  // Python may free the object on a thread that did not enter. Release
  // the claim here. exit() still refuses that thread while the object lives.
  ~ClaimSlot() {
    owner.store(std::thread::id(), std::memory_order_release);
    rgbind::Claim *claim = held.exchange(nullptr, std::memory_order_acq_rel);
    delete claim;
  }

  void exit() {
    const auto self = std::this_thread::get_id();
    if (owner.load(std::memory_order_acquire) != self) {
      throw std::runtime_error("ClaimSlot exit from a thread that does not hold the claim");
    }
    owner.store(std::thread::id(), std::memory_order_release);
    rgbind::Claim *claim = held.exchange(nullptr, std::memory_order_acq_rel);
    delete claim;
  }

  // Native hold. The GIL is released before the atomic claim.
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
using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Loose = nb::ndarray<nb::numpy, nb::device::cpu>;

[[noreturn]] void shape_error(const char *name, const std::string &expected, const std::string &got) {
  const std::string msg = std::string(name) + ": expected shape " + expected + ", got " + got;
  throw nb::value_error(msg.c_str());
}

[[noreturn]] void dtype_error(const char *name) {
  const std::string msg = std::string(name) + ": expected float64 C-contiguous";
  throw nb::type_error(msg.c_str());
}

std::string vec1(std::size_t n) { return "(" + std::to_string(n) + ",)"; }

std::string vec2(std::size_t rows, std::size_t cols) {
  return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

std::string shape_text(const Loose &arr) {
  std::string text = "(";
  for (std::size_t axis = 0; axis < arr.ndim(); ++axis) {
    if (axis != 0) {
      text += ", ";
    }
    text += std::to_string(arr.shape(axis));
  }
  if (arr.ndim() == 1) {
    text += ",";
  }
  text += ")";
  return text;
}

bool is_c_contiguous(const Loose &arr) {
  std::int64_t expect = 1;
  if (arr.stride_ptr() == nullptr) {
    return true;
  }
  for (std::size_t axis = arr.ndim(); axis-- > 0;) {
    const auto n = arr.shape(axis);
    if (n > 1 && arr.stride(axis) != expect) {
      return false;
    }
    if (n > 0) {
      expect *= static_cast<std::int64_t>(n);
    }
  }
  return true;
}

// NumPy's flags see Fortran order. A missing DLPack stride vector is defined
// to mean C order, so the stride walk alone can accept a Fortran array.
bool c_contiguous(nb::handle obj, const Loose &arr) {
  nb::object flags = nb::getattr(obj, "flags", nb::handle());
  if (flags.is_valid()) {
    nb::object bit = nb::getattr(flags, "c_contiguous", nb::handle());
    if (bit.is_valid()) {
      return nb::cast<bool>(bit);
    }
  }
  return is_c_contiguous(arr);
}

Loose loosen(nb::handle obj, const char *name) {
  try {
    return nb::cast<Loose>(obj);
  } catch (const nb::cast_error &) {
    dtype_error(name);
  }
}

nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> take_f64_1d(nb::handle obj,
                                                                                    const char *name,
                                                                                    std::size_t expect_n) {
  const Loose loose = loosen(obj, name);
  if (loose.dtype() != nb::dtype<double>() || !c_contiguous(obj, loose)) {
    dtype_error(name);
  }
  if (loose.ndim() != 1 || static_cast<std::size_t>(loose.shape(0)) != expect_n) {
    shape_error(name, vec1(expect_n), shape_text(loose));
  }
  return nb::cast<nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(obj);
}

nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>
take_f64_2d(nb::handle obj, const char *name, std::size_t expect_rows, std::size_t expect_cols) {
  const Loose loose = loosen(obj, name);
  if (loose.dtype() != nb::dtype<double>() || !c_contiguous(obj, loose)) {
    dtype_error(name);
  }
  if (loose.ndim() != 2 || static_cast<std::size_t>(loose.shape(0)) != expect_rows ||
      static_cast<std::size_t>(loose.shape(1)) != expect_cols) {
    shape_error(name, vec2(expect_rows, expect_cols), shape_text(loose));
  }
  return nb::cast<nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>>(obj);
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
    auto *id_raw = new std::int64_t[m == 0 ? 1 : m];
    auto *x_raw = new double[m * dof == 0 ? 1 : m * dof];
    for (std::size_t k = 0; k < m; ++k) {
      id_raw[k] = job->row_ids[k];
      const std::size_t src = dest->at(id_raw[k]);
      if (dof != 0) {
        std::memcpy(x_raw + k * dof, positions->data() + src * dof, dof * sizeof(double));
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
      throw nb::type_error("eval callback must return (energies, gradients)");
    }
    nb::tuple tup = nb::cast<nb::tuple>(result);
    if (tup.size() != 2) {
      throw nb::value_error("eval callback must return (energies, gradients)");
    }
    auto e = take_f64_1d(tup[0], "energies", m);
    auto g = take_f64_2d(tup[1], "gradients", m, dof);
    for (std::size_t k = 0; k < m; ++k) {
      const std::size_t dst = dest->at(job->row_ids[k]);
      energies[dst] = e(k);
      if (dof != 0 && g.data() != nullptr) {
        std::memcpy(gradients + dst * dof, g.data() + k * dof, dof * sizeof(double));
      }
    }
  }
};

void note_signal(rgbind::CallContext &ctx) {
  if (PyErr_CheckSignals() < 0) {
    ctx.stop.store(true, std::memory_order_release);
    if (!ctx.has_error()) {
      nb::python_error err;
      ctx.capture(err);
    }
  }
}

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
  // between callbacks. More than one worker detaches first.
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

  std::atomic<int> remaining{0};
  std::vector<std::thread> pool;
  pool.reserve(buckets.size());
  nb::gil_scoped_release rel;
  try {
    for (std::size_t worker = 0; worker < buckets.size(); ++worker) {
      if (buckets[worker].empty()) {
        continue;
      }
      remaining.fetch_add(1, std::memory_order_relaxed);
      try {
        pool.emplace_back([&, worker]() {
          for (std::size_t index : buckets[worker]) {
            one(jobs[index], false);
          }
          remaining.fetch_sub(1, std::memory_order_release);
        });
      } catch (...) {
        remaining.fetch_sub(1, std::memory_order_release);
        throw;
      }
    }
  } catch (...) {
    ctx.stop.store(true, std::memory_order_release);
    for (std::thread &worker : pool) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    throw;
  }

  while (remaining.load(std::memory_order_acquire) > 0) {
    {
      nb::gil_scoped_acquire gil;
      note_signal(ctx);
    }
    if (remaining.load(std::memory_order_acquire) == 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  for (std::thread &worker : pool) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

[[noreturn]] void raise_owner_error(const std::invalid_argument &err) {
  std::string text = err.what();
  const std::string prefix = "rgbind::serialise_by_owner: ";
  const auto at = text.find(prefix);
  if (at != std::string::npos) {
    text.erase(at, prefix.size());
  }
  throw nb::value_error(text.c_str());
}

nb::tuple drive_eval(nb::callable callback, IdArray ids, Matrix x, OwnerArray owners, FlagArray reentrant, int threads,
                     ClaimSlot &slot) {
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
      throw nb::value_error("duplicate row_id");
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
    raise_owner_error(err);
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

  // One eval at a time. The claim is an atomic flag, not a mutex.
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
    raise_owner_error(err);
  }
}

int preset_code(const std::string &name) {
  if (name == "min") {
    return RGSADDLE_STENCIL_MIN;
  }
  if (name == "dimer") {
    return RGSADDLE_STENCIL_DIMER;
  }
  if (name == "lanczos") {
    return RGSADDLE_STENCIL_LANCZOS;
  }
  if (name == "art") {
    return RGSADDLE_STENCIL_ART;
  }
  if (name == "eigen-follow") {
    return RGSADDLE_STENCIL_EIGEN;
  }
  if (name == "neb") {
    return RGSADDLE_STENCIL_NEB;
  }
  if (name == "dneb") {
    return RGSADDLE_STENCIL_DNEB;
  }
  if (name == "ci-neb") {
    return RGSADDLE_STENCIL_CI_NEB;
  }
  if (name == "energy-weighted-neb") {
    return RGSADDLE_STENCIL_ENERGY_WEIGHTED_NEB;
  }
  if (name == "zoom-neb") {
    return RGSADDLE_STENCIL_ZOOM_NEB;
  }
  if (name == "autoneb") {
    return RGSADDLE_STENCIL_AUTONEB;
  }
  if (name == "oci-neb") {
    return RGSADDLE_STENCIL_OCI_NEB;
  }
  if (name == "idpp") {
    return RGSADDLE_STENCIL_IDPP;
  }
  if (name == "s-idpp") {
    return RGSADDLE_STENCIL_SIDPP;
  }
  throw nb::value_error("unknown stencil preset");
}

void copy_callback_result(nb::object result, std::size_t rows, std::size_t dof, double *e, double *g) {
  if (!nb::isinstance<nb::tuple>(result)) {
    throw std::invalid_argument("stencil callback must return (energy, gradient)");
  }
  nb::tuple pair = nb::cast<nb::tuple>(result);
  if (pair.size() != 2) {
    throw std::invalid_argument("stencil callback must return (energy, gradient)");
  }
  auto energy = nb::cast<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>(pair[0]);
  auto gradient = nb::cast<nb::ndarray<const double, nb::ndim<2>, nb::c_contig>>(pair[1]);
  if (static_cast<std::size_t>(energy.shape(0)) != rows ||
      static_cast<std::size_t>(gradient.shape(0)) != rows ||
      static_cast<std::size_t>(gradient.shape(1)) != dof) {
    throw std::invalid_argument("stencil callback arrays do not match the position");
  }
  std::memcpy(e, energy.data(), rows * sizeof(double));
  std::memcpy(g, gradient.data(), rows * dof * sizeof(double));
}

void call_python_surface(const nb::callable &fn, std::size_t rows, std::size_t dof, const double *x,
                         double *e, double *g) {
  const std::size_t n = rows * dof == 0 ? 1 : rows * dof;
  auto *raw = new double[n];
  if (rows * dof != 0) {
    std::memcpy(raw, x, rows * dof * sizeof(double));
  }
  nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  const std::size_t shape[2] = {rows, dof};
  nb::ndarray<nb::numpy, double> arr(raw, 2, shape, owner);
  nb::object result = fn(arr);
  copy_callback_result(result, rows, dof, e, g);
}

struct PyStencil {
  rgsaddle::Stencil inner;
};

PyStencil make_point(nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x, const std::string &preset,
                     nb::handle mode, double dr, double force_tol, double max_move, double rotation_tol,
                     std::int64_t max_rotations, std::int64_t krylov_dim, std::int64_t threads,
                     double art_threshold, std::int64_t art_push) {
  if (x.shape(0) <= 0) {
    throw nb::value_error("a point stencil needs a position");
  }
  try {
    rgsaddle::PointBuilder builder(x.data(), static_cast<std::size_t>(x.shape(0)), preset_code(preset));
    if (!mode.is_none()) {
      auto values = nb::cast<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>(mode);
      builder.mode(values.data(), static_cast<std::size_t>(values.shape(0)));
    }
    builder.set_f64(RGSADDLE_STENCIL_KEY_DR, dr);
    builder.set_f64(RGSADDLE_STENCIL_KEY_FORCE_TOL, force_tol);
    builder.set_f64(RGSADDLE_STENCIL_KEY_MAX_MOVE, max_move);
    builder.set_f64(RGSADDLE_STENCIL_KEY_ROTATION_TOL, rotation_tol);
    builder.set_f64(RGSADDLE_STENCIL_KEY_ART_THRESHOLD, art_threshold);
    builder.set_i64(RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, max_rotations);
    builder.set_i64(RGSADDLE_STENCIL_KEY_KRYLOV_DIM, krylov_dim);
    builder.set_i64(RGSADDLE_STENCIL_KEY_THREADS, threads);
    builder.set_i64(RGSADDLE_STENCIL_KEY_ART_PUSH, art_push);
    return PyStencil{builder.build()};
  } catch (const rgsaddle::Error &err) {
    rgsaddle_py_set_last_error(err.what());
    throw nb::value_error(err.what());
  }
}

void make_band(nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x, const std::string &preset) {
  rgsaddle_stencil_config_t *config = rgsaddle_stencil_config_new(preset_code(preset));
  if (config == nullptr) {
    throw nb::value_error(rgsaddle_last_error());
  }
  const std::size_t rows = static_cast<std::size_t>(x.shape(0));
  const std::size_t dof = static_cast<std::size_t>(x.shape(1));
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(config, rows, dof, x.data());
  rgsaddle_stencil_config_free(config);
  if (handle != nullptr) {
    rgsaddle_stencil_free(handle);
  }
  const char *text = rgsaddle_last_error();
  rgsaddle_py_set_last_error(text == nullptr ? "" : text);
  throw nb::value_error(text == nullptr || text[0] == '\0' ? "band presets stay on feat/band-methods-zoom"
                                                           : text);
}

rgsaddle::StencilReport step_python(PyStencil &self, nb::callable fn, std::size_t max_steps, bool run_to_end) {
  rgbind::CallContext ctx;
  rgsaddle::StencilReport report;
  std::exception_ptr pending;
  {
    nb::gil_scoped_release release;
    try {
      auto call = [&](std::size_t rows, std::size_t dof, const std::uint64_t *, const double *x, double *e,
                      double *g) {
        nb::gil_scoped_acquire gil;
        if (!gil.is_valid()) {
          throw std::runtime_error("stencil callback could not attach");
        }
        try {
          call_python_surface(fn, rows, dof, x, e, g);
          if (PyErr_CheckSignals() < 0) {
            nb::python_error err;
            ctx.capture(err);
            throw std::runtime_error("callback");
          }
        } catch (nb::python_error &err) {
          ctx.capture(err);
          throw std::runtime_error("callback");
        }
      };
      report = run_to_end ? self.inner.run(call, max_steps) : self.inner.step(call);
    } catch (...) {
      pending = std::current_exception();
    }
  }
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (pending) {
    try {
      std::rethrow_exception(pending);
    } catch (const rgsaddle::Error &err) {
      rgsaddle_py_set_last_error(err.what());
      if (std::string(err.what()) == "Stencil is in use by another thread") {
        throw rgbind::InUse("Stencil");
      }
      throw nb::value_error(err.what());
    }
  }
  return report;
}

struct ConnectCallback {
  nb::callable *fn;
  rgbind::CallContext *ctx;
};

extern "C" rgsaddle_status_t connect_python_surface(void *user, rgsaddle_surface_request_t *req) {
  auto *state = static_cast<ConnectCallback *>(user);
  if (state == nullptr || req == nullptr || req->positions == nullptr || req->energies == nullptr ||
      req->gradients == nullptr || req->n_images <= 0 || req->n_atoms <= 0) {
    return RGSADDLE_INVALID_PARAMETER;
  }
  const std::size_t rows = static_cast<std::size_t>(req->n_images);
  const std::size_t dof = static_cast<std::size_t>(req->n_atoms) * 3;
  try {
    nb::gil_scoped_acquire gil;
    if (!gil.is_valid()) {
      return RGSADDLE_SURFACE_FAILED;
    }
    call_python_surface(*state->fn, rows, dof, req->positions, req->energies, req->gradients);
    if (PyErr_CheckSignals() < 0) {
      nb::python_error err;
      state->ctx->capture(err);
      return RGSADDLE_SURFACE_FAILED;
    }
    return RGSADDLE_OK;
  } catch (nb::python_error &err) {
    state->ctx->capture(err);
    return RGSADDLE_SURFACE_FAILED;
  } catch (const std::exception &err) {
    rgsaddle_py_set_last_error(err.what());
    return RGSADDLE_SURFACE_FAILED;
  }
}

struct ConnectReportPy {
  std::vector<double> position;
  double energy;
  double force_norm;
  std::string kind;
  int hessian_index;
  int evaluations;
};

ConnectReportPy connect_python(nb::callable fn, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> left,
                               nb::ndarray<const double, nb::ndim<1>, nb::c_contig> right, double force_tol,
                               std::int64_t max_calls, bool quench_nodes) {
  if (left.shape(0) <= 0 || left.shape(0) % 3 != 0 || left.shape(0) != right.shape(0)) {
    throw nb::value_error("connect endpoints must be the same positive multiple of three");
  }
  if (!(force_tol > 0.0) || !std::isfinite(force_tol) || max_calls <= 0) {
    throw nb::value_error("connect force_tol and max_calls must be positive");
  }
  const std::size_t dof = static_cast<std::size_t>(left.shape(0));
  rgsaddle_connect_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.flags = 0;
  config.force_tol = force_tol;
  config.max_calls = max_calls;
  config.quench_nodes = quench_nodes ? 1 : 0;
  std::vector<double> position(dof, 0.0);
  rgsaddle_connect_report_t report{};
  rgbind::CallContext ctx;
  ConnectCallback state{&fn, &ctx};
  rgsaddle_status_t status = RGSADDLE_OK;
  {
    nb::gil_scoped_release release;
    status = rgsaddle_connect(&config, static_cast<std::int64_t>(dof / 3), left.data(), right.data(),
                              connect_python_surface, &state, position.data(), &report);
  }
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (status != RGSADDLE_OK) {
    std::string last = rgsaddle_py_last_error();
    rgsaddle_py_set_last_error("");
    if (last.empty()) {
      const char *name = rgsaddle_status_name(status);
      last = name == nullptr || name[0] == '\0' ? "connect failed" : name;
    }
    throw nb::value_error(last.c_str());
  }
  std::string kind(1, static_cast<char>(report.kind));
  return ConnectReportPy{std::move(position), report.energy, report.force_norm, std::move(kind),
                         report.hessian_index, report.evaluations};
}

// IdppConfig::default in src/idpp.rs: 5000 L-BFGS steps, a gradient L2 of
// 1e-3 Angstrom^-3, and a per-atom step cap of 0.1 Angstrom.
constexpr std::int64_t kIdppMaxIterations = 5000;
constexpr double kIdppGradTol = 1e-3;
constexpr double kIdppMaxMove = 0.1;

nb::ndarray<nb::numpy, double> copy_matrix(const std::vector<double> &values, std::size_t rows,
                                           std::size_t cols);

int32_t projection_code(const std::string &name);

[[noreturn]] void status_error(rgsaddle_status_t status);

nb::ndarray<nb::numpy, double> idpp_path_py(Vector reactant, Vector product, std::int64_t n_images,
                                            nb::handle cell, std::int64_t max_iterations, double grad_tol,
                                            double max_move) {
  if (reactant.shape(0) <= 0 || reactant.shape(0) % 3 != 0 || reactant.shape(0) != product.shape(0)) {
    throw nb::value_error("idpp endpoints must be the same positive multiple of three");
  }
  if (n_images < 3) {
    throw nb::value_error("idpp path needs at least 3 images");
  }
  const auto n_atoms = static_cast<std::int64_t>(reactant.shape(0) / 3);
  std::vector<double> box;
  if (!cell.is_none()) {
    auto view = nb::cast<Matrix>(cell);
    if (view.shape(0) != 3 || view.shape(1) != 3) {
      throw nb::value_error("cell must be 3x3");
    }
    box.assign(view.data(), view.data() + 9);
  }
  const auto rows = static_cast<std::size_t>(n_images);
  const auto dof = static_cast<std::size_t>(reactant.shape(0));
  std::vector<double> out(rows * dof, 0.0);
  const auto status = rgsaddle_idpp_path(n_images, n_atoms, reactant.data(), product.data(),
                                         box.empty() ? nullptr : box.data(), max_iterations, grad_tol,
                                         max_move, out.data());
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return copy_matrix(out, rows, dof);
}

struct AutonebReportPy {
  int status = 0;
  double max_force = 0.0;
  std::int64_t ci_index = -1;
  std::int64_t surface_rows = 0;
  std::int64_t n_images = 0;
  std::int64_t iteration = 0;
  std::vector<double> positions;
  std::size_t dof = 0;
};

AutonebReportPy autoneb_run_py(nb::callable fn, Matrix positions, double k, bool climb,
                               std::int64_t n_simul, std::int64_t n_max, double force_tol,
                               double local_force_tol, std::int64_t local_max_steps,
                               std::int64_t climb_max_steps, const std::string &projection,
                               const std::string &tangent, bool remove_translation,
                               double space_energy_ratio, double max_move, bool idpp, nb::handle cell,
                               nb::handle fixed) {
  if (positions.shape(0) < 2 || positions.shape(1) <= 0 || positions.shape(1) % 3 != 0) {
    throw nb::value_error("positions must be (images >= 2, 3N)");
  }
  if (tangent != "simple" && tangent != "improved" && tangent != "improvedtangent") {
    throw nb::value_error("tangent must be improved, improvedtangent, or simple");
  }
  const auto n_init = static_cast<std::int64_t>(positions.shape(0));
  const auto dof = static_cast<std::size_t>(positions.shape(1));
  const auto n_atoms = static_cast<std::int64_t>(dof / 3);
  if (n_max < n_init) {
    throw nb::value_error("n_max must be at least the number of initial images");
  }
  if (static_cast<std::size_t>(n_max) > std::vector<double>().max_size() / dof) {
    throw nb::value_error("n_max images do not fit in memory");
  }
  std::vector<double> box;
  if (!cell.is_none()) {
    auto view = nb::cast<Matrix>(cell);
    if (view.shape(0) != 3 || view.shape(1) != 3) {
      throw nb::value_error("cell must be 3x3");
    }
    box.assign(view.data(), view.data() + 9);
  }
  std::vector<std::uint8_t> mask;
  if (!fixed.is_none()) {
    auto view =
        nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(fixed);
    if (static_cast<std::int64_t>(view.shape(0)) != n_atoms) {
      throw nb::value_error("fixed must hold one entry per atom");
    }
    mask.assign(view.data(), view.data() + view.shape(0));
  }
  rgsaddle_autoneb_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.flags = 0;
  if (climb) {
    config.flags |= RGSADDLE_AUTONEB_CLIMB;
  }
  if (idpp) {
    config.flags |= RGSADDLE_AUTONEB_IDPP;
  }
  if (!remove_translation) {
    config.flags |= RGSADDLE_AUTONEB_KEEP_TRANSLATION;
  }
  config.projection = projection_code(projection);
  config.tangent = tangent == "simple" ? RGSADDLE_TANGENT_SIMPLE : RGSADDLE_TANGENT_IMPROVED;
  config.n_simul = n_simul;
  config.n_max = n_max;
  config.local_max_steps = local_max_steps;
  config.climb_max_steps = climb_max_steps;
  config.k = k;
  config.force_tol = force_tol;
  config.local_force_tol = local_force_tol;
  config.max_move = max_move;
  config.space_energy_ratio = space_energy_ratio;
  std::vector<double> out(static_cast<std::size_t>(n_max) * dof, 0.0);
  rgsaddle_autoneb_report_t report{};
  rgbind::CallContext ctx;
  ConnectCallback state{&fn, &ctx};
  rgsaddle_status_t status = RGSADDLE_OK;
  {
    nb::gil_scoped_release release;
    status = rgsaddle_autoneb_run(&config, n_init, n_atoms, positions.data(),
                                  box.empty() ? nullptr : box.data(),
                                  mask.empty() ? nullptr : mask.data(), out.data(), &report,
                                  connect_python_surface, &state);
  }
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (status != RGSADDLE_OK) {
    std::string last = rgsaddle_py_last_error();
    rgsaddle_py_set_last_error("");
    if (last.empty()) {
      const char *name = rgsaddle_status_name(status);
      last = name == nullptr || name[0] == '\0' ? "autoneb failed" : name;
    }
    throw nb::value_error(last.c_str());
  }
  if (report.n_images > 0 && static_cast<std::size_t>(report.n_images) < out.size() / dof) {
    out.resize(static_cast<std::size_t>(report.n_images) * dof);
  }
  return AutonebReportPy{report.status, report.max_force, report.ci_index, report.surface_rows,
                         report.n_images, report.iteration, std::move(out), dof};
}

nb::ndarray<nb::numpy, double> copy_vector(const std::vector<double> &values) {
  const std::size_t n = values.empty() ? 1 : values.size();
  auto *raw = new double[n];
  if (!values.empty()) {
    std::memcpy(raw, values.data(), values.size() * sizeof(double));
  }
  nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  const std::size_t shape[1] = {values.size()};
  return nb::ndarray<nb::numpy, double>(raw, 1, shape, owner);
}

bool raise_status(int code) {
  // The status name is a local fallback. Storing it would make the next
  // call reuse this code's name, including after OK.
  std::string last = rgsaddle_py_last_error();
  rgsaddle_py_set_last_error("");
  if (code != RGSADDLE_OK && last.empty()) {
    const char *name = rgsaddle_status_name(static_cast<rgsaddle_status_t>(code));
    if (name != nullptr && name[0] != '\0') {
      last = name;
    }
  }
  const rgbind::MappedStatus mapped =
      rgbind::map_status(static_cast<rgsaddle_status_t>(code), last, "SurfaceAdapter");
  return rgbind::apply_status(mapped, nullptr);
}

[[noreturn]] void status_error(rgsaddle_status_t status) {
  const char *name = rgsaddle_status_name(status);
  throw nb::value_error(name == nullptr || name[0] == '\0' ? "rgsaddle call failed" : name);
}

struct SessionReport {
  int status = 0;
  double max_force = 0.0;
  double curvature = 0.0;
  std::int64_t iteration = 0;
  int evaluations = 0;
  std::int64_t ci_index = -1;
  std::int64_t rotations = 0;
  std::int64_t saddle_index = -1;
};

SessionReport report_from(const rgsaddle_report_t &report) {
  return SessionReport{report.status, report.max_force, report.curvature, report.iteration,
                       report.evaluations, report.ci_index, report.rotations, -1};
}

template <typename Call>
SessionReport run_surface(nb::callable &fn, Call &&call) {
  rgbind::CallContext ctx;
  ConnectCallback state{&fn, &ctx};
  rgsaddle_report_t report{};
  rgsaddle_status_t status = RGSADDLE_OK;
  {
    nb::gil_scoped_release release;
    status = call(connect_python_surface, static_cast<void *>(&state), &report);
  }
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (status != RGSADDLE_OK) {
    status_error(static_cast<rgsaddle_status_t>(status));
  }
  return report_from(report);
}

struct PyMinMode {
  RgsaddleMinMode *handle = nullptr;
  std::int64_t n_atoms = 0;

  PyMinMode() = default;
  PyMinMode(const PyMinMode &) = delete;
  PyMinMode &operator=(const PyMinMode &) = delete;
  PyMinMode(PyMinMode &&other) noexcept : handle(other.handle), n_atoms(other.n_atoms) {
    other.handle = nullptr;
  }
  PyMinMode &operator=(PyMinMode &&other) noexcept {
    if (this != &other) {
      if (handle != nullptr) {
        rgsaddle_minmode_free(handle);
      }
      handle = other.handle;
      n_atoms = other.n_atoms;
      other.handle = nullptr;
    }
    return *this;
  }
  ~PyMinMode() {
    if (handle != nullptr) {
      rgsaddle_minmode_free(handle);
    }
  }
};

struct SaddleIndexPy {
  bool certified = false;
  int index = 0;
  bool converged = false;
  std::vector<double> curvatures;
  std::int64_t evaluations = 0;
  std::int64_t total_evaluations = 0;
  std::int64_t escapes = 0;
};

SaddleIndexPy saddle_index_from(const rgsaddle_saddle_index_t &raw, std::vector<double> curvatures) {
  const auto resolved = static_cast<std::size_t>(std::max(raw.n_curvatures, 0));
  if (curvatures.size() > resolved) {
    curvatures.resize(resolved);
  }
  SaddleIndexPy out;
  out.certified = raw.certified != 0;
  out.index = raw.index;
  out.converged = raw.converged != 0;
  out.curvatures = std::move(curvatures);
  out.evaluations = raw.evaluations;
  out.total_evaluations = raw.total_evaluations;
  out.escapes = raw.escapes;
  return out;
}

std::int32_t saddle_count(std::int64_t value, const char *name) {
  if (value < 0 || value > INT32_MAX) {
    throw nb::value_error((std::string(name) + " must be a count that fits 32 bits").c_str());
  }
  return static_cast<std::int32_t>(value);
}

rgsaddle_saddle_check_t saddle_check_of(std::int64_t lowest, std::int64_t max_actions, double dr,
                                        double residual_tol, double negative_tol,
                                        std::int64_t escape_attempts, double escape_push) {
  rgsaddle_saddle_check_t check{};
  const auto status = rgsaddle_saddle_check_default(&check);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  check.lowest = saddle_count(lowest, "lowest");
  check.max_actions = saddle_count(max_actions, "max_actions");
  check.dr = dr;
  check.residual_tol = residual_tol;
  check.negative_tol = negative_tol;
  check.escape_attempts = saddle_count(escape_attempts, "escape_attempts");
  check.escape_push = escape_push;
  return check;
}

SaddleIndexPy minmode_saddle_index(const PyMinMode &self) {
  rgsaddle_saddle_index_t raw{};
  auto status = rgsaddle_minmode_saddle_index(self.handle, &raw, nullptr, 0);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  std::vector<double> curvatures(static_cast<std::size_t>(std::max(raw.n_curvatures, 0)));
  if (!curvatures.empty()) {
    status = rgsaddle_minmode_saddle_index(self.handle, &raw, curvatures.data(),
                                           static_cast<std::int64_t>(curvatures.size()));
    if (status != RGSADDLE_OK) {
      status_error(status);
    }
  }
  return saddle_index_from(raw, std::move(curvatures));
}

SessionReport with_saddle_index(SessionReport report, const PyMinMode &self) {
  rgsaddle_saddle_index_t raw{};
  if (rgsaddle_minmode_saddle_index(self.handle, &raw, nullptr, 0) == RGSADDLE_OK && raw.certified) {
    report.saddle_index = raw.index;
  }
  return report;
}

void minmode_set_saddle_check(PyMinMode &self, bool enabled, std::int64_t escape_attempts,
                              double escape_push, std::int64_t lowest, std::int64_t max_actions,
                              double dr, double residual_tol, double negative_tol) {
  rgsaddle_status_t status = RGSADDLE_OK;
  if (enabled) {
    const auto check = saddle_check_of(lowest, max_actions, dr, residual_tol, negative_tol,
                                       escape_attempts, escape_push);
    status = rgsaddle_minmode_set_saddle_check(self.handle, &check);
  } else {
    status = rgsaddle_minmode_set_saddle_check(self.handle, nullptr);
  }
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
}

SaddleIndexPy saddle_index_py(nb::callable fn, Vector x, nb::handle gradient, nb::handle mode,
                              nb::handle fixed, std::int64_t lowest, std::int64_t max_actions,
                              double dr, double residual_tol, double negative_tol) {
  if (x.shape(0) <= 0 || x.shape(0) % 3 != 0) {
    throw nb::value_error("position must hold a positive multiple of three values");
  }
  const auto dof = static_cast<std::int64_t>(x.shape(0));
  auto optional_vector = [&](nb::handle value, const char *name) -> std::vector<double> {
    if (value.is_none()) {
      return {};
    }
    auto view = nb::cast<Vector>(value);
    if (static_cast<std::int64_t>(view.shape(0)) != dof) {
      throw nb::value_error((std::string(name) + " length must match the position").c_str());
    }
    return std::vector<double>(view.data(), view.data() + view.shape(0));
  };
  const auto grad = optional_vector(gradient, "gradient");
  const auto seed = optional_vector(mode, "mode");
  std::vector<std::uint8_t> mask;
  if (!fixed.is_none()) {
    auto view = nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(fixed);
    if (static_cast<std::int64_t>(view.shape(0)) != dof / 3) {
      throw nb::value_error("fixed must hold one entry per atom");
    }
    mask.assign(view.data(), view.data() + view.shape(0));
  }
  const auto check = saddle_check_of(lowest, max_actions, dr, residual_tol, negative_tol, 0, 0.1);
  rgbind::CallContext ctx;
  ConnectCallback state{&fn, &ctx};
  rgsaddle_saddle_index_t raw{};
  std::vector<double> curvatures(static_cast<std::size_t>(dof));
  rgsaddle_status_t status = RGSADDLE_OK;
  {
    nb::gil_scoped_release release;
    status = rgsaddle_saddle_index(&check, dof / 3, x.data(), grad.empty() ? nullptr : grad.data(),
                                   seed.empty() ? nullptr : seed.data(),
                                   mask.empty() ? nullptr : mask.data(), connect_python_surface,
                                   &state, &raw, curvatures.data(), dof);
  }
  if (ctx.has_error()) {
    ctx.rethrow_if_set();
  }
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return saddle_index_from(raw, std::move(curvatures));
}

struct PySella {
  RgsaddleSellaSaddle *handle = nullptr;
  std::int64_t n_atoms = 0;

  PySella() = default;
  PySella(const PySella &) = delete;
  PySella &operator=(const PySella &) = delete;
  PySella(PySella &&other) noexcept : handle(other.handle), n_atoms(other.n_atoms) {
    other.handle = nullptr;
  }
  PySella &operator=(PySella &&other) noexcept {
    if (this != &other) {
      if (handle != nullptr) {
        rgsaddle_sella_saddle_free(handle);
      }
      handle = other.handle;
      n_atoms = other.n_atoms;
      other.handle = nullptr;
    }
    return *this;
  }
  ~PySella() {
    if (handle != nullptr) {
      rgsaddle_sella_saddle_free(handle);
    }
  }
};

nb::ndarray<nb::numpy, double> copy_matrix(const std::vector<double> &values, std::size_t rows,
                                           std::size_t cols) {
  const std::size_t n = rows * cols == 0 ? 1 : rows * cols;
  auto *raw = new double[n];
  if (!values.empty() && rows * cols != 0) {
    std::memcpy(raw, values.data(), rows * cols * sizeof(double));
  }
  nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  const std::size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, double>(raw, 2, shape, owner);
}

std::int32_t reparam_code(const std::string &scheme) {
  if (scheme == "linear") {
    return RGSADDLE_REPARAM_LINEAR;
  }
  if (scheme == "not-a-knot" || scheme == "spline") {
    return RGSADDLE_REPARAM_NOT_A_KNOT;
  }
  throw nb::value_error("scheme must be linear, not-a-knot, or spline");
}

std::int32_t band_method_code(const std::string &method) {
  if (method == "fire") {
    return RGSADDLE_METHOD_FIRE;
  }
  if (method == "lbfgs") {
    return RGSADDLE_METHOD_LBFGS;
  }
  if (method == "rtr") {
    return RGSADDLE_METHOD_RTR;
  }
  if (method == "quickmin") {
    return RGSADDLE_METHOD_QUICKMIN;
  }
  if (method == "ode12r") {
    return RGSADDLE_METHOD_ODE12R;
  }
  throw nb::value_error("method must be fire, lbfgs, quickmin, ode12r, or rtr");
}

struct PyBand {
  RgsaddleBand *handle = nullptr;
  std::int64_t n_images = 0;
  std::int64_t n_atoms = 0;

  PyBand() = default;
  PyBand(const PyBand &) = delete;
  PyBand &operator=(const PyBand &) = delete;
  PyBand(PyBand &&other) noexcept
      : handle(other.handle), n_images(other.n_images), n_atoms(other.n_atoms) {
    other.handle = nullptr;
  }
  PyBand &operator=(PyBand &&other) noexcept {
    if (this != &other) {
      if (handle != nullptr) {
        rgsaddle_band_free(handle);
      }
      handle = other.handle;
      n_images = other.n_images;
      n_atoms = other.n_atoms;
      other.handle = nullptr;
    }
    return *this;
  }
  ~PyBand() {
    if (handle != nullptr) {
      rgsaddle_band_free(handle);
    }
  }
};

SaddleIndexPy band_saddle_index(const PyBand &self) {
  rgsaddle_saddle_index_t raw{};
  auto status = rgsaddle_band_saddle_index(self.handle, &raw, nullptr, 0);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  std::vector<double> curvatures(static_cast<std::size_t>(std::max(raw.n_curvatures, 0)));
  if (!curvatures.empty()) {
    status = rgsaddle_band_saddle_index(self.handle, &raw, curvatures.data(),
                                        static_cast<std::int64_t>(curvatures.size()));
    if (status != RGSADDLE_OK) {
      status_error(status);
    }
  }
  return saddle_index_from(raw, std::move(curvatures));
}

SessionReport with_band_saddle_index(SessionReport report, const PyBand &self) {
  rgsaddle_saddle_index_t raw{};
  if (rgsaddle_band_saddle_index(self.handle, &raw, nullptr, 0) == RGSADDLE_OK && raw.certified) {
    report.saddle_index = raw.index;
  }
  return report;
}

void band_set_saddle_check(PyBand &self, bool enabled, std::int64_t escape_attempts,
                           double escape_push, std::int64_t lowest, std::int64_t max_actions,
                           double dr, double residual_tol, double negative_tol) {
  rgsaddle_status_t status = RGSADDLE_OK;
  if (enabled) {
    const auto check = saddle_check_of(lowest, max_actions, dr, residual_tol, negative_tol,
                                       escape_attempts, escape_push);
    status = rgsaddle_band_set_saddle_check(self.handle, &check);
  } else {
    status = rgsaddle_band_set_saddle_check(self.handle, nullptr);
  }
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
}

int32_t projection_code(const std::string &name) {
  if (name == "neb" || name == "improvedtangent" || name == "aseneb" || name == "string") {
    return 1;
  }
  if (name == "eb") {
    return 0;
  }
  if (name == "dneb") {
    return 2;
  }
  throw nb::value_error("projection must be neb, improvedtangent, aseneb, eb, dneb, or string");
}

nb::tuple project_band(Matrix positions, Vector energies, Matrix gradients, double k, bool climb,
                       const std::string &projection, const std::string &tangent,
                       bool remove_translation, double force_tol, nb::handle fixed) {
  if (positions.shape(0) < 3 || positions.shape(1) <= 0 || positions.shape(1) % 3 != 0) {
    throw nb::value_error("positions must be (images >= 3, 3N)");
  }
  const auto rows = static_cast<std::int64_t>(positions.shape(0));
  const auto dof = static_cast<std::int64_t>(positions.shape(1));
  if (energies.shape(0) != positions.shape(0) || gradients.shape(0) != positions.shape(0) ||
      gradients.shape(1) != positions.shape(1)) {
    throw nb::value_error("energies and gradients must match positions");
  }
  if (!(k == k) || !(force_tol >= 0.0) || !(force_tol == force_tol)) {
    throw nb::value_error("spring constant must be finite and force_tol non-negative");
  }
  rgsaddle_band_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.flags = RGSADDLE_BAND_NO_MIN_MODE;
  if (!remove_translation) {
    config.flags |= RGSADDLE_BAND_KEEP_TRANSLATION;
  }
  if (projection == "string") {
    config.flags |= RGSADDLE_BAND_STRING;
  }
  config.tangent = tangent == "simple" ? 0 : 1;
  if (tangent != "simple" && tangent != "improved" && tangent != "improvedtangent") {
    throw nb::value_error("tangent must be improved, improvedtangent, or simple");
  }
  config.spring = 0;
  config.projection = projection_code(projection);
  config.method = RGSADDLE_METHOD_FIRE;
  config.spring_k = k;
  config.ci_trigger_factor = climb ? 1.0 : 0.0;
  config.force_tol = force_tol;
  config.max_move = 0.2;
  std::vector<std::uint8_t> mask;
  if (!fixed.is_none()) {
    auto view = nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(fixed);
    if (static_cast<std::int64_t>(view.shape(0)) != dof / 3) {
      throw nb::value_error("fixed must hold one entry per atom");
    }
    mask.assign(view.data(), view.data() + view.shape(0));
  }
  std::vector<double> forces(static_cast<std::size_t>(rows * dof), 0.0);
  std::int64_t ci = -1;
  const auto status =
      rgsaddle_band_project_fixed(&config, rows, dof / 3, positions.data(), energies.data(), gradients.data(),
                                  mask.empty() ? nullptr : mask.data(), forces.data(), &ci);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return nb::make_tuple(copy_vector(forces), ci);
}

nb::ndarray<nb::numpy, double> reparametrize_py(Matrix positions, const std::string &scheme,
                                                nb::handle anchor, nb::handle cell) {
  if (positions.shape(0) < 2 || positions.shape(1) <= 0 || positions.shape(1) % 3 != 0) {
    throw nb::value_error("positions must be (images >= 2, 3N)");
  }
  std::int32_t code = reparam_code(scheme);
  const std::size_t rows = static_cast<std::size_t>(positions.shape(0));
  const std::size_t dof = static_cast<std::size_t>(positions.shape(1));
  std::int64_t index = -1;
  if (!anchor.is_none()) {
    index = nb::cast<std::int64_t>(anchor);
    if (index <= 0 || index >= static_cast<std::int64_t>(rows) - 1) {
      throw nb::value_error("anchor must name an interior image");
    }
  }
  std::vector<double> box;
  if (!cell.is_none()) {
    auto view = nb::cast<Matrix>(cell);
    if (view.shape(0) != 3 || view.shape(1) != 3) {
      throw nb::value_error("cell must be 3x3");
    }
    box.assign(view.data(), view.data() + 9);
  }
  std::vector<double> out(positions.data(), positions.data() + rows * dof);
  const auto status = rgsaddle_reparametrize(static_cast<std::int64_t>(rows), static_cast<std::int64_t>(dof / 3),
                                             out.data(), index, box.empty() ? nullptr : box.data(), code);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  auto *raw = new double[rows * dof];
  std::memcpy(raw, out.data(), rows * dof * sizeof(double));
  nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
  const std::size_t shape[2] = {rows, dof};
  return nb::ndarray<nb::numpy, double>(raw, 2, shape, owner);
}

nb::tuple halve_raised_energy_py(double previous, double trial, double step) {
  int32_t raised = 0;
  double next = 0.0;
  const auto status = rgsaddle_halve_raised_energy(previous, trial, step, &raised, &next);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return nb::make_tuple(raised != 0, next);
}

double trust_update_py(double radius, double predicted, double actual, double step_norm,
                       double sigma_inc, double sigma_dec, double rho_inc, double rho_dec,
                       double delta_min) {
  double out = 0.0;
  const auto status =
      rgsaddle_trust_update(radius, predicted, actual, step_norm, sigma_inc, sigma_dec, rho_inc,
                            rho_dec, delta_min, &out);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return out;
}

std::int32_t minmode_method_code(const std::string &method) {
  if (method == "fire") {
    return RGSADDLE_METHOD_FIRE;
  }
  if (method == "lbfgs") {
    return RGSADDLE_METHOD_LBFGS;
  }
  throw nb::value_error("method must be fire or lbfgs");
}

PyMinMode make_minmode(Vector x, Vector mode, const std::string &kind, double dr, double force_tol,
                       double max_move, double rotation_tol, std::int64_t max_rotations,
                       std::int64_t krylov_dim, double rotation_stop_length,
                       const std::string &method) {
  if (x.shape(0) <= 0 || x.shape(0) % 3 != 0 || x.shape(0) != mode.shape(0)) {
    throw nb::value_error("position and mode must share a positive multiple of three");
  }
  rgsaddle_minmode_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.kind = kind == "lanczos" ? RGSADDLE_MINMODE_LANCZOS : RGSADDLE_MINMODE_DIMER;
  if (kind != "dimer" && kind != "lanczos") {
    throw nb::value_error("kind must be dimer or lanczos");
  }
  config.method = minmode_method_code(method);
  config.dr = dr;
  config.rotation_tol = rotation_tol;
  config.max_rotations = max_rotations;
  config.krylov_dim = krylov_dim;
  config.force_tol = force_tol;
  config.max_move = max_move;
  config.rotation_angle_tol = 5.0 * 3.14159265358979323846 / 180.0;
  PyMinMode out;
  out.n_atoms = static_cast<std::int64_t>(x.shape(0) / 3);
  out.handle = rgsaddle_minmode_create(&config, out.n_atoms, x.data(), mode.data());
  if (out.handle == nullptr) {
    throw nb::value_error("minimum-mode configuration was refused");
  }
  if (rgsaddle_minmode_set_rotation_stop_length(out.handle, rotation_stop_length) != RGSADDLE_OK) {
    throw nb::value_error("rotation_stop_length must be finite and non-negative");
  }
  return out;
}

void minmode_set_position(PyMinMode &self, Vector x, nb::handle gradient) {
  if (self.handle == nullptr || x.shape(0) != self.n_atoms * 3) {
    throw nb::value_error("position length must stay 3N");
  }
  const double *grad = nullptr;
  if (!gradient.is_none()) {
    auto values = nb::cast<Vector>(gradient);
    if (values.shape(0) != x.shape(0)) {
      throw nb::value_error("gradient length must match the position");
    }
    grad = values.data();
  }
  const auto status = rgsaddle_minmode_set_position(self.handle, x.data(), grad);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
}

PySella make_sella(Vector x, Vector masses, std::int64_t order, double delta, double force_tol) {
  if (x.shape(0) <= 0 || x.shape(0) % 3 != 0 || masses.shape(0) * 3 != x.shape(0)) {
    throw nb::value_error("position is 3N and masses has one entry per atom");
  }
  rgsaddle_sella_saddle_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.delta = delta;
  config.force_tol = force_tol;
  config.force_gate = RGSADDLE_FORCE_MAX_ATOM;
  config.order = order;
  PySella out;
  out.n_atoms = static_cast<std::int64_t>(masses.shape(0));
  out.handle = rgsaddle_sella_saddle_create(&config, out.n_atoms, x.data(), masses.data());
  if (out.handle == nullptr) {
    throw nb::value_error("Sella saddle configuration was refused");
  }
  return out;
}

struct HyperplaneTiContext {
  nb::callable sampler;
  std::string error;
};

std::vector<double> copy_pointer(const double *values, std::size_t count) {
  return std::vector<double>(values, values + count);
}

extern "C" std::int32_t hyperplane_sample_python(
    void *raw, const double *point, const double *normal, std::size_t dimension,
    const rgsaddle_hyperplane_budget_t *budget, rgsaddle_hyperplane_emit_fn emit, void *emit_context) {
  auto *context = static_cast<HyperplaneTiContext *>(raw);
  try {
    nb::object output =
        context->sampler(copy_vector(copy_pointer(point, dimension)), copy_vector(copy_pointer(normal, dimension)),
                         budget->lambda, budget->thermal, budget->seed, budget->thermalize, budget->stride,
                         budget->samples);
    const auto values =
        take_f64_1d(output, "hyperplane sampler result", static_cast<std::size_t>(budget->samples));
    for (std::int64_t index = 0; index < budget->samples; ++index) {
      rgsaddle_hyperplane_observation_t observation{};
      observation.constraint_force = 0.0;
      observation.thermodynamic_derivative = values(static_cast<std::size_t>(index));
      const auto status = emit(emit_context, &observation);
      if (status != RGSADDLE_OK) {
        return status;
      }
    }
    return RGSADDLE_OK;
  } catch (const std::exception &error) {
    context->error = error.what();
    return RGSADDLE_SURFACE_FAILED;
  }
}

nb::tuple hyperplane_ti_python(nb::callable sampler, Vector point, Vector normal, double thermal,
                               std::int64_t nodes, std::int64_t thermalize, std::int64_t stride,
                               std::int64_t samples, std::int64_t blocks, std::uint64_t seed) {
  if (point.shape(0) <= 0 || normal.shape(0) != point.shape(0)) {
    throw nb::value_error("point and normal need one nonzero length");
  }
  HyperplaneTiContext context{std::move(sampler), ""};
  auto *handle = rgsaddle_hyperplane_sampler_create(&context, hyperplane_sample_python);
  if (handle == nullptr) {
    throw std::runtime_error("could not create the hyperplane sampler");
  }
  rgsaddle_hyperplane_ti_config_t config{};
  config.version = RGSADDLE_VERSION_INIT;
  config.thermal = thermal;
  config.nodes = nodes;
  config.thermalize = thermalize;
  config.stride = stride;
  config.samples = samples;
  config.blocks = blocks;
  config.seed = seed;
  rgsaddle_hyperplane_ti_result_t result{};
  const auto status = rgsaddle_hyperplane_thermodynamic_integration(
      handle, point.data(), normal.data(), static_cast<std::size_t>(point.shape(0)), &config, &result);
  rgsaddle_hyperplane_sampler_free(handle);
  if (status != RGSADDLE_OK) {
    if (!context.error.empty()) {
      throw std::runtime_error(context.error);
    }
    status_error(static_cast<rgsaddle_status_t>(status));
  }
  return nb::make_tuple(result.free_energy, result.error, result.evaluations);
}

PyBand make_band_session(Matrix positions, double k, bool climb, double ci_after,
                         double ci_after_rel, const std::string &projection,
                         const std::string &tangent, const std::string &method,
                         std::int64_t memory, double force_tol, double max_move, double time_step,
                         bool remove_translation, bool min_mode, bool string_mode,
                         const std::string &reparametrization, nb::handle cell, nb::handle fixed) {
  if (positions.shape(0) < 3 || positions.shape(1) <= 0 || positions.shape(1) % 3 != 0) {
    throw nb::value_error("positions must be (images >= 3, 3N)");
  }
  if (tangent != "simple" && tangent != "improved" && tangent != "improvedtangent") {
    throw nb::value_error("tangent must be improved, improvedtangent, or simple");
  }
  const auto scheme = reparam_code(reparametrization);
  const auto mcode = band_method_code(method);
  const auto pcode = projection_code(projection);
  std::vector<double> box;
  if (!cell.is_none()) {
    auto view = nb::cast<Matrix>(cell);
    if (view.shape(0) != 3 || view.shape(1) != 3) {
      throw nb::value_error("cell must be 3x3");
    }
    box.assign(view.data(), view.data() + 9);
  }
  const auto rows = static_cast<std::int64_t>(positions.shape(0));
  const auto n_atoms = static_cast<std::int64_t>(positions.shape(1) / 3);
  std::vector<std::uint8_t> mask;
  if (!fixed.is_none()) {
    auto view =
        nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(fixed);
    if (static_cast<std::int64_t>(view.shape(0)) != n_atoms) {
      throw nb::value_error("fixed must hold one entry per atom");
    }
    mask.assign(view.data(), view.data() + view.shape(0));
  }

  rgsaddle_band_config_t config{};
  config.version.major = RGSADDLE_ABI_MAJOR;
  config.version.minor = RGSADDLE_ABI_MINOR;
  config.flags = 0;
  if (!remove_translation) {
    config.flags |= RGSADDLE_BAND_KEEP_TRANSLATION;
  }
  if (!min_mode) {
    config.flags |= RGSADDLE_BAND_NO_MIN_MODE;
  }
  if (climb) {
    config.flags |= RGSADDLE_BAND_CI_AFTER;
    config.ci_trigger_factor = ci_after_rel;
    config.ci_trigger_force = ci_after;
  }
  if (scheme == RGSADDLE_REPARAM_NOT_A_KNOT) {
    config.flags |= RGSADDLE_BAND_SPLINE_REPARAM;
  }
  if (string_mode || projection == "string") {
    config.flags |= RGSADDLE_BAND_STRING;
  }
  config.tangent = tangent == "simple" ? RGSADDLE_TANGENT_SIMPLE : RGSADDLE_TANGENT_IMPROVED;
  config.spring = RGSADDLE_SPRING_UNIFORM;
  config.projection = pcode;
  config.method = mcode;
  config.spring_k = k;
  config.cell = box.empty() ? nullptr : box.data();
  config.force_tol = force_tol;
  config.max_move = max_move;
  config.memory = memory;

  PyBand out;
  out.n_images = rows;
  out.n_atoms = n_atoms;
  out.handle = rgsaddle_band_create(&config, rows, n_atoms, positions.data());
  if (out.handle == nullptr) {
    throw nb::value_error("band configuration was refused");
  }
  if (time_step != 1.0) {
    const auto st = rgsaddle_band_set_time_step(out.handle, time_step);
    if (st != RGSADDLE_OK) {
      status_error(st);
    }
  }
  if (!mask.empty()) {
    const auto st = rgsaddle_band_set_fixed_atoms(out.handle, mask.data(), n_atoms);
    if (st != RGSADDLE_OK) {
      status_error(st);
    }
  }
  return out;
}

SessionReport band_run_py(PyBand &self, nb::callable fn, std::int64_t max_steps) {
  if (max_steps < 1) {
    throw nb::value_error("max_steps must be positive");
  }
  SessionReport last;
  for (std::int64_t step = 0; step < max_steps; ++step) {
    last = with_band_saddle_index(
        run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
          return rgsaddle_band_step(self.handle, surface, user, out);
        }),
        self);
    if (last.status == RGSADDLE_STATUS_CONVERGED) {
      break;
    }
  }
  return last;
}

nb::tuple band_evaluation_py(const PyBand &self) {
  const auto rows = static_cast<std::size_t>(self.n_images);
  const auto dof = static_cast<std::size_t>(self.n_atoms * 3);
  const auto interior = rows - 2;
  std::vector<double> energies(rows, 0.0);
  std::vector<double> gradients(rows * dof, 0.0);
  std::vector<double> projected(interior * dof, 0.0);
  const auto status =
      rgsaddle_band_evaluation(self.handle, energies.data(), gradients.data(), projected.data());
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  return nb::make_tuple(copy_vector(energies), copy_matrix(gradients, rows, dof),
                        copy_matrix(projected, interior, dof));
}

struct PyGrowingString {
  RgsaddleGrowingString *handle = nullptr;
  std::int64_t n_images = 0;
  std::int64_t n_atoms = 0;

  PyGrowingString() = default;
  PyGrowingString(const PyGrowingString &) = delete;
  PyGrowingString &operator=(const PyGrowingString &) = delete;
  PyGrowingString(PyGrowingString &&other) noexcept
      : handle(other.handle), n_images(other.n_images), n_atoms(other.n_atoms) {
    other.handle = nullptr;
  }
  PyGrowingString &operator=(PyGrowingString &&other) noexcept {
    if (this != &other) {
      if (handle != nullptr) {
        rgsaddle_growing_string_free(handle);
      }
      handle = other.handle;
      n_images = other.n_images;
      n_atoms = other.n_atoms;
      other.handle = nullptr;
    }
    return *this;
  }
  ~PyGrowingString() {
    if (handle != nullptr) {
      rgsaddle_growing_string_free(handle);
    }
  }
};

SaddleIndexPy growing_string_saddle_index(const PyGrowingString &self) {
  rgsaddle_saddle_index_t raw{};
  auto status = rgsaddle_growing_string_saddle_index(self.handle, &raw, nullptr, 0);
  if (status != RGSADDLE_OK) {
    status_error(status);
  }
  std::vector<double> curvatures(static_cast<std::size_t>(std::max(raw.n_curvatures, 0)));
  if (!curvatures.empty()) {
    status = rgsaddle_growing_string_saddle_index(self.handle, &raw, curvatures.data(),
                                                  static_cast<std::int64_t>(curvatures.size()));
    if (status != RGSADDLE_OK) {
      status_error(status);
    }
  }
  return saddle_index_from(raw, std::move(curvatures));
}

SessionReport with_growing_string_saddle_index(SessionReport report, const PyGrowingString &self) {
  rgsaddle_saddle_index_t raw{};
  if (rgsaddle_growing_string_saddle_index(self.handle, &raw, nullptr, 0) == RGSADDLE_OK &&
      raw.certified) {
    report.saddle_index = raw.index;
  }
  return report;
}

PyGrowingString make_growing_string_session(Vector reactant, Vector product, std::int64_t n_images,
                                            double frontier_tol, std::int64_t min_steps_per_node,
                                            std::int64_t max_steps_per_node,
                                            std::optional<double> join_tol,
                                            bool climb, double ci_after, double ci_after_rel,
                                            const std::string &method, std::int64_t memory,
                                            double force_tol, double max_move,
                                            bool remove_translation, bool min_mode,
                                            const std::string &reparametrization, nb::handle cell,
                                            nb::handle fixed) {
  if (reactant.shape(0) <= 0 || reactant.shape(0) % 3 != 0 ||
      reactant.shape(0) != product.shape(0)) {
    throw nb::value_error("reactant and product must share a positive multiple of three");
  }
  if (n_images < 4) {
    throw nb::value_error("n_images must be at least 4");
  }
  const auto scheme = reparam_code(reparametrization);
  const auto mcode = band_method_code(method);
  std::vector<double> box;
  if (!cell.is_none()) {
    auto view = nb::cast<Matrix>(cell);
    if (view.shape(0) != 3 || view.shape(1) != 3) {
      throw nb::value_error("cell must be 3x3");
    }
    box.assign(view.data(), view.data() + 9);
  }
  const auto n_atoms = static_cast<std::int64_t>(reactant.shape(0) / 3);
  std::vector<std::uint8_t> mask;
  if (!fixed.is_none()) {
    auto view =
        nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(fixed);
    if (static_cast<std::int64_t>(view.shape(0)) != n_atoms) {
      throw nb::value_error("fixed must hold one entry per atom");
    }
    mask.assign(view.data(), view.data() + view.shape(0));
  }

  rgsaddle_growing_string_config_t config{};
  if (rgsaddle_growing_string_config_default(&config) != RGSADDLE_OK) {
    throw nb::value_error("growing string default configuration failed");
  }
  config.band.flags = RGSADDLE_BAND_STRING;
  if (!remove_translation) {
    config.band.flags |= RGSADDLE_BAND_KEEP_TRANSLATION;
  }
  if (!min_mode) {
    config.band.flags |= RGSADDLE_BAND_NO_MIN_MODE;
  }
  if (climb) {
    config.band.flags |= RGSADDLE_BAND_CI_AFTER;
    config.band.ci_trigger_factor = ci_after_rel;
    config.band.ci_trigger_force = ci_after;
  } else {
    config.band.ci_trigger_factor = 0.0;
    config.band.ci_trigger_force = 0.0;
  }
  if (scheme == RGSADDLE_REPARAM_NOT_A_KNOT) {
    config.band.flags |= RGSADDLE_BAND_SPLINE_REPARAM;
  }
  config.band.method = mcode;
  config.band.cell = box.empty() ? nullptr : box.data();
  config.band.force_tol = force_tol;
  config.band.max_move = max_move;
  config.band.memory = memory;
  config.n_images = n_images;
  config.frontier_tol = frontier_tol;
  config.min_steps_per_node = min_steps_per_node;
  config.max_steps_per_node = max_steps_per_node;
  if (join_tol.has_value()) {
    const auto value = *join_tol;
    if (!std::isfinite(value) || value < 0.0) {
      throw nb::value_error("join_tol must be non-negative or None");
    }
    config.join_tol = value;
  }
  config.reparametrization = scheme;

  PyGrowingString out;
  out.n_images = n_images;
  out.n_atoms = n_atoms;
  out.handle = rgsaddle_growing_string_create(&config, n_atoms, reactant.data(), product.data());
  if (out.handle == nullptr) {
    throw nb::value_error("growing string configuration was refused");
  }
  if (!mask.empty()) {
    const auto st = rgsaddle_growing_string_set_fixed_atoms(out.handle, mask.data(), n_atoms);
    if (st != RGSADDLE_OK) {
      status_error(st);
    }
  }
  return out;
}

SessionReport growing_string_run_py(PyGrowingString &self, nb::callable fn,
                                    std::int64_t max_steps) {
  if (max_steps < 1) {
    throw nb::value_error("max_steps must be positive");
  }
  SessionReport last;
  for (std::int64_t step = 0; step < max_steps; ++step) {
    last = with_growing_string_saddle_index(
        run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
          return rgsaddle_growing_string_step(self.handle, surface, user, out);
        }),
        self);
    if (last.status == RGSADDLE_STATUS_CONVERGED) {
      break;
    }
  }
  return last;
}

nb::ndarray<nb::numpy, std::uint64_t> copy_u64_vector(const std::vector<std::uint64_t> &values) {
  const std::size_t n = values.empty() ? 1 : values.size();
  auto *raw = new std::uint64_t[n];
  if (!values.empty()) {
    std::memcpy(raw, values.data(), values.size() * sizeof(std::uint64_t));
  }
  nb::capsule owner(raw, [](void *ptr) noexcept { delete[] static_cast<std::uint64_t *>(ptr); });
  const std::size_t shape[1] = {values.size()};
  return nb::ndarray<nb::numpy, std::uint64_t>(raw, 1, shape, owner);
}

} // namespace

NB_MODULE(_core, m) {
  m.doc() = "rgsaddle nanobind surface: status mapping and the ASE image callback.";
  m.attr("__version__") = RGSADDLE_PY_VERSION;
  m.def("version", []() { return std::string(RGSADDLE_PY_VERSION); }, "Package version string.");
  m.attr("RGSADDLE_FORCE_L2") = static_cast<int>(RGSADDLE_FORCE_L2);
  m.attr("RGSADDLE_FORCE_LINF") = static_cast<int>(RGSADDLE_FORCE_LINF);
  m.attr("RGSADDLE_FORCE_MAX_ATOM") = static_cast<int>(RGSADDLE_FORCE_MAX_ATOM);
  m.attr("RGSADDLE_MINMODE_DIMER") = static_cast<int>(RGSADDLE_MINMODE_DIMER);
  m.attr("RGSADDLE_MINMODE_LANCZOS") = static_cast<int>(RGSADDLE_MINMODE_LANCZOS);
  m.attr("RGSADDLE_IRC_GS2") = static_cast<int>(RGSADDLE_IRC_GS2);
  m.attr("RGSADDLE_IRC_MOROKUMA") = static_cast<int>(RGSADDLE_IRC_MOROKUMA);
  m.attr("RGSADDLE_IRC_SELLA_SPHERE") = static_cast<int>(RGSADDLE_IRC_SELLA_SPHERE);
  rgbind::install_policy(m);

  nb::class_<ClaimSlot>(m, "ClaimSlot")
      .def(nb::init<>())
      .def("__enter__",
           [](ClaimSlot &slot) -> ClaimSlot & {
             slot.enter();
             return slot;
           },
           nb::rv_policy::reference, "Claim SurfaceAdapter until __exit__.")
      .def(
          "__exit__",
          [](ClaimSlot &slot, nb::handle, nb::handle, nb::handle) {
            slot.exit();
            return false;
          },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(), nb::arg("traceback").none(),
          "Release the claim. Exceptions propagate.");

  m.def("last_error", []() { return std::string(rgsaddle_py_last_error()); },
        "Thread-local slot reserved for the C ABI last-error text. Returns an empty "
        "string unless an error text was set. raise_status reads it and then clears it.");

  m.attr("THERMODYNAMIC_DERIVATIVE") =
      std::string(rgsaddle_thermodynamic_derivative_observable());
  m.def(
      "hyperplane_thermodynamic_integration", &hyperplane_ti_python, nb::arg("sampler"),
      nb::arg("point").noconvert(), nb::arg("normal").noconvert(), nb::arg("thermal") = 0.02585,
      nb::arg("nodes") = 8, nb::arg("thermalize") = 500, nb::arg("stride") = 5,
      nb::arg("samples") = 2000, nb::arg("blocks") = 20, nb::arg("seed") = 1,
      nb::sig("def hyperplane_thermodynamic_integration(sampler: collections.abc.Callable[..., "
              "numpy.typing.NDArray[numpy.float64]], point: numpy.typing.NDArray[numpy.float64], "
              "normal: numpy.typing.NDArray[numpy.float64], thermal: float = 0.02585, nodes: int = 8, "
              "thermalize: int = 500, stride: int = 5, samples: int = 2000, blocks: int = 20, "
              "seed: int = 1) -> tuple[float, float, int]"),
      "Fixed-plane lambda TI through a Python sampler. The callback receives point, normal, lambda, "
      "thermal, seed, thermalize, stride and samples, and returns float64 C-contiguous "
      "V_full - V_harmonic observations.");
  nb::class_<rgsaddle::StencilReport>(m, "StencilReport")
      .def_prop_ro("status", [](const rgsaddle::StencilReport &report) { return report.status; },
                   "0 while running, 1 when converged.")
      .def_prop_ro("max_force", [](const rgsaddle::StencilReport &report) { return report.max_force; },
                   "Maximum absolute force component at the step.")
      .def_prop_ro("curvature", [](const rgsaddle::StencilReport &report) { return report.curvature; },
                   "Curvature along the mode. NaN for a minimiser.")
      .def_prop_ro("iteration", [](const rgsaddle::StencilReport &report) { return report.iteration; },
                   "Steps taken.")
      .def_prop_ro("evaluations", [](const rgsaddle::StencilReport &report) { return report.evaluations; },
                   "Surface evaluations in this call.")
      .def("converged", &rgsaddle::StencilReport::converged, "True when status is converged.");

  nb::class_<PyStencil>(m, "Stencil")
      .def_static(
          "point", &make_point, nb::arg("x").noconvert(), nb::arg("preset") = "dimer",
          nb::arg("mode") = nb::none(), nb::arg("dr") = 1e-3, nb::arg("force_tol") = 1e-3,
          nb::arg("max_move") = 0.2, nb::arg("rotation_tol") = 1e-4, nb::arg("max_rotations") = 20,
          nb::arg("krylov_dim") = 12, nb::arg("threads") = 1, nb::arg("art_threshold") = 0.0,
          nb::arg("art_push") = 20,
          "Point search. dimer and lanczos follow MinModeSession. "
          "The dimer keeps its Polak-Ribiere rotation history across "
          "translations. rotation_angle_tol is the Rust default of five "
          "degrees and is not a Python argument, so the dimer stops on that "
          "angle or on rotation_tol without inferring the classic torque "
          "window. At rotation_tol 1e-4 and looser, lanczos also stops at "
          "two Krylov vectors once the Ritz residual is under 1. A curvature "
          "that is not negative steps by max_move along the mode. Band preset "
          "names raise ValueError.")
      .def_static("band", &make_band, nb::arg("x").noconvert(), nb::arg("preset") = "ci-neb",
                  "Band presets stay on feat/band-methods-zoom and raise ValueError.")
      .def(
          "step", [](PyStencil &self, nb::callable fn) { return step_python(self, fn, 1, false); },
          nb::arg("surface"),
          nb::sig("def step(self, surface: collections.abc.Callable[..., StencilReport]) -> StencilReport"),
          "One step. surface(x) returns (energy, gradient) with x shaped (rows, dof).")
      .def(
          "run",
          [](PyStencil &self, nb::callable fn, std::size_t max_steps) {
            return step_python(self, fn, max_steps, true);
          },
          nb::arg("surface"), nb::arg("max_steps"),
          nb::sig("def run(self, surface: collections.abc.Callable[..., StencilReport], max_steps: int) -> StencilReport"),
          "Loop of step until convergence or max_steps.")
      .def(
          "positions", [](PyStencil &self) { return copy_vector(self.inner.positions()); },
          "Copy of the current position.")
      .def(
          "mode",
          [](PyStencil &self) -> nb::object {
            std::vector<double> values = self.inner.mode();
            if (values.empty()) {
              return nb::none();
            }
            return nb::cast(copy_vector(values));
          },
          nb::sig("def mode(self) -> numpy.typing.NDArray[numpy.float64] | None"),
          "Copy of the mode, or None for a minimiser.")
      .def("curvature", [](PyStencil &self) { return self.inner.curvature(); },
           "Curvature along the mode.");

  nb::class_<ConnectReportPy>(m, "ConnectReport")
      .def_prop_ro("position", [](const ConnectReportPy &report) { return copy_vector(report.position); },
                   nb::rv_policy::move, "Copy of the returned geometry.")
      .def_prop_ro("energy", [](const ConnectReportPy &report) { return report.energy; },
                   "Energy at that geometry.")
      .def_prop_ro("force_norm", [](const ConnectReportPy &report) { return report.force_norm; },
                   "Euclidean norm of the Cartesian gradient.")
      .def_prop_ro("kind", [](const ConnectReportPy &report) { return report.kind; },
                   "s for an accepted saddle, u for an upper bound, f when a segment could not be split.")
      .def_prop_ro("hessian_index", [](const ConnectReportPy &report) { return report.hessian_index; },
                   "Negative eigenvalues of the rigid-projected Hessian, or -1 when it was not computed.")
      .def_prop_ro("evaluations", [](const ConnectReportPy &report) { return report.evaluations; },
                   "Surface evaluations, including a Hessian that decided acceptance.");

  m.def(
      "connect", &connect_python, nb::arg("surface"), nb::arg("left").noconvert(), nb::arg("right").noconvert(),
      nb::arg("force_tol") = 1e-3, nb::arg("max_calls") = static_cast<std::int64_t>(250000),
      nb::arg("quench_nodes") = true,
      nb::sig("def connect(surface: collections.abc.Callable[..., tuple[numpy.typing.NDArray[numpy.float64], "
              "numpy.typing.NDArray[numpy.float64]]], left: numpy.typing.NDArray[numpy.float64], right: "
              "numpy.typing.NDArray[numpy.float64], force_tol: float = 0.001, max_calls: int = 250000, "
              "quench_nodes: bool = True) -> ConnectReport"),
      "Double-ended connect. surface(x) returns (energy, gradient) for one image shaped (1, 3N).");

  m.def(
      "idpp_path", &idpp_path_py, nb::arg("reactant").noconvert(), nb::arg("product").noconvert(),
      nb::arg("n_images"), nb::arg("cell") = nb::none(), nb::arg("max_iterations") = kIdppMaxIterations,
      nb::arg("grad_tol") = kIdppGradTol, nb::arg("max_move") = kIdppMaxMove,
      nb::sig("def idpp_path(reactant: numpy.typing.NDArray[numpy.float64], product: "
              "numpy.typing.NDArray[numpy.float64], n_images: int, cell: "
              "numpy.typing.NDArray[numpy.float64] | None = None, max_iterations: int = 5000, "
              "grad_tol: float = 0.001, max_move: float = 0.1) -> numpy.typing.NDArray[numpy.float64]"),
      "Image-dependent pair potential path (ABI minor 19). reactant and product are flat 3N. "
      "n_images counts the endpoints. cell is the 3x3 row-major lattice, or None in free space. "
      "Interior images are the L-BFGS minimum of the Smidstrup pair objective. The defaults are "
      "IdppConfig::default.");

  nb::class_<AutonebReportPy>(m, "AutonebReport")
      .def_ro("status", &AutonebReportPy::status, "1 when the last stage converged, otherwise 0.")
      .def_ro("max_force", &AutonebReportPy::max_force, "Largest force component of the last stage.")
      .def_ro("ci_index", &AutonebReportPy::ci_index, "Climbing image, or -1.")
      .def_ro("surface_rows", &AutonebReportPy::surface_rows,
              "Rows passed to the surface. One row is one force call.")
      .def_ro("n_images", &AutonebReportPy::n_images, "Images in the finished band, endpoints included.")
      .def_ro("iteration", &AutonebReportPy::iteration, "Steps of the last stage.")
      .def_prop_ro(
          "positions",
          [](const AutonebReportPy &self) {
            const auto rows = static_cast<std::size_t>(std::max(self.n_images, std::int64_t{0}));
            return copy_matrix(self.positions, rows == 0 ? 1 : rows, self.dof == 0 ? 1 : self.dof);
          },
          nb::rv_policy::move, "Copy of the finished band, shaped (n_images, 3N).");

  m.def(
      "autoneb_run", &autoneb_run_py, nb::arg("surface"), nb::arg("positions").noconvert(),
      nb::arg("k") = 0.1, nb::arg("climb") = true, nb::arg("n_simul") = static_cast<std::int64_t>(3),
      nb::arg("n_max") = static_cast<std::int64_t>(7), nb::arg("force_tol") = 0.025,
      nb::arg("local_force_tol") = 0.025, nb::arg("local_max_steps") = static_cast<std::int64_t>(10000),
      nb::arg("climb_max_steps") = static_cast<std::int64_t>(10000), nb::arg("projection") = "neb",
      nb::arg("tangent") = "improved", nb::arg("remove_translation") = false,
      nb::arg("space_energy_ratio") = 0.5, nb::arg("max_move") = 0.2, nb::arg("idpp") = false,
      nb::arg("cell") = nb::none(), nb::arg("fixed") = nb::none(),
      nb::sig("def autoneb_run(surface: collections.abc.Callable[..., tuple[numpy.typing.NDArray[numpy.float64], "
              "numpy.typing.NDArray[numpy.float64]]], positions: numpy.typing.NDArray[numpy.float64], "
              "k: float = 0.1, climb: bool = True, n_simul: int = 3, n_max: int = 7, "
              "force_tol: float = 0.025, local_force_tol: float = 0.025, local_max_steps: int = 10000, "
              "climb_max_steps: int = 10000, projection: str = 'neb', tangent: str = 'improved', "
              "remove_translation: bool = False, space_energy_ratio: float = 0.5, max_move: float = 0.2, "
              "idpp: bool = False, cell: numpy.typing.NDArray[numpy.float64] | None = None, "
              "fixed: numpy.typing.NDArray[numpy.uint8] | None = None) -> AutonebReport"),
      "Run AutoNEB to completion (ABI minor 53). positions is (n_init >= 2, 3N). "
      "surface(x) receives (rows, 3N) and returns (energies, gradients). One row is one force call. "
      "The search stays on the calling thread.");

  nb::class_<SessionReport>(m, "SessionReport")
      .def_ro("status", &SessionReport::status, "0 while running, 1 when converged.")
      .def_ro("max_force", &SessionReport::max_force, "Largest force component, or NaN for an estimate.")
      .def_ro("curvature", &SessionReport::curvature, "Curvature along the mode. 0 for a band projection.")
      .def_ro("iteration", &SessionReport::iteration, "Steps taken.")
      .def_ro("evaluations", &SessionReport::evaluations, "Surface evaluations in this call.")
      .def_ro("ci_index", &SessionReport::ci_index, "Climbing image, or -1.")
      .def_ro("rotations", &SessionReport::rotations, "Dimer rotations or Lanczos actions.")
      .def_ro("saddle_index", &SessionReport::saddle_index,
              "Saddle index certified at the session's current position, or -1.");

  nb::class_<SaddleIndexPy>(m, "SaddleIndex")
      .def_ro("certified", &SaddleIndexPy::certified,
              "True when a certificate holds for the current point; otherwise the counts are "
              "the session's.")
      .def_ro("index", &SaddleIndexPy::index, "Curvatures under -negative_tol; 1 for a first-order saddle.")
      .def_ro("converged", &SaddleIndexPy::converged,
              "Every resolved pair met residual_tol, or the space filled.")
      .def_prop_ro("curvatures", [](const SaddleIndexPy &self) { return copy_vector(self.curvatures); },
                   nb::rv_policy::move, "Lowest curvatures resolved, ascending, energy per length squared.")
      .def_ro("evaluations", &SaddleIndexPy::evaluations, "Surface evaluations of this certificate.")
      .def_ro("total_evaluations", &SaddleIndexPy::total_evaluations,
              "Surface evaluations of every certificate and escape push of the session.")
      .def_ro("escapes", &SaddleIndexPy::escapes, "Escape pushes of the session.");

  m.def(
      "saddle_index", &saddle_index_py, nb::arg("surface"), nb::arg("x").noconvert(),
      nb::arg("gradient") = nb::none(), nb::arg("mode") = nb::none(), nb::arg("fixed") = nb::none(),
      nb::arg("lowest") = 4, nb::arg("max_actions") = 100, nb::arg("dr") = 1e-3,
      nb::arg("residual_tol") = 1e-2, nb::arg("negative_tol") = 1e-3,
      nb::sig("def saddle_index(surface: collections.abc.Callable[..., "
              "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], "
              "x: numpy.typing.NDArray[numpy.float64], gradient: numpy.typing.NDArray[numpy.float64] | "
              "None = None, mode: numpy.typing.NDArray[numpy.float64] | None = None, fixed: "
              "numpy.typing.NDArray[numpy.uint8] | None = None, lowest: int = 4, max_actions: int = 100, "
              "dr: float = 0.001, residual_tol: float = 0.01, negative_tol: float = 0.001) -> SaddleIndex"),
      "Saddle index of x (ABI minor 40): the lowest curvatures on the free coordinates by block "
      "Lanczos on forward-difference Hessian actions, each one surface call. surface(x) is one "
      "image shaped (1, 3N). gradient at x saves one call; mode seeds the space; fixed is one "
      "uint8 per atom, 1 for a fixed atom.");

  m.def(
      "project_band", &project_band, nb::arg("positions").noconvert(), nb::arg("energies").noconvert(),
      nb::arg("gradients").noconvert(), nb::arg("k") = 0.1, nb::arg("climb") = false,
      nb::arg("projection") = "neb", nb::arg("tangent") = "improved",
      nb::arg("remove_translation") = false, nb::arg("force_tol") = 1e-3, nb::arg("fixed") = nb::none(),
      nb::sig("def project_band(positions: numpy.typing.NDArray[numpy.float64], energies: "
              "numpy.typing.NDArray[numpy.float64], gradients: numpy.typing.NDArray[numpy.float64], "
              "k: float = 0.1, climb: bool = False, projection: str = 'neb', tangent: str = 'improved', "
              "remove_translation: bool = False, force_tol: float = 0.001, "
              "fixed: numpy.typing.NDArray[numpy.uint8] | None = None) -> "
              "tuple[numpy.typing.NDArray[numpy.float64], int]"),
      "Project an evaluated band. Returns (forces, ci_index). forces is flat, "
      "one row per image including zero endpoints, in the ASE force convention. "
      "climb selects the interior image by the endpoint-slope rule on this call; "
      "a higher endpoint that slopes into the band does not climb. Units are the "
      "caller's. k is energy per length squared. fixed is one uint8 per atom, 1 "
      "for a fixed atom (ABI minor 39): its force rows are zero, tangents and "
      "springs use the free atoms, and the mean force stays. The endpoints must "
      "agree on every fixed atom.");

  m.def(
      "reparametrize", &reparametrize_py, nb::arg("positions").noconvert(), nb::arg("scheme") = "linear",
      nb::arg("anchor") = nb::none(), nb::arg("cell") = nb::none(),
      nb::sig("def reparametrize(positions: numpy.typing.NDArray[numpy.float64], scheme: str = 'linear', "
              "anchor: int | None = None, cell: numpy.typing.NDArray[numpy.float64] | None = None) -> "
              "numpy.typing.NDArray[numpy.float64]"),
      "Place the interior images of a band at equal arc length and return the new band. "
      "scheme 'linear' places them on the polyline; 'not-a-knot' (or 'spline') on the "
      "not-a-knot cubic spline through the images in their arc length (E, Ren and "
      "Vanden-Eijnden 2007). The first and last rows, and an interior anchor, do not move. "
      "cell is the 3x3 row-major cell; the chords are then minimum images. No surface is evaluated.");

  m.def(
      "halve_raised_energy", &halve_raised_energy_py, nb::arg("previous"), nb::arg("trial"),
      nb::arg("step"),
      nb::sig("def halve_raised_energy(previous: float, trial: float, step: float) -> tuple[bool, float]"),
      "Compare a trial energy with the energy before the step. Returns (raised, new_step). "
      "A trial more than 1e-4 above previous halves step. No surface is evaluated.");

  m.def(
      "trust_update", &trust_update_py, nb::arg("radius"), nb::arg("predicted"), nb::arg("actual"),
      nb::arg("step_norm"), nb::arg("sigma_inc"), nb::arg("sigma_dec"), nb::arg("rho_inc"),
      nb::arg("rho_dec"), nb::arg("delta_min"),
      nb::sig("def trust_update(radius: float, predicted: float, actual: float, step_norm: float, "
              "sigma_inc: float, sigma_dec: float, rho_inc: float, rho_dec: float, delta_min: float) -> float"),
      "Trust radius from a predicted energy change and the change that occurred. "
      "actual is the trial energy minus the energy before the step. A prediction below "
      "1e-14 in absolute value returns radius. The remaining arguments are the Sella schedule. "
      "No surface is evaluated.");

  nb::class_<PyMinMode>(m, "MinMode")
      .def(
          "__init__",
          [](PyMinMode *self, Vector x, Vector mode, const std::string &kind, double dr, double force_tol,
             double max_move, double rotation_tol, std::int64_t max_rotations, std::int64_t krylov_dim,
             double rotation_stop_length, const std::string &method) {
            new (self) PyMinMode(make_minmode(x, mode, kind, dr, force_tol, max_move, rotation_tol,
                                              max_rotations, krylov_dim, rotation_stop_length, method));
          },
          nb::arg("x").noconvert(), nb::arg("mode").noconvert(), nb::arg("kind") = "dimer",
          nb::arg("dr") = 1e-3, nb::arg("force_tol") = 1e-3, nb::arg("max_move") = 0.2,
          nb::arg("rotation_tol") = 1e-4, nb::arg("max_rotations") = 20, nb::arg("krylov_dim") = 12,
          nb::arg("rotation_stop_length") = 0.0, nb::arg("method") = "lbfgs",
          "Dimer or Lanczos minimum-mode session. Lengths are the caller's. A positive "
          "rotation_stop_length stops a rotation once |C| * rotation_stop_length * |sin w| is "
          "at most force_tol (Kastner and Sherwood 2008); 0 turns that stop off. method is the "
          "translation optimizer, lbfgs (the default) or fire.")
      .def(
          "estimate",
          [](PyMinMode &self, nb::callable fn) {
            return with_saddle_index(
                run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
                  return rgsaddle_minmode_estimate(self.handle, surface, user, out);
                }),
                self);
          },
          nb::arg("surface"),
          nb::sig("def estimate(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) "
                  "-> SessionReport"),
          "Refresh the lowest mode. surface(x) is one image shaped (1, 3N).")
      .def(
          "step",
          [](PyMinMode &self, nb::callable fn) {
            return with_saddle_index(
                run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
                  return rgsaddle_minmode_step(self.handle, surface, user, out);
                }),
                self);
          },
          nb::arg("surface"),
          nb::sig("def step(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) "
                  "-> SessionReport"),
          "Estimate the mode, invert the force, and take one translation. A converged step "
          "certifies the saddle index; with escape_attempts set, a step that pushes off a "
          "saddle of index two or more reports status 0.")
      .def(
          "set_position", &minmode_set_position, nb::arg("x").noconvert(), nb::arg("gradient") = nb::none(),
          "Move to a new point. gradient, when given, skips the next centre evaluation.")
      .def(
          "position",
          [](const PyMinMode &self) {
            std::vector<double> out(static_cast<std::size_t>(self.n_atoms * 3));
            const auto status = rgsaddle_minmode_position(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_vector(out);
          },
          "Copy of the current position.")
      .def(
          "mode",
          [](const PyMinMode &self) {
            std::vector<double> out(static_cast<std::size_t>(self.n_atoms * 3));
            const auto status = rgsaddle_minmode_mode(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_vector(out);
          },
          "Copy of the current mode.")
      .def("saddle_index", &minmode_saddle_index,
           "Certificate of the current position (ABI minor 40). On by default; a push or a move "
           "drops it.")
      .def("set_saddle_check", &minmode_set_saddle_check, nb::arg("enabled") = true,
           nb::arg("escape_attempts") = 0, nb::arg("escape_push") = 0.1, nb::arg("lowest") = 4,
           nb::arg("max_actions") = 100, nb::arg("dr") = 1e-3, nb::arg("residual_tol") = 1e-2,
           nb::arg("negative_tol") = 1e-3,
           "Replace the saddle check. enabled=False turns it off. escape_attempts above 0 pushes "
           "a converged point of index two or more by escape_push, the largest per-atom "
           "displacement, doubled on each later attempt, along its extra negative modes and "
           "searches again.");

  nb::class_<PyBand>(m, "Band")
      .def(
          "__init__",
          [](PyBand *self, Matrix positions, double k, bool climb, double ci_after,
             double ci_after_rel, const std::string &projection, const std::string &tangent,
             const std::string &method, std::int64_t memory, double force_tol, double max_move,
             double time_step, bool remove_translation, bool min_mode, bool string,
             const std::string &reparametrization, nb::handle cell, nb::handle fixed) {
            new (self) PyBand(make_band_session(
                positions, k, climb, ci_after, ci_after_rel, projection, tangent, method, memory,
                force_tol, max_move, time_step, remove_translation, min_mode, string,
                reparametrization, cell, fixed));
          },
          nb::arg("positions").noconvert(), nb::arg("k") = 5.0, nb::arg("climb") = true,
          nb::arg("ci_after") = 0.5, nb::arg("ci_after_rel") = 0.0,
          nb::arg("projection") = "neb", nb::arg("tangent") = "improved",
          nb::arg("method") = "fire", nb::arg("memory") = 20, nb::arg("force_tol") = 0.001,
          nb::arg("max_move") = 0.2, nb::arg("time_step") = 1.0,
          nb::arg("remove_translation") = true, nb::arg("min_mode") = false,
          nb::arg("string") = false, nb::arg("reparametrization") = "not-a-knot",
          nb::arg("cell") = nb::none(), nb::arg("fixed") = nb::none(),
          "Stepping band session for NEB and the simplified string method (E, Ren and "
          "Vanden-Eijnden 2007). positions is (n_images >= 3, 3N). string=True evolves "
          "interior images under the full potential force and reparametrizes to equal arc "
          "length on 'not-a-knot' (or 'spline') or 'linear', anchoring an armed climbing image.")
      .def(
          "step",
          [](PyBand &self, nb::callable fn) {
            return with_band_saddle_index(
                run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
                  return rgsaddle_band_step(self.handle, surface, user, out);
                }),
                self);
          },
          nb::arg("surface"),
          nb::sig("def step(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) "
                  "-> SessionReport"),
          "One band or simplified-string step. surface(x) receives (rows, 3N) and returns "
          "(energies, gradients).")
      .def(
          "run", &band_run_py, nb::arg("surface"), nb::arg("max_steps") = 1000,
          nb::sig("def run(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], "
                  "max_steps: int = 1000) -> SessionReport"),
          "Loop step until convergence or max_steps.")
      .def(
          "positions",
          [](const PyBand &self) {
            const auto rows = static_cast<std::size_t>(self.n_images);
            const auto dof = static_cast<std::size_t>(self.n_atoms * 3);
            std::vector<double> out(rows * dof, 0.0);
            const auto status = rgsaddle_band_positions(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_matrix(out, rows, dof);
          },
          "Copy of the current band positions, shaped (n_images, 3N).")
      .def(
          "set_positions",
          [](PyBand &self, Matrix positions) {
            if (static_cast<std::int64_t>(positions.shape(0)) != self.n_images ||
                static_cast<std::int64_t>(positions.shape(1)) != self.n_atoms * 3) {
              throw nb::value_error("positions shape must match the band");
            }
            const auto status = rgsaddle_band_set_positions(self.handle, positions.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          nb::arg("positions").noconvert(),
          "Replace the band positions; unchanged rows keep cached evaluations.")
      .def(
          "evaluation", &band_evaluation_py,
          nb::sig("def evaluation(self) -> tuple[numpy.typing.NDArray[numpy.float64], "
                  "numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]"),
          "Cached (energies, gradients, projected) at the current band.")
      .def(
          "set_string",
          [](PyBand &self, bool enabled, const std::string &reparametrization) {
            const auto scheme = enabled ? reparam_code(reparametrization) : RGSADDLE_REPARAM_NOT_A_KNOT;
            const auto status = rgsaddle_band_set_string(self.handle, enabled ? 1 : 0, scheme);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          nb::arg("enabled") = true, nb::arg("reparametrization") = "not-a-knot",
          "Enable or clear the simplified string method on the band (ABI minor 44).")
      .def(
          "set_free_end",
          [](PyBand &self, bool enabled, const std::string &side, nb::handle target_energy,
             std::int64_t max_cuts, double cut_force, double energy_tol, std::int64_t stability,
             std::int64_t window_offset, bool cubic) {
            std::int32_t side_code = RGSADDLE_FREE_END_LAST;
            if (side == "first") {
              side_code = RGSADDLE_FREE_END_FIRST;
            } else if (side != "last") {
              throw nb::value_error("side must be first or last");
            }
            const bool has_target = !target_energy.is_none();
            const double target_val = has_target ? nb::cast<double>(target_energy) : 0.0;
            const auto status = rgsaddle_band_set_free_end(
                self.handle, enabled ? 1 : 0, side_code, has_target ? 1 : 0, target_val, max_cuts,
                cut_force, energy_tol, stability, window_offset, cubic ? 1 : 0);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          nb::arg("enabled") = true, nb::arg("side") = "last",
          nb::arg("target_energy") = nb::none(), nb::arg("max_cuts") = 2,
          nb::arg("cut_force") = 0.0, nb::arg("energy_tol") = 0.001, nb::arg("stability") = 2,
          nb::arg("window_offset") = 1, nb::arg("cubic") = true,
          "Enable or clear free-end adaptive NEB on the band (ABI minor 45).")
      .def(
          "target_energy_residual",
          [](const PyBand &self) -> nb::object {
            double out = 0.0;
            const auto status = rgsaddle_band_free_end_residual(self.handle, &out);
            if (status == RGSADDLE_NO_EVALUATION) {
              return nb::none();
            }
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return nb::cast(out);
          },
          nb::sig("def target_energy_residual(self) -> float | None"),
          "Signed target-energy residual E(x_end) - E_target from the last free-end step.")
      .def(
          "free_end_cuts",
          [](const PyBand &self) { return rgsaddle_band_free_end_cuts(self.handle); },
          "Number of adaptive cut-and-remesh refinements performed so far (ABI minor 45).")
      .def(
          "set_fixed_atoms",
          [](PyBand &self, FlagArray fixed) {
            if (static_cast<std::int64_t>(fixed.shape(0)) != self.n_atoms) {
              throw nb::value_error("fixed must hold one entry per atom");
            }
            const auto status = rgsaddle_band_set_fixed_atoms(self.handle, fixed.data(), self.n_atoms);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          nb::arg("fixed").noconvert(), "Replace the per-atom fixed mask (1 fixed, 0 free).")
      .def("saddle_index", &band_saddle_index,
           "Certificate of the climbing image at its current position (ABI minor 40).")
      .def("set_saddle_check", &band_set_saddle_check, nb::arg("enabled") = true,
           nb::arg("escape_attempts") = 0, nb::arg("escape_push") = 0.1, nb::arg("lowest") = 4,
           nb::arg("max_actions") = 100, nb::arg("dr") = 0.001, nb::arg("residual_tol") = 0.01,
           nb::arg("negative_tol") = 0.001,
           "Replace the climbing image's saddle check; enabled=False turns it off.")
      .def(
          "reset",
          [](PyBand &self) {
            const auto status = rgsaddle_band_reset(self.handle);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          "Drop optimizer history, climbing state, and all cached surface values.")
      .def(
          "restart",
          [](PyBand &self) {
            const auto status = rgsaddle_band_restart(self.handle);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          "Drop optimizer history and climbing state, keeping cached surface values.");

  nb::class_<PyGrowingString>(m, "GrowingString")
      .def(
          "__init__",
          [](PyGrowingString *self, Vector reactant, Vector product, std::int64_t n_images,
             double frontier_tol, std::int64_t min_steps_per_node,
             std::int64_t max_steps_per_node, std::optional<double> join_tol, bool climb,
             double ci_after, double ci_after_rel, const std::string &method,
             std::int64_t memory, double force_tol, double max_move, bool remove_translation,
             bool min_mode, const std::string &reparametrization, nb::handle cell,
             nb::handle fixed) {
            new (self) PyGrowingString(make_growing_string_session(
                reactant, product, n_images, frontier_tol, min_steps_per_node,
                max_steps_per_node, join_tol, climb, ci_after, ci_after_rel, method, memory,
                force_tol, max_move, remove_translation, min_mode, reparametrization, cell, fixed));
          },
          nb::arg("reactant").noconvert(), nb::arg("product").noconvert(),
          nb::arg("n_images") = 11, nb::arg("frontier_tol") = 0.2,
          nb::arg("min_steps_per_node") = 1, nb::arg("max_steps_per_node") = 0,
          nb::arg("join_tol") = nb::none(), nb::arg("climb") = true, nb::arg("ci_after") = 0.5,
          nb::arg("ci_after_rel") = 0.0, nb::arg("method") = "fire", nb::arg("memory") = 20,
          nb::arg("force_tol") = 0.001, nb::arg("max_move") = 0.2,
          nb::arg("remove_translation") = true, nb::arg("min_mode") = false,
          nb::arg("reparametrization") = "not-a-knot", nb::arg("cell") = nb::none(),
          nb::arg("fixed") = nb::none(),
          "Growing string session (Peters, Heyden, Bell and Chakraborty 2004, ABI minor 44).")
      .def(
          "step",
          [](PyGrowingString &self, nb::callable fn) {
            return with_growing_string_saddle_index(
                run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
                  return rgsaddle_growing_string_step(self.handle, surface, user, out);
                }),
                self);
          },
          nb::arg("surface"),
          nb::sig("def step(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) "
                  "-> SessionReport"),
          "One growing-string or joined simplified-string step.")
      .def(
          "run", &growing_string_run_py, nb::arg("surface"), nb::arg("max_steps") = 1000,
          nb::sig("def run(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]], "
                  "max_steps: int = 1000) -> SessionReport"),
          "Loop step until the fragments join and converge, or until max_steps.")
      .def(
          "n_images",
          [](const PyGrowingString &self) {
            return rgsaddle_growing_string_n_images(self.handle);
          },
          "Active image count (left_nodes + right_nodes while growing, n_images once joined).")
      .def(
          "nodes_added",
          [](const PyGrowingString &self) {
            return rgsaddle_growing_string_nodes_added(self.handle);
          },
          "Nodes the last step grew on the reactant and product fragments (0, 1, or 2).")
      .def(
          "joined",
          [](const PyGrowingString &self) {
            return rgsaddle_growing_string_phase(self.handle) == RGSADDLE_GROWING_STRING_JOINED;
          },
          "True once the reactant and product fragments have joined into a StringSession.")
      .def(
          "positions",
          [](const PyGrowingString &self) {
            const auto rows = static_cast<std::size_t>(rgsaddle_growing_string_n_images(self.handle));
            const auto dof = static_cast<std::size_t>(self.n_atoms * 3);
            std::vector<double> out(rows * dof, 0.0);
            const auto status = rgsaddle_growing_string_positions(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_matrix(out, rows, dof);
          },
          "Copy of the active positions, shaped (active_images, 3N).")
      .def(
          "ids",
          [](const PyGrowingString &self) {
            const auto rows = static_cast<std::size_t>(rgsaddle_growing_string_n_images(self.handle));
            std::vector<std::uint64_t> out(rows, 0);
            const auto status = rgsaddle_growing_string_ids(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_u64_vector(out);
          },
          "Stable uint64 image IDs of the active rows.")
      .def(
          "set_fixed_atoms",
          [](PyGrowingString &self, FlagArray fixed) {
            if (static_cast<std::int64_t>(fixed.shape(0)) != self.n_atoms) {
              throw nb::value_error("fixed must hold one entry per atom");
            }
            const auto status =
                rgsaddle_growing_string_set_fixed_atoms(self.handle, fixed.data(), self.n_atoms);
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
          },
          nb::arg("fixed").noconvert(), "Replace the per-atom fixed mask (1 fixed, 0 free).")
      .def("saddle_index", &growing_string_saddle_index,
           "Certificate of the climbing image once joined and converged (ABI minor 44).");

  nb::class_<PySella>(m, "SellaSaddle")
      .def(
          "__init__",
          [](PySella *self, Vector x, Vector masses, std::int64_t order, double delta, double force_tol) {
            new (self) PySella(make_sella(x, masses, order, delta, force_tol));
          },
          nb::arg("x").noconvert(), nb::arg("masses").noconvert(), nb::arg("order") = 1,
          nb::arg("delta") = 0.1, nb::arg("force_tol") = 1e-3,
          "Order-1 Sella saddle search. masses has one entry per atom, in the caller's mass unit. "
          "delta is the initial trust radius per free coordinate.")
      .def(
          "step",
          [](PySella &self, nb::callable fn) {
            return run_surface(fn, [&](rgsaddle_surface_fn surface, void *user, rgsaddle_report_t *out) {
              return rgsaddle_sella_saddle_step(self.handle, surface, user, out);
            });
          },
          nb::arg("surface"),
          nb::sig("def step(self, surface: collections.abc.Callable[..., "
                  "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]]) "
                  "-> SessionReport"),
          "One partitioned-RFO trust step. surface(x) is one image shaped (1, 3N).")
      .def(
          "position",
          [](const PySella &self) {
            std::vector<double> out(static_cast<std::size_t>(self.n_atoms * 3));
            const auto status = rgsaddle_sella_saddle_position(self.handle, out.data());
            if (status != RGSADDLE_OK) {
              status_error(status);
            }
            return copy_vector(out);
          },
          "Copy of the current Cartesian position.");

  m.def("raise_status", &raise_status, nb::arg("code"),
        "Map an rgsaddle_status_t value. The message is the stored text, or the "
        "status name when that text is empty. The stored text is cleared. True on OK. "
        "False is not used: convergence is not a status code.");

  // Test-only surface. stubgen skips a private submodule, so these
  // names stay off the public stub.
  nb::module_ testing = m.def_submodule("_testing", "Eval-loop probes. Not part of the public surface.");

  nb::class_<HoldGate>(testing, "HoldGate")
      .def(nb::init<>())
      .def("release", &HoldGate::release, "Let the holding thread drop the claim.")
      .def("held", &HoldGate::held, "True once the holding thread owns the claim.");

  testing.def(
      "claim_work", [](ClaimSlot &slot, int hold_ms) { return slot.work(hold_ms); }, nb::arg("slot"),
      nb::arg("hold_ms") = 0, "Claim SurfaceAdapter, hold, and release.");
  testing.def(
      "claim_hold_until", [](ClaimSlot &slot, HoldGate &gate) { slot.hold_until(gate); }, nb::arg("slot"),
      nb::arg("gate"), "Claim and wait, with the GIL released, until gate.release().");
  testing.def("set_last_error", [](const std::string &text) { rgsaddle_py_set_last_error(text.c_str()); },
              nb::arg("text"), "Replace the thread-local last-error text.");
  testing.def("schedule_canonical", &schedule_canonical, nb::arg("ids").noconvert(), nb::arg("owners").noconvert(),
              nb::arg("reentrant").noconvert(), nb::arg("threads"),
              "Canonical owner schedule. The worker count is not part of the text.");
  testing.def("echo_f64", &rgbind::echo_f64, nb::arg("values").noconvert(),
              "Byte-copy a float64 array. -0.0, NaN and subnormals pass through.");
  testing.def(
      "drive_eval", &drive_eval,
      nb::sig("def drive_eval(callback: collections.abc.Callable[[numpy.typing.NDArray[numpy.int64], "
              "numpy.typing.NDArray[numpy.float64]], tuple[numpy.typing.NDArray[numpy.float64], "
              "numpy.typing.NDArray[numpy.float64]]], ids: numpy.typing.NDArray[numpy.int64], x: "
              "numpy.typing.NDArray[numpy.float64], owners: numpy.typing.NDArray[numpy.uint64], reentrant: "
              "numpy.typing.NDArray[numpy.uint8], threads: int, slot: ClaimSlot) -> "
              "tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]"),
      nb::arg("callback"), nb::arg("ids").noconvert(), nb::arg("x").noconvert(), nb::arg("owners").noconvert(),
      nb::arg("reentrant").noconvert(), nb::arg("threads"), nb::arg("slot"),
      "Test eval loop. Groups rows by owner and calls "
      "callback(ids, x) through the trampoline.");
}
