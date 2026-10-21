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
    case RGSADDLE_STALLED:
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
          "Point search. Dimer keeps rotational CG history across translations; Lanczos translations use "
          "relative Ritz-value stopping. Band preset names raise ValueError.")
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
