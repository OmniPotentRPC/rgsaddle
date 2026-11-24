/* Run-to-completion AutoNEB through the C ABI, on a model double well. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(RGSADDLE_AUTONEB_CLIMB == (1ull << 0), "AutoNEB flag bit 0");
_Static_assert(RGSADDLE_AUTONEB_IDPP == (1ull << 1), "AutoNEB flag bit 1");
_Static_assert(RGSADDLE_AUTONEB_KEEP_TRANSLATION == (1ull << 2), "AutoNEB flag bit 2");
_Static_assert(offsetof(rgsaddle_autoneb_config_t, version) == 0, "config opens with the version");
_Static_assert(offsetof(rgsaddle_autoneb_config_t, flags) == sizeof(rgsaddle_version_t),
               "config flags follow the version");
_Static_assert(offsetof(rgsaddle_autoneb_report_t, version) == 0, "report opens with the version");
_Static_assert(offsetof(rgsaddle_autoneb_report_t, flags) == sizeof(rgsaddle_version_t),
               "report flags follow the version");

#define CHECK(cond, msg)                                                                  \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      fprintf(stderr, "FAIL line %d: %s\n", __LINE__, msg);                               \
      return 1;                                                                           \
    }                                                                                     \
  } while (0)

#define N_ATOMS 1
#define DOF (3 * N_ATOMS)
#define N_INIT 3
#define N_MAX 5
#define FORCE_TOL 1e-3
#define CLIMB_STEPS 400

/* E = (x^2 - 1)^2 + 2 y^2 + 2 z^2. Minima at x = -1 and x = 1, and the
   index-1 saddle at the origin, where the barrier is 1. */
typedef struct {
  long calls;
  long rows;
  long fail_after;
  int bad_shape;
} host_t;

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  host_t *host = (host_t *)user;
  host->calls += 1;
  if (host->fail_after > 0 && host->calls > host->fail_after) {
    return RGSADDLE_SURFACE_FAILED;
  }
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != N_ATOMS) {
    host->bad_shape = 1;
    return RGSADDLE_SHAPE;
  }
  host->rows += (long)req->n_images;
  for (long i = 0; i < (long)req->n_images; ++i) {
    const double *p = req->positions + i * DOF;
    double *g = req->gradients + i * DOF;
    const double x = p[0], y = p[1], z = p[2];
    req->energies[i] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
  }
  return RGSADDLE_OK;
}

static void start_band(double *pos) {
  const double rows[N_INIT][DOF] = {{-1.0, 0.0, 0.0}, {0.1, 0.4, 0.0}, {1.0, 0.0, 0.0}};
  memcpy(pos, rows, sizeof rows);
}

static void default_config(rgsaddle_autoneb_config_t *cfg) {
  memset(cfg, 0, sizeof *cfg);
  cfg->version.major = RGSADDLE_ABI_MAJOR;
  cfg->version.minor = RGSADDLE_ABI_MINOR;
  cfg->flags = RGSADDLE_AUTONEB_CLIMB;
  cfg->projection = RGSADDLE_PROJECTION_NEB;
  cfg->tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg->n_simul = 1;
  cfg->n_max = N_MAX;
  cfg->local_max_steps = 400;
  cfg->climb_max_steps = CLIMB_STEPS;
  cfg->k = 5.0;
  cfg->force_tol = FORCE_TOL;
  cfg->local_force_tol = 5e-2;
  cfg->max_move = 0.1;
  cfg->space_energy_ratio = 0.5;
}

static rgsaddle_status_t run(const rgsaddle_autoneb_config_t *cfg, host_t *host, double *out,
                             rgsaddle_autoneb_report_t *report) {
  double start[N_INIT * DOF];
  start_band(start);
  memset(out, 0, sizeof(double) * N_MAX * DOF);
  memset(report, 0, sizeof *report);
  memset(host, 0, sizeof *host);
  return rgsaddle_autoneb_run(cfg, N_INIT, N_ATOMS, start, NULL, NULL, out, report, surface,
                              host);
}

static int test_converged_run(void) {
  rgsaddle_autoneb_config_t cfg;
  rgsaddle_autoneb_report_t report;
  host_t host;
  double out[N_MAX * DOF];
  default_config(&cfg);
  CHECK(run(&cfg, &host, out, &report) == RGSADDLE_OK, "converged run returns OK");
  CHECK(report.version.major == RGSADDLE_ABI_MAJOR && report.version.minor == RGSADDLE_ABI_MINOR,
        "report carries the ABI stamp");
  CHECK(report.status == 1, "climbing window meets its force stop");
  CHECK(report.max_force >= 0.0 && report.max_force <= FORCE_TOL, "max force is under the stop");
  CHECK(report.iteration > 0 && report.iteration <= CLIMB_STEPS, "iteration is within the budget");
  CHECK(report.n_images == N_MAX, "the band grows to n_max images");
  CHECK(report.surface_rows == host.rows && report.surface_rows > 0,
        "surface_rows counts the rows the callback received");
  CHECK(host.calls > 0 && !host.bad_shape, "surface callback ran with the published request");
  CHECK(report.ci_index >= 1 && report.ci_index <= N_MAX - 2, "climbing image is interior");
  CHECK(fabs(out[report.ci_index * DOF]) < 0.05 && fabs(out[report.ci_index * DOF + 1]) < 0.05,
        "climbing image sits on the saddle");
  CHECK(out[0] == -1.0 && out[1] == 0.0 && out[2] == 0.0, "reactant endpoint is kept");
  CHECK(out[(N_MAX - 1) * DOF] == 1.0 && out[(N_MAX - 1) * DOF + 1] == 0.0, "product endpoint is kept");

  /* A second run is the same search: the count and the force repeat. */
  rgsaddle_autoneb_report_t again;
  host_t host_again;
  double out_again[N_MAX * DOF];
  CHECK(run(&cfg, &host_again, out_again, &again) == RGSADDLE_OK, "second run returns OK");
  CHECK(again.iteration == report.iteration && again.max_force == report.max_force &&
            again.surface_rows == report.surface_rows,
        "the report is reproducible");
  CHECK(memcmp(out, out_again, sizeof out) == 0, "the band is reproducible");
  return 0;
}

static int test_exhausted_budget(void) {
  rgsaddle_autoneb_config_t cfg;
  rgsaddle_autoneb_report_t report;
  host_t host;
  double out[N_MAX * DOF];
  default_config(&cfg);
  cfg.climb_max_steps = 2;
  cfg.force_tol = 1e-12;
  CHECK(run(&cfg, &host, out, &report) == RGSADDLE_OK, "an exhausted budget is not an error");
  CHECK(report.status == 0, "the climbing window did not converge");
  CHECK(report.max_force > cfg.force_tol, "max force is over the stop");
  CHECK(report.iteration == cfg.climb_max_steps, "iteration is the step budget");
  CHECK(report.n_images == N_MAX, "the band still reaches n_max images");
  return 0;
}

static int test_without_climb(void) {
  rgsaddle_autoneb_config_t cfg;
  rgsaddle_autoneb_report_t report;
  host_t host;
  double out[N_MAX * DOF];
  default_config(&cfg);
  cfg.flags = 0;
  CHECK(run(&cfg, &host, out, &report) == RGSADDLE_OK, "run without the climb returns OK");
  CHECK(report.n_images == N_MAX, "insertions reach n_max images");
  CHECK(report.ci_index == -1, "no climbing image is reported");
  return 0;
}

static int test_error_paths(void) {
  rgsaddle_autoneb_config_t cfg;
  rgsaddle_autoneb_report_t report;
  host_t host;
  double out[N_MAX * DOF];
  double start[N_INIT * DOF];
  double nan_cell[9] = {NAN, 0, 0, 0, 1, 0, 0, 0, 1};
  start_band(start);
  default_config(&cfg);

  CHECK(rgsaddle_autoneb_run(NULL, N_INIT, N_ATOMS, start, NULL, NULL, out, &report, surface,
                             &host) == RGSADDLE_INVALID_PARAMETER,
        "NULL config");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, NULL, NULL, NULL, out, &report, surface,
                             &host) == RGSADDLE_INVALID_PARAMETER,
        "NULL positions");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, start, NULL, NULL, NULL, &report, surface,
                             &host) == RGSADDLE_INVALID_PARAMETER,
        "NULL output");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, start, NULL, NULL, out, NULL, surface,
                             &host) == RGSADDLE_INVALID_PARAMETER,
        "NULL report");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, start, NULL, NULL, out, &report, NULL,
                             &host) == RGSADDLE_NULL_SURFACE,
        "NULL surface");
  CHECK(rgsaddle_autoneb_run(&cfg, 1, N_ATOMS, start, NULL, NULL, out, &report, surface, &host) ==
            RGSADDLE_SHAPE,
        "one initial image");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, 0, start, NULL, NULL, out, &report, surface, &host) ==
            RGSADDLE_SHAPE,
        "no atoms");
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, start, nan_cell, NULL, out, &report, surface,
                             &host) == RGSADDLE_NON_FINITE,
        "non finite cell");

  rgsaddle_autoneb_config_t bad = cfg;
  bad.version.major = RGSADDLE_ABI_MAJOR + 1u;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_ABI_MISMATCH, "foreign ABI major");
  bad = cfg;
  bad.projection = 99;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_INVALID_PARAMETER, "unknown projection");
  bad = cfg;
  bad.tangent = 99;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_INVALID_PARAMETER, "unknown tangent");
  bad = cfg;
  bad.n_max = -1;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_INVALID_PARAMETER, "negative n_max");
  bad = cfg;
  bad.n_max = N_INIT - 1;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_INVALID_PARAMETER,
        "n_max below the initial band");
  bad = cfg;
  bad.n_simul = N_MAX;
  CHECK(run(&bad, &host, out, &report) == RGSADDLE_INVALID_PARAMETER,
        "window wider than the band");
  CHECK(host.calls == 0, "a refused config never reaches the surface");

  CHECK(run(&cfg, &host, out, &report) == RGSADDLE_OK, "the valid config still runs");
  host_t failing;
  memset(&failing, 0, sizeof failing);
  failing.fail_after = 2;
  CHECK(rgsaddle_autoneb_run(&cfg, N_INIT, N_ATOMS, start, NULL, NULL, out, &report, surface,
                             &failing) == RGSADDLE_SURFACE_FAILED,
        "a surface failure is returned unchanged");
  CHECK(failing.calls > failing.fail_after, "the search reached the failing call");
  return 0;
}

int main(void) {
  if (rgsaddle_abi_version() != (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    fprintf(stderr, "ABI minor disagrees with the header\n");
    return 1;
  }
  if (test_converged_run() || test_exhausted_budget() || test_without_climb() ||
      test_error_paths()) {
    return 1;
  }
  puts("RGSADDLE_AUTONEB_ABI_OK");
  return 0;
}
