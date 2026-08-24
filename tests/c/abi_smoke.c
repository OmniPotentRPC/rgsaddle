/* Drives the rgsaddle C ABI over a double-well band: create, step to
 * convergence, read positions back, reset, free. Built and run by
 * tests/c_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR) {
    return -1;
  }
  const long dof = 3 * (long)req->n_atoms;
  for (long i = 0; i < (long)req->n_images; ++i) {
    const double *p = req->positions + i * dof;
    double *g = req->gradients + i * dof;
    double x = p[0], y = p[1], z = p[2];
    req->energies[i] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
  }
  return 0;
}

int main(void) {
  if (rgsaddle_abi_version() !=
      (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    fprintf(stderr, "abi version mismatch\n");
    return 1;
  }
  rgsaddle_version_t stamp = {0, 0};
  if (rgsaddle_abi_stamp(&stamp) != RGSADDLE_OK || stamp.major != 1) {
    fprintf(stderr, "abi stamp failed\n");
    return 1;
  }
  if (rgsaddle_abi_stamp(NULL) != RGSADDLE_INVALID_PARAMETER) {
    fprintf(stderr, "NULL stamp must fail closed\n");
    return 1;
  }

  const long n_images = 9, n_atoms = 1;
  double pos[9 * 3];
  for (long i = 0; i < n_images; ++i) {
    double t = (double)i / (double)(n_images - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = 0.3 * sin(3.14159265358979 * t);
    pos[i * 3 + 2] = 0.0;
  }

  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.ci_trigger_factor = 0.5;
  cfg.ci_trigger_force = 0.0;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.1;

  /* An unknown major must be refused. */
  rgsaddle_band_config_t bad = cfg;
  bad.version.major = 99;
  if (rgsaddle_band_create(&bad, n_images, n_atoms, pos) != NULL) {
    fprintf(stderr, "unknown major must be refused\n");
    return 1;
  }

  RgsaddleBand *band = rgsaddle_band_create(&cfg, n_images, n_atoms, pos);
  if (!band) {
    fprintf(stderr, "band create failed\n");
    return 1;
  }
  if (rgsaddle_band_step(band, NULL, NULL, NULL) != RGSADDLE_NULL_REPORT) {
    fprintf(stderr, "NULL report must fail closed\n");
    return 1;
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int steps = 0;
  do {
    int rc = rgsaddle_band_step(band, surface, NULL, &rep);
    if (rc != RGSADDLE_OK) {
      fprintf(stderr, "step rc=%d (%s)\n", rc, rgsaddle_status_name(rc));
      return 1;
    }
    ++steps;
  } while (rep.status == RGSADDLE_STATUS_RUNNING && steps < 3000);

  if (rep.status != RGSADDLE_STATUS_CONVERGED) {
    fprintf(stderr, "did not converge, max_force=%g\n", rep.max_force);
    return 1;
  }
  if (rep.version.major != RGSADDLE_ABI_MAJOR) {
    fprintf(stderr, "report not stamped\n");
    return 1;
  }
  if (rep.ci_index < 1 || rep.ci_index > n_images - 2) {
    fprintf(stderr, "climbing image not armed: %lld\n", (long long)rep.ci_index);
    return 1;
  }

  double out[9 * 3];
  if (rgsaddle_band_positions(band, out) != RGSADDLE_OK) {
    fprintf(stderr, "positions failed\n");
    return 1;
  }
  if (fabs(out[0] + 1.0) > 1e-12 || fabs(out[(n_images - 1) * 3] - 1.0) > 1e-12) {
    fprintf(stderr, "endpoints moved\n");
    return 1;
  }
  if (fabs(out[rep.ci_index * 3]) > 0.15) {
    fprintf(stderr, "climbing image off the saddle: %g\n", out[rep.ci_index * 3]);
    return 1;
  }

  if (rgsaddle_band_reset(band) != RGSADDLE_OK) {
    fprintf(stderr, "reset failed\n");
    return 1;
  }
  rgsaddle_band_free(band);
  rgsaddle_band_free(NULL);
  printf("RGSADDLE_C_ABI_OK steps=%d ci=%lld\n", steps, (long long)rep.ci_index);
  return 0;
}
