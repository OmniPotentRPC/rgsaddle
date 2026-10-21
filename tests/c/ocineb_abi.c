/* OCI-NEB C ABI: create, step through the hand-off, read the band. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static rgsaddle_status_t fail(rgsaddle_status_t st, const char *msg) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(st));
  return st;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != 1) {
    return RGSADDLE_SHAPE;
  }
  const long dof = 3;
  for (long i = 0; i < req->n_images; ++i) {
    const double *p = req->positions + i * dof;
    double *g = req->gradients + i * dof;
    const double x = p[0], y = p[1], z = p[2];
    req->energies[i] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
  }
  return RGSADDLE_OK;
}

int main(void) {
  if (rgsaddle_ocineb_step(NULL, NULL, NULL, NULL) != RGSADDLE_NULL_SESSION) {
    return fail(RGSADDLE_NULL_SESSION, "null step");
  }
  if (rgsaddle_ocineb_create(NULL, NULL, NULL, 0, 0, NULL) != NULL) {
    return fail(RGSADDLE_SHAPE, "null create");
  }

  const long n_images = 9, n_atoms = 1;
  double pos[9 * 3];
  memset(pos, 0, sizeof pos);
  for (long i = 0; i < n_images; ++i) {
    const double t = (double)i / (double)(n_images - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = 0.3 * sin(3.141592653589793 * t);
  }

  rgsaddle_band_config_t band;
  memset(&band, 0, sizeof band);
  band.version.major = RGSADDLE_ABI_MAJOR;
  band.version.minor = RGSADDLE_ABI_MINOR;
  band.tangent = RGSADDLE_TANGENT_IMPROVED;
  band.spring = RGSADDLE_SPRING_UNIFORM;
  band.projection = RGSADDLE_PROJECTION_NEB;
  band.method = RGSADDLE_METHOD_FIRE;
  band.spring_k = 5.0;
  band.ci_trigger_factor = 0.5;
  band.force_tol = 1e-3;
  band.max_move = 0.1;

  rgsaddle_minmode_config_t mm;
  memset(&mm, 0, sizeof mm);
  mm.version.major = RGSADDLE_ABI_MAJOR;
  mm.version.minor = RGSADDLE_ABI_MINOR;
  mm.kind = RGSADDLE_MINMODE_DIMER;
  mm.method = RGSADDLE_METHOD_FIRE;
  mm.dr = 1e-3;
  mm.rotation_tol = 1e-4;
  mm.max_rotations = 20;
  mm.krylov_dim = 8;
  mm.force_tol = 1e-3;
  mm.max_move = 0.1;
  mm.difference = RGSADDLE_DIFFERENCE_FORWARD;

  rgsaddle_ocineb_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.trigger_factor = 0.31;
  cfg.angle_tol = 0.5;
  cfg.stability_count = 5;
  cfg.max_mmf_steps = 1000;
  if (rgsaddle_ocineb_create(&band, &mm, &cfg, n_images, n_atoms, pos) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "angle below 1/sqrt(2) was accepted");
  }
  cfg.angle_tol = 0.85;

  RgsaddleOcineb *session =
      rgsaddle_ocineb_create(&band, &mm, &cfg, n_images, n_atoms, pos);
  if (!session) {
    return fail(RGSADDLE_ALLOC, "create");
  }

  int saw_align = 0, saw_minmode = 0, saw_fallback = 0;
  rgsaddle_ocineb_report_t report;
  memset(&report, 0, sizeof report);
  for (int i = 0; i < 4000; ++i) {
    rgsaddle_status_t st = rgsaddle_ocineb_step(session, surface, NULL, &report);
    if (st != RGSADDLE_OK) {
      rgsaddle_ocineb_free(session);
      return fail(st, "step");
    }
    if (report.version.major != RGSADDLE_ABI_MAJOR) {
      rgsaddle_ocineb_free(session);
      return fail(RGSADDLE_ABI_MISMATCH, "report version");
    }
    if (report.phase == RGSADDLE_OCINEB_ALIGN) {
      saw_align = 1;
    }
    if (report.phase == RGSADDLE_OCINEB_MINMODE) {
      saw_minmode = 1;
    }
    if (report.fell_back) {
      saw_fallback = 1;
    }
    if (report.status == RGSADDLE_STATUS_CONVERGED) {
      break;
    }
  }
  if (report.status != RGSADDLE_STATUS_CONVERGED) {
    fprintf(stderr, "did not converge, force=%g iter=%lld\n", report.max_force,
            (long long)report.iteration);
    rgsaddle_ocineb_free(session);
    return 1;
  }
  if (!saw_align || !saw_minmode) {
    fprintf(stderr, "phases align=%d minmode=%d fallback=%d\n", saw_align, saw_minmode,
            saw_fallback);
    rgsaddle_ocineb_free(session);
    return 1;
  }

  double out[9 * 3];
  if (rgsaddle_ocineb_positions(session, out) != RGSADDLE_OK) {
    rgsaddle_ocineb_free(session);
    return fail(RGSADDLE_SHAPE, "positions");
  }
  if (out[0] != -1.0 || out[(n_images - 1) * 3] != 1.0) {
    fprintf(stderr, "endpoints moved\n");
    rgsaddle_ocineb_free(session);
    return 1;
  }
  if (report.ci_index < 1 || report.ci_index >= n_images - 1) {
    fprintf(stderr, "ci index %lld\n", (long long)report.ci_index);
    rgsaddle_ocineb_free(session);
    return 1;
  }
  const double *ci = out + report.ci_index * 3;
  if (fabs(ci[0]) > 0.05 || fabs(ci[1]) > 0.05 || fabs(ci[2]) > 0.05) {
    fprintf(stderr, "saddle %g %g %g\n", ci[0], ci[1], ci[2]);
    rgsaddle_ocineb_free(session);
    return 1;
  }

  if (rgsaddle_ocineb_reset(session) != RGSADDLE_OK) {
    rgsaddle_ocineb_free(session);
    return fail(RGSADDLE_SOLVER, "reset");
  }
  rgsaddle_ocineb_free(session);
  printf("RGSADDLE_OCINEB_ABI_OK force=%.6g iter=%lld\n", report.max_force,
         (long long)report.iteration);
  return 0;
}
