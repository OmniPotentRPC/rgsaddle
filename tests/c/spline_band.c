/* Band session with RGSADDLE_TANGENT_SPLINE on the double well.
 * One step and the converged band match rgsaddle_band_project. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(int st, const char *msg) {
  fprintf(stderr, "FAIL: %s (status=%d %s)\n", msg, st,
          rgsaddle_status_name((rgsaddle_status_t)st));
  return 1;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->image != -1) {
    return RGSADDLE_SHAPE;
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
  return RGSADDLE_OK;
}

static void bow(double *pos, long n_images) {
  for (long i = 0; i < n_images; ++i) {
    double t = (double)i / (double)(n_images - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = 0.3 * sin(3.141592653589793 * t);
    pos[i * 3 + 2] = 0.0;
  }
}

static int forces_match(const rgsaddle_band_config_t *cfg, RgsaddleBand *band, long n_images,
                        long n_atoms) {
  const long dof = 3 * n_atoms;
  const long interior = (n_images - 2) * dof;
  double positions[7 * 3];
  double energies[7];
  double gradients[7 * 3];
  double projected[5 * 3];
  double forces[7 * 3];
  long ci = -1;
  if (n_images != 7 || n_atoms != 1) {
    return fail(RGSADDLE_SHAPE, "fixture size");
  }
  int st = rgsaddle_band_positions(band, positions);
  if (st != RGSADDLE_OK) {
    return fail(st, "positions");
  }
  st = rgsaddle_band_evaluation(band, energies, gradients, projected);
  if (st != RGSADDLE_OK) {
    return fail(st, "evaluation");
  }
  st = rgsaddle_band_project(cfg, n_images, n_atoms, positions, energies, gradients, forces, &ci);
  if (st != RGSADDLE_OK) {
    return fail(st, "project");
  }
  for (long c = 0; c < dof; ++c) {
    if (forces[c] != 0.0 || forces[(n_images - 1) * dof + c] != 0.0) {
      return fail(RGSADDLE_SOLVER, "endpoint force");
    }
  }
  for (long i = 0; i < interior; ++i) {
    long row = i / dof;
    long col = i % dof;
    double want = forces[(row + 1) * dof + col];
    if (fabs(projected[i] - want) > 1e-9) {
      fprintf(stderr, "force[%ld] session=%g project=%g\n", i, projected[i], want);
      return fail(RGSADDLE_SOLVER, "session force differs from the one-shot projection");
    }
  }
  return 0;
}

int main(void) {
  const long n_images = 7;
  const long n_atoms = 1;
  double pos[7 * 3];
  bow(pos, n_images);

  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_NO_MIN_MODE | RGSADDLE_BAND_STRING;
  cfg.tangent = RGSADDLE_TANGENT_SPLINE;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.1;
  if (rgsaddle_band_create(&cfg, n_images, n_atoms, pos) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "string plus spline must be refused");
  }

  cfg.flags = RGSADDLE_BAND_NO_MIN_MODE;
  RgsaddleBand *band = rgsaddle_band_create(&cfg, n_images, n_atoms, pos);
  if (band == NULL) {
    return fail(RGSADDLE_SHAPE, "spline band create");
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int st = rgsaddle_band_step(band, surface, NULL, &rep);
  if (st != RGSADDLE_OK) {
    return fail(st, "first spline step");
  }
  if (forces_match(&cfg, band, n_images, n_atoms) != 0) {
    return 1;
  }

  int steps = 1;
  while (rep.status == RGSADDLE_STATUS_RUNNING && steps < 4000) {
    st = rgsaddle_band_step(band, surface, NULL, &rep);
    if (st != RGSADDLE_OK) {
      return fail(st, "spline step");
    }
    ++steps;
  }
  if (rep.status != RGSADDLE_STATUS_CONVERGED || rep.max_force > 1e-3) {
    fprintf(stderr, "not converged after %d steps, max_force=%g\n", steps, rep.max_force);
    return fail(RGSADDLE_SOLVER, "spline band did not converge");
  }
  if (forces_match(&cfg, band, n_images, n_atoms) != 0) {
    return 1;
  }

  double out[7 * 3];
  st = rgsaddle_band_positions(band, out);
  if (st != RGSADDLE_OK) {
    return fail(st, "final positions");
  }
  if (fabs(out[0] + 1.0) > 1e-12 || fabs(out[(n_images - 1) * 3] - 1.0) > 1e-12) {
    return fail(RGSADDLE_SHAPE, "endpoints moved");
  }
  rgsaddle_band_free(band);
  printf("RGSADDLE_SPLINE_BAND_ABI_OK\n");
  return 0;
}
