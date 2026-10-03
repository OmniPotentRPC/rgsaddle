/* Drives the rgsaddle C ABI over a double-well band: create, step to
 * convergence, read positions back, reset, free. Built and run by
 * tests/c_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int surface_one_calls;

static rgsaddle_status_t fail(rgsaddle_status_t st, const char *msg) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(st));
  return st;
}

static rgsaddle_status_t surface_one(void *user, rgsaddle_surface_request_t *req) {
  int *seen = user;
  if ((req->flags & RGSADDLE_REQ_ONE_IMAGE) == 0 || req->image < 0) {
    return RGSADDLE_SHAPE;
  }
  if (req->image >= req->n_images) {
    return RGSADDLE_SHAPE;
  }
  seen[req->image] += 1;
  surface_one_calls += 1;
  const long dof = 3 * (long)req->n_atoms;
  const double *p = req->positions;
  double *g = req->gradients;
  double x = p[0], y = p[1], z = p[2];
  req->energies[0] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
  g[0] = 4.0 * x * (x * x - 1.0);
  g[1] = 4.0 * y;
  g[2] = 4.0 * z;
  (void)dof;
  return RGSADDLE_OK;
}

static rgsaddle_status_t surface_quad(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->n_images != 1 || req->n_atoms != 1 || req->image != -1) {
    return RGSADDLE_SHAPE;
  }
  const double x = req->positions[0];
  const double y = req->positions[1];
  const double z = req->positions[2];
  req->energies[0] = -x * x + y * y + z * z;
  req->gradients[0] = -2.0 * x;
  req->gradients[1] = 2.0 * y;
  req->gradients[2] = 2.0 * z;
  return RGSADDLE_OK;
}

static int quad_calls;

static rgsaddle_status_t surface_quad_counted(void *user,
                                             rgsaddle_surface_request_t *req) {
  quad_calls += 1;
  return surface_quad(user, req);
}

/* One min-mode session kept across points: set_position with the
 * host's gradient and the previous mode as seed costs one evaluation. */
static rgsaddle_status_t minmode_reuse(void) {
  rgsaddle_minmode_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.kind = RGSADDLE_MINMODE_DIMER;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.dr = 1e-3;
  cfg.rotation_tol = 1e-6;
  cfg.max_rotations = 20;
  cfg.krylov_dim = 3;
  cfg.force_tol = 1e-4;
  cfg.max_move = 0.1;
  cfg.difference = RGSADDLE_DIFFERENCE_FORWARD;
  double x[3] = {0.2, 0.1, -0.1};
  const double invalid_modes[][3] = {
      {0.0, 0.0, 0.0}, {1e-16, 0.0, 0.0}, {NAN, 0.0, 0.0},
      {INFINITY, 0.0, 0.0}, {1e308, 0.0, 0.0}};
  for (size_t i = 0; i < sizeof invalid_modes / sizeof invalid_modes[0]; ++i) {
    RgsaddleMinMode *invalid =
        rgsaddle_minmode_create(&cfg, 1, x, invalid_modes[i]);
    if (invalid) {
      rgsaddle_minmode_free(invalid);
      return fail(RGSADDLE_SOLVER, "minmode create accepted an invalid mode");
    }
  }
  double mode[3] = {0.6, 0.8, 0.0};
  RgsaddleMinMode *mm = rgsaddle_minmode_create(&cfg, 1, x, mode);
  if (!mm) {
    return fail(RGSADDLE_SHAPE, "minmode create");
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  quad_calls = 0;
  rgsaddle_status_t st = rgsaddle_minmode_estimate(mm, surface_quad_counted, NULL, &rep);
  if (st != RGSADDLE_OK || rep.evaluations != quad_calls ||
      fabs(rep.curvature + 2.0) > 1e-8) {
    fprintf(stderr, "estimate st=%d evaluations=%d calls=%d curvature=%g\n", st,
            rep.evaluations, quad_calls, rep.curvature);
    return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "minmode estimate");
  }
  double x2[3] = {0.1, -0.05, 0.02};
  double g2[3] = {-2.0 * x2[0], 2.0 * x2[1], 2.0 * x2[2]};
  if (rgsaddle_minmode_set_position(mm, x2, g2) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "minmode set_position");
  }
  quad_calls = 0;
  st = rgsaddle_minmode_estimate(mm, surface_quad_counted, NULL, &rep);
  if (st != RGSADDLE_OK || quad_calls != 1 || rep.evaluations != 1 || rep.rotations != 0) {
    fprintf(stderr, "warm estimate calls=%d evaluations=%d rotations=%lld\n",
            quad_calls, rep.evaluations, (long long)rep.rotations);
    return fail(RGSADDLE_SOLVER, "warm min-mode estimate must cost one evaluation");
  }
  double seed[3] = {0.0, 0.0, 1.0};
  if (rgsaddle_minmode_set_mode(mm, seed) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "minmode set_mode");
  }
  double back[3];
  if (rgsaddle_minmode_mode(mm, back) != RGSADDLE_OK || fabs(back[2] - 1.0) > 1e-15) {
    return fail(RGSADDLE_SHAPE, "minmode mode after set_mode");
  }
  /* A step from the reused session still climbs. */
  st = rgsaddle_minmode_step(mm, surface_quad, NULL, &rep);
  if (st != RGSADDLE_OK) {
    return fail(st, "minmode step after reuse");
  }
  rgsaddle_minmode_free(mm);
  return RGSADDLE_OK;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR) {
    return RGSADDLE_ABI_MISMATCH;
  }
  if (req->image != -1 || (req->flags & RGSADDLE_REQ_ONE_IMAGE) != 0) {
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

int main(void) {
  rgsaddle_status_t st = RGSADDLE_OK;
  if (rgsaddle_abi_version() !=
      (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    return fail(RGSADDLE_ABI_MISMATCH, "abi version mismatch");
  }
  rgsaddle_version_t stamp = {0, 0};
  st = rgsaddle_abi_stamp(&stamp);
  if (st != RGSADDLE_OK || stamp.major != RGSADDLE_ABI_MAJOR) {
    return fail(st == RGSADDLE_OK ? RGSADDLE_ABI_MISMATCH : st, "abi stamp failed");
  }
  if (rgsaddle_abi_stamp(NULL) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "NULL stamp must fail closed");
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
    return fail(RGSADDLE_ABI_MISMATCH, "unknown major must be refused");
  }

  RgsaddleBand *band = rgsaddle_band_create(&cfg, n_images, n_atoms, pos);
  if (!band) {
    return fail(RGSADDLE_SHAPE, "band create failed");
  }
  if (rgsaddle_band_step(band, NULL, NULL, NULL) != RGSADDLE_NULL_REPORT) {
    return fail(RGSADDLE_NULL_REPORT, "NULL report must fail closed");
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  int steps = 0;
  do {
    st = rgsaddle_band_step(band, surface, NULL, &rep);
    if (st != RGSADDLE_OK) {
      return fail(st, "step");
    }
    ++steps;
  } while (rep.status == RGSADDLE_STATUS_RUNNING && steps < 3000);

  if (rep.status != RGSADDLE_STATUS_CONVERGED) {
    fprintf(stderr, "did not converge, max_force=%g\n", rep.max_force);
    return fail(RGSADDLE_SOLVER, "did not converge");
  }
  if (rep.version.major != RGSADDLE_ABI_MAJOR) {
    return fail(RGSADDLE_ABI_MISMATCH, "report not stamped");
  }
  if (rep.ci_index < 1 || rep.ci_index > n_images - 2) {
    fprintf(stderr, "climbing image not armed: %lld\n", (long long)rep.ci_index);
    return fail(RGSADDLE_SHAPE, "climbing image not armed");
  }

  double out[9 * 3];
  st = rgsaddle_band_positions(band, out);
  if (st != RGSADDLE_OK) {
    return fail(st, "positions");
  }
  if (fabs(out[0] + 1.0) > 1e-12 || fabs(out[(n_images - 1) * 3] - 1.0) > 1e-12) {
    return fail(RGSADDLE_SHAPE, "endpoints moved");
  }
  if (fabs(out[rep.ci_index * 3]) > 0.15) {
    fprintf(stderr, "climbing image off the saddle: %g\n", out[rep.ci_index * 3]);
    return fail(RGSADDLE_SOLVER, "climbing image off the saddle");
  }

  /* One callback per image, so the host can swap orbitals for that
   * image. The endpoints are evaluated once, on the first evaluation;
   * every interior image is evaluated the same number of times. */
  int seen[9] = {0};
  surface_one_calls = 0;
  cfg.flags = RGSADDLE_BAND_PER_IMAGE;
  RgsaddleBand *per = rgsaddle_band_create(&cfg, n_images, n_atoms, pos);
  if (!per) {
    return fail(RGSADDLE_SHAPE, "per-image band create failed");
  }
  st = rgsaddle_band_step(per, surface_one, seen, &rep);
  if (st != RGSADDLE_OK) {
    return fail(st, "per-image step");
  }
  if (seen[0] != 1 || seen[n_images - 1] != 1) {
    fprintf(stderr, "endpoints seen %d and %d times, want 1\n", seen[0],
            seen[n_images - 1]);
    return fail(RGSADDLE_SHAPE, "endpoint evaluation count");
  }
  for (long i = 1; i < n_images - 1; ++i) {
    if (seen[i] != seen[1] || seen[i] < 1) {
      fprintf(stderr, "image %ld seen %d times, image 1 seen %d\n",
              i, seen[i], seen[1]);
      return fail(RGSADDLE_SHAPE, "per-image coverage");
    }
  }
  if (surface_one_calls != 2 + (n_images - 2) * seen[1]) {
    fprintf(stderr, "per-image calls=%d, seen[1]=%d\n", surface_one_calls,
            seen[1]);
    return fail(RGSADDLE_SHAPE, "per-image call count");
  }
  /* A second step touches the interior once each and the endpoints
   * not at all. */
  const int interior_before = seen[1];
  st = rgsaddle_band_step(per, surface_one, seen, &rep);
  if (st != RGSADDLE_OK) {
    return fail(st, "per-image second step");
  }
  if (seen[0] != 1 || seen[n_images - 1] != 1) {
    return fail(RGSADDLE_SHAPE, "endpoints re-evaluated");
  }
  for (long i = 1; i < n_images - 1; ++i) {
    if (seen[i] != interior_before + 1) {
      fprintf(stderr, "image %ld seen %d times after step 2, want %d\n", i,
              seen[i], interior_before + 1);
      return fail(RGSADDLE_SHAPE, "one interior evaluation per step");
    }
  }
  if (rep.evaluations != n_images - 2) {
    fprintf(stderr, "report evaluations %d, want %ld\n", rep.evaluations,
            n_images - 2);
    return fail(RGSADDLE_SHAPE, "report evaluations");
  }

  /* The host resyncs the band it read back and restarts the optimizer:
   * neither costs an evaluation, the next step is one interior pass. */
  double cur[9 * 3];
  st = rgsaddle_band_positions(per, cur);
  if (st != RGSADDLE_OK) {
    return fail(st, "per-image positions");
  }
  double energies[9], gradients[9 * 3], projected[7 * 3];
  st = rgsaddle_band_evaluation(per, energies, gradients, projected);
  if (st != RGSADDLE_OK) {
    return fail(st, "evaluation at the stepped band");
  }
  for (long i = 0; i < n_images; ++i) {
    const double x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2];
    const double e = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    if (fabs(energies[i] - e) > 1e-12) {
      return fail(RGSADDLE_SHAPE, "evaluation energies");
    }
  }
  if (rgsaddle_band_set_positions(per, cur) != RGSADDLE_OK ||
      rgsaddle_band_restart(per) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "resync and restart");
  }
  const int interior_mid = seen[1];
  st = rgsaddle_band_step(per, surface_one, seen, &rep);
  if (st != RGSADDLE_OK) {
    return fail(st, "per-image step after resync");
  }
  if (seen[0] != 1 || seen[1] != interior_mid + 1 ||
      rep.evaluations != n_images - 2) {
    fprintf(stderr, "after resync: seen[0]=%d seen[1]=%d evaluations=%d\n",
            seen[0], seen[1], rep.evaluations);
    return fail(RGSADDLE_SHAPE, "resync must not re-evaluate");
  }
  /* reset is a surface boundary: nothing cached survives. */
  if (rgsaddle_band_reset(per) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "reset");
  }
  if (rgsaddle_band_evaluation(per, NULL, NULL, NULL) != RGSADDLE_NO_EVALUATION) {
    return fail(RGSADDLE_SHAPE, "reset must retire the evaluation");
  }
  rgsaddle_band_free(per);

  st = rgsaddle_band_reset(band);
  if (st != RGSADDLE_OK) {
    return fail(st, "reset");
  }
  rgsaddle_band_free(band);
  rgsaddle_band_free(NULL);

  /* Raw Nichols step on V = -x^2 + y^2, the i-PI displacement. */
  {
    const double gradient[2] = {-0.8, -0.6};
    const double evals[2] = {-2.0, 2.0};
    const double evecs[4] = {1.0, 0.0, 0.0, 1.0};
    double dx[2] = {0.0, 0.0};
    st = rgsaddle_nichols_step(2, 2, gradient, evals, evecs, NULL, 10.0,
                               RGSADDLE_NICHOLS_INDEX1, dx);
    if (st != RGSADDLE_OK) {
      return fail(st, "nichols step");
    }
    if (fabs(dx[0] + 0.4) > 1e-12 || fabs(dx[1] - 0.3) > 1e-12) {
      fprintf(stderr, "nichols dx = %g %g\n", dx[0], dx[1]);
      return fail(RGSADDLE_SOLVER, "nichols step mismatch");
    }
    if (rgsaddle_nichols_step(2, 2, NULL, evals, evecs, NULL, 10.0,
                              RGSADDLE_NICHOLS_INDEX1, dx) !=
        RGSADDLE_INVALID_PARAMETER) {
      return fail(RGSADDLE_INVALID_PARAMETER, "NULL gradient must fail");
    }
    st = rgsaddle_cap_max_abs(2, dx, 0.2);
    if (st != RGSADDLE_OK || fabs(fabs(dx[0]) - 0.2) > 1e-12) {
      return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "trust cap");
    }
  }

  /* Powell update against the i-PI formula on a 2x2 Hessian. */
  {
    double h[4] = {2.0, 0.1, 0.1, 3.0};
    const double s[2] = {0.2, -0.1};
    const double y[2] = {-0.05, 0.4};
    st = rgsaddle_hessian_powell(2, s, y, h);
    if (st != RGSADDLE_OK || fabs(h[0] - 0.976) > 1e-12 || fabs(h[1] - 2.452) > 1e-12 ||
        fabs(h[2] - 2.452) > 1e-12 || fabs(h[3] - 0.904) > 1e-12) {
      fprintf(stderr, "powell H = %g %g %g %g\n", h[0], h[1], h[2], h[3]);
      return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "powell update");
    }
  }

  /* One index-1 step on V = -x^2 + y^2 + z^2 reaches the origin. */
  {
    rgsaddle_index1_config_t icfg;
    memset(&icfg, 0, sizeof icfg);
    icfg.version.major = RGSADDLE_ABI_MAJOR;
    icfg.version.minor = RGSADDLE_ABI_MINOR;
    icfg.update = RGSADDLE_HESS_BOFILL;
    icfg.mode = RGSADDLE_NICHOLS_INDEX1;
    icfg.trust_radius = 1.0;
    icfg.force_tol = 1e-8;
    icfg.fd_dr = 1e-4;
    const double x0[3] = {0.3, 0.2, -0.1};
    const double hess[9] = {-2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
    RgsaddleIndex1 *idx = rgsaddle_index1_create(&icfg, 1, x0, hess, NULL);
    if (!idx) {
      return fail(RGSADDLE_SHAPE, "index1 create failed");
    }
    rgsaddle_report_t irep;
    memset(&irep, 0, sizeof irep);
    int istep = 0;
    do {
      st = rgsaddle_index1_step(idx, surface_quad, NULL, &irep);
      if (st != RGSADDLE_OK) {
        return fail(st, "index1 step");
      }
      ++istep;
    } while (irep.status == RGSADDLE_STATUS_RUNNING && istep < 30);
    if (irep.status != RGSADDLE_STATUS_CONVERGED || irep.curvature >= 0.0) {
      fprintf(stderr, "index1 status=%d force=%g curv=%g steps=%d\n", irep.status,
              irep.max_force, irep.curvature, istep);
      return fail(RGSADDLE_SOLVER, "index1 did not converge");
    }
    double x1[3];
    st = rgsaddle_index1_position(idx, x1);
    if (st != RGSADDLE_OK || fabs(x1[0]) > 1e-8 || fabs(x1[1]) > 1e-8 || fabs(x1[2]) > 1e-8) {
      fprintf(stderr, "index1 x = %g %g %g\n", x1[0], x1[1], x1[2]);
      return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "index1 off the saddle");
    }
    rgsaddle_index1_free(idx);
    rgsaddle_index1_free(NULL);
  }

  {
    const double gradient[2] = {-0.8, -0.6};
    const double hessian[4] = {-2.0, 0.0, 0.0, 2.0};
    double dx[2] = {0.0, 0.0};
    st = rgsaddle_prfo_step(2, gradient, hessian, NULL, 10.0, RGSADDLE_PRFO_INDEX1, dx);
    if (st != RGSADDLE_OK) {
      return fail(st, "prfo step");
    }
    if (fabs(dx[0] + 0.3507810593582122) > 1e-8 || fabs(dx[1] - 0.27698396494843347) > 1e-8) {
      fprintf(stderr, "prfo dx = %g %g\n", dx[0], dx[1]);
      return fail(RGSADDLE_SOLVER, "prfo step mismatch");
    }
  }

  st = minmode_reuse();
  if (st != RGSADDLE_OK) {
    return st;
  }
  printf("RGSADDLE_C_ABI_OK steps=%d ci=%lld\n", steps, (long long)rep.ci_index);
  return RGSADDLE_OK;
}
