/* rgsaddle_band_set_hvp and rgsaddle_minmode_set_hvp (ABI minor 47).
 * Built and run by tests/analytic_hvp_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static const double ANALYTIC_FINITE_DIFFERENCE_GAP = 1e-5;
static const double CURVATURE_GAP = 1e-6;

#define N_IMAGES 4
#define N_ATOMS 1
#define DOF (3 * N_ATOMS)

typedef struct {
  int evals;
  int hvps;
  int expect_dof;
} Counters;

static int fail(const char *msg, rgsaddle_status_t st) {
  const char *detail = rgsaddle_last_error();
  fprintf(stderr, "%s (%s): %s\n", msg, rgsaddle_status_name(st),
          detail ? detail : "");
  return 1;
}

static void quadratic(const double *p, double *e, double *g) {
  const double x = p[0], y = p[1], z = p[2];
  *e = -x * x + 2.0 * y * y + 3.0 * z * z;
  g[0] = -2.0 * x;
  g[1] = 4.0 * y;
  g[2] = 6.0 * z;
}

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  Counters *ctr = user;
  ctr->evals += 1;
  const int64_t rows =
      (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  for (int64_t r = 0; r < rows; ++r) {
    quadratic(req->positions + r * dof, req->energies + r,
              req->gradients + r * dof);
  }
  return RGSADDLE_OK;
}

static int band_hvp(void *user, int64_t image, int64_t n_dof,
                     const double *position, const double *vector,
                     double *action) {
  (void)image;
  (void)position;
  Counters *ctr = user;
  if (n_dof != ctr->expect_dof) {
    return RGSADDLE_SHAPE;
  }
  ctr->hvps += 1;
  action[0] = -2.0 * vector[0];
  action[1] = 4.0 * vector[1];
  action[2] = 6.0 * vector[2];
  return RGSADDLE_OK;
}

static int point_hvp(void *user, int64_t n_dof, const double *position,
                      const double *vector, double *action) {
  (void)position;
  Counters *ctr = user;
  if (n_dof != ctr->expect_dof) {
    return RGSADDLE_SHAPE;
  }
  ctr->hvps += 1;
  action[0] = -2.0 * vector[0];
  action[1] = 4.0 * vector[1];
  action[2] = 6.0 * vector[2];
  return RGSADDLE_OK;
}

static void band_geometry(double *pos) {
  for (int i = 0; i < N_IMAGES; ++i) {
    const double t = (double)i / (double)(N_IMAGES - 1);
    pos[i * DOF] = -1.0 + 2.0 * t;
    pos[i * DOF + 1] = 0.2;
    pos[i * DOF + 2] = -0.15;
  }
}

static rgsaddle_band_config_t band_config(void) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_RTR;
  cfg.spring_k = 0.0;
  cfg.ci_trigger_factor = 0.0;
  cfg.force_tol = 1e-12;
  cfg.max_move = 0.2;
  return cfg;
}

static int step_band(RgsaddleBand *band, Counters *ctr, const char *label) {
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  rgsaddle_status_t st = rgsaddle_band_step(band, surface, ctr, &rep);
  if (st != RGSADDLE_OK) {
    return fail(label, st);
  }
  return 0;
}

static int band_round_trip(void) {
  double pos[N_IMAGES * DOF];
  band_geometry(pos);
  const rgsaddle_band_config_t cfg = band_config();
  Counters analytic = {.evals = 0, .hvps = 0, .expect_dof = DOF};
  Counters finite = {.evals = 0, .hvps = 0, .expect_dof = DOF};

  if (rgsaddle_band_set_hvp(NULL, band_hvp, &analytic, DOF) !=
      RGSADDLE_NULL_BAND) {
    return fail("a null band must be refused", RGSADDLE_NULL_BAND);
  }

  RgsaddleBand *with_hvp = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  RgsaddleBand *without = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  if (!with_hvp || !without) {
    return fail("band create", RGSADDLE_SHAPE);
  }
  if (rgsaddle_band_set_hvp(with_hvp, band_hvp, &analytic, DOF - 1) !=
          RGSADDLE_SHAPE ||
      rgsaddle_band_set_hvp(with_hvp, band_hvp, &analytic, -1) !=
          RGSADDLE_SHAPE) {
    return fail("a band length other than 3 * n_atoms must be refused",
                RGSADDLE_SHAPE);
  }
  if (rgsaddle_band_set_hvp(with_hvp, band_hvp, &analytic, DOF) != RGSADDLE_OK) {
    return fail("band set_hvp", RGSADDLE_OK);
  }
  if (step_band(with_hvp, &analytic, "analytic band step") != 0 ||
      step_band(without, &finite, "finite difference band step") != 0) {
    return 1;
  }
  if (analytic.evals != 2 || analytic.hvps < 1) {
    fprintf(stderr, "analytic band evals %d hvps %d\n", analytic.evals,
            analytic.hvps);
    return 1;
  }
  if (finite.evals <= analytic.evals || finite.hvps != 0) {
    fprintf(stderr, "finite difference band evals %d hvps %d\n", finite.evals,
            finite.hvps);
    return 1;
  }
  double left[N_IMAGES * DOF], right[N_IMAGES * DOF];
  if (rgsaddle_band_positions(with_hvp, left) != RGSADDLE_OK ||
      rgsaddle_band_positions(without, right) != RGSADDLE_OK) {
    return fail("band positions", RGSADDLE_OK);
  }
  for (int i = 0; i < N_IMAGES * DOF; ++i) {
    if (fabs(left[i] - right[i]) > ANALYTIC_FINITE_DIFFERENCE_GAP) {
      fprintf(stderr, "band gap %g at %d\n", fabs(left[i] - right[i]), i);
      return 1;
    }
  }

  analytic.evals = 0;
  analytic.hvps = 0;
  if (rgsaddle_band_set_hvp(with_hvp, NULL, NULL, 0) != RGSADDLE_OK) {
    return fail("clearing the band callback", RGSADDLE_OK);
  }
  if (step_band(with_hvp, &analytic, "cleared band step") != 0) {
    return 1;
  }
  if (analytic.hvps != 0 || analytic.evals <= 2) {
    fprintf(stderr, "cleared band evals %d hvps %d\n", analytic.evals,
            analytic.hvps);
    return 1;
  }
  rgsaddle_band_free(with_hvp);
  rgsaddle_band_free(without);
  return 0;
}

static int minmode_round_trip(void) {
  Counters analytic = {.evals = 0, .hvps = 0, .expect_dof = DOF};
  if (rgsaddle_minmode_set_hvp(NULL, point_hvp, &analytic, DOF) !=
      RGSADDLE_NULL_SESSION) {
    return fail("a null minimum mode session must be refused",
                RGSADDLE_NULL_SESSION);
  }
  rgsaddle_minmode_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.kind = RGSADDLE_MINMODE_LANCZOS;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.dr = 1e-3;
  cfg.rotation_tol = 1e-8;
  cfg.max_rotations = 8;
  cfg.krylov_dim = 3;
  cfg.force_tol = 1e-4;
  cfg.max_move = 0.2;
  const double position[DOF] = {0.2, 0.2, 0.1};
  const double mode[DOF] = {0.1, 1.0, 0.2};
  RgsaddleMinMode *session =
      rgsaddle_minmode_create(&cfg, N_ATOMS, position, mode);
  if (!session) {
    return fail("minimum mode create", RGSADDLE_SHAPE);
  }
  if (rgsaddle_minmode_set_hvp(session, point_hvp, &analytic, 1) !=
          RGSADDLE_SHAPE ||
      rgsaddle_minmode_set_hvp(session, point_hvp, &analytic, DOF + 3) !=
          RGSADDLE_SHAPE) {
    return fail("a minimum mode length other than 3 * n_atoms must be refused",
                RGSADDLE_SHAPE);
  }
  if (rgsaddle_minmode_set_hvp(session, point_hvp, &analytic, DOF) !=
      RGSADDLE_OK) {
    return fail("minimum mode set_hvp", RGSADDLE_OK);
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  for (int i = 0; i < 3; ++i) {
    rgsaddle_status_t st =
        rgsaddle_minmode_estimate(session, surface, &analytic, &rep);
    if (st != RGSADDLE_OK) {
      return fail("analytic minimum mode estimate", st);
    }
  }
  if (fabs(rep.curvature + 2.0) > CURVATURE_GAP || analytic.evals != 1 ||
      analytic.hvps < 3) {
    fprintf(stderr, "curvature %g evals %d hvps %d\n", rep.curvature,
            analytic.evals, analytic.hvps);
    return 1;
  }

  if (rgsaddle_minmode_reset(session) != RGSADDLE_OK) {
    return fail("reset before the length refusal", RGSADDLE_OK);
  }
  analytic.expect_dof = 1;
  analytic.evals = 0;
  analytic.hvps = 0;
  if (rgsaddle_minmode_set_hvp(session, point_hvp, &analytic, DOF) !=
      RGSADDLE_OK) {
    return fail("install a callback that refuses the session length",
                RGSADDLE_OK);
  }
  rgsaddle_status_t mismatch =
      rgsaddle_minmode_estimate(session, surface, &analytic, &rep);
  if (mismatch != RGSADDLE_SHAPE) {
    return fail("a callback that refuses n_dof must surface as SHAPE",
                mismatch);
  }

  if (rgsaddle_minmode_reset(session) != RGSADDLE_OK ||
      rgsaddle_minmode_set_hvp(session, NULL, NULL, -1) != RGSADDLE_OK) {
    return fail("clearing the minimum mode callback", RGSADDLE_OK);
  }
  analytic.expect_dof = DOF;
  analytic.evals = 0;
  analytic.hvps = 0;
  rgsaddle_status_t cleared =
      rgsaddle_minmode_estimate(session, surface, &analytic, &rep);
  if (cleared != RGSADDLE_OK) {
    return fail("cleared minimum mode estimate", cleared);
  }
  if (analytic.hvps != 0 || analytic.evals <= 1) {
    fprintf(stderr, "cleared minimum mode evals %d hvps %d\n", analytic.evals,
            analytic.hvps);
    return 1;
  }
  rgsaddle_minmode_free(session);
  return 0;
}

int main(void) {
  if (band_round_trip() != 0 || minmode_round_trip() != 0) {
    return 1;
  }
  printf("RGSADDLE_ANALYTIC_HVP_ABI_OK\n");
  return 0;
}
