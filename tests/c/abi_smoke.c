/* Drives the rgsaddle C ABI over a double-well band: create, step to
 * convergence, read positions back, reset, free. Built and run by
 * tests/c_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
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
  if (xts_minmode_estimate(NULL, NULL, NULL, NULL) != RGSADDLE_NULL_SESSION) {
    return fail(RGSADDLE_NULL_SESSION, "xts_minmode_estimate null session");
  }
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
  st = xts_minmode_estimate(mm, surface_quad_counted, NULL, &rep);
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

/* Climbing-image requests the batched surface saw. A band configured
 * below minor 22 must not set the flag. */
typedef struct {
  int flag_allowed;
  int calls;
  long image;
} climbing_seen_t;

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  climbing_seen_t *climbing = user;
  if (req->version.major != RGSADDLE_ABI_MAJOR) {
    return RGSADDLE_ABI_MISMATCH;
  }
  if ((req->flags & RGSADDLE_REQ_ONE_IMAGE) != 0 ||
      (req->flags & ~(RGSADDLE_REQ_ONE_IMAGE | RGSADDLE_REQ_CLIMBING_IMAGE)) != 0) {
    return RGSADDLE_SHAPE;
  }
  if ((req->flags & RGSADDLE_REQ_CLIMBING_IMAGE) != 0) {
    if (!climbing->flag_allowed || req->n_images != 1 || req->image < 1) {
      return RGSADDLE_SHAPE;
    }
    climbing->calls += 1;
    climbing->image = (long)req->image;
  } else if (req->image != -1) {
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

static rgsaddle_status_t point_double_well(void *user, int64_t n_dof,
                                           const double *x, double *energy,
                                           double *gradient) {
  (void)user;
  if (n_dof < 1 || !x || !energy || !gradient) {
    return RGSADDLE_INVALID_PARAMETER;
  }
  const double q = x[0];
  *energy = (1.0 - q * q) * (1.0 - q * q);
  for (int64_t i = 0; i < n_dof; ++i) {
    gradient[i] = 0.0;
  }
  gradient[0] = -4.0 * q * (1.0 - q * q);
  return RGSADDLE_OK;
}

static rgsaddle_status_t point_quadratic_saddle(void *user, int64_t n_dof,
                                                const double *x, double *energy,
                                                double *gradient) {
  (void)user;
  if (n_dof != 2 || !x || !energy || !gradient) {
    return RGSADDLE_INVALID_PARAMETER;
  }
  *energy = 0.5 * (-x[0] * x[0] + 4.0 * x[1] * x[1]);
  gradient[0] = -x[0];
  gradient[1] = 4.0 * x[1];
  return RGSADDLE_OK;
}

static rgsaddle_status_t pafi_and_ohtst(void) {
  rgsaddle_pafi_config_t pcfg;
  memset(&pcfg, 0, sizeof pcfg);
  pcfg.version.major = RGSADDLE_ABI_MAJOR;
  pcfg.version.minor = RGSADDLE_ABI_MINOR;
  pcfg.thermal = 0.02;
  pcfg.mobility = 1.0;
  pcfg.time_step = 0.01;
  pcfg.thermalize = 5;
  pcfg.stride = 1;
  pcfg.samples = 16;
  pcfg.blocks = 2;
  pcfg.seed = 3;
  pcfg.fix_com = 0;
  pcfg.retain = 1;
  pcfg.target_error = 0.0;
  pcfg.max_rounds = 1;

  double frames[41];
  for (int i = 0; i < 41; ++i) {
    frames[i] = -1.0 + 2.0 * (double)i / 40.0;
  }
  if (rgsaddle_pafi_create(NULL, 41, 1, frames, NULL, 0) != NULL ||
      rgsaddle_pafi_create(&pcfg, 1, 1, frames, NULL, 0) != NULL ||
      rgsaddle_pafi_create(&pcfg, 41, 0, frames, NULL, 0) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "pafi create bad args");
  }
  rgsaddle_pafi_config_t bad_pcfg = pcfg;
  bad_pcfg.thermal = -1.0;
  if (rgsaddle_pafi_create(&bad_pcfg, 41, 1, frames, NULL, 0) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "pafi create negative thermal");
  }
  bad_pcfg = pcfg;
  bad_pcfg.flags = 1;
  if (rgsaddle_pafi_create(&bad_pcfg, 41, 1, frames, NULL, 0) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "pafi create reserved flags");
  }
  RgsaddlePafi *pafi = rgsaddle_pafi_create(&pcfg, 41, 1, frames, NULL, 0);
  if (!pafi) {
    return fail(RGSADDLE_SHAPE, "pafi create");
  }
  rgsaddle_pafi_result_t pres;
  memset(&pres, 0, sizeof pres);
  if (rgsaddle_pafi_result(NULL, &pres) != RGSADDLE_NULL_SESSION ||
      rgsaddle_pafi_result(pafi, NULL) != RGSADDLE_NULL_REPORT ||
      rgsaddle_pafi_result(pafi, &pres) != RGSADDLE_NO_EVALUATION ||
      rgsaddle_pafi_run(NULL, point_double_well, NULL) != RGSADDLE_NULL_SESSION ||
      rgsaddle_pafi_run(pafi, NULL, NULL) != RGSADDLE_NULL_SURFACE ||
      rgsaddle_pafi_set_config(pafi, &bad_pcfg) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "pafi pre-run guards");
  }
  if (rgsaddle_pafi_run(pafi, point_double_well, NULL) != RGSADDLE_OK ||
      rgsaddle_pafi_result(pafi, &pres) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "pafi run/result");
  }
  if (fabs(pres.barrier - 0.9975) > 1e-12 || pres.dividing != 20) {
    fprintf(stderr, "pafi barrier=%g dividing=%lld\n", pres.barrier,
            (long long)pres.dividing);
    return fail(RGSADDLE_SOLVER, "pafi barrier");
  }
  int64_t pwritten = 0;
  if (rgsaddle_pafi_profile(pafi, NULL, NULL, NULL, NULL, NULL, -1, &pwritten) !=
          RGSADDLE_SHAPE ||
      rgsaddle_pafi_profile(pafi, NULL, NULL, NULL, NULL, NULL, 0, &pwritten) !=
          RGSADDLE_INVALID_PARAMETER ||
      pwritten != 41) {
    return fail(RGSADDLE_SHAPE, "pafi profile query");
  }
  double parcs[41], pgrad[41], pfree[41];
  if (rgsaddle_pafi_profile(pafi, parcs, pgrad, NULL, pfree, NULL, 41,
                            &pwritten) != RGSADDLE_OK ||
      pwritten != 41 || fabs(pfree[20] - 0.9975) > 1e-12) {
    return fail(RGSADDLE_SOLVER, "pafi profile values");
  }
  rgsaddle_pafi_free(pafi);
  rgsaddle_pafi_free(NULL);

  rgsaddle_ohtst_config_t ocfg;
  memset(&ocfg, 0, sizeof ocfg);
  ocfg.version.major = RGSADDLE_ABI_MAJOR;
  ocfg.version.minor = RGSADDLE_ABI_MINOR;
  ocfg.thermal = 0.5;
  ocfg.sample_time_step = 0.1;
  ocfg.sample_friction = 1.0;
  ocfg.sample_burn_in = 300;
  ocfg.sample_stride = 5;
  ocfg.sample_records = 1000;
  ocfg.sample_blocks = 16;
  ocfg.sample_seed = 9;
  ocfg.sample_retain = 512;
  ocfg.sample_integrator = RGSADDLE_OHTST_INTEGRATOR_BAOAB;
  ocfg.shooting_friction = 1.5;
  ocfg.shooting_time_step = 0.05;
  ocfg.shooting_steps = 400;
  ocfg.shooting_check_every = 40;
  ocfg.shooting_commit = 0;
  ocfg.shooting_shots = 4096;
  ocfg.shooting_blocks = 32;
  ocfg.shooting_seed = 17;
  ocfg.counting = RGSADDLE_OHTST_COUNT_FINAL;

  const double point[2] = {0.0, 0.0};
  const double normal[2] = {1.0, 0.0};
  const double zero_normal[2] = {0.0, 0.0};
  if (rgsaddle_ohtst_create(NULL, 2, point, normal) != NULL ||
      rgsaddle_ohtst_create(&ocfg, 2, point, zero_normal) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "ohtst create bad args");
  }
  rgsaddle_ohtst_config_t bad_ocfg = ocfg;
  bad_ocfg.sample_retain = 0;
  if (rgsaddle_ohtst_create(&bad_ocfg, 2, point, normal) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "ohtst create zero retain");
  }
  bad_ocfg = ocfg;
  bad_ocfg.flags = 1;
  if (rgsaddle_ohtst_create(&bad_ocfg, 2, point, normal) != NULL) {
    return fail(RGSADDLE_INVALID_PARAMETER, "ohtst create reserved flags");
  }
  RgsaddleOhtst *ohtst = rgsaddle_ohtst_create(&ocfg, 2, point, normal);
  if (!ohtst) {
    return fail(RGSADDLE_SHAPE, "ohtst create");
  }
  rgsaddle_ohtst_result_t ores;
  memset(&ores, 0, sizeof ores);
  if (rgsaddle_ohtst_result(NULL, &ores) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ohtst_result(ohtst, NULL) != RGSADDLE_NULL_REPORT ||
      rgsaddle_ohtst_result(ohtst, &ores) != RGSADDLE_NO_EVALUATION ||
      rgsaddle_ohtst_run(NULL, point_quadratic_saddle, NULL) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ohtst_run(ohtst, NULL, NULL) != RGSADDLE_NULL_SURFACE ||
      rgsaddle_ohtst_set_config(ohtst, &bad_ocfg) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "ohtst pre-run guards");
  }
  if (rgsaddle_ohtst_run(ohtst, point_quadratic_saddle, NULL) != RGSADDLE_OK ||
      rgsaddle_ohtst_result(ohtst, &ores) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "ohtst run/result");
  }
  const double kramers = sqrt(1.0 + 0.75 * 0.75) - 0.75;
  if (fabs(ores.kappa - kramers) > 4.0 * ores.error + 1e-12) {
    return fail(RGSADDLE_SOLVER, "ohtst kappa");
  }
  int64_t owritten = 0;
  if (rgsaddle_ohtst_curve(ohtst, NULL, NULL, -1, &owritten) != RGSADDLE_SHAPE ||
      rgsaddle_ohtst_curve(ohtst, NULL, NULL, 0, &owritten) !=
          RGSADDLE_INVALID_PARAMETER ||
      owritten < 1 || owritten > 16) {
    return fail(RGSADDLE_SHAPE, "ohtst curve query");
  }
  double times[16], kappas[16];
  if (rgsaddle_ohtst_curve(ohtst, times, kappas, 16, &owritten) != RGSADDLE_OK ||
      fabs(kappas[owritten - 1] - kramers) > 4.0 * ores.error + 1e-12) {
    return fail(RGSADDLE_SOLVER, "ohtst curve values");
  }
  rgsaddle_ohtst_free(ohtst);
  rgsaddle_ohtst_free(NULL);
  return RGSADDLE_OK;
}

/* Every set_config entry point refuses a NULL session first, then a NULL
 * config, then another major, and accepts its own config afterwards. */
static rgsaddle_status_t set_config_statuses(void) {
  rgsaddle_pafi_config_t pcfg;
  memset(&pcfg, 0, sizeof pcfg);
  pcfg.version.major = RGSADDLE_ABI_MAJOR;
  pcfg.version.minor = RGSADDLE_ABI_MINOR;
  pcfg.thermal = 0.02;
  pcfg.mobility = 1.0;
  pcfg.time_step = 0.01;
  pcfg.thermalize = 5;
  pcfg.stride = 1;
  pcfg.samples = 16;
  pcfg.blocks = 2;
  pcfg.seed = 3;
  pcfg.retain = 1;
  pcfg.max_rounds = 1;
  rgsaddle_pafi_config_t other_pcfg = pcfg;
  other_pcfg.version.major = RGSADDLE_ABI_MAJOR + 1u;
  rgsaddle_pafi_config_t flagged_pcfg = pcfg;
  flagged_pcfg.flags = 1;
  const double frames[3] = {-1.0, 0.0, 1.0};
  if (rgsaddle_pafi_create(&other_pcfg, 3, 1, frames, NULL, 0) != NULL) {
    return fail(RGSADDLE_ABI_MISMATCH, "pafi create accepted another major");
  }
  RgsaddlePafi *pafi = rgsaddle_pafi_create(&pcfg, 3, 1, frames, NULL, 0);
  if (!pafi) {
    return fail(RGSADDLE_SHAPE, "pafi create for set_config");
  }
  if (rgsaddle_pafi_set_config(NULL, &pcfg) != RGSADDLE_NULL_SESSION ||
      rgsaddle_pafi_set_config(NULL, &other_pcfg) != RGSADDLE_NULL_SESSION ||
      rgsaddle_pafi_set_config(NULL, NULL) != RGSADDLE_NULL_SESSION ||
      rgsaddle_pafi_set_config(pafi, NULL) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_pafi_set_config(pafi, &other_pcfg) != RGSADDLE_ABI_MISMATCH ||
      rgsaddle_pafi_set_config(pafi, &flagged_pcfg) !=
          RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_pafi_set_config(pafi, &pcfg) != RGSADDLE_OK) {
    rgsaddle_pafi_free(pafi);
    return fail(RGSADDLE_ABI_MISMATCH, "pafi set_config statuses");
  }
  rgsaddle_pafi_free(pafi);

  rgsaddle_ohtst_config_t ocfg;
  memset(&ocfg, 0, sizeof ocfg);
  ocfg.version.major = RGSADDLE_ABI_MAJOR;
  ocfg.version.minor = RGSADDLE_ABI_MINOR;
  ocfg.thermal = 0.5;
  ocfg.sample_time_step = 0.1;
  ocfg.sample_friction = 1.0;
  ocfg.sample_burn_in = 10;
  ocfg.sample_stride = 1;
  ocfg.sample_records = 16;
  ocfg.sample_blocks = 2;
  ocfg.sample_seed = 9;
  ocfg.sample_retain = 4;
  ocfg.sample_integrator = RGSADDLE_OHTST_INTEGRATOR_BAOAB;
  ocfg.shooting_friction = 1.5;
  ocfg.shooting_time_step = 0.05;
  ocfg.shooting_steps = 20;
  ocfg.shooting_check_every = 5;
  ocfg.shooting_shots = 8;
  ocfg.shooting_blocks = 2;
  ocfg.shooting_seed = 17;
  ocfg.counting = RGSADDLE_OHTST_COUNT_FINAL;
  rgsaddle_ohtst_config_t other_ocfg = ocfg;
  other_ocfg.version.major = RGSADDLE_ABI_MAJOR + 1u;
  rgsaddle_ohtst_config_t flagged_ocfg = ocfg;
  flagged_ocfg.flags = 1;
  const double point[2] = {0.0, 0.0};
  const double normal[2] = {1.0, 0.0};
  if (rgsaddle_ohtst_create(&other_ocfg, 2, point, normal) != NULL) {
    return fail(RGSADDLE_ABI_MISMATCH, "ohtst create accepted another major");
  }
  RgsaddleOhtst *ohtst = rgsaddle_ohtst_create(&ocfg, 2, point, normal);
  if (!ohtst) {
    return fail(RGSADDLE_SHAPE, "ohtst create for set_config");
  }
  if (rgsaddle_ohtst_set_config(NULL, &ocfg) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ohtst_set_config(NULL, &other_ocfg) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ohtst_set_config(NULL, NULL) != RGSADDLE_NULL_SESSION ||
      rgsaddle_ohtst_set_config(ohtst, NULL) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_ohtst_set_config(ohtst, &other_ocfg) != RGSADDLE_ABI_MISMATCH ||
      rgsaddle_ohtst_set_config(ohtst, &flagged_ocfg) !=
          RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_ohtst_set_config(ohtst, &ocfg) != RGSADDLE_OK) {
    rgsaddle_ohtst_free(ohtst);
    return fail(RGSADDLE_ABI_MISMATCH, "ohtst set_config statuses");
  }
  rgsaddle_ohtst_free(ohtst);
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
  if (xts_band_step(band, NULL, NULL, NULL) != RGSADDLE_NULL_REPORT) {
    return fail(RGSADDLE_NULL_REPORT, "xts_band_step NULL report must fail closed");
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  climbing_seen_t climbing = {1, 0, -1};
  int steps = 0;
  do {
    st = rgsaddle_band_step(band, surface, &climbing, &rep);
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
  if (climbing.calls < 1 || climbing.image != (long)rep.ci_index) {
    fprintf(stderr, "climbing requests=%d image=%ld ci=%lld\n", climbing.calls,
            climbing.image, (long long)rep.ci_index);
    return fail(RGSADDLE_SHAPE, "refinement requests must carry the climbing image");
  }

  /* A minor 21 config gets the refinement unmarked, with image -1. */
  {
    rgsaddle_band_config_t old = cfg;
    old.version.minor = 21;
    RgsaddleBand *b21 = rgsaddle_band_create(&old, n_images, n_atoms, pos);
    if (!b21) {
      return fail(RGSADDLE_SHAPE, "minor 21 band create failed");
    }
    climbing_seen_t unmarked = {0, 0, -1};
    rgsaddle_report_t r21;
    memset(&r21, 0, sizeof r21);
    int k = 0;
    do {
      st = rgsaddle_band_step(b21, surface, &unmarked, &r21);
      if (st != RGSADDLE_OK) {
        return fail(st, "minor 21 step");
      }
      ++k;
    } while (r21.status == RGSADDLE_STATUS_RUNNING && k < 3000);
    if (r21.status != RGSADDLE_STATUS_CONVERGED) {
      return fail(RGSADDLE_SOLVER, "minor 21 band did not converge");
    }
    rgsaddle_band_free(b21);
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

  /* Zoom-NEB: the band is placed on a window inside the old one, the
   * window ends keep their energies and the climbing image stays armed. */
  {
    rgsaddle_band_config_t zc = cfg;
    zc.flags = RGSADDLE_BAND_NO_MIN_MODE;
    RgsaddleBand *zb = rgsaddle_band_create(&zc, n_images, n_atoms, pos);
    if (!zb) {
      return fail(RGSADDLE_SHAPE, "zoom band create failed");
    }
    if (rgsaddle_band_set_zoom(NULL, 1, 0.5, 1, 0.0, 5, 1) != RGSADDLE_NULL_BAND ||
        rgsaddle_band_set_zoom(zb, 1, 1.0, 1, 0.0, 5, 1) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_zoom(zb, 1, 0.5, 0, 0.0, 5, 1) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_zoom(zb, 1, 0.5, 1, 0.0, 5, 2) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_zoom(zb, 0, 0.0, 0, 0.0, 0, 0) != RGSADDLE_OK ||
        rgsaddle_band_set_zoom(zb, 1, 0.5, 1, 10.0, 2, 1) != RGSADDLE_OK) {
      return fail(RGSADDLE_INVALID_PARAMETER, "zoom setter");
    }
    climbing_seen_t zseen = {1, 0, -1};
    rgsaddle_report_t zr;
    memset(&zr, 0, sizeof zr);
    double zp[9 * 3];
    int k = 0;
    do {
      st = rgsaddle_band_step(zb, surface, &zseen, &zr);
      if (st != RGSADDLE_OK || rgsaddle_band_positions(zb, zp) != RGSADDLE_OK) {
        return fail(st, "zoom step");
      }
    } while (zp[0] == -1.0 && ++k < 500);
    const double lo = zp[0], hi = zp[(n_images - 1) * 3];
    if (!(lo > -1.0 && hi < 1.0) || fabs(zp[(n_images / 2) * 3]) > 1e-9 ||
        zr.ci_index != n_images / 2) {
      fprintf(stderr, "zoom window %g to %g, middle %g, ci %lld\n", lo, hi,
              zp[(n_images / 2) * 3], (long long)zr.ci_index);
      return fail(RGSADDLE_SHAPE, "zoom placement");
    }
    for (long i = 1; i < n_images; ++i) {
      if (!(zp[i * 3] > zp[(i - 1) * 3])) {
        return fail(RGSADDLE_SHAPE, "zoomed images out of order");
      }
    }
    st = rgsaddle_band_step(zb, surface, &zseen, &zr);
    if (st != RGSADDLE_OK || zr.evaluations != 2 * (n_images - 2) || zr.ci_index < 1 ||
        zr.ci_index > n_images - 2) {
      fprintf(stderr, "after zoom: evaluations %d, ci %lld\n", zr.evaluations,
              (long long)zr.ci_index);
      return fail(RGSADDLE_SHAPE, "step after the zoom");
    }
    k = 0;
    while (zr.status == RGSADDLE_STATUS_RUNNING && ++k < 3000) {
      st = rgsaddle_band_step(zb, surface, &zseen, &zr);
      if (st != RGSADDLE_OK) {
        return fail(st, "zoomed band step");
      }
    }
    if (zr.status != RGSADDLE_STATUS_CONVERGED ||
        rgsaddle_band_positions(zb, zp) != RGSADDLE_OK || zp[0] != lo ||
        fabs(zp[zr.ci_index * 3]) > 1e-2) {
      return fail(RGSADDLE_SOLVER, "zoomed band did not reach the saddle");
    }
    rgsaddle_band_free(zb);
  }

  st = minmode_reuse();
  if (st != RGSADDLE_OK) {
    return st;
  }

  st = pafi_and_ohtst();
  if (st != RGSADDLE_OK) {
    return st;
  }

  st = set_config_statuses();
  if (st != RGSADDLE_OK) {
    return st;
  }

  /* Index-k: a bad k is rejected, one dimer step matches the one-mode
   * translation's curvature sign, and the reflected force is the
   * Householder product. */
  {
    rgsaddle_indexk_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.version.major = RGSADDLE_ABI_MAJOR;
    cfg.version.minor = RGSADDLE_ABI_MINOR;
    cfg.index = 0;
    cfg.method = RGSADDLE_MINMODE_DIMER;
    cfg.dr = 1e-3;
    cfg.rotation_tol = 1e-4;
    cfg.max_rotations = 10;
    cfg.force_tol = 1e-6;
    cfg.max_move = 0.1;
    cfg.residual_tol = 1e-6;
    cfg.max_block_iterations = 20;
    const double x0[3] = {0.35, 0.2, -0.1};
    const double mode[3] = {1.0, 0.3, 0.0};
    if (rgsaddle_indexk_create(&cfg, 1, x0, mode) != NULL) {
      return fail(RGSADDLE_INVALID_PARAMETER, "index 0 was accepted");
    }
    cfg.index = 1;
    RgsaddleIndexK *session = rgsaddle_indexk_create(&cfg, 1, x0, mode);
    if (!session) {
      return fail(RGSADDLE_SHAPE, "indexk create");
    }
    rgsaddle_report_t irep;
    memset(&irep, 0, sizeof irep);
    st = rgsaddle_indexk_step(session, surface_quad, NULL, NULL, &irep);
    if (st != RGSADDLE_OK) {
      return fail(st, "indexk step");
    }
    double at[3];
    double modes[3];
    double curvatures[1];
    double translated[3];
    if (rgsaddle_indexk_position(session, at) != RGSADDLE_OK ||
        rgsaddle_indexk_modes(session, modes, curvatures) != RGSADDLE_OK ||
        rgsaddle_indexk_translated(session, translated) != RGSADDLE_OK) {
      return fail(RGSADDLE_SHAPE, "indexk readback");
    }
    if (!(curvatures[0] < 0.0) || irep.curvature != curvatures[0]) {
      fprintf(stderr, "indexk curvature %g report %g\n", curvatures[0], irep.curvature);
      return fail(RGSADDLE_SOLVER, "indexk curvature");
    }
    const double force[3] = {1.0, -2.0, 0.5};
    const double axis[3] = {1.0, 0.0, 0.0};
    double reflected[3];
    st = rgsaddle_reflect_force(3, 1, force, axis, reflected);
    if (st != RGSADDLE_OK || reflected[0] != -1.0 || reflected[1] != -2.0 ||
        reflected[2] != 0.5) {
      return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "reflect force");
    }
    if (rgsaddle_reflect_force(3, 1, force, axis, (double *)force) !=
        RGSADDLE_INVALID_PARAMETER) {
      return fail(RGSADDLE_INVALID_PARAMETER, "aliased reflect was accepted");
    }
    rgsaddle_indexk_free(session);
  }

  /* Simplified string method (ABI minor 44): flag creation, setter validation,
   * spline and linear convergence to the double-well saddle, and project. */
  {
    /* A host stamped with ABI minor 43 may still carry reserved bit 5.
     * The bit must stay inert until ABI minor 44. */
    {
      rgsaddle_band_config_t legacy = cfg;
      legacy.version.minor = 43;
      legacy.flags = RGSADDLE_BAND_STRING | RGSADDLE_BAND_NO_MIN_MODE;
      legacy.ci_trigger_factor = 0.0;
      legacy.spring_k = 0.0;
      legacy.projection = RGSADDLE_PROJECTION_NEB;
      const double c_pos[9] = {
          -1.0, 0.0, 0.0,
          0.4, 0.0, 0.0,
          1.0, 0.0, 0.0,
      };
      const double c_energy[3] = {0.0, 0.7056, 0.0};
      const double c_gradient[9] = {
          0.0, 0.0, 0.0,
          -1.344, 0.0, 0.0,
          0.0, 0.0, 0.0,
      };
      double c_force[9] = {0.0};
      int64_t c_ci = -2;
      st = rgsaddle_band_project(&legacy, 3, 1, c_pos, c_energy, c_gradient, c_force, &c_ci);
      if (st != RGSADDLE_OK || c_ci != -1 || fabs(c_force[3]) > 1e-12) {
        fprintf(stderr, "legacy string bit leaked into ABI 43: fx=%g ci=%lld\n",
                c_force[3], (long long)c_ci);
        return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "legacy string bit");
      }
    }

    if (rgsaddle_band_set_string(NULL, 1, RGSADDLE_REPARAM_NOT_A_KNOT) != RGSADDLE_NULL_BAND) {
      return fail(RGSADDLE_NULL_BAND, "string setter null band");
    }
    rgsaddle_band_config_t rc = cfg;
    rc.method = RGSADDLE_METHOD_RTR;
    rc.flags = RGSADDLE_BAND_NO_MIN_MODE;
    RgsaddleBand *rtr = rgsaddle_band_create(&rc, n_images, n_atoms, pos);
    if (!rtr ||
        rgsaddle_band_set_string(rtr, 1, RGSADDLE_REPARAM_NOT_A_KNOT) !=
            RGSADDLE_INVALID_PARAMETER) {
      return fail(RGSADDLE_INVALID_PARAMETER, "RTR accepted the string setter");
    }
    rgsaddle_band_free(rtr);
    rgsaddle_band_config_t sc = cfg;
    sc.flags = RGSADDLE_BAND_NO_MIN_MODE | RGSADDLE_BAND_STRING | RGSADDLE_BAND_SPLINE_REPARAM;
    sc.ci_trigger_factor = 0.5;
    RgsaddleBand *sb = rgsaddle_band_create(&sc, n_images, n_atoms, pos);
    if (!sb) {
      return fail(RGSADDLE_SHAPE, "string band create failed");
    }
    if (rgsaddle_band_set_string(sb, 2, RGSADDLE_REPARAM_NOT_A_KNOT) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_string(sb, 1, 99) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_string(sb, 0, 99) != RGSADDLE_OK ||
        rgsaddle_band_set_string(sb, 1, RGSADDLE_REPARAM_NOT_A_KNOT) != RGSADDLE_OK) {
      return fail(RGSADDLE_INVALID_PARAMETER, "string setter validation");
    }
    climbing_seen_t sseen = {1, 0, -1};
    rgsaddle_report_t sr;
    memset(&sr, 0, sizeof sr);
    int sk = 0;
    do {
      st = rgsaddle_band_step(sb, surface, &sseen, &sr);
      if (st != RGSADDLE_OK) {
        return fail(st, "string step");
      }
      ++sk;
    } while (sr.status == RGSADDLE_STATUS_RUNNING && sk < 500);
    double sp[9 * 3];
    if (sr.status != RGSADDLE_STATUS_CONVERGED || sr.ci_index < 1 ||
        rgsaddle_band_positions(sb, sp) != RGSADDLE_OK ||
        fabs(sp[sr.ci_index * 3]) > 0.01) {
      return fail(RGSADDLE_SOLVER, "string band did not reach the saddle");
    }
    rgsaddle_band_free(sb);
  }

  /* Growing string method (ABI minor 44): fragment growth, join hand-off,
   * image IDs, and convergence to the double-well saddle. */
  {
    rgsaddle_growing_string_config_t gcfg;
    if (rgsaddle_growing_string_config_default(NULL) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_growing_string_config_default(&gcfg) != RGSADDLE_OK) {
      return fail(RGSADDLE_INVALID_PARAMETER, "growing string default config");
    }
    gcfg.n_images = 7;
    gcfg.frontier_tol = 0.5;
    gcfg.band.force_tol = 0.001;
    gcfg.band.max_move = 0.1;
    const double greact[3] = {-1.0, 0.2, 0.0};
    const double gprod[3] = {1.0, 0.2, 0.0};
    RgsaddleGrowingString *gs =
        rgsaddle_growing_string_create(&gcfg, n_atoms, greact, gprod);
    if (!gs || rgsaddle_growing_string_n_images(gs) != 4 ||
        rgsaddle_growing_string_phase(gs) != RGSADDLE_GROWING_STRING_GROWING) {
      return fail(RGSADDLE_SHAPE, "growing string create");
    }
    uint64_t gids[7];
    if (rgsaddle_growing_string_ids(gs, gids) != RGSADDLE_OK ||
        gids[0] != 0 || gids[1] != 2 || gids[2] != 3 || gids[3] != 1) {
      return fail(RGSADDLE_SHAPE, "growing string initial ids");
    }
    climbing_seen_t gseen = {1, 0, -1};
    rgsaddle_report_t gr;
    memset(&gr, 0, sizeof gr);
    int gk = 0;
    int64_t grown = 0;
    if (rgsaddle_growing_string_nodes_added(gs) != 0 ||
        rgsaddle_growing_string_nodes_added(NULL) != 0) {
      return fail(RGSADDLE_SHAPE, "growing string nodes_added before a step");
    }
    do {
      gr.rotations = -1;
      st = rgsaddle_growing_string_step(gs, surface, &gseen, &gr);
      if (st != RGSADDLE_OK) {
        return fail(st, "growing string step");
      }
      if (gr.rotations != 0) {
        return fail(RGSADDLE_SHAPE, "growing string step reported rotations");
      }
      const int64_t added = rgsaddle_growing_string_nodes_added(gs);
      if (added < 0 || added > 2) {
        return fail(RGSADDLE_SHAPE, "growing string nodes_added out of range");
      }
      grown += added;
      ++gk;
    } while (gr.status == RGSADDLE_STATUS_RUNNING && gk < 500);
    if (grown != 3 || rgsaddle_growing_string_nodes_added(gs) != 0) {
      return fail(RGSADDLE_SHAPE, "growing string grew 3 nodes from 4 to 7 images");
    }
    double gp[7 * 3];
    if (gr.status != RGSADDLE_STATUS_CONVERGED ||
        rgsaddle_growing_string_phase(gs) != RGSADDLE_GROWING_STRING_JOINED ||
        rgsaddle_growing_string_n_images(gs) != 7 || gr.ci_index < 1 ||
        rgsaddle_growing_string_positions(gs, gp) != RGSADDLE_OK ||
        fabs(gp[gr.ci_index * 3]) > 0.02) {
      return fail(RGSADDLE_SOLVER, "growing string did not converge to saddle");
    }
    rgsaddle_growing_string_free(gs);

    {
      rgsaddle_growing_string_config_t bad;
      const double nan = NAN;
      const double inf = INFINITY;
      for (int which = 0; which < 8; ++which) {
        if (rgsaddle_growing_string_config_default(&bad) != RGSADDLE_OK) {
          return fail(RGSADDLE_INVALID_PARAMETER, "growing string default config");
        }
        bad.n_images = 7;
        switch (which) {
          case 0: bad.frontier_tol = 0.0; break;
          case 1: bad.frontier_tol = -0.5; break;
          case 2: bad.frontier_tol = nan; break;
          case 3: bad.min_steps_per_node = 0; break;
          case 4: bad.max_steps_per_node = -1; break;
          case 5: bad.join_tol = nan; break;
          case 6: bad.join_tol = inf; break;
          default: bad.frontier_tol = inf; break;
        }
        RgsaddleGrowingString *refused =
            rgsaddle_growing_string_create(&bad, n_atoms, greact, gprod);
        if (refused) {
          rgsaddle_growing_string_free(refused);
          return fail(RGSADDLE_INVALID_PARAMETER, "growing string accepted an invalid config");
        }
      }
      if (rgsaddle_growing_string_config_default(&bad) != RGSADDLE_OK) {
        return fail(RGSADDLE_INVALID_PARAMETER, "growing string default config");
      }
      bad.n_images = 7;
      if (bad.join_tol != -1.0) {
        return fail(RGSADDLE_INVALID_PARAMETER, "growing string default join tolerance");
      }
      bad.join_tol = 0.0;
      bad.max_steps_per_node = 4;
      RgsaddleGrowingString *kept = rgsaddle_growing_string_create(&bad, n_atoms, greact, gprod);
      if (!kept) {
        return fail(RGSADDLE_INVALID_PARAMETER, "growing string refused a valid config");
      }
      rgsaddle_growing_string_free(kept);
    }
  }

  /* Free-end adaptive NEB (ABI minor 45): setter validation, level-set
   * residual reporting, and saddle convergence. */
  {
    if (rgsaddle_band_set_free_end(NULL, 1, RGSADDLE_FREE_END_LAST, 1, 0.25, 0,
                                   0.0, 0.001, 2, 1, 1) != RGSADDLE_NULL_BAND) {
      return fail(RGSADDLE_NULL_BAND, "free_end null band");
    }
    rgsaddle_band_config_t fc = cfg;
    fc.flags = RGSADDLE_BAND_NO_MIN_MODE;
    fc.ci_trigger_factor = 0.5;
    double fpos[7 * 3];
    for (int i = 0; i < 7; ++i) {
      double t = (double)i / 6.0;
      fpos[3 * i + 0] = -1.0 + 1.6 * t;
      fpos[3 * i + 1] = 0.2 * sin(3.14159265358979 * t);
      fpos[3 * i + 2] = 0.0;
    }
    RgsaddleBand *fb = rgsaddle_band_create(&fc, 7, n_atoms, fpos);
    if (!fb) {
      return fail(RGSADDLE_SHAPE, "free_end band create");
    }
    double fres = 0.0;
    if (rgsaddle_band_free_end_residual(fb, NULL) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_free_end_residual(fb, &fres) != RGSADDLE_NO_EVALUATION ||
        rgsaddle_band_set_free_end(fb, 2, RGSADDLE_FREE_END_LAST, 1, 0.25, 0,
                                   0.0, 0.001, 2, 1, 1) !=
            RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_free_end(fb, 1, 99, 1, 0.25, 0, 0.0, 0.001, 2, 1,
                                   1) != RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_free_end(fb, 1, RGSADDLE_FREE_END_LAST, 1, 0.25, 0,
                                   0.0, 0.001, 0, 1, 1) !=
            RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_free_end(fb, 1, RGSADDLE_FREE_END_LAST, 1, 0.25, 0,
                                   0.0, 0.001, 2, 0, 1) !=
            RGSADDLE_INVALID_PARAMETER ||
        rgsaddle_band_set_free_end(fb, 1, RGSADDLE_FREE_END_LAST, 1, 0.25, 0,
                                   0.0, 0.001, 2, 1, 1) != RGSADDLE_OK) {
      return fail(RGSADDLE_INVALID_PARAMETER, "free_end setter validation");
    }
    climbing_seen_t fseen = {1, 0, -1};
    rgsaddle_report_t fr;
    memset(&fr, 0, sizeof fr);
    int fk = 0;
    do {
      st = rgsaddle_band_step(fb, surface, &fseen, &fr);
      if (st != RGSADDLE_OK) {
        return fail(st, "free_end step");
      }
      ++fk;
    } while (fr.status == RGSADDLE_STATUS_RUNNING && fk < 500);
    if (fr.status != RGSADDLE_STATUS_CONVERGED ||
        rgsaddle_band_free_end_residual(fb, &fres) != RGSADDLE_OK ||
        fabs(fres) > 1e-5) {
      return fail(RGSADDLE_SOLVER, "free_end band residual or convergence");
    }
    rgsaddle_band_free(fb);
  }

  /* A local window with max_cuts > 0 removes images and the evaluation
   * matches the reported positions. */
  {
    if (rgsaddle_band_free_end_cuts(NULL) != 0) {
      return fail(RGSADDLE_NULL_BAND, "free_end_cuts null band");
    }
    rgsaddle_band_config_t cc = cfg;
    cc.flags = RGSADDLE_BAND_NO_MIN_MODE;
    double cpos[5 * 3];
    for (int i = 0; i < 5; ++i) {
      cpos[3 * i + 0] = -0.6 + 0.3 * (double)i;
      cpos[3 * i + 1] = 0.0;
      cpos[3 * i + 2] = 0.0;
    }
    RgsaddleBand *cb = rgsaddle_band_create(&cc, 5, n_atoms, cpos);
    if (!cb) {
      return fail(RGSADDLE_SHAPE, "free_end cut band create");
    }
    if (rgsaddle_band_evaluation(cb, NULL, NULL, NULL) != RGSADDLE_NO_EVALUATION ||
        rgsaddle_band_set_free_end(cb, 1, RGSADDLE_FREE_END_LAST, 0, 0.0, 1, 100.0,
                                   0.001, 1, 1, 0) != RGSADDLE_OK) {
      return fail(RGSADDLE_INVALID_PARAMETER, "free_end cut setup");
    }
    climbing_seen_t cseen = {1, 0, -1};
    rgsaddle_report_t cr;
    memset(&cr, 0, sizeof cr);
    st = rgsaddle_band_step(cb, surface, &cseen, &cr);
    if (st != RGSADDLE_OK || rgsaddle_band_free_end_cuts(cb) < 1) {
      return fail(st == RGSADDLE_OK ? RGSADDLE_SOLVER : st, "free_end cut count");
    }
    double energies[5], gradients[15], projected[9], positions[15];
    if (rgsaddle_band_evaluation(cb, energies, gradients, projected) != RGSADDLE_OK ||
        rgsaddle_band_positions(cb, positions) != RGSADDLE_OK) {
      return fail(RGSADDLE_NO_EVALUATION, "free_end evaluation");
    }
    double x = positions[0];
    double expect = (x * x - 1.0) * (x * x - 1.0);
    if (fabs(energies[0] - expect) > 1e-12) {
      return fail(RGSADDLE_SOLVER, "free_end evaluation energy");
    }
    rgsaddle_band_free(cb);
  }

  printf("RGSADDLE_C_ABI_OK steps=%d ci=%lld\n", steps, (long long)rep.ci_index);
  return RGSADDLE_OK;
}
