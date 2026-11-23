/* C ABI contracts and edge-case tests against librgsaddle.so. */
#include "rgsaddle.h"
#include "main_abi_contract.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(int st, const char *msg) {
  fprintf(stderr, "FAIL: %s (status=%d %s)\n", msg, st,
          rgsaddle_status_name((rgsaddle_status_t)st));
  return 1;
}

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* Muller-Brown 2D embedded in (x, y, z) with harmonic z = 2 z^2. */
static void muller_brown_eval(double x, double y, double z, double *energy,
                              double *gx, double *gy, double *gz) {
  static const double A[4] = {-200.0, -100.0, -170.0, 15.0};
  static const double XX[4] = {-1.0, -1.0, -6.5, 0.7};
  static const double XY[4] = {0.0, 0.0, 11.0, 0.6};
  static const double YY[4] = {-10.0, -10.0, -6.5, 0.7};
  static const double X0[4] = {1.0, 0.0, -0.5, -1.0};
  static const double Y0[4] = {0.0, 0.5, 1.5, 1.0};
  double e = 2.0 * z * z;
  double dx_sum = 0.0;
  double dy_sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    double dx = x - X0[i];
    double dy = y - Y0[i];
    double t = A[i] * exp(XX[i] * dx * dx + XY[i] * dx * dy + YY[i] * dy * dy);
    e += t;
    dx_sum += t * (2.0 * XX[i] * dx + XY[i] * dy);
    dy_sum += t * (XY[i] * dx + 2.0 * YY[i] * dy);
  }
  *energy = e;
  *gx = dx_sum;
  *gy = dy_sum;
  *gz = 4.0 * z;
}

static rgsaddle_status_t mb_surface(void *user,
                                    rgsaddle_surface_request_t *req) {
  int *calls = (int *)user;
  if (calls) {
    *calls += 1;
  }
  int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t i = 0; i < rows; ++i) {
    const double *p = req->positions + 3 * i;
    double *g = req->gradients + 3 * i;
    muller_brown_eval(p[0], p[1], p[2], &req->energies[i], &g[0], &g[1], &g[2]);
  }
  return RGSADDLE_OK;
}

/* 3D index-2 saddle: V(x,y,z) = -x^2 - 0.5 y^2 + 2 z^2. */
static rgsaddle_status_t idx2_surface(void *user,
                                      rgsaddle_surface_request_t *req) {
  int *fail_flag = (int *)user;
  if (fail_flag && *fail_flag) {
    return RGSADDLE_SURFACE_FAILED;
  }
  int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t i = 0; i < rows; ++i) {
    const double *p = req->positions + 3 * i;
    double *g = req->gradients + 3 * i;
    req->energies[i] = -p[0] * p[0] - 0.5 * p[1] * p[1] + p[2] * p[2];
    g[0] = -2.0 * p[0];
    g[1] = -p[1];
    g[2] = 2.0 * p[2];
  }
  return RGSADDLE_OK;
}

static int idx2_hessian(void *user, int64_t n_dof, const double *v,
                        double *action) {
  int *fail_flag = (int *)user;
  if (fail_flag && *fail_flag) {
    return -1;
  }
  if (n_dof != 3) {
    return -2;
  }
  action[0] = -2.0 * v[0];
  action[1] = -1.0 * v[1];
  action[2] = 2.0 * v[2];
  return 0;
}

static rgsaddle_status_t failing_surface(void *user,
                                         rgsaddle_surface_request_t *req) {
  (void)user;
  (void)req;
  return RGSADDLE_SURFACE_FAILED;
}

static int stencil_mb_eval(void *user, size_t rows, size_t dof,
                           const uint64_t *ids, const double *x, double *e,
                           double *g) {
  (void)ids;
  int *mode = (int *)user;
  if (mode && *mode == 1) {
    return -7;
  }
  if (mode && *mode == 2) {
    e[0] = NAN;
    return 0;
  }
  for (size_t r = 0; r < rows; ++r) {
    const double *p = x + r * dof;
    double *gr = g + r * dof;
    muller_brown_eval(p[0], p[1], p[2], &e[r], &gr[0], &gr[1], &gr[2]);
  }
  return 0;
}

static rgsaddle_status_t solid_per_image_cb(
    void *user, rgsaddle_solid_surface_request_t *req) {
  int *state = (int *)user;
  if (state && *state < 0) {
    return RGSADDLE_SURFACE_FAILED;
  }
  if (state && (req->flags & RGSADDLE_REQ_ONE_IMAGE)) {
    *state += 1;
  }
  for (int64_t r = 0; r < ((req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images); ++r) {
    const double *h = req->cells + 9 * r;
    const double a = h[0], b = h[4], c = h[8];
    const double w = (a - 4.0) * (a - 4.4);
    req->energies[r] = 50.0 * w * w + 2.5 * ((b - 4.0) * (b - 4.0) + (c - 4.0) * (c - 4.0));
    double g[9] = {0};
    g[0] = 100.0 * w * (2.0 * a - 8.4);
    g[4] = 5.0 * (b - 4.0);
    g[8] = 5.0 * (c - 4.0);
    double v = a * b * c;
    double *s = req->stresses + 9 * r;
    memset(s, 0, 9 * sizeof(double));
    s[0] = a * g[0] / v;
    s[4] = b * g[4] / v;
    s[8] = c * g[8] / v;
    memset(req->gradients + r * 3 * req->n_atoms, 0, 3 * (size_t)req->n_atoms * sizeof(double));
  }
  return RGSADDLE_OK;
}

static rgsaddle_status_t two_atom_well(void *user,
                                       rgsaddle_surface_request_t *req) {
  (void)user;
  int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t i = 0; i < rows; ++i) {
    const double *p = req->positions + 6 * i;
    double *g = req->gradients + 6 * i;
    double dx = p[3] - p[0] - 1.0;
    double dy = p[4] - p[1];
    double dz = p[5] - p[2];
    req->energies[i] = 0.5 * (dx * dx + dy * dy + dz * dz) + 0.1 * (p[0] * p[0] + p[1] * p[1]);
    g[0] = -dx + 0.2 * p[0];
    g[1] = -dy + 0.2 * p[1];
    g[2] = -dz;
    g[3] = dx;
    g[4] = dy;
    g[5] = dz;
  }
  return RGSADDLE_OK;
}

static rgsaddle_status_t diagonal_surface(void *user,
                                          rgsaddle_surface_request_t *req) {
  (void)user;
  static const double curv[6] = {-2.0, 1.0, 4.0, 8.0, 2.0, 3.0};
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t r = 0; r < rows; ++r) {
    const double *p = req->positions + 6 * r;
    double *g = req->gradients + 6 * r;
    double e = 0.0;
    for (int i = 0; i < 6; ++i) {
      const double c = curv[i];
      e += 0.5 * c * p[i] * p[i];
      g[i] = c * p[i];
    }
    req->energies[r] = e;
  }
  return RGSADDLE_OK;
}

static RgsaddleMinMode *new_dimer_session(double rotation_tol, uint32_t minor, uint64_t flags,
                                          double rotation_angle_tol) {
  rgsaddle_minmode_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = minor;
  cfg.flags = flags;
  cfg.kind = RGSADDLE_MINMODE_DIMER;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.dr = 1e-3;
  cfg.rotation_tol = rotation_tol;
  cfg.max_rotations = 20;
  cfg.krylov_dim = 12;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.2;
  cfg.rotation_angle_tol = rotation_angle_tol;
  cfg.difference = RGSADDLE_DIFFERENCE_FORWARD;
  const double x[6] = {0.3, 0.2, 0.1, 0.4, 0.1, 0.2};
  const double mode[6] = {0.5, 0.5, 0.5, 0.5, 0.0, 0.0};
  return rgsaddle_minmode_create(&cfg, 2, x, mode);
}

static int minmode_rotations(RgsaddleMinMode *session) {
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  if (rgsaddle_minmode_estimate(session, diagonal_surface, NULL, &rep) != RGSADDLE_OK) {
    return -1;
  }
  return (int)rep.rotations;
}

static int test_minmode_rotation_angle_defaults_and_torque_window(void) {
  /* Zero means stop-off only before minor 63. Later minors keep the new rule. */
  RgsaddleMinMode *legacy = new_dimer_session(1e-8, 62, 0, 0.0);
  RgsaddleMinMode *defaulted = new_dimer_session(1e-8, 63, 0, 0.0);
  RgsaddleMinMode *explicit_zero = new_dimer_session(
      1e-8, RGSADDLE_ABI_MINOR, RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL, 0.0);
  RgsaddleMinMode *nonzero = new_dimer_session(1e-8, RGSADDLE_ABI_MINOR, 0, 1e-2);
  CHECK(legacy != NULL && defaulted != NULL && explicit_zero != NULL && nonzero != NULL,
        "minmode create for angle-stop tests");
  const int legacy_rotations = minmode_rotations(legacy);
  const int default_rotations = minmode_rotations(defaulted);
  const int explicit_zero_rotations = minmode_rotations(explicit_zero);
  const int nonzero_rotations = minmode_rotations(nonzero);
  CHECK(legacy_rotations > 0 && default_rotations > 0 && explicit_zero_rotations > 0 &&
            nonzero_rotations > 0,
        "minmode estimate returns rotations");
  CHECK(legacy_rotations > default_rotations, "legacy minor keeps zero as stop-off");
  CHECK(legacy_rotations == explicit_zero_rotations, "explicit zero keeps stop-off");
  CHECK(nonzero_rotations <= legacy_rotations, "nonzero keeps angle stop on");
  CHECK(rgsaddle_minmode_set_torque_window(defaulted, -1) == RGSADDLE_OK,
        "set_torque_window -1");
  CHECK(rgsaddle_minmode_set_torque_window(defaulted, 0) == RGSADDLE_OK,
        "set_torque_window 0");
  CHECK(rgsaddle_minmode_set_torque_window(defaulted, 1) == RGSADDLE_OK,
        "set_torque_window 1");
  rgsaddle_minmode_free(legacy);
  rgsaddle_minmode_free(defaulted);
  rgsaddle_minmode_free(explicit_zero);
  rgsaddle_minmode_free(nonzero);
  return 0;
}

static int test_status_and_gates(void) {
  for (int c = -16; c <= 0; ++c) {
    const char *name = rgsaddle_status_name((rgsaddle_status_t)c);
    CHECK(name != NULL && strlen(name) > 0, "status name non-empty");
  }
  CHECK(strcmp(rgsaddle_status_name((rgsaddle_status_t)-999), "RGSADDLE_UNKNOWN") != 0 ||
            rgsaddle_status_name((rgsaddle_status_t)-999) != NULL,
        "unknown status returns non-null");
  CHECK(strcmp(rgsaddle_force_gate_name(RGSADDLE_FORCE_L2), "L2") == 0, "L2 name");
  CHECK(strcmp(rgsaddle_force_gate_name(RGSADDLE_FORCE_LINF), "LINF") == 0, "LINF name");
  CHECK(strcmp(rgsaddle_force_gate_name(RGSADDLE_FORCE_MAX_ATOM), "MAX_ATOM") == 0, "MAX_ATOM name");
  CHECK(rgsaddle_force_gate_name((rgsaddle_force_gate_t)99) == NULL, "unknown force gate NULL");
  return 0;
}

static int flat_chart(void *ctx, const double *x, int64_t dof, double *q, double *b,
                      int64_t n_coords) {
  (void)ctx;
  if (dof != 2 || n_coords != 2) {
    return 1;
  }
  q[0] = x[0];
  q[1] = x[1];
  b[0] = 1.0;
  b[1] = 0.0;
  b[2] = 0.0;
  b[3] = 1.0;
  return 0;
}

static int test_geodesic_config_and_status(void) {
  rgsaddle_geodesic_config_t config;
  memset(&config, 0xAB, sizeof config);
  CHECK(rgsaddle_geodesic_config_default(&config) == RGSADDLE_OK, "geodesic default fills");
  CHECK(config.version.major == RGSADDLE_ABI_MAJOR, "geodesic default major");
  CHECK(config.version.minor == RGSADDLE_ABI_MINOR, "geodesic default minor");
  CHECK(config.flags == 0u, "geodesic default flags");
  CHECK(config.step_tol == 1e-6, "geodesic default step_tol");
  CHECK(config.grad_tol == 1e-6, "geodesic default grad_tol");
  CHECK(config.max_move == 0.5, "geodesic default max_move");
  CHECK(config.lower_ratio == 0.95, "geodesic default lower_ratio");
  CHECK(config.upper_ratio == 1.1, "geodesic default upper_ratio");
  CHECK(config.max_sweeps == 40, "geodesic default max_sweeps");
  CHECK(config.max_iterations == 40, "geodesic default max_iterations");
  CHECK(config.memory == 8, "geodesic default memory");
  CHECK(config.bound_samples == 10, "geodesic default bound_samples");
  CHECK(config.max_images == 32, "geodesic default max_images");
  CHECK(config.seed == 0u, "geodesic default seed");
  CHECK(rgsaddle_geodesic_config_default(NULL) == RGSADDLE_INVALID_PARAMETER,
        "geodesic default refuses NULL");
  CHECK(RGSADDLE_NOT_CONVERGED == -16, "NOT_CONVERGED value");
  CHECK(strcmp(rgsaddle_status_name(RGSADDLE_NOT_CONVERGED), "NOT_CONVERGED") == 0,
        "NOT_CONVERGED name");

  const double bowed[8] = {0.0, 0.0, 1.0, 2.0, 2.0, 0.5, 3.0, 4.0};
  double out[8];
  double length = 0.0, lower = 0.0, upper = 0.0;
  config.max_images = 8;
  CHECK(rgsaddle_geodesic_path(4, 2, bowed, 2, flat_chart, NULL, NULL, &config, out, &length,
                               &lower, &upper) == RGSADDLE_INVALID_PARAMETER,
        "an image cap above n_images is refused");
  config.max_images = 4;
  int rc = rgsaddle_geodesic_path(4, 2, bowed, 2, flat_chart, NULL, NULL, &config, out, &length,
                                  &lower, &upper);
  CHECK(rc == RGSADDLE_OK, "geodesic path converges on a flat chart");
  CHECK(fabs(length - 5.0) < 1e-6, "flat geodesic length is the 3-4-5 chord");
  CHECK(out[0] == 0.0 && out[1] == 0.0 && out[6] == 3.0 && out[7] == 4.0, "endpoints stay fixed");

  config.max_sweeps = 1;
  config.step_tol = 1e-12;
  rc = rgsaddle_geodesic_path(4, 2, bowed, 2, flat_chart, NULL, NULL, &config, out, &length,
                              &lower, &upper);
  CHECK(rc == RGSADDLE_NOT_CONVERGED, "one sweep is reported as not converged");
  CHECK(isfinite(length) && isfinite(out[2]), "not converged still writes the last iterate");

  config.max_images = 1;
  CHECK(rgsaddle_geodesic_path(4, 2, bowed, 2, flat_chart, NULL, NULL, &config, out, &length,
                               &lower, &upper) == RGSADDLE_INVALID_PARAMETER,
        "an image cap under 2 is refused");
  return 0;
}

static int test_ocineb_lanczos_rtr_mb(void) {
  const long n_images = 7, n_atoms = 1;
  const double b[3] = {-0.050011, 0.466694, 0.0};
  const double c[3] = {0.623499, 0.028038, 0.0};
  double pos[7 * 3];
  for (long i = 0; i < n_images; ++i) {
    double t = (double)i / (double)(n_images - 1);
    pos[3 * i + 0] = (1.0 - t) * b[0] + t * c[0];
    pos[3 * i + 1] = (1.0 - t) * b[1] + t * c[1];
    pos[3 * i + 2] = 0.0;
  }

  rgsaddle_band_config_t band;
  memset(&band, 0, sizeof band);
  band.version.major = RGSADDLE_ABI_MAJOR;
  band.version.minor = RGSADDLE_ABI_MINOR;
  band.flags = RGSADDLE_BAND_PER_IMAGE | RGSADDLE_BAND_SPLINE_REPARAM;
  band.tangent = RGSADDLE_TANGENT_IMPROVED;
  band.spring = RGSADDLE_SPRING_UNIFORM;
  band.projection = RGSADDLE_PROJECTION_NEB;
  band.method = RGSADDLE_METHOD_RTR;
  band.spring_k = 5.0;
  band.ci_trigger_factor = 0.5;
  band.force_tol = 1e-2;
  band.max_move = 0.05;

  rgsaddle_minmode_config_t mm;
  memset(&mm, 0, sizeof mm);
  mm.version.major = RGSADDLE_ABI_MAJOR;
  mm.version.minor = RGSADDLE_ABI_MINOR;
  mm.kind = RGSADDLE_MINMODE_LANCZOS;
  mm.method = RGSADDLE_METHOD_FIRE;
  mm.dr = 1e-3;
  mm.rotation_tol = 1e-4;
  mm.max_rotations = 20;
  mm.krylov_dim = 6;
  mm.force_tol = 1e-2;
  mm.max_move = 0.05;
  mm.difference = RGSADDLE_DIFFERENCE_CENTRAL;

  rgsaddle_ocineb_config_t oci;
  memset(&oci, 0, sizeof oci);
  oci.version.major = RGSADDLE_ABI_MAJOR;
  oci.version.minor = RGSADDLE_ABI_MINOR;
  oci.trigger_factor = 0.31;
  oci.angle_tol = 0.85;
  oci.stability_count = 3;
  oci.max_mmf_steps = 500;
  oci.restore_unhelpful = 1;

  /* Refusal paths. */
  CHECK(rgsaddle_ocineb_create(&band, &mm, &oci, 2, n_atoms, pos) == NULL, "n_images < 3");
  CHECK(rgsaddle_ocineb_create(&band, &mm, &oci, n_images, 0, pos) == NULL, "n_atoms < 1");
  rgsaddle_ocineb_config_t bad_oci = oci;
  bad_oci.version.major = 99;
  CHECK(rgsaddle_ocineb_create(&band, &mm, &bad_oci, n_images, n_atoms, pos) == NULL, "bad major");
  rgsaddle_minmode_config_t bad_mm = mm;
  bad_mm.method = RGSADDLE_METHOD_QUICKMIN;
  CHECK(rgsaddle_ocineb_create(&band, &bad_mm, &oci, n_images, n_atoms, pos) == NULL, "quickmin minmode");

  RgsaddleOcineb *sess = rgsaddle_ocineb_create(&band, &mm, &oci, n_images, n_atoms, pos);
  CHECK(sess != NULL, "create ocineb RTR Lanczos");

  /* Error checks on accessors. */
  rgsaddle_ocineb_report_t rep;
  memset(&rep, 0, sizeof rep);
  CHECK(rgsaddle_ocineb_step(sess, mb_surface, NULL, NULL) == RGSADDLE_NULL_REPORT, "null report");
  CHECK(rgsaddle_ocineb_step(sess, NULL, NULL, &rep) == RGSADDLE_NULL_SURFACE, "null surface");
  CHECK(rgsaddle_ocineb_positions(NULL, pos) == RGSADDLE_NULL_SESSION, "null pos session");
  CHECK(rgsaddle_ocineb_positions(sess, NULL) == RGSADDLE_INVALID_PARAMETER, "null pos out");
  CHECK(rgsaddle_ocineb_reset(NULL) == RGSADDLE_NULL_SESSION, "null reset");
  CHECK(rgsaddle_ocineb_set_saddle_check(NULL, NULL) == RGSADDLE_NULL_SESSION, "null set check");
  CHECK(rgsaddle_ocineb_saddle_index(NULL, NULL, NULL, 0) == RGSADDLE_NULL_SESSION, "null get check");

  rgsaddle_saddle_check_t chk;
  CHECK(rgsaddle_saddle_check_default(&chk) == RGSADDLE_OK, "default check");
  CHECK(rgsaddle_ocineb_set_saddle_check(sess, &chk) == RGSADDLE_OK, "set saddle check");

  int calls = 0;
  for (int i = 0; i < 2000; ++i) {
    rgsaddle_status_t st = rgsaddle_ocineb_step(sess, mb_surface, &calls, &rep);
    if (st != RGSADDLE_OK) {
      return fail(st, "ocineb mb step");
    }
    if (rep.status == RGSADDLE_STATUS_CONVERGED) {
      break;
    }
  }
  CHECK(rep.status == RGSADDLE_STATUS_CONVERGED, "ocineb MB converged");
  CHECK(rep.ci_index >= 1 && rep.ci_index <= n_images - 2, "valid ci_index");

  double out[7 * 3];
  CHECK(rgsaddle_ocineb_positions(sess, out) == RGSADDLE_OK, "read positions");
  const double *saddle = out + rep.ci_index * 3;
  CHECK(fabs(saddle[0] - 0.21248658) < 0.02 && fabs(saddle[1] - 0.29298833) < 0.02,
        "MB saddle reached");

  rgsaddle_saddle_index_t idx;
  double curv[4];
  CHECK(rgsaddle_ocineb_saddle_index(sess, &idx, curv, 4) == RGSADDLE_OK, "read saddle index");
  CHECK(idx.certified == 1 && idx.index == 1 && curv[0] < 0.0, "certified index 1");

  CHECK(rgsaddle_ocineb_set_saddle_check(sess, NULL) == RGSADDLE_OK, "clear saddle check");
  CHECK(rgsaddle_ocineb_reset(sess) == RGSADDLE_OK, "reset ocineb");
  rgsaddle_ocineb_free(sess);
  rgsaddle_ocineb_free(NULL);
  return 0;
}

static int test_indexk_contracts(void) {
  rgsaddle_indexk_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.index = 2;
  cfg.method = RGSADDLE_MINMODE_LANCZOS;
  cfg.dr = 1e-3;
  cfg.rotation_tol = 1e-6;
  cfg.max_rotations = 20;
  cfg.force_tol = 1e-6;
  cfg.max_move = 0.1;
  cfg.residual_tol = 1e-6;
  cfg.max_block_iterations = 30;

  const double x0[3] = {0.3, -0.25, 0.2};
  /* Invalid creates. */
  CHECK(rgsaddle_indexk_create(NULL, 1, x0, NULL) == NULL, "null cfg");
  CHECK(rgsaddle_indexk_create(&cfg, 0, x0, NULL) == NULL, "n_atoms 0");
  CHECK(rgsaddle_indexk_create(&cfg, 1, NULL, NULL) == NULL, "null pos");
  rgsaddle_indexk_config_t bad = cfg;
  bad.version.major = 99;
  CHECK(rgsaddle_indexk_create(&bad, 1, x0, NULL) == NULL, "bad major");
  bad = cfg;
  bad.index = 4;
  CHECK(rgsaddle_indexk_create(&bad, 1, x0, NULL) == NULL, "k > 3N");
  bad = cfg;
  bad.method = 99;
  CHECK(rgsaddle_indexk_create(&bad, 1, x0, NULL) == NULL, "bad method");
  bad = cfg;
  bad.max_block_iterations = 0;
  CHECK(rgsaddle_indexk_create(&bad, 1, x0, NULL) == NULL, "zero max_block_iterations");

  for (int m = 0; m < 3; ++m) {
    cfg.method = (m == 0) ? RGSADDLE_MINMODE_DIMER
                          : (m == 1 ? RGSADDLE_MINMODE_LANCZOS : RGSADDLE_MINMODE_LOBPCG);
    RgsaddleIndexK *sess = rgsaddle_indexk_create(&cfg, 1, x0, NULL);
    CHECK(sess != NULL, "indexk create index=2");
    double tr[3];
    CHECK(rgsaddle_indexk_translated(sess, tr) == RGSADDLE_NO_EVALUATION,
          "translated before first step");

    rgsaddle_report_t rep;
    memset(&rep, 0, sizeof rep);
    CHECK(rgsaddle_indexk_step(NULL, idx2_surface, NULL, NULL, &rep) == RGSADDLE_NULL_SESSION, "null sess");
    CHECK(rgsaddle_indexk_step(sess, idx2_surface, NULL, NULL, NULL) == RGSADDLE_NULL_REPORT, "null rep");
    CHECK(rgsaddle_indexk_step(sess, NULL, NULL, NULL, &rep) == RGSADDLE_NULL_SURFACE, "null surf");

    int fail_flag = 1;
    CHECK(rgsaddle_indexk_step(sess, idx2_surface, idx2_hessian, &fail_flag, &rep) ==
              RGSADDLE_SURFACE_FAILED,
          "surface/hessian fail");
    fail_flag = 0;

    for (int it = 0; it < 400; ++it) {
      rgsaddle_status_t st =
          rgsaddle_indexk_step(sess, idx2_surface, idx2_hessian, &fail_flag, &rep);
      if (st != RGSADDLE_OK) {
        return fail(st, "indexk step");
      }
      if (rep.status == RGSADDLE_STATUS_CONVERGED) {
        break;
      }
    }
    CHECK(rep.status == RGSADDLE_STATUS_CONVERGED, "index2 converged");
    double pos[3], modes[6], curv[2];
    CHECK(rgsaddle_indexk_position(sess, pos) == RGSADDLE_OK, "indexk pos");
    CHECK(rgsaddle_indexk_modes(sess, modes, curv) == RGSADDLE_OK, "indexk modes");
    CHECK(rgsaddle_indexk_translated(sess, tr) == RGSADDLE_OK, "indexk translated");
    CHECK(fabs(pos[0]) < 1e-4 && fabs(pos[1]) < 1e-4 && fabs(pos[2]) < 1e-4, "at origin");
    CHECK(curv[0] < -1.5 && curv[1] < -0.5, "two negative curvatures");

    /* Reuse via set_position with explicit gradient and without. */
    const double x_new[3] = {0.2, 0.1, -0.1};
    const double g_new[3] = {-0.4, -0.1, -0.4};
    CHECK(rgsaddle_indexk_set_position(sess, x_new, g_new) == RGSADDLE_OK, "set_position with grad");
    CHECK(rgsaddle_indexk_set_position(sess, x_new, NULL) == RGSADDLE_OK, "set_position no grad");
    CHECK(rgsaddle_indexk_set_position(NULL, x_new, NULL) == RGSADDLE_NULL_SESSION, "set_position null");
    CHECK(rgsaddle_indexk_set_position(sess, NULL, NULL) == RGSADDLE_INVALID_PARAMETER, "set_position null pos");
    CHECK(rgsaddle_indexk_position(NULL, pos) == RGSADDLE_NULL_SESSION, "pos null sess");
    CHECK(rgsaddle_indexk_position(sess, NULL) == RGSADDLE_INVALID_PARAMETER, "pos null out");
    CHECK(rgsaddle_indexk_modes(NULL, modes, curv) == RGSADDLE_NULL_SESSION, "modes null sess");
    CHECK(rgsaddle_indexk_modes(sess, NULL, curv) == RGSADDLE_INVALID_PARAMETER, "modes null out");
    CHECK(rgsaddle_indexk_translated(NULL, tr) == RGSADDLE_NULL_SESSION, "tr null sess");
    CHECK(rgsaddle_indexk_translated(sess, NULL) == RGSADDLE_INVALID_PARAMETER, "tr null out");

    rgsaddle_indexk_free(sess);
  }
  rgsaddle_indexk_free(NULL);
  CHECK(rgsaddle_reflect_force(0, 1, x0, x0, (double *)x0) == RGSADDLE_INVALID_PARAMETER,
        "reflect invalid n");
  return 0;
}

static int test_solid_per_image_and_errors(void) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_PER_IMAGE;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.ci_trigger_factor = 0.5;
  cfg.force_tol = 1e-4;
  cfg.max_move = 0.05;

  double cells[5 * 9];
  double pos[5 * 3];
  for (int i = 0; i < 5; ++i) {
    double a = 4.0 + 0.1 * i;
    double *h = cells + 9 * i;
    memset(h, 0, 9 * sizeof(double));
    h[0] = a;
    h[4] = 4.0;
    h[8] = 4.0;
    pos[3 * i + 0] = 0.0;
    pos[3 * i + 1] = 0.0;
    pos[3 * i + 2] = 0.0;
  }

  RgsaddleBand *band = rgsaddle_band_create_solid(&cfg, 5, 1, pos, cells, 0.0, 1.0);
  CHECK(band != NULL, "solid per_image create");

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  CHECK(rgsaddle_band_step_solid(NULL, solid_per_image_cb, NULL, &rep) == RGSADDLE_NULL_BAND, "solid null band");
  CHECK(rgsaddle_band_step_solid(band, solid_per_image_cb, NULL, NULL) == RGSADDLE_NULL_REPORT, "solid null rep");
  CHECK(rgsaddle_band_step_solid(band, NULL, NULL, &rep) == RGSADDLE_NULL_SURFACE, "solid null surf");
  CHECK(rgsaddle_band_cells(NULL, cells) == RGSADDLE_NULL_BAND, "cells null band");
  CHECK(rgsaddle_band_cells(band, NULL) == RGSADDLE_INVALID_PARAMETER, "cells null out");
  double jac = 0.0;
  CHECK(rgsaddle_band_solid_jacobian(NULL, &jac) == RGSADDLE_NULL_BAND, "jac null band");
  CHECK(rgsaddle_band_solid_jacobian(band, NULL) == RGSADDLE_INVALID_PARAMETER, "jac null out");

  int neg = -1;
  CHECK(rgsaddle_band_step_solid(band, solid_per_image_cb, &neg, &rep) == RGSADDLE_SURFACE_FAILED,
        "solid callback error");

  int one_image_calls = 0;
  for (int it = 0; it < 2000; ++it) {
    rgsaddle_status_t st = rgsaddle_band_step_solid(band, solid_per_image_cb, &one_image_calls, &rep);
    if (st != RGSADDLE_OK) {
      return fail(st, "solid per_image step");
    }
    if (rep.status == RGSADDLE_STATUS_CONVERGED) {
      break;
    }
  }
  CHECK(rep.status == RGSADDLE_STATUS_CONVERGED, "solid per_image converged");
  CHECK(one_image_calls > 0, "solid per_image invoked with RGSADDLE_REQ_ONE_IMAGE");

  double out_cells[5 * 9];
  CHECK(rgsaddle_band_cells(band, out_cells) == RGSADDLE_OK, "solid cells read");
  CHECK(fabs(out_cells[9 * rep.ci_index] - 4.2) < 1e-2, "climbing cell at 4.2");
  rgsaddle_band_free(band);

  RgsaddleBand *plain = rgsaddle_band_create(&cfg, 3, 1, pos);
  CHECK(rgsaddle_band_cells(plain, out_cells) == RGSADDLE_INVALID_PARAMETER, "cells on plain");
  CHECK(rgsaddle_band_solid_jacobian(plain, &jac) == RGSADDLE_INVALID_PARAMETER, "jac on plain");
  rgsaddle_band_free(plain);
  return 0;
}

static int test_stencil_contracts(void) {
  /* Unknown preset & key errors. */
  CHECK(rgsaddle_stencil_config_new(999) == NULL, "unknown preset");
  CHECK(strlen(rgsaddle_last_error()) > 0, "last error populated");
  CHECK(rgsaddle_stencil_config_set_f64(NULL, RGSADDLE_STENCIL_KEY_DR, 1e-3) == RGSADDLE_NULL_SESSION,
        "f64 null cfg");
  CHECK(rgsaddle_stencil_config_set_i64(NULL, RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, 10) ==
            RGSADDLE_NULL_SESSION,
        "i64 null cfg");
  CHECK(rgsaddle_stencil_config_set_array(NULL, RGSADDLE_STENCIL_KEY_MODE, NULL, 0) ==
            RGSADDLE_NULL_SESSION,
        "array null cfg");

  rgsaddle_stencil_config_t *cfg = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_LANCZOS);
  CHECK(cfg != NULL, "lanczos stencil cfg");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_DR, 1e-3) == RGSADDLE_OK, "set dr");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_FORCE_TOL, 1e-4) == RGSADDLE_OK, "set ftol");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_MAX_MOVE, 0.05) == RGSADDLE_OK, "set max_move");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_ROTATION_TOL, 1e-4) == RGSADDLE_OK, "set rtol");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_ART_THRESHOLD, 0.0) == RGSADDLE_OK, "set art thr");
  CHECK(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, 1.0) ==
            RGSADDLE_INVALID_PARAMETER,
        "wrong f64 key");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, -1) ==
            RGSADDLE_INVALID_PARAMETER,
        "negative i64");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, 20) == RGSADDLE_OK, "set max_rot");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_KRYLOV_DIM, 6) == RGSADDLE_OK, "set krylov");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_THREADS, 1) == RGSADDLE_OK, "set threads");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ART_PUSH, 10) == RGSADDLE_OK, "set art_push");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_DR, 1) ==
            RGSADDLE_INVALID_PARAMETER,
        "wrong i64 key");
  CHECK(rgsaddle_stencil_config_set_i64(cfg, 999, 1) == RGSADDLE_INVALID_PARAMETER, "unknown i64 key");
  CHECK(rgsaddle_stencil_config_set_array(cfg, 999, NULL, 0) == RGSADDLE_INVALID_PARAMETER, "unknown arr key");
  CHECK(rgsaddle_stencil_config_set_array(cfg, RGSADDLE_STENCIL_KEY_MODE, NULL, 3) ==
            RGSADDLE_INVALID_PARAMETER,
        "null arr with n>0");
  const double bad_mode[3] = {NAN, 0.0, 0.0};
  CHECK(rgsaddle_stencil_config_set_array(cfg, RGSADDLE_STENCIL_KEY_MODE, bad_mode, 3) ==
            RGSADDLE_NON_FINITE,
        "nan mode");
  const double seed[3] = {1.0, -0.5, 0.0};
  CHECK(rgsaddle_stencil_config_set_array(cfg, RGSADDLE_STENCIL_KEY_MODE, seed, 3) == RGSADDLE_OK, "valid mode");

  const double start[3] = {0.15, 0.25, 0.0};
  CHECK(rgsaddle_stencil_new(NULL, 1, 3, start) == NULL, "new null cfg");
  CHECK(rgsaddle_stencil_new(cfg, 2, 3, start) == NULL, "new rows!=1");
  CHECK(rgsaddle_stencil_new(cfg, 1, 0, start) == NULL, "new dof==0");
  CHECK(rgsaddle_stencil_new(cfg, 1, 3, NULL) == NULL, "new null x");

  rgsaddle_stencil_t *st = rgsaddle_stencil_new(cfg, 1, 3, start);
  rgsaddle_stencil_config_free(cfg);
  rgsaddle_stencil_config_free(NULL);
  CHECK(st != NULL, "stencil new");

  rgsaddle_stencil_report_t rep;
  memset(&rep, 0, sizeof rep);
  CHECK(rgsaddle_stencil_step(NULL, stencil_mb_eval, NULL, &rep) == RGSADDLE_NULL_SESSION, "step null");
  CHECK(rgsaddle_stencil_step(st, NULL, NULL, &rep) == RGSADDLE_NULL_SURFACE, "step null eval");
  CHECK(rgsaddle_stencil_run(NULL, stencil_mb_eval, NULL, 10, &rep) == RGSADDLE_NULL_SESSION, "run null");
  CHECK(rgsaddle_stencil_run(st, NULL, NULL, 10, &rep) == RGSADDLE_NULL_SURFACE, "run null eval");
  rep.struct_size = 4;
  CHECK(rgsaddle_stencil_step(st, stencil_mb_eval, NULL, &rep) == RGSADDLE_INVALID_PARAMETER,
        "short struct_size");
  rep.struct_size = 0;

  int cb_mode = 1;
  CHECK(rgsaddle_stencil_step(st, stencil_mb_eval, &cb_mode, &rep) == RGSADDLE_SURFACE_FAILED,
        "cb nonzero");
  cb_mode = 2;
  CHECK(rgsaddle_stencil_step(st, stencil_mb_eval, &cb_mode, &rep) == RGSADDLE_NON_FINITE,
        "cb nan");
  cb_mode = 0;

  CHECK(rgsaddle_stencil_run(st, stencil_mb_eval, &cb_mode, 400, &rep) == RGSADDLE_OK, "stencil run");
  CHECK(rep.status == 1, "stencil converged on MB");

  double pos[3], mode[3], curv = 0.0;
  CHECK(rgsaddle_stencil_positions(NULL, pos, 3) == RGSADDLE_NULL_SESSION, "pos null");
  CHECK(rgsaddle_stencil_positions(st, pos, 2) == RGSADDLE_SHAPE, "pos bad len");
  CHECK(rgsaddle_stencil_query(NULL, RGSADDLE_STENCIL_QUERY_POSITIONS, pos, 3) == RGSADDLE_NULL_SESSION,
        "query null");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_POSITIONS, NULL, 3) ==
            RGSADDLE_INVALID_PARAMETER,
        "query null out");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_POSITIONS, pos, 2) == RGSADDLE_SHAPE,
        "query pos shape");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_POSITIONS, pos, 3) == RGSADDLE_OK,
        "query pos");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_MODE, mode, 2) == RGSADDLE_SHAPE,
        "query mode shape");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_MODE, mode, 3) == RGSADDLE_OK,
        "query mode");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_CURVATURE, &curv, 0) == RGSADDLE_SHAPE,
        "query curv shape");
  CHECK(rgsaddle_stencil_query(st, RGSADDLE_STENCIL_QUERY_CURVATURE, &curv, 1) == RGSADDLE_OK,
        "query curv");
  CHECK(rgsaddle_stencil_query(st, 999, &curv, 1) == RGSADDLE_INVALID_PARAMETER, "query unknown");
  CHECK(fabs(pos[0] - 0.21248658) < 0.02 && fabs(pos[1] - 0.29298833) < 0.02 && curv < 0.0,
        "MB saddle and negative curvature");

  rgsaddle_stencil_free(st);
  rgsaddle_stencil_free(NULL);

  /* Minimizer preset has no mode -> QUERY_MODE returns RGSADDLE_INVALID_PARAMETER. */
  rgsaddle_stencil_config_t *mcfg = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_MIN);
  rgsaddle_stencil_t *mst = rgsaddle_stencil_new(mcfg, 1, 3, start);
  rgsaddle_stencil_config_free(mcfg);
  CHECK(rgsaddle_stencil_query(mst, RGSADDLE_STENCIL_QUERY_MODE, mode, 3) ==
            RGSADDLE_INVALID_PARAMETER,
        "minimizer query mode");
  rgsaddle_stencil_free(mst);
  return 0;
}

static int test_band_minmode_index1_and_sessions_cdylib(void) {
  /* Exercise band, minmode, index1, sella, constraints, samd, pes, internal_pes, niggli through cdylib. */
  const double b[3] = {-0.050011, 0.466694, 0.0};
  const double c[3] = {0.623499, 0.028038, 0.0};
  double pos[7 * 3];
  for (int i = 0; i < 7; ++i) {
    double t = (double)i / 6.0;
    pos[3 * i + 0] = (1.0 - t) * b[0] + t * c[0];
    pos[3 * i + 1] = (1.0 - t) * b[1] + t * c[1];
    pos[3 * i + 2] = 0.0;
  }

  rgsaddle_band_config_t bcfg;
  memset(&bcfg, 0, sizeof bcfg);
  bcfg.version.major = RGSADDLE_ABI_MAJOR;
  bcfg.version.minor = RGSADDLE_ABI_MINOR;
  bcfg.flags = RGSADDLE_BAND_NO_MIN_MODE | RGSADDLE_BAND_CI_AFTER;
  bcfg.tangent = RGSADDLE_TANGENT_SIMPLE;
  bcfg.spring = RGSADDLE_SPRING_ONSAGER_MACHLUP;
  bcfg.projection = RGSADDLE_PROJECTION_DNEB;
  bcfg.method = RGSADDLE_METHOD_ODE12R;
  bcfg.spring_k = 5.0;
  bcfg.ci_trigger_factor = 0.0;
  bcfg.ci_trigger_force = 0.5;
  bcfg.force_tol = 1e-2;
  bcfg.max_move = 0.05;

  RgsaddleBand *band = rgsaddle_band_create(&bcfg, 7, 1, pos);
  CHECK(band != NULL, "ode12r OM DNEB create");
  CHECK(rgsaddle_band_set_force_gate(NULL, RGSADDLE_FORCE_L2) == RGSADDLE_NULL_BAND, "gate null band");
  CHECK(rgsaddle_band_set_force_gate(band, 99) == RGSADDLE_INVALID_PARAMETER, "gate invalid");
  CHECK(rgsaddle_band_set_force_gate(band, RGSADDLE_FORCE_L2) == RGSADDLE_OK, "band gate L2");
  CHECK(rgsaddle_band_set_highs(NULL, 1) == RGSADDLE_NULL_BAND, "highs null");
  CHECK(rgsaddle_band_set_highs(band, 2) == RGSADDLE_INVALID_PARAMETER, "highs bad");
  CHECK(rgsaddle_band_set_highs(band, 0) == RGSADDLE_OK, "highs off");
  CHECK(rgsaddle_band_set_exp_precon(NULL, 3.0, 0.1, 1.0, 2.0) == RGSADDLE_NULL_BAND, "exp null");
  /* Single-atom band refuses exp precon. */
  CHECK(rgsaddle_band_set_exp_precon(band, 3.0, 0.1, 1.0, 2.0) == RGSADDLE_INVALID_PARAMETER,
        "exp single atom refused");
  rgsaddle_report_t brep;
  memset(&brep, 0, sizeof brep);
  CHECK(rgsaddle_band_step(band, mb_surface, NULL, &brep) == RGSADDLE_OK, "ode12r OM step");
  CHECK(rgsaddle_band_restart(NULL) == RGSADDLE_NULL_BAND, "restart null");
  CHECK(rgsaddle_band_restart(band) == RGSADDLE_OK, "band restart");
  pos[0] = NAN;
  CHECK(rgsaddle_band_set_positions(band, pos) == RGSADDLE_NON_FINITE, "band nan pos");
  rgsaddle_band_free(band);

  /* 2-atom band with periodic cell + Exp precon + PlainEB + weighted springs. */
  double p2[5 * 6];
  for (int i = 0; i < 5; ++i) {
    double t = (double)i / 4.0;
    p2[6 * i + 0] = 0.0;
    p2[6 * i + 1] = 0.0;
    p2[6 * i + 2] = 0.0;
    p2[6 * i + 3] = 0.8 + 0.4 * t;
    p2[6 * i + 4] = 0.2 * sin(3.14159265358979 * t);
    p2[6 * i + 5] = 0.0;
  }
  const double cell[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const double ks[4] = {2.0, 4.0, 4.0, 2.0};
  bcfg.spring = RGSADDLE_SPRING_WEIGHTED;
  bcfg.spring_ks = ks;
  bcfg.projection = RGSADDLE_PROJECTION_PLAIN_EB;
  bcfg.method = RGSADDLE_METHOD_LBFGS;
  bcfg.cell = cell;
  RgsaddleBand *wband = rgsaddle_band_create(&bcfg, 5, 2, p2);
  CHECK(wband != NULL, "weighted band");
  CHECK(rgsaddle_band_step(wband, two_atom_well, NULL, &brep) == RGSADDLE_OK, "weighted step");
  rgsaddle_band_free(wband);

  /* Uniform 2-atom band with Exp precon. */
  bcfg.spring = RGSADDLE_SPRING_UNIFORM;
  bcfg.spring_ks = NULL;
  bcfg.method = RGSADDLE_METHOD_ODE12R;
  RgsaddleBand *eband = rgsaddle_band_create(&bcfg, 5, 2, p2);
  CHECK(eband != NULL, "2-atom band for exp");
  CHECK(rgsaddle_band_set_exp_precon(eband, -1.0, -1.0, -1.0, -1.0) == RGSADDLE_OK,
        "exp precon defaults");
  CHECK(rgsaddle_band_step(eband, two_atom_well, NULL, &brep) == RGSADDLE_OK, "exp ode12r step");
  rgsaddle_band_free(eband);

  /* MinMode cdylib coverage: set_force_gate, set_highs, set_kappa, set_rotation_stop_length, reset. */
  rgsaddle_minmode_config_t mm;
  memset(&mm, 0, sizeof mm);
  mm.version.major = RGSADDLE_ABI_MAJOR;
  mm.version.minor = RGSADDLE_ABI_MINOR;
  mm.kind = RGSADDLE_MINMODE_LANCZOS;
  mm.method = RGSADDLE_METHOD_LBFGS;
  mm.dr = 1e-3;
  mm.rotation_tol = 1e-4;
  mm.max_rotations = 20;
  mm.krylov_dim = 6;
  mm.force_tol = 1e-4;
  mm.max_move = 0.05;
  double x_mb[3] = {0.15, 0.25, 0.0};
  double m_mb[3] = {1.0, -0.5, 0.0};
  RgsaddleMinMode *ms = rgsaddle_minmode_create(&mm, 1, x_mb, m_mb);
  CHECK(ms != NULL, "minmode create");
  CHECK(rgsaddle_minmode_set_force_gate(NULL, RGSADDLE_FORCE_L2) == RGSADDLE_NULL_SESSION, "mm gate null");
  CHECK(rgsaddle_minmode_set_force_gate(ms, 99) == RGSADDLE_INVALID_PARAMETER, "mm gate bad");
  CHECK(rgsaddle_minmode_set_force_gate(ms, RGSADDLE_FORCE_L2) == RGSADDLE_OK, "mm gate ok");
  CHECK(rgsaddle_minmode_set_highs(NULL, 1) == RGSADDLE_NULL_SESSION, "mm highs null");
  CHECK(rgsaddle_minmode_set_highs(ms, 2) == RGSADDLE_INVALID_PARAMETER, "mm highs bad");
  CHECK(rgsaddle_minmode_set_highs(ms, 0) == RGSADDLE_OK, "mm highs ok");
  CHECK(rgsaddle_minmode_set_rotation_stop_length(ms, 0.1) == RGSADDLE_OK, "mm rot len");
  CHECK(rgsaddle_minmode_set_rotation_stop_length(ms, -1.0) == RGSADDLE_INVALID_PARAMETER, "mm neg rot len");

  rgsaddle_kappa_config_t kcfg;
  CHECK(rgsaddle_kappa_config_default(&kcfg) == RGSADDLE_OK, "kappa default");
  CHECK(rgsaddle_minmode_set_kappa(NULL, &kcfg) == RGSADDLE_NULL_SESSION, "mm kappa null sess");
  CHECK(rgsaddle_minmode_set_kappa(ms, &kcfg) == RGSADDLE_OK, "mm set kappa");
  CHECK(rgsaddle_minmode_set_kappa(ms, NULL) == RGSADDLE_OK, "mm clear kappa");
  CHECK(rgsaddle_minmode_set_position(NULL, x_mb, NULL) == RGSADDLE_NULL_SESSION, "mm setpos null");
  CHECK(rgsaddle_minmode_set_position(ms, NULL, NULL) == RGSADDLE_INVALID_PARAMETER, "mm setpos null x");
  CHECK(rgsaddle_minmode_reset(NULL) == RGSADDLE_NULL_SESSION, "mm reset null");
  CHECK(rgsaddle_minmode_reset(ms) == RGSADDLE_OK, "mm reset");
  CHECK(rgsaddle_minmode_step(ms, mb_surface, NULL, &brep) == RGSADDLE_OK, "mm step after reset");
  rgsaddle_minmode_free(ms);

  /* Index1 reset, hessian before/after, bofill, trust_update, connect errors. */
  rgsaddle_index1_config_t icfg;
  memset(&icfg, 0, sizeof icfg);
  icfg.version.major = RGSADDLE_ABI_MAJOR;
  icfg.version.minor = RGSADDLE_ABI_MINOR;
  icfg.update = RGSADDLE_HESS_POWELL;
  icfg.mode = RGSADDLE_NICHOLS_INDEX1;
  icfg.trust_radius = 0.5;
  icfg.force_tol = 1e-5;
  icfg.fd_dr = 1e-4;
  RgsaddleIndex1 *idx1 = rgsaddle_index1_create(&icfg, 1, x_mb, NULL, NULL);
  CHECK(idx1 != NULL, "index1 create without hessian");
  double h9[9];
  CHECK(rgsaddle_index1_hessian(idx1, h9) == RGSADDLE_SHAPE, "hessian unset");
  CHECK(rgsaddle_index1_step(idx1, mb_surface, NULL, &brep) == RGSADDLE_OK, "index1 fd step");
  CHECK(rgsaddle_index1_hessian(idx1, h9) == RGSADDLE_OK, "hessian after step");
  CHECK(rgsaddle_index1_reset(NULL) == RGSADDLE_NULL_SESSION, "index1 reset null");
  CHECK(rgsaddle_index1_reset(idx1) == RGSADDLE_OK, "index1 reset");
  CHECK(rgsaddle_index1_hessian(idx1, h9) == RGSADDLE_SHAPE, "hessian cleared after reset");
  rgsaddle_index1_free(idx1);

  double h4[4] = {2.0, 0.0, 0.0, 2.0};
  const double s2[2] = {0.1, 0.2}, y2[2] = {0.2, 0.4};
  CHECK(rgsaddle_hessian_bofill(2, s2, y2, h4) == RGSADDLE_OK, "bofill update");
  double rad = 0.0;
  CHECK(rgsaddle_trust_update(0.1, -0.05, -0.05, 0.1, 1.15, 0.65, 1.035, 5.0, 1e-4, &rad) ==
            RGSADDLE_OK &&
            rad > 0.1,
        "trust_update grows");

  rgsaddle_connect_config_t ccfg;
  memset(&ccfg, 0, sizeof ccfg);
  ccfg.version.major = RGSADDLE_ABI_MAJOR;
  ccfg.version.minor = RGSADDLE_ABI_MINOR;
  ccfg.force_tol = 1e-3;
  ccfg.max_calls = 500;
  ccfg.quench_nodes = 0;
  rgsaddle_connect_report_t crep;
  double mb_out[3];
  CHECK(rgsaddle_connect(&ccfg, 1, b, c, failing_surface, NULL, mb_out, &crep) ==
            RGSADDLE_SURFACE_FAILED,
        "connect surface fail");
  CHECK(rgsaddle_connect(&ccfg, 1, b, c, mb_surface, NULL, mb_out, &crep) == RGSADDLE_OK,
        "connect MB quench_nodes=0");
  CHECK(crep.kind == 's' && crep.hessian_index == 1, "connect MB found saddle");

  /* Sella, Constraints, SAMD, PES, InternalPES, Niggli via cdylib. */
  rgsaddle_constraints_config_t cns_cfg = {{RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR}, 0};
  RgsaddleConstraints *cons = rgsaddle_constraints_create(&cns_cfg, 2);
  CHECK(cons != NULL, "constraints create");
  double x2[6] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0};
  const double m2[2] = {1.0, 1.0};
  double target_len = 1.0;
  CHECK(rgsaddle_constraints_fix_com(cons, x2) == RGSADDLE_OK, "fix_com");
  CHECK(rgsaddle_constraints_fix_bond(cons, 0, 1, x2, &target_len) == RGSADDLE_OK, "fix_bond");
  double res_norm = 0.0;
  CHECK(rgsaddle_constraints_residual_norm(cons, x2, &res_norm) == RGSADDLE_OK &&
            fabs(res_norm - 0.2) < 1e-12,
        "residual_norm");
  double vec[6] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double proj[6] = {0};
  CHECK(rgsaddle_constraints_project(cons, x2, vec, proj) == RGSADDLE_OK, "project");
  CHECK(rgsaddle_constraints_retract(cons, x2, vec, proj) == RGSADDLE_OK, "retract");

  rgsaddle_sella_min_config_t sm_cfg = {{RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR}, 0, 0.2, 1e-3, RGSADDLE_FORCE_LINF};
  RgsaddleSellaMin *smin = rgsaddle_sella_min_create(&sm_cfg, 2, x2, m2);
  CHECK(smin != NULL, "sella_min create");
  CHECK(rgsaddle_sella_min_step(smin, two_atom_well, NULL, &brep) == RGSADDLE_OK, "sella_min step");
  CHECK(rgsaddle_sella_min_position(smin, proj) == RGSADDLE_OK, "sella_min pos");
  CHECK(rgsaddle_sella_min_set_hess_update(smin, 1) == RGSADDLE_OK, "sella_min tsbfgs");
  CHECK(rgsaddle_sella_min_set_restricted(smin, 0) == RGSADDLE_OK, "sella_min restricted");
  CHECK(rgsaddle_sella_min_reset(smin) == RGSADDLE_OK, "sella_min reset");
  rgsaddle_sella_min_free(smin);

  RgsaddleSellaMin *smin_on = rgsaddle_sella_min_create_on(&sm_cfg, 2, x2, m2, cons);
  CHECK(smin_on != NULL, "sella_min_on");
  rgsaddle_sella_min_free(smin_on);
  RgsaddleSellaMin *smin_int = rgsaddle_sella_min_create_internal(&sm_cfg, 2, x2, m2, cons);
  CHECK(smin_int != NULL, "sella_min_internal");
  rgsaddle_sella_min_free(smin_int);

  double c9[9] = {5.0, 0.0, 0.0, 1.5, 5.0, 0.0, 0.0, 0.0, 5.0};
  RgsaddleSellaMin *smin_cell = rgsaddle_sella_min_create_cell(&sm_cfg, 2, x2, m2, c9, NULL);
  CHECK(smin_cell != NULL, "sella_min_cell");
  int32_t applied = 0;
  CHECK(rgsaddle_sella_min_set_cell_chart(smin_cell, 1) == RGSADDLE_OK, "cell chart log");
  CHECK(rgsaddle_sella_min_maybe_niggli(smin_cell, 10.0, &applied) == RGSADDLE_OK, "min niggli");
  rgsaddle_sella_min_free(smin_cell);

  rgsaddle_sella_saddle_config_t ss_cfg = {{RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR}, 0, 0.2, 1e-3, RGSADDLE_FORCE_LINF, 1};
  RgsaddleSellaSaddle *ssad = rgsaddle_sella_saddle_create(&ss_cfg, 1, x_mb, m2);
  CHECK(ssad != NULL, "sella_saddle create");
  CHECK(rgsaddle_sella_saddle_seed_mode(ssad, m_mb, -1.0) == RGSADDLE_OK, "seed_mode");
  CHECK(rgsaddle_sella_saddle_set_expand(ssad, 0) == RGSADDLE_OK, "set_expand");
  CHECK(rgsaddle_sella_saddle_set_symmetrized_ritz(ssad, 1) == RGSADDLE_OK, "sym ritz");
  CHECK(rgsaddle_sella_saddle_set_small_system_atom_ceiling(ssad, 1) == RGSADDLE_OK, "atom ceiling");
  CHECK(rgsaddle_sella_saddle_step(ssad, mb_surface, NULL, &brep) == RGSADDLE_OK, "sella_saddle step");
  CHECK(rgsaddle_sella_saddle_position(ssad, mb_out) == RGSADDLE_OK, "sella_saddle pos");
  CHECK(rgsaddle_sella_saddle_reset(ssad) == RGSADDLE_OK, "sella_saddle reset");
  rgsaddle_sella_saddle_free(ssad);

  RgsaddlePes *pes = rgsaddle_pes_create(2, x2, m2, 1, 0);
  CHECK(pes != NULL, "pes create");
  double d6[6] = {0.01, 0.0, 0.0, -0.01, 0.0, 0.0};
  CHECK(rgsaddle_pes_kick(pes, two_atom_well, NULL, d6) == RGSADDLE_OK, "pes kick");
  CHECK(rgsaddle_pes_position(pes, proj) == RGSADDLE_OK, "pes pos");
  CHECK(rgsaddle_pes_set_hess_update(pes, 1) == RGSADDLE_OK, "pes hess update");
  CHECK(rgsaddle_pes_project(pes, vec, proj) == RGSADDLE_OK, "pes project");
  CHECK(rgsaddle_pes_retract(pes, d6, proj) == RGSADDLE_OK, "pes retract");
  CHECK(rgsaddle_pes_transport(pes, x2, vec, proj) == RGSADDLE_OK, "pes transport");
  rgsaddle_pes_free(pes);

  RgsaddleInternalPes *ipes = rgsaddle_internal_pes_create(2, x2, m2, cons);
  CHECK(ipes != NULL, "internal_pes create");
  int64_t n_int = 0;
  CHECK(rgsaddle_internal_pes_n_int(ipes, &n_int) == RGSADDLE_OK && n_int > 0, "n_int");
  double q_int[8] = {0};
  CHECK(rgsaddle_internal_pes_internals(ipes, q_int) == RGSADDLE_OK, "internals");
  CHECK(rgsaddle_internal_pes_position(ipes, proj) == RGSADDLE_OK, "ipes pos");
  double g_int[8] = {0};
  CHECK(rgsaddle_internal_pes_grad(ipes, vec, g_int) == RGSADDLE_OK, "ipes grad");
  double dq[8] = {0};
  CHECK(rgsaddle_internal_pes_kick(ipes, two_atom_well, NULL, dq) == RGSADDLE_OK, "ipes kick");
  CHECK(rgsaddle_internal_pes_set_hess_update(ipes, 1) == RGSADDLE_OK, "ipes hess update");
  rgsaddle_internal_pes_free(ipes);

  double dummy_out[3] = {0};
  CHECK(rgsaddle_place_perp_dummy(2, x2, 0, 1, dummy_out) == RGSADDLE_OK, "place_perp_dummy");
  CHECK(rgsaddle_niggli_reduce(c9, 10.0, &applied) == RGSADDLE_OK, "niggli_reduce");

  rgsaddle_samd_config_t samd_cfg = {{RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR}, 0, 0.05, 1.0, 300.0, 100.0, 10, 1};
  double v0[6] = {0};
  RgsaddleSamd *samd = rgsaddle_samd_create(&samd_cfg, 2, x2, v0, two_atom_well, NULL);
  CHECK(samd != NULL, "samd create");
  double draw[6] = {0.1, -0.1, 0.0, 0.0, 0.1, -0.1};
  CHECK(rgsaddle_samd_step(samd, two_atom_well, NULL, draw, &brep) == RGSADDLE_OK, "samd step");
  CHECK(rgsaddle_samd_position(samd, proj) == RGSADDLE_OK, "samd pos");
  rgsaddle_samd_free(samd);

  rgsaddle_constraints_free(cons);
  return 0;
}

int main(void) {
  if (test_status_and_gates() != 0) {
    return 1;
  }
  if (test_geodesic_config_and_status() != 0) {
    return 1;
  }
  if (test_ocineb_lanczos_rtr_mb() != 0) {
    return 1;
  }
  if (test_indexk_contracts() != 0) {
    return 1;
  }
  if (test_solid_per_image_and_errors() != 0) {
    return 1;
  }
  if (test_stencil_contracts() != 0) {
    return 1;
  }
  if (test_band_minmode_index1_and_sessions_cdylib() != 0) {
    return 1;
  }
  if (test_minmode_rotation_angle_defaults_and_torque_window() != 0) {
    return 1;
  }
  printf("RGSADDLE_CAPI_CONTRACTS_OK\n");
  return 0;
}
