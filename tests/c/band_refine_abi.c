/* Per-step refine stepper at ABI minor 46. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

_Static_assert(RGSADDLE_REFINE_NONE == 0, "refine NONE value");
_Static_assert(RGSADDLE_REFINE_LBFGS == 1, "refine LBFGS value");

static rgsaddle_status_t fail(rgsaddle_status_t st, const char *msg) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(st));
  return st;
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

static void bow(double *pos, long n_images, double height) {
  for (long i = 0; i < n_images; ++i) {
    double t = (double)i / (double)(n_images - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = height * sin(3.141592653589793 * t);
    pos[i * 3 + 2] = 0.0;
  }
}

static int same_positions(const double *a, const double *b, long n) {
  for (long i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

static RgsaddleBand *open_band(const double *pos, long n_images, int32_t method, uint64_t flags) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = flags | RGSADDLE_BAND_NO_MIN_MODE;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = method;
  cfg.spring_k = 5.0;
  cfg.force_tol = 1e-3;
  cfg.max_move = 0.2;
  cfg.memory = 20;
  return rgsaddle_band_create(&cfg, n_images, 1, pos);
}

int main(void) {
  if (rgsaddle_abi_version() != (int)((RGSADDLE_ABI_MAJOR << 16) | 46u)) {
    return fail(RGSADDLE_ABI_MISMATCH, "ABI minor is not 46");
  }
  if (rgsaddle_band_set_refine(NULL, RGSADDLE_REFINE_LBFGS, 0.5, 20) != RGSADDLE_NULL_BAND) {
    return fail(RGSADDLE_NULL_BAND, "null band");
  }
  if (rgsaddle_band_is_refining(NULL) != 0) {
    return fail(RGSADDLE_INVALID_PARAMETER, "null is_refining");
  }

  const long n_images = 7;
  double pos[7 * 3];
  bow(pos, n_images, 0.4);

  RgsaddleBand *band = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, 0);
  if (!band) {
    return fail(RGSADDLE_SHAPE, "create");
  }
  if (rgsaddle_band_set_refine(band, 99, 0.5, 20) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "unknown method");
  }
  if (rgsaddle_band_set_refine(band, RGSADDLE_REFINE_LBFGS, -1.0, 20) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "negative threshold");
  }
  if (rgsaddle_band_set_refine(band, RGSADDLE_REFINE_LBFGS, nan(""), 20) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "NaN threshold");
  }
  if (rgsaddle_band_set_refine(band, RGSADDLE_REFINE_LBFGS, INFINITY, 20) !=
      RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "infinite threshold");
  }
  if (rgsaddle_band_set_refine(band, RGSADDLE_REFINE_NONE, -1.0, 20) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "none ignores the threshold");
  }

  if (rgsaddle_band_set_refine(band, RGSADDLE_REFINE_LBFGS, 1e300, 20) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "set refine");
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof report);
  if (rgsaddle_band_step(band, surface, NULL, &report) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "refine step");
  }
  if (rgsaddle_band_is_refining(band) != 1) {
    return fail(RGSADDLE_SOLVER, "a threshold over every force refines");
  }
  double refined[7 * 3];
  if (rgsaddle_band_positions(band, refined) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "positions");
  }
  rgsaddle_band_free(band);

  RgsaddleBand *alone = open_band(pos, n_images, RGSADDLE_METHOD_LBFGS, 0);
  if (!alone) {
    return fail(RGSADDLE_SHAPE, "lbfgs create");
  }
  memset(&report, 0, sizeof report);
  if (rgsaddle_band_step(alone, surface, NULL, &report) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "lbfgs step");
  }
  double lbfgs[7 * 3];
  if (rgsaddle_band_positions(alone, lbfgs) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "lbfgs positions");
  }
  rgsaddle_band_free(alone);
  if (!same_positions(refined, lbfgs, n_images * 3)) {
    return fail(RGSADDLE_SOLVER, "refine step differs from L-BFGS");
  }

  RgsaddleBand *fire = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, 0);
  if (!fire) {
    return fail(RGSADDLE_SHAPE, "fire create");
  }
  memset(&report, 0, sizeof report);
  if (rgsaddle_band_step(fire, surface, NULL, &report) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "fire step");
  }
  if (rgsaddle_band_is_refining(fire) != 0) {
    return fail(RGSADDLE_SOLVER, "FIRE alone is not refining");
  }
  double fired[7 * 3];
  if (rgsaddle_band_positions(fire, fired) != RGSADDLE_OK) {
    return fail(RGSADDLE_SHAPE, "fire positions");
  }
  rgsaddle_band_free(fire);
  if (same_positions(refined, fired, n_images * 3)) {
    return fail(RGSADDLE_SOLVER, "refine step matched FIRE");
  }

  RgsaddleBand *zero = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, 0);
  RgsaddleBand *half = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, 0);
  if (!zero || !half) {
    return fail(RGSADDLE_SHAPE, "threshold pair");
  }
  if (rgsaddle_band_set_refine(zero, RGSADDLE_REFINE_LBFGS, 0.0, 20) != RGSADDLE_OK ||
      rgsaddle_band_set_refine(half, RGSADDLE_REFINE_LBFGS, 0.5, 20) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "default threshold");
  }
  for (int step = 0; step < 8; ++step) {
    memset(&report, 0, sizeof report);
    if (rgsaddle_band_step(zero, surface, NULL, &report) != RGSADDLE_OK ||
        rgsaddle_band_step(half, surface, NULL, &report) != RGSADDLE_OK) {
      return fail(RGSADDLE_SOLVER, "threshold steps");
    }
  }
  double zero_pos[7 * 3], half_pos[7 * 3];
  rgsaddle_band_positions(zero, zero_pos);
  rgsaddle_band_positions(half, half_pos);
  rgsaddle_band_free(zero);
  rgsaddle_band_free(half);
  if (!same_positions(zero_pos, half_pos, n_images * 3)) {
    return fail(RGSADDLE_SOLVER, "a zero threshold is not 0.5");
  }

  RgsaddleBand *string_band = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, RGSADDLE_BAND_STRING);
  RgsaddleBand *plain_string = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, RGSADDLE_BAND_STRING);
  if (!string_band || !plain_string) {
    return fail(RGSADDLE_SHAPE, "string create");
  }
  if (rgsaddle_band_set_refine(string_band, RGSADDLE_REFINE_LBFGS, 1e300, 20) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "string refine");
  }
  for (int step = 0; step < 6; ++step) {
    memset(&report, 0, sizeof report);
    if (rgsaddle_band_step(string_band, surface, NULL, &report) != RGSADDLE_OK ||
        rgsaddle_band_step(plain_string, surface, NULL, &report) != RGSADDLE_OK) {
      return fail(RGSADDLE_SOLVER, "string step");
    }
    if (rgsaddle_band_is_refining(string_band) != 0) {
      return fail(RGSADDLE_SOLVER, "a string band refined");
    }
  }
  double string_pos[7 * 3], plain_pos[7 * 3];
  rgsaddle_band_positions(string_band, string_pos);
  rgsaddle_band_positions(plain_string, plain_pos);
  rgsaddle_band_free(string_band);
  rgsaddle_band_free(plain_string);
  if (!same_positions(string_pos, plain_pos, n_images * 3)) {
    return fail(RGSADDLE_SOLVER, "a string band used the refine stepper");
  }

  printf("RGSADDLE_BAND_REFINE_ABI_OK\n");
  return 0;
}
