/* Held atoms stay put through the C API, and the new setters accept
 * the built-in defaults. */
#include "rgsaddle.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  const long n = req->n_images;
  const long atoms = req->n_atoms;
  const double *p = req->positions;
  double *g = req->gradients;
  for (long image = 0; image < n; image++) {
    double energy = 0.0;
    for (long atom = 0; atom < atoms; atom++) {
      const long base = (image * atoms + atom) * 3;
      for (int axis = 0; axis < 3; axis++) {
        const double x = p[base + axis];
        energy += 0.5 * x * x;
        g[base + axis] = x;
      }
    }
    req->energies[image] = energy;
  }
  return RGSADDLE_OK;
}

static rgsaddle_band_config_t config(int method) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_NO_MIN_MODE | RGSADDLE_BAND_KEEP_TRANSLATION;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = method;
  cfg.spring_k = 1.0;
  cfg.force_tol = 1e-12;
  cfg.max_move = 0.2;
  cfg.memory = 8;
  return cfg;
}

int main(void) {
  const int images = 5;
  const int atoms = 2;
  double pos[5 * 2 * 3];
  memset(pos, 0, sizeof pos);
  for (int image = 0; image < images; image++) {
    const double t = image / 4.0;
    pos[(image * atoms) * 3 + 0] = 0.3;
    pos[(image * atoms) * 3 + 1] = -0.2;
    pos[(image * atoms) * 3 + 2] = 0.15;
    pos[(image * atoms + 1) * 3 + 0] = -1.0 + 2.0 * t;
    pos[(image * atoms + 1) * 3 + 1] = 0.2;
  }
  rgsaddle_band_config_t cfg = config(RGSADDLE_METHOD_ODE12R);
  RgsaddleBand *band =
      rgsaddle_band_create(&cfg, images, atoms, pos);
  if (band == NULL) {
    fprintf(stderr, "create failed\n");
    return 1;
  }
  if (rgsaddle_band_set_fixed_mask(NULL, NULL, 0) != RGSADDLE_NULL_BAND) {
    fprintf(stderr, "null band was accepted\n");
    return 1;
  }
  int32_t short_mask[1] = {1};
  if (rgsaddle_band_set_fixed_mask(band, short_mask, 1) != RGSADDLE_SHAPE) {
    fprintf(stderr, "short mask was accepted\n");
    return 1;
  }
  int32_t mask[2] = {1, 0};
  if (rgsaddle_band_set_fixed_mask(band, mask, 2) != RGSADDLE_OK) {
    fprintf(stderr, "mask rejected\n");
    return 1;
  }
  if (rgsaddle_band_set_exp_r_cut(band, 3.0) != RGSADDLE_INVALID_PARAMETER) {
    fprintf(stderr, "r_cut before the metric was accepted\n");
    return 1;
  }
  if (rgsaddle_band_set_exp_precon(band, 3.0, 0.1, 1.0, 2.0) != RGSADDLE_OK) {
    fprintf(stderr, "exp precon rejected\n");
    return 1;
  }
  if (rgsaddle_band_set_exp_r_cut(band, 2.5) != RGSADDLE_OK) {
    fprintf(stderr, "absolute r_cut rejected\n");
    return 1;
  }
  if (rgsaddle_band_set_ode12r_tolerances(band, 0.0, 0.0, 0.0, 0.0) !=
      RGSADDLE_OK) {
    fprintf(stderr, "default tolerances rejected\n");
    return 1;
  }
  if (rgsaddle_band_set_ode12r_tolerances(band, NAN, 0.0, 0.01, 2.0) !=
      RGSADDLE_INVALID_PARAMETER) {
    fprintf(stderr, "non-finite rtol was accepted\n");
    return 1;
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof report);
  for (int step = 0; step < 4; step++) {
    if (rgsaddle_band_step(band, surface, NULL, &report) != RGSADDLE_OK) {
      fprintf(stderr, "step %d failed\n", step);
      return 1;
    }
  }
  double out[5 * 2 * 3];
  if (rgsaddle_band_positions(band, out) != RGSADDLE_OK) {
    fprintf(stderr, "positions failed\n");
    return 1;
  }
  for (int image = 0; image < images; image++) {
    for (int axis = 0; axis < 3; axis++) {
      const int at = (image * atoms) * 3 + axis;
      if (out[at] != pos[at]) {
        fprintf(stderr, "held atom moved: image %d axis %d %g -> %g\n", image,
                axis, pos[at], out[at]);
        return 1;
      }
    }
  }
  cfg = config(RGSADDLE_METHOD_FIRE);
  RgsaddleBand *fire = rgsaddle_band_create(&cfg, images, atoms, pos);
  if (fire == NULL) {
    fprintf(stderr, "fire create failed\n");
    return 1;
  }
  if (rgsaddle_band_set_ode12r_tolerances(fire, 0.1, 0.0, 0.01, 2.0) !=
      RGSADDLE_INVALID_PARAMETER) {
    fprintf(stderr, "tolerances on FIRE were accepted\n");
    return 1;
  }
  if (rgsaddle_band_set_fixed_mask(fire, mask, 2) != RGSADDLE_OK) {
    fprintf(stderr, "fire mask rejected\n");
    return 1;
  }
  for (int step = 0; step < 3; step++) {
    if (rgsaddle_band_step(fire, surface, NULL, &report) != RGSADDLE_OK) {
      fprintf(stderr, "fire step %d failed\n", step);
      return 1;
    }
  }
  if (rgsaddle_band_positions(fire, out) != RGSADDLE_OK) {
    fprintf(stderr, "fire positions failed\n");
    return 1;
  }
  for (int image = 0; image < images; image++) {
    for (int axis = 0; axis < 3; axis++) {
      const int at = (image * atoms) * 3 + axis;
      if (out[at] != pos[at]) {
        fprintf(stderr, "FIRE moved a held atom\n");
        return 1;
      }
    }
  }
  rgsaddle_band_free(fire);
  rgsaddle_band_free(band);
  printf("RGSADDLE_FIXED_MASK_OK\n");
  return 0;
}
