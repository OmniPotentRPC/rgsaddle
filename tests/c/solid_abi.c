/* Solid-state band through the C ABI: a cell double well in h00 with
 * minima at 4.0 and 4.4 and its barrier at 4.2. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_ATOMS 2
#define N_IMAGES 5

static const double C = 50.0;
static const double K = 5.0;
static int calls = 0;

static double det3(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

/* E(h) = C ((h00 - 4)(h00 - 4.4))^2 + K/2 ((h11 - 4)^2 + (h22 - 4)^2).
 * sigma = h^T (dE/dh) / V for the right strain h <- h (I + eps). */
static void one(const double *cell, double *energy, double *gradient, double *stress) {
  const double a = cell[0], b = cell[4], c = cell[8];
  const double w = (a - 4.0) * (a - 4.4);
  *energy = C * w * w + 0.5 * K * ((b - 4.0) * (b - 4.0) + (c - 4.0) * (c - 4.0));
  double g[9] = {0};
  g[0] = 2.0 * C * w * (2.0 * a - 8.4);
  g[4] = K * (b - 4.0);
  g[8] = K * (c - 4.0);
  const double v = fabs(det3(cell));
  for (int r = 0; r < 3; ++r) {
    for (int col = 0; col < 3; ++col) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += cell[3 * k + r] * g[3 * k + col];
      }
      stress[3 * r + col] = s / v;
    }
  }
  memset(gradient, 0, 3 * N_ATOMS * sizeof(double));
}

static rgsaddle_status_t surface(void *user, rgsaddle_solid_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR || req->n_atoms != N_ATOMS) {
    return RGSADDLE_ABI_MISMATCH;
  }
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  for (int64_t r = 0; r < rows; ++r) {
    one(req->cells + 9 * r, req->energies + r, req->gradients + 3 * N_ATOMS * r,
        req->stresses + 9 * r);
    ++calls;
  }
  return RGSADDLE_OK;
}

static rgsaddle_status_t atomic(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  (void)req;
  return RGSADDLE_OK;
}

int main(void) {
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
  cfg.force_tol = 1e-4;
  cfg.max_move = 0.05;

  double cells[N_IMAGES * 9];
  double positions[N_IMAGES * 3 * N_ATOMS];
  for (int i = 0; i < N_IMAGES; ++i) {
    const double a = 4.0 + 0.1 * i;
    double *h = cells + 9 * i;
    memset(h, 0, 9 * sizeof(double));
    h[0] = a;
    h[4] = 4.0;
    h[8] = 4.0;
    double *p = positions + 3 * N_ATOMS * i;
    p[0] = p[1] = p[2] = 0.0;
    p[3] = 0.5 * a;
    p[4] = p[5] = 2.0;
  }
  RgsaddleBand *band =
      rgsaddle_band_create_solid(&cfg, N_IMAGES, N_ATOMS, positions, cells, 0.0, 1.0);
  if (band == NULL) {
    printf("create_solid failed\n");
    return 1;
  }
  double jacobian = 0.0;
  if (rgsaddle_band_solid_jacobian(band, &jacobian) != RGSADDLE_OK) {
    return 2;
  }
  /* Endpoint volumes 64 and 70.4, mean 67.2 over 2 atoms: (33.6)^(1/3) sqrt(2). */
  if (fabs(jacobian - cbrt(33.6) * sqrt(2.0)) > 1e-12) {
    printf("jacobian %.15g\n", jacobian);
    return 3;
  }
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  if (rgsaddle_band_step(band, atomic, NULL, &rep) != RGSADDLE_INVALID_PARAMETER) {
    printf("the atomic step was accepted on a solid band\n");
    return 4;
  }
  int status = RGSADDLE_OK;
  for (int it = 0; it < 2000 && rep.status != RGSADDLE_STATUS_CONVERGED; ++it) {
    status = rgsaddle_band_step_solid(band, surface, NULL, &rep);
    if (status != RGSADDLE_OK) {
      printf("step %d: %s\n", it, rgsaddle_status_name(status));
      return 5;
    }
  }
  if (rep.status != RGSADDLE_STATUS_CONVERGED) {
    printf("not converged: max_force %g\n", rep.max_force);
    return 6;
  }
  double out[N_IMAGES * 9];
  if (rgsaddle_band_cells(band, out) != RGSADDLE_OK) {
    return 7;
  }
  if (rep.ci_index < 1 || rep.ci_index > N_IMAGES - 2) {
    printf("no climbing image\n");
    return 8;
  }
  const double *ci = out + 9 * rep.ci_index;
  if (fabs(ci[0] - 4.2) > 1e-3 || fabs(out[0] - 4.0) > 1e-12 ||
      fabs(out[9 * (N_IMAGES - 1)] - 4.4) > 1e-12) {
    printf("climbing cell %.6f endpoints %.6f %.6f\n", ci[0], out[0], out[9 * (N_IMAGES - 1)]);
    return 9;
  }
  for (int i = 0; i < N_IMAGES; ++i) {
    const double *h = out + 9 * i;
    if (h[1] != 0.0 || h[2] != 0.0 || h[5] != 0.0) {
      printf("image %d left the lower-triangular form\n", i);
      return 10;
    }
  }
  rgsaddle_band_free(band);

  /* A fixed-cell band refuses the solid step. */
  double flat[3 * 3] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  RgsaddleBand *plain = rgsaddle_band_create(&cfg, 3, 1, flat);
  if (plain == NULL || rgsaddle_band_step_solid(plain, surface, NULL, &rep) !=
                           RGSADDLE_INVALID_PARAMETER) {
    printf("the solid step was accepted on a fixed-cell band\n");
    return 11;
  }
  if (rgsaddle_band_set_quickmin_variant(plain, RGSADDLE_QUICKMIN_PER_ATOM) !=
      RGSADDLE_INVALID_PARAMETER) {
    printf("a FIRE band took a quick-min variant\n");
    return 12;
  }
  rgsaddle_band_free(plain);
  cfg.method = RGSADDLE_METHOD_QUICKMIN;
  RgsaddleBand *quick = rgsaddle_band_create(&cfg, 3, 1, flat);
  if (quick == NULL ||
      rgsaddle_band_set_quickmin_variant(quick, RGSADDLE_QUICKMIN_PER_ATOM) != RGSADDLE_OK ||
      rgsaddle_band_set_quickmin_variant(quick, RGSADDLE_QUICKMIN_EON) != RGSADDLE_OK ||
      rgsaddle_band_set_quickmin_variant(quick, 7) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_band_set_time_step(quick, 0.25) != RGSADDLE_OK ||
      rgsaddle_band_set_time_step(quick, -1.0) != RGSADDLE_INVALID_PARAMETER) {
    printf("quick-min variants were not taken as documented\n");
    return 13;
  }
  rgsaddle_band_free(quick);
  printf("calls %d iterations %lld climbing cell %.6f\n", calls, (long long)rep.iteration, ci[0]);
  printf("RGSADDLE_SOLID_ABI_OK\n");
  return 0;
}
