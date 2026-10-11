/* Drive the stencil family: CI-NEB, energy-weighted springs, zoom, and a dimer. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail;
static int g_saw_zero;
static int g_saw_end;

static void expect(int cond, const char *what) {
  if (!cond) {
    fprintf(stderr, "FAIL %s (%s)\n", what, rgsaddle_last_error());
    g_fail = 1;
  }
}

static void muller(double x, double y, double *e, double *gx, double *gy) {
  const double a[4] = {-200.0, -100.0, -170.0, 15.0};
  const double aa[4] = {-1.0, -1.0, -6.5, 0.7};
  const double b[4] = {0.0, 0.0, 11.0, 0.6};
  const double c[4] = {-10.0, -10.0, -6.5, 0.7};
  const double x0[4] = {1.0, 0.0, -0.5, -1.0};
  const double y0[4] = {0.0, 0.5, 1.5, 1.0};
  *e = 0.0;
  *gx = 0.0;
  *gy = 0.0;
  for (int term = 0; term < 4; ++term) {
    double dx = x - x0[term];
    double dy = y - y0[term];
    double exponent = aa[term] * dx * dx + b[term] * dx * dy + c[term] * dy * dy;
    double weight = a[term] * exp(exponent);
    double px = 2.0 * aa[term] * dx + b[term] * dy;
    double py = b[term] * dx + 2.0 * c[term] * dy;
    *e += weight;
    *gx += weight * px;
    *gy += weight * py;
  }
}

static int eval_mb(void *user, size_t rows, size_t dof, const uint64_t *ids, const double *x,
                    double *e, double *g) {
  (void)user;
  if (dof != 3 || ids == NULL) {
    return 1;
  }
  for (size_t row = 0; row < rows; ++row) {
    if (ids[row] == 0) {
      g_saw_zero = 1;
    }
    if (ids[row] == 6) {
      g_saw_end = 1;
    }
    double energy, gx, gy;
    muller(x[row * 3], x[row * 3 + 1], &energy, &gx, &gy);
    e[row] = energy;
    g[row * 3] = gx;
    g[row * 3 + 1] = gy;
    g[row * 3 + 2] = 0.0;
  }
  return 0;
}

static void linear_band(double *x, size_t images) {
  const double a[3] = {-0.050010822738, 0.466694104872, 0.0};
  const double b[3] = {0.623499404931, 0.028037758529, 0.0};
  for (size_t i = 0; i < images; ++i) {
    double t = (double)i / (double)(images - 1);
    for (int k = 0; k < 3; ++k) {
      x[i * 3 + k] = a[k] + t * (b[k] - a[k]);
    }
  }
}

static rgsaddle_stencil_report_t blank_report(void) {
  rgsaddle_stencil_report_t report;
  memset(&report, 0, sizeof(report));
  report.struct_size = sizeof(report);
  return report;
}

static void run_band(int32_t preset, int energy_weighted) {
  rgsaddle_stencil_config_t *cfg = rgsaddle_stencil_config_new(preset);
  expect(cfg != NULL, "config");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_THREADS, 1) == RGSADDLE_OK,
         "threads");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE, 1.0e9) ==
             RGSADDLE_OK,
         "trigger");
  if (energy_weighted) {
    expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_SPRING,
                                           RGSADDLE_STENCIL_SPRING_ENERGY_WEIGHTED) == RGSADDLE_OK,
           "ew model");
    expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_K_MIN, 0.972) == RGSADDLE_OK,
           "kmin");
    expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_K_MAX, 9.72) == RGSADDLE_OK,
           "kmax");
  }
  const size_t images = 7;
  double x[21];
  linear_band(x, images);
  g_saw_zero = 0;
  g_saw_end = 0;
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(cfg, images, 3, x);
  expect(handle != NULL, "stencil");
  rgsaddle_stencil_report_t report = blank_report();
  expect(rgsaddle_stencil_step(handle, eval_mb, NULL, &report) == RGSADDLE_OK, "step");
  expect(g_saw_zero && g_saw_end, "endpoint ids");
  double positions[21];
  expect(rgsaddle_stencil_positions(handle, positions, 21) == RGSADDLE_OK, "positions");
  double ids[7];
  expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_IMAGE_IDS, ids, 7) == RGSADDLE_OK,
         "ids");
  expect(ids[0] == 0.0 && ids[6] == 6.0, "stored endpoint ids");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static void run_zoom(void) {
  rgsaddle_stencil_config_t *cfg = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_ZOOM_NEB);
  expect(cfg != NULL, "zoom config");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_STABILITY, 1) == RGSADDLE_OK,
         "stability");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND, 1) == RGSADDLE_OK,
         "force kind");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_FORCE, 1.0e9) == RGSADDLE_OK,
         "zoom force");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_ENDS,
                                         RGSADDLE_STENCIL_ZOOM_ENDS_FIXED) == RGSADDLE_OK,
         "fixed ends");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE, 1.0e300) ==
             RGSADDLE_OK,
         "zoom trigger");
  const size_t images = 7;
  double x[21];
  linear_band(x, images);
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(cfg, images, 3, x);
  expect(handle != NULL, "zoom stencil");
  int saw_zoom = 0;
  for (int step = 0; step < 8; ++step) {
    rgsaddle_stencil_report_t report = blank_report();
    expect(rgsaddle_stencil_step(handle, eval_mb, NULL, &report) == RGSADDLE_OK, "zoom step");
    if (report.phase == RGSADDLE_STENCIL_PHASE_ZOOM) {
      saw_zoom = 1;
      expect(report.zoom_lo >= 0 && report.zoom_hi > report.zoom_lo, "window");
      expect(report.evaluations == 0, "zoom spends no rows");
      double window[2] = {-1.0, -1.0};
      expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW, window, 2) ==
                 RGSADDLE_OK,
             "query window");
      expect(window[0] == (double)report.zoom_lo && window[1] == (double)report.zoom_hi,
             "query matches");
      break;
    }
  }
  expect(saw_zoom, "zoom phase");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static int bowl(void *user, size_t rows, size_t dof, const uint64_t *ids, const double *x, double *e,
                double *g) {
  (void)user;
  (void)ids;
  if (rows != 1 || dof != 2) {
    return 1;
  }
  e[0] = 2.0 * x[0] * x[0] + 5.0 * x[1] * x[1];
  g[0] = 4.0 * x[0];
  g[1] = 10.0 * x[1];
  return 0;
}

static void run_dimer(void) {
  rgsaddle_stencil_config_t *cfg = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_DIMER);
  expect(cfg != NULL, "dimer config");
  const double mode[2] = {1.0, 0.0};
  expect(rgsaddle_stencil_config_set_array(cfg, RGSADDLE_STENCIL_KEY_MODE, mode, 2) == RGSADDLE_OK,
         "mode seed");
  double x[2] = {0.4, 0.3};
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(cfg, 1, 2, x);
  expect(handle != NULL, "dimer");
  rgsaddle_stencil_report_t report = blank_report();
  expect(rgsaddle_stencil_step(handle, bowl, NULL, &report) == RGSADDLE_OK, "dimer step");
  expect(report.phase == RGSADDLE_STENCIL_PHASE_BAND, "point phase");
  double out[2] = {0.0, 0.0};
  expect(rgsaddle_stencil_positions(handle, out, 2) == RGSADDLE_OK, "dimer positions");
  expect(out[1] == 0.3 && out[0] != 0.4, "dimer moved along x");
  double curvature = 0.0;
  expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_CURVATURE, &curvature, 1) ==
             RGSADDLE_OK,
         "curvature");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static void refuse_bad_keys(void) {
  rgsaddle_stencil_config_t *point = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_DIMER);
  expect(rgsaddle_stencil_config_set_f64(point, RGSADDLE_STENCIL_KEY_K, 1.0) != RGSADDLE_OK,
         "band key on a point preset");
  expect(strstr(rgsaddle_last_error(), "band stencil") != NULL, "band message");
  expect(rgsaddle_stencil_config_set_i64(point, 999, 1) != RGSADDLE_OK, "unknown key");
  expect(strstr(rgsaddle_last_error(), "feat/") == NULL, "no branch name");
  rgsaddle_stencil_config_free(point);

  rgsaddle_stencil_config_t *band = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_CI_NEB);
  expect(rgsaddle_stencil_config_set_f64(band, RGSADDLE_STENCIL_KEY_SPRING, 1.0) != RGSADDLE_OK,
         "wrong type");
  expect(strstr(rgsaddle_last_error(), "integer") != NULL, "type message");
  rgsaddle_stencil_config_free(band);

  rgsaddle_stencil_config_t *idpp = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_IDPP);
  double images[6] = {0.0, 0.0, 0.5, 0.0, 1.0, 0.0};
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(idpp, 3, 2, images);
  expect(handle == NULL, "idpp refused");
  const char *text = rgsaddle_last_error();
  expect(text != NULL && strstr(text, "not implemented") != NULL, "idpp message");
  expect(text != NULL && strstr(text, "feat/") == NULL, "idpp names no branch");
  rgsaddle_stencil_config_free(idpp);
}

int main(void) {
  run_band(RGSADDLE_STENCIL_CI_NEB, 0);
  run_band(RGSADDLE_STENCIL_CI_NEB, 1);
  run_zoom();
  run_dimer();
  refuse_bad_keys();
  if (g_fail) {
    return 1;
  }
  printf("RGSADDLE_STENCIL_C_OK\n");
  return 0;
}
