/* Drive the stencil family on Müller-Brown and a point dimer. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static uint64_t g_seen_ids[16];
static size_t g_seen_n;
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
  if (dof != 3) {
    return 1;
  }
  g_seen_n = rows;
  for (size_t row = 0; row < rows && row < 16; ++row) {
    g_seen_ids[row] = ids[row];
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
  RgsaddleStencilConfig *cfg = rgsaddle_stencil_config_new(preset);
  expect(cfg != NULL, "config");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_THREADS, 1) == RGSADDLE_OK,
         "threads");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_STEPPER, 1) == RGSADDLE_OK,
         "lbfgs");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE, 1.0e9) ==
             RGSADDLE_OK,
         "trigger");
  if (energy_weighted) {
    expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_SPRING_MODEL, 1) == RGSADDLE_OK,
           "ew model");
    expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_K_MIN, 0.972) == RGSADDLE_OK,
           "kmin");
    expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_K_MAX, 9.72) == RGSADDLE_OK,
           "kmax");
  }
  const size_t images = 7;
  double x[21];
  linear_band(x, images);
  g_seen_n = 0;
  g_saw_zero = 0;
  g_saw_end = 0;
  RgsaddleStencil *handle = rgsaddle_stencil_new(cfg, images, 3, x);
  expect(handle != NULL, "stencil");
  rgsaddle_stencil_report_t report = blank_report();
  expect(rgsaddle_stencil_step(handle, eval_mb, NULL, &report) == RGSADDLE_OK, "step");
  expect(g_seen_n > 0, "callback rows");
  expect(g_saw_zero && g_saw_end, "endpoint ids");
  double positions[21];
  expect(rgsaddle_stencil_positions(handle, positions) == RGSADDLE_OK, "positions");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static void run_zoom(void) {
  RgsaddleStencilConfig *cfg = rgsaddle_stencil_config_new(RGSADDLE_PRESET_EON_ZOOM);
  expect(cfg != NULL, "zoom config");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_STABILITY, 1) == RGSADDLE_OK,
         "stability");
  expect(rgsaddle_stencil_config_set_i64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_FORCE_KIND, 1) == RGSADDLE_OK,
         "force kind");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_ZOOM_FORCE, 1.0e9) == RGSADDLE_OK,
         "zoom force");
  expect(rgsaddle_stencil_config_set_f64(cfg, RGSADDLE_STENCIL_KEY_CI_TRIGGER_FORCE, 1.0e9) ==
             RGSADDLE_OK,
         "zoom trigger");
  const size_t images = 7;
  double x[21];
  linear_band(x, images);
  RgsaddleStencil *handle = rgsaddle_stencil_new(cfg, images, 3, x);
  expect(handle != NULL, "zoom stencil");
  int saw_zoom = 0;
  for (int step = 0; step < 6; ++step) {
    rgsaddle_stencil_report_t report = blank_report();
    expect(rgsaddle_stencil_step(handle, eval_mb, NULL, &report) == RGSADDLE_OK, "zoom step");
    if (report.phase == RGSADDLE_STENCIL_PHASE_ZOOM) {
      saw_zoom = 1;
      expect(report.zoom_lo >= 0 && report.zoom_hi > report.zoom_lo, "window");
      expect(report.surface_rows == 0, "zoom spends no rows");
      int64_t window[2] = {-1, -1};
      expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_ZOOM_WINDOW, window, 2) ==
                 RGSADDLE_OK,
             "query window");
      expect(window[0] == report.zoom_lo && window[1] == report.zoom_hi, "query matches");
      break;
    }
  }
  expect(saw_zoom, "zoom phase");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static void run_dimer(void) {
  RgsaddleStencilConfig *cfg = rgsaddle_stencil_config_new(RGSADDLE_PRESET_DIMER);
  expect(cfg != NULL, "dimer config");
  double x[3] = {0.25, 0.30, 0.0};
  RgsaddleStencil *handle = rgsaddle_stencil_new(cfg, 1, 3, x);
  expect(handle != NULL, "dimer");
  rgsaddle_stencil_report_t report = blank_report();
  expect(rgsaddle_stencil_step(handle, eval_mb, NULL, &report) == RGSADDLE_OK, "dimer step");
  expect(report.phase == RGSADDLE_STENCIL_PHASE_MINMODE, "minmode phase");
  double mode[3];
  expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_MODE, mode, 3) == RGSADDLE_OK,
         "mode");
  double curvature = 0.0;
  expect(rgsaddle_stencil_query(handle, RGSADDLE_STENCIL_QUERY_CURVATURE, &curvature, 1) ==
             RGSADDLE_OK,
         "curvature");
  rgsaddle_stencil_free(handle);
  rgsaddle_stencil_config_free(cfg);
}

static void refuse_bad_keys(void) {
  RgsaddleStencilConfig *point = rgsaddle_stencil_config_new(RGSADDLE_PRESET_DIMER);
  expect(rgsaddle_stencil_config_set_f64(point, RGSADDLE_STENCIL_KEY_K, 1.0) != RGSADDLE_OK,
         "band key on a point preset");
  expect(strstr(rgsaddle_last_error(), "point stencil") != NULL, "point message");
  expect(rgsaddle_stencil_config_set_i64(point, 999, 1) != RGSADDLE_OK, "unknown key");
  rgsaddle_stencil_config_free(point);

  RgsaddleStencilConfig *band = rgsaddle_stencil_config_new(RGSADDLE_PRESET_CI_NEB);
  expect(rgsaddle_stencil_config_set_f64(band, RGSADDLE_STENCIL_KEY_METHOD, 1.0) != RGSADDLE_OK,
         "wrong type");
  expect(strstr(rgsaddle_last_error(), "integer") != NULL, "type message");
  rgsaddle_stencil_config_free(band);
}

int main(void) {
  run_band(RGSADDLE_PRESET_CI_NEB, 0);
  run_band(RGSADDLE_PRESET_CI_NEB, 1);
  run_zoom();
  run_dimer();
  refuse_bad_keys();
  if (g_fail) {
    return 1;
  }
  printf("RGSADDLE_STENCIL_C_OK\n");
  return 0;
}
