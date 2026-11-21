/* C ABI contract of rgsaddle_pafi_transmission: version stamp, dividing plane. */
#include "rgsaddle.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(int code, const char *msg) {
  fprintf(stderr, "%s (status=%d %s)\n", msg, code,
          rgsaddle_status_name((rgsaddle_status_t)code));
  return 1;
}

/* E = (q^2 - 1)^2 on one coordinate. */
static int quartic(void *user, int64_t n_dof, const double *x, double *energy,
                   double *gradient) {
  (void)user;
  if (n_dof != 1) {
    return RGSADDLE_SHAPE;
  }
  const double q = x[0];
  *energy = (q * q - 1.0) * (q * q - 1.0);
  gradient[0] = 4.0 * q * (q * q - 1.0);
  return 0;
}

typedef struct {
  double free_energy[3];
  double points[3];
  double normals[3];
  int64_t offsets[4];
  double starts[4];
} band_t;

static band_t band(void) {
  band_t b = {{0.0, 1.0, 0.0}, {-1.0, 0.0, 1.0}, {1.0, 1.0, 1.0},
              {0, 1, 3, 4},    {-1.0, 0.0, 0.0, 1.0}};
  return b;
}

static rgsaddle_status_t shoot(const rgsaddle_pafi_transmission_config_t *cfg,
                               const band_t *b,
                               rgsaddle_pafi_transmission_report_t *report) {
  return rgsaddle_pafi_transmission(cfg, 1, 3, b->free_energy, b->points,
                                    b->normals, b->offsets, b->starts, NULL,
                                    quartic, NULL, report);
}

int main(void) {
  rgsaddle_pafi_transmission_config_t cfg;
  rgsaddle_status_t st = rgsaddle_pafi_transmission_config_default(&cfg);
  if (st != RGSADDLE_OK) {
    return fail(st, "config default");
  }
  if (cfg.version.major != RGSADDLE_ABI_MAJOR ||
      cfg.version.minor != RGSADDLE_ABI_MINOR) {
    return fail(RGSADDLE_ABI_MISMATCH, "default config is not stamped");
  }
  cfg.thermal = 0.05;
  cfg.friction = 0.0;
  cfg.time_step = 0.002;
  cfg.steps = 100;
  cfg.check_every = 25;
  cfg.shots = 32;
  cfg.blocks = 4;
  cfg.seed = 7;

  rgsaddle_pafi_transmission_report_t report;
  memset(&report, 0, sizeof report);
  band_t b = band();
  st = shoot(&cfg, &b, &report);
  if (st != RGSADDLE_OK) {
    return fail(st, "interior maximum must run");
  }
  if (report.dividing != 1 || report.shots != 32) {
    return fail(RGSADDLE_SHAPE, "interior maximum selects the middle plane");
  }

  rgsaddle_pafi_transmission_config_t bad = cfg;
  bad.version.minor = RGSADDLE_ABI_MINOR + 1u;
  st = shoot(&bad, &b, &report);
  if (st != RGSADDLE_ABI_MISMATCH) {
    return fail(st, "a minor newer than the library must be refused");
  }
  bad = cfg;
  bad.version.minor = 51u;
  st = shoot(&bad, &b, &report);
  if (st != RGSADDLE_ABI_MISMATCH) {
    return fail(st, "a minor older than the structs must be refused");
  }
  bad = cfg;
  bad.version.major = 99u;
  st = shoot(&bad, &b, &report);
  if (st != RGSADDLE_ABI_MISMATCH) {
    return fail(st, "an unknown major must be refused");
  }
  bad.steps = 0;
  st = shoot(&bad, &b, &report);
  if (st != RGSADDLE_ABI_MISMATCH) {
    return fail(st, "the version is read before the other fields");
  }
  bad = cfg;
  bad.steps = 0;
  st = shoot(&bad, &b, &report);
  if (st != RGSADDLE_INVALID_PARAMETER) {
    return fail(st, "a zero step count is an invalid parameter");
  }

  band_t last = band();
  last.free_energy[0] = 0.0;
  last.free_energy[1] = 0.5;
  last.free_energy[2] = 1.0;
  st = shoot(&cfg, &last, &report);
  if (st != RGSADDLE_INVALID_PARAMETER) {
    return fail(st, "a maximum at the last plane must be refused");
  }
  band_t first = band();
  first.free_energy[0] = 1.0;
  first.free_energy[1] = 0.5;
  first.free_energy[2] = 0.0;
  st = shoot(&cfg, &first, &report);
  if (st != RGSADDLE_INVALID_PARAMETER) {
    return fail(st, "a maximum at the first plane must be refused");
  }

  printf("RGSADDLE_PAFI_TRANSMISSION_ABI_OK\n");
  return 0;
}
