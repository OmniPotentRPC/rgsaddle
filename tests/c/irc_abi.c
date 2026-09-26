/* Drives the IrcSession C ABI on the analytic well: create, step
 * forward and reverse, create_from_surface (Morokuma). Built by
 * tests/c_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR) {
    return -1;
  }
  const long dof = 3 * (long)req->n_atoms;
  for (long i = 0; i < (long)req->n_images; ++i) {
    const double *p = req->positions + i * dof;
    double *g = req->gradients + i * dof;
    memset(g, 0, (size_t)dof * sizeof(double));
    double x = p[0];
    req->energies[i] = (x * x - 1.0) * (x * x - 1.0);
    g[0] = 4.0 * x * (x * x - 1.0);
  }
  return 0;
}

int main(void) {
  const long n_atoms = 1;
  double saddle[3] = {0.0, 0.0, 0.0};
  double masses[1] = {1.0};
  double mode[3] = {1.0, 0.0, 0.0};
  rgsaddle_irc_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.dx = 0.2;
  cfg.force_tol = 1e-3;
  cfg.force_gate = RGSADDLE_FORCE_LINF;
  cfg.max_move = 0.2;
  cfg.max_inner = 10;
  cfg.kind = RGSADDLE_IRC_GS2;
  cfg.direction = RGSADDLE_IRC_FORWARD;

  rgsaddle_irc_config_t bad = cfg;
  bad.version.major = 99;
  if (rgsaddle_irc_create(&bad, n_atoms, saddle, masses, mode) != NULL) {
    fprintf(stderr, "unknown major must be refused\n");
    return 1;
  }

  RgsaddleIrc *irc = rgsaddle_irc_create(&cfg, n_atoms, saddle, masses, mode);
  if (!irc) {
    fprintf(stderr, "irc create failed\n");
    return 1;
  }
  if (rgsaddle_irc_step(irc, NULL, NULL, NULL) != RGSADDLE_NULL_REPORT) {
    fprintf(stderr, "NULL report must fail closed\n");
    return 1;
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  if (rgsaddle_irc_step(irc, surface, NULL, &rep) != RGSADDLE_OK) {
    fprintf(stderr, "irc step failed\n");
    return 1;
  }
  double xf[3];
  if (rgsaddle_irc_position(irc, xf) != RGSADDLE_OK || fabs(xf[0]) < 1e-8) {
    fprintf(stderr, "irc kick stayed at the saddle\n");
    return 1;
  }
  if (rgsaddle_irc_set_direction(irc, RGSADDLE_IRC_REVERSE) != RGSADDLE_OK) {
    fprintf(stderr, "set_direction failed\n");
    return 1;
  }
  if (rgsaddle_irc_step(irc, surface, NULL, &rep) != RGSADDLE_OK) {
    fprintf(stderr, "reverse step failed\n");
    return 1;
  }
  double xr[3];
  if (rgsaddle_irc_position(irc, xr) != RGSADDLE_OK || xf[0] * xr[0] >= 0.0) {
    fprintf(stderr, "F/R must leave opposite ways: %g %g\n", xf[0], xr[0]);
    return 1;
  }

  cfg.kind = RGSADDLE_IRC_MOROKUMA;
  cfg.direction = RGSADDLE_IRC_FORWARD;
  RgsaddleIrc *mor = rgsaddle_irc_create_from_surface(
      &cfg, n_atoms, saddle, masses, mode, surface, NULL);
  if (!mor) {
    fprintf(stderr, "create_from_surface failed\n");
    return 1;
  }
  if (rgsaddle_irc_step(mor, surface, NULL, &rep) != RGSADDLE_OK) {
    fprintf(stderr, "morokuma step failed\n");
    return 1;
  }
  double xm[3];
  if (rgsaddle_irc_position(mor, xm) != RGSADDLE_OK || fabs(xm[0]) < 1e-8) {
    fprintf(stderr, "morokuma kick stayed at the saddle\n");
    return 1;
  }
  if (rgsaddle_irc_reset(mor) != RGSADDLE_OK) {
    fprintf(stderr, "reset failed\n");
    return 1;
  }
  rgsaddle_irc_free(mor);
  rgsaddle_irc_free(irc);
  rgsaddle_irc_free(NULL);
  printf("RGSADDLE_IRC_ABI_OK xf=%g xr=%g xm=%g\n", xf[0], xr[0], xm[0]);
  return 0;
}
