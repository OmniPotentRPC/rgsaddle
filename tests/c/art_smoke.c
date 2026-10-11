/* ART nouveau through the opaque saddle-search configuration. */
#include "rgsaddle.h"

#include <stdio.h>

static rgsaddle_status_t surface(void *user, rgsaddle_surface_request_t *req) {
  (void)user;
  if (req->n_images != 1 || req->n_atoms != 1) {
    return RGSADDLE_SHAPE;
  }
  const double x = req->positions[0];
  const double y = req->positions[1];
  const double z = req->positions[2];
  req->energies[0] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
  req->gradients[0] = 4.0 * x * (x * x - 1.0);
  req->gradients[1] = 4.0 * y;
  req->gradients[2] = 4.0 * z;
  return RGSADDLE_OK;
}

int main(void) {
  RgsaddleSearchConfig *config = rgsaddle_search_config_create();
  if (config == NULL) {
    fprintf(stderr, "config\n");
    return 1;
  }
  if (rgsaddle_search_config_set_push_step(config, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    fprintf(stderr, "zero push accepted\n");
    return 1;
  }
  if (rgsaddle_search_config_set_eigen_threshold(config, -0.01) != RGSADDLE_OK ||
      rgsaddle_search_config_set_push_step(config, 0.3) != RGSADDLE_OK ||
      rgsaddle_search_config_set_activation_distance(config, 0.0) != RGSADDLE_OK ||
      rgsaddle_search_config_set_perp_steps_push(config, 0) != RGSADDLE_OK ||
      rgsaddle_search_config_set_displacement_amplitude(config, 0.2) != RGSADDLE_OK ||
      rgsaddle_search_config_set_max_move(config, 0.3) != RGSADDLE_OK) {
    fprintf(stderr, "setters\n");
    return 1;
  }
  const double minimum[3] = {1.0, 0.0, 0.0};
  const double direction[3] = {-1.0, 0.0, 0.0};
  RgsaddleMinMode *session =
      rgsaddle_minmode_create_search(config, 1, minimum, direction);
  if (session == NULL) {
    fprintf(stderr, "create\n");
    return 1;
  }
  rgsaddle_report_t report;
  rgsaddle_status_t rc = rgsaddle_minmode_step(session, surface, NULL, &report);
  if (rc != RGSADDLE_OK) {
    fprintf(stderr, "step %s\n", rgsaddle_status_name(rc));
    return 1;
  }
  double at[3];
  if (rgsaddle_minmode_position(session, at) != RGSADDLE_OK || !(at[0] < 1.0)) {
    fprintf(stderr, "position\n");
    return 1;
  }
  if (report.status != RGSADDLE_STATUS_RUNNING &&
      report.status != RGSADDLE_STATUS_CONVERGED) {
    fprintf(stderr, "status %d\n", report.status);
    return 1;
  }
  rgsaddle_minmode_free(session);
  rgsaddle_search_config_free(config);
  printf("RGSADDLE_C_ART_OK x=%.6g pushes=%lld\n", at[0],
         (long long)report.ci_index);
  return 0;
}
