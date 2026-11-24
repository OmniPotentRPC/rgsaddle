/* Dynamic NEB at ABI minor 65: setter refusals and the DyNEB mask. */
#include "rgsaddle.h"
#include "main_abi_contract.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static rgsaddle_status_t fail(rgsaddle_status_t st, const char *msg) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(st));
  return st == RGSADDLE_OK ? RGSADDLE_SOLVER : st;
}

static void bow(double *pos, long n_images) {
  for (long i = 0; i < n_images; ++i) {
    double t = (double)i / (double)(n_images - 1);
    pos[i * 3 + 0] = -1.0 + 2.0 * t;
    pos[i * 3 + 1] = 0.4 * sin(3.141592653589793 * t);
    pos[i * 3 + 2] = 0.0;
  }
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

static int same_positions(const double *a, const double *b, long n) {
  for (long i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

/* Whole band: two endpoints around the three interior images of the
 * Rust mask test. The first interior image is cleared; the highest
 * energy image keeps its force. */
static rgsaddle_status_t check_mask(void) {
  const int64_t n_images = 5;
  const int64_t n_atoms = 1;
  double positions[5 * 3] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0,
  };
  double energies[5] = {0.0, 1.0, 5.0, 4.0, 0.0};
  double forces[5 * 3] = {
      0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
  };
  double saved[5 * 3];
  memcpy(saved, forces, sizeof forces);
  uint8_t frozen[5] = {9, 9, 9, 9, 9};

  if (rgsaddle_dyneb_mask(n_images, n_atoms, NULL, energies, forces, 1.0, 1.0, frozen) !=
          RGSADDLE_SHAPE ||
      rgsaddle_dyneb_mask(n_images, n_atoms, positions, NULL, forces, 1.0, 1.0, frozen) !=
          RGSADDLE_SHAPE ||
      rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, NULL, 1.0, 1.0, frozen) !=
          RGSADDLE_SHAPE ||
      rgsaddle_dyneb_mask(2, n_atoms, positions, energies, forces, 1.0, 1.0, frozen) !=
          RGSADDLE_SHAPE ||
      rgsaddle_dyneb_mask(n_images, 0, positions, energies, forces, 1.0, 1.0, frozen) !=
          RGSADDLE_SHAPE) {
    return fail(RGSADDLE_SHAPE, "mask shape");
  }
  if (!same_positions(forces, saved, 15)) {
    return fail(RGSADDLE_SOLVER, "a refused mask wrote forces");
  }

  forces[3] = nan("");
  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, 1.0, frozen) !=
      RGSADDLE_NON_FINITE) {
    return fail(RGSADDLE_NON_FINITE, "non-finite force");
  }
  forces[3] = 0.5;
  positions[0] = INFINITY;
  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, 1.0, frozen) !=
      RGSADDLE_NON_FINITE) {
    return fail(RGSADDLE_NON_FINITE, "non-finite position");
  }
  positions[0] = 0.0;
  energies[2] = nan("");
  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, 1.0, frozen) !=
      RGSADDLE_NON_FINITE) {
    return fail(RGSADDLE_NON_FINITE, "non-finite energy");
  }
  energies[2] = 5.0;
  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, nan(""), 0.0, frozen) !=
          RGSADDLE_NON_FINITE ||
      rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, INFINITY, frozen) !=
          RGSADDLE_NON_FINITE ||
      rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, -INFINITY, 0.0, NULL) !=
          RGSADDLE_NON_FINITE) {
    return fail(RGSADDLE_NON_FINITE, "non-finite threshold");
  }
  if (!same_positions(forces, saved, 15)) {
    return fail(RGSADDLE_SOLVER, "a non-finite threshold wrote forces");
  }

  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, 1.0, NULL) !=
      RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "mask without frozen");
  }
  memcpy(forces, saved, sizeof forces);
  if (rgsaddle_dyneb_mask(n_images, n_atoms, positions, energies, forces, 1.0, 1.0, frozen) !=
      RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "mask");
  }
  for (int c = 0; c < 3; ++c) {
    if (forces[3 + c] != 0.0) {
      return fail(RGSADDLE_SOLVER, "the low image kept its force");
    }
  }
  if (forces[7] != 2.0 || forces[9] != 3.0 || forces[10] != 4.0) {
    return fail(RGSADDLE_SOLVER, "the mask cleared an image it should keep");
  }
  if (frozen[0] != 1 || frozen[1] != 1 || frozen[2] != 0 || frozen[3] != 0 || frozen[4] != 1) {
    fprintf(stderr, "frozen %u %u %u %u %u\n", frozen[0], frozen[1], frozen[2], frozen[3],
            frozen[4]);
    return fail(RGSADDLE_SOLVER, "frozen flags");
  }
  return RGSADDLE_OK;
}

int main(void) {
  if (rgsaddle_abi_version() != (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    return fail(RGSADDLE_ABI_MISMATCH, "ABI minor disagrees with the header");
  }
  if (rgsaddle_band_set_dynamic(NULL, 1, 0.05, 0.0) != RGSADDLE_NULL_BAND) {
    return fail(RGSADDLE_NULL_BAND, "null band");
  }

  const long n_images = 7;
  double pos[7 * 3];
  bow(pos, n_images);
  double kept[7 * 3];
  memcpy(kept, pos, sizeof pos);

  RgsaddleBand *band = open_band(pos, n_images, RGSADDLE_METHOD_FIRE, 0);
  if (!band) {
    return fail(RGSADDLE_SHAPE, "create");
  }
  if (rgsaddle_band_set_dynamic(band, 2, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER ||
      rgsaddle_band_set_dynamic(band, -1, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "enabled flag");
  }
  if (rgsaddle_band_set_dynamic(band, 1, nan(""), 0.0) != RGSADDLE_NON_FINITE ||
      rgsaddle_band_set_dynamic(band, 1, 0.05, nan("")) != RGSADDLE_NON_FINITE ||
      rgsaddle_band_set_dynamic(band, 1, INFINITY, 0.0) != RGSADDLE_NON_FINITE ||
      rgsaddle_band_set_dynamic(band, 1, 0.05, -INFINITY) != RGSADDLE_NON_FINITE) {
    return fail(RGSADDLE_NON_FINITE, "non-finite threshold");
  }
  /* Clearing ignores the thresholds, finite or not. */
  if (rgsaddle_band_set_dynamic(band, 0, nan(""), INFINITY) != RGSADDLE_OK) {
    return fail(RGSADDLE_NON_FINITE, "clearing reads the thresholds");
  }
  if (rgsaddle_band_positions(band, pos) != RGSADDLE_OK || !same_positions(pos, kept, 21)) {
    return fail(RGSADDLE_SOLVER, "a refused setter moved the band");
  }
  if (rgsaddle_band_set_dynamic(band, 1, 0.05, 0.0) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "enable dynamic NEB");
  }
  rgsaddle_report_t report;
  memset(&report, 0, sizeof report);
  if (rgsaddle_band_step(band, surface, NULL, &report) != RGSADDLE_OK) {
    return fail(RGSADDLE_SOLVER, "dynamic step");
  }
  if (rgsaddle_band_set_dynamic(band, 0, 0.0, 0.0) != RGSADDLE_OK) {
    return fail(RGSADDLE_INVALID_PARAMETER, "clear dynamic NEB");
  }
  rgsaddle_band_free(band);

  RgsaddleBand *trust = open_band(pos, n_images, RGSADDLE_METHOD_RTR, 0);
  if (!trust || rgsaddle_band_set_dynamic(trust, 1, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "trust region accepted dynamic NEB");
  }
  rgsaddle_band_free(trust);

  RgsaddleBand *string_band =
      open_band(kept, n_images, RGSADDLE_METHOD_FIRE, RGSADDLE_BAND_STRING);
  if (!string_band ||
      rgsaddle_band_set_dynamic(string_band, 1, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "string method accepted dynamic NEB");
  }
  rgsaddle_band_free(string_band);

  RgsaddleBand *free_end = open_band(kept, n_images, RGSADDLE_METHOD_FIRE, 0);
  if (!free_end ||
      rgsaddle_band_set_free_end(free_end, 1, RGSADDLE_FREE_END_LAST, 0, 0.0, 2, 0.0, 0.001, 2, 1,
                                 1) != RGSADDLE_OK ||
      rgsaddle_band_set_dynamic(free_end, 1, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "free-end band accepted dynamic NEB");
  }
  rgsaddle_band_free(free_end);

  rgsaddle_band_config_t solid_cfg;
  memset(&solid_cfg, 0, sizeof solid_cfg);
  solid_cfg.version.major = RGSADDLE_ABI_MAJOR;
  solid_cfg.version.minor = RGSADDLE_ABI_MINOR;
  solid_cfg.flags = RGSADDLE_BAND_NO_MIN_MODE;
  solid_cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  solid_cfg.spring = RGSADDLE_SPRING_UNIFORM;
  solid_cfg.projection = RGSADDLE_PROJECTION_NEB;
  solid_cfg.method = RGSADDLE_METHOD_FIRE;
  solid_cfg.spring_k = 5.0;
  solid_cfg.force_tol = 1e-3;
  solid_cfg.max_move = 0.2;
  double cells[3 * 9];
  double solid_pos[3 * 3];
  memset(cells, 0, sizeof cells);
  memset(solid_pos, 0, sizeof solid_pos);
  for (int i = 0; i < 3; ++i) {
    cells[9 * i + 0] = 4.0;
    cells[9 * i + 4] = 4.0;
    cells[9 * i + 8] = 4.0;
    solid_pos[3 * i] = 0.1 * i;
  }
  RgsaddleBand *solid =
      rgsaddle_band_create_solid(&solid_cfg, 3, 1, solid_pos, cells, 0.0, 1.0);
  if (!solid || rgsaddle_band_set_dynamic(solid, 1, 0.05, 0.0) != RGSADDLE_INVALID_PARAMETER) {
    return fail(RGSADDLE_INVALID_PARAMETER, "solid band accepted dynamic NEB");
  }
  rgsaddle_band_free(solid);

  rgsaddle_status_t mask = check_mask();
  if (mask != RGSADDLE_OK) {
    return mask;
  }
  printf("RGSADDLE_DYNEB_ABI_OK\n");
  return 0;
}
