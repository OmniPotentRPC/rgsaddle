/* rgsaddle_band_set_dynamic and rgsaddle_dyneb_mask: every refusal and
 * its status, the mask on a hand-built band, and a dynamic band run to
 * convergence on a double well against the plain band. Built and run
 * by tests/dyneb_abi.rs against the cdylib. */
#include "rgsaddle.h"
#include "main_abi_contract.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_IMAGES 9
#define N_ATOMS 1
#define DOF (3 * N_ATOMS)

static int failures = 0;

static void expect(const char *what, rgsaddle_status_t got, rgsaddle_status_t want) {
  if (got != want) {
    fprintf(stderr, "%s: got %s, want %s\n", what, rgsaddle_status_name(got),
            rgsaddle_status_name(want));
    ++failures;
  }
}

static void check(const char *what, int ok) {
  if (!ok) {
    fprintf(stderr, "%s\n", what);
    ++failures;
  }
}

/* V = (x^2 - 1)^2 + 2 y^2 + 2 z^2, minima at (+-1, 0, 0). */
static rgsaddle_status_t well(void *user, rgsaddle_surface_request_t *req) {
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  *(int64_t *)user += rows;
  for (int64_t r = 0; r < rows; ++r) {
    const double *p = req->positions + r * DOF;
    double *g = req->gradients + r * DOF;
    const double x = p[0], y = p[1], z = p[2];
    req->energies[r] = (x * x - 1.0) * (x * x - 1.0) + 2.0 * y * y + 2.0 * z * z;
    g[0] = 4.0 * x * (x * x - 1.0);
    g[1] = 4.0 * y;
    g[2] = 4.0 * z;
  }
  return RGSADDLE_OK;
}

static void bowed(double *pos) {
  for (int i = 0; i < N_IMAGES; ++i) {
    const double t = (double)i / (double)(N_IMAGES - 1);
    pos[i * DOF + 0] = -1.0 + 2.0 * t;
    pos[i * DOF + 1] = 0.4 * sin(3.14159265358979 * t);
    pos[i * DOF + 2] = 0.0;
  }
}

static rgsaddle_band_config_t config(void) {
  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.flags = RGSADDLE_BAND_NO_MIN_MODE;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.ci_trigger_factor = 0.0;
  cfg.force_tol = 0.05;
  cfg.max_move = 0.1;
  return cfg;
}

/* Steps to convergence; returns the surface rows, or -1. */
static int64_t relax(RgsaddleBand *band) {
  int64_t rows = 0;
  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  for (int step = 0; step < 2000; ++step) {
    const rgsaddle_status_t st = rgsaddle_band_step(band, well, &rows, &rep);
    if (st != RGSADDLE_OK) {
      fprintf(stderr, "step %d: %s\n", step, rgsaddle_status_name(st));
      return -1;
    }
    if (rep.status == RGSADDLE_STATUS_CONVERGED) {
      return rows;
    }
  }
  fprintf(stderr, "no convergence, max force %g\n", rep.max_force);
  return -1;
}

static void set_dynamic_refusals(const double *pos) {
  const rgsaddle_band_config_t cfg = config();
  expect("set_dynamic null band", rgsaddle_band_set_dynamic(NULL, 1, 0.05, 0.0),
         RGSADDLE_NULL_BAND);
  expect("set_dynamic null band, enabled 0", rgsaddle_band_set_dynamic(NULL, 0, 0.05, 0.0),
         RGSADDLE_NULL_BAND);

  RgsaddleBand *band = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  if (!band) {
    check("create", 0);
    return;
  }
  expect("enabled 2", rgsaddle_band_set_dynamic(band, 2, 0.05, 0.0),
         RGSADDLE_INVALID_PARAMETER);
  expect("enabled -1", rgsaddle_band_set_dynamic(band, -1, 0.05, 0.0),
         RGSADDLE_INVALID_PARAMETER);
  const double bad_fmax[] = {NAN, INFINITY, -INFINITY, 0.0, -0.05};
  for (size_t i = 0; i < sizeof bad_fmax / sizeof bad_fmax[0]; ++i) {
    expect("bad fmax", rgsaddle_band_set_dynamic(band, 1, bad_fmax[i], 0.0),
           RGSADDLE_INVALID_PARAMETER);
  }
  const double bad_scale[] = {NAN, INFINITY, -INFINITY, -1.0};
  for (size_t i = 0; i < sizeof bad_scale / sizeof bad_scale[0]; ++i) {
    expect("bad scale_fmax", rgsaddle_band_set_dynamic(band, 1, 0.05, bad_scale[i]),
           RGSADDLE_INVALID_PARAMETER);
  }
  expect("enabled 0 ignores the thresholds", rgsaddle_band_set_dynamic(band, 0, NAN, -1.0),
         RGSADDLE_OK);
  expect("dynamic then string", rgsaddle_band_set_dynamic(band, 1, 0.05, 0.0), RGSADDLE_OK);
  expect("string on a dynamic band", rgsaddle_band_set_string(band, 1, RGSADDLE_REPARAM_LINEAR),
         RGSADDLE_INVALID_PARAMETER);
  expect("free end on a dynamic band",
         rgsaddle_band_set_free_end(band, 1, RGSADDLE_FREE_END_LAST, 0, 0.0, 1, 0.1, 1e-3, 2, 1, 0),
         RGSADDLE_INVALID_PARAMETER);
  expect("clear", rgsaddle_band_set_dynamic(band, 0, 0.0, 0.0), RGSADDLE_OK);
  expect("string", rgsaddle_band_set_string(band, 1, RGSADDLE_REPARAM_LINEAR), RGSADDLE_OK);
  expect("dynamic on a string band", rgsaddle_band_set_dynamic(band, 1, 0.05, 0.0),
         RGSADDLE_INVALID_PARAMETER);
  expect("string off", rgsaddle_band_set_string(band, 0, RGSADDLE_REPARAM_LINEAR), RGSADDLE_OK);
  expect("free end",
         rgsaddle_band_set_free_end(band, 1, RGSADDLE_FREE_END_LAST, 0, 0.0, 1, 0.1, 1e-3, 2, 1, 0),
         RGSADDLE_OK);
  expect("dynamic on a free-end band", rgsaddle_band_set_dynamic(band, 1, 0.05, 0.0),
         RGSADDLE_INVALID_PARAMETER);
  rgsaddle_band_free(band);

  rgsaddle_band_config_t rc = cfg;
  rc.method = RGSADDLE_METHOD_RTR;
  RgsaddleBand *rtr = rgsaddle_band_create(&rc, N_IMAGES, N_ATOMS, pos);
  check("create the trust-region band", rtr != NULL);
  if (rtr) {
    expect("dynamic on the trust-region stepper", rgsaddle_band_set_dynamic(rtr, 1, 0.05, 0.0),
           RGSADDLE_INVALID_PARAMETER);
    rgsaddle_band_free(rtr);
  }

  double cells[N_IMAGES * 9];
  memset(cells, 0, sizeof cells);
  for (int i = 0; i < N_IMAGES; ++i) {
    cells[9 * i + 0] = 6.0;
    cells[9 * i + 4] = 6.0;
    cells[9 * i + 8] = 6.0;
  }
  RgsaddleBand *solid = rgsaddle_band_create_solid(&cfg, N_IMAGES, N_ATOMS, pos, cells, 0.0, 1.0);
  check("create the solid-state band", solid != NULL);
  if (solid) {
    expect("dynamic on a solid-state band", rgsaddle_band_set_dynamic(solid, 1, 0.05, 0.0),
           RGSADDLE_INVALID_PARAMETER);
    rgsaddle_band_free(solid);
  }
}

static void set_dynamic_runs(const double *pos) {
  const rgsaddle_band_config_t cfg = config();
  RgsaddleBand *plain = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  RgsaddleBand *dynamic = rgsaddle_band_create(&cfg, N_IMAGES, N_ATOMS, pos);
  if (!plain || !dynamic) {
    check("create", 0);
    return;
  }
  expect("set_dynamic", rgsaddle_band_set_dynamic(dynamic, 1, 0.05, 0.0), RGSADDLE_OK);
  const int64_t plain_rows = relax(plain);
  const int64_t dynamic_rows = relax(dynamic);
  check("both bands converge", plain_rows > 0 && dynamic_rows > 0);
  if (!(dynamic_rows < plain_rows)) {
    fprintf(stderr, "dynamic rows %lld, plain rows %lld\n", (long long)dynamic_rows,
            (long long)plain_rows);
    ++failures;
  }
  double a[N_IMAGES * DOF], b[N_IMAGES * DOF];
  if (rgsaddle_band_positions(plain, a) == RGSADDLE_OK &&
      rgsaddle_band_positions(dynamic, b) == RGSADDLE_OK) {
    double gap = 0.0;
    for (int k = 0; k < N_IMAGES * DOF; ++k) {
      gap = fmax(gap, fabs(a[k] - b[k]));
    }
    check("the dynamic path stays within the band tolerance", gap <= 0.05);
  } else {
    check("positions", 0);
  }
  expect("clear after the run", rgsaddle_band_set_dynamic(dynamic, 0, 0.05, 0.0), RGSADDLE_OK);
  rgsaddle_band_free(plain);
  rgsaddle_band_free(dynamic);
}

#define M_IMAGES 5
#define M_DOF 3

static void mask_input(double *pos, double *energies, double *forces) {
  const double e[M_IMAGES] = {0.0, 1.0, 3.0, 1.0, 0.0};
  const double f[M_IMAGES] = {9.0, 0.01, 0.01, 0.5, 9.0};
  for (int i = 0; i < M_IMAGES; ++i) {
    pos[i * M_DOF + 0] = (double)i;
    pos[i * M_DOF + 1] = 0.0;
    pos[i * M_DOF + 2] = 0.0;
    energies[i] = e[i];
    forces[i * M_DOF + 0] = f[i];
    forces[i * M_DOF + 1] = 0.0;
    forces[i * M_DOF + 2] = 0.0;
  }
}

static void dyneb_mask_refusals(void) {
  double pos[M_IMAGES * M_DOF], energies[M_IMAGES], forces[M_IMAGES * M_DOF], kept[M_IMAGES * M_DOF];
  uint8_t frozen[M_IMAGES];
  mask_input(pos, energies, forces);
  memcpy(kept, forces, sizeof kept);
  memset(frozen, 7, sizeof frozen);

  expect("null positions", rgsaddle_dyneb_mask(M_IMAGES, 1, NULL, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("null energies", rgsaddle_dyneb_mask(M_IMAGES, 1, pos, NULL, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("null forces", rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, NULL, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("two images", rgsaddle_dyneb_mask(2, 1, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("negative images", rgsaddle_dyneb_mask(-5, 1, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("zero atoms", rgsaddle_dyneb_mask(M_IMAGES, 0, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  expect("negative atoms", rgsaddle_dyneb_mask(M_IMAGES, -1, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  /* 3 * n_atoms overflows. */
  expect("3 n_atoms overflows",
         rgsaddle_dyneb_mask(M_IMAGES, INT64_MAX, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  /* n_images * 3 * n_atoms overflows with each factor representable. */
  expect("band length overflows",
         rgsaddle_dyneb_mask(INT64_MAX, 1, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  /* The length fits, but its bytes exceed PTRDIFF_MAX. */
  expect("band bytes exceed PTRDIFF_MAX",
         rgsaddle_dyneb_mask(3, INT64_MAX / 64, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);

  const double bad_fmax[] = {NAN, INFINITY, -INFINITY, 0.0, -0.05};
  for (size_t i = 0; i < sizeof bad_fmax / sizeof bad_fmax[0]; ++i) {
    expect("mask bad fmax",
           rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, forces, bad_fmax[i], 0.0, frozen),
           RGSADDLE_INVALID_PARAMETER);
  }
  const double bad_scale[] = {NAN, INFINITY, -INFINITY, -1.0};
  for (size_t i = 0; i < sizeof bad_scale / sizeof bad_scale[0]; ++i) {
    expect("mask bad scale_fmax",
           rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, forces, 0.05, bad_scale[i], frozen),
           RGSADDLE_INVALID_PARAMETER);
  }

  /* Non-finite data is RGSADDLE_NON_FINITE; a bad threshold outranks it. */
  double nan_pos[M_IMAGES * M_DOF];
  memcpy(nan_pos, pos, sizeof nan_pos);
  nan_pos[2 * M_DOF + 1] = NAN;
  expect("non-finite position",
         rgsaddle_dyneb_mask(M_IMAGES, 1, nan_pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_NON_FINITE);
  expect("bad fmax before non-finite data",
         rgsaddle_dyneb_mask(M_IMAGES, 1, nan_pos, energies, forces, NAN, 0.0, frozen),
         RGSADDLE_INVALID_PARAMETER);
  double inf_energy[M_IMAGES];
  memcpy(inf_energy, energies, sizeof inf_energy);
  inf_energy[M_IMAGES - 1] = INFINITY;
  expect("non-finite energy",
         rgsaddle_dyneb_mask(M_IMAGES, 1, pos, inf_energy, forces, 0.05, 0.0, frozen),
         RGSADDLE_NON_FINITE);
  double nan_forces[M_IMAGES * M_DOF];
  memcpy(nan_forces, forces, sizeof nan_forces);
  nan_forces[0] = NAN;
  expect("non-finite endpoint force",
         rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, nan_forces, 0.05, 0.0, frozen),
         RGSADDLE_NON_FINITE);

  check("a refused mask wrote the forces", memcmp(forces, kept, sizeof kept) == 0);
  for (int i = 0; i < M_IMAGES; ++i) {
    check("a refused mask wrote the frozen flags", frozen[i] == 7);
  }
}

static void dyneb_mask_values(void) {
  double pos[M_IMAGES * M_DOF], energies[M_IMAGES], forces[M_IMAGES * M_DOF];
  uint8_t frozen[M_IMAGES];
  mask_input(pos, energies, forces);
  memset(frozen, 7, sizeof frozen);
  expect("mask", rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, forces, 0.05, 0.0, frozen),
         RGSADDLE_OK);
  /* Image 1 is under fmax and zeroed; image 2 is highest and kept; image
   * 3 is over fmax; the endpoint rows are left as they were. */
  check("endpoint forces untouched", forces[0] == 9.0 && forces[4 * M_DOF] == 9.0);
  check("image 1 zeroed", forces[1 * M_DOF] == 0.0);
  check("highest image keeps its force", forces[2 * M_DOF] == 0.01);
  check("image 3 kept", forces[3 * M_DOF] == 0.5);
  const uint8_t want[M_IMAGES] = {1, 1, 1, 0, 1};
  check("frozen flags", memcmp(frozen, want, sizeof want) == 0);

  /* scale_fmax 20 lifts image 3's threshold to 0.05 * (1 + 1 * 20). */
  mask_input(pos, energies, forces);
  expect("scaled mask", rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, forces, 0.05, 20.0, frozen),
         RGSADDLE_OK);
  check("scaled image 3 zeroed", forces[3 * M_DOF] == 0.0);
  check("scaled highest image kept", forces[2 * M_DOF] == 0.01);
  check("scaled image 3 frozen", frozen[3] == 1);

  mask_input(pos, energies, forces);
  expect("null frozen", rgsaddle_dyneb_mask(M_IMAGES, 1, pos, energies, forces, 0.05, 0.0, NULL),
         RGSADDLE_OK);
  check("null frozen still masks", forces[1 * M_DOF] == 0.0);
}

int main(void) {
  double pos[N_IMAGES * DOF];
  bowed(pos);
  set_dynamic_refusals(pos);
  set_dynamic_runs(pos);
  dyneb_mask_refusals();
  dyneb_mask_values();
  if (failures != 0) {
    fprintf(stderr, "%d dynamic NEB C ABI checks failed\n", failures);
    return 1;
  }
  printf("RGSADDLE_DYNEB_ABI_OK\n");
  return 0;
}
