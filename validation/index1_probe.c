/* Exercise the public index-one ABI for trace and timing comparisons. */
#define _POSIX_C_SOURCE 200809L
#include "rgsaddle.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void require(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

struct surface {
  int64_t n;
  int64_t calls;
  const double *h;
};

static int evaluate(void *user, rgsaddle_surface_request_t *req) {
  struct surface *surface = user;
  int64_t n = surface->n;
  if (req->n_atoms * 3 != n) return -1;
  for (int64_t image = 0; image < req->n_images; ++image) {
    const double *x = req->positions + image * n;
    double *g = req->gradients + image * n;
    double energy = 0;
    for (int64_t i = 0; i < n; ++i) {
      double hx = 0;
      for (int64_t j = 0; j < n; ++j) hx += surface->h[i * n + j] * x[j];
      g[i] = hx + 0.08 * x[i] * x[i] * x[i];
      energy += 0.5 * x[i] * hx + 0.02 * x[i] * x[i] * x[i] * x[i];
    }
    req->energies[image] = energy;
    ++surface->calls;
  }
  return 0;
}

static void emit(const void *data, size_t count, size_t width) {
  require(fwrite(data, width, count, stdout) == count, "trace write failed");
}

static double seconds(void) {
  struct timespec t;
  require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "clock failed");
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void run_case(int64_t n, int mode, int update, int weighted,
                     int finite_difference, double radius, int trace,
                     int repeats) {
  double *x = calloc((size_t)n, sizeof(double));
  double *h = calloc((size_t)(n * n), sizeof(double));
  double *initial = calloc((size_t)(n * n), sizeof(double));
  double *mass = calloc((size_t)n, sizeof(double));
  double *position = calloc((size_t)n, sizeof(double));
  double *hessian = calloc((size_t)(n * n), sizeof(double));
  require(x && h && initial && mass && position && hessian, "allocation failed");
  for (int64_t i = 0; i < n; ++i) {
    x[i] = 0.1 + 0.06 * sin((double)(i + 1));
    mass[i] = weighted ? 1.0 + (double)(i % 5) : 1.0;
    for (int64_t j = 0; j < n; ++j) {
      h[i * n + j] = (i == j ? (i == 0 && mode ? -2.0 : 2.0 + (double)i / (double)n) : 0.0)
                    + 0.02 * cos((double)(i + j)) / (1.0 + (double)llabs(i - j));
      initial[i * n + j] = h[i * n + j];
    }
    initial[i * n + i] += 0.24 * x[i] * x[i];
  }
  rgsaddle_index1_config_t config = {0};
  require(rgsaddle_abi_stamp(&config.version) == 0, "ABI version failed");
  config.update = update;
  config.mode = mode;
  config.trust_radius = radius;
  config.force_tol = 0;
  config.fd_dr = 1e-4;
  struct surface surface = {n, 0, h};
  double checksum = 0;
  double start = seconds();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    RgsaddleIndex1 *session = rgsaddle_index1_create(
        &config, n / 3, x, finite_difference ? NULL : initial,
        weighted ? mass : NULL);
    require(session != NULL, "index-one create failed");
    for (int step = 0; step < (trace ? 6 : 3); ++step) {
      if (trace && step == 3) require(rgsaddle_index1_reset(session) == 0, "reset failed");
      rgsaddle_report_t report = {0};
      require(rgsaddle_index1_step(session, evaluate, &surface, &report) == 0, "step failed");
      require(rgsaddle_index1_position(session, position) == 0, "position failed");
      require(rgsaddle_index1_hessian(session, hessian) == 0, "Hessian failed");
      checksum += position[0] + report.curvature;
      if (trace) {
        int64_t meta[] = {n, mode, update, weighted, finite_difference, step,
                          report.status, report.iteration, surface.calls};
        double values[] = {radius, report.max_force, report.curvature};
        emit(meta, sizeof(meta) / sizeof(meta[0]), sizeof(meta[0]));
        emit(values, sizeof(values) / sizeof(values[0]), sizeof(values[0]));
        emit(position, (size_t)n, sizeof(double));
        emit(hessian, (size_t)(n * n), sizeof(double));
      }
    }
    rgsaddle_index1_free(session);
  }
  double elapsed = seconds() - start;
  if (!trace) printf("%lld,%d,%lld,%.17g,%.17g\n", (long long)n, repeats,
                     (long long)surface.calls, elapsed, checksum);
  free(x); free(h); free(initial); free(mass); free(position); free(hessian);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "trace") == 0) {
    const int64_t sizes[] = {3, 6, 15, 75};
    const double radii[] = {0.01, 0.5};
    for (size_t n = 0; n < sizeof(sizes) / sizeof(sizes[0]); ++n)
      for (int mode = 0; mode < 2; ++mode)
        for (int update = 0; update < 2; ++update)
          for (int weighted = 0; weighted < 2; ++weighted)
            for (int fd = 0; fd < 2; ++fd)
              for (size_t r = 0; r < sizeof(radii) / sizeof(radii[0]); ++r)
                run_case(sizes[n], mode, update, weighted, fd, radii[r], 1, 1);
    return 0;
  }
  if (argc == 4 && strcmp(argv[1], "bench") == 0) {
    int64_t n = strtoll(argv[2], NULL, 10);
    int repeats = atoi(argv[3]);
    require(n >= 3 && n <= 1200 && n % 3 == 0 && repeats > 0, "invalid benchmark dimensions");
    run_case(n, 1, 1, 1, 0, 0.01, 0, repeats);
    return 0;
  }
  fprintf(stderr, "usage: %s trace | bench DIMENSION REPEATS\n", argv[0]);
  return 2;
}
