/* Bain path of a Lennard-Jones crystal through the solid-state band.
 *
 * LJ with epsilon 1 eV and sigma 1 Angstrom, unshifted, times the C^2
 * quintic switch over [2.0, 2.2] Angstrom (rgpot LJConfig cutoff 2.2,
 * switch_width 0.2). The reactant is bcc, the product fcc, both as the
 * two-atom body-centred tetragonal cell diag(a, a, c) repeated
 * nx x ny x nz times. The band climbs to the tetragonal saddle.
 *
 * Every atom sits on an inversion centre, so the path is the cell
 * strain alone. The two-atom cell has no lattice wave vector but k = 0,
 * so its saddle is the scan's. That saddle is unstable to the
 * k = (1/2, 0, 1/2) shuffle: a supercell that admits it (5 x 5 x 4 does)
 * leaves the tetragonal path for a lower one once rounding seeds the
 * mode, and the check below fails there by design.
 *
 * Reference: validation/bain_lj_reference.py, a relaxed scan of the
 * energy per atom over c/a with spacing DQ = 2.5e-4:
 *   bcc     a = 1.228441371,                 E = -7.4148256256 eV/atom
 *   saddle  c/a = 1.09331985,                E = -7.3557899883 eV/atom
 *   fcc     a = 1.100691058, c = 1.556612222, E = -7.6698410342 eV/atom
 *   barrier 0.0590356373 eV/atom
 * The climbing image must reach the scan's c/a within DQ and its
 * barrier within the energy change over one grid step at the saddle,
 * 1.2e-6 eV/atom, plus the band's own force tolerance.
 *
 * Usage: solid_bain_lj [nx ny nz interior force_tol max_iter max_move
 * ci_force trace]. ci_force arms the climbing image once the band force
 * is under it: 0.5 by default, rgsaddle's CiConfig default; inf climbs
 * from the first evaluation. trace > 0 prints every trace-th step. With no arguments it runs the two-atom cell and checks
 * the reference; with arguments it prints the same report for a larger
 * supercell. */
#include "rgsaddle.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RC 2.2
#define WIDTH 0.2
#define A_BCC 1.228441371
#define A_FCC 1.100691058
#define C_FCC 1.556612222
#define E_BCC (-7.4148256256)
#define Q_SADDLE 1.09331985
#define BARRIER 0.0590356373
#define DQ 2.5e-4
#define DE_STEP 1.2e-6

static long long rows_evaluated = 0;

static void inverse3(const double *h, double *inv) {
  const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                     h[2] * (h[3] * h[7] - h[4] * h[6]);
  inv[0] = (h[4] * h[8] - h[5] * h[7]) / det;
  inv[1] = (h[2] * h[7] - h[1] * h[8]) / det;
  inv[2] = (h[1] * h[5] - h[2] * h[4]) / det;
  inv[3] = (h[5] * h[6] - h[3] * h[8]) / det;
  inv[4] = (h[0] * h[8] - h[2] * h[6]) / det;
  inv[5] = (h[2] * h[3] - h[0] * h[5]) / det;
  inv[6] = (h[3] * h[7] - h[4] * h[6]) / det;
  inv[7] = (h[1] * h[6] - h[0] * h[7]) / det;
  inv[8] = (h[0] * h[4] - h[1] * h[3]) / det;
}

static double volume3(const double *h) {
  return fabs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
              h[2] * (h[3] * h[7] - h[4] * h[6]));
}

/* phi(r) and phi'(r) of the switched pair term. */
static void pair(double r, double *phi, double *dphi) {
  const double sr6 = pow(1.0 / r, 6);
  const double v = 4.0 * (sr6 * sr6 - sr6);
  const double dv = 4.0 * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r;
  double t = (r - (RC - WIDTH)) / WIDTH;
  t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  const double s = 1.0 + t * t * t * (-10.0 + t * (15.0 - 6.0 * t));
  const double ds = -30.0 * t * t * (1.0 - t) * (1.0 - t) / WIDTH;
  *phi = v * s;
  *dphi = dv * s + v * ds;
}

/* Energy, gradient and Cauchy stress (1/V) dE/deps of one periodic
 * image, summed over every lattice translation inside the cutoff, so
 * the cell may be narrower than twice the cutoff. Rows of h are the
 * lattice vectors. */
static int lj(int64_t n, const double *x, const double *h, double *energy, double *grad,
              double *stress) {
  double inv[9];
  inverse3(h, inv);
  int reach[3];
  for (int k = 0; k < 3; ++k) {
    const double col = sqrt(inv[k] * inv[k] + inv[3 + k] * inv[3 + k] + inv[6 + k] * inv[6 + k]);
    reach[k] = (int)ceil(RC * col + 0.5);
    /* A cell this thin has left the path; refuse it rather than sum
     * millions of translations. */
    if (reach[k] > 64) {
      return 1;
    }
  }
  double e = 0.0;
  double w[9] = {0};
  memset(grad, 0, (size_t)(3 * n) * sizeof(double));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = i; j < n; ++j) {
      double d0[3];
      for (int c = 0; c < 3; ++c) {
        d0[c] = x[3 * i + c] - x[3 * j + c];
      }
      /* Fold d0 into the cell so the translation loop stays short. */
      double f[3];
      for (int c = 0; c < 3; ++c) {
        f[c] = d0[0] * inv[c] + d0[1] * inv[3 + c] + d0[2] * inv[6 + c];
      }
      for (int c = 0; c < 3; ++c) {
        const double shift = round(f[c]);
        for (int k = 0; k < 3; ++k) {
          d0[k] -= shift * h[3 * c + k];
        }
      }
      const double weight = (i == j) ? 0.5 : 1.0;
      for (int a = -reach[0]; a <= reach[0]; ++a) {
        for (int b = -reach[1]; b <= reach[1]; ++b) {
          for (int c = -reach[2]; c <= reach[2]; ++c) {
            if (i == j && a == 0 && b == 0 && c == 0) {
              continue;
            }
            double d[3];
            for (int k = 0; k < 3; ++k) {
              d[k] = d0[k] + a * h[k] + b * h[3 + k] + c * h[6 + k];
            }
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (r2 >= RC * RC) {
              continue;
            }
            const double r = sqrt(r2);
            double phi, dphi;
            pair(r, &phi, &dphi);
            e += weight * phi;
            const double g = weight * dphi / r;
            for (int k = 0; k < 3; ++k) {
              grad[3 * i + k] += g * d[k];
              grad[3 * j + k] -= g * d[k];
            }
            for (int p = 0; p < 3; ++p) {
              for (int q = 0; q < 3; ++q) {
                w[3 * p + q] += g * d[p] * d[q];
              }
            }
          }
        }
      }
    }
  }
  const double vol = volume3(h);
  *energy = e;
  for (int k = 0; k < 9; ++k) {
    stress[k] = w[k] / vol;
  }
  return 0;
}

static rgsaddle_status_t surface(void *user, rgsaddle_solid_surface_request_t *req) {
  (void)user;
  if (req->version.major != RGSADDLE_ABI_MAJOR) {
    return RGSADDLE_ABI_MISMATCH;
  }
  const int64_t rows = (req->flags & RGSADDLE_REQ_ONE_IMAGE) ? 1 : req->n_images;
  const int64_t dof = 3 * req->n_atoms;
  for (int64_t r = 0; r < rows; ++r) {
    if (lj(req->n_atoms, req->positions + dof * r, req->cells + 9 * r, req->energies + r,
           req->gradients + dof * r, req->stresses + 9 * r) != 0) {
      return RGSADDLE_SURFACE_FAILED;
    }
  }
  rows_evaluated += rows;
  return RGSADDLE_OK;
}

/* The stress the band reads must be the strain derivative of the
 * energy it reads: central differences along each lower-triangular
 * strain of a sheared sample cell. */
static int check_stress(void) {
  double h[9] = {1.2, 0.0, 0.0, 0.03, 1.19, 0.0, -0.02, 0.01, 1.31};
  double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    x[3 + k] = 0.5 * (h[k] + h[3 + k] + h[6 + k]) + 0.01 * (k + 1);
  }
  double e0, g[6], s[9];
  lj(2, x, h, &e0, g, s);
  const double vol = volume3(h);
  const int ij[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 0}, {2, 0}, {2, 1}};
  for (int m = 0; m < 6; ++m) {
    double ep, em, gs[6], ss[9];
    double hp[9], hm[9], xp[6], xm[6];
    const double step = 1e-6;
    for (int sign = -1; sign <= 1; sign += 2) {
      double eps[9] = {0};
      eps[3 * ij[m][0] + ij[m][1]] = sign * step;
      double *ho = sign > 0 ? hp : hm;
      double *xo = sign > 0 ? xp : xm;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          ho[3 * r + c] = h[3 * r + c] + h[3 * r + 0] * eps[c] + h[3 * r + 1] * eps[3 + c] +
                          h[3 * r + 2] * eps[6 + c];
        }
      }
      for (int a = 0; a < 2; ++a) {
        for (int c = 0; c < 3; ++c) {
          xo[3 * a + c] = x[3 * a + c] + x[3 * a + 0] * eps[c] + x[3 * a + 1] * eps[3 + c] +
                          x[3 * a + 2] * eps[6 + c];
        }
      }
    }
    lj(2, xp, hp, &ep, gs, ss);
    lj(2, xm, hm, &em, gs, ss);
    const double numeric = (ep - em) / (2.0 * step) / vol;
    if (fabs(numeric - s[3 * ij[m][0] + ij[m][1]]) > 1e-6) {
      printf("stress (%d,%d): analytic %.10g, central difference %.10g\n", ij[m][0], ij[m][1],
             s[3 * ij[m][0] + ij[m][1]], numeric);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const int check = argc < 2;
  const int nx = argc > 1 ? atoi(argv[1]) : 1;
  const int ny = argc > 2 ? atoi(argv[2]) : 1;
  const int nz = argc > 3 ? atoi(argv[3]) : 1;
  const int interior = argc > 4 ? atoi(argv[4]) : 7;
  const double force_tol = argc > 5 ? atof(argv[5]) : 1e-4;
  const int max_iter = argc > 6 ? atoi(argv[6]) : 5000;
  const double max_move = argc > 7 ? atof(argv[7]) : 0.1;
  const double ci_force = argc > 8 ? atof(argv[8]) : 0.5;
  const int trace = argc > 9 ? atoi(argv[9]) : 0;
  const int64_t n_atoms = 2LL * nx * ny * nz;
  const int64_t n_images = interior + 2;
  const int64_t dof = 3 * n_atoms;

  if (check_stress() != 0) {
    return 1;
  }

  /* Fractional coordinates of the supercell, shared by every image. */
  double *frac = malloc((size_t)dof * sizeof(double));
  int64_t atom = 0;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        for (int b = 0; b < 2; ++b) {
          frac[3 * atom + 0] = (i + 0.5 * b) / nx;
          frac[3 * atom + 1] = (j + 0.5 * b) / ny;
          frac[3 * atom + 2] = (k + 0.5 * b) / nz;
          ++atom;
        }
      }
    }
  }
  /* Linear interpolation of the endpoint cells, fractional
   * coordinates fixed: the initial band eOn builds as well. */
  double *cells = calloc((size_t)(9 * n_images), sizeof(double));
  double *positions = malloc((size_t)(dof * n_images) * sizeof(double));
  for (int64_t i = 0; i < n_images; ++i) {
    const double t = (double)i / (double)(n_images - 1);
    double *h = cells + 9 * i;
    h[0] = nx * ((1.0 - t) * A_BCC + t * A_FCC);
    h[4] = ny * ((1.0 - t) * A_BCC + t * A_FCC);
    h[8] = nz * ((1.0 - t) * A_BCC + t * C_FCC);
    for (int64_t a = 0; a < n_atoms; ++a) {
      for (int c = 0; c < 3; ++c) {
        positions[dof * i + 3 * a + c] = frac[3 * a + c] * h[4 * c];
      }
    }
  }

  rgsaddle_band_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.version.major = RGSADDLE_ABI_MAJOR;
  cfg.version.minor = RGSADDLE_ABI_MINOR;
  cfg.tangent = RGSADDLE_TANGENT_IMPROVED;
  cfg.spring = RGSADDLE_SPRING_UNIFORM;
  cfg.projection = RGSADDLE_PROJECTION_NEB;
  cfg.method = RGSADDLE_METHOD_FIRE;
  cfg.spring_k = 5.0;
  cfg.flags = RGSADDLE_BAND_CI_AFTER;
  cfg.ci_trigger_factor = 0.0;
  cfg.ci_trigger_force = ci_force;
  cfg.force_tol = force_tol;
  cfg.max_move = max_move;

  RgsaddleBand *band =
      rgsaddle_band_create_solid(&cfg, n_images, n_atoms, positions, cells, 0.0, 1.0);
  if (band == NULL) {
    printf("create_solid failed\n");
    return 2;
  }
  double jacobian = 0.0;
  if (rgsaddle_band_solid_jacobian(band, &jacobian) != RGSADDLE_OK) {
    return 3;
  }
  /* J = vbar^(1/3) sqrt(N), vbar the mean endpoint volume per atom
   * (validation/ssneb_bain_cell.py). */
  const double vbar = 0.5 * (0.5 * A_BCC * A_BCC * A_BCC + 0.5 * A_FCC * A_FCC * C_FCC);
  if (fabs(jacobian - cbrt(vbar) * sqrt((double)n_atoms)) > 1e-12 * jacobian) {
    printf("jacobian %.15g, closed form %.15g\n", jacobian, cbrt(vbar) * sqrt((double)n_atoms));
    return 4;
  }

  rgsaddle_report_t rep;
  memset(&rep, 0, sizeof rep);
  for (int it = 0; it < max_iter && rep.status != RGSADDLE_STATUS_CONVERGED; ++it) {
    const rgsaddle_status_t status = rgsaddle_band_step_solid(band, surface, NULL, &rep);
    if (status != RGSADDLE_OK) {
      printf("step %d: %s\n", it, rgsaddle_status_name(status));
      return 5;
    }
    if (trace > 0 && it % trace == 0) {
      double tc[9 * 64];
      if (n_images <= 64 && rgsaddle_band_cells(band, tc) == RGSADDLE_OK) {
        printf("it %5d max_force %.3e ci %lld c/a", it, rep.max_force, (long long)rep.ci_index);
        for (int64_t i = 1; i < n_images - 1; ++i) {
          printf(" %.4f", (tc[9 * i + 8] / nz) / (tc[9 * i] / nx));
        }
        printf("\n");
      }
    }
  }
  const long long force_calls = rows_evaluated;

  double *out_cells = malloc((size_t)(9 * n_images) * sizeof(double));
  double *out_pos = malloc((size_t)(dof * n_images) * sizeof(double));
  if (rgsaddle_band_cells(band, out_cells) != RGSADDLE_OK ||
      rgsaddle_band_positions(band, out_pos) != RGSADDLE_OK) {
    return 6;
  }

  /* Energies, c/a and the largest departures from the tetragonal,
   * fixed-fraction path, image by image. */
  double *grad = malloc((size_t)dof * sizeof(double));
  double stress[9];
  double shear = 0.0, upper = 0.0, square = 0.0, drift = 0.0;
  printf("image   c/a          v (A^3/atom)   E - E_bcc (eV/atom)\n");
  double *energy = malloc((size_t)n_images * sizeof(double));
  for (int64_t i = 0; i < n_images; ++i) {
    const double *h = out_cells + 9 * i;
    const double *x = out_pos + dof * i;
    if (lj(n_atoms, x, h, energy + i, grad, stress) != 0) {
      energy[i] = NAN;
    }
    energy[i] /= (double)n_atoms;
    shear = fmax(shear, fmax(fabs(h[3]), fmax(fabs(h[6]), fabs(h[7]))));
    upper = fmax(upper, fmax(fabs(h[1]), fmax(fabs(h[2]), fabs(h[5]))));
    square = fmax(square, fabs(h[0] / nx - h[4] / ny));
    /* Unwrapped fractional coordinates: a hop between periodic images
     * shows as a jump by one; a drift as anything else. */
    double inv[9];
    inverse3(h, inv);
    for (int64_t a = 0; a < n_atoms; ++a) {
      for (int c = 0; c < 3; ++c) {
        const double s = x[3 * a] * inv[c] + x[3 * a + 1] * inv[3 + c] + x[3 * a + 2] * inv[6 + c];
        drift = fmax(drift, fabs(s - frac[3 * a + c]));
      }
    }
    printf("%5lld   %.8f   %.10f   % .10f%s\n", (long long)i, (h[8] / nz) / (h[0] / nx),
           volume3(h) / (double)n_atoms, energy[i] - energy[0],
           i == rep.ci_index ? "   climbing" : "");
  }
  const int64_t ci = rep.ci_index;
  const double *hci = out_cells + 9 * (ci < 0 ? 0 : ci);
  const double q_ci = (hci[8] / nz) / (hci[0] / nx);
  const double barrier = ci < 0 ? NAN : energy[ci] - energy[0];
  printf("atoms %lld images %lld J %.10f\n", (long long)n_atoms, (long long)n_images, jacobian);
  printf("converged %d iterations %lld force_calls %lld max_force %.3e\n",
         rep.status == RGSADDLE_STATUS_CONVERGED, (long long)rep.iteration, force_calls,
         rep.max_force);
  printf("barrier %.10f eV/atom (scan %.10f) c/a %.8f (scan %.8f)\n", barrier, BARRIER, q_ci,
         Q_SADDLE);
  printf("bcc endpoint %.10f eV/atom (scan %.10f)\n", energy[0], E_BCC);
  printf("shear %.3e upper %.3e a-b %.3e fractional drift %.3e\n", shear, upper, square, drift);

  int failed = 0;
  if (rep.status != RGSADDLE_STATUS_CONVERGED) {
    printf("FAIL: not converged\n");
    failed = 1;
  }
  if (ci < 1 || ci > n_images - 2) {
    printf("FAIL: no climbing image\n");
    failed = 1;
  }
  if (check) {
    if (fabs(energy[0] - E_BCC) > 1e-9) {
      printf("FAIL: bcc endpoint energy\n");
      failed = 1;
    }
    if (!(fabs(barrier - BARRIER) <= DE_STEP)) {
      printf("FAIL: barrier off the scan by %.3e eV/atom\n", barrier - BARRIER);
      failed = 1;
    }
    if (!(fabs(q_ci - Q_SADDLE) <= DQ)) {
      printf("FAIL: c/a off the scan by %.3e\n", q_ci - Q_SADDLE);
      failed = 1;
    }
  }
  /* No rigid rotation, no shear drift, no hop. A hop moves a
   * fractional coordinate by one. The band steps atoms and cell
   * together, but its per-atom step cap can scale an atom and a cell
   * row differently; the restoring force on each site then holds the
   * fractional coordinates to well under 1e-6. */
  if (upper != 0.0 || shear > 1e-10 || square > 1e-10 || drift > 1e-6) {
    printf("FAIL: the band left the tetragonal fixed-fraction path\n");
    failed = 1;
  }
  for (int64_t i = 1; i < n_images; ++i) {
    const int up = i <= ci;
    if (ci > 0 && ((up && energy[i] < energy[i - 1]) || (!up && energy[i] > energy[i - 1]))) {
      printf("FAIL: image %lld breaks the single barrier\n", (long long)i);
      failed = 1;
    }
  }
  rgsaddle_band_free(band);
  free(frac);
  free(cells);
  free(positions);
  free(out_cells);
  free(out_pos);
  free(grad);
  free(energy);
  if (failed) {
    return 7;
  }
  printf("RGSADDLE_BAIN_LJ_OK\n");
  return 0;
}
