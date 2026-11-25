/* Raw C smoke test for the solution-landscape ABI (minor 68). */
#include "rgsaddle.h"
#include "main_abi_contract.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char *msg, rgsaddle_status_t status) {
  fprintf(stderr, "%s (%s)\n", msg, rgsaddle_status_name(status));
  return 1;
}

typedef struct {
  int eval_calls;
  int hessian_calls;
} wells_ctx_t;

static rgsaddle_status_t wells_surface(void *user, rgsaddle_surface_request_t *req) {
  wells_ctx_t *ctx = (wells_ctx_t *)user;
  if (!req || req->version.major != RGSADDLE_ABI_MAJOR || req->n_images != 1 ||
      req->n_atoms != 1 || !req->positions || !req->energies || !req->gradients) {
    return RGSADDLE_INVALID_PARAMETER;
  }
  ctx->eval_calls += 1;
  const double a = req->positions[0];
  const double b = req->positions[1];
  const double c = req->positions[2];
  req->energies[0] = (a * a * a * a - 2.0 * a * a) * 0.25 +
                     (b * b * b * b - b * b) * 0.25 + 0.5 * c * c;
  req->gradients[0] = a * a * a - a;
  req->gradients[1] = b * b * b - 0.5 * b;
  req->gradients[2] = c;
  return RGSADDLE_OK;
}

static int wells_hessian(void *user, int64_t n, const double *x, const double *v,
                         double *out) {
  wells_ctx_t *ctx = (wells_ctx_t *)user;
  if (!ctx || n != 3 || !x || !v || !out) {
    return -1;
  }
  ctx->hessian_calls += 1;
  out[0] = (3.0 * x[0] * x[0] - 1.0) * v[0];
  out[1] = (3.0 * x[1] * x[1] - 0.5) * v[1];
  out[2] = v[2];
  return 0;
}

static int verify_wells_map(RgsaddleLandscape *session, const rgsaddle_landscape_report_t *rep) {
  if (rep->version.major != RGSADDLE_ABI_MAJOR ||
      rep->version.minor != RGSADDLE_ABI_MINOR) {
    return fail("report version mismatch", RGSADDLE_ABI_MISMATCH);
  }
  if (rep->n_nodes != 9 || rep->n_edges != 12 || rep->bound != 0 ||
      rep->evaluations <= 0) {
    fprintf(stderr, "unexpected report: nodes=%lld edges=%lld bound=%d evals=%lld\n",
            (long long)rep->n_nodes, (long long)rep->n_edges, rep->bound,
            (long long)rep->evaluations);
    return 1;
  }

  int indices[9];
  int counts[3] = {0, 0, 0};
  for (int64_t i = 0; i < rep->n_nodes; ++i) {
    double pos[3] = {0.0, 0.0, 0.0};
    double energy = 0.0;
    int32_t idx = -1;
    int32_t n_curv = 0;
    int64_t evals = 0;
    rgsaddle_status_t st = rgsaddle_landscape_node(
        session, i, pos, &energy, &idx, NULL, 0, &n_curv, &evals);
    if (st != RGSADDLE_OK || n_curv != 3 || idx < 0 || idx > 2 || evals <= 0) {
      return fail("node query with NULL curvatures failed", st);
    }
    double short_buf[2];
    if (rgsaddle_landscape_node(session, i, pos, &energy, &idx, short_buf, 2,
                                &n_curv, &evals) != RGSADDLE_SHAPE) {
      return fail("short curvature buffer was not refused", RGSADDLE_SHAPE);
    }
    double curv[3] = {0.0, 0.0, 0.0};
    st = rgsaddle_landscape_node(session, i, pos, &energy, &idx, curv, 3,
                                 &n_curv, &evals);
    if (st != RGSADDLE_OK) {
      return fail("full node read failed", st);
    }
    indices[i] = idx;
    counts[idx] += 1;

    int32_t ncols = -1;
    st = rgsaddle_landscape_modes(session, i, NULL, &ncols);
    if (st != RGSADDLE_OK || ncols != idx) {
      return fail("mode column query failed", st);
    }
    if (ncols > 0) {
      double modes[9];
      st = rgsaddle_landscape_modes(session, i, modes, &ncols);
      if (st != RGSADDLE_OK) {
        return fail("mode read failed", st);
      }
      for (int32_t c = 0; c < ncols; ++c) {
        double norm2 = 0.0;
        for (int r = 0; r < 3; ++r) {
          double v = modes[r * ncols + c];
          norm2 += v * v;
        }
        if (fabs(norm2 - 1.0) > 1e-6) {
          fprintf(stderr, "node %lld col %d not unit: %g\n", (long long)i, c,
                  norm2);
          return 1;
        }
      }
    }
  }
  if (indices[0] != 2 || counts[2] != 1 || counts[1] != 4 || counts[0] != 4) {
    fprintf(stderr, "unexpected saddle counts: %d/%d/%d\n", counts[2],
            counts[1], counts[0]);
    return 1;
  }

  for (int64_t e = 0; e < rep->n_edges; ++e) {
    int64_t parent = -1, child = -1, evals = 0;
    int32_t dir = -1, sign = 0;
    rgsaddle_status_t st = rgsaddle_landscape_edge(
        session, e, &parent, &child, &dir, &sign, &evals);
    if (st != RGSADDLE_OK) {
      return fail("edge read failed", st);
    }
    if (parent < 0 || parent >= rep->n_nodes || child < 0 ||
        child >= rep->n_nodes || (sign != 1 && sign != -1) || evals <= 0) {
      fprintf(stderr, "invalid edge %lld: p=%lld c=%lld sign=%d evals=%lld\n",
              (long long)e, (long long)parent, (long long)child, sign,
              (long long)evals);
      return 1;
    }
    if (indices[parent] != indices[child] + 1) {
      fprintf(stderr, "edge %lld index gap: %d -> %d\n", (long long)e,
              indices[parent], indices[child]);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  if (rgsaddle_abi_version() !=
      (int)((RGSADDLE_ABI_MAJOR << 16) | RGSADDLE_ABI_MINOR)) {
    return fail("ABI version mismatch", RGSADDLE_ABI_MISMATCH);
  }

  rgsaddle_landscape_config_t cfg;
  memset(&cfg, 0, sizeof cfg);
  rgsaddle_status_t st = rgsaddle_landscape_config_default(&cfg);
  if (st != RGSADDLE_OK) {
    return fail("config_default", st);
  }
  if (cfg.version.major != RGSADDLE_ABI_MAJOR ||
      cfg.version.minor != RGSADDLE_ABI_MINOR || cfg.cell != NULL ||
      cfg.push <= 0.0 || cfg.duplicate_distance <= 0.0) {
    return fail("unexpected config defaults", RGSADDLE_INVALID_PARAMETER);
  }

  const double origin[3] = {0.0, 0.0, 0.0};
  if (rgsaddle_landscape_create(NULL, 1, origin, NULL, 0) != NULL ||
      rgsaddle_landscape_create(&cfg, 0, origin, NULL, 0) != NULL ||
      rgsaddle_landscape_create(&cfg, 1, origin, NULL, -1) != NULL ||
      rgsaddle_landscape_create(&cfg, 1, origin, NULL, 4) != NULL ||
      rgsaddle_landscape_create(&cfg, 1, origin, NULL, 1) != NULL) {
    return fail("invalid create inputs were accepted", RGSADDLE_INVALID_PARAMETER);
  }

  cfg.push = 0.5;
  cfg.duplicate_distance = 1.0e-2;
  cfg.max_steps = 800;
  cfg.force_tol = 1.0e-8;
  cfg.relax_force_tol = 1.0e-8;
  cfg.max_move = 0.1;
  cfg.max_atom_move = 0.1;
  cfg.initial_step = 0.05;

  RgsaddleLandscape *analytic = rgsaddle_landscape_create(&cfg, 1, origin, NULL, 0);
  if (!analytic) {
    return fail("create analytic", RGSADDLE_SHAPE);
  }

  double probe_pos[3], probe_e = 0.0;
  int32_t probe_idx = 0, probe_nc = 0, probe_cols = 0;
  int64_t probe_ev = 0, probe_p = 0, probe_c = 0;
  int32_t probe_d = 0, probe_s = 0;
  if (rgsaddle_landscape_node(analytic, 0, probe_pos, &probe_e, &probe_idx,
                              NULL, 0, &probe_nc, &probe_ev) !=
          RGSADDLE_NO_EVALUATION ||
      rgsaddle_landscape_modes(analytic, 0, NULL, &probe_cols) !=
          RGSADDLE_NO_EVALUATION ||
      rgsaddle_landscape_edge(analytic, 0, &probe_p, &probe_c, &probe_d,
                              &probe_s, &probe_ev) != RGSADDLE_NO_EVALUATION) {
    rgsaddle_landscape_free(analytic);
    return fail("queries before run did not return NO_EVALUATION",
                RGSADDLE_NO_EVALUATION);
  }

  wells_ctx_t ctx = {0, 0};
  rgsaddle_landscape_report_t rep;
  memset(&rep, 0xA5, sizeof rep);
  st = rgsaddle_landscape_run(analytic, wells_surface, wells_hessian, &ctx, &rep);
  if (st != RGSADDLE_OK) {
    rgsaddle_landscape_free(analytic);
    return fail("analytic landscape_run", st);
  }
  if (rep.flags != 0u) {
    rgsaddle_landscape_free(analytic);
    return fail("landscape report flags are not zero", RGSADDLE_SOLVER);
  }
  if (ctx.hessian_calls <= 0 || ctx.eval_calls != rep.evaluations) {
    rgsaddle_landscape_free(analytic);
    return fail("evaluation or hessian count mismatch", RGSADDLE_SOLVER);
  }
  if (verify_wells_map(analytic, &rep) != 0) {
    rgsaddle_landscape_free(analytic);
    return 1;
  }
  rgsaddle_landscape_free(analytic);

  /* Finite-difference Hessian fallback (hessian == NULL). */
  RgsaddleLandscape *fd = rgsaddle_landscape_create(&cfg, 1, origin, NULL, 0);
  if (!fd) {
    return fail("create finite-difference", RGSADDLE_SHAPE);
  }
  wells_ctx_t fd_ctx = {0, 0};
  memset(&rep, 0, sizeof rep);
  st = rgsaddle_landscape_run(fd, wells_surface, NULL, &fd_ctx, &rep);
  if (st != RGSADDLE_OK || fd_ctx.hessian_calls != 0 ||
      fd_ctx.eval_calls != rep.evaluations) {
    rgsaddle_landscape_free(fd);
    return fail("finite-difference landscape_run", st);
  }
  if (verify_wells_map(fd, &rep) != 0) {
    rgsaddle_landscape_free(fd);
    return 1;
  }
  rgsaddle_landscape_free(fd);

  /* Bound limit returns RGSADDLE_LANDSCAPE_BOUND and discards pathway. */
  cfg.max_nodes = 2;
  RgsaddleLandscape *bounded = rgsaddle_landscape_create(&cfg, 1, origin, NULL, 0);
  if (!bounded) {
    return fail("create bounded", RGSADDLE_SHAPE);
  }
  wells_ctx_t bctx = {0, 0};
  memset(&rep, 0xA5, sizeof rep);
  st = rgsaddle_landscape_run(bounded, wells_surface, wells_hessian, &bctx, &rep);
  if (st != RGSADDLE_LANDSCAPE_BOUND || rep.flags != 0u || rep.bound != 1 || rep.n_edges != 0 ||
      rep.n_nodes != 2 || rep.evaluations <= 0) {
    rgsaddle_landscape_free(bounded);
    return fail("expected RGSADDLE_LANDSCAPE_BOUND", st);
  }
  if (rgsaddle_landscape_node(bounded, 0, probe_pos, &probe_e, &probe_idx,
                              NULL, 0, &probe_nc, &probe_ev) !=
      RGSADDLE_NO_EVALUATION) {
    rgsaddle_landscape_free(bounded);
    return fail("bounded session kept a pathway", RGSADDLE_SOLVER);
  }
  rgsaddle_landscape_free(bounded);
  rgsaddle_landscape_free(NULL);

  printf("RGSADDLE_LANDSCAPE_ABI_OK\n");
  return 0;
}
