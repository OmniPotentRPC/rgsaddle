/**
 * @file rgsaddle.h
 * @brief C ABI for the rgsaddle band, minimum-mode, and index-1 sessions.
 *
 * The stepping contract in C. The host owns the loop: create a
 * session, call step until it reports converged (or until the host's
 * own policy says otherwise), reset at a surface-epoch boundary,
 * free. There is no run-to-completion entry point.
 *
 * Layout discipline follows dlpack.h, the same as the rest of the
 * stack: every wire struct opens with an rgsaddle_version_t and a
 * flags word, the reader refuses a major it does not know
 * (RGSADDLE_ABI_MISMATCH), fixed-width integers only, and the
 * surface callback takes one request struct so its contract can grow
 * without a new symbol.
 *
 * Units are the caller's; the sessions are unit-agnostic. Positions
 * are unwrapped Cartesian, image-major (image stride 3 * n_atoms).
 */
#ifndef RGSADDLE_H
#define RGSADDLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RGSADDLE_ABI_MAJOR 1u
#define RGSADDLE_ABI_MINOR 4u

/**
 * Band config flags bit 0. When set, each evaluation calls the surface
 * once per image it carries (see rgsaddle_surface_request_t for which
 * images). The host copies that image's saved orbitals and density
 * into the working set, evaluates the force on that image's
 * subcommunicator, and broadcasts the energy and gradient to every
 * rank that entered the step before returning. The library does not
 * call MPI.
 */
#define RGSADDLE_BAND_PER_IMAGE (1ull << 0)

/** Request flags bit 0. positions and gradients are one image. */
#define RGSADDLE_REQ_ONE_IMAGE (1ull << 0)

/** Version head carried by every wire struct. */
typedef struct {
  uint32_t major;
  uint32_t minor;
} rgsaddle_version_t;

#define RGSADDLE_VERSION_INIT                                                  \
  { RGSADDLE_ABI_MAJOR, RGSADDLE_ABI_MINOR }

typedef enum {
  RGSADDLE_OK = 0,
  RGSADDLE_NULL_SESSION = -1,
  RGSADDLE_NULL_BAND = -2,
  RGSADDLE_NULL_SURFACE = -3,
  RGSADDLE_NULL_REPORT = -4,
  RGSADDLE_SHAPE = -5,
  RGSADDLE_SURFACE_FAILED = -6,
  RGSADDLE_NON_FINITE = -7,
  RGSADDLE_SOLVER = -8,
  RGSADDLE_ABI_MISMATCH = -9,
  RGSADDLE_ALLOC = -10,
  RGSADDLE_INVALID_PARAMETER = -11
} rgsaddle_status_t;

typedef enum {
  RGSADDLE_TANGENT_SIMPLE = 0,
  RGSADDLE_TANGENT_IMPROVED = 1
} rgsaddle_tangent_t;

typedef enum {
  RGSADDLE_SPRING_UNIFORM = 0,
  RGSADDLE_SPRING_WEIGHTED = 1,
  RGSADDLE_SPRING_ONSAGER_MACHLUP = 2
} rgsaddle_spring_t;

typedef enum {
  RGSADDLE_PROJECTION_PLAIN_EB = 0,
  RGSADDLE_PROJECTION_NEB = 1,
  RGSADDLE_PROJECTION_DNEB = 2
} rgsaddle_projection_t;

typedef enum {
  RGSADDLE_METHOD_FIRE = 0,
  RGSADDLE_METHOD_LBFGS = 1
} rgsaddle_method_t;

typedef enum {
  RGSADDLE_MINMODE_DIMER = 0,
  RGSADDLE_MINMODE_LANCZOS = 1
} rgsaddle_minmode_t;

typedef enum {
  RGSADDLE_STATUS_RUNNING = 0,
  RGSADDLE_STATUS_CONVERGED = 1
} rgsaddle_run_status_t;

typedef struct RgsaddleBand RgsaddleBand;
typedef struct RgsaddleMinMode RgsaddleMinMode;

/**
 * Host-surface request. The session stamps version and flags.
 *
 * The endpoints of a band never move, so the session evaluates them
 * once: the first evaluation after rgsaddle_band_create or
 * rgsaddle_band_set_positions carries the whole band, and every later
 * evaluation carries only the interior images 1 .. band - 2, in order.
 * rgsaddle_band_reset drops the cached endpoint energies, so the next
 * evaluation carries the whole band again.
 *
 * With flags clear (batched), image is -1 and n_images is the number
 * of images carried by this request: the band length on the first
 * evaluation, band - 2 afterwards. Row r is image r when n_images is
 * the band length and image r + 1 otherwise. positions and gradients
 * are n_images * 3 * n_atoms, energies is n_images.
 *
 * With RGSADDLE_REQ_ONE_IMAGE, image is the band index of the one
 * image carried (endpoints 0 and band - 1 appear only on the first
 * evaluation), positions and gradients are 3 * n_atoms, energies is
 * one value, and n_images is the length of the band.
 *
 * Return RGSADDLE_OK or another rgsaddle_status_t; a non-OK return
 * fails the step with the band unchanged.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int64_t n_images;
  int64_t n_atoms;
  const double *positions;
  double *energies;
  double *gradients;
  int64_t image;
} rgsaddle_surface_request_t;

typedef rgsaddle_status_t (*rgsaddle_surface_fn)(void *user,
                                                 rgsaddle_surface_request_t *req);

/** Band configuration. The caller stamps version and zeroes flags. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t tangent;    /**< rgsaddle_tangent_t */
  int32_t spring;     /**< rgsaddle_spring_t */
  int32_t projection; /**< rgsaddle_projection_t */
  int32_t method;     /**< rgsaddle_method_t */
  double spring_k;
  /** Weighted springs: n_images - 1 constants, else NULL. */
  const double *spring_ks;
  /** Climbing image off when trigger_factor <= 0. */
  double ci_trigger_factor;
  double ci_trigger_force;
  /** Row-major 3x3 cell for minimum-image differences; NULL for none. */
  const double *cell;
  double force_tol;
  double max_move;
  int64_t memory;
} rgsaddle_band_config_t;

/** One step's report. The session stamps version and flags. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t status; /**< rgsaddle_run_status_t */
  int32_t reserved;
  double max_force;
  /** Armed climbing image, or -1. */
  int64_t ci_index;
  int64_t iteration;
  /** Min-mode only: curvature along the lowest mode. */
  double curvature;
  int64_t rotations;
} rgsaddle_report_t;

/** (major << 16) | minor. */
int rgsaddle_abi_version(void);
rgsaddle_status_t rgsaddle_abi_stamp(rgsaddle_version_t *out);
/** Human-readable name for a status code. Never NULL. */
const char *rgsaddle_status_name(rgsaddle_status_t status);

/**
 * Create a band session over n_images x (3 * n_atoms) positions,
 * copied in. Endpoints never move. Returns NULL on invalid shape or
 * an unknown config major.
 */
RgsaddleBand *rgsaddle_band_create(const rgsaddle_band_config_t *config,
                                   int64_t n_images, int64_t n_atoms,
                                   const double *positions);

/** One optimizer step over the assembled band force. */
rgsaddle_status_t rgsaddle_band_step(RgsaddleBand *band,
                                     rgsaddle_surface_fn surface, void *user,
                                     rgsaddle_report_t *out);

/** Copy the current band out (n_images * 3 * n_atoms doubles). */
rgsaddle_status_t rgsaddle_band_positions(const RgsaddleBand *band, double *out);

/**
 * Replace the band; the host may move images between steps. Drops the
 * cached endpoint energies, so the next evaluation carries the whole
 * band.
 */
rgsaddle_status_t rgsaddle_band_set_positions(RgsaddleBand *band,
                                              const double *positions);

/**
 * Drop optimizer history and climbing state at a surface boundary.
 * The cached endpoint energies stay.
 */
rgsaddle_status_t rgsaddle_band_reset(RgsaddleBand *band);

void rgsaddle_band_free(RgsaddleBand *band);

/** Minimum-mode configuration. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t kind;   /**< rgsaddle_minmode_t */
  int32_t method; /**< rgsaddle_method_t */
  double dr;
  double rotation_tol;
  int64_t max_rotations;
  int64_t krylov_dim;
  double force_tol;
  double max_move;
} rgsaddle_minmode_config_t;

/**
 * Create a minimum-mode session at one geometry with an initial mode
 * estimate; both are 3 * n_atoms doubles, copied in.
 */
RgsaddleMinMode *rgsaddle_minmode_create(
    const rgsaddle_minmode_config_t *config, int64_t n_atoms,
    const double *position, const double *mode);

/** Refresh the lowest mode, invert along it, take one step. */
rgsaddle_status_t rgsaddle_minmode_step(RgsaddleMinMode *session,
                                        rgsaddle_surface_fn surface, void *user,
                                        rgsaddle_report_t *out);

rgsaddle_status_t rgsaddle_minmode_position(const RgsaddleMinMode *session,
                                                double *out);
rgsaddle_status_t rgsaddle_minmode_mode(const RgsaddleMinMode *session,
                                        double *out);
rgsaddle_status_t rgsaddle_minmode_reset(RgsaddleMinMode *session);
void rgsaddle_minmode_free(RgsaddleMinMode *session);

typedef enum {
  RGSADDLE_HESS_POWELL = 0,
  RGSADDLE_HESS_BOFILL = 1
} rgsaddle_hess_update_t;

typedef enum {
  RGSADDLE_NICHOLS_MINIMIZE = 0,
  RGSADDLE_NICHOLS_INDEX1 = 1
} rgsaddle_nichols_mode_t;

/**
 * Nichols displacement from a spectrum of the mass-weighted energy
 * Hessian. `gradient` is dE/dx (i-PI `nichols` takes forces, the
 * negation of this vector). `evecs` is row-major `n` by `nmode`,
 * column `k` the eigenvector of `evals[k]`. `nmode` may be smaller
 * than `n` when external modes were dropped. `masses` is per
 * coordinate, or NULL for unit mass.
 *
 * `RGSADDLE_NICHOLS_INDEX1` ignores `trust_radius`; cap the step with
 * `rgsaddle_cap_max_abs`. `RGSADDLE_NICHOLS_MINIMIZE` uses
 * `trust_radius` as i-PI `big_step`.
 */
rgsaddle_status_t rgsaddle_nichols_step(int64_t n, int64_t nmode,
                                        const double *gradient,
                                        const double *evals, const double *evecs,
                                        const double *masses, double trust_radius,
                                        int32_t mode, double *displacement);

typedef enum {
  RGSADDLE_PRFO_MINIMIZE = 0,
  RGSADDLE_PRFO_INDEX1 = 1
} rgsaddle_prfo_t;

/**
 * Restricted-step partitioned RFO. `hessian` is the row-major
 * Cartesian energy Hessian. `gradient` is dE/dx. `masses` is per
 * coordinate, or NULL for unit mass. `mode` is `rgsaddle_prfo_t`.
 * The Euclidean norm of `displacement` is at most `trust_radius`.
 * Index-1 maximizes the lowest mode and minimizes the rest.
 */
rgsaddle_status_t rgsaddle_prfo_step(int64_t n, const double *gradient,
                                     const double *hessian, const double *masses,
                                     double trust_radius, int32_t mode,
                                     double *displacement);

/**
 * Powell symmetric Broyden update of a Cartesian energy Hessian.
 * `step` and `dgradient` have length `n`. `hessian` is row-major
 * `n` by `n` and is updated in place. `dgradient` is the change in
 * dE/dx.
 */
rgsaddle_status_t rgsaddle_hessian_powell(int64_t n, const double *step,
                                          const double *dgradient, double *hessian);

/** Bofill mix of the Powell update and the symmetric rank-one term. */
rgsaddle_status_t rgsaddle_hessian_bofill(int64_t n, const double *step,
                                          const double *dgradient, double *hessian);

/** Scale `step` so its largest absolute entry equals `trust_radius`. */
rgsaddle_status_t rgsaddle_cap_max_abs(int64_t n, double *step, double trust_radius);

typedef struct RgsaddleIndex1 RgsaddleIndex1;

/** Index-1 Newton configuration. The caller stamps version. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t update; /**< rgsaddle_hess_update_t */
  int32_t mode;   /**< rgsaddle_nichols_mode_t */
  /** Max-abs cap on the Cartesian step. */
  double trust_radius;
  double force_tol;
  /** Central-difference step when `hessian` is NULL. */
  double fd_dr;
} rgsaddle_index1_config_t;

/**
 * Index-1 session over one geometry of `3 * n_atoms` coordinates.
 * `hessian` is row-major `dof` by `dof`, or NULL to build it by
 * finite differences on the first step. `masses` is `dof` doubles
 * or NULL for unit mass. For a bead polymer pass `n_atoms * n_beads`.
 */
RgsaddleIndex1 *rgsaddle_index1_create(const rgsaddle_index1_config_t *config,
                                       int64_t n_atoms, const double *position,
                                       const double *hessian, const double *masses);

/** One Nichols step, trust cap, and Hessian update. */
rgsaddle_status_t rgsaddle_index1_step(RgsaddleIndex1 *session,
                                       rgsaddle_surface_fn surface, void *user,
                                       rgsaddle_report_t *out);

rgsaddle_status_t rgsaddle_index1_position(const RgsaddleIndex1 *session, double *out);
/** Copies the Cartesian Hessian. `RGSADDLE_SHAPE` if it is not built. */
rgsaddle_status_t rgsaddle_index1_hessian(const RgsaddleIndex1 *session, double *out);
/** Drop the Hessian and the gradient cache at a surface boundary. */
rgsaddle_status_t rgsaddle_index1_reset(RgsaddleIndex1 *session);
void rgsaddle_index1_free(RgsaddleIndex1 *session);

#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_H */
