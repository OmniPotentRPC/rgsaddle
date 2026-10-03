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
#define RGSADDLE_ABI_MINOR 14u

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
  RGSADDLE_INVALID_PARAMETER = -11,
  /** rgsaddle_band_evaluation: no evaluation at the current band. */
  RGSADDLE_NO_EVALUATION = -12
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

/**
 * Finite difference of the min-mode Hessian actions. Forward costs one
 * gradient per action (the centre is known) with an O(dr) curvature
 * error; central costs two with O(dr^2).
 */
typedef enum {
  RGSADDLE_DIFFERENCE_FORWARD = 0,
  RGSADDLE_DIFFERENCE_CENTRAL = 1
} rgsaddle_difference_t;

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
 * once: the first evaluation after rgsaddle_band_create,
 * rgsaddle_band_reset, or an rgsaddle_band_set_positions that moved an
 * endpoint carries the whole band, and every later evaluation carries
 * only the interior images 1 .. band - 2, in order. The session also
 * keeps the last evaluation: a step that starts where the previous one
 * ended (the host handed the band back unchanged, or only restarted
 * the optimizer) costs one interior evaluation, at its trial point.
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
  /**
   * Geometries the surface evaluated during the step: band image rows
   * (band - 2 on a warm band step, band + band - 2 on the first), or
   * single points for the min-mode session. Zero for index-1.
   */
  int32_t evaluations;
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
 * Replace the band; the host may move images between steps. Rows equal
 * to the session's up to the minimum image (1e-10 per component) are
 * kept as the session holds them, so a host that wraps atoms into its
 * cell and hands the band back changes nothing and loses no cached
 * value. A moved endpoint drops the endpoint energies (the next
 * evaluation carries the whole band); any moved row drops the last
 * evaluation. Optimizer history stays.
 */
rgsaddle_status_t rgsaddle_band_set_positions(RgsaddleBand *band,
                                              const double *positions);

/**
 * Surface boundary: drop optimizer history, climbing state, the
 * endpoint energies, and the last evaluation. Use it when the surface
 * itself changed; the next evaluation carries the whole band.
 */
rgsaddle_status_t rgsaddle_band_reset(RgsaddleBand *band);

/**
 * Optimizer restart on the same surface (after a reparameterization or
 * a host policy reset): drop optimizer history and climbing state, keep
 * the endpoint energies and the last evaluation.
 */
rgsaddle_status_t rgsaddle_band_restart(RgsaddleBand *band);

/**
 * The last evaluation, when it was taken at the current band (after a
 * step, it is the accepted point). Each output may be NULL:
 * energies holds n_images values, gradients n_images * 3 * n_atoms
 * (endpoint rows from the last whole-band evaluation), projected
 * (n_images - 2) * 3 * n_atoms (the interior force the solver stepped
 * on, climbing image included). Returns RGSADDLE_NO_EVALUATION before
 * the first step or after a change that retired it; a host reads these
 * instead of evaluating the band again.
 */
rgsaddle_status_t rgsaddle_band_evaluation(const RgsaddleBand *band,
                                           double *energies, double *gradients,
                                           double *projected);

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
  /**
   * Minor 5: the dimer also stops rotating when an iteration's optimal
   * angle falls under this (radians); 0 disables the test. Ignored
   * when version.minor < 5.
   */
  double rotation_angle_tol;
  /** Minor 5: rgsaddle_difference_t for the Hessian actions. */
  int32_t difference;
  int32_t reserved;
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

/**
 * Refresh the lowest mode at the current point without translating
 * (a host that climbs with its own optimizer). The report carries
 * curvature, rotations, and evaluations (surface calls, the centre
 * included when it was neither cached nor supplied); max_force is NaN.
 * Keep the session across a saddle search: the refreshed mode seeds
 * the next estimate, so a mode that still holds costs one evaluation.
 */
rgsaddle_status_t rgsaddle_minmode_estimate(RgsaddleMinMode *session,
                                            rgsaddle_surface_fn surface,
                                            void *user, rgsaddle_report_t *out);

/**
 * Move the session to a new point (3 * n_atoms doubles). The mode and
 * the optimizer history stay. gradient is the host's energy gradient
 * at that point, or NULL; when given, the next estimate or step skips
 * the centre evaluation.
 */
rgsaddle_status_t rgsaddle_minmode_set_position(RgsaddleMinMode *session,
                                                const double *position,
                                                const double *gradient);

/** Replace the mode seed (3 * n_atoms doubles, normalized on entry). */
rgsaddle_status_t rgsaddle_minmode_set_mode(RgsaddleMinMode *session,
                                            const double *mode);

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

/** Convergence norm for sessions with a selectable force criterion. */
typedef enum {
  RGSADDLE_FORCE_L2 = 0,
  RGSADDLE_FORCE_LINF = 1,
  RGSADDLE_FORCE_MAX_ATOM = 2
} rgsaddle_force_gate_t;
const char *rgsaddle_force_gate_name(rgsaddle_force_gate_t gate);
/** Defaults remain LINF. These setters change the norm without changing config layouts. */
rgsaddle_status_t rgsaddle_band_set_force_gate(RgsaddleBand *band,
                                               rgsaddle_force_gate_t gate);
rgsaddle_status_t rgsaddle_minmode_set_force_gate(RgsaddleMinMode *session,
                                                  rgsaddle_force_gate_t gate);


/** Basin constrained dimer. Excluded directions are consecutive Cartesian rows. */
typedef struct {
  rgsaddle_version_t version;
  double beta;
  double tolerance;
  int64_t max_iterations;
  int64_t krylov_dimension;
  int32_t eigen_kind;
} rgsaddle_kappa_config_t;

typedef struct {
  rgsaddle_version_t version;
  double kappa;
  double tangent_curvature;
  double gamma_parallel;
  double gamma_perpendicular;
  double residual;
  int64_t hessian_actions;
  int64_t tangent_dimension;
} rgsaddle_kappa_report_t;

typedef int (*rgsaddle_hessian_fn)(void *user, int64_t n_dof,
                                  const double *vector, double *action);
rgsaddle_status_t rgsaddle_kappa_config_default(rgsaddle_kappa_config_t *config);
rgsaddle_status_t rgsaddle_minmode_set_kappa(RgsaddleMinMode *session,
    const rgsaddle_kappa_config_t *config);
/** Inputs and outputs do not overlap. Callback errors leave outputs unchanged. */
rgsaddle_status_t rgsaddle_kappa_dimer_force(const rgsaddle_kappa_config_t *config,
    int64_t n_dof, const double *gradient, const double *mode,
    int64_t n_excluded, const double *excluded, rgsaddle_hessian_fn hessian,
    void *user, double *force, rgsaddle_kappa_report_t *report);



typedef enum {
  RGSADDLE_IRC_GS2 = 0,
  RGSADDLE_IRC_MOROKUMA = 1
} rgsaddle_irc_kind_t;

typedef enum {
  RGSADDLE_IRC_FORWARD = 0,
  RGSADDLE_IRC_REVERSE = 1
} rgsaddle_irc_dir_t;

typedef struct RgsaddleIrc RgsaddleIrc;

typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double dx;
  double force_tol;
  rgsaddle_force_gate_t force_gate;
  double max_move;
  int64_t max_inner;
  int32_t kind;      /**< rgsaddle_irc_kind_t */
  int32_t direction; /**< rgsaddle_irc_dir_t */
} rgsaddle_irc_config_t;

/**
 * Create an IRC session. `saddle` and `mode` are 3N, `masses` is N.
 * Returns NULL on invalid shape.
 */
RgsaddleIrc *rgsaddle_irc_create(const rgsaddle_irc_config_t *config,
                                 int64_t n_atoms, const double *saddle,
                                 const double *masses, const double *mode);
/**
 * Same, but the imaginary mode is dest Lanczos on the host surface.
 * `seed` is 3N.
 */
RgsaddleIrc *rgsaddle_irc_create_from_surface(
    const rgsaddle_irc_config_t *config, int64_t n_atoms,
    const double *saddle, const double *masses, const double *seed,
    rgsaddle_surface_fn surface, void *user);
rgsaddle_status_t rgsaddle_irc_step(RgsaddleIrc *session, rgsaddle_surface_fn surface,
                      void *user, rgsaddle_report_t *out);
rgsaddle_status_t rgsaddle_irc_position(const RgsaddleIrc *session, double *out);
rgsaddle_status_t rgsaddle_irc_set_direction(RgsaddleIrc *session, int32_t direction);
rgsaddle_status_t rgsaddle_irc_reset(RgsaddleIrc *session);
void rgsaddle_irc_free(RgsaddleIrc *session);

typedef struct RgsaddleSellaMin RgsaddleSellaMin;
typedef struct RgsaddleConstraints RgsaddleConstraints;

/** Sella order-0 (QN + trust). Default geometry is the rigid quotient. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double delta;
  double force_tol;
  rgsaddle_force_gate_t force_gate;
} rgsaddle_sella_min_config_t;

/**
 * Create a Sella minimum session. `position` is 3N, `masses` is N.
 * Unknown force_gate returns NULL.
 */
RgsaddleSellaMin *rgsaddle_sella_min_create(
    const rgsaddle_sella_min_config_t *config, int64_t n_atoms,
    const double *position, const double *masses);
/** Same, retracting on a live Constraints chart. `cons` is copied. */
RgsaddleSellaMin *rgsaddle_sella_min_create_on(
    const rgsaddle_sella_min_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons);
/** Same, QN in the internals chart (Sella InternalPES). `cons` is copied. */
RgsaddleSellaMin *rgsaddle_sella_min_create_internal(
    const rgsaddle_sella_min_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons);
/**
 * QN in packed [x; cell_params]. `cell` is row-major 3x3. `mask` is
 * 9 ints (nonzero = free); NULL means all nine free.
 */
RgsaddleSellaMin *rgsaddle_sella_min_create_cell(
    const rgsaddle_sella_min_config_t *config, int64_t n_atoms,
    const double *position, const double *masses, const double *cell,
    const int32_t *mask);
/** Packed [q_int; cell_params]. `cons` is copied. */
RgsaddleSellaMin *rgsaddle_sella_min_create_cell_internal(
    const rgsaddle_sella_min_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons, const double *cell,
    const int32_t *mask);
rgsaddle_status_t rgsaddle_sella_min_step(RgsaddleSellaMin *session,
                            rgsaddle_surface_fn surface, void *user,
                            rgsaddle_report_t *out);
rgsaddle_status_t rgsaddle_sella_min_position(const RgsaddleSellaMin *session, double *out);
rgsaddle_status_t rgsaddle_sella_min_reset(RgsaddleSellaMin *session);
/** `update` is 0 = BFGS, 1 = TS-BFGS. Unknown refuses. */
rgsaddle_status_t rgsaddle_sella_min_set_hess_update(RgsaddleSellaMin *session, int32_t update);
/** `restricted` is 0 = TrustRegion, 1 = MaxInternalStep, 2 = RestrictedAtomicStep. */
rgsaddle_status_t rgsaddle_sella_min_set_restricted(RgsaddleSellaMin *session, int32_t restricted);
/** `chart` is 0 = entries, 1 = log-deform. Unknown refuses. */
rgsaddle_status_t rgsaddle_sella_min_set_cell_chart(RgsaddleSellaMin *session, int32_t chart);
/** Niggli-reduce a cell session. `*applied` is 1 if rewritten. */
rgsaddle_status_t rgsaddle_sella_min_maybe_niggli(RgsaddleSellaMin *session,
                                    double angle_threshold, int32_t *applied);
void rgsaddle_sella_min_free(RgsaddleSellaMin *session);

typedef struct RgsaddleSellaSaddle RgsaddleSellaSaddle;

/** Sella order-1 (P-RFO + trust). Default geometry is the rigid quotient. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double delta;
  double force_tol;
  rgsaddle_force_gate_t force_gate;
  int64_t order;
} rgsaddle_sella_saddle_config_t;

RgsaddleSellaSaddle *rgsaddle_sella_saddle_create(
    const rgsaddle_sella_saddle_config_t *config, int64_t n_atoms,
    const double *position, const double *masses);
RgsaddleSellaSaddle *rgsaddle_sella_saddle_create_on(
    const rgsaddle_sella_saddle_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons);
RgsaddleSellaSaddle *rgsaddle_sella_saddle_create_internal(
    const rgsaddle_sella_saddle_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons);
RgsaddleSellaSaddle *rgsaddle_sella_saddle_create_cell(
    const rgsaddle_sella_saddle_config_t *config, int64_t n_atoms,
    const double *position, const double *masses, const double *cell,
    const int32_t *mask);
RgsaddleSellaSaddle *rgsaddle_sella_saddle_create_cell_internal(
    const rgsaddle_sella_saddle_config_t *config, int64_t n_atoms,
    const double *position, const double *masses,
    const RgsaddleConstraints *cons, const double *cell,
    const int32_t *mask);
rgsaddle_status_t rgsaddle_sella_saddle_step(RgsaddleSellaSaddle *session,
                               rgsaddle_surface_fn surface, void *user,
                               rgsaddle_report_t *out);
rgsaddle_status_t rgsaddle_sella_saddle_position(const RgsaddleSellaSaddle *session,
                                   double *out);
rgsaddle_status_t rgsaddle_sella_saddle_reset(RgsaddleSellaSaddle *session);
rgsaddle_status_t rgsaddle_sella_saddle_set_hess_update(RgsaddleSellaSaddle *session,
                                          int32_t update);
rgsaddle_status_t rgsaddle_sella_saddle_set_restricted(RgsaddleSellaSaddle *session,
                                         int32_t restricted);
rgsaddle_status_t rgsaddle_sella_saddle_maybe_niggli(RgsaddleSellaSaddle *session,
                                       double angle_threshold,
                                       int32_t *applied);
/** `expand` is ExpandKind: 0 lanczos, 1 gd, 2 jd0, 3 jd0_alt, 4 mjd0, 5 mjd0_alt. */
rgsaddle_status_t rgsaddle_sella_saddle_set_expand(RgsaddleSellaSaddle *session,
                                     int32_t expand);
void rgsaddle_sella_saddle_free(RgsaddleSellaSaddle *session);

/** Empty Constraints chart. The caller stamps version and zeroes flags. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
} rgsaddle_constraints_config_t;

/**
 * Create an empty Constraints chart on n_atoms. Returns NULL on a
 * null config, an unknown ABI major, or n_atoms < 1.
 */
RgsaddleConstraints *rgsaddle_constraints_create(
    const rgsaddle_constraints_config_t *config, int64_t n_atoms);

/** Fix the fragment COM on every axis. `x` is 3N. */
rgsaddle_status_t rgsaddle_constraints_fix_com(RgsaddleConstraints *cons, const double *x);

/**
 * Fix a bond between atoms `i` and `j`. `x` is 3N. `target` is the
 * length, or NULL for the current length.
 */
rgsaddle_status_t rgsaddle_constraints_fix_bond(RgsaddleConstraints *cons, int64_t i,
                                  int64_t j, const double *x,
                                  const double *target);

rgsaddle_status_t rgsaddle_constraints_residual_norm(const RgsaddleConstraints *cons,
                                       const double *x, double *out);

/** Project `v` onto ker(J) at `x`. `x`, `v`, and `out` are 3N. */
rgsaddle_status_t rgsaddle_constraints_project(const RgsaddleConstraints *cons,
                                 const double *x, const double *v,
                                 double *out);

/** Retract `x` along `v` and restore onto the level set. */
rgsaddle_status_t rgsaddle_constraints_retract(const RgsaddleConstraints *cons,
                                 const double *x, const double *v,
                                 double *out);

void rgsaddle_constraints_free(RgsaddleConstraints *cons);

typedef struct RgsaddlePes RgsaddlePes;

/**
 * Cartesian PES (Sella peswrapper.PES). `position` is 3N, `masses` is N.
 * `proj_trans` / `proj_rot` nonzero hang fix_translation / fix_rotation.
 * Returns NULL on a shape miss.
 */
RgsaddlePes *rgsaddle_pes_create(int64_t n_atoms, const double *position,
                                 const double *masses, int32_t proj_trans,
                                 int32_t proj_rot);
rgsaddle_status_t rgsaddle_pes_position(const RgsaddlePes *pes, double *out);
/** `d` is 3N. */
rgsaddle_status_t rgsaddle_pes_kick(RgsaddlePes *pes, rgsaddle_surface_fn surface, void *user,
                      const double *d);
rgsaddle_status_t rgsaddle_pes_set_hess_update(RgsaddlePes *pes, int32_t update);
/** Project `v` onto ker(J) at the living position. `v` and `out` are 3N. */
rgsaddle_status_t rgsaddle_pes_project(const RgsaddlePes *pes, const double *v, double *out);
/** Retract along `v` and restore onto the trans/rot set. */
rgsaddle_status_t rgsaddle_pes_retract(const RgsaddlePes *pes, const double *v, double *out);
/** Transport `v` from the living point to `x_to`. All 3N. */
rgsaddle_status_t rgsaddle_pes_transport(const RgsaddlePes *pes, const double *x_to,
                           const double *v, double *out);
void rgsaddle_pes_free(RgsaddlePes *pes);

typedef struct RgsaddleInternalPes RgsaddleInternalPes;

/**
 * InternalPES over a Constraints chart. `position` is 3N, `masses` is N.
 * `cons` is copied. Returns NULL on an empty chart or a shape miss.
 */
RgsaddleInternalPes *rgsaddle_internal_pes_create(
    int64_t n_atoms, const double *position, const double *masses,
    const RgsaddleConstraints *cons);
/**
 * Same, with `n_dummy` extra 3-vectors. `cons` covers n_atoms + n_dummy.
 * `dummies` is 3 * n_dummy.
 */
RgsaddleInternalPes *rgsaddle_internal_pes_create_dummies(
    int64_t n_atoms, const double *position, const double *masses,
    const RgsaddleConstraints *cons, int64_t n_dummy,
    const double *dummies);
/**
 * InternalPES from a vocn auto-find primitive set.
 * `position` is 3N, `masses` is N. `bonds` is 2 * n_bonds (i, j),
 * `angles` is 3 * n_angles (i, vertex, k), `dihedrals` is
 * 4 * n_dihedrals. A NULL pointer is valid only when the matching
 * count is 0. Dest does not generate Bond / Angle / Dihedral
 * topology; the host supplies the vocn find payload. Returns NULL
 * on an empty set or a shape miss.
 */
RgsaddleInternalPes *rgsaddle_internal_pes_create_from_find(
    int64_t n_atoms, const double *position, const double *masses,
    int64_t n_bonds, const int64_t *bonds, int64_t n_angles,
    const int64_t *angles, int64_t n_dihedrals,
    const int64_t *dihedrals);
/** Dummy 1 Å off the i-j bond. `out` is 3 doubles. */
rgsaddle_status_t rgsaddle_place_perp_dummy(int64_t n_atoms, const double *x, int64_t i,
                              int64_t j, double *out);
rgsaddle_status_t rgsaddle_internal_pes_n_int(const RgsaddleInternalPes *pes, int64_t *out);
/** `out` holds n_int doubles. */
rgsaddle_status_t rgsaddle_internal_pes_internals(const RgsaddleInternalPes *pes, double *out);
rgsaddle_status_t rgsaddle_internal_pes_position(const RgsaddleInternalPes *pes, double *out);
/** `dq` is n_int. */
rgsaddle_status_t rgsaddle_internal_pes_kick(RgsaddleInternalPes *pes,
                               rgsaddle_surface_fn surface, void *user,
                               const double *dq);
rgsaddle_status_t rgsaddle_internal_pes_set_hess_update(RgsaddleInternalPes *pes,
                                          int32_t update);
/** `g_cart` and `out` are 3N and n_int. */
rgsaddle_status_t rgsaddle_internal_pes_grad(const RgsaddleInternalPes *pes,
                               const double *g_cart, double *out);
void rgsaddle_internal_pes_free(RgsaddleInternalPes *pes);

/**
 * Niggli-reduce a row-major 3x3 cell in place when any angle is more
 * than `angle_threshold` degrees from 90. `*applied` is 1 if rewritten.
 */
rgsaddle_status_t rgsaddle_niggli_reduce(double *cell, double angle_threshold,
                           int32_t *applied);

typedef struct RgsaddleSamd RgsaddleSamd;

/** Sella BDP thermostat. Host supplies the Gaussian draw each step. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double dt;
  double tau;
  double t0;
  double tf;
  int64_t ngen;
  int32_t exponential;
} rgsaddle_samd_config_t;

/**
 * Create a SAMD session. `x` and `v0` are 3N. First surface eval
 * fills the living gradient. Returns NULL on a null config, unknown
 * major, or n_atoms < 1.
 */
RgsaddleSamd *rgsaddle_samd_create(const rgsaddle_samd_config_t *config,
                                   int64_t n_atoms, const double *x,
                                   const double *v0,
                                   rgsaddle_surface_fn surface, void *user);
/** One BDP step. `r` is the 3N Gaussian draw. */
rgsaddle_status_t rgsaddle_samd_step(RgsaddleSamd *session, rgsaddle_surface_fn surface,
                       void *user, const double *r, rgsaddle_report_t *out);
rgsaddle_status_t rgsaddle_samd_position(const RgsaddleSamd *session, double *out);
void rgsaddle_samd_free(RgsaddleSamd *session);



#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_H */
