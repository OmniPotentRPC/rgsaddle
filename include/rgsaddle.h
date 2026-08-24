/**
 * @file rgsaddle.h
 * @brief C ABI for the rgsaddle band and minimum-mode sessions.
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
#define RGSADDLE_ABI_MINOR 9u

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

/** eOn / gpr_optim ConvergenceForceNorm. */
typedef enum {
  RGSADDLE_FORCE_L2 = 0,
  RGSADDLE_FORCE_LINF = 1,
  RGSADDLE_FORCE_MAX_ATOM = 2
} rgsaddle_force_gate_t;

typedef enum {
  RGSADDLE_STATUS_RUNNING = 0,
  RGSADDLE_STATUS_CONVERGED = 1
} rgsaddle_run_status_t;

typedef struct RgsaddleBand RgsaddleBand;
typedef struct RgsaddleMinMode RgsaddleMinMode;

/**
 * Host-surface request. The session stamps version and flags.
 * positions and gradients are n_images * 3 * n_atoms, energies is
 * n_images. Return 0 on success, negative on failure.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int64_t n_images;
  int64_t n_atoms;
  const double *positions;
  double *energies;
  double *gradients;
} rgsaddle_surface_request_t;

typedef int (*rgsaddle_surface_fn)(void *user,
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
  rgsaddle_force_gate_t force_gate;
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
int rgsaddle_abi_stamp(rgsaddle_version_t *out);
/** Human-readable name for a status code. Never NULL. */
const char *rgsaddle_status_name(int status);
/** Name of a force gate. NULL when the discriminant is unknown. */
const char *rgsaddle_force_gate_name(rgsaddle_force_gate_t gate);

/**
 * Create a band session over n_images x (3 * n_atoms) positions,
 * copied in. Endpoints never move. Returns NULL on invalid shape or
 * an unknown config major.
 */
RgsaddleBand *rgsaddle_band_create(const rgsaddle_band_config_t *config,
                                   int64_t n_images, int64_t n_atoms,
                                   const double *positions);

/** One optimizer step over the assembled band force. */
int rgsaddle_band_step(RgsaddleBand *band, rgsaddle_surface_fn surface,
                       void *user, rgsaddle_report_t *out);

/** Copy the current band out (n_images * 3 * n_atoms doubles). */
int rgsaddle_band_positions(const RgsaddleBand *band, double *out);

/** Replace the band; the host may move images between steps. */
int rgsaddle_band_set_positions(RgsaddleBand *band, const double *positions);

/** Drop optimizer history and climbing state at a surface boundary. */
int rgsaddle_band_reset(RgsaddleBand *band);

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
  rgsaddle_force_gate_t force_gate;
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
int rgsaddle_minmode_step(RgsaddleMinMode *session,
                          rgsaddle_surface_fn surface, void *user,
                          rgsaddle_report_t *out);

int rgsaddle_minmode_position(const RgsaddleMinMode *session, double *out);
int rgsaddle_minmode_mode(const RgsaddleMinMode *session, double *out);
int rgsaddle_minmode_reset(RgsaddleMinMode *session);
void rgsaddle_minmode_free(RgsaddleMinMode *session);

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
int rgsaddle_irc_step(RgsaddleIrc *session, rgsaddle_surface_fn surface,
                      void *user, rgsaddle_report_t *out);
int rgsaddle_irc_position(const RgsaddleIrc *session, double *out);
int rgsaddle_irc_set_direction(RgsaddleIrc *session, int32_t direction);
int rgsaddle_irc_reset(RgsaddleIrc *session);
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
int rgsaddle_sella_min_step(RgsaddleSellaMin *session,
                            rgsaddle_surface_fn surface, void *user,
                            rgsaddle_report_t *out);
int rgsaddle_sella_min_position(const RgsaddleSellaMin *session, double *out);
int rgsaddle_sella_min_reset(RgsaddleSellaMin *session);
/** `update` is 0 = BFGS, 1 = TS-BFGS. Unknown refuses. */
int rgsaddle_sella_min_set_hess_update(RgsaddleSellaMin *session, int32_t update);
/** Niggli-reduce a cell session. `*applied` is 1 if rewritten. */
int rgsaddle_sella_min_maybe_niggli(RgsaddleSellaMin *session,
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
int rgsaddle_sella_saddle_step(RgsaddleSellaSaddle *session,
                               rgsaddle_surface_fn surface, void *user,
                               rgsaddle_report_t *out);
int rgsaddle_sella_saddle_position(const RgsaddleSellaSaddle *session,
                                   double *out);
int rgsaddle_sella_saddle_reset(RgsaddleSellaSaddle *session);
int rgsaddle_sella_saddle_set_hess_update(RgsaddleSellaSaddle *session,
                                          int32_t update);
int rgsaddle_sella_saddle_maybe_niggli(RgsaddleSellaSaddle *session,
                                       double angle_threshold,
                                       int32_t *applied);
/** `expand` is ExpandKind: 0 lanczos, 1 gd, 2 jd0, 3 jd0_alt, 4 mjd0, 5 mjd0_alt. */
int rgsaddle_sella_saddle_set_expand(RgsaddleSellaSaddle *session,
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
int rgsaddle_constraints_fix_com(RgsaddleConstraints *cons, const double *x);

/**
 * Fix a bond between atoms `i` and `j`. `x` is 3N. `target` is the
 * length, or NULL for the current length.
 */
int rgsaddle_constraints_fix_bond(RgsaddleConstraints *cons, int64_t i,
                                  int64_t j, const double *x,
                                  const double *target);

int rgsaddle_constraints_residual_norm(const RgsaddleConstraints *cons,
                                       const double *x, double *out);

/** Project `v` onto ker(J) at `x`. `x`, `v`, and `out` are 3N. */
int rgsaddle_constraints_project(const RgsaddleConstraints *cons,
                                 const double *x, const double *v,
                                 double *out);

/** Retract `x` along `v` and restore onto the level set. */
int rgsaddle_constraints_retract(const RgsaddleConstraints *cons,
                                 const double *x, const double *v,
                                 double *out);

void rgsaddle_constraints_free(RgsaddleConstraints *cons);

typedef struct RgsaddleInternalPes RgsaddleInternalPes;

/**
 * InternalPES over a Constraints chart. `position` is 3N, `masses` is N.
 * `cons` is copied. Returns NULL on an empty chart or a shape miss.
 */
RgsaddleInternalPes *rgsaddle_internal_pes_create(
    int64_t n_atoms, const double *position, const double *masses,
    const RgsaddleConstraints *cons);
int rgsaddle_internal_pes_n_int(const RgsaddleInternalPes *pes, int64_t *out);
/** `out` holds n_int doubles. */
int rgsaddle_internal_pes_internals(const RgsaddleInternalPes *pes, double *out);
int rgsaddle_internal_pes_position(const RgsaddleInternalPes *pes, double *out);
/** `dq` is n_int. */
int rgsaddle_internal_pes_kick(RgsaddleInternalPes *pes,
                               rgsaddle_surface_fn surface, void *user,
                               const double *dq);
int rgsaddle_internal_pes_set_hess_update(RgsaddleInternalPes *pes,
                                          int32_t update);
/** `g_cart` and `out` are 3N and n_int. */
int rgsaddle_internal_pes_grad(const RgsaddleInternalPes *pes,
                               const double *g_cart, double *out);
void rgsaddle_internal_pes_free(RgsaddleInternalPes *pes);

/**
 * Niggli-reduce a row-major 3x3 cell in place when any angle is more
 * than `angle_threshold` degrees from 90. `*applied` is 1 if rewritten.
 */
int rgsaddle_niggli_reduce(double *cell, double angle_threshold,
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
int rgsaddle_samd_step(RgsaddleSamd *session, rgsaddle_surface_fn surface,
                       void *user, const double *r, rgsaddle_report_t *out);
int rgsaddle_samd_position(const RgsaddleSamd *session, double *out);
void rgsaddle_samd_free(RgsaddleSamd *session);

#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_H */
