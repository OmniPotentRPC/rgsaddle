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
#define RGSADDLE_ABI_MINOR 0u

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

#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_H */
