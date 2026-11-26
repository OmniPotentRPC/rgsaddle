/**
 * @file rgsaddle.h
 * @brief C ABI for the rgsaddle band, minimum-mode, and index-1 sessions.
 *
 * The stepping contract in C. The host owns the loop: create a
 * session, call step until it reports converged (or until the host's
 * own policy says otherwise), reset at a surface-epoch boundary,
 * free. `rgsaddle_connect` is the run-to-completion double-ended search.
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
#define RGSADDLE_ABI_MINOR 67u

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

/**
 * Band config flags bit 1. The projected force keeps its mean
 * Cartesian part. Without it every image with two or more atoms loses
 * that mean, as in eOn when every atom is free. From ABI minor 39 a
 * band with any atom fixed by rgsaddle_band_set_fixed_atoms keeps the
 * mean whether or not the bit is set; before minor 39 a host with
 * fixed atoms set it and zeroed their forces itself.
 */
#define RGSADDLE_BAND_KEEP_TRANSLATION (1ull << 1)

/**
 * Band config flags bit 2, ABI minor 17. Climbing is on and arms once
 * the band's largest force component is under ci_trigger_force (eOn
 * ci_after) or under ci_trigger_factor times the first evaluation's
 * (eOn ci_after_rel); either may be 0. ci_trigger_force = INFINITY
 * climbs from the first evaluation. Without the bit, climbing is on
 * only for a positive ci_trigger_factor, as before minor 17. A host
 * that wants the Rust default (arm under 0.5) sets the bit, factor 0
 * and force 0.5.
 */
#define RGSADDLE_BAND_CI_AFTER (1ull << 2)

/**
 * Band config flags bit 3, ABI minor 21. No min-mode refinement of the
 * climbing image: the plain CI-NEB of minor 20 and earlier. Without
 * the bit a climbing band hands its climbing image to the OCI-NEB
 * refinement (Goswami, Gunde, and Jonsson, Front. Chem. 14, 1807063
 * (2026)) once that image has held its index for one band step. A
 * Lanczos estimate of the lowest mode must be negative with
 * |mode . tangent| >= 0.85, and then that image climbs alone by P-RFO
 * on a TS-BFGS model fit to a pair-stretch Hessian and the Lanczos
 * vectors, following that mode, while the other images stay. The band
 * resumes when the measured curvature turns positive; a refinement
 * that did not lower the force waits for half the force it started
 * from. rgsaddle_ocineb_create does not read the bit.
 */
#define RGSADDLE_BAND_NO_MIN_MODE (1ull << 3)

/**
 * Band config flags bit 4, ABI minor 38. With RGSADDLE_METHOD_RTR the
 * trust-region trial places the interior images at equal arc length on
 * the not-a-knot cubic spline through the images (E, Ren and
 * Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007)) instead of the
 * polyline. Read only from a config whose version minor is 38 or newer,
 * by rgsaddle_band_create and rgsaddle_ocineb_create.
 */
#define RGSADDLE_BAND_SPLINE_REPARAM (1ull << 4)

/**
 * Band config flags bit 5, ABI minor 44. Simplified string method of E,
 * Ren and Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007): interior
 * images evolve under the full potential force (no spring, no tangent
 * projection) and are placed at equal arc length after each step; an
 * armed climbing image reflects the potential force along the tangent
 * and stays anchored during reparametrization. RGSADDLE_BAND_SPLINE_REPARAM
 * places the images on the not-a-knot cubic spline and leaving it clear
 * places them on the polyline. Read only from a config whose version
 * minor is 44 or newer. Incompatible with RGSADDLE_METHOD_RTR and with
 * rgsaddle_band_create_solid.
 */
#define RGSADDLE_BAND_STRING (1ull << 5)

/**
 * Min-mode config flags bit 0, ABI minor 63. rotation_angle_tol was
 * set explicitly. With the bit clear and version.minor >= 63,
 * rgsaddle_minmode_create keeps the Rust default angle tolerance (five
 * degrees) when rotation_angle_tol is 0. With the bit set, the call
 * reads rotation_angle_tol as given, including 0.
 */
#define RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL (1ull << 0)

/** Request flags bit 0. positions and gradients are one image. */
#define RGSADDLE_REQ_ONE_IMAGE (1ull << 0)

/**
 * Request flags bit 1, ABI minor 23. The request carries the band's
 * climbing image alone, for the min-mode refinement (see
 * RGSADDLE_BAND_NO_MIN_MODE), and image is its band index. Set only on
 * a band whose config version minor is 23 or newer, with
 * RGSADDLE_REQ_ONE_IMAGE for a per-image band and alone (n_images = 1)
 * for a batched one. A host that keys per-image state on the band
 * index reads it here, and an MPI host can give that one image every
 * rank. rgsaddle_ocineb_step sends its climbing image to the point
 * surface as a plain one-geometry request and does not set the bit.
 */
#define RGSADDLE_REQ_CLIMBING_IMAGE (1ull << 1)

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
  RGSADDLE_NO_EVALUATION = -12,
  /**
   * A band step left the band unchanged while its largest force
   * component is above force_tol; the band is unchanged. ABI minor 17.
   */
  RGSADDLE_STALLED = -13,
  /** The constrained-ensemble tail still drifts after equilibration. ABI minor 42. */
  RGSADDLE_NOT_EQUILIBRATED = -14,
  /** The constrained-ensemble tail has too few effective samples. ABI minor 42. */
  RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES = -15,
  /**
   * A geodesic interpolation used all of its sweeps with an image still
   * moving. The outputs hold the last iterate. ABI minor 64.
   */
  RGSADDLE_NOT_CONVERGED = -16
} rgsaddle_status_t;

typedef enum {
  RGSADDLE_TANGENT_SIMPLE = 0,
  RGSADDLE_TANGENT_IMPROVED = 1
} rgsaddle_tangent_t;

typedef enum {
  RGSADDLE_SPRING_UNIFORM = 0,
  RGSADDLE_SPRING_WEIGHTED = 1,
  /**
   * Mandelli and Parrinello springs of constant spring_k. From minor
   * 33 the displacements are F / (2 spring_k) at each interior image,
   * rebuilt from the forces of every evaluation; below minor 33 they
   * were zero. A non-finite spring_k fails band creation.
   */
  RGSADDLE_SPRING_ONSAGER_MACHLUP = 2
} rgsaddle_spring_t;

typedef enum {
  RGSADDLE_PROJECTION_PLAIN_EB = 0,
  RGSADDLE_PROJECTION_NEB = 1,
  RGSADDLE_PROJECTION_DNEB = 2
} rgsaddle_projection_t;

typedef enum {
  RGSADDLE_METHOD_FIRE = 0,
  RGSADDLE_METHOD_LBFGS = 1,
  RGSADDLE_METHOD_RTR = 2, /**< Band trust-region stepper. */
  /** Quick-min band step. Min-mode create returns null for this code. */
  RGSADDLE_METHOD_QUICKMIN = 3,
  /**
   * ODE12r band step (Makri, Ortner, Kermode 2019): adaptive Euler on
   * the projected force, one band evaluation per trial, accepted or
   * not. With rgsaddle_band_set_exp_precon it is ODE12r-Exp. Min-mode
   * create returns null for this code. ABI minor 22.
   */
  RGSADDLE_METHOD_ODE12R = 4
} rgsaddle_method_t;

typedef enum {
  RGSADDLE_MINMODE_DIMER = 0,
  RGSADDLE_MINMODE_LANCZOS = 1,
  /**
   * Block LOBPCG. Valid as rgsaddle_indexk_config_t.method (ABI minor 37).
   * A one-mode session still treats any value other than Lanczos as the dimer.
   */
  RGSADDLE_MINMODE_LOBPCG = 2
} rgsaddle_minmode_t;

/** Equal arc-length placement of rgsaddle_reparametrize (ABI minor 38). */
typedef enum {
  RGSADDLE_REPARAM_LINEAR = 0,     /**< On the polyline through the images. */
  RGSADDLE_REPARAM_NOT_A_KNOT = 1  /**< On the not-a-knot cubic spline in arc length. */
} rgsaddle_reparametrization_t;

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
typedef struct RgsaddleHyperplaneSampler RgsaddleHyperplaneSampler;

/** Hyperplane ensemble request (ABI minor 42). */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double thermal;
  uint64_t seed;
  int64_t thermalize;
  int64_t stride;
  int64_t samples;
  double lambda;
  int32_t positions;
} rgsaddle_hyperplane_budget_t;

/** Borrowed hyperplane observation (ABI minor 42). */
typedef struct {
  double constraint_force;
  const double *position;
  const double *projected_force;
  double thermodynamic_derivative;
} rgsaddle_hyperplane_observation_t;

typedef int32_t (*rgsaddle_hyperplane_emit_fn)(
    void *, const rgsaddle_hyperplane_observation_t *);

typedef int32_t (*rgsaddle_sample_hyperplane_fn)(
    void *, const double *, const double *, size_t,
    const rgsaddle_hyperplane_budget_t *, rgsaddle_hyperplane_emit_fn, void *);

typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double thermal;
  int64_t nodes;
  int64_t thermalize;
  int64_t stride;
  int64_t samples;
  int64_t blocks;
  uint64_t seed;
} rgsaddle_hyperplane_ti_config_t;

typedef struct {
  double free_energy;
  double error;
  int64_t evaluations;
} rgsaddle_hyperplane_ti_result_t;

/**
 * Create a callback sampler. The callback is required; context is passed
 * through unchanged and may be NULL. Returns NULL for a null callback.
 */
RgsaddleHyperplaneSampler *rgsaddle_hyperplane_sampler_create(
    void *context, rgsaddle_sample_hyperplane_fn sample);

/**
 * Free a callback sampler. NULL is accepted. A non-null pointer must have
 * come from rgsaddle_hyperplane_sampler_create and be freed exactly once.
 */
void rgsaddle_hyperplane_sampler_free(RgsaddleHyperplaneSampler *);

/**
 * Integrate V_full - V_harmonic over lambda on a fixed hyperplane.
 * point and normal each hold dimension doubles. config is read and result
 * is written during the call; sampler must remain live. Added in ABI minor
 * 42; stamp config->version with rgsaddle_abi_stamp.
 */
int32_t rgsaddle_hyperplane_thermodynamic_integration(
    RgsaddleHyperplaneSampler *, const double *, const double *, size_t,
    const rgsaddle_hyperplane_ti_config_t *,
    rgsaddle_hyperplane_ti_result_t *);
/** Stable observable name `potential.full_minus_harmonic` for TI records. */
const char *rgsaddle_thermodynamic_derivative_observable(void);

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
 * From ABI minor 65 a dynamic NEB step sends this request, without
 * RGSADDLE_REQ_CLIMBING_IMAGE, for an interior image a neighbour
 * pulled back over its threshold.
 *
 * While the min-mode refinement runs (ABI minor 21, see
 * RGSADDLE_BAND_NO_MIN_MODE) each request carries the climbing image
 * alone: with RGSADDLE_REQ_ONE_IMAGE as above, and batched as one
 * geometry with n_images = 1 and image -1. From a config at minor 23
 * those requests also set RGSADDLE_REQ_CLIMBING_IMAGE, and a batched
 * one then has image set to the climbing image's band index.
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
  /** Climbing image off when trigger_factor <= 0, unless RGSADDLE_BAND_CI_AFTER. */
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
  /**
   * Curvature along the lowest mode: the min-mode session, and a band
   * step of the min-mode refinement (ABI minor 21); 0 on a band step.
   */
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

/**
 * Band with one cell per image. `cells` is n_images * 9 row-major
 * lattice entries. The Jacobian is fixed from the mean endpoint
 * volume. `weight` scales it and must be positive. ABI minor 16.
 */
RgsaddleBand *rgsaddle_band_create_solid(const rgsaddle_band_config_t *config,
                                         int64_t n_images, int64_t n_atoms,
                                         const double *positions,
                                         const double *cells, double pressure,
                                         double weight);

/**
 * Surface request of a solid-state band, ABI minor 16. The rows follow
 * rgsaddle_surface_request_t (batched with image -1, or one image with
 * RGSADDLE_REQ_ONE_IMAGE). Per row the host reads 3 * n_atoms
 * positions and a row-major lower-triangular 3x3 cell, and writes the
 * energy, the 3 * n_atoms gradient and the row-major Cauchy stress
 * sigma = (1/V) dE/deps for the right strain h <- h (I + eps) at fixed
 * fractional coordinates (energy per length cubed, positive in tension
 * and so minus a pressure tensor, the rgpot and eOn convention). One
 * call answers all three, so a potential that returns its stress with
 * its force costs one evaluation per image.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int64_t n_images;
  int64_t n_atoms;
  const double *positions;
  const double *cells;
  double *energies;
  double *gradients;
  double *stresses;
  int64_t image;
} rgsaddle_solid_surface_request_t;

typedef rgsaddle_status_t (*rgsaddle_solid_surface_fn)(
    void *user, rgsaddle_solid_surface_request_t *req);

/**
 * One optimizer step of a band from rgsaddle_band_create_solid: atoms
 * and interior cells move together, endpoint cells stay. A solid-state
 * band refuses rgsaddle_band_step with RGSADDLE_INVALID_PARAMETER, and
 * a fixed-cell band refuses this call the same way. ABI minor 16.
 */
rgsaddle_status_t rgsaddle_band_step_solid(RgsaddleBand *band,
                                           rgsaddle_solid_surface_fn surface,
                                           void *user, rgsaddle_report_t *out);

/** Quick-min variants (rgmin QuickMinOptions). ABI minor 16. */
typedef enum {
  /** FIRE time-step control, one projection over the band (default). */
  RGSADDLE_QUICKMIN_RGMIN = 0,
  /** eOn's quick-min: fixed time step, the velocity kept at the cap. */
  RGSADDLE_QUICKMIN_EON = 1,
  /** RGSADDLE_QUICKMIN_RGMIN with a projection per atom. */
  RGSADDLE_QUICKMIN_PER_ATOM = 2
} rgsaddle_quickmin_t;

/**
 * Select the quick-min variant of a band made with
 * RGSADDLE_METHOD_QUICKMIN; the velocity and time step restart. Any
 * other band returns RGSADDLE_INVALID_PARAMETER. ABI minor 16.
 */
rgsaddle_status_t rgsaddle_band_set_quickmin_variant(RgsaddleBand *band,
                                                     int32_t variant);

/**
 * Initial time step of FIRE and quick-min (rgmin Control::istep, 1 at
 * create); RGSADDLE_QUICKMIN_EON keeps it fixed. The band's optimizer is
 * rebuilt, so its history and the HiGHS selection go. A non-positive or
 * non-finite step returns RGSADDLE_INVALID_PARAMETER. ABI minor 16.
 */
rgsaddle_status_t rgsaddle_band_set_time_step(RgsaddleBand *band,
                                              double time_step);

/** Copy the band's cells out (n_images * 9 doubles). ABI minor 16. */
rgsaddle_status_t rgsaddle_band_cells(const RgsaddleBand *band, double *out);

/**
 * The fixed Jacobian J = (V/N)^(1/3) sqrt(N) weight, V the mean
 * endpoint volume: the length unit of the cell block. ABI minor 16.
 */
rgsaddle_status_t rgsaddle_band_solid_jacobian(const RgsaddleBand *band,
                                               double *out);

/** One optimizer step over the assembled band force. */
rgsaddle_status_t rgsaddle_band_step(RgsaddleBand *band,
                                     rgsaddle_surface_fn surface, void *user,
                                     rgsaddle_report_t *out);

/**
 * xtsci name of rgsaddle_band_step. Same arguments, same status.
 * The band session is the saddle ABI above the minimizer.
 */
rgsaddle_status_t xts_band_step(RgsaddleBand *band, rgsaddle_surface_fn surface,
                                void *user, rgsaddle_report_t *out);

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

/**
 * Project an already evaluated band. One shot, with no optimizer and no
 * arming delay: when the config asks for a climbing image, the image
 * selected by the endpoint-slope rule climbs on this call.
 *
 * positions, energies and gradients cover every image, endpoints
 * included. forces receives the same shape, in the force convention
 * (minus the energy gradient); endpoint rows are zero. *ci_index is
 * the climbing image, or -1. Spring constants are energy per length
 * squared; force_tol is in the host's force unit. ABI minor 35.
 */
rgsaddle_status_t rgsaddle_band_project(const rgsaddle_band_config_t *config,
                                        int64_t n_images, int64_t n_atoms,
                                        const double *positions,
                                        const double *energies,
                                        const double *gradients, double *forces,
                                        int64_t *ci_index);

/**
 * rgsaddle_band_project with a per-atom fixed mask, ABI minor 39.
 * fixed is NULL, which is rgsaddle_band_project, or n_atoms bytes, 1
 * for a fixed atom and 0 for a free one, as rgsaddle_band_set_fixed_atoms
 * reads them. Fixed atoms get zero force rows and the mean force stays.
 * Returns RGSADDLE_INVALID_PARAMETER for a byte other than 0 or 1, every
 * atom fixed, or endpoints that differ on a fixed atom.
 */
rgsaddle_status_t rgsaddle_band_project_fixed(const rgsaddle_band_config_t *config,
                                              int64_t n_images, int64_t n_atoms,
                                              const double *positions,
                                              const double *energies,
                                              const double *gradients,
                                              const uint8_t *fixed, double *forces,
                                              int64_t *ci_index);

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
   * angle falls under this (radians). The Rust default is five degrees
   * (Kästner and Sherwood, J. Chem. Phys. 128, 014106 (2008); eOn
   * converged_angle). For config minors 5 through 62 this field is read
   * as given, and 0 disables the angle test. From minor 63, a value of
   * 0 keeps the Rust default unless
   * RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL is set; with that bit,
   * 0 disables the test. Ignored when version.minor < 5.
   */
  double rotation_angle_tol;
  /** Minor 5: rgsaddle_difference_t for the Hessian actions. */
  int32_t difference;
  int32_t reserved;
} rgsaddle_minmode_config_t;

/**
 * ART nouveau protocol controls, ABI minor 43.
 * Initialize with rgsaddle_artn_config_default before changing fields.
 * Length and curvature values use the host's units.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags; /**< Reserved; set to zero. */
  int64_t lanczos_start;
  double eigenvalue_threshold;
  double basin_push;
  int64_t basin_relaxations;
  int64_t activation_relaxations;
  double relaxation_step;
  int64_t lanczos_actions;
  double curvature_floor;
} rgsaddle_artn_config_t;

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
 * Refresh the lowest mode without translating.
 * A dimer runs its rotation to the tolerance. Lanczos advances one
 * Hessian action and keeps the Krylov basis until set_position,
 * set_mode, or reset; repeat the call until the mode meets the host's
 * test. That basis grows to the free dimension. After k actions the
 * mode also takes its component on the residual direction, the
 * (k+1)-th Lanczos vector, from the Lanczos polynomials at an
 * extrapolated lowest eigenvalue; no surface call is added, and the
 * reported curvature stays the Ritz value of the k vectors. The report carries
 * curvature, rotations, and evaluations (surface calls, the centre
 * included when it was neither cached nor supplied); max_force is NaN.
 * The refreshed mode seeds the next estimate, so a mode that still
 * holds costs one evaluation. A repeated dimer estimate at one
 * geometry restarts rotational CG history. A failed surface callback
 * leaves retained dimer rotation history unchanged.
 */
rgsaddle_status_t rgsaddle_minmode_estimate(RgsaddleMinMode *session,
                                            rgsaddle_surface_fn surface,
                                            void *user, rgsaddle_report_t *out);

/** xtsci name of rgsaddle_minmode_estimate. Same arguments and status. */
rgsaddle_status_t xts_minmode_estimate(RgsaddleMinMode *session,
                                       rgsaddle_surface_fn surface, void *user,
                                       rgsaddle_report_t *out);

/**
 * Move the session to a new point (3 * n_atoms doubles). The mode,
 * translation optimizer history, and dimer rotational CG history stay.
 * gradient is the host's energy gradient at that point, or NULL; when
 * given, the next estimate or step skips the centre evaluation.
 */
rgsaddle_status_t rgsaddle_minmode_set_position(RgsaddleMinMode *session,
                                                const double *position,
                                                const double *gradient);

/**
 * Replace the mode seed (3 * n_atoms doubles, normalized on entry) and
 * clear dimer rotational history and carried Lanczos curvature.
 */
rgsaddle_status_t rgsaddle_minmode_set_mode(RgsaddleMinMode *session,
                                            const double *mode);

rgsaddle_status_t rgsaddle_minmode_position(const RgsaddleMinMode *session,
                                                double *out);
rgsaddle_status_t rgsaddle_minmode_mode(const RgsaddleMinMode *session,
                                        double *out);
/** Clear optimizer/rotation history, carried curvature, and the centre cache. */
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

/**
 * Step length after comparing a trial energy with the energy before the
 * step. A trial more than 1e-4 above `previous` is a rise: `*raised` is 1
 * and `*new_step` is half of `step`. Otherwise `*raised` is 0 and
 * `*new_step` is `step`. A non-finite `previous` is not a rise. No surface
 * is evaluated. ABI minor 36.
 */
rgsaddle_status_t rgsaddle_halve_raised_energy(double previous, double trial, double step,
                                              int32_t *raised, double *new_step);

/**
 * Trust radius after a predicted energy change and the change that
 * occurred. `actual` is the trial energy minus the energy before the step.
 * A prediction whose absolute value is below 1e-14 leaves `*radius_out`
 * equal to `radius`. Otherwise the ratio of `actual` to `predicted` is the
 * Sella schedule named by `sigma_inc`, `sigma_dec`, `rho_inc`, `rho_dec`,
 * and `delta_min`. No surface is evaluated. ABI minor 36.
 */
rgsaddle_status_t rgsaddle_trust_update(double radius, double predicted, double actual,
                                        double step_norm, double sigma_inc, double sigma_dec,
                                        double rho_inc, double rho_dec, double delta_min,
                                        double *radius_out);

/**
 * ABI minor 38. Place the interior images of a band at equal arc length,
 * in place. positions holds n_images rows of 3 * n_atoms doubles; rows 0
 * and n_images - 1 do not move. anchor, when it names an interior image,
 * stays, and each side of it is spaced on its own; a negative anchor is
 * none. cell, when not NULL, is the row-major 3x3 cell and the chords are
 * minimum images. scheme is an rgsaddle_reparametrization_t. The spline
 * is the parabola for three distinct images, and images whose arc lengths
 * coincide to 1e-12 of the total share one knot. A null positions, fewer
 * than two images, a non-finite entry or an unknown scheme returns
 * RGSADDLE_INVALID_PARAMETER with positions unchanged. No surface is
 * evaluated.
 */
rgsaddle_status_t rgsaddle_reparametrize(int64_t n_images, int64_t n_atoms, double *positions,
                                         int64_t anchor, const double *cell, int32_t scheme);

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



/**
 * RGSADDLE_IRC_GS2 is the Gonzalez and Schlegel step: the energy minimum
 * on the mass-weighted sphere of radius dx/2 about a pivot dx/2 down
 * the gradient. RGSADDLE_IRC_SELLA_SPHERE (ABI minor 35) is Sella's
 * step, the minimum on the sphere of radius dx about the last point.
 * A config whose version minor is 34 or older runs the Sella sphere for
 * every kind but RGSADDLE_IRC_MOROKUMA, as those versions did. Minor 34
 * is the curvature probe and leaves IRC kinds unchanged. From minor 35
 * create returns NULL on an unknown kind.
 */
typedef enum {
  RGSADDLE_IRC_GS2 = 0,
  RGSADDLE_IRC_MOROKUMA = 1,
  RGSADDLE_IRC_SELLA_SPHERE = 2
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
  double max_move;   /**< Not read. Kept for the struct layout. */
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
 * Same, but the launch mode is the lowest eigenvector of the
 * mass-weighted Hessian, from Lanczos on the host surface.
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
/** Seed a Cartesian session's Hessian with a 3N mode and its curvature.
 * The mode is projected onto the live tangent space; no surface call is made.
 * Internal and variable-cell sessions refuse Cartesian seeds. */
rgsaddle_status_t rgsaddle_sella_saddle_seed_mode(
    RgsaddleSellaSaddle *session, const double *mode, double curvature);
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
/**
 * ABI minor 24. `enabled` 1 takes the Ritz pairs of each
 * finite-difference diagonalization from the symmetrized products its
 * residual test reads, as Sella does; 0, the default, keeps the mean of
 * V^T AV and its transpose. Other values return
 * RGSADDLE_INVALID_PARAMETER.
 */
rgsaddle_status_t rgsaddle_sella_saddle_set_symmetrized_ritz(
    RgsaddleSellaSaddle *session, int32_t enabled);
/**
 * ABI minor 25. `enabled` 1 bounds only the largest atom move of a
 * Cartesian Sella saddle session, from the first step, when its
 * Euclidean radius delta * n_free starts at 1 or less (ten free
 * coordinates or fewer at the default delta), as Sella's restricted
 * atomic step does, and is the default; 0 keeps the Euclidean radius
 * alone there. Larger searches cap the atom move either way. Other
 * values return RGSADDLE_INVALID_PARAMETER.
 */
rgsaddle_status_t rgsaddle_sella_saddle_set_small_system_atom_ceiling(
    RgsaddleSellaSaddle *session, int32_t enabled);
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



/**
 * Image dependent pair potential initial path.
 *
 * Smidstrup, Pedersen, Stokbro, and Jonsson, J. Chem. Phys. 140,
 * 214106 (2014), doi:10.1063/1.4878664.
 *
 * `reactant` and `product` each hold `3 * n_atoms` doubles. `n_images`
 * counts the endpoints. `out` holds `n_images * 3 * n_atoms` doubles,
 * image-major: row 0 is the reactant and the last row is the product.
 * `cell` is a row-major 3x3 lattice, or NULL in free space. Interior
 * images are relaxed toward interpolated minimum-image pair distances.
 * `max_iterations`, `grad_tol` (L2 norm of the IDPP gradient), and
 * `max_move` (per-atom cap) are the per-image L-BFGS stop. The
 * defaults are 5000, 1e-3, and 0.1. ABI minor 19.
 */
rgsaddle_status_t rgsaddle_idpp_path(int64_t n_images, int64_t n_atoms,
                                     const double *reactant,
                                     const double *product, const double *cell,
                                     int64_t max_iterations, double grad_tol,
                                     double max_move, double *out);

/**
 * Translation rotation internal coordinates with splicing, ABI minor 66.
 *
 * `initial` and `out` must be non-null buffers owned by the caller, each
 * holding `n_images * 3 * n_atoms` doubles in image-major order. `initial`
 * and `out` may be identical for in-place refinement or disjoint; partially
 * overlapping buffers return `RGSADDLE_INVALID_PARAMETER`. `n_images < 2`,
 * `n_atoms < 1`, or dimension overflow returns `RGSADDLE_SHAPE`. The initial
 * path may come from linear or IDPP interpolation. One TRIC system is built
 * per image, low diagnostic segments are interpolated in their internal
 * metrics, and overlapping endpoints are spliced to `splice_tolerance`. The
 * published tolerance is 0.0018 Angstrom. `inversion_tolerance`, `max_move`,
 * and `splice_tolerance` are in angstrom whatever the caller's length unit,
 * and `length_per_angstrom` is the length of one angstrom in the caller's
 * coordinates (1.0 for angstrom, 0.1 for nanometres); `out` is in the
 * caller's coordinates and keeps the first and last image of `initial`
 * exactly. A non-positive or non-finite `length_per_angstrom` returns
 * `RGSADDLE_INVALID_PARAMETER`. A two-image path (which has no interior
 * image), a multi-segment split whose segments have no interior image, a
 * diagnostic maze with no stable route, a coordinate singularity that damped
 * inversion cannot pass, or an inversion or splice that does not
 * converge returns `RGSADDLE_SOLVER`. Invalid controls return
 * `RGSADDLE_INVALID_PARAMETER`; non-finite input or derived geometry returns
 * `RGSADDLE_NON_FINITE`.
 */
rgsaddle_status_t rgsaddle_trics_path(
    int64_t n_images, int64_t n_atoms, const double *initial,
    int64_t max_inversion_iterations, double inversion_tolerance,
    double max_move, int32_t diagnostic_threshold, double splice_tolerance,
    int64_t max_splice_iterations, double length_per_angstrom, double *out);

typedef struct RgsaddleOcineb RgsaddleOcineb;

/** Phase of one OCI-NEB host step. ABI minor 20. */
typedef enum {
  RGSADDLE_OCINEB_BAND = 0,
  RGSADDLE_OCINEB_ALIGN = 1,
  RGSADDLE_OCINEB_MINMODE = 2
} rgsaddle_ocineb_phase_t;

/**
 * Hand-off parameters. The band and minimum-mode configs are separate
 * arguments of rgsaddle_ocineb_create. trigger_factor 0.31 and
 * angle_tol 0.85 are the published values. angle_tol must be at least
 * 1/sqrt(2). stability_count 5 requires the counter to exceed 5.
 * max_mmf_steps 1000. restore_unhelpful 0 restores only on positive
 * curvature. From ABI minor 21 the hand-off is the band's min-mode
 * refinement (see RGSADDLE_BAND_NO_MIN_MODE) with these controls:
 * the trigger is on the climbing image's largest per-atom force, the
 * minimum-mode config supplies only the mode estimate (kind, dr,
 * difference, rotation_tol, rotation_angle_tol, max_rotations,
 * krylov_dim), and the climb is P-RFO, so its method, force_tol and
 * max_move are not read.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double trigger_factor;
  double trigger_force;
  double angle_tol;
  int64_t stability_count;
  int64_t max_mmf_steps;
  int32_t restore_unhelpful;
  int32_t reserved;
} rgsaddle_ocineb_config_t;

/** One OCI-NEB step. The session stamps version and flags. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t status; /**< rgsaddle_run_status_t */
  int32_t phase;  /**< rgsaddle_ocineb_phase_t */
  int32_t fell_back;
  int32_t evaluations;
  double max_force;
  double curvature;
  double alignment;
  double threshold;
  int64_t ci_index;
  int64_t iteration;
  int64_t rotations;
} rgsaddle_ocineb_report_t;

/**
 * Climbing-image band that hands the climbing image to a min-mode
 * climb. Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026),
 * doi:10.3389/fchem.2026.1807063. Positions are n_images x (3 * n_atoms),
 * copied in. Returns NULL on an unknown major, a bad shape, or an
 * alignment tolerance under 1/sqrt(2).
 */
RgsaddleOcineb *rgsaddle_ocineb_create(const rgsaddle_band_config_t *band,
                                       const rgsaddle_minmode_config_t *minmode,
                                       const rgsaddle_ocineb_config_t *config,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions);

/** One band step, one alignment, or one P-RFO climb step. */
rgsaddle_status_t rgsaddle_ocineb_step(RgsaddleOcineb *session,
                                       rgsaddle_surface_fn surface, void *user,
                                       rgsaddle_ocineb_report_t *out);

/** Copy the current band out (n_images * 3 * n_atoms doubles). */
rgsaddle_status_t rgsaddle_ocineb_positions(const RgsaddleOcineb *session,
                                            double *out);

rgsaddle_status_t rgsaddle_ocineb_reset(RgsaddleOcineb *session);
void rgsaddle_ocineb_free(RgsaddleOcineb *session);

/** Select the optional HiGHS feasible-set step. enabled is 0 or 1.
 * No-op in a library built without the highs Cargo feature. */
rgsaddle_status_t rgsaddle_band_set_highs(RgsaddleBand *band, int32_t enabled);
rgsaddle_status_t rgsaddle_minmode_set_highs(RgsaddleMinMode *session, int32_t enabled);
/**
 * ABI minor 32. Classic dimer torque window, apart from rotation_tol.
 * With the window on, a rotation also stops after one iteration once
 * the rotational force is under 1 (eOn torque_max). `window` 1 turns it
 * on, 0 turns it off, and -1, the state of a new session, turns it on
 * when rotation_tol is at least 1e-4 and either the estimator is
 * Lanczos or the improved dimer angle test is disabled
 * (rotation_angle_tol <= 0). From ABI minor 46 an improved dimer with
 * positive rotation_angle_tol leaves the classic window off unless
 * `window` is 1. From ABI minor 63, a caller that leaves
 * rotation_angle_tol at 0 with
 * RGSADDLE_MINMODE_EXPLICIT_ROTATION_ANGLE_TOL clear uses the default
 * improved-dimer angle stop and therefore leaves the classic window off
 * at -1; setting the bit with 0 keeps the old classic-window inference.
 * From ABI minor 41 a Lanczos session reads it too: with the window
 * on, an estimate stops once two Krylov vectors leave the Ritz residual
 * divided by the absolute Ritz value under 1. A Ritz value near zero
 * compares that residual with 1 directly. 0 restores the estimate that
 * runs to its dimensionless residual.
 * Other values return RGSADDLE_INVALID_PARAMETER and leave the session
 * unchanged.
 */
rgsaddle_status_t rgsaddle_minmode_set_torque_window(RgsaddleMinMode *session,
                                                     int32_t window);
/**
 * ABI minor 38. Curvature-aware rotation stop (Kastner and Sherwood
 * 2008). With `length` positive, a dimer rotation stops once
 * |C| * length * |sin w| is at most force_tol, where w = atan(F/|C|)/2
 * is the remaining rotation angle estimated from the rotational force F
 * and the curvature C, and |C| is the larger of the current curvature
 * and the curvature C - F tan w the rotation would reach. Lanczos stops
 * on the same bound for its Ritz residual and Ritz value. The other
 * rotation stops still apply. 0, the state of a new session, turns the
 * stop off. A negative or non-finite length returns
 * RGSADDLE_INVALID_PARAMETER and leaves the session unchanged.
 */
rgsaddle_status_t rgsaddle_minmode_set_rotation_stop_length(RgsaddleMinMode *session,
                                                            double length);

/**
 * ABI minor 43. Write the published ART nouveau defaults.
 */
rgsaddle_status_t rgsaddle_artn_config_default(rgsaddle_artn_config_t *config);

/**
 * ABI minor 43. Enable the ART nouveau push-relax protocol on a
 * Lanczos session. config NULL disables it. Enabling on a dimer or
 * passing invalid controls returns RGSADDLE_INVALID_PARAMETER and
 * leaves the session unchanged. Either successful change restarts
 * search history at the current position and mode.
 */
rgsaddle_status_t rgsaddle_minmode_set_artn(
    RgsaddleMinMode *session, const rgsaddle_artn_config_t *config);

/**
 * Per-image exponential metric (Packwood et al. 2016) inside the band
 * projection, as Makri, Ortner and Kermode 2019, Eq. (9) and (10),
 * write it: each image is driven by -(P^-1 g - t_P (t_P . g)) plus the
 * spring along the P-normalised tangent t_P, and the stepper
 * integrates that force. ODE12r-Exp with RGSADDLE_METHOD_ODE12R; FIRE
 * and L-BFGS step on the same preconditioned force. The band relaxes
 * to the minimum energy path of the metric's geometry (gradient along
 * P t_P), and the band force that convergence and the report measure
 * is the paper's residual -(g - P t_P (t_P . g)) per image, zero on
 * that path where the Euclidean perpendicular force is not. The spring
 * constant enters as the paper's dimensionless kappa = k / mu, and an
 * armed climbing image reflects the preconditioned gradient in the
 * P-metric, -(P^-1 g - 2 t_P (t_P . g)), whose fixed point is the
 * saddle.
 * Pairs within r_cut_factor * r_nn of one another in the same image
 * enter the matrix, with r_nn the largest per-atom nearest-neighbour
 * distance over the interior images, and the matrices are rebuilt
 * when an atom has moved r_nn / 2 since the last build. mu <= 0
 * estimates the energy scale on the first step from one row of the
 * first interior image; a, c_stab and r_cut_factor at or below zero
 * take 3, 0.1 and 2. Refused on a solid-state band, on single-atom
 * images and with a non-uniform spring. ABI minor 22. On a band with
 * fixed atoms (rgsaddle_band_set_fixed_atoms, minor 39) each matrix is
 * built on the free atoms: fixed atoms still count for r_nn and the
 * cutoff, and a free atom's bond to a fixed one adds to its diagonal
 * only, as ASE's Exp does under FixAtoms. P^-1 then acts on the free
 * block and the fixed atoms do not move.
 */
rgsaddle_status_t rgsaddle_band_set_exp_precon(RgsaddleBand *band, double a,
                                               double c_stab, double mu,
                                               double r_cut_factor);

/**
 * Per-atom fixed mask shared by every image, ABI minor 39. mask holds
 * n_atoms bytes, 1 for a fixed atom and 0 for a free one; all zeros
 * clears the mask and leaves the band bitwise as one that never had
 * it. A fixed atom never moves in any band stepper (FIRE, L-BFGS,
 * quick-min, ODE12r, RTR): its gradient reads as zero, so it carries no
 * force in the step, in max_force or in rgsaddle_band_evaluation.
 * Tangents and springs are formed on the free coordinates, the
 * climbing image and the min-mode refinement step the free block only,
 * and the exponential metric of rgsaddle_band_set_exp_precon, Packwood
 * et al. 2016 as Makri, Ortner and Kermode, J. Chem. Phys. 150, 094109
 * (2019), apply it, is built on the free atoms. With any atom fixed the mean force stays, as if
 * RGSADDLE_BAND_KEEP_TRANSLATION were set.
 *
 * Call it after rgsaddle_band_create and between steps. The endpoints
 * must agree on every fixed atom up to the minimum image (1e-10 per
 * component); the interior images take the first image's fixed
 * coordinates, now and after each rgsaddle_band_set_positions, which
 * applies the same endpoint check. Optimizer history, the climbing
 * state and the last evaluation are dropped; the endpoint energies
 * stay.
 *
 * Returns RGSADDLE_NULL_BAND for a null band, RGSADDLE_SHAPE for a
 * null mask or n_atoms other than the band's, and
 * RGSADDLE_INVALID_PARAMETER for a byte other than 0 or 1, every atom
 * fixed, endpoints that differ on a fixed atom, or a solid-state band.
 * On an error the band is unchanged.
 */
rgsaddle_status_t rgsaddle_band_set_fixed_atoms(RgsaddleBand *band,
                                                const uint8_t *mask,
                                                int64_t n_atoms);

/**
 * Zoom-NEB, eOn's zoom_* keys. Once the climbing image has kept its
 * index for `stability` band steps and the largest per-image force is
 * under `after`, the images are placed at equal arc length on a window
 * around it. Each image force is reduced by the band's convergence norm, as
 * eOn reduces it; `after` 0 takes twice force_tol.
 *
 * The window takes the contiguous images above the lower endpoint
 * plus `alpha` of the barrier; `alpha` 0, or a window of the climber
 * alone, takes `offset` images on each side. Unlike eOn, the window
 * then always reaches both neighbours of the climbing image. Its end
 * images become the fixed endpoints, with their energies kept, and
 * climbing stays armed. `cubic` 1 places along cubic Hermite segments,
 * 0 linearly. The band zooms once. Until the climbing image has kept
 * its index for `stability` steps, the min-mode refinement does not
 * start. `enabled` 0 clears the zoom and ignores the rest.
 *
 * Recommended when enabled: alpha 0.5, offset 1, after 0, stability 5,
 * cubic 1. Returns RGSADDLE_NULL_BAND for a null band. Returns
 * RGSADDLE_INVALID_PARAMETER for alpha outside [0, 1), offset or
 * stability under 1, a negative or non-finite after, cubic other than
 * 0 or 1, a solid-state band or a non-uniform spring. ABI minor 31.
 */
rgsaddle_status_t rgsaddle_band_set_zoom(RgsaddleBand *band, int32_t enabled,
                                         double alpha, int64_t offset,
                                         double after, int64_t stability,
                                         int32_t cubic);

/**
 * Select the simplified string method on the band, ABI minor 44.
 * This is the method of E, Ren and Vanden-Eijnden, J. Chem. Phys. 126,
 * 164103 (2007). `enabled` 1
 * evolves interior images under the full potential force (no spring, no
 * tangent projection) and places them at equal arc length after each
 * step according to `scheme` (`rgsaddle_reparametrization_t`:
 * RGSADDLE_REPARAM_LINEAR or RGSADDLE_REPARAM_NOT_A_KNOT), anchoring an
 * armed climbing image; `enabled` 0 returns the band to NEB and ignores
 * `scheme`. Optimizer history and the climbing state restart; cached
 * endpoint energies stay. Returns RGSADDLE_NULL_BAND for a null band and
 * RGSADDLE_INVALID_PARAMETER for `enabled` other than 0 or 1, an unknown
 * `scheme`, a solid-state band, or a trust-region band.
 */
rgsaddle_status_t rgsaddle_band_set_string(RgsaddleBand *band, int32_t enabled,
                                           int32_t scheme);

/**
 * Select dynamic NEB on the band, ABI minor 65.
 * This is the method of Lindgren, Kastlunger and Peterson
 * (J. Chem. Theory Comput. 15, 5787 (2019); ASE 3.29 DyNEB).
 * `enabled` 0 clears it and ignores fmax and scale_fmax. `enabled` 1
 * leaves an image in place once the largest per-atom norm of its
 * projected force is strictly below `fmax`, scaled by
 * `fmax * (1 + distance * scale_fmax)` where distance is the Euclidean
 * norm of the coordinate difference from the highest-energy image.
 * A converged image keeps its last energy and force and stays in the
 * spring and the tangent. It is evaluated again when a neighbour moves
 * enough that the projected force exceeds that threshold. The highest
 * energy image is never given a zero force. The default band, with
 * this unset, evaluates every image.
 *
 * Refused on a solid-state band, the trust-region stepper, the
 * simplified string method and free-end adaptive NEB. A refused call
 * leaves the band unchanged.
 * Returns RGSADDLE_NULL_BAND for a null band,
 * RGSADDLE_INVALID_PARAMETER for enabled other than 0 or 1 or for one
 * of those refused bands, and RGSADDLE_NON_FINITE for a non-finite
 * fmax or scale_fmax when enabled is 1.
 */
rgsaddle_status_t rgsaddle_band_set_dynamic(RgsaddleBand *band, int32_t enabled,
                                            double fmax, double scale_fmax);

/**
 * ASE 3.29 DyNEB mask, ABI minor 65. `positions`, `energies` and
 * `forces` are the whole band, image-major, `n_images * 3 * n_atoms`
 * for the coordinates and the forces. Interior force rows whose
 * largest per-atom norm is strictly below
 * `fmax * (1 + distance * scale_fmax)` are set to zero, except the
 * highest-energy image, whose force is kept. `distance` does not use
 * the minimum image. `frozen`, when non-null, receives `n_images`
 * bytes: 1 when that image would not be moved (the force after this
 * mask is strictly below `fmax`). Both endpoints are 1.
 *
 * Returns RGSADDLE_SHAPE for a null pointer, fewer than 3 images or a
 * non-positive atom count, and RGSADDLE_NON_FINITE for a non-finite
 * position, energy, force, fmax or scale_fmax.
 */
rgsaddle_status_t rgsaddle_dyneb_mask(int64_t n_images, int64_t n_atoms,
                                      const double *positions,
                                      const double *energies, double *forces,
                                      double fmax, double scale_fmax,
                                      uint8_t *frozen);

/**
 * Point stencil. Opaque config, typed keys, one callback.
 * rgsaddle_stencil_run is a loop over rgsaddle_stencil_step.
 * The older band and min-mode sessions stay step-only.
 * Band presets (5 through 13) fail at rgsaddle_stencil_new with
 * "band presets stay on feat/band-methods-zoom" from rgsaddle_last_error()
 * (kept for compatibility); use rgsaddle_band_create for bands.
 * These symbols are additive: opaque handles and
 * rgsaddle_stencil_report_t.struct_size carry layout without RGSADDLE_ABI_MINOR.
 */
typedef struct rgsaddle_stencil_config rgsaddle_stencil_config_t;
typedef struct rgsaddle_stencil rgsaddle_stencil_t;

#define RGSADDLE_STENCIL_MIN 0
#define RGSADDLE_STENCIL_DIMER 1
#define RGSADDLE_STENCIL_LANCZOS 2
#define RGSADDLE_STENCIL_ART 3
#define RGSADDLE_STENCIL_EIGEN 4
#define RGSADDLE_STENCIL_NEB 5
#define RGSADDLE_STENCIL_DNEB 6
#define RGSADDLE_STENCIL_CI_NEB 7
#define RGSADDLE_STENCIL_ENERGY_WEIGHTED_NEB 8
#define RGSADDLE_STENCIL_ZOOM_NEB 9
#define RGSADDLE_STENCIL_AUTONEB 10
#define RGSADDLE_STENCIL_OCI_NEB 11
#define RGSADDLE_STENCIL_IDPP 12
#define RGSADDLE_STENCIL_SIDPP 13

#define RGSADDLE_STENCIL_KEY_DR 1
#define RGSADDLE_STENCIL_KEY_FORCE_TOL 2
#define RGSADDLE_STENCIL_KEY_MAX_MOVE 3
#define RGSADDLE_STENCIL_KEY_ROTATION_TOL 4
#define RGSADDLE_STENCIL_KEY_MAX_ROTATIONS 5
#define RGSADDLE_STENCIL_KEY_KRYLOV_DIM 6
#define RGSADDLE_STENCIL_KEY_THREADS 7
#define RGSADDLE_STENCIL_KEY_ART_THRESHOLD 8
#define RGSADDLE_STENCIL_KEY_ART_PUSH 9
#define RGSADDLE_STENCIL_KEY_MODE 10

#define RGSADDLE_STENCIL_QUERY_POSITIONS 1
#define RGSADDLE_STENCIL_QUERY_MODE 2
#define RGSADDLE_STENCIL_QUERY_CURVATURE 3

typedef int (*rgsaddle_stencil_eval_fn)(void *user, size_t rows, size_t dof,
                                        const uint64_t *ids, const double *x,
                                        double *e, double *g);

/** Leading struct_size is sizeof(rgsaddle_stencil_report_t). Zero means
 * the current size. status is 0 while running and 1 when converged.
 * The function return is the error status. */
typedef struct {
  uint64_t struct_size;
  int32_t status;
  int32_t reserved;
  double max_force;
  double curvature;
  uint64_t iteration;
  uint64_t evaluations;
} rgsaddle_stencil_report_t;

/** Diagnostic for the calling thread. Copy before the next stencil or LI-NEB call. */
const char *rgsaddle_last_error(void);

rgsaddle_stencil_config_t *rgsaddle_stencil_config_new(int32_t preset);
rgsaddle_status_t rgsaddle_stencil_config_set_f64(rgsaddle_stencil_config_t *config,
                                                  int32_t key, double value);
rgsaddle_status_t rgsaddle_stencil_config_set_i64(rgsaddle_stencil_config_t *config,
                                                  int32_t key, int64_t value);
rgsaddle_status_t rgsaddle_stencil_config_set_array(rgsaddle_stencil_config_t *config,
                                                    int32_t key, const double *values,
                                                    size_t n);
void rgsaddle_stencil_config_free(rgsaddle_stencil_config_t *config);

rgsaddle_stencil_t *rgsaddle_stencil_new(const rgsaddle_stencil_config_t *config,
                                         size_t rows, size_t dof, const double *x);
rgsaddle_status_t rgsaddle_stencil_step(rgsaddle_stencil_t *stencil,
                                        rgsaddle_stencil_eval_fn eval, void *user,
                                        rgsaddle_stencil_report_t *out);
rgsaddle_status_t rgsaddle_stencil_run(rgsaddle_stencil_t *stencil,
                                       rgsaddle_stencil_eval_fn eval, void *user,
                                       size_t max_steps, rgsaddle_stencil_report_t *out);
rgsaddle_status_t rgsaddle_stencil_positions(const rgsaddle_stencil_t *stencil, double *out,
                                             size_t n);
rgsaddle_status_t rgsaddle_stencil_query(const rgsaddle_stencil_t *stencil, int32_t key,
                                         double *out, size_t n);
void rgsaddle_stencil_free(rgsaddle_stencil_t *stencil);

/**
 * Double-ended connect, ABI minor 18. The caller stamps version.
 * max_calls is the COMPASS iteration cap and must be positive.
 * force_tol is the Euclidean force stop. quench_nodes nonzero
 * minimizes nodes created by a split before the next segment.
 * Unspecified COMPASS constants stay at the optbench values
 * (alpha0 0.001, history 10, trust radius 0.05).
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double force_tol;
  int64_t max_calls;
  int32_t quench_nodes;
} rgsaddle_connect_config_t;

/** kind is 's', 'u', or 'f'. evaluations counts every surface call. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double energy;
  double force_norm;
  int32_t kind;
  int32_t hessian_index;
  int32_t evaluations;
} rgsaddle_connect_report_t;

typedef struct RgsaddleIndexK RgsaddleIndexK;

/**
 * Index-k minimum-mode configuration, ABI minor 37.
 * method is rgsaddle_minmode_t: dimer, Lanczos, or RGSADDLE_MINMODE_LOBPCG.
 * The translator is FIRE. k == 1 is the one-mode step of that estimator
 * (LOBPCG at k == 1 uses Lanczos). The dimer angle stop is five degrees,
 * the Rust default; this struct has no angle field.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t index; /**< k >= 1 */
  int32_t method;
  double dr;
  double rotation_tol;
  int64_t max_rotations;
  double force_tol;
  double max_move;
  double residual_tol;
  int32_t max_block_iterations;
} rgsaddle_indexk_config_t;

/**
 * Create an index-k session. position holds 3 * n_atoms doubles.
 * modes is row-major (3 * n_atoms) by k, or NULL for the coordinate seed.
 * Returns NULL on an unknown major, a non-positive k, or a bad shape.
 */
RgsaddleIndexK *rgsaddle_indexk_create(const rgsaddle_indexk_config_t *config,
                                       int64_t n_atoms, const double *position,
                                       const double *modes);

/**
 * One index-k step. hessian NULL estimates the action from surface, as
 * rgsaddle_minmode_step does. The report's curvature is the lowest of
 * the k curvatures. Modes and the translated force are read back with
 * rgsaddle_indexk_modes and rgsaddle_indexk_translated.
 */
rgsaddle_status_t rgsaddle_indexk_step(RgsaddleIndexK *session,
                                       rgsaddle_surface_fn surface,
                                       rgsaddle_hessian_fn hessian, void *user,
                                       rgsaddle_report_t *out);

rgsaddle_status_t rgsaddle_indexk_position(const RgsaddleIndexK *session,
                                           double *out);
/** Writes n-by-k row-major modes and k curvatures. */
rgsaddle_status_t rgsaddle_indexk_modes(const RgsaddleIndexK *session,
                                        double *modes, double *curvatures);
/**
 * The effective gradient of the last successful step, length 3 * n_atoms.
 * RGSADDLE_NO_EVALUATION before the first step.
 */
rgsaddle_status_t rgsaddle_indexk_translated(const RgsaddleIndexK *session,
                                             double *out);
rgsaddle_status_t rgsaddle_indexk_set_position(RgsaddleIndexK *session,
                                               const double *position,
                                               const double *gradient);
void rgsaddle_indexk_free(RgsaddleIndexK *session);

/**
 * Householder reflection of force through an orthonormal n-by-k block.
 * modes is row-major. out must not alias force. k == 1 is
 * force - v * dot(force, v) * 2.
 */
rgsaddle_status_t rgsaddle_reflect_force(int64_t n, int64_t k,
                                         const double *force,
                                         const double *modes, double *out);

/**
 * Run the connect search. left and right each hold 3 * n_atoms
 * positions. position receives the returned geometry. The surface is
 * called with one image and writes the energy and the Cartesian
 * gradient. A non-OK surface status fails the search.
 */
rgsaddle_status_t rgsaddle_connect(const rgsaddle_connect_config_t *config,
                                   int64_t n_atoms, const double *left,
                                   const double *right,
                                   rgsaddle_surface_fn surface, void *user,
                                   double *position,
                                   rgsaddle_connect_report_t *report);

/**
 * Saddle-index certification, ABI minor 40. Once a band (its climbing
 * image), a min-mode session or an OCI-NEB session converges, the
 * lowest curvatures on the free coordinates are estimated by block
 * Lanczos on finite-difference Hessian actions, seeded with the
 * search's own mode and pseudo-random vectors; the index is the count
 * under -negative_tol. Fixed atoms (rgsaddle_band_set_fixed_atoms),
 * the rigid motions the band removes, and coordinates that are exactly
 * zero in the gradient and in a forward difference along them, as a
 * host that zeroes the force on its fixed atoms leaves them, are left
 * out. The actions count in the step's evaluations. A band, min-mode
 * or OCI-NEB config stamped with minor 40 or newer certifies by
 * default with rgsaddle_saddle_check_default; an older config does
 * not, until rgsaddle_*_set_saddle_check.
 *
 * block vectors enter the Krylov space per expansion; lowest
 * curvatures are resolved, plus the first non-negative one past every
 * negative curvature, to Ritz residual residual_tol within
 * max_actions actions of displacement dr (difference is
 * rgsaddle_difference_t). seed drives the pseudo-random start
 * vectors. escape_attempts above 0 pushes a point of index two or more
 * by escape_push (largest per-atom displacement; doubled on each later
 * attempt) along its extra negative modes, each to its lower-energy
 * side, and searches again from there seeded with the negative mode
 * closest to the band tangent or the session's mode; the step that
 * pushes reports RGSADDLE_STATUS_RUNNING. 0 reports the index and
 * stops.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  int32_t block;
  int32_t lowest;
  int32_t max_actions;
  int32_t difference; /**< rgsaddle_difference_t */
  double dr;
  double residual_tol;
  double negative_tol;
  uint64_t seed;
  int32_t escape_attempts;
  int32_t reserved;
  double escape_push;
} rgsaddle_saddle_check_t;

/** A certificate, ABI minor 40. The library stamps version and flags. */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  /** 1 when a certificate holds for the current point; else 0 and the
   * counts below are those of the session. */
  int32_t certified;
  /** Curvatures under -negative_tol; 1 for a first-order saddle. */
  int32_t index;
  /** 1 when every resolved pair met residual_tol or the Krylov space
   * filled the free space; 0 when max_actions ran out first. */
  int32_t converged;
  /** Curvatures resolved, ascending; min(n_curvatures, capacity) are
   * written. */
  int32_t n_curvatures;
  /** Surface evaluations of this certificate. */
  int64_t evaluations;
  /** Surface evaluations of every certificate and escape push of the
   * session. */
  int64_t total_evaluations;
  /** Escape pushes of the session. */
  int64_t escapes;
} rgsaddle_saddle_index_t;

/**
 * Fill out with the default saddle check, ABI minor 40. Block 2, lowest
 * 4, max_actions 100, forward differences, dr 1e-3, residual_tol 1e-2,
 * negative_tol 1e-3, escape_attempts 0 and escape_push 0.1. Stamps
 * version and zeroes flags. RGSADDLE_NULL_REPORT for a null out.
 */
rgsaddle_status_t rgsaddle_saddle_check_default(rgsaddle_saddle_check_t *out);

/**
 * Replace the band's saddle check between steps, ABI minor 40; NULL
 * turns it off. The last certificate is dropped and the escape
 * attempts start again; the session's counts stay. Returns
 * RGSADDLE_NULL_BAND, RGSADDLE_ABI_MISMATCH for another major, or
 * RGSADDLE_INVALID_PARAMETER for a non-positive block, lowest, dr or
 * residual_tol, max_actions under block, a negative negative_tol,
 * negative escape_attempts, or escape_attempts with a non-positive
 * escape_push. On an error the band is unchanged.
 */
rgsaddle_status_t rgsaddle_band_set_saddle_check(RgsaddleBand *band,
                                                 const rgsaddle_saddle_check_t *check);

/**
 * The certificate of the climbing image at its current position, ABI
 * minor 40. curvatures may be NULL when capacity is 0. RGSADDLE_NULL_BAND or
 * RGSADDLE_NULL_REPORT for a null band or out, RGSADDLE_SHAPE for a
 * negative capacity or a null curvatures with a positive capacity.
 */
rgsaddle_status_t rgsaddle_band_saddle_index(const RgsaddleBand *band,
                                             rgsaddle_saddle_index_t *out,
                                             double *curvatures, int64_t capacity);

/** rgsaddle_band_set_saddle_check for a min-mode session. */
rgsaddle_status_t rgsaddle_minmode_set_saddle_check(RgsaddleMinMode *session,
                                                    const rgsaddle_saddle_check_t *check);

/** rgsaddle_band_saddle_index for a min-mode session. */
rgsaddle_status_t rgsaddle_minmode_saddle_index(const RgsaddleMinMode *session,
                                                rgsaddle_saddle_index_t *out,
                                                double *curvatures, int64_t capacity);

/** rgsaddle_band_set_saddle_check for an OCI-NEB session. */
rgsaddle_status_t rgsaddle_ocineb_set_saddle_check(RgsaddleOcineb *session,
                                                   const rgsaddle_saddle_check_t *check);

/** rgsaddle_band_saddle_index for an OCI-NEB session. */
rgsaddle_status_t rgsaddle_ocineb_saddle_index(const RgsaddleOcineb *session,
                                               rgsaddle_saddle_index_t *out,
                                               double *curvatures, int64_t capacity);

/**
 * Certify one point, ABI minor 40. position holds 3 * n_atoms values;
 * gradient is its gradient or NULL (one surface call then finds it),
 * mode a direction to seed with or NULL, and fixed n_atoms bytes (1
 * fixed, 0 free) or NULL. surface is called as by
 * rgsaddle_minmode_step. check NULL takes the defaults of
 * rgsaddle_saddle_check_default; its escape fields are not read.
 * Returns RGSADDLE_NULL_REPORT, RGSADDLE_NULL_SURFACE, RGSADDLE_SHAPE
 * for a null position or the capacity errors of
 * rgsaddle_band_saddle_index, RGSADDLE_INVALID_PARAMETER for a bad
 * check or a fixed byte other than 0 or 1, or the surface's error.
 */
rgsaddle_status_t rgsaddle_saddle_index(const rgsaddle_saddle_check_t *check,
                                        int64_t n_atoms, const double *position,
                                        const double *gradient, const double *mode,
                                        const uint8_t *fixed, rgsaddle_surface_fn surface,
                                        void *user, rgsaddle_saddle_index_t *out,
                                        double *curvatures, int64_t capacity);

/** Opaque handle to a growing string session, ABI minor 44. */
typedef struct RgsaddleGrowingString RgsaddleGrowingString;

/** Phase of a growing string session, ABI minor 44. */
typedef enum {
  RGSADDLE_GROWING_STRING_GROWING = 0,
  RGSADDLE_GROWING_STRING_JOINED = 1
} rgsaddle_growing_string_phase_t;

/**
 * Configuration of rgsaddle_growing_string_create, ABI minor 44.
 * Peters, Heyden, Bell, and Chakraborty, J. Chem. Phys. 120, 7877 (2004),
 * doi:10.1063/1.1691018.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  rgsaddle_band_config_t band;
  int64_t n_images;
  double frontier_tol;
  int64_t min_steps_per_node;
  int64_t max_steps_per_node;
  double join_tol;
  int32_t reparametrization;
  int32_t force_gate;
} rgsaddle_growing_string_config_t;

/**
 * Fill out with the default growing-string controls (n_images 11,
 * frontier_tol 0.2, min_steps_per_node 1, not-a-knot spline), ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_config_default(
    rgsaddle_growing_string_config_t *out);

/**
 * Open a growing string session between reactant and product (3 * n_atoms
 * values each), ABI minor 44. Returns NULL when the config or shape is
 * invalid: n_images below 4, a frontier_tol that is not finite and positive,
 * min_steps_per_node below 1, a negative max_steps_per_node (0 means no
 * cap), or a negative join_tol other than -1.0 or a non-finite join_tol
 * (-1.0 means no early join).
 */
RgsaddleGrowingString *rgsaddle_growing_string_create(
    const rgsaddle_growing_string_config_t *config, int64_t n_atoms,
    const double *reactant, const double *product);

/**
 * Advance a growing string session by one step (or one simplified-string
 * step once joined), ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_step(RgsaddleGrowingString *session,
                                               rgsaddle_surface_fn surface,
                                               void *user,
                                               rgsaddle_report_t *out);

/**
 * Active image count of a growing string session (left_nodes + right_nodes
 * while growing, n_images once joined), or 0 for NULL, ABI minor 44.
 */
int64_t rgsaddle_growing_string_n_images(const RgsaddleGrowingString *session);

/**
 * Nodes the last rgsaddle_growing_string_step grew on the reactant and
 * product fragments (0, 1, or 2), or 0 for NULL and before the first step,
 * ABI minor 44. The step report's rotations field is 0 for this session.
 */
int64_t rgsaddle_growing_string_nodes_added(
    const RgsaddleGrowingString *session);

/**
 * Phase of a growing string session (RGSADDLE_GROWING_STRING_GROWING or
 * RGSADDLE_GROWING_STRING_JOINED), or RGSADDLE_NULL_SESSION for NULL,
 * ABI minor 44.
 */
int32_t rgsaddle_growing_string_phase(const RgsaddleGrowingString *session);

/**
 * Copy the active positions (rgsaddle_growing_string_n_images(session) *
 * 3 * n_atoms values) into out, ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_positions(
    const RgsaddleGrowingString *session, double *out);

/**
 * Copy the stable image identifiers (rgsaddle_growing_string_n_images(session)
 * uint64_t values) into out, ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_ids(
    const RgsaddleGrowingString *session, uint64_t *out);

/**
 * Replace the per-atom fixed mask (n_atoms bytes, 1 fixed and 0 free) on
 * a growing string session, ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_set_fixed_atoms(
    RgsaddleGrowingString *session, const uint8_t *fixed, int64_t n_atoms);

/**
 * Read the saddle-index certificate of a joined growing string session,
 * ABI minor 44.
 */
rgsaddle_status_t rgsaddle_growing_string_saddle_index(
    const RgsaddleGrowingString *session, rgsaddle_saddle_index_t *out,
    double *curvatures, int64_t capacity);

/** Free a growing string session, ABI minor 44. */
void rgsaddle_growing_string_free(RgsaddleGrowingString *session);

/** Which terminal image moves on the target-energy level set, ABI minor 45. */
typedef enum {
  RGSADDLE_FREE_END_FIRST = 0,
  RGSADDLE_FREE_END_LAST = 1
} rgsaddle_free_end_side_t;

/**
 * Enable, replace, or clear the free-end adaptive NEB (FEA-NEB) of Zhang,
 * Zhang, Ye, and Zheng, J. Chem. Phys. 145, 094104 (2016),
 * doi:10.1063/1.4962019, on a live band, ABI minor 45. `enabled` 1
 * constrains the terminal image selected by `side` (`RGSADDLE_FREE_END_FIRST`
 * or `RGSADDLE_FREE_END_LAST`) to `E(x_end) = target_energy` (or the initial
 * energy of that terminal image when `has_target_energy` is 0) using rgmin's
 * generic solver level-set constraint, and cuts and remeshes around the
 * detected saddle up to `max_cuts` times; `enabled` 0 restores fixed
 * endpoints.
 */
rgsaddle_status_t rgsaddle_band_set_free_end(
    RgsaddleBand *band, int32_t enabled, int32_t side,
    int32_t has_target_energy, double target_energy, int64_t max_cuts,
    double cut_force, double energy_tol, int64_t stability,
    int64_t window_offset, int32_t cubic);

/**
 * Read the signed target-energy residual `E(x_end) - E_target` from the most
 * recent free-end step into `out`, ABI minor 45. Returns
 * RGSADDLE_INVALID_PARAMETER when `out` is NULL, or RGSADDLE_NO_EVALUATION
 * when free-end mode is off or before the first step.
 */
rgsaddle_status_t rgsaddle_band_free_end_residual(const RgsaddleBand *band,
                                                  double *out);

/**
 * Number of adaptive cut-and-remesh refinements performed so far on a
 * free-end band, or 0 for NULL, ABI minor 45.
 */
int64_t rgsaddle_band_free_end_cuts(const RgsaddleBand *band);

#define RGSADDLE_HAS_LINEB 1

/**
 * Fixed-energy LI-NEB controls for rgsaddle_lineb_solve, ABI minor 47.
 * Initialize with rgsaddle_lineb_config_default before changing fields.
 * Units are the caller's. opt_tol and perp_tol are absolute force norms
 * in mass-weighted coordinates.
 */
typedef struct {
  double opt_tol;
  double perp_tol;
  double spring;
  double endpoint_scale;
  double time_step;
  double max_move;
  double turning_width;
  double hbar;
  double boltzmann;
  int64_t quadrature_pairs;
} rgsaddle_lineb_config_t;

/**
 * Path, temperature and rate summary from rgsaddle_lineb_solve, ABI
 * minor 47. log_rate_over_reactant is NaN when beads was 0.
 */
typedef struct {
  double action;
  double half_period;
  double half_period_dynamics;
  double temperature;
  double temperature_dynamics;
  double log_rate_over_reactant;
  int64_t iterations;
  int64_t optimizer_calls;
  int64_t measure_calls;
  int64_t dynamics_calls;
  int64_t rate_calls;
} rgsaddle_lineb_result_t;

/**
 * Fill out with the solver defaults for the given force tolerances, ABI
 * minor 47. Returns RGSADDLE_INVALID_PARAMETER when out is NULL.
 */
rgsaddle_status_t rgsaddle_lineb_config_default(double opt_tol, double perp_tol,
                                                rgsaddle_lineb_config_t *out);

/**
 * Optimize a fixed-energy tunneling path and optionally its harmonic
 * quantum TST rate, ABI minor 47. Positions are contiguous image, atom,
 * xyz (n_images * n_atoms * 3). atom_masses holds n_atoms positive
 * masses: molecular 0 expands each mass to the three Cartesian
 * coordinates without rigid alignment; molecular 1 aligns every image
 * of the molecule to the first by a quaternion fit; molecular 2 aligns
 * each image to its neighbour. beads 0 returns the path only and writes NaN into
 * log_rate_over_reactant; beads must be 0 or at least 2. An unconverged
 * path returns RGSADDLE_STALLED after writing the last images into
 * optimized_positions. Reuses the host surface callback through
 * CSurface and LinebConfig validation.
 */
rgsaddle_status_t rgsaddle_lineb_solve(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out);

/** Solve with fixed endpoint coefficient κ. Pass NULL for the normalized
 * restraint. All other arguments follow rgsaddle_lineb_solve. ABI minor 47. */
rgsaddle_status_t rgsaddle_lineb_solve_with_kappa(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa);

#define RGSADDLE_LINEB_LINEAR_SEED (1u << 0)
#define RGSADDLE_LINEB_ENDPOINT_CONVERGENCE (1u << 1)
#define RGSADDLE_LINEB_PROJECTED_VERLET (1u << 2)
/* Values of the molecular argument of the LI-NEB entry points. */
#define RGSADDLE_LINEB_MASSES_COORDINATES 0
#define RGSADDLE_LINEB_MASSES_MOLECULE 1
#define RGSADDLE_LINEB_MASSES_MOLECULE_ADJACENT 2
/* Imaginary-time velocity Verlet on the cubic signed image force. */
#define RGSADDLE_LINEB_INTERPOLATED_FORCE_CLOCK (1u << 3)
/* Interpolate Cartesian image Hessians onto the equal-time replicas. */
#define RGSADDLE_LINEB_INTERPOLATED_HESSIAN (1u << 4)
/** Accept surface requests with multiple geometry rows in input order.
 * Both singleton and batched requests carry their row count in n_images.
 * LI-NEB call counts always count geometry rows. ABI minor 48. */
#define RGSADDLE_LINEB_BATCH_SURFACE (1u << 5)

/** Optimize with a choice of initial path, convergence test, and
 * integrator. Combine RGSADDLE_LINEB_LINEAR_SEED,
 * RGSADDLE_LINEB_ENDPOINT_CONVERGENCE, and
 * RGSADDLE_LINEB_PROJECTED_VERLET (ABI minor 47) in options, or pass
 * zero for the default behavior. ABI minor 47. */
rgsaddle_status_t rgsaddle_lineb_solve_options(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options);

/** Disjoint configuration evaluations for one successful solve. */
typedef struct {
  int64_t seed, optimizer, preparation, quadrature, dynamics, rate;
} rgsaddle_lineb_counts_t;

/** Same inputs as solve_options, with optional detailed counters. ABI minor 47. */
rgsaddle_status_t rgsaddle_lineb_solve_counted(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options,
                                       rgsaddle_lineb_counts_t *counts);


#define RGSADDLE_HAS_LINEB_RATE_DIAGNOSTICS 1

/** Counted solve with the number of negative and total retained fluctuation
 * modes. Each optional output points to a distinct int64_t. A path-only
 * solve or error sets the requested mode counts to -1. */
rgsaddle_status_t rgsaddle_lineb_solve_counted_diagnostics(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options,
                                       rgsaddle_lineb_counts_t *counts, int64_t *negative_modes, int64_t *modes);

#define RGSADDLE_HAS_LINEB_ORBIT 1

/** Opt-in observations from the rate's own surface calls. Each supplied buffer
 * has capacity doubles, at least beads * 3 * n_atoms. Positions are Cartesian
 * lengths and gradients are energy per length. Rows on the return leg repeat
 * the measured outbound coordinates and gradients. The maximum and RMS
 * discrete equation-of-motion residuals are in energy per length. Indices are
 * zero-based. With no rate, summary values are NaN and indices are -1. */
typedef struct {
    size_t capacity;
    double *positions;
    double *gradients;
    double max_abs_residual;
    double rms_residual;
    int64_t max_bead;
    int64_t max_coordinate;
} rgsaddle_lineb_orbit_t;

/** Counted solve with optional bead observations. Distinct non-null output
 * buffers are caller-owned. A null orbit uses the ordinary counted solve. */
rgsaddle_status_t rgsaddle_lineb_solve_counted_orbit(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options,
                                       rgsaddle_lineb_counts_t *counts, int64_t *negative_modes, int64_t *modes,
                                       rgsaddle_lineb_orbit_t *orbit);

/** Return-turn agreement for the interpolated-force clock, in the caller's units.
 * launch_image = -1 and the floating fields are NaN without a measured turn. */
typedef struct {
    rgsaddle_version_t version;
    uint64_t flags;
    int64_t launch_image;
    double path_arc_length;
    double return_arc;
    double max_cartesian_displacement;
    double energy_residual;
    double endpoint_energy_difference;
} rgsaddle_lineb_turning_t;

/** Counted orbit solve with optional return-turn geometry and energy checks. */
rgsaddle_status_t rgsaddle_lineb_solve_counted_turning(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options,
                                       rgsaddle_lineb_counts_t *counts, int64_t *negative_modes, int64_t *modes,
                                       rgsaddle_lineb_orbit_t *orbit, rgsaddle_lineb_turning_t *turning);

/** Geometry counts within the rate, and the Cartesian finite-difference
 * displacement used at the path images (NaN for direct or path-only solves). */
typedef struct {
    rgsaddle_version_t version;
    uint64_t flags;
    int64_t curvature_evaluations;
    int64_t bead_evaluations;
    double image_hessian_step;
} rgsaddle_lineb_rate_diagnostics_t;

/** Counted solve with selectable image-Hessian interpolation. A null step
 * uses the native displacement. Optional diagnostic and orbit output buffers
 * add no energy or gradient requests. */
rgsaddle_status_t rgsaddle_lineb_solve_counted_rate(const rgsaddle_lineb_config_t *cfg,
                                       int64_t n_images, int64_t n_atoms,
                                       const double *positions,
                                       const double *atom_masses,
                                       int32_t molecular, double energy,
                                       int64_t max_steps, int64_t beads,
                                       rgsaddle_surface_fn surface, void *user,
                                       double *optimized_positions,
                                       rgsaddle_lineb_result_t *out, const double *endpoint_kappa, uint32_t options,
                                       rgsaddle_lineb_counts_t *counts, int64_t *negative_modes, int64_t *modes,
                                       rgsaddle_lineb_orbit_t *orbit, rgsaddle_lineb_turning_t *turning,
                                       const double *image_hessian_step, rgsaddle_lineb_rate_diagnostics_t *rate_diagnostics);

/** Refine stepper of rgsaddle_band_set_refine, ABI minor 60. */
typedef enum {
  RGSADDLE_REFINE_NONE = 0,
  RGSADDLE_REFINE_LBFGS = 1
} rgsaddle_refine_method_t;

/**
 * Per-step refine stepper, ABI minor 60. `method` is an
 * rgsaddle_refine_method_t. RGSADDLE_REFINE_NONE clears the stepper and
 * does not read `threshold`. RGSADDLE_REFINE_LBFGS steps with L-BFGS
 * when the convergence force is at or under `threshold`, and with the
 * band method otherwise. `memory` is the number of L-BFGS pairs; a
 * value less than or equal to 0 selects 20. A `threshold` of 0 selects
 * the default 0.5, in the host's force units. A negative or non-finite
 * threshold, and an unknown method, are refused. A string, trust-region,
 * solid-state, quick-min, or free-end band keeps its own stepper and
 * ignores this one.
 */
rgsaddle_status_t rgsaddle_band_set_refine(RgsaddleBand *band, int32_t method,
                                           double threshold, int64_t memory);

/**
 * 1 when the last first-order step used the refine stepper, else 0.
 * 0 for NULL, before the first such step, and when no refine stepper
 * is installed. ABI minor 60.
 */
int32_t rgsaddle_band_is_refining(const RgsaddleBand *band);

/**
 * Redundant coordinate chart for geodesic interpolation.
 *
 * Zhu, Thompson, and Martinez, J. Chem. Phys. 150, 164103 (2019).
 * ABI minor 64.
 *
 * `x` holds `dof` Cartesians. The chart writes `n_coords` redundant
 * coordinates into `q` and the row-major Wilson B matrix (`n_coords`
 * rows, `dof` columns) into `b`. Return 0 on success and nonzero when
 * the chart is undefined at `x`.
 */
typedef int (*rgsaddle_geodesic_chart_fn)(void *ctx, const double *x,
                                          int64_t dof, double *q, double *b,
                                          int64_t n_coords);

/**
 * Geodesic interpolation controls. Lengths are in the path's unit.
 * `flags` must be 0. `lower_ratio` is in (0, 1] and `upper_ratio` is
 * at least 1. Counts are at least 1. `bound_samples` is even and at
 * least 2, so the equal partition of a segment includes its Cartesian
 * midpoint. `max_images` is at least 2. Both entry points refuse a cap
 * above their `n_images`, because the output buffer cannot grow; the
 * call returns `RGSADDLE_INVALID_PARAMETER` instead of clamping the cap.
 * `seed` seeds the ten noisy starts of
 * Algorithm 3 in `rgsaddle_geodesic_morse` and takes any value. ABI
 * minor 64.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double step_tol;
  double grad_tol;
  double max_move;
  double lower_ratio;
  double upper_ratio;
  int64_t max_sweeps;
  int64_t max_iterations;
  int64_t memory;
  int64_t bound_samples;
  int64_t max_images;
  uint64_t seed;
} rgsaddle_geodesic_config_t;

/**
 * Fill `out` with the default geodesic controls (step and gradient
 * tolerances 1e-6, max_move 0.5, lower_ratio 0.95, upper_ratio 1.1,
 * 40 sweeps, 40 iterations, memory 8, 10 bound samples, 32 images,
 * seed 0). ABI minor 64.
 */
rgsaddle_status_t rgsaddle_geodesic_config_default(
    rgsaddle_geodesic_config_t *out);

/**
 * Minimize the geodesic length of an existing path. ABI minor 64.
 *
 * `path` and `out` hold `n_images * dof` doubles, image-major, with
 * `n_images` at least 2. The image count is unchanged: `out` has no
 * room for inserted midpoints. `config->max_images` must be at most
 * `n_images`; a larger cap is `RGSADDLE_INVALID_PARAMETER`. The first
 * and last images are copied through.
 * `chart` writes `n_coords` redundant coordinates and their Wilson B
 * matrix. `cell` is a row-major 3x3 lattice, or NULL in free space.
 * With a cell, `dof` is a multiple of 3 and every midpoint, sample, and
 * inserted image follows the minimum image displacement of its two
 * segment ends; the chart applies the same convention to its own
 * distances. `config` NULL selects the defaults, including an image cap
 * of 32, so a NULL config requires `n_images` of at least 32. `length`,
 * `lower`, and
 * `upper` may be NULL; otherwise they receive the Eq. (17) length and
 * the Eq. (B5) and Eq. (B18) bounds. A nonzero chart return is
 * `RGSADDLE_SURFACE_FAILED` when it happens on the starting path.
 * When a round of sweeps uses all of `max_sweeps` with an image still
 * moving by more than `step_tol`, `out`, `length`, `lower`, and `upper`
 * hold the last iterate and the call returns `RGSADDLE_NOT_CONVERGED`.
 * `n_images * dof` and `n_coords * dof` must fit in memory sizes; an
 * overflow returns `RGSADDLE_INVALID_PARAMETER`.
 */
rgsaddle_status_t rgsaddle_geodesic_path(
    int64_t n_images, int64_t dof, const double *path, int64_t n_coords,
    rgsaddle_geodesic_chart_fn chart, void *ctx, const double *cell,
    const rgsaddle_geodesic_config_t *config, double *out, double *length,
    double *lower, double *upper);

/**
 * Morse pair geodesic between two geometries. ABI minor 64.
 *
 * `reactant`, `product`, and each row of `out` hold `3 * n_atoms`
 * doubles. `radii` holds one covalent radius per atom. `cell` is a
 * row-major 3x3 lattice, or NULL in free space. `n_images` counts the
 * endpoints and must be at least 2. `out` holds `n_images * 3 * n_atoms`
 * doubles. `config->max_images` must be at most `n_images`; a larger cap
 * is `RGSADDLE_INVALID_PARAMETER`. Alpha 1.7 and beta 0.01 are Eq. (7)
 * of Zhu, Thompson, and Martinez (2019). `config` NULL selects the
 * defaults, including an image cap of 32, so a NULL config requires
 * `n_images` of at least 32. The scalar
 * outputs follow `rgsaddle_geodesic_path`. `n_images * 3 * n_atoms` and
 * the pair count times `3 * n_atoms` must fit in memory sizes; an
 * overflow returns `RGSADDLE_INVALID_PARAMETER`.
 */
rgsaddle_status_t rgsaddle_geodesic_morse(
    int64_t n_images, int64_t n_atoms, const double *reactant,
    const double *product, const double *radii, const double *cell,
    const rgsaddle_geodesic_config_t *config, double *out, double *length,
    double *lower, double *upper);

/**
 * Mass-weighted finite-difference Vineyard prefactor controls, ABI minor 67.
 * `flags` and `reserved` must be 0. `difference` is `rgsaddle_difference_t`.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double step;
  int32_t difference;
  int32_t reserved;
  double positive_tol;
  double negative_tol;
} rgsaddle_vineyard_config_t;

/**
 * Mass-weighted Vineyard prefactor summary report, ABI minor 67.
 */
typedef struct {
  rgsaddle_version_t version;
  uint64_t flags;
  double prefactor_hz;
  double nu0_thz;
  double unstable_eigenvalue;
  int64_t n_min_modes;
  int64_t n_saddle_modes;
  int64_t evaluations;
} rgsaddle_vineyard_report_t;

/**
 * Look up standard atomic mass in amu by atomic number `z` in [1, 118], ABI minor 67.
 */
rgsaddle_status_t rgsaddle_atomic_mass(int32_t z, double *out);

/**
 * Look up standard atomic mass in amu by IUPAC element symbol, ABI minor 67.
 */
rgsaddle_status_t rgsaddle_atomic_mass_by_symbol(const char *symbol,
                                                 double *out);

/**
 * Fill `out` with default Vineyard finite-difference controls (step 1e-2,
 * central differences, positive_tol 1e-8, negative_tol 1e-4), ABI minor 67.
 */
rgsaddle_status_t rgsaddle_vineyard_config_default(
    rgsaddle_vineyard_config_t *out);

/**
 * Evaluate the harmonic Vineyard prefactor from dense finite-difference
 * Hessians on `surface` at `x_min` and `x_saddle`. `x_min` and `x_saddle`
 * hold `3 * n_atoms` doubles; `atom_masses` holds `n_atoms` positive masses
 * in amu. ABI minor 67.
 */
rgsaddle_status_t rgsaddle_vineyard_fd(
    const rgsaddle_vineyard_config_t *config, int64_t n_atoms,
    const double *x_min, const double *x_saddle, const double *atom_masses,
    rgsaddle_surface_fn surface, void *user,
    rgsaddle_vineyard_report_t *report);

#ifdef __cplusplus
}
#endif

#endif /* RGSADDLE_H */
