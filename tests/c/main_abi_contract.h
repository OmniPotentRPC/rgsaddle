/* The main ABI 1.5 layouts and signatures remain compatible at minor 1.15. */
#ifndef RGSADDLE_MAIN_ABI_CONTRACT_H
#define RGSADDLE_MAIN_ABI_CONTRACT_H
#include "rgsaddle.h"
#include <stddef.h>

_Static_assert(RGSADDLE_ABI_MAJOR == 1u, "main ABI major");
_Static_assert(RGSADDLE_ABI_MINOR == 46u, "ABI minor");
_Static_assert(RGSADDLE_BAND_NO_MIN_MODE == (1ull << 3), "band flag bit 3");
_Static_assert(RGSADDLE_BAND_SPLINE_REPARAM == (1ull << 4), "band flag bit 4");
_Static_assert(RGSADDLE_BAND_STRING == (1ull << 5), "band flag bit 5");
_Static_assert(RGSADDLE_REPARAM_LINEAR == 0, "rgsaddle_reparametrization_t LINEAR");
_Static_assert(RGSADDLE_REPARAM_NOT_A_KNOT == 1, "rgsaddle_reparametrization_t NOT_A_KNOT");
_Static_assert(RGSADDLE_REQ_ONE_IMAGE == (1ull << 0), "request flag bit 0");
_Static_assert(RGSADDLE_REQ_CLIMBING_IMAGE == (1ull << 1), "request flag bit 1");
_Static_assert(RGSADDLE_NOT_EQUILIBRATED == -14, "status NOT_EQUILIBRATED");
_Static_assert(RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES == -15,
               "status INSUFFICIENT_EFFECTIVE_SAMPLES");

typedef struct {
  uint32_t major;
  uint32_t minor;
} abi15_rgsaddle_version_t;
_Static_assert(sizeof(rgsaddle_version_t) == sizeof(abi15_rgsaddle_version_t), "rgsaddle_version_t size");
_Static_assert(_Alignof(rgsaddle_version_t) == _Alignof(abi15_rgsaddle_version_t), "rgsaddle_version_t alignment");
_Static_assert(offsetof(rgsaddle_version_t, major) == offsetof(abi15_rgsaddle_version_t, major), "rgsaddle_version_t.major");
_Static_assert(offsetof(rgsaddle_version_t, minor) == offsetof(abi15_rgsaddle_version_t, minor), "rgsaddle_version_t.minor");

typedef struct {
  abi15_rgsaddle_version_t version;
  uint64_t flags;
  int64_t n_images;
  int64_t n_atoms;
  const double *positions;
  double *energies;
  double *gradients;
  int64_t image;
} abi15_rgsaddle_surface_request_t;
_Static_assert(sizeof(rgsaddle_surface_request_t) == sizeof(abi15_rgsaddle_surface_request_t), "rgsaddle_surface_request_t size");
_Static_assert(_Alignof(rgsaddle_surface_request_t) == _Alignof(abi15_rgsaddle_surface_request_t), "rgsaddle_surface_request_t alignment");
_Static_assert(offsetof(rgsaddle_surface_request_t, version) == offsetof(abi15_rgsaddle_surface_request_t, version), "rgsaddle_surface_request_t.version");
_Static_assert(offsetof(rgsaddle_surface_request_t, flags) == offsetof(abi15_rgsaddle_surface_request_t, flags), "rgsaddle_surface_request_t.flags");
_Static_assert(offsetof(rgsaddle_surface_request_t, n_images) == offsetof(abi15_rgsaddle_surface_request_t, n_images), "rgsaddle_surface_request_t.n_images");
_Static_assert(offsetof(rgsaddle_surface_request_t, n_atoms) == offsetof(abi15_rgsaddle_surface_request_t, n_atoms), "rgsaddle_surface_request_t.n_atoms");
_Static_assert(offsetof(rgsaddle_surface_request_t, positions) == offsetof(abi15_rgsaddle_surface_request_t, positions), "rgsaddle_surface_request_t.positions");
_Static_assert(offsetof(rgsaddle_surface_request_t, energies) == offsetof(abi15_rgsaddle_surface_request_t, energies), "rgsaddle_surface_request_t.energies");
_Static_assert(offsetof(rgsaddle_surface_request_t, gradients) == offsetof(abi15_rgsaddle_surface_request_t, gradients), "rgsaddle_surface_request_t.gradients");
_Static_assert(offsetof(rgsaddle_surface_request_t, image) == offsetof(abi15_rgsaddle_surface_request_t, image), "rgsaddle_surface_request_t.image");

typedef struct {
  abi15_rgsaddle_version_t version;
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
} abi15_rgsaddle_band_config_t;
_Static_assert(sizeof(rgsaddle_band_config_t) == sizeof(abi15_rgsaddle_band_config_t), "rgsaddle_band_config_t size");
_Static_assert(_Alignof(rgsaddle_band_config_t) == _Alignof(abi15_rgsaddle_band_config_t), "rgsaddle_band_config_t alignment");
_Static_assert(offsetof(rgsaddle_band_config_t, version) == offsetof(abi15_rgsaddle_band_config_t, version), "rgsaddle_band_config_t.version");
_Static_assert(offsetof(rgsaddle_band_config_t, flags) == offsetof(abi15_rgsaddle_band_config_t, flags), "rgsaddle_band_config_t.flags");
_Static_assert(offsetof(rgsaddle_band_config_t, tangent) == offsetof(abi15_rgsaddle_band_config_t, tangent), "rgsaddle_band_config_t.tangent");
_Static_assert(offsetof(rgsaddle_band_config_t, spring) == offsetof(abi15_rgsaddle_band_config_t, spring), "rgsaddle_band_config_t.spring");
_Static_assert(offsetof(rgsaddle_band_config_t, projection) == offsetof(abi15_rgsaddle_band_config_t, projection), "rgsaddle_band_config_t.projection");
_Static_assert(offsetof(rgsaddle_band_config_t, method) == offsetof(abi15_rgsaddle_band_config_t, method), "rgsaddle_band_config_t.method");
_Static_assert(offsetof(rgsaddle_band_config_t, spring_k) == offsetof(abi15_rgsaddle_band_config_t, spring_k), "rgsaddle_band_config_t.spring_k");
_Static_assert(offsetof(rgsaddle_band_config_t, spring_ks) == offsetof(abi15_rgsaddle_band_config_t, spring_ks), "rgsaddle_band_config_t.spring_ks");
_Static_assert(offsetof(rgsaddle_band_config_t, ci_trigger_factor) == offsetof(abi15_rgsaddle_band_config_t, ci_trigger_factor), "rgsaddle_band_config_t.ci_trigger_factor");
_Static_assert(offsetof(rgsaddle_band_config_t, ci_trigger_force) == offsetof(abi15_rgsaddle_band_config_t, ci_trigger_force), "rgsaddle_band_config_t.ci_trigger_force");
_Static_assert(offsetof(rgsaddle_band_config_t, cell) == offsetof(abi15_rgsaddle_band_config_t, cell), "rgsaddle_band_config_t.cell");
_Static_assert(offsetof(rgsaddle_band_config_t, force_tol) == offsetof(abi15_rgsaddle_band_config_t, force_tol), "rgsaddle_band_config_t.force_tol");
_Static_assert(offsetof(rgsaddle_band_config_t, max_move) == offsetof(abi15_rgsaddle_band_config_t, max_move), "rgsaddle_band_config_t.max_move");
_Static_assert(offsetof(rgsaddle_band_config_t, memory) == offsetof(abi15_rgsaddle_band_config_t, memory), "rgsaddle_band_config_t.memory");

typedef struct {
  abi15_rgsaddle_version_t version;
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
} abi15_rgsaddle_report_t;
_Static_assert(sizeof(rgsaddle_report_t) == sizeof(abi15_rgsaddle_report_t), "rgsaddle_report_t size");
_Static_assert(_Alignof(rgsaddle_report_t) == _Alignof(abi15_rgsaddle_report_t), "rgsaddle_report_t alignment");
_Static_assert(offsetof(rgsaddle_report_t, version) == offsetof(abi15_rgsaddle_report_t, version), "rgsaddle_report_t.version");
_Static_assert(offsetof(rgsaddle_report_t, flags) == offsetof(abi15_rgsaddle_report_t, flags), "rgsaddle_report_t.flags");
_Static_assert(offsetof(rgsaddle_report_t, status) == offsetof(abi15_rgsaddle_report_t, status), "rgsaddle_report_t.status");
_Static_assert(offsetof(rgsaddle_report_t, evaluations) == offsetof(abi15_rgsaddle_report_t, evaluations), "rgsaddle_report_t.evaluations");
_Static_assert(offsetof(rgsaddle_report_t, max_force) == offsetof(abi15_rgsaddle_report_t, max_force), "rgsaddle_report_t.max_force");
_Static_assert(offsetof(rgsaddle_report_t, ci_index) == offsetof(abi15_rgsaddle_report_t, ci_index), "rgsaddle_report_t.ci_index");
_Static_assert(offsetof(rgsaddle_report_t, iteration) == offsetof(abi15_rgsaddle_report_t, iteration), "rgsaddle_report_t.iteration");
_Static_assert(offsetof(rgsaddle_report_t, curvature) == offsetof(abi15_rgsaddle_report_t, curvature), "rgsaddle_report_t.curvature");
_Static_assert(offsetof(rgsaddle_report_t, rotations) == offsetof(abi15_rgsaddle_report_t, rotations), "rgsaddle_report_t.rotations");

typedef struct {
  abi15_rgsaddle_version_t version;
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
} abi15_rgsaddle_minmode_config_t;
_Static_assert(sizeof(rgsaddle_minmode_config_t) == sizeof(abi15_rgsaddle_minmode_config_t), "rgsaddle_minmode_config_t size");
_Static_assert(_Alignof(rgsaddle_minmode_config_t) == _Alignof(abi15_rgsaddle_minmode_config_t), "rgsaddle_minmode_config_t alignment");
_Static_assert(offsetof(rgsaddle_minmode_config_t, version) == offsetof(abi15_rgsaddle_minmode_config_t, version), "rgsaddle_minmode_config_t.version");
_Static_assert(offsetof(rgsaddle_minmode_config_t, flags) == offsetof(abi15_rgsaddle_minmode_config_t, flags), "rgsaddle_minmode_config_t.flags");
_Static_assert(offsetof(rgsaddle_minmode_config_t, kind) == offsetof(abi15_rgsaddle_minmode_config_t, kind), "rgsaddle_minmode_config_t.kind");
_Static_assert(offsetof(rgsaddle_minmode_config_t, method) == offsetof(abi15_rgsaddle_minmode_config_t, method), "rgsaddle_minmode_config_t.method");
_Static_assert(offsetof(rgsaddle_minmode_config_t, dr) == offsetof(abi15_rgsaddle_minmode_config_t, dr), "rgsaddle_minmode_config_t.dr");
_Static_assert(offsetof(rgsaddle_minmode_config_t, rotation_tol) == offsetof(abi15_rgsaddle_minmode_config_t, rotation_tol), "rgsaddle_minmode_config_t.rotation_tol");
_Static_assert(offsetof(rgsaddle_minmode_config_t, max_rotations) == offsetof(abi15_rgsaddle_minmode_config_t, max_rotations), "rgsaddle_minmode_config_t.max_rotations");
_Static_assert(offsetof(rgsaddle_minmode_config_t, krylov_dim) == offsetof(abi15_rgsaddle_minmode_config_t, krylov_dim), "rgsaddle_minmode_config_t.krylov_dim");
_Static_assert(offsetof(rgsaddle_minmode_config_t, force_tol) == offsetof(abi15_rgsaddle_minmode_config_t, force_tol), "rgsaddle_minmode_config_t.force_tol");
_Static_assert(offsetof(rgsaddle_minmode_config_t, max_move) == offsetof(abi15_rgsaddle_minmode_config_t, max_move), "rgsaddle_minmode_config_t.max_move");
_Static_assert(offsetof(rgsaddle_minmode_config_t, rotation_angle_tol) == offsetof(abi15_rgsaddle_minmode_config_t, rotation_angle_tol), "rgsaddle_minmode_config_t.rotation_angle_tol");
_Static_assert(offsetof(rgsaddle_minmode_config_t, difference) == offsetof(abi15_rgsaddle_minmode_config_t, difference), "rgsaddle_minmode_config_t.difference");
_Static_assert(offsetof(rgsaddle_minmode_config_t, reserved) == offsetof(abi15_rgsaddle_minmode_config_t, reserved), "rgsaddle_minmode_config_t.reserved");

typedef struct {
  abi15_rgsaddle_version_t version;
  uint64_t flags;
  int64_t lanczos_start;
  double eigenvalue_threshold;
  double basin_push;
  int64_t basin_relaxations;
  int64_t activation_relaxations;
  double relaxation_step;
  int64_t lanczos_actions;
  double curvature_floor;
} abi43_rgsaddle_artn_config_t;
_Static_assert(sizeof(rgsaddle_artn_config_t) == sizeof(abi43_rgsaddle_artn_config_t), "rgsaddle_artn_config_t size");
_Static_assert(_Alignof(rgsaddle_artn_config_t) == _Alignof(abi43_rgsaddle_artn_config_t), "rgsaddle_artn_config_t alignment");
_Static_assert(offsetof(rgsaddle_artn_config_t, version) == offsetof(abi43_rgsaddle_artn_config_t, version), "rgsaddle_artn_config_t.version");
_Static_assert(offsetof(rgsaddle_artn_config_t, flags) == offsetof(abi43_rgsaddle_artn_config_t, flags), "rgsaddle_artn_config_t.flags");
_Static_assert(offsetof(rgsaddle_artn_config_t, lanczos_start) == offsetof(abi43_rgsaddle_artn_config_t, lanczos_start), "rgsaddle_artn_config_t.lanczos_start");
_Static_assert(offsetof(rgsaddle_artn_config_t, eigenvalue_threshold) == offsetof(abi43_rgsaddle_artn_config_t, eigenvalue_threshold), "rgsaddle_artn_config_t.eigenvalue_threshold");
_Static_assert(offsetof(rgsaddle_artn_config_t, basin_push) == offsetof(abi43_rgsaddle_artn_config_t, basin_push), "rgsaddle_artn_config_t.basin_push");
_Static_assert(offsetof(rgsaddle_artn_config_t, basin_relaxations) == offsetof(abi43_rgsaddle_artn_config_t, basin_relaxations), "rgsaddle_artn_config_t.basin_relaxations");
_Static_assert(offsetof(rgsaddle_artn_config_t, activation_relaxations) == offsetof(abi43_rgsaddle_artn_config_t, activation_relaxations), "rgsaddle_artn_config_t.activation_relaxations");
_Static_assert(offsetof(rgsaddle_artn_config_t, relaxation_step) == offsetof(abi43_rgsaddle_artn_config_t, relaxation_step), "rgsaddle_artn_config_t.relaxation_step");
_Static_assert(offsetof(rgsaddle_artn_config_t, lanczos_actions) == offsetof(abi43_rgsaddle_artn_config_t, lanczos_actions), "rgsaddle_artn_config_t.lanczos_actions");
_Static_assert(offsetof(rgsaddle_artn_config_t, curvature_floor) == offsetof(abi43_rgsaddle_artn_config_t, curvature_floor), "rgsaddle_artn_config_t.curvature_floor");

typedef struct {
  abi15_rgsaddle_version_t version;
  uint64_t flags;
  int32_t update; /**< rgsaddle_hess_update_t */
  int32_t mode;   /**< rgsaddle_nichols_mode_t */
  /** Max-abs cap on the Cartesian step. */
  double trust_radius;
  double force_tol;
  /** Central-difference step when `hessian` is NULL. */
  double fd_dr;
} abi15_rgsaddle_index1_config_t;
_Static_assert(sizeof(rgsaddle_index1_config_t) == sizeof(abi15_rgsaddle_index1_config_t), "rgsaddle_index1_config_t size");
_Static_assert(_Alignof(rgsaddle_index1_config_t) == _Alignof(abi15_rgsaddle_index1_config_t), "rgsaddle_index1_config_t alignment");
_Static_assert(offsetof(rgsaddle_index1_config_t, version) == offsetof(abi15_rgsaddle_index1_config_t, version), "rgsaddle_index1_config_t.version");
_Static_assert(offsetof(rgsaddle_index1_config_t, flags) == offsetof(abi15_rgsaddle_index1_config_t, flags), "rgsaddle_index1_config_t.flags");
_Static_assert(offsetof(rgsaddle_index1_config_t, update) == offsetof(abi15_rgsaddle_index1_config_t, update), "rgsaddle_index1_config_t.update");
_Static_assert(offsetof(rgsaddle_index1_config_t, mode) == offsetof(abi15_rgsaddle_index1_config_t, mode), "rgsaddle_index1_config_t.mode");
_Static_assert(offsetof(rgsaddle_index1_config_t, trust_radius) == offsetof(abi15_rgsaddle_index1_config_t, trust_radius), "rgsaddle_index1_config_t.trust_radius");
_Static_assert(offsetof(rgsaddle_index1_config_t, force_tol) == offsetof(abi15_rgsaddle_index1_config_t, force_tol), "rgsaddle_index1_config_t.force_tol");
_Static_assert(offsetof(rgsaddle_index1_config_t, fd_dr) == offsetof(abi15_rgsaddle_index1_config_t, fd_dr), "rgsaddle_index1_config_t.fd_dr");

typedef int (*abi15_signature_rgsaddle_abi_version)(void);
_Static_assert(_Generic(&rgsaddle_abi_version, abi15_signature_rgsaddle_abi_version: 1, default: 0), "rgsaddle_abi_version signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_abi_stamp)(rgsaddle_version_t *out);
_Static_assert(_Generic(&rgsaddle_abi_stamp, abi15_signature_rgsaddle_abi_stamp: 1, default: 0), "rgsaddle_abi_stamp signature");

typedef const char * (*abi15_signature_rgsaddle_status_name)(rgsaddle_status_t status);
_Static_assert(_Generic(&rgsaddle_status_name, abi15_signature_rgsaddle_status_name: 1, default: 0), "rgsaddle_status_name signature");

typedef RgsaddleBand * (*abi15_signature_rgsaddle_band_create)(const rgsaddle_band_config_t *config,
                                   int64_t n_images, int64_t n_atoms,
                                   const double *positions);
_Static_assert(_Generic(&rgsaddle_band_create, abi15_signature_rgsaddle_band_create: 1, default: 0), "rgsaddle_band_create signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_step)(RgsaddleBand *band,
                                     rgsaddle_surface_fn surface, void *user,
                                     rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_band_step, abi15_signature_rgsaddle_band_step: 1, default: 0), "rgsaddle_band_step signature");
_Static_assert(_Generic(&xts_band_step, abi15_signature_rgsaddle_band_step: 1, default: 0), "xts_band_step signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_positions)(const RgsaddleBand *band, double *out);
_Static_assert(_Generic(&rgsaddle_band_positions, abi15_signature_rgsaddle_band_positions: 1, default: 0), "rgsaddle_band_positions signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_set_positions)(RgsaddleBand *band,
                                              const double *positions);
_Static_assert(_Generic(&rgsaddle_band_set_positions, abi15_signature_rgsaddle_band_set_positions: 1, default: 0), "rgsaddle_band_set_positions signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_reset)(RgsaddleBand *band);
_Static_assert(_Generic(&rgsaddle_band_reset, abi15_signature_rgsaddle_band_reset: 1, default: 0), "rgsaddle_band_reset signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_restart)(RgsaddleBand *band);
_Static_assert(_Generic(&rgsaddle_band_restart, abi15_signature_rgsaddle_band_restart: 1, default: 0), "rgsaddle_band_restart signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_band_evaluation)(const RgsaddleBand *band,
                                           double *energies, double *gradients,
                                           double *projected);
_Static_assert(_Generic(&rgsaddle_band_evaluation, abi15_signature_rgsaddle_band_evaluation: 1, default: 0), "rgsaddle_band_evaluation signature");

typedef void (*abi15_signature_rgsaddle_band_free)(RgsaddleBand *band);
_Static_assert(_Generic(&rgsaddle_band_free, abi15_signature_rgsaddle_band_free: 1, default: 0), "rgsaddle_band_free signature");

typedef RgsaddleMinMode * (*abi15_signature_rgsaddle_minmode_create)(
    const rgsaddle_minmode_config_t *config, int64_t n_atoms,
    const double *position, const double *mode);
_Static_assert(_Generic(&rgsaddle_minmode_create, abi15_signature_rgsaddle_minmode_create: 1, default: 0), "rgsaddle_minmode_create signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_step)(RgsaddleMinMode *session,
                                        rgsaddle_surface_fn surface, void *user,
                                        rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_minmode_step, abi15_signature_rgsaddle_minmode_step: 1, default: 0), "rgsaddle_minmode_step signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_estimate)(RgsaddleMinMode *session,
                                            rgsaddle_surface_fn surface,
                                            void *user, rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_minmode_estimate, abi15_signature_rgsaddle_minmode_estimate: 1, default: 0), "rgsaddle_minmode_estimate signature");
_Static_assert(_Generic(&xts_minmode_estimate, abi15_signature_rgsaddle_minmode_estimate: 1, default: 0), "xts_minmode_estimate signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_set_position)(RgsaddleMinMode *session,
                                                const double *position,
                                                const double *gradient);
_Static_assert(_Generic(&rgsaddle_minmode_set_position, abi15_signature_rgsaddle_minmode_set_position: 1, default: 0), "rgsaddle_minmode_set_position signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_set_mode)(RgsaddleMinMode *session,
                                            const double *mode);
_Static_assert(_Generic(&rgsaddle_minmode_set_mode, abi15_signature_rgsaddle_minmode_set_mode: 1, default: 0), "rgsaddle_minmode_set_mode signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_position)(const RgsaddleMinMode *session,
                                                double *out);
_Static_assert(_Generic(&rgsaddle_minmode_position, abi15_signature_rgsaddle_minmode_position: 1, default: 0), "rgsaddle_minmode_position signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_mode)(const RgsaddleMinMode *session,
                                        double *out);
_Static_assert(_Generic(&rgsaddle_minmode_mode, abi15_signature_rgsaddle_minmode_mode: 1, default: 0), "rgsaddle_minmode_mode signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_minmode_reset)(RgsaddleMinMode *session);
_Static_assert(_Generic(&rgsaddle_minmode_reset, abi15_signature_rgsaddle_minmode_reset: 1, default: 0), "rgsaddle_minmode_reset signature");

typedef void (*abi15_signature_rgsaddle_minmode_free)(RgsaddleMinMode *session);
_Static_assert(_Generic(&rgsaddle_minmode_free, abi15_signature_rgsaddle_minmode_free: 1, default: 0), "rgsaddle_minmode_free signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_nichols_step)(int64_t n, int64_t nmode,
                                        const double *gradient,
                                        const double *evals, const double *evecs,
                                        const double *masses, double trust_radius,
                                        int32_t mode, double *displacement);
_Static_assert(_Generic(&rgsaddle_nichols_step, abi15_signature_rgsaddle_nichols_step: 1, default: 0), "rgsaddle_nichols_step signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_prfo_step)(int64_t n, const double *gradient,
                                     const double *hessian, const double *masses,
                                     double trust_radius, int32_t mode,
                                     double *displacement);
_Static_assert(_Generic(&rgsaddle_prfo_step, abi15_signature_rgsaddle_prfo_step: 1, default: 0), "rgsaddle_prfo_step signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_hessian_powell)(int64_t n, const double *step,
                                          const double *dgradient, double *hessian);
_Static_assert(_Generic(&rgsaddle_hessian_powell, abi15_signature_rgsaddle_hessian_powell: 1, default: 0), "rgsaddle_hessian_powell signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_hessian_bofill)(int64_t n, const double *step,
                                          const double *dgradient, double *hessian);
_Static_assert(_Generic(&rgsaddle_hessian_bofill, abi15_signature_rgsaddle_hessian_bofill: 1, default: 0), "rgsaddle_hessian_bofill signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_cap_max_abs)(int64_t n, double *step, double trust_radius);
_Static_assert(_Generic(&rgsaddle_cap_max_abs, abi15_signature_rgsaddle_cap_max_abs: 1, default: 0), "rgsaddle_cap_max_abs signature");

typedef RgsaddleIndex1 * (*abi15_signature_rgsaddle_index1_create)(const rgsaddle_index1_config_t *config,
                                       int64_t n_atoms, const double *position,
                                       const double *hessian, const double *masses);
_Static_assert(_Generic(&rgsaddle_index1_create, abi15_signature_rgsaddle_index1_create: 1, default: 0), "rgsaddle_index1_create signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_index1_step)(RgsaddleIndex1 *session,
                                       rgsaddle_surface_fn surface, void *user,
                                       rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_index1_step, abi15_signature_rgsaddle_index1_step: 1, default: 0), "rgsaddle_index1_step signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_index1_position)(const RgsaddleIndex1 *session, double *out);
_Static_assert(_Generic(&rgsaddle_index1_position, abi15_signature_rgsaddle_index1_position: 1, default: 0), "rgsaddle_index1_position signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_index1_hessian)(const RgsaddleIndex1 *session, double *out);
_Static_assert(_Generic(&rgsaddle_index1_hessian, abi15_signature_rgsaddle_index1_hessian: 1, default: 0), "rgsaddle_index1_hessian signature");

typedef rgsaddle_status_t (*abi15_signature_rgsaddle_index1_reset)(RgsaddleIndex1 *session);
_Static_assert(_Generic(&rgsaddle_index1_reset, abi15_signature_rgsaddle_index1_reset: 1, default: 0), "rgsaddle_index1_reset signature");

typedef void (*abi15_signature_rgsaddle_index1_free)(RgsaddleIndex1 *session);
_Static_assert(_Generic(&rgsaddle_index1_free, abi15_signature_rgsaddle_index1_free: 1, default: 0), "rgsaddle_index1_free signature");

typedef rgsaddle_status_t (*abi18_signature_rgsaddle_connect)(
    const rgsaddle_connect_config_t *config, int64_t n_atoms, const double *left,
    const double *right, rgsaddle_surface_fn surface, void *user, double *position,
    rgsaddle_connect_report_t *report);
_Static_assert(_Generic(&rgsaddle_connect, abi18_signature_rgsaddle_connect: 1, default: 0), "rgsaddle_connect signature");

typedef rgsaddle_status_t (*abi19_signature_rgsaddle_idpp_path)(int64_t n_images, int64_t n_atoms,
                                     const double *reactant, const double *product,
                                     const double *cell, int64_t max_iterations,
                                     double grad_tol, double max_move, double *out);
_Static_assert(_Generic(&rgsaddle_idpp_path, abi19_signature_rgsaddle_idpp_path: 1, default: 0), "rgsaddle_idpp_path signature");

typedef RgsaddleOcineb *(*abi20_signature_rgsaddle_ocineb_create)(
    const rgsaddle_band_config_t *band, const rgsaddle_minmode_config_t *minmode,
    const rgsaddle_ocineb_config_t *config, int64_t n_images, int64_t n_atoms,
    const double *positions);
_Static_assert(_Generic(&rgsaddle_ocineb_create, abi20_signature_rgsaddle_ocineb_create: 1, default: 0), "rgsaddle_ocineb_create signature");

typedef rgsaddle_status_t (*abi20_signature_rgsaddle_ocineb_step)(
    RgsaddleOcineb *session, rgsaddle_surface_fn surface, void *user,
    rgsaddle_ocineb_report_t *out);
_Static_assert(_Generic(&rgsaddle_ocineb_step, abi20_signature_rgsaddle_ocineb_step: 1, default: 0), "rgsaddle_ocineb_step signature");

typedef rgsaddle_status_t (*abi20_signature_rgsaddle_ocineb_positions)(
    const RgsaddleOcineb *session, double *out);
_Static_assert(_Generic(&rgsaddle_ocineb_positions, abi20_signature_rgsaddle_ocineb_positions: 1, default: 0), "rgsaddle_ocineb_positions signature");

typedef rgsaddle_status_t (*abi20_signature_rgsaddle_ocineb_reset)(RgsaddleOcineb *session);
_Static_assert(_Generic(&rgsaddle_ocineb_reset, abi20_signature_rgsaddle_ocineb_reset: 1, default: 0), "rgsaddle_ocineb_reset signature");

typedef void (*abi20_signature_rgsaddle_ocineb_free)(RgsaddleOcineb *session);
_Static_assert(_Generic(&rgsaddle_ocineb_free, abi20_signature_rgsaddle_ocineb_free: 1, default: 0), "rgsaddle_ocineb_free signature");

typedef rgsaddle_status_t (*abi24_signature_rgsaddle_sella_saddle_set_symmetrized_ritz)(
    RgsaddleSellaSaddle *session, int32_t enabled);
_Static_assert(_Generic(&rgsaddle_sella_saddle_set_symmetrized_ritz, abi24_signature_rgsaddle_sella_saddle_set_symmetrized_ritz: 1, default: 0), "rgsaddle_sella_saddle_set_symmetrized_ritz signature");

typedef rgsaddle_status_t (*abi25_signature_rgsaddle_sella_saddle_set_small_system_atom_ceiling)(
    RgsaddleSellaSaddle *session, int32_t enabled);
_Static_assert(_Generic(&rgsaddle_sella_saddle_set_small_system_atom_ceiling, abi25_signature_rgsaddle_sella_saddle_set_small_system_atom_ceiling: 1, default: 0), "rgsaddle_sella_saddle_set_small_system_atom_ceiling signature");
_Static_assert(RGSADDLE_IRC_GS2 == 0, "rgsaddle_irc_kind_t GS2");
_Static_assert(RGSADDLE_IRC_MOROKUMA == 1, "rgsaddle_irc_kind_t MOROKUMA");
_Static_assert(RGSADDLE_IRC_SELLA_SPHERE == 2, "rgsaddle_irc_kind_t SELLA_SPHERE (minor 35)");

typedef rgsaddle_status_t (*abi32_signature_rgsaddle_minmode_set_torque_window)(
    RgsaddleMinMode *session, int32_t window);
_Static_assert(_Generic(&rgsaddle_minmode_set_torque_window, abi32_signature_rgsaddle_minmode_set_torque_window: 1, default: 0), "rgsaddle_minmode_set_torque_window signature");

typedef rgsaddle_status_t (*abi38_signature_rgsaddle_minmode_set_rotation_stop_length)(
    RgsaddleMinMode *session, double length);
_Static_assert(_Generic(&rgsaddle_minmode_set_rotation_stop_length, abi38_signature_rgsaddle_minmode_set_rotation_stop_length: 1, default: 0), "rgsaddle_minmode_set_rotation_stop_length signature");

typedef rgsaddle_status_t (*abi43_signature_rgsaddle_artn_config_default)(
    rgsaddle_artn_config_t *config);
_Static_assert(_Generic(&rgsaddle_artn_config_default, abi43_signature_rgsaddle_artn_config_default: 1, default: 0), "rgsaddle_artn_config_default signature");

typedef rgsaddle_status_t (*abi43_signature_rgsaddle_minmode_set_artn)(
    RgsaddleMinMode *session, const rgsaddle_artn_config_t *config);
_Static_assert(_Generic(&rgsaddle_minmode_set_artn, abi43_signature_rgsaddle_minmode_set_artn: 1, default: 0), "rgsaddle_minmode_set_artn signature");

typedef rgsaddle_status_t (*abi38_signature_rgsaddle_reparametrize)(
    int64_t n_images, int64_t n_atoms, double *positions, int64_t anchor, const double *cell,
    int32_t scheme);
_Static_assert(_Generic(&rgsaddle_reparametrize, abi38_signature_rgsaddle_reparametrize: 1, default: 0), "rgsaddle_reparametrize signature");

typedef rgsaddle_status_t (*abi39_signature_rgsaddle_band_set_fixed_atoms)(
    RgsaddleBand *band, const uint8_t *mask, int64_t n_atoms);
_Static_assert(_Generic(&rgsaddle_band_set_fixed_atoms, abi39_signature_rgsaddle_band_set_fixed_atoms: 1, default: 0), "rgsaddle_band_set_fixed_atoms signature");

typedef rgsaddle_status_t (*abi39_signature_rgsaddle_band_project_fixed)(
    const rgsaddle_band_config_t *config, int64_t n_images, int64_t n_atoms,
    const double *positions, const double *energies, const double *gradients,
    const uint8_t *fixed, double *forces, int64_t *ci_index);
_Static_assert(_Generic(&rgsaddle_band_project_fixed, abi39_signature_rgsaddle_band_project_fixed: 1, default: 0), "rgsaddle_band_project_fixed signature");

typedef RgsaddleIndexK *(*abi35_signature_rgsaddle_indexk_create)(
    const rgsaddle_indexk_config_t *config, int64_t n_atoms, const double *position,
    const double *modes);
_Static_assert(_Generic(&rgsaddle_indexk_create, abi35_signature_rgsaddle_indexk_create: 1, default: 0), "rgsaddle_indexk_create signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_indexk_step)(
    RgsaddleIndexK *session, rgsaddle_surface_fn surface, rgsaddle_hessian_fn hessian, void *user,
    rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_indexk_step, abi35_signature_rgsaddle_indexk_step: 1, default: 0), "rgsaddle_indexk_step signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_indexk_position)(const RgsaddleIndexK *session,
                                                                     double *out);
_Static_assert(_Generic(&rgsaddle_indexk_position, abi35_signature_rgsaddle_indexk_position: 1, default: 0), "rgsaddle_indexk_position signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_indexk_modes)(const RgsaddleIndexK *session,
                                                                  double *modes, double *curvatures);
_Static_assert(_Generic(&rgsaddle_indexk_modes, abi35_signature_rgsaddle_indexk_modes: 1, default: 0), "rgsaddle_indexk_modes signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_indexk_translated)(const RgsaddleIndexK *session,
                                                                       double *out);
_Static_assert(_Generic(&rgsaddle_indexk_translated, abi35_signature_rgsaddle_indexk_translated: 1, default: 0), "rgsaddle_indexk_translated signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_indexk_set_position)(
    RgsaddleIndexK *session, const double *position, const double *gradient);
_Static_assert(_Generic(&rgsaddle_indexk_set_position, abi35_signature_rgsaddle_indexk_set_position: 1, default: 0), "rgsaddle_indexk_set_position signature");

typedef void (*abi35_signature_rgsaddle_indexk_free)(RgsaddleIndexK *session);
_Static_assert(_Generic(&rgsaddle_indexk_free, abi35_signature_rgsaddle_indexk_free: 1, default: 0), "rgsaddle_indexk_free signature");

typedef rgsaddle_status_t (*abi35_signature_rgsaddle_reflect_force)(
    int64_t n, int64_t k, const double *force, const double *modes, double *out);
_Static_assert(_Generic(&rgsaddle_reflect_force, abi35_signature_rgsaddle_reflect_force: 1, default: 0), "rgsaddle_reflect_force signature");

typedef struct {
  abi15_rgsaddle_version_t version;
  uint64_t flags;
  int32_t block;
  int32_t lowest;
  int32_t max_actions;
  int32_t difference;
  double dr;
  double residual_tol;
  double negative_tol;
  uint64_t seed;
  int32_t escape_attempts;
  int32_t reserved;
  double escape_push;
} abi40_rgsaddle_saddle_check_t;
_Static_assert(sizeof(rgsaddle_saddle_check_t) == 80, "rgsaddle_saddle_check_t size");
_Static_assert(sizeof(rgsaddle_saddle_check_t) == sizeof(abi40_rgsaddle_saddle_check_t), "rgsaddle_saddle_check_t size");
_Static_assert(_Alignof(rgsaddle_saddle_check_t) == _Alignof(abi40_rgsaddle_saddle_check_t), "rgsaddle_saddle_check_t alignment");
_Static_assert(offsetof(rgsaddle_saddle_check_t, block) == offsetof(abi40_rgsaddle_saddle_check_t, block), "rgsaddle_saddle_check_t.block");
_Static_assert(offsetof(rgsaddle_saddle_check_t, lowest) == offsetof(abi40_rgsaddle_saddle_check_t, lowest), "rgsaddle_saddle_check_t.lowest");
_Static_assert(offsetof(rgsaddle_saddle_check_t, max_actions) == offsetof(abi40_rgsaddle_saddle_check_t, max_actions), "rgsaddle_saddle_check_t.max_actions");
_Static_assert(offsetof(rgsaddle_saddle_check_t, difference) == offsetof(abi40_rgsaddle_saddle_check_t, difference), "rgsaddle_saddle_check_t.difference");
_Static_assert(offsetof(rgsaddle_saddle_check_t, dr) == offsetof(abi40_rgsaddle_saddle_check_t, dr), "rgsaddle_saddle_check_t.dr");
_Static_assert(offsetof(rgsaddle_saddle_check_t, residual_tol) == offsetof(abi40_rgsaddle_saddle_check_t, residual_tol), "rgsaddle_saddle_check_t.residual_tol");
_Static_assert(offsetof(rgsaddle_saddle_check_t, negative_tol) == offsetof(abi40_rgsaddle_saddle_check_t, negative_tol), "rgsaddle_saddle_check_t.negative_tol");
_Static_assert(offsetof(rgsaddle_saddle_check_t, seed) == offsetof(abi40_rgsaddle_saddle_check_t, seed), "rgsaddle_saddle_check_t.seed");
_Static_assert(offsetof(rgsaddle_saddle_check_t, escape_attempts) == offsetof(abi40_rgsaddle_saddle_check_t, escape_attempts), "rgsaddle_saddle_check_t.escape_attempts");
_Static_assert(offsetof(rgsaddle_saddle_check_t, escape_push) == offsetof(abi40_rgsaddle_saddle_check_t, escape_push), "rgsaddle_saddle_check_t.escape_push");

typedef struct {
  abi15_rgsaddle_version_t version;
  uint64_t flags;
  int32_t certified;
  int32_t index;
  int32_t converged;
  int32_t n_curvatures;
  int64_t evaluations;
  int64_t total_evaluations;
  int64_t escapes;
} abi40_rgsaddle_saddle_index_t;
_Static_assert(sizeof(rgsaddle_saddle_index_t) == 56, "rgsaddle_saddle_index_t size");
_Static_assert(sizeof(rgsaddle_saddle_index_t) == sizeof(abi40_rgsaddle_saddle_index_t), "rgsaddle_saddle_index_t size");
_Static_assert(_Alignof(rgsaddle_saddle_index_t) == _Alignof(abi40_rgsaddle_saddle_index_t), "rgsaddle_saddle_index_t alignment");
_Static_assert(offsetof(rgsaddle_saddle_index_t, certified) == offsetof(abi40_rgsaddle_saddle_index_t, certified), "rgsaddle_saddle_index_t.certified");
_Static_assert(offsetof(rgsaddle_saddle_index_t, index) == offsetof(abi40_rgsaddle_saddle_index_t, index), "rgsaddle_saddle_index_t.index");
_Static_assert(offsetof(rgsaddle_saddle_index_t, converged) == offsetof(abi40_rgsaddle_saddle_index_t, converged), "rgsaddle_saddle_index_t.converged");
_Static_assert(offsetof(rgsaddle_saddle_index_t, n_curvatures) == offsetof(abi40_rgsaddle_saddle_index_t, n_curvatures), "rgsaddle_saddle_index_t.n_curvatures");
_Static_assert(offsetof(rgsaddle_saddle_index_t, evaluations) == offsetof(abi40_rgsaddle_saddle_index_t, evaluations), "rgsaddle_saddle_index_t.evaluations");
_Static_assert(offsetof(rgsaddle_saddle_index_t, total_evaluations) == offsetof(abi40_rgsaddle_saddle_index_t, total_evaluations), "rgsaddle_saddle_index_t.total_evaluations");
_Static_assert(offsetof(rgsaddle_saddle_index_t, escapes) == offsetof(abi40_rgsaddle_saddle_index_t, escapes), "rgsaddle_saddle_index_t.escapes");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_saddle_check_default)(
    rgsaddle_saddle_check_t *out);
_Static_assert(_Generic(&rgsaddle_saddle_check_default, abi40_signature_rgsaddle_saddle_check_default: 1, default: 0), "rgsaddle_saddle_check_default signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_band_set_saddle_check)(
    RgsaddleBand *band, const rgsaddle_saddle_check_t *check);
_Static_assert(_Generic(&rgsaddle_band_set_saddle_check, abi40_signature_rgsaddle_band_set_saddle_check: 1, default: 0), "rgsaddle_band_set_saddle_check signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_band_saddle_index)(
    const RgsaddleBand *band, rgsaddle_saddle_index_t *out, double *curvatures, int64_t capacity);
_Static_assert(_Generic(&rgsaddle_band_saddle_index, abi40_signature_rgsaddle_band_saddle_index: 1, default: 0), "rgsaddle_band_saddle_index signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_minmode_set_saddle_check)(
    RgsaddleMinMode *session, const rgsaddle_saddle_check_t *check);
_Static_assert(_Generic(&rgsaddle_minmode_set_saddle_check, abi40_signature_rgsaddle_minmode_set_saddle_check: 1, default: 0), "rgsaddle_minmode_set_saddle_check signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_minmode_saddle_index)(
    const RgsaddleMinMode *session, rgsaddle_saddle_index_t *out, double *curvatures,
    int64_t capacity);
_Static_assert(_Generic(&rgsaddle_minmode_saddle_index, abi40_signature_rgsaddle_minmode_saddle_index: 1, default: 0), "rgsaddle_minmode_saddle_index signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_ocineb_set_saddle_check)(
    RgsaddleOcineb *session, const rgsaddle_saddle_check_t *check);
_Static_assert(_Generic(&rgsaddle_ocineb_set_saddle_check, abi40_signature_rgsaddle_ocineb_set_saddle_check: 1, default: 0), "rgsaddle_ocineb_set_saddle_check signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_ocineb_saddle_index)(
    const RgsaddleOcineb *session, rgsaddle_saddle_index_t *out, double *curvatures,
    int64_t capacity);
_Static_assert(_Generic(&rgsaddle_ocineb_saddle_index, abi40_signature_rgsaddle_ocineb_saddle_index: 1, default: 0), "rgsaddle_ocineb_saddle_index signature");

typedef rgsaddle_status_t (*abi40_signature_rgsaddle_saddle_index)(
    const rgsaddle_saddle_check_t *check, int64_t n_atoms, const double *position,
    const double *gradient, const double *mode, const uint8_t *fixed, rgsaddle_surface_fn surface,
    void *user, rgsaddle_saddle_index_t *out, double *curvatures, int64_t capacity);
_Static_assert(_Generic(&rgsaddle_saddle_index, abi40_signature_rgsaddle_saddle_index: 1, default: 0), "rgsaddle_saddle_index signature");

typedef rgsaddle_status_t (*abi44_signature_rgsaddle_band_set_string)(
    RgsaddleBand *band, int32_t enabled, int32_t scheme);
_Static_assert(_Generic(&rgsaddle_band_set_string, abi44_signature_rgsaddle_band_set_string: 1, default: 0), "rgsaddle_band_set_string signature");

typedef rgsaddle_status_t (*abi44_signature_rgsaddle_growing_string_config_default)(
    rgsaddle_growing_string_config_t *out);
_Static_assert(_Generic(&rgsaddle_growing_string_config_default, abi44_signature_rgsaddle_growing_string_config_default: 1, default: 0), "rgsaddle_growing_string_config_default signature");

typedef RgsaddleGrowingString *(*abi44_signature_rgsaddle_growing_string_create)(
    const rgsaddle_growing_string_config_t *config, int64_t n_atoms, const double *reactant,
    const double *product);
_Static_assert(_Generic(&rgsaddle_growing_string_create, abi44_signature_rgsaddle_growing_string_create: 1, default: 0), "rgsaddle_growing_string_create signature");

typedef rgsaddle_status_t (*abi44_signature_rgsaddle_growing_string_step)(
    RgsaddleGrowingString *session, rgsaddle_surface_fn surface, void *user,
    rgsaddle_report_t *out);
_Static_assert(_Generic(&rgsaddle_growing_string_step, abi44_signature_rgsaddle_growing_string_step: 1, default: 0), "rgsaddle_growing_string_step signature");

typedef int64_t (*abi44_signature_rgsaddle_growing_string_nodes_added)(
    const RgsaddleGrowingString *session);
_Static_assert(_Generic(&rgsaddle_growing_string_nodes_added, abi44_signature_rgsaddle_growing_string_nodes_added: 1, default: 0), "rgsaddle_growing_string_nodes_added signature");

typedef rgsaddle_status_t (*abi45_signature_rgsaddle_band_set_free_end)(
    RgsaddleBand *band, int32_t enabled, int32_t side, int32_t has_target_energy,
    double target_energy, int64_t max_cuts, double cut_force, double energy_tol, int64_t stability,
    int64_t window_offset, int32_t cubic);
_Static_assert(_Generic(&rgsaddle_band_set_free_end, abi45_signature_rgsaddle_band_set_free_end: 1, default: 0), "rgsaddle_band_set_free_end signature");

typedef rgsaddle_status_t (*abi45_signature_rgsaddle_band_free_end_residual)(
    const RgsaddleBand *band, double *out);
_Static_assert(_Generic(&rgsaddle_band_free_end_residual, abi45_signature_rgsaddle_band_free_end_residual: 1, default: 0), "rgsaddle_band_free_end_residual signature");

typedef int64_t (*abi45_signature_rgsaddle_band_free_end_cuts)(
    const RgsaddleBand *band);
_Static_assert(_Generic(&rgsaddle_band_free_end_cuts, abi45_signature_rgsaddle_band_free_end_cuts: 1, default: 0), "rgsaddle_band_free_end_cuts signature");

typedef rgsaddle_status_t (*abi45_signature_rgsaddle_band_set_dynamic)(
    RgsaddleBand *band, int32_t enabled, double fmax, double scale_fmax);
_Static_assert(_Generic(&rgsaddle_band_set_dynamic, abi45_signature_rgsaddle_band_set_dynamic: 1, default: 0), "rgsaddle_band_set_dynamic signature");

typedef rgsaddle_status_t (*abi45_signature_rgsaddle_dyneb_mask)(
    int64_t n_images, int64_t n_atoms, const double *positions, const double *energies,
    double *forces, double fmax, double scale_fmax, uint8_t *frozen);
_Static_assert(_Generic(&rgsaddle_dyneb_mask, abi45_signature_rgsaddle_dyneb_mask: 1, default: 0), "rgsaddle_dyneb_mask signature");

#endif
