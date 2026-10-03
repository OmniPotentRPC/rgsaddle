/* The main ABI 1.5 layouts and signatures remain compatible at minor 1.14. */
#ifndef RGSADDLE_MAIN_ABI_CONTRACT_H
#define RGSADDLE_MAIN_ABI_CONTRACT_H
#include "rgsaddle.h"
#include <stddef.h>

_Static_assert(RGSADDLE_ABI_MAJOR == 1u, "main ABI major");
_Static_assert(RGSADDLE_ABI_MINOR == 14u, "consolidated main ABI minor");

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

#endif
