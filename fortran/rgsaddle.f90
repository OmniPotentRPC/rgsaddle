module rgsaddle
  use, intrinsic :: iso_c_binding
  implicit none
  private

  integer(c_int), parameter, public :: RGSADDLE_ABI_MAJOR = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_ABI_MINOR = 45_c_int
  integer(c_int), parameter, public :: RGSADDLE_OK = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_INVALID_PARAMETER = -11_c_int
  integer(c_int), parameter, public :: RGSADDLE_STATUS_RUNNING = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_STATUS_CONVERGED = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_METHOD_FIRE = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_METHOD_LBFGS = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_MINMODE_DIMER = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_MINMODE_LANCZOS = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_OHTST_INTEGRATOR_BAOAB = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_OHTST_INTEGRATOR_GHMC = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_OHTST_COUNT_FINAL = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_OHTST_COUNT_FIRST_CROSSING = 1_c_int
  integer(c_int64_t), parameter :: RGSADDLE_REQ_ONE_IMAGE = 1_c_int64_t

  type, bind(C), public :: rgsaddle_version_t
    integer(c_int32_t) :: major
    integer(c_int32_t) :: minor
  end type rgsaddle_version_t

  type, bind(C), public :: rgsaddle_surface_request_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    integer(c_int64_t) :: n_images
    integer(c_int64_t) :: n_atoms
    type(c_ptr) :: positions
    type(c_ptr) :: energies
    type(c_ptr) :: gradients
    integer(c_int64_t) :: image
  end type rgsaddle_surface_request_t

  type, bind(C), public :: rgsaddle_band_config_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    integer(c_int32_t) :: tangent
    integer(c_int32_t) :: spring
    integer(c_int32_t) :: projection
    integer(c_int32_t) :: method
    real(c_double) :: spring_k
    type(c_ptr) :: spring_ks
    real(c_double) :: ci_trigger_factor
    real(c_double) :: ci_trigger_force
    type(c_ptr) :: cell
    real(c_double) :: force_tol
    real(c_double) :: max_move
    integer(c_int64_t) :: memory
  end type rgsaddle_band_config_t

  type, bind(C), public :: rgsaddle_minmode_config_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    integer(c_int32_t) :: kind
    integer(c_int32_t) :: method
    real(c_double) :: dr
    real(c_double) :: rotation_tol
    integer(c_int64_t) :: max_rotations
    integer(c_int64_t) :: krylov_dim
    real(c_double) :: force_tol
    real(c_double) :: max_move
    real(c_double) :: rotation_angle_tol
    integer(c_int32_t) :: difference
    integer(c_int32_t) :: reserved
  end type rgsaddle_minmode_config_t

  type, bind(C), public :: rgsaddle_report_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    integer(c_int32_t) :: status
    integer(c_int32_t) :: evaluations
    real(c_double) :: max_force
    integer(c_int64_t) :: ci_index
    integer(c_int64_t) :: iteration
    real(c_double) :: curvature
    integer(c_int64_t) :: rotations
  end type rgsaddle_report_t

  type, bind(C), public :: rgsaddle_pafi_config_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    real(c_double) :: thermal
    real(c_double) :: mobility
    real(c_double) :: time_step
    integer(c_int64_t) :: thermalize
    integer(c_int64_t) :: stride
    integer(c_int64_t) :: samples
    integer(c_int64_t) :: blocks
    integer(c_int64_t) :: seed
    integer(c_int32_t) :: fix_com
    integer(c_int64_t) :: retain
    real(c_double) :: target_error
    integer(c_int64_t) :: max_rounds
  end type rgsaddle_pafi_config_t

  type, bind(C), public :: rgsaddle_pafi_result_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    real(c_double) :: barrier
    real(c_double) :: barrier_error
    integer(c_int64_t) :: dividing
    integer(c_int64_t) :: samples
    integer(c_int64_t) :: evaluations
  end type rgsaddle_pafi_result_t

  type, bind(C), public :: rgsaddle_ohtst_config_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    real(c_double) :: thermal
    real(c_double) :: sample_time_step
    real(c_double) :: sample_friction
    integer(c_int64_t) :: sample_burn_in
    integer(c_int64_t) :: sample_stride
    integer(c_int64_t) :: sample_records
    integer(c_int64_t) :: sample_blocks
    integer(c_int64_t) :: sample_seed
    integer(c_int64_t) :: sample_retain
    integer(c_int32_t) :: sample_integrator
    real(c_double) :: shooting_friction
    real(c_double) :: shooting_time_step
    integer(c_int64_t) :: shooting_steps
    integer(c_int64_t) :: shooting_check_every
    integer(c_int64_t) :: shooting_commit
    integer(c_int64_t) :: shooting_shots
    integer(c_int64_t) :: shooting_blocks
    integer(c_int64_t) :: shooting_seed
    integer(c_int32_t) :: counting
  end type rgsaddle_ohtst_config_t

  type, bind(C), public :: rgsaddle_ohtst_result_t
    type(rgsaddle_version_t) :: version
    integer(c_int64_t) :: flags
    real(c_double) :: kappa
    real(c_double) :: error
    integer(c_int64_t) :: shots
    integer(c_int64_t) :: unassigned
    integer(c_int64_t) :: reactive
    integer(c_int64_t) :: recrossed
    integer(c_int64_t) :: evaluations
  end type rgsaddle_ohtst_result_t

  abstract interface
    integer(c_int) function surface_proc(user, n_images, n_atoms, positions, energies, gradients, image) result(status)
      import :: c_ptr, c_int, c_int64_t, c_double
      type(c_ptr), value :: user
      integer(c_int64_t), value :: n_images
      integer(c_int64_t), value :: n_atoms
      real(c_double), intent(in) :: positions(*)
      real(c_double), intent(inout) :: energies(*)
      real(c_double), intent(inout) :: gradients(*)
      integer(c_int64_t), value :: image
    end function surface_proc
  end interface

  abstract interface
    integer(c_int) function point_surface_proc(user, n_dof, position, energy, gradient) result(status)
      import :: c_ptr, c_int, c_int64_t, c_double
      type(c_ptr), value :: user
      integer(c_int64_t), value :: n_dof
      real(c_double), intent(in) :: position(*)
      real(c_double), intent(inout) :: energy
      real(c_double), intent(inout) :: gradient(*)
    end function point_surface_proc
  end interface

  type, public :: band_handle
    private
    type(c_ptr) :: raw = c_null_ptr
  contains
    final :: band_finalize
    procedure, public :: create => band_create
    procedure, public :: step => band_step
    procedure, public :: step_c => band_step_c
    procedure, public :: positions => band_positions
  end type band_handle

  type, public :: minmode_handle
    private
    type(c_ptr) :: raw = c_null_ptr
  contains
    final :: minmode_finalize
    procedure, public :: create => minmode_create
    procedure, public :: step => minmode_step
    procedure, public :: step_c => minmode_step_c
    procedure, public :: position => minmode_position
    procedure, public :: mode => minmode_mode
  end type minmode_handle

  type, public :: pafi_handle
    private
    type(c_ptr) :: raw = c_null_ptr
  contains
    final :: pafi_finalize
    procedure, public :: create => pafi_create
    procedure, public :: set_config => pafi_set_config
    procedure, public :: run => pafi_run
    procedure, public :: run_c => pafi_run_c
    procedure, public :: result => pafi_result
    procedure, public :: profile => pafi_profile
  end type pafi_handle

  type, public :: ohtst_handle
    private
    type(c_ptr) :: raw = c_null_ptr
  contains
    final :: ohtst_finalize
    procedure, public :: create => ohtst_create
    procedure, public :: set_config => ohtst_set_config
    procedure, public :: run => ohtst_run
    procedure, public :: run_c => ohtst_run_c
    procedure, public :: result => ohtst_result
    procedure, public :: curve => ohtst_curve
  end type ohtst_handle

  procedure(surface_proc), pointer, save :: active_surface => null()
  procedure(point_surface_proc), pointer, save :: active_point_surface => null()
  type(c_ptr), save :: active_user = c_null_ptr

  public :: init_band_config, init_minmode_config, init_pafi_config, init_ohtst_config, &
    abi_stamp, status_name, surface_proc, point_surface_proc

  interface
    function c_rgsaddle_abi_stamp(out) bind(C, name="rgsaddle_abi_stamp") result(status)
      import :: c_int, rgsaddle_version_t
      type(rgsaddle_version_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_abi_stamp

    function c_rgsaddle_band_create(config, n_images, n_atoms, positions) bind(C, name="rgsaddle_band_create") result(handle)
      import :: c_ptr, c_int64_t, rgsaddle_band_config_t, c_double
      type(rgsaddle_band_config_t), intent(in) :: config
      integer(c_int64_t), value :: n_images
      integer(c_int64_t), value :: n_atoms
      real(c_double), intent(in) :: positions(*)
      type(c_ptr) :: handle
    end function c_rgsaddle_band_create

    function c_rgsaddle_band_step(band, surface, user, out) bind(C, name="rgsaddle_band_step") result(status)
      import :: c_ptr, c_funptr, c_int, rgsaddle_report_t
      type(c_ptr), value :: band
      type(c_funptr), value :: surface
      type(c_ptr), value :: user
      type(rgsaddle_report_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_band_step

    function c_rgsaddle_band_positions(band, out) bind(C, name="rgsaddle_band_positions") result(status)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: band
      real(c_double), intent(out) :: out(*)
      integer(c_int) :: status
    end function c_rgsaddle_band_positions

    subroutine c_rgsaddle_band_free(band) bind(C, name="rgsaddle_band_free")
      import :: c_ptr
      type(c_ptr), value :: band
    end subroutine c_rgsaddle_band_free

    function c_rgsaddle_minmode_create(config, n_atoms, position, mode) bind(C, name="rgsaddle_minmode_create") result(handle)
      import :: c_ptr, c_int64_t, rgsaddle_minmode_config_t, c_double
      type(rgsaddle_minmode_config_t), intent(in) :: config
      integer(c_int64_t), value :: n_atoms
      real(c_double), intent(in) :: position(*)
      real(c_double), intent(in) :: mode(*)
      type(c_ptr) :: handle
    end function c_rgsaddle_minmode_create

    function c_rgsaddle_minmode_step(session, surface, user, out) bind(C, name="rgsaddle_minmode_step") result(status)
      import :: c_ptr, c_funptr, c_int, rgsaddle_report_t
      type(c_ptr), value :: session
      type(c_funptr), value :: surface
      type(c_ptr), value :: user
      type(rgsaddle_report_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_minmode_step

    function c_rgsaddle_minmode_position(session, out) bind(C, name="rgsaddle_minmode_position") result(status)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: session
      real(c_double), intent(out) :: out(*)
      integer(c_int) :: status
    end function c_rgsaddle_minmode_position

    function c_rgsaddle_minmode_mode(session, out) bind(C, name="rgsaddle_minmode_mode") result(status)
      import :: c_ptr, c_int, c_double
      type(c_ptr), value :: session
      real(c_double), intent(out) :: out(*)
      integer(c_int) :: status
    end function c_rgsaddle_minmode_mode

    subroutine c_rgsaddle_minmode_free(session) bind(C, name="rgsaddle_minmode_free")
      import :: c_ptr
      type(c_ptr), value :: session
    end subroutine c_rgsaddle_minmode_free

    function c_rgsaddle_pafi_create(config, n_frames, n_dof, frames, arcs, n_arcs) &
        bind(C, name="rgsaddle_pafi_create") result(handle)
      import :: c_ptr, c_int64_t, c_double, rgsaddle_pafi_config_t
      type(rgsaddle_pafi_config_t), intent(in) :: config
      integer(c_int64_t), value :: n_frames
      integer(c_int64_t), value :: n_dof
      real(c_double), intent(in) :: frames(*)
      type(c_ptr), value :: arcs
      integer(c_int64_t), value :: n_arcs
      type(c_ptr) :: handle
    end function c_rgsaddle_pafi_create

    function c_rgsaddle_pafi_set_config(session, config) bind(C, name="rgsaddle_pafi_set_config") result(status)
      import :: c_ptr, c_int, rgsaddle_pafi_config_t
      type(c_ptr), value :: session
      type(rgsaddle_pafi_config_t), intent(in) :: config
      integer(c_int) :: status
    end function c_rgsaddle_pafi_set_config

    function c_rgsaddle_pafi_run(session, surface, user) bind(C, name="rgsaddle_pafi_run") result(status)
      import :: c_ptr, c_funptr, c_int
      type(c_ptr), value :: session
      type(c_funptr), value :: surface
      type(c_ptr), value :: user
      integer(c_int) :: status
    end function c_rgsaddle_pafi_run

    function c_rgsaddle_pafi_result(session, out) bind(C, name="rgsaddle_pafi_result") result(status)
      import :: c_ptr, c_int, rgsaddle_pafi_result_t
      type(c_ptr), value :: session
      type(rgsaddle_pafi_result_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_pafi_result

    function c_rgsaddle_pafi_profile(session, arcs, gradients, gradient_errors, &
        free_energy, free_energy_errors, capacity, written) &
        bind(C, name="rgsaddle_pafi_profile") result(status)
      import :: c_ptr, c_double, c_int64_t, c_int
      type(c_ptr), value :: session
      real(c_double), intent(out) :: arcs(*)
      real(c_double), intent(out) :: gradients(*)
      real(c_double), intent(out) :: gradient_errors(*)
      real(c_double), intent(out) :: free_energy(*)
      real(c_double), intent(out) :: free_energy_errors(*)
      integer(c_int64_t), value :: capacity
      integer(c_int64_t), intent(out) :: written
      integer(c_int) :: status
    end function c_rgsaddle_pafi_profile

    subroutine c_rgsaddle_pafi_free(session) bind(C, name="rgsaddle_pafi_free")
      import :: c_ptr
      type(c_ptr), value :: session
    end subroutine c_rgsaddle_pafi_free

    function c_rgsaddle_ohtst_create(config, n_dof, point, normal) bind(C, name="rgsaddle_ohtst_create") result(handle)
      import :: c_ptr, c_int64_t, c_double, rgsaddle_ohtst_config_t
      type(rgsaddle_ohtst_config_t), intent(in) :: config
      integer(c_int64_t), value :: n_dof
      real(c_double), intent(in) :: point(*)
      real(c_double), intent(in) :: normal(*)
      type(c_ptr) :: handle
    end function c_rgsaddle_ohtst_create

    function c_rgsaddle_ohtst_set_config(session, config) bind(C, name="rgsaddle_ohtst_set_config") result(status)
      import :: c_ptr, c_int, rgsaddle_ohtst_config_t
      type(c_ptr), value :: session
      type(rgsaddle_ohtst_config_t), intent(in) :: config
      integer(c_int) :: status
    end function c_rgsaddle_ohtst_set_config

    function c_rgsaddle_ohtst_run(session, surface, user) bind(C, name="rgsaddle_ohtst_run") result(status)
      import :: c_ptr, c_funptr, c_int
      type(c_ptr), value :: session
      type(c_funptr), value :: surface
      type(c_ptr), value :: user
      integer(c_int) :: status
    end function c_rgsaddle_ohtst_run

    function c_rgsaddle_ohtst_result(session, out) bind(C, name="rgsaddle_ohtst_result") result(status)
      import :: c_ptr, c_int, rgsaddle_ohtst_result_t
      type(c_ptr), value :: session
      type(rgsaddle_ohtst_result_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_ohtst_result

    function c_rgsaddle_ohtst_curve(session, times, kappa_of_t, capacity, written) &
        bind(C, name="rgsaddle_ohtst_curve") result(status)
      import :: c_ptr, c_int, c_double, c_int64_t
      type(c_ptr), value :: session
      real(c_double), intent(out) :: times(*)
      real(c_double), intent(out) :: kappa_of_t(*)
      integer(c_int64_t), value :: capacity
      integer(c_int64_t), intent(out) :: written
      integer(c_int) :: status
    end function c_rgsaddle_ohtst_curve

    subroutine c_rgsaddle_ohtst_free(session) bind(C, name="rgsaddle_ohtst_free")
      import :: c_ptr
      type(c_ptr), value :: session
    end subroutine c_rgsaddle_ohtst_free
  end interface

contains

  function abi_stamp(out) result(status)
    type(rgsaddle_version_t), intent(out) :: out
    integer(c_int) :: status
    status = c_rgsaddle_abi_stamp(out)
  end function abi_stamp

  function status_name(status) result(name)
    integer(c_int), value :: status
    character(len=:), allocatable :: name
    select case (status)
    case (RGSADDLE_OK)
      name = "RGSADDLE_OK"
    case (RGSADDLE_INVALID_PARAMETER)
      name = "RGSADDLE_INVALID_PARAMETER"
    case default
      name = "RGSADDLE_STATUS"
    end select
  end function status_name

  subroutine init_band_config(config)
    type(rgsaddle_band_config_t), intent(out) :: config
    integer(c_int) :: status
    config = rgsaddle_band_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      1_c_int32_t, 0_c_int32_t, 1_c_int32_t, RGSADDLE_METHOD_FIRE, 5.0_c_double, c_null_ptr, &
      0.5_c_double, 0.0_c_double, c_null_ptr, sqrt(epsilon(1.0_c_double)), 0.1_c_double, 10_c_int64_t)
    status = abi_stamp(config%version)
  end subroutine init_band_config

  subroutine init_minmode_config(config, kind)
    type(rgsaddle_minmode_config_t), intent(out) :: config
    integer(c_int32_t), intent(in), optional :: kind
    integer(c_int) :: status
    integer(c_int32_t) :: mode_kind
    mode_kind = RGSADDLE_MINMODE_DIMER
    if (present(kind)) mode_kind = kind
    config = rgsaddle_minmode_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      mode_kind, RGSADDLE_METHOD_FIRE, sqrt(epsilon(1.0_c_double)), sqrt(epsilon(1.0_c_double)), 40_c_int64_t, &
      20_c_int64_t, sqrt(epsilon(1.0_c_double)), 0.1_c_double, 0.0_c_double, 0_c_int32_t, 0_c_int32_t)
    status = abi_stamp(config%version)
  end subroutine init_minmode_config

  subroutine init_pafi_config(config)
    type(rgsaddle_pafi_config_t), intent(out) :: config
    integer(c_int) :: status
    config = rgsaddle_pafi_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      0.02585_c_double, 1.0_c_double, 1.0e-3_c_double, 200_c_int64_t, 5_c_int64_t, 200_c_int64_t, &
      5_c_int64_t, 1_c_int64_t, 1_c_int32_t, 16_c_int64_t, 0.0_c_double, 4_c_int64_t)
    status = abi_stamp(config%version)
  end subroutine init_pafi_config

  subroutine init_ohtst_config(config)
    type(rgsaddle_ohtst_config_t), intent(out) :: config
    integer(c_int) :: status
    config = rgsaddle_ohtst_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      0.02585_c_double, 0.1_c_double, 1.0_c_double, 300_c_int64_t, 5_c_int64_t, 1000_c_int64_t, &
      16_c_int64_t, 1_c_int64_t, 256_c_int64_t, int(RGSADDLE_OHTST_INTEGRATOR_BAOAB, c_int32_t), &
      1.0_c_double, 0.05_c_double, 400_c_int64_t, 40_c_int64_t, 0_c_int64_t, 1024_c_int64_t, &
      16_c_int64_t, 2_c_int64_t, int(RGSADDLE_OHTST_COUNT_FINAL, c_int32_t))
    status = abi_stamp(config%version)
  end subroutine init_ohtst_config

  integer(c_int) function band_create(self, config, n_images, n_atoms, positions) result(status)
    class(band_handle), intent(inout) :: self
    type(rgsaddle_band_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    self%raw = c_rgsaddle_band_create(config, n_images, n_atoms, positions)
    if (c_associated(self%raw)) then
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function band_create

  integer(c_int) function band_step(self, callback, user, out) result(status)
    class(band_handle), intent(inout) :: self
    procedure(surface_proc), pointer, intent(in) :: callback
    type(c_ptr), value, intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    if (present(user)) then
      active_user = user
    else
      active_user = c_null_ptr
    end if
    active_surface => callback
    status = c_rgsaddle_band_step(self%raw, c_funloc(fortran_surface_trampoline), active_user, report)
    if (present(out)) out = report
    active_surface => null()
  end function band_step

  integer(c_int) function band_step_c(self, surface, user, out) result(status)
    class(band_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), value, intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_band_step(self%raw, surface, user_ptr, report)
    if (present(out)) out = report
  end function band_step_c

  integer(c_int) function band_positions(self, positions) result(status)
    class(band_handle), intent(in) :: self
    real(c_double), intent(out) :: positions(*)
    status = c_rgsaddle_band_positions(self%raw, positions)
  end function band_positions

  integer(c_int) function minmode_create(self, config, n_atoms, position, mode) result(status)
    class(minmode_handle), intent(inout) :: self
    type(rgsaddle_minmode_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(in) :: mode(*)
    self%raw = c_rgsaddle_minmode_create(config, n_atoms, position, mode)
    if (c_associated(self%raw)) then
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function minmode_create

  integer(c_int) function minmode_step(self, callback, user, out) result(status)
    class(minmode_handle), intent(inout) :: self
    procedure(surface_proc), pointer, intent(in) :: callback
    type(c_ptr), value, intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    if (present(user)) then
      active_user = user
    else
      active_user = c_null_ptr
    end if
    active_surface => callback
    status = c_rgsaddle_minmode_step(self%raw, c_funloc(fortran_surface_trampoline), active_user, report)
    if (present(out)) out = report
    active_surface => null()
  end function minmode_step

  integer(c_int) function minmode_step_c(self, surface, user, out) result(status)
    class(minmode_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), value, intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_minmode_step(self%raw, surface, user_ptr, report)
    if (present(out)) out = report
  end function minmode_step_c

  integer(c_int) function minmode_position(self, position) result(status)
    class(minmode_handle), intent(in) :: self
    real(c_double), intent(out) :: position(*)
    status = c_rgsaddle_minmode_position(self%raw, position)
  end function minmode_position

  integer(c_int) function minmode_mode(self, mode) result(status)
    class(minmode_handle), intent(in) :: self
    real(c_double), intent(out) :: mode(*)
    status = c_rgsaddle_minmode_mode(self%raw, mode)
  end function minmode_mode

  integer(c_int) function pafi_create(self, config, n_frames, n_dof, frames, arcs) result(status)
    class(pafi_handle), intent(inout) :: self
    type(rgsaddle_pafi_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_frames
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: frames(*)
    real(c_double), intent(in), target :: arcs(:)
    type(c_ptr) :: arcs_ptr
    integer(c_int64_t) :: n_arcs
    arcs_ptr = c_loc(arcs(1))
    n_arcs = size(arcs, kind=c_int64_t)
    self%raw = c_rgsaddle_pafi_create(config, n_frames, n_dof, frames, arcs_ptr, n_arcs)
    if (c_associated(self%raw)) then
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function pafi_create

  integer(c_int) function pafi_set_config(self, config) result(status)
    class(pafi_handle), intent(inout) :: self
    type(rgsaddle_pafi_config_t), intent(in) :: config
    if (.not. c_associated(self%raw)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    status = c_rgsaddle_pafi_set_config(self%raw, config)
  end function pafi_set_config

  integer(c_int) function pafi_run(self, callback, user) result(status)
    class(pafi_handle), intent(inout) :: self
    procedure(point_surface_proc), pointer, intent(in) :: callback
    type(c_ptr), value, intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    active_user = user_ptr
    active_point_surface => callback
    status = c_rgsaddle_pafi_run(self%raw, c_funloc(fortran_point_surface_trampoline), user_ptr)
    active_point_surface => null()
  end function pafi_run

  integer(c_int) function pafi_run_c(self, surface, user) result(status)
    class(pafi_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), value, intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_pafi_run(self%raw, surface, user_ptr)
  end function pafi_run_c

  integer(c_int) function pafi_result(self, out) result(status)
    class(pafi_handle), intent(in) :: self
    type(rgsaddle_pafi_result_t), intent(out) :: out
    status = c_rgsaddle_pafi_result(self%raw, out)
  end function pafi_result

  integer(c_int) function pafi_profile(self, arcs, gradients, gradient_errors, &
      free_energy, free_energy_errors, written) result(status)
    class(pafi_handle), intent(in) :: self
    real(c_double), intent(out) :: arcs(:)
    real(c_double), intent(out) :: gradients(:)
    real(c_double), intent(out) :: gradient_errors(:)
    real(c_double), intent(out) :: free_energy(:)
    real(c_double), intent(out) :: free_energy_errors(:)
    integer(c_int64_t), intent(out), optional :: written
    integer(c_int64_t) :: n_written
    integer(c_int64_t) :: cap
    cap = size(arcs, kind=c_int64_t)
    status = c_rgsaddle_pafi_profile(self%raw, arcs, gradients, gradient_errors, &
      free_energy, free_energy_errors, cap, n_written)
    if (present(written)) written = n_written
  end function pafi_profile

  integer(c_int) function ohtst_create(self, config, n_dof, point, normal) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(rgsaddle_ohtst_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: point(*)
    real(c_double), intent(in) :: normal(*)
    self%raw = c_rgsaddle_ohtst_create(config, n_dof, point, normal)
    if (c_associated(self%raw)) then
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function ohtst_create

  integer(c_int) function ohtst_set_config(self, config) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(rgsaddle_ohtst_config_t), intent(in) :: config
    if (.not. c_associated(self%raw)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    status = c_rgsaddle_ohtst_set_config(self%raw, config)
  end function ohtst_set_config

  integer(c_int) function ohtst_run(self, callback, user) result(status)
    class(ohtst_handle), intent(inout) :: self
    procedure(point_surface_proc), pointer, intent(in) :: callback
    type(c_ptr), value, intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    active_user = user_ptr
    active_point_surface => callback
    status = c_rgsaddle_ohtst_run(self%raw, c_funloc(fortran_point_surface_trampoline), user_ptr)
    active_point_surface => null()
  end function ohtst_run

  integer(c_int) function ohtst_run_c(self, surface, user) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), value, intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_ohtst_run(self%raw, surface, user_ptr)
  end function ohtst_run_c

  integer(c_int) function ohtst_result(self, out) result(status)
    class(ohtst_handle), intent(in) :: self
    type(rgsaddle_ohtst_result_t), intent(out) :: out
    status = c_rgsaddle_ohtst_result(self%raw, out)
  end function ohtst_result

  integer(c_int) function ohtst_curve(self, times, kappa_of_t, written) result(status)
    class(ohtst_handle), intent(in) :: self
    real(c_double), intent(out) :: times(:)
    real(c_double), intent(out) :: kappa_of_t(:)
    integer(c_int64_t), intent(out), optional :: written
    integer(c_int64_t) :: n_written
    integer(c_int64_t) :: cap
    cap = size(times, kind=c_int64_t)
    status = c_rgsaddle_ohtst_curve(self%raw, times, kappa_of_t, cap, n_written)
    if (present(written)) written = n_written
  end function ohtst_curve

  subroutine band_finalize(self)
    type(band_handle), intent(inout) :: self
    if (c_associated(self%raw)) then
      call c_rgsaddle_band_free(self%raw)
      self%raw = c_null_ptr
    end if
  end subroutine band_finalize

  subroutine minmode_finalize(self)
    type(minmode_handle), intent(inout) :: self
    if (c_associated(self%raw)) then
      call c_rgsaddle_minmode_free(self%raw)
      self%raw = c_null_ptr
    end if
  end subroutine minmode_finalize

  subroutine pafi_finalize(self)
    type(pafi_handle), intent(inout) :: self
    if (c_associated(self%raw)) then
      call c_rgsaddle_pafi_free(self%raw)
      self%raw = c_null_ptr
    end if
  end subroutine pafi_finalize

  subroutine ohtst_finalize(self)
    type(ohtst_handle), intent(inout) :: self
    if (c_associated(self%raw)) then
      call c_rgsaddle_ohtst_free(self%raw)
      self%raw = c_null_ptr
    end if
  end subroutine ohtst_finalize

  integer(c_int) function fortran_surface_trampoline(user, req_ptr) bind(C) result(status)
    type(c_ptr), value :: user
    type(c_ptr), value :: req_ptr
    type(rgsaddle_surface_request_t), pointer :: req
    real(c_double), pointer :: positions(:)
    real(c_double), pointer :: energies(:)
    real(c_double), pointer :: gradients(:)
    integer(c_int64_t) :: n_pos, n_eval
    if (.not. associated(active_surface)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(req_ptr, req)
    if (.not. associated(req)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (iand(req%flags, RGSADDLE_REQ_ONE_IMAGE) /= 0_c_int64_t) then
      n_pos = 3_c_int64_t * req%n_atoms
      n_eval = 1_c_int64_t
    else
      n_pos = req%n_images * 3_c_int64_t * req%n_atoms
      n_eval = req%n_images
    end if
    call c_f_pointer(req%positions, positions, [int(n_pos, kind=kind(0))])
    call c_f_pointer(req%energies, energies, [int(n_eval, kind=kind(0))])
    call c_f_pointer(req%gradients, gradients, [int(n_pos, kind=kind(0))])
    status = active_surface(user, req%n_images, req%n_atoms, positions, energies, gradients, req%image)
  end function fortran_surface_trampoline

  integer(c_int) function fortran_point_surface_trampoline(user, n_dof, position_ptr, energy, gradient_ptr) bind(C) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    type(c_ptr), value :: position_ptr
    real(c_double), intent(out) :: energy
    type(c_ptr), value :: gradient_ptr
    real(c_double), pointer :: position(:)
    real(c_double), pointer :: gradient(:)
    if (.not. associated(active_point_surface)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(position_ptr, position, [int(n_dof, kind=kind(0))])
    call c_f_pointer(gradient_ptr, gradient, [int(n_dof, kind=kind(0))])
    status = active_point_surface(user, n_dof, position, energy, gradient)
  end function fortran_point_surface_trampoline

end module rgsaddle
