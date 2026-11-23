module rgsaddle
  use, intrinsic :: iso_c_binding
  implicit none
  private

  integer(c_int), parameter, public :: RGSADDLE_ABI_MAJOR = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_ABI_MINOR = 63_c_int
  integer(c_int), parameter, public :: RGSADDLE_OK = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_NULL_SESSION = -1_c_int
  integer(c_int), parameter, public :: RGSADDLE_NULL_BAND = -2_c_int
  integer(c_int), parameter, public :: RGSADDLE_NULL_SURFACE = -3_c_int
  integer(c_int), parameter, public :: RGSADDLE_NULL_REPORT = -4_c_int
  integer(c_int), parameter, public :: RGSADDLE_SHAPE = -5_c_int
  integer(c_int), parameter, public :: RGSADDLE_SURFACE_FAILED = -6_c_int
  integer(c_int), parameter, public :: RGSADDLE_NON_FINITE = -7_c_int
  integer(c_int), parameter, public :: RGSADDLE_SOLVER = -8_c_int
  integer(c_int), parameter, public :: RGSADDLE_ABI_MISMATCH = -9_c_int
  integer(c_int), parameter, public :: RGSADDLE_ALLOC = -10_c_int
  integer(c_int), parameter, public :: RGSADDLE_INVALID_PARAMETER = -11_c_int
  integer(c_int), parameter, public :: RGSADDLE_NO_EVALUATION = -12_c_int
  integer(c_int), parameter, public :: RGSADDLE_STALLED = -13_c_int
  integer(c_int), parameter, public :: RGSADDLE_NOT_EQUILIBRATED = -14_c_int
  integer(c_int), parameter, public :: RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES = -15_c_int
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
  integer(c_int64_t), parameter, public :: RGSADDLE_BAND_PER_IMAGE = 1_c_int64_t
  integer(c_int64_t), parameter, public :: RGSADDLE_BAND_CI_AFTER = 4_c_int64_t
  integer(c_int64_t), parameter, public :: RGSADDLE_REQ_ONE_IMAGE = 1_c_int64_t
  ! `error stop` code for `b = a` on a handle. Assignment would copy the
  ! raw pointer and both finalizers would free it.
  integer, parameter, public :: RGSADDLE_HANDLE_NOT_ASSIGNABLE = 17
  ! floor((2**63 - 1) / 3): the largest atom count whose three Cartesian
  ! coordinates still fit in a signed 64-bit length.
  integer(c_int64_t), parameter :: max_atoms_for_triplet = 3074457345618258602_c_int64_t

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
    ! C `uint64_t` bit pattern. Values above 2^63-1 are negative here;
    ! `uint64_bits` builds them. The bits are passed through unchanged.
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
    ! C `uint64_t` bit pattern; see `seed` on `rgsaddle_pafi_config_t`.
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
    ! C `uint64_t` bit pattern; see `seed` on `rgsaddle_pafi_config_t`.
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
      ! PAFI and OH-TST may evaluate callbacks concurrently on rayon worker
      ! threads. Host state reachable through `user` must be thread-safe.
      type(c_ptr), value :: user
      integer(c_int64_t), value :: n_dof
      real(c_double), intent(in) :: position(*)
      real(c_double), intent(inout) :: energy
      real(c_double), intent(inout) :: gradient(*)
    end function point_surface_proc
  end interface

  type, public :: band_handle
    private
    type(c_ptr), allocatable :: raw
    type(c_ptr) :: owner
  contains
    final :: band_finalize
    procedure, public :: create => band_create
    procedure, public :: step => band_step
    procedure, public :: step_c => band_step_c
    procedure, public :: positions => band_positions
    procedure, public :: release => release_band
    procedure, public :: move_from => move_from_band
  end type band_handle

  type, public :: minmode_handle
    private
    type(c_ptr), allocatable :: raw
    type(c_ptr) :: owner
  contains
    final :: minmode_finalize
    procedure, public :: create => minmode_create
    procedure, public :: step => minmode_step
    procedure, public :: step_c => minmode_step_c
    procedure, public :: position => minmode_position
    procedure, public :: mode => minmode_mode
    procedure, public :: release => release_minmode
    procedure, public :: move_from => move_from_minmode
  end type minmode_handle

  type, public :: pafi_handle
    private
    type(c_ptr), allocatable :: raw
    type(c_ptr) :: owner
  contains
    final :: pafi_finalize
    procedure, public :: create => pafi_create
    procedure, public :: set_config => pafi_set_config
    procedure, public :: run => pafi_run
    procedure, public :: run_c => pafi_run_c
    procedure, public :: result => pafi_result
    procedure, public :: profile => pafi_profile
    procedure, public :: release => release_pafi
    procedure, public :: move_from => move_from_pafi
  end type pafi_handle

  type, public :: ohtst_handle
    private
    type(c_ptr), allocatable :: raw
    type(c_ptr) :: owner
  contains
    final :: ohtst_finalize
    procedure, public :: create => ohtst_create
    procedure, public :: set_config => ohtst_set_config
    procedure, public :: run => ohtst_run
    procedure, public :: run_c => ohtst_run_c
    procedure, public :: result => ohtst_result
    procedure, public :: curve => ohtst_curve
    procedure, public :: release => release_ohtst
    procedure, public :: move_from => move_from_ohtst
  end type ohtst_handle

  ! One call's callback and the host pointer, passed as the C user
  ! argument. Workers read it; the caller keeps it alive until the
  ! C call returns. There is no module-level callback.
  type :: callback_box
    procedure(surface_proc), pointer, nopass :: surface => null()
    procedure(point_surface_proc), pointer, nopass :: point => null()
    ! PAFI and OH-TST callbacks can run concurrently across rayon workers.
    ! `user` therefore points to host-owned thread-safe state.
    type(c_ptr) :: user = c_null_ptr
  end type callback_box


  interface assignment(=)
    module procedure assign_band
    module procedure assign_minmode
    module procedure assign_pafi
    module procedure assign_ohtst
  end interface

  public :: assignment(=)
  public :: init_band_config, init_minmode_config, init_pafi_config, init_ohtst_config, &
    abi_stamp, status_name, surface_proc, point_surface_proc, uint64_bits

  interface
    function c_malloc(size) bind(C, name="malloc") result(ptr)
      import :: c_ptr, c_size_t
      integer(c_size_t), value :: size
      type(c_ptr) :: ptr
    end function c_malloc

    function c_rgsaddle_abi_stamp(out) bind(C, name="rgsaddle_abi_stamp") result(status)
      import :: c_int, rgsaddle_version_t
      type(rgsaddle_version_t), intent(out) :: out
      integer(c_int) :: status
    end function c_rgsaddle_abi_stamp

    function c_rgsaddle_status_name(status) bind(C, name="rgsaddle_status_name") result(ptr)
      import :: c_int, c_ptr
      integer(c_int), value :: status
      type(c_ptr) :: ptr
    end function c_rgsaddle_status_name

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

  recursive function abi_stamp(out) result(status)
    type(rgsaddle_version_t), intent(out) :: out
    integer(c_int) :: status
    status = c_rgsaddle_abi_stamp(out)
  end function abi_stamp

  recursive function status_name(status) result(name)
    integer(c_int), value :: status
    character(len=:), allocatable :: name
    type(c_ptr) :: ptr
    character(len=1, kind=c_char), pointer :: chars(:)
    integer :: n, i
    ptr = c_rgsaddle_status_name(status)
    if (.not. c_associated(ptr)) then
      name = "UNKNOWN"
      return
    end if
    call c_f_pointer(ptr, chars, [64])
    n = 0
    do i = 1, 64
      if (chars(i) == c_null_char) exit
      n = i
    end do
    allocate(character(len=n) :: name)
    do i = 1, n
      name(i:i) = chars(i)
    end do
  end function status_name

  recursive subroutine init_band_config(config)
    type(rgsaddle_band_config_t), intent(out) :: config
    integer(c_int) :: status
    ! Rust `BandConfig::default`: climbing armed by force threshold 0.5
    ! (RGSADDLE_BAND_CI_AFTER), trigger_factor 0, and max_move 0.2.
    config = rgsaddle_band_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), RGSADDLE_BAND_CI_AFTER, &
      1_c_int32_t, 0_c_int32_t, 1_c_int32_t, RGSADDLE_METHOD_FIRE, 5.0_c_double, c_null_ptr, &
      0.0_c_double, 0.5_c_double, c_null_ptr, sqrt(epsilon(1.0_c_double)), 0.2_c_double, 10_c_int64_t)
    status = abi_stamp(config%version)
  end subroutine init_band_config

  recursive subroutine init_minmode_config(config, kind)
    type(rgsaddle_minmode_config_t), intent(out) :: config
    integer(c_int32_t), intent(in), optional :: kind
    integer(c_int) :: status
    integer(c_int32_t) :: mode_kind
    ! No C ABI default-initializer exists for this struct. The numbers
    ! are MinModeConfig::default: dr and force_tol 1e-3, rotation_tol
    ! 1e-4, 20 rotations, Krylov 12, max_move 0.2, and the five-degree
    ! angle stop (zero would disable it).
    real(c_double), parameter :: default_dr = 1.0_c_double / 1000.0_c_double
    real(c_double), parameter :: default_rotation_tol = 1.0_c_double / 10000.0_c_double
    real(c_double), parameter :: default_force_tol = 1.0_c_double / 1000.0_c_double
    real(c_double), parameter :: default_max_move = 0.2_c_double
    real(c_double), parameter :: default_rotation_angle_tol = &
      5.0_c_double * acos(-1.0_c_double) / 180.0_c_double
    mode_kind = RGSADDLE_MINMODE_DIMER
    if (present(kind)) mode_kind = kind
    config = rgsaddle_minmode_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      mode_kind, RGSADDLE_METHOD_FIRE, default_dr, default_rotation_tol, 20_c_int64_t, &
      12_c_int64_t, default_force_tol, default_max_move, default_rotation_angle_tol, &
      0_c_int32_t, 0_c_int32_t)
    status = abi_stamp(config%version)
  end subroutine init_minmode_config

  recursive subroutine init_pafi_config(config)
    type(rgsaddle_pafi_config_t), intent(out) :: config
    integer(c_int) :: status
    config = rgsaddle_pafi_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      0.02585_c_double, 1.0_c_double, 1.0e-3_c_double, 200_c_int64_t, 5_c_int64_t, 200_c_int64_t, &
      5_c_int64_t, 1_c_int64_t, 1_c_int32_t, 16_c_int64_t, 0.0_c_double, 4_c_int64_t)
    status = abi_stamp(config%version)
  end subroutine init_pafi_config

  recursive subroutine init_ohtst_config(config)
    type(rgsaddle_ohtst_config_t), intent(out) :: config
    integer(c_int) :: status
    ! Rust defaults: LangevinConfig::default + TransmissionConfig::default.
    config = rgsaddle_ohtst_config_t(rgsaddle_version_t(0_c_int32_t, 0_c_int32_t), 0_c_int64_t, &
      0.02585_c_double, 0.05_c_double, 1.0_c_double, 500_c_int64_t, 5_c_int64_t, 2000_c_int64_t, &
      20_c_int64_t, 1_c_int64_t, 16_c_int64_t, int(RGSADDLE_OHTST_INTEGRATOR_BAOAB, c_int32_t), &
      0.0_c_double, 0.05_c_double, 400_c_int64_t, 40_c_int64_t, 0_c_int64_t, 256_c_int64_t, &
      16_c_int64_t, 1_c_int64_t, int(RGSADDLE_OHTST_COUNT_FIRST_CROSSING, c_int32_t))
    status = abi_stamp(config%version)
  end subroutine init_ohtst_config

  recursive integer(c_int) function band_create(self, config, n_images, n_atoms, positions) result(status)
    class(band_handle), intent(inout) :: self
    type(rgsaddle_band_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    type(c_ptr) :: ptr
    logical :: shape_ok
    shape_ok = n_images >= 3_c_int64_t .and. n_atoms >= 1_c_int64_t
    if (shape_ok) then
      if (n_atoms > max_atoms_for_triplet / n_images) shape_ok = .false.
    end if
    if (config%version%major /= int(RGSADDLE_ABI_MAJOR, c_int32_t) .or. .not. shape_ok) then
      status = create_status(c_null_ptr, config%version%major, shape_ok)
      return
    end if
    ptr = c_rgsaddle_band_create(config, n_images, n_atoms, positions)
    status = create_status(ptr, config%version%major, .true.)
    if (status == RGSADDLE_OK) then
      call band_finalize(self)
      call store_raw(self%raw, ptr)
      call store_owner(self%owner)
    end if
  end function band_create

  recursive integer(c_int) function band_step(self, callback, user, out) result(status)
    class(band_handle), intent(inout) :: self
    procedure(surface_proc), pointer, intent(in) :: callback
    type(c_ptr), intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(callback_box), target :: box
    box%surface => callback
    box%point => null()
    box%user = c_null_ptr
    if (present(user)) box%user = user
    status = c_rgsaddle_band_step(raw_ptr(self%raw), c_funloc(fortran_surface_trampoline), c_loc(box), report)
    if (present(out)) out = report
  end function band_step

  recursive integer(c_int) function band_step_c(self, surface, user, out) result(status)
    class(band_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_band_step(raw_ptr(self%raw), surface, user_ptr, report)
    if (present(out)) out = report
  end function band_step_c

  recursive integer(c_int) function band_positions(self, positions) result(status)
    class(band_handle), intent(in) :: self
    real(c_double), intent(out) :: positions(*)
    status = c_rgsaddle_band_positions(raw_ptr(self%raw), positions)
  end function band_positions

  recursive integer(c_int) function minmode_create(self, config, n_atoms, position, mode) result(status)
    class(minmode_handle), intent(inout) :: self
    type(rgsaddle_minmode_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(in) :: mode(*)
    type(c_ptr) :: ptr
    logical :: shape_ok
    shape_ok = n_atoms >= 1_c_int64_t
    if (shape_ok) then
      if (n_atoms > max_atoms_for_triplet) shape_ok = .false.
    end if
    if (config%version%major /= int(RGSADDLE_ABI_MAJOR, c_int32_t) .or. .not. shape_ok) then
      status = create_status(c_null_ptr, config%version%major, shape_ok)
      return
    end if
    ptr = c_rgsaddle_minmode_create(config, n_atoms, position, mode)
    status = create_status(ptr, config%version%major, .true.)
    if (status == RGSADDLE_OK) then
      call minmode_finalize(self)
      call store_raw(self%raw, ptr)
      call store_owner(self%owner)
    end if
  end function minmode_create

  recursive integer(c_int) function minmode_step(self, callback, user, out) result(status)
    class(minmode_handle), intent(inout) :: self
    procedure(surface_proc), pointer, intent(in) :: callback
    type(c_ptr), intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(callback_box), target :: box
    box%surface => callback
    box%point => null()
    box%user = c_null_ptr
    if (present(user)) box%user = user
    status = c_rgsaddle_minmode_step(raw_ptr(self%raw), c_funloc(fortran_surface_trampoline), c_loc(box), report)
    if (present(out)) out = report
  end function minmode_step

  recursive integer(c_int) function minmode_step_c(self, surface, user, out) result(status)
    class(minmode_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), intent(in), optional :: user
    type(rgsaddle_report_t), intent(out), optional :: out
    type(rgsaddle_report_t) :: report
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_minmode_step(raw_ptr(self%raw), surface, user_ptr, report)
    if (present(out)) out = report
  end function minmode_step_c

  recursive integer(c_int) function minmode_position(self, position) result(status)
    class(minmode_handle), intent(in) :: self
    real(c_double), intent(out) :: position(*)
    status = c_rgsaddle_minmode_position(raw_ptr(self%raw), position)
  end function minmode_position

  recursive integer(c_int) function minmode_mode(self, mode) result(status)
    class(minmode_handle), intent(in) :: self
    real(c_double), intent(out) :: mode(*)
    status = c_rgsaddle_minmode_mode(raw_ptr(self%raw), mode)
  end function minmode_mode

  recursive integer(c_int) function pafi_create(self, config, n_frames, n_dof, frames, arcs) result(status)
    class(pafi_handle), intent(inout) :: self
    type(rgsaddle_pafi_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_frames
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: frames(*)
    real(c_double), intent(in), contiguous, target, optional :: arcs(:)
    type(c_ptr) :: arcs_ptr, ptr
    integer(c_int64_t) :: n_arcs
    logical :: shape_ok
    arcs_ptr = c_null_ptr
    n_arcs = 0_c_int64_t
    if (present(arcs)) then
      n_arcs = size(arcs, kind=c_int64_t)
      if (n_arcs > 0_c_int64_t) then
        arcs_ptr = c_loc(arcs(1))
      end if
    end if
    shape_ok = n_frames >= 2_c_int64_t .and. n_dof >= 1_c_int64_t
    if (shape_ok) then
      if (n_dof > huge(n_dof) / n_frames) shape_ok = .false.
    end if
    if (config%version%major /= int(RGSADDLE_ABI_MAJOR, c_int32_t) .or. .not. shape_ok) then
      status = create_status(c_null_ptr, config%version%major, shape_ok)
      return
    end if
    ptr = c_rgsaddle_pafi_create(config, n_frames, n_dof, frames, arcs_ptr, n_arcs)
    status = create_status(ptr, config%version%major, .true.)
    if (status == RGSADDLE_OK) then
      call pafi_finalize(self)
      call store_raw(self%raw, ptr)
      call store_owner(self%owner)
    end if
  end function pafi_create

  recursive integer(c_int) function pafi_set_config(self, config) result(status)
    class(pafi_handle), intent(inout) :: self
    type(rgsaddle_pafi_config_t), intent(in) :: config
    type(c_ptr) :: ptr
    ptr = raw_ptr(self%raw)
    if (.not. c_associated(ptr)) then
      status = RGSADDLE_NULL_SESSION
      return
    end if
    status = c_rgsaddle_pafi_set_config(ptr, config)
  end function pafi_set_config

  recursive integer(c_int) function pafi_run(self, callback, user) result(status)
    class(pafi_handle), intent(inout) :: self
    procedure(point_surface_proc), pointer, intent(in) :: callback
    type(c_ptr), intent(in), optional :: user
    type(callback_box), target :: box
    box%surface => null()
    box%point => callback
    box%user = c_null_ptr
    if (present(user)) box%user = user
    status = c_rgsaddle_pafi_run(raw_ptr(self%raw), c_funloc(fortran_point_surface_trampoline), c_loc(box))
  end function pafi_run

  recursive integer(c_int) function pafi_run_c(self, surface, user) result(status)
    class(pafi_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_pafi_run(raw_ptr(self%raw), surface, user_ptr)
  end function pafi_run_c

  recursive integer(c_int) function pafi_result(self, out) result(status)
    class(pafi_handle), intent(in) :: self
    type(rgsaddle_pafi_result_t), intent(out) :: out
    status = c_rgsaddle_pafi_result(raw_ptr(self%raw), out)
  end function pafi_result

  recursive integer(c_int) function pafi_profile(self, arcs, gradients, gradient_errors, &
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
    n_written = 0_c_int64_t
    cap = min(size(arcs, kind=c_int64_t), size(gradients, kind=c_int64_t), &
      size(gradient_errors, kind=c_int64_t), size(free_energy, kind=c_int64_t), &
      size(free_energy_errors, kind=c_int64_t))
    status = c_rgsaddle_pafi_profile(raw_ptr(self%raw), arcs, gradients, gradient_errors, &
      free_energy, free_energy_errors, cap, n_written)
    if (present(written)) written = n_written
  end function pafi_profile

  recursive integer(c_int) function ohtst_create(self, config, n_dof, point, normal) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(rgsaddle_ohtst_config_t), intent(in) :: config
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: point(*)
    real(c_double), intent(in) :: normal(*)
    type(c_ptr) :: ptr
    logical :: shape_ok
    shape_ok = n_dof >= 1_c_int64_t
    if (config%version%major /= int(RGSADDLE_ABI_MAJOR, c_int32_t) .or. .not. shape_ok) then
      status = create_status(c_null_ptr, config%version%major, shape_ok)
      return
    end if
    ptr = c_rgsaddle_ohtst_create(config, n_dof, point, normal)
    status = create_status(ptr, config%version%major, .true.)
    if (status == RGSADDLE_OK) then
      call ohtst_finalize(self)
      call store_raw(self%raw, ptr)
      call store_owner(self%owner)
    end if
  end function ohtst_create

  recursive integer(c_int) function ohtst_set_config(self, config) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(rgsaddle_ohtst_config_t), intent(in) :: config
    type(c_ptr) :: ptr
    ptr = raw_ptr(self%raw)
    if (.not. c_associated(ptr)) then
      status = RGSADDLE_NULL_SESSION
      return
    end if
    status = c_rgsaddle_ohtst_set_config(ptr, config)
  end function ohtst_set_config

  recursive integer(c_int) function ohtst_run(self, callback, user) result(status)
    class(ohtst_handle), intent(inout) :: self
    procedure(point_surface_proc), pointer, intent(in) :: callback
    type(c_ptr), intent(in), optional :: user
    type(callback_box), target :: box
    box%surface => null()
    box%point => callback
    box%user = c_null_ptr
    if (present(user)) box%user = user
    status = c_rgsaddle_ohtst_run(raw_ptr(self%raw), c_funloc(fortran_point_surface_trampoline), c_loc(box))
  end function ohtst_run

  recursive integer(c_int) function ohtst_run_c(self, surface, user) result(status)
    class(ohtst_handle), intent(inout) :: self
    type(c_funptr), value :: surface
    type(c_ptr), intent(in), optional :: user
    type(c_ptr) :: user_ptr
    user_ptr = c_null_ptr
    if (present(user)) user_ptr = user
    status = c_rgsaddle_ohtst_run(raw_ptr(self%raw), surface, user_ptr)
  end function ohtst_run_c

  recursive integer(c_int) function ohtst_result(self, out) result(status)
    class(ohtst_handle), intent(in) :: self
    type(rgsaddle_ohtst_result_t), intent(out) :: out
    status = c_rgsaddle_ohtst_result(raw_ptr(self%raw), out)
  end function ohtst_result

  recursive integer(c_int) function ohtst_curve(self, times, kappa_of_t, written) result(status)
    class(ohtst_handle), intent(in) :: self
    real(c_double), intent(out) :: times(:)
    real(c_double), intent(out) :: kappa_of_t(:)
    integer(c_int64_t), intent(out), optional :: written
    integer(c_int64_t) :: n_written
    integer(c_int64_t) :: cap
    n_written = 0_c_int64_t
    cap = min(size(times, kind=c_int64_t), size(kappa_of_t, kind=c_int64_t))
    status = c_rgsaddle_ohtst_curve(raw_ptr(self%raw), times, kappa_of_t, cap, n_written)
    if (present(written)) written = n_written
  end function ohtst_curve

  recursive subroutine band_finalize(self)
    type(band_handle), intent(inout) :: self
    if (allocated(self%raw)) then
      if (c_associated(self%raw) .and. should_free_owner(self%owner)) call c_rgsaddle_band_free(self%raw)
      deallocate(self%raw)
    end if
    self%owner = c_null_ptr
  end subroutine band_finalize

  recursive subroutine minmode_finalize(self)
    type(minmode_handle), intent(inout) :: self
    if (allocated(self%raw)) then
      if (c_associated(self%raw) .and. should_free_owner(self%owner)) call c_rgsaddle_minmode_free(self%raw)
      deallocate(self%raw)
    end if
    self%owner = c_null_ptr
  end subroutine minmode_finalize

  recursive subroutine pafi_finalize(self)
    type(pafi_handle), intent(inout) :: self
    if (allocated(self%raw)) then
      if (c_associated(self%raw) .and. should_free_owner(self%owner)) call c_rgsaddle_pafi_free(self%raw)
      deallocate(self%raw)
    end if
    self%owner = c_null_ptr
  end subroutine pafi_finalize

  recursive subroutine ohtst_finalize(self)
    type(ohtst_handle), intent(inout) :: self
    if (allocated(self%raw)) then
      if (c_associated(self%raw) .and. should_free_owner(self%owner)) call c_rgsaddle_ohtst_free(self%raw)
      deallocate(self%raw)
    end if
    self%owner = c_null_ptr
  end subroutine ohtst_finalize

  recursive integer(c_int) function fortran_surface_trampoline(user, req_ptr) &
      bind(C, name="rgsaddle_fortran_surface_trampoline") result(status)
    type(c_ptr), value :: user
    type(c_ptr), value :: req_ptr
    type(rgsaddle_surface_request_t), pointer :: req
    type(callback_box), pointer :: box
    real(c_double), pointer :: positions(:)
    real(c_double), pointer :: energies(:)
    real(c_double), pointer :: gradients(:)
    integer(c_int64_t) :: n_pos, n_eval
    logical :: one_image
    call c_f_pointer(req_ptr, req)
    if (.not. associated(req)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (iand(req%flags, RGSADDLE_REQ_ONE_IMAGE) /= 0_c_int64_t) then
      one_image = .true.
    else
      one_image = .false.
    end if
    status = buffer_extent(req%n_images, req%n_atoms, one_image, n_pos, n_eval)
    if (status /= RGSADDLE_OK) return
    call c_f_pointer(user, box)
    if (.not. associated(box)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (.not. associated(box%surface)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(req%positions, positions, [int(n_pos, kind=kind(0))])
    call c_f_pointer(req%energies, energies, [int(n_eval, kind=kind(0))])
    call c_f_pointer(req%gradients, gradients, [int(n_pos, kind=kind(0))])
    ! n_eval is the number of rows in the buffers: 1 for ONE_IMAGE,
    ! the request's row count otherwise. req%image is the band index
    ! for ONE_IMAGE and for a batched climbing-image refinement
    ! (which also carries one row), or -1 for an ordinary batch.
    status = box%surface(box%user, n_eval, req%n_atoms, positions, energies, gradients, req%image)
  end function fortran_surface_trampoline

  recursive integer(c_int) function fortran_point_surface_trampoline(user, n_dof, position_ptr, energy, gradient_ptr) &
      bind(C, name="rgsaddle_fortran_point_trampoline") result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    type(c_ptr), value :: position_ptr
    real(c_double), intent(out) :: energy
    type(c_ptr), value :: gradient_ptr
    type(callback_box), pointer :: box
    real(c_double), pointer :: position(:)
    real(c_double), pointer :: gradient(:)
    energy = 0.0_c_double
    if (n_dof < 1_c_int64_t .or. n_dof > int(huge(0), c_int64_t)) then
      status = RGSADDLE_SHAPE
      return
    end if
    call c_f_pointer(user, box)
    if (.not. associated(box)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (.not. associated(box%point)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(position_ptr, position, [int(n_dof, kind=kind(0))])
    call c_f_pointer(gradient_ptr, gradient, [int(n_dof, kind=kind(0))])
    status = box%point(box%user, n_dof, position, energy, gradient)
  end function fortran_point_surface_trampoline

  pure recursive function raw_ptr(slot) result(ptr)
    type(c_ptr), allocatable, intent(in) :: slot
    type(c_ptr) :: ptr
    if (allocated(slot)) then
      ptr = slot
    else
      ptr = c_null_ptr
    end if
  end function raw_ptr

  recursive subroutine store_raw(slot, ptr)
    type(c_ptr), allocatable, intent(inout) :: slot
    type(c_ptr), value :: ptr
    if (allocated(slot)) deallocate(slot)
    if (c_associated(ptr)) then
      allocate(slot)
      slot = ptr
    end if
  end subroutine store_raw

  recursive integer(c_int) function create_status(handle, major, shape_ok) result(status)
    type(c_ptr), intent(in) :: handle
    integer(c_int32_t), intent(in) :: major
    logical, intent(in) :: shape_ok
    if (major /= int(RGSADDLE_ABI_MAJOR, c_int32_t)) then
      status = RGSADDLE_ABI_MISMATCH
    else if (.not. shape_ok) then
      status = RGSADDLE_SHAPE
    else if (.not. c_associated(handle)) then
      status = RGSADDLE_INVALID_PARAMETER
    else
      status = RGSADDLE_OK
    end if
  end function create_status

  recursive integer(c_int) function buffer_extent(n_images, n_atoms, one_image, n_pos, n_eval) result(status)
    integer(c_int64_t), intent(in) :: n_images, n_atoms
    logical, intent(in) :: one_image
    integer(c_int64_t), intent(out) :: n_pos, n_eval
    integer(c_int64_t) :: rows, atoms3
    integer(c_int64_t), parameter :: default_huge = int(huge(0), c_int64_t)
    n_pos = 0_c_int64_t
    n_eval = 0_c_int64_t
    if (n_images < 1_c_int64_t .or. n_atoms < 1_c_int64_t) then
      status = RGSADDLE_SHAPE
      return
    end if
    if (one_image) then
      rows = 1_c_int64_t
    else
      rows = n_images
    end if
    if (n_atoms > max_atoms_for_triplet) then
      status = RGSADDLE_SHAPE
      return
    end if
    atoms3 = 3_c_int64_t * n_atoms
    if (rows > huge(rows) / atoms3) then
      status = RGSADDLE_SHAPE
      return
    end if
    n_pos = rows * atoms3
    n_eval = rows
    if (n_pos > default_huge .or. n_eval > default_huge) then
      status = RGSADDLE_SHAPE
      return
    end if
    status = RGSADDLE_OK
  end function buffer_extent

  pure recursive function uint64_bits(high, low) result(bits)
    integer(c_int64_t), intent(in) :: high, low
    integer(c_int64_t) :: bits
    integer(c_int64_t), parameter :: mask32 = int(z'FFFFFFFF', c_int64_t)
    bits = ior(shiftl(iand(high, mask32), 32), iand(low, mask32))
  end function uint64_bits

  recursive subroutine release_band(self)
    class(band_handle), intent(inout) :: self
    call band_finalize(self)
  end subroutine release_band

  recursive subroutine release_minmode(self)
    class(minmode_handle), intent(inout) :: self
    call minmode_finalize(self)
  end subroutine release_minmode

  recursive subroutine release_pafi(self)
    class(pafi_handle), intent(inout) :: self
    call pafi_finalize(self)
  end subroutine release_pafi

  recursive subroutine release_ohtst(self)
    class(ohtst_handle), intent(inout) :: self
    call ohtst_finalize(self)
  end subroutine release_ohtst

  recursive subroutine move_from_band(self, src)
    class(band_handle), intent(inout) :: self
    type(band_handle), intent(inout) :: src
    if (c_associated(raw_ptr(self%raw), raw_ptr(src%raw))) return
    call band_finalize(self)
    if (allocated(src%raw)) then
      call move_alloc(src%raw, self%raw)
      self%owner = src%owner
      src%owner = c_null_ptr
    end if
  end subroutine move_from_band

  recursive subroutine move_from_minmode(self, src)
    class(minmode_handle), intent(inout) :: self
    type(minmode_handle), intent(inout) :: src
    if (c_associated(raw_ptr(self%raw), raw_ptr(src%raw))) return
    call minmode_finalize(self)
    if (allocated(src%raw)) then
      call move_alloc(src%raw, self%raw)
      self%owner = src%owner
      src%owner = c_null_ptr
    end if
  end subroutine move_from_minmode

  recursive subroutine move_from_pafi(self, src)
    class(pafi_handle), intent(inout) :: self
    type(pafi_handle), intent(inout) :: src
    if (c_associated(raw_ptr(self%raw), raw_ptr(src%raw))) return
    call pafi_finalize(self)
    if (allocated(src%raw)) then
      call move_alloc(src%raw, self%raw)
      self%owner = src%owner
      src%owner = c_null_ptr
    end if
  end subroutine move_from_pafi

  recursive subroutine move_from_ohtst(self, src)
    class(ohtst_handle), intent(inout) :: self
    type(ohtst_handle), intent(inout) :: src
    if (c_associated(raw_ptr(self%raw), raw_ptr(src%raw))) return
    call ohtst_finalize(self)
    if (allocated(src%raw)) then
      call move_alloc(src%raw, self%raw)
      self%owner = src%owner
      src%owner = c_null_ptr
    end if
  end subroutine move_from_ohtst

  recursive subroutine store_owner(slot)
    type(c_ptr), intent(inout) :: slot
    slot = alloc_owner_flag()
  end subroutine store_owner

  recursive logical function should_free_owner(owner) result(free_it)
    type(c_ptr), intent(in) :: owner
    integer(c_signed_char), pointer :: flag
    free_it = .true.
    if (.not. c_associated(owner)) return
    call c_f_pointer(owner, flag)
    if (.not. associated(flag)) return
    free_it = flag /= 0_c_signed_char
    flag = 0_c_signed_char
  end function should_free_owner

  recursive function alloc_owner_flag() result(slot)
    type(c_ptr) :: slot
    integer(c_signed_char), pointer :: flag
    slot = c_malloc(int(1, c_size_t))
    if (c_associated(slot)) then
      call c_f_pointer(slot, flag)
      if (associated(flag)) flag = 1_c_signed_char
    end if
  end function alloc_owner_flag

  recursive subroutine assign_band(lhs, rhs)
    type(band_handle), intent(inout) :: lhs
    type(band_handle), intent(in) :: rhs
    if (c_associated(raw_ptr(lhs%raw), raw_ptr(rhs%raw))) error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
    error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
  end subroutine assign_band

  recursive subroutine assign_minmode(lhs, rhs)
    type(minmode_handle), intent(inout) :: lhs
    type(minmode_handle), intent(in) :: rhs
    if (c_associated(raw_ptr(lhs%raw), raw_ptr(rhs%raw))) error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
    error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
  end subroutine assign_minmode

  recursive subroutine assign_pafi(lhs, rhs)
    type(pafi_handle), intent(inout) :: lhs
    type(pafi_handle), intent(in) :: rhs
    if (c_associated(raw_ptr(lhs%raw), raw_ptr(rhs%raw))) error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
    error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
  end subroutine assign_pafi

  recursive subroutine assign_ohtst(lhs, rhs)
    type(ohtst_handle), intent(inout) :: lhs
    type(ohtst_handle), intent(in) :: rhs
    if (c_associated(raw_ptr(lhs%raw), raw_ptr(rhs%raw))) error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
    error stop RGSADDLE_HANDLE_NOT_ASSIGNABLE
  end subroutine assign_ohtst

end module rgsaddle
