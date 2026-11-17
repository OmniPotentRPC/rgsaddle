module rgsaddle
  use, intrinsic :: iso_c_binding
  implicit none
  private

  integer(c_int), parameter, public :: RGSADDLE_ABI_MAJOR = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_ABI_MINOR = 41_c_int
  integer(c_int), parameter, public :: RGSADDLE_OK = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_INVALID_PARAMETER = -11_c_int
  integer(c_int), parameter, public :: RGSADDLE_STATUS_RUNNING = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_STATUS_CONVERGED = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_METHOD_FIRE = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_METHOD_LBFGS = 1_c_int
  integer(c_int), parameter, public :: RGSADDLE_MINMODE_DIMER = 0_c_int
  integer(c_int), parameter, public :: RGSADDLE_MINMODE_LANCZOS = 1_c_int
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
  end type pafi_handle

  type, public :: ohtst_handle
    private
    type(c_ptr) :: raw = c_null_ptr
  contains
    final :: ohtst_finalize
    procedure, public :: create => ohtst_create
  end type ohtst_handle

  procedure(surface_proc), pointer, save :: active_surface => null()
  type(c_ptr), save :: active_user = c_null_ptr

  public :: init_band_config, init_minmode_config, abi_stamp, status_name, surface_proc

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

  integer(c_int) function pafi_create(self) result(status)
    class(pafi_handle), intent(inout) :: self
    self%raw = c_null_ptr
    status = RGSADDLE_INVALID_PARAMETER
  end function pafi_create

  integer(c_int) function ohtst_create(self) result(status)
    class(ohtst_handle), intent(inout) :: self
    self%raw = c_null_ptr
    status = RGSADDLE_INVALID_PARAMETER
  end function ohtst_create

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
    self%raw = c_null_ptr
  end subroutine pafi_finalize

  subroutine ohtst_finalize(self)
    type(ohtst_handle), intent(inout) :: self
    self%raw = c_null_ptr
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

end module rgsaddle
