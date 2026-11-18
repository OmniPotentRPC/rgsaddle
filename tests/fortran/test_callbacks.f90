module rgsaddle_fortran_callback_fixtures
  use, intrinsic :: iso_c_binding
  use rgsaddle
  implicit none
  private

  integer(c_int64_t), parameter, public :: SURFACE_CTX_MAGIC = 7042001_c_int64_t
  integer(c_int64_t), parameter, public :: POINT_CTX_MAGIC = 7042002_c_int64_t
  integer(c_int64_t), parameter, public :: OUTER_CTX_MAGIC = 7042003_c_int64_t

  type, public :: surface_ctx
    integer(c_int64_t) :: magic = 0_c_int64_t
    type(c_ptr) :: self_ptr = c_null_ptr
    real(c_double) :: center(3) = 0.0_c_double
    real(c_double) :: scale = 1.0_c_double
    integer(c_int64_t) :: calls = 0_c_int64_t
  end type surface_ctx

  type, public :: point_ctx
    integer(c_int64_t) :: magic = 0_c_int64_t
    type(c_ptr) :: self_ptr = c_null_ptr
    real(c_double) :: scale = 1.0_c_double
    real(c_double) :: shift = 0.0_c_double
  end type point_ctx

  type, public :: nested_surface_ctx
    integer(c_int64_t) :: magic = 0_c_int64_t
    type(c_ptr) :: self_ptr = c_null_ptr
    real(c_double) :: target_x = 0.0_c_double
    integer(c_int64_t) :: outer_calls = 0_c_int64_t
    integer(c_int64_t) :: inner_band_calls = 0_c_int64_t
    integer(c_int64_t) :: inner_minmode_calls = 0_c_int64_t
  end type nested_surface_ctx

  public :: shifted_surface, null_user_surface, c_shifted_surface, &
    scaled_pafi_surface, scaled_ohtst_surface, null_user_point_surface, &
    c_scaled_point_surface, nested_outer_band_surface, &
    nested_outer_pafi_surface, nested_outer_ohtst_surface

contains

  pure recursive subroutine eval_shifted_well(row, center, scale, energy, grad)
    real(c_double), intent(in) :: row(3), center(3)
    real(c_double), value :: scale
    real(c_double), intent(out) :: energy
    real(c_double), intent(out) :: grad(3)
    real(c_double) :: dx, dy, dz
    dx = row(1) - center(1)
    dy = row(2) - center(2)
    dz = row(3) - center(3)
    energy = scale * ((dx * dx - 1.0_c_double) ** 2 + 2.0_c_double * dy * dy + 2.0_c_double * dz * dz)
    grad(1) = scale * 4.0_c_double * dx * (dx * dx - 1.0_c_double)
    grad(2) = scale * 4.0_c_double * dy
    grad(3) = scale * 4.0_c_double * dz
  end subroutine eval_shifted_well

  recursive integer(c_int) function shifted_surface(user, n_images, n_atoms, &
      positions, energies, gradients, image) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    real(c_double), intent(inout) :: energies(*)
    real(c_double), intent(inout) :: gradients(*)
    integer(c_int64_t), value :: image
    type(surface_ctx), pointer :: ctx
    integer :: i, n_eval
    real(c_double) :: row(3), grad(3), energy
    if (.not. c_associated(user)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, ctx)
    if (.not. associated(ctx)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (ctx%magic /= SURFACE_CTX_MAGIC .or. .not. c_associated(user, ctx%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_atoms /= 1_c_int64_t .or. n_images < 1_c_int64_t .or. image < -1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    ctx%calls = ctx%calls + 1_c_int64_t
    n_eval = int(n_images, kind=kind(0))
    do i = 1, n_eval
      row = positions(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_shifted_well(row, ctx%center, ctx%scale, energy, grad)
      energies(i) = energy
      gradients(3 * (i - 1) + 1:3 * (i - 1) + 3) = grad
    end do
    status = RGSADDLE_OK
  end function shifted_surface

  recursive integer(c_int) function null_user_surface(user, n_images, n_atoms, &
      positions, energies, gradients, image) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    real(c_double), intent(inout) :: energies(*)
    real(c_double), intent(inout) :: gradients(*)
    integer(c_int64_t), value :: image
    integer :: i, n_eval
    real(c_double) :: row(3), grad(3), energy
    if (c_associated(user)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_atoms /= 1_c_int64_t .or. n_images < 1_c_int64_t .or. image < -1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    n_eval = int(n_images, kind=kind(0))
    do i = 1, n_eval
      row = positions(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_shifted_well(row, [0.0_c_double, 0.0_c_double, 0.0_c_double], 1.0_c_double, energy, grad)
      energies(i) = energy
      gradients(3 * (i - 1) + 1:3 * (i - 1) + 3) = grad
    end do
    status = RGSADDLE_OK
  end function null_user_surface

  recursive integer(c_int) function c_shifted_surface(user, req_ptr) bind(C) result(status)
    type(c_ptr), value :: user
    type(c_ptr), value :: req_ptr
    type(rgsaddle_surface_request_t), pointer :: req
    type(surface_ctx), pointer :: ctx
    real(c_double), pointer :: pos(:), ene(:), grad(:)
    integer(c_int64_t) :: npos, neval
    integer :: i
    real(c_double) :: row(3), g(3), energy
    if (.not. c_associated(user)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, ctx)
    if (ctx%magic /= SURFACE_CTX_MAGIC .or. .not. c_associated(user, ctx%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(req_ptr, req)
    if (.not. associated(req)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (iand(req%flags, RGSADDLE_REQ_ONE_IMAGE) /= 0_c_int64_t) then
      neval = 1_c_int64_t
    else
      neval = req%n_images
    end if
    npos = 3_c_int64_t * req%n_atoms * neval
    call c_f_pointer(req%positions, pos, [int(npos, kind=kind(0))])
    call c_f_pointer(req%energies, ene, [int(neval, kind=kind(0))])
    call c_f_pointer(req%gradients, grad, [int(npos, kind=kind(0))])
    ctx%calls = ctx%calls + 1_c_int64_t
    do i = 1, int(neval, kind=kind(0))
      row = pos(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_shifted_well(row, ctx%center, ctx%scale, energy, g)
      ene(i) = energy
      grad(3 * (i - 1) + 1:3 * (i - 1) + 3) = g
    end do
    status = RGSADDLE_OK
  end function c_shifted_surface

  recursive integer(c_int) function scaled_pafi_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    type(point_ctx), pointer :: ctx
    real(c_double) :: u
    integer :: i
    if (.not. c_associated(user) .or. n_dof < 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, ctx)
    if (ctx%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, ctx%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    u = position(1) - ctx%shift
    energy = ctx%scale * (1.0_c_double - u * u) ** 2
    do i = 1, int(n_dof, kind=kind(0))
      gradient(i) = 0.0_c_double
    end do
    gradient(1) = -4.0_c_double * ctx%scale * u * (1.0_c_double - u * u)
    status = RGSADDLE_OK
  end function scaled_pafi_surface

  recursive integer(c_int) function scaled_ohtst_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    type(point_ctx), pointer :: ctx
    real(c_double) :: dx, y
    if (.not. c_associated(user) .or. n_dof /= 2_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, ctx)
    if (ctx%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, ctx%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    dx = position(1) - ctx%shift
    y = position(2)
    energy = 0.5_c_double * (-ctx%scale * dx * dx + 4.0_c_double * y * y)
    gradient(1) = -ctx%scale * dx
    gradient(2) = 4.0_c_double * y
    status = RGSADDLE_OK
  end function scaled_ohtst_surface

  recursive integer(c_int) function null_user_point_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    real(c_double) :: x
    if (c_associated(user)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_dof == 1_c_int64_t) then
      x = position(1)
      energy = (1.0_c_double - x * x) ** 2
      gradient(1) = -4.0_c_double * x * (1.0_c_double - x * x)
      status = RGSADDLE_OK
    else if (n_dof == 2_c_int64_t) then
      energy = 0.5_c_double * (-position(1) ** 2 + 4.0_c_double * position(2) ** 2)
      gradient(1) = -position(1)
      gradient(2) = 4.0_c_double * position(2)
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function null_user_point_surface

  recursive integer(c_int) function c_scaled_point_surface(user, n_dof, pos_ptr, energy, grad_ptr) &
      bind(C) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    type(c_ptr), value :: pos_ptr
    real(c_double), intent(out) :: energy
    type(c_ptr), value :: grad_ptr
    type(point_ctx), pointer :: ctx
    real(c_double), pointer :: pos(:), grad(:)
    real(c_double) :: u
    energy = 0.0_c_double
    if (.not. c_associated(user) .or. n_dof < 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, ctx)
    if (ctx%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, ctx%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(pos_ptr, pos, [int(n_dof, kind=kind(0))])
    call c_f_pointer(grad_ptr, grad, [int(n_dof, kind=kind(0))])
    if (n_dof == 1_c_int64_t) then
      u = pos(1) - ctx%shift
      energy = ctx%scale * (1.0_c_double - u * u) ** 2
      grad(1) = -4.0_c_double * ctx%scale * u * (1.0_c_double - u * u)
      status = RGSADDLE_OK
    else if (n_dof == 2_c_int64_t) then
      u = pos(1) - ctx%shift
      energy = 0.5_c_double * (-ctx%scale * u * u + 4.0_c_double * pos(2) * pos(2))
      grad(1) = -ctx%scale * u
      grad(2) = 4.0_c_double * pos(2)
      status = RGSADDLE_OK
    else
      status = RGSADDLE_INVALID_PARAMETER
    end if
  end function c_scaled_point_surface

  recursive integer(c_int) function nested_outer_band_surface(user, n_images, n_atoms, &
      positions, energies, gradients, image) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    real(c_double), intent(inout) :: energies(*)
    real(c_double), intent(inout) :: gradients(*)
    integer(c_int64_t), value :: image
    type(nested_surface_ctx), pointer :: outer
    type(surface_ctx), target :: inner_band_ctx, inner_mm_ctx
    type(rgsaddle_band_config_t) :: bcfg
    type(rgsaddle_minmode_config_t) :: mcfg
    type(band_handle) :: inner_band
    type(minmode_handle) :: inner_mm
    type(rgsaddle_report_t) :: brep, mrep
    procedure(surface_proc), pointer :: icb
    real(c_double) :: bpos(3, 5), start(3), mode(3), saddle(3)
    real(c_double) :: row(3), grad(3), energy, t
    integer(c_int) :: st
    integer :: i, k, n_eval

    if (.not. c_associated(user)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, outer)
    if (outer%magic /= OUTER_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_atoms /= 1_c_int64_t .or. n_images < 1_c_int64_t .or. image < -1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    outer%outer_calls = outer%outer_calls + 1_c_int64_t

    ! Step an inner band session (nested band_step -> fortran_surface_trampoline)
    ! with a distinct user pointer.
    inner_band_ctx%magic = SURFACE_CTX_MAGIC
    inner_band_ctx%self_ptr = c_loc(inner_band_ctx)
    inner_band_ctx%center = [-0.5_c_double, 0.25_c_double, 0.0_c_double]
    inner_band_ctx%scale = 1.25_c_double
    inner_band_ctx%calls = 0_c_int64_t
    do k = 1, 5
      t = real(k - 1, c_double) / 4.0_c_double
      bpos(1, k) = inner_band_ctx%center(1) - 1.0_c_double + 2.0_c_double * t
      bpos(2, k) = inner_band_ctx%center(2) + 0.3_c_double * sin(acos(-1.0_c_double) * t)
      bpos(3, k) = inner_band_ctx%center(3)
    end do
    call init_band_config(bcfg)
    bcfg%max_move = 0.1_c_double
    st = inner_band%create(bcfg, 5_c_int64_t, 1_c_int64_t, bpos)
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    icb => shifted_surface
    st = inner_band%step(icb, c_loc(inner_band_ctx), brep)
    if (st /= RGSADDLE_OK .or. inner_band_ctx%calls < 1_c_int64_t) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    outer%inner_band_calls = outer%inner_band_calls + inner_band_ctx%calls

    ! Converge an inner min-mode session (nested minmode_step -> fortran_surface_trampoline)
    ! to recover the shifted saddle coordinate used by the outer potential.
    inner_mm_ctx%magic = SURFACE_CTX_MAGIC
    inner_mm_ctx%self_ptr = c_loc(inner_mm_ctx)
    inner_mm_ctx%center = [outer%target_x, 0.0_c_double, 0.0_c_double]
    inner_mm_ctx%scale = 1.0_c_double
    inner_mm_ctx%calls = 0_c_int64_t
    start = [outer%target_x + 0.3_c_double, 0.15_c_double, -0.1_c_double]
    mode = [0.9_c_double, 0.2_c_double, 0.1_c_double]
    call init_minmode_config(mcfg, int(RGSADDLE_MINMODE_DIMER, c_int32_t))
    mcfg%force_tol = 1.0e-4_c_double
    mcfg%max_move = 0.1_c_double
    st = inner_mm%create(mcfg, 1_c_int64_t, start, mode)
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    mrep%status = RGSADDLE_STATUS_RUNNING
    do k = 1, 500
      st = inner_mm%step(icb, c_loc(inner_mm_ctx), mrep)
      if (st /= RGSADDLE_OK) then
        status = st
        return
      end if
      if (mrep%status == RGSADDLE_STATUS_CONVERGED) exit
    end do
    if (mrep%status /= RGSADDLE_STATUS_CONVERGED .or. inner_mm_ctx%calls < 1_c_int64_t) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    st = inner_mm%position(saddle)
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    if (abs(saddle(1) - outer%target_x) > 1.0e-2_c_double) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    outer%inner_minmode_calls = outer%inner_minmode_calls + inner_mm_ctx%calls

    ! Confirm the outer user context was not clobbered by either nested run.
    if (outer%magic /= OUTER_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if

    n_eval = int(n_images, kind=kind(0))
    do i = 1, n_eval
      row = positions(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_shifted_well(row, [saddle(1), 0.0_c_double, 0.0_c_double], 1.0_c_double, energy, grad)
      energies(i) = energy
      gradients(3 * (i - 1) + 1:3 * (i - 1) + 3) = grad
    end do
    status = RGSADDLE_OK
  end function nested_outer_band_surface

  recursive integer(c_int) function nested_outer_pafi_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    type(point_ctx), pointer :: outer
    type(point_ctx), target :: inner_ctx
    type(rgsaddle_pafi_config_t) :: icfg
    type(rgsaddle_pafi_result_t) :: ires
    type(pafi_handle) :: inner
    procedure(point_surface_proc), pointer :: icb
    real(c_double) :: iframes(21), u
    integer(c_int) :: st
    integer :: k

    if (.not. c_associated(user) .or. n_dof /= 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, outer)
    if (outer%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if

    inner_ctx%magic = POINT_CTX_MAGIC
    inner_ctx%self_ptr = c_loc(inner_ctx)
    inner_ctx%scale = 1.75_c_double
    inner_ctx%shift = -0.2_c_double
    do k = 1, 21
      iframes(k) = inner_ctx%shift - 1.0_c_double + 2.0_c_double * real(k - 1, c_double) / 20.0_c_double
    end do
    call init_pafi_config(icfg)
    icfg%thermal = 0.02_c_double
    icfg%time_step = 0.01_c_double
    icfg%thermalize = 1_c_int64_t
    icfg%stride = 1_c_int64_t
    icfg%samples = 4_c_int64_t
    icfg%blocks = 2_c_int64_t
    icfg%seed = 5_c_int64_t
    icfg%fix_com = 0_c_int32_t
    icfg%retain = 1_c_int64_t
    icfg%max_rounds = 1_c_int64_t
    st = inner%create(icfg, 21_c_int64_t, 1_c_int64_t, iframes)
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    icb => scaled_pafi_surface
    st = inner%run(icb, c_loc(inner_ctx))
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    st = inner%result(ires)
    if (st /= RGSADDLE_OK .or. ires%evaluations < 1_c_int64_t) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    if (abs(ires%barrier - inner_ctx%scale) > 5.0e-2_c_double) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if

    if (outer%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    u = position(1) - outer%shift
    energy = outer%scale * (1.0_c_double - u * u) ** 2
    gradient(1) = -4.0_c_double * outer%scale * u * (1.0_c_double - u * u)
    status = RGSADDLE_OK
  end function nested_outer_pafi_surface

  recursive integer(c_int) function nested_outer_ohtst_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    type(point_ctx), pointer :: outer
    type(point_ctx), target :: inner_ctx
    type(rgsaddle_ohtst_config_t) :: icfg
    type(rgsaddle_ohtst_result_t) :: ires
    type(ohtst_handle) :: inner
    procedure(point_surface_proc), pointer :: icb
    real(c_double) :: pt(2), nrm(2), dx, y
    integer(c_int) :: st

    if (.not. c_associated(user) .or. n_dof /= 2_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    call c_f_pointer(user, outer)
    if (outer%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if

    inner_ctx%magic = POINT_CTX_MAGIC
    inner_ctx%self_ptr = c_loc(inner_ctx)
    inner_ctx%scale = 2.25_c_double
    inner_ctx%shift = -0.1_c_double
    pt = [-0.1_c_double, 0.0_c_double]
    nrm = [1.0_c_double, 0.0_c_double]
    call init_ohtst_config(icfg)
    icfg%thermal = 0.5_c_double
    icfg%sample_burn_in = 2_c_int64_t
    icfg%sample_stride = 1_c_int64_t
    icfg%sample_records = 4_c_int64_t
    icfg%sample_blocks = 2_c_int64_t
    icfg%sample_retain = 2_c_int64_t
    icfg%shooting_steps = 4_c_int64_t
    icfg%shooting_check_every = 2_c_int64_t
    icfg%shooting_shots = 4_c_int64_t
    icfg%shooting_blocks = 2_c_int64_t
    st = inner%create(icfg, 2_c_int64_t, pt, nrm)
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    icb => scaled_ohtst_surface
    st = inner%run(icb, c_loc(inner_ctx))
    if (st /= RGSADDLE_OK) then
      status = st
      return
    end if
    st = inner%result(ires)
    if (st /= RGSADDLE_OK .or. ires%evaluations < 1_c_int64_t) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if

    if (outer%magic /= POINT_CTX_MAGIC .or. .not. c_associated(user, outer%self_ptr)) then
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    dx = position(1) - outer%shift
    y = position(2)
    energy = 0.5_c_double * (-outer%scale * dx * dx + 4.0_c_double * y * y)
    gradient(1) = -outer%scale * dx
    gradient(2) = 4.0_c_double * y
    status = RGSADDLE_OK
  end function nested_outer_ohtst_surface

end module rgsaddle_fortran_callback_fixtures

program test_callbacks
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support, only: check_status
  use rgsaddle_fortran_callback_fixtures
  implicit none

  call check_band_user_forwarding()
  call check_minmode_user_forwarding()
  call check_pafi_user_forwarding()
  call check_ohtst_user_forwarding()
  call check_nested_surface_reentrancy()
  call check_nested_point_reentrancy()
  write(*, '(a)') 'PASS callbacks and re-entrancy'

contains

  subroutine check_band_user_forwarding()
    integer, parameter :: n_images = 9
    type(surface_ctx), target :: ctx, c_ctx
    type(rgsaddle_band_config_t) :: config
    type(band_handle) :: session, omitted, raw_c
    type(rgsaddle_report_t) :: report
    real(c_double) :: start(3, n_images), band(3, n_images), zero_start(3, n_images)
    real(c_double) :: t, err
    procedure(surface_proc), pointer :: cb, null_cb
    integer(c_int) :: status
    integer :: i, step, ci

    ctx%magic = SURFACE_CTX_MAGIC
    ctx%self_ptr = c_loc(ctx)
    ctx%center = [0.25_c_double, -0.15_c_double, 0.10_c_double]
    ctx%scale = 1.5_c_double
    ctx%calls = 0_c_int64_t

    do i = 1, n_images
      t = real(i - 1, c_double) / real(n_images - 1, c_double)
      start(1, i) = ctx%center(1) - 1.0_c_double + 2.0_c_double * t
      start(2, i) = ctx%center(2) + 0.4_c_double * sin(acos(-1.0_c_double) * t)
      start(3, i) = ctx%center(3)
      zero_start(1, i) = -1.0_c_double + 2.0_c_double * t
      zero_start(2, i) = 0.4_c_double * sin(acos(-1.0_c_double) * t)
      zero_start(3, i) = 0.0_c_double
    end do

    call init_band_config(config)
    config%force_tol = 1.0e-4_c_double
    config%max_move = 0.1_c_double
    config%ci_trigger_force = 0.5_c_double

    status = session%create(config, int(n_images, c_int64_t), 1_c_int64_t, start)
    call check_status("create band user", status)
    cb => shifted_surface
    report%status = RGSADDLE_STATUS_RUNNING
    do step = 1, 4000
      status = session%step(cb, c_loc(ctx), report)
      call check_status("band user step", status)
      if (report%status == RGSADDLE_STATUS_CONVERGED) exit
    end do
    if (report%status /= RGSADDLE_STATUS_CONVERGED .or. ctx%calls < 1_c_int64_t) then
      write(*, '(a,1x,i0,1x,i0)') 'FAIL band user convergence', report%status, ctx%calls
      stop 2
    end if
    status = session%positions(band)
    call check_status("band user positions", status)
    ci = int(report%ci_index, kind=kind(0)) + 1
    err = sum(abs(band(:, ci) - ctx%center))
    if (err > 2.0e-2_c_double) then
      write(*, '(a,1x,3f12.6)') 'FAIL band user saddle', band(:, ci)
      stop 2
    end if

    status = omitted%create(config, int(n_images, c_int64_t), 1_c_int64_t, zero_start)
    call check_status("create band omitted user", status)
    null_cb => null_user_surface
    status = omitted%step(null_cb, out=report)
    call check_status("band omitted user step", status)

    c_ctx%magic = SURFACE_CTX_MAGIC
    c_ctx%self_ptr = c_loc(c_ctx)
    c_ctx%center = ctx%center
    c_ctx%scale = 1.2_c_double
    c_ctx%calls = 0_c_int64_t
    status = raw_c%create(config, int(n_images, c_int64_t), 1_c_int64_t, start)
    call check_status("create band step_c user", status)
    status = raw_c%step_c(c_funloc(c_shifted_surface), c_loc(c_ctx), report)
    call check_status("band step_c user", status)
    if (c_ctx%calls < 1_c_int64_t) stop 2
  end subroutine check_band_user_forwarding

  subroutine check_minmode_user_forwarding()
    type(surface_ctx), target :: ctx, c_ctx
    type(rgsaddle_minmode_config_t) :: config
    type(minmode_handle) :: session, omitted, raw_c
    type(rgsaddle_report_t) :: report
    real(c_double) :: start(3), seed(3), pos(3), zero_start(3)
    procedure(surface_proc), pointer :: cb, null_cb
    integer(c_int) :: status
    integer :: step

    ctx%magic = SURFACE_CTX_MAGIC
    ctx%self_ptr = c_loc(ctx)
    ctx%center = [-0.20_c_double, 0.30_c_double, -0.10_c_double]
    ctx%scale = 2.0_c_double
    ctx%calls = 0_c_int64_t

    start = ctx%center + [0.35_c_double, 0.20_c_double, -0.15_c_double]
    zero_start = [0.35_c_double, 0.20_c_double, -0.15_c_double]
    seed = [0.8_c_double, 0.2_c_double, 0.1_c_double]

    call init_minmode_config(config, int(RGSADDLE_MINMODE_DIMER, c_int32_t))
    config%force_tol = 1.0e-3_c_double
    config%max_move = 0.1_c_double

    status = session%create(config, 1_c_int64_t, start, seed)
    call check_status("create minmode user", status)
    cb => shifted_surface
    report%status = RGSADDLE_STATUS_RUNNING
    do step = 1, 4000
      status = session%step(cb, c_loc(ctx), report)
      call check_status("minmode user step", status)
      if (report%status == RGSADDLE_STATUS_CONVERGED) exit
    end do
    if (report%status /= RGSADDLE_STATUS_CONVERGED .or. ctx%calls < 1_c_int64_t) then
      write(*, '(a,1x,i0,1x,i0)') 'FAIL minmode user convergence', report%status, ctx%calls
      stop 3
    end if
    status = session%position(pos)
    call check_status("minmode user position", status)
    if (sum(abs(pos - ctx%center)) > 2.0e-2_c_double) then
      write(*, '(a,1x,3f12.6)') 'FAIL minmode user saddle', pos
      stop 3
    end if

    status = omitted%create(config, 1_c_int64_t, zero_start, seed)
    call check_status("create minmode omitted user", status)
    null_cb => null_user_surface
    status = omitted%step(null_cb, out=report)
    call check_status("minmode omitted user step", status)

    c_ctx%magic = SURFACE_CTX_MAGIC
    c_ctx%self_ptr = c_loc(c_ctx)
    c_ctx%center = ctx%center
    c_ctx%scale = 1.5_c_double
    c_ctx%calls = 0_c_int64_t
    status = raw_c%create(config, 1_c_int64_t, start, seed)
    call check_status("create minmode step_c user", status)
    status = raw_c%step_c(c_funloc(c_shifted_surface), c_loc(c_ctx), report)
    call check_status("minmode step_c user", status)
    if (c_ctx%calls < 1_c_int64_t) stop 3
  end subroutine check_minmode_user_forwarding

  subroutine check_pafi_user_forwarding()
    integer, parameter :: n_frames = 21
    type(point_ctx), target :: ctx, c_ctx
    type(rgsaddle_pafi_config_t) :: config
    type(rgsaddle_pafi_result_t) :: result
    type(pafi_handle) :: session, omitted, raw_c
    real(c_double) :: frames(n_frames), zero_frames(n_frames)
    procedure(point_surface_proc), pointer :: cb, null_cb
    integer(c_int) :: status
    integer :: i

    ctx%magic = POINT_CTX_MAGIC
    ctx%self_ptr = c_loc(ctx)
    ctx%scale = 2.5_c_double
    ctx%shift = 0.25_c_double

    do i = 1, n_frames
      zero_frames(i) = -1.0_c_double + 2.0_c_double * real(i - 1, c_double) / real(n_frames - 1, c_double)
      frames(i) = zero_frames(i) + ctx%shift
    end do

    call init_pafi_config(config)
    config%thermal = 0.02_c_double
    config%mobility = 1.0_c_double
    config%time_step = 0.01_c_double
    config%thermalize = 5_c_int64_t
    config%stride = 1_c_int64_t
    config%samples = 60_c_int64_t
    config%blocks = 5_c_int64_t
    config%seed = 7_c_int64_t
    config%fix_com = 0_c_int32_t
    config%retain = 1_c_int64_t
    config%target_error = 0.0_c_double
    config%max_rounds = 1_c_int64_t

    status = session%create(config, int(n_frames, c_int64_t), 1_c_int64_t, frames)
    call check_status("create pafi user", status)
    cb => scaled_pafi_surface
    status = session%run(cb, c_loc(ctx))
    call check_status("pafi user run", status)
    status = session%result(result)
    call check_status("pafi user result", status)
    if (abs(result%barrier - ctx%scale) > 5.0e-2_c_double) then
      write(*, '(a,1x,f12.6,1x,f12.6)') 'FAIL pafi user barrier', result%barrier, ctx%scale
      stop 4
    end if

    status = omitted%create(config, int(n_frames, c_int64_t), 1_c_int64_t, zero_frames)
    call check_status("create pafi omitted user", status)
    null_cb => null_user_point_surface
    status = omitted%run(null_cb)
    call check_status("pafi omitted user run", status)

    c_ctx%magic = POINT_CTX_MAGIC
    c_ctx%self_ptr = c_loc(c_ctx)
    c_ctx%scale = 3.25_c_double
    c_ctx%shift = ctx%shift
    status = raw_c%create(config, int(n_frames, c_int64_t), 1_c_int64_t, frames)
    call check_status("create pafi run_c user", status)
    status = raw_c%run_c(c_funloc(c_scaled_point_surface), c_loc(c_ctx))
    call check_status("pafi run_c user", status)
    status = raw_c%result(result)
    call check_status("pafi run_c result", status)
    if (abs(result%barrier - c_ctx%scale) > 5.0e-2_c_double) stop 4
  end subroutine check_pafi_user_forwarding

  subroutine check_ohtst_user_forwarding()
    type(point_ctx), target :: ctx, c_ctx
    type(rgsaddle_ohtst_config_t) :: config
    type(rgsaddle_ohtst_result_t) :: result
    type(ohtst_handle) :: session, omitted, raw_c
    real(c_double) :: point(2), zero_point(2), normal(2)
    real(c_double) :: omega_b, eta, kramers, margin
    procedure(point_surface_proc), pointer :: cb, null_cb
    integer(c_int) :: status

    ctx%magic = POINT_CTX_MAGIC
    ctx%self_ptr = c_loc(ctx)
    ctx%scale = 2.25_c_double
    ctx%shift = 0.30_c_double

    point = [ctx%shift, 0.0_c_double]
    zero_point = [0.0_c_double, 0.0_c_double]
    normal = [1.0_c_double, 0.0_c_double]

    call init_ohtst_config(config)
    config%thermal = 0.5_c_double
    config%sample_time_step = 0.1_c_double
    config%sample_friction = 1.0_c_double
    config%sample_burn_in = 200_c_int64_t
    config%sample_stride = 4_c_int64_t
    config%sample_records = 600_c_int64_t
    config%sample_blocks = 12_c_int64_t
    config%sample_seed = 9_c_int64_t
    config%sample_retain = 256_c_int64_t
    config%shooting_friction = 1.5_c_double
    config%shooting_time_step = 0.05_c_double
    config%shooting_steps = 300_c_int64_t
    config%shooting_check_every = 30_c_int64_t
    config%shooting_commit = 0_c_int64_t
    config%shooting_shots = 2048_c_int64_t
    config%shooting_blocks = 16_c_int64_t
    config%shooting_seed = 17_c_int64_t

    status = session%create(config, 2_c_int64_t, point, normal)
    call check_status("create ohtst user", status)
    cb => scaled_ohtst_surface
    status = session%run(cb, c_loc(ctx))
    call check_status("ohtst user run", status)
    status = session%result(result)
    call check_status("ohtst user result", status)

    omega_b = sqrt(ctx%scale)
    eta = config%shooting_friction / (2.0_c_double * omega_b)
    kramers = sqrt(1.0_c_double + eta * eta) - eta
    margin = 4.0_c_double * result%error + epsilon(1.0_c_double)
    if (abs(result%kappa - kramers) > margin) then
      write(*, '(a,1x,f12.6,1x,f12.6,1x,f12.6)') 'FAIL ohtst user kappa', result%kappa, kramers, margin
      stop 5
    end if

    config%sample_burn_in = 10_c_int64_t
    config%sample_records = 32_c_int64_t
    config%sample_blocks = 4_c_int64_t
    config%sample_retain = 16_c_int64_t
    config%shooting_steps = 40_c_int64_t
    config%shooting_check_every = 10_c_int64_t
    config%shooting_shots = 64_c_int64_t
    config%shooting_blocks = 4_c_int64_t
    status = omitted%create(config, 2_c_int64_t, zero_point, normal)
    call check_status("create ohtst omitted user", status)
    null_cb => null_user_point_surface
    status = omitted%run(null_cb)
    call check_status("ohtst omitted user run", status)

    c_ctx%magic = POINT_CTX_MAGIC
    c_ctx%self_ptr = c_loc(c_ctx)
    c_ctx%scale = ctx%scale
    c_ctx%shift = ctx%shift
    status = raw_c%create(config, 2_c_int64_t, point, normal)
    call check_status("create ohtst run_c user", status)
    status = raw_c%run_c(c_funloc(c_scaled_point_surface), c_loc(c_ctx))
    call check_status("ohtst run_c user", status)
  end subroutine check_ohtst_user_forwarding

  subroutine check_nested_surface_reentrancy()
    integer, parameter :: n_images = 7
    type(nested_surface_ctx), target :: outer
    type(rgsaddle_band_config_t) :: config
    type(band_handle) :: session
    type(rgsaddle_report_t) :: report
    real(c_double) :: start(3, n_images), band(3, n_images), t
    procedure(surface_proc), pointer :: cb
    integer(c_int) :: status
    integer :: i, step, ci

    outer%magic = OUTER_CTX_MAGIC
    outer%self_ptr = c_loc(outer)
    outer%target_x = 0.15_c_double
    outer%outer_calls = 0_c_int64_t
    outer%inner_band_calls = 0_c_int64_t
    outer%inner_minmode_calls = 0_c_int64_t

    do i = 1, n_images
      t = real(i - 1, c_double) / real(n_images - 1, c_double)
      start(1, i) = outer%target_x - 1.0_c_double + 2.0_c_double * t
      start(2, i) = 0.35_c_double * sin(acos(-1.0_c_double) * t)
      start(3, i) = 0.0_c_double
    end do

    call init_band_config(config)
    config%force_tol = 1.0e-3_c_double
    config%max_move = 0.1_c_double
    config%ci_trigger_force = 0.5_c_double

    status = session%create(config, int(n_images, c_int64_t), 1_c_int64_t, start)
    call check_status("create nested outer band", status)
    cb => nested_outer_band_surface
    report%status = RGSADDLE_STATUS_RUNNING
    do step = 1, 200
      status = session%step(cb, c_loc(outer), report)
      call check_status("nested outer band step", status)
      if (report%status == RGSADDLE_STATUS_CONVERGED) exit
    end do
    if (report%status /= RGSADDLE_STATUS_CONVERGED) then
      write(*, '(a,1x,i0)') 'FAIL nested outer band convergence', step
      stop 6
    end if
    if (outer%outer_calls < 2_c_int64_t .or. outer%inner_band_calls < 2_c_int64_t .or. &
        outer%inner_minmode_calls < 4_c_int64_t) then
      write(*, '(a,3(1x,i0))') 'FAIL nested call counts', outer%outer_calls, &
        outer%inner_band_calls, outer%inner_minmode_calls
      stop 6
    end if
    status = session%positions(band)
    call check_status("nested outer band positions", status)
    ci = int(report%ci_index, kind=kind(0)) + 1
    if (abs(band(1, ci) - outer%target_x) > 2.0e-2_c_double .or. &
        abs(band(2, ci)) > 2.0e-2_c_double .or. abs(band(3, ci)) > 2.0e-2_c_double) then
      write(*, '(a,1x,3f12.6)') 'FAIL nested outer band saddle', band(:, ci)
      stop 6
    end if
  end subroutine check_nested_surface_reentrancy

  subroutine check_nested_point_reentrancy()
    integer, parameter :: n_frames = 21
    type(point_ctx), target :: pafi_outer, ohtst_outer
    type(rgsaddle_pafi_config_t) :: pcfg
    type(rgsaddle_pafi_result_t) :: pres
    type(pafi_handle) :: pafi
    type(rgsaddle_ohtst_config_t) :: ocfg
    type(rgsaddle_ohtst_result_t) :: ores
    type(ohtst_handle) :: ohtst
    real(c_double) :: frames(n_frames), pt(2), nrm(2)
    procedure(point_surface_proc), pointer :: pcb, ocb
    integer(c_int) :: status
    integer :: i

    pafi_outer%magic = POINT_CTX_MAGIC
    pafi_outer%self_ptr = c_loc(pafi_outer)
    pafi_outer%scale = 1.5_c_double
    pafi_outer%shift = 0.1_c_double
    do i = 1, n_frames
      frames(i) = pafi_outer%shift - 1.0_c_double + 2.0_c_double * real(i - 1, c_double) / real(n_frames - 1, c_double)
    end do
    call init_pafi_config(pcfg)
    pcfg%thermal = 0.02_c_double
    pcfg%time_step = 0.01_c_double
    pcfg%thermalize = 1_c_int64_t
    pcfg%stride = 1_c_int64_t
    pcfg%samples = 4_c_int64_t
    pcfg%blocks = 2_c_int64_t
    pcfg%seed = 11_c_int64_t
    pcfg%fix_com = 0_c_int32_t
    pcfg%retain = 1_c_int64_t
    pcfg%max_rounds = 1_c_int64_t
    status = pafi%create(pcfg, int(n_frames, c_int64_t), 1_c_int64_t, frames)
    call check_status("create nested outer pafi", status)
    pcb => nested_outer_pafi_surface
    status = pafi%run(pcb, c_loc(pafi_outer))
    call check_status("run nested outer pafi", status)
    status = pafi%result(pres)
    call check_status("result nested outer pafi", status)
    if (abs(pres%barrier - pafi_outer%scale) > 5.0e-2_c_double) then
      write(*, '(a,1x,f12.6)') 'FAIL nested outer pafi barrier', pres%barrier
      stop 7
    end if

    ohtst_outer%magic = POINT_CTX_MAGIC
    ohtst_outer%self_ptr = c_loc(ohtst_outer)
    ohtst_outer%scale = 1.0_c_double
    ohtst_outer%shift = 0.2_c_double
    pt = [ohtst_outer%shift, 0.0_c_double]
    nrm = [1.0_c_double, 0.0_c_double]
    call init_ohtst_config(ocfg)
    ocfg%thermal = 0.5_c_double
    ocfg%sample_burn_in = 2_c_int64_t
    ocfg%sample_stride = 1_c_int64_t
    ocfg%sample_records = 6_c_int64_t
    ocfg%sample_blocks = 2_c_int64_t
    ocfg%sample_retain = 3_c_int64_t
    ocfg%shooting_steps = 6_c_int64_t
    ocfg%shooting_check_every = 3_c_int64_t
    ocfg%shooting_shots = 6_c_int64_t
    ocfg%shooting_blocks = 2_c_int64_t
    status = ohtst%create(ocfg, 2_c_int64_t, pt, nrm)
    call check_status("create nested outer ohtst", status)
    ocb => nested_outer_ohtst_surface
    status = ohtst%run(ocb, c_loc(ohtst_outer))
    call check_status("run nested outer ohtst", status)
    status = ohtst%result(ores)
    call check_status("result nested outer ohtst", status)
    if (ores%evaluations < 1_c_int64_t) stop 7
  end subroutine check_nested_point_reentrancy

end program test_callbacks
