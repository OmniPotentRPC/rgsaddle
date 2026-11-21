module rgsaddle_fortran_test_support
  use, intrinsic :: iso_c_binding
  use rgsaddle
  implicit none
  private

  real(c_double), parameter :: prefactor(4) = [ -200.0_c_double, -100.0_c_double, -170.0_c_double, 15.0_c_double ]
  real(c_double), parameter :: xx(4) = [ -1.0_c_double, -1.0_c_double, -6.5_c_double, 0.7_c_double ]
  real(c_double), parameter :: xy(4) = [ 0.0_c_double, 0.0_c_double, 11.0_c_double, 0.6_c_double ]
  real(c_double), parameter :: yy(4) = [ -10.0_c_double, -10.0_c_double, -6.5_c_double, 0.7_c_double ]
  real(c_double), parameter :: x0(4) = [ 1.0_c_double, 0.0_c_double, -0.5_c_double, -1.0_c_double ]
  real(c_double), parameter :: y0(4) = [ 0.0_c_double, 0.5_c_double, 1.5_c_double, 1.0_c_double ]

  integer, save :: one_image_hits = 0
  integer, save :: one_image_bad = 0

  public :: fill_linear_band, fill_double_well_band, fortran_surface, c_surface, &
    double_well_surface, per_image_double_well, reset_one_image_counts, &
    one_image_hits, one_image_bad, pafi_along_x_surface, &
    pafi_widening_surface, ohtst_quadratic_surface, close_enough, check_status, &
    check_i64, run_minmode, assert_minmode_defaults

contains

  subroutine muller_brown(x, y, energy, gx, gy)
    real(c_double), value, intent(in) :: x, y
    real(c_double), intent(out) :: energy, gx, gy
    integer :: i
    real(c_double) :: dx, dy, term
    energy = 0.0_c_double
    gx = 0.0_c_double
    gy = 0.0_c_double
    do i = 1, 4
      dx = x - x0(i)
      dy = y - y0(i)
      term = prefactor(i) * exp(xx(i) * dx * dx + xy(i) * dx * dy + yy(i) * dy * dy)
      energy = energy + term
      gx = gx + term * (2.0_c_double * xx(i) * dx + xy(i) * dy)
      gy = gy + term * (xy(i) * dx + 2.0_c_double * yy(i) * dy)
    end do
  end subroutine muller_brown

  subroutine eval_point(row, energy, gradient)
    real(c_double), intent(in) :: row(3)
    real(c_double), intent(out) :: energy
    real(c_double), intent(out) :: gradient(3)
    real(c_double) :: gx, gy
    call muller_brown(row(1), row(2), energy, gx, gy)
    energy = energy + 2.0_c_double * row(3) * row(3)
    gradient(1) = gx
    gradient(2) = gy
    gradient(3) = 4.0_c_double * row(3)
  end subroutine eval_point

  subroutine reset_one_image_counts()
    one_image_hits = 0
    one_image_bad = 0
  end subroutine reset_one_image_counts

  subroutine fill_double_well_band(positions, n_images)
    integer, value, intent(in) :: n_images
    real(c_double), intent(out) :: positions(3, n_images)
    integer :: i
    real(c_double) :: t
    do i = 1, n_images
      t = real(i - 1, c_double) / real(n_images - 1, c_double)
      positions(1, i) = -1.0_c_double + 2.0_c_double * t
      positions(2, i) = 0.4_c_double * sin(acos(-1.0_c_double) * t)
      positions(3, i) = 0.0_c_double
    end do
  end subroutine fill_double_well_band

  subroutine double_well_point(row, energy, gradient)
    real(c_double), intent(in) :: row(3)
    real(c_double), intent(out) :: energy
    real(c_double), intent(out) :: gradient(3)
    real(c_double) :: x
    x = row(1)
    energy = (x * x - 1.0_c_double) * (x * x - 1.0_c_double) + &
      2.0_c_double * row(2) * row(2) + 2.0_c_double * row(3) * row(3)
    gradient(1) = 4.0_c_double * x * (x * x - 1.0_c_double)
    gradient(2) = 4.0_c_double * row(2)
    gradient(3) = 4.0_c_double * row(3)
  end subroutine double_well_point

  subroutine fill_linear_band(positions, n_images)
    integer, value, intent(in) :: n_images
    real(c_double), intent(out) :: positions(3, n_images)
    integer :: i
    real(c_double) :: t
    real(c_double), parameter :: a(3) = [ -0.558223634633_c_double, 1.44172584195_c_double, 0.0_c_double ]
    real(c_double), parameter :: b(3) = [ 0.623499404931_c_double, 0.028037758529_c_double, 0.0_c_double ]
    do i = 1, n_images
      t = real(i - 1, c_double) / real(n_images - 1, c_double)
      positions(:, i) = (1.0_c_double - t) * a + t * b
    end do
  end subroutine fill_linear_band

  recursive integer(c_int) function fortran_surface(user, n_images, n_atoms, positions, energies, gradients, image) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    real(c_double), intent(inout) :: energies(*)
    real(c_double), intent(inout) :: gradients(*)
    integer(c_int64_t), value :: image
    integer :: i
    integer :: n_eval
    real(c_double) :: row(3), grad(3), energy
    if (c_associated(user)) continue
    status = RGSADDLE_OK
    if (n_atoms /= 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_images == 1_c_int64_t .and. image >= 0_c_int64_t) then
      n_eval = 1
    else
      n_eval = int(n_images, kind=kind(0))
    end if
    do i = 1, n_eval
      row = positions(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_point(row, energy, grad)
      energies(i) = energy
      gradients(3 * (i - 1) + 1:3 * (i - 1) + 3) = grad
    end do
  end function fortran_surface

  recursive integer(c_int) function double_well_surface(user, n_images, n_atoms, &
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
    if (c_associated(user)) continue
    if (n_atoms /= 1_c_int64_t .or. n_images < 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (n_images > int(huge(0), c_int64_t)) then
      status = RGSADDLE_SHAPE
      return
    end if
    n_eval = int(n_images, kind=kind(0))
    do i = 1, n_eval
      row = positions(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call double_well_point(row, energy, grad)
      energies(i) = energy
      gradients(3 * (i - 1) + 1:3 * (i - 1) + 3) = grad
    end do
    if (image < -1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    status = RGSADDLE_OK
  end function double_well_surface

  recursive integer(c_int) function per_image_double_well(user, n_images, n_atoms, &
      positions, energies, gradients, image) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_images
    integer(c_int64_t), value :: n_atoms
    real(c_double), intent(in) :: positions(*)
    real(c_double), intent(inout) :: energies(*)
    real(c_double), intent(inout) :: gradients(*)
    integer(c_int64_t), value :: image
    if (image >= 0_c_int64_t .and. n_images /= 1_c_int64_t) then
      one_image_bad = one_image_bad + 1
      status = RGSADDLE_SURFACE_FAILED
      return
    end if
    if (image >= 0_c_int64_t) one_image_hits = one_image_hits + 1
    status = double_well_surface(user, n_images, n_atoms, positions, energies, gradients, image)
  end function per_image_double_well

  recursive integer(c_int) function c_surface(user, req_ptr) bind(C) result(status)
    type(c_ptr), value :: user
    type(c_ptr), value :: req_ptr
    type(rgsaddle_surface_request_t), pointer :: req
    real(c_double), pointer :: pos(:)
    real(c_double), pointer :: ene(:)
    real(c_double), pointer :: grad(:)
    integer(c_int64_t) :: npos, neval
    integer :: i
    real(c_double) :: row(3), g(3), energy
    if (c_associated(user)) continue
    call c_f_pointer(req_ptr, req)
    if (.not. associated(req)) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    if (iand(req%flags, 1_c_int64_t) /= 0_c_int64_t) then
      npos = 3_c_int64_t * req%n_atoms
      neval = 1_c_int64_t
    else
      npos = req%n_images * 3_c_int64_t * req%n_atoms
      neval = req%n_images
    end if
    call c_f_pointer(req%positions, pos, [int(npos, kind=kind(0))])
    call c_f_pointer(req%energies, ene, [int(neval, kind=kind(0))])
    call c_f_pointer(req%gradients, grad, [int(npos, kind=kind(0))])
    status = RGSADDLE_OK
    do i = 1, int(neval, kind=kind(0))
      row = pos(3 * (i - 1) + 1:3 * (i - 1) + 3)
      call eval_point(row, energy, g)
      ene(i) = energy
      grad(3 * (i - 1) + 1:3 * (i - 1) + 3) = g
    end do
  end function c_surface

  recursive integer(c_int) function pafi_along_x_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    integer :: i
    real(c_double) :: x
    if (c_associated(user)) continue
    if (n_dof < 1_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    x = position(1)
    energy = (1.0_c_double - x * x) * (1.0_c_double - x * x)
    do i = 1, int(n_dof, kind=kind(0))
      gradient(i) = 0.0_c_double
    end do
    gradient(1) = -4.0_c_double * x * (1.0_c_double - x * x)
    status = RGSADDLE_OK
  end function pafi_along_x_surface

  recursive integer(c_int) function pafi_widening_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    real(c_double) :: x, y, well, stiffness
    if (c_associated(user)) continue
    if (n_dof /= 2_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    x = position(1)
    y = position(2)
    well = (1.0_c_double - x * x) * (1.0_c_double - x * x)
    stiffness = 1.5_c_double + 0.5_c_double * x
    energy = well + 0.5_c_double * stiffness * y * y
    gradient(1) = -4.0_c_double * x * (1.0_c_double - x * x) + 0.25_c_double * y * y
    gradient(2) = stiffness * y
    status = RGSADDLE_OK
  end function pafi_widening_surface

  recursive integer(c_int) function ohtst_quadratic_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    if (c_associated(user)) continue
    if (n_dof /= 2_c_int64_t) then
      status = RGSADDLE_INVALID_PARAMETER
      return
    end if
    energy = 0.5_c_double * (-position(1) * position(1) + 4.0_c_double * position(2) * position(2))
    gradient(1) = -position(1)
    gradient(2) = 4.0_c_double * position(2)
    status = RGSADDLE_OK
  end function ohtst_quadratic_surface

  logical function close_enough(a, b, scale) result(ok)
    real(c_double), value, intent(in) :: a, b, scale
    real(c_double) :: tol
    tol = scale * sqrt(epsilon(1.0_c_double))
    ok = abs(a - b) <= tol * max(1.0_c_double, abs(a), abs(b))
  end function close_enough

  subroutine check_status(label, status)
    character(len=*), intent(in) :: label
    integer(c_int), value, intent(in) :: status
    if (status /= RGSADDLE_OK) then
      write(*, '(a,1x,a,1x,i0,1x,a)') 'FAIL', trim(label), status, trim(status_name(status))
      stop 1
    end if
  end subroutine check_status

  subroutine assert_minmode_defaults()
    type(rgsaddle_minmode_config_t) :: config
    real(c_double) :: angle
    call init_minmode_config(config)
    angle = 5.0_c_double * acos(-1.0_c_double) / 180.0_c_double
    if (abs(config%dr - 1.0_c_double / 1000.0_c_double) > epsilon(1.0_c_double)) stop 8
    if (abs(config%rotation_tol - 1.0_c_double / 10000.0_c_double) > epsilon(1.0_c_double)) stop 8
    if (config%max_rotations /= 20_c_int64_t) stop 8
    if (config%krylov_dim /= 12_c_int64_t) stop 8
    if (abs(config%force_tol - 1.0_c_double / 1000.0_c_double) > epsilon(1.0_c_double)) stop 8
    if (abs(config%max_move - 0.2_c_double) > epsilon(1.0_c_double)) stop 8
    if (abs(config%rotation_angle_tol - angle) > 1.0_c_double / 1000000.0_c_double) then
      write(*, '(a,1x,f16.8)') 'FAIL rotation_angle_tol', config%rotation_angle_tol
      stop 8
    end if
    if (config%rotation_angle_tol <= 0.0_c_double) stop 8
  end subroutine assert_minmode_defaults

  subroutine run_minmode(kind, label)
    integer(c_int32_t), value, intent(in) :: kind
    character(len=*), intent(in) :: label
    type(rgsaddle_minmode_config_t) :: config
    type(minmode_handle) :: session, moved
    type(rgsaddle_report_t) :: report
    real(c_double) :: start(3), seed(3), position(3), mode(3)
    real(c_double) :: gap
    integer(c_int) :: status
    integer :: step
    procedure(surface_proc), pointer :: cb

    call assert_minmode_defaults()
    start = [0.35_c_double, 0.2_c_double, -0.15_c_double]
    seed = [0.8_c_double, 0.2_c_double, 0.1_c_double]
    call init_minmode_config(config, kind)
    config%force_tol = 1.0_c_double / 1000.0_c_double
    config%max_move = 0.1_c_double
    status = session%create(config, 1_c_int64_t, start, seed)
    call check_status("create " // trim(label), status)
    cb => double_well_surface
    report%status = RGSADDLE_STATUS_RUNNING
    do step = 1, 4000
      status = session%step(cb, c_null_ptr, report)
      call check_status(trim(label) // " step", status)
      if (report%status == RGSADDLE_STATUS_CONVERGED) exit
    end do
    if (report%status /= RGSADDLE_STATUS_CONVERGED) then
      write(*, '(a,1x,a,1x,i0)') 'FAIL', trim(label), step
      stop 2
    end if
    if (.not. (report%curvature < 0.0_c_double)) then
      write(*, '(a,1x,a,1x,f12.6)') 'FAIL curvature', trim(label), report%curvature
      stop 2
    end if
    position = 42.0_c_double
    mode = -42.0_c_double
    status = session%position(position)
    call check_status(trim(label) // " position", status)
    status = session%mode(mode)
    call check_status(trim(label) // " mode", status)
    if (abs(position(1)) > 2.0_c_double / 100.0_c_double .or. &
        abs(position(2)) > 2.0_c_double / 100.0_c_double .or. &
        abs(position(3)) > 2.0_c_double / 100.0_c_double) then
      write(*, '(a,1x,a,1x,3f12.6)') 'FAIL saddle', trim(label), position
      stop 3
    end if
    if (abs(mode(1)) < 0.9_c_double .or. abs(mode(2)) > 0.2_c_double .or. abs(mode(3)) > 0.2_c_double) then
      write(*, '(a,1x,a,1x,3f12.6)') 'FAIL mode', trim(label), mode
      stop 3
    end if
    gap = abs(position(1) - mode(1)) + abs(position(2) - mode(2)) + abs(position(3) - mode(3))
    if (gap < 0.5_c_double) then
      write(*, '(a,1x,a)') 'FAIL position matches mode', trim(label)
      stop 3
    end if
    call moved%move_from(session)
    status = session%position(position)
    call check_i64(trim(label) // " moved-from", status, RGSADDLE_NULL_SESSION)
    status = moved%position(position)
    call check_status(trim(label) // " moved", status)
    if (abs(position(1)) > 2.0_c_double / 100.0_c_double) stop 4
    write(*, '(a,1x,a,1x,i0,1x,a)') 'PASS', trim(label), step, 'steps'
  end subroutine run_minmode

  subroutine check_i64(label, got, expect)
    character(len=*), intent(in) :: label
    integer(c_int), value, intent(in) :: got
    integer(c_int), value, intent(in) :: expect
    if (got /= expect) then
      write(*, '(a,1x,a,1x,i0,1x,a,1x,i0,1x,a)') 'FAIL', trim(label), got, &
        'expected', expect, trim(status_name(got))
      stop 1
    end if
  end subroutine check_i64

end module rgsaddle_fortran_test_support
