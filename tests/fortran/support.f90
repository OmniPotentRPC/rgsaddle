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

  public :: fill_linear_band, fortran_surface, c_surface, pafi_along_x_surface, &
    pafi_widening_surface, ohtst_quadratic_surface, close_enough, check_status

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

  integer(c_int) function fortran_surface(user, n_images, n_atoms, positions, energies, gradients, image) result(status)
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

  integer(c_int) function c_surface(user, req_ptr) bind(C) result(status)
    type(c_ptr), value :: user
    type(c_ptr), value :: req_ptr
    type(rgsaddle_surface_request_t), pointer :: req
    real(c_double), pointer :: pos(:)
    real(c_double), pointer :: ene(:)
    real(c_double), pointer :: grad(:)
    integer(c_int64_t) :: npos, neval
    integer :: i
    real(c_double) :: row(3), g(3), energy
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

  integer(c_int) function pafi_along_x_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    integer :: i
    real(c_double) :: x
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

  integer(c_int) function pafi_widening_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
    real(c_double) :: x, y, well, stiffness
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

  integer(c_int) function ohtst_quadratic_surface(user, n_dof, position, energy, gradient) result(status)
    type(c_ptr), value :: user
    integer(c_int64_t), value :: n_dof
    real(c_double), intent(in) :: position(*)
    real(c_double), intent(inout) :: energy
    real(c_double), intent(inout) :: gradient(*)
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

end module rgsaddle_fortran_test_support
