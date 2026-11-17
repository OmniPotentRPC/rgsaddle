program test_lanczos
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type(rgsaddle_minmode_config_t) :: config
  type(minmode_handle) :: session_fortran, session_c
  type(rgsaddle_report_t) :: report_fortran, report_c
  real(c_double) :: start(3), seed(3)
  real(c_double) :: pos_fortran(3), pos_c(3), mode_fortran(3), mode_c(3)
  integer(c_int) :: status
  integer :: step
  procedure(surface_proc), pointer :: cb

  start = [0.35_c_double, 0.4_c_double, -0.3_c_double]
  seed = [0.8_c_double, 0.5_c_double, 0.1_c_double]
  call init_minmode_config(config, int(RGSADDLE_MINMODE_LANCZOS, c_int32_t))
  config%force_tol = 1.0_c_double / 1000.0_c_double
  config%max_move = 0.1_c_double
  config%krylov_dim = 20_c_int64_t

  status = session_fortran%create(config, 1_c_int64_t, start, seed)
  call check_status("create fortran lanczos", status)
  status = session_c%create(config, 1_c_int64_t, start, seed)
  call check_status("create c lanczos", status)

  cb => fortran_surface
  do step = 1, 4000
    status = session_fortran%step(cb, c_null_ptr, report_fortran)
    call check_status("fortran lanczos step", status)
    status = session_c%step_c(c_funloc(c_surface), c_null_ptr, report_c)
    call check_status("c lanczos step", status)
    if (report_fortran%status == RGSADDLE_STATUS_CONVERGED .and. &
        report_c%status == RGSADDLE_STATUS_CONVERGED) exit
  end do

  status = session_fortran%position(pos_fortran)
  call check_status("fortran lanczos position", status)
  status = session_c%position(pos_c)
  call check_status("c lanczos position", status)
  status = session_fortran%mode(mode_fortran)
  call check_status("fortran lanczos mode", status)
  status = session_c%mode(mode_c)
  call check_status("c lanczos mode", status)

  call compare_vectors(pos_fortran, pos_c)
  call compare_vectors(mode_fortran, mode_c)
  call compare_reports(report_fortran, report_c)
  write(*, '(a,1x,i0,1x,a)') 'PASS lanczos after', step, 'steps'

contains

  subroutine compare_vectors(left, right)
    real(c_double), intent(in) :: left(3), right(3)
    integer :: i
    do i = 1, 3
      if (.not. close_enough(left(i), right(i), 1024.0_c_double)) stop 2
    end do
  end subroutine compare_vectors

  subroutine compare_reports(left, right)
    type(rgsaddle_report_t), intent(in) :: left, right
    if (left%status /= right%status) stop 3
    if (.not. close_enough(left%max_force, right%max_force, 2048.0_c_double)) stop 3
    if (.not. close_enough(left%curvature, right%curvature, 2048.0_c_double)) stop 3
  end subroutine compare_reports

end program test_lanczos
