program test_neb
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  integer, parameter :: n_images = 11
  integer(c_int64_t), parameter :: n_atoms = 1_c_int64_t
  type(rgsaddle_band_config_t) :: config
  type(band_handle) :: session_fortran, session_c
  type(rgsaddle_report_t) :: report_fortran, report_c
  real(c_double) :: start(3, n_images)
  real(c_double) :: band_fortran(3, n_images), band_c(3, n_images)
  integer(c_int) :: status
  integer :: step
  procedure(surface_proc), pointer :: cb

  call fill_linear_band(start, n_images)
  call init_band_config(config)
  config%max_move = 0.05_c_double
  config%ci_trigger_force = 0.5_c_double

  status = session_fortran%create(config, int(n_images, c_int64_t), n_atoms, start)
  call check_status("create fortran neb", status)
  status = session_c%create(config, int(n_images, c_int64_t), n_atoms, start)
  call check_status("create c neb", status)

  cb => fortran_surface
  do step = 1, 4000
    status = session_fortran%step(cb, c_null_ptr, report_fortran)
    call check_status("fortran neb step", status)
    status = session_c%step_c(c_funloc(c_surface), c_null_ptr, report_c)
    call check_status("c neb step", status)
    if (report_fortran%status == RGSADDLE_STATUS_CONVERGED .and. &
        report_c%status == RGSADDLE_STATUS_CONVERGED) exit
  end do

  status = session_fortran%positions(band_fortran)
  call check_status("fortran neb positions", status)
  status = session_c%positions(band_c)
  call check_status("c neb positions", status)

  call compare_bands(band_fortran, band_c)
  call compare_reports(report_fortran, report_c)
  write(*, '(a,1x,i0,1x,a)') 'PASS neb after', step, 'steps'

contains

  subroutine compare_bands(left, right)
    real(c_double), intent(in) :: left(3, n_images), right(3, n_images)
    integer :: i
    do i = 1, n_images
      if (.not. close_enough(left(1, i), right(1, i), 1024.0_c_double)) stop 2
      if (.not. close_enough(left(2, i), right(2, i), 1024.0_c_double)) stop 2
      if (.not. close_enough(left(3, i), right(3, i), 1024.0_c_double)) stop 2
    end do
  end subroutine compare_bands

  subroutine compare_reports(left, right)
    type(rgsaddle_report_t), intent(in) :: left, right
    if (left%status /= right%status) stop 3
    if (left%ci_index /= right%ci_index) stop 3
    if (.not. close_enough(left%max_force, right%max_force, 2048.0_c_double)) stop 3
    if (.not. close_enough(left%curvature, right%curvature, 2048.0_c_double)) stop 3
  end subroutine compare_reports

end program test_neb
