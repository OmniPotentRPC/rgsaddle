program test_handle_leaks
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  integer :: i
  type(band_handle) :: session
  type(rgsaddle_band_config_t) :: config
  real(c_double) :: images(3, 3)
  integer(c_int) :: status

  images = 0.0_c_double
  images(1, 2) = 0.5_c_double
  images(1, 3) = 1.0_c_double
  call init_band_config(config)

  do i = 1, 100
    status = session%create(config, 3_c_int64_t, 1_c_int64_t, images)
    call check_status("leak create", status)
    call session%release()
  end do

  write(*, '(a)') 'PASS handle leak loop'
end program test_handle_leaks
