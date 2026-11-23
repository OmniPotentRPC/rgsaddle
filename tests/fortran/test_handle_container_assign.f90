program test_handle_container_assign
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type :: band_owner
    type(band_handle) :: session
  end type band_owner

  type(band_owner) :: a, b
  type(rgsaddle_band_config_t) :: config
  real(c_double) :: images(3, 3)
  integer(c_int) :: status

  images = 0.0_c_double
  images(1, 2) = 0.5_c_double
  images(1, 3) = 1.0_c_double
  call init_band_config(config)
  status = a%session%create(config, 3_c_int64_t, 1_c_int64_t, images)
  call check_status("container create", status)

  b = a
  call a%session%release()
  call b%session%release()

  write(*, '(a)') 'PASS handle container assignment'
end program test_handle_container_assign
