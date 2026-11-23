program test_lanczos
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  call run_minmode(int(RGSADDLE_MINMODE_LANCZOS, c_int32_t), 'lanczos')
end program test_lanczos
