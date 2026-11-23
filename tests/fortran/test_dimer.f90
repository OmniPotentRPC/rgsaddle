program test_dimer
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  call run_minmode(int(RGSADDLE_MINMODE_DIMER, c_int32_t), 'dimer')
end program test_dimer
