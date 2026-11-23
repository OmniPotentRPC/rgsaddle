program test_ohtst
  use, intrinsic :: iso_c_binding
  use rgsaddle
  implicit none

  type(ohtst_handle) :: session
  integer(c_int) :: status

  status = session%create()
  if (status /= RGSADDLE_INVALID_PARAMETER) then
    write(*, '(a,1x,i0)') 'FAIL oh-tst expected unsupported status, got', status
    stop 1
  end if

  write(*, '(a)') 'PASS oh-tst unsupported in current C ABI'
end program test_ohtst
