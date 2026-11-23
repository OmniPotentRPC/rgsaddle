program test_assign
  use, intrinsic :: iso_c_binding
  use rgsaddle
  implicit none

  character(len=32) :: which
  if (command_argument_count() /= 1) then
    write(*, '(a)') 'FAIL usage: test_assign band|minmode|pafi|ohtst'
    stop 2
  end if
  call get_command_argument(1, which)
  select case (trim(which))
  case ('band')
    call assign_band()
  case ('minmode')
    call assign_minmode()
  case ('pafi')
    call assign_pafi()
  case ('ohtst')
    call assign_ohtst()
  case default
    write(*, '(a)') 'FAIL unknown handle'
    stop 2
  end select
  write(*, '(a)') 'FAIL assignment returned'
  stop 3

contains

  subroutine assign_band()
    type(band_handle) :: a, b
    type(rgsaddle_band_config_t) :: config
    real(c_double) :: images(3, 3)
    integer(c_int) :: status
    images = 0.0_c_double
    images(1, 2) = 1.0_c_double
    images(1, 3) = 2.0_c_double
    call init_band_config(config)
    status = a%create(config, 3_c_int64_t, 1_c_int64_t, images)
    if (status /= RGSADDLE_OK) stop 4
    b = a
  end subroutine assign_band

  subroutine assign_minmode()
    type(minmode_handle) :: a, b
    type(rgsaddle_minmode_config_t) :: config
    real(c_double) :: point(3), mode(3)
    integer(c_int) :: status
    point = 0.0_c_double
    mode = [1.0_c_double, 0.0_c_double, 0.0_c_double]
    call init_minmode_config(config)
    status = a%create(config, 1_c_int64_t, point, mode)
    if (status /= RGSADDLE_OK) stop 4
    b = a
  end subroutine assign_minmode

  subroutine assign_pafi()
    type(pafi_handle) :: a, b
    type(rgsaddle_pafi_config_t) :: config
    real(c_double) :: frames(2)
    integer(c_int) :: status
    frames = [-1.0_c_double, 1.0_c_double]
    call init_pafi_config(config)
    status = a%create(config, 2_c_int64_t, 1_c_int64_t, frames)
    if (status /= RGSADDLE_OK) stop 4
    b = a
  end subroutine assign_pafi

  subroutine assign_ohtst()
    type(ohtst_handle) :: a, b
    type(rgsaddle_ohtst_config_t) :: config
    real(c_double) :: point(2), normal(2)
    integer(c_int) :: status
    point = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]
    call init_ohtst_config(config)
    status = a%create(config, 2_c_int64_t, point, normal)
    if (status /= RGSADDLE_OK) stop 4
    b = a
  end subroutine assign_ohtst

end program test_assign
