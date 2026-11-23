program test_handle_container_assign
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type :: band_owner
    type(band_handle) :: session
  end type band_owner

  type :: minmode_owner
    type(minmode_handle) :: session
  end type minmode_owner

  type :: pafi_owner
    type(pafi_handle) :: session
  end type pafi_owner

  type :: ohtst_owner
    type(ohtst_handle) :: session
  end type ohtst_owner

  character(len=32) :: which
  if (command_argument_count() /= 1) then
    write(*, '(a)') 'FAIL usage: test_handle_container_assign band|minmode|pafi|ohtst'
    stop 2
  end if
  call get_command_argument(1, which)
  select case (trim(which))
  case ('band')
    call assign_band_owner()
  case ('minmode')
    call assign_minmode_owner()
  case ('pafi')
    call assign_pafi_owner()
  case ('ohtst')
    call assign_ohtst_owner()
  case default
    write(*, '(a)') 'FAIL unknown handle'
    stop 2
  end select
  write(*, '(a)') 'FAIL container assignment returned'
  stop 3

contains

  subroutine assign_band_owner()
    type(band_owner) :: a, b
    type(rgsaddle_band_config_t) :: config
    real(c_double) :: images(3, 3)
    integer(c_int) :: status
    images = 0.0_c_double
    images(1, 2) = 0.5_c_double
    images(1, 3) = 1.0_c_double
    call init_band_config(config)
    status = a%session%create(config, 3_c_int64_t, 1_c_int64_t, images)
    call check_status("band container create", status)
    b = a
    call a%session%release()
    call b%session%release()
  end subroutine assign_band_owner

  subroutine assign_minmode_owner()
    type(minmode_owner) :: a, b
    type(rgsaddle_minmode_config_t) :: config
    real(c_double) :: point(3), mode(3)
    integer(c_int) :: status
    point = 0.0_c_double
    mode = [1.0_c_double, 0.0_c_double, 0.0_c_double]
    call init_minmode_config(config)
    status = a%session%create(config, 1_c_int64_t, point, mode)
    call check_status("minmode container create", status)
    b = a
    call a%session%release()
    call b%session%release()
  end subroutine assign_minmode_owner

  subroutine assign_pafi_owner()
    type(pafi_owner) :: a, b
    type(rgsaddle_pafi_config_t) :: config
    real(c_double) :: frames(2)
    integer(c_int) :: status
    frames = [-1.0_c_double, 1.0_c_double]
    call init_pafi_config(config)
    status = a%session%create(config, 2_c_int64_t, 1_c_int64_t, frames)
    call check_status("pafi container create", status)
    b = a
    call a%session%release()
    call b%session%release()
  end subroutine assign_pafi_owner

  subroutine assign_ohtst_owner()
    type(ohtst_owner) :: a, b
    type(rgsaddle_ohtst_config_t) :: config
    real(c_double) :: point(2), normal(2)
    integer(c_int) :: status
    point = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]
    call init_ohtst_config(config)
    status = a%session%create(config, 2_c_int64_t, point, normal)
    call check_status("ohtst container create", status)
    b = a
    call a%session%release()
    call b%session%release()
  end subroutine assign_ohtst_owner

end program test_handle_container_assign
