program test_ohtst
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type(ohtst_handle) :: session, tilted
  type(rgsaddle_ohtst_config_t) :: config
  type(rgsaddle_ohtst_result_t) :: result, tilted_result
  real(c_double) :: point(2), normal(2), tilted_normal(2)
  real(c_double) :: times(32), curve(32)
  real(c_double) :: kramers, theta, geometric, expected_tilted
  real(c_double) :: margin
  integer(c_int64_t) :: written
  integer(c_int) :: status
  procedure(point_surface_proc), pointer :: cb

  call init_ohtst_config(config)
  config%thermal = 0.5_c_double
  config%sample_time_step = 0.1_c_double
  config%sample_friction = 1.0_c_double
  config%sample_burn_in = 300_c_int64_t
  config%sample_stride = 5_c_int64_t
  config%sample_records = 1000_c_int64_t
  config%sample_blocks = 16_c_int64_t
  config%sample_seed = 9_c_int64_t
  config%sample_retain = 512_c_int64_t
  config%sample_integrator = int(RGSADDLE_OHTST_INTEGRATOR_BAOAB, c_int32_t)
  config%shooting_friction = 1.5_c_double
  config%shooting_time_step = 0.05_c_double
  config%shooting_steps = 400_c_int64_t
  config%shooting_check_every = 40_c_int64_t
  config%shooting_commit = 0_c_int64_t
  config%shooting_shots = 4096_c_int64_t
  config%shooting_blocks = 32_c_int64_t
  config%shooting_seed = 17_c_int64_t
  config%counting = int(RGSADDLE_OHTST_COUNT_FINAL, c_int32_t)

  point = [0.0_c_double, 0.0_c_double]
  normal = [1.0_c_double, 0.0_c_double]
  status = session%create(config, 2_c_int64_t, point, normal)
  call check_status("create ohtst", status)
  cb => ohtst_quadratic_surface
  status = session%run(cb, c_null_ptr)
  call check_status("run ohtst", status)
  status = session%result(result)
  call check_status("result ohtst", status)

  kramers = sqrt(1.0_c_double + (0.75_c_double * 0.75_c_double)) - 0.75_c_double
  margin = 4.0_c_double * result%error + epsilon(1.0_c_double)
  if (abs(result%kappa - kramers) > margin) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL ohtst kappa', result%kappa, &
      'expected', kramers, 'margin', margin
    stop 2
  end if

  status = session%curve(times, curve, written)
  call check_status("curve ohtst", status)
  if (written < 1_c_int64_t) then
    write(*, '(a)') 'FAIL ohtst empty curve'
    stop 3
  end if
  if (abs(curve(written) - kramers) > margin) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL ohtst curve end', curve(written), 'expected', kramers
    stop 3
  end if

  theta = 25.0_c_double * acos(-1.0_c_double) / 180.0_c_double
  tilted_normal = [cos(theta), sin(theta)]
  status = tilted%create(config, 2_c_int64_t, point, tilted_normal)
  call check_status("create tilted ohtst", status)
  status = tilted%run(cb, c_null_ptr)
  call check_status("run tilted ohtst", status)
  status = tilted%result(tilted_result)
  call check_status("result tilted ohtst", status)
  geometric = sqrt(cos(theta) * cos(theta) - 0.25_c_double * sin(theta) * sin(theta))
  expected_tilted = kramers * geometric
  margin = 4.0_c_double * tilted_result%error + epsilon(1.0_c_double)
  if (abs(tilted_result%kappa - expected_tilted) > margin) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL tilted ohtst kappa', &
      tilted_result%kappa, 'expected', expected_tilted, 'margin', margin
    stop 4
  end if

  config%sample_integrator = int(RGSADDLE_OHTST_INTEGRATOR_GHMC, c_int32_t)
  config%counting = int(RGSADDLE_OHTST_COUNT_FIRST_CROSSING, c_int32_t)
  status = tilted%set_config(config)
  call check_status("set_config tilted ohtst", status)
  status = tilted%run(cb, c_null_ptr)
  call check_status("run tilted ghmc ohtst", status)
  status = tilted%result(tilted_result)
  call check_status("result tilted ghmc ohtst", status)
  margin = 4.0_c_double * tilted_result%error + epsilon(1.0_c_double)
  if (abs(tilted_result%kappa - expected_tilted) > margin) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL tilted ghmc kappa', &
      tilted_result%kappa, 'expected', expected_tilted, 'margin', margin
    stop 5
  end if

  write(*, '(a,1x,f10.6,1x,a,1x,f10.6)') 'PASS ohtst kappa', result%kappa, 'tilted', tilted_result%kappa
end program test_ohtst
