program test_pafi
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type(pafi_handle) :: session
  type(rgsaddle_pafi_config_t) :: config
  type(rgsaddle_pafi_result_t) :: result
  integer, parameter :: n_frames = 41
  real(c_double) :: frames(n_frames), arcs(n_frames)
  real(c_double) :: grad(n_frames), grad_err(n_frames), free_energy(n_frames), free_error(n_frames)
  integer(c_int64_t) :: written
  integer(c_int) :: status
  integer :: i
  procedure(point_surface_proc), pointer :: cb

  call init_pafi_config(config)
  config%thermal = 0.02_c_double
  config%mobility = 1.0_c_double
  config%time_step = 0.01_c_double
  config%thermalize = 5_c_int64_t
  config%stride = 1_c_int64_t
  config%samples = 200_c_int64_t
  config%blocks = 5_c_int64_t
  config%seed = 3_c_int64_t
  config%fix_com = 0_c_int32_t
  config%retain = 1_c_int64_t
  config%target_error = 0.0_c_double
  config%max_rounds = 1_c_int64_t

  do i = 1, n_frames
    frames(i) = -1.0_c_double + real(i - 1, c_double) / real(n_frames - 1, c_double)
    arcs(i) = 2.0_c_double * real(i - 1, c_double) / real(n_frames - 1, c_double)
  end do

  status = session%create(config, int(n_frames, c_int64_t), 1_c_int64_t, frames, arcs)
  call check_status("create pafi", status)
  cb => pafi_along_x_surface
  status = session%run(cb, c_null_ptr)
  call check_status("run pafi", status)
  status = session%result(result)
  call check_status("result pafi", status)
  if (.not. (abs(result%barrier - 1.0_c_double) < 2.0e-2_c_double .and. &
      abs(result%barrier - 1.0_c_double) < 4.0_c_double * max(result%barrier_error, 5.0e-3_c_double))) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL pafi barrier mismatch', result%barrier, &
      'error', result%barrier_error
    stop 2
  end if
  status = session%profile(arcs, grad, grad_err, free_energy, free_error, written)
  call check_status("profile pafi", status)
  if (written /= int(n_frames, c_int64_t)) then
    write(*, '(a,1x,i0)') 'FAIL pafi profile length', written
    stop 3
  end if
  if (.not. (abs(free_energy(n_frames)) < huge(1.0_c_double))) then
    write(*, '(a,1x,f12.6)') 'FAIL pafi terminal free energy is non-finite', free_energy(n_frames)
    stop 4
  end if

  write(*, '(a,1x,f10.6,1x,a)') 'PASS pafi barrier', result%barrier, 'on analytic line'
end program test_pafi
