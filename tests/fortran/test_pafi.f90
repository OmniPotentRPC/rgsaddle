program test_pafi
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  type(pafi_handle) :: session, widening
  type(rgsaddle_pafi_config_t) :: config
  type(rgsaddle_pafi_result_t) :: result, w_result
  integer, parameter :: n_frames = 41
  integer, parameter :: n_wide = 11
  real(c_double) :: frames(n_frames), arcs(n_frames)
  real(c_double) :: grad(n_frames), grad_err(n_frames), free_energy(n_frames), free_error(n_frames)
  real(c_double) :: w_frames(2, n_wide), w_arcs(n_wide), w_grad(n_wide), w_grad_err(n_wide)
  real(c_double) :: w_free(n_wide), w_free_err(n_wide)
  real(c_double) :: x, exact_grad, exact_free, truth, miss
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
    frames(i) = -1.0_c_double + 2.0_c_double * real(i - 1, c_double) / real(n_frames - 1, c_double)
    arcs(i) = 2.0_c_double * real(i - 1, c_double) / real(n_frames - 1, c_double)
  end do

  status = session%create(config, int(n_frames, c_int64_t), 1_c_int64_t, frames, arcs)
  call check_status("create pafi", status)
  cb => pafi_along_x_surface
  status = session%run(cb, c_null_ptr)
  call check_status("run pafi", status)
  status = session%result(result)
  call check_status("result pafi", status)
  if (result%dividing /= 20_c_int64_t) then
    write(*, '(a,1x,i0)') 'FAIL pafi dividing index', result%dividing
    stop 2
  end if
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
  do i = 1, n_frames
    x = frames(i)
    exact_grad = -4.0_c_double * x * (1.0_c_double - x * x)
    exact_free = (1.0_c_double - x * x) * (1.0_c_double - x * x)
    if (abs(grad(i) - exact_grad) > 1.0e-10_c_double .or. &
        abs(free_energy(i) - exact_free) > 5.0e-3_c_double) then
      write(*, '(a,1x,i0,1x,f12.6,1x,f12.6)') 'FAIL pafi profile point', i, free_energy(i), exact_free
      stop 4
    end if
  end do

  call init_pafi_config(config)
  config%thermal = 0.05_c_double
  config%mobility = 1.0_c_double
  config%time_step = 0.004_c_double
  config%thermalize = 400_c_int64_t
  config%stride = 8_c_int64_t
  config%samples = 600_c_int64_t
  config%blocks = 6_c_int64_t
  config%seed = 11_c_int64_t
  config%fix_com = 0_c_int32_t
  config%retain = 2_c_int64_t
  config%target_error = 0.0_c_double
  config%max_rounds = 1_c_int64_t

  do i = 1, n_wide
    w_frames(1, i) = -1.0_c_double + real(i - 1, c_double) / real(n_wide - 1, c_double)
    w_frames(2, i) = 0.0_c_double
  end do

  status = widening%create(config, int(n_wide, c_int64_t), 2_c_int64_t, w_frames)
  call check_status("create widening pafi", status)
  cb => pafi_widening_surface
  status = widening%run(cb, c_null_ptr)
  call check_status("run widening pafi", status)
  status = widening%result(w_result)
  call check_status("result widening pafi", status)
  truth = 1.0_c_double + 0.5_c_double * config%thermal * log(1.5_c_double)
  miss = abs(w_result%barrier - truth)
  if (.not. (miss < 2.0e-2_c_double .and. miss < 4.0_c_double * max(w_result%barrier_error, 5.0e-3_c_double))) then
    write(*, '(a,1x,f12.6,1x,a,1x,f12.6,1x,a,1x,f12.6)') 'FAIL widening barrier', w_result%barrier, &
      'truth', truth, 'error', w_result%barrier_error
    stop 5
  end if
  status = widening%profile(w_arcs, w_grad, w_grad_err, w_free, w_free_err, written)
  call check_status("profile widening pafi", status)
  if (written /= int(n_wide, c_int64_t)) stop 6
  do i = 1, n_wide
    x = w_frames(1, i)
    exact_free = (1.0_c_double - x * x) * (1.0_c_double - x * x) + &
      0.5_c_double * config%thermal * log(1.5_c_double + 0.5_c_double * x)
    if (abs(w_free(i) - exact_free) > 4.0_c_double * max(w_free_err(i), 5.0e-3_c_double)) then
      write(*, '(a,1x,i0,1x,f12.6,1x,f12.6)') 'FAIL widening profile', i, w_free(i), exact_free
      stop 7
    end if
  end do

  write(*, '(a,1x,f10.6,1x,a,1x,f10.6)') 'PASS pafi barrier', result%barrier, 'widening', w_result%barrier
end program test_pafi
