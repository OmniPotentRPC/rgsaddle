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
  ! A 1D path gives a deterministic series with 2.2 effective samples.
  config%blocks = 2_c_int64_t
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

  call check_arcs_and_config()
  call check_short_profile()
  write(*, '(a,1x,f10.6,1x,a,1x,f10.6)') 'PASS pafi barrier', result%barrier, 'widening', w_result%barrier

contains

  subroutine cheap_config(out)
    type(rgsaddle_pafi_config_t), intent(out) :: out
    call init_pafi_config(out)
    out%thermal = 0.02_c_double
    out%mobility = 1.0_c_double
    out%time_step = 0.01_c_double
    out%thermalize = 1_c_int64_t
    out%stride = 1_c_int64_t
    out%samples = 8_c_int64_t
    out%blocks = 2_c_int64_t
    out%seed = 3_c_int64_t
    out%fix_com = 0_c_int32_t
    out%retain = 1_c_int64_t
    out%target_error = 0.0_c_double
    out%max_rounds = 1_c_int64_t
  end subroutine cheap_config

  subroutine check_arcs_and_config()
    type(pafi_handle) :: partial, full, tuned
    type(rgsaddle_pafi_config_t) :: cfg
    type(rgsaddle_pafi_result_t) :: partial_result, full_result, first, second
    integer, parameter :: n = 11
    integer, parameter :: n_arc = 4
    real(c_double) :: frames(n), full_arcs(n), near(n_arc)
    real(c_double) :: bad(3), wide(2, n)
    real(c_double) :: g1(n), g2(n), free_e(n), free_err(n), grad_err(n)
    integer(c_int64_t) :: n_written
    integer(c_int) :: st
    integer :: k
    procedure(point_surface_proc), pointer :: pcb
    pcb => pafi_along_x_surface
    do k = 1, n
      frames(k) = -1.0_c_double + 2.0_c_double * real(k - 1, c_double) / real(n - 1, c_double)
      full_arcs(k) = 2.0_c_double * real(k - 1, c_double) / real(n - 1, c_double)
    end do
    near = [0.0_c_double, 0.05_c_double, 0.10_c_double, 0.15_c_double]
    bad = [0.0_c_double, 1.0_c_double, 0.5_c_double]
    call cheap_config(cfg)
    st = partial%create(cfg, int(n, c_int64_t), 1_c_int64_t, frames, bad)
    if (st == RGSADDLE_OK) then
      write(*, '(a,1x,i0)') 'FAIL invalid arcs accepted', st
      stop 8
    end if
    st = partial%create(cfg, int(n, c_int64_t), 1_c_int64_t, frames, near)
    call check_status("create partial arcs", st)
    st = partial%run(pcb, c_null_ptr)
    call check_status("run partial arcs", st)
    st = partial%result(partial_result)
    call check_status("result partial arcs", st)
    st = full%create(cfg, int(n, c_int64_t), 1_c_int64_t, frames, full_arcs)
    call check_status("create full arcs", st)
    st = full%run(pcb, c_null_ptr)
    call check_status("run full arcs", st)
    st = full%result(full_result)
    call check_status("result full arcs", st)
    if (.not. (partial_result%barrier < 0.4_c_double .and. full_result%barrier > 0.7_c_double)) then
      write(*, '(a,1x,f12.6,1x,f12.6)') 'FAIL arcs did not change the barrier', &
        partial_result%barrier, full_result%barrier
      stop 8
    end if

    ! A 1D path has no perpendicular direction, so the Langevin seed
    ! cannot move the tangent gradient. The widening well's x gradient
    ! depends on y^2, and the high bit of the seed is a uint64 pattern.
    pcb => pafi_widening_surface
    do k = 1, n
      wide(1, k) = -1.0_c_double + 2.0_c_double * real(k - 1, c_double) / real(n - 1, c_double)
      wide(2, k) = 0.0_c_double
    end do
    call cheap_config(cfg)
    cfg%thermal = 0.2_c_double
    cfg%time_step = 0.05_c_double
    cfg%thermalize = 100_c_int64_t
    cfg%stride = 10_c_int64_t
    cfg%samples = 64_c_int64_t
    cfg%blocks = 8_c_int64_t
    cfg%seed = uint64_bits(int(z'80000000', c_int64_t), 1_c_int64_t)
    st = tuned%create(cfg, int(n, c_int64_t), 2_c_int64_t, wide)
    call check_status("create high seed", st)
    st = tuned%run(pcb, c_null_ptr)
    call check_status("run high seed", st)
    st = tuned%profile(full_arcs, g1, grad_err, free_e, free_err, n_written)
    call check_status("profile high seed", st)
    cfg%seed = 1_c_int64_t
    st = tuned%set_config(cfg)
    call check_status("set seed", st)
    st = tuned%run(pcb, c_null_ptr)
    call check_status("run low seed", st)
    st = tuned%profile(full_arcs, g2, grad_err, free_e, free_err, n_written)
    call check_status("profile low seed", st)
    if (all(abs(g1 - g2) < 1.0_c_double / 1000000.0_c_double)) then
      write(*, '(a,1x,f16.8)') 'FAIL seed did not change the profile', maxval(abs(g1 - g2))
      stop 9
    end if

    cfg%samples = 64_c_int64_t
    st = tuned%set_config(cfg)
    call check_status("set few samples", st)
    st = tuned%run(pcb, c_null_ptr)
    call check_status("run few samples", st)
    st = tuned%result(first)
    call check_status("result few samples", st)
    cfg%samples = 640_c_int64_t
    st = tuned%set_config(cfg)
    call check_status("set many samples", st)
    st = tuned%run(pcb, c_null_ptr)
    call check_status("run many samples", st)
    st = tuned%result(second)
    call check_status("result many samples", st)
    if (second%evaluations < first%evaluations * 4_c_int64_t) then
      write(*, '(a,1x,i0,1x,i0)') 'FAIL set_config ignored', first%evaluations, second%evaluations
      stop 9
    end if
    cfg%thermal = -1.0_c_double
    st = tuned%set_config(cfg)
    if (st == RGSADDLE_OK) then
      write(*, '(a)') 'FAIL invalid config accepted'
      stop 9
    end if
  end subroutine check_arcs_and_config

  subroutine check_short_profile()
    real(c_double) :: one_arc(1), one_g(1), one_ge(1), one_f(1), one_fe(1)
    integer(c_int64_t) :: n_written
    integer(c_int) :: st
    n_written = -1_c_int64_t
    one_arc = 0.0_c_double
    st = session%profile(one_arc, one_g, one_ge, one_f, one_fe, n_written)
    if (st /= RGSADDLE_INVALID_PARAMETER) then
      write(*, '(a,1x,i0)') 'FAIL short profile status', st
      stop 10
    end if
    if (n_written /= int(n_frames, c_int64_t)) then
      write(*, '(a,1x,i0)') 'FAIL short profile written', n_written
      stop 10
    end if
  end subroutine check_short_profile

end program test_pafi
