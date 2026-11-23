program test_neb
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  integer, parameter :: n_images = 11
  integer(c_int64_t), parameter :: n_atoms = 1_c_int64_t
  real(c_double), parameter :: saddle_tol = 2.0_c_double / 100.0_c_double
  type(rgsaddle_band_config_t) :: config
  type(band_handle) :: session
  type(rgsaddle_report_t) :: report
  real(c_double) :: start(3, n_images)
  real(c_double) :: band(3, n_images)
  real(c_double) :: climbed
  integer(c_int) :: status
  integer :: step, i, ci
  procedure(surface_proc), pointer :: cb

  call fill_double_well_band(start, n_images)
  call init_band_config(config)
  config%force_tol = 1.0_c_double / 10000.0_c_double
  config%max_move = 0.1_c_double
  config%ci_trigger_force = 0.5_c_double

  status = session%create(config, int(n_images, c_int64_t), n_atoms, start)
  call check_status("create neb", status)
  call reset_one_image_counts()
  cb => batched_climbing_double_well
  report%status = RGSADDLE_STATUS_RUNNING
  do step = 1, 4000
    status = session%step(cb, c_null_ptr, report)
    call check_status("neb step", status)
    if (report%status == RGSADDLE_STATUS_CONVERGED) exit
  end do
  if (report%status /= RGSADDLE_STATUS_CONVERGED) then
    write(*, '(a,1x,i0)') 'FAIL neb did not converge', step
    stop 2
  end if
  if (report%ci_index < 1 .or. report%ci_index > n_images - 2) then
    write(*, '(a,1x,i0)') 'FAIL neb climbing index', report%ci_index
    stop 2
  end if
  if (batched_climbing_hits < 1 .or. batched_climbing_bad /= 0) then
    write(*, '(a,1x,i0,1x,i0)') 'FAIL batched climbing requests', &
      batched_climbing_hits, batched_climbing_bad
    stop 2
  end if

  band = 42.0_c_double
  status = session%positions(band)
  call check_status("neb positions", status)
  ci = int(report%ci_index, kind=kind(0)) + 1
  climbed = abs(band(1, ci)) + abs(band(2, ci)) + abs(band(3, ci))
  if (climbed > saddle_tol) then
    write(*, '(a,1x,i0,1x,3f12.6)') 'FAIL neb saddle', ci, band(:, ci)
    stop 3
  end if
  if (abs(band(1, 1) + 1.0_c_double) > saddle_tol .or. abs(band(1, n_images) - 1.0_c_double) > saddle_tol) then
    write(*, '(a,1x,3f12.6,1x,3f12.6)') 'FAIL neb endpoints', band(:, 1), band(:, n_images)
    stop 3
  end if
  do i = 1, n_images
    if (abs(band(1, i) - 42.0_c_double) < 1.0_c_double) then
      write(*, '(a,1x,i0)') 'FAIL neb positions unchanged', i
      stop 4
    end if
    ! The spring (k = 5) balances a shallow residual bow. The initial
    ! path bows out to 0.4, so this is still far inside that arc.
    if (abs(band(2, i)) > 5.0_c_double * saddle_tol .or. abs(band(3, i)) > saddle_tol) then
      write(*, '(a,1x,i0,1x,3f12.6)') 'FAIL neb left the x axis', i, band(:, i)
      stop 4
    end if
  end do

  call check_one_image(config, start)
  write(*, '(a,1x,i0,1x,a,1x,i0)') 'PASS neb after', step, 'steps climbing', ci

contains

  subroutine check_one_image(cfg, initial)
    type(rgsaddle_band_config_t), intent(in) :: cfg
    real(c_double), intent(in) :: initial(3, n_images)
    type(rgsaddle_band_config_t) :: local
    type(band_handle) :: per
    type(rgsaddle_report_t) :: rep
    real(c_double) :: out(3, n_images)
    integer(c_int) :: st
    integer :: k
    procedure(surface_proc), pointer :: pcb
    local = cfg
    local%flags = ior(local%flags, RGSADDLE_BAND_PER_IMAGE)
    st = per%create(local, int(n_images, c_int64_t), n_atoms, initial)
    call check_status("create per-image neb", st)
    call reset_one_image_counts()
    pcb => per_image_double_well
    do k = 1, 3
      st = per%step(pcb, c_null_ptr, rep)
      call check_status("per-image neb step", st)
    end do
    if (one_image_hits < 1 .or. one_image_bad /= 0) then
      write(*, '(a,1x,i0,1x,i0)') 'FAIL one-image requests', one_image_hits, one_image_bad
      stop 5
    end if
    out = 42.0_c_double
    st = per%positions(out)
    call check_status("per-image positions", st)
    if (maxval(abs(out - initial)) < 1.0_c_double / 1000.0_c_double) then
      write(*, '(a)') 'FAIL per-image band did not move'
      stop 5
    end if
  end subroutine check_one_image

end program test_neb
