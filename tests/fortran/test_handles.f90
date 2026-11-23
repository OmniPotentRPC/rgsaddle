program test_handles
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  call check_new_statuses()
  call check_create_status()
  call check_set_config_status()
  call check_default_configs_match_rust()
  call check_create_keeps_live_handle_on_invalid_args()
  call check_band_release()
  call check_moves()
  call check_trampoline_shape()
  write(*, '(a)') 'PASS handles'

contains

  subroutine check_new_statuses()
    if (RGSADDLE_NOT_EQUILIBRATED /= -14_c_int) stop 1
    if (RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES /= -15_c_int) stop 1
  end subroutine check_new_statuses

  subroutine check_create_status()
    type(rgsaddle_band_config_t) :: band_cfg
    type(rgsaddle_minmode_config_t) :: mode_cfg
    type(rgsaddle_pafi_config_t) :: pafi_cfg
    type(rgsaddle_ohtst_config_t) :: ohtst_cfg
    type(band_handle) :: band
    type(minmode_handle) :: mode
    type(pafi_handle) :: pafi
    type(ohtst_handle) :: ohtst
    real(c_double) :: images(3, 3), point(3), guess(3), frames(2), plane(2), normal(2)
    integer(c_int) :: status
    images = 0.0_c_double
    images(1, 2) = 0.5_c_double
    images(1, 3) = 1.0_c_double
    point = 0.0_c_double
    guess = [1.0_c_double, 0.0_c_double, 0.0_c_double]
    frames = [-1.0_c_double, 1.0_c_double]
    plane = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]

    call init_band_config(band_cfg)
    band_cfg%version%major = 9_c_int32_t
    status = band%create(band_cfg, 3_c_int64_t, 1_c_int64_t, images)
    call check_i64("band major", status, RGSADDLE_ABI_MISMATCH)
    call init_band_config(band_cfg)
    status = band%create(band_cfg, 2_c_int64_t, 1_c_int64_t, images)
    call check_i64("band shape", status, RGSADDLE_SHAPE)

    call init_minmode_config(mode_cfg)
    mode_cfg%version%major = 4_c_int32_t
    status = mode%create(mode_cfg, 1_c_int64_t, point, guess)
    call check_i64("minmode major", status, RGSADDLE_ABI_MISMATCH)
    call init_minmode_config(mode_cfg)
    status = mode%create(mode_cfg, 0_c_int64_t, point, guess)
    call check_i64("minmode shape", status, RGSADDLE_SHAPE)

    call init_pafi_config(pafi_cfg)
    pafi_cfg%version%major = 3_c_int32_t
    status = pafi%create(pafi_cfg, 2_c_int64_t, 1_c_int64_t, frames)
    call check_i64("pafi major", status, RGSADDLE_ABI_MISMATCH)
    call init_pafi_config(pafi_cfg)
    status = pafi%create(pafi_cfg, 1_c_int64_t, 1_c_int64_t, frames)
    call check_i64("pafi shape", status, RGSADDLE_SHAPE)

    call init_ohtst_config(ohtst_cfg)
    ohtst_cfg%version%major = 0_c_int32_t
    status = ohtst%create(ohtst_cfg, 2_c_int64_t, plane, normal)
    call check_i64("ohtst major", status, RGSADDLE_ABI_MISMATCH)
    call init_ohtst_config(ohtst_cfg)
    status = ohtst%create(ohtst_cfg, 0_c_int64_t, plane, normal)
    call check_i64("ohtst shape", status, RGSADDLE_SHAPE)
  end subroutine check_create_status

  subroutine check_set_config_status()
    type(pafi_handle) :: pafi, pafi_null
    type(ohtst_handle) :: ohtst, ohtst_null
    type(rgsaddle_pafi_config_t) :: pafi_cfg, pafi_other
    type(rgsaddle_ohtst_config_t) :: ohtst_cfg, ohtst_other
    real(c_double) :: frames(2), point(2), normal(2)
    integer(c_int) :: status

    frames = [-1.0_c_double, 1.0_c_double]
    point = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]

    call init_pafi_config(pafi_cfg)
    pafi_other = pafi_cfg
    pafi_other%version%major = 9_c_int32_t
    status = pafi_null%set_config(pafi_cfg)
    call check_i64("pafi set config null handle", status, RGSADDLE_NULL_SESSION)
    status = pafi_null%set_config(pafi_other)
    call check_i64("pafi set config null handle major", status, RGSADDLE_NULL_SESSION)
    status = pafi%create(pafi_cfg, 2_c_int64_t, 1_c_int64_t, frames)
    call check_status("pafi set config create", status)
    status = pafi%set_config(pafi_other)
    call check_i64("pafi set config major", status, RGSADDLE_ABI_MISMATCH)
    status = pafi%set_config(pafi_cfg)
    call check_status("pafi set config", status)
    call pafi%release()
    status = pafi%set_config(pafi_other)
    call check_i64("pafi set config released major", status, RGSADDLE_NULL_SESSION)

    call init_ohtst_config(ohtst_cfg)
    ohtst_other = ohtst_cfg
    ohtst_other%version%major = 9_c_int32_t
    status = ohtst_null%set_config(ohtst_cfg)
    call check_i64("ohtst set config null handle", status, RGSADDLE_NULL_SESSION)
    status = ohtst_null%set_config(ohtst_other)
    call check_i64("ohtst set config null handle major", status, RGSADDLE_NULL_SESSION)
    status = ohtst%create(ohtst_cfg, 2_c_int64_t, point, normal)
    call check_status("ohtst set config create", status)
    status = ohtst%set_config(ohtst_other)
    call check_i64("ohtst set config major", status, RGSADDLE_ABI_MISMATCH)
    status = ohtst%set_config(ohtst_cfg)
    call check_status("ohtst set config", status)
    call ohtst%release()
    status = ohtst%set_config(ohtst_other)
    call check_i64("ohtst set config released major", status, RGSADDLE_NULL_SESSION)
  end subroutine check_set_config_status

  subroutine check_band_release()
    integer(c_int64_t), parameter :: n_atoms = 20000_c_int64_t
    integer(c_int64_t), parameter :: n_images = 3_c_int64_t
    type, bind(C) :: mallinfo2_t
      integer(c_size_t) :: arena, ordblks, smblks, hblks, hblkhd
      integer(c_size_t) :: usmblks, fsmblks, uordblks, fordblks, keepcost
    end type mallinfo2_t
    interface
      type(mallinfo2_t) function c_mallinfo2() bind(C, name="mallinfo2")
        import :: mallinfo2_t
      end function c_mallinfo2
    end interface
    type(band_handle) :: session
    type(rgsaddle_band_config_t) :: config
    type(mallinfo2_t) :: before, created, released
    real(c_double), allocatable :: positions(:, :, :)
    real(c_double) :: probe(3)
    integer(c_size_t) :: grew, dropped
    integer(c_int) :: status
    allocate(positions(3, n_atoms, n_images))
    positions = 0.0_c_double
    positions(1, :, 1) = 0.0_c_double
    positions(1, :, 2) = 0.5_c_double
    positions(1, :, 3) = 1.0_c_double
    call init_band_config(config)
    before = c_mallinfo2()
    status = session%create(config, n_images, n_atoms, positions)
    call check_status("create large band", status)
    created = c_mallinfo2()
    grew = (created%uordblks + created%hblkhd) - (before%uordblks + before%hblkhd)
    if (grew < 500000_c_size_t) then
      write(*, '(a,1x,i0)') 'FAIL band allocation not visible', grew
      stop 2
    end if
    call session%release()
    released = c_mallinfo2()
    dropped = (created%uordblks + created%hblkhd) - (released%uordblks + released%hblkhd)
    if (dropped * 2_c_size_t < grew) then
      write(*, '(a,1x,i0,1x,i0)') 'FAIL band_free kept the session', grew, dropped
      stop 2
    end if
    status = session%positions(probe)
    call check_i64("released band", status, RGSADDLE_NULL_BAND)
  end subroutine check_band_release

  subroutine check_default_configs_match_rust()
    type(rgsaddle_band_config_t) :: bcfg
    type(rgsaddle_ohtst_config_t) :: ocfg
    type(rgsaddle_pafi_config_t) :: pcfg
    type(rgsaddle_minmode_config_t) :: mcfg
    call init_band_config(bcfg)
    call check_i64("band flags", int(bcfg%flags, c_int), int(RGSADDLE_BAND_CI_AFTER, c_int))
    if (.not. close_enough(bcfg%ci_trigger_factor, 0.0_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(bcfg%ci_trigger_force, 0.5_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(bcfg%force_tol, 1.0e-3_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(bcfg%max_move, 0.2_c_double, 1.0_c_double)) stop 6

    call init_ohtst_config(ocfg)
    if (.not. close_enough(ocfg%sample_time_step, 0.05_c_double, 1.0_c_double)) stop 6
    if (ocfg%sample_burn_in /= 500_c_int64_t) stop 6
    if (ocfg%sample_records /= 2000_c_int64_t) stop 6
    if (ocfg%sample_blocks /= 20_c_int64_t) stop 6
    if (ocfg%sample_retain /= 16_c_int64_t) stop 6
    if (.not. close_enough(ocfg%shooting_friction, 0.0_c_double, 1.0_c_double)) stop 6
    if (ocfg%shooting_shots /= 256_c_int64_t) stop 6
    if (ocfg%shooting_seed /= 1_c_int64_t) stop 6
    if (ocfg%counting /= int(RGSADDLE_OHTST_COUNT_FIRST_CROSSING, c_int32_t)) stop 6

    call init_pafi_config(pcfg)
    if (.not. close_enough(pcfg%thermal, 0.02585_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(pcfg%mobility, 1.0_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(pcfg%time_step, 1.0e-3_c_double, 1.0_c_double)) stop 6
    if (pcfg%thermalize /= 200_c_int64_t) stop 6
    if (pcfg%stride /= 5_c_int64_t) stop 6
    if (pcfg%samples /= 200_c_int64_t) stop 6
    if (pcfg%blocks /= 5_c_int64_t) stop 6
    if (pcfg%seed /= 1_c_int64_t) stop 6
    if (pcfg%fix_com /= 1_c_int32_t) stop 6
    if (pcfg%retain /= 16_c_int64_t) stop 6
    if (.not. close_enough(pcfg%target_error, 0.0_c_double, 1.0_c_double)) stop 6
    if (pcfg%max_rounds /= 4_c_int64_t) stop 6

    call init_minmode_config(mcfg)
    if (mcfg%kind /= int(RGSADDLE_MINMODE_DIMER, c_int32_t)) stop 6
    if (.not. close_enough(mcfg%dr, 1.0e-3_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(mcfg%rotation_tol, 1.0e-4_c_double, 1.0_c_double)) stop 6
    if (mcfg%max_rotations /= 20_c_int64_t) stop 6
    if (mcfg%krylov_dim /= 12_c_int64_t) stop 6
    if (.not. close_enough(mcfg%force_tol, 1.0e-3_c_double, 1.0_c_double)) stop 6
    if (.not. close_enough(mcfg%max_move, 0.2_c_double, 1.0_c_double)) stop 6
    if (mcfg%difference /= 0_c_int32_t) stop 6
    if (mcfg%method /= int(RGSADDLE_METHOD_FIRE, c_int32_t)) stop 6
    if (.not. close_enough(mcfg%rotation_angle_tol, &
        5.0_c_double * acos(-1.0_c_double) / 180.0_c_double, 1.0_c_double)) stop 6
  end subroutine check_default_configs_match_rust

  subroutine check_create_keeps_live_handle_on_invalid_args()
    type(band_handle) :: band
    type(minmode_handle) :: mode
    type(pafi_handle) :: pafi
    type(ohtst_handle) :: ohtst
    type(rgsaddle_band_config_t) :: bcfg
    type(rgsaddle_minmode_config_t) :: mcfg
    type(rgsaddle_pafi_config_t) :: pcfg
    type(rgsaddle_ohtst_config_t) :: ocfg
    type(rgsaddle_ohtst_result_t) :: oresult
    type(rgsaddle_pafi_result_t) :: presult
    real(c_double) :: images(3, 3), out(3, 3)
    real(c_double) :: point(3), seed(3), mode_out(3)
    real(c_double) :: frames(2), plane(2), normal(2)
    integer(c_int) :: status

    images = 0.0_c_double
    images(1, 2) = 0.5_c_double
    images(1, 3) = 1.0_c_double
    point = 0.0_c_double
    seed = [1.0_c_double, 0.0_c_double, 0.0_c_double]
    frames = [-1.0_c_double, 1.0_c_double]
    plane = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]

    call init_band_config(bcfg)
    status = band%create(bcfg, 3_c_int64_t, 1_c_int64_t, images)
    call check_status("band live create", status)
    status = band%create(bcfg, 2_c_int64_t, 1_c_int64_t, images)
    call check_i64("band invalid create", status, RGSADDLE_SHAPE)
    status = band%positions(out)
    call check_status("band alive after invalid create", status)

    call init_minmode_config(mcfg)
    status = mode%create(mcfg, 1_c_int64_t, point, seed)
    call check_status("minmode live create", status)
    status = mode%create(mcfg, 0_c_int64_t, point, seed)
    call check_i64("minmode invalid create", status, RGSADDLE_SHAPE)
    status = mode%mode(mode_out)
    call check_status("minmode alive after invalid create", status)

    call init_pafi_config(pcfg)
    status = pafi%create(pcfg, 2_c_int64_t, 1_c_int64_t, frames)
    call check_status("pafi live create", status)
    status = pafi%create(pcfg, 1_c_int64_t, 1_c_int64_t, frames)
    call check_i64("pafi invalid create", status, RGSADDLE_SHAPE)
    status = pafi%result(presult)
    call check_i64("pafi alive after invalid create", status, RGSADDLE_NO_EVALUATION)

    call init_ohtst_config(ocfg)
    status = ohtst%create(ocfg, 2_c_int64_t, plane, normal)
    call check_status("ohtst live create", status)
    status = ohtst%create(ocfg, 0_c_int64_t, plane, normal)
    call check_i64("ohtst invalid create", status, RGSADDLE_SHAPE)
    status = ohtst%result(oresult)
    call check_i64("ohtst alive after invalid create", status, RGSADDLE_NO_EVALUATION)
  end subroutine check_create_keeps_live_handle_on_invalid_args

  subroutine check_moves()
    call move_band()
    call move_pafi()
    call move_ohtst()
  end subroutine check_moves

  subroutine move_band()
    type(band_handle) :: src, dest
    type(rgsaddle_band_config_t) :: config
    real(c_double) :: images(3, 3), out(3, 3)
    integer(c_int) :: status
    images = 0.0_c_double
    images(1, 2) = 0.5_c_double
    images(1, 3) = 1.0_c_double
    call init_band_config(config)
    status = src%create(config, 3_c_int64_t, 1_c_int64_t, images)
    call check_status("move band create", status)
    call dest%move_from(src)
    status = src%positions(out)
    call check_i64("band source", status, RGSADDLE_NULL_BAND)
    status = dest%positions(out)
    call check_status("band dest", status)
    if (abs(out(1, 3) - 1.0_c_double) > 1.0_c_double / 1000000.0_c_double) stop 3
  end subroutine move_band

  subroutine move_pafi()
    type(pafi_handle) :: src, dest
    type(rgsaddle_pafi_config_t) :: config
    type(rgsaddle_pafi_result_t) :: result
    real(c_double) :: frames(2)
    integer(c_int) :: status
    frames = [-1.0_c_double, 1.0_c_double]
    call init_pafi_config(config)
    status = src%create(config, 2_c_int64_t, 1_c_int64_t, frames)
    call check_status("move pafi create", status)
    call dest%move_from(src)
    status = src%result(result)
    call check_i64("pafi source", status, RGSADDLE_NULL_SESSION)
    status = dest%result(result)
    call check_i64("pafi dest", status, RGSADDLE_NO_EVALUATION)
  end subroutine move_pafi

  subroutine move_ohtst()
    type(ohtst_handle) :: src, dest
    type(rgsaddle_ohtst_config_t) :: config
    type(rgsaddle_ohtst_result_t) :: result
    real(c_double) :: point(2), normal(2)
    integer(c_int) :: status
    point = 0.0_c_double
    normal = [1.0_c_double, 0.0_c_double]
    call init_ohtst_config(config)
    status = src%create(config, 2_c_int64_t, point, normal)
    call check_status("move ohtst create", status)
    call dest%move_from(src)
    status = src%result(result)
    call check_i64("ohtst source", status, RGSADDLE_NULL_SESSION)
    status = dest%result(result)
    call check_i64("ohtst dest", status, RGSADDLE_NO_EVALUATION)
  end subroutine move_ohtst

  subroutine check_trampoline_shape()
    interface
      integer(c_int) function rgsaddle_fortran_surface_trampoline(user, req) bind(C) result(status)
        import :: c_int, c_ptr
        type(c_ptr), value :: user
        type(c_ptr), value :: req
      end function rgsaddle_fortran_surface_trampoline
      integer(c_int) function rgsaddle_fortran_point_trampoline(user, n_dof, position, energy, gradient) &
          bind(C) result(status)
        import :: c_int, c_ptr, c_int64_t, c_double
        type(c_ptr), value :: user
        integer(c_int64_t), value :: n_dof
        type(c_ptr), value :: position
        real(c_double), intent(out) :: energy
        type(c_ptr), value :: gradient
      end function rgsaddle_fortran_point_trampoline
    end interface
    type(rgsaddle_surface_request_t), target :: req
    real(c_double) :: energy
    integer(c_int) :: status
    req%version%major = 1_c_int32_t
    req%version%minor = int(RGSADDLE_ABI_MINOR, c_int32_t)
    req%flags = RGSADDLE_REQ_ONE_IMAGE
    req%n_images = 4_c_int64_t
    req%n_atoms = 2_c_int64_t ** 30
    req%positions = c_null_ptr
    req%energies = c_null_ptr
    req%gradients = c_null_ptr
    req%image = 1_c_int64_t
    status = rgsaddle_fortran_surface_trampoline(c_null_ptr, c_loc(req))
    call check_i64("surface shape", status, RGSADDLE_SHAPE)
    energy = 1.0_c_double
    status = rgsaddle_fortran_point_trampoline(c_null_ptr, 2_c_int64_t ** 31, c_null_ptr, energy, c_null_ptr)
    call check_i64("point shape", status, RGSADDLE_SHAPE)
  end subroutine check_trampoline_shape

end program test_handles
