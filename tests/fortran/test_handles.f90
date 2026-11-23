program test_handles
  use, intrinsic :: iso_c_binding
  use rgsaddle
  use rgsaddle_fortran_test_support
  implicit none

  call check_new_statuses()
  call check_create_status()
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
    req%version%minor = 45_c_int32_t
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
