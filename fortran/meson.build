fortran_moduledir = get_option('includedir') / 'rgsaddle' / 'fortran'

rgsaddle_fortran_lib = static_library(
    'rgsaddle_fortran',
    files('rgsaddle.f90'),
    include_directories: rgsaddle_inc,
    install: true,
)

rgsaddle_fortran_dep = declare_dependency(
    include_directories: include_directories('.'),
    link_with: rgsaddle_fortran_lib,
    version: meson.project_version(),
)
meson.override_dependency('rgsaddle-fortran', rgsaddle_fortran_dep)

if not meson.is_subproject()
    install_data(
        meson.current_build_dir() / 'rgsaddle.mod',
        install_dir: fortran_moduledir,
    )

    pkg = import('pkgconfig')
    pkg.generate(
        name: 'rgsaddle-fortran',
        description: 'Fortran 2008 ISO C binding interfaces for the rgsaddle C ABI',
        version: meson.project_version(),
        libraries: rgsaddle_fortran_lib,
        requires: ['rgsaddle-headers'],
        variables: [
            'moduledir=' + get_option('prefix') / fortran_moduledir,
        ],
    )

    cmake_dir = get_option('libdir') / 'cmake' / 'rgsaddleFortran'
    cfg = configuration_data()
    cfg.set('RGSADDLE_FORTRAN_VERSION', meson.project_version())
    cfg.set('RGSADDLE_FORTRAN_MODULE_DIR', get_option('prefix') / fortran_moduledir)
    cfg.set('RGSADDLE_FORTRAN_INCLUDE_DIR', get_option('prefix') / get_option('includedir'))
    cfg.set('RGSADDLE_FORTRAN_LIB_DIR', get_option('prefix') / get_option('libdir'))

    configure_file(
        input: files('cmake/rgsaddleFortranConfig.cmake.in'),
        output: 'rgsaddleFortranConfig.cmake',
        configuration: cfg,
        install: true,
        install_dir: cmake_dir,
    )
    configure_file(
        input: files('cmake/rgsaddleFortranConfigVersion.cmake.in'),
        output: 'rgsaddleFortranConfigVersion.cmake',
        configuration: cfg,
        install: true,
        install_dir: cmake_dir,
    )
endif
