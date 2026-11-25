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
    py = find_program('python3', 'python')
    rgsaddle_fortran_mod = custom_target(
        'rgsaddle_fortran_mod',
        output: 'rgsaddle.mod',
        depends: rgsaddle_fortran_lib,
        build_by_default: true,
        install: true,
        install_dir: fortran_moduledir,
        command: [
            py,
            '-c',
            'import os, shutil, sys\n' +
            'root, dst = sys.argv[1], sys.argv[2]\n' +
            'for dirpath, _, names in os.walk(root):\n' +
            '    if "rgsaddle.mod" in names:\n' +
            '        src = os.path.join(dirpath, "rgsaddle.mod")\n' +
            '        if os.path.abspath(src) != os.path.abspath(dst):\n' +
            '            shutil.copyfile(src, dst)\n' +
            '        sys.exit(0)\n' +
            'sys.exit("rgsaddle.mod not found under " + root)\n',
            meson.current_build_dir(),
            '@OUTPUT@',
        ],
    )

    pkg = import('pkgconfig')
    # `${prefix}` stays literal so pkg-config recomputes it from the
    # .pc location. `-lrgsaddle` is the cargo cdylib; it is not installed
    # by meson, and the link line must also have its directory.
    pkg.generate(
        name: 'rgsaddle-fortran',
        description: 'Fortran 2008 ISO C binding interfaces for the rgsaddle C ABI',
        version: meson.project_version(),
        libraries: [rgsaddle_fortran_lib, '-lrgsaddle'],
        requires: ['rgsaddle-headers'],
        subdirs: 'rgsaddle/fortran',
        variables: [
            'moduledir=${prefix}/' + fortran_moduledir,
        ],
    )

    cmake_dir = get_option('libdir') / 'cmake' / 'rgsaddleFortran'
    # One `..` per component of the package dir reaches the prefix.
    # includedir and libdir are the relative install directories.
    _prefix_ups = []
    foreach _part : cmake_dir.split('/')
        _prefix_ups += '..'
    endforeach
    cfg = configuration_data()
    cfg.set('RGSADDLE_FORTRAN_VERSION', meson.project_version())
    cfg.set('RGSADDLE_PREFIX_FROM_CONFIG', '/'.join(_prefix_ups))
    cfg.set('RGSADDLE_FORTRAN_MODULE_REL', fortran_moduledir)
    cfg.set('RGSADDLE_FORTRAN_INCLUDE_REL', get_option('includedir'))
    cfg.set('RGSADDLE_FORTRAN_LIB_REL', get_option('libdir'))

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
