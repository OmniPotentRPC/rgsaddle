project(
    'rgsaddle',
    version: '0.6.0',
    license: 'MIT',
    meson_version: '>=0.63.0',
)

# Headers only. The runtime is the Rust cdylib or staticlib (cargo build
# --features capi, or cargo-c); hosts dlopen it or link it themselves.
# tests/package_version.rs keeps the version above equal to Cargo.toml's.

rgsaddle_inc = include_directories('include')

rgsaddle_dep = declare_dependency(
    include_directories: rgsaddle_inc,
    version: meson.project_version(),
)
# The name and variable of the rgsaddle-headers packagefiles overlay that
# consumers carried before this file existed.
rgsaddle_headers_dep = rgsaddle_dep

meson.override_dependency('rgsaddle', rgsaddle_dep)
meson.override_dependency('rgsaddle-headers', rgsaddle_dep)

fortran_opt = get_option('fortran')
if fortran_opt.enabled()
    add_languages('fortran', native: false, required: true)
    subdir('fortran')
endif

if not meson.is_subproject()
    install_headers('include/rgsaddle.h')
    install_headers(
        'include/rgsaddle/session.hpp',
        'include/rgsaddle/stencil.hpp',
        subdir: 'rgsaddle',
    )

    # cargo-c installs rgsaddle.pc with the library, so the headers-only
    # file takes the overlay's name and the architecture-independent dir.
    pkg = import('pkgconfig')
    pkg.generate(
        name: 'rgsaddle-headers',
        description: 'C ABI and C++ headers for the rgsaddle band, minimum-mode, and index-1 sessions',
        version: meson.project_version(),
        url: 'https://github.com/OmniPotentRPC/rgsaddle',
        install_dir: get_option('datadir') / 'pkgconfig',
    )
endif
