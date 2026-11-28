The top-level `meson.build` gives a C or C++ project the rgsaddle headers: `rgsaddle.h`, `rgsaddle/session.hpp` and `rgsaddle/stencil.hpp`. It does not build the Rust library. The host links the library that `cargo build --features capi` or `cargo cinstall` produces, or loads it at run time. The headers are the hand-written contract, the same files cargo-c installs, so there is nothing to generate.

The meson project is `rgsaddle`, and its version is the crate's. `tests/package_version.rs` fails when `meson.build` and `Cargo.toml` disagree. The dependency answers to two names, `rgsaddle-headers` and `rgsaddle`, and the variable `rgsaddle_headers_dep` holds it.


# Pull the headers in with a wrap

Pin a full commit, as the crate pins its own git dependencies. Save this as `subprojects/rgsaddle-headers.wrap`:

```ini
[wrap-git]
directory = rgsaddle-headers
url = https://github.com/OmniPotentRPC/rgsaddle.git
revision = <full 40-hex commit>
depth = 1

[provide]
dependency_names = rgsaddle-headers, rgsaddle
```

Then ask for the dependency by either name:

```meson
rgsaddle_dep = dependency('rgsaddle-headers')
```

The explicit form that names the subproject and its variable also works:

```meson
rgsaddle_dep = dependency(
    'rgsaddle-headers',
    fallback: ['rgsaddle-headers', 'rgsaddle_headers_dep'],
)
```

As a subproject, rgsaddle installs nothing into the parent's prefix.


# Move off the packagefiles overlay

A project that carried `subprojects/packagefiles/rgsaddle-headers/meson.build` deletes that directory and the `patch_directory` line of its wrap, and moves `revision` to a commit that has `meson.build`. Its `dependency()` call stays as it was. The overlay only exposed `include/rgsaddle.h`; the subproject now exposes `include/`, so `#include <rgsaddle/session.hpp>` also resolves.


# Use installed headers through pkg-config

From a checkout:

```sh
meson setup build --prefix "$PREFIX"
meson install -C build
```

That installs the headers under `include` and `rgsaddle-headers.pc` under `share/pkgconfig`, the directory for files with no architecture. The file has `Cflags` and no `Libs`. Put that directory on `PKG_CONFIG_PATH`, and `dependency('rgsaddle-headers')` finds it before any fallback.

`cargo cinstall` writes `rgsaddle.pc`, which also links `-lrgsaddle`. The meson file has another name so that the two can share a prefix. Installed, `dependency('rgsaddle')` is therefore the cargo-c library. As a subproject, the `rgsaddle` name is the headers alone.


# Load the library at run time

A host that loads the library with `dlopen`, as gpridge does through `RGSADDLE_LIBRARY`, types each lookup with the header's declaration and needs no link line:

```cpp
#include <rgsaddle.h>
#include <dlfcn.h>

void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
auto* band_create = reinterpret_cast<decltype(rgsaddle_band_create)*>(
    dlsym(lib, "rgsaddle_band_create"));
```

Call `rgsaddle_abi_stamp` first and refuse a library whose major differs from `RGSADDLE_ABI_MAJOR` or whose minor is below what the host uses. The inline helpers of `rgsaddle/session.hpp` that only fill structs, such as `rgsaddle::band_config()`, need no symbol either. The wrapper classes call the library directly and need it linked.


# What CI checks

The `meson-consumer` job builds `ci/meson-consumer`, a C++ project that makes the explicit `dependency()` call above, once through a wrap-git at the commit under test and once through an installed `rgsaddle-headers.pc` with fallbacks off. Each run checks which of the two it got and its version, loads the cdylib, and steps a climbing band on a double well to the saddle. The job runs on meson 0.63.0, the `meson_version` floor, and on the current release. To run it locally:

```sh
cargo build --features capi
ci/meson_consumer.sh subproject target/debug/librgsaddle.so
ci/meson_consumer.sh pkgconfig target/debug/librgsaddle.so
```

The subproject mode clones the checkout, so it sees committed files only. The consumer loads the library with `dlopen`; CI runs it on Linux.
