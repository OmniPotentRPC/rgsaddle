cmake_minimum_required(VERSION 3.15...3.30)
project(rgsaddle_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  add_compile_options(/fp:strict)
else()
  add_compile_options(-ffp-contract=off)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Nightly rustc, an explicit target triple, and a std rebuilt with
# the thread sanitizer. Rust_TOOLCHAIN and Rust_CARGO_TARGET are read
# when Corrosion is included, so they are set first. The workflow passes
# the dated nightly that matches the image clang.
option(RGSADDLE_TSAN "Build the Rust staticlib with -Zsanitizer=thread." OFF)
if(RGSADDLE_TSAN)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_rgsaddle_tsan_target "aarch64-unknown-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(_rgsaddle_tsan_target "x86_64-unknown-linux-gnu")
  else()
    message(FATAL_ERROR
      "RGSADDLE_TSAN has no target triple for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  if(NOT Rust_TOOLCHAIN)
    set(Rust_TOOLCHAIN "nightly" CACHE STRING
      "Rustup toolchain used when RGSADDLE_TSAN is ON")
  endif()
  set(Rust_CARGO_TARGET "${_rgsaddle_tsan_target}" CACHE STRING
    "Explicit cargo target triple for the sanitizer staticlib" FORCE)
  message(STATUS "RGSADDLE_TSAN toolchain ${Rust_TOOLCHAIN} target ${_rgsaddle_tsan_target}")
endif()

include(FetchContent)
FetchContent_Declare(
  corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
)
FetchContent_MakeAvailable(corrosion)

# The staticlib is the C ABI (feature capi). Python calls rgsaddle_status_name
# and nothing else from it. Session entry points stay off the Python surface.
corrosion_import_crate(
  MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  CRATES rgsaddle
  CRATE_TYPES staticlib
  FEATURES capi
)

if(RGSADDLE_TSAN)
  corrosion_set_cargo_flags(rgsaddle --locked -Zbuild-std=std,panic_abort)
  corrosion_add_target_rustflags(rgsaddle -Zsanitizer=thread)
else()
  corrosion_set_cargo_flags(rgsaddle --locked)
endif()

# FREE_THREADED sets Py_mod_gil = Py_MOD_GIL_NOT_USED on a
# free-threaded interpreter. On a GIL interpreter nanobind keeps
# STABLE_ABI and ignores FREE_THREADED. cp312-abi3 covers 3.12+ GIL.
# cp314t and cp315t are version-specific free-threaded wheels.
nanobind_add_module(_core FREE_THREADED STABLE_ABI NB_STATIC bind/ext.cpp bind/stencil.cpp)
target_include_directories(_core PRIVATE include)
target_link_libraries(_core PRIVATE rgsaddle Threads::Threads)
if(NOT DEFINED SKBUILD_PROJECT_VERSION OR SKBUILD_PROJECT_VERSION STREQUAL "")
  message(FATAL_ERROR "SKBUILD_PROJECT_VERSION is required")
endif()
target_compile_definitions(_core PRIVATE "RGSADDLE_PY_VERSION=\"${SKBUILD_PROJECT_VERSION}\"")

# The stub generator imports rgsaddle._core, so the extension has
# to sit inside a package directory on PYTHON_PATH.
set(_core_stage "${CMAKE_CURRENT_BINARY_DIR}/rgsaddle")
file(MAKE_DIRECTORY "${_core_stage}")
file(WRITE "${_core_stage}/__init__.py" "")
# A generator expression stops the Visual Studio generator from appending
# a per-configuration directory. Stubgen imports rgsaddle._core from
# this directory, so the extension cannot land in Release/.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<1:${_core_stage}>"
  RUNTIME_OUTPUT_DIRECTORY "$<1:${_core_stage}>")

if(NOT MSVC)
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Windows free-threaded builds define Py_GIL_DISABLED when the
# Python headers did not.
if(WIN32 AND NB_FREE_THREADED)
  target_compile_definitions(_core PRIVATE Py_GIL_DISABLED=1)
endif()

# Hidden visibility is the default above. --exclude-libs,ALL drops
# the staticlib from the dynamic table. The version script (Linux) and
# the export list (macOS) leave only PyInit__core.
if(UNIX AND NOT APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.map"
"{\n  global:\n    PyInit__core;\n  local:\n    *;\n};\n")
  target_link_options(_core PRIVATE
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/core_exports.map")
endif()
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt" "_PyInit__core\n")
  target_link_options(_core PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/core_exports.txt")
endif()
# A Rust staticlib exports every public symbol. On Windows the shared
# library limits that set with a module-definition file.
# __declspec(dllexport) symbols from the static nanobind library are not
# removed by the .def file, so the export directory is rewritten after
# the link.
if(MSVC)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/core_exports.def" "EXPORTS\n  PyInit__core\n")
  target_link_options(_core PRIVATE
    "/DEF:${CMAKE_CURRENT_BINARY_DIR}/core_exports.def")
  add_custom_command(
    TARGET _core POST_BUILD
    COMMAND "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ci/limit_pe_exports.py"
            "$<TARGET_FILE:_core>"
            "PyInit__core"
    VERBATIM)
endif()

# py-tsan.yml counts __tsan_func_entry in librust_core_demo.a.
# The archive name stays until rgpot's workflow takes it as an input.
# Corrosion writes the static archive to the build directory as
# librgsaddle.a. The rgsaddle CMake target is an interface library, so
# TARGET_FILE does not apply to it. Copying by path avoids that, and
# avoids a cycle through the imported rgsaddle-static target.
if(RGSADDLE_TSAN)
  add_custom_target(rgsaddle_tsan_archive_name ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/librgsaddle.a"
            "${CMAKE_BINARY_DIR}/librust_core_demo.a"
    DEPENDS cargo-build_rgsaddle
    VERBATIM)
endif()

nanobind_add_stub(
  _core_stub
  MODULE rgsaddle._core
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
  PYTHON_PATH "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS _core
)

# LIBRARY is the .so on Unix. RUNTIME is the .pyd on Windows.
install(TARGETS _core
  LIBRARY DESTINATION rgsaddle
  RUNTIME DESTINATION rgsaddle)
