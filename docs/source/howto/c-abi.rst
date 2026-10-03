Use the C and C++ interfaces
============================

``include/rgsaddle.h`` declares the C application binary interface (ABI), version 1.15. It preserves the main band, minimum-mode and index-1 layouts and adds intrinsic reaction coordinate (IRC), Sella, constraint and geometry controls. Include this header rather than duplicating its structures. Check ``rgsaddle_abi_stamp`` before constructing a session.

A surface callback returns ``rgsaddle_status_t`` and receives one ``rgsaddle_surface_request_t``. Positions and gradients are image-major Cartesian triples. The host supplies the energy and gradient arrays for every requested image; the library performs no Message Passing Interface (MPI) communication.

Create an opaque session, call its ``step`` entry, inspect ``rgsaddle_report_t``, and free it. Check each function status before reading outputs. The host owns search limits and the surface lifetime. A reset marks a surface change. The report's ``evaluations`` field counts the surface rows consumed by the step.

Band and minimum-mode force norms are selected by ``rgsaddle_band_set_force_gate`` and ``rgsaddle_minmode_set_force_gate``. Their configuration layouts have no member for that norm. The optional feasible-step controls are ``rgsaddle_band_set_highs`` and ``rgsaddle_minmode_set_highs``, with 0 or 1 as the flag.

``include/rgsaddle/session.hpp`` provides the C++ owning wrappers. ``Band``, ``MinMode`` and ``Irc`` release their opaque handles automatically and translate failing statuses into exceptions. ``tests/c/main_abi_contract.h`` checks the retained main layouts and signatures; the C and C++ smoke executables exercise the official header and library together.
