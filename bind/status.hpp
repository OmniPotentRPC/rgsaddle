// MIT License
// Copyright 2026 Rohit Goswami

/**
 * @file status.hpp
 * @brief Map every rgsaddle_status_t onto the rgbind status table.
 *
 * The switch is exhaustive. Build this translation unit with -Wswitch
 * -Werror so a new enumerator cannot fall through unnoticed.
 */

#pragma once

#include <rgsaddle.h>

#include <rgbind/rgbind.hpp>

namespace rgbind {

template <> struct status_traits<rgsaddle_status_t> {
  static int code(rgsaddle_status_t value) {
    switch (value) {
    case RGSADDLE_OK:
      return static_cast<int>(Status::ok);
    case RGSADDLE_SHAPE:
      return static_cast<int>(Status::build);
    case RGSADDLE_INVALID_PARAMETER:
      return static_cast<int>(Status::key);
    case RGSADDLE_SURFACE_FAILED:
      return static_cast<int>(Status::callback);
    case RGSADDLE_NON_FINITE:
      return static_cast<int>(Status::numerical);
    case RGSADDLE_NULL_SESSION:
    case RGSADDLE_NULL_BAND:
    case RGSADDLE_NULL_SURFACE:
    case RGSADDLE_NULL_REPORT:
    case RGSADDLE_SOLVER:
    case RGSADDLE_STALLED:
    case RGSADDLE_ABI_MISMATCH:
    case RGSADDLE_ALLOC:
    case RGSADDLE_NO_EVALUATION:
    case RGSADDLE_NOT_EQUILIBRATED:
    case RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES:
    case RGSADDLE_LANDSCAPE_BOUND:
      return static_cast<int>(Status::other);
    }
    return static_cast<int>(Status::other);
  }
};

} // namespace rgbind
