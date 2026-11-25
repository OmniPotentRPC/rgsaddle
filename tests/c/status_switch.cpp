#define RGBIND_NO_PYTHON
#include "status.hpp"

int main() {
  const int ok = rgbind::status_traits<rgsaddle_status_t>::code(RGSADDLE_OK);
  const int equilibration =
      rgbind::status_traits<rgsaddle_status_t>::code(RGSADDLE_NOT_EQUILIBRATED);
  const int samples = rgbind::status_traits<rgsaddle_status_t>::code(
      RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES);
  const int converged =
      rgbind::status_traits<rgsaddle_status_t>::code(RGSADDLE_NOT_CONVERGED);
  if (ok != static_cast<int>(rgbind::Status::ok) ||
      converged == static_cast<int>(rgbind::Status::ok)) {
    return 1;
  }
  if (equilibration != static_cast<int>(RGSADDLE_NOT_EQUILIBRATED) ||
      samples != static_cast<int>(RGSADDLE_INSUFFICIENT_EFFECTIVE_SAMPLES)) {
    return 2;
  }
  if (equilibration == static_cast<int>(rgbind::Status::other) ||
      samples == static_cast<int>(rgbind::Status::other) ||
      equilibration == samples) {
    return 3;
  }
  return 0;
}
