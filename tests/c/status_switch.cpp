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
  if (ok != 0 || equilibration == 0 || samples == 0 || converged == 0) {
    return 1;
  }
  return 0;
}
