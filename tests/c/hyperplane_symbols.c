/* Link the four hyperplane entry points. They are declared for every
 * build. Without the ohtst feature, creation returns NULL, free is a
 * no-op, integration fails, and the observable name is unchanged. */
#include "rgsaddle.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  if (rgsaddle_hyperplane_sampler_create(NULL, NULL) != NULL) {
    return 1;
  }
  rgsaddle_hyperplane_sampler_free(NULL);
  if (rgsaddle_hyperplane_thermodynamic_integration(NULL, NULL, NULL, 0, NULL,
                                                    NULL) == 0) {
    return 2;
  }
  const char *name = rgsaddle_thermodynamic_derivative_observable();
  if (name == NULL || strcmp(name, "potential.full_minus_harmonic") != 0) {
    return 3;
  }
  printf("RGSADDLE_HYPERPLANE_SYMBOLS_OK\n");
  return 0;
}
