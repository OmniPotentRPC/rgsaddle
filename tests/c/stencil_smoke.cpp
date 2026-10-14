/* Point stencil over the C ABI. Built by tests/stencil_cxx.rs. */
#include "rgsaddle/stencil.hpp"

#include <cstdio>
#include <string>
#include <vector>

static void bowl(std::size_t rows, std::size_t dof, const std::uint64_t *, const double *x,
                 double *e, double *g) {
  if (rows != 1 || dof != 2) {
    throw rgsaddle::Error("shape");
  }
  const double x0 = x[0];
  const double y = x[1];
  e[0] = 2.0 * x0 * x0 + 5.0 * y * y;
  g[0] = 4.0 * x0;
  g[1] = 10.0 * y;
}

int main() {
  const double x[2] = {0.4, 0.3};
  const double mode[2] = {1.0, 0.0};
  auto search = rgsaddle::PointBuilder(x, 2, RGSADDLE_STENCIL_DIMER)
                    .mode(mode, 2)
                    .set_i64(RGSADDLE_STENCIL_KEY_MAX_ROTATIONS, 4)
                    .build();
  rgsaddle::StencilReport report = search.step(bowl);
  std::vector<double> out = search.positions();
  if (out.size() != 2 || out[1] != 0.3 || out[0] == 0.4 || report.iteration < 1) {
    std::fprintf(stderr, "dimer step\n");
    return 1;
  }

  const double images[6] = {0.0, 0.0, 0.5, 0.0, 1.0, 0.0};
  rgsaddle_stencil_config_t *config = rgsaddle_stencil_config_new(RGSADDLE_STENCIL_CI_NEB);
  if (config == nullptr) {
    std::fprintf(stderr, "config\n");
    return 1;
  }
  rgsaddle_stencil_t *handle = rgsaddle_stencil_new(config, 3, 2, images);
  if (handle != nullptr) {
    rgsaddle_stencil_free(handle);
    std::fprintf(stderr, "band handle\n");
    return 1;
  }
  const char *text = rgsaddle_last_error();
  if (text == nullptr || std::string(text).find("feat/band-methods-zoom") == std::string::npos) {
    std::fprintf(stderr, "missing seam: %s\n", text == nullptr ? "" : text);
    return 1;
  }
  rgsaddle_stencil_config_free(config);

  std::printf("RGSADDLE_STENCIL_CXX_OK y=%.17g iter=%llu\n", out[1],
              static_cast<unsigned long long>(report.iteration));
  return 0;
}
