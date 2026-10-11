/* C++ stencil wrapper: a dimer step and one climbing-image band step. */
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

static void band_well(std::size_t rows, std::size_t dof, const std::uint64_t *ids, const double *x,
                      double *e, double *g) {
  if (dof != 2 || ids == nullptr) {
    throw rgsaddle::Error("shape");
  }
  for (std::size_t row = 0; row < rows; ++row) {
    const double px = x[row * 2];
    const double py = x[row * 2 + 1];
    e[row] = (px * px - 1.0) * (px * px - 1.0) + py * py;
    g[row * 2] = 4.0 * px * (px * px - 1.0);
    g[row * 2 + 1] = 2.0 * py;
  }
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
  bool threw = false;
  try {
    rgsaddle::PointBuilder(x, 2, RGSADDLE_STENCIL_DIMER).set_f64(RGSADDLE_STENCIL_KEY_K_MIN, 0.972);
  } catch (const rgsaddle::Error &err) {
    threw = std::string(err.what()).find("band stencil") != std::string::npos;
  }
  if (!threw) {
    std::fprintf(stderr, "point accepted a band key\n");
    return 1;
  }

  const double images[6] = {-1.0, 0.0, 0.0, 0.2, 1.0, 0.0};
  auto band = rgsaddle::BandBuilder(images, 3, 2, RGSADDLE_STENCIL_ENERGY_WEIGHTED_NEB)
                  .set_f64(RGSADDLE_STENCIL_KEY_K_MIN, 0.972)
                  .set_f64(RGSADDLE_STENCIL_KEY_K_MAX, 9.72)
                  .build();
  rgsaddle::StencilReport stepped = band.step(band_well);
  if (stepped.iteration < 1 || band.positions().size() != 6 || band.rows() != 3) {
    std::fprintf(stderr, "band step\n");
    return 1;
  }
  std::vector<std::uint64_t> ids = band.image_ids();
  if (ids.size() != 3 || ids[0] != 0 || ids[2] != 2) {
    std::fprintf(stderr, "band ids\n");
    return 1;
  }

  std::printf("RGSADDLE_STENCIL_CXX_OK y=%.17g iter=%llu\n", out[1],
              static_cast<unsigned long long>(report.iteration));
  return 0;
}
