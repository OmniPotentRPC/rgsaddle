/* C++ stencil builder: one CI-NEB step on the double well. */
#include "rgsaddle/session.hpp"

#include <cstdio>
#include <vector>

namespace {

int well(uint64_t const* ids, std::size_t rows, std::size_t dof, double const* x, double* energy,
         double* gradient) {
    (void)ids;
    if (dof != 3) {
        return 1;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        double px = x[row * 3];
        double py = x[row * 3 + 1];
        double pz = x[row * 3 + 2];
        energy[row] = (px * px - 1.0) * (px * px - 1.0) + 2.0 * py * py + 2.0 * pz * pz;
        gradient[row * 3] = 4.0 * px * (px * px - 1.0);
        gradient[row * 3 + 1] = 4.0 * py;
        gradient[row * 3 + 2] = 4.0 * pz;
    }
    return 0;
}

}  // namespace

int main() {
    std::vector<double> band;
    for (int image = 0; image < 5; ++image) {
        double t = static_cast<double>(image) / 4.0;
        band.push_back(-1.0 + 2.0 * t);
        band.push_back(0.0);
        band.push_back(0.0);
    }
    auto stencil = rgsaddle::Stencil::band(band, 5, 3)
                       .energy_weighted(0.972, 9.72)
                       .threads(1)
                       .build();
    auto surface = [](uint64_t const* ids, std::size_t rows, std::size_t dof, double const* x,
                      double* energy, double* gradient) {
        return well(ids, rows, dof, x, energy, gradient);
    };
    auto report = stencil.step(surface);
    if (report.iteration < 1) {
        std::fprintf(stderr, "stencil step did not advance\n");
        return 1;
    }
    auto positions = stencil.positions();
    if (positions.size() != band.size()) {
        std::fprintf(stderr, "position length\n");
        return 1;
    }
    bool thrown = false;
    try {
        auto point = rgsaddle::Stencil::point(std::vector<double>{0.0, 0.0, 0.0});
        auto rejected = point.energy_weighted(1.0, 2.0).build();
        (void)rejected;
    } catch (std::runtime_error const&) {
        thrown = true;
    }
    if (!thrown) {
        std::fprintf(stderr, "point spring was accepted\n");
        return 1;
    }
    std::printf("RGSADDLE_STENCIL_CXX_OK\n");
    return 0;
}
