#pragma once

#include <cmath>
#include <random>
#include <variant>

namespace queue_sim {

struct ExponentialDist {
    double mu;  // rate parameter; E[X] = 1/mu
    explicit ExponentialDist(double mu) : mu(mu) {}

    double sample(std::mt19937_64 &rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return -(1.0 / mu) * std::log(1.0 - u(rng));
    }
};

struct UniformDist {
    double a, b;
    UniformDist(double a, double b) : a(a), b(b) {}

    double sample(std::mt19937_64 &rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return (b - a) * u(rng) + a;
    }
};

struct BoundedParetoDist {
    double k, p, alpha, C;
    BoundedParetoDist(double k, double p, double alpha)
        : k(k), p(p), alpha(alpha),
          C(std::pow(k, alpha) / (1.0 - std::pow(k / p, alpha))) {}

    double sample(std::mt19937_64 &rng) const {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return std::pow(-u(rng) / C + std::pow(k, -alpha), -1.0 / alpha);
    }
};

using Distribution = std::variant<ExponentialDist, UniformDist, BoundedParetoDist>;

inline double sample(Distribution &dist, std::mt19937_64 &rng) {
    return std::visit([&rng](auto &d) { return d.sample(rng); }, dist);
}

}  // namespace queue_sim
