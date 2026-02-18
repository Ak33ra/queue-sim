#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "distributions.hpp"
#include "server.hpp"

namespace queue_sim {

struct ReplicationRawResult {
    std::vector<double> raw_N;
    std::vector<double> raw_T;
};

// SplitMix64 one-round (Steele / Vigna) — matches Python _splitmix64.
inline uint64_t splitmix64(uint64_t x) {
    static constexpr uint64_t PHI = 0x9E3779B97F4A7C15ULL;
    x = (x + PHI);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline uint64_t derive_seed(uint64_t base_seed, uint64_t index) {
    static constexpr uint64_t PHI = 0x9E3779B97F4A7C15ULL;
    return splitmix64(base_seed + index * PHI);
}

class QueueSystem {
public:
    std::vector<std::shared_ptr<Server>> servers;
    Distribution arrivalDist;
    std::vector<std::vector<double>> transitionMatrix;
    double T = 0.0;

    QueueSystem(std::vector<std::shared_ptr<Server>> servers,
                Distribution arrivalDist,
                std::vector<std::vector<double>> transitionMatrix = {})
        : servers(std::move(servers)),
          arrivalDist(std::move(arrivalDist)),
          transitionMatrix(std::move(transitionMatrix)) {}

    void addServer(std::shared_ptr<Server> server) {
        servers.push_back(std::move(server));
    }

    void updateTransitionMatrix(std::vector<std::vector<double>> M) {
        transitionMatrix = std::move(M);
    }

    std::pair<double, double> sim(int num_events = 1000000,
                                  int seed = -1,
                                  int warmup = 0) {
        std::mt19937_64 rng;
        if (seed >= 0) {
            rng.seed(static_cast<std::mt19937_64::result_type>(seed));
        } else {
            std::random_device rd;
            rng.seed(rd());
        }

        verifyTransitionMatrix();

        for (auto &s : servers) {
            s->setRNG(&rng);
            s->reset();
        }

        int num_completions = 0;
        double ttna = sample(arrivalDist, rng);  // time to next arrival
        int state = 0;        // total jobs in the network

        // -- warmup phase (no accumulation) ----------------------------------
        if (warmup > 0) {
            int warmup_done = 0;
            while (warmup_done < warmup) {
                double ttnc = minTTNC();
                double ttne = std::min(ttnc, ttna);
                std::vector<int> completed;
                for (int i = 0; i < static_cast<int>(servers.size()); ++i) {
                    if (servers[i]->update(ttne)) {
                        completed.push_back(i);
                    }
                }
                for (int idx : completed) {
                    int dest = routeJob(idx, rng);
                    if (dest >= static_cast<int>(servers.size())) {
                        warmup_done += 1;
                        state -= 1;
                    } else {
                        servers[dest]->arrival();
                    }
                }
                if (ttna <= ttnc) {
                    state += 1;
                    servers[0]->arrival();
                    ttna = sample(arrivalDist, rng);
                } else {
                    ttna -= ttne;
                }
            }
        }

        // -- measurement phase -----------------------------------------------
        double area_n = 0.0;
        double clock = 0.0;

        while (num_completions < num_events) {
            double ttnc = minTTNC();
            double ttne = std::min(ttnc, ttna);

            clock += ttne;
            area_n += static_cast<double>(state) * ttne;

            // Advance all servers, collect indices of those that completed
            std::vector<int> completed;
            for (int i = 0; i < static_cast<int>(servers.size()); ++i) {
                if (servers[i]->update(ttne)) {
                    completed.push_back(i);
                }
            }

            // Route completed jobs
            for (int idx : completed) {
                int dest = routeJob(idx, rng);
                if (dest >= static_cast<int>(servers.size())) {
                    num_completions += 1;
                    state -= 1;
                } else {
                    servers[dest]->arrival();
                }
            }

            // Handle arrival if it fires at or before the next completion
            if (ttna <= ttnc) {
                state += 1;
                servers[0]->arrival();
                ttna = sample(arrivalDist, rng);
            } else {
                ttna -= ttne;
            }
        }

        double mean_n = area_n / clock;
        double mean_t = area_n / std::max(1, num_completions);
        T = mean_t;
        return {mean_n, mean_t};
    }

    ReplicationRawResult replicate(int n_replications = 30,
                                   int num_events = 1000000,
                                   int seed = -1,
                                   int warmup = 0) {
        uint64_t base_seed;
        if (seed >= 0) {
            base_seed = static_cast<uint64_t>(seed);
        } else {
            std::random_device rd;
            base_seed = static_cast<uint64_t>(rd()) |
                        (static_cast<uint64_t>(rd()) << 32);
        }

        ReplicationRawResult result;
        result.raw_N.reserve(n_replications);
        result.raw_T.reserve(n_replications);

        for (int i = 0; i < n_replications; ++i) {
            uint64_t rep_seed = derive_seed(base_seed, static_cast<uint64_t>(i));
            auto [n, t] = sim(num_events, static_cast<int>(rep_seed & 0x7FFFFFFF), warmup);
            result.raw_N.push_back(n);
            result.raw_T.push_back(t);
        }
        return result;
    }

private:
    void verifyTransitionMatrix() const {
        if (transitionMatrix.empty()) return;

        int n_servers = static_cast<int>(servers.size());
        int n_rows = static_cast<int>(transitionMatrix.size());
        if (n_rows != n_servers) {
            throw std::invalid_argument(
                "Transition matrix must have " + std::to_string(n_servers) +
                " rows, got " + std::to_string(n_rows));
        }
        for (int i = 0; i < n_rows; ++i) {
            if (static_cast<int>(transitionMatrix[i].size()) != n_servers + 1) {
                throw std::invalid_argument(
                    "Transition matrix row " + std::to_string(i) +
                    " must have " + std::to_string(n_servers + 1) +
                    " columns, got " +
                    std::to_string(transitionMatrix[i].size()));
            }
            double row_sum = 0.0;
            for (double v : transitionMatrix[i]) row_sum += v;
            if (std::abs(row_sum - 1.0) > 1e-9) {
                throw std::invalid_argument(
                    "Transition matrix row " + std::to_string(i) +
                    " sums to " + std::to_string(row_sum) + ", expected 1.0");
            }
        }
    }

    double minTTNC() const {
        double m = std::numeric_limits<double>::infinity();
        for (const auto &s : servers) {
            m = std::min(m, s->queryTTNC());
        }
        return m;
    }

    int routeJob(int server_idx, std::mt19937_64 &rng) {
        if (transitionMatrix.empty()) {
            return server_idx + 1;
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        double acc = 0.0;
        const auto &row = transitionMatrix[server_idx];
        for (int i = 0; i < static_cast<int>(row.size()); ++i) {
            acc += row[i];
            if (r < acc) return i;
        }
        // Numerical safety: fall through → exit
        return static_cast<int>(servers.size());
    }
};

}  // namespace queue_sim
