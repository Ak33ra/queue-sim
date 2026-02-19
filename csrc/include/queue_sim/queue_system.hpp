#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "distributions.hpp"
#include "event_log.hpp"
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
    std::vector<double> response_times;
    EventLog event_log;

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
                                  int warmup = 0,
                                  bool track_response_times = false,
                                  bool track_events = false) {
        verifyTransitionMatrix();
        uint64_t resolved_seed;
        if (seed >= 0) {
            resolved_seed = static_cast<uint64_t>(seed);
        } else {
            std::random_device rd;
            resolved_seed = static_cast<uint64_t>(rd()) |
                            (static_cast<uint64_t>(rd()) << 32);
        }
        response_times.clear();
        std::vector<double>* rt_ptr = nullptr;
        if (track_response_times) {
            response_times.reserve(num_events);
            rt_ptr = &response_times;
        }
        event_log.clear();
        EventLog* el_ptr = nullptr;
        if (track_events) {
            event_log.reserve(num_events * 2);
            el_ptr = &event_log;
        }
        auto [mean_n, mean_t] = sim_internal(
            servers, arrivalDist, transitionMatrix, num_events,
            resolved_seed, warmup, rt_ptr, el_ptr);
        T = mean_t;
        return {mean_n, mean_t};
    }

    ReplicationRawResult replicate(int n_replications = 30,
                                   int num_events = 1000000,
                                   int seed = -1,
                                   int warmup = 0,
                                   int n_threads = 0) {
        uint64_t base_seed;
        if (seed >= 0) {
            base_seed = static_cast<uint64_t>(seed);
        } else {
            std::random_device rd;
            base_seed = static_cast<uint64_t>(rd()) |
                        (static_cast<uint64_t>(rd()) << 32);
        }

        verifyTransitionMatrix();

        int actual_threads = n_threads;
        if (actual_threads <= 0) {
            actual_threads = static_cast<int>(
                std::thread::hardware_concurrency());
            if (actual_threads <= 0) actual_threads = 1;
        }
        actual_threads = std::min(actual_threads, n_replications);

        ReplicationRawResult result;
        result.raw_N.resize(n_replications);
        result.raw_T.resize(n_replications);

        auto worker = [&](int start, int end) {
            // Clone servers once for this thread
            std::vector<std::shared_ptr<Server>> local_servers;
            local_servers.reserve(servers.size());
            for (const auto& s : servers) {
                local_servers.push_back(s->clone());
            }

            for (int i = start; i < end; ++i) {
                uint64_t rep_seed =
                    derive_seed(base_seed, static_cast<uint64_t>(i));
                auto [n, t] = sim_internal(
                    local_servers, arrivalDist, transitionMatrix,
                    num_events, rep_seed, warmup);
                result.raw_N[i] = n;
                result.raw_T[i] = t;
            }
        };

        if (actual_threads == 1) {
            worker(0, n_replications);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(actual_threads);
            int chunk = n_replications / actual_threads;
            int remainder = n_replications % actual_threads;
            int start = 0;
            for (int t = 0; t < actual_threads; ++t) {
                int end = start + chunk + (t < remainder ? 1 : 0);
                threads.emplace_back(worker, start, end);
                start = end;
            }
            for (auto& th : threads) {
                th.join();
            }
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

    static double minTTNC(
            const std::vector<std::shared_ptr<Server>>& srvs) {
        double m = std::numeric_limits<double>::infinity();
        for (const auto &s : srvs) {
            m = std::min(m, s->queryTTNC());
        }
        return m;
    }

    static int routeJob(int server_idx, std::mt19937_64 &rng,
                         const std::vector<std::vector<double>>& tm,
                         int n_servers) {
        if (tm.empty()) {
            return server_idx + 1;
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        double acc = 0.0;
        const auto &row = tm[server_idx];
        for (int i = 0; i < static_cast<int>(row.size()); ++i) {
            acc += row[i];
            if (r < acc) return i;
        }
        // Numerical safety: fall through -> exit
        return n_servers;
    }

    static std::pair<double, double> sim_internal(
            std::vector<std::shared_ptr<Server>>& srvs,
            Distribution arrival_dist,
            const std::vector<std::vector<double>>& tm,
            int num_events,
            uint64_t seed,
            int warmup,
            std::vector<double>* response_times = nullptr,
            EventLog* event_log = nullptr) {
        std::mt19937_64 rng(seed);
        int n_servers = static_cast<int>(srvs.size());

        for (auto &s : srvs) {
            s->setRNG(&rng);
            s->reset();
        }

        int num_completions = 0;
        double ttna = sample(arrival_dist, rng);
        int state = 0;

        // -- warmup phase (no accumulation) ----------------------------------
        if (warmup > 0) {
            int warmup_done = 0;
            while (warmup_done < warmup) {
                double ttnc = minTTNC(srvs);
                double ttne = std::min(ttnc, ttna);
                std::vector<int> completed;
                for (int i = 0; i < n_servers; ++i) {
                    if (srvs[i]->update(ttne)) {
                        completed.push_back(i);
                    }
                }
                for (int idx : completed) {
                    int dest = routeJob(idx, rng, tm, n_servers);
                    if (dest >= n_servers) {
                        warmup_done += 1;
                        state -= 1;
                    } else {
                        srvs[dest]->num_arrivals += 1;
                        if (srvs[dest]->is_full()) {
                            srvs[dest]->num_rejected += 1;
                            warmup_done += 1;
                            state -= 1;
                        } else {
                            srvs[dest]->arrival();
                        }
                    }
                }
                if (ttna <= ttnc) {
                    srvs[0]->num_arrivals += 1;
                    if (srvs[0]->is_full()) {
                        srvs[0]->num_rejected += 1;
                    } else {
                        state += 1;
                        srvs[0]->arrival();
                    }
                    ttna = sample(arrival_dist, rng);
                } else {
                    ttna -= ttne;
                }
            }
        }

        // Clear per-server rejection counters so measurement reflects
        // only the measurement phase.
        for (auto &s : srvs) {
            s->num_rejected = 0;
            s->num_arrivals = 0;
        }

        // -- measurement phase -----------------------------------------------
        double area_n = 0.0;
        double clock = 0.0;

        while (num_completions < num_events) {
            double ttnc = minTTNC(srvs);
            double ttne = std::min(ttnc, ttna);

            clock += ttne;
            area_n += static_cast<double>(state) * ttne;

            std::vector<int> completed;
            for (int i = 0; i < n_servers; ++i) {
                if (srvs[i]->update(ttne)) {
                    completed.push_back(i);
                }
            }

            for (int idx : completed) {
                int dest = routeJob(idx, rng, tm, n_servers);
                if (dest >= n_servers) {
                    num_completions += 1;
                    state -= 1;
                    if (response_times) {
                        response_times->push_back(
                            srvs[idx]->_last_response_time);
                    }
                    if (event_log) {
                        event_log->push(clock, EventLog::DEPARTURE, idx, EventLog::SYSTEM_EXIT, state);
                    }
                } else {
                    srvs[dest]->num_arrivals += 1;
                    if (srvs[dest]->is_full()) {
                        srvs[dest]->num_rejected += 1;
                        num_completions += 1;
                        state -= 1;
                        if (event_log) {
                            event_log->push(clock, EventLog::REJECTION, idx, dest, state);
                        }
                    } else {
                        srvs[dest]->arrival();
                        if (event_log) {
                            event_log->push(clock, EventLog::ROUTE, idx, dest, state);
                        }
                    }
                }
            }

            if (ttna <= ttnc) {
                srvs[0]->num_arrivals += 1;
                if (srvs[0]->is_full()) {
                    srvs[0]->num_rejected += 1;
                    if (event_log) {
                        event_log->push(clock, EventLog::REJECTION, EventLog::EXTERNAL, 0, state);
                    }
                } else {
                    state += 1;
                    srvs[0]->arrival();
                    if (event_log) {
                        event_log->push(clock, EventLog::ARRIVAL, EventLog::EXTERNAL, 0, state);
                    }
                }
                ttna = sample(arrival_dist, rng);
            } else {
                ttna -= ttne;
            }
        }

        double mean_n = area_n / clock;
        double mean_t = area_n / std::max(1, num_completions);
        return {mean_n, mean_t};
    }
};

}  // namespace queue_sim
