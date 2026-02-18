#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "server.hpp"

namespace queue_sim {

class SRPT : public Server {
public:
    // min-heap: (remaining, arrival_time) — sorted by remaining first
    using Job = std::pair<double, double>;
    std::priority_queue<Job, std::vector<Job>, std::greater<Job>> jobs;
    double _running_arrival_time = 0.0;

    explicit SRPT(Distribution sizeDist, int buffer_capacity = -1)
        : Server(std::move(sizeDist), 1, buffer_capacity) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<SRPT>(sizeDist, buffer_capacity);
    }

    void reset() override {
        Server::reset();
        jobs = {};
        _running_arrival_time = 0.0;
    }

    double nextJob() override {
        auto [remaining, arr] = jobs.top();
        jobs.pop();
        _running_arrival_time = arr;
        return remaining;
    }

    void updateET() override {
        double t = clock - _running_arrival_time;
        _last_response_time = t;
        double n = static_cast<double>(num_completions);
        T = T * (n - 1.0) / n + t / n;
    }

    void arrival() override {
        if (state > 0) {
            jobs.push({TTNC, _running_arrival_time});
        }
        jobs.push({sample(sizeDist, *rng), clock});
        auto [remaining, arr] = jobs.top();
        jobs.pop();
        TTNC = remaining;
        _running_arrival_time = arr;
        state += 1;
    }

    // Critical: updateET() BEFORE nextJob() so we read the
    // completing job's arrival time, not the next job's.
    bool update(double time_elapsed) override {
        TTNC -= time_elapsed;
        clock += time_elapsed;
        if (TTNC <= 0.0) {
            state -= 1;
            num_completions += 1;
            updateET();
            TTNC = (state > 0) ? nextJob()
                               : std::numeric_limits<double>::infinity();
            return true;
        }
        return false;
    }
};

}  // namespace queue_sim
