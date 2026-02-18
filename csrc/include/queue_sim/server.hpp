#pragma once

#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>

#include "distributions.hpp"

namespace queue_sim {

class Server {
public:
    Distribution sizeDist;
    std::mt19937_64 *rng = nullptr;

    double clock = 0.0;
    double TTNC = std::numeric_limits<double>::infinity();
    double T = 0.0;
    int num_completions = 0;
    int state = 0;
    std::deque<double> arrivalTimes;

    int num_servers;
    int buffer_capacity;
    int num_rejected = 0;
    int num_arrivals = 0;
    double _last_response_time = 0.0;

    explicit Server(Distribution sizeDist, int num_servers = 1,
                    int buffer_capacity = -1)
        : sizeDist(std::move(sizeDist)), num_servers(num_servers),
          buffer_capacity(buffer_capacity) {
        if (buffer_capacity == 0)
            throw std::invalid_argument(
                "buffer_capacity must be >= 1 or -1 (unlimited)");
    }
    virtual ~Server() = default;
    virtual std::shared_ptr<Server> clone() const = 0;

    void setRNG(std::mt19937_64 *r) { rng = r; }

    bool is_full() const {
        return buffer_capacity >= 0 && state >= buffer_capacity;
    }

    virtual void reset() {
        clock = 0.0;
        TTNC = std::numeric_limits<double>::infinity();
        T = 0.0;
        num_completions = 0;
        state = 0;
        num_rejected = 0;
        num_arrivals = 0;
        _last_response_time = 0.0;
        arrivalTimes.clear();
    }

    virtual double nextJob() = 0;

    virtual void updateET() {
        double t = clock - arrivalTimes.front();
        arrivalTimes.pop_front();
        _last_response_time = t;
        double n = static_cast<double>(num_completions);
        T = T * (n - 1.0) / n + t / n;
    }

    virtual void arrival() {
        arrivalTimes.push_back(clock);
        if (state == 0) {
            TTNC = nextJob();
        }
        state += 1;
    }

    double queryTTNC() const { return TTNC; }

    virtual bool update(double time_elapsed) {
        TTNC -= time_elapsed;
        clock += time_elapsed;
        if (TTNC <= 0.0) {
            state -= 1;
            TTNC = (state > 0) ? nextJob()
                               : std::numeric_limits<double>::infinity();
            num_completions += 1;
            updateET();
            return true;
        }
        return false;
    }
};

}  // namespace queue_sim
