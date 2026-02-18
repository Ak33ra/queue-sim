#pragma once

#include <algorithm>
#include <functional>
#include <queue>
#include <vector>

#include "server.hpp"

namespace queue_sim {

class SRPT : public Server {
public:
    // min-heap: shortest remaining time on top
    std::priority_queue<double, std::vector<double>, std::greater<double>> jobs;

    explicit SRPT(Distribution sizeDist, int buffer_capacity = -1)
        : Server(std::move(sizeDist), 1, buffer_capacity) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<SRPT>(sizeDist, buffer_capacity);
    }

    void reset() override {
        Server::reset();
        // Clear the priority queue
        jobs = {};
    }

    double nextJob() override {
        double top = jobs.top();
        jobs.pop();
        return top;
    }

    void updateET() override {
        // SRPT reorders jobs — FIFO-based E[T] tracker is invalid
    }

    void arrival() override {
        if (state > 0) {
            jobs.push(TTNC);
        }
        jobs.push(sample(sizeDist, *rng));
        TTNC = jobs.top();
        jobs.pop();
        state += 1;
    }
};

}  // namespace queue_sim
