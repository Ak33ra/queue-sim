#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "server.hpp"

namespace queue_sim {

class PS : public Server {
public:
    std::vector<double> remaining;
    std::vector<double> jobArrivals;

    explicit PS(Distribution sizeDist) : Server(std::move(sizeDist)) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<PS>(sizeDist);
    }

    void reset() override {
        Server::reset();
        remaining.clear();
        jobArrivals.clear();
    }

    double nextJob() override {
        return sample(sizeDist, *rng);
    }

    void updateET() override {
        // PS computes response times directly in update(); no-op here.
    }

    void arrival() override {
        remaining.push_back(sample(sizeDist, *rng));
        jobArrivals.push_back(clock);
        state += 1;
        recalcTTNC();
    }

    bool update(double dt) override {
        TTNC -= dt;
        clock += dt;
        if (state == 0) return false;

        double work = dt / state;
        for (auto &r : remaining) r -= work;

        if (TTNC <= 0.0) {
            auto it = std::min_element(remaining.begin(), remaining.end());
            int idx = static_cast<int>(it - remaining.begin());
            double response_time = clock - jobArrivals[idx];
            remaining.erase(it);
            jobArrivals.erase(jobArrivals.begin() + idx);
            state -= 1;
            num_completions += 1;
            double n = static_cast<double>(num_completions);
            T = T * (n - 1.0) / n + response_time / n;
            recalcTTNC();
            return true;
        }
        return false;
    }

private:
    void recalcTTNC() {
        if (remaining.empty()) {
            TTNC = std::numeric_limits<double>::infinity();
            return;
        }
        double min_rem =
            *std::min_element(remaining.begin(), remaining.end());
        TTNC = min_rem * state;
    }
};

}  // namespace queue_sim
