#pragma once

#include <algorithm>
#include <deque>
#include <limits>
#include <vector>

#include "server.hpp"

namespace queue_sim {

class FCFS : public Server {
public:
    // Multi-server state (only used when num_servers > 1)
    std::vector<double> channelRemaining;
    std::vector<double> channelArrivals;
    std::deque<double> waitQueue;

    explicit FCFS(Distribution sizeDist, int num_servers = 1)
        : Server(std::move(sizeDist), num_servers) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<FCFS>(sizeDist, num_servers);
    }

    void reset() override {
        Server::reset();
        channelRemaining.clear();
        channelArrivals.clear();
        waitQueue.clear();
    }

    double nextJob() override {
        return sample(sizeDist, *rng);
    }

    void updateET() override {
        if (num_servers == 1) {
            Server::updateET();
            return;
        }
        // For k>1, jobs depart out of arrival order — no-op.
        // Response time is computed directly in update().
    }

    void arrival() override {
        if (num_servers == 1) {
            Server::arrival();
            return;
        }
        state += 1;
        if (static_cast<int>(channelRemaining.size()) < num_servers) {
            // Free channel available — start immediately
            channelRemaining.push_back(sample(sizeDist, *rng));
            channelArrivals.push_back(clock);
            recalcTTNC();
        } else {
            // All channels busy — queue
            waitQueue.push_back(clock);
        }
    }

    bool update(double time_elapsed) override {
        if (num_servers == 1) {
            return Server::update(time_elapsed);
        }

        clock += time_elapsed;

        // Subtract elapsed time from all active channels
        for (auto &r : channelRemaining) r -= time_elapsed;
        TTNC -= time_elapsed;

        if (TTNC <= 0.0) {
            // Find the channel with minimum remaining (the one that completed)
            auto it = std::min_element(
                channelRemaining.begin(), channelRemaining.end());
            int idx = static_cast<int>(it - channelRemaining.begin());

            // Compute response time for the departing job
            double response_time = clock - channelArrivals[idx];
            num_completions += 1;
            double n = static_cast<double>(num_completions);
            T = T * (n - 1.0) / n + response_time / n;

            // Remove completed channel
            channelRemaining.erase(it);
            channelArrivals.erase(channelArrivals.begin() + idx);
            state -= 1;

            // Pull from wait queue if non-empty
            if (!waitQueue.empty()) {
                double arrTime = waitQueue.front();
                waitQueue.pop_front();
                channelRemaining.push_back(sample(sizeDist, *rng));
                channelArrivals.push_back(arrTime);
                state += 1;
            }

            recalcTTNC();
            return true;
        }
        return false;
    }

private:
    void recalcTTNC() {
        if (channelRemaining.empty()) {
            TTNC = std::numeric_limits<double>::infinity();
            return;
        }
        TTNC = *std::min_element(
            channelRemaining.begin(), channelRemaining.end());
    }
};

}  // namespace queue_sim
