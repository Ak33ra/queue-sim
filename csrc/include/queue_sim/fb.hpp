#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "server.hpp"

namespace queue_sim {

class FB : public Server {
public:
    struct Job {
        double remaining;
        double attained;
        double arrivalTime;
    };

    std::vector<Job> jobs;

    explicit FB(Distribution sizeDist) : Server(std::move(sizeDist)) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<FB>(sizeDist);
    }

    void reset() override {
        Server::reset();
        jobs.clear();
    }

    double nextJob() override {
        return sample(sizeDist, *rng);
    }

    void updateET() override {
        // FB computes response times directly in update(); no-op here.
    }

    void arrival() override {
        jobs.push_back({sample(sizeDist, *rng), 0.0, clock});
        state += 1;
        recalcTTNC();
    }

    bool update(double dt) override {
        TTNC -= dt;
        clock += dt;
        if (jobs.empty()) return false;

        // Find active set (minimum attained service)
        double min_att = jobs[0].attained;
        for (const auto &j : jobs)
            min_att = std::min(min_att, j.attained);

        int numActive = 0;
        for (const auto &j : jobs) {
            if (j.attained <= min_att + 1e-12) numActive++;
        }

        // Apply work to active jobs only
        double work = dt / numActive;
        for (auto &j : jobs) {
            if (j.attained <= min_att + 1e-12) {
                j.remaining -= work;
                j.attained += work;
            }
        }

        if (TTNC <= 0.0) {
            // Check for completion (remaining ≈ 0)
            for (auto it = jobs.begin(); it != jobs.end(); ++it) {
                if (it->remaining <= 1e-12) {
                    double response_time = clock - it->arrivalTime;
                    jobs.erase(it);
                    state -= 1;
                    num_completions += 1;
                    double n = static_cast<double>(num_completions);
                    T = T * (n - 1.0) / n + response_time / n;
                    recalcTTNC();
                    return true;
                }
            }
            // Level crossing — active set expanded, recalculate
            recalcTTNC();
        }
        return false;
    }

private:
    void recalcTTNC() {
        if (jobs.empty()) {
            TTNC = std::numeric_limits<double>::infinity();
            return;
        }

        double min_att = jobs[0].attained;
        for (const auto &j : jobs)
            min_att = std::min(min_att, j.attained);

        int numActive = 0;
        double min_rem_active = std::numeric_limits<double>::infinity();
        double next_level = std::numeric_limits<double>::infinity();

        for (const auto &j : jobs) {
            if (j.attained <= min_att + 1e-12) {
                numActive++;
                min_rem_active = std::min(min_rem_active, j.remaining);
            } else {
                next_level = std::min(next_level, j.attained);
            }
        }

        double time_to_completion = min_rem_active * numActive;
        double time_to_crossing = (next_level - min_att) * numActive;
        TTNC = std::min(time_to_completion, time_to_crossing);
    }
};

}  // namespace queue_sim
