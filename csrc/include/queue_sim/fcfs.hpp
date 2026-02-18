#pragma once

#include "server.hpp"

namespace queue_sim {

class FCFS : public Server {
public:
    explicit FCFS(Distribution sizeDist) : Server(std::move(sizeDist)) {}

    double nextJob() override {
        return sample(sizeDist, *rng);
    }
};

}  // namespace queue_sim
