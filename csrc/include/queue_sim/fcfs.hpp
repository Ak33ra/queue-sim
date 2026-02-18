#pragma once

#include "server.hpp"

namespace queue_sim {

class FCFS : public Server {
public:
    explicit FCFS(Distribution sizeDist) : Server(std::move(sizeDist)) {}

    std::shared_ptr<Server> clone() const override {
        return std::make_shared<FCFS>(sizeDist);
    }

    double nextJob() override {
        return sample(sizeDist, *rng);
    }
};

}  // namespace queue_sim
