#pragma once
#include <string>
#include <vector>

namespace queue_sim {

struct EventLog {
    // -- Event kind constants (match Python EventLog) --
    static constexpr const char* ARRIVAL    = "arrival";
    static constexpr const char* DEPARTURE  = "departure";
    static constexpr const char* ROUTE      = "route";
    static constexpr const char* REJECTION  = "rejection";

    // -- Special server indices --
    static constexpr int EXTERNAL    = -1;
    static constexpr int SYSTEM_EXIT = -1;

    std::vector<double> times;
    std::vector<std::string> kinds;
    std::vector<int> from_servers;
    std::vector<int> to_servers;
    std::vector<int> states;

    void push(double time, const char* kind,
              int from_server, int to_server, int state) {
        times.push_back(time);
        kinds.emplace_back(kind);
        from_servers.push_back(from_server);
        to_servers.push_back(to_server);
        states.push_back(state);
    }

    void clear() {
        times.clear(); kinds.clear();
        from_servers.clear(); to_servers.clear();
        states.clear();
    }

    void reserve(size_t n) {
        times.reserve(n); kinds.reserve(n);
        from_servers.reserve(n); to_servers.reserve(n);
        states.reserve(n);
    }

    size_t size() const { return times.size(); }
};

}  // namespace queue_sim
