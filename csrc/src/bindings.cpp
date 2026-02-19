#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "queue_sim/distributions.hpp"
#include "queue_sim/event_log.hpp"
#include "queue_sim/fcfs.hpp"
#include "queue_sim/queue_system.hpp"
#include "queue_sim/server.hpp"
#include "queue_sim/srpt.hpp"
#include "queue_sim/ps.hpp"
#include "queue_sim/fb.hpp"

namespace py = pybind11;
using namespace queue_sim;

PYBIND11_MODULE(_queue_sim_cpp, m) {
    m.doc() = "C++ backend for queue_sim — hot-path event loop";

    // -- Distributions -------------------------------------------------------

    py::class_<ExponentialDist>(m, "ExponentialDist")
        .def(py::init<double>(), py::arg("mu"));

    py::class_<UniformDist>(m, "UniformDist")
        .def(py::init<double, double>(), py::arg("a"), py::arg("b"));

    py::class_<BoundedParetoDist>(m, "BoundedParetoDist")
        .def(py::init<double, double, double>(),
             py::arg("k"), py::arg("p"), py::arg("alpha"));

    // -- Server (abstract — not directly constructible) ----------------------

    py::class_<Server, std::shared_ptr<Server>>(m, "Server")
        .def_readonly("T", &Server::T)
        .def_readonly("num_completions", &Server::num_completions)
        .def_readonly("state", &Server::state)
        .def_readonly("clock", &Server::clock)
        .def_readonly("num_servers", &Server::num_servers)
        .def_readonly("buffer_capacity", &Server::buffer_capacity)
        .def_readonly("num_rejected", &Server::num_rejected)
        .def_readonly("num_arrivals", &Server::num_arrivals)
        .def("is_full", &Server::is_full)
        .def("queryTTNC", &Server::queryTTNC);

    // -- FCFS ----------------------------------------------------------------

    py::class_<FCFS, Server, std::shared_ptr<FCFS>>(m, "FCFS")
        .def(py::init([](py::object dist, int num_servers,
                         int buffer_capacity) -> FCFS {
            if (py::isinstance<ExponentialDist>(dist))
                return FCFS(dist.cast<ExponentialDist>(), num_servers,
                            buffer_capacity);
            if (py::isinstance<UniformDist>(dist))
                return FCFS(dist.cast<UniformDist>(), num_servers,
                            buffer_capacity);
            if (py::isinstance<BoundedParetoDist>(dist))
                return FCFS(dist.cast<BoundedParetoDist>(), num_servers,
                            buffer_capacity);
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"), py::arg("num_servers") = 1,
            py::arg("buffer_capacity") = -1);

    // -- SRPT ----------------------------------------------------------------

    py::class_<SRPT, Server, std::shared_ptr<SRPT>>(m, "SRPT")
        .def(py::init([](py::object dist, int buffer_capacity) -> SRPT {
            if (py::isinstance<ExponentialDist>(dist))
                return SRPT(dist.cast<ExponentialDist>(), buffer_capacity);
            if (py::isinstance<UniformDist>(dist))
                return SRPT(dist.cast<UniformDist>(), buffer_capacity);
            if (py::isinstance<BoundedParetoDist>(dist))
                return SRPT(dist.cast<BoundedParetoDist>(), buffer_capacity);
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"), py::arg("buffer_capacity") = -1);

    // -- PS ------------------------------------------------------------------

    py::class_<PS, Server, std::shared_ptr<PS>>(m, "PS")
        .def(py::init([](py::object dist, int num_servers,
                         int buffer_capacity) -> PS {
            if (py::isinstance<ExponentialDist>(dist))
                return PS(dist.cast<ExponentialDist>(), num_servers,
                          buffer_capacity);
            if (py::isinstance<UniformDist>(dist))
                return PS(dist.cast<UniformDist>(), num_servers,
                          buffer_capacity);
            if (py::isinstance<BoundedParetoDist>(dist))
                return PS(dist.cast<BoundedParetoDist>(), num_servers,
                          buffer_capacity);
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"), py::arg("num_servers") = 1,
            py::arg("buffer_capacity") = -1);

    // -- FB ------------------------------------------------------------------

    py::class_<FB, Server, std::shared_ptr<FB>>(m, "FB")
        .def(py::init([](py::object dist, int buffer_capacity) -> FB {
            if (py::isinstance<ExponentialDist>(dist))
                return FB(dist.cast<ExponentialDist>(), buffer_capacity);
            if (py::isinstance<UniformDist>(dist))
                return FB(dist.cast<UniformDist>(), buffer_capacity);
            if (py::isinstance<BoundedParetoDist>(dist))
                return FB(dist.cast<BoundedParetoDist>(), buffer_capacity);
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"), py::arg("buffer_capacity") = -1);

    // -- EventLog ------------------------------------------------------------

    py::class_<EventLog>(m, "EventLog")
        .def_readonly("times", &EventLog::times)
        .def_readonly("kinds", &EventLog::kinds)
        .def_readonly("from_servers", &EventLog::from_servers)
        .def_readonly("to_servers", &EventLog::to_servers)
        .def_readonly("states", &EventLog::states)
        .def("__len__", &EventLog::size)
        .def_property_readonly_static("ARRIVAL", [](py::object) { return EventLog::ARRIVAL; })
        .def_property_readonly_static("DEPARTURE", [](py::object) { return EventLog::DEPARTURE; })
        .def_property_readonly_static("ROUTE", [](py::object) { return EventLog::ROUTE; })
        .def_property_readonly_static("REJECTION", [](py::object) { return EventLog::REJECTION; })
        .def_property_readonly_static("EXTERNAL", [](py::object) { return EventLog::EXTERNAL; })
        .def_property_readonly_static("SYSTEM_EXIT", [](py::object) { return EventLog::SYSTEM_EXIT; });

    // -- ReplicationRawResult ------------------------------------------------

    py::class_<ReplicationRawResult>(m, "ReplicationRawResult")
        .def_readonly("raw_N", &ReplicationRawResult::raw_N)
        .def_readonly("raw_T", &ReplicationRawResult::raw_T);

    // -- QueueSystem ---------------------------------------------------------

    py::class_<QueueSystem>(m, "QueueSystem")
        .def(py::init([](std::vector<std::shared_ptr<Server>> servers,
                         py::object arrivalDist,
                         std::vector<std::vector<double>> tm) -> QueueSystem {
            if (py::isinstance<ExponentialDist>(arrivalDist))
                return QueueSystem(std::move(servers),
                                   arrivalDist.cast<ExponentialDist>(),
                                   std::move(tm));
            if (py::isinstance<UniformDist>(arrivalDist))
                return QueueSystem(std::move(servers),
                                   arrivalDist.cast<UniformDist>(),
                                   std::move(tm));
            if (py::isinstance<BoundedParetoDist>(arrivalDist))
                return QueueSystem(std::move(servers),
                                   arrivalDist.cast<BoundedParetoDist>(),
                                   std::move(tm));
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }),
             py::arg("servers"),
             py::arg("arrivalfn"),
             py::arg("transitionMatrix") = std::vector<std::vector<double>>{})
        .def("sim", &QueueSystem::sim,
             py::arg("num_events") = 1000000,
             py::arg("seed") = -1,
             py::arg("warmup") = 0,
             py::arg("track_response_times") = false,
             py::arg("track_events") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("replicate", &QueueSystem::replicate,
             py::arg("n_replications") = 30,
             py::arg("num_events") = 1000000,
             py::arg("seed") = -1,
             py::arg("warmup") = 0,
             py::arg("n_threads") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("addServer", &QueueSystem::addServer)
        .def("updateTransitionMatrix", &QueueSystem::updateTransitionMatrix)
        .def_readonly("T", &QueueSystem::T)
        .def_readonly("response_times", &QueueSystem::response_times)
        .def_readonly("event_log", &QueueSystem::event_log);
}
