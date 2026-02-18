#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "queue_sim/distributions.hpp"
#include "queue_sim/fcfs.hpp"
#include "queue_sim/queue_system.hpp"
#include "queue_sim/server.hpp"
#include "queue_sim/srpt.hpp"

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
        .def("queryTTNC", &Server::queryTTNC);

    // -- FCFS ----------------------------------------------------------------

    py::class_<FCFS, Server, std::shared_ptr<FCFS>>(m, "FCFS")
        .def(py::init([](py::object dist) -> FCFS {
            // Accept any of the three distribution types
            if (py::isinstance<ExponentialDist>(dist))
                return FCFS(dist.cast<ExponentialDist>());
            if (py::isinstance<UniformDist>(dist))
                return FCFS(dist.cast<UniformDist>());
            if (py::isinstance<BoundedParetoDist>(dist))
                return FCFS(dist.cast<BoundedParetoDist>());
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"));

    // -- SRPT ----------------------------------------------------------------

    py::class_<SRPT, Server, std::shared_ptr<SRPT>>(m, "SRPT")
        .def(py::init([](py::object dist) -> SRPT {
            if (py::isinstance<ExponentialDist>(dist))
                return SRPT(dist.cast<ExponentialDist>());
            if (py::isinstance<UniformDist>(dist))
                return SRPT(dist.cast<UniformDist>());
            if (py::isinstance<BoundedParetoDist>(dist))
                return SRPT(dist.cast<BoundedParetoDist>());
            throw py::type_error(
                "Expected ExponentialDist, UniformDist, or BoundedParetoDist");
        }), py::arg("sizefn"));

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
        .def_readonly("T", &QueueSystem::T);
}
