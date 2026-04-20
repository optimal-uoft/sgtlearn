#include <pybind11/pybind11.h>

#include "sgtlearn/adder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<sgtlearn::Adder>(m, "Adder")
        .def(py::init<>())
        .def("add", &sgtlearn::Adder::add);
}
