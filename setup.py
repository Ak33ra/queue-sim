"""Build configuration for the C++ extension module."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "_queue_sim_cpp",
        ["csrc/src/bindings.cpp"],
        include_dirs=["csrc/include"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
