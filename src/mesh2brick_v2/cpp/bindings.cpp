#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lego_block_graph.hpp"

namespace py = pybind11;

py::dict solve_voxels(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ,
                     int max_iter = 100,
                     uint32_t seed = 12345) {
    auto buf = occ.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("voxels must be a 3D numpy array with shape (X,Y,Z)");
    }

    int X = static_cast<int>(buf.shape[0]);
    int Y = static_cast<int>(buf.shape[1]);
    int Z = static_cast<int>(buf.shape[2]);

    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);

    auto g = LegoBlockGraph::from_occ_u8(ptr, X, Y, Z, seed);
    g.merge_to_maximal();
    int A = g.generate_single_component_analysis(max_iter);
    //g.test_stability_analysis();
    auto out = g.export_result();

    py::list py_bricks;
    for (auto &b : out.bricks) {
        py_bricks.append(py::make_tuple(b[0], b[1], b[2], b[3], b[4]));
    }

    py::dict d;
    d["components"] = A;
    d["bricks"] = py_bricks;
    return d;
}

PYBIND11_MODULE(voxel2brick_cpp, m) {
    m.doc() = "C++ accelerated LegoBlockGraph for voxel-to-brick optimization";

    m.def("solve", &solve_voxels,
          py::arg("voxels"),
          py::arg("max_iter") = 100,
          py::arg("seed") = 12345,
          "Solve lego brick decomposition from voxel occupancy array (X,Y,Z).");
}
