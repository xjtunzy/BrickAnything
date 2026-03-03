cmake_minimum_required(VERSION 3.14)
project(voxel2brick_cpp LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# 1. 配置 Eigen (Header-only)
# ----------------------------------------------------------------------------
# 指向你下载的 external/eigen 目录
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen")

if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Eigen not found at ${EIGEN3_INCLUDE_DIR}. Please check the path.")
endif()

# ----------------------------------------------------------------------------
# 2. 配置 pybind11 (处理找不到 pybind11Config.cmake 的问题)
# ----------------------------------------------------------------------------
# 尝试通过 Python 自动获取 pybind11 的 CMake 路径
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
    message(STATUS "Found pybind11 via python: ${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 REQUIRED)

# ----------------------------------------------------------------------------
# 3. 创建 pybind11 模块
# ----------------------------------------------------------------------------
pybind11_add_module(voxel2brick_cpp
    bindings.cpp
    lego_block_graph.cpp
)

# ----------------------------------------------------------------------------
# 4. 配置包含路径
# ----------------------------------------------------------------------------
target_include_directories(voxel2brick_cpp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}    # 包含 lego_block_graph.hpp
    ${EIGEN3_INCLUDE_DIR}          # 包含 Eigen
)

# ----------------------------------------------------------------------------
# 5. (可选) 如果需要链接某些库，可以在此处添加
# ----------------------------------------------------------------------------
# target_link_libraries(voxel2brick_cpp PRIVATE ...)
