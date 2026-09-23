cmake_minimum_required(VERSION 3.16)
project(nn_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NN_BUILD_TESTS "Build unit tests" ON)
option(NN_BUILD_APPS "Build example apps" ON)

add_library(nn_core STATIC
	src/tensor.cpp
	src/autograd/value.cpp
	src/autograd/ops.cpp
	src/nn/linear.cpp
	src/nn/conv2d.cpp
	src/nn/pool2d.cpp
	src/nn/activation.cpp
	src/nn/sequential.cpp
	src/optim/sgd.cpp
	src/data/dataloader.cpp
)

target_include_directories(nn_core PUBLIC include)

if(NN_BUILD_TESTS)
	enable_testing()
	include(${CMAKE_SOURCE_DIR}/cmake/FetchDoctest.cmake)
	include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
endif()

if(NN_BUILD_APPS)
	add_subdirectory(apps)
endif()

if(NN_BUILD_TESTS)
	add_subdirectory(tests)
endif()
