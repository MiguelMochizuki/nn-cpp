cmake_minimum_required(VERSION 3.16)
project(nn_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NN_BUILD_TESTS "Build unit tests" ON)
option(NN_BUILD_APPS "Build example apps" ON)

add_library(nn_core STATIC
	src/tensor.cpp
)

target_include_directories(nn_core PUBLIC include)

if(NN_BUILD_APPS)
	add_subdirectory(apps)
endif()

if(NN_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
