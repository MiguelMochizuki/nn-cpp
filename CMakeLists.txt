cmake_minimum_required(VERSION 3.16)
project(nn_cpp VERSION 1.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NN_BUILD_TESTS "Build unit tests" ON)
option(NN_BUILD_APPS "Build example apps" ON)
option(NN_NATIVE "Compile with -march=native" OFF)

add_library(nn_core STATIC
	src/tensor.cpp
	src/random.cpp
	src/kernels/gemm.cpp
	src/kernels/parallel.cpp
	src/autograd/value.cpp
	src/autograd/ops.cpp
	src/nn/linear.cpp
	src/nn/embedding.cpp
	src/nn/layernorm.cpp
	src/nn/positional.cpp
	src/nn/conv2d.cpp
	src/nn/pool2d.cpp
	src/nn/dropout.cpp
	src/nn/batchnorm.cpp
	src/nn/activation.cpp
	src/nn/sequential.cpp
	src/checkpoint.cpp
	src/optim/sgd.cpp
	src/optim/adam.cpp
	src/optim/adamw.cpp
	src/optim/scheduler.cpp
	src/optim/plateau.cpp
	src/optim/clip.cpp
	src/metrics.cpp
	src/data/dataloader.cpp
	src/data/fer2013.cpp
	src/data/subset.cpp
	src/data/transform.cpp
)

target_include_directories(nn_core PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(nn_core PROPERTIES EXPORT_NAME core)

find_package(Threads REQUIRED)
target_link_libraries(nn_core PUBLIC Threads::Threads)

if(NN_NATIVE)
	if(MSVC)
		message(WARNING "NN_NATIVE has no effect on MSVC; pass /arch:AVX2 yourself")
	else()
		target_compile_options(nn_core PUBLIC -march=native)
	endif()
	target_compile_definitions(nn_core PUBLIC NN_NATIVE_BUILD=1)
else()
	target_compile_definitions(nn_core PUBLIC NN_NATIVE_BUILD=0)
endif()

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

# Install: the library, its headers, the apps and a package config for find_package(nn_cpp)
install(TARGETS nn_core EXPORT nn_cppTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/nn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(NN_BUILD_APPS)
	install(TARGETS xor_demo train_mnist train_fer2013 bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set(NN_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/nn_cpp)
install(EXPORT nn_cppTargets NAMESPACE nn:: DESTINATION ${NN_CMAKE_DIR})
configure_package_config_file(cmake/nn_cppConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/nn_cppConfig.cmake INSTALL_DESTINATION ${NN_CMAKE_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/nn_cppConfigVersion.cmake
	COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nn_cppConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/nn_cppConfigVersion.cmake DESTINATION ${NN_CMAKE_DIR})
