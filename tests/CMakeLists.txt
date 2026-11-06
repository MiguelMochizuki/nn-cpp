add_executable(nn_tests
	test_tensor.cpp
	test_random.cpp
	test_autograd.cpp
	test_nn.cpp
	test_transformer.cpp
	test_optim.cpp
	test_dataloader.cpp
	test_metrics.cpp
	test_plateau.cpp
	test_checkpoint.cpp
	test_kernels.cpp
	test_parallel.cpp
	test_training_e2e.cpp
)
target_link_libraries(nn_tests PRIVATE nn_core doctest::doctest)

target_compile_definitions(nn_tests PRIVATE NN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

doctest_discover_tests(nn_tests)
