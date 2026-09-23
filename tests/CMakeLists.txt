add_executable(nn_tests
	test_tensor.cpp
	test_autograd.cpp
	test_nn.cpp
	test_optim.cpp
	test_dataloader.cpp
)
target_link_libraries(nn_tests PRIVATE nn_core doctest::doctest)

doctest_discover_tests(nn_tests)
