add_executable(xor_demo xor/xor_demo.cpp)
target_link_libraries(xor_demo PRIVATE nn_core)

add_executable(train_mnist
	mnist/idx_dataset.cpp
	mnist/mnist_dataset.cpp
	mnist/train_mnist.cpp
)
target_link_libraries(train_mnist PRIVATE nn_core)
target_include_directories(train_mnist PRIVATE mnist)

if(NN_BUILD_TESTS)
	add_executable(mnist_tests
		mnist/idx_dataset.cpp
		mnist/mnist_dataset.cpp
		mnist/test_mnist_data.cpp
	)
	target_link_libraries(mnist_tests PRIVATE nn_core doctest::doctest)
	target_include_directories(mnist_tests PRIVATE mnist)
	doctest_discover_tests(mnist_tests)
endif()
