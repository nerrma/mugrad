add_executable("test_linear" "test_linear.cpp")

target_link_libraries("test_linear" PRIVATE test_main)
target_compile_options("test_linear" PRIVATE -Wno-error -Wno-self-assign-overloaded)
add_test("test.linear" "test_linear")

add_executable("test_tensor" "test_tensor.cpp")

target_link_libraries("test_tensor" PRIVATE test_main)
target_compile_options("test_tensor" PRIVATE -Wno-error -Wno-self-assign-overloaded)
add_test("test.tensor" "test_tensor")
