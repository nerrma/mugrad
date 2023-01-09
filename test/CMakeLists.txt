set(legal_library_types "" STATIC SHARED MODULE OBJECT)
list(FIND legal_library_types "test_main" library_type_result)
add_library("test_main" test_main.cpp "test_main.cpp")

target_include_directories(test_main PUBLIC .)

add_subdirectory(main)
