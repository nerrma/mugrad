cmake_minimum_required(VERSION 3.23.1)

project(mugrad)

add_library(${PROJECT_NAME} INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

# Project configuration
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config/cmake")
enable_testing()
include(CTest)

#include(add-targets)

include_directories(include)

add_subdirectory(source)
add_subdirectory(test)
