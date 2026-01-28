cmake_minimum_required(VERSION 3.19)

project(efanna2e)

#OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)

set(DATA_PATH "data" CACHE PATH "Path to datasets directory")
message(STATUS "Data Path: ${DATA_PATH}")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmark)
