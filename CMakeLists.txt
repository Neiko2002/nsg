cmake_minimum_required(VERSION 3.15)

project(efanna2e)

#OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)

add_subdirectory(src)
add_subdirectory(tests)
