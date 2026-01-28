cmake_minimum_required(VERSION 3.19)

project(nsg_benchmark)

set(NSG_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)

# Benchmark executable
add_executable(nsg_build_and_test
    ${PROJECT_SOURCE_DIR}/src/nsg_build_and_test.cpp)

# Link against the main library (defined in root)
target_link_libraries(nsg_build_and_test PUBLIC efanna2e OpenMP::OpenMP_CXX Boost::headers)

target_include_directories(nsg_build_and_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(nsg_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")

