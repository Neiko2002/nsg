add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search PRIVATE ${PROJECT_NAME})
