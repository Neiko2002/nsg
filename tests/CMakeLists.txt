add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_index_sift1m test_nsg_index_sift1m.cpp)
target_link_libraries(test_nsg_index_sift1m PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_index_sift1m_evaluation test_nsg_index_sift1m_evaluation.cpp)
target_link_libraries(test_nsg_index_sift1m_evaluation PRIVATE ${PROJECT_NAME})

add_executable(test_nsg_index_sift1m_stats test_nsg_index_sift1m_stats.cpp)
target_link_libraries(test_nsg_index_sift1m_stats PRIVATE ${PROJECT_NAME})

