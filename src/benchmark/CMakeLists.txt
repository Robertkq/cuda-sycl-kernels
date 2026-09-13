
find_package(CLI11 REQUIRED)

add_library(benchmark STATIC
    benchmark.cpp
)

target_include_directories(benchmark PUBLIC 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/benchmark
)

target_link_libraries(benchmark PRIVATE
    CLI11::CLI11
)

if(TBB_LIBRARY)
  target_link_libraries(benchmark PUBLIC ${TBB_LIBRARY})
endif()