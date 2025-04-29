cmake_minimum_required(VERSION 3.10)

project(DistributedArray)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
   kaval
   GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
   GIT_TAG f01703d
   SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)
FetchContent_MakeAvailable(kaval)

add_subdirectory(external/KaMPIng)
add_subdirectory(external/KaGen)
add_subdirectory(external/CLI11)

add_executable(distributed_graph_cli src/main.cpp)
add_executable(run_bfs src/run_bfs.cpp)
add_executable(run_cc src/run_cc.cpp)

target_link_libraries(distributed_graph_cli PRIVATE kamping::kamping)
target_link_libraries(distributed_graph_cli PRIVATE KaGen::KaGen)
target_link_libraries(distributed_graph_cli PRIVATE CLI11::CLI11)
target_link_libraries(run_bfs PRIVATE kamping::kamping)
target_link_libraries(run_bfs PRIVATE KaGen::KaGen)
target_link_libraries(run_bfs PRIVATE CLI11::CLI11)
target_link_libraries(run_cc PRIVATE kamping::kamping)
target_link_libraries(run_cc PRIVATE KaGen::KaGen)
target_link_libraries(run_cc PRIVATE CLI11::CLI11)    