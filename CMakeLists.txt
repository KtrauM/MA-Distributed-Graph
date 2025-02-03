cmake_minimum_required(VERSION 3.10)

project(DistributedArray)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.1
)

FetchContent_MakeAvailable(kamping)

add_executable(myapp distribution_strategy.hpp distributed_array.hpp main.cpp)
target_link_libraries(myapp PRIVATE kamping::kamping)