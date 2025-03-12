cmake_minimum_required(VERSION 3.10)

project(DistributedArray)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# KaMPIng 
include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.1.1
)

FetchContent_MakeAvailable(kamping)

# KaGen
include(FetchContent)
FetchContent_Declare(KaGen 
  GIT_REPOSITORY https://github.com/sebalamm/KaGen.git 
  GIT_TAG master)

FetchContent_MakeAvailable(KaGen)
# set_property(DIRECTORY "${KaGen_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES) # optional

add_executable(myapp primitives/distribution_strategy.hpp primitives/distributed_array.hpp main.cpp
        algorithms/distributed/bfs.hpp
        algorithms/distributed/page_rank.hpp
        algorithms/distributed/connected_component.hpp
        algorithms/distributed/label_propogation.hpp)
target_link_libraries(myapp PRIVATE kamping::kamping)
target_link_libraries(myapp PUBLIC KaGen::KaGen)