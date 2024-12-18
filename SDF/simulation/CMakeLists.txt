cmake_minimum_required(VERSION 3.10)
project(husky_panda_sdf_simulation)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
set(CUDACXX /usr/local/cuda-11.8/bin/nvcc)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
roscpp
std_msgs
sensor_msgs
)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)


## Declare a catkin package
catkin_package(
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

## Declare the executable target
add_executable(env_dist_check_node src/env_dist_check_node.cpp)

## Add dependencies to ensure correct build order
add_dependencies(env_dist_check_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Link the target against the necessary libraries
target_link_libraries(env_dist_check_node
"${TORCH_LIBRARIES}"
  Eigen3::Eigen
  ${catkin_LIBRARIES}
)
