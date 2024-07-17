#cmake which simply run nvcc -O3 main.cpp -o main


cmake_minimum_required(VERSION 3.17)
project(knn_cpu LANGUAGES CXX)

find_package(CUDA REQUIRED)
cuda_add_executable(main main.cpp)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CXX)
SET( CMAKE_CXX_FLAGS  "-O3")