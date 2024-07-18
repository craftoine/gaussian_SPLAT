#cmake which simply run nvcc -I/usr/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui main.cu -O3 --expt-relaxed-constexpr


cmake_minimum_required(VERSION 3.17)
project(render_transmittance LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
add_executable(main main.cu)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CUDA)
set( CMAKE_CXX_FLAGS  "-O3")
set( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
find_package(OpenCV REQUIRED)
target_link_libraries(main PRIVATE opencv_core opencv_videoio opencv_highgui)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)