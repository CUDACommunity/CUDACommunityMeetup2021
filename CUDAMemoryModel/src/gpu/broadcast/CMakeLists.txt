cmake_minimum_required(VERSION 3.17)
project(CUDABroadcast CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDABroadcast main.cu)

set_target_properties(
        CUDABroadcast
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(CUDABroadcast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86 -O3>)
