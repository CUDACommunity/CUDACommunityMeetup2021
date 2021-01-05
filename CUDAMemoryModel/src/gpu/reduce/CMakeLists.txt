cmake_minimum_required(VERSION 3.17)
project(CUDAReduce CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDAReduce main.cu)
target_compile_options(CUDAReduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>)

set_target_properties(
        CUDAReduce
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)