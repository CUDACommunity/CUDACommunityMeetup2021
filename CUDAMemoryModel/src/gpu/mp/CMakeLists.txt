cmake_minimum_required(VERSION 3.17)
project(CUDAMessageChain CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CUDAMessageChain main.cu)

set_target_properties(
        CUDAMessageChain
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)