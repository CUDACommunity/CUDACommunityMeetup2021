cmake_minimum_required(VERSION 3.17)
project(CUDAScan CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(CUDAScan main.cu)

target_compile_options(CUDAScan  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>)

set_target_properties(
        CUDAScan
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
