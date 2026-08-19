cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(bumblebee LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bumblebee src/main.cpp src/hello.cu)

target_compile_options(bumblebee PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-freflection>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
