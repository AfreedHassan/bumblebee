cmake_minimum_required(VERSION 3.30)
project(bumblebee LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bumblebee src/main.cpp src/hello.cu)

# Target GPU architectures (Ampere=86, Hopper=90)
set_target_properties(bumblebee PROPERTIES
    CUDA_ARCHITECTURES "86;90"
)

target_compile_options(bumblebee PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-freflection>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
