cmake_minimum_required(VERSION 3.30)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(bumblebee LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bumblebee src/main.cc src/hello.cu)
target_include_directories(bumblebee PRIVATE include)
target_link_libraries(bumblebee PRIVATE CUDA::cudart)

target_compile_options(bumblebee PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-freflection>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

include(CTest)

if(BUILD_TESTING)
    add_executable(host_buffer_test tests/bumblebee/memory/host_buffer_test.cc)
    target_include_directories(host_buffer_test PRIVATE include)
    target_link_libraries(host_buffer_test PRIVATE CUDA::cudart)
    target_compile_options(host_buffer_test PRIVATE -Wall -Wextra)
    add_test(NAME host_buffer COMMAND host_buffer_test)

    add_executable(device_buffer_test tests/bumblebee/memory/device_buffer_test.cc)
    target_include_directories(device_buffer_test PRIVATE include)
    target_link_libraries(device_buffer_test PRIVATE CUDA::cudart)
    target_compile_options(device_buffer_test PRIVATE -Wall -Wextra)
    add_test(NAME device_buffer COMMAND device_buffer_test)
    set_tests_properties(device_buffer PROPERTIES SKIP_RETURN_CODE 77)
endif()
