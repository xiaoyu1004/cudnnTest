cmake_minimum_required(VERSION 3.10)
project(cudnnTest)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

option(ENABLE_CUDNN "" ON)
if(ENABLE_CUDNN)
    add_definitions(-DENABLE_CUDNN)
endif()

add_executable(reduceTest ./test/reduce5DTest.cpp ./src/reduce/reduce5D.cu)
target_include_directories(reduceTest PRIVATE src /usr/local/cuda/include)
target_link_libraries(reduceTest PUBLIC cudnn)