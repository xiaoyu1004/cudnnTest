cmake_minimum_required(VERSION 3.10)
project(cudnnTest)
# enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 86)

option(ENABLE_CUDNN "" OFF)

set(ROOT_PATH "/usr/local/corex/" CACHE PATH "cuda installation root dir")
set(CUDA_TOOLKIT_ROOT_DIR ${ROOT_PATH} CACHE PATH "root path to cuda/lib64 cuda/bin etc")

set(COMPILER_PATH ${ROOT_PATH}/bin CACHE PATH "compiler binary path")
set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++ CACHE PATH "CXX compiler binary path")
set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang CACHE PATH "C compiler binary path")
set(CUDA_ARCH "ivcore11" CACHE STRING "CUDA architecture.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)

cuda_add_executable(reduceTest ./test/reduce5DTest.cpp ./src/reduce/reduce5D.cu)
target_include_directories(reduceTest PRIVATE src)
target_link_libraries(reduceTest PRIVATE cudart)

if(ENABLE_CUDNN)
    add_definitions(-DENABLE_CUDNN)
    target_include_directories(reduceTest PRIVATE src /usr/local/corex/include)
    # target_link_directories(reduceTest PRIVATE BEFORE /home/yuhongzhang/projects/ixdnn/build/lib)
    target_link_libraries(reduceTest PUBLIC /home/yuhongzhang/projects/ixdnn/build/lib/libcudnn.so)
endif()