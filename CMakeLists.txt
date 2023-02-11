cmake_minimum_required(VERSION 3.18)
project(cusparse_cholesky VERSION 0.0.1 DESCRIPTION
  "Benchmark cusparse Cholesky solve times" LANGUAGES C CUDA)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

find_package(CUDAToolkit REQUIRED)

set(SOURCES src/cholbench.c src/cusparse.c)

add_library(cholbench SHARED ${SOURCES})
target_include_directories(cholbench PRIVATE src)
target_link_libraries(cholbench CUDA::cudart CUDA::cusparse CUDA::cusolver)
set_target_properties(cholbench PROPERTIES VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/cholbench.h
  CUDA_SEPARABLE_COMPILATION ON)

add_executable(driver bin/driver.c)
target_link_libraries(driver cholbench)
target_include_directories(driver PUBLIC src)

find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format COMMAND ${CLANG_FORMAT} -i
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format ...")
endif()
