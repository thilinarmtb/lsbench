cmake_minimum_required(VERSION 3.18)
project(cusparse_cholesky VERSION 0.0.1 DESCRIPTION
  "Benchmark cusparse Cholesky solve times" LANGUAGES C)

option(ENABLE_HYPRE "Enable Hypre support." ON)
option(ENABLE_CUSPARSE "Enable cusaprse support." ON)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(SOURCES src/cholbench.c src/cholbench-csr.c)
if (ENABLE_CUSPARSE)
  set(SOURCES ${SOURCES} src/cusparse.c)
endif()
if (ENABLE_HYPRE)
  set(SOURCES ${SOURCES} src/hypre.c)
endif()

add_library(cholbench SHARED ${SOURCES})
target_include_directories(cholbench PRIVATE src)
set_target_properties(cholbench PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/cholbench.h)

if (ENABLE_CUSPARSE)
  include(libs/cusparse.cmake)
  target_compile_definitions(cholbench PRIVATE -DCHOLBENCH_CUSPARSE)
endif()
if (ENABLE_HYPRE)
  include(libs/hypre.cmake)
  target_compile_definitions(cholbench PRIVATE -DCHOLBENCH_HYPRE)
endif()

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
