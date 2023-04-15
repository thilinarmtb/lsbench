cmake_minimum_required(VERSION 3.18)
project(cusparse_cholesky VERSION 0.0.1 DESCRIPTION
  "Benchmark linear solve times" LANGUAGES C CXX)

option(ENABLE_CUSPARSE "Enable cusaprse support." OFF)
option(ENABLE_HYPRE "Enable Hypre support." ON)
option(ENABLE_AMGX "Enable AmgX support." OFF)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(SOURCES src/lsbench.c src/lsbench-csr.c)
if (ENABLE_CUSPARSE)
  set(SOURCES ${SOURCES} src/cusparse.c)
endif()
if (ENABLE_HYPRE)
  set(SOURCES ${SOURCES} src/hypre.c)
endif()
if (ENABLE_AMGX)
  set(SOURCES ${SOURCES} src/amgx.c)
endif()

add_library(lsbench SHARED ${SOURCES})
target_include_directories(lsbench PRIVATE src)
set_target_properties(lsbench PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/lsbench.h)

if (ENABLE_CUSPARSE)
  include(libs/cusparse.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_CUSPARSE)
endif()
if (ENABLE_HYPRE)
  include(libs/hypre.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_HYPRE)
endif()
if (ENABLE_AMGX)
  include(libs/amgx.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_AMGX)
endif()

add_executable(driver bin/driver.c)
target_link_libraries(driver lsbench)
target_include_directories(driver PUBLIC src)

find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format COMMAND ${CLANG_FORMAT} -i
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format ...")
endif()

install(TARGETS lsbench driver RUNTIME DESTINATION bin LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
