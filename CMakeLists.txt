cmake_minimum_required(VERSION 3.18)
project(cusparse_cholesky VERSION 0.0.1 DESCRIPTION
  "Benchmark linear solve times" LANGUAGES C CXX)

option(ENABLE_CUSPARSE "Enable cusparse support." OFF)
option(ENABLE_HYPRE "Enable Hypre support." ON)
option(ENABLE_AMGX "Enable AmgX support." OFF)
option(ENABLE_CHOLMOD "Enable Cholmod support." OFF)
option(ENABLE_PARALMOND "Enable parAlmond support." OFF)
option(ENABLE_ROCALUTION "Enable rocAluction support." ON)

option(HYPRE_CUDA_ENABLED "Enable HYPRE CUDA support" OFF)
option(HYPRE_HIP_ENABLED "Enable HYPRE HIP support" ON)
option(HYPRE_DPCPP_ENABLED "Enable HYPRE SYCL support" OFF)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(SOURCES src/lsbench.c src/lsbench-csr.c src/cusparse.c src/hypre.c
  src/amgx.c src/cholmod.c src/paralmond.cpp)

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
if (ENABLE_CHOLMOD)
  include(libs/suitesparse.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_CHOLMOD)
endif()
if (ENABLE_PARALMOND)
  include(libs/paralmond.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_PARALMOND)
endif()
if (ENABLE_ROCALUTION)
  include(libs/rocalution.cmake)
  target_compile_definitions(lsbench PRIVATE -DLSBENCH_ROCALUTION)
endif()

add_executable(driver bin/driver.c)
target_link_libraries(driver lsbench)
target_include_directories(driver PRIVATE src)
set_target_properties(driver PROPERTIES VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER src/lsbench.h)

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
