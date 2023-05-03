include(ExternalProject)

set(SUITESPARSE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(SUITESPARSE_LIBDIR ${SUITESPARSE_INSTALL_DIR}/lib64)
set(SUITESPARSE_BINDIR ${SUITESPARSE_INSTALL_DIR}/bin)
set(SUITESPARSE_INCDIR ${SUITESPARSE_INSTALL_DIR}/include)

ExternalProject_Add(ss_config
  URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.0.1.tar.gz
  SOURCE_DIR "${CMAKE_BINARY_DIR}/SuiteSparse"
  SOURCE_SUBDIR SuiteSparse_config
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${SUITESPARSE_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

ExternalProject_Add(ss_colamd
  DOWNLOAD_COMMAND ""
  SOURCE_DIR "${CMAKE_BINARY_DIR}/SuiteSparse"
  SOURCE_SUBDIR COLAMD
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${SUITESPARSE_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

ExternalProject_Add(ss_amd
  DOWNLOAD_COMMAND ""
  SOURCE_DIR "${CMAKE_BINARY_DIR}/SuiteSparse"
  SOURCE_SUBDIR AMD
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${SUITESPARSE_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

ExternalProject_Add(ss_cholmod
  DOWNLOAD_COMMAND ""
  SOURCE_DIR "${CMAKE_BINARY_DIR}/SuiteSparse"
  SOURCE_SUBDIR CHOLMOD
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${SUITESPARSE_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

add_dependencies(ss_colamd ss_config)
add_dependencies(ss_amd ss_config)
add_dependencies(ss_cholmod ss_amd ss_colamd)

add_dependencies(lsbench ss_cholmod)
target_link_libraries(lsbench PRIVATE 
  ${SUITESPARSE_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}cholmod${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${SUITESPARSE_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}colamd${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${SUITESPARSE_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}amd${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${SUITESPARSE_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}suitesparseconfig${CMAKE_SHARED_LIBRARY_SUFFIX})
target_include_directories(lsbench PRIVATE ${SUITESPARSE_INCDIR})