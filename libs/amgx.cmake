include(ExternalProject)

set(AMGX_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(AMGX_LIBDIR ${AMGX_INSTALL_DIR}/lib)
set(AMGX_INCDIR ${AMGX_INSTALL_DIR}/include)

find_package(CUDAToolkit 11.0 REQUIRED)

ExternalProject_Add(AMGX_DEVICE
  URL https://github.com/NVIDIA/AMGX/archive/refs/tags/v2.2.0.tar.gz
  BUILD_ALWAYS ON
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_NO_MPI=TRUE
    -DCMAKE_INSTALL_PREFIX=${AMGX_INSTALL_DIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

add_dependencies(lsbench AMGX_DEVICE)
target_link_libraries(lsbench PRIVATE
  ${AMGX_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}amgxsh${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${AMGX_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}amgx${CMAKE_STATIC_LIBRARY_SUFFIX}
  CUDA::cudart CUDA::curand CUDA::cublas CUDA::cusparse CUDA::cusolver) 
target_include_directories(lsbench PRIVATE ${AMGX_INCDIR})
