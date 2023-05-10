include(ExternalProject)

set(HYPRE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(HYPRE_SOURCE_DIR ${CMAKE_BINARY_DIR}/3rd_party/hypre)
set(HYPRE_LIBDIR ${HYPRE_INSTALL_DIR}/lib)
set(HYPRE_INCDIR ${HYPRE_INSTALL_DIR}/include)

if (HYPRE_CUDA_ENABLED)
  set(HYPRE_CUDA_SM 70)
  set(HYPRE_ENABLE_DEVICE_MALLOC_ASYNC OFF)
  find_package(CUDAToolkit 11.0 REQUIRED)
  if(CUDA_VERSION VERSION_GREATER_EQUAL 11.1.0)
    set(HYPRE_CUDA_SM 70 80)
  endif()
  if(CUDA_VERSION VERSION_GREATER_EQUAL 11.2.0)
    set(HYPRE_ENABLE_DEVICE_MALLOC_ASYNC ON)
  endif()
  
  ExternalProject_Add(HYPRE_DEVICE
    URL https://github.com/yslan/hypre/archive/refs/tags/v2.27.1.tar.gz
    SOURCE_DIR ${HYPRE_SOURCE_DIR}
    SOURCE_SUBDIR "src"
    BUILD_ALWAYS ON
    CMAKE_CACHE_ARGS -DHYPRE_CUDA_SM:STRING=${HYPRE_CUDA_SM}
    CMAKE_ARGS -DHYPRE_ENABLE_SHARED=OFF
      -DHYPRE_WITH_MPI=OFF
      -DHYPRE_ENABLE_MIXEDINT=ON
      -DHYPRE_ENABLE_SINGLE=ON
      -DHYPRE_WITH_CUDA=ON
      -DHYPRE_WITH_GPU_AWARE_MPI=OFF
      -DHYPRE_ENABLE_CUSPARSE=ON
      -DHYPRE_ENABLE_DEVICE_MALLOC_ASYNC=${HYPRE_ENABLE_DEVICE_MALLOC_ASYNC}
      -DHYPRE_BUILD_TYPE=RelWithDebInfo
      -DHYPRE_INSTALL_PREFIX=${HYPRE_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=${HYPRE_LIBDIR}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_C_VISIBILITY_PRESET=hidden
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden
      -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
      -DCMAKE_ENABLE_EXPORTS=TRUE
      -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}
  )
  
  add_dependencies(lsbench HYPRE_DEVICE)
  target_link_libraries(lsbench PUBLIC
    ${HYPRE_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}HYPRE${CMAKE_STATIC_LIBRARY_SUFFIX}
    CUDA::cudart CUDA::curand CUDA::cublas CUDA::cusparse CUDA::cusolver) 
  target_include_directories(lsbench PRIVATE ${HYPRE_INCDIR})
elseif (HYPRE_HIP_ENABLED)
  find_package(HIP REQUIRED)
  find_package(rocrand REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocSOLVER REQUIRED)
  find_package(rocSPARSE REQUIRED)
  find_package(hipSPARSE REQUIRED)

  ExternalProject_Add(HYPRE_DEVICE
    URL https://github.com/yslan/hypre/archive/refs/tags/v2.27.1.tar.gz
    SOURCE_DIR ${HYPRE_SOURCE_DIR}
    SOURCE_SUBDIR "src"
    BUILD_ALWAYS ON
    CMAKE_ARGS -DHYPRE_ENABLE_SHARED=OFF
      -DHYPRE_WITH_MPI=OFF
      -DHYPRE_ENABLE_MIXEDINT=ON
      -DHYPRE_ENABLE_SINGLE=ON
      -DHYPRE_WITH_HIP=ON
      -DHYPRE_WITH_GPU_AWARE_MPI=OFF
      # -DHYPRE_ENABLE_ROCSPARSE=ON
      -DHYPRE_ENABLE_DEVICE_MALLOC_ASYNC=${HYPRE_ENABLE_DEVICE_MALLOC_ASYNC}
      -DHYPRE_BUILD_TYPE=RelWithDebInfo
      -DHYPRE_INSTALL_PREFIX=${HYPRE_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=${HYPRE_LIBDIR}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_C_VISIBILITY_PRESET=hidden
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden
      # -DCMAKE_HIP_VISIBILITY_PRESET=hidden
      # -DCMAKE_HIP_HOST_COMPILER=${CMAKE_CXX_COMPILER}
  )

  add_dependencies(lsbench HYPRE_DEVICE)
  target_link_libraries(lsbench PUBLIC
    ${HYPRE_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}HYPRE${CMAKE_STATIC_LIBRARY_SUFFIX}
    hip::host roc::rocrand roc::hipblas roc::rocblas roc::rocsolver
    roc::rocsparse roc::hipsparse)
  target_include_directories(lsbench PRIVATE ${HYPRE_INCDIR})
elseif (HYPRE_DPCPP_ENABLED)
  message(FATAL_ERROR "HYPRE wrapper build does not support DPCPP!")
else()
  ExternalProject_Add(HYPRE_CPU
    URL https://github.com/yslan/hypre/archive/refs/tags/v2.27.1.tar.gz
    SOURCE_DIR ${HYPRE_SOURCE_DIR}
    SOURCE_SUBDIR "src"
    BUILD_ALWAYS ON
    CMAKE_ARGS -DHYPRE_ENABLE_SHARED=OFF
      -DHYPRE_WITH_MPI=OFF
      -DHYPRE_ENABLE_MIXEDINT=ON
      -DHYPRE_ENABLE_SINGLE=ON
      -DHYPRE_BUILD_TYPE=RelWithDebInfo
      -DHYPRE_INSTALL_PREFIX=${HYPRE_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=${HYPRE_LIBDIR}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_FLAGS_RELWITHDEBINFO=${CMAKE_C_FLAGS_RELWITHDEBINFO}
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_C_VISIBILITY_PRESET=hidden
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden
  )
  add_dependencies(lsbench HYPRE_CPU)
  target_link_libraries(lsbench PUBLIC
    ${HYPRE_LIBDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}HYPRE${CMAKE_STATIC_LIBRARY_SUFFIX})
  target_include_directories(lsbench PRIVATE ${HYPRE_INCDIR})
endif()
