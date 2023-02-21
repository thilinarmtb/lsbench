enable_language(CUDA)
find_package(CUDAToolkit 11.0 REQUIRED)
target_link_libraries(cholbench PRIVATE CUDA::cudart CUDA::cusparse
  CUDA::cusolver)
set_target_properties(cholbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
