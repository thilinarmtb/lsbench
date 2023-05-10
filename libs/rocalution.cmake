include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
  rocalution
  GIT_REPOSITORY https://github.com/thilinarmtb/rocALUTION
  GIT_TAG fixes_for_rocm_5_3_0
)
FetchContent_GetProperties(rocalution)
if (NOT rocalution_POPULATED)
  FetchContent_Populate(rocalution)
endif()

ExternalProject_Add(ext_rocalution
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${rocalution_SOURCE_DIR}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${rocalution_BINARY_DIR}
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  -DCMAKE_CXX_COMPILER=hipcc
)

add_dependencies(lsbench ext_rocalution)
#target_link_libraries(lsbench PRIVATE roc::rocalution)
