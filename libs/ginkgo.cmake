include(FetchContent)
FetchContent_Declare(
  ginkgo
  GIT_REPOSITORY https://github.com/ginkgo-project/ginkgo.git
  GIT_TAG develop
)
FetchContent_GetProperties(ginkgo)
if (NOT ginkgo_POPULATED)
  FetchContent_Populate(ginkgo)
  set(GINKGO_BUILD_MPI OFF)
  set(GINKGO_BUILD_HWLOC OFF)
  set(GINKGO_BUILD_HIP OFF)
  add_subdirectory(${ginkgo_SOURCE_DIR} ${ginkgo_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_dependencies(lsbench Ginkgo::ginkgo)
target_link_libraries(lsbench PRIVATE Ginkgo::ginkgo)
