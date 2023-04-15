# FetchContent_Declare(
#   occa_content
#   GIT_REPOSITORY https://github.com/libocca/occa.git
#   GIT_TAG v1.3.0
# )
# FetchContent_GetProperties(occa_content)
# if (NOT occa_content_POPULATED)
#   FetchContent_Populate(occa_content)
# endif()

# FetchContent_Declare(
#   paranumal_content
#   GIT_REPOSITORY https://github.com/thilinarmtb/libparanumal.git
#   GIT_TAG main
# )
# FetchContent_GetProperties(paranumal_content)
# if (NOT paranumal_content_POPULATED)
#   FetchContent_Populate(paranumal_content)
# endif()

set(OCCA_SOURCE_DIR "$ENV{HOME}/Repos/libparanumal/occa")
add_subdirectory(${OCCA_SOURCE_DIR} ${CMAKE_BINARY_DIR}/occa)

set(LIBP_SOURCE_DIR "$ENV{HOME}/Repos/libparanumal")
add_subdirectory(${LIBP_SOURCE_DIR}/libs ${CMAKE_BINARY_DIR}/libparanumal/libs)

add_dependencies(lsbench parAlmond linearSolver linAlg core libocca)
target_link_libraries(lsbench PUBLIC parAlmond linearSolver linAlg core libocca)
target_include_directories(lsbench PRIVATE ${LIBP_SOURCE_DIR}/include)
