# Copyright (c) 2024 Advanced Micro Devices, Inc.
include(CMakePrintHelpers)

message("Hellooooo_torch_cpp_gpu")
set(target custom_ops_dml_gpu)
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)
source_group(
  TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SRC}
)

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
message(CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")
message(target="${target}")
set(LINK_LIBS "")
add_library(${target} STATIC ${SRC})

# download nuget to restore the dependency packages
set(NUGET_EXE ${CMAKE_BINARY_DIR}/nuget/nuget.exe)
set(NUGET_PACKAGES_DIR ${PROJECT_SOURCE_DIR}/external/packages)
if(NOT EXISTS ${NUGET_EXE})
  message(STATUS "Downloading NuGet...")
  file(DOWNLOAD https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
       ${NUGET_EXE}
  )
endif()

# run nuget command to retore from packages.config which lists the packages. the
# packages are saved in external folder in the root directory
message(STATUS "Restoring nuget packages...")
execute_process(
  COMMAND ${NUGET_EXE} restore ${PROJECT_SOURCE_DIR}/packages.config
          -PackagesDirectory ${NUGET_PACKAGES_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/nuget
)

# add preprocessor to use the latest DML version
target_compile_definitions(${target} PRIVATE DML_TARGET_VERSION_USE_LATEST)
target_include_directories(
  ${target}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/external/packages/Microsoft.AI.DirectML.1.15.0/include
    .
)
target_link_libraries(
  ${target} PRIVATE ${LINK_LIBS} "d3d12.lib" "dxgi.lib" "directml.lib"
)
message(STATUS "exiting cmake in torch_cpp/gpu")
