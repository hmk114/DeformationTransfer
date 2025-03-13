cmake_minimum_required(VERSION 3.22)

project(DeformTrans CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wno-deprecated-enum-enum-conversion)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(src/deps/eigen-3.4.0)
include_directories(src/deps/OpenMesh-11.0.0/src)
find_package(OpenMesh REQUIRED PATHS src/deps/OpenMesh-11.0.0/build)

add_executable(deformtrans src/main.cpp)
target_link_directories(deformtrans PRIVATE src/deps/OpenMesh-11.0.0/build/Build/lib)
target_link_libraries(deformtrans OpenMeshCore)
