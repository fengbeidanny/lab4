cmake_minimum_required(VERSION 3.12)
project(bicgstab C CXX Fortran)

find_package(OpenMP REQUIRED)
# Uncomment below is MPI env is available
find_package(MPI REQUIRED)

# Set Compiler
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_Fortran_COMPILER "gfortran")

# Set flags
set(CMAKE_C_FLAGS "-Og -g")
set(CMAKE_CXX_FLAGS "-Og -g")
set(CMAKE_Fortran_FLAGS "-Og -g")

# Include headers
include_directories(include)
include_directories(${MPI_INCLUDE_PATH})


# Add common source files
file(GLOB COMMON_SOURCES "src/*.cpp")

# Create an object library for common sources
add_library(judger OBJECT ${COMMON_SOURCES})

# Add C implementation
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.c")
  add_executable(bicgstab src/bicgstab/solver.c $<TARGET_OBJECTS:judger>)

  if(OpenMP_C_FOUND)
    target_link_libraries(bicgstab PUBLIC OpenMP::OpenMP_C)
  endif()

  target_link_libraries(bicgstab PUBLIC MPI::MPI_C)
endif()

# Add Fortran implementation
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.f90")
  add_executable(bicgstab-fortran src/bicgstab/solver.f90 $<TARGET_OBJECTS:judger>)
  if(OpenMP_Fortran_FOUND)
    target_link_libraries(bicgstab-fortran PUBLIC OpenMP::OpenMP_Fortran)
  endif()
endif()
