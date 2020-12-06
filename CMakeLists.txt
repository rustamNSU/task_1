cmake_minimum_required(VERSION 3.17)
project(task_1)

set(CMAKE_CXX_STANDARD 17)
# MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

file(GLOB_RECURSE SOURCES MatrixVector.cpp)
file(GLOB_RECURSE HEADERS MatrixVector.h)

add_library(MatrixVector ${SOURCES})
target_include_directories(MatrixVector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(task_1 main.cpp)
target_link_libraries(task_1 MatrixVector)