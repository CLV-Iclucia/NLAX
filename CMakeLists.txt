cmake_minimum_required(VERSION 3.24)
project(NLAX)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found!")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
    add_definitions(-DOPENMP_ENABLED)
    cmake_host_system_information(RESULT CPU_CORE_NUM QUERY NUMBER_OF_LOGICAL_CORES)
    message(STATUS "${CPU_CORE_NUM} CPU logical cores detected.")
    add_definitions(-DNCPU_CORES=${CPU_CORE_NUM})
else()
    message(FATAL_ERROR "OpenMP not found.")
endif ()

#find_package(benchmark REQUIRED)
#if(benchmark_FOUND)
#    message(STATUS "Google Benchmark found at ${benchmark_DIR}!")
#    add_subdirectory(benchmark)
#else()
#    message(FATAL_ERROR "Google Benchmark not found.")
#endif()

set(NLAX_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/include")
set(NLAX_SOURCE_PATH "${PROJECT_SOURCE_DIR}/src")
set(NLAX_EXAMPLE_PATH "${PROJECT_SOURCE_DIR}/examples")

include_directories(${NLAX_INCLUDE_PATH})

file(GLOB NLAX_HEADERS ${NLAX_INCLUDE_PATH})
file(GLOB NLAX_SOURCES ${NLAX_SOURCE_PATH})

add_executable(VecOp ${NLAX_EXAMPLE_PATH}/VecOp.cpp ${NLAX_SOURCES} ${NLAX_HEADERS})
add_executable(classical_iterative_methods ${NLAX_EXAMPLE_PATH}/classical_iterative_methods.cpp
                ${NLAX_SOURCES} ${NLAX_HEADERS})
add_executable(CG_Hilbert ${NLAX_EXAMPLE_PATH}/CG_Hilbert.cpp
        ${NLAX_SOURCES} ${NLAX_HEADERS})