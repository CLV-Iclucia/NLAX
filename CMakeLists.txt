cmake_minimum_required(VERSION 3.24)
project(NLAX)

set(CMAKE_CXX_STANDARD 17)

include_directories("./include")

add_executable(NLAX main.cpp Solver.cpp Solver.h include/nlax_types.h)
