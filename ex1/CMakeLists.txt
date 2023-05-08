cmake_minimum_required(VERSION 3.6)

project(exercise_1)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/cmake CACHE PATH "Path to installed Eigen")

# #cmake .. -DDEBUG=ON  
option(DEBUG "Use" OFF)

configure_file(Config.h.in ${PROJECT_SOURCE_DIR}/Config.h)

set(CMAKE_CXX_STANDARD 14)

set(SOURCES ex1.cpp)

find_package(Eigen3 REQUIRED)

add_executable(exercise_1 ${SOURCES})
target_include_directories(exercise_1 PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(exercise_1 general Eigen3::Eigen)

