cmake_minimum_required(VERSION 3.6)

project(exercise_3)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/cmake CACHE PATH "Path to installed Eigen")
# #cmake .. -DDEBUG=ON  
option(DEBUG "Use" OFF)
# option(EX1 "exercise_3-1" OFF)
# option(EX2 "exercise_3-1" OFF)

set(CMAKE_CXX_STANDARD 17)


# set(SOURCES ex1.cpp)
set(SOURCES ex1.cpp)

find_package(Eigen3 REQUIRED)

add_executable(exercise_3 ${SOURCES})
target_include_directories(exercise_3 PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(exercise_3 general Eigen3::Eigen)

