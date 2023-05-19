cmake_minimum_required(VERSION 3.6)

project(exercise_2)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/cmake CACHE PATH "Path to installed Eigen")
# #cmake .. -DDEBUG=ON  
option(DEBUG OFF)
# option(EX1 "exercise_1-1" OFF)
# option(EX2 "exercise_1-1" OFF)

configure_file(Config.h.in ${PROJECT_SOURCE_DIR}/Config.h)

set(CMAKE_CXX_STANDARD 14)


# set(SOURCES ex1.cpp)
set(SOURCES ex2.cpp)

find_package(Eigen3 REQUIRED)

add_executable(exercise_2 ${SOURCES})
target_include_directories(exercise_2 PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(exercise_2 general Eigen3::Eigen)

