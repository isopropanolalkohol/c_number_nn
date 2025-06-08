cmake_minimum_required(VERSION 3.31)
project(c_number_nn C)

set(CMAKE_C_STANDARD 11)

file(GLOB SOURCES "src/*.c")

add_executable(c_number_nn ${SOURCES})

target_include_directories(c_number_nn PRIVATE include)
target_link_libraries(c_number_nn m)
