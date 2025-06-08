cmake_minimum_required(VERSION 3.31)
project(c_number_nn C)

set(CMAKE_C_STANDARD 11)

file(GLOB SOURCES "src/*.c")
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)

include_directories(${SDL2_INCLUDE_DIRS})
link_directories(${SDL2_LIBRARY_DIRS})
add_executable(c_number_nn ${SOURCES})

target_include_directories(c_number_nn PRIVATE include)
target_link_libraries(c_number_nn m ${SDL2_LIBRARIES})
