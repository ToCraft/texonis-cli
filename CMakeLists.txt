cmake_minimum_required(VERSION 3.10)
set(LIBRARY_NAME texonis)
set(PROJECT_NAME ${LIBRARY_NAME}-cli)
project(${PROJECT_NAME})

file(GLOB_RECURSE sources      src/*)

add_library(${LIBRARY_NAME} SHARED ${sources})
add_executable(${PROJECT_NAME} src/main.cpp)

target_compile_options(${PROJECT_NAME} PUBLIC -std=c++17 -Wall -Wfloat-conversion)

target_include_directories(${PROJECT_NAME} PUBLIC src)

# Add llama.cpp
add_subdirectory(llama.cpp)
target_link_libraries(${LIBRARY_NAME} PUBLIC llama)

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARY_NAME})

# Install the executable and library
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# CPack
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "dev.tocraft")
set(CPACK_PACKAGE_NAME "Texonis-CLI")
set(CPACK_PACKAGE_VERSION "1.0.0")

include(CPack)
