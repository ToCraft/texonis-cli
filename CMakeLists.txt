cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME texonis-cli)
project(${PROJECT_NAME})


file(GLOB_RECURSE sources      src/main/* src/main/*)
file(GLOB_RECURSE sources_test src/test/*)
file(GLOB_RECURSE data src/resources/*)


add_executable(${PROJECT_NAME} ${sources} ${data})

target_compile_options(${PROJECT_NAME} PUBLIC -std=c++1y -Wall -Wfloat-conversion)

target_include_directories(${PROJECT_NAME} PUBLIC src/main)

file(COPY ${data} DESTINATION src/resources)

add_subdirectory(llama.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC
  llama
)

install(TARGETS "${PROJECT_NAME}" DESTINATION bin)
install(DIRECTORY src/resources/ DESTINATION "${PROJECT_NAME}")

set(CPACK_PACKAGE_NAME "Texonis-CLI")
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)
