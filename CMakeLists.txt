cmake_minimum_required(VERSION 3.0) # setting this is required
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


find_package(GTest)

if(GTEST_FOUND)
  add_executable(unit_tests ${sources_test} ${sources})

  # This define is added to prevent collision with the main.
  # It might be better solved by not adding the source with the main to the
  # testing target.
  target_compile_definitions(unit_tests PUBLIC UNIT_TESTS)

  # This allows us to use the executable as a link library, and inherit all 
  # linker options and library dependencies from it, by simply adding it as dependency.
  set_target_properties(example PROPERTIES ENABLE_EXPORTS on)

  target_link_libraries(unit_tests PUBLIC
    ${GTEST_BOTH_LIBRARIES}
    example
  )

  target_include_directories(unit_tests PUBLIC
    ${GTEST_INCLUDE_DIRS} # doesn't do anything on linux
  )
  
endif()


install(TARGETS "${PROJECT_NAME}" DESTINATION "${PROJECT_NAME}")

install(DIRECTORY resources DESTINATION "${PROJECT_NAME}")

set(CPACK_PACKAGE_NAME "Texonis-CLI")
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)
