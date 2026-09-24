cmake_minimum_required(VERSION 3.20)
project(vro_indexes_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CONTRACT 7: -O3 -march=native. Fall back to -mcpu=native if the compiler
# rejects -march=native (older clang on Apple Silicon).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" VRO_HAS_MARCH_NATIVE)
if(VRO_HAS_MARCH_NATIVE)
  set(VRO_ARCH_FLAG "-march=native")
else()
  set(VRO_ARCH_FLAG "-mcpu=native")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${VRO_ARCH_FLAG} -DNDEBUG")

find_package(Threads REQUIRED)

add_library(vro_core STATIC
  src/common.cpp src/npy.cpp src/distance.cpp src/kmeans.cpp src/splitmix.cpp
  src/flat.cpp src/ivf.cpp src/pq.cpp src/ivf_pq.cpp src/hnsw.cpp src/diskann.cpp)
target_include_directories(vro_core PUBLIC src)
target_include_directories(vro_core SYSTEM PUBLIC third_party)
target_compile_options(vro_core PUBLIC -Wall -Wextra)
target_link_libraries(vro_core PUBLIC Threads::Threads)

add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE vro_core)

add_executable(tests tests/test_main.cpp)
target_link_libraries(tests PRIVATE vro_core)

# Tests read data/processed/dev from the repository root.
enable_testing()
get_filename_component(VRO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
foreach(t npy splitmix flat kmeans json)
  add_test(NAME ${t} COMMAND tests ${t} ${VRO_REPO_ROOT} $<TARGET_FILE:bench>)
endforeach()

# One test executable per index, so Wave 2 agents edit only their own file.
# A placeholder returns 77, which ctest reports as "skipped".
foreach(t ivf pq hnsw ivf_pq diskann)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE vro_core)
  add_test(NAME ${t} COMMAND test_${t} ${VRO_REPO_ROOT} $<TARGET_FILE:bench>)
  set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
