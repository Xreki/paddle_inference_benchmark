cmake_minimum_required(VERSION 3.0)
project(paddle_inference_benchmark CXX C)

option(USE_SHARED "Linking shared library priority" ON)
option(USE_GPU    "Linking library of GPU version"  OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Fluid)
if(NOT PADDLE_FOUND)
  find_package(Paddle)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(PADDLE_FLUID_FOUND)
  if(USE_GPU)
    add_definitions(-DPADDLE_WITH_CUDA)
  endif()
  add_executable(bench_fluid fluid/inference_helper.cc fluid/main.cc)
  target_link_libraries(bench_fluid
      ${PADDLE_LIBRARIES}
      ${PADDLE_THIRD_PARTY_LIBRARIES}
      -lrt -ldl -lpthread)
elseif(PADDLE_FOUND)
  add_executable(bench_v2 v2/inference_helper.cc v2/main.cc)
  target_link_libraries(bench_v2
      ${PADDLE_LIBRARIES}
      ${PADDLE_THIRD_PARTY_LIBRARIES}
      -lrt -ldl -lpthread)
endif()
