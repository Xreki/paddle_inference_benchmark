# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

set(PADDLE_FOUND OFF)
set(PADDLE_FLUID_FOUND OFF)

set(PADDLE_ROOT $ENV{PADDLE_ROOT} CACHE PATH "Paddle Path")
if(NOT PADDLE_ROOT)
  message(FATAL_ERROR "Set PADDLE_ROOT as your root directory installed PaddlePaddle")
endif()
set(THIRD_PARTY_ROOT ${PADDLE_ROOT}/third_party)

if(USE_GPU)
  set(CUDA_ROOT $ENV{CUDA_ROOT} CACHE PATH "CUDA root Path")
  set(CUDNN_ROOT $ENV{CUDNN_ROOT} CACHE PATH "CUDNN root Path")
  set(NCCL_ROOT $ENV{NCCL_ROOT} CACHE PATH "NCCL root Path")
endif()

find_path(PADDLE_INC_DIR NAMES paddle/fluid/inference/io.h PATHS ${PADDLE_ROOT})
find_library(PADDLE_FLUID_SHARED_LIB NAMES "libpaddle_fluid.so" PATHS
    ${PADDLE_ROOT}/paddle/fluid/inference)
find_library(PADDLE_FLUID_STATIC_LIB NAMES "libpaddle_fluid.a" PATHS
    ${PADDLE_ROOT}/paddle/fluid/inference)
if(USE_SHARED AND PADDLE_INC_DIR AND PADDLE_FLUID_SHARED_LIB)
  set(PADDLE_FOUND ON)
  set(PADDLE_FLUID_FOUND ON)
  add_definitions(-DUSE_PADDLE_FLUID)
  add_library(paddle_fluid_shared SHARED IMPORTED)
  set_target_properties(paddle_fluid_shared PROPERTIES IMPORTED_LOCATION
                        ${PADDLE_FLUID_SHARED_LIB})
  set(PADDLE_LIBRARIES paddle_fluid_shared)
  message(STATUS "Found PaddlePaddle Fluid (include: ${PADDLE_INC_DIR}; "
          "library: ${PADDLE_FLUID_SHARED_LIB}")
elseif(PADDLE_INC_DIR AND PADDLE_FLUID_STATIC_LIB)
  set(PADDLE_FOUND ON)
  set(PADDLE_FLUID_FOUND ON)
  add_definitions(-DUSE_PADDLE_FLUID)
  add_library(paddle_fluid_static STATIC IMPORTED)
  set_target_properties(paddle_fluid_static PROPERTIES IMPORTED_LOCATION
                        ${PADDLE_FLUID_STATIC_LIB})
  set(PADDLE_LIBRARIES paddle_fluid_static)
  message(STATUS "Found PaddlePaddle Fluid (include: ${PADDLE_INC_DIR}; "
          "library: ${PADDLE_FLUID_STATIC_LIB}")
else()
  set(PADDLE_FOUND OFF)
  set(PADDLE_FLUID_FOUND OFF)
  message(WARNING "Cannot find PaddlePaddle Fluid under ${PADDLE_ROOT}")
  return()
endif()

include_directories(${PADDLE_INC_DIR})

# including directory of third_party libraries
set(PADDLE_THIRD_PARTY_INC_DIRS)
function(third_party_include TARGET_NAME HEADER_NAME TARGET_DIRNAME)
  find_path(PADDLE_${TARGET_NAME}_INC_DIR NAMES ${HEADER_NAME} PATHS
            ${TARGET_DIRNAME}
            NO_DEFAULT_PATH)
  if(PADDLE_${TARGET_NAME}_INC_DIR)
    message(STATUS "Found PaddlePaddle third_party including directory: " ${PADDLE_${TARGET_NAME}_INC_DIR})
    set(PADDLE_THIRD_PARTY_INC_DIRS ${PADDLE_THIRD_PARTY_INC_DIRS} ${PADDLE_${TARGET_NAME}_INC_DIR} PARENT_SCOPE)
  endif()
endfunction()

third_party_include(glog glog/logging.h ${THIRD_PARTY_ROOT}/install/glog/include)
third_party_include(protobuf google/protobuf/message.h ${THIRD_PARTY_ROOT}/install/protobuf/include)
third_party_include(gflags gflags/gflags.h ${THIRD_PARTY_ROOT}/install/gflags/include)
third_party_include(eigen unsupported/Eigen/CXX11/Tensor ${THIRD_PARTY_ROOT}/eigen3)
third_party_include(boost boost/config.hpp ${THIRD_PARTY_ROOT}/boost)
if(USE_GPU)
  third_party_include(cuda cuda.h ${CUDA_ROOT}/include)
  third_party_include(cudnn cudnn.h ${CUDNN_ROOT}/include)
  third_party_include(nccl nccl.h ${NCCL_ROOT}/include)
  add_definitions(-DPADDLE_WITH_CUDA)
endif()

message(STATUS "PaddlePaddle need to include these third party directories: ${PADDLE_THIRD_PARTY_INC_DIRS}")
include_directories(${PADDLE_THIRD_PARTY_INC_DIRS})

set(PADDLE_THIRD_PARTY_LIBRARIES)
function(third_party_library TARGET_NAME TARGET_DIRNAME)
  set(library_names ${ARGN})
  foreach(lib ${library_names})
    string(REGEX REPLACE "^lib" "" lib_noprefix ${lib})
    if(${lib} MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}$")
      set(libtype STATIC)
      string(REGEX REPLACE "${CMAKE_STATIC_LIBRARY_SUFFIX}$" "" libname ${lib_noprefix})
    elseif(${lib} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
      set(libtype SHARED)
      string(REGEX REPLACE "${CMAKE_SHARED_LIBRARY_SUFFIX}$" "" libname ${lib_noprefix})
    else()
      message(FATAL_ERROR "Unknown library type: ${lib}")
    endif()
    find_library(${libname}_LIBRARY NAMES "${lib}" PATHS
        ${TARGET_DIRNAME}
        NO_DEFAULT_PATH)
    if(${libname}_LIBRARY)
      add_library(${libname} ${libtype} IMPORTED)
      set_target_properties(${libname} PROPERTIES IMPORTED_LOCATION ${${libname}_LIBRARY})
      set(PADDLE_THIRD_PARTY_LIBRARIES ${PADDLE_THIRD_PARTY_LIBRARIES} ${libname} PARENT_SCOPE)
      message(STATUS "Found PaddlePaddle third_party library: " ${${libname}_LIBRARY})
    else()
      message(WARNING "Cannot find ${lib} under ${THIRD_PARTY_ROOT}")
    endif()
  endforeach()
  set(${TARGET_NAME}_FOUND ON PARENT_SCOPE)
endfunction()

third_party_library(mklml ${THIRD_PARTY_ROOT}/install/mklml/lib libiomp5.so libmklml_intel.so)
if(NOT USE_SHARED)
  third_party_library(glog ${THIRD_PARTY_ROOT}/install/glog/lib libglog.a)
  third_party_library(protobuf ${THIRD_PARTY_ROOT}/install/protobuf/lib libprotobuf.a)
  third_party_library(gflags ${THIRD_PARTY_ROOT}/install/gflags/lib libgflags.a)
  if(NOT mklml_FOUND)
    third_party_library(openblas ${THIRD_PARTY_ROOT}/install/openblas/lib libopenblas.a)
  endif()
  third_party_library(zlib ${THIRD_PARTY_ROOT}/install/zlib/lib libz.a)
  third_party_library(snappystream ${THIRD_PARTY_ROOT}/install/snappystream/lib libsnappystream.a)
  third_party_library(snappy ${THIRD_PARTY_ROOT}/install/snappy/lib libsnappy.a)
  if(USE_GPU)
    third_party_library(cudart ${CUDA_ROOT}/lib64 libcudart.so)
  endif()
endif()
