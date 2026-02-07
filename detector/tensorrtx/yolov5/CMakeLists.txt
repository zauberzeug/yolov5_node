cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt - auto-detect or use custom path
if(DEFINED ENV{TENSORRT_DIR})
    include_directories($ENV{TENSORRT_DIR}/include/)
    link_directories($ENV{TENSORRT_DIR}/lib/)
elseif(EXISTS "/workspace/shared/TensorRT-10.2.0.19/include/")
    include_directories(/workspace/shared/TensorRT-10.2.0.19/include/)
    link_directories(/workspace/shared/TensorRT-10.2.0.19/lib/)
else()
    # Jetson JetPack: TensorRT is in system paths
    include_directories(/usr/include/aarch64-linux-gnu/)
    link_directories(/usr/lib/aarch64-linux-gnu/)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_definitions(-DAPI_EXPORTS)
add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_det yolov5_det.cpp ${SRCS})
target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
target_link_libraries(yolov5_det myplugins)
target_link_libraries(yolov5_det ${OpenCV_LIBS})
