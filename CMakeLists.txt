cmake_minimum_required(VERSION 2.8)

project(Test)
# TARGET_ARCH values: soc, x86
# soc means soc mode
# x86 means pcie mode

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH soc)
endif ()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall")
set(CMAKE_CXX_STANDARD 14)

if (${TARGET_ARCH} STREQUAL "x86")
    message("${TARGET_ARCH} mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "soc")
    message("${TARGET_ARCH} mode, starting......")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(SDK_ROOT /media/psf/Home/work/sgnnsdk/soc-sdk)
endif ()

# use ffmpeg
set(FFMPEG_DIR ${SDK_ROOT}/opt/sophon/sophon-ffmpeg-latest/lib/cmake)
MESSAGE(STATUS "ffmpeg_dir=${FFMPEG_DIR}")
set(FFMPEG_INCLUDE_DIRS ${SDK_ROOT}/opt/sophon/sophon-ffmpeg-latest/include)
set(FFMPEG_LIB_DIRS ${SDK_ROOT}/opt/sophon/sophon-ffmpeg-latest/lib)
set(FFMPEG_LIBS avcodec avformat avutil avfilter avdevice swscale swresample)

# use opencv
set(OpenCV_DIR ${SDK_ROOT}/opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS ${SDK_ROOT}/opt/sophon/sophon-opencv-latest/include/opencv4)
set(OpenCV_LIB_DIRS ${SDK_ROOT}/opt/sophon/sophon-opencv-latest/lib)
set(OpenCV_LIBS opencv_calib3d opencv_core opencv_dnn opencv_features2d opencv_flann opencv_freetype
        opencv_gapi opencv_highgui opencv_imgcodecs opencv_imgproc opencv_ml opencv_objdetect opencv_photo
        opencv_stitching opencv_video opencv_videoio)

# libsophon
set(libsophon_INCLUDE_DIRS ${SDK_ROOT}/opt/sophon/libsophon-current/include)
set(libsophon_LIB_DIRS ${SDK_ROOT}/opt/sophon/libsophon-current/lib)
set(libsophon_LIBS bmrt bmcv bmlib)

set(include_dirs ${libsophon_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(link_dirs ${FFMPEG_LIB_DIRS} ${OpenCV_LIB_DIRS} ${libsophon_LIB_DIRS})
set(link_libraries ${OpenCV_LIBS} ${FFMPEG_LIBS} ${libsophon_LIBS})

# debug information
message(STATUS "include_dirs = ${include_dirs}")
message(STATUS "link_dirs = ${link_dirs}")
message(STATUS "link_libraries = ${link_libraries}")

# source files
set(SRC_FILES test_main.cpp base_model.cpp yolov5.cpp)

# exe
INCLUDE_DIRECTORIES(${include_dirs})
LINK_DIRECTORIES(${link_dirs})

INCLUDE_DIRECTORIES(bmutility)
ADD_SUBDIRECTORY(bmutility)

add_executable(test_yolo.${TARGET_ARCH} ${SRC_FILES})
target_link_libraries(test_yolo.${TARGET_ARCH} bmutility ${link_libraries} pthread)
