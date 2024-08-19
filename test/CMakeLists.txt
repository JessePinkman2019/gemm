cmake_minimum_required(VERSION 3.10)
project(GEMM)

# 设置项目路径
set(SRC_DIR ${CMAKE_SOURCE_DIR}/../source/backend/arm/kernel)
set(UTIL_DIR ${CMAKE_SOURCE_DIR}/../source/utils)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/../output)

# 编译器设置
set(CMAKE_CXX_COMPILER "/home/lixiang/code/hexagon-sdk/4.3.0.0.htp/tools/android-ndk-r23b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android23-clang++")
set(CMAKE_CXX_FLAGS "-O2 -Wall")

# 包含头文件目录
include_directories(${SRC_DIR} ${UTIL_DIR})

add_definitions(-DGEMM_M=128)
add_definitions(-DGEMM_N=128)
add_definitions(-DGEMM_K=128)
add_definitions(-DGEMM_UNROLL_M=4)
add_definitions(-DGEMM_UNROLL_N=4)
add_definitions(-DGEMM_UNROLL_K=16)
add_definitions(-DMIN_MATRIX_SIZE=128)
add_definitions(-DMAX_MATRIX_SIZE=1024)
add_definitions(-DMATRIX_SIZE_STEP=128)
add_definitions(-DNREPEATS=1)


# 搜索所有cpp文件
file(GLOB KERNEL_SOURCES "${SRC_DIR}/*.cpp")
file(GLOB UTIL_SOURCES "${UTIL_DIR}/*.cpp")
file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/main.cpp")

# 创建一个目标文件
add_executable(main.x ${KERNEL_SOURCES} ${UTIL_SOURCES} ${TEST_SOURCES})

# 链接数学库
target_link_libraries(main.x m)

# 设置目标创建位置
set_target_properties(main.x PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR}
)

# 添加清理任务
add_custom_target(clean_up
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
)

add_custom_target(clean_all
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/../plot/*.png
)
