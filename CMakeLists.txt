cmake_minimum_required(VERSION 3.15) 
project(
    main     
    LANGUAGES CXX C    
    DESCRIPTION "sam2_image"
)

# 设置编译器标准和模式
set(CMAKE_CXX_STANDARD 20) # 设置为 C++ 20 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求 C++ 20
set(CMAKE_BUILD_TYPE "Debug") # 调试模式
# set(CMAKE_BUILD_TYPE "Release") # 如果需要释放模式可切换

# 输出路径配置
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin) # 可执行文件输出到 bin 目录
set(CMAKE_BUILD_PARALLEL_LEVEL 8) # 并行构建

# 包含目录
include_directories(${PROJECT_DIR}/include/)

# 自动检索代码文件
file(GLOB SRC_FILES 
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# 查找库依赖
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

# 添加可执行文件目标
add_executable(${CMAKE_PROJECT_NAME} ${SRC_FILES})

# 链接库
target_link_libraries(${CMAKE_PROJECT_NAME} onnxruntime)
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBRARIES})
