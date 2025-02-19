cmake_minimum_required(VERSION 3.1) 
project(main LANGUAGES CXX CUDA)

# 设置编译器标准和模式
set(CMAKE_CXX_STANDARD 20) # 设置为 C++ 20 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制要求 C++ 20
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-g)

# 查找 CUDA 环境
find_package(CUDA 11.8 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 查找 OpenCV 环境
find_package(OpenCV REQUIRED)

# 配置 TensorRT
include_directories(/opt/TensorRT-8.5.2.2/include)
include_directories(/opt/TensorRT-8.5.2.2/samples/common)
set(TENSORRT_LIB_PATH /opt/TensorRT-8.5.2.2/lib)
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# 配置 ONNX Runtime
# 替换为你的 ONNX Runtime 的头文件和库文件路径
set(ONNXRUNTIME_INCLUDE_DIR "/opt/onnxruntime-linux-x64-gpu-1.15.1/include")  
set(ONNXRUNTIME_LIB_PATH "/opt/onnxruntime-linux-x64-gpu-1.15.1/lib")      
file(GLOB ONNXRUNTIME_LIBS "${ONNXRUNTIME_LIB_PATH}/*.so")  


# 将所有头文件路径添加到 INCLUDE_DIRS 列表
list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}  # 添加 ONNX Runtime 的头文件路径
        include
        )

# 将所有库路径添加到 ALL_LIBS 列表
list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${OpenCV_LIBRARIES}
        ${LIBS}  # TensorRT 的库
        ${ONNXRUNTIME_LIBS}  # ONNX Runtime 的库
        )

# 将包含目录添加到项目中
include_directories(${INCLUDE_DIRS})

# 定义调试宏
add_definitions(-DDEBUG)

# 添加目标文件
set(SRC_DIR /home/sam2_c++_muti/src)
add_executable(${PROJECT_NAME}
        ${SRC_DIR}/main.cpp
        ${SRC_DIR}/Sam2.cpp
        ${SRC_DIR}/utils.cpp
        ${SRC_DIR}/yolov8Predictor.cpp
        include
        )
target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS})

#target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS} /home/sam2_c++_muti/src)
# SET(SRC_FILES ${SRC_DIR}/Sam2.cpp)
# add_library(Sam2 SHARED ${SRC_FILES})
# 将库路径添加到目标链接


# 链接相关的库
target_link_libraries(${PROJECT_NAME}
        PRIVATE nvinfer 
        PRIVATE nvinfer_plugin 
        PRIVATE ${OpenCV_LIBS} 
        PRIVATE ${ONNXRUNTIME_LIBS}  # 添加 ONNX Runtime 的库
        PRIVATE ${CUDA_LIBRARIES}
        #Sam2
)