#include "SAM2.h"
#include<string>
#include <filesystem>
#include"yolov8Predictor.h"
#include <onnxruntime_cxx_api.h>
namespace fs = std::filesystem;

void sam2() {
    std::vector<std::string> onnx_paths{
        "D:/sam2/segment-anything-2-main/tools/conver_model/conver_tiny_encoder.onnx",
        "D:/sam2/segment-anything-2-main/tools/conver_model/conver_tiny_decoder.onnx",
    };
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths, false);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cout << ("error：{}", error);
        return;
    }

    int type = 0;//0点作为提示 1框作为提示
    cv::Rect prompt_box = { 1087,1200,1000,1000 };//xywh
    cv::Point prompt_point = { 835, 352 };
    sam2->setparms(type, prompt_box, prompt_point); // 在原始图像上的box,point
    cv::Mat frame;
    size_t idx = 0;
    std::string window_name = "frame";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1980, 1080); // 例如将窗口大小设置为 640x480
    frame = cv::imread("E:/LYX_date/yanwo_cover/1_yanwo_cover_data/Image_20240925171424731.jpg");
    auto start = std::chrono::high_resolution_clock::now();
    auto result = sam2->inference(frame);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ("frame = {},duration = {}ms", idx++, duration) << std::endl;
    if (result.index() == 0) {
        std::string text = "frame = " + std::to_string(idx) + ", fps = " + std::to_string(1000.0f / duration);
        cv::putText(frame, text, cv::Point{ 30,40 }, 1, 2, cv::Scalar(0, 0, 255), 2);
        cv::imshow(window_name, frame);
        int key = cv::waitKey(0);
    }
    else {
        std::string error = std::get<std::string>(result);
        std::cout << ("error：{}", error);
    }
}


void yolo_sam2() {
    const std::vector<std::string> classNames = { "conver" };
    YOLOPredictor predictor{ nullptr };
    bool isGPU = false;
    std::string modelPath = "D:/sam2/ultralytics-main/ultralytics-main/runs/detect/train27/weights/best.onnx"; // YOLO模型路径
    std::vector<std::string> onnx_paths{
        "D:/sam2/segment-anything-2-main/tools/conver_tiny_encoder.onnx",
        "D:/sam2/segment-anything-2-main/tools/conver_tiny_decoder.onnx",
    }; // SAM2模型路径
    std::string inputFolder = "E:/LYX_date/yanwo_cover/1_yanwo_cover_data";  // 输入文件夹
    std::string outputFolder = "E:/LYX_date/yanwo_cover/c++_sam2/"; // 输出文件夹
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    try {
        predictor = YOLOPredictor(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
        std::cout << "YOLO initialization successful。" << std::endl;
        assert(classNames.size() == predictor.classNums);
    }
    catch (const std::exception& e) {
        std::cerr << "YOLO initialization failed: " << e.what() << std::endl;
        return;
    }
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths, false);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cerr << "SAM2 initialization failed: " << error << std::endl;
        return;
    }
    std::cout << "SAM2 initialization successful。" << std::endl;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (!entry.is_regular_file()) continue;
        std::string filePath = entry.path().string();
        if (filePath.find(".jpg") == std::string::npos &&
            filePath.find(".jpeg") == std::string::npos &&
            filePath.find(".png") == std::string::npos) {
            continue;
        }
        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            std::cerr << "Image Load Failure: " << filePath << std::endl;
            continue;
        }
        std::vector<Yolov8Result> results = predictor.predict(image);
        for (int idx = 0; idx < results.size(); ++idx) {
            cv::Point prompt_point = {
                results[idx].box.x + results[idx].box.width / 2,
                results[idx].box.y + results[idx].box.height / 2
            };
            sam2->setparms(0, results[idx].box, prompt_point); // 使用点提示,框为1
            cv::Mat sam2_frame = image.clone();
            auto inferenceResult = sam2->inference(sam2_frame);
            if (inferenceResult.index() == 0) {
                std::string outputFileName = outputFolder + entry.path().stem().string() + "_result.jpg";
                cv::imwrite(outputFileName, sam2_frame);
                std::cout << "results are saved to the: " << outputFileName << std::endl;
            }
            else {
                std::string error = std::get<std::string>(inferenceResult);
                std::cerr << "Segmentation Failure: " << error << std::endl;
            }
        }
    }
    std::cout << "ok" << std::endl;
}

bool isGpuAvailable() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(ORT_CUDA)
        session_options.AppendExecutionProvider_CUDA(0); // 使用第0号 GPU
#else
        std::cerr << "CUDA is not enabled in this build of ONNX Runtime." << std::endl;
        return false;
#endif
        const ORTCHAR_T* modelPath = L"D:/sam2/SAM2Export-main/image_encoder.onnx";
        Ort::Session test_session(env, modelPath, session_options);

        // 如果能够成功创建 session，则说明 GPU 可用
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "GPU is not available: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char const* argv[]) {
    sam2();
    return 0;
}






