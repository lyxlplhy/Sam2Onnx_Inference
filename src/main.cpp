#include "SAM2.h"
#include<string>
#include <filesystem>
#include"yolov8Predictor.h"
#include <onnxruntime_cxx_api.h>
namespace fs = std::filesystem;

void sam2() {
    std::vector<std::string> onnx_paths{
       "D:/sam2/segment-anything-2-main/tools/onnx/conver_tiny_encoder.onnx",
        "D:/sam2/segment-anything-2-main/tools/onnx/conver_tiny_decoder.onnx",
    }; // SAM2模型路径
    std::string frame_path = "E:/LYX_date/yanwo_cover/1_yanwo_cover_data/Image_20240925140308619.jpg";
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths,false);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cout << ("eror：{}", error);
        return;
    }
    int type = 0;//1点作为提示 0框作为提示
    std::vector<cv::Rect> prompt_box = { { 560,455,500,500 },{1407,1076,500,500} };//xywh
    std::vector<cv::Point> prompt_point = { { 794, 686 } ,{1617,1298} };
    sam2->setparms(type, prompt_box, prompt_point); // 在原始图像上的box,point
    cv::Mat frame;
    size_t idx = 1;
    std::string window_name = "frame";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1980,1080 ); // 例如将窗口大小设置为 640x480
    frame = cv::imread(frame_path);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = sam2->inference(frame);
    result = sam2->inference(frame);
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
            std::cout << ("错误：{}", error);
        }
 }


void yolo_sam2() {
    const std::vector<std::string> classNames = { "conver" };
    YOLOPredictor predictor{ nullptr };
    bool isGPU = false;
    std::string modelPath = "D:/sam2/ultralytics-main/ultralytics-main/runs/detect/train8/weights/best.onnx"; // YOLO模型路径
    std::vector<std::string> onnx_paths{
        "D:/sam2/segment-anything-2-main/tools/onnx/conver_tiny_encoder.onnx",
         "D:/sam2/segment-anything-2-main/tools/onnx/conver_tiny_decoder.onnx",
    }; // SAM2模型路径
    std::string inputFolder = "E:/LYX_date/hegai1/1";  // 输入文件夹
    std::string outputFolder = "E:/LYX_date/hegai1/1_out/"; // 输出文件夹
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    try {
        predictor = YOLOPredictor(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
        std::cout << "YOLO initialize ok!!。" << std::endl;
        assert(classNames.size() == predictor.classNums);
    }
    catch (const std::exception& e) {
        std::cerr << "YOLO initialize failed!!: " << e.what() << std::endl;
        return;
    }
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths, isGPU);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cerr << "SAM2 initialize failed!!: " << error << std::endl;
        return;
    }
    std::cout << "SAM2 initialize ok!!。" << std::endl;
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
            std::cerr << "can not imread mat: " << filePath << std::endl;
            continue;
        }
        std::cout << "precess mat: " << filePath << std::endl;
        std::vector<Yolov8Result> results = predictor.predict(image);
        std::vector<cv::Point> prompt_points = {};
        std::vector<cv::Rect> prompt_boxes = {};
        for (int idx = 0; idx < results.size(); ++idx) {
            cv::Point prompt_point = {
                results[idx].box.x + results[idx].box.width / 2,
                results[idx].box.y + results[idx].box.height / 2
            };
            prompt_points.push_back(prompt_point);
            prompt_boxes.push_back(results[idx].box);
        }
        sam2->setparms(0, prompt_boxes, prompt_points); // 使用点提示
        cv::Mat sam2_frame = image.clone();
        auto inferenceResult = sam2->inference(sam2_frame);
        if (inferenceResult.index() == 0) {
            std::string outputFileName = outputFolder + entry.path().stem().string() + "_result.jpg";
            cv::imwrite(outputFileName, sam2_frame);
            std::cout << "分割结果保存到: " << outputFileName << std::endl;
        }
        else {
            std::string error = std::get<std::string>(inferenceResult);
            std::cerr << "分割失败: " << error << std::endl;
        }
    }
    std::cout << "seg ok" << std::endl;
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

    yolo_sam2();
    return 0;
}




