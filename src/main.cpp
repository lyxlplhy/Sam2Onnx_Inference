#include <iostream>
#include "SAM2.h"
#include<string>
#include <filesystem>
#include<regex>
#include"yolov8Predictor.h"
#include <onnxruntime_cxx_api.h>

void sam2() {
    std::vector<std::string> onnx_paths{
        "D:/sam2/segment-anything-2-main/tools/conver_tiny_encoder.onnx",
        "D:/sam2/segment-anything-2-main/tools/conver_tiny_encoder.onnx",
    };
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths,false);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cout << ("错误：{}", error);
        return;
    }

    int type = 1;
    cv::Rect prompt_box = { 1087,1200,1000,1000 };//xywh
    cv::Point prompt_point = { 1373, 1682 };
    sam2->setparms(type, prompt_box, prompt_point); // 在原始图像上的box,point
    std::string video_path = "E:/LYX_date/SAM_data/SAM_data/sam2_lianxu/2.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    //************************************************************
    cv::Mat frame;
    size_t idx = 0;
    std::string window_name = "frame";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1980,1080 ); // 例如将窗口大小设置为 640x480
    while (true) {
        frame = cv::imread("E:/LYX_date/mapImage/mapImage/00003595_999613_1107_1206_970_968_0.980981.jpg");
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sam2->inference(frame);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << ("frame = {},duration = {}ms", idx++, duration) << std::endl;
        if (result.index() == 0) {
            //std::string text = std::format("frame = {},fps={:.1f}", idx, 1000.0f / duration);
            std::string text = "frame = " + std::to_string(idx) + ", fps = " + std::to_string(1000.0f / duration);
            cv::putText(frame, text, cv::Point{ 30,40 }, 1, 2, cv::Scalar(0, 0, 255), 2);
            cv::imshow(window_name, frame);
            int key = cv::waitKey(1000);
            if (key == 'q' || key == 27) break;
        }
        else {
            std::string error = std::get<std::string>(result);
            std::cout << ("错误：{}", error);
            break;
        }
    }
    capture.release();
}

void yolo()
{
    const std::vector<std::string> classNames = {"conver"};
    YOLOPredictor predictor{ nullptr };
    bool isGPU = false;
    std::string modelPath = "D:/sam2/ultralytics-main/ultralytics-main/runs/detect/train4/weights/best.onnx";//yolo
    std::vector<std::string> onnx_paths{
        "D:/sam2/segment-anything-2-main/tools/conver_model/conver_tiny_encoder.onnx",
        "D:/sam2/segment-anything-2-main/tools/conver_model/conver_tiny_decoder.onnx",
    };//sam2
    std::string Filename = "E:/LYX_date/yanwo_cover/1_yanwo_cover_data/Image_20240925140308619.jpg";
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold=0.5f;
    try
    {
        predictor = YOLOPredictor(modelPath, isGPU,
            confThreshold,
            iouThreshold,
            maskThreshold);
        std::cout << "Model was initialized." << std::endl;
        assert(classNames.size() == predictor.classNums);
        std::regex pattern(".+\\.(jpg|jpeg|png|gif)$");
        std::cout << "Start predicting..." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    auto sam2 = std::make_unique<SAM2>();
    auto r = sam2->initialize(onnx_paths, false);
    if (r.index() != 0) {
        std::string error = std::get<std::string>(r);
        std::cout << ("错误：{}", error);
        return;
    }
    int type = 1;//0用框，1用点
    cv::Mat image = cv::imread(Filename);
    std::vector<Yolov8Result> result = predictor.predict(image);
    //utils::visualizeDetection(image, result, classNames);
    for (int idx = 0;idx < result.size();++idx)
    {
        cv::Point prompt_point = { result[idx].box.x+result[idx].box.width/2,result[idx].box.y + result[idx].box.height / 2 };
        sam2->setparms(type, result[idx].box, prompt_point); // 在原始图像上的box,point
        cv::Mat sam2_frame = cv::imread(Filename);
        auto result = sam2->inference(sam2_frame);
        if (result.index() == 0) {
            std::string text = "frame = " + std::to_string(idx) ;
            cv::putText(sam2_frame, text, cv::Point{ 30,40 }, 1, 2, cv::Scalar(0, 0, 255), 2);
            cv::imshow("window_name", sam2_frame);
            int key = cv::waitKey(0);
            if (key == 'q' || key == 27) break;
        }
        else {
            std::string error = std::get<std::string>(result);
            std::cout << ("错误：{}", error);
            break;
        }

    }

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

    yolo();
    return 0;
}




