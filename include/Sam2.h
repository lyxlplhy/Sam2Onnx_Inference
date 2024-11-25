#pragma once
#pragma once
#include "Model.h"
#include <fstream>
#include <variant>

class SAM2 :public yo::Model {

    static const size_t BUFFER_SIZE = 15;

    struct SubStatus {
        std::vector<Ort::Value> maskmem_features;
        std::vector<Ort::Value> maskmem_pos_enc;
        std::vector<Ort::Value> temporal_code;
    };
    struct InferenceStatus {
        int32_t current_frame = 0;
        std::vector<Ort::Value> obj_ptr_first;
        yo::FixedSizeQueue<SubStatus, 7> status_recent;
        yo::FixedSizeQueue<Ort::Value, BUFFER_SIZE> obj_ptr_recent;
    };
    struct ParamsSam2 {
        uint type = 0; // 0使用box，1使用point
        cv::Rect prompt_box;
        cv::Point prompt_point;
    };
private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;
    std::vector<cv::Mat> input_images;
    ParamsSam2 parms;
    InferenceStatus infer_status;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    void PrintTensorShape(const Ort::Value& tensor, const std::string& tensor_name);
    //Env
    Ort::Env img_encoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "img_encoder");
    Ort::Env img_decoder_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "img_decoder");

    //onnx会话配置相关
    Ort::Session* img_encoder_session = nullptr;
    Ort::Session* img_decoder_session = nullptr;

    //options
    Ort::SessionOptions img_encoder_options = Ort::SessionOptions();
    Ort::SessionOptions img_decoder_options = Ort::SessionOptions();

    //输入相关
    std::vector<yo::Node> img_encoder_input_nodes;
    std::vector<yo::Node> img_decoder_input_nodes;

    //输出相关
    std::vector<yo::Node> img_encoder_output_nodes;
    std::vector<yo::Node> img_decoder_output_nodes;
protected:
    void preprocess(cv::Mat& image) override;
    void postprocess(std::vector<Ort::Value>& output_tensors) override;

    std::variant<std::vector<Ort::Value>, std::string> img_encoder_infer(std::vector<Ort::Value>&);
    std::variant<std::vector<Ort::Value>, std::string> img_decoder_infer(std::vector<Ort::Value>&);
    float extract_score_from_tensor(const Ort::Value& output_tensor);
public:
    SAM2() {};
    SAM2(const SAM2&) = delete;// 删除拷贝构造函数
    SAM2& operator=(const SAM2&) = delete;// 删除赋值运算符
    ~SAM2() {
        if (img_encoder_session != nullptr) delete img_encoder_session;
        if (img_decoder_session != nullptr) delete img_decoder_session;
    };
    int setparms(ParamsSam2 parms);
    int setparms(int type, cv::Rect prompt_box, cv::Point prompt_point);
    std::variant<bool, std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda) override;
    std::variant<bool, std::string> inference(cv::Mat& image) override;
};