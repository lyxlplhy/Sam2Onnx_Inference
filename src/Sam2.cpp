#include "Sam2.h"
#include <variant>
#include <cstdlib> // 包含 mbstowcs_s

std::variant<bool, std::string> SAM2::initialize(std::vector<std::string>& onnx_paths, bool is_cuda) {
    // image_encoder,image_decoder,
    assert(onnx_paths.size() == 2);
    auto is_file = [](const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
        };
    for (const auto& path : onnx_paths) {
        if (!is_file(path))
            //return std::format("Model file dose not exist.file:{}", path);
            return "Model file dose not exist.file:\n " + path;
    }
    this->img_encoder_options.SetIntraOpNumThreads(2); //设置线程数量
    this->img_decoder_options.SetIntraOpNumThreads(2); //设置线程数量
    //***********************************************************
    if (is_cuda) {
        try {
            OrtCUDAProviderOptions options;
            options.device_id = 0;
            options.arena_extend_strategy = 0;
            options.gpu_mem_limit = SIZE_MAX;
            options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            options.do_copy_in_default_stream = 1;
            this->img_encoder_options.AppendExecutionProvider_CUDA(options);
            this->img_decoder_options.AppendExecutionProvider_CUDA(options);
            std::cout << ("Using CUDA...");
        }
        catch (const std::exception& e) {
            std::string error(e.what());
            return error;
        }
    }
    else {
        std::cout << ("Using CPU...");
    }
    try {
#ifdef _WIN32
        unsigned len_img_encoder = onnx_paths[0].size() * 2; // 预留字节数
        unsigned len_img_decoder = onnx_paths[1].size() * 2; // 预留字节数
        setlocale(LC_CTYPE, ""); //必须调用此函数,本地化
        wchar_t* p_img_encoder = new wchar_t[len_img_encoder]; // 申请一段内存存放转换后的字符串
        wchar_t* p_img_decoder = new wchar_t[len_img_decoder]; // 申请一段内存存放转换后的字符串

        size_t convertedChars = 0; // 用于存储转换的字符数
        errno_t err; // 用于存储错误代码
        err = mbstowcs_s(&convertedChars, p_img_encoder, len_img_encoder, onnx_paths[0].c_str(), _TRUNCATE);
        if (err != 0) {
            std::cerr << "转换 p_img_encoder 失败，错误代码: " << err << std::endl;
        }

        err = mbstowcs_s(&convertedChars, p_img_decoder, len_img_decoder, onnx_paths[1].c_str(), _TRUNCATE);
        if (err != 0) {
            std::cerr << "转换 p_img_decoder 失败，错误代码: " << err << std::endl;
        }
        std::wstring wstr_img_encoder(p_img_encoder);
        std::wstring wstr_img_decoder(p_img_decoder);

        delete[] p_img_encoder,p_img_decoder; // 释放申请的内存

        img_encoder_session = new Ort::Session(img_encoder_env, wstr_img_encoder.c_str(), this->img_encoder_options);
        img_decoder_session = new Ort::Session(img_decoder_env, wstr_img_decoder.c_str(), this->img_decoder_options);

#else
        img_encoder_session = new Ort::Session(img_encoder_env, (const char*)onnx_paths[0].c_str(), this->img_encoder_options);
        img_decoder_session = new Ort::Session(img_decoder_env, (const char*)onnx_paths[1].c_str(), this->img_decoder_options);
#endif  
    }
    catch (const std::exception& e) {
        //return std::format("Failed to load model. Please check your onnx file!");
        return "Failed to load model. Please check your onnx file!\n";

    }
    Ort::AllocatorWithDefaultOptions allocator;
    size_t const img_encoder_input_num = this->img_encoder_session->GetInputCount();
    size_t const img_decoder_input_num = this->img_decoder_session->GetInputCount();
    std::cout << ("[{},{}]", img_encoder_input_num, img_decoder_input_num);

    //记录网络输出输出的名字
    for (size_t index = 0; index < img_encoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_encoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_encoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for (size_t j = 0;j < input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        strncpy_s(node.name, name_length, name, name_length);
        this->img_encoder_input_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_input_num; index++) {
        Ort::AllocatedStringPtr input_name_Ptr = this->img_decoder_session->GetInputNameAllocated(index, allocator);
        Ort::TypeInfo input_type_info = this->img_decoder_session->GetInputTypeInfo(index);
        auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for (size_t j = 0;j < input_dims.size();j++) node.dim.push_back(input_dims.at(j));
        char* name = input_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        //strncpy(node.name, name, name_length);
        strncpy_s(node.name, name_length, name, name_length);

        this->img_decoder_input_nodes.push_back(node);
    }

    size_t const img_encoder_output_num = this->img_encoder_session->GetOutputCount();
    size_t const img_decoder_output_num = this->img_decoder_session->GetOutputCount();

    //记录网络输入输出的张量形状
    for (size_t index = 0; index < img_encoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_encoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_encoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for (size_t j = 0;j < output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        //strncpy(node.name, name, name_length);
        strncpy_s(node.name, name_length, name, name_length);
        this->img_encoder_output_nodes.push_back(node);
    }
    for (size_t index = 0; index < img_decoder_output_num; index++) {
        Ort::AllocatedStringPtr output_name_Ptr = this->img_decoder_session->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo output_type_info = this->img_decoder_session->GetOutputTypeInfo(index);
        auto output_dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
        yo::Node node;
        for (size_t j = 0;j < output_dims.size();j++) node.dim.push_back(output_dims.at(j));
        char* name = output_name_Ptr.get();
        size_t name_length = strlen(name) + 1;
        node.name = new char[name_length];
        //strncpy(node.name, name, name_length);
        strncpy_s(node.name, name_length, name, name_length);

        this->img_decoder_output_nodes.push_back(node);
    }
    std::cout << ("----------------img_encoder------------------");
    for (const auto& outputs : img_encoder_input_nodes) {
        std::cout << ("{}=[", outputs.name);
        for (size_t i = 0;i < outputs.dim.size() - 1;i++)  std::cout << ("{},", outputs.dim[i]);
        std::cout << ("{}]", outputs.dim[outputs.dim.size() - 1]);
    }
    std::cout << ("-----------------img_decoder-----------------");
    for (const auto& outputs : img_decoder_input_nodes) {
        std::cout << ("{}=[", outputs.name);
        for (size_t i = 0;i < outputs.dim.size() - 1;i++)  std::cout << ("{},", outputs.dim[i]);
        std::cout << ("{}]", outputs.dim[outputs.dim.size() - 1]);
    }
    std::cout << ("----------------------------------");
    this->is_inited = true;
    std::cout << ("initialize ok!!");
    return true;
}


std::variant<bool, std::string> SAM2::inference(cv::Mat& image) {
    if (image.empty() || !is_inited) return "image can not empyt!";
    this->ori_img = &image;
    // 图片预处理
    try {
        this->preprocess(image); // 
    }
    catch (const std::exception& e) {
        return "Image preprocess failed!";
    }
    // 图片编码器，输入图片
    std::vector<Ort::Value> img_encoder_tensor;
    img_encoder_tensor.push_back(std::move(Ort::Value::CreateTensor<float>(
        memory_info,
        this->input_images[0].ptr<float>(),
        this->input_images[0].total(),
        this->img_encoder_input_nodes[0].dim.data(),
        this->img_encoder_input_nodes[0].dim.size()))
    );
    //*****************************img_encoder**********************************

    auto start1 = std::chrono::high_resolution_clock::now();
    auto result_0 = this->img_encoder_infer(img_encoder_tensor);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "img_encoder_infer" << duration1 << std::endl;
    if (result_0.index() != 0) return std::get<std::string>(result_0);
    auto& img_encoder_out = std::get<0>(result_0); // 'high_res_feats_0', 'high_res_feats_1', 'image_embed'
    auto start = std::chrono::high_resolution_clock::now();
    auto result_2 = this->img_decoder_infer(img_encoder_out);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "img_decoder_infer" << duration << std::endl;
    if (result_2.index() != 0) return std::get<std::string>(result_2);
    auto& img_decoder_out = std::get<0>(result_2); // obj_ptr,mask_for_mem,pred_mask
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(std::move(img_decoder_out[0])); //pred_mask
    try {
        this->postprocess(output_tensors);
        float score = extract_score_from_tensor(img_decoder_out[1]);
        std::cout << score;
    }
    catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    this->infer_status.current_frame++;
    return true;
}


// input [1,3,1024,1024]
// output: 
//      pix_feat        [1,256,64,64]
//      high_res_feat0  [1,32,256,256]
//      high_res_feat1  [1,64,128,128]
//      vision_feats    [1,256,64,64]
//      vision_pos_embed [4096,1,256]
std::variant<std::vector<Ort::Value>, std::string> SAM2::img_encoder_infer(std::vector<Ort::Value>& input_tensor) {
    std::vector<const char*> input_names, output_names;
    for (auto& node : this->img_encoder_input_nodes)  input_names.push_back(node.name);
    for (auto& node : this->img_encoder_output_nodes) output_names.push_back(node.name);

    std::vector<Ort::Value> img_encoder_out;
    try {
        img_encoder_out = std::move(this->img_encoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()));
    }
    catch (const std::exception& e) {
        std::string error(e.what());
        //return std::format("ERROR: img_encoder_infer failed!!\n {}", error);
        return "ERROR: img_encoder_infer failed!!\n" + error;

    }
    return img_encoder_out;
}

std::variant<std::vector<Ort::Value>, std::string> SAM2::img_decoder_infer(std::vector<Ort::Value>& mem_attention_out) {

    std::vector<const char*> input_names, output_names;
    for (auto& node : this->img_decoder_input_nodes)  input_names.push_back(node.name);
    for (auto& node : this->img_decoder_output_nodes) output_names.push_back(node.name);
    // point_coords,point_labels,frame_size,image_embed,high_res_feats_0,high_res_feats_1
    std::vector<Ort::Value> input_tensor; // 8
    auto box = parms.prompt_box;
    auto point = parms.prompt_point;
    // 变化bbox比例
    box.x = 1024 * ((float)box.x / ori_img->cols);
    box.y = 1024 * ((float)box.y / ori_img->rows);
    box.width = 1024 * ((float)box.width / ori_img->cols);
    box.height = 1024 * ((float)box.height / ori_img->rows);
    point.x = 1024 * ((float)point.x / ori_img->cols);
    point.y = 1024 * ((float)point.y / ori_img->rows);
    std::vector<float>point_val, point_labels;
    if (parms.type == 0) {
        point_val = { (float)box.x,(float)box.y,(float)box.x + box.width,(float)box.y + box.height };//xyxy
        point_labels = { 2,3 };
        this->img_decoder_input_nodes[3].dim = { 1,2,2 };
        this->img_decoder_input_nodes[4].dim = { 1,2 };
    }
    else if (parms.type == 1) {
        point_val = { (float)point.x,(float)point.y };//xy
        point_labels = { 1 };
        this->img_decoder_input_nodes[3].dim = { 1,1,2 };
        this->img_decoder_input_nodes[4].dim = { 1,1 };
    }
    this->img_decoder_input_nodes[5].dim = { 1,1,256,256 };
    this->img_decoder_input_nodes[6].dim = { 1 };
    std::vector<int> frame_size = { ori_img->rows,ori_img->cols };

    int num_labels = 1;     // 1
    int height = 256;       // 256
    int width = 256;        // 256
    std::vector<float> mask_input(num_labels * 1 * height * width, 0.0f);
    std::vector<int64_t> mask_input_shape = { num_labels, 1, height, width };
    std::vector<float> has_mask_input = { 0.0f };
    //***************************************************************
    input_tensor.push_back(std::move(mem_attention_out[2]));    // image_embed
    input_tensor.push_back(std::move(mem_attention_out[0]));    // high_res_feats_0
    input_tensor.push_back(std::move(mem_attention_out[1]));    // high_res_feats_1
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, point_val.data(), point_val.size(),this->img_decoder_input_nodes[3].dim.data(),this->img_decoder_input_nodes[3].dim.size()));//point_coords
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, point_labels.data(), point_labels.size(),this->img_decoder_input_nodes[4].dim.data(),this->img_decoder_input_nodes[4].dim.size()));//point_labels
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, mask_input.data(), mask_input.size(), this->img_decoder_input_nodes[5].dim.data(), this->img_decoder_input_nodes[5].dim.size()));//mask_input
    input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, has_mask_input.data(), has_mask_input.size(), this->img_decoder_input_nodes[6].dim.data(), this->img_decoder_input_nodes[6].dim.size()));//has_mask_input
    input_tensor.push_back(Ort::Value::CreateTensor<int>(memory_info, frame_size.data(), frame_size.size(),this->img_decoder_input_nodes[7].dim.data(),this->img_decoder_input_nodes[7].dim.size()));//orig_im_size
    //***************************************************
    std::vector<Ort::Value> img_decoder_out;
    try {
        img_decoder_out = std::move(this->img_decoder_session->Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensor.data(),
            input_tensor.size(),
            output_names.data(),
            output_names.size()));
    }
    catch (const std::exception& e) {
        std::string error(e.what());
        //return std::format("ERROR: img_decoder_infer failed!!\n {}", error);
        return "ERROR: img_decoder_infer failed!!\n " + error;

    }
    return img_decoder_out;
}


//2024.11.21更新前处理版本，增加减均值除方差
void SAM2::preprocess(cv::Mat& image) {
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        std::cout << "Converted single-channel image to 3-channel BGR." << std::endl;
    }
    std::vector<cv::Mat> mats{ image };
    cv::Mat blob = cv::dnn::blobFromImages(mats, 1 / 255.0, cv::Size(1024, 1024), cv::Scalar(0, 0, 0), true, false);
    float mean[3] = { 0.485, 0.456, 0.406 };
    float std[3] = { 0.229, 0.224, 0.225 };
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel(blob.size[2], blob.size[3], CV_32F, blob.ptr(0, c)); // 提取第 c 通道
        channel -= mean[c]; 
        channel /= std[c];  
    }
    input_images.clear();
    this->input_images.emplace_back(blob);
}

//2024.11.21更新后处理版本,去除不必要操作
void SAM2::postprocess(std::vector<Ort::Value>& output_tensors) {
    float* output = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat outimg(this->ori_img->size(), CV_32FC1, output);
    cv::Mat dst;
    outimg.convertTo(dst, CV_8UC1, 255);
    cv::threshold(dst, dst, 0, 255, cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));//开运算
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, element);
    std::vector<std::vector<cv::Point>> contours; // 不一定是1
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(*ori_img, contours, -1, cv::Scalar(50, 250, 20), 2, cv::LINE_AA);
    cv::rectangle(*ori_img, parms.prompt_box, cv::Scalar(0, 0, 255), 2);

}

float SAM2::extract_score_from_tensor(const Ort::Value& output_tensor) {
    const Ort::TensorTypeAndShapeInfo& tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();
    const float* data = output_tensor.GetTensorData<float>();

    if (shape.size() == 2 && shape[0] == 1) {
        return data[0]; 
    }
    else if (shape.size() == 1) {
        return data[0];
    }
    else {
        std::cerr << "Unexpected tensor shape: ";
        for (int64_t dim : shape) {
            std::cerr << dim << " ";
        }
        std::cerr << std::endl;
        return -1.0f;  
    }
}

int SAM2::setparms(ParamsSam2 parms) {
    this->parms = std::move(parms);
    return 1;
}
int SAM2::setparms(int type, cv::Rect prompt_box, cv::Point prompt_point)
{
    this->parms.type = type;
    this->parms.prompt_box = prompt_box;
    this->parms.prompt_point = prompt_point;
    return 1;
}


