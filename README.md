# Sam2Onnx_Inference Muti_object_seg
SAM2多提示框/点分割实现文件包含Model.h、Sam2.h、Sam2.cpp和测试代码main.cpp，支持float32、float16推理

## 功能：
  * sam2仅支持对同一张图像，一次输入单个提示框进行单个目标分割。本项目提供一次可输入多个提示框、点对一张图像不同物理进行分割。
## 依赖
  * opencv=4.8
  * onnxruntime=1.18.0
  * cuda=12.4
  * cudnn


