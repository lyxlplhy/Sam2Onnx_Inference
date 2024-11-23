# OrtInference
文件包含Model.h、Sam2.h、Sam2.cpp和测试代码main.cpp

目前完成：
  * Sam2单帧照片的目标分割，使用点或者框作为提示信息(节省推理效率，去除了视频推理中记忆模块的使用)
  * Sam2所使用onnx权重格式生成代码[Sam2Onnx_Inference](https://github.com/lyxlplhy/Sam2-collection/blob/master/README.md#onnx%E5%AF%BC%E5%87%BA)

# 依赖
  * opencv=4.8
  * onnxruntime=1.18.0
  * cuda=12.4
  * cudnn

# 参考
  * Sam2:[https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
  * OrtInference：[https://github.com/Aimol-l/OrtInference?tab=readme-ov-file](https://github.com/Aimol-l/OrtInference?tab=readme-ov-file)
  * ONNX-SAM2-Segment-Anything:[https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)
  * Sam2Onnx_Inference: [https://github.com/lyxlplhy/Sam2Onnx_Inference](https://github.com/lyxlplhy/Sam2-collection/blob/master)

Sam2Onnx_Inference
