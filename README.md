# Sam2Onnx_Inference
SAM2分割实现文件包含Model.h、Sam2.h、Sam2.cpp和测试代码main.cpp，支持float32、float16推理
![1_out00000008_206448_result](https://github.com/user-attachments/assets/05bee99a-06c6-4ecc-a1f3-d78f4f81ad26)
![Uploading 1_out00000010_206608_result.jpg…]()


## 功能：
  * Sam2单帧照片，单目标分割，使用点或者框作为提示信息(节省推理效率，去除了视频推理中记忆模块的使用)
  * Sam2所使用onnx权重格式生成代码[Sam2Onnx_Inference](https://github.com/lyxlplhy/Sam2-collection?tab=readme-ov-file#onnx%E5%AF%BC%E5%87%BA)
  * 新增yolo目标检测的结果作为sam2分割提示点，后续将完善代码(2024-11-27)

## 依赖
  * opencv=4.8
  * onnxruntime=1.18.0
  * cuda=12.4
  * cudnn

## 参考
  * Sam2:[https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
  * OrtInference：[https://github.com/Aimol-l/OrtInference?tab=readme-ov-file](https://github.com/Aimol-l/OrtInference?tab=readme-ov-file)
  * ONNX-SAM2-Segment-Anything:[https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)
  * Sam2-collection: [https://github.com/lyxlplhy/Sam2-collection/tree/master](https://github.com/lyxlplhy/Sam2-collection/tree/master)

