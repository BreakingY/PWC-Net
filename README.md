# PWC-Net
- 本项目基于 https://github.com/sniklaus/pytorch-pwc (PWC-Net 的 PyTorch 复现版本)进行开发，新增功能为 ONNX 模型导出、TensorRT推理加速。
# 功能
- 去掉自定义算子，新增 ONNX 模型导出
    - python run.py
    - 查看光流结果： python view_flo.py
- onnx推理demo
    - python infer_onnx.py
- tensorrt推理
    - /data/sunkx/TensorRT-10.4.0.26/bin/trtexec --onnx=./pwcnet.onnx --minShapes=input1:1x3x448x1024,input2:1x3x448x1024 --optShapes=input1:4x3x448x1024,input2:4x3x448x1024 --maxShapes=input1:4x3x448x1024,input2:4x3x448x1024 --saveEngine=pwcnet.engine --fp16
    - make
    - ./infer_trt pwcnet.engine video/test.mp4 video && ./infer_trt pwcnet.engine images picture