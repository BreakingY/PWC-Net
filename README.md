# PWC-Net
- 本项目基于 https://github.com/sniklaus/pytorch-pwc (PWC-Net 的 PyTorch 复现版本)进行开发，新增功能为 ONNX 模型导出、TensorRT推理加速。
# 功能
- 去掉自定义算子，新增 ONNX 模型导出
    - python run.py
    - 查看光流结果： python view_flo.py
- onnx推理
    - python infer_onnx.py
- tensorrt推理
    - /data/sunkx/TensorRT-10.4.0.26/bin/trtexec --onnx=./pwcnet.onnx --minShapes=input1:1x3x448x1024,input2:1x3x448x1024 --optShapes=input1:4x3x448x1024,input2:4x3x448x1024 --maxShapes=input1:4x3x448x1024,input2:4x3x448x1024 --saveEngine=pwcnet.engine --fp16
    - make -f Makefile_trt
    - ./infer_trt pwcnet.engine video/test.mp4 video && ./infer_trt pwcnet.engine images picture
- 晟腾CANN推理
    - 测试版本：8.2.RC1 8.5.0
    - atc --model=./pwcnet.onnx --framework=5 --input_shape="input1:-1,3,448,1024;input2:-1,3,448,1024" --dynamic_batch_size="1,2,3,4" --insert_op_conf=./insert_op.cfg --output=pwcnet --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16
    - make -f Makefile_cann
    - ./infer_cann pwcnet.om video/test.mp4 video && ./infer_cann pwcnet.om images picture