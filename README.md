# PWC-Net
- 本项目基于 https://github.com/sniklaus/pytorch-pwc (PWC-Net 的 PyTorch 复现版本)进行开发，新增功能为 ONNX 模型导出、TensorRT(适配10.4、8.5、8.4)、CANN推理加速。
# 功能
- 去掉自定义算子，新增 ONNX 模型导出
    - python run.py --backend=<trt104;trt85;trt84;cann> --trt_plugin=<false;true>(使用TensorRT插件替换Correlation和Backwarp算子，实现加速)
    - 查看光流结果： python view_flo.py
    - backend=trt84的时候需要执行python fix_trt84_onnx.py <pwcnet_trt84.onnx;pwcnet_trt84_plugin.onnx>修改onnx，TensorRT使用修复后的onnx
- onnx推理
    - python infer_onnx.py
- tensorrt推理
    - 编译插件(trt85为例,兼容trt104、trt85、trt84)
        * cd plugin
        * mkdir build && cd build
        * export PATH=$PATH:/usr/local/cuda/bin && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
        * cmake -DTRT_PATH=/data/sunkx/TensorRT-8.5.1.7 ..
        * make
    - 非插件版本(trt85为例)：
        * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/sunkx/TensorRT-8.5.1.7/lib
        * /data/sunkx/TensorRT-8.5.1.7/bin/trtexec --onnx=pwcnet_trt85.onnx --minShapes=input1:1x3x384x768,input2:1x3x384x768 --optShapes=input1:4x3x384x768,input2:4x3x384x768 --maxShapes=input1:4x3x384x768,input2:4x3x384x768  --saveEngine=pwcnet_trt85.engine --fp16
    - 插件版本(trt85为例)：
        * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/sunkx/TensorRT-8.5.1.7/lib
        * /data/sunkx/TensorRT-8.5.1.7/bin/trtexec --onnx=pwcnet_trt85_plugin.onnx --minShapes=input1:1x3x384x768,input2:1x3x384x768 --optShapes=input1:4x3x384x768,input2:4x3x384x768 --maxShapes=input1:4x3x384x768,input2:4x3x384x768 --plugins=./plugin/build/libpwc_net_plugin.so --saveEngine=pwcnet_trt85_plugin.engine --fp16
    - make -f Makefile_trt TRT_VERSION=<TRT_10;TRT_84_85> TRT_PATH=</data/sunkx/TensorRT-10.4.0.26;/data/sunkx/TensorRT-8.5.1.7;/usr>
    - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/plugin/build/:/usr/local/opencvgpu/lib/
    - ./infer_trt <engine> video/test.mp4 video && ./infer_trt <engine> images picture
- 晟腾CANN推理
    - 测试版本：8.2.RC1 8.5.0
    - atc --model=./pwcnet_cann.onnx --framework=5 --input_shape="input1:-1,3,384,768;input2:-1,3,384,768" --dynamic_batch_size="1,2,3,4" --insert_op_conf=./insert_op.cfg --output=pwcnet --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16
    - make -f Makefile_cann
    - ./infer_cann pwcnet.om video/test.mp4 video && ./infer_cann pwcnet.om images picture
