#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>
#include <cmath>
#include <thread>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include "NvInfer.h"
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif
class Logger: public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if(severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;
char* ReadFromPath(std::string eng_path,int &model_size){
    std::ifstream file(eng_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << eng_path << " error!" << std::endl;
        return nullptr;
    }
    char *trt_model_stream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if(!trt_model_stream){
        return nullptr;
    }
    file.read(trt_model_stream, size);
    file.close();
    model_size = size;
    return trt_model_stream;
}
void resize_GPU(int orig_h, int orig_w, void *img_buffer, void *out_buffer,int new_h, int new_w)
{
    Npp8u *pu8_src = static_cast<Npp8u*>(img_buffer);
    Npp8u *pu8_dst = static_cast<Npp8u*>(out_buffer);
    NppiSize npp_src_size{orig_w, orig_h};
    NppiSize npp_dst_size{new_w, new_h};
    int ret = nppiResize_8u_C3R(pu8_src, orig_w * 3, npp_src_size, NppiRect{0, 0, orig_w, orig_h},
                                pu8_dst, new_w * 3, npp_dst_size, NppiRect{0, 0, new_w, new_h},
                                NPPI_INTER_LINEAR);
    if(ret != 0){
        std::cerr << "nppiResize_8u_C3R error: " << ret << std::endl;
        return;
    }
#if 0
    cv::Mat img_cpu(new_h, new_w, CV_8UC3);
    size_t bytes = new_w * new_h * 3 * sizeof(Npp8u);
    CUDA_CHECK(cudaMemcpy(img_cpu.data, out_buffer, bytes, cudaMemcpyDeviceToHost));
    if(!cv::imwrite("output.jpg", img_cpu)){
        std::cerr << "Failed to save image"  << std::endl;
    } 
#endif
    return;
}
void PreprocessImage_GPU(cv::Mat &img, void *buffer, int input_h, int input_w, cudaStream_t stream){
    void *img_buffer = nullptr;
    int ret = 0;
    int orig_h = img.rows;
    int orig_w = img.cols;
    CUDA_CHECK(cudaMalloc(&img_buffer, orig_h * orig_w * 3));
    void *img_ptr = img.data;
    CUDA_CHECK(cudaMemcpyAsync(img_buffer, img_ptr, orig_h * orig_w * 3, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    resize_GPU(orig_h, orig_w, img_buffer, buffer, input_h, input_w);
    
    // pwc-net use BGR
    // BGR-->RGB
    // Npp8u *pu8_rgb = nullptr;
    // CUDA_CHECK(cudaMalloc(&pu8_rgb, input_h * input_w * 3));
    // int aOrder[3] = {2, 1, 0};
    // NppiSize size = {input_w, input_h};
    // NppStatus ret = nppiSwapChannels_8u_C3R((Npp8u*)buffer, input_w * 3, pu8_rgb, input_w * 3, size, aOrder);
    // if(ret != 0){
    //     std::cerr << "nppiSwapChannels_8u_C3R error: " << ret << std::endl;
    // }

    // 转 float 并归一化
    Npp8u *ptr_float = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr_float, input_h * input_w * 3 * sizeof(float)));
    NppiSize fsize = {input_w, input_h};
    ret = nppiConvert_8u32f_C3R((Npp8u *)buffer, input_w * 3, (Npp32f*)ptr_float, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiConvert_8u32f_C3R error: " << ret << std::endl;
    }
    Npp32f aConstants[3] = {1.f / 255.f, 1.f / 255.f,1.f / 255.f};
    ret = nppiMulC_32f_C3IR(aConstants, (Npp32f*)ptr_float, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiMulC_32f_C3IR error: " << ret << std::endl;
    }

    // HWC TO CHW
    NppiSize chw_size = {input_w, input_h};
    float* buffer_chw = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer_chw, input_h * input_w * 3 * sizeof(float)));
    Npp32f* dst_planes[3];
    dst_planes[0] = (Npp32f*)buffer_chw;
    dst_planes[1] = (Npp32f*)buffer_chw + input_h * input_w;
    dst_planes[2] = (Npp32f*)buffer_chw + input_h * input_w * 2;
    ret = nppiCopy_32f_C3P3R((Npp32f*)ptr_float, input_w * 3 * sizeof(float), dst_planes, input_w * sizeof(float), chw_size);
    if (ret != 0) {
        std::cerr << "nppiCopy_32f_C3P3R error: " << ret << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(buffer, buffer_chw, input_h * input_w * 3 * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(buffer_chw));
    CUDA_CHECK(cudaFree(img_buffer));
    // CUDA_CHECK(cudaFree(pu8_rgb));
    CUDA_CHECK(cudaFree(ptr_float));
    return;
}
size_t CountElement(const nvinfer1::Dims &dims, int batch_zise)
{
    int64_t total = batch_zise;
    for (int32_t i = 1; i < dims.nbDims; ++i){
        total *= dims.d[i];
    }
    return static_cast<size_t>(total);
}
// test version TensorRT-10.4.0.26
int Inference(nvinfer1::IExecutionContext* context, void** buffers, void* output, int one_output_len, const int batch_size, int channel, int input_h, int input_w, 
                std::vector<std::pair<int, std::string>> in_tensor_info, std::vector<std::pair<int, std::string>> out_tensor_info, cudaStream_t stream){
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = batch_size;
    trt_in_dims.d[1] = channel;
    trt_in_dims.d[2] = input_h;
    trt_in_dims.d[3] = input_w;
    context->setInputShape(in_tensor_info[0].second.c_str(), trt_in_dims);
    context->setInputShape(in_tensor_info[1].second.c_str(), trt_in_dims);
    if(!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed!" << std::endl;
        return -2;
    }
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[out_tensor_info[0].first], batch_size * one_output_len * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
cv::Mat FlowToHSV(const cv::Mat& flow)
{
    cv::Mat flow_split[2];
    cv::split(flow, flow_split);

    cv::Mat fx = flow_split[0];
    cv::Mat fy = flow_split[1];

    cv::Mat mag, ang;
    cv::cartToPolar(fx, fy, mag, ang, true); // angle in degrees

    cv::Mat hsv[3];

    // H
    hsv[0] = ang * 0.5;

    // S
    hsv[1] = cv::Mat::ones(ang.size(), CV_32F) * 255;

    // V
    cv::normalize(mag, hsv[2], 0, 255, cv::NORM_MINMAX);

    cv::Mat hsv_merge;
    cv::merge(hsv, 3, hsv_merge);

    hsv_merge.convertTo(hsv_merge, CV_8UC3);

    cv::Mat bgr;
    cv::cvtColor(hsv_merge, bgr, cv::COLOR_HSV2BGR);

    return bgr;
}
cv::Mat PostprocessFlowSingle(float* flow_ptr, int flow_h, int flow_w, int orig_h, int orig_w, int net_h, int net_w){
    // flow_ptr: [2, H, W]
    cv::Mat flow_x(flow_h, flow_w, CV_32F, flow_ptr);
    cv::Mat flow_y(flow_h, flow_w, CV_32F, flow_ptr + flow_h * flow_w);

    cv::Mat flow_x_resized, flow_y_resized;
    cv::resize(flow_x, flow_x_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
    cv::resize(flow_y, flow_y_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    flow_x_resized *= (float)orig_w / net_w;
    flow_y_resized *= (float)orig_h / net_h;

    std::vector<cv::Mat> channels = {flow_x_resized, flow_y_resized};
    cv::Mat flow_hwc;
    cv::merge(channels, flow_hwc);

    return flow_hwc;
}
int PictureInfer(cudaStream_t &stream, std::vector<std::pair<int, std::string>> &in_tensor_info, std::vector<std::pair<int, std::string>> &out_tensor_info,
                nvinfer1::IExecutionContext* context, void* buffers[3], float* output, int input_h, int input_w, std::string path){
    int test_batch = 2;
    int buffer_idx = 0;
    char* input_ptr_one = static_cast<char*>(buffers[in_tensor_info[0].first]);
    char* input_ptr_two = static_cast<char*>(buffers[in_tensor_info[1].first]);
    cv::Mat img_one = cv::imread(path + std::string("/one.png"));
    cv::Mat img_two = cv::imread(path + std::string("/two.png"));
    int orig_h = img_one.rows;
    int orig_w = img_one.cols;
    for(int i = 0; i < test_batch; i++){
        PreprocessImage_GPU(img_one, input_ptr_one + buffer_idx, input_h, input_w, stream);
        PreprocessImage_GPU(img_two, input_ptr_two + buffer_idx, input_h, input_w, stream);
        buffer_idx += input_h * input_w * 3 * sizeof(float);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Inference(context, buffers, (void*)output, input_h * input_w * 2, test_batch, 3, input_h, input_w, in_tensor_info, out_tensor_info, stream);
    auto end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " time: " << duration << "ms" << std::endl;

    for(int i = 0; i < test_batch; i++){
        float* flow_ptr = output + i * (input_h * input_w * 2);

        cv::Mat flow = PostprocessFlowSingle(flow_ptr, input_h, input_w, orig_h, orig_w, input_h, input_w);

        cv::Mat vis = FlowToHSV(flow);
        std::string save_name = "flow_trt_" + std::to_string(i) + ".jpg";
        cv::imwrite(save_name, vis);
    }
    return 0;
}
int VideoInfer(cudaStream_t &stream, std::vector<std::pair<int, std::string>> &in_tensor_info, std::vector<std::pair<int, std::string>> &out_tensor_info,
                nvinfer1::IExecutionContext* context, void* buffers[2], float* output, int input_h, int input_w, std::string path){
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return -1;
    }
    cv::Mat img_one;
    bool ret = cap.read(img_one);
    if(!ret){
        std::cout << "视频读取完毕或出现错误" << std::endl;
        return -1;
    }
    int orig_h = img_one.rows;
    int orig_w = img_one.cols;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << std::endl;
    cv::VideoWriter writer;
    const char* save_name = "flow_trt.mp4";
    writer.open(save_name, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(orig_w, orig_h));
    bool over_flag = false;
    while(!over_flag){
        std::vector<std::tuple<float, float, float>> res_pre;
        int buffer_idx = 0;
        char* input_ptr_one = static_cast<char*>(buffers[in_tensor_info[0].first]);
        char* input_ptr_two = static_cast<char*>(buffers[in_tensor_info[1].first]);
        cv::Mat img_two;
        ret = cap.read(img_two);
        if(!ret){
            std::cout << "视频读取完毕或出现错误" << std::endl;
            over_flag = true;
            break;
        }
        PreprocessImage_GPU(img_one, input_ptr_one + buffer_idx, input_h, input_w, stream);
        PreprocessImage_GPU(img_two, input_ptr_two + buffer_idx, input_h, input_w, stream);

        auto start = std::chrono::high_resolution_clock::now();
        Inference(context, buffers, (void*)output, input_h * input_w * 2, 1, 3, input_h, input_w, in_tensor_info, out_tensor_info, stream);
        auto end = std::chrono::high_resolution_clock::now();
        long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " time: " << duration << "ms" << std::endl;

        img_one = img_two.clone();
        float* flow_ptr = output;
        cv::Mat flow = PostprocessFlowSingle(flow_ptr, input_h, input_w, orig_h, orig_w, input_h, input_w);
        cv::Mat vis = FlowToHSV(flow);
        writer.write(vis);
    }
    std::cout << "Saved: " << save_name << std::endl;
    cap.release();
    writer.release();
    return 0;
}
int main(int argc, char **argv){
    if(argc < 3){
        std::cerr << "./bin eng_path video/test.mp4 or images video/picture" << std::endl;
        return 0;
    }
    const char *eng_path = argv[1];
    const char *media_path = argv[2];
    std::string media_type = argv[3];
    int device_id = 0;
    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	assert(runtime != nullptr);
    int model_size = 0;
    char *trt_model_stream = ReadFromPath(eng_path,model_size);
    assert(trt_model_stream != nullptr);

    auto engine{runtime->deserializeCudaEngine(trt_model_stream, model_size)};
	assert(engine != nullptr);

    auto context{engine->createExecutionContext()};
	assert(context != nullptr);
    delete []trt_model_stream;

    int num_bindings = engine->getNbIOTensors();
	std::cout << "input/output : " << num_bindings << std::endl;
	std::vector<std::pair<int, std::string>> in_tensor_info;
	std::vector<std::pair<int, std::string>> out_tensor_info;
    for (int i = 0; i < num_bindings; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_tensor_info.push_back({i, std::string(tensor_name)});
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_tensor_info.push_back({i, std::string(tensor_name)});
    }
    for(int idx = 0; idx < in_tensor_info.size(); idx++){
        nvinfer1::Dims in_dims=context->getTensorShape(in_tensor_info[idx].second.c_str());
        std::cout << "input: " << in_tensor_info[idx].second.c_str() << std::endl;
        for(int i = 0; i < in_dims.nbDims; i++){
            std::cout << "dims [" << i << "]: " << in_dims.d[i] << std::endl;
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(in_tensor_info[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            std::cout << "类型为 int32" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            std::cout << "类型为 int64" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            std::cout << "类型为 float" << std::endl;
        }
        std::cout << std::endl;
    }
    for(int idx = 0; idx < out_tensor_info.size(); idx++){
        nvinfer1::Dims out_dims=context->getTensorShape(out_tensor_info[idx].second.c_str());
        std::cout << "output: " << out_tensor_info[idx].second.c_str() << std::endl;
        for(int i = 0; i < out_dims.nbDims; i++){
            std::cout << "dims [" << i << "]: " << out_dims.d[i] << std::endl;
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(out_tensor_info[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            std::cout << "类型为 int32" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            std::cout << "类型为 int64" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            std::cout << "类型为 float" << std::endl;
        }
        std::cout << std::endl;
    }
    assert(in_tensor_info.size() == 2);
    assert(out_tensor_info.size() == 1);
    

    int batch_size = 4; // trtexex转模型设置的最大batch
    nvinfer1::Dims in_dims = context->getTensorShape(in_tensor_info[0].second.c_str()); // input1和input2尺寸一致
    nvinfer1::Dims out_dims = context->getTensorShape(out_tensor_info[0].second.c_str()); // output
    size_t max_in_size_byte = CountElement(in_dims, batch_size) * sizeof(float); // batch_size * input_h * input_w * 3 * sizeof(float)
    size_t max_out_size_byte = CountElement(out_dims, batch_size) * sizeof(float); // batch_size * input_h_flow * input_w_flow * 2 * sizeof(float)
    // in_dims.d[0] dynamic batch_size == -1 
    int channel = in_dims.d[1];
    int input_h = in_dims.d[2];
	int input_w = in_dims.d[3];
    std::cout << "batch_size:" << batch_size << " channel:" << channel << " input_h:" << input_h << " input_w:" << input_w << std::endl;

    // out_dims.d[0] dynamic batch_size == -1
    int channel_flow = out_dims.d[1];
    int input_h_flow = out_dims.d[2];
	int input_w_flow = out_dims.d[3];
    std::cout << "batch_size_flow:" << batch_size << " channel_flow:" << channel_flow << " input_h_flow:" << input_h_flow << " input_w_flow:" << input_w_flow << std::endl;
	
    void* buffers[3] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[0].first], max_in_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[1].first], max_in_size_byte));

	CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[0].first], max_out_size_byte));
    float* output = new float[max_out_size_byte];
    // set in/out tensor address
    context->setInputTensorAddress(in_tensor_info[0].second.c_str(), buffers[in_tensor_info[0].first]);
    context->setInputTensorAddress(in_tensor_info[1].second.c_str(), buffers[in_tensor_info[1].first]);
    context->setOutputTensorAddress(out_tensor_info[0].second.c_str(), buffers[out_tensor_info[0].first]);
    
    if(media_type == std::string("picture"))
        PictureInfer(stream, in_tensor_info, out_tensor_info, context, buffers, output, input_h, input_w, media_path);
    else if(media_type == std::string("video"))
        VideoInfer(stream, in_tensor_info, out_tensor_info, context, buffers, output, input_h, input_w, media_path);
    else
        std::cout << "must be picture or video" << std::endl;
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    CUDA_CHECK(cudaFree(buffers[2]));
    delete []output;
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}