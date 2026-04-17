#include <iostream>
#include <string>
#include <vector>
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#include <ops/acl_dvpp.h>
#include <opencv2/opencv.hpp>
#ifndef CHECK_ACL
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif

#ifndef CHECK_DVPP_MPI
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif
#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)
std::string GetType(aclDataType type){
    std::string type_str;
    switch (type)
    {
    case ACL_DT_UNDEFINED:
        type_str = std::string("ACL_DT_UNDEFINED");
        break;
    case ACL_FLOAT:
        type_str = std::string("ACL_FLOAT");
        break;
    case ACL_FLOAT16:
        type_str = std::string("ACL_FLOAT16");
        break;
    case ACL_INT8:
        type_str = std::string("ACL_INT8");
        break;
    case ACL_INT32:
        type_str = std::string("ACL_INT32");
        break;
    case ACL_UINT8:
        type_str = std::string("ACL_UINT8");
        break;
    case ACL_INT16:
        type_str = std::string("ACL_INT16");
        break;
    case ACL_UINT16:
        type_str = std::string("ACL_UINT16");
        break;
    case ACL_UINT32:
        type_str = std::string("ACL_UINT32");
        break;
    case ACL_INT64:
        type_str = std::string("ACL_INT64");
        break;
    case ACL_UINT64:
        type_str = std::string("ACL_UINT64");
        break;
    case ACL_DOUBLE:
        type_str = std::string("ACL_DOUBLE");
        break;
    case ACL_BOOL:
        type_str = std::string("ACL_BOOL");
        break;
    case ACL_STRING:
        type_str = std::string("ACL_STRING");
        break;
    case ACL_COMPLEX64 :
        type_str = std::string("ACL_COMPLEX64");
        break;
    case ACL_COMPLEX128:
        type_str = std::string("ACL_COMPLEX128");
        break;
    case ACL_BF16:
        type_str = std::string("ACL_BF16");
        break;
    case ACL_INT4:
        type_str = std::string("ACL_INT4");
        break;
    case ACL_UINT1:
        type_str = std::string("ACL_UINT1");
        break;
    case ACL_COMPLEX32:
        type_str = std::string("ACL_COMPLEX32");
        break;
    case ACL_HIFLOAT8:
        type_str = std::string("ACL_HIFLOAT8");
        break;
    case ACL_FLOAT8_E5M2:
        type_str = std::string("ACL_FLOAT8_E5M2");
        break;
    case ACL_FLOAT8_E4M3FN:
        type_str = std::string("ACL_FLOAT8_E4M3FN");
        break;
    case ACL_FLOAT8_E8M0:
        type_str = std::string("ACL_FLOAT8_E8M0");
        break;
    case ACL_FLOAT6_E3M2:
        type_str = std::string("ACL_FLOAT6_E3M2");
        break;
    case ACL_FLOAT6_E2M3:
        type_str = std::string("ACL_FLOAT6_E2M3");
        break;
    case ACL_FLOAT4_E2M1:
        type_str = std::string("ACL_FLOAT4_E2M1");
        break;
    case ACL_FLOAT4_E1M2:
        type_str = std::string("ACL_FLOAT4_E1M2");
        break;
    default:
        break;
    }
    return type_str;
}
void PreprocessImage(cv::Mat &img, void *buffer, int input_h, int input_w,  hi_vpc_chn channel_id_resize, aclrtStream stream){
    void *img_buffer = nullptr;
    int ret = 0;
    int orig_h = img.rows;
    int orig_w = img.cols;
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &img_buffer, orig_h * orig_w * 3));
    void *img_ptr = img.data;

    CHECK_ACL(aclrtMemcpyAsync(img_buffer, orig_h * orig_w * 3, img_ptr, orig_h * orig_w * 3, ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    hi_vpc_pic_info input_pic;
    input_pic.picture_width = orig_w;
    input_pic.picture_height = orig_h;
    input_pic.picture_format = HI_PIXEL_FORMAT_BGR_888;
    input_pic.picture_width_stride = orig_w * 3;
    input_pic.picture_height_stride = orig_h;
    input_pic.picture_buffer_size = orig_h * orig_w * 3;
    input_pic.picture_address = img_buffer;

    hi_vpc_pic_info output_pic;
    output_pic.picture_width = input_w;
    output_pic.picture_height = input_h;
    output_pic.picture_format = HI_PIXEL_FORMAT_BGR_888;
    output_pic.picture_width_stride = input_w * 3;
    output_pic.picture_height_stride = input_h;
    output_pic.picture_buffer_size = input_h * input_w * 3;
    output_pic.picture_address = buffer;

    uint32_t task_id = 0;
    hi_double fx = (double)input_w / orig_w;
    hi_double fy = (double)input_h / orig_h;
    CHECK_DVPP_MPI(hi_mpi_vpc_resize(channel_id_resize, &input_pic, &output_pic, fx, fy, 0, &task_id, -1));
    CHECK_DVPP_MPI(hi_mpi_vpc_get_process_result(channel_id_resize, task_id, -1));
    CHECK_DVPP_MPI(hi_mpi_dvpp_free(img_buffer));
    return;
}
int Inference(aclrtStream stream, uint32_t model_id,
                aclmdlDataset *input, aclDataBuffer* data_buf_inputs[2], void *input_addr_imgs[2],
                aclmdlDataset *output, std::vector<void*> output_buf, 
                int input_h, int input_w, int batch_size, int dynamic_batch_idx, float* output_flow)
{
    CHECK_ACL(aclUpdateDataBuffer(data_buf_inputs[0], input_addr_imgs[0], batch_size * input_w * input_h * 3));
    CHECK_ACL(aclUpdateDataBuffer(data_buf_inputs[1], input_addr_imgs[1], batch_size * input_w * input_h * 3));
    CHECK_ACL(aclmdlSetDynamicBatchSize(model_id, input, dynamic_batch_idx, batch_size));
    CHECK_ACL(aclmdlExecuteAsync(model_id, input, output, stream));
    int output_flow_len = input_h * input_w * 2 * sizeof(float);
    memset(output_flow, 0, batch_size *  output_flow_len);
    CHECK_ACL(aclrtMemcpyAsync(output_flow, batch_size *  output_flow_len , output_buf[0], batch_size * output_flow_len, ACL_MEMCPY_DEVICE_TO_HOST, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    return 0;
}
cv::Mat FlowToHSV(const cv::Mat& flow)
{
    cv::Mat flow_split[2];
    cv::split(flow, flow_split);
    cv::Mat fx = flow_split[0];
    cv::Mat fy = flow_split[1];
    cv::Mat mag, ang;
    cv::cartToPolar(fx, fy, mag, ang, true);
    cv::Mat hsv[3];
    cv::Mat hue = cv::Mat::zeros(ang.size(), CV_32F);
    for (int y = 0; y < ang.rows; y++){
        const float* ang_ptr = ang.ptr<float>(y);
        float* hue_ptr = hue.ptr<float>(y);

        for (int x = 0; x < ang.cols; x++){
            float a = ang_ptr[x];

            float rad = a * CV_PI / 180.0f;

            float h = 0.0f;

            if (rad < CV_PI / 2){
                h = (rad / (CV_PI / 2)) * 30.0f;
            }
            else if (rad < CV_PI){
                h = 30.0f + ((rad - CV_PI / 2) / (CV_PI / 2)) * 90.0f;
            }
            else{
                h = 120.0f + ((rad - CV_PI) / CV_PI) * 60.0f;
            }

            hue_ptr[x] = h;
        }
    }
    hsv[0] = hue;

    cv::Mat mag_norm;
    cv::normalize(mag, mag_norm, 0, 255, cv::NORM_MINMAX);
    hsv[1] = mag_norm;

    hsv[2] = cv::Mat::ones(mag.size(), CV_32F) * 255;

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
int PictureInfer(aclrtStream stream, uint32_t model_id, hi_vpc_chn channel_id_resize,
                aclmdlDataset *input, aclDataBuffer* data_buf_inputs[2], void *input_addr_imgs[2],
                aclmdlDataset *output, std::vector<void*> output_buf,
                size_t dynamic_batch_idx,
                float* output_flow, int input_h, int input_w, std::string path){
    int test_batch = 2;
    int buffer_idx = 0;
    char* input_ptr_one = static_cast<char*>(input_addr_imgs[0]);
    char* input_ptr_two = static_cast<char*>(input_addr_imgs[1]);
    cv::Mat img_one = cv::imread(path + std::string("/one.png"));
    cv::Mat img_two = cv::imread(path + std::string("/two.png"));
    int orig_h = img_one.rows;
    int orig_w = img_one.cols;
    for(int i = 0; i < test_batch; i++){
        PreprocessImage(img_one, input_ptr_one + buffer_idx, input_h, input_w, channel_id_resize, stream);
        PreprocessImage(img_two, input_ptr_two + buffer_idx, input_h, input_w, channel_id_resize, stream);
        buffer_idx += input_h * input_w * 3;
    }

    auto start = std::chrono::high_resolution_clock::now();
    Inference(stream, model_id, input, data_buf_inputs, input_addr_imgs, output, output_buf, input_h, input_w, test_batch, dynamic_batch_idx, output_flow);
    auto end = std::chrono::high_resolution_clock::now();
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << " time: " << duration << "ms" << std::endl;

    for(int i = 0; i < test_batch; i++){
        float* flow_ptr = output_flow + i * (input_h * input_w * 2);

        cv::Mat flow = PostprocessFlowSingle(flow_ptr, input_h, input_w, orig_h, orig_w, input_h, input_w);

        cv::Mat vis = FlowToHSV(flow);
        std::string save_name = "flow_trt_" + std::to_string(i) + ".jpg";
        cv::imwrite(save_name, vis);
    }
    return 0;
}
int VideoInfer(aclrtStream stream, uint32_t model_id, hi_vpc_chn channel_id_resize,
                aclmdlDataset *input, aclDataBuffer* data_buf_inputs[2], void *input_addr_imgs[2],
                aclmdlDataset *output, std::vector<void*> output_buf,
                size_t dynamic_batch_idx,
                float* output_flow, int input_h, int input_w, std::string path)
{
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
        char* input_ptr_one = static_cast<char*>(input_addr_imgs[0]);
        char* input_ptr_two = static_cast<char*>(input_addr_imgs[1]);
        cv::Mat img_two;
        ret = cap.read(img_two);
        if(!ret){
            std::cout << "视频读取完毕或出现错误" << std::endl;
            over_flag = true;
            break;
        }
        PreprocessImage(img_one, input_ptr_one + buffer_idx, input_h, input_w, channel_id_resize, stream);
        PreprocessImage(img_two, input_ptr_two + buffer_idx, input_h, input_w, channel_id_resize, stream);

        auto start = std::chrono::high_resolution_clock::now();
        Inference(stream, model_id, input, data_buf_inputs, input_addr_imgs, output, output_buf, input_h, input_w, 1, dynamic_batch_idx, output_flow);
        auto end = std::chrono::high_resolution_clock::now();
        long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " time: " << duration << "ms" << std::endl;

        img_one = img_two.clone();
        float* flow_ptr = output_flow;
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
    uint32_t model_id;
    aclmdlDesc *model_desc = nullptr;
    aclmdlDataset *output = nullptr;
    size_t outputs_num;
    std::vector<void*> output_buf;
    std::vector<size_t> output_size;
    std::vector<aclDataBuffer*> output_data_buf;
    aclmdlDataset *input = nullptr;
    aclDataBuffer* data_buf_inputs[2] = {nullptr, nullptr};
    void *input_addr_imgs[2] = {nullptr, nullptr};
    uint32_t input_num;
    size_t aipp_index_1 = -1;
    size_t aipp_index_2 = -1;
    void *input_AIPPs[2] = {nullptr, nullptr};
    size_t dynamic_batch_idx;
    void *input_batch = nullptr;
    aclrtStream stream;
    hi_vpc_chn channel_id_resize;
    int input_h;
    int input_w;
    int input_h_flow;
    int input_w_flow;
    int batch_size = 4;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_DVPP_MPI(hi_mpi_sys_init()); // hi_mpi_sys_init 必须在aclrtSetDevice之后
    CHECK_ACL(aclmdlLoadFromFile(eng_path, &model_id));
    model_desc = aclmdlCreateDesc();
    if(model_desc == nullptr){
        std::cout << "aclmdlCreateDesc error" << std::endl;
        return -1;
    }
    CHECK_ACL(aclmdlGetDesc(model_desc, model_id));
    output = aclmdlCreateDataset();
    if (output == nullptr) {
        std::cout << "aclmdlCreateDataset ouput error" << std::endl;
        return -1;
    }

    input_num = aclmdlGetNumInputs(model_desc);
    /*
    intput name: input1type: ACL_UINT8
    intput name: input2type: ACL_UINT8
    intput name: ascend_mbatch_shape_datatype: ACL_INT32
    intput name: ascend_dynamic_aipp_datatype: ACL_UINT8
    intput name: ascend_dynamic_aipp_data_input2_0_aipptype: ACL_UINT8
    output name: PartitionedCall_/model/Resize_Resize_12:0:outputtype: ACL_FLOAT
    */
    for (size_t i = 0; i < input_num; ++i) {
        std::cout << "intput name: " << aclmdlGetInputNameByIndex(model_desc, i) << " type: " << GetType(aclmdlGetInputDataType(model_desc, i)) << std::endl;
    }

    outputs_num = aclmdlGetNumOutputs(model_desc); // 1个输出
    for (size_t i = 0; i < outputs_num; ++i) {
        std::cout << "output name: " << aclmdlGetOutputNameByIndex(model_desc, i) << " type: " << GetType(aclmdlGetOutputDataType(model_desc, i)) << std::endl;
        size_t buf_size = aclmdlGetOutputSizeByIndex(model_desc, i);
        void *buffer = nullptr;
        CHECK_ACL(aclrtMalloc(&buffer, buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
        aclDataBuffer* data_buf = aclCreateDataBuffer(buffer, buf_size);
        if (data_buf == nullptr) {
            std::cout << "aclCreateDataBuffer error" << std::endl;
            exit(0);
        }
        CHECK_ACL(aclmdlAddDatasetBuffer(output, data_buf));
        output_buf.push_back(buffer); // 第i个输出的显存地址
        output_size.push_back(buf_size); // 第i个输出的显存大小
        output_data_buf.push_back(data_buf); // 包装了第i个输出显存地址的输出描述
    }
    // AIPP
    // CHECK_ACL(aclmdlGetInputIndexByName(model_desc, ACL_DYNAMIC_AIPP_NAME, &aipp_index)); // 仅仅适用单个AIPP
    // size_t data_len_aipp = aclmdlGetInputSizeByIndex(model_desc, aipp_index);
    // CHECK_ACL(aclrtMalloc(&input_AIPP, data_len_aipp, ACL_MEM_MALLOC_NORMAL_ONLY));
    aipp_index_1 = 3;
    aipp_index_2 = 4;
    size_t data_len_aipp_1 = aclmdlGetInputSizeByIndex(model_desc, aipp_index_1);
    CHECK_ACL(aclrtMalloc(&input_AIPPs[0], data_len_aipp_1, ACL_MEM_MALLOC_NORMAL_ONLY));
    size_t data_len_aipp_2 = aclmdlGetInputSizeByIndex(model_desc, aipp_index_2);
    CHECK_ACL(aclrtMalloc(&input_AIPPs[1], data_len_aipp_2, ACL_MEM_MALLOC_NORMAL_ONLY));
    
    
    // 动态batch
    // CHECK_ACL(aclmdlGetInputIndexByName(model_desc, ACL_DYNAMIC_TENSOR_NAME, &dynamic_batch_idx));
    dynamic_batch_idx = 2;
    size_t data_len = aclmdlGetInputSizeByIndex(model_desc, dynamic_batch_idx);
    CHECK_ACL(aclrtMalloc(&input_batch, data_len, ACL_MEM_MALLOC_NORMAL_ONLY));

    hi_vpc_chn_attr st_chn_attr {};
    st_chn_attr.attr = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_sys_create_chn(&channel_id_resize, &st_chn_attr));

    // input1 和 input2尺寸一样
    aclmdlIODims input_dims1, input_dims2;
    CHECK_ACL(aclmdlGetInputDims(model_desc, 0, &input_dims1));
    CHECK_ACL(aclmdlGetInputDims(model_desc, 1, &input_dims2));
    input_dims1.dims[2] > 0 ? input_h = input_dims1.dims[2]: input_h = 448;
    input_dims1.dims[3] > 0 ? input_w = input_dims1.dims[3]: input_w = 1024;
    std::cout << "input_h: " << input_h << " input_w: " << input_w << std::endl;

    aclmdlIODims output_dims;
    CHECK_ACL(aclmdlGetOutputDims(model_desc, 0, &output_dims));
    output_dims.dims[2] > 0 ? input_h_flow = output_dims.dims[2]: input_h_flow = 448;
    output_dims.dims[3] > 0 ? input_w_flow = output_dims.dims[3]: input_w_flow = 1024;
    std::cout << "input_h_flow: " << input_h_flow << " input_w_flow: " << input_w_flow << std::endl;
    assert(input_h == input_h_flow);
    assert(input_w_flow == input_w_flow);

    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_imgs[0], batch_size * input_h * input_w * 3));
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_imgs[1], batch_size * input_h * input_w * 3));
    CHECK_ACL(aclrtCreateStream(&stream));

    float *output_flow = new float[input_h_flow * input_w_flow * 2 * batch_size];

    // set input
    input = aclmdlCreateDataset();
    if (input == nullptr) {
        std::cout << "aclmdlCreateDataset input error" << std::endl;
        return -1;
    }
    for (size_t i = 0; i < input_num; ++i) {
        size_t buf_size = aclmdlGetInputSizeByIndex(model_desc, i);
        aclDataBuffer* data_buf;
        if(i == 0 || i == 1){ // input1 input2
            data_buf = aclCreateDataBuffer(nullptr,0);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            if(i == 0){
                data_buf_inputs[0] = data_buf;
            }
            else{
                data_buf_inputs[1] = data_buf;
            }
        }
        else if(i == aipp_index_1 || i == aipp_index_2){
            if(i == aipp_index_1){
                data_buf = aclCreateDataBuffer(input_AIPPs[0], buf_size);
            }
            else{
                data_buf = aclCreateDataBuffer(input_AIPPs[1], buf_size);
            }
            
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            aclmdlAIPP* aipp_param_tensor = aclmdlCreateAIPP(batch_size);
            if(aipp_param_tensor == nullptr){
                std::cout << "aclmdlCreateAIPP error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlSetAIPPInputFormat(aipp_param_tensor, ACL_RGB888_U8)); // 设置输入格式, 
            // CHECK_ACL(aclmdlSetAIPPRbuvSwapSwitch(aipp_param_tensor, 1)); // 通道（R/B 、U/V）交换开关 0-不交换 1-交换。 注意：pwc-net输入格式是BGR,需要交换RB通道!!!,但是预处理的时候用的就是cv::Mat的BGR，所以这里不需要交换了
            CHECK_ACL(aclmdlSetAIPPSrcImageSize(aipp_param_tensor, input_w, input_h)); // 输入图像尺寸
            /*
            设置 AIPP（图像预处理引擎）中的缩放参数 
            int8_t scfSwitch: 缩放开关。非零值-启用缩放；0-禁用缩放。
            int32_t scfInputSizeW: 输入图像的宽度，用于缩放计算。
            int32_t scfInputSizeH: 输入图像的高度，用于缩放计算。
            int32_t scfOutputSizeW: 输出图像的目标宽度。
            int32_t scfOutputSizeH: 输出图像的目标高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPScfParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的裁剪参数
            int8_t cropSwitch: 裁剪开关。非零值-启用裁剪 0-禁用裁剪。
            int32_t cropStartPosW: 裁剪区域的起始水平位置（X 坐标）。
            int32_t cropStartPosH: 裁剪区域的起始垂直位置（Y 坐标）。
            int32_t cropSizeW: 裁剪区域的宽度。
            int32_t cropSizeH: 裁剪区域的高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPCropParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的填充参数
            int8_t paddingSwitch: 填充开关。非零值-启用填充；0-禁用填充。
            int32_t paddingSizeTop: 顶部填充的大小。
            int32_t paddingSizeBottom: 底部填充的大小。
            int32_t paddingSizeLeft: 左侧填充的大小。
            int32_t paddingSizeRight: 右侧填充的大小。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPPaddingParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            图像预处理归一化操作过程如下：
            pixel_out_chx(i)=[pixel_in_chx(i)-mean_chn_i-min_chn_i]*var_reci_chn_i
            */
            for(int idx = 0; idx < batch_size; idx++){
                float dtcPixelMeanChni0 = 0.0 * 255.0;
                float dtcPixelMeanChni1 = 0.0 * 255.0;
                float dtcPixelMeanChni2 = 0.0 * 255.0;
                CHECK_ACL(aclmdlSetAIPPDtcPixelMean(aipp_param_tensor, dtcPixelMeanChni0, dtcPixelMeanChni1, dtcPixelMeanChni2, 0, idx));
            }
            for(int idx = 0; idx < batch_size; idx++){
                CHECK_ACL(aclmdlSetAIPPDtcPixelMin(aipp_param_tensor, 0.0, 0.0, 0.0, 0.0, idx));
            }
            for(int idx = 0; idx < batch_size; idx++){
                float dtcPixelVarReciChn0 = 1.0 / 255.0;
                float dtcPixelVarReciChn1 = 1.0 / 255.0;
                float dtcPixelVarReciChn2 = 1.0 / 255.0;
                CHECK_ACL(aclmdlSetAIPPPixelVarReci(aipp_param_tensor, dtcPixelVarReciChn0, dtcPixelVarReciChn1, dtcPixelVarReciChn2, 1.0, idx));
            }
            if(i == aipp_index_1){
                CHECK_ACL(aclmdlSetInputAIPP(model_id, input, aipp_index_1, aipp_param_tensor));
            }
            else{
                CHECK_ACL(aclmdlSetInputAIPP(model_id, input, aipp_index_2, aipp_param_tensor));
            }
            CHECK_ACL(aclmdlDestroyAIPP(aipp_param_tensor));
        }
        else if(i == dynamic_batch_idx){
            data_buf = aclCreateDataBuffer(input_batch, buf_size);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
        }
        else{
            std::cout << "input error" << std::endl;
        }
    }
    if(media_type == std::string("picture"))
        PictureInfer(stream, model_id, channel_id_resize,
                input, data_buf_inputs, input_addr_imgs,
                output, output_buf,
                dynamic_batch_idx,
                output_flow, input_h,  input_w, media_path);
    else if(media_type == std::string("video"))
        VideoInfer(stream, model_id, channel_id_resize,
                input, data_buf_inputs, input_addr_imgs,
                output, output_buf,
                dynamic_batch_idx,
                output_flow, input_h,  input_w, media_path);
    else
        std::cout << "must be picture or video" << std::endl;

    CHECK_ACL(aclmdlUnload(model_id));
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); ++i) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(output, i);
        void* data = aclGetDataBufferAddr(data_buffer);
        CHECK_ACL(aclrtFree(data));
        CHECK_ACL(aclDestroyDataBuffer(data_buffer));
        data_buffer = nullptr;
    }
    CHECK_ACL(aclmdlDestroyDataset(output));
    CHECK_DVPP_MPI(hi_mpi_dvpp_free(input_addr_imgs[0]));
    CHECK_DVPP_MPI(hi_mpi_dvpp_free(input_addr_imgs[1]));
    CHECK_ACL(aclrtFree(input_AIPPs[0]));
    CHECK_ACL(aclrtFree(input_AIPPs[1]));
    CHECK_ACL(aclrtFree(input_batch));
    CHECK_ACL(aclmdlDestroyDesc(model_desc));
    CHECK_DVPP_MPI(hi_mpi_vpc_destroy_chn(channel_id_resize));
    delete []output_flow;
    return 0;
}