#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#if 1
namespace {

// ============================================================
// Rearrange
//
// input : [B, C, H, W]
// output: [B, H+8, W+8, C]
//
// input1 and input2 are rearranged in one kernel to reduce
// kernel launch overhead.
// ============================================================

__global__ void kernel_Correlation_rearrange(
    int n,
    const float* input1,
    const float* input2,
    float* output1,
    float* output2,
    int channels,
    int height,
    int width)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    const int batch = blockIdx.z;
    const int channel = blockIdx.y;

    const int inputOffset =
        ((batch * channels + channel) * height * width) + index;

    const float value1 = input1[inputOffset];
    const float value2 = input2[inputOffset];

    const int paddedY = index / width + 4;
    const int paddedX = index % width + 4;

    const int paddedWidth = width + 8;

    const int rearrange =
        paddedY * paddedWidth + paddedX;

    const int outputOffset =
        ((batch * (height + 8) * (width + 8) + rearrange)
         * channels)
        + channel;

    output1[outputOffset] = value1;
    output2[outputOffset] = value2;
}


// ============================================================
// Correlation
//
// One block processes one spatial position.
//
// 3 warps are used:
//
//   warp 0 -> output channels  0 ~ 26
//   warp 1 -> output channels 27 ~ 53
//   warp 2 -> output channels 54 ~ 80
//
// Each warp computes one correlation channel at a time.
//
// This removes the global block-wide synchronization used by
// the original implementation and uses warp shuffle reduction.
// ============================================================

__global__ void kernel_Correlation_updateOutput(
    const float* __restrict__ rbot0,
    const float* __restrict__ rbot1,
    float* __restrict__ output,
    int channels,
    int height,
    int width)
{
    extern __shared__ float patchData[];

    const int thread = threadIdx.x;

    const int warpId = thread >> 5;
    const int laneId = thread & 31;

    const int x = blockIdx.x;
    const int y = blockIdx.y;
    const int batch = blockIdx.z;

    const int paddedWidth = width + 8;

    const int x1 = x + 4;
    const int y1 = y + 4;

    // --------------------------------------------------------
    // Load center pixel from rbot0 into shared memory.
    //
    // All three warps cooperate on this load.
    // --------------------------------------------------------

    const int centerBase =
        ((batch * (height + 8) + y1) * paddedWidth + x1)
        * channels;

    for (int channel = thread;
         channel < channels;
         channel += blockDim.x)
    {
        patchData[channel] = rbot0[centerBase + channel];
    }

    __syncthreads();

    // --------------------------------------------------------
    // 81 correlation channels.
    //
    // 3 warps work independently.
    // Each warp handles 27 output channels.
    // --------------------------------------------------------

    const int firstOutputChannel = warpId * 27;

    for (int k = 0; k < 27; ++k) {

        const int outputChannel =
            firstOutputChannel + k;

        const int dx =
            outputChannel % 9 - 4;

        const int dy =
            outputChannel / 9 - 4;

        const int x2 = x1 + dx;
        const int y2 = y1 + dy;

        const int rbot1Base =
            ((batch * (height + 8) + y2) * paddedWidth + x2)
            * channels;

        // ----------------------------------------------------
        // Each lane processes:
        //
        // lane
        // lane + 32
        // lane + 64
        // ...
        //
        // Same channel accumulation order as before inside
        // each lane.
        // ----------------------------------------------------

        float value = 0.0f;

        for (int channel = laneId;
             channel < channels;
             channel += 32)
        {
            value +=
                patchData[channel] *
                rbot1[rbot1Base + channel];
        }

        // ----------------------------------------------------
        // Warp reduction.
        //
        // No shared-memory reduction and no block-wide
        // synchronization are required.
        // ----------------------------------------------------

        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value += __shfl_down_sync(0xffffffff, value, 2);
        value += __shfl_down_sync(0xffffffff, value, 1);

        if (laneId == 0) {

            const int outputIndex =
                batch * 81 * height * width
                + outputChannel * height * width
                + y * width
                + x;

            output[outputIndex] =
                value / static_cast<float>(channels);
        }
    }
}

} // namespace


// ============================================================
// Launch
// ============================================================

void launch_correlation(
    const float* input1,
    const float* input2,
    float* output,
    void* workspace,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
    const int spatialSize = height * width;

    const size_t rbotElements =
        static_cast<size_t>(batch) *
        (height + 8) *
        (width + 8) *
        channels;

    const size_t rbotBytes =
        rbotElements * sizeof(float);

    float* rbot0 =
        static_cast<float*>(workspace);

    float* rbot1 =
        rbot0 + rbotElements;

    // --------------------------------------------------------
    // Original PWC-Net uses new_zeros() for both tensors.
    //
    // TensorRT workspace is not guaranteed to be initialized,
    // therefore padding must be explicitly zeroed.
    // --------------------------------------------------------

    cudaMemsetAsync(
        rbot0,
        0,
        rbotBytes,
        stream);

    cudaMemsetAsync(
        rbot1,
        0,
        rbotBytes,
        stream);


    // --------------------------------------------------------
    // Rearrange input1 + input2 together.
    // --------------------------------------------------------

    {
        const dim3 block(32, 1, 1);

        const dim3 grid(
            (spatialSize + block.x - 1) / block.x,
            channels,
            batch);

        kernel_Correlation_rearrange<<<
            grid,
            block,
            0,
            stream>>>(
                spatialSize,
                input1,
                input2,
                rbot0,
                rbot1,
                channels,
                height,
                width);
    }


    // --------------------------------------------------------
    // Calculate 9 x 9 correlation.
    //
    // 3 warps = 96 threads.
    // --------------------------------------------------------

    {
        const dim3 block(96, 1, 1);

        const dim3 grid(
            width,
            height,
            batch);

        const size_t sharedMemorySize =
            static_cast<size_t>(channels) *
            sizeof(float);

        kernel_Correlation_updateOutput<<<
            grid,
            block,
            sharedMemorySize,
            stream>>>(
                rbot0,
                rbot1,
                output,
                channels,
                height,
                width);
    }
}
#endif
namespace {

__global__ void kernel_Backwarp(
    const float* __restrict__ input,
    const float* __restrict__ flow,
    float* __restrict__ output,
    int channels,
    int height,
    int width)
{
    const int spatialSize =
        height * width;

    const int elementsPerBatch =
        channels * spatialSize;

    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= elementsPerBatch) {
        return;
    }

    const int batch =
        blockIdx.z;

    const int channel =
        index / spatialSize;

    const int spatialIndex =
        index % spatialSize;

    const int y =
        spatialIndex / width;

    const int x =
        spatialIndex % width;


    // --------------------------------------------------------
    // Equivalent to backwarp_v2:
    //
    // hor = linspace(-1, 1, W)
    // ver = linspace(-1, 1, H)
    // --------------------------------------------------------

    const float hor =
        -1.0f +
        static_cast<float>(x) *
        (2.0f / static_cast<float>(width - 1));

    const float ver =
        -1.0f +
        static_cast<float>(y) *
        (2.0f / static_cast<float>(height - 1));


    // --------------------------------------------------------
    // Flow
    //
    // flow layout:
    // [B, 2, H, W]
    // --------------------------------------------------------

    const int flowOffset =
        batch * 2 * spatialSize +
        spatialIndex;

    const float flowX =
        flow[flowOffset];

    const float flowY =
        flow[flowOffset + spatialSize];


    // --------------------------------------------------------
    // Normalize flow
    //
    // backwarp_v2:
    //
    // flow[:, 0:1] * (2 / (W - 1))
    // flow[:, 1:2] * (2 / (H - 1))
    // --------------------------------------------------------

    const float normalizedFlowX =
        flowX *
        (2.0f / static_cast<float>(width - 1));

    const float normalizedFlowY =
        flowY *
        (2.0f / static_cast<float>(height - 1));


    // --------------------------------------------------------
    // Add flow to normalized grid
    // --------------------------------------------------------

    const float gridX =
        hor + normalizedFlowX;

    const float gridY =
        ver + normalizedFlowY;


    // --------------------------------------------------------
    // normalized -> pixel coordinates
    //
    // backwarp_v2:
    //
    // x = (gridX + 1) * ((W - 1) / 2)
    // y = (gridY + 1) * ((H - 1) / 2)
    // --------------------------------------------------------

    const float srcX =
        (gridX + 1.0f) *
        (static_cast<float>(width - 1) / 2.0f);

    const float srcY =
        (gridY + 1.0f) *
        (static_cast<float>(height - 1) / 2.0f);


    // --------------------------------------------------------
    // floor
    // --------------------------------------------------------

    const float x0f =
        floorf(srcX);

    const float y0f =
        floorf(srcY);

    const float x1f =
        x0f + 1.0f;

    const float y1f =
        y0f + 1.0f;


    // --------------------------------------------------------
    // Bilinear weights
    // --------------------------------------------------------

    const float wx =
        srcX - x0f;

    const float wy =
        srcY - y0f;


    // --------------------------------------------------------
    // Neighbor validity
    //
    // Exactly follows backwarp_v2.
    // --------------------------------------------------------

    const bool valid00 =
        x0f >= 0.0f &&
        x0f < static_cast<float>(width) &&
        y0f >= 0.0f &&
        y0f < static_cast<float>(height);

    const bool valid01 =
        x1f >= 0.0f &&
        x1f < static_cast<float>(width) &&
        y0f >= 0.0f &&
        y0f < static_cast<float>(height);

    const bool valid10 =
        x0f >= 0.0f &&
        x0f < static_cast<float>(width) &&
        y1f >= 0.0f &&
        y1f < static_cast<float>(height);

    const bool valid11 =
        x1f >= 0.0f &&
        x1f < static_cast<float>(width) &&
        y1f >= 0.0f &&
        y1f < static_cast<float>(height);


    // --------------------------------------------------------
    // Clamp indices for safe memory access
    // --------------------------------------------------------

    int x0 =
        static_cast<int>(x0f);

    int x1 =
        static_cast<int>(x1f);

    int y0 =
        static_cast<int>(y0f);

    int y1 =
        static_cast<int>(y1f);

    x0 = max(0, min(width - 1, x0));
    x1 = max(0, min(width - 1, x1));

    y0 = max(0, min(height - 1, y0));
    y1 = max(0, min(height - 1, y1));


    // --------------------------------------------------------
    // Input layout:
    //
    // [B, C, H, W]
    // --------------------------------------------------------

    const size_t batchOffset =
        static_cast<size_t>(batch) *
        channels *
        spatialSize;

    const size_t channelOffset =
        static_cast<size_t>(channel) *
        spatialSize;


    const size_t index00 =
        batchOffset +
        channelOffset +
        static_cast<size_t>(y0) * width +
        x0;

    const size_t index01 =
        batchOffset +
        channelOffset +
        static_cast<size_t>(y0) * width +
        x1;

    const size_t index10 =
        batchOffset +
        channelOffset +
        static_cast<size_t>(y1) * width +
        x0;

    const size_t index11 =
        batchOffset +
        channelOffset +
        static_cast<size_t>(y1) * width +
        x1;


    // --------------------------------------------------------
    // Gather
    //
    // Invalid neighbors contribute zero.
    // --------------------------------------------------------

    const float v00 =
        valid00 ? input[index00] : 0.0f;

    const float v01 =
        valid01 ? input[index01] : 0.0f;

    const float v10 =
        valid10 ? input[index10] : 0.0f;

    const float v11 =
        valid11 ? input[index11] : 0.0f;


    // --------------------------------------------------------
    // Bilinear interpolation
    // --------------------------------------------------------

    const float outputValue =
        v00 * (1.0f - wx) * (1.0f - wy)
        + v01 * wx * (1.0f - wy)
        + v10 * (1.0f - wx) * wy
        + v11 * wx * wy;


    // --------------------------------------------------------
    // Same mask calculation as backwarp_v2
    // --------------------------------------------------------

    const float maskOutput =
        (valid00 ? 1.0f : 0.0f)
            * (1.0f - wx)
            * (1.0f - wy)
        + (valid01 ? 1.0f : 0.0f)
            * wx
            * (1.0f - wy)
        + (valid10 ? 1.0f : 0.0f)
            * (1.0f - wx)
            * wy
        + (valid11 ? 1.0f : 0.0f)
            * wx
            * wy;


    // --------------------------------------------------------
    // backwarp_v2:
    //
    // tenMask = (mask_output > 0.999).to(dtype)
    // --------------------------------------------------------

    const float mask =
        maskOutput > 0.999f
            ? 1.0f
            : 0.0f;


    output[
        batchOffset +
        channelOffset +
        static_cast<size_t>(spatialIndex)
    ] =
        outputValue * mask;
}

} // namespace


void launch_backwarp(
    const float* input,
    const float* flow,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
    const int spatialSize =
        height * width;

    const int elementsPerBatch =
        channels * spatialSize;

    constexpr int blockSize = 256;

    const int gridX =
        (elementsPerBatch + blockSize - 1)
        / blockSize;

    const dim3 block(
        blockSize,
        1,
        1);

    const dim3 grid(
        gridX,
        1,
        batch);

    kernel_Backwarp<<<
        grid,
        block,
        0,
        stream>>>(
            input,
            flow,
            output,
            channels,
            height,
            width);
}