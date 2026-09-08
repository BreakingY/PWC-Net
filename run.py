# Copyright (C) 2026 sunkx
# Licensed under the GNU General Public License v3.0
#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch

'''
修改:去掉自定义算子
'''
# try:
#     from .correlation import correlation # the custom cost volume layer
# except:
#     sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'default' # 'default', or 'chairs-things'
args_strOne = './images/one.png'
args_strTwo = './images/two.png'
args_strOut = './out.flo'

# 后端：
# trt104
# trt85
# trt84
# cann 
args_strBackend = 'trt104'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'one=',
    'two=',
    'out=',
    'backend=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
    if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
    if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
    if strOption == '--backend' and strArg != '': args_strBackend = strArg
VALID_BACKENDS = {'trt104', 'trt85', 'trt84', 'cann'}

if args_strBackend not in VALID_BACKENDS:
    raise ValueError(
        f'Unsupported backend: {args_strBackend}. '
        f'Available backends: {sorted(VALID_BACKENDS)}'
    )

print(f'[INFO] Backend: {args_strBackend}')
def print_backend_info():

    if args_strBackend in ('trt104', 'trt85'):
        correlation_version = 'v1'
        backwarp_version = 'v1'

    elif args_strBackend == 'trt84':
        correlation_version = 'v2'
        backwarp_version = 'v2'

    elif args_strBackend == 'cann':
            correlation_version = 'v3'
            backwarp_version = 'v3'

    else:
        raise RuntimeError(f'Unsupported backend: {args_strBackend}')

    print('=' * 60)
    print('PWC-Net backend configuration')
    print('=' * 60)
    print(f'Backend            : {args_strBackend}')
    print(f'torch_correlation  : {correlation_version}')
    print(f'backwarp           : {backwarp_version}')
    print('=' * 60)
print_backend_info()
# end
##########################################################
'''
修改:使用torch算子实现
'''
import torch.nn.functional as F
# v1 tensorrt推荐使用v1 适配TensorRT10.4和TensorRT8.5
def torch_correlation_v1(tenOne, tenTwo):
    B, C, H, W = tenOne.shape

    # 1. pad to keep spatial alignment
    tenTwo = F.pad(tenTwo, (4, 4, 4, 4))  # [B, C, H+8, W+8]

    # 2. unfold patches
    patches = F.unfold(tenTwo, kernel_size=9, padding=0, stride=1)
    # [B, C*81, H*W]

    # 3. reshape safely
    patches = patches.view(B, C, 81, H * W)
    tenOne_flat = tenOne.view(B, C, H * W)

    # 4. correlation (dot over channel)
    corr = (tenOne_flat.unsqueeze(2) * patches).mean(dim=1)

    # 5. reshape back
    corr = corr.view(B, 81, H, W)

    return corr
# v2 适配TensorRT8.4(Jetson)
def torch_correlation_v2(tenOne, tenTwo):
    """
    不使用 F.pad / F.unfold 的实现。

    保持与原始 torch_correlation 完全一致：

        channel 顺序:
            dy = -4 ... +4
            dx = -4 ... +4

        对每一个输出位置:

            shifted[y, x] = tenTwo[y + dy, x + dx]

        越界部分补 0。

    输入:
        tenOne: [B, C, H, W]
        tenTwo: [B, C, H, W]

    输出:
        [B, 81, H, W]
    """

    B, C, H, W = tenOne.shape

    outputs = []

    # -------------------------------------------------------------------------
    # F.unfold 的 81 个 channel 顺序：
    #
    # dy = -4 ... +4
    # dx = -4 ... +4
    #
    # channel = (dy + 4) * 9 + (dx + 4)
    # -------------------------------------------------------------------------

    for dy in range(-4, 5):

        # =====================================================================
        # Y 方向 shift
        #
        # 目标：
        #
        #     shifted_y[y, x] = tenTwo[y + dy, x]
        #
        # 越界补 0
        # =====================================================================

        if dy > 0:

            # 例如 dy = +1:
            #
            # shifted:
            #
            #   tenTwo[1]
            #   tenTwo[2]
            #   ...
            #   tenTwo[H-1]
            #   0

            shifted_y = torch.cat([tenTwo[:, :, dy:, :],torch.zeros_like(tenTwo[:, :, :dy, :])],dim=2)

        elif dy < 0:

            # 例如 dy = -1:
            #
            # shifted:
            #
            #   0
            #   tenTwo[0]
            #   tenTwo[1]
            #   ...
            #   tenTwo[H-2]

            k = -dy

            shifted_y = torch.cat([torch.zeros_like(tenTwo[:, :, :k, :]),tenTwo[:, :, :H-k, :]],dim=2)
        else:
            shifted_y = tenTwo

        # =====================================================================
        # X 方向 shift
        # =====================================================================

        for dx in range(-4, 5):

            # -----------------------------------------------------------------
            # 目标：
            #
            # shifted[y, x] = shifted_y[y, x + dx]
            #
            # 越界补 0
            # -----------------------------------------------------------------

            if dx > 0:

                # 例如 dx = +1:
                #
                # shifted:
                #
                #   tenTwo[:, 1:]
                #   0

                shifted = torch.cat([shifted_y[:, :, :, dx:],torch.zeros_like(shifted_y[:, :, :, :dx])],dim=3)

            elif dx < 0:

                # 例如 dx = -1:
                #
                # shifted:
                #
                #   0
                #   tenTwo[:, :-1]

                k = -dx

                shifted = torch.cat([torch.zeros_like(shifted_y[:, :, :, :k]),shifted_y[:, :, :, :W-k]],dim=3)

            else:

                shifted = shifted_y

            # -----------------------------------------------------------------
            # correlation
            # -----------------------------------------------------------------

            corr = (tenOne * shifted).mean(dim=1)

            outputs.append(corr)

    # [81, B, H, W]
    #
    # stack(dim=1):
    #
    # [B, 81, H, W]
    return torch.stack(outputs,dim=1)
# v3 针对晟腾优化
def torch_correlation_v3(tenOne, tenTwo):
    B, C, H, W = tenOne.shape

    # pad once
    tenTwo = F.pad(tenTwo, (4, 4, 4, 4))  # [B, C, H+8, W+8]

    corr = tenOne.new_zeros(B, 81, H, W)

    idx = 0
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            shifted = tenTwo[:, :, 
                             (4 + dy):(4 + dy + H),
                             (4 + dx):(4 + dx + W)]
            
            # same computation as unfold version
            corr[:, idx] = (tenOne * shifted).mean(dim=1)
            idx += 1

    return corr


def torch_correlation(tenOne, tenTwo):

    if args_strBackend in ('trt104', 'trt85'):
        return torch_correlation_v1(tenOne,tenTwo)

    elif args_strBackend == 'trt84':
        return torch_correlation_v2(tenOne,tenTwo)

    elif args_strBackend == 'cann':
        return torch_correlation_v3(tenOne,tenTwo)

    else:
        raise RuntimeError(f'Unsupported backend: {args_strBackend}')
##########################################################
"""
# v0:原项目代码
backwarp_tenGrid = {}
backwarp_tenPartial = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask
"""
'''
修改:导出onnx
'''
# v1 tensorrt推荐使用v1 适配TensorRT10.4和TensorRT8.5
def backwarp_v1(tenInput, tenFlow):
    B, C, H, W = tenInput.shape
    _, _, Hf, Wf = tenFlow.shape

    device = tenInput.device
    dtype = tenInput.dtype

    hor = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, -1, H, -1)

    ver = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, -1, -1, W)

    grid = torch.cat([hor, ver], 1)

    flow = torch.cat([tenFlow[:, 0:1] * (2.0 / (W - 1.0)), tenFlow[:, 1:2] * (2.0 / (H - 1.0))], 1)


    mask = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    tenInput = torch.cat([tenInput, mask], 1)

    output = F.grid_sample(tenInput, (grid + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    tenMask = output[:, -1:, :, :]
    tenMask = (tenMask > 0.999).to(dtype)

    return output[:, :-1, :, :] * tenMask
# v2 适配TensorRT8.4
def backwarp_v2(tenInput, tenFlow):
    B, C, H, W = tenInput.shape
    _, _, Hf, Wf = tenFlow.shape

    assert H == Hf and W == Wf

    device = tenInput.device
    dtype = tenInput.dtype

    # -------------------------------------------------------------------------
    # 1. 构造和 backwarp_v1 完全一致的 normalized grid
    # -------------------------------------------------------------------------

    hor = torch.linspace(-1.0,1.0,W,device=device,dtype=dtype).view(1, 1, 1, W).expand(B, -1, H, -1)

    ver = torch.linspace(-1.0,1.0,H,device=device,dtype=dtype).view(1, 1, H, 1).expand(B, -1, -1, W)

    grid = torch.cat([hor, ver], 1)

    # -------------------------------------------------------------------------
    # 2. 和 v1 完全一致的 flow normalized 坐标
    # -------------------------------------------------------------------------

    flow = torch.cat([tenFlow[:, 0:1] * (2.0 / (W - 1.0)),tenFlow[:, 1:2] * (2.0 / (H - 1.0))], 1)

    grid = grid + flow

    # -------------------------------------------------------------------------
    # 3. normalized coordinates -> pixel coordinates
    #
    # align_corners=True:
    #
    # x_pixel = (x_normalized + 1) * (W - 1) / 2
    # y_pixel = (y_normalized + 1) * (H - 1) / 2
    # -------------------------------------------------------------------------

    x = ((grid[:, 0:1] + 1.0)* ((W - 1.0) / 2.0))

    y = ((grid[:, 1:2] + 1.0)* ((H - 1.0) / 2.0))

    # -------------------------------------------------------------------------
    # 4. 双线性插值四个邻居
    # -------------------------------------------------------------------------

    x0 = torch.floor(x)
    y0 = torch.floor(y)

    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # interpolation weight
    wx = x - x0
    wy = y - y0

    # -------------------------------------------------------------------------
    # 5. 四个邻居分别判断是否有效
    #
    # 这是和之前版本最大的区别。
    #
    # 例如：
    #
    # y = 383.0007
    #
    # y0 = 383       -> 有效
    # y1 = 384       -> 无效
    #
    # grid_sample padding_mode='zeros'
    # 会让 y1 对应的采样值为 0，
    # 但不会让整个采样点直接变成 0。
    # -------------------------------------------------------------------------

    valid00 = ((x0 >= 0.0) &(x0 < float(W)) &(y0 >= 0.0) &(y0 < float(H)))

    valid01 = ((x1 >= 0.0) &(x1 < float(W)) &(y0 >= 0.0) &(y0 < float(H)))

    valid10 = ((x0 >= 0.0) &(x0 < float(W)) &(y1 >= 0.0) &(y1 < float(H)))

    valid11 = ((x1 >= 0.0) &(x1 < float(W)) &(y1 >= 0.0) &(y1 < float(H)))

    # -------------------------------------------------------------------------
    # 6. 越界坐标 clamp
    #
    # 只是为了让 gather 能够正常访问。
    # 真正是否有效由上面的 valid mask 决定。
    # -------------------------------------------------------------------------

    x0_safe = x0.clamp(0, W - 1).long()
    x1_safe = x1.clamp(0, W - 1).long()

    y0_safe = y0.clamp(0, H - 1).long()
    y1_safe = y1.clamp(0, H - 1).long()

    # -------------------------------------------------------------------------
    # 7. 转换成 flatten index
    #
    # input:
    #   [B, C, H, W]
    #
    # flatten:
    #   [B, C*H*W]
    #
    # index:
    #   y * W + x
    # -------------------------------------------------------------------------

    base = (torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, C, H, W))

    channel = (torch.arange(C, device=device).view(1, C, 1, 1).expand(B, C, H, W))

    # 注意：
    # 这里的 base/channel 只是构造 batch/channel offset。
    #
    # 最终 index 对应：
    #
    # ((b * C + c) * H + y) * W + x

    index00 = (((base * C + channel) * H + y0_safe.expand(B, C, H, W))* W+ x0_safe.expand(B, C, H, W)).long()

    index01 = (((base * C + channel) * H + y0_safe.expand(B, C, H, W))* W+ x1_safe.expand(B, C, H, W)).long()

    index10 = (((base * C + channel) * H + y1_safe.expand(B, C, H, W))* W+ x0_safe.expand(B, C, H, W)).long()

    index11 = (((base * C + channel) * H + y1_safe.expand(B, C, H, W))* W+ x1_safe.expand(B, C, H, W)).long()

    # -------------------------------------------------------------------------
    # 8. flatten input
    # -------------------------------------------------------------------------

    input_flat = tenInput.reshape(-1)

    # -------------------------------------------------------------------------
    # 9. gather 四个邻居
    # -------------------------------------------------------------------------

    v00 = torch.gather(input_flat,0,index00.reshape(-1)).reshape(B, C, H, W)

    v01 = torch.gather(input_flat,0,index01.reshape(-1)).reshape(B, C, H, W)

    v10 = torch.gather(input_flat,0,index10.reshape(-1)).reshape(B, C, H, W)

    v11 = torch.gather(input_flat,0,index11.reshape(-1)).reshape(B, C, H, W)

    # -------------------------------------------------------------------------
    # 10. 越界邻居按照 padding_mode='zeros' 处理
    # -------------------------------------------------------------------------

    v00 = v00 * valid00.to(dtype)
    v01 = v01 * valid01.to(dtype)
    v10 = v10 * valid10.to(dtype)
    v11 = v11 * valid11.to(dtype)

    # -------------------------------------------------------------------------
    # 11. 双线性插值
    # -------------------------------------------------------------------------

    output = (v00 * (1.0 - wx) * (1.0 - wy)+ v01 * wx * (1.0 - wy)+ v10 * (1.0 - wx) * wy+ v11 * wx * wy)

    # -------------------------------------------------------------------------
    # 12. 和 v1 一样构造 mask
    #
    # 注意：
    # 这里不能使用简单的坐标 valid mask。
    # v1 是先对 mask 做 grid_sample，
    # 然后：
    #
    # tenMask = (tenMask > 0.999)
    #
    # 因此这里也需要按照同样的方式计算 mask。
    # -------------------------------------------------------------------------

    mask = torch.ones((B, 1, H, W),device=device,dtype=dtype)

    # mask 是全 1，因此只需要计算它对应的双线性插值。
    mask_output = (valid00.to(dtype) * (1.0 - wx) * (1.0 - wy)+ valid01.to(dtype) * wx * (1.0 - wy)+ valid10.to(dtype) * (1.0 - wx) * wy+ valid11.to(dtype) * wx * wy)

    tenMask = (mask_output > 0.999).to(dtype)

    return output * tenMask
# v3 针对晟腾优化
def backwarp_v3(tenInput, tenFlow):
    B, C, H, W = tenInput.shape
    device = tenInput.device
    dtype = tenInput.dtype

    hor = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ver = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(ver, hor, indexing='ij')

    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    flow = torch.cat([
        tenFlow[:, 0:1] * (2.0 / (W - 1.0)),
        tenFlow[:, 1:2] * (2.0 / (H - 1.0))
    ], dim=1)

    grid = grid + flow.permute(0, 2, 3, 1)

    output = F.grid_sample(
        tenInput,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return output


def backwarp(tenInput, tenFlow):
    if args_strBackend in ('trt104', 'trt85'):
        return backwarp_v1(tenInput, tenFlow)

    elif args_strBackend == 'trt84':
        return backwarp_v2(tenInput, tenFlow)
    
    elif args_strBackend == 'cann':
        return backwarp_v3(tenInput, tenFlow)

    else:
        return backwarp_v1(tenInput, tenFlow)

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None
                    '''
                    修改:去掉自定义算子,使用torch算子
                    '''
                    # tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)
                    tenVolume = torch.nn.functional.leaky_relu(input=torch_correlation(tenOne, tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])
                    '''
                    修改:去掉自定义算子,使用torch算子
                    '''
                    # tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)
                    tenVolume = torch.nn.functional.leaky_relu(input=torch_correlation(tenOne, backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end
                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + args_strModel + '.pytorch', file_name='pwc-' + args_strModel).items() })
        state_dict = torch.load('./network-' + args_strModel + '.pytorch', map_location='cpu')
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict.items()})
    # end

    def forward(self, tenOne, tenTwo):
        intHeight, intWidth = tenOne.shape[2], tenOne.shape[3]
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        # return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
        '''
        修复: flow尺寸
        '''
        flow = (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
        flow = torch.nn.functional.interpolate(flow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        scale_w = float(intWidth) / float(flow.shape[3])
        scale_h = float(intHeight) / float(flow.shape[2])
        flow = torch.stack([flow[:, 0] * scale_w, flow[:, 1] * scale_h], dim=1)
        return flow
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if netNetwork is None:
        # netNetwork = Network().cuda().train(False)
        netNetwork = Network().to(device).train(False)
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    # tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    # tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth).to(device)
    tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth).to(device)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()
# end
##########################################################
class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tenOne, tenTwo):
        flow = self.model(tenOne, tenTwo)
        return flow


def export_onnx(weight_path='./network-default.pytorch', onnx_path='./pwcnet.onnx', model_type='default', opset=17):
    """
    Export PWC-style model to ONNX with dynamic batch support
    """

    model = Network()
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict({
        k.replace('module', 'net'): v
        for k, v in state_dict.items()
    })
    model.eval()
    model.cpu()

    wrapper = OnnxWrapper(model)
    wrapper.eval()


    dummy_one = torch.randn(1, 3, 384, 768, dtype=torch.float32)
    dummy_two = torch.randn(1, 3, 384, 768, dtype=torch.float32)

    dynamic_axes = {
        'input1': {0: 'batch'},
        'input2': {0: 'batch'},
        'output': {0: 'batch'}
    }

    torch.onnx.export(
        wrapper,
        (dummy_one, dummy_two),
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input1', 'input2'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False,
        keep_initializers_as_inputs=False
    )

    print(f"[OK] ONNX exported to: {onnx_path}")

if __name__ == '__main__':
    # BGR + CHW + float + 归一化
    tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenOne, tenTwo)

    objOutput = open(args_strOut, 'wb')

    numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objOutput)
    numpy.array([ tenOutput.shape[2], tenOutput.shape[1] ], numpy.int32).tofile(objOutput)
    numpy.array(tenOutput.numpy(force=True).transpose(1, 2, 0), numpy.float32).tofile(objOutput)

    objOutput.close()
    onnx_path = f'./pwcnet_{args_strBackend}.onnx'
    export_onnx(weight_path='./network-default.pytorch', onnx_path=onnx_path)
# end