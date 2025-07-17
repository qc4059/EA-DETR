# import torch
# import torch.nn as nn

# # Polar Coordinate Circular Dynamic Snake Convolution极坐标环形动态蛇形卷积
# __all__ = ['C3k2_PCCDSConv']


# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))


# class DySnakeConv(nn.Module):
#     def __init__(self, inc, ouc, k=3) -> None:
#         super().__init__()

#         self.conv_0 = Conv(inc, ouc, k)
#         self.conv_x = DSConv(inc, ouc, 0, k)
#         self.conv_y = DSConv(inc, ouc, 1, k)

#     def forward(self, x):
#         return torch.cat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], dim=1)


# class DSConv(nn.Module):
#     def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True):
#         """
#         The Dynamic Snake Convolution
#         :param in_ch: input channel
#         :param out_ch: output channel
#         :param kernel_size: the size of kernel
#         :param morph: the morphology of the convolution kernel is mainly divided into two types
#                         along the x-axis (0) and the y-axis (1) (see the paper for details)
#         :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
#         """
#         super(DSConv, self).__init__()
#         # use the <offset_conv> to learn the deformable offset
#         self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
#         self.bn = nn.BatchNorm2d(2 * kernel_size)
#         self.kernel_size = kernel_size

#         # two types of the DSConv (along x-axis and y-axis)
#         self.dsc_conv_x = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(kernel_size, 1),
#             stride=(kernel_size, 1),
#             padding=0,
#         )
#         self.dsc_conv_y = nn.Conv2d(
#             in_ch,
#             out_ch,
#             kernel_size=(1, kernel_size),
#             stride=(1, kernel_size),
#             padding=0,
#         )

#         self.gn = nn.GroupNorm(out_ch // 4, out_ch)
#         self.act = Conv.default_act

#         self.morph = morph
#         self.if_offset = if_offset

#         # 可学习参数
#         self.radius_elastic = nn.Parameter(torch.tensor(1.0))
#         self.extend_scope = nn.Parameter(torch.tensor(1.0))

#     def forward(self, f):
#         # 参数约束
#         self.radius_elastic.data = torch.clamp(self.radius_elastic.data, min=0.1, max=10.0)
#         self.extend_scope.data = torch.clamp(self.extend_scope.data, min=0.1, max=10.0)

#         offset = self.offset_conv(f)
#         offset = self.bn(offset)
#         # We need a range of deformation between -1 and 1 to mimic the snake's swing
#         offset = torch.tanh(offset)
#         input_shape = f.shape
#         dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph, self.radius_elastic)
#         deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
#         if self.morph == 0:
#             x = self.dsc_conv_x(deformed_feature.type(f.dtype))
#             x = self.gn(x)
#             x = self.act(x)
#             return x
#         else:
#             x = self.dsc_conv_y(deformed_feature.type(f.dtype))
#             x = self.gn(x)
#             x = self.act(x)
#             return x


# # Core code, for ease of understanding, we mark the dimensions of input and output next to the code
# class DSC(object):
#     def __init__(self, input_shape, kernel_size, extend_scope, morph, radius_elastic):
#         self.num_points = kernel_size
#         self.width = input_shape[2]
#         self.height = input_shape[3]
#         self.morph = morph
#         self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
#         self.radius_elastic = radius_elastic

#         # define feature map shape
#         """
#         B: Batch size  C: Channel  W: Width  H: Height
#         """
#         self.num_batch = input_shape[0]
#         self.num_channels = input_shape[1]

#     """
#     input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
#     output_x: [B,1,W,K*H]   coordinate map
#     output_y: [B,1,K*W,H]   coordinate map
#     """

#     def _coordinate_map_3D(self, offset, if_offset):
#         device = offset.device
#         # offset
#         y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

#         y_center = torch.arange(0, self.width).repeat([self.height])
#         y_center = y_center.reshape(self.height, self.width)
#         y_center = y_center.permute(1, 0)
#         y_center = y_center.reshape([-1, self.width, self.height])
#         y_center = y_center.repeat([self.num_points, 1, 1]).float()
#         y_center = y_center.unsqueeze(0)

#         x_center = torch.arange(0, self.height).repeat([self.width])
#         x_center = x_center.reshape(self.width, self.height)
#         x_center = x_center.permute(0, 1)
#         x_center = x_center.reshape([-1, self.width, self.height])
#         x_center = x_center.repeat([self.num_points, 1, 1]).float()
#         x_center = x_center.unsqueeze(0)

#         if self.morph == 0:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: only need 0
#                 x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
#             """
#             y = torch.linspace(0, 0, 1)
#             x = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )

#             y, x = torch.meshgrid(y, x, indexing='ij')
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(device)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(device)

#             y_offset_new = y_offset.detach().clone()

#             if if_offset:
#                 y_offset = y_offset.permute(1, 0, 2, 3)
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)

#                 # The center position remains unchanged and the rest of the positions begin to swing
#                 # This part is quite simple. The main idea is that "offset is an iterative process"
#                 y_offset_new[center] = 0
#                 for index in range(1, center):
#                     y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
#                     y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
#                 y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
#                 y_new = y_new.add(y_offset_new.mul(self.extend_scope))

#             # Convert to polar coordinates
#             r = torch.sqrt(x_new ** 2 + y_new ** 2)
#             theta = torch.atan2(y_new, x_new)

#             # Radius elastic adjustment
#             r = r * self.radius_elastic

#             # Harmonic angle optimization
#             r = r + torch.sin(4 * theta) * self.extend_scope

#             # Dynamic range adjustment
#             dynamic_scope = torch.mean(torch.abs(offset), dim=(1, 2, 3), keepdim=True)
#             r = r + dynamic_scope * torch.sin(theta) * self.extend_scope

#             # Convert back to Cartesian coordinates
#             x_new = r * torch.cos(theta)
#             y_new = r * torch.sin(theta)

#             y_new = y_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, self.num_points, 1, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, self.num_points * self.width, 1 * self.height
#             ])
#             return y_new, x_new

#         else:
#             """
#             Initialize the kernel and flatten the kernel
#                 y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
#                 x: only need 0
#             """
#             y = torch.linspace(
#                 -int(self.num_points // 2),
#                 int(self.num_points // 2),
#                 int(self.num_points),
#             )
#             x = torch.linspace(0, 0, 1)

#             y, x = torch.meshgrid(y, x, indexing='ij')
#             y_spread = y.reshape(-1, 1)
#             x_spread = x.reshape(-1, 1)

#             y_grid = y_spread.repeat([1, self.width * self.height])
#             y_grid = y_grid.reshape([self.num_points, self.width, self.height])
#             y_grid = y_grid.unsqueeze(0)

#             x_grid = x_spread.repeat([1, self.width * self.height])
#             x_grid = x_grid.reshape([self.num_points, self.width, self.height])
#             x_grid = x_grid.unsqueeze(0)

#             y_new = y_center + y_grid
#             x_new = x_center + x_grid

#             y_new = y_new.repeat(self.num_batch, 1, 1, 1)
#             x_new = x_new.repeat(self.num_batch, 1, 1, 1)

#             y_new = y_new.to(device)
#             x_new = x_new.to(device)
#             x_offset_new = x_offset.detach().clone()

#             if if_offset:
#                 x_offset = x_offset.permute(1, 0, 2, 3)
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3)
#                 center = int(self.num_points // 2)
#                 x_offset_new[center] = 0
#                 for index in range(1, center):
#                     x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
#                     x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
#                 x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
#                 x_new = x_new.add(x_offset_new.mul(self.extend_scope))

#             # Convert to polar coordinates
#             r = torch.sqrt(x_new ** 2 + y_new ** 2)
#             theta = torch.atan2(y_new, x_new)

#             # Radius elastic adjustment
#             r = r * self.radius_elastic

#             # Harmonic angle optimization
#             r = r + torch.sin(4 * theta) * self.extend_scope

#             # Dynamic range adjustment
#             dynamic_scope = torch.mean(torch.abs(offset), dim=(1, 2, 3), keepdim=True)
#             r = r + dynamic_scope * torch.sin(theta) * self.extend_scope

#             # Convert back to Cartesian coordinates
#             x_new = r * torch.cos(theta)
#             y_new = r * torch.sin(theta)

#             y_new = y_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             y_new = y_new.permute(0, 3, 1, 4, 2)
#             y_new = y_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             x_new = x_new.reshape(
#                 [self.num_batch, 1, self.num_points, self.width, self.height])
#             x_new = x_new.permute(0, 3, 1, 4, 2)
#             x_new = x_new.reshape([
#                 self.num_batch, 1 * self.width, self.num_points * self.height
#             ])
#             return y_new, x_new

#     """
#     input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
#     output: [N,1,K*D,K*W,K*H]  deformed feature map
#     """

#     def _bilinear_interpolate_3D(self, input_feature, y, x):
#         device = input_feature.device
#         y = y.reshape([-1]).float()
#         x = x.reshape([-1]).float()

#         zero = torch.zeros([]).int()
#         max_y = self.width - 1
#         max_x = self.height - 1

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y)
#         y1 = torch.clamp(y1, zero, max_y)
#         x0 = torch.clamp(x0, zero, max_x)
#         x1 = torch.clamp(x1, zero, max_x)

#         input_feature_flat = input_feature.flatten()
#         input_feature_flat = input_feature_flat.reshape(
#             self.num_batch, self.num_channels, self.width, self.height)
#         input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
#         input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
#         dimension = self.height * self.width

#         base = torch.arange(self.num_batch) * dimension
#         base = base.reshape([-1, 1]).float()

#         repeat = torch.ones([self.num_points * self.width * self.height
#                              ]).unsqueeze(0)
#         repeat = repeat.float()

#         base = torch.matmul(base, repeat)
#         base = base.reshape([-1])

#         base = base.to(device)

#         base_y0 = base + y0 * self.height
#         base_y1 = base + y1 * self.height

#         # top rectangle of the neighbourhood volume
#         index_a0 = base_y0 - base + x0
#         index_c0 = base_y0 - base + x1

#         # bottom rectangle of the neighbourhood volume
#         index_a1 = base_y1 - base + x0
#         index_c1 = base_y1 - base + x1

#         # get 8 grid values
#         value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
#         value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
#         value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
#         value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

#         # find 8 grid locations
#         y0 = torch.floor(y).int()
#         y1 = y0 + 1
#         x0 = torch.floor(x).int()
#         x1 = x0 + 1

#         # clip out coordinates exceeding feature map volume
#         y0 = torch.clamp(y0, zero, max_y + 1)
#         y1 = torch.clamp(y1, zero, max_y + 1)
#         x0 = torch.clamp(x0, zero, max_x + 1)
#         x1 = torch.clamp(x1, zero, max_x + 1)

#         x0_float = x0.float()
#         x1_float = x1.float()
#         y0_float = y0.float()
#         y1_float = y1.float()

#         vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
#         vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
#         vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
#         vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

#         outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
#                    value_c1 * vol_c1)

#         if self.morph == 0:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 self.num_points * self.width,
#                 1 * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         else:
#             outputs = outputs.reshape([
#                 self.num_batch,
#                 1 * self.width,
#                 self.num_points * self.height,
#                 self.num_channels,
#             ])
#             outputs = outputs.permute(0, 3, 1, 2)
#         return outputs

#     def deform_conv(self, input, offset, if_offset):
#         y, x = self._coordinate_map_3D(offset, if_offset)
#         deformed_feature = self._bilinear_interpolate_3D(input, y, x)
#         return deformed_feature


# class Bottleneck(nn.Module):
#     """Standard bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class Bottleneck_DySnakeConv(Bottleneck):
#     """Standard bottleneck with DySnakeConv."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv2 = DySnakeConv(c_, c2, k[1])
#         self.cv3 = Conv(c2 * 3, c2, k=1)

#     def forward(self, x):
#         """'forward()' applies the YOLOv5 FPN to input data."""
#         return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))


# class C3(nn.Module):
#     """CSP Bottleneck with 3 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# class C3k_DSConv(C3):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# class C3k2_PCCDSConv(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k_DSConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in
#             range(n)
#         )
#     # 在特征提取时用DSConv,在辅助特征融合时换回原先的Bottleneck


# if __name__ == "__main__":
#     # Generating Sample image
#     image_size = (1, 64, 240, 240)
#     image = torch.rand(*image_size)

#     # Model
#     mobilenet_v1 = C3k2_PCCDSConv(64, 64, c3k=True)

#     out = mobilenet_v1(image)
#     print(out.size())


import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warnings
import glob
import os
import cv2
from packaging import version

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics import YOLO, __version__
    print(f"✅ 导入成功, Ultralytics v{__version__}")
except ImportError:
    print("⚠️ 未能从ultralytics导入DetectionValidator，尝试从本地文件导入")

# 版本检查
if version.parse(__version__) < version.parse("8.0.0"):
    raise EnvironmentError("需要 Ultralytics 8.0.0 或更高版本")

def draw_custom(image, boxes, scores, labels, conf_thres):
    """
    自定义绘制函数，使用指定的颜色和字体样式标注检测结果
    
    参数:
        image (PIL.Image): 原始图像对象
        boxes (list): 边界框坐标列表
        scores (list): 置信度分数列表
        labels (list): 类别标签列表
        conf_thres (float): 置信度阈值
    """
    # 定义类别名称映射
    class_names = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor"
    }
    
    # 定义类别颜色映射
    class_colors = {
        0: (255, 0, 0),        # 红色 - pedestrian
        1: (0, 255, 0),        # 绿色 - people
        2: (0, 0, 255),        # 蓝色 - bicycle
        3: (230, 180, 80),     # 黄色 - car
        4: (255, 0, 255),      # 紫色 - van
        5: (0, 255, 255),      # 青色 - truck
        6: (255, 165, 0),      # 橙色 - tricycle
        7: (128, 0, 128),      # 深紫色 - awning-tricycle
        8: (0, 128, 128),     # 蓝绿色 - bus
        9: (128, 128, 0),      # 橄榄色 - motor
    }
    
    # 默认颜色（当类别超出定义范围时使用）
    default_color = (200, 200, 200)  # 灰色
    
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体（Windows系统通常有Arial字体）
    try:
        # 查找系统字体（优先使用Arial）
        font_path = None
        possible_fonts = [
            "arial.ttf", 
            "arialbd.ttf", 
            "ariblk.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf"
        ]
        
        for font_candidate in possible_fonts:
            if os.path.exists(font_candidate):
                font_path = font_candidate
                break
        
        if font_path:
            # 根据图像尺寸动态设置字体大小 - 缩小字体尺寸
            min_dim = min(image.width, image.height)
            # 减少字体大小比例
            font_size = max(10, int(min_dim * 0.02))  # 从0.03减小到0.02
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 尝试使用PIL内置字体
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 按置信度排序，确保高置信度的目标显示在上层
    indices = np.argsort(scores)[::-1]
    
    for idx in indices:
        if scores[idx] < conf_thres:
            continue
            
        b = boxes[idx]
        label_id = int(labels[idx])
        
        # 获取类别名称（如果找不到则显示"Unknown"）
        class_name = class_names.get(label_id, f"Class {label_id}")
        
        # 获取类别颜色
        color = class_colors.get(label_id, default_color)
        
        # 绘制边框 - 减小边框宽度
        draw.rectangle(
            list(b),
            outline=color,
            width=2  # 从3减小到2
        )
        
        # 准备文本（使用类别名称和置信度）
        text = f"{class_name}: {scores[idx]:.2f}"
        
        # 获取文本尺寸
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            # 旧版PIL使用不同方法
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        
        # 确保文本在图像范围内 - 减小边距
        text_x = max(3, min(b[0], image.width - text_width - 3))
        text_y = max(3, min(b[1] - text_height - 3, image.height - text_height - 3))
        
        # 绘制文本背景矩形 - 减小内边距
        bg_rect = [
            text_x - 1,  # 从-2减小到-1
            text_y - 1,  # 从-2减小到-1
            text_x + text_width + 1,  # 从+2减小到+1
            text_y + text_height + 1  # 从+2减小到+1
        ]
        draw.rectangle(
            bg_rect,
            fill=color
        )
        
        # 绘制文本
        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=(255, 255, 255)  # 白色文本
        )
    
    return image

def detect_and_visualize(model_path, image_paths, output_dir="detections", imgsz=640, conf_thres=0.25, device='', 
                         measure_fps=False, fps_image=None, fps_iterations=100):
    """
    加载YOLOv11模型，对图片/视频进行目标检测并可视化结果
    
    参数:
        model_path (str): 模型权重文件路径
        image_paths (list): 待检测图片/视频路径列表
        output_dir (str): 结果保存目录
        imgsz (int): 输入尺寸
        conf_thres (float): 置信度阈值
        device (str): 指定计算设备
        measure_fps (bool): 是否执行专业FPS测量
        fps_image (str): FPS测量用的图像路径
        fps_iterations (int): FPS测量的迭代次数
    """
    # 支持的图像和视频格式
    supported_image_suffix = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
    supported_video_suffix = ['.mp4', '.avi', '.mov', '.mkv', '.gif']
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 加载模型
    try:
        model = YOLO(model_path)
        if device:
            model = model.to(device)
        print(f"✅ 模型加载成功, 设备: {model.device.type}")
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("❌ 显存不足! 尝试减小imgsz值或使用--half")
        else:
            print(f"❌ 模型加载失败: {e}")
        return
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return
    
    # ==============================================
    # 专业FPS测量功能
    # ==============================================
    def measure_fps(model, image, num_iterations=100, warmup_iterations=10):
        """
        执行专业级的FPS测量，包括预热和统计计算
        
        参数:
            model (YOLO): 加载的YOLO模型
            image (np.array): 输入图像(numpy数组)
            num_iterations (int): 测量迭代次数
            warmup_iterations (int): GPU预热迭代次数
            
        返回:
            dict: 包含详细性能指标的字典
        """
        import time
        import numpy as np
        
        # 预热GPU
        print(f"🔥 预热GPU ({warmup_iterations}次迭代)...")
        for _ in range(warmup_iterations):
            model.predict(image, verbose=False)
        
        # 初始化计时器
        total_preprocess = []
        total_inference = []
        total_postprocess = []
        total_fps = []
        
        # 主测量循环
        print(f"📊 开始FPS测量 ({num_iterations}次迭代)...")
        for i in range(num_iterations):
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 执行推理
            results = model.predict(image, verbose=False)
            result = results[0]  # 只取第一个结果
            
            # 计算各阶段时间
            if hasattr(result, 'speed'):
                speed = result.speed
                pre_time = speed.get('preprocess', 0)
                inf_time = speed.get('inference', 0)
                post_time = speed.get('postprocess', 0)
                total_time = pre_time + inf_time + post_time
                
                total_preprocess.append(pre_time)
                total_inference.append(inf_time)
                total_postprocess.append(post_time)
                total_fps.append(1000 / total_time if total_time > 0 else float('inf'))
        
        # 计算统计指标
        stats = {
            'preprocess_avg': np.mean(total_preprocess) if total_preprocess else 0,
            'preprocess_std': np.std(total_preprocess) if total_preprocess else 0,
            'inference_avg': np.mean(total_inference) if total_inference else 0,
            'inference_std': np.std(total_inference) if total_inference else 0,
            'postprocess_avg': np.mean(total_postprocess) if total_postprocess else 0,
            'postprocess_std': np.std(total_postprocess) if total_postprocess else 0,
            'fps_avg': np.mean(total_fps) if total_fps else 0,
            'fps_std': np.std(total_fps) if total_fps else 0,
            'min_latency': min(total_preprocess) + min(total_inference) + min(total_postprocess),
            'max_latency': max(total_preprocess) + max(total_inference) + max(total_postprocess),
            'iterations': num_iterations,
            'warmup': warmup_iterations,
            # 添加范围数据到返回的字典中
            'preprocess_min': min(total_preprocess) if total_preprocess else 0,
            'preprocess_max': max(total_preprocess) if total_preprocess else 0,
            'inference_min': min(total_inference) if total_inference else 0,
            'inference_max': max(total_inference) if total_inference else 0,
            'postprocess_min': min(total_postprocess) if total_postprocess else 0,
            'postprocess_max': max(total_postprocess) if total_postprocess else 0
        }
        
        # 计算总体延迟
        total_latencies = [p + i + po for p, i, po in zip(total_preprocess, total_inference, total_postprocess)]
        stats['latency_avg'] = np.mean(total_latencies)
        stats['latency_std'] = np.std(total_latencies)
        stats['p95_latency'] = np.percentile(total_latencies, 95)
        
        return stats
    
    # 如果启用了FPS测量
    if measure_fps:
        print("\n🔬 开始专业级FPS测量...")
        
        # 准备测试图像
        test_image = None
        if fps_image:
            try:
                img = Image.open(fps_image).convert('RGB')
                test_image = np.array(img)
                print(f"📷 使用指定图像进行FPS测量: {fps_image}")
            except Exception as e:
                print(f"⚠️ 无法加载指定图像: {e}")
                test_image = None
        
        # 如果未指定图像或加载失败，使用默认图像
        if test_image is None:
            # 尝试从输入路径获取第一张图像
            if image_paths:
                try:
                    img = Image.open(image_paths[0]).convert('RGB')
                    test_image = np.array(img)
                    print(f"📷 使用输入路径的第一张图像进行FPS测量")
                except:
                    # 生成随机测试图像
                    test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                    print("📷 使用随机生成的图像进行FPS测量")
            else:
                # 生成随机测试图像
                test_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
                print("📷 使用随机生成的图像进行FPS测量")
        
        # 设置测量参数
        warmup = max(10, fps_iterations // 10)  # 预热次数为总次数的10%
        
        # 执行测量
        fps_stats = measure_fps(model, test_image, fps_iterations, warmup)
        
        # 打印结果
        print("\n" + "="*70)
        print(f"{'FPS Measurement Results':^70}")
        print("="*70)
        print(f"{'Stage':<15}{'Avg (ms)':>10}{'Std Dev':>10}{'Range (ms)':>15}")
        print(f"{'-'*70}")
        
        # 使用fps_stats字典中的值
        preprocess_range = f"{fps_stats['preprocess_min']:.2f}-{fps_stats['preprocess_max']:.2f}"
        inference_range = f"{fps_stats['inference_min']:.2f}-{fps_stats['inference_max']:.2f}"
        postprocess_range = f"{fps_stats['postprocess_min']:.2f}-{fps_stats['postprocess_max']:.2f}"
        total_range = f"{fps_stats['min_latency']:.2f}-{fps_stats['max_latency']:.2f}"
        
        print(f"{'Preprocess':<15}{fps_stats['preprocess_avg']:10.2f}{fps_stats['preprocess_std']:10.2f}{preprocess_range:>15}")
        print(f"{'Inference':<15}{fps_stats['inference_avg']:10.2f}{fps_stats['inference_std']:10.2f}{inference_range:>15}")
        print(f"{'Postprocess':<15}{fps_stats['postprocess_avg']:10.2f}{fps_stats['postprocess_std']:10.2f}{postprocess_range:>15}")
        print(f"{'-'*70}")
        print(f"{'Total Latency':<15}{fps_stats['latency_avg']:10.2f}{fps_stats['latency_std']:10.2f}{total_range:>15}")
        print(f"{'FPS':<15}{fps_stats['fps_avg']:10.2f}{fps_stats['fps_std']:10.2f}")
        print(f"{'95% Latency':<15}{fps_stats['p95_latency']:10.2f}ms")
        print("="*70)
        print(f"测量迭代次数: {fps_iterations} | 预热迭代次数: {warmup}")
        
        # 提前返回，不再执行常规检测
        return
    
    # ==============================================
    # 常规检测和可视化功能
    # ==============================================
    
    # 处理每个输入文件
    valid_paths = []
    for path in image_paths:
        # 处理通配符路径
        expanded_paths = glob.glob(path, recursive=True)
        if not expanded_paths:
            print(f"⚠️ 未找到匹配的文件: {path}")
            continue
            
        for expanded_path in expanded_paths:
            file_path = Path(expanded_path)
            if not file_path.exists():
                print(f"⚠️ 文件不存在: {file_path}")
                continue
                
            suffix = file_path.suffix.lower()
            if suffix in supported_image_suffix or suffix in supported_video_suffix:
                valid_paths.append(str(file_path))
            else:
                print(f"⚠️ 不支持的文件格式: {file_path} (仅支持 {supported_image_suffix + supported_video_suffix})")
    
    if not valid_paths:
        print("❌ 没有有效的文件可处理")
        return
        
    print(f"🔍 开始检测: {len(valid_paths)}个有效文件")
    print(f"🎯 置信度阈值: {conf_thres}")
    
    for i, path in enumerate(valid_paths):
        print(f"\n📄 处理 {i+1}/{len(valid_paths)}: {Path(path).name}")
        
        # 执行推理
        try:
            results = model.predict(
                path, 
                imgsz=imgsz,
                conf=conf_thres,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
            )
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            continue
        
        # 处理每个结果 (视频可能有多个帧)
        for j, result in enumerate(results):
            # 获取原始图像（转换为PIL格式）
            if hasattr(result, 'orig_img'):
                orig_img = result.orig_img
                if orig_img is not None:
                    # 将BGR转为RGB
                    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(orig_img_rgb)
                else:
                    # 尝试直接加载图像
                    try:
                        img_pil = Image.open(path).convert("RGB")
                    except:
                        print(f"⚠️ 无法加载图像: {path}")
                        continue
            else:
                print(f"⚠️ 结果对象没有orig_img属性")
                continue
            
            # 提取检测结果
            boxes = []
            scores = []
            labels = []
            
            if result.boxes is not None and len(result.boxes):
                for bbox, cls, score in zip(result.boxes.xyxy.cpu().numpy(), 
                                           result.boxes.cls.cpu().numpy(), 
                                           result.boxes.conf.cpu().numpy()):
                    if score < conf_thres:
                        continue
                        
                    boxes.append(bbox)
                    scores.append(score)
                    labels.append(cls)
                
                # 打印检测统计
                print(f"📊 检测到目标数: {len(boxes)}")
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                for cls_id, count in zip(unique_labels, counts):
                    cls_id = int(cls_id)
                    # 使用自定义类名
                    class_names = [
                        "pedestrian", "people", "bicycle", "car", "van",
                        "truck", "tricycle", "awning-tricycle", "bus", "motor"
                    ]
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                    print(f"  {class_name}: {count}个")
            else:
                print("⚠️ 未检测到任何目标")
                boxes = []
                scores = []
                labels = []
            
            # 使用自定义绘制函数
            try:
                annotated_img = draw_custom(
                    img_pil.copy(), 
                    boxes, 
                    scores, 
                    labels, 
                    conf_thres
                )
                
                # 生成输出文件名
                stem = Path(path).stem
                suffix = f"_{j}" if len(results) > 1 else ""
                output_file = output_path / f"{stem}{suffix}_detect.jpg"
                
                # 保存可视化结果
                annotated_img.save(output_file)
                print(f"💾 结果保存至: {output_file}")
                
            except Exception as e:
                print(f"❌ 自定义绘制失败: {e}")
                continue
            
            if hasattr(result, 'speed'):
                speed = result.speed
                total_time = speed.get('preprocess', 0) + speed.get('inference', 0) + speed.get('postprocess', 0)
                fps = 1000 / total_time if total_time > 0 else float('inf')
                
                print(f"⚡ 速度 - 预: {speed.get('preprocess', 0):.2f}ms, "
                    f"推: {speed.get('inference', 0):.2f}ms, "
                    f"后: {speed.get('postprocess', 0):.2f}ms, "
                    f"总时间: {total_time:.2f}ms, FPS: {fps:.2f}")
                
    print("\n✅ 所有文件处理完成!")

def validate(model_path, data_yaml, imgsz=640):
    """执行YOLOv11在VisDrone上的验证"""
    args = {
        'model': model_path,   # 模型权重路径
        'data': data_yaml,     # 数据集配置文件
        'batch': 16,           # 与训练一致的batch size
        'imgsz': imgsz,        # 与训练一致的图像尺寸
        'conf': 0.001,         # VisDrone专用：低置信度阈值检测小目标
        'iou': 0.6,            # 适合密集场景的IoU阈值
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'name': 'YOLOv11-VisDrone-Val',
        'exist_ok': True,      # 覆盖已有结果
        'save_json': True,     # 保存COCO格式结果
        'save_txt': True,      # 保存预测结果文本
        'task': 'detect',      # 检测任务
        'agnostic_nms': True,  # VisDrone专用：类别无关的NMS
        'max_det': 300,        # VisDrone专用：每张图最多检测目标数
        'plots': True,         # 生成可视化图表
        'half': True,          # 半精度推理
        'workers': 4,          # 与训练一致的数据加载线程数
        'single_cls': False,   # 多类别检测
    }
    
    print(f"🚀 开始验证 YOLOv11 在 VisDrone 上的性能")
    print(f"📦 模型权重: {model_path}")
    print(f"📊 数据集配置: {data_yaml}")
    
    try:
        validator = DetectionValidator(args=args)
        validator()  # 执行验证
    except Exception as e:
        print(f"❌ 验证器执行失败: {e}")
        return None, None
    
    # 确保metrics对象存在
    if hasattr(validator, 'metrics') and validator.metrics is not None:
        return validator, validator.metrics.results_dict
    else:
        raise RuntimeError("验证器未生成有效的metrics结果")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv11 VisDrone 验证与检测')
    
    parser.add_argument('--weights', type=str,
                        default=r"E:/Learning/深度学习/YoLo系列/v11/runs/train/YOLOv11-VisDrone25/weights/best.pt", 
                        help='模型权重路径')
    
    parser.add_argument('--data', type=str, 
                        default=r"E:/Learning/深度学习/YoLo系列/v11/ultralytics/cfg/datasets/VisDrone.yaml",
                        help='VisDrone数据集YAML路径')
    
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
                        
    # 图片/视频检测参数
    parser.add_argument('--source', type=str, action='append', default=[],
                        help='检测图片/视频路径（支持通配符如*.jpg）')
                        
    parser.add_argument('--conf', type=float, default=0.6,
                        help='检测置信度阈值')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='指定计算设备 (如: 0,1 或 "cpu")')
    
    args = parser.parse_args()
    
    # 如果没有提供source参数，使用默认值
    if not args.source:
        args.source = ["E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/论文图/检测图/9999952_00000_d_0000238.jpg"]
        # args.source = []
    
    # 执行模式选择
    if args.source:
        # 单张图片/视频检测模式
        detect_and_visualize(
            model_path=args.weights,
            image_paths=args.source,
            imgsz=args.imgsz,
            conf_thres=args.conf,
            device=args.device if args.device else None
        )
    else:
        # 完整验证集评估模式
        try:
            validator, results = validate(args.weights, args.data, args.imgsz)
        except Exception as e:
            print(f"❌ 验证过程中发生错误: {e}")
            exit(1)
        
        # 检查结果是否有效
        if not results:
            print("❌ 未获得有效的验证结果")
            exit(1)
            
        # 打印关键指标
        print("\n📊 验证结果:")
        if hasattr(validator.metrics, 'keys') and validator.metrics.keys:
            for key in validator.metrics.keys:
                value = results.get(key, float('nan'))
                print(f"{key}: {value:.4f}")
        else:
            for key, value in results.items():
                print(f"{key}: {value:.4f}")

        # 保存结果到文本文件
        result_file = Path(args.weights).parent.parent / 'validation_results.txt'
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                # 保存速度信息
                if hasattr(validator.metrics, 'speed'):
                    speed = validator.metrics.speed
                    total_time = speed.get('preprocess', 0) + speed.get('inference', 0) + speed.get('postprocess', 0)
                    
                    if total_time > 0:
                        total_fps = 1000 / total_time
                        f.write(f"端到端FPS: {total_fps:.2f}\n")
                    
                    # 分别记录各阶段耗时
                    for stage in ['preprocess', 'inference', 'postprocess']:
                        time_ms = speed.get(stage, 'N/A')
                        f.write(f"{stage}时间: {time_ms:.2f}ms/img\n")
                
                # 保存指标结果
                if hasattr(validator.metrics, 'keys') and validator.metrics.keys:
                    for key in validator.metrics.keys:
                        value = results.get(key, float('nan'))
                        f.write(f"{key}: {value:.4f}\n")
                else:
                    for key, value in results.items():
                        f.write(f"{key}: {value:.4f}\n")
                        
            print(f"\n💾 结果已保存至: {result_file}")
            
        except Exception as e:
            print(f"❌ 保存结果时出错: {e}")