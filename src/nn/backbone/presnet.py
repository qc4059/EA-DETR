'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d, DepthWiseConv

from src.core import register


__all__ = ['PResNet']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', dual_conv_g=1):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut    # 这个参数的作用是控制 是否对输入x 做维度调整   True 时，不需要调整 直接与x相加

        if not shortcut:     # 第二层残差 stride = 1
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),                            # 这里将平均池  替换为 最大池化
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        if stride==2:  # stride ==2 时 替换使用卷积进行下采样,使用SPD进行下采样
            self.branch2a = space_to_depth()                                                     
            self.branch2b = ConvNormLayer(4*ch_in, ch_out, 1, 1, act=None)                           

        else:   # stride==1 的情况
            self.branch2a = ConvNormLayer(ch_in, ch_out, 3, 1, act=act)
            self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)

        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):       # stage:2 64->64 stage3:64->128 stage4:128->256   count = 2   传递分组参数
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act,)   # 传递分组参数
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class DwBlock(nn.Module):  #128  256
    def __init__(self, ch_in, ch_out, stride, act):
        super().__init__()

        self.branch3a = space_to_depth()  
        self.branch3b =  DepthWiseConv(ch_in*4, ch_in*4, stride=1, act=act) 
        self.branch3c = ConvNormLayer(ch_in*4, ch_out, 1, 1)  # 1x1卷积调整通道数

        self.shortcut = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            ConvNormLayer(ch_in, ch_out, 1, 1, act=None)
        )

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        residual = x
        out = self.branch3a(x)
        out = self.branch3b(out)
        out = self.branch3c(out)
        residual = self.shortcut(residual) 
        out = self.act(residual+out)                                      # 在残差连接之后 添加 非线性激活函数
        return out


import torch
import torch.nn as nn


class SingleAttention(nn.Module):
    def __init__(self, ch_in, ch_out, act, head_dim=None):
        super().__init__()
        assert ch_in == ch_out, "Input and output channels must be equal in single-head attention"
        
        self.ch_out = ch_out
        self.norm = nn.LayerNorm(ch_in)
        
        # 生成Q/K/V的1x1卷积
        self.to_qkv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=3 * ch_out,
            kernel_size=1
        )
        self.scale = ch_out ** -0.5
        
        # 残差缩放因子
        self.gamma = nn.Parameter(torch.ones(1))  
        
        # 激活函数定义
        # self.act = nn.Identity() if act is None else get_activation(act)

    def build_2d_sincos_position_embedding(self, w, h, device):
        """ 动态生成位置编码 """
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')  # x对应宽度，y对应高度
        
        pos_dim = self.ch_out // 4  # 拆分4部分：sin_w, cos_w, sin_h, cos_h
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1. / (10000. ** omega)
        
        # 计算各分量
        out_w = grid_w.reshape(-1, 1) @ omega[None]  # [H*W, pos_dim]
        out_h = grid_h.reshape(-1, 1) @ omega[None]
        
        pos_embed = torch.cat([
            out_w.sin(), out_w.cos(),
            out_h.sin(), out_h.cos()
        ], dim=1)  # [H*W, C_out]
        
        return pos_embed.unsqueeze(0)  # [1, H*W, C_out]

    def forward(self, x):
        N, C, H, W = x.shape
        identity = x
        
        # LayerNorm处理 (前置)
        x = x.permute(0, 2, 3, 1).contiguous()  # 添加contiguous确保内存连续
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 生成位置编码
        pos_embed = self.build_2d_sincos_position_embedding(W, H, x.device)
        
        # 生成Q/K/V
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑维度并融合位置编码
        q = q.view(N, C, -1).permute(0, 2, 1)
        k = k.view(N, C, -1)
        v = v.view(N, C, -1).permute(0, 2, 1)
        
        q = q + pos_embed
        k = k + pos_embed.permute(0, 2, 1)
        
        # 注意力计算
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # 恢复空间维度
        out = out.permute(0, 2, 1).view(N, C, H, W)
        
        # 残差连接 + 可学习缩放
        return identity + self.gamma * out


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)



class ElAN(nn.Module):   # 256 256
    def __init__(self, ch_in, ch_out, act):
        super().__init__()
        ch_temp = ch_out // 2
        self.cv0 = ConvNormLayer(ch_in, ch_temp, 1, 1, act=act)         
        self.cv1 = ConvNormLayer(ch_in, ch_temp, 1, 1, act=act)
        # self.cv2 = ConvNormLayer(ch_in, ch_temp, 1, 1, act=act)
        self.attention = SingleAttention(ch_temp, ch_temp, act=act)
        self.cv3 =  ConvNormLayer(ch_temp, ch_temp, 3, 1, act=act)
        self.cv4 = ConvNormLayer(ch_temp, ch_temp, 3, 1, act=act)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.out = ConvNormLayer(ch_temp, ch_out, 1, 1, act=act)

    def forward(self, x):   
        residual = x
        x1 = self.cv0(x)
        out = self.cv3(x1)
        out = self.cv4(out)
        residual = self.cv1(residual)
        att_out = self.attention(residual)
        return self.out(self.act(att_out+out))
    

class GHSA_Blocks(nn.Module):
    def __init__(self, DwBlock, ch_in, ch_out, act='silu'):
        super().__init__()
        # Stage4/5 的下采样与注意力模块
        # self.bn = nn.BatchNorm2d(ch_in)
        self.dw_block = DwBlock(ch_in, ch_out, stride=2, act=act)
        self.elan= ElAN(ch_out, ch_out, act)


    def forward(self, x):
        # x = self.bn(x)
        x = self.dw_block(x)     # 下采样 [N,256,40,40] → [N,512,20,20]
        x = self.elan(x)    # 自注意力处理 [N,512,20,20]
        # x = self.cglu(x)         # 门控融合 [N,512,20,20]
        return x



# SPD
class space_to_depth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(block_size)

    def forward(self, x):
        return self.unshuffle(x)
    

@register
class PResNet(nn.Module):
    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False,):     # 新增分组参数 
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],                                    
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        self.spd = space_to_depth()

        ch_out_list = [64, 128, 256, 512]

        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.spd_conv = ConvNormLayer(32*4, 64, 1, 1, act=act)  # 假设 conv1 输出通道为64

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):  # num_stages = 4
            if i < 2:   # stage2 和 stage3 使用基于Dual Conv的 Basic Block
                stage_num = i + 2
                self.res_layers.append(                   # 2，2，2，2      # stage_num 残差网络的阶段编号        传递分组参数
                    Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
                )
                ch_in = _out_channels[i]   # 64，128，256，512
            else:  # Stage4 和 Stage5 使用 自写的Blocks
                self.res_layers.append(
                    GHSA_Blocks(DwBlock, ch_in, ch_out_list[i], act=act)
                )
                ch_in = _out_channels[i]   # 64，128，256，512

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        # if pretrained:
        #     state = torch.hub.load_state_dict_from_url(donwload_url[depth])
        #     self.load_state_dict(state)
        #     print(f'Load PResNet{depth} state_dict')
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

        # x 的 维度 (4,3,640,640)
    def forward(self, x):
        # conv1 维度 (4,32,320,320)
        conv1 = self.conv1(x)  #  使用卷积进行下采样
        # x 维度 (4,128,160,160)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = [] # 主干网络输出 C3、C4、C5
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:  # 指定返回的层索引，例如C3、C4、C5
                outs.append(x)
        return outs


















# class SingleAttention(nn.Module):
#     def __init__(self,ch_in, ch_out, act, selected_channels=64):
#         super().__init__()
#         self.selected_channels = selected_channels
        
#         # 定义Q/K/V的生成层（每个通道独立映射）
#         self.to_qkv = nn.Conv2d(
#             in_channels=selected_channels,  # 输入为前64个通道
#             out_channels=3 * selected_channels,  # 输出Q/K/V各64通道
#             kernel_size=1  # 1x1卷积相当于全连接
#         )
        
#         # 缩放因子
#         self.scale = selected_channels ** -0.5

#         self.branch4a = ConvNormLayer(selected_channels, selected_channels,1,1) # 对前64通道的特征图做1x1的卷积
#         channels = ch_in - selected_channels
#         self.branch4b = ConvNormLayer(channels, channels,1,1)  # 对剩下的通道也做 1x1 的卷积

#     def forward(self, x):
#         # 输入x形状: [N, 256, 40, 40]
#         # 步骤1: 直接提取前64个通道
#         x_slice = x[:, :self.selected_channels, :, :]  # 输出形状: [N, 64, 40, 40]
#         x_slice_1x1 = self.branch4a(x_slice)
#         # 步骤2: 生成Q/K/V（通过1x1卷积）
#         qkv = self.to_qkv(x_slice)  # 输出形状: [N, 3*64, 40, 40]
        
#         # 拆分为Q, K, V
#         q, k, v = torch.chunk(qkv, 3, dim=1)  # 每个形状: [N, 64, 40, 40]
        
#         # 步骤3: 展平空间维度为序列
#         N, C, H, W = q.shape
#         q = q.view(N, C, H * W).permute(0, 2, 1)  # [N, 1600, 64]
#         k = k.view(N, C, H * W).permute(0, 2, 1)  # [N, 1600, 64]
#         v = v.view(N, C, H * W).permute(0, 2, 1)  # [N, 1600, 64]
        
#         # 步骤4: 计算注意力权重
#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, 1600, 1600]
#         attn = torch.softmax(attn, dim=-1)
        
#         # 步骤5: 聚合V并恢复空间维度
#         out = torch.matmul(attn, v)  # [N, 1600, 64]
#         out = out.permute(0, 2, 1).view(N, C, H, W)  # [N, 64, 40, 40]
        
#         out = out + x_slice_1x1

#         x_slice2 = x[:, self.selected_channels:, :, :]  #  剩下的特征图
#         x_slice2_1x1 = self.branch4b(x_slice2)   # 做1x1卷积
#         out = torch.cat([out, x_slice2_1x1], dim=1)  # 按通道拼接两部分的特征
#         out = out + x  # 类似残差连接
#         return out