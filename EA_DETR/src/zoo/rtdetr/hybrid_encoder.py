'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register
from .DCNv2 import DepthWiseConv

__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()  # 不对输入进行任何处理，返回输入值

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class space_to_depth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(block_size)

    def forward(self, x):
        return self.unshuffle(x)

class EDF_FAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride =1 , act="silu"):
        super().__init__()
        
        # 空间注意力路径 (L-path)
        self.spatial_att = nn.Sequential(
            ConvNormLayer(in_channels, out_channels, 1, 1, act=act),

            DepthWiseConv(out_channels, out_channels, stride=1, act=act),
            DepthWiseConv(out_channels, out_channels, stride=1, act=act),
        )
        
        # 通道注意力路径 (R-path)
        self.conv_reduce = ConvNormLayer(in_channels, out_channels, 1, 1, act=act)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 修正后的1D卷积层 (输入通道保持为out_channels)
        self.conv1d_7 = nn.Conv1d(
            in_channels=out_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=7,
            padding=3,
            groups=out_channels  # 深度可分离卷积
        )
        self.conv1d_5 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            padding=2,
            groups=out_channels
        )
        self.conv1d_3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels
        )
        
        self.conv_aggregate = ConvNormLayer(out_channels * 3, out_channels, 1, 1, act=act)
        self.sigmoid = nn.Sigmoid()
        self.csp_layer = CSPRepLayer(out_channels, out_channels, round(3), act=act, expansion=1)

    def forward(self, x):
        # 空间注意力
        L = self.spatial_att(x)  # [B, C, H, W]
        
        # 通道注意力
        x_reduced = self.conv_reduce(x)  # [B, C, H, W]
        
        # 保持通道维度为C
        pooled = self.adaptive_avg_pool(x_reduced)  # [B, C, 1, 1]
        pooled = pooled.squeeze(-1).squeeze(-1)     # [B, C]
        pooled = pooled.unsqueeze(-1)               # [B, C, 1] = 输入形状
        
        # 并行处理三个卷积 (保持通道维度)
        out_7 = self.conv1d_7(pooled)  # [B, C, 1]  （8,256,1）
        out_5 = self.conv1d_5(pooled)  # [B, C, 1]
        out_3 = self.conv1d_3(pooled)  # [B, C, 1]
        
        # 拼接结果
        concatenated = torch.cat([out_7, out_5, out_3], dim=1)  # [8, 256, 3]
        concatenated = concatenated.unsqueeze(-1) # [B, 3*C, 1, 1]
        
        R = self.conv_aggregate(concatenated)  # 
        R = R.expand_as(L)  # 扩展至与L相同的维度 
        
        M = self.sigmoid(L + R)  # 空间+通道注意力融合
        out = x * M  # 特征校准
        out = self.csp_layer(out)  # 特征增强
        return out

@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.act = act 

                # 预先定义EDF_FAM模块
        self.edf_fam = EDF_FAM(
            in_channels= hidden_dim,  # 根据实际输入通道数调整
            out_channels= hidden_dim,
            act=act
        )

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential( # 输入通道数 输出通道数  
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        self.Concat_conv1 = ConvNormLayer(hidden_dim*6, hidden_dim, 1, 1, act=act)   # 用来处理P2层与其他层拼接后的特征的卷积

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.last_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):   #  len(in_channels)为4    _取值为3,2,1 
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.last_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))                 # 有添加
            if _== 1:
                self.fpn_blocks.append(
                    CSPRepLayer(hidden_dim * 6, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                )
            else:
                self.fpn_blocks.append(
                    CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                )
 
        # self.new_fpn_blocks = RepLayer()                                                      #  如何设计多尺度拼接           ++++++++++++++++++
        self.spd = space_to_depth()
        # bottom-up pan
        # 统一的下采样模块定义
        self.downsample_convs = nn.ModuleList([
            ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            for _ in range(len(in_channels)-2)  # 只需要两个下采样
        ])

        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
        self.BiFPN = CSPRepLayer(hidden_dim * 3, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)  # 这里的self_in_channels [256, 512, 1024, 2048]      需确定：是否需要四个尺度  确定
        # 这里的proj_feats 存储了四个张量S2、S3、S4、S5  并且将张量的深度统一调整为 256
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:  # 编码器层数
            for i, enc_ind in enumerate(self.use_encoder_idx):   # self.use_encoder_idx = [3]作用是取出S5级别 张量
                h, w = proj_feats[enc_ind].shape[2:] # 取出S5 尺度的张量的宽高
                # flatten [B, C, H, W] to [B, HxW, C]  【4,HxW,256】
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)  # 
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(   # 位置编码 （1,HxW,256）
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # 将S5输入编码器，并得到输出结果memory 将编码器的输出结果 存入proj_feats[2]最后一层 并将维度在此换为(4,256,18,18)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous() #contiguous() 确保张量在内存中连续存储
                # print([x.is_contiguous() for x in proj_feats ])
                inner_outs = self.build_fpn(proj_feats)
                outs = self.build_pan(inner_outs, proj_feats)
    
                return outs

    def build_fpn(self, proj_feats):
        inner_outs = [proj_feats[-1]]
        for idx in reversed(range(1, len(self.in_channels))):   # idx 为 3 2 1
            feat_high = self.lateral_convs[-idx](inner_outs[0])
            inner_outs[0] = feat_high
            feat_low = proj_feats[idx-1]
            
            # 统一上采样处理
            if idx > 1:
                feat_high = F.interpolate(feat_high, scale_factor=2, mode='nearest')
            
            # 动态拼接特征
            concat_features = [feat_high, feat_low]                     
            if idx == 1:
                del(concat_features[1])
                concat_features.append(self.spd(proj_feats[idx-1]))
                feat_S3 = proj_feats[idx]
                concat_features.append(feat_S3)             # torch.cat(concat_features, dim=1) 维度 80x80x1536
                features = self.Concat_conv1(torch.cat(concat_features, dim=1))           # 1x1卷积 处理拼接过后的特征                   
                inner_out = self.edf_fam(features)
                # inner_out = self.fpn_blocks[-idx](torch.cat(concat_features, dim=1))
            else:
                inner_out = self.fpn_blocks[-idx](torch.cat(concat_features, dim=1))

            if idx == 1:
                inner_outs[0] = inner_out
            else:
                inner_outs.insert(0,inner_out)

        return inner_outs


    def build_pan(self, inner_outs, proj_feats):
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 2):  # 0 1 2
            feat_low = outs[-1]
            feat_low = self.last_convs[idx](feat_low)
            outs[idx] = feat_low
            if idx==0:
                feat_S4 = proj_feats[2]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # 使用3x3的卷积进行下采样
            if idx==0:
                out = self.BiFPN(torch.concat([downsample_feat, feat_high, feat_S4], dim=1))
            else:
                out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        out = self.last_convs[2](outs[-1])    
        outs[-1] = out
        return outs


        # # broadcasting and fusion  这部分是 进行多尺度 融合的步骤
        # inner_outs = [proj_feats[-1]]  # 这个就是编码器的输出
        # for idx in range(len(self.in_channels) - 1, 0, -1):  # range(2,0,-1)  只循环两次 ids 取 2,1               in_channels = 4 时 取 3 2 1
        #     # feat_high 就是编码器的输出 维度为 (4,256,18,18)
        #     feat_high = inner_outs[0]
        #     feat_low = proj_feats[idx - 1] # S4 的维度
        #     feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high) # 经过1x1的卷积 维度:(4,256,18,18)

        #     if idx!=1:         # 当对S2 S3 和 上面处理后的特征 进行融合的时候， 不对feat_high进行上采样 保证维度对齐
        #         inner_outs[0] = feat_high
        #         upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest') # 对S5进行上采样 维度为（4,256,36,36) 与S4 维度匹配 进行融合
        #     # concat 将上采样后的 S5 和 S4 在通道上进行拼接 拼接后的维度为：(4,512,36,36))
        #     if idx ==1:
        #         feat_low = self.spd(feat_low)    # 使用 SPD对S2 进行下采样 到 80x80
        #         feat_S3 = proj_feats[idx]  # 取出S3 
        #         inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low, feat_S3], dim=1))  # 拼接 融合
        #     else:
        #         inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1)) # 维度:(4,256,36,36)
        #     if idx!=1 : 
        #         inner_outs.insert(0, inner_out)
        #     else:
        #         inner_outs[0] = inner_out


        # outs = [inner_outs[0]]
        # for idx in range(len(self.in_channels) - 2):  # 0 1 2
        #     feat_low = outs[-1]
        #     feat_low = self.last_convs[idx](feat_low)
        #     outs[idx] = feat_low
        #     if idx==0:
        #         feat_S4 = proj_feats[2]
        #     feat_high = inner_outs[idx + 1]
        #     downsample_feat = self.downsample_convs[idx](feat_low)  # 使用3x3的卷积进行下采样
        #     if idx==0:
        #         out = self.BiFPN(torch.concat([downsample_feat, feat_high, feat_S4], dim=1))
        #     else:
        #         out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
        #     outs.append(out)
        # out = self.last_convs[2](outs[-1])    
        # outs[-1] = out
        # return outs