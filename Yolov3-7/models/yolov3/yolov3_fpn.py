import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov3_basic import Conv, ConvBlocks


# Yolov3FPN
class Yolov3FPN(nn.Module):
    def __init__(self,
                 in_dims=[256, 512, 1024],
                 width=1.0,
                 depth=1.0,
                 out_dim=None,
                 act_type='silu',
                 norm_type='BN'):
        super(Yolov3FPN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims

        # P5 -> P4                        1024       512              silu                bn   
        self.top_down_layer_1 = ConvBlocks(c5, int(512*width), act_type=act_type, norm_type=norm_type)
        self.reduce_layer_1 = Conv(int(512*width), int(256*width), k=1, act_type=act_type, norm_type=norm_type)

        # P4 -> P3
        self.top_down_layer_2 = ConvBlocks(c4 + int(256*width), int(256*width), act_type=act_type, norm_type=norm_type)
        self.reduce_layer_2 = Conv(int(256*width), int(128*width), k=1, act_type=act_type, norm_type=norm_type)

        # P3
        self.top_down_layer_3 = ConvBlocks(c3 + int(128*width), int(128*width), act_type=act_type, norm_type=norm_type)

        # output proj layers
        if out_dim is not None:     # out_dim 是统一通道数的 将每一个ConvBlock 的输出 转为同一个通道数 
            # output proj layers
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
                        for in_dim in [int(128 * width), int(256 * width), int(512 * width)]
                        ])
            self.out_dim = [out_dim] * 3

        else:
            self.out_layers = None
            self.out_dim = [int(128 * width), int(256 * width), int(512 * width)]


    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.top_down_layer_1(c5)
        # print(p5.size())                # p5 [1, 512, 20, 20]
        # p4/16
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)  # 上采样操作
        # print(p5_up.size())
        p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))
        #print(p4.size())                  p4 [1, 256, 40, 40]
        # P3/8
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
        # print(p4_up.size())
        p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))
        #print(p3.size())                  p3 [1, 128, 80, 80]
        out_feats = [p3, p4, p5]

        # output proj layers
        if self.out_layers is not None:     # out_layers的作用是 可 指定P3 P4 P5的特征图的输出
            # output proj layers
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    # build neck
    if model == 'yolov3_fpn':
        fpn_net = Yolov3FPN(in_dims=in_dims,
                            out_dim=out_dim,
                            width=cfg['width'],
                            depth=cfg['depth'],
                            act_type=cfg['fpn_act'],
                            norm_type=cfg['fpn_norm']
                            )
    else:
        raise NotImplementedError('FPN {} not implemented.'.format(cfg['fpn']))        
    return fpn_net
