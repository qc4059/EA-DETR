import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
import torchvision.transforms as transforms
from PIL import Image


class CyclicShift(nn.Module):     #对输入张量进行 循环位移
    def __init__(self, displacement):   # displacement 是偏移量（位移量）
        super().__init__()
        self.displacement = displacement

    def forward(self, x):   #roll 是 torch 中的一个函数，用于将张量沿指定dims维度进行循环位移
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2)) #shifts表示在指定维度上的位移步长
    # 这里 x 的维度是什么？


class Residual(nn.Module):
    def __init__(self, fn):   # 这里fn是什么？  定义类时传入的参数
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):    # 这里的x是什么？类的实例化后，调用方法时传入的参数
        return self.fn(x, **kwargs) + x
        
class PreNorm(nn.Module):     #Layer Norm  层归一化
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #nn.LayerNorm是nn模块中的一个类， self.norm是 实例  #dim是输入张量中最后一个维度的小大
        self.fn = fn    # fn 是自定义的一个函数操作  一开始是WindowAttention类，这里是对类的实例化

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 这里调用实例化的类，并传入归一化好的 x，归一化好的 x 的维度依然是(1,56,56,96)
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(    #这个nn.Sequential类似于 Transforms.Compose用于图像处理的模块
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)    #传入 输入数据x后依次执行nn.Sequential中的步骤
    

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances
def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances




class WindowAttention(nn.Module):
    #                  96     3       32        flase      7                  true
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads  # 96  所有注意力头的维度

        self.heads = heads      # 注意力中 头 的数量
        self.scale = head_dim ** -0.5    # head_dim 一个头的维度,  这里scale的作用是 缩放每个头内的 QK
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2   # 对滑动窗口进行 偏移 半个窗口单位 窗口为7 偏移量为3
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            # 对窗口进行偏移后，有些图片像素不是相邻，对不相邻的图片片段进行 掩码
            # 不相邻图片 主要集中 在 底侧、右侧窗口
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        #                        96       288
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):   # x 为归一化好的x (1,56,56,96)
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # 此时的 qkv 是元组 (tensor[1,56,56,96], tensor[1,56,56,96], tensor[1,56,56,96])
        qkv = self.to_qkv(x).chunk(3, dim=-1)   # 这里qkv(x)的结果是 (1,56,56,288) chunk一下变为 三个 (1,56,56,96) 分别对应 Q、K、V

        nw_h = n_h // self.window_size   # nw_h = 8
        nw_w = n_w // self.window_size   # nw_w = 8
        # 这里的 map 函数是将qkv作为参数 传入 lambda函数进行相应操作 
        q, k, v = map(
            # lambda 中 :前是参数 :后是定义的类似于某种函数方法     
            #                       1     56          56      96     1 3      64         49    32 
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv) #qkv是一个元组，元组内元素依次作为t参数传入
        # 为什么要将q、k、v的维度从(1,56,56,96)变到(1,3,64,49,32)？因为要将96深度，分为三个注意力头，分别处理32深度的
        # 这里q、k、v 的维度都为 torch.Size([1, 3, 64, 49, 32])
        # 为什么q、k、v 的维度是五维？ 你仔细想，49个32维的，49表示一个窗口内的token，每个头处理32深度，每个头要处理64个窗口，一共有3个注意力头要处理这些。
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale # dots:(1,3,64,49,49) #一个窗口内的q、k进行点积之后，生成一个49*49大小的相似度矩阵，每个头一共有64个49相似度矩阵，三个头有3，64，49，49
        # (64,49,32)表示一个注意力头，在整个大的token像素中32维的q、k、v 

        # 
        if self.relative_pos_embedding:
            # Convert relative_indices to long type to use as indices
            dots += self.pos_embedding[self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long()]
        else:
            dots += self.pos_embedding

        # 在qk计算之后的 相似度上 进行 掩码
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1) #缩放相似度矩阵值之后，进行softmax处理，跟transformer注意力一样的

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v) #跟上面一样的操作(1,3,64,49,49)  (1,3,64,49,32)V
        # out: (1,3,64,49,32)

        # 这一步是要将 变化的张量 恢复到与输入一致的形状 
        #                     1 3   8     8     7  7    32   1   8    7     8    7    3  32
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # out -> (1,56,56,96)

        #   
        out = self.to_out(out)   # out = (1,56,56,96)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    #                   96    3        32      288                   7              true
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        # 下面将类的实例化作为参数进行传递，所执行的步骤如下：
        # PreNorm：对输入x 进行Layers Normalization
        # WindowAttention：计算窗口注意力
        # Residual：将窗口注意力的输出 与 x 进行相加
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        # 下面将类的实例化作为参数进行传递，所执行的步骤如下：
        # 1. PreNorm：对输入 x 进行Layers Normalization
        # 2. FeedForward：对 x 进行前馈神经网络的处理
        # 3. Residual：将前馈神经网络的输出 与 x 进行相加
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):   #(1,56,56,96)
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
    


class PatchMerging(nn.Module):
    # 输入通道数、输出通道数、下采样倍数  eg:downscaling_factor=2 表示将原图的 2x2 区域划分为一个patch
    #                       3            96               4                 
    def __init__(self, in_channels, out_channels, downscaling_factor): 
        super().__init__()
        self.downscaling_factor = downscaling_factor
        #这里nn.Unfold是一个类对其实例化的操作，先初始化类中一些参数，随后在self.patch_merge(x)传入x时，会调用nn.Unfold中的一个方法，这个方法会实现将原始输入图片数据进行分割多个patch的操作
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        #这里in_channels * downscaling_factor ** 2 ---3*4*4 = 48（输入）96（输出） 
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):  # (1,3,224,224)
        b, c, h, w = x.shape  #（批次，深度，图片高，图片宽）
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor  #计划下采样后的图片维度 例如：224//4 = 56，将图片分割为56*56个patch，每个patch16个小方格，深度为48
        #view更改张量维度，permute 换一下张量中维度的位置 例如：（1,48,56,56） 经过permute（1,56,56,48）
        # 经过nn.Unfold操作后 x 的维度为 (1,48,3136) 经过view 维度变为 (1,48,56,56)
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1) # 再经过permute x 维度变为（1,56,56,48）
        # 对分割后的patch转换后的token（1,56,56,48）->（1,56,56,96）进行线性变化，
        # 这里很让人百思不得其解，nn.Linear()一般接受的二维张量，但是这里接受的是四维张量，你会怎么想？
        # 当输入一个多维张量时，它会自动处理最后一维。因此，当输入是 (1, 56, 56, 48) 时，nn.Linear 只会对最后的 48 维进行线性变换，将其变成 96 维，而保持前面的维度不变
        # print("前",x.size())   # (1,56,56,48)
        x = self.linear(x) 
        # print("后",x.size())   # (1,56,56,96)
        return x
    

class StageModule(nn.Module):
    #                       3             96            2           4                3          32          7
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        # 这里layers 指的是  block的数量，例如 layers=2 表示 W-MSA 和 SW-MSA 各一个
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        #                                        3                         96
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                #                  96                 3               32                    288
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))
  
    def forward(self, x):     # 初始x (1,3,224,224)
        x = self.patch_partition(x)   # x的维度：(1,56,56,96)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
    

class SwinTransformer(nn.Module):
    #                         96     2,2,6,2  3,6,12,24
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()
        # hidden_dimension 表示的是每个阶段输入图像数据的 通道数 一阶段96 二阶段 192 三阶段 384 四阶段 768
        #                                 3                     96                       2
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        # self.mlp_head = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 8),
            # nn.Linear(hidden_dim * 8, num_classes)
        # )
        self.cv1 = nn.Conv2d(768,768,2)
        self.cv2 = nn.Conv2d(768,512,1)

    def forward(self, img):
        x = self.stage1(img)   # img (1,3,224,224)  x(1,56,56,96)
        x = self.stage2(x)    #x x(1,56,56,96)    x()
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.cv1(x)
        x = self.cv2(x)
        # x = x.mean(dim=[2, 3])
        # return self.mlp_head(x)
        # x = x.permute(0, 3, 2, 1)
        x = x.permute(0, 1, 3, 2)
        # print(x.size()) 
        return x
    
def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


# 初始实例化的时候，会自动初始类中的__init__初始化部分
net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)
image_01 = Image.open("E:/Learning/Data/VOCdevkit/VOC2007/JPEGImages/000001.jpg")
transform = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor()
])
dummy_x = transform(image_01)
dummy_x = dummy_x.unsqueeze(0)  # dummy_x 形状(1,3,224,224)
# 在调用实例的时候，会自动执行实例中的 forward 方法
logits = net(dummy_x)  # (1,3)
# print(net)
# print(logits)

