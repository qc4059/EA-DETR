import torch
import torch.nn as nn

__all__ = ['SCINet']

class SCINet(nn.Module):
    def __init__(self, channels=3, layers=3):
        super(SCINet, self).__init__()
        kernel_size = 3
        padding = (kernel_size - 1) // 2  # 计算padding

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, padding=padding),
            nn.ReLU()
        )

        # 为每个块创建独立的卷积模块
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=padding),
                # nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            self.blocks.append(block)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for block in self.blocks:
            fea = fea + block(fea)  # 残差连接
        fea = self.out_conv(fea)

        # 向量化条件判断
        input_mean = torch.mean(input, dim=(1,2,3), keepdim=True)
        alpha = torch.ones_like(input_mean)  # 默认1.0
        alpha = torch.where(input_mean < 0.2, 2.0, alpha)
        alpha = torch.where((input_mean >= 0.2) & (input_mean < 0.5), 1.5, alpha)
        # alpha = torch.ones_like(input_mean) * 0.5  # 固定系数

        illu = alpha * fea + input
        illu = torch.clamp(illu, 0.0, 1.0)  # 更合理的范围限制
        return illu


