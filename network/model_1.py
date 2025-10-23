
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DoubleConv3D, self).__init__()
        if isinstance(kernel_size, int):
            assert kernel_size % 2 == 1, f"卷积核大小必须是奇数，当前为{kernel_size}"
            padding = (kernel_size-1)//2
        else:
            assert all(k % 2 == 1 for k in kernel_size), f"所有卷积核维度必须是奇数，当前为{kernel_size}"
            padding = tuple((k-1)//2 for k in kernel_size)
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet3D_4Layer(nn.Module):
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512], 
                 time_kernels=[3, 3, 5, 5], spatial_kernels=[(7,5), (5,3), 3, 3]):
        """
        3D UNet网络
        
        Args:
            in_channels: 输入通道数
            channels: 各层通道数配置
            time_kernels: 各层时间维度的卷积核大小
            spatial_kernels: 各层空间维度的卷积核大小
        """
        super(UNet3D_4Layer, self).__init__()
        
        assert len(channels) == 4,
        assert len(time_kernels) == 4,
        assert len(spatial_kernels) == 4,
        
        self.channels = channels
        self.time_kernels = time_kernels
        self.spatial_kernels = spatial_kernels
        
        kernel_sizes = []
        for i in range(4):
            t_kernel = time_kernels[i]
            s_kernel = spatial_kernels[i]
            
            if isinstance(s_kernel, int):
                kernel_sizes.append((t_kernel, s_kernel, s_kernel))
            else:
                kernel_sizes.append((t_kernel, s_kernel[0], s_kernel[1]))

        self.enc1 = DoubleConv3D(in_channels, channels[0], kernel_sizes[0])
        self.enc2 = DoubleConv3D(channels[0], channels[1], kernel_sizes[1])
        self.enc3 = DoubleConv3D(channels[1], channels[2], kernel_sizes[2])
        self.enc4 = DoubleConv3D(channels[2], channels[3], kernel_sizes[3])
        

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        

        self.up3 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = DoubleConv3D(channels[2] * 2, channels[2], kernel_sizes[2])  # 注意通道数翻倍
        
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = DoubleConv3D(channels[1] * 2, channels[1], kernel_sizes[1])
        
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = DoubleConv3D(channels[0] * 2, channels[0], kernel_sizes[0])
        

        self.final_conv = nn.Conv3d(channels[0], in_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        original_shape = x.shape  # (b, t, c, h, w)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        enc1 = self.enc1(x)  # (b, 64, t, h, w)
        enc2 = self.enc2(self.pool(enc1))  # (b, 128, t, h/2, w/2)
        enc3 = self.enc3(self.pool(enc2))  # (b, 256, t, h/4, w/4)
        enc4 = self.enc4(self.pool(enc3))  # (b, 512, t, h/8, w/8)
        

        dec3 = self.up3(enc4)  # (b, 256, t, h/4, w/4)
        dec3 = torch.cat([dec3, enc3], dim=1)  #  (b, 512, t, h/4, w/4)
        dec3 = self.dec3(dec3)  # (b, 256, t, h/4, w/4)
        
        dec2 = self.up2(dec3)  # (b, 128, t, h/2, w/2)
        dec2 = torch.cat([dec2, enc2], dim=1)  #  (b, 256, t, h/2, w/2)
        dec2 = self.dec2(dec2)  # (b, 128, t, h/2, w/2)
        
        dec1 = self.up1(dec2)  # (b, 64, t, h, w)
        dec1 = torch.cat([dec1, enc1], dim=1)  #  (b, 128, t, h, w)
        dec1 = self.dec1(dec1)  # (b, 64, t, h, w)
        
        output = self.final_conv(dec1)  # (b, c, t, h, w)

        output = self.sigmoid(output)
        
        output = output.permute(0, 2, 1, 3, 4)
        
        output = output.reshape(original_shape)
        
        return output

def test_unet3d():
    model = UNet3D_4Layer(
        in_channels=3, 
        channels=[64, 128, 256, 512], 
        time_kernels=[3, 7, 15, 21], 
        spatial_kernels=[(5,7), (3,5), 3, 3] 
    )
    
    test_input = torch.randn(1, 40, 3, 16, 144) * 100
    print(f"input shape: {test_input}")
    
    with torch.no_grad():
        output = model(test_input)
        print(f"output shape: {output}")
    
    # 验证输入输出形状一致
    assert test_input.shape == output.shape, "输入输出形状不匹配"
    print("✓ 输入输出形状一致")
    
    # 打印模型参数和配置
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 打印各层卷积核配置
    print("\n各层卷积核配置:")
    for i in range(4):
        t_kernel = model.time_kernels[i]
        s_kernel = model.spatial_kernels[i]
        if isinstance(s_kernel, tuple):
            kernel_size = (t_kernel, s_kernel[0], s_kernel[1])
        else:
            kernel_size = (t_kernel, s_kernel, s_kernel)
        print(f"第{i+1}层: 时间核={t_kernel}, 空间核={s_kernel}, 总核大小={kernel_size}")
    
    return model

if __name__ == "__main__":
    model = test_unet3d()