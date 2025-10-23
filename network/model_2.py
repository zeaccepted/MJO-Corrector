
import torch
import torch.nn as nn
import xarray as xr
import numpy as np




def _denormalize_data(normalized_data, opt):
    
    if opt == 0:
        norm_params = {
            'olr': (-195.36105346679688, 110.78022766113281),
            'u850': (-22.693273441277675, 31.37643406305719),
            'u200': (-62.916534423828125, 43.55909729003906)
        }
    elif opt == 1:
        norm_params = {
            'olr': (-195.36105346679688, 135.9539794921875),
            'u850': (-26.091207777050208, 35.59424326704047),
            'u200': (-63.94963836669922, 49.02571105957031)
        }
    elif opt == 2:
        norm_params = {
            'olr': (-200, 115.91719055175781),
            'u850': (-24.21541445442865, 33.67539313901426),
            'u200': (-62.916534423828125, 49.02571105957031)
        }
    
    else:
        raise ValueError("opt must be 0, 1, or 2")
    denormalized_data = normalized_data
    
    # OLR反归一化 (通道0)
    min_val, max_val = norm_params['olr']
    denormalized_data[:, :, 0] = normalized_data[:, :, 0] * (max_val - min_val) + min_val
    
    # U850反归一化 (通道1)
    min_val, max_val = norm_params['u850']
    denormalized_data[:, :, 1] = normalized_data[:, :, 1] * (max_val - min_val) + min_val
    
    # U200反归一化 (通道2)
    min_val, max_val = norm_params['u200']
    denormalized_data[:, :, 2] = normalized_data[:, :, 2] * (max_val - min_val) + min_val
    
    return denormalized_data



def _calculate_rmm(data, opt):


    data = _denormalize_data(data, opt)


    with xr.open_dataset("/home/sunyuze/08_mjo_corr_new/data/truth.nc") as ds:
        pattern = ds['pattern'].values.reshape(2, -1)
        std = ds['std'].values
        factor = ds['factor'].values

    
    # 经度平均
    data = torch.nanmean(data, dim=3)

    # 标准化
    std_tensor = torch.from_numpy(std).to(data.device)
    data /= std_tensor[None, None, :, None]
    
    # 调整维度
    data = data.view((data.shape[0], data.shape[1], -1))
    
    # 矩阵乘法计算RMM
    pattern_tensor = torch.from_numpy(pattern).to(data.device)
    factor_tensor = torch.from_numpy(factor).to(data.device)
    data_swap = data.permute(0, 2, 1)
    rmm_data = torch.matmul(pattern_tensor, data_swap) / factor_tensor[:, None]
    
    # RMM归一化
    min_val = torch.tensor(-4.271076202392578, device=data.device)
    max_val = torch.tensor(3.9197659492492676, device=data.device)
    rmm_normalized = (rmm_data - min_val) / (max_val - min_val)
    rmm_normalized = rmm_normalized.permute(0, 2, 1)
    
    return rmm_normalized



class MinimalLSTMCorrection(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, opt = 0, num_layers = 4):
        super(MinimalLSTMCorrection, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.opt = opt
        
        
        
    def forward(self, x):

        rmm = _calculate_rmm(x[:,:,:,1:16,:], self.opt)
        lstm_out, _ = self.lstm(rmm)
        correction = self.output_layer(lstm_out)
        return rmm + correction



# 测试代码
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 4
    time_steps = 40
    channels = 3  # 例如：OLR, U850, U200
    height = 16
    width = 144
    
    # 创建模型
    model = MinimalLSTMCorrection(opt = 0)
    # model = OptimizedLSTMCorrection(opt = 0)
    
    # 测试输入
    x = torch.randn(batch_size, time_steps, channels, height, width)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 (4, 10, 2)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    