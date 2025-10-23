import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import os
from network.model_1 import UNet3D_4Layer
from network.model_2 import MinimalLSTMCorrection
import xarray as xr
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import shap
import random

from scipy import stats


class MJODataset(Dataset):
    def __init__(self, opt=0, dataopt="sequential", split="train", leadtime=40, season="all"):

        self.root_path = '/home/sunyuze/08_mjo_corr_new'
        self.file_truth = os.path.join(root_path, 'data', 'truth.nc')

        if opt == 0:
            self.forecast_file = os.path.join(self.root_path, 'data', 'ecmwf.nc')
            self.norm_params = {
                'olr': (-195.36105346679688, 110.78022766113281),
                'u850': (-22.693273441277675, 31.37643406305719),
                'u200': (-62.916534423828125, 43.55909729003906)
            }
        elif opt == 1:
            self.forecast_file = os.path.join(self.root_path, 'data', 'bcc.nc')
            self.norm_params = {
                'olr': (-195.36105346679688, 135.9539794921875),
                'u850': (-26.091207777050208, 35.59424326704047),
                'u200': (-63.94963836669922, 49.02571105957031)
            }
        elif opt == 2:
            self.forecast_file = os.path.join(self.root_path, 'data', 'ncep.nc')
            self.norm_params = {
                'olr': (-200, 115.91719055175781),
                'u850': (-24.21541445442865, 33.67539313901426),
                'u200': (-62.916534423828125, 49.02571105957031)
            }
        
        else:
            raise ValueError("opt must be 0, 1, or 2")

        self.opt = opt
        self.dataopt = dataopt
        self.split = split
        self.leadtime = leadtime
        self.season = season

        self._load_data()

        self._split_dataset()

        print(f"{split} dataset (opt={opt}, {dataopt}): "
              f"forecast shape = {self.forecast_data.shape}, "
              f"truth shape = {self.truth_data.shape}, "
              f"rmm shape = {self.rmm_data.shape}")

    def _load_data(self):
        ds_forecast = xr.open_dataset(self.forecast_file)
        self.forecast_data = np.stack([
            ds_forecast['olr'].values,
            ds_forecast['u'][:, :, 0, :, :].values,  # u850
            ds_forecast['u'][:, :, 2, :, :].values  # u200
        ], axis=2)
        forecast_times = ds_forecast['time'].values
        self.forecast_times = ds_forecast['time'].values  # 保存时间信息用于季节判断
        ds_forecast.close()

        ds_truth = xr.open_dataset(self.file_truth)
        self.truth_data = np.stack([
            ds_truth['olr'].values,
            ds_truth['u'][:, 0, :, :].values,
            ds_truth['u'][:, 2, :, :].values
        ], axis=1)
        truth_times = ds_truth['time'].values
        self.rmm_data = ds_truth['rmm'].values
        ds_truth.close()

        self.truth_data = self._expand_truth_to_forecast_shape(self.truth_data, truth_times, forecast_times)
        self.rmm_data = self._expand_rmm_to_forecast_shape(self.rmm_data, truth_times, forecast_times)

        print(f"Forecast: {self.forecast_data.shape}, Truth: {self.truth_data.shape}, rmm: {self.rmm_data.shape}")

    def _expand_truth_to_forecast_shape(self, truth_data, truth_times, forecast_times):
        expanded_truth = []

        for i, forecast_time in enumerate(forecast_times):
            start_idx = np.where(truth_times == forecast_time)[0][0]
            end_idx = start_idx + self.leadtime
            if end_idx <= len(truth_times):
                truth_sequence = truth_data[start_idx:end_idx]
                expanded_truth.append(truth_sequence)
            else:
                continue

        return np.array(expanded_truth)

    def _expand_rmm_to_forecast_shape(self, rmm_data, truth_times, forecast_times):
        expanded_rmm = []

        for i, forecast_time in enumerate(forecast_times):
            start_idx = np.where(truth_times == forecast_time)[0]
            if len(start_idx) == 0:
                continue
            start_idx = start_idx[0]
            end_idx = start_idx + self.leadtime

            if end_idx <= len(truth_times):
                rmm_sequence = rmm_data[:, start_idx:end_idx]  # shape: (leadtime, 2)
                expanded_rmm.append(rmm_sequence)

        return np.array(expanded_rmm)  # shape: (b, t, 2)


    def _get_season_indices(self, indices):
        """从给定的索引中筛选出指定季节的索引"""
        if self.season == "all":
            return indices
        
        # 获取对应的时间
        selected_times = self.forecast_times[indices]
        
        if isinstance(selected_times[0], np.datetime64):
            dates = [pd.Timestamp(t).to_pydatetime() for t in selected_times]
        else:
            dates = selected_times
        
        season_indices = []
        
        for i, (orig_idx, date) in enumerate(zip(indices, dates)):
            month = date.month
            
            if self.season == "winter":
                if month in [10, 11,12, 1, 2]:
                    season_indices.append(orig_idx)
            elif self.season == "summer":
                if month in [5,6, 7, 8,9]:
                    season_indices.append(orig_idx)
        
        return np.array(season_indices)


    def _split_dataset(self):
        total_samples = self.forecast_data.shape[0]

        if self.dataopt == "sequential":
            if self.split == "train":
                split_index = int(total_samples * 0.8)
                indices = np.arange(split_index)
            else:  # test
                split_index = int(total_samples * 0.8)
                indices = np.arange(split_index, total_samples)
            
            if self.season != "all":
                indices = self._get_season_indices(indices)
            
            print(f"{self.split} indices: {indices[:10]}...{indices[-10:]}")
            print(f"Total {len(indices)} samples")

        elif self.dataopt == "interleaved":
            test_indices = np.round(np.linspace(0, total_samples - 1, 365)).astype(int)

            if self.split == "train":
                train_mask = np.ones(total_samples, dtype=bool)
                train_mask[test_indices] = False
                indices = train_mask
            else:  # test
                indices = test_indices
            print(f"{self.split} indices: {indices[:10]}...{indices[-10:]}")
            print(f"Total {len(indices)} samples")
        else:
            raise ValueError(f"Unknown dataopt: {self.dataopt}")

        self.forecast_data = self.forecast_data[indices, :self.leadtime]
        self.truth_data = self.truth_data[indices, :self.leadtime]
        self.rmm_data = self.rmm_data[indices, :self.leadtime]

        self.forecast_data = self._normalize_data(self.forecast_data)
        self.truth_data = self._normalize_data(self.truth_data)
        self.rmm_data = self._normalize_rmm(self.rmm_data)

    def _normalize_data(self, data):

        normalized_data = np.zeros_like(data)

        min_val, max_val = self.norm_params['olr']
        data_clipped = np.clip(data[:, :, 0], -200, None) if self.opt == 1 else data[:, :, 0]
        normalized_data[:, :, 0] = (data_clipped - min_val) / (max_val - min_val)

        min_val, max_val = self.norm_params['u850']
        normalized_data[:, :, 1] = (data[:, :, 1] - min_val) / (max_val - min_val)

        min_val, max_val = self.norm_params['u200']
        normalized_data[:, :, 2] = (data[:, :, 2] - min_val) / (max_val - min_val)

        return normalized_data

    def _denormalize_data(self, normalized_data):
        denormalized_data = np.zeros_like(normalized_data)
        
        # OLR反归一化 (通道0)
        min_val, max_val = self.norm_params['olr']
        denormalized_data[:, :, 0] = normalized_data[:, :, 0] * (max_val - min_val) + min_val
        
        # U850反归一化 (通道1)
        min_val, max_val = self.norm_params['u850']
        denormalized_data[:, :, 1] = normalized_data[:, :, 1] * (max_val - min_val) + min_val
        
        # U200反归一化 (通道2)
        min_val, max_val = self.norm_params['u200']
        denormalized_data[:, :, 2] = normalized_data[:, :, 2] * (max_val - min_val) + min_val
        
        return denormalized_data
    

    def _normalize_rmm(self, rmm_data):

        min_val = -4.271076202392578
        max_val = 3.9197659492492676
        return (rmm_data - min_val) / (max_val - min_val)

    def _denormalize_rmm(self, rmm_data):

        min_val = -4.271076202392578
        max_val = 3.9197659492492676
        return rmm_data * (max_val - min_val) + min_val

    def __len__(self):
        return self.forecast_data.shape[0]

    def __getitem__(self, idx):
        forecast = torch.tensor(self.forecast_data[idx], dtype=torch.float32)
        truth = torch.tensor(self.truth_data[idx], dtype=torch.float32)
        rmm = torch.tensor(self.rmm_data[idx], dtype=torch.float32)
        rmm = rmm.transpose(0, 1)

        return forecast, rmm, truth



# 主程序
if __name__ == "__main__":

    def randomseed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    randomseed(1)


    # 超参数

    root_path = '/home/sunyuze/08_mjo_corr_new'
    file_truth = os.path.join(root_path, 'data', 'truth.nc')

    batch_size = 8
    learning_rate = 0.001
    num_epochs = 300
    opt = 2
    a = 0.2
    b = 0.8
    dataopt = "sequential"  #sequential #interleaved
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    print("data load")
    test_dataset = MJODataset(opt=opt, dataopt=dataopt, split="test", season="all")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    forecast, rmm, truth = test_dataset[0]
    print(f"Sample shapes - Forecast: {forecast.shape}, Truth: {truth.shape}, RMM: {rmm.shape}")




    print('model init')  
    model_unet = UNet3D_4Layer(
        in_channels=3, 
        channels=[64, 128, 256, 512], 
        time_kernels=[3, 7, 15, 21],  # 前两层3天，后两层5天
        spatial_kernels=[(7,5), (5,3), 3, 3]  # 保持原来的空间卷积核设置
    ).to(device)


    model = MinimalLSTMCorrection(opt = opt, num_layers = 4, hidden_dim=32).to(device)


    unet_model_path = "/home/sunyuze/08_mjo_corr_new/model_1_new/model_68.pth"
    model_path = "/home/sunyuze/09_mjo_rmm_corr/model_save/model_2_pcc_loss/model_57.pth"
    model_unet.load_state_dict(torch.load(unet_model_path))
    model.load_state_dict(torch.load(model_path))
    print('load model from ',unet_model_path)
    print('load model from ',model_path)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    true_rmm_all = []
    true_var_all = []
    var_output_all = []
    rmm_output_all = []
    forecast_all = []
    rmm_forecast_all = []

    print('start testing')
    with torch.no_grad():
        for forecast, true_rmm, true_var  in test_dataloader:
            forecast = forecast.to(device)  
            true_rmm = true_rmm.to(device)         
            true_var = true_var.to(device)   

            var_outputs = model_unet(forecast[:,:,:,0:16,:])
            
            rmm_output = model(var_outputs)

            loss = criterion(rmm_output, true_rmm)   # First task loss


            loss = loss.cpu().numpy()

            var_outputs = var_outputs.cpu().numpy()
            rmm_output = rmm_output.cpu().numpy()
            true_rmm = true_rmm.cpu().numpy()
            true_var = true_var.cpu().numpy()
            forecast = forecast.cpu().numpy()



            true_rmm = np.array(true_rmm)
            rmm_output = np.array(rmm_output)
            var_outputs = np.array(var_outputs)



            var_outputs = test_dataset._denormalize_data(var_outputs)
            true_var = test_dataset._denormalize_data(true_var)
            forecast = test_dataset._denormalize_data(forecast)
            true_rmm = test_dataset._denormalize_rmm(true_rmm)
            rmm_output = test_dataset._denormalize_rmm(rmm_output)
            

            var_output_all.append(var_outputs[:,:,:,:,:])
            rmm_output_all.append(rmm_output[:,:])
            true_rmm_all.append(true_rmm[:,:])
            true_var_all.append(true_var[:,:,:,:,:])
            forecast_all.append(forecast[:,:,:,:,:])

    

    #算重建场RMSE
    var_output_all = np.concatenate(var_output_all, axis=0)
    rmm_output_all = np.concatenate(rmm_output_all, axis=0)
    true_rmm_all = np.concatenate(true_rmm_all, axis=0)
    true_var_all = np.concatenate(true_var_all, axis=0)
    forecast_all = np.concatenate(forecast_all, axis=0)

    var_output_all = np.array(var_output_all)
    rmm_output_all = np.array(rmm_output_all)
    true_rmm_all = np.array(true_rmm_all)
    true_var_all = np.array(true_var_all)
    forecast_all = np.array(forecast_all)

    print('var_output_all',var_output_all.shape)
    print('rmm_output_all',rmm_output_all.shape)
    print('true_rmm_all',true_rmm_all.shape)
    print('true_var_all',true_var_all.shape)
    print('forecast_all',forecast_all.shape)

    


    criterion = nn.MSELoss()
    # 原代码
    def test():
        # 1 读取EOF模态
        with xr.open_dataset("/home/sunyuze/08_mjo_corr_new/data/truth.nc") as ds:
            date = ds.coords['time'].values.astype('datetime64[D]')
            pattern = ds['pattern'].values.reshape(2, -1)
            std = ds['std'].values
            factor = ds['factor'].values
        with xr.open_dataset(test_dataset.file_truth) as ds:
            rmm = ds['rmm'].values

        # 2 计算评估值
        ans_corr = np.full((40, 1), np.nan, dtype=np.float64)
        ans_rmse = np.full((40, 1), np.nan, dtype=np.float64)
        ans_msss = np.full((40, 1), np.nan, dtype=np.float64)
        ans_bcor_forecast = np.full((40, 1), np.nan, dtype=np.float64)
        ans_bcor_correction = np.full((40, 1), np.nan, dtype=np.float64)
        with xr.open_dataset(test_dataset.forecast_file) as ds:
            # 初始化
            n = ds.sizes['time']
            ansx = np.full((len(test_dataset), 2, 40), np.nan, dtype=np.float64)
            ansy = np.full((len(test_dataset), 2, 40), np.nan, dtype=np.float64)


            data = forecast_all[:,:,:,1:16,:]  #var_output_all  #forecast_all
            data = np.nanmean(data, axis=3)
            data /= std[None, None, :, None]
            data = data.reshape((data.shape[0:2] + (-1,)))
            for k in range(len(test_dataset)):

                rmm_swap = true_rmm_all[k, :,:]
                rmm_swap = np.swapaxes(rmm_swap, -1, -2)
                # if k == 0:
                #     print(rmm_swap)
                ansx[k, :, :] = rmm_swap
                ansy[k, :, :] = np.matmul(pattern, data[k, :, :].T, dtype=np.float64) / factor[:, None]

            rmm_forecast_all = ansy.swapaxes(-1, -2)
            ansx -= np.mean(ansx, axis=0)[None, :, :]
            ansy -= np.mean(ansy, axis=0)[None, :, :]
            ansx /= np.std(ansx, axis=0)[None, :, :]
            ansy /= np.std(ansy, axis=0)[None, :, :]
            
            # corr = np.mean(np.sum(ansx * ansy, axis=1), axis=0) / np.sqrt(
            #     np.mean(np.sum(ansx * ansx, axis=1), axis=0) * np.mean(np.sum(ansy * ansy, axis=1), axis=0))

            ansx_reshaped = np.transpose(ansx, (0, 2, 1))  # 形状: b × 40 × 2
            ansy_reshaped = np.transpose(ansy, (0, 2, 1))  # 形状: b × 40 × 2

            rmse = np.sqrt(np.mean(np.sum((ansx_reshaped - ansy_reshaped) ** 2, axis=2), axis=0))

            pred_labels = np.argmax(ansx, axis=1) 
            true_labels = np.argmax(ansy, axis=1) 

            ansx_reshaped = np.transpose(ansx, (0, 2, 1))
            ansy_reshaped = np.transpose(ansy, (0, 2, 1))

            # 向量化计算每一天的相关系数
            numerator = np.mean(np.sum(ansx_reshaped * ansy_reshaped, axis=2), axis=0)  # 形状: 40
            denominator = np.sqrt(np.mean(np.sum(ansx_reshaped * ansx_reshaped, axis=2), axis=0) * 
                                np.mean(np.sum(ansy_reshaped * ansy_reshaped, axis=2), axis=0))  # 形状: 40

            corr = numerator / denominator  # 形状: 40





            # msss
            mse_f_t = np.mean(np.sum((ansx - ansy) ** 2, axis=1), axis=0)  # shape (40,)
    
            # MSE_c for each timestep: sum over the 2 variables, then average over batch  
            mse_c_t = np.mean(np.sum(ansx ** 2, axis=1), axis=0)           # shape (40,)
            
            # MSSS for each timestep
            msss = 1 - (mse_f_t / mse_c_t)



            # bcor

            def calculate_daily_cor(obs, forecast):
                b, t, _ = obs.shape
                
                daily_cor = []
                
                for day in range(t):
                    # 获取第day天的所有batch数据
                    O_day = obs[:, day, :]  # shape: (b, 2)
                    F_day = forecast[:, day, :]  # shape: (b, 2)
                    
                    # 按照公式计算分子
                    numerator = np.sum(O_day[:, 0] * F_day[:, 0] + O_day[:, 1] * F_day[:, 1])
                    
                    # 计算分母的两个部分
                    denom1 = np.sqrt(np.sum(O_day[:, 0]**2 + O_day[:, 1]**2))
                    denom2 = np.sqrt(np.sum(F_day[:, 0]**2 + F_day[:, 1]**2))
                    
                    # 计算COR
                    cor = numerator / (denom1 * denom2)
                    daily_cor.append(cor)
                
                return np.array(daily_cor)



            bcor_forecast = calculate_daily_cor(true_rmm_all, rmm_forecast_all)  # r12
            bcor_correction = calculate_daily_cor(true_rmm_all, rmm_output_all)  # r13
            bcor_fore_corr = calculate_daily_cor(rmm_forecast_all, rmm_output_all)  # r23

            print(bcor_forecast.shape)
            print(bcor_correction.shape)
            print(bcor_fore_corr.shape)

            # 参数设置
            n = 273  # 样本量
            # 定义三个置信水平的临界值
            confidence_levels = {
                '90%': 1.645,
                '95%': 1.96
            }

            # 存储结果
            results = []
            confidence_codes = []  # 存储置信水平代码

            # 对每一天进行计算
            for i in range(len(bcor_forecast)):
                r12 = bcor_forecast[i]    # 预测与真实的相关系数
                r13 = bcor_correction[i]  # 修正与真实的相关系数
                r23 = bcor_fore_corr[i]   # 预测与修正的相关系数
                
                # Fisher Z 变换
                Z12 = 0.5 * np.log((1 + r12) / (1 - r12))
                Z13 = 0.5 * np.log((1 + r13) / (1 - r13))
                
                # Z值差异 (比较r12和r13，即预测vs修正与真实的相关性)
                Z_diff = Z12 - Z13
                
                ES = np.sqrt(((n-3) * (1-(r12 ** 2 + r13 ** 2)/2) ** 2)/((1-r23) * (2-3*(r12 **2 + r13 ** 2)/2 + r23 * (r12 ** 2 + r13 ** 2)/2)))
                
                # 检验统计量
                Z_test = Z_diff * ES
                
                # 判断三个置信水平的显著性并确定置信水平代码
                # 0: 不显著, 1: 90%, 2: 95%
                if abs(Z_test) > confidence_levels['95%']:
                    conf_code = 2
                    highest_conf = '95%'
                elif abs(Z_test) > confidence_levels['90%']:
                    conf_code = 1
                    highest_conf = '90%'
                else:
                    conf_code = 0
                    highest_conf = '不显著'
                
                # 存储置信水平代码
                confidence_codes.append(conf_code)
                
                # 存储结果
                results.append({
                    'day': i + 1,
                    'r12_forecast': r12,
                    'r13_correction': r13,
                    'r23_fore_corr': r23,
                    'Z12': Z12,
                    'Z13': Z13,
                    'Z_diff': Z_diff,
                    'SE': ES,
                    'Z_test': Z_test,
                    'confidence_code': conf_code,
                    'highest_confidence': highest_conf
                })
                
                # 打印每天结果
                print(f"第{i+1:2d}天: r12={r12:6.3f}, r13={r13:6.3f}, r23={r23:6.3f}, Z_test={Z_test:7.3f}, 置信水平={conf_code} ({highest_conf})")

            # 竖着打印置信水平代码
            print(f"\n40天置信水平代码 (0:不显著, 1:90%, 2:95%):")
            print("天数: 置信水平代码")
            for i, conf_code in enumerate(confidence_codes):
                print(conf_code)

            # 统计各置信水平的天数
            count_0 = confidence_codes.count(0)
            count_1 = confidence_codes.count(1)
            count_2 = confidence_codes.count(2)
            count_3 = confidence_codes.count(3)

            print(f"\n统计总结:")
            print(f"不显著 (0): {count_0} 天 ({count_0/len(results)*100:.1f}%)")
            print(f"80%置信水平 (1): {count_1} 天 ({count_1/len(results)*100:.1f}%)")
            print(f"90%置信水平 (2): {count_2} 天 ({count_2/len(results)*100:.1f}%)")
            print(f"95%置信水平 (3): {count_3} 天 ({count_3/len(results)*100:.1f}%)")

            # 可选：将置信水平代码保存为数组
            confidence_array = np.array(confidence_codes)
            print(f"\n置信水平代码数组: {confidence_array}")





            del ansx, ansy
            # 存储数据
            ans_corr[:, 0] = corr
            ans_rmse[:, 0] = rmse
            ans_msss[:, 0] = msss
            ans_bcor_forecast[:, 0] = bcor_forecast
            ans_bcor_correction[:, 0] = bcor_correction
            del corr, rmse
        # 3 输出结果
        print("ans_bcor_forecast values:")
        for value in ans_bcor_forecast[:, 0]:
            print(f"{value:.6f}")

        print("ans_bcor_correction values:")
        for value in ans_bcor_correction[:, 0]:
            print(f"{value:.6f}")

        print("Correlation values:")
        for value in ans_corr[:, 0]:
            print(f"{value:.6f}")
            
        print("Correlation values:")
        for value in ans_rmse[:, 0]:
            print(f"{value:.6f}")


        print("msss:")
        for value in ans_msss[:, 0]:
            print(f"{value:.6f}")



    test()



    