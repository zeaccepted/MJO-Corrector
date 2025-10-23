#!/usr/bin/env python3
# coding=utf-8
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import os
from network.model_1 import UNet3D_4Layer
import xarray as xr
import random


class MJODataset(Dataset):
    def __init__(self, opt=0, dataopt="sequential", split="train", leadtime=40):

        self.root_path = '/home/sunyuze/08_mjo_corr_new'
        self.file_truth = os.path.join(self.root_path, 'data', 'truth.nc')

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
        ds_forecast.close()

        ds_truth = xr.open_dataset(self.file_truth)
        self.truth_data = np.stack([
            ds_truth['olr'].values,
            ds_truth['u'][:, 0, :, :].values,
            ds_truth['u'][:, 2, :, :].values
        ], axis=1)
        truth_times = ds_truth['time'].values
        self.rmm_data = ds_truth['rmm'].values
        self.amp_data = ds_truth['amp'].values
        ds_truth.close()

        self.truth_data = self._expand_truth_to_forecast_shape(self.truth_data, truth_times, forecast_times)
        self.rmm_data = self._expand_rmm_to_forecast_shape(self.rmm_data, truth_times, forecast_times)
        self.amp_data = self._expand_amp_to_forecast_shape(self.amp_data, truth_times, forecast_times)


        print(f"Forecast: {self.forecast_data.shape}, Truth: {self.truth_data.shape}, rmm: {self.rmm_data.shape}, amp:{self.amp_data.shape}")

    def _expand_truth_to_forecast_shape(self, truth_data, truth_times, forecast_times):
        expanded_truth = []

        for i, forecast_time in enumerate(forecast_times):
            start_idx = np.where(truth_times == forecast_time)[0][0]
            if i == 0 or i == 1:
                print(start_idx)
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
            if i == 0 or i == 1:
                print(start_idx)
            start_idx = start_idx[0]
            end_idx = start_idx + self.leadtime

            if end_idx <= len(truth_times):
                rmm_sequence = rmm_data[:, start_idx:end_idx]  # shape: (leadtime, 2)
                expanded_rmm.append(rmm_sequence)

        return np.array(expanded_rmm)  # shape: (b, t, 2)

    def _expand_amp_to_forecast_shape(self, amp_data, truth_times, forecast_times):
        expanded_amp = []

        for i, forecast_time in enumerate(forecast_times):
            start_idx = np.where(truth_times == forecast_time)[0]
            if len(start_idx) == 0:
                continue
            if i == 0 or i == 1:
                print(start_idx)
            start_idx = start_idx[0]
            end_idx = start_idx + self.leadtime

            if end_idx <= len(truth_times):
                amp_sequence = amp_data[start_idx:end_idx]  # shape: (leadtime, 2)
                expanded_amp.append(amp_sequence)

        return np.array(expanded_amp)  # shape: (b, t, 2)

    def _split_dataset(self):
        total_samples = self.forecast_data.shape[0]

        if self.dataopt == "sequential":
            if self.split == "train":
                split_index = int(total_samples * 0.8)
                indices = np.arange(split_index)
            else: 
                split_index = int(total_samples * 0.8)
                indices = np.arange(split_index, total_samples)
            
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
        self.amp_data = self.amp_data[indices, :self.leadtime]

        self.forecast_data = self._normalize_data(self.forecast_data)
        self.truth_data = self._normalize_data(self.truth_data)
        self.rmm_data = self._normalize_rmm(self.rmm_data)
        self.amp_data = self._normalize_amp(self.amp_data)

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

    def _normalize_rmm(self, rmm_data):

        min_val = -3.5036380290985107
        max_val = 3.5641696453094482
        return (rmm_data - min_val) / (max_val - min_val)

    def _normalize_amp(self, rmm_data):

        min_val = 0
        max_val = 4.3
        return (rmm_data - min_val) / (max_val - min_val)

    def __len__(self):
        return self.forecast_data.shape[0]

    def __getitem__(self, idx):
        forecast = torch.tensor(self.forecast_data[idx], dtype=torch.float32)
        truth = torch.tensor(self.truth_data[idx], dtype=torch.float32)
        rmm = torch.tensor(self.rmm_data[idx], dtype=torch.float32)
        amp = torch.tensor(self.amp_data[idx], dtype=torch.float32)
        rmm = rmm.transpose(0, 1)

        return forecast, rmm, truth, amp








def train(model, dataloader, optimizer, criterion, device ,a, b):
    model.train()
    running_loss = 0.0
    for forecast, true_rmm, true_var, amp  in dataloader:
        forecast = forecast.to(device)  
        true_rmm = true_rmm.to(device)         
        true_var = true_var.to(device)   
        amp = amp.to(device)   

        optimizer.zero_grad()
        var_output = model(forecast[:,:,:,0:16,:])
        loss = criterion(var_output, true_var[:,:,:,0:16,:])   # First task loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def test(model, dataloader, criterion, device, a, b):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for forecast, true_rmm, true_var, amp  in dataloader:
            forecast = forecast.to(device)  
            true_rmm = true_rmm.to(device)         
            true_var = true_var.to(device)  
            amp = amp.to(device)    

            optimizer.zero_grad()
            
            var_output = model(forecast[:,:,:,0:16,:])
            loss = criterion(var_output, true_var[:,:,:,0:16,:])

            running_loss += loss.item()
    return running_loss / len(dataloader)




if __name__ == "__main__":

    def randomseed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    randomseed(1)


    root_path = '/home/sunyuze/08_mjo_corr_new'
    file_truth = os.path.join(root_path, 'data', 'truth.nc')

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 300
    opt = 0
    a = 1.0
    b = 0.0
    dataopt = "sequential"  #sequential #interleaved
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    print("data load")
    train_dataset = MJODataset(opt=opt, dataopt=dataopt, split="train")
    test_dataset = MJODataset(opt=opt, dataopt=dataopt, split="test")


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    forecast, rmm, truth, amp= train_dataset[0]
    print(f"Sample shapes - Forecast: {forecast.shape}, Truth: {truth.shape}, RMM: {rmm.shape}")




    print('model init')
    
    model = UNet3D_4Layer(
        in_channels=3, 
        channels=[64, 128, 256, 512], 
        time_kernels=[3, 7, 15, 21], 
        spatial_kernels=[(5,7), (3,5), 3, 3] 
    ).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    print('start training')
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device, a, b)
        test_loss = test(model, test_dataloader, criterion, device, a, b)


        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        model_save_dir = f'/home/sunyuze/08_mjo_corr_new/model_1'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'model_{epoch}.pth')

        torch.save(model.state_dict(), model_save_path)

        print(f'model saved to {model_save_path}')

    print('finish')
