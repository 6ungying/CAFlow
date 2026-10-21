# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import enum
import pandas as pd 
import cv2
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .jpeg import jpeg_decode, jpeg_encode
from .blur import Deblurring
from .superresolution import build_sr_bicubic, build_sr_pool
from .inpaint import get_center_mask, load_freeform_masks
from project_paths import DATA_DIR

from ipdb import set_trace as debug


DATA_ROOT = DATA_DIR


def _data_path(*parts):
    """Build a portable path under the repository's data directory."""
    return str(DATA_ROOT.joinpath(*parts))


class AllCorrupt(enum.IntEnum):
    JPEG_5 = 0
    JPEG_10 = 1
    BLUR_UNI = 2
    BLUR_GAUSS = 3
    SR4X_POOL = 4
    SR4X_BICUBIC = 5
    INPAINT_CENTER = 6
    INPAINT_FREE1020 = 7
    INPAINT_FREE2030 = 8

class MixtureCorruptMethod:
    def __init__(self, opt, device="cpu"):

        # ===== blur ====
        self.blur_uni = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, opt.image_size, device)

        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, device)

        # ===== sr4x ====
        factor = 4
        self.sr4x_pool = build_sr_pool(factor, device, opt.image_size)
        self.sr4x_bicubic = build_sr_bicubic(factor, device, opt.image_size)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')

        # ===== inpaint ====
        self.center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1, 1, 256, 256]
        self.free1020_masks = torch.from_numpy((load_freeform_masks("freeform1020"))) # [10000, 1, 256, 256]
        self.free2030_masks = torch.from_numpy((load_freeform_masks("freeform2030"))) # [10000, 1, 256, 256]

    def jpeg(self, img, qf):
        return jpeg_decode(jpeg_encode(img, qf), qf)

    def blur(self, img, kernel):
        img = (img + 1) / 2
        if kernel == "uni":
            _img = self.blur_uni.H(img).reshape(*img.shape)
        elif kernel == "gauss":
            _img = self.blur_gauss.H(img).reshape(*img.shape)
        # [0,1] -> [-1,1]
        return _img * 2 - 1

    def sr4x(self, img, filter):
        b, c, w, h = img.shape
        if filter == "pool":
            _img = self.sr4x_pool.H(img).reshape(b, c, w // 4, h // 4)
        elif filter == "bicubic":
            _img = self.sr4x_bicubic.H(img).reshape(b, c, w // 4, h // 4)

        # scale to original image size for I2SB
        return self.upsample(_img)

    def inpaint(self, img, mask_type, mask_index=None):
        if mask_type == "center":
            mask = self.center_mask
        elif mask_type == "free1020":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free1020_masks))
            mask = self.free1020_masks[[mask_index]]
        elif mask_type == "free2030":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free2030_masks))
            mask = self.free2030_masks[[mask_index]]
        return img * (1. - mask) + mask * torch.randn_like(img)

    def mixture(self, img, corrupt_idx, mask_index=None):
        if corrupt_idx == AllCorrupt.JPEG_5:
            corrupt_img = self.jpeg(img, 5)
        elif corrupt_idx == AllCorrupt.JPEG_10:
            corrupt_img = self.jpeg(img, 10)
        elif corrupt_idx == AllCorrupt.BLUR_UNI:
            corrupt_img = self.blur(img, "uni")
        elif corrupt_idx == AllCorrupt.BLUR_GAUSS:
            corrupt_img = self.blur(img, "gauss")
        elif corrupt_idx == AllCorrupt.SR4X_POOL:
            corrupt_img = self.sr4x(img, "pool")
        elif corrupt_idx == AllCorrupt.SR4X_BICUBIC:
            corrupt_img = self.sr4x(img, "bicubic")
        elif corrupt_idx == AllCorrupt.INPAINT_CENTER:
            corrupt_img = self.inpaint(img, "center")
        elif corrupt_idx == AllCorrupt.INPAINT_FREE1020:
            corrupt_img = self.inpaint(img, "free1020", mask_index=mask_index)
        elif corrupt_idx == AllCorrupt.INPAINT_FREE2030:
            corrupt_img = self.inpaint(img, "free2030", mask_index=mask_index)
        return corrupt_img


class MixtureCorruptDatasetTrain(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetTrain, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        rand_idx = np.random.choice(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), rand_idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y


class MixtureCorruptDatasetVal(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetVal, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        idx = index % len(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), idx, mask_index=idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y


class floodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(floodDataset, self).__init__()
        self.opt = opt
        self.dem_folder = _data_path('dems', 'dem_png')
        
        # CA4D 資料夾: 根據測試/訓練模式選擇不同資料集
        # - Multi: 多 DEM 訓練 (30 DEMs × 86 scenarios)
        # - single_train: 單 DEM 訓練 (1 DEM × 182 scenarios)
        # - single_test: 單 DEM 測試 (1 DEM × 15 scenarios)
        # [FIXED] 正確的路徑應該是 data/ca4d/ca4d/{dataset}
        if test:
            self.ca4d_folder = _data_path('ca4d', 'ca4d', 'Multi')
        else:
            # 訓練模式: 使用 Multi (多 DEM) 或 single_train (單 DEM)
            # 可以根據 opt.ca4d_dataset 參數選擇
            ca4d_dataset = getattr(opt, 'ca4d_dataset', 'Multi') if opt is not None else 'Multi'
            self.ca4d_folder = _data_path('ca4d', 'ca4d', ca4d_dataset)
        
        self.use_ca4d = getattr(opt, 'ca4d', False) if opt is not None else False

        self.dem_stat = _data_path('dems', 'dem_png', 'elevation_stats.csv')
        self.dem_stat = pd.read_csv(self.dem_stat)
        
        # ===== 整合 CSV 讀取：載入地形深度數據 =====
        # 從 maxmin_duv.csv 讀取每個地形的最大深度 metadata
        maxmin_csv_path = _data_path('maxmin_duv.csv')
        if os.path.exists(maxmin_csv_path):
            terrain_df = pd.read_csv(maxmin_csv_path)
            self.terrain_depths = dict(zip(terrain_df['terrain'].astype(int), terrain_df['depth_max']))
            print(f"[Dataset] [OK] Loaded terrain depths from CSV: {len(self.terrain_depths)} terrains")
        else:
            self.terrain_depths = {}
            print(f"[Dataset] [WARNING] maxmin_duv.csv not found, using default depth 4.0m")
        
        self.test = test

        self.flood_path = _data_path('dems', 'train', 'd')
        rainfall_path = _data_path('dems', 'scenario_rainfall.csv')
        
        # 使用所有可用的 DEM 資料夾進行訓練
        all_available_dems = [int(f) for f in os.listdir(self.flood_path) if os.path.isdir(os.path.join(self.flood_path, f))]
        
        # 取得 test_rain_list（測試用的雨型ID）
        test_rain_list = getattr(opt, 'test_rain_list', None) if opt is not None else None
        
        if test:
            # 測試模式: 優先使用命令列指定的 test_dem_list
            if hasattr(opt, 'test_dem_list') and opt.test_dem_list is not None:
                dem_folder = opt.test_dem_list
                print(f"[Test mode] Using specified DEMs: {dem_folder}")
            else:
                # 沒指定就使用預設
                dem_folder = [8, 29, 62]
                print(f"[Test mode] Using default test DEMs: {dem_folder}")
        else:
            # 訓練模式: 使用所有可用 DEM,但排除測試用的 DEM
            if hasattr(opt, 'test_dem_list') and opt.test_dem_list is not None:
                # 從所有可用 DEM 中排除測試 DEM
                dem_folder = [dem for dem in all_available_dems if dem not in opt.test_dem_list]
                print(f"[Training mode] Excluding test DEMs {opt.test_dem_list}")
                print(f"[Training mode] Using DEMs: {sorted(dem_folder)}")
            else:
                # 沒指定測試 DEM,使用全部
                dem_folder = all_available_dems
                print(f"[Training mode] Using all available DEMs: {sorted(dem_folder)}")
            
            # 打印雨型篩選信息
            if test_rain_list is not None:
                print(f"[Training mode] Excluding test rainfall scenarios: {test_rain_list}")

        rainfall = pd.read_csv(rainfall_path)
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []

        val = False
        # Iterate through each column
        for dem_num in dem_folder:
            for col in rainfall.columns:
                if col == 'time':
                    continue
                col_num = int(col.split("_")[1])
                
                # [ADDED] 在訓練模式排除測試雨型，在測試模式只選擇測試雨型
                if test:
                    # 測試模式: 如果指定了 test_rain_list，只使用這些雨型
                    if test_rain_list is not None and col_num not in test_rain_list:
                        continue
                else:
                    # 訓練模式: 如果指定了 test_rain_list，排除這些雨型
                    if test_rain_list is not None and col_num in test_rain_list:
                        continue
                
                if (val and col_num not in [2]) or (not val and col_num in []):
                    continue
                cell_values = []
                # Iterate through each row in the current column
                for row in range(25):
                    cell_value = rainfall.iloc[row][col]
                    cell_values.append(np.ceil(cell_value))
                    # make it a len 24 list if not append 0 in front
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    sum_rainfall = sum(temp[:])

                    # PIFF pairs CA4D_001 with TUFLOW_001; CA4D has no t=0 input.
                    if self.use_ca4d and row == 0:
                        continue
                    
                    if not test and sum_rainfall <= 5:
                        if np.random.rand() < 0.8:
                            continue
                    
                    skip_sample = False

                    # 只在啟用時才檢查 CA4D
                    if self.use_ca4d:
                        dem_str = str(dem_num)
                        # CA4D 使用小寫 rf (不是 RF)
                        rf_scenario = f"rf{col_num:02d}"
                        # CA4D timestep 從 001 開始 (row+1)
                        time_str = f"{row:03d}"
                        
                        # CA4D 資料夾結構: ca4d_folder/d/{DEM}/rf{scenario}/ca4d_{DEM}_rf{scenario}_d_{timestep}.png
                        path_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                                f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
                        if not os.path.exists(path_d):
                            skip_sample = True
                    
                    if skip_sample:
                        continue
                    
                    # 檢查 Ground Truth
                    dem_str = str(dem_num)
                    if col_num < 100:
                        folder_name = f"RF{col_num:02d}"
                    else:
                        folder_name = f"RF{col_num}"
                    
                    flood_name = f"{dem_str}_{folder_name}_d_{row:03d}_00.png"
                    flood_path = os.path.join(self.flood_path, dem_str, folder_name, flood_name)

                    if not os.path.exists(flood_path):
                        continue
                    
                    rainfall_cum_value.append(temp[:])
                    # col_num is the rainfall index, and row is the time index
                    cell_positions.append((dem_num, col_num, row))   

        # [MODIFIED] 在測試模式添加 t=0 初始狀態樣本（預測 0-24 小時）
        if test and not self.use_ca4d:
            # 提取唯一的 (dem, col) 組合
            unique_scenarios = {}
            for dem_num, col_num, row in cell_positions:
                key = (dem_num, col_num)
                if key not in unique_scenarios:
                    unique_scenarios[key] = True
            
            # 為每個情景添加一個 t=0 初始狀態樣本 (row=-1 標記)
            t0_samples = []
            t0_rainfall = []
            for (dem_num, col_num) in unique_scenarios.keys():
                first_rainfall = None
                for i, (d, c, r) in enumerate(cell_positions):
                    if d == dem_num and c == col_num:
                        first_rainfall = rainfall_cum_value[i]
                        break
                
                if first_rainfall is not None:
                    t0_samples.append((dem_num, col_num, -1))  # -1 標記為 t=0
                    t0_rainfall.append(first_rainfall)
            
            # 在開頭插入 t=0 樣本
            cell_positions = t0_samples + cell_positions
            rainfall_cum_value = t0_rainfall + rainfall_cum_value
            
            print(f"[Test mode] Added {len(t0_samples)} t=0 initial state samples")
            print(f"[Test mode] Total samples: {len(cell_positions)} (t=0 ~ t=24 小時)")
            
            # 列印測試配置信息
            if test_rain_list is not None:
                print(f"[Test mode] Using specified rainfall scenarios: {test_rain_list}")
        
        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        
        # [ADDED] 驗證並計算雨型篩選的效果
        if not test and test_rain_list is not None:
            # 計算原始應該有的樣本數（如果不篩選雨型）
            total_rainfall_scenarios = len([c for c in rainfall.columns if c != 'time'])
            excluded_rain_count = len(test_rain_list)
            included_rain_count = total_rainfall_scenarios - excluded_rain_count
            
            # 估算樣本統計
            total_dems_count = len(dem_folder)
            avg_samples_per_scenario = len(self.cell_positions) // (total_dems_count * included_rain_count + 1)  # +1 避免除以 0
            
            estimated_total_without_rain_filter = total_dems_count * total_rainfall_scenarios * avg_samples_per_scenario
            estimated_excluded_samples = estimated_total_without_rain_filter - len(self.cell_positions)
            
            print(f"[Training Statistics]")
            print(f"  Total rainfall scenarios: {total_rainfall_scenarios}")
            print(f"  Excluded rainfall scenarios: {excluded_rain_count} {test_rain_list}")
            print(f"  Included rainfall scenarios: {included_rain_count}")
            print(f"  Estimated samples excluded by rain filter: ~{estimated_excluded_samples // (total_dems_count + 1)} (~{excluded_rain_count/total_rainfall_scenarios*100:.1f}%)")
        
        # 打印訓練資料統計
        total_scenarios = len(dem_folder) * len([c for c in rainfall.columns if c != 'time']) * len(rainfall)
        
        # 新增訓練配置信息
        if not test and test_rain_list is not None:
            print(f"[Training mode] Excluding test rainfall scenarios: {test_rain_list}")
        
        print(f"[OK] Dataset initialized: {len(self.cell_positions)} valid samples")
        print(f"     I2SB x1: {'CA4D' if self.use_ca4d else 'None'}; conditioning: DEM")
        print(f"     (Filtered {total_scenarios - len(self.cell_positions)} samples with missing files)")
        
        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __find_flood_image(self, cell_position, flood_path):
        dem, col, row = cell_position
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        # 正確的檔案名稱格式: {dem}_RF{scenario}_d_{timestep}_00.png
        image_name = f"{dem_folder_name}_{folder_name}_d_{row:03d}_00.png"
        image_path = os.path.join(flood_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_dem_image(self, cell_position):
        dem_num = cell_position[0]
        # 從專案 data 根目錄定位 DEM
        dem_path = Path(self.dem_folder) / f'{dem_num}.png'
        return str(dem_path)

    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        cell_position = self.cell_positions[index]
        rainfall = self.rainfall[index]
        
        # [MODIFIED] 處理 t=0 初始狀態 (row == -1)
        if cell_position[2] == -1:
            # t=0 時刻：回傳全零狀態
            dem_path = self.__find_dem_image(cell_position)
            dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
            
            if dem_image is None:
                raise FileNotFoundError(f"Failed to load DEM image: {dem_path}")
            
            if len(dem_image.shape) == 3:
                dem_image = dem_image[:,:,0]
            
            dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
            min_elev = int(dem_cur_state['Min Elevation'].iloc[0])
            max_elev = int(dem_cur_state['Max Elevation'].iloc[0])
            real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
            dem_image = (real_height - (-3)) / (125 + 3) * 255
            dem_image = np.clip(dem_image, 0, 255)
            dem_image = np.array(dem_image, dtype=np.uint8)
            dem_image = self.transform(dem_image)
            dem_image = (dem_image - 0.18) / 0.22
            
            # 初始狀態：全零
            zero_image = torch.zeros((1, 256, 256))
            zero_rainfall = np.zeros(24, dtype=np.int64)
            ca4d_image = None
            
            max_depth = self.terrain_depths.get(cell_position[0], 4.0)
            dem_id = cell_position[0]
            
            return (
                zero_image, dem_image, None, zero_rainfall,
                "t0_initial", ca4d_image, max_depth, dem_id,
            )
        
        # 正常邏輯（t=1~24）
        dem_path = self.__find_dem_image(cell_position)
        
        # ===== 獲取地形深度 metadata =====
        dem_id = cell_position[0]
        max_depth = self.terrain_depths.get(dem_id, 4.0)  # 預設 4.0m
        
        # 讀取 DEM
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM image: {dem_path}")
        
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:,:,0]
        
        dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
        min_elev = int(dem_cur_state['Min Elevation'].iloc[0])
        max_elev = int(dem_cur_state['Max Elevation'].iloc[0])
        real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
        dem_image = (real_height - (-3)) / (125 + 3) * 255
        dem_image = np.clip(dem_image, 0, 255)
        dem_image = np.array(dem_image, dtype=np.uint8)
        
        # Load paired CA4D depth as x1.
        dem_str = str(cell_position[0])
        col_num = cell_position[1]
        row_num = cell_position[2]
        
        # 初始化為 None
        ca4d_image = None
        
        # 只在啟用時才載入 CA4D
        if self.use_ca4d:
            # CA4D 格式: ca4d_{DEM}_rf{scenario}_{channel}_{timestep}.png
            # 注意: scenario 是小寫 rf (不是 RF), 且沒有 _00 後綴
            rf_scenario = f"rf{col_num:02d}"  # 小寫 rf
            time_str = f"{row_num:03d}"   # timestep (001, 002, ...)
            
            # CA4D 資料夾結構: ca4d_folder/d/{DEM}/rf{scenario}/ca4d_{DEM}_rf{scenario}_d_{timestep}.png
            path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario, 
                                           f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")
            ca4d_d = cv2.imread(path_ca4d_d, cv2.IMREAD_GRAYSCALE)
            if ca4d_d is None:
                raise FileNotFoundError(f"CA4D depth image not found: {path_ca4d_d}")

            ca4d_d = self.transform(ca4d_d)
            ca4d_image = (ca4d_d - 0.986) / 0.0405

        # 讀取 Ground Truth
        rainfall = np.array(rainfall, dtype=np.int64)

        flood_path = self.__find_flood_image(cell_position, self.flood_path)

        flood_image = cv2.imread(flood_path, cv2.IMREAD_UNCHANGED)

        if flood_image is None:
             raise FileNotFoundError(f"TUFLOW depth image not found: {flood_path}")
        
        # [MODIFIED] Mask 設為 None
        # flood_binary_mask = (flood_image <= 250).astype('uint8') ...
        flood_binary_mask = None
        # Convert to proper dtype
        flood_image = np.array(flood_image, dtype=np.uint8)

        # Apply transforms
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)

        # Normalize
        dem_image = (dem_image - 0.18) / 0.22
        flood_image = (flood_image - 0.986) / 0.0405
        
        return (
            flood_image, dem_image, flood_binary_mask, rainfall,
            flood_path, ca4d_image, max_depth, dem_id,
        )

class singleDEMFloodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(singleDEMFloodDataset, self).__init__()
        self.opt = opt
        self.test = test
        
        self.use_ca4d = getattr(opt, 'ca4d', False) if opt is not None else False
        
        dem_path = _data_path('tainan_dem.png')
        self.dem = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        
        # [MODIFIED] CA4D Folder - 使用 ca4d 資料夾結構
        if test:
            self.ca4d_folder = _data_path('ca4d', 'single_test')
        else:
            # 訓練模式使用 single_train
            self.ca4d_folder = _data_path('ca4d', 'single_train')

        self.flood_path = _data_path('train', 'depth')

        rainfall_path = _data_path('train_rf.csv')
        if test:
            rainfall_path = _data_path('test_rf.csv')
            self.flood_path = _data_path('test', 'depth')

        rainfall = pd.read_csv(rainfall_path)
        rainfall = rainfall.iloc[:, :]

        rainfall_cum_value = []
        cell_positions = []
        
        val = False
        for col in rainfall.columns:
            if col == 'time':
                continue
            col_num = int(col.split("_")[1])
            if (val and col_num not in [2]) or (not val and col_num in []):
                continue
            cell_values = []
            for row in range(len(rainfall)):
                cell_value = rainfall.iloc[row][col]
                cell_values.append(np.ceil(cell_value))
                temp = [0] * (24 - len(cell_values))
                temp.extend(cell_values)
                if len(temp) == 25:
                    temp = temp[1:]

                sum_rainfall = sum(temp[:])
                # PIFF pairs CA4D_001 with TUFLOW_001; CA4D has no t=0 input.
                if self.use_ca4d and row == 0:
                    continue
                dem_num = 1
                dem_str = str(dem_num)
                rf_scenario = f"rf{col_num:02d}"
                time_str = f"{row:03d}"
                
                path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                           f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")

                if self.use_ca4d and not os.path.exists(path_ca4d_d):
                    continue
                
                # 檢查 Ground Truth
                if col_num < 100:
                    folder_name = f"RF{col_num:02d}"
                else:
                    folder_name = f"RF{col_num}"
                
                flood_name = f"{folder_name}_d_{row:03d}_00.png"
                flood_file = os.path.join(self.flood_path, folder_name, flood_name)

                if not os.path.exists(flood_file):
                    continue
                
                # [RESTORED] 所有檔案都存在，記錄此樣本
                rainfall_cum_value.append(temp)
                cell_positions.append((col_num, row))

        # [MODIFIED] 在測試模式添加 t=0 初始狀態樣本（預測 0-24 小時）
        if test and not self.use_ca4d:
            # 提取唯一的入流點
            unique_cols = set()
            for col_num, row in cell_positions:
                unique_cols.add(col_num)
            
            # 為每個入流點添加一個 t=0 初始狀態樣本 (row=-1 標記)
            t0_samples = []
            t0_rainfall = []
            for col_num in sorted(unique_cols):
                first_rainfall = None
                for i, (c, r) in enumerate(cell_positions):
                    if c == col_num:
                        first_rainfall = rainfall_cum_value[i]
                        break
                
                if first_rainfall is not None:
                    t0_samples.append((col_num, -1))  # -1 標記為 t=0
                    t0_rainfall.append(first_rainfall)
            
            # 在開頭插入 t=0 樣本
            cell_positions = t0_samples + cell_positions
            rainfall_cum_value = t0_rainfall + rainfall_cum_value
            
            print(f"[Test mode] Added {len(t0_samples)} t=0 initial state samples")
            print(f"[Test mode] Total samples: {len(cell_positions)} (t=0 ~ t=24 小時)")
        
        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions
        print(f"singleDEMFloodDataset initialized: {len(self.cell_positions)} samples")
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __find_image(self, cell_position, flood_path):
        col, row = cell_position
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                folder_name = f"RF{col:02d}"
            else:
                folder_name = f"RF{col}"
        image_name = f"{folder_name}_d_{row:03d}_00.png"
        image_path = os.path.join(flood_path, folder_name, image_name)
        return image_path
    
    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        dem_num = 1  # singleDEM 固定為 1
        dem_str = str(dem_num)
        
        cell_position = self.cell_positions[index]
        rainfall = self.rainfall[index]
        rainfall = np.array(rainfall, dtype=np.int64)
        
        col_num = cell_position[0]
        row_num = cell_position[1]
        
        # [MODIFIED] 處理 t=0 初始狀態 (row == -1)
        if row_num == -1:
            # t=0 時刻：回傳全零狀態
            dem_image = self.dem
            dem_image = self.transform(dem_image)
            dem_image = (dem_image - 0.547) / 0.282
            
            # 初始狀態：全零
            zero_image = torch.zeros((1, 256, 256))
            zero_rainfall = np.zeros(24, dtype=np.int64)
            ca4d_image = None
            
            max_depth = 4.0
            dem_id = 1
            
            return (
                zero_image, dem_image, None, zero_rainfall,
                "t0_initial", ca4d_image, max_depth, dem_id,
            )
        
        # 正常邏輯（t=1~24）
        # ===== 讀取 DEM =====
        dem_image = self.dem
        dem_image = self.transform(dem_image)
        dem_image = (dem_image - 0.547) / 0.282
        
        # ===== 讀取 CA4D 水深，作為 I2SB x1 =====
        rf_scenario = f"rf{col_num:02d}"
        time_str = f"{row_num:03d}"
        
        path_ca4d_d = os.path.join(self.ca4d_folder, 'd', dem_str, rf_scenario,
                                   f"ca4d_{dem_str}_{rf_scenario}_d_{time_str}.png")

        ca4d_image = None
        if self.use_ca4d:
            ca4d_d = cv2.imread(path_ca4d_d, cv2.IMREAD_GRAYSCALE)
            if ca4d_d is None:
                raise FileNotFoundError(f"CA4D depth image not found: {path_ca4d_d}")
            ca4d_d = self.transform(ca4d_d)
            ca4d_image = (ca4d_d - 0.987) / 0.0343
        
        # ===== 讀取 Ground Truth depth =====
        if col_num < 100:
            folder_name = f"RF{col_num:02d}"
        else:
            folder_name = f"RF{col_num}"
        
        image_path = self.__find_image(cell_position, self.flood_path)

        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if flood_image is None:
            raise FileNotFoundError(f"TUFLOW depth image not found: {image_path}")

        # Mask 設為 None
        binary_mask = None
        flood_image = np.array(flood_image, dtype=np.uint8)
        
        # 轉換為 Tensor
        flood_image = self.transform(flood_image)

        # 標準化
        flood_image = (flood_image - 0.987) / 0.0343

        max_depth = 4.0
        dem_id = 1

        return (
            flood_image, dem_image, binary_mask, rainfall,
            image_path, ca4d_image, max_depth, dem_id,
        )


class yilanDataset(Dataset):
    def __init__(self, opt, test_dem_list=None):
        super(yilanDataset, self).__init__()
        self.opt = opt
        
        # ===== 模型配置標誌 =====
        self.use_ca4d = getattr(opt, 'ca4d', False) if opt is not None else False
        
        # 資料根路徑
        base_path = _data_path('yilan_new')
        
        # ===== 路徑設定 =====
        self.dem_folder = os.path.join(base_path, 'dem')
        self.tuflow_d_folder = os.path.join(base_path, 'tuflow', 'd')
        self.ca4d_folder = os.path.join(base_path, 'ca4d')
        
        # ===== 載入統計資訊 =====
        maxmin_dem_path = os.path.join(base_path, 'maxmin_dem.csv')
        self.dem_stats = pd.read_csv(maxmin_dem_path)
        dem_dict = dict(zip(self.dem_stats['terrain'].astype(str), 
                           zip(self.dem_stats['min_elevation'], self.dem_stats['max_elevation'])))
        
        maxmin_duv_path = os.path.join(base_path, 'maxmin_duv.csv')
        duv_stats = pd.read_csv(maxmin_duv_path)
        self.terrain_depths = dict(zip(duv_stats['terrain'].astype(str), duv_stats['depth_max']))
        print(f"[yilanDataset] Loaded terrain depths: {self.terrain_depths}")
        
        rainfall_path = os.path.join(base_path, 'scenario_rainfall.csv')
        rainfall_df = pd.read_csv(rainfall_path)
        
        # ===== [FIXED] 與 floodDataset 一致的降雨處理邏輯 =====
        self.cell_positions = []  # (dem_name, rf_scenario, timestep)
        self.rainfall_values = []  # [FIXED] 儲存 24 步序列 (與 floodDataset 一致)
        self.dem_info = {}
        
        dem_names = ['yilan01', 'yilan02', 'yilan03']
        
        for dem_name in dem_names:
            dem_png_path = os.path.join(self.dem_folder, f'{dem_name}.png')
            if not os.path.exists(dem_png_path):
                print(f"[WARNING] DEM not found: {dem_png_path}")
                continue
            
            if dem_name in dem_dict:
                min_elev, max_elev = dem_dict[dem_name]
                self.dem_info[dem_name] = {'min': min_elev, 'max': max_elev}
            
            # ===== [FIXED] 與 floodDataset 相同的遍歷邏輯 =====
            for col_idx, col in enumerate(rainfall_df.columns):
                if col == 'time':
                    continue
                
                try:
                    flow_num = int(col.split('_')[1])
                except:
                    continue
                
                # [FIXED] 逐時間步遍歷並累積降雨
                cell_values = []
                for row_idx in range(25):
                    # 取得該時間步的降雨值
                    cell_value = rainfall_df.iloc[row_idx][col]
                    
                    # [FIXED] 向上取整 (與 floodDataset 一致)
                    cell_values.append(np.ceil(cell_value))
                    
                    # [FIXED] 構造 24 步序列 (與 floodDataset 相同)
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    
                    if self.use_ca4d:
                        if row_idx == 0:
                            continue
                        ca4d_depth_path = self.__find_ca4d_depth(dem_name, flow_num, row_idx)
                        if not os.path.exists(ca4d_depth_path):
                            continue

                    # 儲存樣本
                    self.cell_positions.append((dem_name, flow_num, row_idx))
                    self.rainfall_values.append(np.array(temp[:], dtype=np.int64))  # [FIXED] 儲存 24 步序列
        
        print(f"[yilanDataset] Initialized with {len(self.cell_positions)} samples")
        print(f"              Terrain: yilan1, yilan2, yilan3")
        print(f"              Scenarios: 10 inflow points × 25 timesteps (0-24 hours)")
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.cell_positions)
    
    def __find_dem_image(self, dem_name):
        """根據 DEM 名稱找到對應的 DEM 圖像路徑"""
        dem_path = os.path.join(self.dem_folder, f'{dem_name}.png')
        return dem_path
    
    def __find_tuflow_depth(self, dem_name, flow_num, timestep):
        """找到 TUFLOW 水深真值圖像。"""
        filename = f"{dem_name}_RF{flow_num:02d}_d_{timestep:03d}_00.png"
        rf_folder = f"RF{flow_num:02d}"
        image_path = os.path.join(self.tuflow_d_folder, dem_name, rf_folder, filename)
        return image_path
    
    def __find_ca4d_depth(self, dem_name, flow_num, timestep):
        """找到作為 I2SB x1 的 CA4D 水深圖像。"""
        rf_scenario = f"rf{flow_num:02d}"
        time_str = f"{timestep:03d}"
        return os.path.join(
            self.ca4d_folder, 'd', dem_name, rf_scenario,
            f"ca4d_{dem_name}_{rf_scenario}_d_{time_str}.png",
        )
    
    def __getitem__(self, index):
        dem_name, flow_num, timestep = self.cell_positions[index]
        rainfall_array = self.rainfall_values[index]  # [FIXED] 取得 24 步序列
        
        # ===== 讀取 DEM =====
        dem_path = self.__find_dem_image(dem_name)
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM: {dem_path}")
        
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:, :, 0]
        
        if dem_name in self.dem_info:
            min_elev = self.dem_info[dem_name]['min']
            max_elev = self.dem_info[dem_name]['max']
            elev_range = max_elev - min_elev if max_elev != min_elev else 1
            real_height = dem_image / 255.0 * elev_range + min_elev
            dem_normalized = (real_height - (-3)) / (125 + 3) * 255
            dem_image = np.clip(dem_normalized, 0, 255).astype(np.uint8)
        
        # ===== 讀取 TUFLOW 真值 =====
        d_path = self.__find_tuflow_depth(dem_name, flow_num, timestep)
        flood_image = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)

        if flood_image is None:
            raise FileNotFoundError(f"TUFLOW depth image not found: {d_path}")
        
        flood_image = np.array(flood_image, dtype=np.uint8)
        
        # ===== 讀取 CA4D 水深，作為 I2SB x1 =====
        ca4d_image = None
        if self.use_ca4d:
            ca4d_path = self.__find_ca4d_depth(dem_name, flow_num, timestep)
            ca4d_d = cv2.imread(ca4d_path, cv2.IMREAD_GRAYSCALE)
            if ca4d_d is None:
                raise FileNotFoundError(f"CA4D depth image not found: {ca4d_path}")
            ca4d_d = self.transform(ca4d_d)
            ca4d_image = (ca4d_d - 0.9880) / 0.0279
        
        # ===== 影像標準化 =====
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        
        dem_image = (dem_image - 0.1623) / 0.1836
        flood_image = (flood_image - 0.9880) / 0.0279
        
        max_depth = self.terrain_depths.get(dem_name, 4.0)
        dem_id = int(dem_name.replace('yilan', ''))
        
        return (
            flood_image, dem_image, None, rainfall_array,
            d_path, ca4d_image, max_depth, dem_id,
        )
    
class hsinchuDataset(Dataset):
    def __init__(self, opt, test_dem_list=None):
        super(hsinchuDataset, self).__init__()
        self.opt = opt
        
        # ===== 模型配置標誌 =====
        self.use_ca4d = getattr(opt, 'ca4d', False) if opt is not None else False
        
        # 資料根路徑
        base_path = _data_path('hsinchu')
        
        # ===== 路徑設定 =====
        self.dem_folder = os.path.join(base_path, 'dem')
        self.tuflow_d_folder = os.path.join(base_path, 'tuflow', 'd')
        self.ca4d_folder = os.path.join(base_path, 'ca4d')
        
        # ===== 載入統計資訊 =====
        maxmin_dem_path = os.path.join(base_path, 'maxmin_dem.csv')
        self.dem_stats = pd.read_csv(maxmin_dem_path)
        dem_dict = dict(zip(self.dem_stats['terrain'].astype(str), 
                           zip(self.dem_stats['min_elevation'], self.dem_stats['max_elevation'])))
        
        maxmin_duv_path = os.path.join(base_path, 'maxmin_duv.csv')
        duv_stats = pd.read_csv(maxmin_duv_path)
        self.terrain_depths = dict(zip(duv_stats['terrain'].astype(str), duv_stats['depth_max']))
        print(f"[yilanDataset] Loaded terrain depths: {self.terrain_depths}")
        
        rainfall_path = os.path.join(base_path, 'scenario_rainfall.csv')
        rainfall_df = pd.read_csv(rainfall_path)
        
        # ===== [FIXED] 與 floodDataset 一致的降雨處理邏輯 =====
        self.cell_positions = []  # (dem_name, rf_scenario, timestep)
        self.rainfall_values = []  # [FIXED] 儲存 24 步序列 (與 floodDataset 一致)
        self.dem_info = {}
        
        dem_names = ['hsinchu01']
        
        for dem_name in dem_names:
            dem_png_path = os.path.join(self.dem_folder, f'{dem_name}.png')
            if not os.path.exists(dem_png_path):
                print(f"[WARNING] DEM not found: {dem_png_path}")
                continue
            
            if dem_name in dem_dict:
                min_elev, max_elev = dem_dict[dem_name]
                self.dem_info[dem_name] = {'min': min_elev, 'max': max_elev}
            
            # ===== [FIXED] 與 floodDataset 相同的遍歷邏輯 =====
            for col_idx, col in enumerate(rainfall_df.columns):
                if col == 'time':
                    continue
                
                try:
                    flow_num = int(col.split('_')[1])
                except:
                    continue
                
                # [FIXED] 逐時間步遍歷並累積降雨
                cell_values = []
                for row_idx in range(25):
                    # 取得該時間步的降雨值
                    cell_value = rainfall_df.iloc[row_idx][col]
                    
                    # [FIXED] 向上取整 (與 floodDataset 一致)
                    cell_values.append(np.ceil(cell_value))
                    
                    # [FIXED] 構造 24 步序列 (與 floodDataset 相同)
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    
                    if self.use_ca4d:
                        if row_idx == 0:
                            continue
                        ca4d_depth_path = self.__find_ca4d_depth(dem_name, flow_num, row_idx)
                        if not os.path.exists(ca4d_depth_path):
                            continue

                    # 儲存樣本
                    self.cell_positions.append((dem_name, flow_num, row_idx))
                    self.rainfall_values.append(np.array(temp[:], dtype=np.int64))  # [FIXED] 儲存 24 步序列
        
        print(f"[yilanDataset] Initialized with {len(self.cell_positions)} samples")
        print(f"              Terrain: yilan1, yilan2, yilan3")
        print(f"              Scenarios: 10 inflow points × 25 timesteps (0-24 hours)")
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.cell_positions)
    
    def __find_dem_image(self, dem_name):
        """根據 DEM 名稱找到對應的 DEM 圖像路徑"""
        dem_path = os.path.join(self.dem_folder, f'{dem_name}.png')
        return dem_path
    
    def __find_tuflow_depth(self, dem_name, flow_num, timestep):
        """找到 TUFLOW 水深真值圖像。"""
        filename = f"{dem_name}_RF{flow_num:02d}_d_{timestep:03d}_00.png"
        rf_folder = f"RF{flow_num:02d}"
        image_path = os.path.join(self.tuflow_d_folder, dem_name, rf_folder, filename)
        return image_path
    
    def __find_ca4d_depth(self, dem_name, flow_num, timestep):
        """找到作為 I2SB x1 的 CA4D 水深圖像。"""
        rf_scenario = f"rf{flow_num:02d}"
        time_str = f"{timestep:03d}"
        return os.path.join(
            self.ca4d_folder, 'd', dem_name, rf_scenario,
            f"ca4d_{dem_name}_{rf_scenario}_d_{time_str}.png",
        )
    
    def __getitem__(self, index):
        dem_name, flow_num, timestep = self.cell_positions[index]
        rainfall_array = self.rainfall_values[index]  # [FIXED] 取得 24 步序列
        
        # ===== 讀取 DEM =====
        dem_path = self.__find_dem_image(dem_name)
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)
        
        if dem_image is None:
            raise FileNotFoundError(f"Failed to load DEM: {dem_path}")
        
        if len(dem_image.shape) == 3:
            dem_image = dem_image[:, :, 0]
        
        if dem_name in self.dem_info:
            min_elev = self.dem_info[dem_name]['min']
            max_elev = self.dem_info[dem_name]['max']
            elev_range = max_elev - min_elev if max_elev != min_elev else 1
            real_height = dem_image / 255.0 * elev_range + min_elev
            dem_normalized = (real_height - (-3)) / (125 + 3) * 255
            dem_image = np.clip(dem_normalized, 0, 255).astype(np.uint8)
        
        # ===== 讀取 TUFLOW 真值 =====
        d_path = self.__find_tuflow_depth(dem_name, flow_num, timestep)
        flood_image = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)

        if flood_image is None:
            raise FileNotFoundError(f"TUFLOW depth image not found: {d_path}")
        
        flood_image = np.array(flood_image, dtype=np.uint8)
        
        # ===== 讀取 CA4D 水深，作為 I2SB x1 =====
        ca4d_image = None
        if self.use_ca4d:
            ca4d_path = self.__find_ca4d_depth(dem_name, flow_num, timestep)
            ca4d_d = cv2.imread(ca4d_path, cv2.IMREAD_GRAYSCALE)
            if ca4d_d is None:
                raise FileNotFoundError(f"CA4D depth image not found: {ca4d_path}")
            ca4d_d = self.transform(ca4d_d)
            ca4d_image = (ca4d_d - 0.9751028) / 0.0256081
        
        # ===== 影像標準化 =====
        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)
        
        dem_image = (dem_image - 0.279147398) / 0.14481558
        flood_image = (flood_image - 0.9751028) / 0.0256081

        max_depth = self.terrain_depths.get(dem_name, 4.0)
        dem_id = int(dem_name.replace('hsinchu', ''))
        
        return (
            flood_image, dem_image, None, rainfall_array,
            d_path, ca4d_image, max_depth, dem_id,
        )
