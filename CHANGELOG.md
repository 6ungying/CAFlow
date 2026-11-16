# 變更日誌

## 2026-11-16

### 訓練與推論

- 統一使用離散 I2SB residual 訓練目標與 posterior 採樣，檢查 timestep 索引，移除舊的 continuous 採樣路徑。
- 新增訓練設定契約，檢查 checkpoint 的目標、資料分割及驗證設定；不相容的舊 checkpoint 需重新訓練。
- 改善梯度累積，依實際有效樣本數平均，將網路與降雨嵌入一起裁剪梯度及更新 EMA，防止非有限數值造成無效更新。
- 依完整驗證集的水深 RMSE 儲存 best.pt，另保留 latest.pt、訓練步數、optimizer、scheduler、EMA 與最佳分數供續訓。
- 加入訓練參數與既有實驗目錄檢查，目前限定單 GPU、FP32、256×256 pixel-space；預設 batch size 32、microbatch 4。
- 推論預設使用 best.pt 與 I2SB 採樣，繼承 checkpoint 的資料模式和水深尺度，檢查降雨 token 範圍，避免多 DEM 模式默默改用單 DEM 資料。

### 資料分割與驗證

- 新增可重現的降雨事件訓練／驗證分割，支援指定驗證事件、比例與 seed，檢查事件重疊與空集合。
- 新增以公尺計算、依像素加權的 MAE／RMSE，評估模型及 CA4D 基準，涵蓋驗證集最後一個不足批次。
- 驗證採固定隨機種子並保留訓練 RNG 狀態；多 DEM 要求水深校正資料，單 DEM 可設定最大水深。
- 配對 CA4D 資料缺漏時明確報錯，不再以零值補齊。

### Windows 環境與文件

- 以 PowerShell 訓練及模型下載腳本取代原有 shell 腳本，支援 Python 路徑與 GPU 數量設定。
- 更新 VS Code 偵錯及 Conda 環境設定，固定 Python 3.10、PyTorch 2.0.1／CUDA 11.8 等相依版本，讓 Conda 共用 requirements.txt。
- 預設資料路徑改為 C:\Users\Hank\Desktop\data，仍可透過 CAFLOW_DATA_DIR 覆寫；相關繪圖字型改用 Arial。
- 更新 README、操作指南與模型儲存說明，並新增本變更日誌及 README 連結。

### 驗證紀錄

- 新增 14 項回歸測試，涵蓋離散目標、posterior、採樣、資料分割、水深指標、梯度累積、EMA、checkpoint 與異常資料處理。
- 本次在 flooddiff 環境執行 `python -m unittest discover -s tests -v`，14 項全部通過。
- 本次未執行完整 GPU 訓練或完整資料集評估。
