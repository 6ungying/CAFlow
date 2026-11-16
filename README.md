# CAFlow

變更紀錄請參閱 [CHANGELOG.md](CHANGELOG.md)。

## 修正後的 I2SB 訓練與驗證

訓練目標固定為 `(xt - x0) / std_fwd[step]`，推論使用相同離散 timestep 的
I2SB posterior，預設 `--sampling-method i2sb`。舊 continuous／速度場 checkpoint
不相容，必須以新實驗名稱重新訓練；程式會拒絕把舊權重誤用於新版目標。
新訓練也會拒絕覆寫已有檔案的實驗目錄。

共用預設為 Batch Size 32、Microbatch 4，每次 optimizer 更新累積 8 個 microbatch。

```powershell
conda activate flooddiff
python train.py --name flood-single-i2sb --use-single-dem
python sample.py --ckpt flood-single-i2sb --use-single-dem --checkpoint best.pt --nfe 10 --batch-size 8
```

預設以 `--split-seed 0` 按完整降雨事件保留約 10% 作驗證，同一雨型的不同時刻
不會跨入訓練與驗證。也可透過 `--val-rain-list 21,33,48` 明確指定。
本機單 DEM 預設切分為訓練 3,936 筆、驗證 432 筆；獨立測試集仍為 360 筆。
多 DEM 的 validation 也是未見雨型；未見地形測試須另以 `--test-dem-list` 留出。

每 1,000 個成功 optimizer steps 及訓練結束時，完整遍歷驗證集，以固定
`--validation-seed 12345` 和 `--val-nfe 10` 計算模型與原始 CA4D 的
`rmse_m`、`mae_m`。不對預測作影像量化或數值截斷，再計算公尺尺度誤差。
單 DEM 以白色為乾地、黑色為 4 公尺換算（`--single-max-depth 4`）；多 DEM
使用 `maxmin_duv.csv` 中各地形的 `depth_max`，需確保與 PNG 編碼一致。

`results/<name>/best.pt` 依最低 validation RMSE 選模，`latest.pt` 用於續訓。
checkpoint 同時保存 step、optimizer、scheduler、EMA、最佳 RMSE 及目標／資料切分設定。
續訓須沿用原資料切分、驗證 NFE 與其他方法設定，例如：

```powershell
python train.py --ckpt flood-single-i2sb --use-single-dem
```

`--num-itr` 為包含已完成步數的訓練嘗試上限；非有限 microbatch 不參與梯度平均，
非有限梯度會跳過整次更新。UNet、DEM Adapter、Rainfall Embedder 共用梯度裁剪，
文字 log 與 TensorBoard 皆記錄有效樣本平均 loss。
新版自動選模使用 `Runner.evaluation()` 的水深指標；舊 `compute_metrices.py`
仍屬於 ImageNet 評估工具，不作水深選模依據。
驗證不提供 early stopping，預設仍須由使用者設定適當訓練長度。
續訓恢復 optimizer step，但尚未恢復 DataLoader／RNG 的逐位元相同取樣順序。

目前支援單 GPU、FP32、256×256 pixel-space 訓練；不使用 `--clip-denoise`，因為水深
採用 z-score，合法數值不受限於 `[-1,1]`。完整訓練尚未執行，修正不代表精度提升已獲證實。

回歸測試：

```powershell
python -B -m unittest discover -s tests -v
```

## 環境設定（Windows / NVIDIA GPU）

已驗證的環境：Python 3.10、PyTorch 2.0.1 + CUDA 11.8、torchvision 0.15.2。
本機已存在可用的 `flooddiff` Conda 環境，可直接啟用。

```powershell
conda activate flooddiff
```

在其他電腦建立環境時，請於專案根目錄執行：

```powershell
conda env create -f requirements.yaml
conda activate flooddiff
```

更新既有環境的套件：

```powershell
python -m pip install -r requirements.txt
```

檢查環境與程式入口：

```powershell
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python train.py --help
python sample.py --help
python compute_metrices.py --help
```

VS Code 請以「Python: Select Interpreter」選擇 `flooddiff`。
訓練與推論參數請參閱 [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md)。

`requirements.txt` 包含目前 CAFlow 訓練、推論、前處理與評估所需的套件，
`requirements.yaml` 共用同一份清單。
保留的上游 `i2sb/VQGAN/taming/` 與部分 `i2sb/base/` 範例仍有 `model.*` 等舊匯入路徑；
這些非主要入口模組未納入執行驗證，若要使用需另外整理路徑及其選用相依套件。
環境驗證不代表已完成整輪模型訓練或資料集評估。

## Windows 執行方式

預設資料目錄為 `C:\Users\Hank\Desktop\data`，程式與 VS Code 偵錯設定均已使用此位置。
需要更換時，可先在 PowerShell 執行：

```powershell
$env:CAFLOW_DATA_DIR = "C:\Users\Hank\Desktop\data"
```

啟動單一 DEM 訓練（預設使用 1 張 GPU）：

```powershell
conda activate flooddiff
.\scripts\train.ps1 --name flood-single-i2sb --use-single-dem --batch-size 32
```

`train.ps1` 會轉交參數給 CAFlow 訓練入口；原本逐一執行九種影像修復任務的 Linux 腳本已移除。
下載上游影像修復模型時可執行：

```powershell
.\scripts\download_checkpoint.ps1 -Name inpaint-center
```

這個下載工具提供上游影像修復模型，CAFlow 水深模型仍使用自行訓練的 checkpoint。
如 PowerShell 阻擋本機腳本，可在本次視窗允許後再執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
