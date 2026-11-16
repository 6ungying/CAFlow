# CAFlow 深度預測模型指南

## 模型資料流

目前模型只處理水深：

- x0：TUFLOW 水深真值，1 通道
- x1：CA4D 水深，1 通道
- cond：DEM 地形，1 通道
- rainfall：24 個時間步的降雨序列

模型固定以 CA4D 作為 I2SB x1，並以 DEM、降雨序列與 timestep
作為條件。

## 資料集模式

- --use-single-dem：單一 DEM（台南）
- --test-dem-list：多 DEM，逗號分隔
- --test-rain-list：保留作測試的降雨情境，逗號分隔
- --use-yilan：宜蘭區域評估
- --use-hsinchu：新竹區域評估

資料預設位於 `C:\Users\Hank\Desktop\data`。如需更改，可在 PowerShell 設定：

~~~powershell
$env:CAFLOW_DATA_DIR = "C:\Users\Hank\Desktop\data"
~~~

## 訓練

### 單一 DEM

~~~powershell
python train.py `
  --name flood-single-i2sb `
  --use-single-dem `
  --ca4d `
  --batch-size 32
~~~

### 多 DEM

~~~powershell
python train.py `
  --name flood-multi-i2sb-new `
  --test-dem-list 17,42,54 `
  --test-rain-list 9,20,33 `
  --ca4d `
  --batch-size 32
~~~

## 推論

推論時會從 checkpoint 的 options.pkl 同步 CA4D 設定；DEM 固定作為
spatial conditioning。

### 單一 DEM

~~~powershell
python sample.py `
  --ckpt flood-single-i2sb `
  --use-single-dem `
  --nfe 10 `
  --batch-size 8
~~~

### 多 DEM

~~~powershell
python sample.py `
  --ckpt flood-multi-i2sb-new `
  --test-dem-list 17,42,54 `
  --test-rain-list 9,20,33 `
  --nfe 10 `
  --batch-size 8
~~~

### 區域評估

~~~powershell
python sample.py --ckpt flood-multi-i2sb-new --use-yilan --nfe 10 --batch-size 8
python sample.py --ckpt flood-multi-i2sb-new --use-hsinchu --nfe 10 --batch-size 8
~~~

## 輸出位置

訓練模型：

~~~text
results/<experiment-name>/
├── latest.pt
└── options.pkl
~~~

各模型的訓練紀錄與誤差圖：

~~~text
.log/<experiment-name>/
├── log.txt
└── training_error.png
~~~

`log.txt` 採追加寫入，同名模型續訓時會保留先前內容。Sampling 與
metrics 則分別寫入同一模型資料夾下的 `sample_log.txt` 與
`metrics_log.txt`，不會覆蓋訓練紀錄。

推論影像：

~~~text
results/<checkpoint-name>/test3_nfe<NFE>_<sampling-method>/
~~~

可用 CAFLOW_RESULTS_DIR 改變輸出根目錄，或直接使用 VS Code
launch.json 中已整理好的設定。
