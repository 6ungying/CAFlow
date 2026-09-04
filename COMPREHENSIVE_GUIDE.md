# CAFlow 深度預測模型指南

## 模型資料流

目前模型只處理水深：

- x0：TUFLOW 水深真值，1 通道
- x1：CA4D 水深，1 通道
- cond：DEM 地形，1 通道
- rainfall：24 個時間步的降雨序列
- spm：選用的空間先驗圖，1 通道

有效模型配置為：

1. CA4D baseline
2. CA4D + SPM

## 資料集模式

- --use-single-dem：單一 DEM（台南）
- --test-dem-list：多 DEM，逗號分隔
- --test-rain-list：保留作測試的降雨情境，逗號分隔
- --use-yilan：宜蘭區域評估
- --use-hsinchu：新竹區域評估

資料預設放在專案的 data/。如需使用其他位置，可設定
CAFLOW_DATA_DIR。

## 訓練

### 單一 DEM：CA4D baseline

~~~bash
python train.py \
  --name flood-single-ca4d \
  --use-single-dem \
  --ca4d \
  --batch-size 32
~~~

### 單一 DEM：CA4D + SPM

~~~bash
python train.py \
  --name flood-single-spm01 \
  --use-single-dem \
  --ca4d \
  --spm \
  --batch-size 32
~~~

### 多 DEM：CA4D baseline

~~~bash
python train.py \
  --name flood-multi-ca4d-new \
  --test-dem-list 17,42,54 \
  --test-rain-list 9,20,33 \
  --ca4d \
  --batch-size 32
~~~

### 多 DEM：CA4D + SPM

~~~bash
python train.py \
  --name flood-multi-spm-new \
  --test-dem-list 17,42,54 \
  --test-rain-list 9,20,33 \
  --ca4d \
  --spm \
  --batch-size 32
~~~

## 推論

推論時會從 checkpoint 的 options.pkl 同步 CA4D 與 SPM 設定。

### 單一 DEM

~~~bash
python sample.py \
  --ckpt flood-single-ca4d \
  --use-single-dem \
  --nfe 10 \
  --batch-size 8
~~~

### 多 DEM

~~~bash
python sample.py \
  --ckpt flood-multi-spm-new \
  --test-dem-list 17,42,54 \
  --test-rain-list 9,20,33 \
  --nfe 10 \
  --batch-size 8
~~~

### 區域評估

~~~bash
python sample.py --ckpt flood-multi-ca4d-new --use-yilan --nfe 10 --batch-size 8
python sample.py --ckpt flood-multi-ca4d-new --use-hsinchu --nfe 10 --batch-size 8
~~~

## 輸出位置

訓練模型：

~~~text
results/<experiment-name>/
├── latest.pt
└── options.pkl
~~~

推論影像：

~~~text
results/<checkpoint-name>/test3_nfe<NFE>_<sampling-method>/
~~~

可用 CAFLOW_RESULTS_DIR 改變輸出根目錄，或直接使用 VS Code
launch.json 中已整理好的設定。
