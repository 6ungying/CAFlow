# 模型儲存與續訓

## 儲存位置

模型預設儲存在專案的 results/：

~~~text
results/<experiment-name>/
├── latest.pt
└── options.pkl
~~~

experiment-name 來自訓練參數 --name。可透過 CAFLOW_RESULTS_DIR
將整個輸出目錄移到其他磁碟。

## Checkpoint 內容

latest.pt 包含：

- 深度預測網路權重
- 降雨 embedding 權重
- EMA 權重
- optimizer 狀態
- learning-rate scheduler 狀態

options.pkl 保存建立模型及資料集所需的訓練參數，例如 CA4D、DEM
conditioning、資料切分與 batch size。

## 繼續訓練

~~~bash
python train.py --ckpt <checkpoint-name>
~~~

程式會讀取：

~~~text
results/<checkpoint-name>/latest.pt
~~~

例如：

~~~bash
python train.py --ckpt flood-multi-ca4d-new
~~~

## 推論

~~~bash
python sample.py --ckpt <checkpoint-name> --nfe 10 --batch-size 8
~~~

也可直接把 checkpoint 資料夾的完整路徑傳給 --ckpt。

## 備份

備份時應一起保存整個實驗目錄，確保 latest.pt 與 options.pkl
保持配對。
