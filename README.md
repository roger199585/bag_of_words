# Run With Docker 

1. Download [preprocess data](https://drive.google.com/file/d/1nX0lq07XnB8w8OH09Nk616ZQK5nnrijp/view?usp=sharing)
2. Download [dataset](https://drive.google.com/file/d/1J752eYmySgCPJJ7HHAMy8KC9vkN_cxoa/view?usp=sharing)
2. Clone the code from github `git clone https://github.com/roger199585/bag_of_words.git`
3. The preprocess data should rename to `preprocessData` and put it under `bag_of_words`
4. Also `dataset` should be move under `bag_of_words`
5. Build Dockerfile `docker build -t cgvlab:ad .`
6. Run training with docker :
    - With old docker driver `docker run --runtime=nvidia -e CUDA_VISIBLE_DEVICE=0,1,2,3 -e TASK=1 --ipc=host -v <code folder>:/ws_data/nctu/bag_of_words -v <result folder>:/result -it --rm cgvlab:ad`
    - With latest docker driver `docker run --gpus=all -e TASK=1 --ipc=host -v <code folder>:/ws_data/nctu/bag_of_words -v <result folder>:/result -it --rm cgvlab:ad`



### 手動執行，如果要透過 docker 跑的話下面不用理他
## 資料處理
> 首先如果你沒有資料庫的話要先下載下來
> 下載方法如下方所示
```
cd dataset
wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
tar xvf mvtec_anomaly_detection.tar.xz
```

## Preprocess
> 這邊包含了 5 個步驟
> 1. 先把圖片轉換乘 1024 的大小
> 2. 把圖片切成 64 x 64 的小圖給 vgg19 抽取特徵
> 3. 將步驟 2 的小圖進行分群
> 4. 生成 groundtruth
> 5. 這步我其實還是不知道在幹嘛，我相信你比我清楚
- 執行方式只需跑 `./preprocess.sh`
- 如果遇到他說那個檔案不能跑的話先執行 `chmod +x preprocess.sh`
- preprocess.sh 裡面記得更改成你要跑的資料庫

## Train model (真 運氣遊戲的部分)
> 參數部分請自己調整，如果你懶的話也可以把這個指令加到 preprocess.sh 的最後，這樣 `preprocess.sh` 就會一條龍的幫你執行完，但是效果要看你的運氣囉
`CUDA_VISIBLE_DEVICES=0,1 python model_weightSample.py --train_batch 16 --kmeans=128 --data=wood --type=good --epoch 40 &`

## 備註
- 基本上檔案結構我沒什麼改，只有把所有前處理資料都塞進去 preprocessData 裡面，此外有一個 config.py 的檔案，裡面有你的 root 也就是你的 bag_of_words 這個資料夾的絕對路徑，如果你要搬去其他地方執行的話記得要更改這個地方 
- preprocess.py and convert.py 裡面的 dataset 路徑也要記得修改

## Progress
- 我們的 dataset 一共有 15 種不同的類型，目前剩下最後兩種 (wood, zipper) 還沒跑實驗



## 環境變數設定
ROOT=bag_of_words 這個專案的位置
RESULT_PATH=<存 Tensorboard log 的位置>
PYTHONPATH=$ROOT