import time
import pickle
import argparse
import numpy as np
from PIL import Image

from sklearn import metrics
from scipy.ndimage.measurements import label

from torch.utils.data import Dataset, DataLoader
import dataloaders

from config import ROOT

class PRO_curve():
    def __init__(self, y_pred, y_true, FPR_boundary=0.3, spacing=0.005):
        # Data Groung Truth and Model predictions
        self.y_pred = self.norm(y_pred)
        self.y_true = y_true

        # fpr and pro score with different threshold
        self.FPRS = []
        self.PROS = []

        # Initial hyper-parameter setting
        self.FPR_boundary = FPR_boundary
        self.spacing = spacing

        self.score = 0

    def norm(self, t):
        return t / t.max()
    
    def threshold_anomaly_images(self, y_pred, threshold):
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0

        return y_pred

    def compute_metrics_image(self, prediction, idx):
        # 找出 connected components (aka anomalous regions)
        structure = np.ones((3, 3), dtype=np.int)

        labeled, n_components = label(self.y_true[idx], structure)
                
        # 計算 false positive pixels 數量
        num_false_positives = np.sum(prediction[labeled==0])
        
        # 計算所有異常區域內的重疊面積
        all_region_overlaps = []
        for i in range(0, n_components):
            # 計算 connected component (異常區域) 的面積，同一張圖可能存在多個異常區域
            component_size = len(self.y_true[idx][labeled==(i+1)])

            # 計算異常區域內正確被預測出來的比例
            num_true_positives = np.sum(prediction[labeled==(i+1)])
            region_overlap = num_true_positives / component_size
            all_region_overlaps.append(region_overlap)
                    
        return all_region_overlaps, num_false_positives

    def compute_metrics_dataset(self, predictions):
        """
            Args:
                predictions: 這是已經砍 threshold 的預測數值，其代表意義為 threshold=? 的時候模型預測出來的 mask(因此他與 self.y_pred 並不一樣，self.y_pred 是原始的 feature loss 經過 overlapping 的計算後的數值)
        """
        # FPR 試用者個 dataset 中的所有 pixel 去計算的
        # overlap 是每塊異常區域獨立計算 Ex. 圖一有兩塊異常區域，第一塊重疊率是 0.5 第二塊重疊率是 0.8，則圖一的 overlap = (0.5 + 0.8) / 2
        all_region_overlaps = []
        total_num_FP = 0
        total_num_background_pixels = 0
                
        for i in range(0, self.y_true.shape[0]):
            region_overlaps, num_FP = self.compute_metrics_image(predictions[i], i)
            all_region_overlaps.extend(region_overlaps)
                    
            total_num_FP += num_FP
            total_num_background_pixels += len(self.y_true[i][self.y_true[i]==0])
            
        PRO = np.mean(all_region_overlaps)
        FPR = total_num_FP / total_num_background_pixels
        
        # print(f'FPR={FPR}, PRO={PRO}')
        self.FPRS.append(FPR)
        self.PROS.append(PRO)
    
    def calculate_PRO_score(self):
        # 計算 PRO curve 的面積

        area = 0
        candidate = []
        for i in range(1, len(self.PROS) - 1):
            height = np.abs(self.FPRS[i - 1] - self.FPRS[i])

            # 如果出現連續且相同的 FPR 值得話其對應的 PRO 要全部加起來取平均
            if height != 0:
                if len(candidate) > 0:
                    candidate.append(self.PROS[i-1])
                    width1 = np.array(candidate).mean()
                    candidate = []
                else:
                    width1 = self.PROS[i-1]
                width2 = self.PROS[i]
                
                area += (width1 + width2) * height / 2
            else: 
                candidate.append(self.PROS[i-1])
                candidate.append(self.PROS[i])

        print(f'FPR={self.FPRS[len(self.FPRS) - 2]}')
        self.score = area / self.FPR_boundary

    def calculate_fpr(self, pred, threshold):
        prediction = self.threshold_anomaly_images(pred, threshold)       
        # 找出 connected components (aka anomalous regions)
        structure = np.ones((3, 3), dtype=np.int)
        labeled, n_components = label(self.y_true, structure)
                
        # 計算 false positive pixels 數量
        num_false_positives = np.sum(prediction[labeled==0])

        return num_false_positives

    def get_best_threshold(self):
        threshold = 1
        lastFPR = 1

        while abs(fpr - 0.3) > 0.00001:
            total_num_FP = 0
            total_num_background_pixels = 0
                    
            for i in range(0, self.y_true.shape[0]):
                fpr_pixel = self.calculate_fpr(self.y_pred[i].copy(), self.y_true[i])
                        
                total_num_FP += fpr_pixel
                total_num_background_pixels += len(self.y_true[i][self.y_true[i]==0])
                
            fpr = total_num_FP / total_num_background_pixels 

    def test_all_threshold(self):
        for threshold in np.arange(1.0, 0.0, -1 * self.spacing):
            threshed_y_pred = self.threshold_anomaly_images(self.y_pred.copy(), threshold)

            self.compute_metrics_dataset(threshed_y_pred)
            
            if self.FPRS[len(self.FPRS) - 1] > 0.2999:
                break
    
    def getScore(self):
        return self.score
    
    def getFPRS(self):
        return self.FPRS
    
    def getPROS(self):
        return self.PROS


"""
* calculate the PRO score

Usage: 
    1. `evalTool = PRO_curve(y_pred, y_true, spacing=0.001)`: create an new object
    2. `evalTool.test_all_threshold()`: find out the PRO curve under FPR=30
    3. `evalTool.calculate_PRO_score()`: calculate the area under PRO cureve
    4. `evalTool.getScore()`: get the area under PRO curve under FPR=30
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--index', type=str, default='10')
    parser.add_argument('--kmeans', type=str, default='128')
    parser.add_argument('--resolution', type=str, default='4')
    args = parser.parse_args()
    
    y_pred = pickle.load(open(f"{ ROOT }/Results/testing_multiMap/AE/{ args.data }/{ args.resolution }/all/{ args.kmeans }_img_all_feature_{ args.index }_Origin.pickle", "rb"))
    y_pred = y_pred.reshape(-1, 1024, 1024)
    # gt image
    y_true = []
    mask_path = f"{ ROOT }/dataset/{ args.data }/ground_truth_resize/all"
    mask_dataset = dataloaders.MaskLoader(mask_path)
    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=False)
    for (index, mask) in mask_loader:
        y_true.append(mask[0, 0,:,:].numpy())
    y_true = np.array(y_true)


    evalTool = PRO_curve(y_pred, y_true, spacing=0.001)
    start = time.time()
    evalTool.test_all_threshold()
    evalTool.calculate_PRO_score()
    # evalTool.printFPR()
    print(evalTool.getScore())
    print(f'spend {time.time() - start}')
