import cv2
import numpy as np

"""
Usage

from visualize import errorMap

errorMap = errorMap('./MVTec/mask', False)

result = errorMap.generateMap(params)
errorMap.saveMap(result, pathToSave)

"""
class errorMap():
    def __init__(self, maskRoot="./", inverse=False, autoSave=False):
        """
        maskRoot: 你放 mask 的資料夾位置，可以放相對路徑也可以放絕對路徑
        inverse: True 代表 0 是 mask 掉的區域， False 代表 1 是 mask 掉的區域
        autoSave: 自動幫你最後完成的那一張圖片存下來，新的圖片會蓋掉舊的
        """
        self.maskRoot = maskRoot
        self.autoSave = autoSave
        self.inverse = inverse
    
    def generateMap(self, image, ground_truth, pred_indexs):
        """
        image: origin image(nparray)
        ground_truth: gt_mask(nparray)
        pred_indexs: predicted error arrea index, this will be a list
        """
        assert isinstance(ground_truth, str) or len(ground_truth.shape) == 2, 'Mask should be an 2d nparray(256, 256) or string'
        assert len(image.shape) == 3, 'Input image should be an RGB image(256, 256, 3)'
        assert isinstance(pred_indexs, list), 'pred_indexs should be a one hot list'

        image = self.denormalize(image, 0) if np.any(image[:,:,:] < 0) else image if np.any(image[:,:,:] > 1) else self.denormalize(image, 1)

        if isinstance(ground_truth, str):
            ground_truth = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE)

        image = cv2.UMat(np.array(image, dtype=np.uint8))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 解決變色問題

        # 把 ground truth 的區域用綠色填滿(GT size 記得要 resize)
        gt_contours, _ = cv2.findContours(ground_truth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, gt_contours, -1, (0,255,0), -1)

        # 將你預測的區域在圖片上圈出
        mask = self.mergeMask(pred_indexs)
        pred_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, pred_contours, -1, (0,0,255), 1)

        if self.autoSave:
            self.saveMap(image)
            
        return image
    
    def mergeMask(self, pred_indexs):
        total_mask = np.ones((256, 256), dtype="uint8")

        print(f'sum: {sum(pred_indexs)}')
        for index, value in enumerate(pred_indexs):
            # print(f'index:{index} / value:{value}')
            """ abnormal """ 
            if value == 0:
                name = 'mask' + str(index)
                mask = cv2.imread("{}/{}.png".format(self.maskRoot, name), cv2.IMREAD_GRAYSCALE)
                mask[mask == 255] = 1

                total_mask *= mask

        if self.inverse:
            total_mask[total_mask == 1] = 255
            total_mask = np.invert(total_mask)

        return total_mask
        
    def saveMap(self, image, dist="./errorMap.png"):
        """
        image: np array of the image
        dis: Path that you are going to save the image
        """

        cv2.imwrite(dist, image)
    
    def denormalize(self, image, mode=0):
        """
        mode == 0: image rnage [-1, 1]
        mode == 1: image range [0, 1]
        """
        if mode == 0:
            return np.around(image * 127.5 + 127.5).astype(int)
        elif mode == 1:
            return (image * 255).astype(int)
        else:
            print("mode: {} is not support".format(mode))