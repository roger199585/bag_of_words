from sklearn import metrics
import numpy as np

class evalTool():
    def __init__(self, partialRatial=0.3):
        self.partialRatial = partialRatial

    def pROC(self, y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y, score)

        nearestIndex = np.argmin(abs(fpr - self.partialRatial))
        
        tpr = tpr[:nearestIndex]
        fpr = fpr[:nearestIndex]

        area = 0
        for index in range(1, nearestIndex):
            width1 = tpr[index] - fpr[index]
            width2 = tpr[index - 1] - fpr[index - 1]
            height = fpr[index] - fpr[index]

            area += (width1 + width2) * height / 2
        return area



