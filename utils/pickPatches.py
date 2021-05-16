import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm 

from skimage.measure import compare_ssim

import ei
ei.patch(select=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--root', type=str, default='./')
    # parser.add_argument('--data', type=str, default='bottle')
    args = parser.parse_args()

    images = os.listdir(args.root)
    images = sorted(images, key=lambda a: int(a[:-4]))

    diff = []

    for j in range(args.start+1, len(images)):
        im1 = cv2.imread(args.root + images[args.start])
        im2 = cv2.imread(args.root + images[j])

        im1 = im1.astype(int)
        im2 = im1.astype(int)
        # hsv1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
        # hsv2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

        # dh = min(abs(im1[:, :, 0] - im2[:, :, 0]).mean(), 360 - abs(im1[:, :, 0] - im2[:, :, 0]).mean()) / 180.0
        # ds = abs(im1[:, :, 1] - im2[:, :, 1]).mean()
        # dv = (abs(im1[:, :, 2] - im2[:, :, 2]) / 255.0).mean()

        # distance = math.sqrt(dh*dh+ds*ds+dv*dv)
        _mean = np.abs(im1 - im2).mean()
        _max = np.abs(im1 - im2).max()
        diff.append({
            # "diff": distance,
            "diff": _mean,
            "max": _max,
            # "sim": compare_ssim(im1, im2, multichannel=True),
            "i": args.start,
            "j": j
        })
        
    diff = np.array(diff)
    np.save('/mnt/train-data1/corn/diff/patches_rgb_diff_start_{}'.format(args.start), diff)
