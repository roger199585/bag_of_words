import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

from skimage.measure import compare_ssim

import ei
ei.patch(select=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--root', type=str, default='/mnt/train-data1/corn')
    args = parser.parse_args()

    threshold = 30

    print('Merge calculated data')
    train_count = 0
    val_count = 0

    images = os.listdir(args.root + '/bottle')
    images = sorted(images, key=lambda a: int(a[:-4]))

    for i in tqdm(range(0, 53503)):
        diff = np.load(args.root +'/diff/'+'patches_rgb_diff_start_'+str(i)+'.npy', allow_pickle=True)
        
        diff = sorted(diff, key=lambda x: x['diff']) 

        for i in range(len(diff)):
            # HSV distance 需要小於一定的數值，並且 ssim 的相似度需要大於 0.8 才可以被挑選出來
            if diff[i]['diff'] > 0.5 and diff[i]['diff'] < threshold:
                im1 = cv2.imread(args.root + '/bottle/' + images[diff[i]['i']])
                im2 = cv2.imread(args.root + '/bottle/' + images[diff[i]['j']])

                if compare_ssim(im1, im2, multichannel=True) > 0.8:
                    if random.randint(1, 10) % 10 == 0:
                        os.system(f"cp { args.root }/bottle/{ diff[i]['i'] }.png { args.root }/bottle_patch/val/{ str(val_count) }_1.png")
                        os.system(f"cp { args.root }/bottle/{ diff[i]['j'] }.png { args.root }/bottle_patch/val/{ str(val_count) }_2.png")
                        val_count += 1
                    else:
                        os.system(f"cp { args.root }/bottle/{ diff[i]['i'] }.png { args.root }/bottle_patch/train/{ str(train_count) }_1.png")
                        os.system(f"cp { args.root }/bottle/{ diff[i]['j'] }.png { args.root }/bottle_patch/train/{ str(train_count) }_2.png")
                        train_count += 1
