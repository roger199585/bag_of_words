"""
    Author: Corn

    Update: 2020/12/24
    History: 
        2020/12/24 -> 將 pretrained autoencoder 所萃取出來的特徵進行 BoW，以及 kmeans 的分群

    Description: 將 pretrained autoencoder 所萃取出來的特徵進行 BoW，以及 kmeans 的分群
"""

""" STD Library """
import os
import sys
import pickle
import argparse
import numpy as np

""" sklearn Library"""
from sklearn.cluster import MiniBatchKMeans

""" Custom Library """
from config import ROOT

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--resolution', type=int, default=4, help='Dimension of the autoencoder\'s latent code resolution')
    args = parser.parse_args()

    """ read preprocess patches """
    chunks_path = f"{ ROOT }/preprocessData/chunks/AE/{ args.data }/{ args.resolution}/chunks_{ args.data }_train.pickle"
    chunks = pickle.load(open(chunks_path, "rb"))
    
    print(np.array(chunks).shape)

    new_feature = np.array(chunks)
    
    kmeans = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=100)
    kmeans.fit(new_feature)

    print(new_feature.shape)
    
    """ Check file and folders and auto create """
    if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution }"):
        print('create', f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution }")
        os.makedirs(f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution }")
    
    """ save files """
    save_kmeans  = f"{ ROOT }/preprocessData/kmeans/AE/{ args.data }/{ args.resolution }/AE_{ str(args.kmeans) }.pickle"
    
    with open(save_kmeans, 'wb') as write:
        pickle.dump(kmeans, write)
        