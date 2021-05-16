"""
    Author: Corn

    Update: 2021/5/4

    Description: 透過 PCA 降維之後再用 kmeans 去進行分群
"""

""" STD Library """
import os
import sys
import pickle
import argparse
import numpy as np

""" sklearn Library"""
# Try MDS, TSNE, DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

import umap

""" Custom Library """
from config import ROOT

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    parser.add_argument('--kmeans', type=int, default=128, help='number of kmeans clusters')
    args = parser.parse_args()

    """ read preprocess patches """
    chunks_path = f"{ ROOT }/preprocessData/chunks/ssl/chunks_{ args.data }_train.pickle"
    chunks = pickle.load(open(chunks_path, "rb"))
    chunks = np.array(chunks)
    print(chunks.shape)

    # """ dimension reduction """ 
    dim_reduction = PCA(n_components=args.dim, copy=True)

    new_feature = dim_reduction.fit_transform(chunks)
    
    times = 0
    while True:
        count = np.zeros(args.kmeans)
        kmeans = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=args.batch)
        kmeans.fit(new_feature)

        for num in range(new_feature.shape[0]):
            label = kmeans.predict(new_feature[num].reshape(1,-1))
            label = label.item()
            count[label] += 1
        
        print(count)

        if 0 in count:
            print("fail!!!!!")
            times += 1
            continue
        else:
            print("success!")
            break

    print(f"Retry { times } times ")
    print(new_feature.shape)
    
    """ Check file and folders and auto create """
    if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }"):
        print('create', f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }")
        os.makedirs(f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/chunks/ssl/{ args.dim_reduction }/"):
        print('create', f"{ ROOT }/preprocessData/chunks/ssl/{ args.dim_reduction }/")
        os.makedirs(f"{ ROOT }/preprocessData/chunks/ssl/{ args.dim_reduction }/")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/"):
        print('create', f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/")
        os.makedirs(f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/")
    
    """ save files """
    save_kmeans  = f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }/ssl_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    save_feature = f"{ ROOT }/preprocessData/chunks/ssl/{ args.dim_reduction }/{ args.data }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    save_PCA     = f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/ssl_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    
    with open(save_kmeans, 'wb') as write:
        pickle.dump(kmeans, write)
    
    with open(save_PCA, 'wb') as write:
        pickle.dump(dim_reduction, write)

    with open(save_feature, 'wb') as write:
        pickle.dump(new_feature, write)
        