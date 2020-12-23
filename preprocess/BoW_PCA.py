"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/3
    History: 
        2020/12/2 -> code refactor
        2020/12/3 -> code refactor and add description
        2020/12/23 -> 新增其他的降維方式

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
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN

""" Custom Library """
from config import ROOT

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    """ Set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--dim_reduction', type=str, default='PCA')
    args = parser.parse_args()

    """ read preprocess patches """
    chunks_path = f"{ ROOT }/preprocessData/chunks/{ str(args.model) }/chunks_{ args.data }_train.pickle"
    chunks = pickle.load(open(chunks_path, "rb"))
    
    print(np.array(chunks).shape)

    # """ dimension reduction """ 
    if args.dim_reduction == 'PCA':
        dim_reduction = PCA(n_components=args.dim, copy=True)
    elif args.dim_reduction == 'MDS':
        dim_reduction = MDS(n_components=args.dim)
    elif args.dim_reduction == 'TSNE':
        dim_reduction = TSNE(n_components=args.dim, perplexity=50, method='exact')
    elif args.dim_reduction == 'DBSCAN':
        pass
        # dim_reduction = DBSCAN(n_components=args.dim)

    new_feature = dim_reduction.fit_transform(chunks)
    
    kmeans = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=args.batch)
    kmeans.fit(new_feature)

    print(new_feature.shape)
    
    """ Check file and folders and auto create """
    if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }"):
        print('create', f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }")
        os.makedirs(f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/chunks/{ args.model }/{ args.dim_reduction }/"):
        print('create', f"{ ROOT }/preprocessData/chunks/{ args.model }/{ args.dim_reduction }/")
        os.makedirs(f"{ ROOT }/preprocessData/chunks/{ args.model }/{ args.dim_reduction }/")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/"):
        print('create', f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/")
        os.makedirs(f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/")
    
    """ save files """
    save_kmeans  = f"{ ROOT }/preprocessData/kmeans/{ args.dim_reduction }/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    save_feature = f"{ ROOT }/preprocessData/chunks/{ args.model }/{ args.dim_reduction }/{ args.data }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    save_PCA     = f"{ ROOT }/preprocessData/{ args.dim_reduction }/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle"
    
    with open(save_kmeans, 'wb') as write:
        pickle.dump(kmeans, write)
    
    with open(save_PCA, 'wb') as write:
        pickle.dump(dim_reduction, write)

    with open(save_feature, 'wb') as write:
        pickle.dump(new_feature, write)
        