"""
    Author: Yong Yu Chen
    Collaborator: Corn

    Update: 2020/12/2
    History: 
        2020/12/2 -> code refactor

    Description: 
"""

""" STD Library """
import os
import pickle
import argparse
import numpy as np

""" sklearn Library"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

""" Custom Library """
import ipydbg
from config import ROOT

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":
    """ set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

    """ read preprocess patches """
    chunks_path = f"{ ROOT }/preprocessData/chunks/{ str(args.model) }/chunks_{ args.data }_train.pickle"
    chunks = pickle.load(open(chunks_path, "rb"))
    
    print(np.array(chunks).shape)

    """ dimension reduction """ 
    pca = PCA(n_components=args.dim, copy=True)
    new_feature = pca.fit_transform(chunks)
    kmeans = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=args.batch)
    kmeans.fit(new_feature)

    print(new_feature.shape)
    
    """" Check file and folders and auto create """
    if not os.path.isdir(f"{ ROOT }/preprocessData/kmeans/{ args.data }"):
        print('create', f"{ ROOT }/preprocessData/kmeans/{ args.data }")
        os.makedirs(f"{ ROOT }/preprocessData/kmeans/{ args.data }")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/chunks/{ args.model }/PCA/"):
        print('create', f"{ ROOT }/preprocessData/chunks/{ args.model }/PCA/")
        os.makedirs(f"{ ROOT }/preprocessData/chunks/{ args.model }/PCA/")
        
    if not os.path.isdir(f"{ ROOT }/preprocessData/PCA/{ args.data }/"):
        print('create', f"{ ROOT }/preprocessData/PCA/{ args.data }/")
        os.makedirs(f"{ ROOT }/preprocessData/PCA/{ args.data }/")
    
    """ save files """ 
    save_kmeans  = f"{ ROOT }/preprocessData/kmeans/{ args.data }/{ args.model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle")
    save_feature = f"{ ROOT }/preprocessData/chunks/{ args.model }/PCA/{ args.data }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle")
    save_PCA     = f"{ ROOT }/preprocessData/PCA/{ args.data }/{ args/model }_{ str(args.kmeans) }_{ str(args.batch) }_{ str(args.dim) }.pickle")
    
    with open(save_kmeans, 'wb') as write:
        pickle.dump(kmeans, write)
    
    with open(save_PCA, 'wb') as write:
        pickle.dump(pca, write)

    with open(save_feature, 'wb') as write:
        pickle.dump(new_feature, write)
        


        
