from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import random
import torch
import ipydbg
import argparse
import pickle
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import ROOT

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == "__main__":

    # ipydbg.selectable = True
    # ipydbg.patch()

    torch.backends.cudnn.benchmark = True

    """ set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

    chunks_path = ROOT + '/preprocessData/chunks/' + str(args.model) + '/chunks_' + args.data + '_train.pickle'
    chunks = pickle.load(open(chunks_path, "rb"))
    
    print(np.array(chunks).shape)
    
    """ dimension reduction """ 
    pca = PCA(n_components=args.dim, copy=True)
    new_feature = pca.fit_transform(chunks)

    print(new_feature.shape)

    kmeans = MiniBatchKMeans(n_clusters=args.kmeans, batch_size=args.batch)
    kmeans.fit(new_feature)
    
    """ save files """ 
    save_kmeans = "{}/preprocessData/kmeans/{}/{}_{}_{}_{}.pickle".format(
        ROOT,
        args.data,
        args.model, 
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    save_feature = "{}/preprocessData/chunks/{}/PCA/{}_{}_{}_{}.pickle".format(
        ROOT,
        args.model,
        args.data,
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    save_PCA = "{}/preprocessData/PCA/{}/{}_{}_{}_{}.pickle".format(
        ROOT,
        args.data,
        args.model,
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    
    if not os.path.isdir('{}/preprocessData/kmeans/{}'.format(ROOT, args.data)):
        print('create', '{}/preprocessData/kmeans/{}'.format(ROOT, args.data))
        os.makedirs('{}/preprocessData/kmeans/{}'.format(ROOT, args.data))
        
    if not os.path.isdir("{}/preprocessData/chunks/{}/PCA/".format(ROOT, args.model)):
        print('create', "{}/preprocessData/chunks/{}/PCA/".format(ROOT, args.model))
        os.makedirs("{}/preprocessData/chunks/{}/PCA/".format(ROOT, args.model))
        
    if not os.path.isdir("{}/preprocessData/PCA/{}/".format(ROOT, args.data)):
        print('create', "{}/preprocessData/PCA/{}/".format(ROOT, args.data))
        os.makedirs("{}/preprocessData/PCA/{}/".format(ROOT, args.data))
    
    with open(save_kmeans, 'wb') as write:
        pickle.dump(kmeans, write)
    
    with open(save_PCA, 'wb') as write:
        pickle.dump(pca, write)

    with open(save_feature, 'wb') as write:
        pickle.dump(new_feature, write)
        


        
