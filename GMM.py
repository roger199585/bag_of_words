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
from sklearn.decomposition import PCA
from sklearn import mixture

if __name__ == "__main__":

    # ipydbg.selectable = True
    # ipydbg.patch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    """ set parameters """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmeans', type=int, default=16, help='number of kmeans clusters')
    parser.add_argument('--data', type=str, default='bottle')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--model', type=str, default='vgg19')
    args = parser.parse_args()

    chunks_path = './chunks/' + str(args.model) + '/chunks_' + args.data + '_train.pickle'
    chunks = pickle.load(open(chunks_path, "rb"))
    
    print(np.array(chunks).shape)
    chunks = np.squeeze(chunks)
    
    """ dimension reduction """ 
    pca = PCA(n_components=args.dim, copy=True)
    new_feature = pca.fit_transform(chunks)
    print(new_feature.shape)

    """ apply Gaussian Mixture Model """
    gmm = mixture.GaussianMixture(n_components=args.kmeans)
    gmm.fit(new_feature)

    """ save files """ 
    save_gmm = "gmm/{}/{}_{}_{}_{}.pickle".format(
        args.data,
        args.model, 
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    save_feature = "chunks/{}/PCA/{}_{}_{}_{}_gmm.pickle".format(
        args.model,
        args.data,
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    save_PCA = "PCA/{}/{}_{}_{}_{}_gmm.pickle".format(
        args.data,
        args.model,
        str(args.kmeans),
        str(args.batch),
        str(args.dim)
    )
    
    with open(save_gmm, 'wb') as write:
        pickle.dump(gmm, write)

    with open(save_feature, 'wb') as write:
        pickle.dump(new_feature, write)

    with open(save_PCA, 'wb') as write:
        pickle.dump(pca, write)