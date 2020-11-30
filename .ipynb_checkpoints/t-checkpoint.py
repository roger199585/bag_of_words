import pickle
from config import ROOT

center_features_path = "{}/preprocessData/cluster_center/128/bottle.pickle".format(ROOT)
cluster_features = pickle.load(open(center_features_path, "rb"))

print(cluster_features[0])