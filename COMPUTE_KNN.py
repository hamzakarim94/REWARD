import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import os
feat_path = "ucf_feats/"
save_loc = 'KNN_UCF_UNI_6/'
k = 20 #number of nieghbours
np.random.seed(2)
normal_vids = pd.read_csv("train_normal.txt",header=None)
anomaly_vids = pd.read_csv("train_anomaly.txt",header=None)
NORMAL_FEATURES = []
for vid in normal_vids.values:
    path = vid[0].split('/')[1].split(".")[0]
    feats = np.load(feat_path+path + '.npy', allow_pickle=True)
    NORMAL_FEATURES.append(feats)
NORMAL_FEATURES = np.concatenate(NORMAL_FEATURES)
for i,li in enumerate(anomaly_vids.values):
    path = li[0].split('/')[1].split(".")[0]
    feats = np.load(feat_path+path+'.npy', allow_pickle=True)


    distance_matrix = pairwise_distances(feats, NORMAL_FEATURES, metric='euclidean')

    sorted_distance_matrix = np.sort(distance_matrix, axis=1)
    average_distances = np.mean(sorted_distance_matrix[:, :k], axis=1)

    if os.path.exists(save_loc) and os.path.isdir(save_loc):
        np.save(save_loc + path + '.npy', average_distances)
    else:
        os.mkdir(save_loc)
        np.save(save_loc + path + '.npy', average_distances)
    print("DISTANCES COMPUTED = ",str(i)+"/"+str(len(anomaly_vids.values)))


    #for indices_list in indexes:

