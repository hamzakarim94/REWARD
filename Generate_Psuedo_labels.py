import pandas as pd
import numpy as np
from UTILS import MLP,divide_chunks,compute_Dt
import random
import torch
import math

ANOMALY_features= []
anomaly_labels = []
pseudo_labels = []
initial = 'KNN_UCF_UNI_6/'# KNN directory
feat_path = "ucf_feats/" # features directory
if 'UCF'  in initial or 'ucf' in initial:
    normal_vids = pd.read_csv("train_normal.txt", header=None)
    NORMAL_FEATURES = []
    Normal_labels = []
    for vid in normal_vids.values:
        path = vid[0].split('/')[1].split(".")[0]
        feats = np.load(feat_path + path + '.npy', allow_pickle=True)
        NORMAL_FEATURES.extend(feats)
        Normal_labels.extend([0] * len(feats))
        for ind in range(len(feats)):
            pseudo_labels.append([path.split(".")[0]+"_"+str(ind)+".mp4",0])

    train_file = pd.read_csv("train_anomaly.txt",header=None)
    train_file = train_file.values
else:
    train_file = pd.read_csv('XD_train.csv',header=None)
    train_file = train_file.values

for file in train_file:#rain_file.values:
    if 'UCF' in initial or 'ucf' in initial:
        file_name = file[0].split('/')[1].split(".")[0]+".npy"
    else:
        file_name = file[0]+'_uni.npy'

    df = np.load(initial + file_name)
    st = 0
    st_arr = []

    anomalous_dists = df.reshape(len(df)).tolist()
    #remove 20% of segments from start and end
    if len(df) <= 10:
        s = 0
        l = 0
    else:
        drop_lenght = math.floor(len(df) * 0.2)
        s = drop_lenght
        l = drop_lenght
        l=0
        s=0
    if s > 0:
        del anomalous_dists[:s]
    if l > 0:
        anomalous_dists = anomalous_dists[:-l]
    #####################################################################################################'''
    #CUMSUM Smoothing
    anomalous_dists = (anomalous_dists - np.min(anomalous_dists)) / (np.max(anomalous_dists) - np.min(anomalous_dists))
    e_alpha = sum(anomalous_dists) / len(anomalous_dists)
    for e, anom in enumerate(anomalous_dists):
        if e == 0:
            st = max((st + anom - e_alpha), 0)
        else:
            st = max((st + anom - e_alpha), 0)
        st_arr.append(st)

    if max(st_arr) > 0:
        st_arr = (st_arr - np.min(st_arr)) / (np.max(st_arr) - np.min(st_arr))
        for ind,seg in enumerate(st_arr):
            if seg > 0.8: #Lamba = 0.8
                #grope = grope + str(ind+s) + '/'
                anomaly_feat = np.load(feat_path+file_name, allow_pickle=True)
                ANOMALY_features.append(anomaly_feat[ind+s])
                anomaly_labels.append(1)

Combined_features = NORMAL_FEATURES + ANOMALY_features
Combined_labels = Normal_labels + anomaly_labels
temp = list(zip(Combined_features, Combined_labels))
random.shuffle(temp)
features, labels = zip(*temp)
features, labels = list(features), list(labels)
features = list(divide_chunks(features, 256))
labels = list(divide_chunks(labels, 256))

#Train the intermediate MLP
mlp = MLP()
mlp.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.BCELoss().to(device=device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001,weight_decay=0.0001)
for i in range(3):
    for i,batch_list in enumerate(features):
        label = labels[i]
        label = np.array(label)
        batch_list = np.array(batch_list)
        mlp.to(device=device)
        input = torch.from_numpy(batch_list)
        input = input.type(torch.float32)
        size = input.size()
        input.requires_grad = True
        input= input.to(device =device)
        label = torch.from_numpy(label).to(device = device)
        label = label.reshape((len(label), 1))
        label = label.type(torch.float32)
        preds = mlp(input)
        optimizer.zero_grad()
        loss = loss_fn(preds,label)
        print(loss.item())
        loss.backward()
        optimizer.step()

mlp.eval()
for file in train_file:
    file_name = file[0].split('/')[1].split(".")[0] + ".npy"
    input = np.load(feat_path+file_name, allow_pickle=True)
    input = torch.from_numpy(input)
    input = input.type(torch.float32).to(device)
    with torch.no_grad():
        preds = mlp(input)
    preds = preds.cpu().numpy()
    if len(preds) <= 10:
            s = 0
            l = 0
    else:
        drop_lenght = math.floor(len(preds)*0.2)
        s=drop_lenght
        l=drop_lenght
    anomalous_dists = preds.tolist()
    #Compute Dt
    anomalous_dists,avg = compute_Dt(anomalous_dists,s,l)
    for ind, seg in enumerate(anomalous_dists):
        if seg > avg:
            file_name= file[0].split('/')[1].split(".")[0]+"_"+str(s+ind)+".mp4"
            pseudo_labels.append([file_name,1])
random.shuffle(pseudo_labels)
df = pd.DataFrame(pseudo_labels)
df.to_csv("Psuedo_labels.csv",header=None,index=False)