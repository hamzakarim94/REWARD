from Datasets import CustomVideoDataset
import torch
from torch import nn
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
from slowfast.models import build_model
import pickle
import numpy as np
sig = torch.nn.Sigmoid()
from test import testing
def training(model,dataloader,loss_fn,optimizer):
    model.train()
    total_loss =0
    for cur_iter,data,video_data in dataloader:
        if data == None or video_data == None:
            break
        video_data = np.array(video_data)
        labels = torch.tensor(video_data[:, 1].astype(np.float32)).to("cuda:0")
        data = torch.concatenate(data)
        shape = data.size()
        data = torch.reshape(data, (1, shape[0], shape[1], shape[2], shape[3], shape[4]))
        out = sig(model(data.to("cuda:0")))
        labels = torch.reshape(labels,out.size())
        loss = loss_fn(out, labels.to('cuda:0'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss+loss.item()
        print("average loss: ",total_loss/(cur_iter))
    return model



if __name__ == "__main__":
    batch_size = 1
    epochs = 5
    best_metric = 0
    Metric = "auc" #Select Metric
    with open('gt.pkl', 'rb') as file:
        loaded_gt = pickle.load(file)
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    block_list = ['blocks4']#Only train block4
    for name, param in model.named_parameters():
        for block in block_list:
            if block in name:
                param.requires_grad = True
                break
            else:
                param.requires_grad = False
                break

    model.to(device="cuda:0")
    train_data_loader = CustomVideoDataset(data="Psuedo_labels.csv", batch_size=32, test=False,cfg=cfg, prefix="ucf_segmented_videos")
    test_data_loader = CustomVideoDataset(data="testing_data.csv", batch_size=1, test=True,cfg=cfg, prefix=".")
    for epoch in range(epochs):
        loss_fn = nn.BCELoss().to('cuda:0')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005,weight_decay=0.005)
        mpdel = training(model=model,dataloader=train_data_loader,loss_fn=loss_fn,optimizer=optimizer)
        auc,AP = testing(model=model,dataloader=test_data_loader,gt = loaded_gt)
        if Metric == "auc":
            if auc > best_metric:
                best_metric = auc
                torch.save(model.state_dict(), "best_model_+"+str(auc)+"_"+ str(epoch)+ "_.pth")
        else:
            if AP > best_metric:
                best_metric = AP
                torch.save(model.state_dict(), "best_model_+" + str(AP) +"_"+ str(epoch)+ "_.pth")
