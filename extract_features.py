from Datasets import CustomVideoDataset
import torch
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
from slowfast.models import build_model
import os
import numpy as np
from UTILS import Identity

def extract_features(model,dataloader,extract_folder):
    model.eval()

    for data,video_path in dataloader:

        #shape = video.size()
        #video = torch.reshape(video,(1,shape[0],shape[1],shape[2],shape[3],shape[4]))
        #data = torch.stack(data)
        if data == None or video_path == None:
            break

        label_data =video_path[0].split('/')[-1].split('.')[0]
        print(label_data)
        if os.path.exists(extract_folder + label_data+ '.npy'):
            continue

        data = data[0]
        out = []
        for segment in data:
            segment = torch.reshape(segment,(1,1,segment.shape[0],segment.shape[1],segment.shape[2],segment.shape[3]))
            with torch.no_grad():
                out.append(model(segment.to("cuda:0")))
        out = torch.concatenate(out)
        out = out.cpu()
        out = out.cpu().detach().numpy()
        if os.path.exists(extract_folder) and os.path.isdir(extract_folder):
            np.save(extract_folder+label_data + '.npy', out)
        else:
            os.mkdir(extract_folder)
            np.save(extract_folder + label_data + '.npy', out)

    return
if __name__ == "__main__":
    extract_folder = "ucf_feats/" #Saves features to this Folder
    data_loader = CustomVideoDataset(data="UCF_extract_features.csv",batch_size=1,test=True,prefix=".",video_prefix="ucf_segmented_videos/")# video_prefix is the directory where the segmented videos are saved
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.head = Identity()
    model.eval()
    model.to(device="cuda:0")
    extract_features(model=model,dataloader=data_loader,extract_folder=extract_folder)
