from Datasets import CustomVideoDataset
import torch
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
from slowfast.models import build_model
import pickle
from sklearn import metrics
import numpy as np
sig = torch.nn.Sigmoid()
def testing(model,dataloader,gt):
    model.eval()
    y_preds = []
    y_trues = []
    for cur_iter, data,video_path in dataloader:

        #shape = video.size()
        #video = torch.reshape(video,(1,shape[0],shape[1],shape[2],shape[3],shape[4]))
        #data = torch.stack(data)
        if data == None or video_path == None:
            break
        label_data =video_path[0].split('/')[-1].split('.')[0]
        print(label_data)
        true = gt[label_data + '_i3d']
        data = data[0]
        out = []
        for segment in data:
            segment = torch.reshape(segment,(1,1,segment.shape[0],segment.shape[1],segment.shape[2],segment.shape[3]))
            with torch.no_grad():
                out.append(sig(model(segment.to("cuda:0"))))
        out = torch.concatenate(out)
        out = out.cpu()
        out = out.cpu().detach().numpy()
        segmented_preds = np.array_split(true, len(out))
        for i, segs in enumerate(segmented_preds):
            segs = np.full(segs.shape, out[i][0])
            segmented_preds[i] = segs
        pred = np.concatenate(segmented_preds)
        y_trues.append(true)
        y_preds.append(pred)
    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)
    auc = metrics.roc_auc_score(y_trues, y_preds)
    AP = metrics.average_precision_score(y_trues, y_preds)
    print("AUC ",auc)
    print("AP ",AP)
    return auc,AP
if __name__ == "__main__":
    with open('gt.pkl', 'rb') as file:
        loaded_gt = pickle.load(file)
    test_data_loader = CustomVideoDataset(data="testing_data.csv", batch_size=1, test=True,cfg=cfg, prefix=".")
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    model.to(device="cuda:0")
    testing(model=model,dataloader=test_data_loader,gt = loaded_gt)
