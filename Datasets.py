import pandas as pd
import torch
import cv2
import numpy as np
import os
import decord
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def denormalize(tensor, mean, std):
    tensor = tensor.cpu()
    size = tensor.size()
    # tensor = torch.reshape(tensor,(size[1],size[2],size[3],size[4]))
    tensor = torch.permute(tensor, (1, 2, 3, 0))
    if type(std) == list:
        std = torch.tensor(std)
        tensor = tensor * std.cpu()
    if type(mean) == list:
        mean = torch.tensor(mean)
        tensor = tensor + mean.cpu()
    tensor = tensor * 255.0
    tensor_float = tensor.to("cuda:0")
    tensor = tensor.to(torch.uint8)

    return tensor.numpy(), tensor_float


def sample_frames(self,video_path, frame_count):
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path)

    # Get total number of frames in the video
    total_frames = len(vr)
    if total_frames > 10000: #If a video is too long, it might overload the memory, to avoid this we restrict maximum number of total frames to 10k
        total_frames = 10000
    if frame_count == None:
        if total_frames >= 6*self.cfg.DATA.NUM_FRAMES:#divide videos into segments each having 32 frames
            frame_indices = np.linspace(0, total_frames - 1, int(total_frames/(6*self.cfg.DATA.NUM_FRAMES))*self.cfg.DATA.NUM_FRAMES, dtype=np.int32)
        else:
            frame_indices = np.linspace(0, total_frames - 1, int(total_frames / (self.cfg.DATA.NUM_FRAMES)) * self.cfg.DATA.NUM_FRAMES, dtype=np.int32)
    else:
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, frame_count, dtype=np.int32)

    # List to store sampled frames
    sampled_frames = []

    # Iterate through each frame index and extract the frame
    for frame_index in frame_indices:
        # Read the frame
        frame = vr[frame_index].asnumpy()

        h, w, _ = frame.shape
        frame = frame[(h - 224) // 2:(h + 224) // 2, (w - 224) // 2:(w + 224) // 2]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        sampled_frames.append(frame)

    return sampled_frames


class CustomVideoDataset():
    def __init__(self, data, batch_size=1, prefix='UCFCRIME',test = False,cfg=None,video_prefix=None,transform=None):
        self.path_a = data
        self.batch_size = batch_size
        self.transform = transform
        self.prefix = prefix
        self.cfg = cfg
        df_a = pd.read_csv(self.path_a,header=None)
        df_a = df_a.values.squeeze().tolist()
        self.files_a = df_a
        self.index_a = 0
        self.test = test
        self.video_prefix = video_prefix
    def __len__(self):
        return min(len(self.files_a), len(self.files_b)) // self.batch_size

    def __getitem__(self, idx):
        if self.index_a >= len(self.files_a):
            return None, None, None

        batch_files_a = self.files_a[self.index_a:self.index_a + self.batch_size]
        if self.test == True:
            videos_a = self.load_videos(batch_files_a,self.prefix,self.video_prefix,frame_count=None)
        else:
            videos_a = self.load_videos(batch_files_a, self.prefix,self.video_prefix,frame_count=self.cfg.DATA.NUM_FRAMES)
        self.index_a += self.batch_size

        return int(self.index_a/self.batch_size),videos_a,batch_files_a

    def load_videos(self, file_list,prefix,video_prefix,frame_count):
        video_list = []
        for file in file_list:
            if self.test == True:
                video = self.read_video(prefix + "/" + file, frame_count=frame_count)
            else:
                video = self.read_video(prefix+"/"+file[0],frame_count=frame_count)
            if self.transform:
                video = self.transform(video)
            video = video.view(int(video.shape[0]/self.cfg.DATA.NUM_FRAMES), self.cfg.DATA.NUM_FRAMES, 224, 224, 3)
            video = torch.permute(video,(0,4,1,2,3))
            if video_prefix != None:
                self.save_video(video_prefix, file.split("/")[-1], video)
            video_list.append(video)
        return video_list

    def read_video(self, video_path,frame_count):
        sampled_frames = sample_frames(video_path=video_path, frame_count=frame_count)
        sampled_frames_tensor = torch.stack([torch.tensor(frame) for frame in sampled_frames])
        normalized_frames = tensor_normalize(sampled_frames_tensor, mean, std)
        return normalized_frames

    def save_video(self,path_prefix,video_name,video):
        for c,clip in enumerate(video):
            clip = denormalize(clip,mean,std)[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
            fps = 5  # Frames per second
            height, width = clip.shape[1], clip.shape[2]
            if not os.path.exists(path_prefix):
                os.mkdir(path_prefix)
            # Create the VideoWriter object
            out = cv2.VideoWriter(path_prefix+video_name.split(".")[0]+"_"+str(c)+".mp4", fourcc, fps, (width, height))

            # Iterate through the frames and write them to the video file
            for i in range(clip.shape[0]):
                frame = clip[i]
                # Write the frame to the video file
                out.write(frame)

            # Release the VideoWriter object
            out.release()
    def reset(self):
        self.index_a = 0
