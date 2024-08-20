# REWARD
REWARD: Real-Time Weakly Labeled Video Anomaly Detection
## Extracting Features
To Extract features, use **extract_features.py**. Place the dataset in the main folder.
Use `CustomVideoDataset(data="UCF_extract_features.csv",batch_size=1,test=True,prefix=".",video_prefix="ucf_segmented_videos/")` accordingly. 
Data is supposed to be the data file that points to the videos, leave batch_size as 1, you can add prefix = "path_to_data/" is data is not in main directory. video_prefix is the directory you want to save your segmented videos in. Running extract features will extract features and produce semented video clips at the same time.
## Computing KNN Distances
To generate KNN distances, Use **Compute_KNN.py**. feat_path is the directory you save the features in while save_loc is the directory to save the KNN distances. You can change the value of **k** to change the number of nieghbours.
## Computing Psuedo Lables
To generate Psuedo Labels, use **Generate_Psuedo_Labels.py** and set the KNN save directory (`initial = 'KNN_UCF_UNI_6/'`) and the features save directory (`feat_path = "ucf_feats/"`) to the one used previously.
## Train and test
Run **train.py** and **test.py** to train and test using the generated psuedo labels. Use `batch_size=1` when testing.
## Checkpoints
Below are the links for the checkpoints trained on UCF-Crime and XD-Violence.<br/> 
Ucf-crime: https://drive.google.com/file/d/14HQKEoLl3ZinlTb1PmdMzWnmS67qcyQb/view?usp=sharing\n
XD-Violence: https://drive.google.com/file/d/1o5AW5QFFpPpIfnHGznVn-maAooORRKFg/view?usp=sharing\n
Note that due to a data corruption issue we lost the orignal checkpoint for UCF-Crime, hence the model linked here has a slightly less auc at 86.4 instead of 86.9.

