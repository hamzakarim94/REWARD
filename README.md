# REWARD
REWARD: Real-Time Weakly Labeled Video Anomaly Detection
To Extract features, use extract_features.py. Place the dataset in the main folder.
Use CustomVideoDataset(data="UCF_extract_features.csv",batch_size=1,test=True,prefix=".",video_prefix="ucf_segmented_videos/") accordingly. Data is supposed to be the data file that points to the videos, leave batch_size as 1, you can add prefix = "path_to_data/" is data is not in main directory. video_prefix is the directory you want to save your segmented videos in. Running extract features will extract features and produce semented video clips at the same time.
To generate KNN distances, Use Compute_KNN.py. feat_path is the directory you save the features in while save_loc is the directory to save the KNN distances.
To generate Psuedo Labels, use Generate_Psuedo_Labels.py and set the KNN save directory and the features save directory to the one used previously.
Run train.py and test.py to train and test using the generated psuedolabels.
