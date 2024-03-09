import os
from os.path import join
import numpy as np
import cv2
import pandas as pd

def normalize_pose(image_name,class_,train_or_val):
    results = [image_name, class_]
    h, w, channel = cv2.imread(f"poses_data/{train_or_val}/{class_}/{image_name}").shape
    with open(f"poses_data/{train_or_val}/{class_}_json/{image_name.split('.')[0]}.npy", 'rb') as f:
        pose_data = np.load(f)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
        for pairs in pose_data:
            norm_x, norm_y = round(pairs[0]/w ,2), round(pairs[1]/h, 2)
            results.extend([norm_x,norm_y])
    return results

a = []
r = "new_poses_data"
classes_ = ["pose","not_pose"]
for set in ["train","val"]:
    for class_ in classes_:
        for image_name in os.listdir(join(r,set,class_)):
            l = normalize_pose(image_name, class_,set)
            a.append(l)    
    my_df = pd.DataFrame(a)
    my_df.to_csv(f'small_{set}_poses_normalized_no_feature_selection.csv', index=False, header=False)




