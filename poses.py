from keypoints_detectron2 import pose_dir
import os,time
from os.path import join

def detectron_poses():
    in_dir = "/media/HDD2/VITON/pose_classification/poses_data/val/not_pose"
    out_dir = "/media/HDD2/VITON/pose_classification/poses_data/val/not_pose_img"
    json_dir = "/media/HDD2/VITON/pose_classification/poses_data/val/not_pose_json"
    start = time.time()
    pose_dir(in_dir, out_dir, json_dir)
    end = time.time()
    data = {
		"text": f"processed {len(os.listdir(json_dir))} images in {round((end-start),2)} seconds"
	}

    return data


detectron_poses()