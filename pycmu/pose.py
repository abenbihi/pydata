import os, argparse
import glob
import time

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

def get_gt_pose(args):
    """Get pose from Extended-CMU-Seasons for each survey when available.
    It holds the rotation quaternion from camera to world and the camera CENTER
    not the translation from world to camera.
    """
    if args.survey_id == -1: # it is already split so there is nothing to do
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_db"%(args.slice_id,
                args.slice_id, args.cam_id)
        pose_fn_l  = ["%s/slice%d/ground-truth-database-images-slice%d.txt"%(
            args.img_dir, args.slice_id, args.slice_id)]
    else:
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_%d"%(args.slice_id,
                args.slice_id, args.cam_id, args.survey_id)
        pose_fn_l  = glob.glob("%s/slice%d/camera-poses/*.txt"%(args.img_dir, args.slice_id))
    
    if not os.path.exists("%s/fn.txt"%out_dir):
        print("There is no such survey: %d_c%d_%d"%(
            args.slice_id, args.cam_id, args.survey_id))
        return 0
    
    all_pose_l = []
    for pose_fn in pose_fn_l:
        if os.stat(pose_fn).st_size == 0:
            continue
        all_pose_l.append(np.loadtxt(pose_fn, dtype=str))
    if len(all_pose_l) == 0:
        print("There is no gt pose available for survey %d_c%d_%d"%(
            args.slice_id, args.cam_id, args.survey_id))
        fn_v = np.loadtxt("%s/fn.txt"%out_dir, dtype=str)
        fn_v = fn_v.reshape(fn_v.shape[0], 1)
        mock_pose = np.array(["-1.0 -1.0 -1.0 -1.0 -1.0 -1.0 1.0"]).reshape(1,1)
        mock_pose = np.tile(mock_pose, (fn_v.shape[0],1))
        mock_pose = np.hstack((fn_v, mock_pose))
        np.savetxt("%s/pose.txt"%out_dir, mock_pose, fmt="%s")
        return(0)
    
    pose_v = np.vstack(all_pose_l)
    fn_v = np.loadtxt("%s/fn.txt"%out_dir, dtype=str)
    fn_v = np.array([l.split("/")[-1] for l in fn_v])
    #print(pose_v.shape)
    null_pose = -1 * np.ones(7)
    # TODO: avoid for loop

    pose_l = [] 
    for fn in fn_v:
        idx = np.where(pose_v[:,0] == fn)[0]
        if idx.size == 0:
            #print(idx)
            #input("NULL")
            pose_l.append(null_pose)
        else:
            #input("%d"%idx)
            pose_l.append(pose_v[idx, 1:])

    pose_v = np.vstack(pose_l)
    fn_v = np.loadtxt("%s/fn.txt"%out_dir, dtype=str)
    pose_v = np.hstack((np.expand_dims(fn_v, 1), pose_v))
    np.savetxt("%s/pose.txt"%out_dir, pose_v, fmt="%s")
    
           
if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--slice_id', type=int, required=True)
    parser.add_argument('--cam_id', type=int)
    parser.add_argument('--survey_id', type=int)
    args = parser.parse_args()
    
    get_gt_pose(args)
