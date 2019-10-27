"""Here I go again, playing with 3D geometry."""

import argparse
from datetime import datetime
from math import pi,cos,sin 
import os
import time

import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from pyquaternion import Quaternion

import tools.angles

def gen_init_model(args):
    year = '20%d'%(args.survey_id/10000)
    
    good_poses_fn = 'pysymphony/meta/poses/%d.txt'%(args.survey_id)
    good_poses = np.loadtxt(good_poses_fn)
    
    colmap_dir = 'pysymphony/meta/colmap_prior/%d'%(args.surve_id)
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)
    
    # empty 3D points
    colmap_f = open('%s/points3D.txt'%colmap_dir, 'w')
    colmap_f.close()
    
    # camera params (intrinsics)
    cam_f = open('%s/cameras.txt'%colmap_dir, 'w')
    camera_id = 1
    cam_f.write("%d PINHOLE 700 480 780.170806 708.378535 317.745657 246.801583\n"%camera_id)
    cam_f.close()
    
    # camera extrinsincs i.e. image params
    colmap_f = open('%s/images.txt'%colmap_dir, 'w')
    
    # list of images to use 
    img_l_f = open('%s/img_list.txt'%(colmap_dir), 'w')
    
   
    # get lines of queried imgs
    query_id_v = np.arange(args.id_start, args.id_stop, args.id_iter)
    time_idx_v = np.in1d(good_poses[:,0], query_id_v).nonzero()[0]
    print('\ntime_idx_v', good_poses[time_idx_v,0])
    img_id_v = good_poses[time_idx_v,0].astype(np.int)
    img_poses_v = good_poses[time_idx_v,1:]
    
    for i, img_id in enumerate(img_id_v):
        seq = int(img_id/1000)
    
        qw, qx, qy, qz, tx, ty, tz = img_poses_v[i,:]
    
        # colmap format
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = i + 1 # !! Must be >0, not >=0 !!
        #camera_id = i + 1
        if args.bag_mode: # old nomenclature
            img_fn = '%04d/%04d.jpg'%(seq, img_id%1000)
        else:
            img_fn = '%04d/%02d%03d.jpg'%(seq, seq, img_id%1000)
    
        colmap_f.write('%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %d %s\n\n' 
                %(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, img_fn))
        img_l_f.write('%s\n'%img_fn)
    
    colmap_f.close()
    img_l_f.close()
        
    
    print('%s survey_id:%s data_count:%d' %(datetime.now(), args.survey_id,
        img_id_v.shape[0]))

    
def gen_init_model_from_spline_optposes(args):
    """
    I use Cedric's code to interpolate good poses from optposes.csv on each
    image.
    """
    year = '20%d'%(args.survey_id/10000)
    
    good_poses_fn = 'pysymphony/meta/poses/%d.txt'%(args.survey_id)
    good_poses = np.loadtxt(good_poses_fn, delimiter=',')
    
    colmap_dir = 'pysymphony/meta/colmap_prior/%d'%(args.surve_id)
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)
    
    # empty 3D points
    colmap_f = open('%s/points3D.txt'%colmap_dir, 'w')
    colmap_f.close()
    
    # camera params (intrinsics)
    cam_f = open('%s/cameras.txt'%colmap_dir, 'w')
    camera_id = 1
    cam_f.write("%d PINHOLE 700 480 780.170806 708.378535 317.745657 246.801583\n"%camera_id)
    cam_f.close()
    
    # camera extrinsincs i.e. image params
    colmap_f = open('%s/images.txt'%colmap_dir, 'w')
    
    # list of images to use 
    img_l_f = open('%s/img_list.txt'%(colmap_dir), 'w')
    
   
    # get lines of queried imgs
    query_id_v = np.arange(args.id_start, args.id_stop, args.id_iter)
    time_idx_v = np.in1d(good_poses[:,3], query_id_v).nonzero()[0]
    print('\ntime_idx_v', good_poses[time_idx_v,3])
    img_id_v = good_poses[time_idx_v,3].astype(np.int)
    img_poses_v = good_poses[time_idx_v,4:]

    print_island = 0
    
    for i, img_id in enumerate(img_id_v):
        seq = int(img_id/1000)

        x, y, rot, pan = img_poses_v[i,:]
    
        # transformation from boat to world frame
        R_w_b = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0,             0,          1]])
        t_w_b = np.array([x,y,0]) # t: world -> camera

        T_w_b = np.eye(4)
        T_w_b[:3,:3] = R_w_b
        T_w_b[:3,3] =  t_w_b
        #pose_l.append(T_w_b)

        T_b_w = np.linalg.inv(T_w_b)
        R_b_w = T_b_w[:3,:3]
        t_b_w = T_b_w[:3,3]
        
        #print(img_pan_l[i]) 
        # transformation from boat to camera
        # TODO: Handling the island part is KO. This is not a priority for now
        # but someday you should solve this bug.
        if np.abs(pan + np.pi/2) < 1e-1:
             # island, camera is turned the other way
            if print_island == 0:
                print('NOW island island')
                print_island = 1
            R_c_b = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]])
            break
        else: # not island
            R_c_b = np.array([
                [-1, 0, 0],
                [0, 0, -1],
                [0, -1, 0]])

        t_c_b = 0
        T_c_b = np.eye(4)
        T_c_b[:3,:3] = R_c_b
        T_c_b[:3,3] = t_c_b

        T_c_w = np.dot(T_c_b, T_b_w)
        R_c_w = T_c_w[:3,:3]
        t_c_w = T_c_w[:3,3]


        qw, qx, qy, qz = Quaternion(matrix=R_c_w)
        tx, ty, tz = t_c_w[0], t_c_w[1], t_c_w[2]

        # colmap format
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = i + 1 # !! Must be >0, not >=0 !!
        #camera_id = i + 1
        if args.bag_mode: # old nomenclature
            img_fn = '%04d/%04d.jpg'%(seq, img_id%1000)
        else:
            img_fn = '%04d/%02d%03d.jpg'%(seq, seq, img_id%1000)
    
        colmap_f.write('%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %d %s\n\n' 
                %(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, img_fn))
        img_l_f.write('%s\n'%img_fn)
    
    colmap_f.close()
    img_l_f.close()
        
    
    print('%s survey_id:%s data_count:%d' %(datetime.now(), args.survey_id,
        img_id_v.shape[0]))


def test_gen(pose_l):
    # plot T
    X, Y, Z, U, V = [],[],[],[],[]
    for T in pose_l:
        X.append(T[0,3])
        Y.append(T[1,3])
        Z.append(T[2,3])
        R = T[0:3,0:3]
        
        out = tools.angles.so3_to_euler(R)
        rr = out[0,0]
        pp = out[0,1]
        yy = out[0,2] 
        U.append(cos(yy))
        V.append(sin(yy))
            
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    U = np.array(U)
    V = np.array(V)
    
    fig = plt.figure(figsize=(20,20))
    Q = plt.quiver(X, Y, U, V, units='width')
    plt.savefig('toto.png')
    plt.close()
    toto = cv2.imread('toto.png')
    cv2.imshow('toto', toto)
    cv2.waitKey(0)
   

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_trial', type=int, help='')
    parser.add_argument('--survey_id', type=int, required=True, default=151027)
    parser.add_argument('--id_start', type=int, help='Start with this id of this survey_id. 5 digits.')
    parser.add_argument('--id_stop', type=int, help='Stop with this img id of this survey_id. 5 digits.')
    parser.add_argument('--id_iter', type=int, help='Sample every id_iter img.')
    parser.add_argument('--bag_mode', type=int, help='Set to 1 if you use the old nomenclature.')
    args = parser.parse_args()

    #gen_init_model(args)
    gen_init_model_from_spline_optposes(args)

