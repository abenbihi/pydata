"""Set of scripts to manage data from/for colmap."""
import os, argparse
import glob

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

import tools.angles
import tools.read_model

def pose2colmap(args):
    """Writes img poses from the CMU-Seasons dataset to colmap format. 
    
        Use the same camera convention but expects the img pose with the 
        following file format: 
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

        Useful for when you want to run colmap on CMU-Seasons and use the known
        poses.
    """
    if args.survey_id == -1:
        survey_dir = "pycmu/meta/surveys/%d/%d_c%d_db/"%(
            args.slice_id, args.slice_id, args.cam_id)
    else:
        survey_dir = "pycmu/meta/surveys/%d/%d_c%d_%d/"%(
            args.slice_id, args.slice_id, args.cam_id, args.survey_id)
    survey_m = np.loadtxt("%s/pose.txt"%survey_dir, dtype=str)
    
    colmap_dir = "%s/colmap_prior"%survey_dir # output dir
    print("colmap_dir: %s"%colmap_dir)
    if not os.path.exists(colmap_dir):
        os.makedirs(colmap_dir)
    
    # output: empty 3D points
    colmap_f = open('%s/points3D.txt'%colmap_dir, 'w')
    colmap_f.close()
    
    # output: camera params (intrinsics)
    camera_id = 1 # camera count i.e. it is not the same id as c0 or c1
    cam_f = open('%s/cameras.txt'%colmap_dir, 'w')
    
    #if cam_id==0:
    #    cam_f.write('%d OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571\n'%camera_id)
    #    #cam_f.write("%d PINHOLE 1024 768 868.99 866.06 525.94 420.04\n"%camera_id)
    #    #print("%d PINHOLE 1024 768 868.99,866.06,525.94,420.04\n"%data_count)
    #else:
    #    #cam_f.write("%d PINHOLE 1024 768 873.38 876.49 529.32 397.27\n"%camera_id)
    #    cam_f.write("%d OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579\n"%camera_id)
    #cam_f.close()
    
    # output: camera extrinsincs i.e. image params
    colmap_f = open('%s/images.txt'%colmap_dir, 'w')

    # output: image list to use 
    img_l_f = open('%s/image_list.txt'%colmap_dir, 'w')
  
    data_count = 0
    for l in survey_m:
        img_fn = l[0]
        img_id = int(img_fn.split("_")[1])
        qw, qx, qy, qz, cx, cy, cz = [float(ll) for ll in l[1:]]
        #if (qw==-1):
        #    continue

        R = tools.angles.quat2mat([qw, qx, qy, qz]) # cam -> world
        c = np.array([cx, cy, cz]) # cam center
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = c

        t = -np.dot(R, c)
        tx, ty, tz = t # cam translation

        data_count += 1
        colmap_f.write('%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %d %s\n\n' 
            %(data_count, qw, qx, qy, qz, tx, ty, tz, camera_id, img_fn))
        img_l_f.write('%s\n' %(img_fn) )
        
        if args.cam_id==0:
            cam_f.write('%d OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571\n'%camera_id)
            #cam_f.write("%d PINHOLE 1024 768 868.99 866.06 525.94 420.04\n"%camera_id)
            #print("%d PINHOLE 1024 768 868.99,866.06,525.94,420.04\n"%data_count)
        else:
            cam_f.write("%d OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579\n"%camera_id)
            #cam_f.write("%d PINHOLE 1024 768 873.38 876.49 529.32 397.27\n"%camera_id)
        camera_id += 1

    cam_f.close()
    colmap_f.close()
    img_l_f.close()

    ## manual matches
    ## It specifies which image to match during the feature matching step.
    ## By default colmap does exhaustive matching but if you already know that
    ## some images do not overlap, it will save time to specify the matches.
    #overlap = 10 # number of consecutive img that overlap, you may adjust it
    #match_fn = '%s/image_pairs_to_match_intra.txt'%colmap_dir
    #match_l_f = open(match_fn, 'w')
    #img_fn_l = [l.split("\n")[0].split(" ")[-1] for l in
    #        open('%s/images.txt'%colmap_dir).readlines() if l!='\n']
    #img_num = len(img_fn_l)
    #
    #for i, img_fn in enumerate(img_fn_l):
    #    ref_img_fn = img_fn_l[i]
    #    start = min(i+1, img_num - 1)
    #    end = min(i+1+overlap, img_num-1)
    #    for j in range(start, end):
    #        match_l_f.write('%s %s\n'%(img_fn_l[i], img_fn_l[j]))
    #match_l_f.close()



def read_depth_colmap(slice_id, cam_id, survey_id, colmap_ws, mode,
        save_type=np.float32, save_fmt="txt", save_visu=False, display=False):
    """Converts colmap-format bin depth/normal maps to img/txt format.

    Args:
        mode: {depth, normal}
    """
    if survey_id == -1:
        ws_dir = "%s/%d_%d_db/"%(colmap_ws, slice_id, cam_id)
    else:
        ws_dir = "%s/%d_%d_%d/"%(colmap_ws, slice_id, cam_id, 
                survey_id)
    img_fn_v = np.loadtxt('%s//mano/image_list.txt'%ws_dir, dtype=str)
    depth_dir = '%s/dense/stereo/depth_maps/'%ws_dir
    
    out_depth_dir = '%s/dense/stereo/depth_txt/'%ws_dir
    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)
    
    min_l = [] # list of min depth of depth map
    max_l = [] # list of max depth of depth map
    fn_l = []
    for i, img_root_fn in enumerate(img_fn_v):
        if i%20==0:
            print("%d/%d"%(i, img_fn_v.shape[0]))
        root_fn = img_root_fn.split("/")[-1].split(".")[0]
        fn_l.append(root_fn)
        if mode == "normal":
            fn = '%s/%s.photometric.bin'%(depth_dir, img_root_fn)
        elif mode == "depth":
            depth_fn = '%s/%s.geometric.bin'%(depth_dir, img_root_fn)
        else:
            raise ValueError("Unknown mode: %s"%mode)

        depth_map = tools.read_model.read_array(depth_fn)
        
        # copied from
        # https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py 
        min_depth, max_depth = np.percentile(depth_map, [5, 95])
        if min_depth <= 1e-5:
            min_depth = 0
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        min_l.append(min_depth)
        max_l.append(max_depth)

        # save depth map in img format
        if save_visu:
            plt.figure()
            plt.imshow(depth_map)
            plt.colorbar()
            plt_fn = "%s/%s.png"%(out_depth_dir, root_fn)
            print(plt_fn)
            plt.title('%s'%root_fn)
            plt.savefig(plt_fn)
            plt.close()
        
        root_fn = img_root_fn.split("/")[-1].split(".")[0]
        if save_type == np.float32:
            # save depth map in txt file with float32 precision
            out_fn = "%s/%s.txt"%(out_depth_dir, root_fn)
            print(out_fn)
            np.savetxt(out_fn, depth_map)
        else:
            # remap
            old_min, old_max = min_depth, max_depth
            new_min, new_max = 0, np.iinfo(save_type).max
            depth_map_u = new_min + (depth_map - old_min)*(new_max - new_min)/(
                    old_max - old_min)
            depth_map_u = depth_map_u.astype(save_type)
            out_fn = "%s/%s.%s"%(out_depth_dir, root_fn, save_fmt)
            if save_fmt == "txt":
                np.savetxt(out_fn, depth_map_u, fmt="%d")
            elif save_fmt == "png":
                cv2.imwrite(out_fn, depth_map_u)

        # display the depth map img
        if display and save_visu:
            toto = cv2.imread(plt_fn)
            cv2.imshow('toto', toto)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                exit(0)
        
    np.savetxt("%s/min.txt"%out_depth_dir, np.array(min_l), fmt="%.8f")
    np.savetxt("%s/max.txt"%out_depth_dir, np.array(max_l), fmt="%.8f")
    np.savetxt("%s/fn.txt"%out_depth_dir, np.array(fn_l), fmt="%s")

def test_depth_precision(slice_id, cam_id, survey_id, colmap_ws, mode):
    """Computes the RMSE introduced in the depth map when saving with uint16
    precision instead of float32."""
    depth_dir = '%s/%d_%d_%d/dense/stereo/depth_maps/'%(
            colmap_ws, slice_id, cam_id, survey_id) # colmap depth
    img_fn_v = np.loadtxt('%s/%d_%d_%d/mano/image_list.txt'%(
        colmap_ws, slice_id, cam_id, survey_id), dtype=str) 

    for root_fn in img_fn_v:
        if mode == "normal":
            fn = '%s/%s.photometric.bin'%(depth_dir, root_fn)
        elif mode == "depth":
            depth_fn = '%s/%s.geometric.bin'%(depth_dir, root_fn)
        else:
            raise ValueError("Unknown mode: %s"%mode)

        if not os.path.exists(depth_fn):
            continue
        depth_map = tools.read_model.read_array(depth_fn)
        
        # copied from
        # https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py 
        min_ = np.min(depth_map)
        max_ = np.max(depth_map)
        print("min_: %.8f\tmax_: %.8f"%(min_, max_))

        min_depth, max_depth = np.percentile(depth_map, [5, 95])
        print("min_depth: %.8f\tmax_depth: %.8f"%(min_depth, max_depth))
        if min_depth <= -1e-5:
            raise ValueError("min_depth<=0: %.15f"%min_depth)

        if min_depth <= 1e-5:
            min_depth = 0
            #print("Manual min_depth=0")
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        # remap
        type_ = np.uint16
        old_min, old_max = min_depth, max_depth
        new_min, new_max = 0, np.iinfo(type_).max
        depth_map_u = new_min + (depth_map - old_min)*(new_max - new_min)/(
                old_max - old_min)
        depth_map_u = depth_map_u.astype(type_)

        # to file
        np.savetxt("toto.txt", depth_map_u, fmt="%d")
        cv2.imwrite("toto.png", depth_map_u)
    
        # reconstruct float depth map from the uint one
        # online
        new_min, new_max = min_depth, max_depth
        old_min, old_max = 0, np.iinfo(type_).max
        depth_new = new_min + (depth_map_u - old_min)*(new_max - new_min)/(
                old_max - old_min)
        error = np.sqrt(np.sum((depth_new - depth_map)**2))
        precision = error / np.mean(depth_new)
        print("error_online: %.3f \t precision: %.3f"%(error, precision))
    
        # from txt file
        depth_map_u = np.loadtxt("toto.txt", dtype=type_)
        new_min, new_max = min_depth, max_depth
        old_min, old_max = 0, np.iinfo(type_).max
        depth_new = new_min + (depth_map_u - old_min)*(new_max - new_min)/(
                old_max - old_min)
        error = np.sqrt(np.sum((depth_new - depth_map)**2))
        precision = error / np.mean(depth_new)
        print("error_from_txt: %.3f\tprecision: %.3f"%(error, precision))

        # from txt img
        depth_map_u = cv2.imread("toto.png", cv2.IMREAD_UNCHANGED)
        new_min, new_max = min_depth, max_depth
        old_min, old_max = 0, np.iinfo(type_).max
        depth_new = new_min + (depth_map_u - old_min)*(new_max - new_min)/(
                old_max - old_min)
        error = np.sqrt(np.sum((depth_new - depth_map)**2))
        precision = error / np.mean(depth_new)
        input("error_from_img: %.3f\tprecision: %.3f"%(error, precision))



def colmap2txt(slice_id, cam_id, survey_id, colmap_ws):
    """Converts bin colmap depth to txt depth map."""
    if survey_id == -1:
        ws_dir = "%s/%d_%d_db/"%(colmap_ws, slice_id, cam_id)
    else:
        ws_dir = "%s/%d_%d_%d/"%(colmap_ws, slice_id, cam_id, 
                survey_id)

    depth_dir = "%s/dense/stereo/depth_maps/"%ws_dir
    out_depth_dir = '%s/dense/stereo/depth_txt/'%ws_dir
    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)
    img_fn_v = np.loadtxt('%s//mano/image_list.txt'%ws_dir, dtype=str)
    
    # convert bin colmap-format camera pose to txt format
    pose_fn = '%s/cmu/%d_%d_%d/dense/sparse/images.bin'%(
        COLMAP_WS_DIR, slice_id, cam_id, survey_id)
    images = read_poses_binary(pose_fn)
    pose_f = open('%s/poses.txt'%out_depth_dir, 'w')
    for k,v in images.items():
        pose_f.write('%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n'%(
            (img_fn_v[k-1],)+ tuple(np.hstack(v))))
    pose_f.close()

    # get depth maps
    read_depth_colmap(slice_id, cam_id, survey_id, img_dir, colmap_ws, "depth")



def plot_colmap_pose(slice_id, cam_id, survey_id, colmap_ws):
    depth_dir = '%s/%d_%d_%d/dense/stereo/depth_maps/'%(
            colmap_ws, slice_id, cam_id, survey_id)
    out_depth_dir = '%s/cmu/%d_%d_%d/dense/stereo/depth_txt/'%(
            colmap_ws, slice_id, cam_id, survey_id)
    if not os.path.exists(out_depth_dir):
        os.makedirs(out_depth_dir)
    
    # get camera pose
    poses_str_l = [l.split("\n")[0] for l in
            open('%s/poses.txt'%out_depth_dir).readlines()]
    
    poses_l = []
    for l in poses_str_l:
        qw, qx, qy, qz, tx, ty, tz = [float(ll) for ll in l.split(',')[1:]]
        R_c_w =  tools.angles.quat2mat([qw, qx, qy, qz])
        t_c_w = np.array([tx, ty, tz])
        T_c_w = np.eye(4) # camera -> world
        T_c_w[:3,:3] = R_c_w
        T_c_w[:3,3] = t_c_w

        T_w_c = np.linalg.inv(T_c_w) # world -> camera
        poses_l.append(T_w_c)

    # plot T
    X, Y, Z, U, V = [],[],[],[],[]
    for T in poses_l:
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
    
    fig = plt.figure(figsize=(40,40))
    Q = plt.quiver(X, Y, U, V, units='width')
    plt.savefig('toto.png')
    plt.close()
    toto = cv2.imread('toto.png')
    cv2.imshow('toto', toto)
    cv2.waitKey(0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_id", type=int, required=True)
    parser.add_argument("--cam_id", type=int, required=True)
    parser.add_argument("--survey_id", type=int, required=True)
    parser.add_argument("--colmap_ws", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()
    
    # generate colmap ws to run colmap from known poses
    pose2colmap(args)
    
    #gen_match_manual(args)
    #gen_match_inter(args)
    #test_match_inter(args)

    # convert colmap-format (bin) depth/normal maps to img/txt format
    
    ## save colmap depth to txt file with float32 precision
    #read_depth_colmap(args.slice_id, args.cam_id, args.survey_id,
    #        args.colmap_ws, "depth", save_visu=True, display=False)

    ## save colmap depth to png file with uint16 precision
    #read_depth_colmap(args.slice_id, args.cam_id, args.survey_id,
    #        args.colmap_ws, "depth", save_type=np.uint16, save_fmt="png")
    
    ## computes the error introduced by the bin->png/txt conversion
    #test_depth_precision(args.slice_id, args.cam_id, args.survey_id, args.colmap_ws, "depth")
    
    ## converts bin colmap depth and pose to txt
    #colmap2txt(args.slice_id, args.cam_id, args.survey_id, args.colmap_ws)

