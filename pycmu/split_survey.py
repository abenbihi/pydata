import os, argparse
import glob

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

#import cst

def get_img_order(slice_id, mode='database', save=False):
    """Display sequence of img for cam0 and cam1.
    
    Args:
        slice_id: slice number
        mode: {database, query}
        save: save img order to file
    """
    img_dir = '%s/slice%d/%s/'%(cst.EXT_IMG_DIR, slice_id, mode)
    img_fn_l = ['slice%d/%s/%s'%(slice_id, mode, l) for l in sorted(os.listdir(img_dir)) ]
    
    # split cam0 and cam1 imgs
    c0_l = [l for l in img_fn_l if l.split("_")[2] == 'c0']
    c1_l = [l for l in img_fn_l if l.split("_")[2] == 'c1']
    
    # sort c0_l by timestamps
    time0_l = np.array([l.split("_")[3].split("us")[0] for l in c0_l])
    time0_order = np.argsort(time0_l)
    c0_sorted = np.array(c0_l)[time0_order]

    # sort c1_l by timestamps
    time1_l = np.array([l.split("_")[3].split("us")[0] for l in c1_l])
    time1_order = np.argsort(time1_l)
    c1_sorted = np.array(c1_l)[time1_order]
    
    if save:
        np.savetxt('meta/img_order/%d_%s_c0.txt'%(slice_id, mode), c0_sorted, fmt='%s')
        np.savetxt('meta/img_order/%d_%s_c1.txt'%(slice_id, mode), c1_sorted, fmt='%s')

    return c0_sorted, c1_sorted


def show_ordered_img(slice_id, mode='database'):
    """Display sequence of img for cam0 and cam1."""
    c0_sorted, c1_sorted = get_img_order(slice_id, mode, False)
    for c0_root_fn, c1_root_fn in zip(c0_sorted, c1_sorted):
        c0_fn = '%s/%s'%(cst.EXT_IMG_DIR, c0_root_fn) 
        c1_fn = '%s/%s'%(cst.EXT_IMG_DIR, c1_root_fn)
        print("c0_fn: %s\nc1_fn: %s\n"%(c0_fn, c1_fn))
        img0, img1 = cv2.imread(c0_fn), cv2.imread(c1_fn)
        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)


def get_survey_mano(slice_id, cam_id, mode='database'):
    """Interactive code to split queries into surveys.

    Display sequence of img for cam0 and cam1.
    """
    
    out_dir = 'meta/surveys/%s/fn'%slice_id
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_l = {}
    img_l[0] = np.loadtxt('meta/img_order/%d_%s_c0.txt'%(slice_id, mode), dtype=str)
    img_l[1] = np.loadtxt('meta/img_order/%d_%s_c1.txt'%(slice_id, mode), dtype=str)
   

    if mode == 'database': # it is already split so there is nothing to do
        np.savetxt('meta/surveys/%s/fn/c0_db.txt'%slice_id, img_l[0], fmt='%s')
        np.savetxt('meta/surveys/%s/fn/c1_db.txt'%slice_id, img_l[1], fmt='%s')
    else:
        survey_count = 0
        img_count, img_count_prev = 0,0
        while True:
            if img_count == (img_l[cam_id].shape[0]):
                # save last survey
                print("Survey %d: %d -> %d"%(survey_count, img_count_prev, img_count))
                np.savetxt('meta/surveys/%s/fn/c%d_%d.txt'%(slice_id, cam_id, survey_count),
                        img_l[cam_id][img_count_prev:img_count+1], fmt='%s')
                #np.savetxt('meta/surveys/%s/fn/c1_%d.txt'%(slice_id, survey_count),
                #        img_l[1][img_count_prev:img_count+1], fmt='%s')
                img_count_prev = img_count
                survey_count += 1
                break
            print('Cam %d\t%d\t%s'%(cam_id, img_count, img_l[cam_id][img_count]))

            print('img_fn: %s/%s'%(cst.EXT_IMG_DIR, img_l[cam_id][img_count]))
            img0 = cv2.imread('%s/%s'%(cst.EXT_IMG_DIR, img_l[cam_id][img_count]))
            cv2.imshow('img0', img0)
            k = cv2.waitKey(0) & 0xFF

            if k==ord("q"): # stop
                exit(0)
            elif k==ord("n"): # next img
                img_count +=1 
                continue
            elif k==ord("p"): # previous img
                if img_count >0:
                    img_count -= 1
                else:
                    print("Already first img. Go to next one")
                continue
            elif k==ord("k"): # you are at the end of a survey, split
                print("Survey %d: %d -> %d"%(survey_count, img_count_prev, img_count))
                
                np.savetxt('meta/surveys/%s/fn/c%d_%d.txt'%(slice_id, cam_id, survey_count),
                        img_l[cam_id][img_count_prev:img_count+1], fmt='%s')

                #np.savetxt('meta/surveys/%s/fn/c1_%d.txt'%(slice_id, survey_count),
                #        img_l[1][img_count_prev:img_count+1], fmt='%s')

                img_count_prev = img_count + 1
                survey_count += 1


def get_survey_auto(img_dir, slice_id, mode='database'):
    """Auto code to split queries into surveys based on camera poses."""
    out_dir = 'pycmu/meta/surveys/%d/'%slice_id
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # load ground-truth poses
    if mode == 'database':
        meta_v = np.loadtxt('%s/slice%d/ground-truth-database-images-slice%d.txt'
                %(img_dir, slice_id, slice_id), dtype=str)
        fn_v = meta_v[:,0]
        pose_v = meta_v[:,1:]
        print('pose_v.shape: ', pose_v.shape)
    elif mode == 'query':
        # gather gt metas
        meta_fn_l = glob.glob('%s/slice%d/camera-poses/*txt'%(img_dir, slice_id))
        meta_l = []
        for meta_fn in meta_fn_l:
            #print(meta_fn)
            if os.stat(meta_fn).st_size == 0:
                continue
            meta_l.append(np.loadtxt(meta_fn, dtype=str))
        if len(meta_l) == 0:
            print("Error: no ground-truth for this slice so I can't use it.")
            exit(1)

        meta_v = np.vstack(meta_l)
        #print(meta_v.shape)
        fn_v = meta_v[:,0]
        #pose_v = meta_v[:,5:].astype(np.float32)
        pose_v = meta_v[:,1:]
        print('pose_v.shape: ', pose_v.shape)

    # split c0 and c1
    fn0_v = np.array([l for l in fn_v if l.split("_")[2] == 'c0'])
    fn1_v = np.array([l for l in fn_v if l.split("_")[2] == 'c1'])
    idx0_v = np.in1d(fn_v, fn0_v).nonzero()[0]
    pose0_v = pose_v[idx0_v]
    idx1_v = np.in1d(fn_v, fn1_v).nonzero()[0]
    pose1_v = pose_v[idx1_v]

    # sort by timestamps
    time0_v = np.array([l.split("_")[3].split("us")[0] for l in fn0_v])
    time0_order = np.argsort(time0_v)
    fn0_v_sorted = fn0_v[time0_order]
    pose0_v_sorted = pose0_v[time0_order]

    time1_v = np.array([l.split("_")[3].split("us")[0] for l in fn1_v])
    time1_order = np.argsort(time1_v)
    fn1_v_sorted = fn1_v[time1_order]
    pose1_v_sorted = pose1_v[time1_order]


    if mode == 'database': # nothing to do but save as is
        fn0_v_sorted = np.array(['slice%d/database/%s'%(slice_id, l) for l in
            fn0_v_sorted])
        fn0_v_sorted = np.expand_dims(fn0_v_sorted, 1)
        np.savetxt('%s/c0_db.txt'%out_dir, np.hstack((fn0_v_sorted,
            pose0_v_sorted)), fmt='%s')

        fn1_v_sorted = np.array(['slice%d/database/%s'%(slice_id, l) for l in
            fn1_v_sorted])
        fn1_v_sorted = np.expand_dims(fn1_v_sorted, 1)
        np.savetxt('%s/c1_db.txt'%out_dir, np.hstack((fn1_v_sorted,
            pose1_v_sorted)), fmt='%s')
    elif mode == 'query': # meta holds several surveys, so split them
        # cam0
        fn0_v_sorted = np.array(['slice%d/query/%s'%(slice_id, l) for l in
            fn0_v_sorted])
        idx = 0
        img_num = fn0_v.shape[0]
        xy0 = pose0_v_sorted[0,5:7].astype(np.float32)

        survey_id = 0
        survey_l = [np.hstack((fn0_v_sorted[0], pose0_v_sorted[0,:]))]
        eps = 100
        for idx in range(1,img_num):
            xy1 = pose0_v_sorted[idx,5:7].astype(np.float32)
            d = np.sum( (xy0 - xy1)**2)
            #print('idx: %d\td: %.3f'%(idx, d))
            if np.sum( (xy0 - xy1)**2) > eps: # end of survey
                np.savetxt('%s/c0_%d.txt'%(out_dir, survey_id), np.vstack(survey_l), fmt='%s')
                survey_id += 1
                survey_l = []
            
            survey_l.append(np.hstack((fn0_v_sorted[idx], pose0_v_sorted[idx,:])))
            xy0 = xy1

        # cam1 (TODO: eliminate code duplication)
        fn1_v_sorted = np.array(['slice%d/query/%s'%(slice_id, l) for l in
            fn1_v_sorted])
        idx = 0
        img_num = fn1_v.shape[0]
        xy0 = pose1_v_sorted[0,5:7].astype(np.float32)

        survey_id = 0
        survey_l = [np.hstack((fn1_v_sorted[0], pose1_v_sorted[0,:]))]
        for idx in range(1,img_num):
            xy1 = pose1_v_sorted[idx,5:7].astype(np.float32)
            d = np.sum( (xy0 - xy1)**2)
            #print('idx: %d\td: %.3f'%(idx, d))
            if np.sum( (xy0 - xy1)**2) > eps: # end of survey
                np.savetxt('%s/c1_%d.txt'%(out_dir, survey_id), np.vstack(survey_l), fmt='%s')
                survey_id += 1
                survey_l = []
            
            survey_l.append(np.hstack((fn1_v_sorted[idx], pose1_v_sorted[idx,:])))
            xy0 = xy1
    
    else:
        print("Error: I don't know this mode: %s"%mode)
        exit(1)


def show_survey(img_dir, slice_id, cam_id, survey_id):
    print('\nslice_id: %d\tcam_id: %d\tsurvey_id: %d'%(slice_id, cam_id, survey_id))
   
    if survey_id == -1:
        survey_m = np.loadtxt('pycmu/meta/surveys/%d/c%d_db.txt'%(slice_id, cam_id), dtype=str)[:,0]
    else:
        survey_m = np.loadtxt('pycmu/meta/surveys/%d/c%d_%d.txt'%(slice_id, cam_id, survey_id), dtype=str)[:,0]

    data_count = 0
    for l in survey_m:
        img_fn = '%s/%s'%(img_dir, l)
        print(img_fn)
        if not os.path.exists(img_fn):
            print("Error: %s : no such file or directory"%img_fn)
            exit(1)

        img = cv2.imread(img_fn)
        cv2.imshow('img', img)
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)


if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--slice_id', type=int, required=True)
    parser.add_argument('--cam_id', type=int)
    parser.add_argument('--survey_id', type=int)
    args = parser.parse_args()

    if args.survey_id == -1:
        mode = 'database' # 'query'
    else:
        mode = 'query'

    if (0==1):
        show_ordered_img(args.slice_id, mode)

    if (0==1): # split slice into surveys (auto)
        get_survey_auto(args.img_dir, args.slice_id, 'database')
        get_survey_auto(args.img_dir, args.slice_id, 'query')
 
    if (1==1): # show specific survey
        show_survey(args.img_dir, args.slice_id, args.cam_id, args.survey_id)
