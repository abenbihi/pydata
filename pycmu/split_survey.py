import os, argparse
import glob
import time

import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

def get_img_order(args):
    """Display sequence of img for cam0 and cam1.
    
    Args:
        slice_id: slice number
        mode: {database, query}
        save: save img order to file
    """
    # get unordered img list
    if args.survey_id == -1:
        img_dir = '%s/slice%d/database/'%(args.img_dir, args.slice_id)
        img_fn_l = ["slice%d/database/%s"%(args.slice_id, l) for l in sorted(os.listdir(img_dir)) ]
    else:
        img_dir = '%s/slice%d/query/'%(args.img_dir, args.slice_id)
        img_fn_l = ["slice%d/query/%s"%(args.slice_id, l) for l in sorted(os.listdir(img_dir)) ]
    
    # pick cam0 or cam1 imgs
    img_fn_l = [l for l in img_fn_l if l.split("_")[2] == 'c%d'%args.cam_id]
    
    # sort images by timestamps
    time_l = np.array([l.split("_")[3].split("us")[0] for l in 
        img_fn_l]).astype(np.int64)
    time_order = np.argsort(time_l)
    img_fn_l = np.array(img_fn_l)[time_order]
    time_l = time_l[time_order]

    return time_l, img_fn_l


def get_survey_auto_time(args):#mode='database'):
    """Interactive code to split queries into surveys.
    Display sequence of img for cam0 and cam1.
    """
    time_l, img_fn_l = get_img_order(args)

    if args.survey_id == -1: # it is already split so there is nothing to do
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_db"%(args.slice_id,
                args.slice_id, args.cam_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savetxt("%s/fn.txt"%out_dir, img_fn_l, fmt='%s')
    else: # manually specify splits between surveys
        time_post_l = np.roll(time_l, -1, 0)
        #print(time_l[:10])
        #print(time_post_l[:10])
        time_diff = (time_post_l - time_l)[:-1]/1e9
        splits = np.where(time_diff>500)[0]
        #print(splits)
        split_prev = 0
        for survey_id, split in enumerate(splits):
            out_dir = "pycmu/meta/surveys/%d/%d_c%d_%d"%(args.slice_id,
                    args.slice_id, args.cam_id, survey_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            #print(split_prev, split)
            sub_fn_l = img_fn_l[split_prev:split+1]
            split_prev = split+1
            np.savetxt("%s/fn.txt"%out_dir, sub_fn_l, fmt='%s')

        if (0==1): # debug
            plt.figure()
            plt.scatter(np.arange(time_diff.shape[0]), time_diff)
            plt.savefig("toto.png")
            plt.close()
            toto = cv2.imread("toto.png")
            cv2.imshow("toto", toto)
            cv2.waitKey(0)

def test_get_survey_auto_time(args):
    """ """
    for slice_id in range(6,7):
        #if slice_id == 21:
        #    continue
        for survey_id in range(0, 4):
            print("\nslice_id: %d"%slice_id)
            #print("survey_id: %d"%survey_id)
            fn = "pycmu/meta/surveys/%d/%d_c%d_%d/fn.txt"%(slice_id,
                        slice_id, args.cam_id, survey_id)
            root_fn_v = np.loadtxt(fn, dtype=str)
            img_num = root_fn_v.shape[0]
            idx_v = list(range(0, img_num, 1)) + [img_num-1]
            for i in idx_v:
                print("%d/%d"%(i, img_num))
                img_fn = "%s/%s"%(args.img_dir, root_fn_v[i])
                img = cv2.imread(img_fn)
                cv2.imshow("img", img)
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    exit(0)

            time.sleep(2)



def show_ordered_img(args):
    """Display sequence of img for cam0 and cam1."""
    args.cam_id = 0
    c0_sorted = get_img_order(args)
    args.cam_id = 1
    c1_sorted = get_img_order(args)
    for c0_root_fn, c1_root_fn in zip(c0_sorted, c1_sorted):
        c0_fn = '%s/%s'%(args.img_dir, c0_root_fn) 
        c1_fn = '%s/%s'%(args.img_dir, c1_root_fn)
        print("c0_fn: %s\nc1_fn: %s\n"%(c0_fn, c1_fn))
        img0, img1 = cv2.imread(c0_fn), cv2.imread(c1_fn)
        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        stop_show = cv2.waitKey(0) & 0xFF
        if stop_show == ord("q"):
            exit(0)


def get_survey_mano(args):#mode='database'):
    """Interactive code to split queries into surveys.
    Display sequence of img for cam0 and cam1.
    """
    # prepare output dir and get order img list
    if args.survey_id == -1:
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_db"%(args.slice_id,
                args.slice_id, args.cam_id)
        img_fn_l = get_img_order(args)
    else:
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_%d"%(args.slice_id,
                args.slice_id, args.cam_id, args.survey_id)
        img_fn_l = get_img_order(args)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.survey_id == 0: # it is already split so there is nothing to do
        np.savetxt("%s/fn.txt"%out_dir, img_fn_l, fmt='%s')
    #else: # manually specify splits between surveys
    #    survey_count = 0
    #    img_count, img_count_prev = 0,0
    #    while True:
    #        if img_count == (img_l[cam_id].shape[0]):
    #            # save last survey
    #            print("Survey %d: %d -> %d"%(survey_count, img_count_prev, img_count))
    #            np.savetxt('meta/surveys/%s/fn/c%d_%d.txt'%(slice_id, cam_id, survey_count),
    #                    img_l[cam_id][img_count_prev:img_count+1], fmt='%s')
    #            #np.savetxt('meta/surveys/%s/fn/c1_%d.txt'%(slice_id, survey_count),
    #            #        img_l[1][img_count_prev:img_count+1], fmt='%s')
    #            img_count_prev = img_count
    #            survey_count += 1
    #            break
    #        print('Cam %d\t%d\t%s'%(cam_id, img_count, img_l[cam_id][img_count]))

    #        print('img_fn: %s/%s'%(cst.EXT_IMG_DIR, img_l[cam_id][img_count]))
    #        img0 = cv2.imread('%s/%s'%(cst.EXT_IMG_DIR, img_l[cam_id][img_count]))
    #        cv2.imshow('img0', img0)
    #        k = cv2.waitKey(0) & 0xFF

    #        if k==ord("q"): # stop
    #            exit(0)
    #        elif k==ord("n"): # next img
    #            img_count +=1 
    #            continue
    #        elif k==ord("p"): # previous img
    #            if img_count >0:
    #                img_count -= 1
    #            else:
    #                print("Already first img. Go to next one")
    #            continue
    #        elif k==ord("k"): # you are at the end of a survey, split
    #            print("Survey %d: %d -> %d"%(survey_count, img_count_prev, img_count))
    #            
    #            np.savetxt('meta/surveys/%s/fn/c%d_%d.txt'%(slice_id, cam_id, survey_count),
    #                    img_l[cam_id][img_count_prev:img_count+1], fmt='%s')

    #            #np.savetxt('meta/surveys/%s/fn/c1_%d.txt'%(slice_id, survey_count),
    #            #        img_l[1][img_count_prev:img_count+1], fmt='%s')

    #            img_count_prev = img_count + 1
    #            survey_count += 1


def get_survey_auto(args):
    """Auto code to split queries into surveys based on camera poses."""
    if args.survey_id == -1:
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_db"%(args.slice_id,
                args.slice_id, args.cam_id)
        img_fn_l = get_img_order(args)
    else:
        out_dir = "pycmu/meta/surveys/%d/%d_c%d_%d"%(args.slice_id,
                args.slice_id, args.cam_id, args.survey_id)
        img_fn_l = get_img_order(args)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load ground-truth poses
    if args.survey_id == -1:
        meta_v = np.loadtxt('%s/slice%d/ground-truth-database-images-slice%d.txt'
                %(args.img_dir, args.slice_id, args.slice_id), dtype=str)
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


#def get_survey_auto(img_dir, slice_id, mode='database'):
#    """Auto code to split queries into surveys based on camera poses."""
#    if args.survey_id == -1:
#        out_dir = "pycmu/meta/surveys/%d/%d_c%d_db"%(args.slice_id,
#                args.slice_id, args.cam_id)
#        img_fn_l = get_img_order(args)
#    else:
#        out_dir = "pycmu/meta/surveys/%d/%d_c%d_%d"%(args.slice_id,
#                args.slice_id, args.cam_id, args.survey_id)
#        img_fn_l = get_img_order(args)
#    if not os.path.exists(out_dir):
#        os.makedirs(out_dir)
#
#
#    # load ground-truth poses
#    if args.survey_id == -1:
#        meta_v = np.loadtxt('%s/slice%d/ground-truth-database-images-slice%d.txt'
#                %(args.img_dir, args.slice_id, args.slice_id), dtype=str)
#        fn_v = meta_v[:,0]
#        pose_v = meta_v[:,1:]
#        print('pose_v.shape: ', pose_v.shape)
#    elif mode == 'query':
#        # gather gt metas
#        meta_fn_l = glob.glob('%s/slice%d/camera-poses/*txt'%(img_dir, slice_id))
#        meta_l = []
#        for meta_fn in meta_fn_l:
#            #print(meta_fn)
#            if os.stat(meta_fn).st_size == 0:
#                continue
#            meta_l.append(np.loadtxt(meta_fn, dtype=str))
#        if len(meta_l) == 0:
#            print("Error: no ground-truth for this slice so I can't use it.")
#            exit(1)
#
#        meta_v = np.vstack(meta_l)
#        #print(meta_v.shape)
#        fn_v = meta_v[:,0]
#        #pose_v = meta_v[:,5:].astype(np.float32)
#        pose_v = meta_v[:,1:]
#        print('pose_v.shape: ', pose_v.shape)
#
#    # split c0 and c1
#    fn0_v = np.array([l for l in fn_v if l.split("_")[2] == 'c0'])
#    fn1_v = np.array([l for l in fn_v if l.split("_")[2] == 'c1'])
#    idx0_v = np.in1d(fn_v, fn0_v).nonzero()[0]
#    pose0_v = pose_v[idx0_v]
#    idx1_v = np.in1d(fn_v, fn1_v).nonzero()[0]
#    pose1_v = pose_v[idx1_v]
#
#    # sort by timestamps
#    time0_v = np.array([l.split("_")[3].split("us")[0] for l in fn0_v])
#    time0_order = np.argsort(time0_v)
#    fn0_v_sorted = fn0_v[time0_order]
#    pose0_v_sorted = pose0_v[time0_order]
#
#    time1_v = np.array([l.split("_")[3].split("us")[0] for l in fn1_v])
#    time1_order = np.argsort(time1_v)
#    fn1_v_sorted = fn1_v[time1_order]
#    pose1_v_sorted = pose1_v[time1_order]
#
#
#    if mode == 'database': # nothing to do but save as is
#        fn0_v_sorted = np.array(['slice%d/database/%s'%(slice_id, l) for l in
#            fn0_v_sorted])
#        fn0_v_sorted = np.expand_dims(fn0_v_sorted, 1)
#        np.savetxt('%s/c0_db.txt'%out_dir, np.hstack((fn0_v_sorted,
#            pose0_v_sorted)), fmt='%s')
#
#        fn1_v_sorted = np.array(['slice%d/database/%s'%(slice_id, l) for l in
#            fn1_v_sorted])
#        fn1_v_sorted = np.expand_dims(fn1_v_sorted, 1)
#        np.savetxt('%s/c1_db.txt'%out_dir, np.hstack((fn1_v_sorted,
#            pose1_v_sorted)), fmt='%s')
#    elif mode == 'query': # meta holds several surveys, so split them
#        # cam0
#        fn0_v_sorted = np.array(['slice%d/query/%s'%(slice_id, l) for l in
#            fn0_v_sorted])
#        idx = 0
#        img_num = fn0_v.shape[0]
#        xy0 = pose0_v_sorted[0,5:7].astype(np.float32)
#
#        survey_id = 0
#        survey_l = [np.hstack((fn0_v_sorted[0], pose0_v_sorted[0,:]))]
#        eps = 100
#        for idx in range(1,img_num):
#            xy1 = pose0_v_sorted[idx,5:7].astype(np.float32)
#            d = np.sum( (xy0 - xy1)**2)
#            #print('idx: %d\td: %.3f'%(idx, d))
#            if np.sum( (xy0 - xy1)**2) > eps: # end of survey
#                np.savetxt('%s/c0_%d.txt'%(out_dir, survey_id), np.vstack(survey_l), fmt='%s')
#                survey_id += 1
#                survey_l = []
#            
#            survey_l.append(np.hstack((fn0_v_sorted[idx], pose0_v_sorted[idx,:])))
#            xy0 = xy1
#
#        # cam1 (TODO: eliminate code duplication)
#        fn1_v_sorted = np.array(['slice%d/query/%s'%(slice_id, l) for l in
#            fn1_v_sorted])
#        idx = 0
#        img_num = fn1_v.shape[0]
#        xy0 = pose1_v_sorted[0,5:7].astype(np.float32)
#
#        survey_id = 0
#        survey_l = [np.hstack((fn1_v_sorted[0], pose1_v_sorted[0,:]))]
#        for idx in range(1,img_num):
#            xy1 = pose1_v_sorted[idx,5:7].astype(np.float32)
#            d = np.sum( (xy0 - xy1)**2)
#            #print('idx: %d\td: %.3f'%(idx, d))
#            if np.sum( (xy0 - xy1)**2) > eps: # end of survey
#                np.savetxt('%s/c1_%d.txt'%(out_dir, survey_id), np.vstack(survey_l), fmt='%s')
#                survey_id += 1
#                survey_l = []
#            
#            survey_l.append(np.hstack((fn1_v_sorted[idx], pose1_v_sorted[idx,:])))
#            xy0 = xy1
#    
#    else:
#        print("Error: I don't know this mode: %s"%mode)
#        exit(1)


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

    if (0==1):
        show_ordered_img(args.slice_id, mode)

    #if (0==1): # split slice into surveys (auto)
    #    get_survey_auto(args.img_dir, args.slice_id, 'database')
    #    #get_survey_auto(args.img_dir, args.slice_id, 'query')
 
    if (1==1): # split slice into surveys (auto)
        get_survey_auto_time(args)
        #test_get_survey_auto_time(args)

    if (0==1): # show specific survey
        show_survey(args.img_dir, args.slice_id, args.cam_id, args.survey_id)
