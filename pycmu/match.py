import os, argparse

import cv2
import numpy as np

import tools.angles
import tools.read_model


def gen_match_manual(args):
    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_db/"%( args.slice_id,
            args.slice_id, args.cam_id)
    db_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
    db_num = db_fn_v.shape[0]

    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_%d/"%( args.slice_id,
            args.slice_id, args.cam_id, args.survey_id)
    q_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
    
    db_idx_prev = 0
    match_inter_l = []
    for q_idx, q_fn in enumerate(q_fn_v):
        q_img = cv2.imread("%s/%s"%(args.img_dir, q_fn))
        q_img = cv2.resize(q_img, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_AREA)
        next_img = False
        
        db_idx = db_idx_prev
        while True:
            if db_idx == db_num:
                print("You're at the end of the db survey.")
            #    break
            #print(db_idx_prev)
            if next_img == True:
                break
            db_fn = db_fn_v[db_idx]
            #match_inter_l.append("%s %s"%(q_fn, db_fn))

            db_img = cv2.imread("%s/%s"%(args.img_dir, db_fn))
            db_img = cv2.resize(db_img, None, fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_AREA)
            cv2.imshow("img", np.hstack((q_img, db_img)))
            k = (cv2.waitKey(0) & 0xFF)

            if k==ord("q"): # stop
                exit(0)
            elif k==ord("n"): # next img
                db_idx += 1
                continue
            elif k==ord("p"): # previous img
                if db_idx >0:
                    db_idx -= 1
                else:
                    print("Already first img. Go to next one")
                continue
            elif k==ord("k"): # previous img
                match_inter_l.append("%d %d"%(q_idx, db_idx))
                next_img = True
                db_idx_prev = db_idx + 1
                continue

    out_fn = "%s/image_pairs_manual.txt"%survey_dir 
    np.savetxt(out_fn, np.array(match_inter_l), fmt="%s")


def gen_match_inter(args):
    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_db/"%( args.slice_id,
            args.slice_id, args.cam_id)
    db_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
    db_num = db_fn_v.shape[0]

    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_%d/"%( args.slice_id,
            args.slice_id, args.cam_id, args.survey_id)
    q_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)

    pairs_fn = "%s/image_pairs_manual.txt"%survey_dir 
    pairs = np.loadtxt(pairs_fn, dtype=np.int)

    match_inter_l = []
    for pair in pairs:
        q_idx, db_idx = pair
        q_fn = q_fn_v[q_idx]
        for idx in range(max(0, db_idx-5), min(db_num, db_idx+5)):
            db_fn = db_fn_v[idx]
            match_inter_l.append("%s %s"%(q_fn, db_fn))
    
    out_fn = "%s/image_pairs_to_match_inter.txt"%survey_dir 
    np.savetxt(out_fn, np.array(match_inter_l), fmt="%s")


def test_match_inter(args):
    """ """
    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_db/"%( args.slice_id,
            args.slice_id, args.cam_id)
    db_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
    db_num = db_fn_v.shape[0]

    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_%d/"%( args.slice_id,
            args.slice_id, args.cam_id, args.survey_id)
    q_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)

    out_fn = "%s/image_pairs_to_match_inter.txt"%survey_dir 
    match_inter_v = np.loadtxt(out_fn, dtype=str)
    
    for i, match in enumerate(match_inter_v):
        #if i % 10 !=0:
        #    continue
        q_fn, db_fn = match
        q_img = cv2.imread("%s/%s"%(args.img_dir, q_fn))
        q_img = cv2.resize(q_img, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_AREA)
        db_img = cv2.imread("%s/%s"%(args.img_dir, db_fn))
        db_img = cv2.resize(db_img, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_AREA)
        cv2.imshow("img", np.hstack((q_img, db_img)))
        if (cv2.waitKey(0) & 0xFF) == ord("q"):
            exit(0)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice_id", type=int, required=True)
    parser.add_argument("--cam_id", type=int, required=True)
    parser.add_argument("--survey_id", type=int, required=True)
    parser.add_argument("--colmap_ws", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()
    
    #gen_match_manual(args)
    #gen_match_inter(args)
    test_match_inter(args)


#def gen_match_inter(args):
#    """ """
#    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_db/"%( args.slice_id,
#            args.slice_id, args.cam_id)
#    db_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
#    db_num = db_fn_v.shape[0]
#
#    survey_dir = "pycmu/meta/surveys/%d/%d_c%d_%d/"%( args.slice_id,
#            args.slice_id, args.cam_id, args.survey_id)
#    q_fn_v = np.loadtxt("%s/fn.txt"%survey_dir, dtype=str)
#
#    match_inter_l = []
#    for q_idx, q_fn in enumerate(q_fn_v):
#        q_img = cv2.imread("%s/%s"%(args.img_dir, q_fn))
#        q_img = cv2.resize(q_img, None, fx=0.5, fy=0.5,
#                interpolation=cv2.INTER_AREA)
#        next_img = False
#
#        for db_idx in range(max(q_idx-10, 0), min(q_idx+10, db_num)):
#            if next_img:
#                break
#            db_fn = db_fn_v[db_idx]
#            match_inter_l.append("%s %s"%(q_fn, db_fn))
#
#            db_img = cv2.imread("%s/%s"%(args.img_dir, db_fn))
#            db_img = cv2.resize(db_img, None, fx=0.5, fy=0.5,
#                    interpolation=cv2.INTER_AREA)
#            cv2.imshow("img", np.hstack((q_img, db_img)))
#            k = (cv2.waitKey(0) & 0xFF)
#            while True:
#                if k == ord("q"):
#                    exit(0)
#                    break
#                elif k == ord("b"):
#                    next_img = True
#                    break
#                elif k == ord("k"):
#                    match_inter_l.append("%s %s"%(q_fn, db_fn))
#                    break
#                else:
#                    print("Wrong key. Press {q, b, k}")
#                    k = (cv2.waitKey(0) & 0xFF)
#                    break
#
#    out_fn = "%s/image_pairs_to_match_inter.txt"%survey_dir 
#    np.savetxt(out_fn, np.array(match_inter_l), fmt="%s")


