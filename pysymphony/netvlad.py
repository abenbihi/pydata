"""Functions to generate netvlad data for training."""
import argparse
import os
import glob

import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

import cst
import retrieval.tools as tools

def gen_dataset(args, splits_d, whichSet):
    sample_pose = True
    
    # TODO: refine these values
    # posDistThr: distance in meters which defines potential positives
    posDistThr = 1 # meters
    posDistSqThr = posDistThr ** 2
    # nonTrivPosDistSqThr: squared distance in meters which defines the potential positives used for training
    nonTrivPosDistSqThr = 2 # meters

    out_dir = '%s/datasets/netvlad/%d/%s'%(cst.SCRIPT_DIR, args.data_id, whichSet)
    if os.path.exists(out_dir):
        answer = input("This dataset already exists. Do you want to delete it ?: %s\n"%(out_dir))
        if answer=='n':
            exit(0)
    else:
        os.makedirs(out_dir)
    
    dbImage, qImage = [], []
    utmDb, utmQ = [], []

    for survey_id in splits_d['%s_db'%whichSet]:
        img_l, pose_l = tools.sample_survey(survey_id, args.id_iter, whichSet, sample_pose)
        dbImage += img_l
        utmDb += pose_l

    for survey_id in splits_d['%s_q'%whichSet]:
        img_l, pose_l = tools.sample_survey(survey_id, args.id_iter, whichSet, sample_pose)
        qImage += img_l
        utmQ += pose_l
    

    np.savetxt('%s/dbImage.txt'%out_dir, np.vstack(dbImage), fmt='%s')
    np.savetxt('%s/qImage.txt'%out_dir, np.vstack(qImage), fmt='%s')
    np.savetxt('%s/utmDb.txt'%out_dir, np.vstack(utmDb))
    np.savetxt('%s/utmQ.txt'%out_dir, np.vstack(utmQ))
    np.savetxt('%s/meta.txt'%out_dir, np.array([posDistThr, posDistSqThr, nonTrivPosDistSqThr]))
    np.savetxt('%s/mode.txt'%out_dir, np.array([whichSet]), fmt='%s')



class Metadata():
    def __init__(self, split_dir):

        self.split_dir = split_dir

        self.dbImage = np.loadtxt('%s/dbImage.txt'%split_dir, dtype=str)
        self.utmDb = np.loadtxt('%s/utmDb.txt'%split_dir)

        self.qImage = np.loadtxt('%s/qImage.txt'%split_dir, dtype=str)
        self.utmQ = np.loadtxt('%s/utmQ.txt'%split_dir)
        
        self.meta = np.loadtxt('%s/meta.txt'%split_dir)

        self.posDistThr             = self.meta[0]
        self.posDistSqThr           = self.meta[1]
        self.nonTrivPosDistSqThr    = self.meta[2]
        self.dist_pos = self.meta[2]

    def filter(self, idx_to_keep):
        self.qImage = self.qImage[idx_to_keep]
        self.utmQ = self.utmQ[idx_to_keep, :]

    def save(self):
        np.savetxt('%s/qImage.txt'%self.split_dir, self.qImage, fmt='%s')
        np.savetxt('%s/utmQ.txt'%self.split_dir, self.utmQ)

   

def filter_queries(args):
    """
    Queries without matching img in db are useless for both training and
    validation.
    """
    for split_name in ['train', 'val']:
        split_dir = '%s/datasets/netvlad/%d/%s'%(cst.SCRIPT_DIR, args.data_id, split_name)
        metadata = Metadata(split_dir)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(metadata.utmDb)

        # list of array of db idx matching a query
        # nontrivial_positives[i] = list of db img idx matching the i-th query
        nontrivial_positives = list(knn.radius_neighbors(metadata.utmQ,
                radius=metadata.dist_pos, return_distance=False))
        #print(nontrivial_positives) # [i]=array([ 0,  1,  2,  3,  4,  5])
        
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        queries_idx = np.where(np.array([len(x) for x in nontrivial_positives])>0)[0]
        #metadata.utmQ = metadata.utmQ[queries_idx,:]
        #metadata.qImage = metadata.qImage[queries_idx]
        #num_queries = queries_idx.shape[0]
        
        metadata.filter(queries_idx)
        metadata.save()
        
        # debug
        if (1==1):
            toto  = np.array([i for i,l in enumerate(nontrivial_positives) if len(l)>0])
            toto_sum = np.sum( (toto - queries_idx))
            if toto_sum!=0:
                print(toto_sum)
                print("Error somewhere in dataset")
                exit(1)
        nontrivial_positives = [l for l in nontrivial_positives if len(l)>0]


def prepare_timelapse(args):
    """Sample images of survey as a basis for the timelapse."""
    
    out_dir = '%s/datasets/retrieval/%d/'%(cst.SCRIPT_DIR, args.data_id)
    if os.path.exists(out_dir):
        answer = input("This dataset already exists. Do you want to delete it ?: %s\n"%(out_dir))
        if answer=='n':
            exit(0)
    else:
        os.makedirs(out_dir)
    
    sample_pose = True
    img_fn_l, pose_l = tools.sample_survey(args.survey_id, args.id_iter,
            'val', sample_pose)
    
    np.savetxt('%s/db_img.txt'%out_dir, np.vstack(img_fn_l), fmt='%s')
    np.savetxt('%s/db_pose.txt'%out_dir, np.vstack(pose_l))


if __name__=='__main__':

    if (1==1):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_id', type=int, required=True, default=0)
        parser.add_argument('--id_iter', type=int, default=100)
        parser.add_argument('--survey_id', type=int, default=160808)
        args = parser.parse_args()

        prepare_timelapse(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, required=True, default=0)
    parser.add_argument('--id_iter', type=int, default=100)
    args = parser.parse_args()
   
    splits_d = {}
    splits_d['train_db'] = [150408, 150730, 150929, 151214]
    splits_d['train_q'] = [170217, 160411, 170725, 161127]
    gen_dataset(args, splits_d, 'train')
    
 
    splits_d['val_db'] = [150429]
    splits_d['val_q'] = [150216, 150723, 151027, 150421]
    gen_dataset(args, splits_d, 'val')

    # filter out queries that do not have matching db images
    filter_queries(args)

    start_d = {}
    start_d[survey0_id] = 2265
    #show_matches(survey0_id, survey1_id, island_d, start_d)

    netvlad_data_id = 4
    #netvlad_val(netvlad_data_id, survey0_id, survey1_id, island_d, start_d)

    netvlad_trial = 7
    netvlad_show_output(netvlad_trial, netvlad_data_id)


