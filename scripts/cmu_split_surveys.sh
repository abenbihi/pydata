#!/bin/sh

. ./scripts/export_path.sh

# get database file list
survey_id=-1
if [ 0 -eq 1 ]; then
  #for slice_id in 6 7 8
  for slice_id in 18 19 20 21 22 23 24 25
  do
    for cam_id in 0 1 
    do
      echo "slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
      python3 -m pycmu.split_survey \
        --img_dir "$IMG_DIR" \
        --slice_id "$slice_id" \
        --survey_id "$survey_id" \
        --cam_id "$cam_id"

      if [ "$?" -ne 0 ]; then
        echo "Error in run slice_id: "$slice_id"\tcam_id: "$cam_id""
        exit 1
      fi

    done
  done
fi

# query file list
if [ 1 -eq 1 ]; then
  #for slice_id in 18 19 20 21 22 23 24 25
  #for slice_id in 6 #18 19 20 21 22 23 24 25
  for slice_id in 24
  do
    for cam_id in 0 1 
    do
      echo "slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
      python3 -m pycmu.split_survey \
        --img_dir "$IMG_DIR" \
        --slice_id "$slice_id" \
        --survey_id 0 \
        --cam_id "$cam_id"

      if [ "$?" -ne 0 ]; then
        echo "Error in run slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
        exit 1
      fi

    done
  done
fi


# test survey splits
if [ 0 -eq 1 ]; then
  for slice_id in 6 # 19 20 21 22 23 24 25
  do
    for cam_id in 0  
    do
      for survey_id in 0 #1 2 3 4 5 6 7 8 9 
      do
        echo "slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
        python3 -m pycmu.split_survey \
          --img_dir "$IMG_DIR" \
          --slice_id "$slice_id" \
          --survey_id "$survey_id" \
          --cam_id "$cam_id"

        if [ "$?" -ne 0 ]; then
          echo "Error in run slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
          exit 1
        fi
      done
    done
  done
fi
