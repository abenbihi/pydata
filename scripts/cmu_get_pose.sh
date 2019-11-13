#!/bin/sh

. ./scripts/export_path.sh

if [ 1 -eq 1 ]; then
  for slice_id in 18 19 20 #22 23 24 25
  #for slice_id in 24
  do
    if [ "$slice_id" -eq 21 ]; then
      echo "Abort: slice 21 need to be done manually because it is f***** up"
      exit 1
    fi
    for cam_id in 0 1 
    do
      echo "slice_id: "$slice_id"\tcam_id: "$cam_id""
      for survey_id in -1 0 1 2 3 4 5 6 7 8 9
      do
        echo "slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
        python3 -m pycmu.pose \
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
  done
fi

if [ 0 -eq 1 ]; then
  for slice_id in 21
  do
    for cam_id in 0 1 
    do
      echo "slice_id: "$slice_id"\tcam_id: "$cam_id""
      for survey_id in -1 1 3 4 7
      do
        echo "slice_id: "$slice_id"\tcam_id: "$cam_id"\tsurvey_id: "$survey_id""
        python3 -m pycmu.pose \
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
  done

fi
