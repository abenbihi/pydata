#!/bin/sh

. ./scripts/export_path.sh

#for slice_id in 22 23 24 25
if [ 1 -eq 1 ]; then
  for slice_id in 18 19 20 21 22 23 24 25
  do
    for cam_id in 0 1
    do
      for survey_id in -1 #0 1 2 3 4 5 6 7 8 9
      do
        if [ "$slice_id" -eq 24 ] || [ "$slice_id" -eq 25 ]; then
          if [ "$cam_id" -eq 1 ] && [ "$survey_id" -eq 8 ]; then
            echo "This traversal has no ground-truth pose."
            continue
          fi
        fi

        echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"
        python3 -m pycmu.colmap_prior \
          --slice_id "$slice_id" \
          --cam_id "$cam_id" \
          --survey_id "$survey_id" \
          --img_dir "$IMG_DIR" \
          --colmap_ws "$COLMAP_WS"

        if [ "$?" -ne 0 ]; then
          echo "Error in run slice_id: "$slice_id"\tcam_id: "$cam_id""
          exit 1
        fi
      done
    done
  done
fi

#for slice_id in 18 19 20 21
#do
#  for cam_id in 0 1 
#  do
#    survey_id=-1
#    echo "\n\n** Slice "$slice_id" Cam "$cam_id" Survey "$survey_id" **"
#    python3 -m pycmu.colmap \
#      --slice_id "$slice_id" \
#      --cam_id "$cam_id" \
#      --survey_id "$survey_id" \
#      --colmap_ws "$COLMAP_WS"
#  done
#done
#
