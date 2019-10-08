#!/bin/sh

# 0
IMG_DIR=/home/abenbihi/ws/datasets/Extended-CMU-Seasons/
COLMAP_WS=/home/abenbihi/ws/datasets/colmap/cmu/
# 1
#COLMAP_WS=/home/gpu_user/assia/ws/datasets/colmap/cmu/

slice_id=24
cam_id=0
survey_id=0

for slice_id in 23
do
  python3 -m pycmu.split_survey \
    --img_dir "$IMG_DIR" \
    --slice_id "$slice_id" \
    --survey_id 0 \
    --cam_id 0
done
