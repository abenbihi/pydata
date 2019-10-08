#!/bin/sh

#COLMAP_WS=/home/abenbihi/ws/datasets/colmap/cmu/
COLMAP_WS=/home/gpu_user/assia/ws/datasets/colmap/cmu/

slice_id=24
cam_id=0
survey_id=0

python3 -m pycmu.colmap \
    --slice_id "$slice_id" \
    --cam_id "$cam_id" \
    --survey_id "$survey_id" \
    --colmap_ws "$COLMAP_WS" 
