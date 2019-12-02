#!/bin/sh
# TODO: specify your machine here
MACHINE=1

if [ "$MACHINE" -eq 0 ]; then
  WS_DIR=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  WS_DIR=/home/gpu_user/assia/ws/
else
  echo "test_lake.sh: Get your MTF MACHINE correct"
  exit 1
fi

IMG_DIR="$WS_DIR"datasets/Extended-CMU-Seasons/
IMG_UNDISTORTED_DIR="$WS_DIR"datasets/Extended-CMU-Seasons-Undistorted/
COLMAP_BIN="$WS_DIR"tools/colmap/build/src/exe/colmap

