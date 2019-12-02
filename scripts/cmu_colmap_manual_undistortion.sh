#!/bin/sh

## TODO: specify your path here
#MACHINE=1
#if [ "$MACHINE" -eq 0 ]; then
#  WS_PATH=/home/abenbihi/ws/
#  IMG_DIR=/mnt/data_drive/dataset/CMU-Seasons/images/
#elif [ "$MACHINE" -eq 1 ]; then
#  WS_PATH=/home/gpu_user/assia/ws/
#  #IMG_DIR="$WS_PATH"/datasets/Extended-CMU-Seasons/
#  IMG_DIR="$WS_PATH"datasets/Extended-CMU-Seasons-Undistorted/
#else
#  echo "test_lake.sh: Get your MTF MACHINE correct"
#  exit 1
#fi
#COLMAP_BIN="$WS_PATH"tools/colmap/build/src/exe/colmap
##IMG_DIR=/mnt/data_drive/dataset/Extended-CMU-Seasons/
#

. ./scripts/export_path.sh

IMG_DIR="$IMG_UNDISTORTED_DIR"

if [ "$#" -eq 0 ]; then 
  echo "Arguments: "
  echo "1: slice"
  echo "2: camera id"
  echo "3: survey id"
  exit 1
fi


if [ "$#" -ne 3 ]; then 
  echo "Error: bad number of arguments"
  echo "1: slice"
  echo "2: camera id"
  echo "3: survey id"
  exit 1
fi

slice_id="$1"
cam_id="$2"
survey_id="$3"

if [ "$survey_id" -eq -1 ]; then
  survey_id=db
fi

project_name="$slice_id"/"$slice_id"_c"$cam_id"_"$survey_id"
colmap_ws=pycmu/res/colmap/"$project_name"/

# used for CMU seasons ONLY, because the images from Extended-CMU-Seasons are
# not undistorted !!!!
if [ "$cam_id" -eq 0 ]; then 
  camera_model=PINHOLE
  camera_params=868.99,866.06,525.94,420.04

  #camera_model=OPENCV
  #camera_params=868.993378,866.063001,525.942323,420.042529,-0.399431,0.188924,0.000153,0.000571
elif [ "$cam_id" -eq 1 ]; then 
  camera_model=PINHOLE
  camera_params=873.38,876.49,529.32,397.27
else
  echo "Error: Wrong cam_id="$cam_id" != {0,1}."
  exit 1
fi

if [ 1 -eq 1 ]; then 
  rm -f  "$colmap_ws"/database.db
  rm -rf "$colmap_ws"/sparse
  rm -rf "$colmap_ws"/dense
  if [ $? -ne 0 ]; then
    echo "Error when deleting previous colmap workspace"
    exit 1
  fi

  mkdir -p "$colmap_ws"/sparse/
  mkdir -p "$colmap_ws"/sparse/
  mkdir -p "$colmap_ws"/dense

  if [ $? -ne 0 ]; then
    echo "Error when creating colmap workspace"
    exit 1
  fi
fi

if [ 1 -eq 1 ]; then
  # import known poses
  if [ -d "$colmap_ws"/colmap_prior ]; then
    rm -rf "$colmap_ws"/colmap_prior/
  fi

  cp -r pycmu/meta/surveys/"$project_name"/colmap_prior/ \
    "$colmap_ws"/colmap_prior
  if [ $? -ne 0 ]; then
    echo "Error when copying prior information."
    exit 1
  fi
fi

#echo "$IMG_DIR"
#echo "$colmap_ws"
#exit 0

if [ 1 -eq 1 ]; then
  ## feature extraction with known camera params on masked imgs, from directory
  "$COLMAP_BIN" feature_extractor \
    --database_path "$colmap_ws"/database.db \
    --image_path "$IMG_DIR" \
    --image_list_path "$colmap_ws"/colmap_prior/image_list.txt \
    --ImageReader.camera_model "$camera_model" \
    --ImageReader.camera_params "$camera_params" 
  if [ $? -ne 0 ]; then
    echo "Error during feature_extractor."
    exit 1
  fi
fi


if [ 1 -eq 1 ]; then
  ### feature matching with custom matches. Here I specify image pairs to match.
  ## I think you can also specify features inliers directly.
  #"$COLMAP_BIN" matches_importer \
  #  --database_path $colmap_ws/database.db \
  #  --match_list_path $colmap_ws/mano/match_list.txt \
  #  --match_type pairs

  "$COLMAP_BIN" exhaustive_matcher \
    --database_path "$colmap_ws"/database.db

  if [ $? -ne 0 ]; then
    echo "Error during matcher."
    exit 1
  fi
fi

if [ 1 -eq 1 ]; then
  echo "$IMG_DIR"
  # for when you know the camera pose beforehand
  "$COLMAP_BIN" point_triangulator \
    --database_path $colmap_ws/database.db \
    --image_path "$IMG_DIR" \
    --input_path "$colmap_ws"/colmap_prior/ \
    --output_path "$colmap_ws"/sparse/
  if [ $? -ne 0 ]; then
    echo "Error during point_triangulator."
    exit 1
  fi
fi


#if [ 0 -eq 1 ]; then
#  # for when you have no prior on the camera pose, img list
#  #"$COLMAP_BIN" mapper \
#  #  --database_path $colmap_ws/database.db \
#  #  --image_path "$IMG_DIR" \
#  #  --image_list_path "$colmap_ws"/colmap_prior/image_list.txt \
#  #  --output_path $colmap_ws/sparse/
#
#  # for when you have no prior on the camera pose, from dir
#  "$COLMAP_BIN" mapper \
#    --database_path $colmap_ws/database.db \
#    --image_path "$colmap_ws"img \
#    --output_path $colmap_ws/sparse/
#
#  if [ $? -ne 0 ]; then
#    echo "Error during mapper."
#    exit 1
#  fi
#fi

#if [ 0 -eq 1 ]; then
#    
#    mkdir -p "$colmap_ws"/dense/images
#    while read -r line
#    do
#        cp "$IMG_DIR""$line" "$colmap_ws"/dense/images/
#    done < "$colmap_ws"/colmap_prior/image_list.txt
#    cp -r "$colmap_ws"/sparse "$colmap_ws"/dense/
#fi


if [ 1 -eq 1 ]; then
  "$COLMAP_BIN" image_undistorter \
    --image_path "$IMG_DIR" \
    --input_path "$colmap_ws"/sparse/ \
    --output_path "$colmap_ws"/dense/

  if [ $? -ne 0 ]; then
    echo "Error during image_undistorter."
    exit 1
  fi
fi

if [ 1 -eq 1 ]; then
  "$COLMAP_BIN" patch_match_stereo \
    --workspace_path "$colmap_ws"/dense/

  if [ $? -ne 0 ]; then
    echo "Error during patch_match_stereo."
    exit 1
  fi
fi

#if [ 1 -eq 1 ]; then
#  "$COLMAP_BIN" stereo_fusion \
#    --workspace_path "$colmap_ws"/dense/ \
#    --output_path "$colmap_ws"/dense/fused.ply
#  if [ $? -ne 0 ]; then
#    echo "Error during stereo_fusion."
#    exit 1
#  fi
#fi

