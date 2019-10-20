#!/bin/sh

. ./scripts/export_path.sh

COLMAP_BIN="$WS_DIR"tools/colmap/build/src/exe/colmap

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

project_name="$slice_id"_c"$cam_id"_"$survey_id"
colmap_ws=pycmu/res/colmap/"$project_name"/

# used for CMU seasons ONLY, because the images from Extended-CMU-Seasons are
# not undistorted !!!!
camera_model=OPENCV
if [ "$cam_id" -eq 0 ]; then 
  #camera_model=PINHOLE
  #camera_params=868.99,866.06,525.94,420.04
  camera_params=868.993378,866.063001,525.942323,420.042529,-0.399431,0.188924,0.000153,0.000571
elif [ "$cam_id" -eq 1 ]; then 
  #camera_model=PINHOLE
  #camera_params=873.38,876.49,529.32,397.27
  camera_params=873.382641,876.489513,529.324138,397.272397,-0.397066,0.181925,0.000176,-0.000579
else
  echo "Error: Wrong cam_id="$cam_id" != {0,1}."
  exit 1
fi

# prepare workspace
if [ 1 -eq 1 ]; then 
  rm -rf "$colmap_ws"
  if [ $? -ne 0 ]; then
    echo "Error when deleting previous colmap workspace"
    exit 1
  fi

  mkdir -p "$colmap_ws"/sparse/
  mkdir -p "$colmap_ws"/dense/
  if [ $? -ne 0 ]; then
    echo "Error when creating colmap workspace"
    exit 1
  fi
fi

# import known poses
if [ 1 -eq 1 ]; then
  if [ -d "$colmap_ws"/colmap_prior ]; then
    rm -rf "$colmap_ws"/colmap_prior/
  fi

  cp -r pycmu/meta/surveys/"$slice_id"/"$slice_id"_c"$cam_id"_"$survey_id"/colmap_prior/ \
    "$colmap_ws"/colmap_prior
  if [ $? -ne 0 ]; then
    echo "Error when copying prior information."
    exit 1
  fi
fi


if [ 1 -eq 1 ]; then
  ## feature extraction with known camera params on masked imgs, from directory
  echo "IMG_DIR: "$IMG_DIR""
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
  #  --match_list_path $colmap_ws/colmap_prior/match_list.txt \
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


if [ 0 -eq 1 ]; then
  # for when you have no prior on the camera pose, img list
  #"$COLMAP_BIN" mapper \
  #  --database_path $colmap_ws/database.db \
  #  --image_path "$IMG_DIR" \
  #  --image_list_path "$colmap_ws"/colmap_prior/image_list.txt \
  #  --output_path $colmap_ws/sparse/

  # for when you have no prior on the camera pose, from dir
  "$COLMAP_BIN" mapper \
    --database_path $colmap_ws/database.db \
    --image_path "$colmap_ws"img \
    --output_path $colmap_ws/sparse/

  if [ $? -ne 0 ]; then
    echo "Error during mapper."
    exit 1
  fi
fi

if [ 0 -eq 1 ]; then
    
    mkdir -p "$colmap_ws"/dense/images
    while read -r line
    do
        cp "$IMG_DIR""$line" "$colmap_ws"/dense/images/
    done < "$colmap_ws"/colmap_prior/image_list.txt
    cp -r "$colmap_ws"/sparse "$colmap_ws"/dense/
fi


if [ 0 -eq 1 ]; then
  #"$COLMAP_BIN" image_undistorter \
  #  --image_path "$IMG_DIR" \
  #  --input_path "$colmap_ws"/sparse/0 \
  #  --output_path "$colmap_ws"/dense/

  "$COLMAP_BIN" image_undistorter \
    --image_path "$IMG_DIR" \
    --input_path "$colmap_ws"/sparse/ \
    --output_path "$colmap_ws"/dense/

  if [ $? -ne 0 ]; then
    echo "Error during image_undistorter."
    exit 1
  fi
fi

if [ 0 -eq 1 ]; then
  "$COLMAP_BIN" patch_match_stereo \
    --workspace_path "$colmap_ws"/dense/

  if [ $? -ne 0 ]; then
    echo "Error during patch_match_stereo."
    exit 1
  fi
fi

if [ 0 -eq 1 ]; then
  "$COLMAP_BIN" stereo_fusion \
    --workspace_path "$colmap_ws"/dense/ \
    --output_path "$colmap_ws"/dense/fused.ply
  if [ $? -ne 0 ]; then
    echo "Error during stereo_fusion."
    exit 1
  fi
fi



