#!/bin/sh

. ./scripts/export_path.sh

COLMAP_BIN="$WS_DIR"tools/colmap/build/src/exe/colmap

if [ "$#" -eq 0 ]; then 
  echo "Arguments: "
  echo "1: slice"
  echo "2: camera id"
  echo "3: survey id"
  echo "4. min_num_matches"
  exit 1
fi


if [ "$#" -ne 4 ]; then 
  echo "Error: bad number of arguments"
  echo "1: slice"
  echo "2: camera id"
  echo "3: survey id"
  echo "4. min_num_matches"
  exit 1
fi

slice_id="$1"
cam_id="$2"
survey_id="$3"
min_num_matches="$4"

if [ "$survey_id" -eq -1 ]; then # db survey
  project_name="$slice_id"_c"$cam_id"_db
else # q survey
  project_name="$slice_id"_c"$cam_id"_"$survey_id"
fi
colmap_ws="$PYDATA_DIR"pycmu/res/colmap/"$project_name"/
database_path="$colmap_ws"database.db
#match_list_path="$WASABI_DIR"meta/cmu/colmap/image_pairs_to_match/"$project_name".txt
match_list_path="$PYDATA_DIR"pycmu/meta/surveys/"$slice_id"/"$project_name"/colmap_prior/image_pairs_to_match_intra.txt

echo "database_path: "$database_path""
echo "match_list_path: "$match_list_path""
#if [ "$survey_id" -eq -1 ]; then # db survey
  python3 -m pycmu.export_inlier_pairs \
    --database_path "$database_path" \
    --match_list_path "$match_list_path" \
    --min_num_matches "$min_num_matches"
#else # q survey

#fi
