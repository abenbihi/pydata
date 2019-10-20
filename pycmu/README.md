# CMU-Seasons
This repo holds python tools to manipulate the CMU-Seasons data for tasks such
as image retrieval (VLAD, NetVLAD), localisation (WASABI), reconstruction (COLMAP).

### Download data

Set the constant `EXT_IMG_DIR` in `cst.py`

    EXT_IMG_DIR = /full/path/to/Extended-CMU-Seasons

### Split slices into surveys


Run the function `get_survey_auto`.
    python -m splut_survey --slice_id 24


### COLMAP

Generate known pose in colmap format. This allows colmap to start from the
known poses when generating the sparse reconstruction.

    python -m colmap --slice_id 24 --cam_id 0 --survey_id 0


### NetVLAD


```bash
./scripts/cmu_split_surveys.sh
./scripts/cmu_get_pose.sh

./scripts/cmu_colmap_prior.sh
./scripts/colmap.sh 6 0 -1

./scripts/export_inlier_pairs.sh 6 0 -1 15

./scripts/cmu_match.sh

```
