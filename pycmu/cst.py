
# colmap output dir
MACHINE = 0
if MACHINE == 0:
    DATASET_DIR = '/home/abenbihi/ws/datasets/'
    WS_DIR = '/home/abenbihi/ws/'
    EXT_IMG_DIR = '%s/datasets/Extended-CMU-Seasons/'%WS_DIR
elif MACHINE == 1:
    DATASET_DIR = '/home/gpu_user/assia/ws/datasets/'
    WS_DIR = '/home/gpu_user/assia/ws/'
    EXT_IMG_DIR = '/home/gpu_user/assia/ws/datasets/Extended-CMU-Seasons-Undistorted/'
elif MACHINE == 2:
    WS_DIR = '/opt/BenbihiAssia/ws/'
    DATASET_DIR = '%s/datasets/'%WS_DIR
    EXT_IMG_DIR = '%s/datasets/Extended-CMU-Seasons/'%WS_DIR
    #DATA_DIR = '/home/abenbihi/ws/datasets/CMU-Seasons/'
else:
    print('Get you MTF MACHINE macro correct !')
    exit(1)

COLMAP_WS_DIR = '%s/datasets/colmap'%WS_DIR

