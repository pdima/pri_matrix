import pandas as pd
import pims
import config
import os
from pprint import pprint

res_map = {}

training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)

for i, fn in enumerate(training_set_labels_ds_full.filename):
    v = pims.Video(os.path.join(config.RAW_VIDEO_DIR, fn))
    resolution = tuple(v.frame_shape)

    res_map[resolution] = res_map.get(resolution, 0) + 1

    if i % 1000 == 0:
        print(i)
        pprint(res_map)
