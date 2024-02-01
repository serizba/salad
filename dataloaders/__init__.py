import os
from pathlib import Path


__DEFAULT = '/material/data/VPR/'
if 'DATAROOT' in os.environ:
    DATAROOT = os.environ['DATAROOT']
    if DATAROOT is None:
        DATAROOT = __DEFAULT
else:
    DATAROOT = __DEFAULT


def make_path(dataset_name: str):
    return str(Path(DATAROOT).joinpath(dataset_name)) + os.sep
