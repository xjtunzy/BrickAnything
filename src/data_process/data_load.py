import objaverse
import os
import multiprocessing


new_base = "/mnt/nas/yanfeng/data/n_project/objaverse"
objaverse.BASE_PATH = os.path.expanduser(new_base)
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, objaverse.__version__)


uids = objaverse.load_uids()
#annotations = objaverse.load_annotations(uids[:10])

processes = multiprocessing.cpu_count()

objects = objaverse.load_objects(
    uids=uids,
    download_processes=processes
)
print(objects)