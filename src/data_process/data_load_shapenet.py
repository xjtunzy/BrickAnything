
from huggingface_hub import snapshot_download
import os, glob, zipfile

local_dir = "/mnt/nas/yanfeng/data/n_project/ShapeNet"  # 你想存放的位置

# snapshot_download(
#     repo_id="ShapeNet/ShapeNetCore",
#     repo_type="dataset",
#     local_dir=local_dir,
#     token=os.environ["HF_TOKEN"],
#     local_dir_use_symlinks=False,
# )

for z in glob.glob(os.path.join(local_dir, "*.zip")):
    out = os.path.join(local_dir, os.path.splitext(os.path.basename(z))[0])
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(z, "r") as f:
        f.extractall(out)
print("done")