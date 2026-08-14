# BrickAnything
Official code of **BrickAnything：Geometry-Conditioned Buildable Brick Generation with Structure-Aware Tokenization**.

**BrickAnything** is a geometry-conditioned generative framework for creating physically buildable brick assemblies from 3D shapes. By combining structure-aware tree modeling, buildability-aware preference optimization, and efficient structural rollback, BrickAnything generates brick structures with improved geometric fidelity, validity, and physical stability.

<p align="center">
  <img src="assets/results.png" width="1000">
</p>

## Getting Started

### Set Up the Environment
The code has been tested on Ubuntu 20.04.6 LTS with CUDA Toolkit 11.7 and an NVIDIA GeForce RTX 4090 (24 GB). 

Let us begin by creating the conda environment:
```bash
git clone --recursive https://github.com/xjtunzy/BrickAnything.git
cd BrickAnything
conda create -n BrickAnything python=3.10.20
conda activate BrickAnything
pip install -r requirements.txt
```
```
pip install flash-attn==2.7.3
pip install bpy==4.0.0
```
> **Note:** `flash-attn==2.7.3` and `bpy==4.0.0` may require pre-built wheels depending on your Python, CUDA, and platform versions.  
> - FlashAttention wheels: [website](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3)  
> - Blender `bpy` wheels: [website](https://download.blender.org/pypi/bpy/)
>
> Please select the wheel compatible with your environment rather than blindly using the commands below.

Running stability analysis requires a Gurobi license to use Gurobi. Academics may request a free license from the [Gurobi website](https://www.gurobi.com/product/download-center); after obtaining the license, place it in your home directory or another recommended location.

Download the LDraw parts library: `cd ~ && wget https://library.ldraw.org/library/updates/complete.zip && unzip complete.zip`
Alternatively, you can place the LDraw library in a custom directory and set: `export LDRAW_LIBRARY_PATH=/path/to/ldraw` Using the `LDRAW_LIBRARY_PATH` environment variable is recommended.

Install the ImportLDraw submodule with `git submodule update --init`.

Download this [background exr file](https://drive.google.com/file/d/1Yux0sEqWVpXGMT9Z5J094ISfvxhH-_5K/view) and place it in the `ImportLDraw/loadldraw` subdirectory.

Download Michelangelo's point encoder from [website](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) and put it into `src/brickanything_train/miche/checkpoints/aligned_shape_latents/shapevae-256.ckpt`.

Download the BrickAnything's model weight from [website](https://huggingface.co/niels-peter/BrickAnything/blob/main) and put it in your home, then modify the `ckpt_path` in`model_config/opt_tree_mode.yaml`

### Run inference
```bash
# mesh condition
# Single mesh 
python src/main.py \
  --config model_config/opt_tree_mode.yaml \
  --input_path path_to_target_obj \
  --out_dir inference_out \
  --input_type mesh \
  --mc \
  --mc_level 7 \
  --do_render
# Directory containing multiple meshes
python src/main.py \
  --config model_config/opt_tree_mode.yaml \
  --input_dir path_to_target_obj \
  --out_dir inference_out \
  --input_type mesh \
  --mc \
  --mc_level 7 \
  --do_render
```
```bash
# pointcloud condition
# Single pointcloud 
python src/main.py \
  --config model_config/opt_tree_mode.yaml \
  --input_path pc_example \ 
  --out_dir inference_out \
  --input_type pc_normal \
  --do_render
# Dir of several pointcloud
python src/main.py \
  --config model_config/opt_tree_mode.yaml \
  --input_dir pc_examples \ 
  --out_dir inference_out \
  --input_type pc_normal \
  --do_render
```
## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{ni2026brickanything,
  title         = {BrickAnything: Geometry-Conditioned Buildable Brick Generation with Structure-Aware Tokenization},
  author        = {Zhengyang Ni and Feng Yan and Yu Guo and Fei Wang},
  year          = {2026},
  eprint        = {2605.26182},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```
## Acknowledgements
Our code is based on these wonderful repos:
- [MeshAnything](http://github.com/buaacyw/MeshAnything)
- [MeshAnythingV2](https://github.com/buaacyw/MeshAnythingV2)
- [BrickGPT](https://github.com/AvaLovelace1/BrickGPT)
- [StableLego](https://github.com/intelligent-control-lab/StableLego)
