# Getting Started

## 1. Set Up the Environment
Create the conda environment:
```bash
conda create -n BrickAnything python=3.10.20
conda activate BrickAnything
pip install -r requirements.txt
pip install flashattention=2.7.3
pip install bpy=4.0.0
```
Running stability analysis requires a Gurobi license to use Gurobi. Academics may request a free license from the [Gurobi website](https://www.gurobi.com/product/download-center); after obtaining the license, place it in your home directory or another recommended location.

Download the LDraw parts library: `cd ~ && wget https://library.ldraw.org/library/updates/complete.zip && unzip complete.zip`
If you wish to put the LDraw parts library in a different directory, set the environment variable LDRAW_LIBRARY_PATH to the path of the ldraw directory: `export LDRAW_LIBRARY_PATH=path/to/ldraw`,which one I recommend.

Download Michelangelo's point encoder from [website](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) and put it into `brickanything_train/miche/checkpoints/aligned_shape_latents/shapevae-256.ckpt`.

# Acknowledgement
