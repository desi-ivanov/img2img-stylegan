## StyleGAN2-UNet
A style-based GAN with UNet-guided synthesis.

Some examples from a GAN trained on [UniToPatho](https://github.com/EIDOSLAB/UNITOPATHO):

![multi-style-generation](./docs/multi-style-generation.jpg)

https://user-images.githubusercontent.com/20824840/198903666-7a130b1d-4bc3-49d5-8f41-e964515b2adb.mp4

## Training new networks
### Data preparation
You need to put your training images into one folder and your segmentation masks into another folder. The names of the images and masks must be paired together in a lexicographical order.

### Training
To train a network (or resume training), you must specify the path to the segmentation masks through the `seg_data` option and additionally provide the RGB colors for each class through the `seg_colors` option encoded in JSON format.

```bash
python3 train.py \
  --resume=$HOME/training-runs/00000-train-auto1-resumecustom/network-snapshot-001000.pkl \
  --outdir=$HOME/training-runs \
  --data=$HOME/datasets/crops2-512-filtered/train \
  --seg_data=$HOME/datasets/segmented-512-filtered/train \
  --seg_colors="[[0,0,255],[0,128,0],[0,191,191],[191,0,191],[255,0,0],[255,255,255]]" \
  --image_snapshot_ticks=1 \
  --wandb_project=sgunet \
  --snap=50 \
  --batch=16 \
  --batch_gpu=4 \
  --gpus=1
```

In this example, the results are saved to a newly created directory `~/training-runs/<ID>-mydataset-auto1`, controlled by `--outdir`. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (controlled by `--snap`). For each pickle, it also evaluates FID (controlled by `--metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed).

The name of the output directory reflects the training configuration. For example, `00000-mydataset-auto1` indicates that the *base configuration* was `auto1`, meaning that the hyperparameters were selected automatically for training on one GPU. The base configuration is controlled by `--cfg`:

| Base config           | Description
| :-------------------- | :----------
| `auto`&nbsp;(default) | Automatically select reasonable defaults based on resolution and GPU count. Serves as a good starting point for new datasets but does not necessarily lead to optimal results.
| `stylegan2`           | Reproduce results for StyleGAN2 config F at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `paper256`            | Reproduce results for FFHQ and LSUN Cat at 256x256 using 1, 2, 4, or 8 GPUs.
| `paper512`            | Reproduce results for BreCaHAD and AFHQ at 512x512 using 1, 2, 4, or 8 GPUs.
| `paper1024`           | Reproduce results for MetFaces at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `cifar`               | Reproduce results for CIFAR-10 (tuned configuration) using 1 or 2 GPUs.

The training configuration can be further customized with additional command line options:

* `--aug=noaug` disables ADA.
* `--cond=1` enables class-conditional training (requires a dataset with labels).
* `--mirror=1` amplifies the dataset with x-flips. Often beneficial, even with ADA.
* `--resume=ffhq1024 --snap=10` performs transfer learning from FFHQ trained at 1024x1024.
* `--resume=~/training-runs/<NAME>/network-snapshot-<INT>.pkl` resumes a previous training run.
* `--gamma=10` overrides R1 gamma. We recommend trying a couple of different values for each new dataset.
* `--aug=ada --target=0.7` adjusts ADA target value (default: 0.6).
* `--augpipe=blit` enables pixel blitting but disables all other augmentations.
* `--augpipe=bgcfnc` enables all available augmentations (blit, geom, color, filter, noise, cutout).

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list.

## Inference Server
You can spin up an API server that can be used for inference. Refer to [srv.py](./srv.py) for more details.

Example usage:
```bash
export SG_SEG_DATASET_PATH="$HOME/datasets/segmented-512-filtered/train"
export SG_REAL_DATASET_PATH="$HOME/datasets/crops2-512-filtered/train"
export SG_MODEL_PATH="$HOME/training-runs/00000-train-auto1-resumecustom/network-snapshot-001000.pkl"
export SG_SEG_COLORS_JSON="[[0,0,255],[0,128,0],[0,191,191],[191,0,191],[255,0,0],[255,255,255]]"
FLASK_APP=srv.py python -m flask run --host=0.0.0.0
```

## Acknowledgements

This work was supported by [EIDOSLab](https://github.com/EIDOSlab). The code is heavily based on [StyleGAN2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).
