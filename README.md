Pixel Art Diffusion

This repository contains code for training pixel-art diffusion models.

Run `uv sync` to install dependencies (project uses requirements.txt).

## Usage

Install dependencies:

uv sync

Start training (example):

python train.py --data_path ./data --image_size 16 --batch_size 128

For higher-quality models (more memory / compute):

python train.py --base_channels 128 --num_res_blocks 3 --channel_mults "1,2,4"

View training metrics with TensorBoard (logs default to ./runs_cond):

tensorboard --logdir ./runs_cond --port 6006

Open http://localhost:6006 in your browser to view the dashboard.
