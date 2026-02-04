# Lejurobot Challenge

## Setup

Create conda environment:
```bash
conda create -n lejurobot python=3.11
conda activate lejurobot
```

Install package:
```bash
pip install .
```

Install FFmpeg (required for torchcodec video backend):
```bash
conda install -n lejurobot -c conda-forge "ffmpeg<8" -y
```

**Note:** FFmpeg version 8+ is not compatible with torchcodec. Use version 7 or earlier (the command above installs FFmpeg 6.1.1 which is compatible).

## Data

Download the dataset from:
https://huggingface.co/datasets/LejuRobotics/kuavo_data_challenge_icra

## Resources

- Official competition repository: https://github.com/LejuRobotics/kuavo_data_challenge/tree/icra
- Simulator: https://github.com/LejuRobotics/kuavo-ros-opensource/tree/opensource/kuavo-data-challenge-icra
