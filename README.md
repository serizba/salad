# Optimal Transport Aggregation for Visual Place Recognition

Code for visual place recognitions using DINOv2 SALAD.

## Setup

Install required dependencies:
```python
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning prettytable faiss-gpu pytorch-metric-learning torchmetrics pandas
```

## Dataset

For training, download [GSV-Cities](https://github.com/amaralibey/gsv-cities) dataset, for evaluation download the desired datasets ([MSLS](https://github.com/FrederikWarburg/mapillary_sls), [NordLand](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), [SPED](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), [Pittsburgh](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/))

## Train

For training DINOv2 SALAD on GSV-Cities run:
```bash
python3 main.py
```

## Evaluation

You can download a pretrained DINOv2 SALAD model from [here](https://drive.google.com/file/d/1pIGThm04pSo2mJIYoBqhzhyqlqzJxOES/view?usp=sharing). For evaluating run:

```bash
python3 eval.py --ckpt_path 'dino_salad.ckpt' --image_size 322 322 --batch_size 256 --val_datasets MSLS
```