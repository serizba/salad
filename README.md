## Inference-only branch. Without dependencies. Without training and evaluation code.

![DINOv2 SALAD](assets/dino_salad_title.png)
# Optimal Transport Aggregation for Visual Place Recognition
Sergio Izquierdo, Javier Civera

Code and models for Optimal Transport Aggregation for Visual Place Recognition (DINOv2 SALAD).

## Summary

We introduce DINOv2 SALAD, a Visual Place Recognition model that achieves state-of-the-art results on common benchmarks. We introduce two main contributions:
 - Using a finetuned DINOv2 encoder to get richer and more powerful features.
 - A new aggregation technique based on optimal transport to create a global descriptor based on optimal transport. This aggregation extends NetVLAD to consider feature-to-cluster relations as well as cluster-to-features. Besides, it includes a dustbin to discard uninformative features.

For more details, check the paper at [arXiv](https://arxiv.org/abs/2311.15937).

![Method](assets/method.jpg)

## Setup (inference-only mode)

Create a ready to run environment with:
```bash
conda env create -f environment.yml
```
or
'''bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
'''

To quickly test and use our model, you can use Torch Hub:
```python
import torch
model = torch.hub.load("serizba/salad:inference-no-deps", "dinov2_salad")
model.eval()
model.cuda()
```



## Acknowledgements
This code is based on the amazing work of:
 - [MixVPR](https://github.com/amaralibey/MixVPR)
 - [GSV-Cities](https://github.com/amaralibey/gsv-cities)
 - [DINOv2](https://github.com/facebookresearch/dinov2)

## Cite
Here is the bibtex to cite our paper
```
@InProceedings{Izquierdo_CVPR_2024_SALAD,
    author    = {Izquierdo, Sergio and Civera, Javier},
    title     = {Optimal Transport Aggregation for Visual Place Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
}
```
