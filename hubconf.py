dependencies = ['torch']

import torch
from vpr_model import VPRModel
from models.backbones.dinov2 import DINOV2_ARCHS


def dinov2_salad(
        backbone : str = "dinov2_vitb14",
        pretrained=True,
        backbone_args=None,
        agg_args=None,
    ) -> torch.nn.Module:
    """Return a DINOv2 SALAD model.
    
    Args:
        backbone (str): DINOv2 encoder to use ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14').
        pretrained (bool): If True, returns a model pre-trained on GSV-Cities (only available for 'dinov2_vitb14').
        backbone_args (dict): Arguments for the backbone (check models.backbones.dinov2).
        agg_args (dict): Arguments for the aggregation module (check models.aggregators.salad).
    Return:
        model (torch.nn.Module): the model.
    """
    assert backbone in DINOV2_ARCHS.keys(), f"Parameter `backbone` is set to {backbone} but it must be one of {list(DINOV2_ARCHS.keys())}"
    assert not pretrained or backbone == "dinov2_vitb14", f"Parameter `pretrained` can only be set to True if backbone is 'dinov2_vitb14', but it is set to {backbone}"


    backbone_args = backbone_args or {
        'num_trainable_blocks': 4,
        'return_token': True,
        'norm_layer': True,
    }
    agg_args = agg_args or {
        'num_channels': DINOV2_ARCHS[backbone],
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256,
    }
    model = VPRModel(
        backbone_arch=backbone,
        backbone_config=backbone_args,
        agg_arch='SALAD',
        agg_config=agg_args,
    )
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            f'https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt',
            map_location=torch.device('cpu')
        )
    )
    return model